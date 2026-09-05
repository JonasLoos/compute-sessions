#!/usr/bin/env python3
"""compute-sessions runner — the sbatch job script that activates one session on a compute node. Uploaded to <remote_base>/runner.py at cluster setup; stdlib-only (python3.8+).

Starts an Apptainer container running sshd, opens a reverse-SSH tunnel back to the login node (Unix socket in the session dir on the shared FS), and restores/archives the project venv via tar+zstd snapshots — the live venv stays on node-local /tmp, because a CUDA torch venv is ~4.7 GB / ~22k tiny files and loose venv files are murder on a parallel FS. After activation it idle-monitors the session (established sshd connections, `.pid` heartbeats of tracked commands, and the `.activity` sentinel touched by host-side file tools) and writes the final status on the way out.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Isolation policy — CONSTANT bind list. Edit here and re-upload runner.py to
# change what's visible inside sessions. Intentionally NOT driven by
# per-session config (which an agent can write) so changing the isolation
# surface requires SSH access to the cluster.
#
# Paths are relative to the invoking user's $HOME and bound at the same path
# inside the container. Entries whose source doesn't exist are created first so
# the caches actually persist across sessions.
# ---------------------------------------------------------------------------
ISOLATION_BINDS = [
    # HF model cache + auth token
    ".cache/huggingface",
    ".local/share/huggingface",
    # uv: package cache + uv-managed Python installs (both persisted across sessions).
    ".cache/uv",
    ".local/share/uv",
    # Wandb: config + local run staging
    ".config/wandb",
    ".local/share/wandb",
]

IDLE_POLL_SECONDS = 10
# A `.pid` refreshed within this window counts as a live command. Must exceed run.py's heartbeat interval (10s) by a comfortable margin so one missed touch or FS-mtime lag never makes a live command look idle.
PID_FRESH_SECONDS = 60
HEARTBEAT_PERSIST_SECONDS = 60


# Snapshot helper — runs INSIDE the container, spawned detached by run.py's command epilogue. Lives in the session dir (bound read-only at /cs-helpers) so iterating on it only needs a runner re-upload, not an image rebuild.
# State-based: snapshots whenever the live freeze hash differs from the marker, which is written only on publish (here) or restore (at activation) — a snapshot that dies midway simply retries on the next command, and any install path is caught. The no-change exit is near-instant.
# Registers as a tracked cmd_venvsnap_* command (.start/.pid heartbeat/.exit in /cs-logs): visible in list_commands, killable, and the heartbeat keeps the idle monitor from reaping the session mid-snapshot.
# Kept as bash (not python): it runs inside the container image, which deliberately ships no system python — only uv-managed project interpreters.
VENV_SNAPSHOT_SH = r"""#!/bin/bash
set -u
MARKER=/cs-venv/.cs-freeze-hash
LOCKDIR=/tmp/cs-venvsnap.lock
command -v uv >/dev/null 2>&1 || exit 0
[ -d /cs-venv ] && [ -d /cs-venv-cache ] || exit 0
# Hash = installed packages + interpreter version (pyvenv.cfg) — a python bump without package changes must also re-snapshot.
freeze_hash() { { uv pip freeze 2>/dev/null; grep -i '^version' /cs-venv/venv/pyvenv.cfg 2>/dev/null; } | sha256sum | cut -c1-16; }
# An empty venv is nothing worth caching — and skipping it means a wiped venv can never tombstone over good snapshots.
[ -n "$(uv pip freeze 2>/dev/null | head -c 1)" ] || exit 0
cur=$(freeze_hash)
[ "$cur" = "$(cat "$MARKER" 2>/dev/null)" ] && exit 0
mkdir "$LOCKDIR" 2>/dev/null || exit 0  # another snapshot is in flight; the next command epilogue retries
# Signal traps so a kill still runs the EXIT trap — otherwise a stale lock blocks all future snapshots for the job's lifetime. The minimal EXIT trap covers the setup window; the full one replaces it below.
trap 'exit 143' TERM; trap 'exit 130' INT; trap 'exit 129' HUP
trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT
cid="cmd_venvsnap_$(date +%s)_$$"
b="/cs-logs/$cid"
: > "$b.start"; : > "$b.out"; : > "$b.err"; echo $$ > "$b.pid"
( while kill -0 $$ 2>/dev/null; do touch "$b.pid"; sleep 10; done ) &
trap 'ec=$?; echo "$ec" > "$b.exit"; rmdir "$LOCKDIR" 2>/dev/null' EXIT
{
  tag=nolock
  if [ -f /workdir/uv.lock ]; then tag=$(sha256sum /workdir/uv.lock | cut -c1-16); fi
  tmp="/cs-venv-cache/.snap.$cid.tmp"
  nice -n 19 tar -C /cs-venv -I 'zstd -T0 -3' -cf "$tmp" . || { rm -f "$tmp"; exit 1; }
  # Discard if the venv changed mid-tar; the marker stays stale, so the next command epilogue retries cleanly.
  [ "$(freeze_hash)" = "$cur" ] || { rm -f "$tmp"; echo "venv changed mid-snapshot; discarded"; exit 0; }
  mv -f "$tmp" "/cs-venv-cache/$tag.$cur.$(date +%s).tar.zst"
  echo "$cur" > "$MARKER"
  echo "published snapshot tagged $tag"
  # Keep the newest 3 snapshots (each ~GBs; parallel FS often runs near-full) and sweep orphaned publish temps (dotfiles, so the glob above never matches them; age-gated so a live temp survives).
  ls -t /cs-venv-cache/*.tar.zst 2>/dev/null | tail -n +4 | xargs -r rm -f
  find /cs-venv-cache -maxdepth 1 -name '.snap.*.tmp' -mmin +180 -delete 2>/dev/null || true
} > "$b.out" 2> "$b.err"
"""


def utcnow():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg):
    sys.stderr.write(msg.rstrip() + "\n")
    sys.stderr.flush()


class Session:
    """Paths + atomic config.json access for one session."""

    def __init__(self, base, session_id):
        self.base = base
        self.id = session_id
        self.dir = base / "sessions" / session_id
        self.config_path = self.dir / "config.json"
        self.workdir = self.dir / "workdir"
        self.logdir = self.dir / "logs"
        self.home_dir = self.dir / "home"
        self.sshd_dir = self.dir / ".sshd"
        self.helpers_dir = self.dir / ".helpers"
        self.socket = self.dir / "session.sock"
        self.activity = self.dir / ".activity"

    def config(self):
        return json.loads(self.config_path.read_text())

    def config_update(self, _expect=None, **fields):
        """Read-modify-write with an atomic replace. config.json has concurrent writers (this runner + the host-side cs client, which also writes via temp + rename), so a reader can never observe a half-written file; last-writer-wins on races is acceptable for these heartbeat-ish fields. `_expect` maps field -> tuple of acceptable current values; when the record doesn't match, nothing is written and False is returned (compare-and-set for the final status write)."""
        d = self.config()
        if _expect and any(d.get(k) not in vals for k, vals in _expect.items()):
            return False
        d.update(fields)
        tmp = self.config_path.with_name(f"config.json.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(d, indent=2))
        os.replace(tmp, self.config_path)
        return True


def pick_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def detect_cuda_version():
    """Host driver's CUDA version (e.g. "12.9"), so project code can pick a matching torch wheel instead of guessing. Parsed from the nvidia-smi header (`--query-gpu=cuda_version` isn't universally supported). Empty string on CPU-only nodes."""
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return ""
    m = re.search(r"CUDA Version: ([0-9.]+)", out)
    return m.group(1) if m else ""


def detect_gpu_models():
    """Human-readable model(s) of the GPU(s) this job actually got (e.g. "NVIDIA A100 80GB PCIe", "2x NVIDIA H100"), from `nvidia-smi -L`. The job's cgroup normally scopes this to the allocated devices. The requested gpu_type is only a constraint filter, so recording the real allocation lets show/list report actual hardware. Empty string when no GPU/driver.

    MIG nodes break the cgroup assumption: every parent GPU stays visible and the actual allocation is a MIG slice listed underneath (observed: a 1-GPU request reported as "8x NVIDIA A100 80GB PCIe" while the job held one 40GB slice — the agent then OOM'd trusting the phantom 80GB). MIG device lines therefore win over parent lines, filtered to CUDA_VISIBLE_DEVICES when the scheduler set it to MIG UUIDs."""
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return ""
    parent = ""
    migs = []  # (display name, uuid)
    names = []
    for line in out.splitlines():
        m = re.match(r"GPU \d+: (.+?) \(", line)
        if m:
            parent = m.group(1).strip()
            names.append(parent)
            continue
        m = re.match(r"\s+MIG (\S+)\s+Device\s+\d+: \(UUID: (MIG-[^)]+)\)", line)
        if m:
            migs.append((f"MIG {m.group(1)}" + (f" ({parent})" if parent else ""), m.group(2)))
    if migs:
        visible = [t.strip() for t in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if t.strip().startswith("MIG-")]
        selected = [name for name, uuid in migs if uuid in visible] or [name for name, _ in migs]
        names = selected
    if not names:
        return ""
    if len(set(names)) == 1:
        return names[0] if len(names) == 1 else f"{len(names)}x {names[0]}"
    return ", ".join(names)


def prepare_sshd(sess, port, setenv):
    """Session-local sshd config + host key + authorized_keys, written under sess.sshd_dir (bind-mounted into the container at /sshd)."""
    sess.sshd_dir.mkdir(parents=True, exist_ok=True)
    host_key = sess.sshd_dir / "host_key"
    if not host_key.is_file():
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-f", host_key, "-N", "", "-q"], check=True)

    # The in-container sshd accepts any key that can already log into the cluster — i.e. the user's own ~/.ssh/authorized_keys. That means the same laptop key used to reach the login node authenticates into the container too, without shipping an extra keypair.
    auth = sess.sshd_dir / "authorized_keys"
    src_auth = Path.home() / ".ssh" / "authorized_keys"
    if not src_auth.is_file():
        raise SystemExit(f"fatal: {src_auth} does not exist on the cluster — add your machine's public key there before creating sessions.")
    shutil.copyfile(src_auth, auth)
    auth.chmod(0o600)

    lines = [
        f"Port {port}",
        "ListenAddress 127.0.0.1",
        "HostKey /sshd/host_key",
        "PidFile /sshd/sshd.pid",
        "AuthorizedKeysFile /sshd/authorized_keys",
        "PasswordAuthentication no",
        "PubkeyAuthentication yes",
        "UsePAM no",
        "PermitRootLogin no",
        "StrictModes no",
        "Subsystem sftp internal-sftp",
    ]
    # OpenSSH scrubs the environment on each new session, so env vars given to the sshd process are only visible to sshd itself — not to the shells spawned for incoming `run` connections. SetEnv makes them reach those shells. All vars MUST share ONE SetEnv line: sshd applies the first-obtained-value rule per keyword, so multiple `SetEnv` lines keep only the first and silently drop the rest (verified with `sshd -T` on OpenSSH 9.6). The forwarded values are space-free, so a single space-separated line is safe.
    pairs = " ".join(f"{k}={v}" for k, v in setenv.items() if v)
    if pairs:
        lines.append(f"SetEnv {pairs}")
    (sess.sshd_dir / "sshd_config").write_text("\n".join(lines) + "\n")


def wait_for_ssh_banner(port, timeout, procs):
    """Wait until something on 127.0.0.1:<port> answers with an SSH banner — reading the banner proves sshd itself is up, not just that the port accepts. Breaks out early if any of `procs` died (real crashes fail fast instead of burning the window)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for p in procs:
            if p.poll() is not None:
                return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2) as s:
                s.settimeout(2)
                if s.recv(64).startswith(b"SSH-"):
                    return True
        except OSError:
            pass
        time.sleep(0.25)
    return False


def established_connections(port):
    """Count established TCP connections involving `port`, parsed from /proc/net/tcp{,6} (no ss/netstat dependency). Apptainer shares the host network namespace, so this sees the in-container sshd directly."""
    n = 0
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(path).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            f = line.split()
            if len(f) < 4 or f[3] != "01":  # 01 = ESTABLISHED
                continue
            try:
                lport = int(f[1].rsplit(":", 1)[1], 16)
                rport = int(f[2].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if lport == port or rport == port:
                n += 1
    return n


def is_busy(sess, port):
    """Activity = an established SSH connection to the session sshd, OR a live user command tracked by a .pid file in the logs dir (an unfinished `run` — .pid written, matching .exit not yet written, mtime recently refreshed), OR a fresh .activity mtime (touched by the host-side cs client on file ops, which are otherwise invisible here).

    Liveness is read from the .pid file's MTIME, not `kill -0` on its contents: the container has a private PID namespace, so the recorded PID is meaningless to this host-side process. run.py's inner bash refreshes the mtime every 10s from *inside* the container and stops within 10s of dying by any means — mtime is namespace-independent, so the host-side read is correct."""
    if established_connections(port) > 0:
        return True
    now = time.time()
    try:
        pidfiles = list(sess.logdir.glob("*.pid"))
    except OSError:
        pidfiles = []
    for pidfile in pidfiles:
        # Command ids contain no dots, so with_suffix cleanly swaps .pid → .exit.
        if pidfile.with_suffix(".exit").exists():
            continue
        try:
            if now - pidfile.stat().st_mtime < PID_FRESH_SECONDS:
                return True
        except OSError:
            continue
    try:
        if now - sess.activity.stat().st_mtime < PID_FRESH_SECONDS:
            return True
    except OSError:
        pass
    return False


def monitor_loop(sess, procs, port, idle_timeout_minutes):
    """Block until a child process dies or the session idles out. Returns a human-readable reason. Polls every ~2s for child death (a dead sshd/tunnel must not hang the job) and every IDLE_POLL_SECONDS for activity — a brief connect/disconnect between coarser checks would be missed. Persists the last-active heartbeat to config.json at most once per HEARTBEAT_PERSIST_SECONDS so the host-side cs client can compute `seconds_until_idle_deactivate` without hammering the FS."""
    timeout = idle_timeout_minutes * 60
    last_active = time.time()
    last_check = 0.0
    last_persisted = 0.0
    while True:
        time.sleep(2)
        for p in procs:
            if p.poll() is not None:
                return f"process exited: {p.args[0]} (rc={p.returncode})"
        now = time.time()
        if now - last_check >= IDLE_POLL_SECONDS:
            last_check = now
            if is_busy(sess, port):
                last_active = now
            if now - last_persisted >= HEARTBEAT_PERSIST_SECONDS:
                try:
                    sess.config_update(last_activity_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(last_active)))
                except Exception as exc:
                    log(f"heartbeat persist failed (ignored): {exc}")
                last_persisted = now
            if now - last_active >= timeout:
                return f"idle timeout reached ({idle_timeout_minutes} min)"


def _sha256_16(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _restore_venv(sess, image, venv_dir, venv_cache):
    """Restore /cs-venv from the project's newest snapshot: prefer one tagged with the current uv.lock hash, else take the newest as a stale base for `uv sync` to delta-fix, else start empty. Snapshot identity is the OBSERVED venv state — a `uv pip freeze` hash — not the lock hash: `uv pip install` changes the venv without the lock, and a lock edit without a sync changes no venv state. The lock hash only picks the restore candidate ("nolock" for lock-less projects). Archive names are <locktag>.<freezehash>.<epoch>.tar.zst; the current state's hash is mirrored at /cs-venv/.cs-freeze-hash (at the mount root, outside uv's /cs-venv/venv) so the helper can skip unchanged states."""
    lock = sess.workdir / "uv.lock"
    lock_tag = _sha256_16(lock) if lock.is_file() else "nolock"

    def newest(pattern):
        snaps = sorted(venv_cache.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return snaps[0] if snaps else None

    snap = newest(f"{lock_tag}.*.tar.zst")
    kind = "exact"
    if snap is None:
        snap = newest("*.tar.zst")
        kind = "stale"
    if snap is None:
        return
    # Unpack inside the container — zstd is guaranteed in the image, not on compute nodes. Plain directory binds have no single-writer constraint, so this pre-exec can't conflict with the sshd exec that follows.
    rc = subprocess.run([
        "apptainer", "exec", "--containall", "--writable-tmpfs",
        "-B", f"{venv_dir}:/cs-venv",
        "-B", f"{snap}:/cs-snapshot.tar.zst:ro",
        image, "tar", "-I", "zstd -T0", "-xf", "/cs-snapshot.tar.zst", "-C", "/cs-venv",
    ]).returncode
    if rc == 0:
        # Re-seed the freeze marker from the snapshot filename so the first command's epilogue doesn't re-archive an unchanged venv.
        (venv_dir / ".cs-freeze-hash").write_text(snap.name.split(".")[1] + "\n")
        log(f"venv: restored {kind} snapshot {snap.name}")
    else:
        log(f"venv: restore failed from {snap} — starting empty")
        shutil.rmtree(venv_dir, ignore_errors=True)
        venv_dir.mkdir(parents=True, exist_ok=True)


def activate(sess, cfg, port, login, procs, cleanups):
    """Start the Apptainer container (sshd in the foreground) and the reverse tunnel, registering them in `procs`/`cleanups` as they come up — so main()'s finally-block can tear down partial activations too."""
    home = Path.home()
    image = Path(cfg.get("image", ""))
    if not image.is_absolute():
        image = sess.base / image

    # Pre-create cache dirs under $HOME so first-session binds actually persist. Without this, a fresh account has no ~/.cache/uv; the bind would be skipped/created-empty and uv would write to the ephemeral per-session home instead, losing its cache on every deactivation.
    for rel in ISOLATION_BINDS:
        (home / rel).mkdir(parents=True, exist_ok=True)
    sess.home_dir.mkdir(parents=True, exist_ok=True)

    # Env passthrough. Host shell/user creds are NOT forwarded (tools pick those up from bound on-disk config). We do forward:
    #   - CUDA_VISIBLE_DEVICES: SLURM-allocated GPU(s). Apptainer --nv only wires the driver/libs; without this, torch sees every GPU on the node even though cgroups only permit access to one.
    #   - CS_CUDA_VERSION: host driver's CUDA version, so pyproject / user code can pick a matching torch wheel.
    # SLURM_* are deliberately NOT forwarded: compute-sessions abstracts the scheduler away, and CUDA_VISIBLE_DEVICES already conveys the GPU slice.
    setenv = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "CS_CUDA_VERSION": detect_cuda_version(),
    }

    # Venv hibernation: the live venv is a plain directory on node-local /tmp; persistence comes from tar+zstd snapshots in the per-project cache on the shared FS.
    venv_cache = sess.base / "venv-cache" / cfg.get("project_id", "unnamed")
    venv_dir = Path(f"/tmp/cs-venv.{sess.id}")
    venv_dir.mkdir(parents=True, exist_ok=True)
    venv_cache.mkdir(parents=True, exist_ok=True)
    _restore_venv(sess, image, venv_dir, venv_cache)

    sess.helpers_dir.mkdir(parents=True, exist_ok=True)
    helper = sess.helpers_dir / "venv-snapshot.sh"
    helper.write_text(VENV_SNAPSHOT_SH)
    helper.chmod(0o755)

    prepare_sshd(sess, port, setenv)

    binds = []
    for rel in ISOLATION_BINDS:
        binds += ["-B", str(home / rel)]
    binds += ["-B", "/tmp"]  # node-local fast SSD — per-job, auto-cleaned by SLURM
    binds += ["-B", f"{sess.workdir}:/workdir"]
    # Per-command log files (.out/.err/.exit/.pid) live on the shared FS so the host-side `cs logs` can read them. Inside the container $HOME is an ephemeral per-session mount, so the host logs path is not reachable by its absolute name — bind it at the stable container path run.py writes to.
    binds += ["-B", f"{sess.logdir}:/cs-logs"]
    # Hibernation binds: live venv (node-local), snapshot cache (the helper publishes there via temp + atomic rename), and the helper (read-only). The cache path derives from config.json's project_id, which is not agent-writable (the session dir is never bound into the container).
    binds += ["-B", f"{venv_dir}:/cs-venv"]
    binds += ["-B", f"{venv_cache}:/cs-venv-cache"]
    binds += ["-B", f"{sess.helpers_dir}:/cs-helpers:ro"]
    binds += ["-B", f"{sess.sshd_dir}:/sshd"]

    envs = []
    for k, v in setenv.items():
        if v:
            envs += ["--env", f"{k}={v}"]

    # Launch sshd directly via apptainer exec (foreground container).
    # --containall: no host $HOME, no host /tmp, empty env by default. Binds & env vars above are the only holes punched into that isolation.
    # --writable-tmpfs gives a small scratch layer (sshd privsep dirs, auto-created bind destinations); its ~16 MB cap is fine because the venv lives on a plain bind, not in the writable layer.
    # We deliberately do NOT use `apptainer instance start`: the background sinit daemon can get reaped shortly after sbatch launches the step (SLURM proctrack/cgroup + PrologFlags=Contain interaction), which silently kills sshd with SIGTERM. Running sshd directly under `apptainer exec` ties the container's lifetime to sshd's — no daemon for SLURM to reap.
    # -e: sshd log output goes to stderr (→ the runner log) instead of syslog; the image has no syslog daemon, so without -e pre-auth errors are silently dropped.
    sshd_proc = subprocess.Popen(
        ["apptainer", "exec", "--nv", "--containall", "--writable-tmpfs",
         "--home", f"{sess.home_dir}:{home}"]
        + binds + envs
        + [str(image), "/usr/sbin/sshd", "-f", "/sshd/sshd_config", "-D", "-e"]
    )
    procs.append(sshd_proc)

    if not wait_for_ssh_banner(port, 30, [sshd_proc]):
        raise SystemExit(f"fatal: sshd did not come up on port {port} within 30s")

    # Reverse tunnel: Unix socket on the login node's shared FS -> local sshd port. The internal key's authorized_keys entry is forwarding-only (command="/bin/false"), so this -N session is all it can do.
    internal_key = sess.base / "cluster-internal"

    def unlink_socket():
        # The socket file is on the shared FS — unlink it locally (an `ssh … rm` would need an internal-key command session, which PAM may deny; the -N forwarding is exempt).
        sess.socket.unlink(missing_ok=True)

    # A socket file left by a previous activation that died without cleanup (node failure, OOM-kill of this runner) must go BEFORE the tunnel starts: the readiness poll below would otherwise see it and declare the session active while the forward is still being set up — or has failed, since the login node's sshd refuses to bind over an existing file unless its own sshd_config sets StreamLocalBindUnlink (a client-side `-o StreamLocalBindUnlink` does not apply to -R sockets; the server creates them).
    unlink_socket()
    cleanups.append(unlink_socket)
    tunnel_proc = subprocess.Popen([
        "ssh", "-o", "ControlMaster=no",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-i", str(internal_key),
        "-N", "-R", f"{sess.socket}:127.0.0.1:{port}",
        login,
    ])
    procs.append(tunnel_proc)

    # Wait until the -R socket actually exists before declaring the session active — status=active must mean "reachable". The socket lives in the session dir on the shared FS, so poll it locally — NOT via ssh (login-node PAM may deny internal-key command sessions while the session-less -N tunnel is unaffected).
    deadline = time.time() + 30
    while time.time() < deadline:
        if sess.socket.exists():
            break
        if tunnel_proc.poll() is not None:
            break
        time.sleep(0.5)
    if not sess.socket.exists():
        raise SystemExit(f"fatal: reverse-tunnel socket {sess.socket} did not appear within 30s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="absolute remote_base on the cluster")
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--login", required=True, help="login-node ssh alias for the reverse tunnel, resolvable from compute nodes")
    ap.add_argument("--backend", help=argparse.SUPPRESS)  # accepted and ignored: a 0.4 client still passes `--backend slurm`, and must keep activating against this runner while the client is being upgraded
    args = ap.parse_args()

    sess = Session(Path(args.base), args.session_id)
    sess.workdir.mkdir(parents=True, exist_ok=True)
    sess.logdir.mkdir(parents=True, exist_ok=True)
    cfg = sess.config()
    idle_timeout_minutes = int(cfg.get("idle_timeout_minutes") or 20)

    # Convert TERM/INT/HUP (scancel, deactivate, wall-clock kill) into SystemExit so the finally-block cleanup always runs. The flag lets the final status write distinguish a deliberate cancel while still activating (→ inactive) from an activation that died on its own (→ failed).
    signalled = {"yes": False}

    def _on_signal(signum, frame):
        signalled["yes"] = True
        sys.exit(128 + signum)

    for signo in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signo, _on_signal)

    procs = []
    cleanups = []
    try:
        port = pick_port()
        activate(sess, cfg, port, args.login, procs, cleanups)

        # last_activity_at is stamped fresh in the same write that flips status to active: a previous activation's heartbeat would otherwise linger in config.json and make seconds_until_idle_deactivate read 0 until the monitor's first ~60s persist.
        now = utcnow()
        sess.config_update(
            status="active",
            job_id=os.environ.get("SLURM_JOB_ID", ""),
            node=socket.gethostname(),
            sshd_port=port,
            gpu=detect_gpu_models() or None,
            last_activated_at=now,
            last_activity_at=now,
        )
        reason = monitor_loop(sess, procs, port, idle_timeout_minutes)
        log(f"shutting down: {reason}")
    finally:
        for p in procs:
            try:
                p.terminate()
            except OSError:
                pass
        for p in procs:
            try:
                p.wait(timeout=15)
            except Exception:
                try:
                    p.kill()
                except OSError:
                    pass
        for fn in cleanups:
            try:
                fn()
            except Exception as exc:
                log(f"cleanup failed (ignored): {exc}")
        # A run that dies before flipping to active never finished activating — record `failed` (not `inactive`) so clients can tell a broken activation from a clean deactivate (a signalled exit IS a deliberate cancel, even while pending). `show` surfaces this log's tail for failed sessions.
        # Compare-and-set on the job id: `cs deactivate` waits only briefly for this teardown, and a `cs activate` issued right after it has already re-submitted the session — its pending record (new job_id) must not be overwritten with `inactive`, which used to strand the new allocation unwatched. An empty/absent job_id is accepted too: the client stamps it only after sbatch returns, so a very early failure may see it unset.
        try:
            own_job = os.environ.get("SLURM_JOB_ID", "")
            status = "failed" if (sess.config().get("status") == "pending" and not signalled["yes"]) else "inactive"
            if not sess.config_update({"job_id": ("", None, own_job)}, status=status, last_deactivated_at=utcnow()):
                log(f"final status write skipped: the session was re-activated as job {sess.config().get('job_id')}")
        except Exception as exc:
            log(f"final status write failed: {exc}")


if __name__ == "__main__":
    main()
