#!/usr/bin/env python3
"""compute-sessions runner — the sbatch job script that activates one session on a compute node. Uploaded to <remote_base>/runner.py at cluster setup; stdlib-only (python3.8+).

Starts an Apptainer container running sshd, opens a reverse-SSH tunnel back to the login node (Unix socket in the session dir on the shared FS), and restores/archives the project venv via tar+zstd snapshots — the live venv stays on node-local /tmp, because a CUDA torch venv is ~4.7 GB / ~22k tiny files and loose venv files are murder on a parallel FS. After activation it idle-monitors the session (`.pid` heartbeats of tracked commands, and the `.activity` sentinel touched by host-side file tools and command launches) and writes the final status on the way out.
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


# Isolation policy — CONSTANT bind list. Edit here and re-upload runner.py to change what's visible inside sessions. Intentionally NOT driven by per-session config (which an agent can write), so changing the isolation surface requires SSH access to the cluster.
# Paths are relative to the invoking user's $HOME and bound at the same path inside the container. Entries whose source doesn't exist are created first so the caches actually persist across sessions.
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
# stop_reason for a job the scheduler ended at its wall-clock limit; the client names the partition.
TIME_LIMIT_REASON = "the partition's time limit"
# A `cs run --deactivate-after` command ends the session once every `.exit` is this old (and nothing else runs): a command's venv-snapshot epilogue registers within this window, and a streaming client reads the end first.
RELEASE_SETTLE_SECONDS = 20
# A node whose GPU fails the start-up check is left out of new activations this long (read by the login-node helper).
BAD_NODE_SECONDS = 24 * 3600

# Start-up GPU check, run with the job's CUDA_VISIBLE_DEVICES before the container starts: initialise CUDA through the driver library (stdlib ctypes — no torch on the node), then per device a context, an allocation, a memset and a synchronize. Catches GPUs that nvidia-smi lists but that fail every kernel ("busy or unavailable", "no CUDA-capable device"). Exit 0 = fine or nothing to check, 3 = CUDA reported a failure (the reason on stdout); anything else is a fault of the check itself, which never blames the node.
GPU_CHECK = r"""
import ctypes, sys
try:
    cu = ctypes.CDLL("libcuda.so.1")
except OSError:
    sys.exit(0)
def check(what, rc):
    if rc:
        name = ctypes.c_char_p()
        cu.cuGetErrorName(rc, ctypes.byref(name))
        print("%s failed: %s" % (what, name.value.decode() if name.value else rc))
        sys.exit(3)
check("cuInit", cu.cuInit(0))
n = ctypes.c_int()
check("cuDeviceGetCount", cu.cuDeviceGetCount(ctypes.byref(n)))
if n.value < 1:
    print("no CUDA device is visible")
    sys.exit(3)
for i in range(n.value):
    dev, ctx, ptr = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_uint64()
    check("cuDeviceGet(%d)" % i, cu.cuDeviceGet(ctypes.byref(dev), i))
    check("cuDevicePrimaryCtxRetain(%d)" % i, cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev))
    check("cuCtxSetCurrent(%d)" % i, cu.cuCtxSetCurrent(ctx))
    check("cuMemAlloc(%d)" % i, cu.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(1 << 20)))
    check("cuMemsetD8(%d)" % i, cu.cuMemsetD8_v2(ptr, ctypes.c_ubyte(1), ctypes.c_size_t(1 << 20)))
    check("cuCtxSynchronize(%d)" % i, cu.cuCtxSynchronize())
    cu.cuMemFree_v2(ptr)
    cu.cuDevicePrimaryCtxRelease_v2(dev)
"""


# Snapshot helper — runs INSIDE the container, spawned detached by the `cs run` launcher's epilogue. Lives in the session dir (bound read-only at /cs-helpers) so iterating on it only needs a runner re-upload, not an image rebuild.
# State-based: snapshots whenever the live freeze hash differs from the marker, which is written only on publish/adopt (here) or restore (at activation) — a snapshot that dies midway simply retries on the next command, and any install path is caught. The no-change exit is near-instant.
# Parallel sessions of one project share the cache, so a state another session already published is adopted instead of archived again, and a cross-session claim dir (heartbeat-touched; taken over once stale, since a killed owner never runs its EXIT trap) lets only one session archive a given state.
# Registers as a tracked cmd_venvsnap_* command (.start/.pid heartbeat/.exit in /cs-logs): visible in list_commands, killable, and the heartbeat keeps the idle monitor from reaping the session mid-snapshot.
# Kept as bash (not python): it runs inside the container image, which deliberately ships no system python — only uv-managed project interpreters.
VENV_SNAPSHOT_SH = r"""#!/bin/bash
set -u
VENV=/cs-venv
CACHE=/cs-venv-cache
LOGS=/cs-logs
WORKDIR=/workdir
LOCKDIR=/tmp/cs-venvsnap.lock
MARKER=$VENV/.cs-freeze-hash
command -v uv >/dev/null 2>&1 || exit 0
[ -d "$VENV" ] && [ -d "$CACHE" ] || exit 0
# No venv yet: `uv pip freeze` would fall back to some other interpreter and publish a near-empty archive.
[ -f "$VENV/venv/pyvenv.cfg" ] || exit 0
# Hash = installed packages + interpreter version (pyvenv.cfg) — a python bump without package changes must also re-snapshot.
freeze_hash() { { uv pip freeze 2>/dev/null; grep -i '^version' "$VENV/venv/pyvenv.cfg" 2>/dev/null; } | sha256sum | cut -c1-16; }
# An empty venv is nothing worth caching — and skipping it means a wiped venv can never tombstone over good snapshots.
[ -n "$(uv pip freeze 2>/dev/null | head -c 1)" ] || exit 0
cur=$(freeze_hash)
[ "$cur" = "$(cat "$MARKER" 2>/dev/null)" ] && exit 0
tag=nolock
if [ -f "$WORKDIR/uv.lock" ]; then tag=$(sha256sum "$WORKDIR/uv.lock" | cut -c1-16); fi
# This exact state is already published: adopt it. The touch makes restore (newest first) and the keep-3 cleanup treat it as the newest.
adopt() { local s; s=$(ls -t "$CACHE/$tag.$cur".*.tar.zst 2>/dev/null | head -n 1); [ -n "$s" ] || return 1; touch -c "$s"; echo "$cur" > "$MARKER"; }
adopt && exit 0
# One snapshot at a time in this session; the next command epilogue retries. A SIGKILLed owner (`cs kill -s KILL`) never runs its EXIT trap, so a lock whose owner is gone is taken over.
lock() { mkdir "$LOCKDIR" 2>/dev/null && echo $$ > "$LOCKDIR/pid"; }
if ! lock; then
  kill -0 "$(cat "$LOCKDIR/pid" 2>/dev/null)" 2>/dev/null && exit 0
  rm -rf "$LOCKDIR"; lock || exit 0
fi
# Signal traps so a kill still runs the EXIT trap. The minimal EXIT trap covers the setup window; the full one replaces it below.
trap 'exit 143' TERM; trap 'exit 130' INT; trap 'exit 129' HUP
CLAIM="$CACHE/.claim.$tag.$cur"
owned=
trap 'rm -rf "$LOCKDIR"; [ -z "$owned" ] || rm -rf "$CLAIM"' EXIT
claim() { mkdir "$CLAIM" 2>/dev/null || return 1; owned=1; echo "$(hostname) $$" > "$CLAIM/owner" 2>/dev/null; return 0; }
if ! claim; then
  # A claim touched within 2 min has a live owner; the next command's epilogue adopts its snapshot. Two sessions taking over the same stale claim at once can both archive — harmless, just one extra archive.
  [ -n "$(find "$CLAIM" -maxdepth 0 -mmin -2 2>/dev/null)" ] && exit 0
  rm -rf "$CLAIM"; claim || exit 0
fi
# Another session may have published between the first check and the claim.
adopt && exit 0
cid="cmd_venvsnap_$(date +%s)_$$"
b="$LOGS/$cid"
: > "$b.start"; : > "$b.out"; : > "$b.err"; echo $$ > "$b.pid"
( while kill -0 $$ 2>/dev/null; do touch -c "$b.pid" "$CLAIM"; sleep 10; done ) &
trap 'ec=$?; echo "$ec" > "$b.exit"; rm -rf "$LOCKDIR" "$CLAIM"' EXIT
{
  tmp="$CACHE/.snap.$cid.tmp"
  nice -n 19 tar -C "$VENV" -I 'zstd -T0 -3' -cf "$tmp" . || { rm -f "$tmp"; exit 1; }
  # Discard if the venv changed mid-tar; the marker stays stale, so the next command epilogue retries cleanly.
  [ "$(freeze_hash)" = "$cur" ] || { rm -f "$tmp"; echo "venv changed mid-snapshot; discarded"; exit 0; }
  mv -f "$tmp" "$CACHE/$tag.$cur.$(date +%s).tar.zst"
  echo "$cur" > "$MARKER"
  echo "published snapshot tagged $tag"
  # Keep the newest 3 snapshots (each ~GBs; parallel FS often runs near-full) and sweep orphaned publish temps (dotfiles, so the glob above never matches them; age-gated so a live temp survives).
  ls -t "$CACHE"/*.tar.zst 2>/dev/null | tail -n +4 | xargs -r rm -f
  find "$CACHE" -maxdepth 1 -name '.snap.*.tmp' -mmin +180 -delete 2>/dev/null || true
} > "$b.out" 2> "$b.err"
"""


def utcnow():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg):
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg.rstrip() + "\n")
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
        # Commands queued while the session was pending (launch scripts written by the login-node helper).
        self.queue = self.dir / "queue"
        # The runner's own login to the container sshd, for launching queued commands. The session dir is never bound into the container.
        self.runner_key = self.dir / ".runner_key"

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


def scheduler_reason(sess):
    """Why the scheduler signalled this job: slurmstepd writes "CANCELLED … DUE TO TIME LIMIT" (or PREEMPTION) into the job's output file before it signals."""
    path = sess.logdir / f"runner-{os.environ.get('SLURM_JOB_ID', '')}.out"
    try:
        with path.open("rb") as fh:
            fh.seek(max(0, path.stat().st_size - 65536))
            tail = fh.read().decode(errors="replace")
    except OSError:
        tail = ""
    if "DUE TO TIME LIMIT" in tail:
        return TIME_LIMIT_REASON
    if "DUE TO PREEMPTION" in tail:
        return "preempted by the scheduler"
    return "signalled by the scheduler (time limit, preemption or scancel)"


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
    # Newer drivers print "CUDA UMD Version" (user-mode driver) where older ones print "CUDA Version".
    m = re.search(r"CUDA (?:UMD )?Version:\s*([0-9.]+)", out)
    return m.group(1) if m else ""


def detect_gpu_models():
    """Human-readable model(s) of the GPU(s) this job actually got (e.g. "NVIDIA A100 80GB PCIe", "2x NVIDIA H100"), from `nvidia-smi -L`. The job's cgroup normally scopes this to the allocated devices. The requested gpu_type is only a constraint filter, so recording the real allocation lets show/list report actual hardware. Empty string when no GPU/driver.

    MIG nodes break the cgroup assumption: every parent GPU stays visible and the actual allocation is a MIG slice listed underneath, so a 1-slice job would read as "8x NVIDIA A100 80GB PCIe". MIG device lines therefore win over parent lines, filtered to CUDA_VISIBLE_DEVICES when the scheduler set it to MIG UUIDs."""
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
    for k in (sess.runner_key, sess.runner_key.with_name(sess.runner_key.name + ".pub")):
        k.unlink(missing_ok=True)
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-f", sess.runner_key, "-N", "", "-q", "-C", "cs-runner"], check=True)
    with auth.open("a") as fh:
        fh.write('\nfrom="127.0.0.1" ' + sess.runner_key.with_name(sess.runner_key.name + ".pub").read_text().strip() + "\n")
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
        # Every cs stream into the session is a session channel on one shared client connection, and parallel streams of one session are normal.
        "MaxSessions 100",
        # Drop a client that stopped answering (laptop asleep) instead of keeping its dead connection around.
        "ClientAliveInterval 15",
        "ClientAliveCountMax 3",
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


def command_live(sess):
    """Whether a tracked command runs: a `.pid` without its `.exit` whose mtime is fresh.

    Liveness is read from the .pid file's MTIME, not `kill -0` on its contents: the container has a private PID namespace, so the recorded PID is meaningless to this host-side process. run.py's inner bash refreshes the mtime every 10s from *inside* the container and stops within 10s of dying by any means — mtime is namespace-independent, so the host-side read is correct."""
    now = time.time()
    try:
        pidfiles = list(sess.logdir.glob("*.pid"))
    except OSError:
        return False
    for pidfile in pidfiles:
        # Command ids contain no dots, so with_suffix cleanly swaps .pid → .exit.
        if pidfile.with_suffix(".exit").exists():
            continue
        try:
            if now - pidfile.stat().st_mtime < PID_FRESH_SECONDS:
                return True
        except OSError:
            continue
    return False


def recent_file_ops(sess):
    """Whether the host-side cs client touched `.activity` lately (workdir file ops and container commands, which are otherwise invisible here)."""
    try:
        return time.time() - sess.activity.stat().st_mtime < PID_FRESH_SECONDS
    except OSError:
        return False


def release_reason(sess):
    """Why the session ends now because of `cs run --deactivate-after`: every flagged command of this activation has ended (`.exit` settled, or a heartbeat gone stale), and no other command runs or waits in the queue. None otherwise."""
    try:
        flagged = [p.name[:-len(".release")] for p in sess.logdir.glob("*.release")]
    except OSError:
        return None
    if not flagged or any(sess.queue.glob("cmd_*")):
        return None
    now = time.time()
    for cid in flagged:
        if not (sess.logdir / f"{cid}.exit").exists():
            try:
                if now - (sess.logdir / f"{cid}.pid").stat().st_mtime < PID_FRESH_SECONDS:
                    return None
            except OSError:
                pass
    # Any command that just ended may still start its venv-snapshot epilogue, and a client may still be reading its end.
    for exitfile in sess.logdir.glob("*.exit"):
        try:
            if now - exitfile.stat().st_mtime < RELEASE_SETTLE_SECONDS:
                return None
        except OSError:
            continue
    for pidfile in sess.logdir.glob("*.pid"):
        try:
            if not pidfile.with_suffix(".exit").exists() and now - pidfile.stat().st_mtime < PID_FRESH_SECONDS:
                return None
        except OSError:
            continue
    return f"{', '.join(sorted(flagged))} ended (cs run --deactivate-after)"


def container_login(sess, port):
    """argv of a login shell in the session container that reads its script from stdin, as the runner (its own key, from the node itself)."""
    import pwd
    return ["ssh", "-F", "/dev/null", "-i", str(sess.runner_key), "-p", str(port), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ControlMaster=no",
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", f"{pwd.getpwuid(os.getuid()).pw_name}@127.0.0.1", "bash -l -s"]


def launch_queued(sess, port):
    """Launch the commands queued while the session was pending, oldest first, through the runner's own login to the container sshd (the same launch script a client sends). A launch that fails is recorded as the command's exit (125) and stderr, so the client sees it."""
    try:
        entries = sorted((p for p in sess.queue.iterdir() if re.match(r"^cmd_[A-Za-z0-9_]+$", p.name)), key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    argv = container_login(sess, port)
    for entry in entries:
        cid = entry.name
        claimed = entry.with_name(f".launching.{cid}")
        try:
            os.rename(entry, claimed)  # a concurrent `cs kill`/`cs deactivate` dequeue renames too: one of the two wins
        except OSError:
            continue
        problem = "the session ended while the command was being started"
        try:
            p = subprocess.run(argv, input=claimed.read_bytes(), capture_output=True, timeout=120)
            out = p.stdout.decode(errors="replace").splitlines()
            problem = None if "S" in out else (next((l[2:] for l in out if l.startswith("F ")), None) or p.stderr.decode(errors="replace").strip() or f"ssh exit {p.returncode}")
        except Exception as exc:
            problem = str(exc)
        finally:
            # Also on the way out (a signal mid-launch): a claimed command must end up started or failed, never stuck as queued.
            if problem:
                fail_queued(sess, cid, problem)
            else:
                log(f"queue: launched {cid}")
            claimed.unlink(missing_ok=True)


def fail_queued(sess, cid, problem):
    """Record a queued command that could not be started as exited 125 with the reason on stderr. The logs dir is writable from the container, so nothing here follows a symlink."""
    log(f"queue: could not launch {cid}: {problem}")
    for ext, text in ((".err", f"cs: the queued command could not be started: {problem}\n"), (".pid", ""), (".exit", "125\n")):
        try:
            fd = os.open(sess.logdir / f"{cid}{ext}", os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
            with os.fdopen(fd, "w") as fh:
                fh.write(text)
        except OSError:
            pass
    try:
        (sess.logdir / f"{cid}.queued").unlink()
    except OSError:
        pass


def gpu_check():
    """None when the allocated GPUs work (or there is nothing to check, or the check itself broke — logged), else why not. A hung check counts as a failed GPU; it is killed without waiting for it (a process stuck in the driver may never exit)."""
    p = subprocess.Popen([sys.executable, "-I", "-c", GPU_CHECK], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        p.kill()
        return "the CUDA check did not finish within 90 s"
    if p.returncode == 3:
        return out.strip() or "CUDA reported a failure"
    if p.returncode != 0:
        log(f"gpu check: could not run (exit {p.returncode}): {(err.strip().splitlines() or [''])[-1]}")
    return None


def record_bad_node(sess, node, reason):
    """Remember a node whose GPU failed the check: the login-node helper leaves it out of new activations for BAD_NODE_SECONDS. Entries older than that are dropped on the way."""
    path = sess.base / "bad-nodes.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    now = time.time()
    data = {n: e for n, e in data.items() if isinstance(e, dict) and now - float(e.get("at") or 0) < BAD_NODE_SECONDS}
    data[node] = {"at": now, "reason": reason, "session": sess.id}
    tmp = path.with_name(f"bad-nodes.json.tmp.{socket.gethostname()}.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def monitor_loop(sess, procs, port, idle_timeout_minutes):
    """Block until a child process dies or the session idles out. Returns a human-readable reason. Polls every ~2s for child death (a dead sshd/tunnel must not hang the job) and every IDLE_POLL_SECONDS for activity. Persists the last-active heartbeat to config.json at most once per HEARTBEAT_PERSIST_SECONDS so the host-side cs client can compute `seconds_until_idle_deactivate` without hammering the FS — and `busy` (a live command holds the idle timer) as soon as it changes, so the client shows no countdown that is not running."""
    timeout = idle_timeout_minutes * 60
    last_active = time.time()
    last_check = 0.0
    last_persisted = 0.0
    busy_persisted = None
    while True:
        time.sleep(2)
        for p in procs:
            if p.poll() is not None:
                return f"{'the container' if p.args[0] == 'apptainer' else 'the reverse tunnel'} exited (rc={p.returncode})"
        now = time.time()
        if now - last_check >= IDLE_POLL_SECONDS:
            last_check = now
            # Only a running command is "busy"; recent file ops also hold the timer. An open ssh connection does not: the client keeps one open for a minute after each use.
            busy = command_live(sess)
            if busy or recent_file_ops(sess):
                last_active = now
            if now - last_persisted >= HEARTBEAT_PERSIST_SECONDS or busy != busy_persisted:
                try:
                    sess.config_update(last_activity_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(last_active)), busy=busy)
                    busy_persisted = busy
                except Exception as exc:
                    log(f"heartbeat persist failed (ignored): {exc}")
                last_persisted = now
            if now - last_active >= timeout:
                return f"idle timeout ({idle_timeout_minutes} min without activity)"
            launch_queued(sess, port)
            released = release_reason(sess)
            if released:
                return released


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

    # A socket file left by a previous activation that died without cleanup must go BEFORE the tunnel starts: the readiness poll below would otherwise see it and declare the session active while the forward is still being set up — or has failed, since the login node's sshd refuses to bind over an existing file (a client-side StreamLocalBindUnlink does not apply to -R sockets; the server creates them).
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
    args = ap.parse_args()

    # The job's wall-clock limit counts from here; recorded with the flip to active.
    job_started_at = utcnow()
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
    reason = "the runner stopped unexpectedly"
    try:
        if int((cfg.get("resources") or {}).get("gpus") or 0) > 0:
            problem = gpu_check()
            if problem:
                node = os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0]
                try:
                    record_bad_node(sess, node, problem)
                except Exception as exc:
                    log(f"could not record the bad node: {exc}")
                raise SystemExit(f"fatal: the GPU check failed on {node} ({problem}) — new activations avoid this node for 24 h; `cs activate` retries elsewhere")
            log("gpu check: passed")
        port = pick_port()
        activate(sess, cfg, port, args.login, procs, cleanups)
        # `.release` marks of earlier activations' commands must not end this one; a launch an earlier runner did not finish (hard kill) failed.
        for stale in sess.logdir.glob("*.release"):
            stale.unlink(missing_ok=True)
        for stale in sess.queue.glob(".launching.cmd_*"):
            fail_queued(sess, stale.name[len(".launching."):], "an earlier activation ended while the command was being started")
            stale.unlink(missing_ok=True)

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
            job_started_at=job_started_at,
        )
        launch_queued(sess, port)
        reason = monitor_loop(sess, procs, port, idle_timeout_minutes)
        log(f"shutting down: {reason}")
    except SystemExit as exc:
        if not signalled["yes"] and isinstance(exc.code, str):
            reason = exc.code[len("fatal: "):] if exc.code.startswith("fatal: ") else exc.code
        raise
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        raise
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
        # Compare-and-set on the job id: `cs deactivate` waits only briefly for this teardown, and a `cs activate` issued right after may already have re-submitted the session — its pending record (new job_id) must not be overwritten with `inactive`. An empty/absent job_id is accepted too: the client stamps it only after sbatch returns.
        # stop_reason tells `cs show` and dead-command messages why the session ended. A signal is the scheduler's: `cs deactivate` records itself before it cancels, so its reason is kept; otherwise the job log names the time limit or a preemption, else it was a manual scancel or similar.
        try:
            own_job = os.environ.get("SLURM_JOB_ID", "")
            cur = sess.config()
            status = "failed" if (cur.get("status") == "pending" and not signalled["yes"]) else "inactive"
            if signalled["yes"]:
                reason = cur.get("stop_reason") or scheduler_reason(sess)
            if not sess.config_update({"job_id": ("", None, own_job)}, status=status, last_deactivated_at=utcnow(), stop_reason=reason):
                log(f"final status write skipped: the session was re-activated as job {sess.config().get('job_id')}")
        except Exception as exc:
            log(f"final status write failed: {exc}")


if __name__ == "__main__":
    main()
