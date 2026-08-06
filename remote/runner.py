#!/usr/bin/env python3
"""compute-sessions runner — activates one session on a compute source. Uploaded to <remote_base>/ at source setup together with its backend modules (runner_slurm.py / runner_docker.py); stdlib-only (python3.8+).

This file is the entry point plus everything backend-independent: session paths + atomic config.json access, sshd preparation, readiness/liveness probes, the idle monitor, and the activate/cleanup scaffolding. The backend module contributes only `activate(sess, cfg, port, args, procs, cleanups) -> job_id`:
  runner_slurm.py   sbatch job script on a compute node — Apptainer + reverse-SSH tunnel + venv snapshots.
  runner_docker.py  detached process on an always-on ssh host — Docker container with sshd published on 127.0.0.1.
  runner_vast.py    NOT dispatched from here — it is its own entry point, baked into the vast session image (remote/Dockerfile.vast) as the container entrypoint; it imports the shared helpers from this file.

After activation, the runner idle-monitors the session (established sshd connections, `.pid` heartbeats of tracked commands, and the `.activity` sentinel touched by host-side file tools) and writes the final status on the way out.
"""
import argparse
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
# Isolation policy — CONSTANT bind list, shared by both backends. Edit here and
# re-upload runner.py to change what's visible inside sessions. Intentionally
# NOT driven by per-session config (which an agent can write) so changing the
# isolation surface requires SSH access to the source.
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

    def config_update(self, **fields):
        """Read-modify-write with an atomic replace. config.json has concurrent writers (this runner + the host-side cs client, which also writes via temp + rename), so a reader can never observe a half-written file; last-writer-wins on races is acceptable for these heartbeat-ish fields."""
        d = self.config()
        d.update(fields)
        tmp = self.config_path.with_name(f"config.json.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(d, indent=2))
        os.replace(tmp, self.config_path)


def pick_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def detect_cuda_version():
    """Host driver's CUDA version (e.g. "12.9"), so project code can pick a matching torch wheel instead of guessing. Parsed from the nvidia-smi header (`--query-gpu=cuda_version` isn't universally supported). Empty string on CPU-only hosts."""
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return ""
    m = re.search(r"CUDA Version: ([0-9.]+)", out)
    return m.group(1) if m else ""


def detect_gpu_models():
    """Human-readable model(s) of the GPU(s) this activation actually got (e.g. "NVIDIA A100 80GB PCIe", "2x NVIDIA H100"), from `nvidia-smi -L`. On slurm the job's cgroup normally scopes this to the allocated devices; on a docker host it's the whole machine's. The requested gpu_type is only a constraint filter, so recording the real allocation lets show/list report actual hardware. Empty string when no GPU/driver.

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


def prepare_sshd(sess, port, listen_address, setenv, authorized_keys_text=None, permit_root=False, conf_dir="/sshd"):
    """Session-local sshd config + host key + authorized_keys, written under sess.sshd_dir. `conf_dir` is that directory's path AS SEEN BY the sshd process — "/sshd" when the dir is bind-mounted into a container (slurm/docker), the real path when sshd runs beside us (vast)."""
    sess.sshd_dir.mkdir(parents=True, exist_ok=True)
    host_key = sess.sshd_dir / "host_key"
    if not host_key.is_file():
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-f", host_key, "-N", "", "-q"], check=True)

    auth = sess.sshd_dir / "authorized_keys"
    if authorized_keys_text is not None:
        # Keys handed in explicitly (vast: the creating machine's public keys, delivered via env var — a rented box has no pre-existing trust relationship to inherit from).
        auth.write_text(authorized_keys_text.rstrip() + "\n")
    else:
        # The in-container sshd accepts any key that can already log into the source host — i.e. the user's own ~/.ssh/authorized_keys. That means the same laptop key used to reach the source authenticates into the container too, without shipping an extra keypair.
        src_auth = Path.home() / ".ssh" / "authorized_keys"
        if not src_auth.is_file():
            raise SystemExit(f"fatal: {src_auth} does not exist on the source — add your machine's public key there before creating sessions.")
        shutil.copyfile(src_auth, auth)
    auth.chmod(0o600)

    lines = [
        f"Port {port}",
        f"ListenAddress {listen_address}",
        f"HostKey {conf_dir}/host_key",
        f"PidFile {conf_dir}/sshd.pid",
        f"AuthorizedKeysFile {conf_dir}/authorized_keys",
        "PasswordAuthentication no",
        "PubkeyAuthentication yes",
        "UsePAM no",
        f"PermitRootLogin {'prohibit-password' if permit_root else 'no'}",
        "StrictModes no",
        "Subsystem sftp internal-sftp",
    ]
    # OpenSSH scrubs the environment on each new session, so env vars given to the sshd process are only visible to sshd itself — not to the shells spawned for incoming `run` connections. SetEnv makes them reach those shells. All vars MUST share ONE SetEnv line: sshd applies the first-obtained-value rule per keyword, so multiple `SetEnv` lines keep only the first and silently drop the rest (verified with `sshd -T` on OpenSSH 9.6). The forwarded values are space-free, so a single space-separated line is safe.
    pairs = " ".join(f"{k}={v}" for k, v in setenv.items() if v)
    if pairs:
        lines.append(f"SetEnv {pairs}")
    (sess.sshd_dir / "sshd_config").write_text("\n".join(lines) + "\n")


def wait_for_ssh_banner(port, timeout, procs):
    """Wait until something on 127.0.0.1:<port> answers with an SSH banner.

    A plain connect-success test is not enough for docker: docker-proxy starts listening the moment the container starts, before sshd inside is ready, and would accept-then-reset. Reading the banner proves sshd itself is up. Breaks out early if any of `procs` died (real crashes fail fast instead of burning the window)."""
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
    """Count established TCP connections involving `port`, parsed from /proc/net/tcp{,6} (no ss/netstat dependency). For slurm this sees the in-container sshd directly (apptainer shares the host netns); for docker it sees the host-side leg through docker-proxy."""
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


def monitor_loop(sess, procs, port, idle_timeout_minutes, deadline=None):
    """Block until a child process dies, the session idles out, or `deadline` (epoch seconds; optional) passes. Returns a human-readable reason. The deadline is a hard age cap independent of activity — vast sessions use it to bound their worst-case rental cost. Polls every ~2s for child death (a dead sshd/tunnel must not hang the job) and every IDLE_POLL_SECONDS for activity — a brief connect/disconnect between coarser checks would be missed. Persists the last-active heartbeat to config.json at most once per HEARTBEAT_PERSIST_SECONDS so the host-side cs client can compute `seconds_until_idle_deactivate` without hammering the FS."""
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
        if deadline is not None and now >= deadline:
            return "hard session-age limit reached"
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


def ensure_bind_sources():
    """Pre-create cache dirs under $HOME so first-session binds actually persist. Without this, a fresh account has no ~/.cache/uv; the bind would be skipped/created-empty and uv would write to the ephemeral per-session home instead, losing its cache on every deactivation."""
    for rel in ISOLATION_BINDS:
        (Path.home() / rel).mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["slurm", "docker"])
    ap.add_argument("--base", required=True, help="absolute remote_base on this source")
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--login", default="", help="slurm only: login-node ssh alias for the reverse tunnel, resolvable from compute nodes")
    args = ap.parse_args()

    base = Path(args.base)
    sess = Session(base, args.session_id)
    sess.workdir.mkdir(parents=True, exist_ok=True)
    sess.logdir.mkdir(parents=True, exist_ok=True)
    cfg = sess.config()
    idle_timeout_minutes = int(cfg.get("idle_timeout_minutes") or 20)

    # The backend modules live next to this file in remote_base — but under sbatch this script runs as a spool-dir COPY, so its own directory is useless for imports; remote_base must be on sys.path explicitly.
    sys.path.insert(0, str(base))
    if args.backend == "slurm":
        from runner_slurm import activate
    else:
        from runner_docker import activate

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
        job_id = activate(sess, cfg, port, args, procs, cleanups)

        # last_activity_at is stamped fresh in the same write that flips status to active: a previous activation's heartbeat would otherwise linger in config.json and make seconds_until_idle_deactivate read 0 until the monitor's first ~60s persist.
        now = utcnow()
        sess.config_update(
            status="active",
            job_id=job_id,
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
        try:
            status = "failed" if (sess.config().get("status") == "pending" and not signalled["yes"]) else "inactive"
            sess.config_update(status=status, last_deactivated_at=utcnow())
        except Exception as exc:
            log(f"final status write failed: {exc}")


if __name__ == "__main__":
    main()
