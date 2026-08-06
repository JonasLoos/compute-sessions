"""slurm backend for the compute-sessions runner (imported by runner.py, which sbatch executes as the job script on a compute node).

Starts an Apptainer container running sshd, opens a reverse-SSH tunnel back to the login node (Unix socket in the session dir on the shared FS), and restores/archives the project venv via tar+zstd snapshots — the live venv stays on node-local /tmp, because a CUDA torch venv is ~4.7 GB / ~22k tiny files and loose venv files are murder on a parallel FS.
"""
import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path

from runner import ISOLATION_BINDS, detect_cuda_version, ensure_bind_sources, log, prepare_sshd, wait_for_ssh_banner


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


def activate(sess, cfg, port, args, procs, cleanups):
    """Start the Apptainer container (sshd in the foreground) and the reverse tunnel, registering them in `procs`/`cleanups` as they come up — so runner.py's finally-block can tear down partial activations too. Returns the job handle."""
    home = Path.home()
    image = Path(cfg.get("image", ""))
    if not image.is_absolute():
        image = sess.base / image

    ensure_bind_sources()
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

    prepare_sshd(sess, port, "127.0.0.1", setenv)

    binds = []
    for rel in ISOLATION_BINDS:
        src = home / rel
        if src.exists():
            binds += ["-B", str(src)]
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

    # Reverse tunnel: Unix socket on the login node's shared FS -> local sshd port. StreamLocalBindUnlink removes a stale socket file if present. The internal key's authorized_keys entry is forwarding-only (command="/bin/false"), so this -N session is all it can do.
    internal_key = sess.base / "cluster-internal"

    def unlink_socket():
        # The socket file is on the shared FS — unlink it locally (an `ssh … rm` would need an internal-key command session, which PAM may deny; the -N forwarding is exempt).
        sess.socket.unlink(missing_ok=True)

    cleanups.append(unlink_socket)
    tunnel_proc = subprocess.Popen([
        "ssh", "-o", "ControlMaster=no",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "StreamLocalBindUnlink=yes",
        "-o", "ServerAliveInterval=30",
        "-i", str(internal_key),
        "-N", "-R", f"{sess.socket}:127.0.0.1:{port}",
        args.login,
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

    return os.environ.get("SLURM_JOB_ID", "")
