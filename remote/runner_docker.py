"""docker backend for the compute-sessions runner (imported by runner.py, which is launched detached over ssh on an always-on host, e.g. a gaming PC's WSL).

Starts a Docker container running sshd with its port published on the host's 127.0.0.1 — the cs client forwards to it over the same ssh connection, so no reverse tunnel or internal key is needed. The venv lives in a persistent per-project directory (local disk, no snapshot machinery).
"""
import os
import shutil
import subprocess
from pathlib import Path

from runner import ISOLATION_BINDS, detect_cuda_version, ensure_bind_sources, prepare_sshd, wait_for_ssh_banner


def activate(sess, cfg, port, args, procs, cleanups):
    """Start the session's Docker container (sshd in the foreground, port published on the host's 127.0.0.1), registering it in `procs`/`cleanups` — so runner.py's finally-block can tear down partial activations too. Returns the job handle (this runner's pid)."""
    home = Path.home()
    image = cfg.get("image", "")

    ensure_bind_sources()
    sess.home_dir.mkdir(parents=True, exist_ok=True)

    # No snapshot machinery here: the host disk is persistent and local, so the venv just lives in a per-project directory bound at the same /cs-venv path the image's UV_PROJECT_ENVIRONMENT points to.
    venv_dir = sess.base / "venvs" / cfg.get("project_id", "unnamed")
    venv_dir.mkdir(parents=True, exist_ok=True)

    # No CUDA_VISIBLE_DEVICES: there is no scheduler slicing the machine — the session gets every GPU `--gpus all` exposes.
    setenv = {"CS_CUDA_VERSION": detect_cuda_version()}
    # sshd must listen on all container interfaces — connections arrive via docker's published-port proxy, not loopback.
    prepare_sshd(sess, port, "0.0.0.0", setenv)

    name = f"cs-{sess.id}"

    # A stale container from a crashed previous activation would collide on the name. The same force-remove doubles as this activation's cleanup (registered before launch so even a failed `docker run` gets swept).
    def remove_container():
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    remove_container()
    cleanups.append(remove_container)

    docker_args = [
        "docker", "run", "--rm", "--name", name,
        # Run as the invoking user so workdir/log/venv writes stay owned (and deletable/rsyncable) by them. The image bakes a matching passwd entry at build time (see remote/Dockerfile build args) — sshd needs getpwnam to resolve the login user.
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-p", f"127.0.0.1:{port}:{port}",
        "-v", f"{sess.workdir}:/workdir",
        "-v", f"{sess.logdir}:/cs-logs",
        "-v", f"{venv_dir}:/cs-venv",
        "-v", f"{sess.sshd_dir}:/sshd",
        # Ephemeral per-session home at the user's real home path, so the cache binds below (same absolute paths) land correctly inside it.
        "-v", f"{sess.home_dir}:{home}",
    ]
    for rel in ISOLATION_BINDS:
        src = home / rel
        if src.exists():
            docker_args += ["-v", f"{src}:{src}"]
    if shutil.which("nvidia-smi"):
        docker_args += ["--gpus", "all"]
    docker_args += [image, "/usr/sbin/sshd", "-f", "/sshd/sshd_config", "-D", "-e"]

    # docker run stays in the foreground with --sig-proxy (default): our TERM reaches sshd as container PID 1, and --rm removes the container when it exits.
    proc = subprocess.Popen(docker_args)
    procs.append(proc)

    if not wait_for_ssh_banner(port, 60, [proc]):
        raise SystemExit(f"fatal: container sshd did not come up on 127.0.0.1:{port} within 60s")

    return str(os.getpid())
