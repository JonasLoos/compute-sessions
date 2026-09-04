from __future__ import annotations

import fcntl
import os
import re
import secrets
import shlex
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from compute_sessions.config import Config
from compute_sessions.errors import RemoteError
from compute_sessions.paths import RemotePaths


# Wall-clock ceilings for the local ssh subprocesses. These are safety nets against a wedged ControlMaster, a stale NFS mount, or a hung host — NOT normal operating limits (every call below completes in well under a second on the happy path). Without them a single blackholed connection hangs the whole cs invocation indefinitely, blinding the caller in the meantime.
_CONTROL_TIMEOUT = 60.0      # ssh control-plane ops: -O check/forward/cancel/exit, master open
_HOST_CMD_TIMEOUT = 180.0    # one login-node shell command (cat config, squeue, sbatch …)
# In-container execs are short on the happy path: the `run` launcher returns as soon as the command is detached, and the kill scripts poll for at most ~3s.
_CONTAINER_EXEC_TIMEOUT = 60.0
# rsync moves real data (datasets, weights) and can legitimately run for many minutes, so we deliberately do NOT bound it by wall clock. Instead rsync's own --timeout aborts a transfer that stalls (no bytes moved) for this long — catching a wedged link without killing a slow-but-progressing one.
_RSYNC_IO_TIMEOUT = 120


def _run_capture(
    args: list[str],
    *,
    timeout: float,
    input_text: str | None = None,
    best_effort: bool = False,
) -> "subprocess.CompletedProcess[str]":
    """`subprocess.run` with capture+text+errors=replace and a hard wall-clock timeout.

    A timed-out call is converted to a typed RemoteError (instead of a bare TimeoutExpired) so callers fail fast with a clear message. With `best_effort=True` a timeout is swallowed into a returncode-124 result instead — for teardown paths (control-master exit, tunnel cancel) that ignore the outcome and must never raise just because cleanup was slow.
    """
    try:
        return subprocess.run(
            args,
            input=input_text,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        if best_effort:
            return subprocess.CompletedProcess(
                args, 124, stdout="", stderr=f"timed out after {timeout:g}s"
            )
        # Trim the command in the message: the full ssh invocation (options + wrapped remote script) runs to many hundreds of characters of noise for the agent reading the error; the head identifies the call well enough.
        cmd_str = " ".join(shlex.quote(a) for a in args)
        if len(cmd_str) > 160:
            cmd_str = cmd_str[:160] + f"… [+{len(cmd_str) - 160} chars]"
        raise RemoteError(
            f"remote command timed out after {timeout:g}s (transient network/host hiccups are the usual cause — reads are safe to retry; after a create/activate timeout check `cs list` first, the submission may have gone through)",
            command=cmd_str,
            stderr=exc.stderr if isinstance(exc.stderr, str) else "",
            exit_code=None,
        ) from exc


@dataclass
class CompletedRemote:
    returncode: int
    stdout: str
    stderr: str


# Inner ssh runs with UserKnownHostsFile=/dev/null, so at the default LogLevel it prints a "Warning: Permanently added …" line to stderr on every call. We want real connection errors visible, not that one noise line — filter it.
def _filter_ssh_noise(stderr: str) -> str:
    if not stderr:
        return stderr
    return "".join(line for line in stderr.splitlines(keepends=True) if not line.startswith("Warning: Permanently added"))


# Signatures of a dead tunnel path (nc on the ProxyCommand side, or the outer ssh giving up on a half-open transport). Matched alongside exit code 255 before we trigger a tunnel rebuild + retry — we don't want to retry real command failures.
_TUNNEL_FAILURE_HINTS = (
    "Connection refused",
    "kex_exchange_identification",
    "Connection closed by",
    "Connection reset by",
    "No such file or directory",
)


def _looks_like_tunnel_failure(stderr: str) -> bool:
    if not stderr:
        return False
    return any(hint in stderr for hint in _TUNNEL_FAILURE_HINTS)


def _lock_path(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return os.path.join(tempfile.gettempdir(), f"cs-{os.getuid()}-{safe}.lock")


@contextmanager
def _process_lock(path: str) -> Iterator[None]:
    """Cross-process critical section via flock on a sidecar file. Concurrent compute-sessions processes are the normal case (any number of `cs` invocations, possibly from parallel agents) — the threading locks in HostAccess only serialize within one process, but ControlMaster startup and the tunnel cancel/unlink/forward sequence race across processes too. Lock files are tiny and left behind; flock releases on fd close, so no cleanup or staleness handling is needed."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


class HostAccess:
    """Wraps `ssh <login node>` with a persistent ControlMaster connection, plus rsync and the per-session tunnel into a session's container.

    One instance per process is the intended pattern. The first call opens the master; subsequent calls re-use the shared socket.
    """

    # Outer shell on the host (bash) sources ~/.bashrc on every ssh command invocation — can't be disabled from the client side. If the user's bashrc prints a banner to stdout, it pollutes our command output. We prefix every command with a marker echo and strip everything before it. Markers are randomized per-instance so user command output can't ever collide with the sentinel and get silently stripped.

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.host = cfg.host
        self._control_path = os.path.expanduser(cfg.control_path)
        self._master_started = False
        self._remote_home: str | None = None
        self._paths_cached: RemotePaths | None = None
        self._tunnels: dict[str, str] = {}
        # Serialize tunnel setup/teardown so two threads can't interleave the cancel/unlink/forward sequence for the same session and kill each other's live forward.
        self._tunnel_lock = threading.Lock()
        # Serialize first-time master startup: two concurrent calls would both see _master_started=False and both run `ssh -M`, and the loser errors out a call that should have succeeded. Separate from _tunnel_lock because session_tunnel() calls open() while holding that lock.
        self._open_lock = threading.Lock()
        self._container_user_cached: str | None = None
        nonce = secrets.token_hex(8)
        self._stdout_marker = f"__CS_STDOUT_{nonce}__"
        self._stderr_marker = f"__CS_STDERR_{nonce}__"

    def _wrap(self, cmd: str) -> str:
        """Prefix stdout/stderr with sentinel lines so dotfile banners can be stripped."""
        return (
            f"printf '%s\\n' {shlex.quote(self._stdout_marker)}; "
            f"printf '%s\\n' {shlex.quote(self._stderr_marker)} 1>&2; "
            f"{cmd}"
        )

    def _strip(self, stdout: str, stderr: str) -> tuple[str, str]:
        def _cut(s: str, marker: str) -> str:
            idx = s.find(marker)
            if idx < 0:
                return s
            nl = s.find("\n", idx)
            return s[nl + 1 :] if nl >= 0 else ""
        return _cut(stdout, self._stdout_marker), _cut(stderr, self._stderr_marker)

    # ---------- connection management ----------

    def _ssh_base(self) -> list[str]:
        return [
            "ssh",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"ControlPath={self._control_path}",
            "-o", "ControlMaster=auto",
            "-o", "ControlPersist=10m",
        ]

    def container_user(self) -> str:
        """Resolve the in-container login name: the configured `container_user`, else the remote `whoami` (which equals the ssh user, which equals the Unix user the container runtime runs sshd as)."""
        if self.cfg.container_user:
            return self.cfg.container_user
        if self._container_user_cached is None:
            res = self.run("whoami", check=True)
            self._container_user_cached = res.stdout.strip()
        return self._container_user_cached

    def open(self) -> None:
        if self._master_started:
            return
        with self._open_lock:
            if self._master_started:
                return
            Path(self._control_path).parent.mkdir(parents=True, exist_ok=True)
            # Cross-process serialization: two processes racing `ssh -M` on the same socket would make the loser error out a call that should have succeeded; the loser now waits and finds the winner's live master via `-O check`.
            with _process_lock(_lock_path(f"master-{self.host}")):
                # Check if a master is already live.
                check = _run_capture(
                    self._ssh_base() + ["-O", "check", self.host],
                    timeout=_CONTROL_TIMEOUT,
                )
                if check.returncode == 0:
                    self._master_started = True
                    return
                # Start it in background (no command; will run until ControlPersist expires).
                start = _run_capture(
                    self._ssh_base() + ["-M", "-N", "-f", self.host],
                    timeout=_CONTROL_TIMEOUT,
                )
                if start.returncode != 0:
                    raise RemoteError(
                        f"could not open ssh ControlMaster to {self.host}",
                        command=" ".join(self._ssh_base() + ["-M", "-N", "-f", self.host]),
                        stderr=start.stderr,
                        exit_code=start.returncode,
                    )
                self._master_started = True

    def close(self) -> None:
        if not self._master_started:
            return
        _run_capture(
            self._ssh_base() + ["-O", "exit", self.host],
            timeout=_CONTROL_TIMEOUT,
            best_effort=True,
        )
        self._master_started = False
        # Exiting the master kills all its forwards; drop the cache so a future master (re-opened lazily) rebuilds tunnels instead of returning stale paths.
        with self._tunnel_lock:
            self._tunnels.clear()

    # ---------- shell exec on the login node ----------

    def remote_home(self) -> str:
        """Resolve and cache the remote $HOME. Needed because we can't rely on shell tilde expansion through shlex.quote'd arguments."""
        if self._remote_home is None:
            self.open()
            # Use a bootstrap ssh call that bypasses our own run() (which relies on this value once we start using absolute paths in commands). Bootstrap: the outer shell sources .bashrc (can't prevent that), which may print banners. Use the marker strip to recover clean HOME.
            inner = self._wrap("echo \"$HOME\"")
            full = self._ssh_base() + [
                self.host,
                f"bash -lc {shlex.quote(inner)}",
            ]
            proc = _run_capture(full, timeout=_HOST_CMD_TIMEOUT)
            stdout, _ = self._strip(proc.stdout, proc.stderr)
            home = stdout.strip()
            if proc.returncode != 0 or not home:
                raise RemoteError(
                    f"could not resolve $HOME on {self.host}",
                    command="echo $HOME",
                    stderr=proc.stderr,
                    exit_code=proc.returncode,
                )
            self._remote_home = home
        return self._remote_home

    @property
    def paths(self) -> RemotePaths:
        """RemotePaths rooted at `remote_base` resolved to an absolute path (leading ~ expanded). Cached."""
        if self._paths_cached is None:
            base = self.cfg.remote_base
            if base.startswith("~/"):
                base = self.remote_home() + base[1:]
            elif base == "~":
                base = self.remote_home()
            self._paths_cached = RemotePaths(base)
        return self._paths_cached

    def run(self, cmd: str, *, check: bool = True, input_text: str | None = None) -> CompletedRemote:
        self.open()
        # ssh joins argv[2:] into a single command string for the remote login shell, which then re-parses. Quote the whole command once so the outer shell treats it as one word and passes it intact to `bash -c`.
        inner = self._wrap(cmd)
        full = self._ssh_base() + [self.host, f"bash -lc {shlex.quote(inner)}"]
        proc = _run_capture(full, timeout=_HOST_CMD_TIMEOUT, input_text=input_text)
        stdout, stderr = self._strip(proc.stdout, proc.stderr)
        result = CompletedRemote(
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )
        if check and proc.returncode != 0:
            raise RemoteError(
                "remote command failed",
                command=cmd,
                stderr=stderr,
                exit_code=proc.returncode,
            )
        return result

    # ---------- rsync ----------

    def _rsync(self, args: list[str], input_text: str | None = None) -> str:
        """Run rsync over the ControlMaster connection; returns stdout+stderr combined."""
        self.open()
        ssh_cmd = " ".join(shlex.quote(x) for x in self._ssh_base())
        full = ["rsync", "-a", f"--timeout={_RSYNC_IO_TIMEOUT}", "-e", ssh_cmd, *args]
        proc = subprocess.run(full, input=input_text, capture_output=True, text=True, errors="replace")
        if proc.returncode != 0:
            raise RemoteError(
                "rsync failed",
                command=" ".join(shlex.quote(a) for a in full),
                stderr=proc.stderr,
                exit_code=proc.returncode,
            )
        return proc.stdout + proc.stderr

    def rsync_files(self, local_root: Path, rel_files: Iterable[str], remote_dir: str, *, follow_symlinks: bool = False) -> str:
        """rsync a list of files (relative to local_root) into remote_dir.

        With `follow_symlinks=True`, dereferences symlinks so the linked file contents are copied (rsync `-L`). Default preserves symlinks as-is — broken or out-of-tree links won't transfer their targets.

        Returns the rsync stderr/stdout combined, for the caller to parse.
        """
        files = list(rel_files)
        if not files:
            return ""
        args = ["-L"] if follow_symlinks else []
        args += ["--itemize-changes", "--stats", "--files-from=-", str(local_root) + "/", f"{self.host}:{remote_dir}/"]
        return self._rsync(args, input_text="\n".join(files) + "\n")

    def rsync_from(self, remote_path: str, local_path: Path, *, contents_only: bool = False) -> None:
        """Copy `remote_path` to `local_path` via rsync.

        `--safe-links` makes rsync skip symlinks whose targets resolve outside the source tree. Without it, an agent inside the session container could place a symlink in the workdir pointing at the remote user's `~/.ssh/id_rsa` (or any other user-readable file), and the copy would dereference it server-side and exfiltrate the target. Plain rsync also preserves symlinks as-is rather than following them, which prevents the same attack via in-tree links.

        With `contents_only=True` (only meaningful for directories), copies the *contents* of `remote_path` into `local_path` (rsync `src/ dst/` semantics — no nested `dst/src/` directory). Caller must ensure `local_path` exists as a directory.
        """
        src = f"{remote_path}/" if contents_only else remote_path
        dst = f"{local_path}/" if contents_only else str(local_path)
        self._rsync(["--safe-links", f"{self.host}:{src}", dst])

    def rsync_to(self, local_path: Path, remote_path: str, *, contents_only: bool = False) -> None:
        """Copy `local_path` to `remote_path` via rsync.

        `--safe-links` skips symlinks whose targets resolve outside the source tree, so a symlink inside an uploaded directory can't be dereferenced to pull in out-of-tree files. This mirrors `rsync_from`'s download-side handling (and matches `sync`, which likewise preserves links rather than following them).

        With `contents_only=True` (only meaningful for directories), copies the *contents* of `local_path` into `remote_path` (rsync `src/ dst/` semantics — no nested `remote/<basename>/` directory). Caller must ensure `remote_path` exists as a directory.
        """
        src = f"{local_path}/" if contents_only else str(local_path)
        dst = f"{remote_path}/" if contents_only else remote_path
        self._rsync(["--safe-links", src, f"{self.host}:{dst}"])

    # ---------- exec inside the container through the session tunnel ----------

    def _local_tunnel_path(self, session_id: str) -> str:
        """Deterministic local-side path for a session's `-L` forward.

        Computed the same way whether the tunnel is cached or not, so close_tunnel() can clear stale on-disk sockets even after a process restart wiped the in-memory cache.
        """
        return os.path.join(
            tempfile.gettempdir(), f"cs-{os.getuid()}-{session_id}.sock"
        )

    def session_tunnel(self, session_id: str) -> str:
        """Open (and cache) a local Unix socket that forwards to the session's sshd via the ControlMaster.

        The remote side of the `-L` forward is the reverse-tunnel Unix socket the runner creates in the session dir on the login node's shared FS. We use SSH's native streamlocal forwarding (`-L /local.sock:/remote.sock`) instead of piping `nc` through a remote shell, so no remote shell runs for this tunnel — which bypasses any PAM messages, bashrc output, or other stdout pollution on the host that would otherwise corrupt the SSH protocol stream.
        """
        with self._tunnel_lock:
            if session_id in self._tunnels:
                return self._tunnels[session_id]
            self.open()
            target = self.paths.socket(session_id)
            # Local socket path — short, tmp, collision-free per session+user.
            local = self._local_tunnel_path(session_id)
            # Cross-process lock: without it, two processes interleaving this cancel/unlink/forward sequence for the same session can cancel each other's just-created forward and unlink a live socket. Serialized, the end state is one live forward at the shared path, whichever process ran last.
            with _process_lock(local + ".lock"):
                # The ControlMaster persists across Python processes (ControlPersist=10m), so a previous process may have already registered this forward. `ssh -O forward` is a no-op if the forward exists, returning 0 without creating a new local socket. Cancel any prior registration first.
                _run_capture(
                    self._ssh_base() + ["-O", "cancel", "-L", f"{local}:{target}", self.host],
                    timeout=_CONTROL_TIMEOUT,
                    best_effort=True,
                )
                try:
                    os.unlink(local)
                except FileNotFoundError:
                    pass
                cmd = self._ssh_base() + ["-O", "forward", "-L", f"{local}:{target}", self.host]
                proc = _run_capture(cmd, timeout=_CONTROL_TIMEOUT)
                if proc.returncode != 0 or not os.path.exists(local):
                    raise RemoteError(
                        "could not open session tunnel forward",
                        command=" ".join(shlex.quote(c) for c in cmd),
                        stderr=proc.stderr or f"local socket {local} not created",
                        exit_code=proc.returncode,
                    )
            self._tunnels[session_id] = local
            return local

    def close_tunnel(self, session_id: str) -> None:
        """Tear down a per-session `-L` forward and drop its cache entry.

        Call when the remote endpoint may have changed (e.g. after deactivate, or defensively before activate — a re-activation creates a fresh socket). Without this, a cached tunnel would return a stale local socket path and the next `run` would hit `nc: … Connection refused`.

        Works even when the cache is empty: re-derives the local path from session_id and tries to cancel/unlink anyway. This covers a fresh process inheriting a stale forward + stale socket file from an earlier one (the ControlMaster outlives any single cs invocation), with no in-memory handle to them.

        Best-effort: the underlying ssh/unlink calls are not checked (a missing forward or file is fine).
        """
        with self._tunnel_lock:
            local = self._tunnels.pop(session_id, None) or self._local_tunnel_path(session_id)
            target = self.paths.socket(session_id)
            # Same cross-process lock as session_tunnel — a teardown racing another process's setup must not unlink the socket that setup just created.
            with _process_lock(local + ".lock"):
                _run_capture(
                    self._ssh_base() + ["-O", "cancel", "-L", f"{local}:{target}", self.host],
                    timeout=_CONTROL_TIMEOUT,
                    best_effort=True,
                )
                try:
                    os.unlink(local)
                except FileNotFoundError:
                    pass

    def exec_in_container(self, session_id: str, cmd: str) -> CompletedRemote:
        """SSH into the session container via its tunnel.

        Opens a local forward (once, cached) and runs ssh against it. No remote shell is ever involved in the transport.
        """
        def _build_outer() -> list[str]:
            local_sock = self.session_tunnel(session_id)
            proxy_cmd = f"nc -U {shlex.quote(local_sock)}"
            # Inner ssh uses the user's default keys/agent. The container's sshd authorized_keys is populated from the cluster's ~/.ssh/authorized_keys, so the same key that logs into the login node logs into the container.
            return [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", f"ProxyCommand={proxy_cmd}",
                f"{self.container_user()}@localhost",
                cmd,
            ]

        proc = _run_capture(_build_outer(), timeout=_CONTAINER_EXEC_TIMEOUT)
        # Exit 255 + one of these transport-layer messages means ssh (or its ProxyCommand) couldn't reach sshd — not a command failure. Common cause: cached tunnel outlived the endpoint (session restart, ControlMaster recycled, stale local socket file). Tear down the cached tunnel and retry once; session_tunnel() will rebuild the `-L` forward from scratch.
        if proc.returncode == 255 and _looks_like_tunnel_failure(proc.stderr):
            self.close_tunnel(session_id)
            proc = _run_capture(_build_outer(), timeout=_CONTAINER_EXEC_TIMEOUT)
        return CompletedRemote(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=_filter_ssh_noise(proc.stderr),
        )
