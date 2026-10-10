from __future__ import annotations

import fcntl
import functools
import json
import logging
import os
import re
import secrets
import shlex
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from compute_sessions.config import Config
from compute_sessions.errors import RemoteError, SessionError, UsageError


# Wall-clock ceilings for the local ssh subprocesses — safety nets against a wedged master or a hung host, not normal operating limits. Without them one blackholed connection hangs the whole cs invocation.
_CONTROL_TIMEOUT = 60.0      # master open / check
_HELPER_TIMEOUT = 180.0      # one login-node helper request
# rsync moves real data and can legitimately run for many minutes, so it is not bounded by wall clock; its own --timeout aborts a transfer that moves no bytes for this long.
_RSYNC_IO_TIMEOUT = 120
# Transient transfer failures — rsync's own timeout (30) or a dropped connection (12 protocol stream, 10 socket I/O, 255 ssh transport) — are resumed; anything else (23 partial transfer, 3 file selection, 11 file I/O …) is a real error a retry would only repeat.
_RSYNC_RETRY_CODES = frozenset({10, 12, 30, 255})
_RSYNC_RETRY_PAUSES = (10, 30)  # seconds before the 2nd and 3rd attempt
# Received bytes of an interrupted file are kept in this sibling dir on the receiving side, so a retry resumes instead of restarting from zero. Never under the final name: a truncated file must not be mistakable for a complete one.
_RSYNC_PARTIAL_DIR = ".rsync-partial"
# Container connections are shared for this long after their last use: back-to-back commands skip the handshake. The open connection counts as activity for the idle monitor, so keep it short.
_CONTAINER_PERSIST = "60s"

# The helper source travels on stdin ahead of the request; this bootstrap (the only thing on the command line) reads exactly that many bytes and runs them. -I: no cwd/user-site/PYTHON* on sys.path, so a stray `json.py` on the login node cannot hijack the import.
_BOOT = "import sys;exec(compile(sys.stdin.buffer.read(int(sys.argv[1])),'cs-helper','exec'))"
_HELPER_SOURCE = Path(__file__).with_name("login_helper.py").read_bytes()

log = logging.getLogger(__name__)


@contextmanager
def _process_lock(path: str) -> Iterator[None]:
    """Cross-process critical section via flock on a sidecar file (concurrent cs processes are normal). flock releases on fd close, so no cleanup or staleness handling is needed."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as exc:
        # Typically a sandbox that blocks writes there; ssh's control socket needs the same directory.
        raise RemoteError(f"cannot create {path} ({exc.strerror}) — cs needs write access to {os.path.dirname(path)} for its ssh connection; in a sandbox, run cs outside it") from exc
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _run(args: list[str], *, timeout: float, input: bytes | None = None) -> subprocess.CompletedProcess:
    """subprocess.run with captured bytes and a hard timeout (→ RemoteError). Without input, stdin is /dev/null: ssh forwards whatever stdin it gets, so an inherited pipe would be silently drained (`… | while read s; do cs deactivate "$s"; done`)."""
    try:
        return subprocess.run(args, input=input, stdin=subprocess.DEVNULL if input is None else None, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RemoteError(
            f"remote command timed out after {timeout:g}s (usually a transient network/host hiccup — reads are safe to retry; after a create/activate timeout check `cs list` first, the submission may have gone through)",
            command=" ".join(args[:2]) + " …",
        ) from exc


@functools.cache
def _ssh_version() -> tuple[int, int]:
    out = subprocess.run(["ssh", "-V"], stdin=subprocess.DEVNULL, capture_output=True, text=True).stderr
    m = re.search(r"OpenSSH_(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


class HostAccess:
    """The login node: one shared ControlMaster connection, the per-request helper, rsync over the master, and the ssh command line into a session's container (through the master, via the session's reverse-tunnel socket)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.host = cfg.host
        self._control_path = os.path.expanduser(cfg.control_path)
        self._master_ok = False
        # Learned from every helper reply: the login user (= the container's sshd user).
        self.user: str | None = None

    def _ssh_base(self, *, master: bool = True) -> list[str]:
        # The keepalives make a connection that died silently (laptop sleep, network switch) fail within ~45 s instead of hanging every cs command on it; they take effect when the master (re)starts.
        opts = ["StrictHostKeyChecking=accept-new", f"ControlPath={self._control_path}", "ServerAliveInterval=15", "ServerAliveCountMax=3", "ConnectTimeout=15"]
        opts += ["ControlMaster=auto", "ControlPersist=1h"] if master else ["ControlMaster=no"]
        return ["ssh"] + [x for o in opts for x in ("-o", o)]

    def open(self) -> None:
        if self._master_ok:
            return
        control_dir = Path(self._control_path).parent
        control_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", self.host)
        # Two processes racing `ssh -M` on one socket would make the loser fail; serialized, the loser finds the winner's master. The lock sits next to the control socket, out of other users' reach.
        with _process_lock(str(control_dir / f"cs-master-{safe}.lock")):
            if _run(self._ssh_base() + ["-O", "check", self.host], timeout=_CONTROL_TIMEOUT).returncode != 0:
                start = _run(self._ssh_base() + ["-M", "-N", "-f", self.host], timeout=_CONTROL_TIMEOUT)
                if start.returncode != 0:
                    raise RemoteError(f"could not connect to {self.host}", stderr=start.stderr.decode(errors="replace"), exit_code=start.returncode)
        self._master_ok = True

    # ---------- the login-node helper ----------

    def _helper_argv(self, nsrc: int) -> list[str]:
        return self._ssh_base() + [self.host, f"python3 -I -c {shlex.quote(_BOOT)} {nsrc}"]

    def call(self, op: str, *, retry: bool = True, timeout: float = _HELPER_TIMEOUT, **params) -> dict:
        """One helper request: all login-node work of `op`, one round trip. A transport failure that produced no reply (e.g. a master whose connection died during laptop sleep) is re-sent once on a fresh connection — callers pass retry=False for operations that must not run twice (create, clone, submit)."""
        marker = f"__CS_REPLY_{secrets.token_hex(8)}__"
        stdin = _HELPER_SOURCE + json.dumps({"op": op, "marker": marker, "base": self.cfg.remote_base, **params}).encode()
        for attempt in (1, 2):
            self.open()
            proc = _run(self._helper_argv(len(_HELPER_SOURCE)), timeout=timeout, input=stdin)
            out = proc.stdout.decode(errors="replace")
            i = out.find(marker + "\n")
            if i >= 0:
                break
            stderr = proc.stderr.decode(errors="replace")
            if not (retry and attempt == 1 and proc.returncode == 255):
                raise RemoteError(f"login-node request {op!r} failed", stderr=stderr, exit_code=proc.returncode)
            log.debug("helper transport failure, retrying: %s", stderr.strip())
            self._master_ok = False
            time.sleep(1)
        try:
            # The reply is one JSON line; anything after it (a shell EXIT trap's echo) is noise.
            reply = json.loads(out[i + len(marker) + 1:].partition("\n")[0])
        except ValueError as exc:
            raise RemoteError(f"login-node request {op!r} returned a malformed reply: {exc}", stderr=proc.stderr.decode(errors="replace")) from exc
        if "error" in reply:
            kind, msg = reply["kind"], reply["error"]
            if kind == "usage":
                raise UsageError(msg)
            if kind == "infer":
                raise _inference_error(reply)
            if kind == "session":
                raise SessionError(msg)
            raise RemoteError(f"login-node request {op!r} failed: {msg}")
        self.user = reply.pop("user")
        return reply

    # ---------- rsync ----------

    def _remote(self, path: str) -> str:
        return f"{self.host}:{path}"

    def _rsync_transport(self) -> list[str]:
        return ["-e", " ".join(shlex.quote(x) for x in self._ssh_base())]

    def _rsync(self, args: list[str], input: str | None = None) -> str:
        """rsync over the master; returns stdout+stderr. Transient failures are retried with the partial file resumed; a retry that makes no progress fails the same way, so a genuine outage costs at most the pauses."""
        self.open()
        full = ["rsync", "-a", f"--timeout={_RSYNC_IO_TIMEOUT}", f"--partial-dir={_RSYNC_PARTIAL_DIR}", *self._rsync_transport(), *args]
        for attempt, pause in enumerate((*_RSYNC_RETRY_PAUSES, None), 1):
            proc = subprocess.run(full, input=input, stdin=subprocess.DEVNULL if input is None else None, capture_output=True, text=True, errors="replace")
            if proc.returncode == 0:
                return proc.stdout + proc.stderr
            if pause is None or proc.returncode not in _RSYNC_RETRY_CODES:
                break
            reason = next((l for l in proc.stderr.splitlines() if l.strip()), f"exit {proc.returncode}")
            log.warning("cs: transfer interrupted (%s) — resuming in %ds (attempt %d)", reason.strip(), pause, attempt + 1)
            time.sleep(pause)
        raise RemoteError("rsync failed" + (f" after {attempt} attempts" if attempt > 1 else ""), command=" ".join(shlex.quote(a) for a in full), stderr=proc.stderr, exit_code=proc.returncode)

    def rsync_files(self, local_root: Path, rel_files: Iterable[str], remote_dir: str, *, follow_symlinks: bool = False) -> str:
        """rsync a list of files (relative to local_root) into remote_dir; returns rsync's itemized output. `follow_symlinks` copies link targets (rsync -L) instead of the links."""
        files = list(rel_files)
        if not files:
            return ""
        args = ["-L"] if follow_symlinks else []
        args += ["--itemize-changes", "--stats", "--files-from=-", f"{local_root}/", self._remote(f"{remote_dir}/")]
        return self._rsync(args, input="\n".join(files) + "\n")

    def rsync_from(self, remote_path: str, local_path: Path, *, contents_only: bool = False) -> None:
        """Copy a workdir path to the local machine. `--safe-links` skips symlinks whose targets leave the source tree, so a link placed in the workdir from inside the container cannot exfiltrate e.g. `~/.ssh/id_rsa`. `contents_only` = rsync's `src/ dst/`."""
        src = f"{remote_path}/" if contents_only else remote_path
        dst = f"{local_path}/" if contents_only else str(local_path)
        self._rsync(["--safe-links", self._remote(src), dst])

    def rsync_to(self, local_path: Path, remote_path: str, *, contents_only: bool = False) -> None:
        """Copy a local path into the workdir (`--safe-links` as for downloads; rsync creates the last component of the destination)."""
        src = f"{local_path}/" if contents_only else str(local_path)
        dst = f"{remote_path}/" if contents_only else remote_path
        self._rsync(["--safe-links", src, self._remote(dst)])

    # ---------- into the container ----------

    def container_control_path(self, session_id: str) -> str:
        return os.path.join(os.path.dirname(self._control_path), f"cs-{session_id}")

    def container_argv(self, session_id: str, socket: str, remote_cmd: str) -> list[str]:
        """ssh into the session container through its reverse-tunnel socket on the login node: the ProxyCommand is a stdio forward (`ssh -W`) through the login master — nothing local to manage, and a re-activation's new socket is simply the next connection's target. The connection itself is shared (ControlMaster=auto) for _CONTAINER_PERSIST after its last use."""
        if _ssh_version() < (9, 4):
            raise RemoteError(f"reaching a session container needs OpenSSH 9.4 or newer on this machine (`ssh -W` to a Unix socket); found {'.'.join(map(str, _ssh_version()))}")
        # `%` doubled: ssh expands %-tokens in ProxyCommand, and the login ControlPath carries its own (%r@%h:%p) for the inner ssh to expand.
        proxy = " ".join(shlex.quote(a) for a in self._ssh_base(master=False) + ["-W", socket, self.host]).replace("%", "%%")
        opts = [
            "ControlMaster=auto", f"ControlPath={self.container_control_path(session_id)}", f"ControlPersist={_CONTAINER_PERSIST}",
            "ServerAliveInterval=15", "ServerAliveCountMax=3", "ConnectTimeout=15",
            # The container's host key is per session and its sshd is reachable only through our own tunnel.
            "StrictHostKeyChecking=no", "UserKnownHostsFile=/dev/null",
            # VERBOSE logs "Authenticated to …" once a login succeeds — run.py tells from it whether a failed connection could have run anything.
            "LogLevel=VERBOSE",
            f"ProxyCommand={proxy}",
        ]
        user = self.cfg.container_user or self.user
        return ["ssh"] + [x for o in opts for x in ("-o", o)] + ["-l", user, f"cs-{session_id}", remote_cmd]

    def close_container(self, session_id: str) -> None:
        """End the session's shared container connection (best effort) — after a deactivate it only points at a dead endpoint."""
        cp = self.container_control_path(session_id)
        if os.path.exists(cp):
            try:
                _run(["ssh", "-o", f"ControlPath={cp}", "-O", "exit", f"cs-{session_id}"], timeout=_CONTROL_TIMEOUT)
            except RemoteError:
                pass


def _inference_error(reply: dict) -> UsageError:
    """The cwd project has several sessions and no unique live one: list them, most recent first."""
    from compute_sessions.models import SessionInfo, fmt_duration
    infos = [SessionInfo.from_json(d) for d in reply["sessions"]]
    pid = reply["project_id"]

    def age(i: SessionInfo) -> float:
        t = i.to_enriched_json().get("time_in_status_seconds")
        return t if t is not None else float("inf")

    def row(i: SessionInfo) -> str:
        return f"{i.session_id} ({i.status.value} {fmt_duration(i.to_enriched_json().get('time_in_status_seconds'))})"

    infos.sort(key=age)
    if reply["live"]:
        return UsageError(f"project {pid!r} has {len(infos)} live sessions — pass one: {', '.join(row(i) for i in infos)}")
    more = f"; {len(infos) - 4} more via `cs list`" if len(infos) > 4 else ""
    return UsageError(f"project {pid!r} has {len(infos)} sessions but none is live — pass one, most recent first: {', '.join(row(i) for i in infos[:4])}{more}")
