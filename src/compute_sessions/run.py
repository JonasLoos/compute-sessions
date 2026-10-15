"""Commands inside a session container: launching, live output, kill.

`cs run` and `cs logs -f` hold one ssh connection into the container (see HostAccess.container_argv) running a small bash follower, which reads the command's log files where they are written — no shared-FS delay, no login-node load — and sends them as frames:

    O <offset> <len>\\n<bytes> / E …   stdout/stderr bytes (at most 64 KB per frame)
    S                                 the launch in this request is done
    F <message>                       the launch failed before the command started
    H                                 keepalive (every 15 s without output)
    X [<code>]                        exited (after the last bytes); no code = unreadable `.exit`
    K                                 died without recording an exit (after the last bytes)
    T                                 the round ended (~5 min): reconnect from the received offsets
    M / P                             no such command / it exists (sent before following)

Reconnecting is the normal path: every round ends, and a dropped connection resumes from the byte offsets the client has. A command is launched again ONLY when the failed connection never reached the container's sshd — `run` stays non-idempotent.
"""
from __future__ import annotations

import codecs
import os
import re
import secrets
import select
import shlex
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterator

from compute_sessions import session
from compute_sessions.errors import ComputeSessionsError, RemoteError, SessionError, UsageError


# Accepts POSIX signal names (with or without SIG prefix) or numeric IDs. Whitelisted rather than regex-shaped: a typo like "FAKETERM" would otherwise fail with a cryptic shell-level error in the container.
_VALID_SIGNALS: frozenset[str] = frozenset({
    "HUP", "INT", "QUIT", "ILL", "TRAP", "ABRT", "BUS", "FPE", "KILL",
    "USR1", "SEGV", "USR2", "PIPE", "ALRM", "TERM", "STKFLT", "CHLD",
    "CONT", "STOP", "TSTP", "TTIN", "TTOU", "URG", "XCPU", "XFSZ",
    "VTALRM", "PROF", "WINCH", "IO", "PWR", "SYS",
})


def validate_signal(signal: str) -> None:
    if signal.isdigit():
        if not 1 <= int(signal) <= 64:
            raise UsageError(f"invalid signal number: {signal!r}")
        return
    if signal.removeprefix("SIG") not in _VALID_SIGNALS:
        raise UsageError(f"invalid signal: {signal!r}")


# The command refreshes its `.pid` mtime this often while alive — the liveness signal for the runner's idle monitor and for readers on the login node (a container PID means nothing outside the container's PID namespace). Must stay well below the freshness window everyone uses.
_HEARTBEAT_SECONDS = 10
_PID_FRESH_SECONDS = 60
# How often a `cs run --after` command checks whether the commands it waits for have ended.
_AFTER_POLL_SECONDS = 1
# Session log dir and workdir inside the container (bind mounts of the session dir on the shared FS).
_LOGS = "/cs-logs"
_WORKDIR = "/workdir"
_ROUND_SECONDS = 300
_KEEPALIVE_SECONDS = 15
# Written when the container's sshd starts. A recorded PID is only valid in the container that launched the command — after a re-activation the new PID namespace reuses the numbers — so kill never signals a command whose `.start` is older than this file.
_SSHD_PID = "/sshd/sshd.pid"
# No frame for this long (keepalives come every 15 s) means a dead connection or a stuck follower: reconnect.
_FRAME_TIMEOUT = 45
_MAX_FAILURES = 8          # backoff 2, 4, 8, 16, 30, 30, 30 s: about 2 minutes
# A connection that failed in the SSH handshake (tunnel socket missing or refused, sshd not answering) never ran anything, so a launch may be sent again. ssh runs with LogLevel=VERBOSE, which also logs "Authenticated to …" once a login succeeded; and a failed shared-connection attempt that falls back to a direct one leaves only the direct one's handshake error. Mux/forwarding errors alone are not enough: OpenSSH may fall back after them and run the command.
_HANDSHAKE_FAILED = ("kex_exchange_identification", "ssh_exchange_identification", "Connection closed by UNKNOWN", "banner exchange")
_SSH_NOISE = ("Warning: Permanently added", "Authenticated to ")

def _never_reached_sshd(returncode: int, got_output: bool, stderr: str) -> bool:
    return returncode == 255 and not got_output and "Authenticated to " not in stderr and any(s in stderr for s in _HANDSHAKE_FAILED)


def new_command_id() -> str:
    return "cmd_" + secrets.token_hex(8)


def _validate_command_id(command_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_]{1,128}", command_id):
        raise UsageError(f"invalid command id: {command_id!r}")


def _inner(command_id: str, after: list[str] | tuple[str, ...] = ()) -> str:
    """The detached bash that runs the user command, read from `.cmd` — never from its own arguments, so `pgrep -f`/`pkill -f` patterns in the command cannot match this wrapper (job control in the launcher gives this bash its own process group, so its PID is the PGID `cs kill` signals):
      - TERM/INT/HUP traps exit with 128+N, so the EXIT trap sees the signal instead of $?=0, and the EXIT trap writes `.exit` on every exit path. The subshell traps them the same way: bash runs a trap only after the foreground process ends, so a command that shuts down gracefully on TERM (saving a checkpoint) finishes — and its last output is streamed — before `.exit` is written; a shell loop still stops after the current step. USR1/USR2/ALRM (e.g. a checkpoint request) are no-ops for this bash and the subshell: they reach the command's own processes without ending the wrapper.
      - Heartbeat: a background loop touches `.pid` every _HEARTBEAT_SECONDS while this bash lives (`kill -0 $$` runs inside the container, where the PID is valid) and stops within that interval after any death, SIGKILL/OOM included. Job control puts it in a process group of its own, so after `.exit` a populated command group means processes of the command itself (see _KILL_SCRIPT).
      - The command runs in a subshell through `eval --` (a command may start with '-'), so it is parsed on its own (a trailing `# comment` or a heredoc cannot break the wrapper, a syntax error exits 2 with bash's message in the command's stderr), and its `exit`, `exec`, `set -e` or own EXIT trap end only the subshell.
      - `.exit` is written before the venv-snapshot epilogue (runner.py), which never delays the result; the epilogue is skipped when the session's runner bound no helper.
      - `after` (`cs run --after`): the command starts only once each listed command has exited 0, waiting while it is queued or its heartbeat is fresh; `.wait` marks the wait. As soon as one has failed or died, it exits 125 without running, the reason on its stderr. The heartbeat runs during the wait, so the waiting command holds the session like a running one."""
    b = f"{_LOGS}/{command_id}"
    out, err, exit_f, pid_f, cmd_f, wait_f = (shlex.quote(f"{b}.{x}") for x in ("out", "err", "exit", "pid", "cmd", "wait"))
    wait = ""
    if after:
        # Each round checks every listed command, so one that failed ends the wait at once. An empty `.exit` is still being written while the heartbeat is fresh.
        wait = (
            f"printf '%s ' {' '.join(after)} > {wait_f}; "
            f"while :; do w=; for d in {' '.join(after)}; do "
            f"f={shlex.quote(_LOGS)}/$d; c=; [ -s \"$f.exit\" ] && IFS= read -r c < \"$f.exit\"; "
            f"if [ -n \"$c\" ]; then [ \"$c\" = 0 ] && continue; r=\"exited $c\"; "
            f"elif [ -e \"$f.queued\" ] || {{ [ -e \"$f.pid\" ] && [ $(($(date +%s) - $(stat -c %Y \"$f.pid\" 2>/dev/null || stat -f %m \"$f.pid\" 2>/dev/null || echo 0))) -lt {_PID_FRESH_SECONDS} ]; }}; then w=1; continue; "
            f"else r='ended without an exit code (killed, dropped from the queue, or its session ended)'; fi; "
            f"rm -f {wait_f}; echo \"cs: not started — $d $r\" >> {err}; exit 125; done; "
            f"[ -z \"$w\" ] && break; sleep {_AFTER_POLL_SECONDS}; done; "
            f"rm -f {wait_f}; "
        )
    return (
        f"trap 'exit 143' TERM; trap 'exit 130' INT; trap 'exit 129' HUP; trap : USR1 USR2 ALRM; "
        f"trap 'ec=$?; echo \"$ec\" > {exit_f}' EXIT; "
        f"echo $$ > {pid_f}; "
        f"set -m; ( while kill -0 $$ 2>/dev/null; do touch {pid_f}; sleep {_HEARTBEAT_SECONDS}; done ) & set +m; "
        f"{wait}"
        f"IFS= read -r -d '' __cs_cmd < {cmd_f}; "
        f"( trap 'exit 143' TERM; trap 'exit 130' INT; trap 'exit 129' HUP; trap : USR1 USR2 ALRM; eval -- \"$__cs_cmd\" ) > {out} 2> {err}; ec=$?; "
        f"echo \"$ec\" > {exit_f}; "
        f"if [ -x /cs-helpers/venv-snapshot.sh ]; then setsid /cs-helpers/venv-snapshot.sh </dev/null >/dev/null 2>&1 & fi; "
        f"exit $ec"
    )


# Sent on stdin to `bash -l -s` (login shell: the image's profile sets PATH, uv, CUDA env). Not on the command line: the follower lives as long as the command, and `pkill -f <pattern>`/`ps | grep` inside the session must not match it. The whole script is one function called on the last line, and its body starts with `exec </dev/null`, so bash has read all of it before anything runs and no child can eat the rest of the script from stdin.
# Portable to bash 3.2 and BSD stat, so the tests run on macOS too.
_SCRIPT = r"""cs_main() {
exec </dev/null
B=@BASE@
printf '%s\n' @MARK@ || return
if [ @LAUNCH@ = 1 ]; then
  # `.pid` exists from here on, so a missing `.pid` always means "no such command". `.start` is the launch time (`.pid`'s mtime is the heartbeat); `.cmd` the command text for `cs commands`; `.release` asks the runner to end the session once the command is done; `.queued` marks a command waiting for the activation.
  if mkdir -p @LOGS@ && cd @WORKDIR@ && : > "$B.out" && : > "$B.err" && printf '%s' @COMMAND@ > "$B.cmd" && : > "$B.pid" && : > "$B.start" && rm -f "$B.exit" "$B.queued" "$B.wait" && { [ @RELEASE@ = 0 ] || : > "$B.release"; }; then
    set -m
    bash -c @INNER@ </dev/null >/dev/null 2>&1 &
    disown $! 2>/dev/null
    set +m
    printf 'S\n' || return
  else
    printf 'F could not prepare the command log files in the container\n'
    return
  fi
fi
[ -e "$B.pid" ] || { printf 'M\n'; return; }
printf 'P\n' || return
[ @FOLLOW@ = 1 ] || return
if stat -c %s "$B.pid" >/dev/null 2>&1; then cs_st() { stat -c '%s %Y' "$@" 2>/dev/null; }; else cs_st() { stat -f '%z %m' "$@" 2>/dev/null; }; fi
oo=@OO@; eo=@EO@; t0=$SECONDS; now0=$(date +%s); sent=$SECONDS; seen=$SECONDS
cs_send() {
  local tag=$1 f=$2 sz=$3 off n
  if [ "$tag" = O ]; then off=$oo; else off=$eo; fi
  while [ "$sz" -gt "$off" ]; do
    n=$((sz - off)); [ "$n" -gt 65536 ] && n=65536
    printf '%s %s %s\n' "$tag" "$off" "$n" || return 1
    tail -c +$((off + 1)) "$f" | head -c "$n" || return 1
    off=$((off + n)); sent=$SECONDS; seen=$SECONDS
  done
  if [ "$tag" = O ]; then oo=$off; else eo=$off; fi
}
cs_drain() {
  set -- $(cs_st "$B.out" "$B.err" "$B.pid")
  osz=${1:-0}; esz=${3:-0}; pmt=${6:-0}
  if [ $# -ne 6 ]; then
    set -- $(cs_st "$B.out"); osz=${1:-0}
    set -- $(cs_st "$B.err"); esz=${1:-0}
    set -- $(cs_st "$B.pid"); pmt=${2:-0}
  fi
  cs_send O "$B.out" "$osz" && cs_send E "$B.err" "$esz"
}
cs_code() {
  code=; [ -s "$B.exit" ] && IFS= read -r code < "$B.exit"
  case $code in ''|*[!0-9]*) code= ;; esac
}
# Alive = fresh heartbeat AND (no PID yet, i.e. starting, OR the PID exists). The heartbeat condition matters: a command from an earlier activation keeps its old `.pid`, and in this container that PID can belong to another process.
cs_alive() {
  local pid=
  [ $((now0 + SECONDS - t0 - pmt)) -lt @FRESH@ ] || return 1
  IFS= read -r pid < "$B.pid" 2>/dev/null
  case $pid in ''|*[!0-9]*) return 0 ;; esac
  kill -0 "$pid" 2>/dev/null
}
while :; do
  cs_code
  cs_drain || return
  if [ -n "$code" ]; then printf 'X %s\n' "$code"; return; fi
  if ! cs_alive; then
    # A normal exit can land between the two checks.
    cs_code
    cs_drain || return
    if [ -n "$code" ]; then printf 'X %s\n' "$code"; elif [ -e "$B.exit" ]; then printf 'X\n'; else printf 'K\n'; fi
    return
  fi
  if [ $((SECONDS - t0)) -ge @ROUND@ ]; then printf 'T\n'; return; fi
  if [ $((SECONDS - sent)) -ge @KEEPALIVE@ ]; then printf 'H\n' || return; sent=$SECONDS; fi
  idle=$((SECONDS - seen))
  # Poll fast while output is recent (short commands finish within 0.1 s of the end), slower once the command has been quiet for a while.
  if [ $idle -lt 10 ]; then sleep 0.1; elif [ $idle -lt 60 ]; then sleep 0.25; else sleep 0.5; fi
done
}
cs_main
"""


def container_script(command_id: str, marker: str, *, command: str | None = None, follow: bool = True, offsets: tuple[int, int] = (0, 0), round_seconds: int = _ROUND_SECONDS, keepalive: int = _KEEPALIVE_SECONDS, release: bool = False, after: list[str] | tuple[str, ...] = ()) -> bytes:
    """The stdin script for one container connection: optionally launch `command` (with `release`: the session ends once it is done; with `after`: it starts once those commands have succeeded), then follow (or just check that the command exists)."""
    for cid in after:
        _validate_command_id(cid)
    subs = {
        "@BASE@": shlex.quote(f"{_LOGS}/{command_id}"), "@MARK@": shlex.quote(marker), "@LOGS@": shlex.quote(_LOGS), "@WORKDIR@": shlex.quote(_WORKDIR),
        "@LAUNCH@": "1" if command is not None else "0", "@FOLLOW@": "1" if follow else "0", "@RELEASE@": "1" if release else "0",
        "@INNER@": shlex.quote(_inner(command_id, after)), "@COMMAND@": shlex.quote(command) if command is not None else "''",
        "@OO@": str(offsets[0]), "@EO@": str(offsets[1]), "@ROUND@": str(round_seconds), "@KEEPALIVE@": str(keepalive), "@FRESH@": str(_PID_FRESH_SECONDS),
    }
    # surrogateescape: a command-line argument that is not valid UTF-8 passes through as its original bytes.
    return _fill(_SCRIPT, subs).encode(errors="surrogateescape")


def _fill(script: str, subs: dict[str, str]) -> str:
    """Substitute @PLACEHOLDERS@ in one pass — a substituted value (the user command) may itself contain placeholder-like text."""
    return re.sub("|".join(subs), lambda m: subs[m.group(0)], script)


class FrameError(ComputeSessionsError):
    pass


class Frames:
    """Incremental parser of the follower's output. Everything before the marker line (login-shell noise) is dropped; a frame whose payload has not fully arrived stays buffered."""

    _SIMPLE = {"S", "F", "H", "X", "K", "T", "M", "P"}

    def __init__(self, marker: str):
        self.marker = marker.encode() + b"\n"
        self.buf = bytearray()
        self.started = False

    def feed(self, data: bytes) -> list[tuple]:
        self.buf += data
        events: list[tuple] = []
        if not self.started:
            i = self.buf.find(self.marker)
            if i < 0:
                del self.buf[:-len(self.marker)]
                return events
            del self.buf[:i + len(self.marker)]
            self.started = True
        while True:
            nl = self.buf.find(b"\n")
            if nl < 0:
                if len(self.buf) > 4096:
                    raise FrameError("malformed stream from the container (no frame header)")
                return events
            head = self.buf[:nl].decode("ascii", "replace").split(" ")
            if head[0] in ("O", "E"):
                if len(head) != 3 or not head[1].isdigit() or not head[2].isdigit():
                    raise FrameError(f"malformed frame header from the container: {head!r}")
                end = nl + 1 + int(head[2])
                if len(self.buf) < end:
                    return events
                events.append((head[0], int(head[1]), bytes(self.buf[nl + 1:end])))
                del self.buf[:end]
            elif head[0] in self._SIMPLE:
                events.append((head[0], " ".join(head[1:])))
                del self.buf[:nl + 1]
            else:
                raise FrameError(f"unexpected frame from the container: {head!r}")


def collapse_cr(line: bytes) -> bytes:
    """A carriage-return-overwritten line (tqdm bars, download meters) reduced to its final state; a CRLF ending loses only its \\r."""
    if line.endswith(b"\r"):
        line = line[:-1]
    i = line.rfind(b"\r")
    return line[i + 1:] if i >= 0 else line


# A held line that keeps being overwritten with \r (a tqdm bar) is shown as a line of its own at most this often, so a reader of a pipe or file sees progress before the line ends.
PROGRESS_EVERY = 30.0


class _Sink:
    def __init__(self, name: str, tail: int | None, raw: bool | None):
        self.name = name
        self.tail = tail
        self.raw = raw
        self.partial = b""
        self.shown: bytes | None = None  # the held line's state last shown as a snapshot line
        self.shown_at: float | None = None
        self.lines: deque[bytes] = deque(maxlen=tail) if tail else deque()
        # Output that is not UTF-8 is data (`cs run 'cat model.bin' > model.bin`): from the first invalid byte on, nothing is collapsed or held back.
        self.text = codecs.getincrementaldecoder("utf-8")()
        self.binary = False
        self.last = b"\n"

    def _file(self):
        return getattr(sys, self.name)

    def _out(self, data: bytes) -> None:
        if not data:
            return
        f = self._file()
        f.flush()
        f.buffer.write(data)
        f.buffer.flush()
        self.last = data[-1:]

    def write(self, data: bytes) -> None:
        if self.raw is None:
            self.raw = self.tail is None and self._file().isatty()
        if not self.binary:
            try:
                self.text.decode(data)
            except UnicodeDecodeError:
                self.binary = True
        if self.raw or (self.binary and not self.tail):
            data, self.partial = self.partial + data, b""
            self._out(data)
            return
        *complete, partial = (self.partial + data).split(b"\n")
        if complete:
            lines = [(line if self.binary else collapse_cr(line)) + b"\n" for line in complete]
            if self.shown is not None and lines[0] == self.shown + b"\n":
                lines = lines[1:]  # the progress line ended in the state last shown
            self.shown, self.shown_at = None, None
            self._lines(lines)
        state = None
        if not self.binary:
            # The newest state a \r has completed (the text after the last \r may still be half-written).
            last = partial.rfind(b"\r")
            if last >= 0:
                state = partial[partial.rfind(b"\r", 0, last) + 1:last]
            # Only what follows the held line's last \r can survive the collapse (a trailing \r may still be a CRLF's): drop the rest now, so \r-only progress updates are not held and re-copied whole.
            partial = partial[partial.rfind(b"\r", 0, len(partial) - 1) + 1:]
        self.partial = partial
        now = time.monotonic()
        if state and not self.tail and state != self.shown and (self.shown_at is None or now - self.shown_at >= PROGRESS_EVERY):
            self._out(state + b"\n")
            self.shown, self.shown_at = state, now

    def _lines(self, lines: list[bytes]) -> None:
        if self.tail:
            self.lines.extend(lines)
        else:
            self._out(b"".join(lines))

    def flush(self) -> None:
        if self.partial and not (self.shown is not None and collapse_cr(self.partial) == self.shown):
            self._lines([self.partial if self.binary else collapse_cr(self.partial)])
        self.partial = b""
        if self.lines:
            self._out(b"".join(self.lines))
            self.lines.clear()


STDERR_SEPARATOR = b"cs: --- stderr (printed after stdout, not interleaved) ---\n"


class Output:
    """A command's output on the local stdout/stderr. A terminal gets the bytes as they are (it renders \\r progress bars itself); a file or pipe — what agents read — gets whole lines with \\r-overwrites collapsed, a partial line held until it completes (unless the output turns out to be binary) — a held progress line's current state shown as a line of its own at most every PROGRESS_EVERY seconds. With `tail`, each stream is buffered to its last N lines and printed by flush() — the in-cs replacement for `| tail`, which would replace the command's exit code."""

    def __init__(self, tail: int | None = None, raw: bool | None = None):
        self.sinks = {"O": _Sink("stdout", tail, raw), "E": _Sink("stderr", tail, raw)}

    def write(self, tag: str, data: bytes) -> None:
        if data:
            self.sinks[tag].write(data)

    def flush(self) -> None:
        o, e = self.sinks["O"], self.sinks["E"]
        # Only with --tail are the streams buffered and printed one after the other.
        both = o.tail and (o.lines or o.partial) and (e.lines or e.partial)
        o.flush()
        if both:
            self.separator()
        e.flush()

    def separator(self) -> None:
        """Marks where stderr starts when the two streams are printed one after the other instead of as they were written."""
        self.sinks["E"]._out(STDERR_SEPARATOR)

    def at_line_start(self) -> bool:
        """Whether both streams end with a complete line — a status message printed after a held progress line must not be glued onto it."""
        return all(s.last == b"\n" for s in self.sinks.values())


@dataclass
class Target:
    """A reachable session container: the session id and its reverse-tunnel socket on the login node."""
    session_id: str
    socket: str


class _Connection:
    """One ssh connection into the container running a stdin script."""

    def __init__(self, cluster, target: Target, script: bytes, marker: str, frame_timeout: float):
        self.frames = Frames(marker)
        self.frame_timeout = frame_timeout
        self.errf = tempfile.TemporaryFile()
        # Own session: the terminal's Ctrl-C must not reach ssh or the shared connection's ProxyCommand (we stop the client ourselves). stderr goes to a file, not a pipe: a ProxyCommand started by this ssh lives on with the shared connection and keeps its stderr open.
        self.proc = subprocess.Popen(cluster.access.container_argv(target.session_id, target.socket, "bash -l -s"), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errf, start_new_session=True)
        self.got_bytes = False
        self.timed_out = False
        self.error: str | None = None
        try:
            self.proc.stdin.write(script)
            self.proc.stdin.close()
        except OSError:
            pass

    def events(self) -> Iterator[tuple]:
        fd = self.proc.stdout.fileno()
        while True:
            ready, _, _ = select.select([fd], [], [], self.frame_timeout)
            if not ready:
                self.timed_out = True
                return
            data = os.read(fd, 65536)
            if not data:
                return
            self.got_bytes = True
            yield from self.frames.feed(data)

    def close(self) -> None:
        """Stop the ssh client (first, so a slow connection cannot deliver frames after its successor started) and collect its exit status and stderr."""
        if self.proc.poll() is None:
            try:
                self.proc.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc.stdout.close()
        self.errf.seek(0)
        self.stderr = self.errf.read().decode(errors="replace")
        self.errf.close()

    def before_login(self) -> bool:
        return _never_reached_sshd(self.proc.returncode, self.got_bytes, self.stderr)

    def reason(self) -> str:
        if self.error:
            return self.error
        if self.timed_out:
            return f"no data from the container for {_FRAME_TIMEOUT}s"
        lines = [l.strip() for l in self.stderr.splitlines() if l.strip() and not l.startswith(_SSH_NOISE)]
        return lines[-1] if lines else f"connection ended (ssh exit {self.proc.returncode})"


class StreamLost(ComputeSessionsError):
    """The container stayed unreachable through the whole backoff. `launched`: True (the command keeps running remotely), False (it was never started), None (unknown)."""

    def __init__(self, msg: str, launched: bool | None):
        super().__init__(msg)
        self.launched = launched


class NotConfirmed(SessionError):
    pass


def stream(cluster, target: Target, command_id: str, out: Output, *, command: str | None = None, follow: bool = True, offsets: tuple[int, int] = (0, 0),
           on_launch: Callable[[], None] | None = None, on_retry: Callable[[str], None] | None = None,
           round_seconds: int = _ROUND_SECONDS, frame_timeout: float = _FRAME_TIMEOUT, release: bool = False, after: list[str] | tuple[str, ...] = ()) -> tuple[str, int | None]:
    """Launch `command` (if given) and/or follow `command_id` until it settles; returns (status, exit_code) with status exited/killed/missing, or ("launched", None) when not following.

    After a failed connection, one helper request tells whether to keep trying the container: if the session is no longer active (its job is checked), or the command is already settled on the shared FS, the result comes from there (after the FS delay); otherwise the container is retried from the current offsets with backoff. Output comes only from the container while the session is active."""
    oo, eo = offsets
    launch = "pending" if command is not None else "done"   # pending → unknown (the connection broke after login) or done (S seen)
    failures = 0
    missing_checks = 0
    sid = target.session_id

    def emit(tag: str, off: int, data: bytes) -> None:
        nonlocal oo, eo
        cur = oo if tag == "O" else eo
        if off > cur:
            raise FrameError(f"gap in the {tag} stream ({off} > {cur})")
        data = data[cur - off:]  # overlap with bytes already received is dropped; offsets never move back
        out.write(tag, data)
        if tag == "O":
            oo = cur + len(data)
        else:
            eo = cur + len(data)

    def final_read() -> dict:
        time.sleep(3)  # compute-node writes reach the login node's view with up to ~2 s delay
        r = session.logs(cluster, [sid, command_id], since=(oo, eo))
        if follow:  # `cs run -d` prints the command id, nothing else
            out.write("O", r["out"])
            out.write("E", r["err"])
        return r

    while True:
        marker = f"__CS_STREAM_{secrets.token_hex(8)}__"
        script = container_script(command_id, marker, command=command if launch == "pending" else None, follow=follow, offsets=(oo, eo), round_seconds=round_seconds, release=release, after=after)
        conn = _Connection(cluster, target, script, marker, frame_timeout)
        ended = None
        try:
            for ev in conn.events():
                tag = ev[0]
                if tag in ("O", "E"):
                    emit(*ev)
                    failures = 0
                elif tag == "S" or (tag == "P" and launch == "unknown"):
                    # P after an uncertain launch: the command exists after all.
                    launch = "done"
                    if on_launch:
                        on_launch()
                    if not follow:
                        return "launched", None
                elif tag == "F":
                    raise SessionError(f"could not launch the command: {ev[1]}")
                elif tag == "H":
                    failures = 0
                elif tag == "X":
                    return "exited", int(ev[1]) if ev[1].lstrip("-").isdigit() else None
                elif tag == "K":
                    return "killed", None
                elif tag in ("T", "M"):
                    ended = tag
                    break
        except FrameError as exc:
            conn.error = str(exc)
        finally:
            conn.close()
        if ended == "T":
            failures = 0
            continue
        if ended == "M":
            if launch == "done":
                return "missing", None
            # The connection that carried the launch broke after login: the launcher may be mid-way. Look again for a few seconds, never relaunch.
            missing_checks += 1
            if missing_checks > 5:
                raise NotConfirmed(f"launch not confirmed: command {command_id} not found yet — check `cs commands {sid}` before running again")
            time.sleep(1)
            continue
        if launch == "pending" and not conn.before_login():
            launch = "unknown"
        failures += 1
        if failures >= _MAX_FAILURES:
            raise StreamLost(f"giving up after repeated connection failures ({conn.reason()})", {"done": True, "pending": False}.get(launch))
        try:
            r = cluster.call("logs", [sid, command_id], squeue=True, status_only=True)
        except RemoteError as exc:
            reason = f"{conn.reason()}; login node: {str(exc).splitlines()[0]}"
        else:
            reason = conn.reason()
            gone = r["session_status"] != "active"
            if launch == "pending" and gone:
                raise SessionError(f"session {sid} is {r['session_status']} — the command was not started")
            if launch != "pending" and (gone or r["status"] in ("exited", "killed")):
                r = final_read()
                if r["status"] == "missing":
                    if launch == "unknown":
                        raise NotConfirmed(f"launch not confirmed: command {command_id} not found — check `cs commands {sid}` before running again")
                    return "missing", None
                if launch == "unknown" and on_launch:
                    on_launch()
                return ("exited", r.get("exit_code")) if r["status"] == "exited" else ("killed", None)
        if on_retry:
            on_retry(reason)
        time.sleep(min(30, 2 ** failures))


def _container_capture(cluster, target: Target, remote_cmd: str, timeout: float = 60) -> tuple[int, str, str]:
    """Run a short command in the container and capture its output; a before-login failure (nothing ran) is retried."""
    for attempt in range(3):
        with tempfile.TemporaryFile() as errf:
            try:
                p = subprocess.run(cluster.access.container_argv(target.session_id, target.socket, remote_cmd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errf, start_new_session=True, timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                raise RemoteError(f"the session container did not answer within {timeout:g}s") from exc
            errf.seek(0)
            err = errf.read().decode(errors="replace")
        if not _never_reached_sshd(p.returncode, bool(p.stdout), err) or attempt == 2:
            return p.returncode, p.stdout.decode(errors="replace"), "".join(l for l in err.splitlines(keepends=True) if not l.startswith(_SSH_NOISE))
        time.sleep(1)
    raise AssertionError


# `cs kill` in the container. `cs_state` tells how to treat one command: only a plausible PID is ever signalled — `kill -- -1` would signal every process of the user, and 0 the killer's own group (a leading zero or an overlong number could alias them) — and its recorded PID names its process group only for a command of this activation (`.start` newer than sshd's pidfile — a re-activation's PID namespace reuses the numbers) whose heartbeat is fresh. After `.exit`, a still-populated group means survivors of the command (background children, or processes that ignored the signal that ended its shell; the heartbeat has a group of its own) — but only while no process holds the PID itself: the leader has exited, so that would be a newcomer. A signalled group must empty within 3 s to count as exited. `mt` falls back to BSD stat for the tests on macOS. With @ID@ empty, every command of this activation is signalled (internal venv snapshots excepted).
_KILL_SCRIPT = r"""mt() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo 0; }
cs_state() {
  local pid now
  [ -e "$1.pid" ] || { echo missing; return; }
  [ @SSHD_PID@ -nt "$1.start" ] && { echo old; return; }
  IFS= read -r pid < "$1.pid"
  case $pid in ''|*[!0-9]*|0*|1) pid= ;; esac
  [ ${#pid} -le 7 ] || pid=
  if [ -z "$pid" ]; then if [ -f "$1.exit" ]; then echo done; else echo starting; fi; return; fi
  now=$(date +%s)
  if [ -f "$1.exit" ]; then
    if ! kill -0 "$pid" 2>/dev/null && kill -0 -- "-$pid" 2>/dev/null; then echo "orphans $pid"; else echo done; fi
  elif [ $((now - $(mt "$1.pid"))) -lt @FRESH@ ]; then echo "live $pid"
  else echo dead
  fi
}
cd @LOGS@ || exit 0
if [ -n @ID@ ]; then
  set -- $(cs_state @ID@)
  case $1 in
    live)
      kill -@SIG@ -- "-$2" 2>/dev/null || { echo kill_failed; exit; }
      for _ in $(seq 1 30); do
        kill -0 -- "-$2" 2>/dev/null || { echo exited; exit; }
        sleep 0.1
      done
      if [ -f @ID@.exit ]; then echo survivors; else echo running; fi ;;
    orphans) if kill -@SIG@ -- "-$2" 2>/dev/null; then echo killed_orphans; else echo already_exited; fi ;;
    done) echo already_exited ;;
    *) echo "$1" ;;
  esac
  exit
fi
shopt -s nullglob
for p in *.pid; do
  c=${p%.pid}
  case $c in cmd_venvsnap_*) continue ;; esac
  set -- $(cs_state "$c")
  case $1 in live|orphans)
    if kill -@SIG@ -- "-$2" 2>/dev/null; then echo "$c,killed"; elif kill -0 -- "-$2" 2>/dev/null; then echo "$c,kill_failed"; fi ;;
  esac
done
"""


def kill_script(command_id: str, signal: str) -> str:
    validate_signal(signal)
    return _fill(_KILL_SCRIPT, {"@SSHD_PID@": shlex.quote(_SSHD_PID), "@FRESH@": str(_PID_FRESH_SECONDS), "@LOGS@": shlex.quote(_LOGS), "@ID@": shlex.quote(command_id), "@SIG@": signal})


def _kill(cluster, target: Target, command_id: str, signal: str) -> str:
    rc, stdout, stderr = _container_capture(cluster, target, f"bash -c {shlex.quote(kill_script(command_id, signal))}")
    if rc == 255 and not stdout.strip():
        raise RemoteError("could not reach the session container", stderr=stderr, exit_code=rc)
    return stdout


def kill_command(cluster, target: Target, command_id: str, signal: str = "TERM") -> str:
    """Signal one command's process group (the launcher's job control made its PID the PGID) and wait up to 3 s for it to go. Returns: exited | running (still alive after 3 s) | survivors (its shell exited, but processes of it are still alive after 3 s) | already_exited | killed_orphans (signalled what an exited command left running) | old (from an earlier activation) | dead (died without an exit) | missing | starting | kill_failed."""
    _validate_command_id(command_id)
    lines = _kill(cluster, target, command_id, signal).split()
    return lines[-1] if lines else "kill_failed"


def kill_all(cluster, target: Target, signal: str = "TERM") -> dict:
    """Signal every running command of this activation (and the orphans of finished ones). Returns {killed, errors}: command ids, and {command_id, state} for failed deliveries. A command whose process group is already gone (died untrappably, never wrote `.exit`) is skipped."""
    killed, errors = [], []
    for line in _kill(cluster, target, "", signal).splitlines():
        cid, _, outcome = line.strip().partition(",")
        if outcome == "killed":
            killed.append(cid)
        elif outcome:
            errors.append({"command_id": cid, "state": outcome})
    return {"killed": killed, "errors": errors}
