from __future__ import annotations

import secrets
import shlex

from compute_sessions.backends import Backend
from compute_sessions.errors import RemoteError, SessionError
from compute_sessions.models import RunResult, SessionStatus
from compute_sessions.paths import validate_command_id, validate_session_id
from compute_sessions.session import logs_with_access


# Accepts POSIX signal names (with or without SIG prefix) or numeric IDs. Whitelisted rather than regex-shaped: a typo like "FAKETERM" would otherwise be passed to `kill -FAKETERM` in the container and fail with a cryptic shell-level error rather than a clear validation message up front.
_VALID_SIGNALS: frozenset[str] = frozenset({
    "HUP", "INT", "QUIT", "ILL", "TRAP", "ABRT", "BUS", "FPE", "KILL",
    "USR1", "SEGV", "USR2", "PIPE", "ALRM", "TERM", "STKFLT", "CHLD",
    "CONT", "STOP", "TSTP", "TTIN", "TTOU", "URG", "XCPU", "XFSZ",
    "VTALRM", "PROF", "WINCH", "IO", "PWR", "SYS",
})


def _validate_signal(signal: str) -> None:
    if signal.isdigit():
        n = int(signal)
        if not 1 <= n <= 64:
            raise SessionError(f"invalid signal number: {signal!r}")
        return
    name = signal.removeprefix("SIG")
    if name not in _VALID_SIGNALS:
        raise SessionError(f"invalid signal: {signal!r}")


# How often the inner bash refreshes its `.pid` mtime while alive, so the runner-side idle monitor can detect a live command without a (namespace-invalid) `kill -0`. Must stay comfortably below the monitor's freshness window (runner.py: a `.pid` touched within the last 60s counts as busy) so a single missed touch or FS-mtime lag never makes a live command look idle.
_HEARTBEAT_SECONDS = 10

# The heartbeat subshell (and its in-flight `sleep`) lives in the command's process group for up to _HEARTBEAT_SECONDS after the command exits — so the kill tools' orphan check must ignore a process group whose `.exit` is younger than this, or a kill right after a fast command reports its heartbeat as `killed_orphans` (observed: `cs kill` on an idle session claiming `killed: cmd_…`). Real orphans persist; a later kill still reaps them.
_ORPHAN_GRACE_SECONDS = _HEARTBEAT_SECONDS + 5


# Path of the session's log directory *inside the container*. runner.py bind-mounts the host-side logs dir here. The host path (paths.logs_dir) is unreachable by its absolute name inside the container because the container gets an ephemeral home mount over $HOME.
_CONTAINER_LOGS_DIR = "/cs-logs"


def _new_command_id() -> str:
    return "cmd_" + secrets.token_hex(8)


def _build_detached_launch(command: str, command_id: str) -> str:
    """Build a `bash -lc '...'` invocation that launches the user command detached in the session container and returns immediately.

    Detachment strategy:
      - `set -m` (job control) in the outer bash: each backgrounded command starts in its own process group, so the inner bash's PID equals its PGID. The kill tool then targets `-$pid` to reach the whole tree.
      - `nohup` + stdin from /dev/null: inner bash ignores SIGHUP and has no tty, so our ssh disconnect doesn't propagate.
      - `disown $!`: outer bash drops the job, so when the outer script exits it neither waits for it nor sends SIGHUP via its job table.

    The inner bash writes its own `$$` to `.pid` before running the user command, then writes `$?` to `.exit` when done. The outer script checks `.exit` once and prints one final status line:

        DONE <exit_code>   # finished within the launch round-trip (instant commands)
        RUNNING            # still going; the caller streams via wait_for_command

    Idle-monitor heartbeat: the inner bash also backgrounds a loop that `touch`es `.pid` every `_HEARTBEAT_SECONDS` while it's alive. The runner-side idle monitor can't use the recorded PID for liveness — the container has a private PID namespace, so `$$` here is namespaced and meaningless to a host-side `kill -0`. Instead the monitor treats a `.pid` whose mtime is fresh (and which has no `.exit` yet) as a live command. The loop's `kill -0 $$` runs *inside* the container, where the PID is valid, so the touches stop within `_HEARTBEAT_SECONDS` of the bash dying by any means — normal exit, signal, or SIGKILL/OOM — and the session then idles out normally instead of either reaping a live command or leaking the allocation.

    Venv-hibernation epilogue: after `.exit` is recorded, the inner bash spawns `/cs-helpers/venv-snapshot.sh` detached — it archives `/cs-venv` when the command changed the venv's installed packages (see remote/runner.py). Guarded by an executable test: docker sources keep the venv on persistent local disk instead, so they bind no helper and the epilogue is a no-op there.

    Paths are individually quoted; the user command is not re-quoted (it must be a valid shell fragment, same contract as before).
    """
    base = f"{_CONTAINER_LOGS_DIR}/{command_id}"
    out = shlex.quote(f"{base}.out")
    err = shlex.quote(f"{base}.err")
    exit_f = shlex.quote(f"{base}.exit")
    pid_f = shlex.quote(f"{base}.pid")
    start_f = shlex.quote(f"{base}.start")
    lq_logs = shlex.quote(_CONTAINER_LOGS_DIR)

    # Inner (single-quoted via shlex below so $$/$? are expanded by the inner bash, not the outer launcher). Sequence:
    #   - Explicit TERM/INT/HUP traps so bash exits with 128+N instead of being killed mid-wait; otherwise the EXIT trap sees $?=0 for a signalled death.
    #   - EXIT trap writes $? to .exit on every exit path — normal completion or signal — so `logs` always sees a final exit_code.
    #   - Record our own PID (== PGID thanks to outer `set -m`) for the kill tool.
    #   - Heartbeat: background a loop that refreshes the `.pid` mtime every `_HEARTBEAT_SECONDS` while we're alive, so the runner-side idle monitor sees the live command (see _build_detached_launch docstring). `$$` in the subshell is the inner bash's PID (bash keeps $$ as the main shell PID inside ( )), so `kill -0 $$` stops the loop once this bash dies by any means. It shares our process group, so the kill tool's `kill -- -$pid` reaps it too. `touch` only bumps mtime — the recorded PID is preserved.
    #   - Run the user command with streams redirected to log files.
    inner = (
        f"trap 'exit 143' TERM; "
        f"trap 'exit 130' INT; "
        f"trap 'exit 129' HUP; "
        f"trap 'ec=$?; echo \"$ec\" > {exit_f}' EXIT; "
        f"echo $$ > {pid_f}; "
        f"( while kill -0 $$ 2>/dev/null; do touch {pid_f}; sleep {_HEARTBEAT_SECONDS}; done ) & "
        f"{{ {command}; }} > {out} 2> {err}; ec=$?; "
        # Record the exit code before the epilogue so it never delays `run`'s result; the EXIT trap re-writes the same value.
        f"echo \"$ec\" > {exit_f}; "
        # Venv-hibernation epilogue (slurm sources only — see docstring): spawn the snapshot helper detached (setsid → own process group, so the kill tool's `kill -- -$pid` works on it). It exits near-instantly when the venv is unchanged; signal deaths skip this line and the next command's epilogue catches up (the helper is state-based).
        f"if [ -x /cs-helpers/venv-snapshot.sh ]; then setsid /cs-helpers/venv-snapshot.sh </dev/null >/dev/null 2>&1 & fi; "
        f"exit $ec"
    )

    # `A && B && ... & C` backgrounds the whole list, not just the last command — so use `;` + `set -e` to run setup synchronously and only background the `nohup ... &` line.
    script = (
        f"set -em; "  # -e: abort on setup error; -m: new pgrp per bg job
        f"mkdir -p {lq_logs}; "
        f"cd /workdir; "
        # Create `.pid` synchronously here (the inner bash overwrites it with its real $$ a moment later) so it exists the instant this launcher returns. That way a *missing* `.pid` unambiguously means "no such command" rather than "launched but the async inner hasn't written it yet" — which lets logs()/list_commands distinguish `missing` from `running`/`killed`.
        # `.start` is the launch timestamp for list_commands: written once, never touched — unlike `.pid`, whose mtime is the heartbeat.
        f": > {out}; : > {err}; : > {pid_f}; : > {start_f}; rm -f {exit_f}; "
        f"nohup bash -c {shlex.quote(inner)} < /dev/null > /dev/null 2>&1 & "
        f"disown $! 2>/dev/null || true; "
        f"if [ -f {exit_f} ]; then printf 'DONE %s\\n' \"$(cat {exit_f})\"; "
        f"else printf 'RUNNING\\n'; fi"
    )
    return f"bash -lc {shlex.quote(script)}"


def _parse_launch_status(stdout: str) -> tuple[str, int | None]:
    """Parse the final status line emitted by the launch script."""
    status_line = ""
    for line in stdout.splitlines():
        line = line.strip()
        if line == "RUNNING" or line.startswith("DONE "):
            status_line = line
    if status_line.startswith("DONE "):
        rest = status_line[len("DONE "):].strip()
        try:
            return "exited", int(rest)
        except ValueError:
            return "exited", None
    if status_line == "RUNNING":
        return "running", None
    return "", None


def run(
    backend: Backend,
    session_id: str,
    command: str,
) -> RunResult:
    """Launch `command` detached inside the session container and return the launch snapshot.

    Requires an ACTIVE session (the CLI waits out PENDING itself before calling). Instant commands that finish within the launch round-trip come back complete (`status="exited"`, `exit_code` set); everything else returns `status="running"` with a `cursor` — the caller streams from there via `wait_for_command(since=cursor)`.
    """
    validate_session_id(session_id)
    # An empty command would expand to `{ ; } > out 2> err`, a bash syntax error, and surface as an opaque "could not launch command". Reject it up front with a clear message.
    if not command or not command.strip():
        raise SessionError("command must be a non-empty shell command")
    info = backend.read_info(session_id)

    # ACTIVE-without-sshd_port is a narrow window during activation (the runner flipped status before writing sshd_port) — report it like PENDING; a retry moments later succeeds.
    if info.status == SessionStatus.PENDING or (info.status == SessionStatus.ACTIVE and not info.sshd_port):
        raise SessionError(f"session {session_id} is still pending — the source has not allocated it yet (a busy cluster, a constrained request like a pinned gpu type, or a slow image pull can take a while). Wait with `cs show -w`, or `cs deactivate` to cancel.")
    if info.status != SessionStatus.ACTIVE:
        if info.status == SessionStatus.FAILED:
            raise SessionError(f"session {session_id} failed during activation — `cs show` has the failure log tail; `cs activate` retries")
        raise SessionError(f"session {session_id} is {info.status.value}; `cs activate` first")
    # First contact after an activation may still owe workdir setup (vast: fresh instances boot empty; the create-time sync couldn't run because no instance existed yet).
    backend.ensure_ready(info)
    access = backend.work_access(info)

    command_id = _new_command_id()
    target = backend.tunnel_target(info)

    launch_cmd = _build_detached_launch(command, command_id)
    try:
        res = access.exec_in_container(session_id, target, launch_cmd)
    except RemoteError as exc:
        # A wall-clock timeout here almost always means the launch went through and only the status echo was cut off. Surface the command_id so the caller can pick the command back up instead of re-issuing it (which would start a duplicate).
        if "timed out" in str(exc):
            raise SessionError(
                f"the launch call timed out, but the command was started detached and is likely "
                f"still running as {command_id!r} — check `cs logs {session_id} {command_id}` or "
                f"`cs commands` before re-issuing (a retry would launch a duplicate)"
            ) from exc
        raise
    status, exit_code = _parse_launch_status(res.stdout)
    # Read whatever has been captured so far. For instant commands this is the complete output.
    log_data = logs_with_access(access, session_id, command_id)
    if not status:
        # No DONE/RUNNING line came back — usually the command killed the launcher itself (e.g. `pkill -f <pat>` self-matches: the launcher embeds the command in its argv). The command still started, so trust the sentinel-derived status from logs(); only a `missing` command genuinely never launched.
        if log_data["status"] == "missing":
            raise SessionError(f"could not launch command: stdout={res.stdout!r} stderr={res.stderr!r}")
        status, exit_code = log_data["status"], log_data.get("exit_code")
    elif status == "running" and log_data["status"] in ("exited", "killed"):
        # The launch probe and the log read are separate round-trips; a fast command settles between them. The log read is the later word — reporting "running" here (with no cursor, since settled results carry none) made the CLI resume streaming from offset 0 and emit the whole output a second time.
        status, exit_code = log_data["status"], log_data.get("exit_code")
    return RunResult(
        command_id=command_id,
        status=status,
        exit_code=exit_code,
        stdout=log_data["stdout"],
        stderr=log_data["stderr"],
        cursor=log_data.get("cursor"),
    )


def kill_all_running(
    backend: Backend,
    session_id: str,
    signal: str = "TERM",
) -> dict:
    """Signal every command tracked as running for this session.

    Iterates `.pid` files in the session's logs dir. Each live pid is signalled via its process group (the same `kill -- -$pid` trick `kill_command` uses). A command that already recorded `.exit` is normally skipped — but if its process group still has members (backgrounded children that outlived the tracked command), those orphans are signalled too and the command is reported as killed. A pid whose process group is already gone (no `.exit`, empty group — i.e. the `killed` state from an untrappable SIGKILL/OOM/session death) is skipped, not reported as an error.
    Returns `{killed, skipped, errors}`: lists of `command_id`s by outcome.
    """
    validate_session_id(session_id)
    _validate_signal(signal)
    info = backend.read_info(session_id)
    if info.status != SessionStatus.ACTIVE or not info.sshd_port:
        raise SessionError(f"session {session_id} is not active (status={info.status.value}); cannot send signal to container process")
    access = backend.work_access(info)

    logs_dir = shlex.quote(f"{_CONTAINER_LOGS_DIR}")
    # Iterate inside the container so we hit the same pidspace `kill_command` targets. Emit one CSV line per command_id: `<command_id>,<outcome>`.
    script = (
        f"shopt -s nullglob; cd {logs_dir} || exit 0; "
        f"for pidfile in *.pid; do "
        f"  cmd=${{pidfile%.pid}}; "
        f"  pid=$(cat \"$pidfile\" 2>/dev/null); "
        # A recorded `.exit` normally means done — but backgrounded children of the command can outlive it in its process group. Signal the group if it still has members (the orphan case observed as leftover GPU-hogging processes); only report already_exited when the group is empty too. A fresh `.exit` is exempt — see _ORPHAN_GRACE_SECONDS.
        f"  if [ -f \"${{cmd}}.exit\" ]; then "
        f"    case $pid in ''|*[!0-9]*) printf '%s,already_exited\\n' \"$cmd\"; continue;; esac; "
        f"    if [ $(( $(date +%s) - $(stat -c %Y \"${{cmd}}.exit\") )) -lt {_ORPHAN_GRACE_SECONDS} ]; then printf '%s,already_exited\\n' \"$cmd\"; continue; fi; "
        f"    if kill -0 -- -$pid 2>/dev/null && kill -{signal} -- -$pid 2>/dev/null; then printf '%s,killed_orphans\\n' \"$cmd\"; "
        f"    else printf '%s,already_exited\\n' \"$cmd\"; fi; continue; "
        f"  fi; "
        # Only a plain numeric pid may reach `kill -- -$pid` — .pid lives in container-writable /cs-logs, so its contents are not trusted.
        f"  case $pid in ''|*[!0-9]*) printf '%s,no_pid\\n' \"$cmd\"; continue;; esac; "
        f"  if kill -{signal} -- -$pid 2>/dev/null; then printf '%s,killed\\n' \"$cmd\"; "
        # A failed signal followed by an empty process group (`kill -0` also fails) means the command already died untrappably — SIGKILL/OOM/session death — so it left a `.pid` but never wrote `.exit`. That's the `killed` state logs()/list_commands report, not a kill error: emit `already_dead` so it's skipped rather than surfaced as `kill_failed`. (Mirrors kill_command's empty-pgroup check.)
        f"  elif ! kill -0 -- -$pid 2>/dev/null; then printf '%s,already_dead\\n' \"$cmd\"; "
        f"  else printf '%s,kill_failed\\n' \"$cmd\"; fi; "
        f"done"
    )
    cmd = f"bash -lc {shlex.quote(script)}"
    res = access.exec_in_container(session_id, backend.tunnel_target(info), cmd)
    killed: list[str] = []
    skipped: list[str] = []
    errors: list[dict] = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if "," not in line:
            continue
        cid, outcome = line.split(",", 1)
        if outcome in ("killed", "killed_orphans"):
            killed.append(cid)
        # no_pid counts as skipped here, not an error: in a session-wide sweep it's just a command whose pid was never recorded (e.g. a launch the OOM-killer beat to the write) — nothing to deliver a signal to. kill_command keeps reporting it, where a bad COMMAND_ID is the likely cause.
        elif outcome in ("already_exited", "already_dead", "no_pid"):
            skipped.append(cid)
        else:
            errors.append({"command_id": cid, "state": outcome})
    return {"signal": signal, "killed": killed, "skipped": skipped, "errors": errors}


def kill_command(
    backend: Backend,
    session_id: str,
    command_id: str,
    signal: str = "TERM",
) -> dict:
    """Send a signal to a previously-launched command's process group.

    The launcher starts the inner bash under `set -m` (job control), so its PID equals the PGID of a fresh process group — `kill -- -$pid` delivers the signal to the whole tree. Requires an active session (container processes are only reachable via the session tunnel).

    After delivering the signal, waits up to 3 seconds for the command to actually exit before returning.

    Returns {command_id, signal, state} where `state` is one of:
        "exited"          — signal delivered and the command is gone (`.exit` recorded, or — for SIGKILL, which can't be trapped — its process group is empty)
        "running"         — signal delivered, command still alive after 3s
        "already_exited"  — command already finished, nothing left of its process group
        "killed_orphans"  — command already finished, but backgrounded children it left behind were still alive; its process group was signalled
        "no_pid"          — no pid file (bad command_id or wiped)
        "kill_failed"     — kill(2) returned non-zero (pid gone, perms, ...)
    """
    validate_session_id(session_id)
    validate_command_id(command_id)
    _validate_signal(signal)

    info = backend.read_info(session_id)
    if info.status != SessionStatus.ACTIVE or not info.sshd_port:
        raise SessionError(f"session {session_id} is not active (status={info.status.value}); cannot send signal to container process")
    access = backend.work_access(info)

    # Runs inside the container — use the in-container bind path, not the host path (the host logs dir is not reachable by its absolute name here).
    base = f"{_CONTAINER_LOGS_DIR}/{command_id}"
    pid_path = shlex.quote(f"{base}.pid")
    exit_path = shlex.quote(f"{base}.exit")

    script = (
        f"pid=$(cat {pid_path} 2>/dev/null || true); "
        # A recorded `.exit` normally means done — but backgrounded children can outlive the tracked command in its process group (they'd otherwise keep holding the GPU with no handle to reach them). Signal the group if it still has members; only report already_exited when it's empty. A fresh `.exit` is exempt — see _ORPHAN_GRACE_SECONDS.
        f"if [ -f {exit_path} ]; then "
        f"  case $pid in ''|*[!0-9]*) echo already_exited; exit 0;; esac; "
        f"  if [ $(( $(date +%s) - $(stat -c %Y {exit_path}) )) -lt {_ORPHAN_GRACE_SECONDS} ]; then echo already_exited; exit 0; fi; "
        f"  if kill -0 -- -$pid 2>/dev/null && kill -{signal} -- -$pid 2>/dev/null; then echo killed_orphans; "
        f"  else echo already_exited; fi; exit 0; "
        f"fi; "
        # Only a plain numeric pid may reach `kill -- -$pid` — .pid lives in container-writable /cs-logs, so its contents are not trusted.
        f"case $pid in ''|*[!0-9]*) echo no_pid; exit 0;; esac; "
        # Target the process group (negative pid). The launcher's `set -m` gave the inner bash its own process group, so $pid is the PGID of the whole command tree.
        f"if ! kill -{signal} -- -$pid 2>/dev/null; then echo kill_failed; exit 0; fi; "
        # Poll up to 3s (30 × 0.1s) for the command to be gone. A trappable signal (TERM/INT/HUP) lets the inner bash write `.exit`; SIGKILL can't be trapped and writes nothing, so also treat an empty process group as exited. `kill -0 -- -$pid` is valid here because this runs *inside* the container, where $pid lives (a host-side check would hit the wrong PID namespace).
        f"for _ in $(seq 1 30); do "
        f"if [ -f {exit_path} ]; then echo exited; exit 0; fi; "
        f"kill -0 -- -$pid 2>/dev/null || {{ echo exited; exit 0; }}; "
        f"sleep 0.1; "
        f"done; "
        f"echo running"
    )
    cmd = f"bash -lc {shlex.quote(script)}"
    res = access.exec_in_container(session_id, backend.tunnel_target(info), cmd)
    lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    state = lines[-1] if lines else "kill_failed"
    return {"command_id": command_id, "signal": signal, "state": state}
