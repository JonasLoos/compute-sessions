"""`cs` — the command-line frontend; a thin adapter over the core modules (session/run/backends).

Concurrent cs processes are safe and normal (parallel agents, backgrounded runs) — state lives on the sources. On top of the core the CLI adds session-id prefix matching, cwd-based session inference, and live output streaming with exit-code propagation.

Exit codes: `cs run` / `cs logs` exit with the remote command's code; 137 stands in for a command that died without recording an exit (OOM / kill / session death); `cs kill` exits 1 when the signal could not be delivered; 125 means cs itself failed (docker's convention for tool-level failure — a remote command can also exit 125, but the stderr message disambiguates); 2 is a usage error (click's code, shared by cs's own argument validation via `_usage`).
"""

from __future__ import annotations

import json
import re
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NoReturn

import click

from compute_sessions import run as run_mod
from compute_sessions import session as sess
from compute_sessions.backends import Backend, make_backend
from compute_sessions.config import load_config
from compute_sessions.errors import ComputeSessionsError, RemoteError, SessionError
from compute_sessions.models import SessionInfo, SessionStatus
from compute_sessions.project import project_id as derive_project_id
from compute_sessions.session import CreateRequest

EXIT_TOOL_FAILURE = 125
EXIT_KILLED = 137

# Server-side wait per streaming round. Each round is a poll loop of short ssh exchanges (not one blocking exec), so long rounds are transport-safe — they just mean fewer exchanges while the command is quiet.
_FOLLOW_ROUND_SECONDS = 60
_PENDING_ROUND_SECONDS = 30

# Config is loaded at import time so the per-source vocabulary can render into --help. A missing/broken config must not take `cs --help` down with it — commands surface the error when actually run.
try:
    _CFG = load_config()
    _BACKENDS: dict[str, Backend] = {
        name: make_backend(scfg, _CFG.control_path) for name, scfg in _CFG.sources.items()
    }
    _CFG_ERROR: Exception | None = None
except Exception as exc:
    _CFG, _BACKENDS, _CFG_ERROR = None, {}, exc


def _fail(msg: str) -> NoReturn:
    click.echo(f"cs: {msg}", err=True)
    sys.exit(EXIT_TOOL_FAILURE)


def _usage(msg: str) -> NoReturn:
    """A malformed invocation exits 2 (click's usage-error code, as documented in --help) — distinct from 125, so an agent can tell "I called it wrong" from "cs failed"."""
    click.echo(f"cs: {msg}", err=True)
    sys.exit(2)


def _require_config() -> None:
    if _CFG is None:
        _fail(f"config error: {_CFG_ERROR}")


def _backend_for(session_id: str) -> Backend:
    src = sess.source_of(session_id)
    if src not in _BACKENDS:
        raise SessionError(f"unknown source {src!r}; configured sources: {sorted(_BACKENDS)}")
    return _BACKENDS[src]


def _all_infos() -> list[SessionInfo]:
    rows: list[SessionInfo] = []
    for b in _BACKENDS.values():
        rows.extend(b.list_infos(None))
    return rows


# Bare hex tails ("8ec5" for hydra_8ec59e89a5) are how agents naturally abbreviate sessions — accepting them everywhere saves the retry observed when the source prefix was demanded.
_HEX_TOKEN_RE = re.compile(r"[0-9a-f]{4,10}")


def _norm_token(token: str) -> str:
    """Ids survive copy-paste: strip stray whitespace and lowercase the hex tail ("92A0 " → "92a0"). The source name keeps its case — it's a config key."""
    token = token.strip()
    name, sep, tail = token.partition("_")
    return f"{name}_{tail.lower()}" if sep else token.lower()


def _looks_like_session(token: str) -> bool:
    """Whether a positional token is meant as a session id: `<configured-source>_...`, or a bare hex tail prefix that matches a known session. Commands with trailing free-form args (run/upload/ls/...) use this to decide if their first arg is the session — and a token shaped like a session id that then fails to resolve is an error, never silently folded into the command/path args (a typo'd id must not run `hydra_typo echo hi` as a shell command). A bare hex token is the reverse: it only claims the session slot when a session actually matches, so ordinary commands/paths never get swallowed."""
    token = _norm_token(token)
    name, sep, _ = token.partition("_")
    if sep and name in _BACKENDS:
        return True
    if _HEX_TOKEN_RE.fullmatch(token):
        return any(i.session_id.partition("_")[2].startswith(token) for i in _all_infos())
    return False


def _note_other_project(info: SessionInfo) -> None:
    """A resolved session from another project is legal but often a mistyped prefix — one stderr line instead of silently reading a neighbour project's workdir."""
    if info.project_id != derive_project_id(Path.cwd()):
        click.echo(f"cs: note — {info.session_id} belongs to another project ({info.project_path or info.project_id})", err=True)


def _resolve_session(token: str | None) -> str:
    """Resolve an optional session token to a full id: exact id, unique prefix (cwd-project sessions win ties), or — with no token — the cwd project's only (or only live) session."""
    _require_config()
    if token is not None:
        token = _norm_token(token) or None  # an empty/whitespace token means "infer from cwd", same as no token
    if token is not None:
        try:
            src = sess.source_of(token)
        except SessionError:
            src = None
        if src in _BACKENDS:
            # Fast path: a full id needs one read, not a listing of every source.
            try:
                info = _BACKENDS[src].read_info(token)
                _note_other_project(info)
                return token
            except ComputeSessionsError:
                pass
        # Prefixes match against the full id or the bare hex tail ("8ec5" == "hydra_8ec5…").
        infos = _all_infos()
        matches = sorted({
            i.session_id for i in infos
            if i.session_id.startswith(token) or i.session_id.partition("_")[2].startswith(token)
        })
        if len(matches) > 1:
            # A prefix that's ambiguous globally but unique within the cwd project means the project's own session — short prefixes must not stop working just because other projects accumulated sessions.
            pid = derive_project_id(Path.cwd())
            own = sorted({i.session_id for i in infos if i.project_id == pid and i.session_id in matches})
            if len(own) == 1:
                return own[0]
            shown = ", ".join(matches[:6]) + (f", … +{len(matches) - 6} more" if len(matches) > 6 else "")
            raise SessionError(f"session prefix {token!r} is ambiguous: {shown}")
        if matches:
            _note_other_project(next(i for i in infos if i.session_id == matches[0]))
            return matches[0]
        raise SessionError(f"no session matches {token!r} — `cs list` shows known sessions")
    pid = derive_project_id(Path.cwd())
    mine = [i for i in _all_infos() if i.project_id == pid]
    if not mine:
        raise SessionError(f"no sessions for project {pid!r} — create one with `cs create`, or pass a session id")
    if len(mine) == 1:
        return mine[0].session_id
    live = [i for i in mine if i.status in (SessionStatus.ACTIVE, SessionStatus.PENDING)]
    if len(live) == 1:
        return live[0].session_id
    # Most-recent-first, capped listing: a project accumulates dead sessions over weeks, and the old wall of 16 ids (twice per compound command) buried the actual problem — usually "nothing is live".
    def _row(i: SessionInfo) -> str:
        t = i.to_enriched_json().get("time_in_status_seconds")
        return f"{i.session_id} ({i.status.value} {_dur(t)})"

    if live:
        raise SessionError(f"project {pid!r} has {len(live)} live sessions — pass one: {', '.join(_row(i) for i in live)}")
    mine.sort(key=lambda i: (lambda t: t if t is not None else float("inf"))(i.to_enriched_json().get("time_in_status_seconds")))
    more = f"; {len(mine) - 4} more via `cs list`" if len(mine) > 4 else ""
    raise SessionError(
        f"project {pid!r} has {len(mine)} sessions but none is live — pass one, most recent first: "
        f"{', '.join(_row(i) for i in mine[:4])}{more}"
    )


def _split_leading_session(args: tuple[str, ...]) -> tuple[str | None, list[str]]:
    """Split an optional leading session token off a positional-args tuple."""
    rest = list(args)
    if rest and _looks_like_session(rest[0]):
        return rest[0], rest[1:]
    return None, rest


def _dur(seconds: object) -> str:
    if not isinstance(seconds, (int, float)):
        return "-"
    s = int(seconds)
    if s >= 86400:
        return f"{s // 86400}d{(s % 86400) // 3600}h"
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join([fmt.format(*headers)] + [fmt.format(*r) for r in rows])


def _emit(res: dict) -> None:
    """Write a logs/wait_for_command result's streams through to the local streams."""
    out, err = res.get("stdout", ""), res.get("stderr", "")
    if out:
        sys.stdout.write(out)
        sys.stdout.flush()
    if err:
        sys.stderr.write(err)
        sys.stderr.flush()


class _TailEmitter:
    """Emit-callable that buffers each stream to its last N lines and replays them when the command settles — the in-cs replacement for `cs run … | tail -N`, which caps context at the cost of the exit code (the pipeline returns tail's status; observed as a wall-clock-killed job reading 'exit 0') and SIGPIPEs the stream on early-exit filters."""

    def __init__(self, n: int):
        self.n = n
        self._out = ""
        self._err = ""

    def __call__(self, res: dict) -> None:
        trim = lambda s: "".join(s.splitlines(keepends=True)[-self.n:])
        self._out = trim(self._out + res.get("stdout", ""))
        self._err = trim(self._err + res.get("stderr", ""))

    def flush(self) -> None:
        _emit({"stdout": self._out, "stderr": self._err})
        self._out = self._err = ""


def _exit_for(status: str, exit_code: int | None, backend: Backend | None = None, session_id: str | None = None) -> NoReturn:
    if status == "exited":
        sys.exit(exit_code if exit_code is not None else 1)
    if status == "killed":
        click.echo(f"cs: command died without recording an exit — {_death_cause(backend, session_id)}", err=True)
        sys.exit(EXIT_KILLED)
    _fail(
        f"no trace of this command (status={status})"
        + (f" — `cs commands {session_id}` lists recorded commands" if session_id else "")
    )


def _death_cause(backend: Backend | None, session_id: str | None) -> str:
    """Disambiguate a silent command death by checking whether the session survived it: a live session means the command alone was killed (usually the OOM-killer), a dead one means the session took the command down with it. One extra round-trip, only on this rare path; show() also reconciles a gone job so a session the OOM-killer took out whole doesn't read as still active."""
    if backend is None or session_id is None:
        return "OOM, kill, or session death"
    try:
        info = sess.show(backend, session_id)["info"]
    except Exception:
        return "OOM, kill, or session death"
    if info["status"] == "active":
        # --mem only exists on sources with a scheduler/renter; suggesting it on a docker source sends the agent into a resource-args error.
        mem_hint = "" if backend.source.type == "docker" else " or reactivate with more --mem"
        return f"the session is still active, so most likely the OOM-killer (or an explicit kill) — retry with a smaller workload{mem_hint}"
    wall = _walltime_hit(info)
    if wall:
        return wall
    return f"the session is {info['status']}, so it ended (idle timeout, deactivate, or job death) and took the command with it"


def _walltime_hit(info: dict) -> str | None:
    """Slurm partition names encode their wall-clock limit as a suffix (gpu-2h, gpu-7d). An activation that lasted right up to that limit died to the scheduler's kill, not OOM or idling — name it, because agents otherwise re-run the same job on the same too-short partition (observed twice: multi-hour chains killed at exactly 5h00m, read as generic session death)."""
    partition = (info.get("resources") or {}).get("partition") or ""
    m = re.search(r"(\d+)([hd])$", partition)
    if not m:
        return None
    limit = int(m.group(1)) * (3600 if m.group(2) == "h" else 86400)
    try:
        activated = datetime.strptime(info["last_activated_at"], "%Y-%m-%dT%H:%M:%SZ")
        ended = datetime.strptime(info["last_deactivated_at"], "%Y-%m-%dT%H:%M:%SZ")
    except (KeyError, TypeError, ValueError):
        return None
    if (ended - activated).total_seconds() < limit * 0.97:
        return None
    return (
        f"the session hit the {partition} partition's {m.group(1)}{m.group(2)} wall-clock limit "
        f"({_dur(limit)} after activation) and was killed with the command — reactivate on a longer partition for jobs this size"
    )


def _res_summary(resources: dict) -> str:
    """Compact echo of an activation's resolved resources. Printed on create/activate so a silently-reused partition is visible — activate keeping the previous gpu-2h partition killed a 4h job at 2h02m without the agent ever seeing which partition it was on."""
    parts = [str(resources[k]) for k in ("partition", "gpu_type") if resources.get(k)]
    if resources.get("gpus") is not None:
        parts.append(f"{resources['gpus']} gpu")
    if resources.get("mem"):
        parts.append(f"{resources['mem']}G")
    return ", ".join(parts)


def _stream_retry(failures: int, exc: RemoteError, resume_hint: str) -> None:
    """Backoff for a transient connection failure inside a streaming loop; re-raises after 8 consecutive failures. A network blip mid-training-run must not read as a failure — the remote side keeps going either way."""
    if failures >= 8:
        click.echo(f"cs: giving up after repeated connection failures — {resume_hint}", err=True)
        raise exc
    click.echo(f"cs: connection hiccup ({str(exc).splitlines()[0]}); retrying…", err=True)
    time.sleep(min(30, 2 ** failures))


def _follow(backend: Backend, session_id: str, command_id: str, cursor: str, emit=_emit) -> tuple[str, int | None]:
    """Stream a running command's output until it settles; returns (status, exit_code). Each round blocks server-side until new output appears or the command settles.

    A `missing` result is retried like a connection failure, not returned: the probe reads sentinel files over ssh, and a transient failure there is indistinguishable from genuinely wiped logs — but this command was just seen running, so transient is overwhelmingly more likely. Killing the stream here was observed to fail multi-hour jobs as exit 125 while the remote command kept running for hours.
    """
    failures = 0
    while True:
        try:
            res = sess.wait_for_command(
                backend, session_id, command_id,
                timeout_seconds=_FOLLOW_ROUND_SECONDS, since=cursor,
            )
            if res["status"] == "missing":
                raise RemoteError(f"command {command_id} probe came back empty (status=missing)")
            failures = 0
        except RemoteError as exc:
            failures += 1
            _stream_retry(failures, exc, f"reattach with: cs logs -f {session_id} {command_id}")
            continue
        emit(res)
        if res["status"] == "running":
            cursor = res.get("cursor") or cursor
            continue
        return res["status"], res.get("exit_code")


def _stream_and_exit(backend: Backend, session_id: str, command_id: str, cursor: str, emit=_emit) -> NoReturn:
    flush = getattr(emit, "flush", lambda: None)
    # The finally-flush covers the giving-up RemoteError path too — buffered --tail output must not vanish exactly when the stream dies (flush is idempotent, so the explicit calls below stay for message ordering).
    try:
        try:
            status, exit_code = _follow(backend, session_id, command_id, cursor, emit=emit)
        except KeyboardInterrupt:
            flush()
            click.echo(
                f"\ncs: detached — the command keeps running. Reattach: cs logs -f {session_id} {command_id}; stop: cs kill {session_id} {command_id}",
                err=True,
            )
            sys.exit(130)
        flush()
        _exit_for(status, exit_code, backend, session_id)
    finally:
        flush()


def _wait_active(backend: Backend, session_id: str) -> dict:
    """Block until the session leaves PENDING, echoing queue state to stderr; exits (125) on failure. Returns the final show() snapshot. Transient connection failures are retried like the streaming loops — an ssh blip during a long queue wait must not abort a create/activate whose job then allocates unwatched (observed: an orphaned H100 idling away after one broken pipe at 7m45s of queueing)."""
    started = time.monotonic()
    failures = 0
    rounds = 0
    last_reason: str | None = None
    try:
        while True:
            try:
                detail = sess.show(backend, session_id, wait_seconds=_PENDING_ROUND_SECONDS)
                failures = 0
            except RemoteError as exc:
                failures += 1
                _stream_retry(failures, exc, f"keep waiting with: cs show -w {session_id}")
                continue
            status = detail["info"]["status"]
            if status == "active":
                gpu = detail["info"].get("gpu")
                click.echo(f"cs: {session_id} active" + (f" — {gpu}" if gpu else ""), err=True)
                return detail
            if status != "pending":
                tail = detail.get("failure_log_tail")
                if tail:
                    click.echo(tail.rstrip(), err=True)
                _fail(f"session {session_id} is {status}")
            reason = (detail.get("job") or {}).get("reason")
            # Every round for the first ~2 minutes, then only on a changed reason or every ~5 minutes — a 62-minute queue wait used to spam 124 identical lines into whatever captures stderr.
            rounds += 1
            if rounds <= 4 or reason != last_reason or rounds % 10 == 0:
                click.echo(
                    f"cs: waiting for allocation ({_dur(time.monotonic() - started)})"
                    + (f" — {reason}" if reason else ""),
                    err=True,
                )
            last_reason = reason
    except KeyboardInterrupt:
        click.echo(
            f"\ncs: still pending — keep waiting: cs show -w {session_id}; cancel: cs deactivate {session_id}",
            err=True,
        )
        sys.exit(130)


# Config-derived --help fragments (callers guard on _CFG being loaded). Source-type-specific lines are gated on that type actually being configured — notes about backends the user doesn't have would just burn attention or actively mislead.
def _source_types() -> tuple[bool, bool]:
    types = {s.type for s in _CFG.sources.values()}
    return "slurm" in types, "vast" in types


def _resource_arg_lines() -> list[str]:
    """Per-flag doc lines for the create/activate/clone resource flags."""
    slurm, vast = _source_types()
    lines: list[str] = []
    if slurm:
        lines.append("--partition: slurm only (suffix = hard wall-clock limit that kills the whole session when it elapses; suffix-less partitions like *-test have short limits of their own — `cs show`'s time_left shows what remains; omit for the source default).")
    if slurm or vast:
        gpus_range = " / ".join(r for cond, r in ((slurm, "slurm 0-16 (cpu-* partitions always run with 0)"), (vast, "vast 1-8")) if cond)
        lines.append(f"--gpus: {gpus_range}; default 1.")
        why_omit = " / ".join(w for cond, w in ((slurm, "faster allocation"), (vast, "the cheapest offer")) if cond)
        lines.append(f"--gpu-type: omit unless the task truly needs a specific GPU — any-gpu requests get {why_omit}. To pin, use a type from the source's list (OR with `|`). The actually-assigned model shows in `cs list`/`cs show` (gpu).")
        mem_kinds = "; ".join(m for cond, m in ((slurm, "slurm: allocation (omit for the source default)"), (vast, "vast: minimum system RAM of the rented machine")) if cond)
        lines.append(f"--mem: GB — {mem_kinds}.")
    else:
        lines.append("--partition/--gpus/--gpu-type/--mem: not used by any configured source — leave unset.")
    lines.append(
        "--idle-timeout: auto-shutdown after N minutes with no live `run` command and no workdir file ops (sync/upload/download/ls); reading logs does not reset the timer."
        + (" On vast this destroys the instance (billing stops; workdir is lost)." if vast else "")
    )
    return lines


def _container_notes() -> str:
    """How to work inside the session container — rendered into `cs run --help`."""
    _, vast = _source_types()
    venv_note = " (not on vast — fresh instance each activation)" if vast else ""
    return (
        "No system python — use `uv run`. Write scratch to /workdir. The shared HF cache at `~/.cache/huggingface` should be used for hf downloads. "
        f"The project venv lives at /cs-venv/venv (`uv` targets it automatically — no `uv venv` needed, just `uv sync`/`uv pip install`) and persists across deactivate/activate{venv_note}. "
        "Torch: the driver runs wheels up to CUDA $CS_CUDA_VERSION (env var set in every GPU session) — pin torch's index to match (e.g. `[tool.uv.sources]` → https://download.pytorch.org/whl/cu128); the default wheels may target newer CUDA and fail to run."
    )


def _resource_epilog() -> str:
    if _CFG is None:
        return ""
    # One line per configured source (name + its backend's resource vocabulary), then the per-flag lines.
    source_lines = [
        f"- {name}{' (default)' if name == _CFG.default_source else ''}: {b.describe()}"
        for name, b in _BACKENDS.items()
    ]
    # \b disables click's paragraph rewrapping for the block that follows.
    return "\b\nResource flags — valid values depend on the source:\n" + "\n".join(f"  {line}" for line in source_lines + _resource_arg_lines())


def _resource_options(f):
    f = click.option("--idle-timeout", "idle_timeout_minutes", type=int, default=20, show_default=True, metavar="MIN", help="Auto-shutdown after MIN idle minutes.")(f)
    f = click.option("--mem", type=int, default=None, metavar="GB", help="Memory in GB.")(f)
    f = click.option("--gpu-type", default=None, help="Pin GPU type(s), OR with '|'.")(f)
    f = click.option("--gpus", type=int, default=None, help="GPU count.")(f)
    f = click.option("--partition", default=None, help="Slurm partition.")(f)
    return f


@click.group(
    # help_option_names is inherited by subcommand contexts, so -h works everywhere.
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Manage compute sessions — interactive container jobs on your configured compute sources. Most commands take the session as an optional first argument: a full id, a unique prefix (the bare hex tail works: '8ec5' for hydra_8ec5…), or nothing (the cwd project's only (live) session is used).",
    epilog="""\
\b
Lifecycle: pending -> active -> inactive (cs activate re-enters) | failed
(cs show has the failure log tail; cs activate retries). File commands
(sync, upload, download, ls, logs, commands) work in any state.

\b
Files: `cs sync` mirrors the project's non-ignored files into the workdir
(one-way, local->source); .gitignored files (datasets, weights, .env) need
`cs upload`. Remote paths are relative to the workdir.

\b
Exit codes: cs run/logs exit with the remote command's code; 137 = command
died without recording an exit (OOM/kill/session death); cs kill exits 1 if
the signal was not delivered; 125 = cs itself failed; 2 = usage error.
""",
)
@click.version_option(package_name="compute-sessions", prog_name="cs")
def cli() -> None:
    pass


@cli.command("list")
@click.option("--all", "all_projects", is_flag=True, help="All projects, not just the cwd's.")
@click.option("--json", "as_json", is_flag=True, help="JSON rows instead of a table.")
def list_cmd(all_projects: bool, as_json: bool) -> None:
    """List sessions (the cwd project's by default)."""
    _require_config()
    pid = None if all_projects else derive_project_id(Path.cwd())
    # The project id derives from the git origin and can re-key when a remote is added — orphaning every earlier session from a plain `cs list` (observed: 10 sessions holding all of a project's artifacts invisible, and an agent planning a 40-hour re-upload of data that was already there). Filter client-side and flag same-directory sessions under other ids.
    cwd = str(Path.cwd().resolve())
    former: dict[str, int] = {}
    rows = []
    for b in _BACKENDS.values():
        for i in b.list_infos(None):
            if pid is not None and i.project_id != pid:
                if i.project_path == cwd:
                    former[i.project_id] = former.get(i.project_id, 0) + 1
                continue
            row = i.to_agent_json()
            if all_projects:
                row["project_id"] = i.project_id
            rows.append(row)
    # Live sessions first, most recently changed first within each status — backend iteration order is arbitrary.
    status_rank = {"active": 0, "pending": 1, "inactive": 2, "failed": 3}
    rows.sort(key=lambda r: (status_rank.get(r["status"], 9), r.get("time_in_status_seconds") or 0))
    for fid, n in sorted(former.items()):
        click.echo(f"cs: {n} session(s) for this directory live under a former project id {fid!r} — `cs list --all` shows them", err=True)
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        click.echo(f"no sessions{'' if all_projects else f' for project {pid!r} (try --all)'}", err=True)
        return
    headers = ["SESSION", "STATUS", "IN-STATUS", "GPU", "RESOURCES", "IDLE-IN"]
    if all_projects:
        headers.append("PROJECT")
    table_rows = []
    for r in rows:
        res = " ".join(
            str(r[k]) for k in ("partition", "gpus", "gpu_type", "mem") if r.get(k) not in (None, "")
        )
        line = [
            r["session_id"],
            r["status"],
            _dur(r.get("time_in_status_seconds")),
            r.get("gpu") or "-",
            res or "-",
            _dur(r.get("seconds_until_idle_deactivate")),
        ]
        if all_projects:
            line.append(r.get("project_id", "-"))
        table_rows.append(line)
    click.echo(_table(headers, table_rows))


@cli.command(epilog=_resource_epilog())
@click.option("--source", default=None, help=f"Compute source (default: {_CFG.default_source if _CFG else 'from config'}).")
@_resource_options
@click.option("--no-wait", is_flag=True, help="Submit and return immediately instead of waiting for activation (block later with cs show -w).")
def create(source: str | None, partition: str | None, gpus: int | None, gpu_type: str | None, mem: int | None, idle_timeout_minutes: int, no_wait: bool) -> None:
    """Create a session for the cwd project and start it. Prints the session id on stdout."""
    _require_config()
    name = source or _CFG.default_source
    if name not in _BACKENDS:
        _fail(f"unknown source {name!r}; configured sources: {sorted(_BACKENDS)}")
    req = CreateRequest(
        partition=partition, gpus=gpus, gpu_type=gpu_type, mem=mem,
        idle_timeout_minutes=idle_timeout_minutes,
    )
    info = sess.create(_BACKENDS[name], _CFG, req, Path.cwd())
    res = _res_summary(info.resources)
    click.echo(f"cs: created {info.session_id} (pending{' — ' + res if res else ''})", err=True)
    if not no_wait:
        _wait_active(_BACKENDS[name], info.session_id)
    click.echo(info.session_id)


@cli.command(short_help="Start or resume a session.", epilog=_resource_epilog())
@click.argument("session", required=False)
@_resource_options
@click.option("--no-wait", is_flag=True, help="Submit and return immediately instead of waiting for activation (block later with cs show -w).")
def activate(session: str | None, partition: str | None, gpus: int | None, gpu_type: str | None, mem: int | None, idle_timeout_minutes: int, no_wait: bool) -> None:
    """Start (or resume) a session; resources are chosen per activation."""
    sid = _resolve_session(session)
    b = _backend_for(sid)
    info = sess.activate(
        b, sid,
        partition=partition, gpus=gpus, gpu_type=gpu_type, mem=mem,
        idle_timeout_minutes=idle_timeout_minutes,
    )
    res = _res_summary(info.resources)
    click.echo(f"cs: {sid} pending{' — ' + res if res else ''}", err=True)
    if not no_wait:
        _wait_active(b, sid)
    click.echo(sid)


# Rendered per config: the vast billing/data warning would mislead on a setup with no vast source.
_VAST_CONFIGURED = _CFG is not None and any(s.type == "vast" for s in _CFG.sources.values())


@cli.command(
    short_help="Stop a session (workdir persists; activate re-enters).",
    help="Stop a session (workdir and logs persist; re-enter with `cs activate`)."
    + (" On vast sources this DESTROYS the rented instance — billing stops, the workdir is gone; download results first." if _VAST_CONFIGURED else ""),
)
@click.argument("session", required=False)
def deactivate(session: str | None) -> None:
    sid = _resolve_session(session)
    info = sess.deactivate(_backend_for(sid), sid)
    click.echo(f"{sid} {info.status.value}")


@cli.command(short_help="Clone a session into a new one on the same source.", epilog=_resource_epilog())
@click.argument("session", required=False)
@_resource_options
@click.option("--no-wait", is_flag=True, help="Submit and return immediately instead of waiting for activation (block later with cs show -w).")
def clone(session: str | None, partition: str | None, gpus: int | None, gpu_type: str | None, mem: int | None, idle_timeout_minutes: int, no_wait: bool) -> None:
    """Clone a session's workdir into a new session on the same source (server-side copy). Prints the new session id on stdout."""
    sid = _resolve_session(session)
    b = _backend_for(sid)
    req = CreateRequest(partition=partition, gpus=gpus, gpu_type=gpu_type, mem=mem, idle_timeout_minutes=idle_timeout_minutes)
    info = sess.clone(b, _CFG, sid, req)
    res = _res_summary(info.resources)
    click.echo(f"cs: created {info.session_id} (pending{' — ' + res if res else ''})", err=True)
    if not no_wait:
        _wait_active(b, info.session_id)
    click.echo(info.session_id)


@cli.command(short_help="Full session detail: job state, health, failure log.")
@click.argument("session", required=False)
@click.option("-w", "--wait", is_flag=True, help="Block while the session is pending.")
@click.option("--json", "as_json", is_flag=True, help="JSON output.")
def show(session: str | None, wait: bool, as_json: bool) -> None:
    """Full detail for one session: resources, live job state, health, failure log tail."""
    sid = _resolve_session(session)
    b = _backend_for(sid)
    # Same anti-spam cadence as _wait_active: every round for ~2 minutes, then only on a changed queue reason or every ~5 minutes — `cs show -w` through a long queue wait otherwise floods stderr with identical lines.
    started = time.monotonic()
    rounds = 0
    last_reason: str | None = None
    while True:
        detail = sess.show(b, sid, wait_seconds=_PENDING_ROUND_SECONDS if wait else 0)
        if not (wait and detail["info"]["status"] == "pending"):
            break
        reason = (detail.get("job") or {}).get("reason")
        rounds += 1
        if rounds <= 4 or reason != last_reason or rounds % 10 == 0:
            click.echo(
                f"cs: still pending ({_dur(time.monotonic() - started)})"
                + (f" — {reason}" if reason else ""),
                err=True,
            )
        last_reason = reason
    if as_json:
        click.echo(json.dumps(detail, indent=2))
        return
    for k, v in detail["info"].items():
        if v is None:
            continue
        click.echo(f"{k}: {json.dumps(v) if isinstance(v, (dict, list)) else v}")
    for key in ("job", "health"):
        if key in detail:
            click.echo(f"{key}: " + ", ".join(f"{k}={v}" for k, v in detail[key].items() if v is not None))
    if "hint" in detail:
        click.echo(f"hint: {detail['hint']}")
    if "failure_log_tail" in detail:
        click.echo("failure log tail:\n" + detail["failure_log_tail"].rstrip())


# ignore_unknown_options keeps unknown flags (`--lr 0.1`) in ARGS; allow_interspersed_args=False stops option parsing at the first positional, so a `-d`/`-h` INSIDE the user command stays part of the command instead of being captured by click.
@cli.command(context_settings={"ignore_unknown_options": True, "allow_interspersed_args": False}, epilog=("\b\nContainer environment:\n" + _container_notes() if _CFG else None))
@click.argument("args", nargs=-1, required=True)
@click.option("-d", "--detach", is_flag=True, help="Launch and print the command id instead of streaming.")
@click.option("--tail", "tail_n", type=int, default=None, metavar="N", help="Print only the last N lines of each stream, once the command ends (output is withheld until then). The full log stays retrievable via cs logs.")
def run(args: tuple[str, ...], detach: bool, tail_n: int | None) -> None:
    """Run a command in a session: cs run [SESSION] CMD [ARGS...]

    Streams output until the command exits and exits with its code; Ctrl-C detaches (the command keeps running). Runs in /workdir via the container shell. A single CMD argument is a raw shell fragment (pipes/redirects work): cs run 'python train.py 2>&1 | tee train.log'. Multiple arguments are quoted individually (exec-style): cs run python -c 'print("hi")'.

    Never pipe cs run's OWN output through filters like `| tail` / `| grep -q` — the pipeline's exit code replaces the command's, and an early-closing filter kills the stream. To cap output, use --tail N.

    Long-running commands: prefer running cs run itself in the background (shell '&', or your harness's background-task facility) — it streams, survives connection blips, and exits with the command's code. Use -d when the local cs process may not outlive the job (laptop sleep) or for servers, then follow with cs logs -f. Killing cs never kills the remote command.
    """
    _require_config()
    token, rest = _split_leading_session(args)
    if not rest:
        _usage("no command given — usage: cs run [SESSION] CMD [ARGS...]")
    sid = _resolve_session(token)
    b = _backend_for(sid)
    # One arg = raw shell fragment; several = exec-style (each arg quoted). Plain " ".join would let the remote shell re-split args containing spaces.
    command = rest[0] if len(rest) == 1 else shlex.join(rest)
    info = b.read_info(sid)
    if info.status == SessionStatus.PENDING:
        _wait_active(b, sid)
    elif info.status == SessionStatus.FAILED:
        _fail(f"session {sid} failed to activate — `cs show {sid}` has the log tail; `cs activate {sid}` retries")
    elif info.status != SessionStatus.ACTIVE:
        _fail(f"session {sid} is {info.status.value} — `cs activate {sid}` first (resources are chosen per activation)")
    result = run_mod.run(b, sid, command)
    click.echo(f"cs: {result.command_id} on {sid}", err=True)
    if detach:
        click.echo(result.command_id)
        return
    emit = _TailEmitter(tail_n) if tail_n else _emit
    # The launch snapshot's output must be emitted here: the cursor already covers those bytes, so the streaming loop below will never return them again.
    emit({"stdout": result.stdout, "stderr": result.stderr})
    if result.status != "running":
        getattr(emit, "flush", lambda: None)()
        _exit_for(result.status, result.exit_code, b, sid)
    _stream_and_exit(b, sid, result.command_id, result.cursor or "0:0", emit=emit)


@cli.command()
@click.argument("args", nargs=-1)
@click.option("-f", "--follow", is_flag=True, help="Stream until the command exits; exit with its code.")
@click.option("--tail", "tail_n", type=int, default=None, metavar="N", help="Only the last N lines of each stream.")
def logs(args: tuple[str, ...], follow: bool, tail_n: int | None) -> None:
    """Read a command's output: cs logs [SESSION] [COMMAND_ID]

    COMMAND_ID defaults to the session's newest RUNNING command, else its most recent one (`cs commands` lists them). stdout/stderr go to the local stdout/stderr; a finished command's exit code becomes the exit code.
    """
    token, rest = _split_leading_session(args)
    if len(rest) > 1:
        _usage("too many arguments — usage: cs logs [SESSION] [COMMAND_ID]")
    sid = _resolve_session(token)
    b = _backend_for(sid)
    command_id = rest[0] if rest else None
    if command_id is None:
        # Prefer the newest running command over the newest command: a short-lived probe launched after a long job must not hijack the default (observed: `cs logs` returning a dead 2-second probe while the 4-hour fit ran on, costing several turns of confusion). Say which one was picked and in what state, so a surprising default is visible.
        cmds = sess.list_commands(b, sid, limit=20).get("commands") or []
        if not cmds:
            _fail(f"no commands recorded for {sid}")
        chosen = next((c for c in cmds if c["status"] == "running"), cmds[0])
        command_id = chosen["command_id"]
        state = chosen["status"] + (f" {chosen['exit_code']}" if chosen.get("exit_code") is not None else "")
        click.echo(f"cs: {command_id} ({state})", err=True)
    res = sess.logs(b, sid, command_id, max_lines=tail_n)
    _emit(res)
    if res["status"] == "running":
        if not follow:
            click.echo(f"cs: still running — follow with: cs logs -f {sid} {command_id}", err=True)
            sys.exit(0)
        _stream_and_exit(b, sid, command_id, res["cursor"])
    _exit_for(res["status"], res.get("exit_code"), b, sid)


@cli.command()
@click.argument("args", nargs=-1)
@click.option("-s", "--signal", "sig", default="TERM", show_default=True, help="Signal to send (TERM, KILL, INT, ... or numeric).")
def kill(args: tuple[str, ...], sig: str) -> None:
    """Stop command(s): cs kill [SESSION] [COMMAND_ID]

    Without COMMAND_ID, signals every running command in the session. Targets the whole process group, so backgrounded children are covered. Exits 1 if the signal could not be delivered (no pid recorded, or kill failed).
    """
    token, rest = _split_leading_session(args)
    if len(rest) > 1:
        _usage("too many arguments — usage: cs kill [SESSION] [COMMAND_ID]")
    sid = _resolve_session(token)
    b = _backend_for(sid)
    if rest:
        out = run_mod.kill_command(b, sid, rest[0], sig)
        # "exited" alone reads as "it had already exited before the kill" — that case is "already_exited"; disambiguate the delivered-signal outcomes.
        label = {"exited": "signalled — exited", "running": "signalled — still running after 3s"}.get(out["state"], out["state"])
        click.echo(f"{out['command_id']}: {label}")
        if out["state"] in ("no_pid", "kill_failed"):
            sys.exit(1)
    else:
        out = run_mod.kill_all_running(b, sid, sig)
        # Finished history is noise here — only what was killed (or failed to be) matters. (An earlier per-id skip dump ran 8-9KB; even the count read as "something was ignored".)
        parts = []
        if out["killed"]:
            parts.append(f"killed: {', '.join(out['killed'])}")
        parts.extend(f"{e['command_id']}: {e['state']}" for e in out["errors"])
        click.echo("; ".join(parts) or "nothing to kill — no running commands")
        if out["errors"]:
            sys.exit(1)


@cli.command(short_help="Mirror non-ignored project files into the workdir.")
@click.argument("session", required=False)
@click.option("--rebuild-manifest", is_flag=True, help="Rewrite the manifest without deleting (recovery after out-of-band remote edits).")
@click.option("--follow-symlinks", is_flag=True, help="Transfer linked file contents instead of the links.")
def sync(session: str | None, rebuild_manifest: bool, follow_symlinks: bool) -> None:
    """Mirror the project's non-ignored files into the session workdir (one-way, local->source)."""
    sid = _resolve_session(session)
    res = sess.sync(_backend_for(sid), sid, rebuild_manifest=rebuild_manifest, follow_symlinks=follow_symlinks)
    click.echo(f"added {len(res.added)}, updated {len(res.updated)}, deleted {len(res.deleted_outdated)} outdated, {res.bytes_transferred} bytes")
    for e in res.cleanup_errors:
        click.echo(f"cs: cleanup error: {e}", err=True)


def _split_sources_dest(rest: list[str], usage: str) -> tuple[list[str], str | None]:
    """Split transfer args into (sources, dest). One arg = source only; two = source + dest; more = the last must name a directory with a trailing `/` — the only unambiguous multi-source form."""
    if len(rest) == 1:
        return rest, None
    if len(rest) == 2:
        return rest[:1], rest[1]
    if not rest[-1].endswith("/"):
        _usage(f"{usage} — with several sources the destination must be a directory ending in '/'")
    return rest[:-1], rest[-1]


@cli.command(short_help="Copy into the workdir: cs upload LOCAL... [REMOTE]")
@click.argument("args", nargs=-1, required=True)
def upload(args: tuple[str, ...]) -> None:
    """Copy into the workdir, bypassing .gitignore: cs upload [SESSION] LOCAL... [REMOTE]

    For datasets, weights, .env — anything sync skips. LOCAL is relative to the project dir, REMOTE to the workdir (default: the local basename). cp conventions: REMOTE ending in '/' (or an existing dir) receives the source inside it; otherwise it names the result. A directory is placed as a directory; 'dir/' as LOCAL merges its contents into REMOTE. Several LOCALs need a REMOTE ending in '/'. Destination directories must already exist (as with cp).
    """
    token, rest = _split_leading_session(args)
    if not rest:
        _usage("usage: cs upload [SESSION] LOCAL... [REMOTE]")
    sid = _resolve_session(token)
    sources, dest = _split_sources_dest(rest, "usage: cs upload [SESSION] LOCAL... [REMOTE]")
    b = _backend_for(sid)
    for src in sources:
        out = sess.upload(b, sid, src, dest or "")
        click.echo(f"{out['local_path']} -> {out['remote_path']}")


@cli.command(short_help="Copy out of the workdir: cs download REMOTE... LOCAL")
@click.argument("args", nargs=-1, required=True)
def download(args: tuple[str, ...]) -> None:
    """Copy out of the workdir: cs download [SESSION] REMOTE... LOCAL

    REMOTE is relative to the workdir, LOCAL to the project dir. cp conventions: LOCAL ending in '/' (or an existing dir) receives the source inside it; otherwise it names the result. A directory is placed as a directory; 'dir/' as REMOTE merges its contents into LOCAL. Several REMOTEs need a LOCAL ending in '/'. Destination directories must already exist (as with cp).
    """
    token, rest = _split_leading_session(args)
    if len(rest) < 2:
        _usage("usage: cs download [SESSION] REMOTE... LOCAL")
    sid = _resolve_session(token)
    sources, dest = _split_sources_dest(rest, "usage: cs download [SESSION] REMOTE... LOCAL")
    b = _backend_for(sid)
    for src in sources:
        out = sess.download(b, sid, src, dest)
        click.echo(f"{out['remote_path']} -> {out['local_path']}")


@cli.command()
@click.argument("args", nargs=-1)
def ls(args: tuple[str, ...]) -> None:
    """List workdir contents: cs ls [SESSION] [PATH]"""
    token, rest = _split_leading_session(args)
    if len(rest) > 1:
        _usage("too many arguments — usage: cs ls [SESSION] [PATH]")
    sid = _resolve_session(token)
    click.echo(sess.ls(_backend_for(sid), sid, rest[0] if rest else ""), nl=False)


@cli.command("commands")
@click.argument("session", required=False)
@click.option("--limit", type=int, default=20, show_default=True, help="Maximum number of commands to list.")
@click.option("--json", "as_json", is_flag=True, help="JSON output.")
def commands_cmd(session: str | None, limit: int, as_json: bool) -> None:
    """List a session's recorded commands, newest first."""
    sid = _resolve_session(session)
    listing = sess.list_commands(_backend_for(sid), sid, limit=limit)
    if as_json:
        click.echo(json.dumps(listing, indent=2))
        return
    rows = [
        [c["command_id"], c["status"], str(c.get("exit_code", "-")), c.get("started_at", "-"), c.get("exited_at", "-")]
        for c in listing.get("commands", [])
    ]
    if not rows:
        click.echo(f"no commands recorded for {sid} ({listing.get('total', 0)} total)", err=True)
        return
    click.echo(_table(["COMMAND", "STATUS", "EXIT", "STARTED", "EXITED"], rows))
    total = listing.get("total", len(rows))
    if total > len(rows):
        click.echo(f"({len(rows)} of {total} — raise --limit for more)", err=True)


def _warn_if_stale_install() -> None:
    """When cs runs as a uv tool installed from a local directory (recorded in the tool's uv-receipt.toml), compare the installed package sources against that directory and nag on drift. A four-day drift once shipped a skill doc advertising `cs tail` and exec-style quoting against a binary that had neither — every silent arg-mangling failure in the July transcript study traces to it. Content comparison rather than versions (repo edits rarely bump the version); local-FS reads only; best-effort."""
    try:
        pkg_dir = Path(__file__).resolve().parent
        receipt = pkg_dir.parents[3] / "uv-receipt.toml"  # <tool-root>/lib/pythonX.Y/site-packages/compute_sessions
        if not receipt.is_file():
            return
        import tomllib
        reqs = tomllib.loads(receipt.read_text()).get("tool", {}).get("requirements", [])
        src_dir = next((Path(r["directory"]) for r in reqs if isinstance(r, dict) and r.get("directory")), None)
        if src_dir is None:
            return
        src_pkg = src_dir / "src" / "compute_sessions"
        if not src_pkg.is_dir():
            return
        names = {p.name for p in pkg_dir.glob("*.py")} | {p.name for p in src_pkg.glob("*.py")}
        for name in sorted(names):
            installed, source = pkg_dir / name, src_pkg / name
            if not installed.is_file() or not source.is_file() or installed.read_bytes() != source.read_bytes():
                click.echo(
                    f"cs: this installed build differs from its source repo at {src_dir} — "
                    f"reinstall: uv tool install --force --from {src_dir} compute-sessions",
                    err=True,
                )
                return
    except Exception:
        return


def main() -> None:
    _warn_if_stale_install()
    try:
        cli(prog_name="cs")
    except ComputeSessionsError as exc:
        click.echo(f"cs: {exc}", err=True)
        sys.exit(EXIT_TOOL_FAILURE)


if __name__ == "__main__":
    main()
