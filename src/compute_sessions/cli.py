"""`cs` — the command-line frontend; a thin adapter over the core modules (session/run/cluster).

Concurrent cs processes are safe and normal (parallel agents, backgrounded runs) — state lives on the cluster. Every command does its login-node work in one request (session selection included: exact id, prefix, or cwd-project inference happen in the helper); `cs run` / `cs logs -f` stream live output through one connection into the container.

Exit codes: `cs run` / `cs logs` exit with the remote command's code; 137 stands in for a command that died without recording an exit (OOM / kill / session death); `cs kill` exits 1 when the signal could not be delivered; 2 is an invalid request (click's usage-error code, shared by cs's own validation via `_usage`/UsageError: bad arguments, an unknown or ambiguous session/command/path, a session in the wrong state); 125 means cs itself failed (docker's convention for tool-level failure — a remote command can also exit 125, but the stderr message disambiguates).
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, NoReturn

import click

from compute_sessions import run as run_mod
from compute_sessions import session as sess
from compute_sessions.cluster import Cluster
from compute_sessions.config import load_config
from compute_sessions.errors import ComputeSessionsError, RemoteError, UsageError
from compute_sessions.models import SessionInfo, SessionStatus, fmt_duration as _dur, partition_limit_seconds
from compute_sessions.run import Output, Target
from compute_sessions.session import ResourceRequest

EXIT_TOOL_FAILURE = 125
EXIT_KILLED = 137
# `cs list` shows every live session but only this many of the most recently stopped ones.
LIST_INACTIVE = 20
# SIGTERM/SIGHUP to cs act like Ctrl-C (see main); a detach then exits with 128 + the signal.
_interrupt_exit = 130

# Config is loaded at import time so the cluster's resource vocabulary can render into --help. A missing/broken config must not take `cs --help` down with it — commands surface the error when actually run.
try:
    _CFG = load_config()
    _CLUSTER: Cluster | None = Cluster(_CFG)
    _CFG_ERROR: Exception | None = None
except Exception as exc:
    _CFG, _CLUSTER, _CFG_ERROR = None, None, exc


def _fail(msg: str) -> NoReturn:
    click.echo(f"cs: {msg}", err=True)
    sys.exit(EXIT_TOOL_FAILURE)


def _usage(msg: str) -> NoReturn:
    """A malformed invocation exits 2 (click's usage-error code, as documented in --help) — distinct from 125, so an agent can tell "I called it wrong" from "cs failed"."""
    click.echo(f"cs: {msg}", err=True)
    sys.exit(2)


def _note_other_project(sid: str, project: str) -> None:
    click.echo(f"cs: note — {sid} belongs to another project ({project})", err=True)


def _cluster() -> Cluster:
    if _CLUSTER is None:
        _fail(f"config error: {_CFG_ERROR}")
    _CLUSTER.on_note = _note_other_project
    return _CLUSTER


def _token(session: str | None) -> str | None:
    """An empty/whitespace session argument means "infer from the cwd", like none."""
    return session if session and session.strip() else None


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join(line.rstrip() for line in [fmt.format(*headers)] + [fmt.format(*r) for r in rows])


def _short(text: str | None, width: int = 60) -> str:
    """One table cell of free text: whitespace (newlines too) collapsed, cut to `width`."""
    if text is None:
        return "-"
    t = " ".join(text.split())
    return t if len(t) <= width else t[:width - 1] + "…"


def _exit_for(status: str, exit_code: int | None, session_id: str | None = None, *, earlier_activation: bool = False) -> NoReturn:
    if status == "exited":
        sys.exit(exit_code if exit_code is not None else 1)
    if status == "killed":
        click.echo(f"cs: command died without recording an exit — {_death_cause(session_id, earlier_activation)}", err=True)
        sys.exit(EXIT_KILLED)
    _fail(
        f"no trace of this command (status={status})"
        + (f" — `cs commands {session_id}` lists recorded commands" if session_id else "")
    )


def _death_cause(session_id: str | None, earlier_activation: bool = False) -> str:
    """Disambiguate a silent command death by whether the activation it ran in survived it: a live one means the command alone was killed (usually the OOM-killer), an ended one took the command down with it — named by the recorded stop reason. One extra request, only on this rare path; show() also reconciles a gone job."""
    if session_id is None:
        return "OOM, kill, or session death"
    try:
        info = sess.show(_cluster(), session_id, probe=False)["info"]
    except Exception:
        return "OOM, kill, or session death"
    if earlier_activation or info["status"] == "pending":
        return "it ran in an earlier activation of the session, which ended while it ran (cs deactivate, idle timeout, time limit or job death)"
    if info["status"] == "active":
        return "the session is still active, so most likely the OOM-killer (or an explicit kill) — retry with a smaller workload or reactivate with more --mem"
    cause = _stop_cause(info)
    if _time_limit_hit(info):
        return f"the session hit {cause} and was killed with the command — reactivate on a longer partition for jobs this size"
    return f"the session is {info['status']} ({cause or 'idle timeout, deactivate, or job death'}), so it ended and took the command with it"


# stop_reason values the runner writes for a scheduler signal (remote/runner.py).
_TIME_LIMIT_REASON = "the partition's time limit"
_SCHEDULER_SIGNAL = "signalled by the scheduler"


def _time_limit_hit(info: dict) -> bool:
    """Whether the last activation ended at its wall-clock limit: the runner read it from the job log, or — for an unnamed scheduler signal — the activation lasted right up to the limit its partition name encodes."""
    reason = info.get("stop_reason") or ""
    return reason == _TIME_LIMIT_REASON or ((not reason or reason.startswith(_SCHEDULER_SIGNAL)) and _walltime_hit(info))


def _stop_cause(info: dict) -> str | None:
    """Why the session's last activation ended: the recorded stop_reason, with the partition named when it was the time limit."""
    if _time_limit_hit(info):
        partition = (info.get("resources") or {}).get("partition")
        return f"the {partition} partition's time limit" if partition else _TIME_LIMIT_REASON
    return info.get("stop_reason")


def _walltime_hit(info: dict) -> bool:
    """Partition names encode their wall-clock limit as a suffix (cpu-9m, gpu-2h, gpu-7d). An activation that lasted right up to that limit died to the scheduler's kill, not OOM or idling — name it, so the same job is not re-run on the same too-short partition."""
    limit = partition_limit_seconds((info.get("resources") or {}).get("partition"))
    if limit is None:
        return False
    try:
        activated = datetime.strptime(info["last_activated_at"], "%Y-%m-%dT%H:%M:%SZ")
        ended = datetime.strptime(info["last_deactivated_at"], "%Y-%m-%dT%H:%M:%SZ")
    except (KeyError, TypeError, ValueError):
        return False
    return (ended - activated).total_seconds() >= limit * 0.97


def _start_label(start: str | None) -> str | None:
    """The scheduler's start forecast for a pending session, compact: "~start 15:30" today, else with the date. The scheduler reports its own (cluster-local) time."""
    if not start:
        return None
    day, _, clock = start.partition("T")
    return f"~start {clock[:5]}" if day == datetime.now().strftime("%Y-%m-%d") else f"~start {day[5:]} {clock[:5]}"


def _res_summary(resources: dict) -> str:
    """Compact resources with units ("gpu-5h 1gpu 80gb|h100 42G 4cpu"), as create/activate echo them and list/show print them."""
    parts = [resources["partition"]] if resources.get("partition") else []
    if resources.get("gpus") is not None:
        parts.append(f"{resources['gpus']}gpu")
    if resources.get("gpu_type"):
        parts.append(resources["gpu_type"])
    if resources.get("mem"):
        parts.append(f"{resources['mem']}G")
    if resources.get("cpus"):
        parts.append(f"{resources['cpus']}cpu")
    return " ".join(parts)


def _retry_pause(failures: int, exc: Exception, resume_hint: str) -> None:
    """Backoff for a transient connection failure in a waiting loop; re-raises after 8 consecutive failures (~2 min). A network blip must not abort a wait whose job then allocates unwatched."""
    if failures >= 8:
        click.echo(f"cs: giving up after repeated connection failures — {resume_hint}", err=True)
        raise exc
    click.echo(f"cs: connection hiccup ({str(exc).splitlines()[0]}); retrying…", err=True)
    time.sleep(min(30, 2 ** failures))


def _wait_pending(session_id: str, label: str) -> dict:
    """Block while the session is pending, echoing queue state to stderr: every scheduler refresh (30 s) for the first ~2 minutes, then only on a changed reason or every ~5 minutes. Returns the first non-pending detail."""
    started = time.monotonic()
    last = {"reason": None, "printed": None}

    def on_poll(detail: dict, _elapsed: float, fresh: bool) -> None:
        if not fresh:
            return
        elapsed = time.monotonic() - started
        job = detail.get("job") or {}
        reason = job.get("reason")
        if elapsed < 120 or reason != last["reason"] or elapsed - last["printed"] >= 300:
            click.echo(f"cs: {label} ({_dur(elapsed)})" + "".join(f" — {x}" for x in (reason, _start_label(job.get("estimated_start"))) if x), err=True)
            last["printed"] = elapsed
        last["reason"] = reason

    failures = 0
    while True:
        try:
            return sess.wait_until_settled(_cluster(), session_id, on_poll)
        except RemoteError as exc:
            failures += 1
            _retry_pause(failures, exc, f"keep waiting with: cs show -w {session_id}")


def _wait_active(session_id: str) -> dict:
    """Wait out PENDING; exits 125 when the activation fails. Ctrl-C leaves the session queued."""
    try:
        detail = _wait_pending(session_id, "waiting for allocation")
    except KeyboardInterrupt:
        click.echo(f"\ncs: still pending — keep waiting: cs show -w {session_id}; cancel: cs deactivate {session_id}", err=True)
        sys.exit(_interrupt_exit)
    status = detail["info"]["status"]
    if status == "active":
        gpu = detail["info"].get("gpu")
        click.echo(f"cs: {session_id} active" + (f" — {gpu}" if gpu else ""), err=True)
        return detail
    tail = detail.get("failure_log_tail")
    if tail:
        click.echo(tail.rstrip(), err=True)
    _fail(f"session {session_id} is {status}")


def _stream_and_exit(target: Target, command_id: str, out: Output, *, command: str | None = None, offsets: tuple[int, int] = (0, 0), on_launch: Callable[[], None] | None = None, release: bool = False) -> NoReturn:
    """Run/follow through the container and exit with the command's code. Ctrl-C detaches (the command keeps running)."""
    sid = target.session_id
    launched = {"yes": command is None}

    def launch_note() -> None:
        launched["yes"] = True
        if on_launch:
            on_launch()

    hint = f"cs logs -f {sid} {command_id}"
    try:
        try:
            status, code = run_mod.stream(_cluster(), target, command_id, out, command=command, offsets=offsets, on_launch=launch_note, release=release,
                                          on_retry=lambda reason: click.echo(f"cs: connection hiccup ({reason}); retrying…", err=True))
        finally:
            out.flush()
    except KeyboardInterrupt:
        if launched["yes"]:
            click.echo(f"\ncs: detached — the command keeps running. Reattach: {hint}; stop: cs kill {sid} {command_id}", err=True)
        else:
            click.echo(f"\ncs: interrupted before the launch was confirmed — `cs commands {sid}` shows whether {command_id} started", err=True)
        sys.exit(_interrupt_exit)
    except run_mod.StreamLost as exc:
        _fail(f"{exc} — " + {True: f"the command keeps running; reattach with: {hint}", False: "the command was not started"}.get(exc.launched, f"`cs commands {sid}` shows whether {command_id} started"))
    if status == "exited" and code == 127 and command and command.split(maxsplit=1)[0] in ("python", "python3", "pip", "pip3"):
        click.echo("cs: the container has no system python — use `uv run python …` / `uv pip …`", err=True)
    _exit_for(status, code, sid)


# How to work inside the session container — rendered into `cs run --help`.
_CONTAINER_NOTES = (
    "No system python — use `uv run`. Write scratch to /workdir. The shared HF cache at `~/.cache/huggingface` should be used for hf downloads. "
    "The project venv lives at /cs-venv/venv (`uv` targets it automatically — no `uv venv` needed, just `uv sync`/`uv pip install`) and persists across deactivate/activate. "
    "Torch: the driver runs wheels up to CUDA $CS_CUDA_VERSION (env var set in every GPU session) — pin torch's index to match (e.g. `[tool.uv.sources]` → https://download.pytorch.org/whl/cu128); the default wheels may target newer CUDA and fail to run."
)


def _resource_epilog() -> str:
    """Config-derived --help block for the create/activate/clone resource flags (empty when the config failed to load)."""
    if _CFG is None:
        return ""
    lines = [
        f"--partition: one of {', '.join(_CFG.partitions)} (default {_CFG.default_partition}). The suffix is a hard wall-clock limit that kills the whole session when it elapses; suffix-less partitions like *-test have short limits of their own — `cs show`'s time_left shows what remains.",
        "--gpus: 0-16 (cpu-* partitions always run with 0); default 1.",
        (f"--gpu-type: omit unless the task truly needs a specific GPU — any-gpu requests allocate faster. To pin, use one of {', '.join(_CFG.gpu_types)} (OR with `|`). The actually-assigned model shows in `cs list`/`cs show` (gpu)."
         if _CFG.gpu_types else "--gpu-type: no gpu types configured — leave unset."),
        f"--mem: GB (default {_CFG.default_mem}).",
        "--cpus: CPU cores (default: the cluster's, often 1-2) — set it for CPU-heavy work (data processing, CPU-bound sweeps).",
        "--idle-timeout: auto-shutdown after N minutes with no live `run` command and no workdir file ops (sync/upload/download/ls); reading logs does not reset the timer.",
        "activate/clone without --partition/--gpus/--gpu-type/--mem/--cpus reuse the session's previous resources; any of them starts from the defaults above.",
    ]
    # \b disables click's paragraph rewrapping for the block that follows.
    return f"\b\nResource flags ({_CFG.host}):\n" + "\n".join(f"  {line}" for line in lines)


def _resource_options(f):
    f = click.option("--idle-timeout", "idle_timeout_minutes", type=int, default=None, metavar="MIN", help="Auto-shutdown after MIN idle minutes (default: the session's previous value, else 20).")(f)
    f = click.option("--cpus", type=int, default=None, metavar="N", help="CPU cores.")(f)
    f = click.option("--mem", type=int, default=None, metavar="GB", help="Memory in GB.")(f)
    f = click.option("--gpu-type", default=None, help="Pin GPU type(s), OR with '|'.")(f)
    f = click.option("--gpus", type=int, default=None, help="GPU count.")(f)
    f = click.option("--partition", default=None, help="Slurm partition.")(f)
    return f


_NO_WAIT = click.option("--no-wait", is_flag=True, help="Submit and return immediately instead of waiting for activation (block later with cs show -w).")


@click.group(
    # help_option_names is inherited by subcommand contexts, so -h works everywhere.
    context_settings={"help_option_names": ["-h", "--help"]},
    help=f"Manage compute sessions — interactive container jobs on your SLURM cluster. Most commands take the session as an optional first argument: a full id, a unique prefix (the bare hex tail works: '8ec5' for {_CFG.session_prefix if _CFG else '<host>'}_8ec5…), or nothing (the cwd project's only (live) session is used).",
    epilog="""\
\b
Lifecycle: pending -> active -> inactive (cs activate re-enters) | failed
(cs show has the failure log tail; cs activate retries). File commands
(sync, upload, download, ls, logs, commands) work in any state.

\b
Files: `cs sync` mirrors the project's non-ignored files into the workdir
(one-way, local->cluster); .gitignored files (datasets, weights, .env) need
`cs upload`. Remote paths are relative to the workdir.

\b
Exit codes: cs run/logs exit with the remote command's code; 137 = command
died without recording an exit (OOM/kill/session death); cs kill exits 1 if
the signal was not delivered; 2 = invalid request (bad arguments, unknown
session/command/path, or a session in the wrong state); 125 = cs failed.
""",
)
@click.version_option(package_name="compute-sessions", prog_name="cs")
def cli() -> None:
    pass


@cli.command("list")
@click.option("--all", "all_projects", is_flag=True, help="All projects, not just the cwd's.")
@click.option("--json", "as_json", is_flag=True, help="JSON rows instead of a table.")
def list_cmd(all_projects: bool, as_json: bool) -> None:
    """List sessions (the cwd project's by default): every live one, and the most recently stopped ones."""
    cluster = _cluster()
    pid, cwd = cluster.project()
    # The project id derives from the git origin and can re-key when a remote is added — orphaning every earlier session from a plain `cs list`. Same-directory sessions under other ids are flagged.
    former: dict[str, int] = {}
    live, stopped = [], []
    for d in cluster.call("list", all=all_projects)["sessions"]:
        i = SessionInfo.from_json(d)
        if not all_projects and i.project_id != pid:
            if i.project_path == cwd:
                former[i.project_id] = former.get(i.project_id, 0) + 1
            continue
        row = i.to_agent_json()
        if d.get("estimated_start"):
            row["estimated_start"] = d["estimated_start"]
        if all_projects:
            row["project_id"] = i.project_id
        (live if i.status in (SessionStatus.ACTIVE, SessionStatus.PENDING) else stopped).append((i, row))
    # Active before pending, most recently changed first; then the most recently stopped.
    live.sort(key=lambda p: (p[0].status != SessionStatus.ACTIVE, p[1].get("time_in_status_seconds") or 0))
    stopped.sort(key=lambda p: p[0].last_deactivated_at or p[0].created_at, reverse=True)
    rows = [r for _, r in live + stopped[:LIST_INACTIVE]]
    for fid, n in sorted(former.items()):
        click.echo(f"cs: {n} session(s) for this directory live under a former project id {fid!r} — `cs list --all` shows them", err=True)
    if len(stopped) > LIST_INACTIVE:
        click.echo(f"cs: {len(stopped) - LIST_INACTIVE} older inactive/failed session(s) not shown", err=True)
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        click.echo(f"no sessions{'' if all_projects else f' for project {pid!r} (try --all)'}", err=True)
        return
    headers = ["SESSION", "STATUS", "IN-STATUS", "GPU", "RESOURCES", "IDLE-IN", "TIME-LEFT"]
    if all_projects:
        headers.append("PROJECT")
    table_rows = []
    for r in rows:
        res = _res_summary(r)
        line = [r["session_id"], r["status"], _dur(r.get("time_in_status_seconds")), r.get("gpu") or "-", res or "-", "busy" if r.get("busy") else _dur(r.get("seconds_until_idle_deactivate")), _start_label(r.get("estimated_start")) or _dur(r.get("time_left_seconds"))]
        if all_projects:
            line.append(r.get("project_id", "-"))
        table_rows.append(line)
    click.echo(_table(headers, table_rows))


@cli.command(epilog=_resource_epilog())
@_resource_options
@_NO_WAIT
def create(partition: str | None, gpus: int | None, gpu_type: str | None, mem: int | None, cpus: int | None, idle_timeout_minutes: int | None, no_wait: bool) -> None:
    """Create a session for the cwd project and start it. Prints the session id on stdout."""
    req = ResourceRequest(partition=partition, gpus=gpus, gpu_type=gpu_type, mem=mem, cpus=cpus, idle_timeout_minutes=idle_timeout_minutes)
    info = sess.create(_cluster(), req, Path.cwd())
    click.echo(f"cs: created {info.session_id} (pending — {_res_summary(info.resources)})", err=True)
    if not no_wait:
        _wait_active(info.session_id)
    click.echo(info.session_id)


@cli.command(short_help="Start or resume a session.", epilog=_resource_epilog())
@click.argument("session", required=False)
@_resource_options
@_NO_WAIT
def activate(session: str | None, partition: str | None, gpus: int | None, gpu_type: str | None, mem: int | None, cpus: int | None, idle_timeout_minutes: int | None, no_wait: bool) -> None:
    """Start (or resume) a session. Resources are chosen per activation: without resource flags, the previous activation's are reused."""
    req = ResourceRequest(partition=partition, gpus=gpus, gpu_type=gpu_type, mem=mem, cpus=cpus, idle_timeout_minutes=idle_timeout_minutes)
    info = sess.activate(_cluster(), _token(session), req)
    sid = info.session_id
    click.echo(f"cs: {sid} pending — {_res_summary(info.resources)}", err=True)
    if not no_wait:
        _wait_active(sid)
    click.echo(sid)


@cli.command(short_help="Stop a session (workdir persists; activate re-enters).")
@click.argument("session", required=False)
def deactivate(session: str | None) -> None:
    """Stop a session (workdir and logs persist; re-enter with `cs activate`). Running commands end with it."""
    def on_stopped(cmds: list[dict], dropped: list[str]) -> None:
        if cmds:
            click.echo("cs: stopping running command(s): " + "; ".join(f"{c['command_id']} ({_short(c['command'], 50)})" for c in cmds), err=True)
        if dropped:
            click.echo(f"cs: dropped queued command(s): {', '.join(dropped)}", err=True)
    info = sess.deactivate(_cluster(), _token(session), on_stopped)
    click.echo(f"{info.session_id} {info.status.value}")


@cli.command(short_help="Clone a session's workdir into a new session.", epilog=_resource_epilog())
@click.argument("session", required=False)
@_resource_options
@_NO_WAIT
def clone(session: str | None, partition: str | None, gpus: int | None, gpu_type: str | None, mem: int | None, cpus: int | None, idle_timeout_minutes: int | None, no_wait: bool) -> None:
    """Clone a session's workdir into a new session (server-side copy). Prints the new session id on stdout."""
    req = ResourceRequest(partition=partition, gpus=gpus, gpu_type=gpu_type, mem=mem, cpus=cpus, idle_timeout_minutes=idle_timeout_minutes)
    info = sess.clone(_cluster(), _token(session), req)
    click.echo(f"cs: created {info.session_id} (pending — {_res_summary(info.resources)})", err=True)
    if not no_wait:
        _wait_active(info.session_id)
    click.echo(info.session_id)


# Record fields that are plumbing, or raw inputs of the derived fields shown instead (time left, time in status): nothing an agent acts on.
_SHOW_HIDDEN = ("image", "job_id", "sshd_port", "job_started_at", "submitted_at", "last_activity_at")


@cli.command(short_help="Full session detail: job state, health, failure log.")
@click.argument("session", required=False)
@click.option("-w", "--wait", is_flag=True, help="Block while the session is pending.")
@click.option("--json", "as_json", is_flag=True, help="JSON output.")
def show(session: str | None, wait: bool, as_json: bool) -> None:
    """Full detail for one session: resources, live job state, health, failure log tail."""
    detail = sess.show(_cluster(), _token(session))
    sid = detail.pop("session_id")
    if wait and detail["info"]["status"] == "pending":
        _wait_pending(sid, "still pending")
        detail = sess.show(_cluster(), sid)
        detail.pop("session_id")
    if detail["info"]["status"] in ("inactive", "failed"):
        detail["info"]["stop_reason"] = _stop_cause(detail["info"])
    for k in _SHOW_HIDDEN:
        detail["info"].pop(k, None)
    if detail["info"]["status"] not in ("pending", "active"):
        detail.pop("job", None)  # a stopped session has no job; its scheduler state says nothing
    if as_json:
        click.echo(json.dumps(detail, indent=2))
        return
    for k, v in detail["info"].items():
        if v is None:
            continue
        click.echo(f"{k}: {_res_summary(v) if k == 'resources' else json.dumps(v) if isinstance(v, (dict, list)) else v}")
    for key in ("job", "health"):
        if key in detail:
            click.echo(f"{key}: " + ", ".join(f"{k}={v}" for k, v in detail[key].items() if v is not None))
    if "hint" in detail:
        click.echo(f"hint: {detail['hint']}")
    if "failure_log_tail" in detail:
        click.echo("failure log tail:\n" + detail["failure_log_tail"].rstrip())


def _container(args: tuple[str, ...], **params) -> tuple[dict, list[str]]:
    """Resolve a `[SESSION] …` command against the cluster (marking activity) — (reply, remaining args)."""
    r = _cluster().call("session", args, maybe=True, touch=True, **params)
    return r, r["rest"]


def _require_target(r: dict) -> Target:
    if not r.get("socket"):
        _usage(f"session {r['sid']} is not active (status={r['info']['status']}); cannot reach container processes")
    return Target(r["sid"], r["socket"])


def _run_options(rest: list[str], detach: bool, tail_n: int | None, release: bool) -> tuple[list[str], bool, int | None, bool]:
    """cs run's own options also work right after the session id (`cs run SID -d CMD`): a command never starts with '-', so a leading '-' word there is cs's. A leading `--` ends them."""
    rest = list(rest)
    while rest and rest[0].startswith("-"):
        opt = rest.pop(0)
        if opt == "--":
            break
        if opt in ("-d", "--detach"):
            detach = True
        elif opt == "--deactivate-after":
            release = True
        elif opt == "--tail" or opt.startswith("--tail="):
            val = opt.partition("=")[2] if "=" in opt else (rest.pop(0) if rest else "")
            try:
                tail_n = int(val)
            except ValueError:
                _usage(f"--tail needs a number, got {val!r}")
        else:
            _usage(f"no such option: {opt} — usage: cs run [SESSION] [-d] [--tail N] [--deactivate-after] CMD [ARGS...]")
    return rest, detach, tail_n, release


# ignore_unknown_options keeps unknown flags (`--lr 0.1`) in ARGS; allow_interspersed_args=False stops option parsing at the first positional, so a `-d`/`-h` INSIDE the user command stays part of the command instead of being captured by click.
@cli.command(context_settings={"ignore_unknown_options": True, "allow_interspersed_args": False}, epilog="\b\nContainer environment:\n" + _CONTAINER_NOTES)
@click.argument("args", nargs=-1, required=True)
@click.option("-d", "--detach", is_flag=True, help="Launch and print the command id instead of streaming.")
@click.option("--tail", "tail_n", type=int, default=None, metavar="N", help="Print only the last N lines of each stream, once the command ends (output is withheld until then). The full log stays retrievable via cs logs.")
@click.option("--deactivate-after", "release", is_flag=True, help="Deactivate the session once this command has ended and nothing else runs — no GPU idles until the idle timeout, even if this machine sleeps.")
def run(args: tuple[str, ...], detach: bool, tail_n: int | None, release: bool) -> None:
    """Run a command in a session: cs run [SESSION] CMD [ARGS...]

    Streams output until the command exits and exits with its code; Ctrl-C detaches (the command keeps running). Runs in /workdir via the container shell. A single CMD argument is a raw shell fragment (pipes/redirects work): cs run 'python train.py 2>&1 | tee train.log'. Multiple arguments are quoted individually (exec-style): cs run python -c 'print("hi")'.

    Never pipe cs run's OWN output through filters like `| tail` / `| grep -q` — the pipeline's exit code replaces the command's, and an early-closing filter kills the stream. To cap output, use --tail N.

    Long-running commands: prefer running cs run itself in the background (shell '&', or your harness's background-task facility) — it streams, survives connection blips, and exits with the command's code. Use -d when the local cs process may not outlive the job (laptop sleep) or for servers, then follow with cs logs -f. Killing cs never kills the remote command. On a pending session, -d queues the command: it starts on its own once the session is active.

    Local stdin is not forwarded: upload input files and read them in the workdir.
    """
    r, rest = _container(args)
    rest, detach, tail_n, release = _run_options(rest, detach, tail_n, release)
    if tail_n is not None and tail_n <= 0:
        _usage("--tail must be a positive integer")
    if detach and tail_n is not None:
        _usage("--tail has no effect with -d — read the end later with cs logs --tail N")
    if not rest or not "".join(rest).strip():
        _usage("no command given — usage: cs run [SESSION] CMD [ARGS...]")
    sid = r["sid"]
    # One arg = raw shell fragment; several = exec-style (each arg quoted). Plain " ".join would let the remote shell re-split args containing spaces.
    command = rest[0] if len(rest) == 1 else shlex.join(rest)
    status = r["info"]["status"]
    command_id = run_mod.new_command_id()
    if detach and status == SessionStatus.PENDING.value:
        if sess.queue(_cluster(), sid, command_id, command, release)["queued"]:
            click.echo(f"cs: {command_id} queued on {sid} — it starts when the session is active", err=True)
            click.echo(command_id)
            return
        r = _cluster().call("session", [sid], touch=True)
        status = r["info"]["status"]
    if status == SessionStatus.PENDING.value or (status == SessionStatus.ACTIVE.value and not r.get("socket")):
        _wait_active(sid)
        r = _cluster().call("session", [sid], touch=True)
    elif status == SessionStatus.FAILED.value:
        _usage(f"session {sid} failed to activate — `cs show {sid}` has the log tail; `cs activate {sid}` retries")
    elif status != SessionStatus.ACTIVE.value:
        _usage(f"session {sid} is {status} — `cs activate {sid}` first (resources are chosen per activation)")
    target = _require_target(r)
    on_launch = lambda: click.echo(f"cs: {command_id} on {sid}", err=True)  # noqa: E731
    if detach:
        try:
            run_mod.stream(_cluster(), target, command_id, Output(), command=command, follow=False, on_launch=on_launch, release=release,
                           on_retry=lambda reason: click.echo(f"cs: connection hiccup ({reason}); retrying…", err=True))
        except run_mod.StreamLost as exc:
            _fail(f"{exc} — " + ("the command was not started" if exc.launched is False else f"`cs commands {sid}` shows whether {command_id} started"))
        click.echo(command_id)
        return
    _stream_and_exit(target, command_id, Output(tail_n), command=command, on_launch=on_launch, release=release)


def _wait_queued(sid: str, command_id: str, tail_n: int | None) -> dict:
    """Block until a queued command has been launched (the session must become active first); returns its first logs reply."""
    click.echo(f"cs: {command_id} is queued — waiting for {sid} to start it", err=True)
    active_since = None
    while True:
        r = sess.logs(_cluster(), [sid, command_id], max_lines=tail_n)
        if r["status"] != "queued":
            return r
        if r["session_status"] == SessionStatus.PENDING.value:
            _wait_pending(sid, "waiting for allocation")
            active_since = None
        elif r["session_status"] != SessionStatus.ACTIVE.value:
            _fail(f"session {sid} is {r['session_status']} — {command_id} stays queued for its next activation (`cs activate {sid}`; `cs kill {sid} {command_id}` drops it)")
        else:
            # The runner starts queued commands within seconds of the activation.
            active_since = active_since or time.monotonic()
            if time.monotonic() - active_since > 120:
                _fail(f"{command_id} is still queued 2 min after {sid} became active — its runner did not start it; `cs kill {sid} {command_id}` drops it, then run it again")
            time.sleep(2)


@cli.command()
@click.argument("args", nargs=-1)
@click.option("-f", "--follow", is_flag=True, help="Stream until the command exits; exit with its code.")
@click.option("--tail", "tail_n", type=int, default=None, metavar="N", help="Only the last N lines of each stream (with -f: printed once the command ends).")
def logs(args: tuple[str, ...], follow: bool, tail_n: int | None) -> None:
    """Read a command's output: cs logs [SESSION] [COMMAND_ID]

    COMMAND_ID (a unique prefix like cmd_7c71 works) defaults to the session's newest RUNNING command, else its most recent one (`cs commands` lists them). stdout/stderr go to the local stdout/stderr; a finished command's exit code becomes the exit code.
    """
    if tail_n is not None and tail_n <= 0:
        _usage("--tail must be a positive integer")
    cluster = _cluster()
    r = sess.logs(cluster, list(args), max_lines=tail_n, resolve=True)
    sid, command_id = r["sid"], r["cid"]
    if r.get("chosen"):
        # Say which command the default picked and in what state, so a surprising default is visible.
        click.echo(f"cs: {command_id} ({r['chosen']})", err=True)
    if r["status"] == "queued":
        if not follow:
            click.echo(f"cs: {command_id} is queued — it starts when {sid} is active", err=True)
            sys.exit(0)
        r = _wait_queued(sid, command_id, tail_n)
    # With --tail and -f the cap applies to the whole follow: buffered, printed once the command settles.
    out = Output(tail_n if follow else None)
    out.write("O", r["out"])
    if r["out"] and r["err"] and not follow:
        out.flush()
        out.separator()
    out.write("E", r["err"])
    status = r["status"]
    if follow and (status == "running" or (status == "killed" and r.get("session_status") != "active")):
        if status == "running" and r.get("socket"):
            _stream_and_exit(Target(sid, r["socket"]), command_id, out, offsets=tuple(r["offsets"]))
        # The session ended (taking the command with it): final read after the shared-FS delay.
        time.sleep(3)
        r = sess.logs(cluster, [sid, command_id], since=tuple(r["offsets"]))
        out.write("O", r["out"])
        out.write("E", r["err"])
        status = "killed" if r["status"] == "running" else r["status"]
    out.flush()
    if not out.at_line_start():
        click.echo("", err=True)
    if status == "running":
        click.echo(f"cs: still running — follow with: cs logs -f {sid} {command_id}", err=True)
        sys.exit(0)
    _exit_for(status, r.get("exit_code"), sid, earlier_activation=bool(r.get("earlier_activation")))


@cli.command()
@click.argument("args", nargs=-1)
@click.option("-s", "--signal", "sig", default="TERM", show_default=True, help="Signal to send (TERM, KILL, INT, ... or numeric).")
def kill(args: tuple[str, ...], sig: str) -> None:
    """Stop command(s): cs kill [SESSION] [COMMAND_ID]

    COMMAND_ID may be a unique prefix like cmd_7c71 (keep the `cmd_`: a bare hex token may name a session). Without COMMAND_ID, signals every running command in the session. Targets the whole process group, so backgrounded children are covered. Exits 1 if the signal could not be delivered (no pid recorded, or kill failed).
    """
    run_mod.validate_signal(sig)
    r, rest = _container(args, cid=True)
    if len(rest) > 1:
        _usage("too many arguments — usage: cs kill [SESSION] [COMMAND_ID]")
    if not r.get("socket"):
        # Not running: queued commands can still be dropped.
        dropped = sess.dequeue(_cluster(), [r["sid"]] + rest)["dropped"]
        if dropped:
            click.echo("\n".join(f"{c}: removed from the queue" for c in dropped))
            return
    target = _require_target(r)
    if rest:
        cid = rest[0]
        state = run_mod.kill_command(_cluster(), target, cid, sig)
        if state == "missing" and sess.dequeue(_cluster(), [r["sid"], cid])["dropped"]:
            state = "dequeued"
        label = {
            "exited": "signalled — exited",
            "running": "signalled — still running after 3s (it handles or ignores the signal; -s KILL forces it)",
            "survivors": f"signalled — its shell exited, but processes of it are still running after 3s (they handle or ignore the signal); `cs kill -s KILL {r['sid']} {cid}` stops them",
            "already_exited": "already exited — nothing to signal",
            "killed_orphans": "signalled the processes it left running after it exited",
            "old": "ran in an earlier activation of the session — no longer running",
            "dead": "died without recording an exit — nothing to signal",
            "missing": "no such command",
            "dequeued": "removed from the queue",
            "starting": "not started yet — the signal was not sent; retry in a moment",
            "kill_failed": "the signal could not be delivered",
        }.get(state, state)
        click.echo(f"{cid}: {label}")
        if state in ("missing", "starting", "kill_failed"):
            sys.exit(1)
    else:
        out = run_mod.kill_all(_cluster(), target, sig)
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
    """Mirror the project's non-ignored files into the session workdir (one-way, local->cluster)."""
    _, res = sess.sync(_cluster(), _token(session), rebuild_manifest=rebuild_manifest, follow_symlinks=follow_symlinks)
    click.echo(f"added {len(res.added)}, updated {len(res.updated)}, removed {len(res.deleted_outdated)} ({res.bytes_transferred} bytes transferred)")
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
    # Each source lands at DEST/<its name>: two of one name would silently overwrite each other ('dir/' sources merge their contents instead).
    names = [src.rstrip().rstrip("/").rsplit("/", 1)[-1] for src in rest[:-1] if not src.rstrip().endswith("/")]
    dup = next((n for n in names if names.count(n) > 1), None)
    if dup:
        _usage(f"several sources are named {dup!r} — they would overwrite each other in {rest[-1]}; copy them one at a time to different paths")
    return rest[:-1], rest[-1]


@cli.command(short_help="Copy into the workdir: cs upload LOCAL... [REMOTE]")
@click.argument("args", nargs=-1, required=True)
def upload(args: tuple[str, ...]) -> None:
    """Copy into the workdir, bypassing .gitignore: cs upload [SESSION] LOCAL... [REMOTE]

    For datasets, weights, .env — anything sync skips. LOCAL is relative to the project dir (absolute paths must stay inside it or an agent scratchpad: /tmp/claude-*, /tmp/codex-*, /tmp/agent-*), REMOTE to the workdir (default: the local basename). cp conventions: REMOTE ending in '/' (or an existing dir) receives the source inside it; otherwise it names the result — except that an existing dir with the source dir's own name is merged into, so repeating an upload does not nest. A directory is placed as a directory; 'dir/' as LOCAL merges its contents into REMOTE. Several LOCALs need a REMOTE ending in '/'. Missing destination directories are created.
    """
    usage = "usage: cs upload [SESSION] LOCAL... [REMOTE]"
    r, rest = _container(args)
    if not rest:
        _usage(usage)
    sources, dest = _split_sources_dest(rest, usage)
    info = SessionInfo.from_json(r["info"])
    for src in sources:
        out = sess.upload(_cluster(), r["sid"], info, src, dest or "")
        click.echo(f"{out['local_path']} -> {out['remote_path']}")


@cli.command(short_help="Copy out of the workdir: cs download REMOTE... LOCAL")
@click.argument("args", nargs=-1, required=True)
def download(args: tuple[str, ...]) -> None:
    """Copy out of the workdir: cs download [SESSION] REMOTE... LOCAL

    REMOTE is relative to the workdir (no wildcards), LOCAL to the project dir (absolute paths must stay inside it or an agent scratchpad: /tmp/claude-*, /tmp/codex-*, /tmp/agent-*). cp conventions: LOCAL ending in '/' (or an existing dir) receives the source inside it; otherwise it names the result — except that an existing dir with the source dir's own name is merged into, so repeating a download does not nest. A directory is placed as a directory; 'dir/' as REMOTE merges its contents into LOCAL. Several REMOTEs need a LOCAL ending in '/'. Destination directories must already exist (as with cp).
    """
    usage = "usage: cs download [SESSION] REMOTE... LOCAL"
    r, rest = _container(args)
    if len(rest) < 2:
        _usage(usage)
    sources, dest = _split_sources_dest(rest, usage)
    info = SessionInfo.from_json(r["info"])
    for src in sources:
        out = sess.download(_cluster(), r["sid"], info, src, dest)
        click.echo(f"{out['remote_path']} -> {out['local_path']}")


@cli.command()
@click.argument("args", nargs=-1)
def ls(args: tuple[str, ...]) -> None:
    """List workdir contents: cs ls [SESSION] [PATH]"""
    click.echo(sess.ls(_cluster(), list(args)), nl=False)


@cli.command("cp", short_help="Copy between session workdirs: cs cp SRC:PATH... DST:PATH")
@click.argument("args", nargs=-1, required=True)
def cp_cmd(args: tuple[str, ...]) -> None:
    """Copy workdir paths between sessions, on the cluster: cs cp SESSION:PATH... SESSION:PATH

    Sessions may be in any state. Missing destination directories are created. `SESSION:` alone is the workdir root.
    """
    usage = "usage: cs cp SESSION:PATH... SESSION:PATH"
    if len(args) < 2:
        _usage(usage)
    parsed = []
    for a in args:
        sid, sep, path = a.partition(":")
        if not sep or not sid.strip():
            _usage(f"{usage} — {a!r} names no session")
        parsed.append((sid.strip(), path))
    if len({s for s, _ in parsed[:-1]}) > 1:
        _usage(f"{usage} — all sources must be in one session")
    dst, dest = parsed[-1]
    sources, dest = _split_sources_dest([p for _, p in parsed[:-1]] + [dest or "./"], usage)
    out = sess.copy(_cluster(), parsed[0][0], sources, dst, dest)
    for s, d in out["copied"]:
        click.echo(f"{out['src']}:{s} -> {out['sid']}:{d}")


@cli.command(short_help="Delete stopped sessions.")
@click.argument("sessions", nargs=-1, required=True)
def delete(sessions: tuple[str, ...]) -> None:
    """Delete stopped sessions with their workdirs and logs: cs delete SESSION..."""
    out = sess.delete(_cluster(), list(sessions))
    for sid in out["deleted"]:
        click.echo(f"{sid} deleted (recoverable from {out['trash']} for {out['keep_days']} days)")


@cli.command("commands")
@click.argument("session", required=False)
@click.option("--limit", type=int, default=20, show_default=True, help="Maximum number of commands to list.")
@click.option("--json", "as_json", is_flag=True, help="JSON output.")
def commands_cmd(session: str | None, limit: int, as_json: bool) -> None:
    """List a session's recorded commands, newest first."""
    listing = sess.list_commands(_cluster(), _token(session), limit=limit)
    sid = listing.pop("sid")
    listing.pop("note", None)
    if as_json:
        click.echo(json.dumps(listing, indent=2))
        return
    rows = [
        [c["command_id"], c["status"], str(c.get("exit_code", "-")), c.get("started_at", "-"), c.get("exited_at", "-"), _short(c.get("command"))]
        for c in listing.get("commands", [])
    ]
    if not rows:
        click.echo(f"no commands recorded for {sid} ({listing.get('total', 0)} total)", err=True)
        return
    click.echo(_table(["COMMAND", "STATUS", "EXIT", "STARTED", "EXITED", "CMD"], rows))
    total = listing.get("total", len(rows))
    if total > len(rows):
        click.echo(f"({len(rows)} of {total} — raise --limit for more)", err=True)


def _asset_dirs() -> tuple[Path, Path]:
    """(cluster-side files dir, skills dir). Wheel installs carry both under compute_sessions/assets/; a repo checkout falls back to the live top-level dirs so dev runs see current files."""
    pkg = Path(__file__).resolve().parent
    bundled = pkg / "assets"
    if (bundled / "remote" / "runner.py").is_file():
        return bundled / "remote", bundled / "skills"
    root = pkg.parent.parent
    return root / "remote", root / "skills"


@cli.command()
def assets() -> None:
    """Print the directory of cluster-side files shipped with this build (runner.py, Apptainer.def) — what setup scps to / builds on the cluster. Re-upload/rebuild from here after upgrading compute-sessions."""
    remote_dir, _ = _asset_dirs()
    if not (remote_dir / "runner.py").is_file():
        _fail(f"bundled assets missing at {remote_dir} (broken install?)")
    click.echo(remote_dir)


@cli.command("install-skills")
@click.option("--dir", "target", default=None, metavar="DIR", help="Install into this skills directory only (default: ~/.agents/skills, plus ~/.claude/skills when ~/.claude exists).")
def install_skills(target: str | None) -> None:
    """Copy the agent skills shipped with this build (compute-sessions usage + setup-cluster) into the standard skills directories, replacing any prior copies: ~/.agents/skills — the cross-agent location read by Codex, Gemini CLI, Cursor, Copilot, opencode, Amp, Goose, Windsurf, and others — and ~/.claude/skills for Claude Code, which does not read ~/.agents. Re-run after upgrading compute-sessions."""
    _, skills_dir = _asset_dirs()
    srcs = [p for p in sorted(skills_dir.iterdir()) if (p / "SKILL.md").is_file()] if skills_dir.is_dir() else []
    if not srcs:
        _fail(f"no skills found at {skills_dir} (broken install?)")
    if target:
        roots = [Path(target).expanduser()]
    else:
        roots = [Path.home() / ".agents" / "skills"]
        if (Path.home() / ".claude").is_dir():
            roots.append(Path.home() / ".claude" / "skills")
        else:
            click.echo("no ~/.claude — skipped the Claude Code copy (use --dir ~/.claude/skills to force)", err=True)
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        for src in srcs:
            dest = root / src.name
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            elif dest.is_dir():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            click.echo(dest)


def _warn_if_stale_install() -> None:
    """When cs runs as a uv tool installed from a local directory (recorded in the tool's uv-receipt.toml), compare the installed package sources against that directory and report drift — a stale install silently disagrees with the repo's skill docs. Content comparison rather than versions (repo edits rarely bump the version); local-FS reads only; best-effort. Only `cs --version` checks: on every command, the warning reached agents that cannot act on it (any uncommitted edit in the repo triggers it) and taught them to discard stderr."""
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


def _interrupt(signum, frame) -> None:
    """SIGTERM/SIGHUP (a harness stopping a task, a closed terminal) act like Ctrl-C: the KeyboardInterrupt paths print how to reattach, and subprocess cleanup stops a running rsync instead of leaving it uploading."""
    global _interrupt_exit
    _interrupt_exit = 128 + signum
    raise KeyboardInterrupt


def main() -> None:
    for signo in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signo, _interrupt)
    if sys.argv[1:2] == ["--version"]:
        _warn_if_stale_install()
    try:
        cli(prog_name="cs")
    except UsageError as exc:
        _usage(str(exc))
    except ComputeSessionsError as exc:
        click.echo(f"cs: {exc}", err=True)
        sys.exit(EXIT_TOOL_FAILURE)


if __name__ == "__main__":
    main()
