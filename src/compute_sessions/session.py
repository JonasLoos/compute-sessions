from __future__ import annotations

import json
import logging
import re
import secrets
import shlex
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from compute_sessions.cluster import Cluster
from compute_sessions.errors import SessionError
from compute_sessions.models import SessionInfo, SessionStatus, SyncResult
from compute_sessions.paths import validate_command_id, validate_session_id
from compute_sessions.project import project_id as derive_project_id
from compute_sessions.sync import sync_session


log = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ResourceRequest:
    """Per-activation resource request. None = unset (filled from the config defaults by Cluster.resolve_resources)."""
    partition: str | None = None
    gpus: int | None = None
    gpu_type: str | None = None
    mem: int | None = None
    idle_timeout_minutes: int = 20


def mark_activity(cluster: Cluster, session_id: str) -> None:
    """Touch the session's `.activity` sentinel; the idle monitor counts a fresh mtime as activity. Workdir file tools run outside the container-side view — without this, an agent in a download/analyze phase between runs loses its allocation and the next `run` pays a fresh one. NOT called by show/list (status polling shouldn't keep a session alive) or by the log readers (logs/wait_for_command/list_commands): a live command already registers via its `.pid` heartbeat, and tailing the logs of a finished command — a human's `cs logs -f`, an agent re-reading results — shouldn't hold hardware either. Best-effort."""
    try:
        sentinel = f"{cluster.access.paths.session_dir(session_id)}/.activity"
        cluster.access.run(f"touch {shlex.quote(sentinel)}", check=False)
    except Exception as exc:
        log.debug("could not mark activity for %s: %s", session_id, exc)


def _ensure_session_cap(cluster: Cluster, project_id_val: str) -> None:
    cap = cluster.cfg.max_sessions_per_repo
    if cap <= 0:
        return
    live = [s for s in cluster.list_infos(project_id_val) if s.status in (SessionStatus.ACTIVE, SessionStatus.PENDING)]
    if len(live) >= cap:
        raise SessionError(
            f"repo {project_id_val!r} already has {len(live)} active/pending session(s); limit is {cap} — "
            f"deactivate one ({', '.join(s.session_id for s in live[:4])}) or raise max_sessions_per_repo in the config"
        )


def _new_info(cluster: Cluster, project_id_val: str, project_path: str) -> SessionInfo:
    """Identity-only record for a fresh session, persisted with its dirs. The resource fields (resources/idle_timeout/image) are resolved, validated, and persisted by activate()."""
    info = SessionInfo(
        session_id=f"{cluster.cfg.session_prefix}_{secrets.token_hex(5)}",
        project_id=project_id_val,
        project_path=project_path,
        created_at=_utcnow(),
    )
    cluster.init_session_dirs(info.session_id)
    cluster.write_info(info)
    return info


def _resolve_request(cluster: Cluster, req: ResourceRequest) -> dict:
    """Validate a resource request against the config (defaults filled in) without touching the cluster. create()/clone() call this BEFORE writing anything: validation used to happen inside activate(), after the record was persisted and the project synced, so an invalid `--gpu-type` left a `pending` record with no job behind (observed repeatedly once a gpu type was dropped from the allowlist while agents kept passing it)."""
    resolved = cluster.resolve_resources(partition=req.partition, gpus=req.gpus, gpu_type=req.gpu_type, mem=req.mem)
    # Upper bound is 7 days — anything past the longest realistic allocation can never take effect anyway.
    if not 1 <= req.idle_timeout_minutes <= 10080:
        raise SessionError(f"idle_timeout_minutes must be between 1 and 10080, got {req.idle_timeout_minutes}")
    return resolved


@contextmanager
def _fail_if_incomplete(cluster: Cluster, info: SessionInfo):
    """Mark a freshly created session FAILED if its create/clone does not run to completion (broken sync, sbatch error, Ctrl-C). The record is persisted before the slow steps, so an interruption used to leave it `pending` with no job — a state nothing could clear: show() only reconciles against a job, deactivate() returned early without one, and the zombie still counted toward the session cap and blocked cwd session inference. Cancels the job if the interruption landed between submit and the final record write. Best-effort — the original error is what the caller sees."""
    try:
        yield
    except BaseException:
        try:
            latest = cluster.read_info(info.session_id)
            cluster.cancel(latest)
            latest.status = SessionStatus.FAILED
            latest.last_deactivated_at = _utcnow()
            cluster.write_info(latest)
        except Exception as exc:
            log.debug("could not mark %s failed after an incomplete create: %s", info.session_id, exc)
        raise


def create(cluster: Cluster, req: ResourceRequest, cwd: Path) -> SessionInfo:
    cwd = cwd.resolve()
    project_id_val = derive_project_id(cwd)
    _resolve_request(cluster, req)
    _ensure_session_cap(cluster, project_id_val)
    info = _new_info(cluster, project_id_val, str(cwd))
    with _fail_if_incomplete(cluster, info):
        sync_session(cluster.access, info.session_id, cwd)
        # Non-blocking: resolve resources, submit, and return PENDING. The first `run` blocks until the session is ACTIVE; tools that only touch cluster-side files (sync, logs, ls, download, upload) work without waiting. No live job exists yet, so activate() never refuses here.
        return activate(cluster, info.session_id, req)


def clone(cluster: Cluster, source_session_id: str, req: ResourceRequest) -> SessionInfo:
    """Create a new session for the same project as `source_session_id`, seeded with a server-side copy of its workdir (contents + sync manifest — no laptop round-trip), then submit its activation like create() does.

    Command logs are NOT copied — they are the source session's history. The copy is a snapshot of the workdir as it is right now; commands still running in the source keep writing to the source only.
    """
    validate_session_id(source_session_id)
    src = cluster.read_info(source_session_id)
    _resolve_request(cluster, req)
    _ensure_session_cap(cluster, src.project_id)
    info = _new_info(cluster, src.project_id, src.project_path)
    with _fail_if_incomplete(cluster, info):
        # Copy before submitting the activation so a fast allocation can't start running commands against a half-copied workdir.
        cluster.copy_workdir(source_session_id, info.session_id)
        return activate(cluster, info.session_id, req)


def sync(cluster: Cluster, session_id: str, *, rebuild_manifest: bool = False, follow_symlinks: bool = False) -> SyncResult:
    validate_session_id(session_id)
    info = cluster.read_info(session_id)
    mark_activity(cluster, session_id)
    return sync_session(
        cluster.access, session_id, project_cwd(info),
        rebuild_manifest=rebuild_manifest,
        follow_symlinks=follow_symlinks,
    )


def project_cwd(info: SessionInfo) -> Path:
    """Resolve the local project directory stored on a session. Raises if the stored path no longer exists."""
    p = Path(info.project_path)
    if not p.is_dir():
        raise SessionError(
            f"session {info.session_id} project_path {info.project_path!r} "
            "no longer exists on this machine"
        )
    return p


def activate(cluster: Cluster, session_id: str, req: ResourceRequest) -> SessionInfo:
    """Submit the activation and return immediately with status PENDING.

    Resource args are validated with config defaults filled in. The container image comes from the config, not a per-call arg.

    Refuses with SessionError if a job from a prior activation is still live — an allocation can't be resized in place and a second submit would duplicate it; deactivate first, then activate to restart with new resources.

    Does not wait for allocation or for sshd to come up — callers wait via `show(wait_seconds=…)` (blocks while pending; the CLI's `cs show -w` / create-wait loop) before calling `run`.
    """
    validate_session_id(session_id)
    info = cluster.read_info(session_id)

    if info.job_id and cluster.job_state(info)["state"] in ("pending", "running"):
        raise SessionError(
            f"session {session_id} is already active (job_id={info.job_id}); "
            f"deactivate it first, then activate to restart with new resources."
        )

    resolved = _resolve_request(cluster, req)

    # Clear any stale tunnel from a previous activation of this session. Implicit deaths (idle timeout, scancel, node failure) don't run deactivate()'s teardown, so a cached tunnel can outlive its endpoint. Without this, the first `run` after re-activate can hit a stale endpoint → connection refused.
    cluster.access.close_tunnel(session_id)

    # Persist the resolved resources so the runner reads the right image / idle timeout and show()/list() reflect this activation's request.
    info.resources = resolved
    info.idle_timeout_minutes = req.idle_timeout_minutes
    info.image = cluster.cfg.image
    # Drop the previous activation's runtime state: a leftover heartbeat makes seconds_until_idle_deactivate read 0 until the runner stamps a fresh value, and a leftover node/sshd_port/gpu would show through `show` while the new activation is still pending. The runner rewrites all of these when the session goes active.
    info.last_activity_at = None
    info.node = None
    info.sshd_port = None
    info.gpu = None
    prev_status = info.status
    info.status = SessionStatus.PENDING
    # Persist BEFORE submitting — the runner reads config.json (image, idle timeout, project_id) as its first act, and a fast allocation could otherwise race a stale config.
    cluster.write_info(info)

    try:
        info.job_id = cluster.submit(info)
    except BaseException:
        # sbatch refused the job (or Ctrl-C landed here): put the record back, or it lingers as `pending` with no job — a state nothing reconciles, since there is no job to observe. A record that was already job-less pending (a fresh create) becomes FAILED instead.
        info.status = SessionStatus.FAILED if prev_status == SessionStatus.PENDING else prev_status
        cluster.write_info(info)
        raise
    cluster.write_info(info)
    return info


def deactivate(cluster: Cluster, session_id: str) -> SessionInfo:
    validate_session_id(session_id)
    info = cluster.read_info(session_id)
    # Tear down any cached tunnel for this session regardless of whether there's a job to cancel. After deactivate the endpoint is gone, so a later activate creates a fresh one — a kept cache entry would make the next `run` hit a stale endpoint and fail with "Connection refused".
    cluster.access.close_tunnel(session_id)
    if not info.job_id:
        # Nothing to cancel — but a record stuck in PENDING without a job (a create that died before submitting, on a client older than the _fail_if_incomplete guard) must still be clearable here: no job means nothing else ever reconciles it.
        if info.status == SessionStatus.PENDING:
            info.status = SessionStatus.INACTIVE
            info.last_deactivated_at = _utcnow()
            cluster.write_info(info)
        return info
    cluster.cancel(info)
    # The runner's cleanup writes status=inactive. Poll briefly so the caller sees the post-cancellation state instead of the pre-cancel snapshot. For PENDING sessions the runner never ran, so no cleanup fires — fall through to the explicit write below.
    latest = info
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            latest = cluster.read_info(session_id)
        except Exception as exc:
            log.debug("could not read latest config during deactivate: %s", exc)
            break
        if latest.status not in (SessionStatus.ACTIVE, SessionStatus.PENDING):
            break
        time.sleep(0.5)
    # The cleanup didn't run / hasn't finished within the window — mark inactive ourselves so the session doesn't linger in a transient state. FAILED is terminal and excluded.
    if latest.status in (SessionStatus.ACTIVE, SessionStatus.PENDING):
        latest.status = SessionStatus.INACTIVE
        cluster.write_info(latest)
    return latest


def show(cluster: Cluster, session_id: str, *, wait_seconds: float = 0.0) -> dict:
    """No wait cap: the pending-wait below is a poll loop of short read_info round-trips, safe at any timeout."""
    if wait_seconds < 0:
        raise SessionError(f"wait_seconds must be >= 0, got {wait_seconds!r}")
    info = cluster.read_info(session_id)
    # With wait_seconds, block while the session is PENDING (polling the session record) and return the full snapshot once it settles. A job that dies without running its cleanup stays "pending" the full window — the gone-job reconciliation below then flips it to failed.
    if wait_seconds > 0 and info.status == SessionStatus.PENDING:
        deadline = time.monotonic() + wait_seconds
        while info.status == SessionStatus.PENDING and time.monotonic() < deadline:
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            info = cluster.read_info(session_id)
    result: dict = {"info": info.to_enriched_json()}
    if info.job_id:
        state = cluster.job_state(info)
        result["job"] = state
        if state["state"] == "gone":
            # The job is gone but the session record never caught up — the runner died without running its cleanup (early crash while PENDING; node failure or OOM-SIGKILL while ACTIVE). Reconcile so `list` stops reporting a dead session and `run`/`activate` report a clear state error instead of an opaque connection failure.
            reconciled = {
                SessionStatus.PENDING: SessionStatus.FAILED,
                SessionStatus.ACTIVE: SessionStatus.INACTIVE,
            }.get(info.status)
            if reconciled is not None:
                info.status = reconciled
                cluster.write_info(info)
                result["info"] = info.to_enriched_json()
    # Placed after the gone-job reconciliation above so a just-died job reads failed, not pending-with-hint.
    if info.status == SessionStatus.PENDING:
        result["hint"] = (
            "allocation can take a while when the cluster is busy or the request "
            "is constrained (e.g. a pinned gpu type) — keep waiting with "
            "`cs show -w`, or `cs deactivate` to cancel and retry with "
            "different resources"
        )
    if info.status == SessionStatus.FAILED:
        # A failed activation is otherwise a black box — surface the runner log tail so the agent can see the cause without shell access to the cluster.
        tail = cluster.failure_log_tail(info)
        if tail.strip():
            result["failure_log_tail"] = tail
    if info.status == SessionStatus.ACTIVE:
        result["health"] = cluster.probe(info)
    return result


# A command's completion is normally recorded in its `.exit` file by the inner bash's EXIT trap. But an untrappable death — SIGKILL, OOM, or the session dying mid-run — never writes `.exit`, so `.exit`-presence alone can't tell "still running" from "died silently". The `.pid` heartbeat does: the inner bash touches `.pid` every run._HEARTBEAT_SECONDS (10s) while alive, so a `.pid` with no sibling `.exit` whose mtime has gone stale means the command is gone. This mirrors the runner-side idle monitor (a `.pid` touched within ~60s counts as busy) — a host-side `kill -0` can't be used because the container has a private PID namespace. Kept comfortably above the 10s heartbeat so a single missed touch or FS-mtime lag never mislabels a live command as dead.
_PID_STALE_SECONDS = 60


def _classify_command_status(probe: str) -> tuple[str, int | None]:
    """Map the `(status, exit_code)` probe emitted by the shell snippet below into a status.

    The snippet prints one of:
      `exited <code>` — `.exit` present (code may be empty → exit_code None).
      `age <seconds>` — no `.exit`, `.pid` present; age = seconds since its last heartbeat.
      `missing`       — neither file (unknown command_id, or logs wiped).

    A fresh `.pid` → `running`; a stale one → `killed` (died without recording an exit).
    """
    probe = probe.strip()
    if probe.startswith("exited"):
        rest = probe[len("exited"):].strip()
        try:
            return "exited", int(rest)
        except ValueError:
            return "exited", None
    if probe.startswith("age"):
        rest = probe[len("age"):].strip()
        try:
            age = int(rest)
        except ValueError:
            age = 0
        return ("running" if age <= _PID_STALE_SECONDS else "killed"), None
    return "missing", None


# Shell snippet (login-node side) that probes a command's `.exit`/`.pid` state in one round-trip. `{base}` is the quoted log-file stem; `.exit`/`.pid` are appended outside the quotes (shell concatenates). Parsed by `_classify_command_status`. The `stat -c` (GNU) → `stat -f` (BSD/macOS) fallback keeps the age probe working when the tests run it against a local directory.
def _status_probe_cmd(base_quoted: str) -> str:
    return (
        f"e={base_quoted}.exit; p={base_quoted}.pid; "
        f'if [ -f "$e" ]; then printf "exited %s" "$(cat "$e" 2>/dev/null)"; '
        f'elif [ -f "$p" ]; then printf "age %s" '
        f'"$(( $(date +%s) - $(stat -c %Y "$p" 2>/dev/null || stat -f %m "$p" 2>/dev/null || echo 0) ))"; '
        'else printf "missing"; fi'
    )


# Status probe plus the current byte size of each stream file, one `SIZES <out> <err>` line followed by the status text. The sizes feed the `since` cursor: wait_for_command uses them to return early on new output, logs_with_access to report the cursor for the next poll. `$osz`/`$esz` are expanded unquoted in printf so BSD wc's space-padding can't smuggle extra fields into the line.
def _sizes_probe_cmd(base_quoted: str) -> str:
    return (
        f"o={base_quoted}.out; er={base_quoted}.err; osz=0; esz=0; "
        f'[ -f "$o" ] && osz=$(wc -c < "$o"); [ -f "$er" ] && esz=$(wc -c < "$er"); '
        f"printf 'SIZES %s %s\\n' $osz $esz; "
        + _status_probe_cmd(base_quoted)
    )


def _parse_since(since: str | None) -> tuple[int, int]:
    """Parse a `since` cursor ("<out_bytes>:<err_bytes>", as returned in the `cursor` field) into per-stream byte offsets."""
    if not since:
        return 0, 0
    out_s, sep, err_s = since.partition(":")
    try:
        oo, eo = int(out_s), int(err_s)
        if not sep or oo < 0 or eo < 0:
            raise ValueError
    except ValueError:
        raise SessionError(
            f"invalid since cursor {since!r} — pass the `cursor` value a previous "
            f"run/logs call returned (format '<int>:<int>')"
        )
    return oo, eo


# Collapse carriage-return-overwritten output (tqdm bars, HF download meters) to each line's final state — a shell pipe stage appended to the stream reads. Must run REMOTELY: subprocess's text mode applies universal-newline translation, so by the time output reaches Python every \r has already become \n and the overwrites are indistinguishable from real lines. First expression drops a line-trailing \r (CRLF endings) so the second — keep only what follows the last \r — can't blank those lines. The \r chars are embedded literally (BSD sed has no \r escape). The raw log files keep everything; only the returned view is condensed.
_COLLAPSE_CR_PIPE = " | sed -e " + shlex.quote("s/\r$//") + " -e " + shlex.quote("s/.*\r//")


def logs_with_access(
    access,
    session_id: str,
    command_id: str,
    max_lines: int | None = None,
    since: str | None = None,
) -> dict:
    """`logs` against an access object — the shared core of logs()/run().

    One round-trip: a single script emits stream sizes, the status probe, and both stream contents separated by a per-call sentinel. With `since` (a cursor from a previous call), only bytes past the recorded offsets are returned; a still-running command's result carries the next `cursor`. `max_lines` caps each returned stream to its last N lines (of the returned slice) with dropped-line counts.
    """
    validate_session_id(session_id)
    validate_command_id(command_id)
    if max_lines is not None and max_lines <= 0:
        raise SessionError(f"max_lines must be a positive integer, got {max_lines!r}")
    oo, eo = _parse_since(since)
    base = f"{access.paths.logs_dir(session_id)}/{command_id}"
    sep = f"__CS_SEP_{secrets.token_hex(8)}__"
    n = int(max_lines) if max_lines is not None else 0
    cap = f" | tail -n {n}" if n else ""
    # Line counts of the returned slice are only needed to compute the dropped-lines counters, so the extra read is gated on max_lines.
    count_lines = (
        f'[ -f "$o" ] && ol=$(tail -c +{oo + 1} "$o" | wc -l); '
        f'[ -f "$er" ] && el=$(tail -c +{eo + 1} "$er" | wc -l); '
        if n else ""
    )
    # The content reads are capped (head -c) at the byte sizes measured for META: a live command can append between the wc and the tail, and returning bytes past the reported cursor makes the next since=cursor read re-emit them — observed as `cs run` printing a command's whole output block twice when it was written in one burst right at snapshot time (uv's error path).
    script = (
        f"o={shlex.quote(base)}.out; er={shlex.quote(base)}.err; osz=0; esz=0; ol=0; el=0; "
        f'[ -f "$o" ] && osz=$(wc -c < "$o"); [ -f "$er" ] && esz=$(wc -c < "$er"); '
        + count_lines
        + "printf 'META %s %s %s %s\\n' $osz $esz $ol $el; "
        + _status_probe_cmd(shlex.quote(base)) + "; "
        f"printf '\\n%s\\n' {shlex.quote(sep)}; "
        f'if [ -f "$o" ] && [ "$osz" -gt {oo} ]; then tail -c +{oo + 1} "$o" | head -c $((osz-{oo})){cap}{_COLLAPSE_CR_PIPE}; fi; '
        f"printf '\\n%s\\n' {shlex.quote(sep)}; "
        f'if [ -f "$er" ] && [ "$esz" -gt {eo} ]; then tail -c +{eo + 1} "$er" | head -c $((esz-{eo})){cap}{_COLLAPSE_CR_PIPE}; fi'
    )
    r = access.run(script, check=False)
    head, out_part, err_part = (r.stdout.split(f"\n{sep}\n") + ["", ""])[:3]
    meta_line, _, probe = head.partition("\n")
    osz = esz = out_total = err_total = 0
    fields = meta_line.split()
    if len(fields) == 5 and fields[0] == "META":
        osz, esz, out_total, err_total = (int(f) if f.isdigit() else 0 for f in fields[1:])
    status, exit_code = _classify_command_status(probe)
    result: dict = {
        "stdout": out_part,
        "stderr": err_part,
        "status": status,
    }
    if exit_code is not None:
        result["exit_code"] = exit_code
    if max_lines is not None:
        result["stdout_truncated_lines"] = max(0, out_total - out_part.count("\n"))
        result["stderr_truncated_lines"] = max(0, err_total - err_part.count("\n"))
    # The cursor only matters while more output can still arrive; an exited/killed command's next read would be empty anyway.
    if status == "running":
        result["cursor"] = f"{osz}:{esz}"
    return result


def logs(
    cluster: Cluster,
    session_id: str,
    command_id: str,
    max_lines: int | None = None,
    since: str | None = None,
) -> dict:
    """Read stdout/stderr and completion state for a previously-launched command.

    `status` is one of: `exited` (`.exit` recorded; `exit_code` set), `running` (`.pid` heartbeat still fresh), `killed` (heartbeat went stale with no `.exit` — SIGKILL/OOM/session death), or `missing` (no trace — unknown command_id or wiped logs). `exit_code` is only present when a code was recorded (i.e. `exited`).

    When `max_lines` is set, only the last `max_lines` lines of each stream are returned, and the response includes `stdout_truncated_lines` / `stderr_truncated_lines` counts (lines dropped from the front) so the agent can tell how much output was skipped.

    `since` is the `cursor` from a previous run/logs call on this command: only output past it is returned, and a still-running result carries the next `cursor`.
    """
    validate_session_id(session_id)
    cluster.read_info(session_id)  # a typo'd session id should raise, not read an empty log dir
    return logs_with_access(cluster.access, session_id, command_id, max_lines=max_lines, since=since)


def wait_for_command(
    cluster: Cluster,
    session_id: str,
    command_id: str,
    *,
    timeout_seconds: int,
    poll_interval_seconds: float = 1.0,
    max_lines: int | None = None,
    since: str | None = None,
) -> dict:
    """Block until `command_id` finishes (or `timeout_seconds` elapses), polling the shared FS from the login node.

    Returns the same shape as `logs()`: `{stdout, stderr, status, [exit_code, cursor, stdout_truncated_lines, stderr_truncated_lines]}`. Returns as soon as the command settles — `exited` (`.exit` appeared) or `killed` (`.pid` heartbeat went stale with no `.exit`: SIGKILL/OOM/session death) — else `running` when the timeout elapses first. With `since` (a cursor from a previous call), also returns as soon as NEW output past the cursor appears, so each poll of a chatty command comes back promptly and informative instead of sitting out the full window.

    Polls `.exit` (and the `.pid` heartbeat) without in-container exec, so it works even if the session deactivated mid-run (both files persist on the shared FS).

    No timeout cap: each poll round-trip is short, so any timeout is transport-safe — the CLI's streaming follow passes 60s+ rounds.
    """
    validate_session_id(session_id)
    validate_command_id(command_id)
    if timeout_seconds <= 0:
        raise SessionError(f"timeout_seconds must be positive, got {timeout_seconds!r}")
    if poll_interval_seconds < 0.2:
        poll_interval_seconds = 0.2
    oo, eo = _parse_since(since)
    cluster.read_info(session_id)
    acc = cluster.access
    base_q = shlex.quote(f"{acc.paths.logs_dir(session_id)}/{command_id}")
    deadline = time.monotonic() + timeout_seconds
    while True:
        # Cheap FS probe — much lighter than reading both log streams every poll. We settle on `.exit` (clean finish) OR a stale `.pid` heartbeat (the command died without recording an exit — SIGKILL/OOM/session death — so waiting out the full timeout would be pointless) OR, when a cursor was given, stream growth past it. Once settled, read the full result via logs_with_access, which re-classifies the same way.
        probe = acc.run(_sizes_probe_cmd(base_q), check=False)
        sizes_line, _, probe_txt = probe.stdout.partition("\n")
        settled = _classify_command_status(probe_txt)[0] != "running"
        if not settled and since is not None:
            fields = sizes_line.split()
            if len(fields) == 3 and fields[0] == "SIZES":
                osz, esz = (int(f) if f.isdigit() else 0 for f in fields[1:])
                settled = osz > oo or esz > eo
        if settled or time.monotonic() >= deadline:
            return logs_with_access(acc, session_id, command_id, max_lines=max_lines, since=since)
        # Sleep no longer than the remaining budget so we return promptly when the timeout fires between polls.
        remaining = deadline - time.monotonic()
        time.sleep(min(poll_interval_seconds, max(0.0, remaining)))


# Agent scratchpad roots (Claude Code: /tmp/claude-<uid>/…; codex/other agents analogous) are allowed for upload/download despite living outside the project — agents naturally stage helper scripts and fetch results there, and these are agent-owned temp dirs with nothing to exfiltrate or clobber. macOS resolves /tmp to /private/tmp, hence the optional prefix.
_SCRATCH_ROOT_RE = re.compile(r"^(?:/private)?/tmp/(?:claude|codex|agent)-[^/]+/")


def _resolve_local_inside_project(local: Path, info: SessionInfo, label: str) -> Path:
    """Resolve `local` against the session's project_path (a relative path is taken relative to it, mirroring `remote_path`'s workdir-relative contract) and refuse if the result escapes both that project_path and the known agent scratchpad roots.

    The download/upload tools must not be usable as arbitrary local read/write primitives — a prompt-injected agent could otherwise overwrite `~/.ssh/authorized_keys` (download) or exfiltrate `~/.ssh/id_rsa` (upload). Resolving both sides first means a symlink inside the project that points outside still gets rejected.
    """
    if not info.project_path:
        raise SessionError(
            f"session {info.session_id} has no stored project_path — recreate the session"
        )
    project = Path(info.project_path).expanduser().resolve()
    local = local.expanduser()
    resolved = (local if local.is_absolute() else project / local).resolve()
    try:
        resolved.relative_to(project)
    except ValueError:
        if not _SCRATCH_ROOT_RE.match(str(resolved)):
            raise SessionError(
                f"{label} {str(resolved)!r} is outside the session's project_path "
                f"{str(project)!r} and not in an agent scratchpad (/tmp/claude-*, "
                f"/tmp/codex-*); refusing to operate"
            )
    return resolved


# Destination semantics shared by download/upload, following cp/rsync conventions (transcript evidence: the old "dest names the file / dirs merge contents" contract silently merged 4.6GB into a results dir, overwrote tracked files, and turned three files into one file named like the intended dir):
#   - A dest that ends with `/` or already exists as a directory receives the source INSIDE it (dest/<basename>).
#   - Otherwise the dest names the resulting file/dir itself.
#   - Directory contents only merge directly into dest when the SOURCE ends with `/` (rsync's src/ idiom; also `.`/empty), or when dest is an existing dir of the same basename (so repeat `upload data data` stays idempotent instead of nesting data/data).
#   - Destination directories must already exist — like cp, and unlike an implicit mkdir -p, which turned typo'd destinations into silently-created fresh directories.
def _resolve_dest(src_base: str, dest: str, *, src_is_dir: bool, src_trailing_slash: bool, dest_trailing_slash: bool, dest_is_dir: bool) -> str:
    if src_is_dir and src_trailing_slash:
        return dest  # explicit contents-merge
    if dest_trailing_slash or (dest_is_dir and not (src_is_dir and src_base == dest.rstrip("/").rsplit("/", 1)[-1])):
        return f"{dest.rstrip('/')}/{src_base}"
    return dest


def _display_local(p: Path, project: Path) -> str:
    """Local paths echo relative to project_path when inside it; scratchpad paths (allowed by _resolve_local_inside_project) stay absolute instead of crashing relative_to()."""
    try:
        return str(p.relative_to(project))
    except ValueError:
        return str(p)


def download(cluster: Cluster, session_id: str, remote_rel: str, local_path: Path | str) -> dict:
    """Copy a file or directory from the session workdir to the local machine.

    cp/rsync destination conventions — see _resolve_dest: `local_path` ending with `/` (or an existing directory) receives the remote file/dir inside it; otherwise it names the result. A remote directory is placed AS a directory; its contents merge directly into `local_path` only when `remote_rel` ends with `/`. The remote path is resolved relative to the session workdir (same cwd as `run`/`ls`).
    """
    validate_session_id(session_id)
    info = cluster.read_info(session_id)
    acc = cluster.access
    paths = acc.paths
    raw_local = str(local_path)
    resolved = _resolve_local_inside_project(Path(raw_local), info, "download local_path")
    mark_activity(cluster, session_id)
    remote = paths.resolve_workdir_relative(session_id, remote_rel)
    probe = acc.run(
        f"if [ -d {shlex.quote(remote)} ]; then echo dir; "
        f"elif [ -e {shlex.quote(remote)} ]; then echo file; "
        f"else echo missing; fi",
        check=False,
    )
    kind = probe.stdout.strip()
    if kind == "missing":
        raise SessionError(f"remote path does not exist: {remote}")
    # cp conventions include the failure: a destination directory that doesn't exist is an error, not an implicit mkdir -p — a typo'd path must not silently become a fresh directory holding the only copy of the results.
    if raw_local.rstrip().endswith("/") and not resolved.is_dir():
        raise SessionError(f"local destination {raw_local!r} is not an existing directory — create it first or drop the trailing '/'")
    dest = Path(_resolve_dest(
        remote.rstrip("/").rsplit("/", 1)[-1], str(resolved),
        src_is_dir=kind == "dir",
        src_trailing_slash=remote_rel.strip() in ("", ".") or remote_rel.rstrip().endswith("/"),
        dest_trailing_slash=raw_local.rstrip().endswith("/"),
        dest_is_dir=resolved.is_dir(),
    ))
    if not dest.parent.is_dir():
        raise SessionError(f"local destination {raw_local!r}: parent directory {dest.parent} does not exist — create it first")
    try:
        if kind == "dir":
            if dest.is_file():
                raise SessionError(f"local destination {dest} exists as a file but the remote is a directory — remove it or pick another path")
            dest.mkdir(exist_ok=True)
            acc.rsync_from(remote, dest, contents_only=True)
        else:
            if dest.is_dir():
                raise SessionError(f"local destination {dest} exists as a directory but the remote is a file — pick another path")
            acc.rsync_from(remote, dest)
    except OSError as exc:
        raise SessionError(f"local destination {raw_local!r} is unusable: {exc}") from exc
    # Echo both sides relative to their roots (local→project_path, remote→workdir); a trailing `/` marks a directory result so the caller sees what kind of thing landed where.
    project = Path(info.project_path).expanduser().resolve()
    return {"local_path": _display_local(dest, project) + ("/" if kind == "dir" else ""), "remote_path": remote.removeprefix(paths.workdir(session_id)).lstrip("/")}


def upload(cluster: Cluster, session_id: str, local_path: Path | str, remote_rel: str) -> dict:
    """Upload a file or directory into the session workdir (no .gitignore filtering).

    cp/rsync destination conventions — see _resolve_dest: `remote_rel` ending with `/` (or an existing remote directory) receives the local file/dir inside it; otherwise it names the result. A local directory is placed AS a directory; its contents merge directly into `remote_rel` only when `local_path` ends with `/`. Empty `remote_rel` = the local basename at the workdir root.
    """
    validate_session_id(session_id)
    info = cluster.read_info(session_id)
    acc = cluster.access
    paths = acc.paths
    raw_local = str(local_path)
    resolved = _resolve_local_inside_project(Path(raw_local), info, "upload local_path")
    mark_activity(cluster, session_id)
    if not resolved.exists():
        raise SessionError(f"local path does not exist: {resolved}")
    rel = (remote_rel or "").strip()
    if not rel or rel == ".":
        rel = resolved.name
    remote = paths.resolve_workdir_relative(session_id, rel)
    # One probe round-trip so an existing remote dir gets place-inside semantics instead of being silently merged into / overwritten; the same probe checks the parent dir for the validation below.
    probe = acc.run(
        f"if [ -d {shlex.quote(remote)} ]; then echo dir; "
        f"elif [ -e {shlex.quote(remote)} ]; then echo file; "
        f"else echo missing; fi; "
        f"[ -d {shlex.quote(remote.rsplit('/', 1)[0])} ] && echo parent-ok || echo parent-missing",
        check=False,
    ).stdout.split()
    kind = probe[0] if probe else "missing"
    # cp conventions include the failure: a destination directory that doesn't exist is an error, not an implicit mkdir -p — a typo'd path must not silently become a fresh directory.
    if kind == "missing" and rel.rstrip().endswith("/"):
        raise SessionError(f"remote destination {rel!r} is not an existing directory — create it first or drop the trailing '/'")
    src_is_dir = resolved.is_dir()
    dest = _resolve_dest(
        resolved.name, remote,
        src_is_dir=src_is_dir,
        src_trailing_slash=raw_local.rstrip().endswith("/"),
        dest_trailing_slash=rel.rstrip().endswith("/"),
        dest_is_dir=kind == "dir",
    )
    if dest == remote and kind == "missing" and "parent-ok" not in probe:
        raise SessionError(f"remote destination {rel!r}: parent directory does not exist — create it first")
    if src_is_dir:
        if kind == "file" and dest == remote:
            raise SessionError(f"remote destination {rel!r} exists as a file but the local path is a directory — pick another path")
        acc.run(f"mkdir -p {shlex.quote(dest)}")
        acc.rsync_to(resolved, dest, contents_only=True)
    else:
        acc.rsync_to(resolved, dest)
    project = Path(info.project_path).expanduser().resolve()
    return {"local_path": _display_local(resolved, project), "remote_path": dest.removeprefix(paths.workdir(session_id)).lstrip("/") + ("/" if src_is_dir else "")}


def ls(cluster: Cluster, session_id: str, remote_rel: str = "") -> str:
    validate_session_id(session_id)
    cluster.read_info(session_id)
    mark_activity(cluster, session_id)
    remote = cluster.access.paths.resolve_workdir_relative(session_id, remote_rel)
    # A nonexistent path is an expected probe result (agents scan old sessions for artifacts) — answer it with a clear message instead of the raw ssh failure dump a checked `ls` would produce.
    res = cluster.access.run(
        f"if [ -e {shlex.quote(remote)} ]; then ls -la {shlex.quote(remote)}; else echo __CS_MISSING__; fi",
        check=False,
    )
    if res.stdout.strip() == "__CS_MISSING__":
        raise SessionError(f"path not found in session {session_id} workdir: {remote_rel or '.'!r}")
    if res.returncode != 0:
        raise SessionError(f"ls failed for {remote_rel or '.'!r}: {res.stderr.strip() or 'unknown error'}")
    return res.stdout


# Remote helper: inline Python piped to python3 on the login node. For each cmd_* id: .start mtime → started_at (NOT .pid — the heartbeat refreshes its mtime every 10s, which would read as "moments before exit"), .exit mtime → exited_at, .exit contents → exit code.
#
# Paginates on the remote side — a hot session with thousands of commands would otherwise ship its whole history over ssh (and into an agent's context) just to recover a recent command_id.
_LIST_COMMANDS_PY = '''\
import json, os, sys, time
from datetime import datetime, timezone

logdir = sys.argv[1]
limit = int(sys.argv[2])   # 0 = unlimited
offset = int(sys.argv[3])
stale = int(sys.argv[4])   # .pid heartbeat staleness threshold (seconds)
now = time.time()

def iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

rows = {}
for name in os.listdir(logdir):
    if not name.startswith("cmd_"):
        continue
    # Housekeeping commands spawned by the run epilogue (runner.py's venv-snapshot helper) — internal noise, always hidden. They stay killable by id and inspectable via ps in the container or the raw log files.
    if name.startswith("cmd_venvsnap_"):
        continue
    base, _, ext = name.rpartition(".")
    if ext not in ("start", "pid", "exit"):
        continue
    path = os.path.join(logdir, name)
    mtime = os.path.getmtime(path)
    row = rows.setdefault(base, {
        "command_id": base,
        "status": "running",
        "exit_code": None,
        "started_at": None,
        "exited_at": None,
        "_pid_age": None,
    })
    if ext == "start":
        row["started_at"] = iso(mtime)
    elif ext == "pid":
        row["_pid_age"] = now - mtime
    else:
        row["status"] = "exited"
        row["exited_at"] = iso(mtime)
        with open(path) as fh:
            content = fh.read().strip()
        # Tolerate garbage — /cs-logs is writable from inside the container, so .exit contents are not trusted.
        row["exit_code"] = int(content) if content.lstrip("-").isdigit() else None

# A command still tagged "running" but whose .pid heartbeat has gone stale (or has no .pid at all) died without recording an exit — SIGKILL/OOM/session death. Report "killed" instead of a perpetual "running". Matches logs()/_classify_command_status.
for row in rows.values():
    age = row.pop("_pid_age", None)
    if row["status"] == "running" and (age is None or age > stale):
        row["status"] = "killed"

ordered = sorted(
    rows.values(),
    key=lambda r: r["started_at"] or r["exited_at"] or "",
    reverse=True,
)
end = (offset + limit) if limit > 0 else None
sliced = ordered[offset:end] if end is not None else ordered[offset:]
# Omit null fields (exit_code/exited_at while running, started_at when the .start sentinel is gone) — absence is less noise for the agent than explicit nulls.
sliced = [{k: v for k, v in r.items() if v is not None} for r in sliced]
print(json.dumps({
    "commands": sliced,
    "total": len(ordered),
    "offset": offset,
    "limit": limit if limit > 0 else None,
}))
'''


def list_commands(cluster: Cluster, session_id: str, *, limit: int | None = 50, offset: int = 0) -> dict:
    """Enumerate commands recorded under the session's logs directory.

    Returns `{commands, total, offset, limit}`. `commands` is the paginated slice (newest first); each row is `{command_id, status, exit_code, started_at, exited_at}` with null fields omitted (e.g. no exit_code/exited_at while running), where `status` is `exited`, `running`, or `killed` (heartbeat went stale with no `.exit` — SIGKILL/OOM, or the session died mid-run). Logs persist across deactivation, so historical commands appear too.

    Internal venv-snapshot commands (`cmd_venvsnap_*`) are filtered out (and excluded from `total`). `limit=None` returns all rows; the default (50) keeps the listing compact for agent consumption.
    """
    validate_session_id(session_id)
    # A typo should raise, not return [].
    cluster.read_info(session_id)
    limit_arg = max(0, int(limit)) if limit is not None else 0
    offset_arg = max(0, int(offset))
    res = cluster.access.run(
        f"python3 - {shlex.quote(cluster.access.paths.logs_dir(session_id))} "
        f"{shlex.quote(str(limit_arg))} {shlex.quote(str(offset_arg))} "
        f"{shlex.quote(str(_PID_STALE_SECONDS))}",
        input_text=_LIST_COMMANDS_PY,
    )
    return json.loads(res.stdout)
