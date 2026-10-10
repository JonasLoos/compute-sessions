"""Session operations. Each does its cluster-side work as one login-node helper request (login_helper.py); the local side keeps the file list, local path checks, and the transfers."""
from __future__ import annotations

import base64
import re
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from compute_sessions.cluster import Cluster
from compute_sessions.errors import SessionError
from compute_sessions.models import SessionInfo, SessionStatus, SyncResult
from compute_sessions.project import project_id as derive_project_id
from compute_sessions.sync import parse_rsync_output, plan_sync


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _token(token: str | None) -> list[str]:
    return [token] if token is not None else []


@dataclass
class ResourceRequest:
    """Per-activation resource request. None = unset (filled from the config defaults by Cluster.resolve_resources)."""
    partition: str | None = None
    gpus: int | None = None
    gpu_type: str | None = None
    mem: int | None = None
    idle_timeout_minutes: int = 20


def _resolve_request(cluster: Cluster, req: ResourceRequest) -> dict:
    """Validate a resource request against the config before anything is written: an invalid `--gpu-type` must fail before a record exists."""
    resolved = cluster.resolve_resources(partition=req.partition, gpus=req.gpus, gpu_type=req.gpu_type, mem=req.mem)
    if not 1 <= req.idle_timeout_minutes <= 10080:
        raise SessionError(f"idle_timeout_minutes must be between 1 and 10080, got {req.idle_timeout_minutes}")
    return resolved


def _submit(cluster: Cluster, args: list[str], req: ResourceRequest, resources: dict, manifest: list[str] | None = None) -> SessionInfo:
    cfg = cluster.cfg
    params = dict(resources=resources, idle_timeout_minutes=req.idle_timeout_minutes, image=cfg.image, login=cfg.login_host or cfg.host, exclude_nodes=list(cfg.exclude_nodes))
    if manifest is not None:
        params["manifest"] = manifest
    info = SessionInfo.from_json(cluster.call("submit", args, retry=False, **params)["info"])
    # A previous activation's shared container connection points at a dead endpoint.
    cluster.access.close_container(info.session_id)
    return info


def _new_record(cluster: Cluster, project_id: str = "", project_path: str = "") -> dict:
    return SessionInfo(session_id=f"{cluster.cfg.session_prefix}_{secrets.token_hex(5)}", project_id=project_id, project_path=project_path, created_at=_utcnow()).to_json()


def _create_then(cluster: Cluster, record: dict, start: Callable[[], dict], finish: Callable[[dict], SessionInfo]) -> SessionInfo:
    """Run a create/clone request, then the rest of the setup, and mark the fresh session failed if it does not complete (broken transfer, scheduler refusal, Ctrl-C) — a jobless pending record is a state nothing reconciles. An interrupted first request may still complete on the login node after the client is gone: the abort waits for it (or pre-empts it with a failed record). A refusal of the first request (session cap) wrote nothing."""
    started = False
    try:
        r = start()
        started = True
        return finish(r)
    except BaseException as exc:
        if started or not isinstance(exc, SessionError):
            try:
                cluster.access.call("abort", sid=record["session_id"], record=record)
            except Exception:
                pass
        raise


def create(cluster: Cluster, req: ResourceRequest, cwd: Path) -> SessionInfo:
    """Create a session for the project at `cwd`: record, initial sync, and the first activation (returns PENDING)."""
    cwd = cwd.resolve()
    resources = _resolve_request(cluster, req)
    files = plan_sync(cwd)
    record = _new_record(cluster, derive_project_id(cwd), str(cwd))

    def finish(r: dict) -> SessionInfo:
        if files:
            cluster.access.rsync_files(cwd, files, r["workdir"])
        return _submit(cluster, [r["sid"]], req, resources, manifest=files)
    return _create_then(cluster, record, lambda: cluster.access.call("create", retry=False, record=record, cap=cluster.cfg.max_sessions_per_repo), finish)


def clone(cluster: Cluster, token: str | None, req: ResourceRequest) -> SessionInfo:
    """A new session for the same project, seeded with a server-side copy of the source's workdir (contents + sync manifest, no laptop round trip), then activated like create(). Logs stay with the source."""
    resources = _resolve_request(cluster, req)
    record = _new_record(cluster)
    return _create_then(cluster, record, lambda: cluster.call("clone", _token(token), retry=False, timeout=3600, record=record, cap=cluster.cfg.max_sessions_per_repo),
                        lambda r: _submit(cluster, [r["sid"]], req, resources))


def activate(cluster: Cluster, token: str | None, req: ResourceRequest) -> SessionInfo:
    """Submit the activation and return PENDING. Refuses while a previous job is live (an allocation can't be resized in place: deactivate first)."""
    return _submit(cluster, _token(token), req, _resolve_request(cluster, req))


def deactivate(cluster: Cluster, token: str | None) -> SessionInfo:
    r = cluster.call("deactivate", _token(token))
    sid = r["sid"]
    cluster.access.close_container(sid)
    info = SessionInfo.from_json(r["info"])
    if not r.get("job_id"):
        return info
    # The runner's teardown writes the final status; wait briefly for it so the caller sees the real post-cancel state. A job that was still queued never ran a runner — the finish call marks it inactive (compare-and-set on the job id: a re-activation meanwhile owns the record).
    for _ in range(10):
        time.sleep(0.5)
        info = SessionInfo.from_json(cluster.call("show", [sid])["info"])
        if info.status not in (SessionStatus.ACTIVE, SessionStatus.PENDING):
            return info
    return SessionInfo.from_json(cluster.access.call("deactivate", sid=sid, finish=r["job_id"])["info"])


_PENDING_HINT = "allocation can take a while when the cluster is busy or the request is constrained (e.g. a pinned gpu type) — keep waiting with `cs show -w`, or `cs deactivate` to cancel and retry with different resources"


def show(cluster: Cluster, token: str | None, *, squeue: bool = True, probe: bool = True) -> dict:
    """Full detail: the enriched record, live job state (a gone job is reconciled), health, failure log tail, and a hint while pending."""
    r = cluster.call("show", _token(token), squeue=squeue, probe=probe)
    info = SessionInfo.from_json(r["info"])
    detail: dict = {"session_id": r["sid"], "info": info.to_enriched_json()}
    for k in ("job", "health", "failure_log_tail"):
        if r.get(k):
            detail[k] = r[k]
    if info.status == SessionStatus.PENDING:
        detail["hint"] = _PENDING_HINT
    return detail


def wait_until_settled(cluster: Cluster, session_id: str, on_poll: Callable[[dict, float, bool], None] | None = None, sleep=time.sleep) -> dict:
    """Block while the session is pending; returns the first non-pending show() detail. Short requests every 2 s (5 s after the first 2 minutes) — activation is noticed within a few seconds — while the scheduler is asked at most every 30 s, so a long queue wait costs the controller nothing extra. `on_poll(detail, elapsed, fresh_job_state)` sees each round."""
    start = time.monotonic()
    last_squeue: float | None = None
    while True:
        now = time.monotonic()
        squeue = last_squeue is None or now - last_squeue >= 30
        detail = show(cluster, session_id, squeue=squeue, probe=False)
        if squeue:
            last_squeue = now
        if detail["info"]["status"] != "pending":
            return detail
        if on_poll:
            on_poll(detail, now - start, squeue)
        sleep(2 if now - start < 120 else 5)


def project_cwd(info: SessionInfo) -> Path:
    """The session's local project directory (fixed at create time); raises if it no longer exists."""
    p = Path(info.project_path)
    if not p.is_dir():
        raise SessionError(f"session {info.session_id} project_path {info.project_path!r} no longer exists on this machine")
    return p


def sync(cluster: Cluster, token: str | None, *, rebuild_manifest: bool = False, follow_symlinks: bool = False) -> tuple[str, SyncResult]:
    """Mirror the project's non-ignored files into the workdir. Paths the previous sync placed and the project no longer has are removed BEFORE the transfer — a path whose type changed locally (dir → file or back) must be unwound first, since rsync refuses to write a file over a non-empty directory. The manifest is written after the transfer: if rsync fails, the next sync repeats the (idempotent) deletions. `rebuild_manifest` skips the deletions and just rewrites the manifest (recovery after out-of-band edits)."""
    r = cluster.call("sync_prepare", _token(token))
    sid = r["sid"]
    cwd = project_cwd(SessionInfo.from_json(r["info"]))
    files = plan_sync(cwd)
    old = set(r["manifest"])
    deleted = [] if rebuild_manifest else sorted(old - set(files))
    errors = cluster.access.call("sync_delete", sid=sid, paths=deleted)["errors"] if deleted else []
    added, updated, nbytes = parse_rsync_output(cluster.access.rsync_files(cwd, files, r["workdir"], follow_symlinks=follow_symlinks) if files else "")
    if rebuild_manifest or set(files) != old:
        cluster.access.call("sync_finish", sid=sid, files=files)
    return sid, SyncResult(added=sorted(added), updated=sorted(updated), deleted_outdated=deleted, bytes_transferred=nbytes, cleanup_errors=errors)


def logs(cluster: Cluster, args: list[str], *, since: tuple[int, int] | None = None, max_lines: int | None = None) -> dict:
    """A command's output and state: `cs logs [SESSION] [COMMAND_ID]` args. Status exited | running | killed | missing; `out`/`err` are bytes from `since` (and only their last `max_lines` lines). A log bigger than one reply arrives in several requests."""
    r = cluster.call("logs", args, maybe=True, since=list(since or (0, 0)), max_lines=max_lines)
    out, err = [r["out"]], [r["err"]]
    while r.pop("more", False):
        r.update(cluster.call("logs", [r["sid"], r["cid"]], since=r["offsets"]))
        out.append(r["out"])
        err.append(r["err"])
    r["out"], r["err"] = (b"".join(base64.b64decode(c) for c in chunks) for chunks in (out, err))
    return r


def list_commands(cluster: Cluster, token: str | None, *, limit: int | None = 50) -> dict:
    """{commands, total, …}: the session's recorded commands, newest first — {command_id, status, [exit_code, started_at, exited_at]} with status exited | running | killed. Internal venv-snapshot commands are hidden."""
    return cluster.call("commands", _token(token), limit=max(0, int(limit or 0)))


def ls(cluster: Cluster, args: list[str]) -> str:
    return cluster.call("ls", args, maybe=True)["listing"]


# Agent scratchpad roots (Claude Code: /tmp/claude-<uid>/…; codex/other agents analogous) are allowed for upload/download despite living outside the project — agents stage helper scripts and fetch results there, and these are agent-owned temp dirs with nothing to exfiltrate or clobber. macOS resolves /tmp to /private/tmp, hence the optional prefix.
_SCRATCH_ROOT_RE = re.compile(r"^(?:/private)?/tmp/(?:claude|codex|agent)-[^/]+/")


def _resolve_local_inside_project(local: Path, info: SessionInfo, label: str) -> Path:
    """Resolve `local` against the session's project_path (relative paths are taken relative to it, mirroring the workdir-relative remote side) and refuse a result outside both the project and the agent scratchpad roots — download/upload must not become arbitrary local read/write primitives (a prompt-injected agent could otherwise overwrite `~/.ssh/authorized_keys` or exfiltrate `~/.ssh/id_rsa`). Resolving first means a symlink inside the project pointing outside is still refused."""
    if not info.project_path:
        raise SessionError(f"session {info.session_id} has no stored project_path — recreate the session")
    project = Path(info.project_path).expanduser().resolve()
    local = local.expanduser()
    resolved = (local if local.is_absolute() else project / local).resolve()
    try:
        resolved.relative_to(project)
    except ValueError:
        if not _SCRATCH_ROOT_RE.match(str(resolved)):
            raise SessionError(
                f"{label} {str(resolved)!r} is outside the session's project_path {str(project)!r} and not in an agent scratchpad (/tmp/claude-*, /tmp/codex-*); refusing to operate"
            )
    return resolved


# Destination semantics shared by download/upload, following cp/rsync conventions:
#   - A dest that ends with `/` or already exists as a directory receives the source INSIDE it (dest/<basename>).
#   - Otherwise the dest names the resulting file/dir itself.
#   - Directory contents only merge directly into dest when the SOURCE ends with `/` (rsync's src/ idiom; also `.`/empty), or when dest is an existing dir of the same basename (so a repeated `upload data data` stays idempotent instead of nesting data/data).
#   - Destination directories must already exist — like cp; an implicit mkdir -p turns a typo'd destination into a silently created directory.
def _resolve_dest(src_base: str, dest: str, *, src_is_dir: bool, src_trailing_slash: bool, dest_trailing_slash: bool, dest_is_dir: bool) -> str:
    if src_is_dir and src_trailing_slash:
        return dest  # explicit contents-merge
    if dest_trailing_slash or (dest_is_dir and not (src_is_dir and src_base == dest.rstrip("/").rsplit("/", 1)[-1])):
        return f"{dest.rstrip('/')}/{src_base}" if dest.rstrip("/") else src_base
    return dest


def _display_local(p: Path, project: Path) -> str:
    """Local paths echo relative to project_path when inside it; scratchpad paths stay absolute."""
    try:
        return str(p.relative_to(project))
    except ValueError:
        return str(p)


def _path_check(cluster: Cluster, sid: str, rel: str, child: str | None = None) -> dict:
    return cluster.access.call("path_check", sid=sid, path=rel, child=child)


def download(cluster: Cluster, sid: str, info: SessionInfo, remote_rel: str, local_path: str) -> dict:
    """Copy a workdir file or directory to the local machine (cp/rsync destination conventions, see _resolve_dest). The transfer reads from the verified real path inside the workdir."""
    raw_local = str(local_path)
    resolved = _resolve_local_inside_project(Path(raw_local), info, "download local_path")
    pc = _path_check(cluster, sid, remote_rel)
    if pc["kind"] == "missing":
        raise SessionError(f"remote path does not exist: {pc['rel'] or '.'}")
    if raw_local.rstrip().endswith("/") and not resolved.is_dir():
        raise SessionError(f"local destination {raw_local!r} is not an existing directory — create it first or drop the trailing '/'")
    dest = Path(_resolve_dest(
        (pc["rel"] or "workdir").rsplit("/", 1)[-1], str(resolved),
        src_is_dir=pc["kind"] == "dir",
        src_trailing_slash=remote_rel.strip() in ("", ".") or remote_rel.rstrip().endswith("/"),
        dest_trailing_slash=raw_local.rstrip().endswith("/"),
        dest_is_dir=resolved.is_dir(),
    ))
    if not dest.parent.is_dir():
        raise SessionError(f"local destination {raw_local!r}: parent directory {dest.parent} does not exist — create it first")
    try:
        if pc["kind"] == "dir":
            if dest.is_file():
                raise SessionError(f"local destination {dest} exists as a file but the remote is a directory — remove it or pick another path")
            dest.mkdir(exist_ok=True)
            cluster.access.rsync_from(pc["real"], dest, contents_only=True)
        else:
            if dest.is_dir():
                raise SessionError(f"local destination {dest} exists as a directory but the remote is a file — pick another path")
            cluster.access.rsync_from(pc["real"], dest)
    except OSError as exc:
        raise SessionError(f"local destination {raw_local!r} is unusable: {exc}") from exc
    project = Path(info.project_path).expanduser().resolve()
    return {"local_path": _display_local(dest, project) + ("/" if pc["kind"] == "dir" else ""), "remote_path": pc["rel"]}


def upload(cluster: Cluster, sid: str, info: SessionInfo, local_path: str, remote_rel: str) -> dict:
    """Copy a local file or directory into the workdir, no .gitignore filtering (cp/rsync destination conventions, see _resolve_dest). Empty `remote_rel` = the local basename at the workdir root. The transfer writes to the verified real path inside the workdir."""
    raw_local = str(local_path)
    resolved = _resolve_local_inside_project(Path(raw_local), info, "upload local_path")
    if not resolved.exists():
        raise SessionError(f"local path does not exist: {resolved}")
    rel = (remote_rel or "").strip()
    if not rel or rel == ".":
        rel = resolved.name
    pc = _path_check(cluster, sid, rel, child=resolved.name)
    kind, remote = pc["kind"], pc["rel"]
    if kind == "missing" and rel.endswith("/"):
        raise SessionError(f"remote destination {rel!r} is not an existing directory — create it first or drop the trailing '/'")
    src_is_dir = resolved.is_dir()
    dest = _resolve_dest(resolved.name, remote, src_is_dir=src_is_dir, src_trailing_slash=raw_local.rstrip().endswith("/"), dest_trailing_slash=rel.endswith("/"), dest_is_dir=kind == "dir")
    if dest == remote and kind == "missing" and not pc["parent_ok"]:
        raise SessionError(f"remote destination {rel!r}: parent directory does not exist — create it first")
    dest_real = pc["real"] if dest == remote else f"{pc['real']}/{resolved.name}"
    if src_is_dir:
        if (kind if dest == remote else pc["child_kind"]) == "file":
            raise SessionError(f"remote destination {dest!r} exists as a file but the local path is a directory — pick another path")
        cluster.access.rsync_to(resolved, dest_real, contents_only=True)
    else:
        cluster.access.rsync_to(resolved, dest_real)
    project = Path(info.project_path).expanduser().resolve()
    return {"local_path": _display_local(resolved, project), "remote_path": dest + ("/" if src_is_dir else "")}
