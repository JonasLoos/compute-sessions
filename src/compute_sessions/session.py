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
from compute_sessions.errors import RemoteError, SessionError, UsageError
from compute_sessions.login_helper import resolve_dest
from compute_sessions.models import SessionInfo, SessionStatus, SyncResult
from compute_sessions.project import project_id as derive_project_id
from compute_sessions.sync import parse_rsync_output, plan_sync


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ResourceRequest:
    """Per-activation resource request. None = unset: filled from the config defaults by Cluster.resolve_resources — or, when no resource is set at all on a re-activation or clone, the session's previous resources are reused (idle timeout: the previous one, else 20 min)."""
    partition: str | None = None
    gpus: int | None = None
    gpu_type: str | None = None
    mem: int | None = None
    cpus: int | None = None
    idle_timeout_minutes: int | None = None


def _resolve_request(cluster: Cluster, req: ResourceRequest, *, reuse: bool = False) -> dict | None:
    """Validate a resource request against the config before anything is written: an invalid `--gpu-type` must fail before a record exists. With `reuse` and no resource given, None: the login node reuses the session's previous resources."""
    if req.idle_timeout_minutes is not None and not 1 <= req.idle_timeout_minutes <= 10080:
        raise UsageError(f"idle_timeout_minutes must be between 1 and 10080, got {req.idle_timeout_minutes}")
    if reuse and all(v is None for v in (req.partition, req.gpus, req.gpu_type, req.mem, req.cpus)):
        return None
    return cluster.resolve_resources(partition=req.partition, gpus=req.gpus, gpu_type=req.gpu_type, mem=req.mem, cpus=req.cpus)


def _submit(cluster: Cluster, args: list[str], req: ResourceRequest, resources: dict | None, manifest: list[str] | None = None) -> SessionInfo:
    cfg = cluster.cfg
    params = dict(resources=resources, idle_timeout_minutes=req.idle_timeout_minutes, image=cfg.image, login=cfg.login_host or cfg.host, exclude_nodes=list(cfg.exclude_nodes))
    if manifest is not None:
        params["manifest"] = manifest
    info = SessionInfo.from_json(cluster.call("submit", args, retry=False, **params)["info"])
    # A previous activation's shared container connection points at a dead endpoint.
    cluster.access.close_container(info.session_id)
    return info


def _new_record(cluster: Cluster, project_id: str = "", project_path: str = "") -> dict:
    # Not live until the submit makes it pending: a create that dies before (client killed, network gone, so the abort cannot reach the cluster either) must not leave a jobless pending record, a state nothing reconciles.
    return SessionInfo(session_id=f"{cluster.cfg.session_prefix}_{secrets.token_hex(5)}", project_id=project_id, project_path=project_path, created_at=_utcnow(), status=SessionStatus.INACTIVE).to_json()


def _create_then(cluster: Cluster, record: dict, start: Callable[[], dict], finish: Callable[[dict], SessionInfo]) -> SessionInfo:
    """Run a create/clone request, then the rest of the setup, and mark the fresh session failed if it does not complete (broken transfer, scheduler refusal, Ctrl-C). An abort that cannot reach the cluster leaves the record inactive. An interrupted first request may still complete on the login node after the client is gone: the abort waits for it (or pre-empts it with a failed record). A refusal of the first request (session cap) wrote nothing."""
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


def clone(cluster: Cluster, sid: str, req: ResourceRequest) -> SessionInfo:
    """A new session for the same project, seeded with a server-side copy of the source's workdir (contents + sync manifest, no laptop round trip), then activated like create(). Logs stay with the source."""
    resources = _resolve_request(cluster, req, reuse=True)
    record = _new_record(cluster)
    return _create_then(cluster, record, lambda: cluster.call("clone", [sid], retry=False, timeout=3600, record=record, cap=cluster.cfg.max_sessions_per_repo),
                        lambda r: _submit(cluster, [r["sid"]], req, resources))


def activate(cluster: Cluster, sid: str, req: ResourceRequest) -> SessionInfo:
    """Submit the activation and return PENDING. Refuses while a previous job is live (an allocation can't be resized in place: deactivate first). Without resource flags, the previous activation's resources are reused."""
    return _submit(cluster, [sid], req, _resolve_request(cluster, req, reuse=True))


def deactivate(cluster: Cluster, sid: str, on_stopped: Callable[[list[dict], list[str]], None] | None = None) -> SessionInfo:
    """`on_stopped` sees the commands that were still running (command_id, command) — the cancel ends them — and the ids of queued commands that were dropped."""
    r = cluster.call("deactivate", [sid])
    sid = r["sid"]
    cluster.access.close_container(sid)
    if (r.get("stopped") or r.get("dropped")) and on_stopped:
        on_stopped(r.get("stopped") or [], r.get("dropped") or [])
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


def show(cluster: Cluster, sid: str, *, squeue: bool = True, probe: bool = True) -> dict:
    """Full detail: the enriched record, live job state (a gone job is reconciled), health, failure log tail, and a hint while pending."""
    r = cluster.call("show", [sid], squeue=squeue, probe=probe)
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


def sync(cluster: Cluster, sid: str, *, rebuild_manifest: bool = False, follow_symlinks: bool = False) -> tuple[str, SyncResult]:
    """Mirror the project's non-ignored files into the workdir. Paths the previous sync placed and the project no longer has are removed BEFORE the transfer — a path whose type changed locally (dir → file or back) must be unwound first, since rsync refuses to write a file over a non-empty directory. The manifest is written after the transfer, and also when it fails: a partial transfer may have placed any of the files, and only manifest entries are ever removed again (the deletions are idempotent). `rebuild_manifest` skips the deletions and just rewrites the manifest (recovery after out-of-band edits)."""
    r = cluster.call("sync_prepare", [sid])
    sid = r["sid"]
    cwd = project_cwd(SessionInfo.from_json(r["info"]))
    files = plan_sync(cwd)
    old = set(r["manifest"])
    deleted = [] if rebuild_manifest else sorted(old - set(files))
    errors = cluster.access.call("sync_delete", sid=sid, paths=deleted)["errors"] if deleted else []
    try:
        output = cluster.access.rsync_files(cwd, files, r["workdir"], follow_symlinks=follow_symlinks) if files else ""
    except RemoteError as exc:
        if set(files) != old:
            cluster.access.call("sync_finish", sid=sid, files=files)
        # rsync will not delete a non-empty directory that stands where the project now has a file (or link).
        blocked = re.findall(r"could not make way for new [^:]+: (.+)", exc.stderr or "")
        if blocked:
            paths = ", ".join(dict.fromkeys(p.strip() for p in blocked))
            raise SessionError(f"sync stopped: the workdir has a directory where the project now has a file ({paths}), and it holds files sync did not place — remove or move it, e.g. cs run {sid} 'rm -r <path>', then sync again") from exc
        raise
    added, updated, nbytes = parse_rsync_output(output)
    if rebuild_manifest or set(files) != old:
        cluster.access.call("sync_finish", sid=sid, files=files)
    return sid, SyncResult(added=sorted(added), updated=sorted(updated), deleted_outdated=deleted, bytes_transferred=nbytes, cleanup_errors=errors)


def logs(cluster: Cluster, args: list[str], *, since: tuple[int, int] | None = None, max_lines: int | None = None, resolve: bool = False) -> dict:
    """A command's output and state: `cs logs SESSION COMMAND_ID` args. Status exited | running | killed | missing; `out`/`err` are bytes from `since` (and only their last `max_lines` lines). A log bigger than one reply arrives in several requests. `resolve`: the id comes from the user — an unknown id is a UsageError."""
    r = cluster.call("logs", args, since=list(since or (0, 0)), max_lines=max_lines, resolve=resolve)
    out, err = [r["out"]], [r["err"]]
    while r.pop("more", False):
        r.update(cluster.call("logs", [r["sid"], r["cid"]], since=r["offsets"]))
        out.append(r["out"])
        err.append(r["err"])
    r["out"], r["err"] = (b"".join(base64.b64decode(c) for c in chunks) for chunks in (out, err))
    return r


def list_commands(cluster: Cluster, sid: str, *, limit: int | None = 50) -> dict:
    """{commands, total, …}: the session's recorded commands, newest first — {command_id, status, [exit_code, started_at, exited_at]} with status exited | running | killed. Internal venv-snapshot commands are hidden."""
    return cluster.call("commands", [sid], limit=max(0, int(limit or 0)))


def queue(cluster: Cluster, sid: str, command_id: str, command: str, release: bool = False, after: list[str] | tuple[str, ...] = ()) -> dict:
    """Queue a command on a pending session; the runner launches it once the session is active. `queued` is False when the session stopped being pending meanwhile (launch it directly then)."""
    from compute_sessions.run import container_script
    script = container_script(command_id, "__CS_QUEUED__", command=command, follow=False, release=release, after=after)
    return cluster.access.call("queue", retry=False, sid=sid, cid=command_id, command=command, script=base64.b64encode(script).decode())


def dequeue(cluster: Cluster, args: list[str]) -> dict:
    """Drop a queued command: `SESSION COMMAND_ID`."""
    return cluster.call("dequeue", args)


def copy(cluster: Cluster, src: str, paths: list[str], dst: str, dest: str) -> dict:
    """Copy workdir paths of session `src` into session `dst` (cp conventions, see resolve_dest), on the cluster."""
    return cluster.call("copy", retry=False, timeout=3600, src=src, paths=paths, dst=dst, dest=dest)


def delete(cluster: Cluster, sids: list[str]) -> dict:
    """Move stopped sessions into the cluster-side trash (exact ids only)."""
    return cluster.access.call("delete", retry=False, sids=sids)


def ls(cluster: Cluster, args: list[str]) -> str:
    return cluster.call("ls", args)["listing"]


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
            raise UsageError(
                f"{label} {str(resolved)!r} is outside the session's project_path {str(project)!r} and not in an agent scratchpad (/tmp/claude-*, /tmp/codex-*); refusing to operate"
            )
    return resolved


def _display_local(p: Path, project: Path) -> str:
    """Local paths echo relative to project_path when inside it; scratchpad paths stay absolute."""
    try:
        return str(p.relative_to(project))
    except ValueError:
        return str(p)


def _path_check(cluster: Cluster, sid: str, rel: str, child: str | None = None, make_dirs: bool = False) -> dict:
    return cluster.access.call("path_check", sid=sid, path=rel, child=child, make_dirs=make_dirs)


def download(cluster: Cluster, sid: str, info: SessionInfo, remote_rel: str, local_path: str) -> dict:
    """Copy a workdir file or directory to the local machine (cp/rsync destination conventions, see _resolve_dest). The transfer reads from the verified real path inside the workdir."""
    raw_local = str(local_path)
    resolved = _resolve_local_inside_project(Path(raw_local), info, "download local_path")
    pc = _path_check(cluster, sid, remote_rel)
    if pc["kind"] == "missing":
        glob = " (wildcards are not expanded — pass the paths, or the directory that holds them)" if any(c in remote_rel for c in "*?[") else ""
        raise UsageError(f"remote path does not exist: {pc['rel'] or '.'}{glob}")
    if raw_local.rstrip().endswith("/") and not resolved.is_dir():
        raise UsageError(f"local destination {raw_local!r} is not an existing directory ({resolved}) — create it first (local paths are relative to the project dir)")
    dest = Path(resolve_dest(
        (pc["rel"] or "workdir").rsplit("/", 1)[-1], str(resolved),
        src_is_dir=pc["kind"] == "dir",
        src_trailing_slash=remote_rel.strip() in ("", ".") or remote_rel.rstrip().endswith("/"),
        dest_trailing_slash=raw_local.rstrip().endswith("/"),
        dest_is_dir=resolved.is_dir(),
    ))
    if not dest.parent.is_dir():
        raise UsageError(f"local destination {raw_local!r}: parent directory {dest.parent} does not exist — create it first (local paths are relative to the project dir)")
    try:
        if pc["kind"] == "dir":
            if dest.is_file():
                raise UsageError(f"local destination {dest} exists as a file but the remote is a directory — remove it or pick another path")
            dest.mkdir(exist_ok=True)
            cluster.access.rsync_from(pc["real"], dest, contents_only=True)
        else:
            if dest.is_dir():
                raise UsageError(f"local destination {dest} exists as a directory but the remote is a file — pick another path")
            cluster.access.rsync_from(pc["real"], dest)
    except OSError as exc:
        raise SessionError(f"local destination {raw_local!r} is unusable: {exc}") from exc
    project = Path(info.project_path).expanduser().resolve()
    return {"local_path": _display_local(dest, project) + ("/" if pc["kind"] == "dir" else ""), "remote_path": pc["rel"]}


def upload(cluster: Cluster, sid: str, info: SessionInfo, local_path: str, remote_rel: str) -> dict:
    """Copy a local file or directory into the workdir, no .gitignore filtering (cp/rsync destination conventions, see _resolve_dest). Empty `remote_rel` = the local basename at the workdir root. A symlink's content is sent under the link's own name, like cp. The transfer writes to the verified real path inside the workdir."""
    raw_local = str(local_path)
    resolved = _resolve_local_inside_project(Path(raw_local), info, "upload local_path")
    if not resolved.exists():
        raise UsageError(f"local path does not exist: {resolved}")
    name = Path(raw_local.strip()).name
    if name in ("", ".", ".."):
        name = resolved.name
    rel = (remote_rel or "").strip()
    if not rel or rel == ".":
        rel = name
    pc = _path_check(cluster, sid, rel, child=name, make_dirs=True)
    kind, remote = pc["kind"], pc["rel"]
    src_is_dir = resolved.is_dir()
    dest = resolve_dest(name, remote, src_is_dir=src_is_dir, src_trailing_slash=raw_local.rstrip().endswith("/"), dest_trailing_slash=rel.endswith("/"), dest_is_dir=kind == "dir")
    dest_real = pc["real"] if dest == remote else f"{pc['real']}/{name}"
    if src_is_dir:
        if (kind if dest == remote else pc["child_kind"]) == "file":
            raise UsageError(f"remote destination {dest!r} exists as a file but the local path is a directory — pick another path")
        cluster.access.rsync_to(resolved, dest_real, contents_only=True)
    else:
        cluster.access.rsync_to(resolved, dest_real)
    project = Path(info.project_path).expanduser().resolve()
    return {"local_path": _display_local(resolved, project), "remote_path": dest + ("/" if src_is_dir else "")}
