from __future__ import annotations

import base64
import json
import logging
import secrets
import shlex
import socket
import threading
import time
from pathlib import Path

from compute_sessions.config import SourceConfig
from compute_sessions.errors import RemoteError, SessionError
from compute_sessions.models import SessionInfo, SessionStatus
from compute_sessions.paths import RemotePaths
from compute_sessions.ssh import HostAccess, InstanceAccess, LocalAccess
from compute_sessions.sync import sync_session
from compute_sessions.vast_api import Ledger, VastClient, parse_filter_clauses


log = logging.getLogger(__name__)


def read_config_json(access, paths: RemotePaths, session_id: str) -> SessionInfo:
    config_path = paths.config_file(session_id)
    # Distinguish "session not found" (typed error the caller can act on) from any other cat failure. Otherwise the raw RemoteError bubbles up with "cat: ... No such file or directory" and callers can't tell whether the session was GC'd, never created, or whether something else went wrong.
    try:
        res = access.run(f"cat {shlex.quote(config_path)}", check=True)
    except RemoteError as exc:
        if exc.exit_code == 1 and exc.stderr and "No such file or directory" in exc.stderr:
            raise SessionError(
                f"session {session_id!r} not found on source {access.source.name!r} — "
                f"config.json missing at {config_path}. It may have been "
                "garbage-collected or never created. `cs list` shows "
                "currently-known sessions."
            ) from exc
        raise
    return SessionInfo.from_json(json.loads(res.stdout))


def write_config_json(access, paths: RemotePaths, info: SessionInfo) -> None:
    body = json.dumps(info.to_json(), indent=2)
    final = paths.config_file(info.session_id)
    # Atomic write: stream into a unique temp file in the same dir, then `mv` it over the target. config.json has several concurrent writers (this host process plus the runner), and a plain `cat >` truncates-then-writes — a concurrent reader could observe a half-written file. `mv` within one dir is atomic; a per-write nonce keeps concurrent writers from colliding on the temp name.
    tmp = f"{final}.tmp.{secrets.token_hex(6)}"
    q_tmp = shlex.quote(tmp)
    q_final = shlex.quote(final)
    access.run(
        f"cat > {q_tmp} && mv -f {q_tmp} {q_final} || {{ rm -f {q_tmp}; exit 1; }}",
        input_text=body,
    )


def list_config_jsons(access, project_id_filter: str | None = None) -> list[SessionInfo]:
    paths = access.paths
    # Concatenate each config.json with a NUL separator. JSON text can't contain raw NULs, so this round-trips cleanly regardless of content.
    cmd = (
        f"for d in {shlex.quote(paths.sessions_root())}/*/; do "
        f"  [ -f \"$d/config.json\" ] && cat \"$d/config.json\" && printf '\\0'; "
        f"done"
    )
    res = access.run(cmd, check=False)
    if res.returncode != 0:
        return []
    out: list[SessionInfo] = []
    for chunk in res.stdout.split("\x00"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            out.append(SessionInfo.from_json(json.loads(chunk)))
        except Exception as exc:
            log.warning("skipping unparsable session config: %s", exc)
            continue
    if project_id_filter is not None:
        out = [s for s in out if s.project_id == project_id_filter]
    return out


class Backend:
    """Source-type-specific operations behind one interface: session-record I/O, resource resolution, submit/poll/cancel of the activation, and routing of file/exec traffic to the right access object.

    Two access roles, identical for slurm/docker but split for vast:
      state       — where session records (config.json, the registry) live. Must outlive activations.
      work/logs   — where the workdir, command logs and the session sshd live. For vast that's the rented instance itself, which dies on deactivate.
    """

    # Whether the source-side runner writes the session's final status on shutdown (slurm/docker). Vast instances vaporize on destroy, so the local side records it instead.
    runner_finalizes = True
    # Whether create() can sync the project into the workdir right away. A vast workdir doesn't exist until the instance boots — first contact after activation populates it instead (ensure_ready).
    syncs_at_create = True

    def __init__(self, source: SourceConfig, access):
        self.source = source
        self.access = access

    @property
    def name(self) -> str:
        return self.source.name

    # ---------- access routing ----------

    @property
    def state(self):
        """Access for session records."""
        return self.access

    def work_access(self, info: SessionInfo):
        """Access for workdir operations (sync/upload/download/ls) and in-container exec. May raise SessionError when the session has no live filesystem (vast, no instance)."""
        return self.access

    def logs_access(self, info: SessionInfo):
        """Access for command-log reads (logs/list_commands). Never raises for lack of a live instance — logs outlive activations on every source."""
        return self.access

    # ---------- session-record I/O ----------

    def read_info(self, session_id: str) -> SessionInfo:
        return read_config_json(self.state, self.state.paths, session_id)

    def write_info(self, info: SessionInfo) -> None:
        write_config_json(self.state, self.state.paths, info)

    def list_infos(self, project_id_filter: str | None = None) -> list[SessionInfo]:
        return list_config_jsons(self.state, project_id_filter)

    def init_session_dirs(self, session_id: str) -> None:
        paths = self.state.paths
        self.state.run(
            f"mkdir -p {shlex.quote(paths.workdir(session_id))} "
            f"{shlex.quote(paths.logs_dir(session_id))}"
        )

    def copy_workdir(self, src_session_id: str, dst_session_id: str) -> None:
        """Server-side copy of one session's workdir contents (plus its sync manifest, so later `sync` calls keep tracking the mirrored files) into another's — the data never leaves the source. Both sessions live under the same base, so plain `cp -a` on the source host does it."""
        p = self.state.paths
        self.state.run(
            f"cp -a {shlex.quote(p.workdir(src_session_id))}/. {shlex.quote(p.workdir(dst_session_id))}/ && "
            f"{{ cp {shlex.quote(p.manifest(src_session_id))} {shlex.quote(p.manifest(dst_session_id))} 2>/dev/null || true; }}"
        )

    # ---------- lifecycle ----------

    def resolve_resources(self, *, partition: str | None, gpus: int | None, gpu_type: str | None, mem: int | None) -> dict:
        """Validate a per-activation resource request; return the dict stored on SessionInfo.resources. All args arrive as None when the caller didn't specify them."""
        raise NotImplementedError

    def submit(self, info: SessionInfo) -> str:
        """Start the activation (config.json is already persisted); return the job handle (slurm job id / runner pid / vast instance id)."""
        raise NotImplementedError

    def job_state(self, info: SessionInfo) -> dict:
        """Live state of the current activation's job: {state, reason, node}. `state` ∈ {"pending", "running", "gone", "unknown"} — "gone" means the source has no trace of the job (finished or never ran), "unknown" means the query itself failed/timed out."""
        raise NotImplementedError

    def cancel(self, info: SessionInfo) -> None:
        """Stop the activation (best-effort; the runner's cleanup writes the final status)."""
        raise NotImplementedError

    def tunnel_target(self, info: SessionInfo) -> str | None:
        """Remote side of the ssh `-L` forward to the session's sshd — a Unix socket path or `127.0.0.1:<port>`. None when no tunnel is involved (vast: direct connection) or the session never activated."""
        raise NotImplementedError

    def probe(self, info: SessionInfo) -> dict:
        """Health-check the session's sshd endpoint; returns {reachable, [detail]}."""
        raise NotImplementedError

    def describe(self) -> str:
        """One-line summary of this source and its resource vocabulary, rendered into cs --help."""
        raise NotImplementedError

    # ---------- hooks with sensible defaults ----------

    def ensure_ready(self, info: SessionInfo) -> None:
        """Called by `run` once the session is ACTIVE, before executing. Vast uses it to populate the fresh instance's empty workdir."""

    def after_sync(self, info: SessionInfo) -> None:
        """Called after an explicit `cs sync` completed."""

    def reset_transport(self, info: SessionInfo) -> None:
        """Drop any cached transport state for this session (stale tunnels / instance connections) — called before activate and during deactivate."""
        self.access.close_tunnel(info.session_id, self.tunnel_target(info))

    def failure_log_tail(self, info: SessionInfo) -> str:
        """Tail of the most recent runner log, for FAILED sessions."""
        logs_q = shlex.quote(self.state.paths.logs_dir(info.session_id))
        res = self.state.run(
            f"f=$(ls -t {logs_q}/runner* 2>/dev/null | head -1); "
            f"[ -n \"$f\" ] && tail -n 30 \"$f\"; true",
            check=False,
        )
        return res.stdout


class SlurmBackend(Backend):
    """SLURM cluster: sessions are sbatch jobs running an Apptainer container, reached via a reverse-SSH tunnel socket on the login node's shared FS."""

    def resolve_resources(self, *, partition: str | None, gpus: int | None, gpu_type: str | None, mem: int | None) -> dict:
        cfg = self.source
        partition = partition or cfg.default_partition
        if partition not in cfg.partitions:
            raise SessionError(
                f"invalid partition {partition!r} for source {cfg.name!r}; "
                f"must be one of {sorted(cfg.partitions)}"
            )

        # cpu-* partitions have no GPUs, so gpus is forced to 0 there. The default gpus=1 suits gpu partitions; a cpu session simply drops it (an explicit gpus>0 on a cpu partition is dropped, not rejected — you picked a CPU partition).
        gpus = 1 if gpus is None else gpus
        is_gpu_partition = not partition.startswith("cpu-")
        resolved_gpus = gpus if is_gpu_partition else 0
        if not 0 <= resolved_gpus <= 16:
            raise SessionError(f"gpus must be between 0 and 16, got {resolved_gpus}")

        # gpu_type → sbatch --constraint. Only meaningful when actually requesting GPUs. Validate every `|`-separated token against the source's allowlist — the value flows into the sbatch shell command, so an unvetted string would be a shell-injection vector.
        if gpu_type is not None:
            gpu_type = gpu_type.strip() or None
        if gpu_type is not None:
            if resolved_gpus == 0:
                raise SessionError(
                    f"gpu_type={gpu_type!r} requires a partition with gpus>0 "
                    f"(got partition={partition!r}, gpus={resolved_gpus})"
                )
            tokens = [t.strip() for t in gpu_type.split("|")]
            bad = [t for t in tokens if t not in cfg.gpu_types]
            if not tokens or "" in tokens or bad:
                raise SessionError(
                    f"invalid gpu_type {gpu_type!r} for source {cfg.name!r}; each '|'-separated "
                    f"value must be one of {sorted(cfg.gpu_types)}"
                )
            gpu_type = "|".join(tokens)

        mem = cfg.default_mem if mem is None else mem
        if not 1 <= mem <= 6000:
            raise SessionError(f"mem must be between 1 and 6000 GB, got {mem}")

        return {"partition": partition, "gpus": resolved_gpus, "gpu_type": gpu_type, "mem": mem}

    def submit(self, info: SessionInfo) -> str:
        paths = self.access.paths
        r = info.resources
        # partition/gpu_type are validated against allowlists in resolve_resources, but we still shlex.quote every flag as defense-in-depth — the whole string is handed to `bash -lc`. Wall-clock limit is implied by the partition, so no --time.
        flags: list[str] = [
            "--parsable",
            shlex.quote(f"--job-name=cs-{info.session_id}"),
            shlex.quote(f"--partition={r['partition']}"),
        ]
        if self.source.exclude_nodes:
            flags.append(shlex.quote(f"--exclude={','.join(self.source.exclude_nodes)}"))
        if r.get("gpus"):
            flags.append(shlex.quote(f"--gpus-per-node={r['gpus']}"))
        if r.get("gpu_type"):
            flags.append(shlex.quote(f"--constraint={r['gpu_type']}"))
        flags.append(shlex.quote(f"--mem={r['mem']}G"))
        flags.append(shlex.quote(f"--output={paths.logs_dir(info.session_id)}/runner-%j.out"))
        login = self.source.login_host or self.source.host
        runner_args = (
            f"--backend slurm --base {shlex.quote(paths.base)} "
            f"--session-id {shlex.quote(info.session_id)} --login {shlex.quote(login)}"
        )
        cmd = (
            "sbatch --export=ALL "
            + " ".join(flags)
            + f" {shlex.quote(paths.runner())} {runner_args}"
        )
        res = self.access.run(cmd)
        job_id = res.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise SessionError(f"unexpected scheduler response while submitting: {res.stdout!r}")
        return job_id

    def job_state(self, info: SessionInfo) -> dict:
        if not info.job_id:
            return {"state": "gone", "reason": None, "node": None}
        # `squeue -o %R` is overloaded: for PENDING jobs it's the pending reason (Priority, Resources, ...); for RUNNING jobs it's the nodelist. Split into separate fields so the caller doesn't have to condition on state to interpret the value. `%L` is the remaining wall clock — surfaced so agents can see a job's 5h partition window closing instead of discovering it as a silent mid-run kill.
        #
        # `timeout 5s` guards against a stuck slurmctld (observed: `show`/`list` calls hanging long enough to blind the agent). Tag a sentinel so we can distinguish timeout (state="unknown") from a healthy empty result (state="gone").
        sq = self.access.run(
            f"timeout 5s squeue -h -j {shlex.quote(info.job_id)} -o '%T|%L|%R' 2>/dev/null; "
            f"rc=$?; if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then echo __CS_SQUEUE_TIMEOUT__; fi; "
            f"exit 0",
            check=False,
        )
        text = sq.stdout
        if "__CS_SQUEUE_TIMEOUT__" in text:
            return {"state": "unknown", "reason": "scheduler query timed out", "node": None}
        line = text.strip()
        if not line:
            # sbatch --parsable returns only after SLURM has queued the job, so an empty squeue result means the job has finished — safe to treat as gone without a grace period.
            return {"state": "gone", "reason": None, "node": None}
        state, time_left, col = (line.split("|", 2) + ["", ""])[:3]
        if state == "RUNNING":
            return {"state": "running", "reason": None, "node": col, "time_left": time_left or None}
        if state in ("PENDING", "CONFIGURING"):
            return {"state": "pending", "reason": col, "node": None, "time_left": time_left or None}
        return {"state": "pending", "reason": f"{state}: {col}", "node": None}

    def cancel(self, info: SessionInfo) -> None:
        if info.job_id:
            self.access.run(f"scancel {shlex.quote(info.job_id)}", check=False)

    def tunnel_target(self, info: SessionInfo) -> str | None:
        return self.access.paths.socket(info.session_id)

    def probe(self, info: SessionInfo) -> dict:
        """Connect with nc to the reverse-tunnel Unix socket on the login node and read the SSH identification banner sshd sends on accept. A valid `SSH-2.0-…` line means both the reverse tunnel and the in-container sshd are alive."""
        quoted = shlex.quote(self.access.paths.socket(info.session_id))
        cmd = (
            f"if [ ! -S {quoted} ]; then echo MISSING; exit 0; fi; "
            f"timeout 3 nc -U {quoted} </dev/null 2>&1 | head -c 256 || true"
        )
        res = self.access.run(cmd, check=False)
        out = res.stdout.strip()
        if out == "MISSING":
            return {"reachable": False, "detail": "endpoint not created yet (session may still be starting)"}
        banner = out.splitlines()[0] if out else ""
        if banner.startswith("SSH-"):
            return {"reachable": True}
        return {"reachable": False, "detail": out or "no response"}

    def describe(self) -> str:
        cfg = self.source
        parts = [f"slurm cluster ({cfg.host}); partitions: {', '.join(cfg.partitions)} (default {cfg.default_partition})"]
        if cfg.gpu_types:
            parts.append(f"gpu types: {', '.join(cfg.gpu_types)} (OR with '|')")
        parts.append(f"mem in GB (default {cfg.default_mem})")
        if cfg.description:
            parts.append(cfg.description)
        return "; ".join(parts)


class DockerBackend(Backend):
    """Always-on ssh host (e.g. a gaming PC running WSL): sessions are Docker containers started by a detached runner.py process on the host, with sshd published on the host's 127.0.0.1. No scheduler — the whole machine is yours, so there are no resource args."""

    def resolve_resources(self, *, partition: str | None, gpus: int | None, gpu_type: str | None, mem: int | None) -> dict:
        given = {k: v for k, v in {"partition": partition, "gpus": gpus, "gpu_type": gpu_type, "mem": mem}.items() if v is not None}
        if given:
            raise SessionError(
                f"source {self.source.name!r} is a docker host with no scheduler — it takes no "
                f"resource args (got {given}); the session gets the whole machine"
            )
        return {}

    def submit(self, info: SessionInfo) -> str:
        paths = self.access.paths
        log_path = f"{paths.logs_dir(info.session_id)}/runner.out"
        # Detached launch that survives the ssh exec: nohup + full stdio redirect (ssh would otherwise hang on the open pipe), orphaned to init when the remote shell exits. `$!` is the python pid — the runner's job handle.
        cmd = (
            f"nohup python3 {shlex.quote(paths.runner())} --backend docker "
            f"--base {shlex.quote(paths.base)} --session-id {shlex.quote(info.session_id)} "
            f"> {shlex.quote(log_path)} 2>&1 < /dev/null & echo $!"
        )
        res = self.access.run(cmd)
        pid = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
        if not pid.isdigit():
            raise SessionError(f"unexpected response while launching runner: {res.stdout!r}")
        return pid

    def job_state(self, info: SessionInfo) -> dict:
        if not info.job_id:
            return {"state": "gone", "reason": None, "node": None}
        # The runner is a plain host process (no PID-namespace mismatch — that only affects PIDs recorded *inside* containers). Match on argv, not bare pid liveness, so a recycled pid can't read as a live runner.
        res = self.access.run(f"ps -p {shlex.quote(info.job_id)} -o args= 2>/dev/null || true", check=False)
        args = res.stdout.strip()
        if "runner.py" in args and info.session_id in args:
            return {"state": "running", "reason": None, "node": self.source.host}
        return {"state": "gone", "reason": None, "node": None}

    def cancel(self, info: SessionInfo) -> None:
        # TERM the runner (its cleanup stops the container and writes the final status); force-remove the container as a fallback for the runner-died-but-container-survived case. Both idempotent.
        parts = []
        if info.job_id:
            parts.append(f"kill -TERM {shlex.quote(info.job_id)} 2>/dev/null")
        parts.append(f"docker rm -f cs-{shlex.quote(info.session_id)} >/dev/null 2>&1")
        self.access.run("; ".join(parts) + "; true", check=False)

    def tunnel_target(self, info: SessionInfo) -> str | None:
        return f"127.0.0.1:{info.sshd_port}" if info.sshd_port else None

    def probe(self, info: SessionInfo) -> dict:
        """Read the sshd banner from the published port on the host. Uses bash /dev/tcp so the host needs no netcat."""
        if not info.sshd_port:
            return {"reachable": False, "detail": "no sshd port recorded (session may still be starting)"}
        port = int(info.sshd_port)
        cmd = f"timeout 3 bash -c 'exec 3<>/dev/tcp/127.0.0.1/{port} && head -c 64 <&3' 2>&1 || true"
        res = self.access.run(cmd, check=False)
        out = res.stdout.strip()
        if out.startswith("SSH-"):
            return {"reachable": True}
        return {"reachable": False, "detail": out or "no response"}

    def describe(self) -> str:
        cfg = self.source
        extra = f" — {cfg.description}" if cfg.description else ""
        return f"always-on docker host ({cfg.host}){extra}; no resource args (sessions get the whole machine)"


# ---------------------------------------------------------------------------
# vast.ai
# ---------------------------------------------------------------------------

_SYNCED_SENTINEL = "/cs-state/.synced"


def _local_pubkeys() -> str:
    """All local ssh public keys, newline-joined — authorized inside the instance so whichever identity the user's ssh offers works."""
    keys = sorted({p.read_text().strip() for p in Path("~/.ssh").expanduser().glob("*.pub") if p.is_file()})
    keys = [k for k in keys if k]
    if not keys:
        raise SessionError(
            "no ssh public keys found in ~/.ssh — generate one (ssh-keygen -t ed25519) "
            "so rented instances can authorize this machine"
        )
    return "\n".join(keys)


def _mapped_ssh_port(inst: dict) -> int | None:
    """External port vast mapped to the instance's internal :22. `ports` is a docker-style dict, only populated once the container runs."""
    try:
        return int(inst["ports"]["22/tcp"][0]["HostPort"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def build_offer_filters(cfg: SourceConfig, resources: dict, max_price: float) -> dict:
    """Filter objects for the vast offer search: structural constraints from the resource request + config, ANDed with the config's offer_filter clauses."""
    filters: dict = {
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "external": {"eq": False},
        "num_gpus": {"eq": resources["gpus"]},
        "dph_total": {"lte": round(max_price, 4)},
        "disk_space": {"gte": cfg.disk},
        # Direct (non-proxied) ssh needs machines with open ports; ask for headroom beyond our one mapped port.
        "direct_port_count": {"gte": 2},
    }
    if resources.get("gpu_type"):
        # Config tokens use underscores (they must survive whitespace-splitting in filter strings); vast gpu_name values use spaces.
        names = [t.replace("_", " ") for t in resources["gpu_type"].split("|")]
        filters["gpu_name"] = {"in": names} if len(names) > 1 else {"eq": names[0]}
    if resources.get("mem"):
        filters["cpu_ram"] = {"gte": resources["mem"] * 1024}  # vast reports MB
    for field, ops in parse_filter_clauses(cfg.offer_filter).items():
        filters.setdefault(field, {}).update(ops)
    return filters


class VastBackend(Backend):
    """vast.ai: every activation rents a fresh on-demand instance that runs the session image directly (args mode — remote/runner_vast.py is the entrypoint: sshd on :22, idle monitor, and self-destruct via the vast-injected per-instance API key).

    Split state by design: session records live on THIS machine (LocalAccess) because instances are destroyed on deactivate — destroy is the only vast billing cutoff, so nothing durable can live on one. Workdir and command logs live on the instance and die with it; logs are salvaged into the local session dir on deactivate and on observed failures. read_info() reconciles the local record against reality: pending→active once the on-instance runner reports ready, heartbeat refresh while active, active→inactive / pending→failed when the instance is gone.
    """

    runner_finalizes = False
    syncs_at_create = False

    _PENDING_REFRESH_SECONDS = 2.0
    _ACTIVE_REFRESH_SECONDS = 10.0

    def __init__(self, source: SourceConfig, access: LocalAccess, control_path: str):
        super().__init__(source, access)
        self._control_path = control_path
        self.client = VastClient(source)
        self.ledger = Ledger(Path(access.paths.base) / f"{source.name}-ledger.jsonl")
        self._lock = threading.Lock()
        self._instances: dict[str, InstanceAccess] = {}
        self._refreshed: dict[str, float] = {}
        self._ready: set[tuple[str, str]] = set()

    # ---------- access routing ----------

    def _instance_access(self, info: SessionInfo) -> InstanceAccess | None:
        if not (info.job_id and info.node and info.sshd_port):
            return None
        with self._lock:
            acc = self._instances.get(info.session_id)
            if acc is not None and (acc.host, acc.port) == (info.node, int(info.sshd_port)):
                return acc
            acc = self._make_instance_access(info.node, int(info.sshd_port))
            self._instances[info.session_id] = acc
            return acc

    def _make_instance_access(self, ip: str, port: int) -> InstanceAccess:
        return InstanceAccess(self.source, self._control_path, ip, port, self.source.container_user or "root")

    def work_access(self, info: SessionInfo):
        if info.status in (SessionStatus.ACTIVE, SessionStatus.PENDING):
            acc = self._instance_access(info)
            if acc is not None:
                return acc
        raise SessionError(
            f"session {info.session_id} has no reachable instance (status={info.status.value}) — vast workdirs "
            f"are ephemeral and die with the instance; activate the session, then sync/upload again"
        )

    def logs_access(self, info: SessionInfo):
        if info.status == SessionStatus.ACTIVE:
            acc = self._instance_access(info)
            if acc is not None:
                return acc
        # Not live: serve the copies salvaged into the local session dir at deactivate/failure time.
        return self.state

    def init_session_dirs(self, session_id: str) -> None:
        # Local record dir + logs dir (the salvage target). The workdir only exists on an instance.
        paths = self.state.paths
        self.state.run(
            f"mkdir -p {shlex.quote(paths.session_dir(session_id))} "
            f"{shlex.quote(paths.logs_dir(session_id))}"
        )

    def copy_workdir(self, src_session_id: str, dst_session_id: str) -> None:
        raise SessionError(
            "clone is not supported on vast sources — workdirs are instance-bound and die with "
            "the instance; download the needed files and upload them into a new session instead"
        )

    def reset_transport(self, info: SessionInfo) -> None:
        with self._lock:
            acc = self._instances.pop(info.session_id, None)
        if acc is not None:
            acc.close()
        self._ready.discard((info.session_id, str(info.job_id)))

    # ---------- record reconciliation ----------

    def read_info(self, session_id: str) -> SessionInfo:
        info = super().read_info(session_id)
        if not info.job_id:
            return info
        interval = {
            SessionStatus.PENDING: self._PENDING_REFRESH_SECONDS,
            SessionStatus.ACTIVE: self._ACTIVE_REFRESH_SECONDS,
        }.get(info.status)
        now = time.monotonic()
        if interval is None or now - self._refreshed.get(session_id, 0.0) < interval:
            return info
        self._refreshed[session_id] = now
        try:
            if info.status == SessionStatus.PENDING:
                self._reconcile_pending(info)
            else:
                self._reconcile_active(info)
        except (RemoteError, SessionError) as exc:
            log.debug("vast reconcile for %s failed (ignored): %s", session_id, exc)
        return super().read_info(session_id)

    def _read_instance_record(self, acc: InstanceAccess) -> dict | None:
        """The on-instance runner's own config.json — the authority for runner readiness and the idle heartbeat."""
        try:
            res = acc.run("cat /cs-state/config.json 2>/dev/null", check=False)
        except RemoteError:
            return None
        if res.returncode != 0 or not res.stdout.strip():
            return None
        try:
            return json.loads(res.stdout)
        except ValueError:
            return None

    def _reconcile_pending(self, info: SessionInfo) -> None:
        inst = self.client.show_instance(info.job_id)
        if inst is None:
            # Never came up and already gone (cancel_unavail rejection, console destroy, ...).
            self.ledger.close(info.job_id, self.source.max_session_hours)
            info.status = SessionStatus.FAILED
            self.write_info(info)
            return
        if str(inst.get("actual_status") or "") != "running":
            return
        ip, port = inst.get("public_ipaddr"), _mapped_ssh_port(inst)
        if not (ip and port):
            return
        acc = self._make_instance_access(str(ip).strip(), port)
        remote = self._read_instance_record(acc)
        if not remote or remote.get("status") != "active":
            return  # container is up but the runner hasn't finished booting sshd yet
        with self._lock:
            self._instances[info.session_id] = acc
        info.status = SessionStatus.ACTIVE
        info.node = str(ip).strip()
        info.sshd_port = port
        info.last_activated_at = remote.get("last_activated_at")
        info.last_activity_at = remote.get("last_activity_at")
        self.write_info(info)

    def _reconcile_active(self, info: SessionInfo) -> None:
        acc = self._instance_access(info)
        remote = self._read_instance_record(acc) if acc is not None else None
        if remote:
            hb = remote.get("last_activity_at")
            if hb and hb != info.last_activity_at:
                info.last_activity_at = hb
                self.write_info(info)
            return
        # Unreachable — find out whether the instance still exists at all (idle self-destruct is the normal cause).
        if self.job_state(info)["state"] == "gone":
            info.status = SessionStatus.INACTIVE
            self.write_info(info)
            with self._lock:
                self._instances.pop(info.session_id, None)

    def list_infos(self, project_id_filter: str | None = None) -> list[SessionInfo]:
        rows = super().list_infos(project_id_filter)
        stale = [i for i in rows if i.job_id and i.status in (SessionStatus.PENDING, SessionStatus.ACTIVE)]
        if stale:
            # One API call reconciles every listed session (and the ledger) — cheaper than per-session show_instance, and it keeps `list` honest about instances that self-destructed while nobody was looking.
            try:
                live_ids = {str(i.get("id")) for i in self.client.list_instances()}
            except RemoteError as exc:
                log.debug("vast list reconcile failed (ignored): %s", exc)
                return rows
            self.ledger.reconcile(live_ids, self.source.max_session_hours)
            for info in stale:
                if str(info.job_id) not in live_ids:
                    info.status = SessionStatus.FAILED if info.status == SessionStatus.PENDING else SessionStatus.INACTIVE
                    self.write_info(info)
        return rows

    # ---------- lifecycle ----------

    def resolve_resources(self, *, partition: str | None, gpus: int | None, gpu_type: str | None, mem: int | None) -> dict:
        cfg = self.source
        if partition is not None:
            raise SessionError(
                f"source {cfg.name!r} rents vast.ai instances — there are no partitions; "
                f"it takes gpus, gpu_type, and mem (minimum system RAM in GB)"
            )
        gpus = 1 if gpus is None else gpus
        if not 1 <= gpus <= 8:
            raise SessionError(f"gpus must be between 1 and 8, got {gpus}")
        if gpu_type is not None:
            gpu_type = gpu_type.strip() or None
        if gpu_type is None:
            gpu_type = cfg.default_gpu_type or None
        if gpu_type is not None:
            if not cfg.gpu_types:
                raise SessionError(
                    f"source {cfg.name!r} has no gpu_types configured — add allowed vast gpu_name "
                    f"values (underscored, e.g. \"RTX_4090\") to the source config to pin GPU types"
                )
            tokens = [t.strip() for t in gpu_type.split("|")]
            bad = [t for t in tokens if t not in cfg.gpu_types]
            if not tokens or "" in tokens or bad:
                raise SessionError(
                    f"invalid gpu_type {gpu_type!r} for source {cfg.name!r}; each '|'-separated "
                    f"value must be one of {sorted(cfg.gpu_types)}"
                )
            gpu_type = "|".join(tokens)
        if mem is not None and not 1 <= mem <= 4000:
            raise SessionError(f"mem must be between 1 and 4000 GB, got {mem}")
        return {"gpus": gpus, "gpu_type": gpu_type, "mem": mem}

    def submit(self, info: SessionInfo) -> str:
        cfg = self.source
        # Spend gates. The live API list is the authority for the concurrent-rate cap (it sees instances from every machine/session, including leaked ones); the local ledger estimates the calendar month.
        live = self.client.list_instances()
        self.ledger.reconcile({str(i.get("id")) for i in live}, cfg.max_session_hours)
        month = self.ledger.month_spend()
        if month >= cfg.max_total_spend:
            raise SessionError(
                f"monthly spend limit reached on source {cfg.name!r}: ~${month:.2f} of ${cfg.max_total_spend:.2f} "
                f"accrued this calendar month (estimate from {self.ledger.path}) — raise max_total_spend in the config or wait"
            )
        mine = [i for i in live if str(i.get("label") or "").startswith("cs-")]
        running_rate = sum(float(i.get("dph_total") or 0) for i in mine)
        budget_rate = cfg.max_hourly_spend - running_rate
        if budget_rate <= 0:
            raise SessionError(
                f"hourly spend cap reached on source {cfg.name!r}: {len(mine)} running instance(s) already bill "
                f"${running_rate:.2f}/h of the ${cfg.max_hourly_spend:.2f}/h cap — deactivate one first"
            )
        max_price = min(cfg.max_instance_price, budget_rate)
        offers = self.client.search_offers(build_offer_filters(cfg, info.resources, max_price))
        if not offers:
            r = info.resources
            raise SessionError(
                f"no vast offer matches: gpus={r['gpus']}, gpu_type={r.get('gpu_type') or 'any'}, "
                f"price ≤ ${max_price:.2f}/h, disk ≥ {cfg.disk}GB, offer_filter {cfg.offer_filter!r} — "
                f"loosen gpu_type, the price caps, or the offer_filter"
            )
        offer = offers[0]
        env = {
            "-p 22:22": "1",  # opens internal :22; vast maps it to a random external port
            "CS_SESSION_ID": info.session_id,
            "CS_IDLE_TIMEOUT_MINUTES": str(info.idle_timeout_minutes),
            "CS_MAX_HOURS": str(cfg.max_session_hours),
            "CS_AUTHORIZED_KEYS_B64": base64.b64encode(_local_pubkeys().encode()).decode(),
        }
        body = {
            "client_id": "me",
            "image": cfg.image,
            "disk": cfg.disk,
            "runtype": "args",  # no vast injection: the image's entrypoint (runner_vast.py) is in full control
            "env": env,
            "label": f"cs-{info.session_id}",
            "cancel_unavail": True,  # error out instead of leaving a stopped-but-storage-billing instance when scheduling fails
        }
        instance_id = self.client.create_instance(int(offer["id"]), body)
        self.ledger.start(instance_id, info.session_id, float(offer.get("dph_total") or 0))
        return instance_id

    def job_state(self, info: SessionInfo) -> dict:
        if not info.job_id:
            return {"state": "gone", "reason": None, "node": None}
        try:
            inst = self.client.show_instance(info.job_id)
        except RemoteError as exc:
            return {"state": "unknown", "reason": f"vast API query failed: {exc}", "node": None}
        if inst is None:
            self.ledger.close(info.job_id, self.source.max_session_hours)
            return {"state": "gone", "reason": None, "node": None}
        status = str(inst.get("actual_status") or "")
        extra = {"gpu_name": inst.get("gpu_name"), "dph_total": inst.get("dph_total")}
        if status == "running":
            return {"state": "running", "reason": inst.get("status_msg"), "node": inst.get("public_ipaddr"), **extra}
        if status in ("", "loading"):
            reason = inst.get("status_msg") or "provisioning (image pull can take minutes on a cold machine)"
            return {"state": "pending", "reason": reason, "node": inst.get("public_ipaddr"), **extra}
        # exited/stopped/offline/unknown are dead ends that keep billing storage — salvage the container log, then destroy the corpse.
        self._salvage_api_logs(info)
        try:
            self.client.destroy_instance(info.job_id)
        except RemoteError as exc:
            return {"state": "unknown", "reason": f"instance is {status} and could not be destroyed: {exc}", "node": None, **extra}
        self.ledger.close(info.job_id, self.source.max_session_hours)
        return {"state": "gone", "reason": f"instance was {status} ({inst.get('status_msg') or 'no status message'}); destroyed it", "node": None, **extra}

    def cancel(self, info: SessionInfo) -> None:
        if not info.job_id:
            return
        # Salvage what should outlive the instance — command logs over ssh (for post-mortem `logs`/`list_commands`) and the container log via the API (for runner-level failures) — then destroy. Salvage is best-effort; the destroy is NOT: a live instance the user believes dead is the most expensive bug this backend can have, so failures propagate.
        acc = self._instance_access(info)
        if acc is not None:
            try:
                local = Path(self.state.paths.logs_dir(info.session_id))
                local.mkdir(parents=True, exist_ok=True)
                acc.rsync_from("/cs-logs", local, contents_only=True)
            except Exception as exc:
                log.debug("log salvage for %s failed (ignored): %s", info.session_id, exc)
        self._salvage_api_logs(info)
        self.client.destroy_instance(info.job_id)
        self.ledger.stop(info.job_id, info.session_id)
        with self._lock:
            self._instances.pop(info.session_id, None)

    def _salvage_api_logs(self, info: SessionInfo) -> None:
        """Best-effort copy of the container (docker) log into the local session logs dir — named runner-vast.log so the standard failure-tail glob picks it up."""
        try:
            text = self.client.instance_logs(info.job_id, tail=200)
            if text.strip():
                local = Path(self.state.paths.logs_dir(info.session_id))
                local.mkdir(parents=True, exist_ok=True)
                (local / "runner-vast.log").write_text(text)
        except Exception as exc:
            log.debug("api log salvage for %s failed (ignored): %s", info.session_id, exc)

    def tunnel_target(self, info: SessionInfo) -> str | None:
        return None  # direct connection; no tunnel leg exists

    def probe(self, info: SessionInfo) -> dict:
        if not (info.node and info.sshd_port):
            return {"reachable": False, "detail": "no instance endpoint recorded yet"}
        try:
            with socket.create_connection((info.node, int(info.sshd_port)), timeout=5) as s:
                s.settimeout(3)
                if s.recv(64).startswith(b"SSH-"):
                    return {"reachable": True}
                return {"reachable": False, "detail": "connected but no SSH banner"}
        except OSError as exc:
            return {"reachable": False, "detail": str(exc)}

    # ---------- workdir bootstrap ----------

    def ensure_ready(self, info: SessionInfo) -> None:
        """First contact after an activation: populate the fresh instance's empty /workdir. An instance-side sentinel makes this once-per-activation (it dies with the instance, exactly matching the workdir it describes); the in-memory set skips the ssh probe on later runs from this process."""
        key = (info.session_id, str(info.job_id))
        if key in self._ready:
            return
        acc = self.work_access(info)
        probe = acc.run(f"test -e {_SYNCED_SENTINEL} && echo yes || echo no", check=False).stdout.strip()
        if probe != "yes":
            project = Path(info.project_path).expanduser()
            if not project.is_dir():
                raise SessionError(
                    f"session {info.session_id}'s project_path {info.project_path!r} no longer exists on this "
                    f"machine — cannot populate the fresh instance workdir; sync/upload from a valid project"
                )
            sync_session(acc, acc.paths, info.session_id, project)
            acc.run(f"touch {_SYNCED_SENTINEL}", check=False)
        self._ready.add(key)

    def after_sync(self, info: SessionInfo) -> None:
        try:
            self.work_access(info).run(f"touch {_SYNCED_SENTINEL}", check=False)
        except SessionError:
            return
        self._ready.add((info.session_id, str(info.job_id)))

    def describe(self) -> str:
        cfg = self.source
        parts = ["vast.ai rentals (fresh instance per activation, ephemeral disk)"]
        if cfg.description:
            parts.insert(0, cfg.description)
        if cfg.gpu_types:
            parts.append(f"gpu types: {', '.join(cfg.gpu_types)} (default {cfg.default_gpu_type or 'cheapest'})")
        parts.append(
            f"caps: ${cfg.max_instance_price:g}/h/instance, ${cfg.max_hourly_spend:g}/h total, "
            f"${cfg.max_total_spend:g}/month, self-destruct after {cfg.max_session_hours}h"
        )
        return "; ".join(parts)


def make_backend(source: SourceConfig, control_path: str) -> Backend:
    if source.type == "vast":
        return VastBackend(source, LocalAccess(source), control_path)
    access = HostAccess(source, control_path)
    return {"slurm": SlurmBackend, "docker": DockerBackend}[source.type](source, access)
