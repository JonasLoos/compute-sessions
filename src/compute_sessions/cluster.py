"""SLURM-side operations: session-record I/O under `remote_base/sessions/`, resource validation, and submit/poll/cancel/probe of a session's sbatch job. Sessions are sbatch jobs running an Apptainer container, reached via a reverse-SSH tunnel socket on the login node's shared FS."""
from __future__ import annotations

import json
import logging
import secrets
import shlex

from compute_sessions.config import Config
from compute_sessions.errors import RemoteError, SessionError
from compute_sessions.models import SessionInfo
from compute_sessions.ssh import HostAccess


log = logging.getLogger(__name__)

# Login-node helper for Cluster.patch_info: {"fields": {...}, "expect": {...}} on stdin, the config.json path as argv[1]; prints `ok` or `stale`.
_PATCH_INFO_PY = """\
import json, os, sys
path = sys.argv[1]
req = json.load(sys.stdin)
with open(path) as fh:
    d = json.load(fh)
if any(d.get(k) != v for k, v in req["expect"].items()):
    print("stale")
    sys.exit(0)
d.update(req["fields"])
tmp = path + ".tmp." + str(os.getpid())
with open(tmp, "w") as fh:
    json.dump(d, fh, indent=2)
os.replace(tmp, path)
print("ok")
"""


# `squeue %T` states after which the job holds no allocation and its runner (if it ever ran) is gone or in its final teardown. COMPLETING is included: the node is being released and a new activation may be submitted; the runner's final status write is guarded against clobbering a re-activated record (runner.py).
_TERMINAL_JOB_STATES = frozenset({
    "COMPLETED", "COMPLETING", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED",
    "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "REVOKED", "SPECIAL_EXIT",
})


class Cluster:
    def __init__(self, cfg: Config, access: HostAccess | None = None):
        self.cfg = cfg
        self.access = access if access is not None else HostAccess(cfg)

    # ---------- session-record I/O ----------

    def read_info(self, session_id: str) -> SessionInfo:
        config_path = self.access.paths.config_file(session_id)
        # Distinguish "session not found" (typed error the caller can act on) from any other cat failure. Otherwise the raw RemoteError bubbles up with "cat: ... No such file or directory" and callers can't tell whether the session was GC'd, never created, or whether something else went wrong.
        try:
            res = self.access.run(f"cat {shlex.quote(config_path)}", check=True)
        except RemoteError as exc:
            if exc.exit_code == 1 and exc.stderr and "No such file or directory" in exc.stderr:
                raise SessionError(
                    f"session {session_id!r} not found — config.json missing at {config_path}. "
                    "It may have been deleted or never created. `cs list` shows known sessions."
                ) from exc
            raise
        return SessionInfo.from_json(json.loads(res.stdout))

    def write_info(self, info: SessionInfo) -> None:
        body = json.dumps(info.to_json(), indent=2)
        final = self.access.paths.config_file(info.session_id)
        # Atomic write: stream into a unique temp file in the same dir, then `mv` it over the target. config.json has several concurrent writers (this process plus the runner), and a plain `cat >` truncates-then-writes — a concurrent reader could observe a half-written file. `mv` within one dir is atomic; a per-write nonce keeps concurrent writers from colliding on the temp name.
        tmp = f"{final}.tmp.{secrets.token_hex(6)}"
        q_tmp = shlex.quote(tmp)
        q_final = shlex.quote(final)
        self.access.run(
            f"cat > {q_tmp} && mv -f {q_tmp} {q_final} || {{ rm -f {q_tmp}; exit 1; }}",
            input_text=body,
        )

    def patch_info(self, session_id: str, fields: dict, *, expect: dict | None = None) -> bool:
        """Compare-and-set update of a few fields of config.json, done in one round-trip on the login node (read, check, update, atomic replace — the same temp+rename the runner uses).

        `write_info` overwrites the whole record from an in-memory object that may be stale by the time it lands; the runner writes to the same file (status/node/sshd_port on activation, heartbeats, the final status). Every client write that follows a slow step — sbatch, a squeue query, the deactivate poll — goes through here instead, touching only the fields it owns. `expect` maps fields to the values they must still hold, else nothing is written and False is returned: a `show` that decided a job was gone must not mark a session that was re-activated meanwhile.
        """
        config_path = self.access.paths.config_file(session_id)
        payload = json.dumps({"fields": fields, "expect": expect or {}})
        res = self.access.run(f"python3 -c {shlex.quote(_PATCH_INFO_PY)} {shlex.quote(config_path)}", input_text=payload)
        return res.stdout.strip() == "ok"

    def list_infos(self, project_id_filter: str | None = None) -> list[SessionInfo]:
        # Concatenate each config.json with a NUL separator. JSON text can't contain raw NULs, so this round-trips cleanly regardless of content.
        # The trailing `true` matters: a `for` loop exits with its last iteration's status, so a session dir without a config.json (a create that died between mkdir and the first record write) sorting last would fail the whole command and hide every session from `cs list` and session inference.
        cmd = (
            f"for d in {shlex.quote(self.access.paths.sessions_root())}/*/; do "
            f"  [ -f \"$d/config.json\" ] && cat \"$d/config.json\" && printf '\\0'; "
            f"done; true"
        )
        res = self.access.run(cmd, check=False)
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

    def init_session_dirs(self, session_id: str) -> None:
        p = self.access.paths
        self.access.run(f"mkdir -p {shlex.quote(p.workdir(session_id))} {shlex.quote(p.logs_dir(session_id))}")

    def copy_workdir(self, src_session_id: str, dst_session_id: str) -> None:
        """Server-side copy of one session's workdir contents (plus its sync manifest, so later `sync` calls keep tracking the mirrored files) into another's — the data never leaves the cluster."""
        p = self.access.paths
        self.access.run(
            f"cp -a {shlex.quote(p.workdir(src_session_id))}/. {shlex.quote(p.workdir(dst_session_id))}/ && "
            f"{{ cp {shlex.quote(p.manifest(src_session_id))} {shlex.quote(p.manifest(dst_session_id))} 2>/dev/null || true; }}"
        )

    def failure_log_tail(self, info: SessionInfo) -> str:
        """Tail of the most recent runner log, for FAILED sessions."""
        logs_q = shlex.quote(self.access.paths.logs_dir(info.session_id))
        res = self.access.run(
            f"f=$(ls -t {logs_q}/runner* 2>/dev/null | head -1); "
            f"[ -n \"$f\" ] && tail -n 30 \"$f\"; true",
            check=False,
        )
        return res.stdout

    # ---------- lifecycle ----------

    def resolve_resources(self, *, partition: str | None, gpus: int | None, gpu_type: str | None, mem: int | None) -> dict:
        """Validate a per-activation resource request; return the dict stored on SessionInfo.resources. All args arrive as None when the caller didn't specify them."""
        cfg = self.cfg
        partition = partition or cfg.default_partition
        if partition not in cfg.partitions:
            raise SessionError(f"invalid partition {partition!r}; must be one of {sorted(cfg.partitions)}")

        # cpu-* partitions have no GPUs, so gpus is forced to 0 there. The default gpus=1 suits gpu partitions; a cpu session simply drops it (an explicit gpus>0 on a cpu partition is dropped, not rejected — you picked a CPU partition).
        gpus = 1 if gpus is None else gpus
        resolved_gpus = gpus if not partition.startswith("cpu-") else 0
        if not 0 <= resolved_gpus <= 16:
            raise SessionError(f"gpus must be between 0 and 16, got {resolved_gpus}")

        # gpu_type → sbatch --constraint. Only meaningful when actually requesting GPUs. Validate every `|`-separated token against the allowlist — the value flows into the sbatch shell command, so an unvetted string would be a shell-injection vector.
        if gpu_type is not None:
            gpu_type = gpu_type.strip() or None
        if gpu_type is not None:
            if resolved_gpus == 0:
                raise SessionError(f"gpu_type={gpu_type!r} requires a partition with gpus>0 (got partition={partition!r}, gpus={resolved_gpus})")
            tokens = [t.strip() for t in gpu_type.split("|")]
            bad = [t for t in tokens if t not in cfg.gpu_types]
            if not tokens or "" in tokens or bad:
                raise SessionError(f"invalid gpu_type {gpu_type!r}; each '|'-separated value must be one of {sorted(cfg.gpu_types)}")
            gpu_type = "|".join(tokens)

        mem = cfg.default_mem if mem is None else mem
        if not 1 <= mem <= 6000:
            raise SessionError(f"mem must be between 1 and 6000 GB, got {mem}")

        return {"partition": partition, "gpus": resolved_gpus, "gpu_type": gpu_type, "mem": mem}

    def submit(self, info: SessionInfo) -> str:
        """sbatch the runner for this session (config.json is already persisted); returns the slurm job id."""
        paths = self.access.paths
        r = info.resources
        # partition/gpu_type are validated against allowlists in resolve_resources, but we still shlex.quote every flag as defense-in-depth — the whole string is handed to `bash -lc`. Wall-clock limit is implied by the partition, so no --time.
        flags: list[str] = [
            "--parsable",
            shlex.quote(f"--job-name=cs-{info.session_id}"),
            shlex.quote(f"--partition={r['partition']}"),
        ]
        if self.cfg.exclude_nodes:
            flags.append(shlex.quote(f"--exclude={','.join(self.cfg.exclude_nodes)}"))
        if r.get("gpus"):
            flags.append(shlex.quote(f"--gpus-per-node={r['gpus']}"))
        if r.get("gpu_type"):
            flags.append(shlex.quote(f"--constraint={r['gpu_type']}"))
        flags.append(shlex.quote(f"--mem={r['mem']}G"))
        flags.append(shlex.quote(f"--output={paths.logs_dir(info.session_id)}/runner-%j.out"))
        login = self.cfg.login_host or self.cfg.host
        runner_args = f"--base {shlex.quote(paths.base)} --session-id {shlex.quote(info.session_id)} --login {shlex.quote(login)}"
        cmd = "sbatch --export=ALL " + " ".join(flags) + f" {shlex.quote(paths.runner())} {runner_args}"
        res = self.access.run(cmd)
        job_id = res.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise SessionError(f"unexpected scheduler response while submitting: {res.stdout!r}")
        return job_id

    def job_state(self, info: SessionInfo) -> dict:
        """Live state of the current activation's job: {state, reason, node, [time_left]}. `state` ∈ {"pending", "running", "gone", "unknown"} — "gone" means slurm has no trace of the job or reports it in a terminal state (finished, cancelled, never ran), "unknown" means the query itself failed (controller unreachable, timeout) — callers must not treat that as gone."""
        if not info.job_id:
            return {"state": "gone", "reason": None, "node": None}
        # `squeue -o %R` is overloaded: for PENDING jobs it's the pending reason (Priority, Resources, ...); for RUNNING jobs it's the nodelist. Split into separate fields so the caller doesn't have to condition on state to interpret the value. `%L` is the remaining wall clock — surfaced so agents can see a job's 5h partition window closing instead of discovering it as a silent mid-run kill.
        #
        # `timeout 5s` guards against a stuck slurmctld (observed: `show`/`list` calls hanging long enough to blind the agent). stderr is captured too: squeue exits 1 both for a purged job ("Invalid job id specified" — genuinely gone) and for an unreachable controller — the latter used to read as an empty result and mark live sessions failed/inactive, so the exit code and message travel back and only the purged-job case counts as gone.
        sq = self.access.run(
            f"out=$(timeout 5s squeue -h -j {shlex.quote(info.job_id)} -o '%T|%L|%R' 2>&1); rc=$?; "
            f"printf '%s\\n' \"$out\"; printf '__CS_RC__ %s\\n' \"$rc\"; exit 0",
            check=False,
        )
        body, _, trailer = sq.stdout.rstrip("\n").rpartition("\n")
        if not trailer.startswith("__CS_RC__ "):
            return {"state": "unknown", "reason": f"unexpected scheduler response: {sq.stdout.strip()[:200]!r}", "node": None}
        rc = trailer.split()[1]
        text = body.strip()
        if rc in ("124", "137"):
            return {"state": "unknown", "reason": "scheduler query timed out", "node": None}
        if rc != "0":
            if "Invalid job id specified" in text:
                return {"state": "gone", "reason": None, "node": None}
            first = next((l.strip() for l in text.splitlines() if l.strip()), f"squeue exit {rc}")
            return {"state": "unknown", "reason": f"scheduler query failed: {first}", "node": None}
        if not text:
            # sbatch --parsable returns only after SLURM has queued the job, so an empty squeue result means the job has finished — safe to treat as gone without a grace period.
            return {"state": "gone", "reason": None, "node": None}
        state, time_left, col = (text.split("|", 2) + ["", ""])[:3]
        if state == "RUNNING":
            return {"state": "running", "reason": None, "node": col, "time_left": time_left or None}
        if state in ("PENDING", "CONFIGURING"):
            return {"state": "pending", "reason": col, "node": None, "time_left": time_left or None}
        if state in _TERMINAL_JOB_STATES:
            return {"state": "gone", "reason": f"{state}: {col}" if col else state, "node": None}
        # Anything else (SUSPENDED, STOPPED, REQUEUED, RESIZING, …) still holds or will hold an allocation: report it as pending so activate() refuses to submit a duplicate.
        return {"state": "pending", "reason": f"{state}: {col}", "node": None}

    def cancel(self, info: SessionInfo) -> None:
        """scancel the activation (best-effort; the runner's cleanup writes the final status)."""
        if info.job_id:
            self.access.run(f"scancel {shlex.quote(info.job_id)}", check=False)

    def probe(self, info: SessionInfo) -> dict:
        """Health-check the session's sshd: connect with nc to the reverse-tunnel Unix socket on the login node and read the SSH identification banner sshd sends on accept. A valid `SSH-2.0-…` line means both the reverse tunnel and the in-container sshd are alive. Returns {reachable, [detail]}."""
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
