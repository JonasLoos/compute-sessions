from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SessionStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"


def _seconds_since(iso_utc: str | None) -> int | None:
    """Parse an ISO-8601 UTC timestamp ('…Z') and return seconds elapsed since it, or None."""
    if not iso_utc:
        return None
    try:
        ts = datetime.strptime(iso_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return max(0, int((datetime.now(timezone.utc) - ts).total_seconds()))


# Fields surfaced in the compact agent view (`SessionInfo.to_agent_json`), in addition to the flattened `resources` dict.
_AGENT_FIELDS = (
    "session_id",
    "status",
    "created_at",
    "idle_timeout_minutes",
    "time_in_status_seconds",
    "seconds_until_idle_deactivate",
    "gpu",
)


@dataclass
class SessionInfo:
    # Identity
    session_id: str    # "<prefix>_<hex>"
    project_id: str
    project_path: str  # absolute path on the local machine, fixed at create time
    created_at: str    # ISO-8601 UTC

    # Resource request for the *most recent* activation (partition/gpus/gpu_type/mem). Resolved and validated at activate() time, not fixed at create() — so a restart can request different resources.
    resources: dict[str, Any] = field(default_factory=dict)
    idle_timeout_minutes: int = 0
    image: str = ""

    # Runtime state — populated by runner.py on the cluster.
    status: SessionStatus = SessionStatus.PENDING
    job_id: str | None = None    # slurm job id
    node: str | None = None
    sshd_port: int | None = None
    # Human-readable model of the actually-allocated GPU(s), recorded by the runner at activation from `nvidia-smi -L` (e.g. "NVIDIA A100 80GB PCIe", "2x NVIDIA H100"). Distinct from resources["gpu_type"], which is the *requested* constraint filter — an unconstrained request can land on anything.
    gpu: str | None = None
    last_activated_at: str | None = None
    last_deactivated_at: str | None = None
    # Wall-clock time the idle monitor last observed activity (ssh connection, live tracked `run`, or a file-tool op). Updated ~once per minute by the runner's idle monitor while the session is ACTIVE. Drives `seconds_until_idle_deactivate` in the enriched view.
    last_activity_at: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Plain serialization — used both to persist to disk and as the base for the enriched form. Must not include time-sensitive derived fields, or they'd get baked into the on-disk config.json."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def to_enriched_json(self) -> dict[str, Any]:
        """Serialization with derived, time-sensitive fields added on top of `to_json`:
          - `time_in_status_seconds` — seconds since the session entered its current `status` (anchored at `last_activated_at` for active, `last_deactivated_at` for inactive, else null).
          - `seconds_until_idle_deactivate` — for ACTIVE sessions, an estimate of how long until the idle monitor reaps the session if no further activity occurs. Computed as `idle_timeout_minutes*60 - (now - last_activity_at)`, clamped to [0, idle_timeout_minutes*60]. Null when not active or when the runner hasn't written a heartbeat yet.
        """
        d = self.to_json()
        anchor: str | None = None
        if self.status == SessionStatus.ACTIVE:
            anchor = self.last_activated_at
        elif self.status == SessionStatus.INACTIVE:
            anchor = self.last_deactivated_at
        d["time_in_status_seconds"] = _seconds_since(anchor)

        idle_remaining: int | None = None
        if self.status == SessionStatus.ACTIVE and self.last_activity_at:
            since = _seconds_since(self.last_activity_at)
            if since is not None:
                cap = self.idle_timeout_minutes * 60
                idle_remaining = max(0, cap - since)
        d["seconds_until_idle_deactivate"] = idle_remaining
        return d

    def to_agent_json(self) -> dict[str, Any]:
        """Compact per-session view (`cs list` rows and --json output).

        Keeps the fields an agent uses to pick and monitor a session, flattens `resources` into the row, and drops internal plumbing (project_id, job_id, node, sshd_port, image) and the raw activation timestamps — `time_in_status_seconds` and `seconds_until_idle_deactivate` already summarize those. Null fields are omitted (e.g. `gpu_type` when unconstrained, the idle countdown outside ACTIVE). `cs show` returns the full enriched view for debugging.
        """
        enriched = self.to_enriched_json()
        out = {k: enriched[k] for k in _AGENT_FIELDS if enriched[k] is not None}
        out.update({k: v for k, v in self.resources.items() if v is not None})
        return out

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "SessionInfo":
        status = SessionStatus(d.get("status", "pending"))
        return cls(
            session_id=d["session_id"],
            project_id=d["project_id"],
            project_path=d.get("project_path", ""),
            created_at=d["created_at"],
            resources=dict(d.get("resources", {})),
            idle_timeout_minutes=int(d.get("idle_timeout_minutes", 0)),
            image=d.get("image", ""),
            status=status,
            job_id=str(d["job_id"]) if d.get("job_id") not in (None, "") else None,
            node=d.get("node"),
            sshd_port=d.get("sshd_port"),
            gpu=d.get("gpu"),
            last_activated_at=d.get("last_activated_at"),
            last_deactivated_at=d.get("last_deactivated_at"),
            last_activity_at=d.get("last_activity_at"),
        )


@dataclass
class SyncResult:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    # Files the session previously placed via sync that are no longer present in the local project (so they've been removed from workdir). Never includes files the session itself created — only files tracked by the per-session .sync-manifest.
    deleted_outdated: list[str] = field(default_factory=list)
    bytes_transferred: int = 0
    # Per-path messages from the cleanup phase (e.g. permission errors). The rsync and manifest-write phases still complete when this is non-empty, so partial progress is visible to the caller instead of being swallowed by an exception.
    cleanup_errors: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    command_id: str
    status: str  # "running" | "exited" | "killed" (settled between the launch probe and the log read)
    exit_code: int | None  # None while status == "running"
    stdout: str
    stderr: str
    # Byte-offset cursor ("<out>:<err>") for incremental follow-up reads via wait_for_command(since=…). Only set while the command is still running.
    cursor: str | None = None
