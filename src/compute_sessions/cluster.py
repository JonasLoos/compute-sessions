"""The cluster as the client sees it: the config, the login-node connection, session selection context for helper requests, and validation of resource requests (which needs only the config)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from compute_sessions.config import Config
from compute_sessions.errors import SessionError
from compute_sessions.project import project_id
from compute_sessions.ssh import HostAccess


class Cluster:
    def __init__(self, cfg: Config, access: HostAccess | None = None):
        self.cfg = cfg
        self.access = access if access is not None else HostAccess(cfg)
        self._project: tuple[str, str] | None = None
        # Called once per session with (session id, its project) when an explicitly named session belongs to another project — legal, but often a mistyped prefix.
        self.on_note: Callable[[str, str], None] | None = None
        self._noted: set[str] = set()

    def project(self) -> tuple[str, str]:
        """(project id, absolute path) of the cwd — the context for session inference and the other-project note."""
        if self._project is None:
            cwd = Path.cwd().resolve()
            self._project = project_id(cwd), str(cwd)
        return self._project

    def call(self, op: str, args: list[str] | tuple[str, ...] = (), *, maybe: bool = False, **params) -> dict:
        """A helper request with session selection: `args` are the command's positional args, whose first is the session token (with `maybe`: only if it looks like one)."""
        pid, cwd = self.project()
        reply = self.access.call(op, args=list(args), maybe=maybe, prefix=self.cfg.session_prefix, project_id=pid, cwd=cwd, **params)
        if reply.get("note") and self.on_note and reply["sid"] not in self._noted:
            self._noted.add(reply["sid"])
            self.on_note(reply["sid"], reply["note"])
        return reply

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

        # gpu_type → sbatch --constraint. Only meaningful when actually requesting GPUs. Every `|`-separated token is checked against the allowlist.
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
