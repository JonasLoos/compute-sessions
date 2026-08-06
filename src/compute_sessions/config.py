from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from compute_sessions.errors import ConfigError


DEFAULT_CONFIG_PATH = "~/.config/compute-sessions/config.toml"

# Source names prefix session ids ("<source>_<hex>", parsed by splitting on the first underscore), so they must be plain lowercase alphanumerics with no separators.
_SOURCE_NAME_RE = re.compile(r"^[a-z][a-z0-9]*$")
# Partition / gpu-type tokens flow into scheduler shell commands. They come from the trusted config file, but constrain the charset anyway (defense-in-depth alongside the shlex.quote at use sites).
_TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]+$")

SOURCE_TYPES = ("slurm", "docker", "vast")


@dataclass(frozen=True)
class SourceConfig:
    """One compute source (a `[sources.<name>]` table in config.toml)."""
    name: str
    type: str          # "slurm" (scheduler + Apptainer) | "docker" (always-on ssh host + Docker) | "vast" (GPU instance rented on vast.ai per activation)
    host: str = ""     # ssh alias of the source's entry host (connection details live in ~/.ssh/config); unused for vast — each activation rents a fresh instance with its own address
    remote_base: str = "~/.compute-sessions"   # session-state base dir: on the source host for slurm/docker; on THIS machine for vast (instances are ephemeral, so state must outlive them)
    image: str = ""    # slurm: .sif path (relative to remote_base if not absolute); docker: local image tag; vast: public registry ref with the runner baked in (remote/Dockerfile.vast)
    container_user: str = ""   # in-container sshd login; empty resolves to the remote `whoami` (slurm/docker) / "root" (vast)
    description: str = ""      # free-text note rendered into cs create/activate --help — cost, hardware, quirks (e.g. "gaming PC: free, 1x RTX 4090, one session at a time")
    # slurm only
    partitions: tuple[str, ...] = ()
    default_partition: str = ""
    default_mem: int = 42             # GB
    login_host: str = ""              # login-node alias as resolvable FROM compute nodes; empty = same as `host`
    exclude_nodes: tuple[str, ...] = ()   # nodes to never schedule on (sbatch --exclude), e.g. temporarily broken ones
    # slurm + vast
    gpu_types: tuple[str, ...] = ()   # slurm: sbatch --constraint values; vast: allowed gpu_name filters (underscored, e.g. "RTX_4090"); empty = gpu_type arg unsupported (vast: any GPU)
    # vast only
    api_key_file: str = "~/.config/vastai/vast_api_key"   # file holding the vast.ai API key (one line; default = where the vastai CLI keeps it)
    disk: int = 32                             # GB of instance disk to rent
    default_gpu_type: str = ""                 # gpu_name filter when the caller doesn't pin gpu_type; empty = any GPU the other filters allow
    offer_filter: str = "verified=true reliability>0.98"   # vast offer-search clauses ANDed with the structural filters (gpu/price/disk/ports); setting it replaces this quality baseline
    max_instance_price: float = 0.0            # $/hr ceiling for a single instance (offer-search filter); required
    max_hourly_spend: float = 0.0              # $/hr ceiling summed over all running instances of this source; required
    max_total_spend: float = 0.0               # $ ceiling per calendar month (local ledger estimate); required
    max_session_hours: int = 12                # hard self-destruct age for an instance, whatever its activity


@dataclass(frozen=True)
class Config:
    sources: dict[str, SourceConfig]
    default_source: str
    max_sessions_per_repo: int = 3    # per source; 0 = unlimited
    control_path: str = "~/.ssh/cm-%r@%h:%p"   # OpenSSH ControlMaster socket path template


def _source_from_table(name: str, t: dict) -> SourceConfig:
    if not _SOURCE_NAME_RE.fullmatch(name):
        raise ConfigError(f"invalid source name {name!r}: must match {_SOURCE_NAME_RE.pattern} (it prefixes session ids)")
    stype = t.get("type", "")
    if stype not in SOURCE_TYPES:
        raise ConfigError(f"source {name!r}: type must be one of {SOURCE_TYPES}, got {stype!r}")
    if stype != "vast" and not t.get("host"):
        raise ConfigError(f"source {name!r}: `host` (ssh alias) is required")
    if not t.get("image"):
        raise ConfigError(f"source {name!r}: `image` is required")
    partitions = tuple(t.get("partitions", []))
    gpu_types = tuple(t.get("gpu_types", []))
    exclude_nodes = tuple(t.get("exclude_nodes", []))
    for tok in (*partitions, *gpu_types, *exclude_nodes):
        if not _TOKEN_RE.fullmatch(str(tok)):
            raise ConfigError(f"source {name!r}: invalid partition/gpu_type/exclude_nodes token {tok!r}")
    if stype == "slurm":
        if not partitions:
            raise ConfigError(f"source {name!r}: slurm sources need a non-empty `partitions` list")
        default_partition = t.get("default_partition", partitions[0])
        if default_partition not in partitions:
            raise ConfigError(f"source {name!r}: default_partition {default_partition!r} not in partitions")
    else:
        default_partition = ""

    try:
        max_instance_price = float(t.get("max_instance_price", 0.0))
        max_hourly_spend = float(t.get("max_hourly_spend", 0.0))
        max_total_spend = float(t.get("max_total_spend", 0.0))
        disk = int(t.get("disk", 32))
        max_session_hours = int(t.get("max_session_hours", 12))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"source {name!r}: numeric field has a non-numeric value: {exc}") from exc
    default_gpu_type = str(t.get("default_gpu_type", ""))
    if stype == "vast":
        # Spend limits are mandatory for rented GPUs — a missing cap must fail loudly at config time, not bill silently at run time.
        if not (max_instance_price > 0 and max_hourly_spend > 0 and max_total_spend > 0):
            raise ConfigError(
                f"source {name!r}: vast sources require positive spend limits: max_instance_price "
                f"($/hr per instance), max_hourly_spend ($/hr across running instances), "
                f"max_total_spend ($ per calendar month)"
            )
        if default_gpu_type and default_gpu_type not in gpu_types:
            raise ConfigError(f"source {name!r}: default_gpu_type {default_gpu_type!r} not in gpu_types")
        if not 8 <= disk <= 4000:
            raise ConfigError(f"source {name!r}: disk must be 8-4000 GB, got {disk}")
        if not 1 <= max_session_hours <= 336:
            raise ConfigError(f"source {name!r}: max_session_hours must be 1-336, got {max_session_hours}")

    return SourceConfig(
        name=name,
        type=stype,
        host=t.get("host", ""),
        remote_base=t.get("remote_base", "~/.compute-sessions"),
        image=t["image"],
        container_user=t.get("container_user", ""),
        description=t.get("description", ""),
        partitions=partitions,
        default_partition=default_partition,
        gpu_types=gpu_types,
        default_mem=int(t.get("default_mem", 42)),
        login_host=t.get("login_host", ""),
        exclude_nodes=exclude_nodes,
        api_key_file=t.get("api_key_file", "~/.config/vastai/vast_api_key"),
        disk=disk,
        default_gpu_type=default_gpu_type,
        offer_filter=t.get("offer_filter", "verified=true reliability>0.98"),
        max_instance_price=max_instance_price,
        max_hourly_spend=max_hourly_spend,
        max_total_spend=max_total_spend,
        max_session_hours=max_session_hours,
    )


def config_path() -> Path:
    return Path(os.environ.get("COMPUTE_SESSIONS_CONFIG", DEFAULT_CONFIG_PATH)).expanduser()


def load_config(path: Path | None = None) -> Config:
    """Load and validate the config file. Sources live in `[sources.<name>]` tables; global settings at top level. No env-var fallbacks — the file is the single source of truth (the `setup-compute-source` skill writes it)."""
    path = path or config_path()
    if not path.is_file():
        raise ConfigError(
            f"config file not found at {path}. Create it (see README.md) or run the "
            f"setup-compute-source skill to configure your first compute source."
        )
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc

    tables = data.get("sources", {})
    if not isinstance(tables, dict) or not tables:
        raise ConfigError(f"{path} defines no [sources.<name>] tables")
    sources = {name: _source_from_table(name, t) for name, t in tables.items()}

    default_source = data.get("default_source", next(iter(sources)))
    if default_source not in sources:
        raise ConfigError(f"default_source {default_source!r} is not a configured source ({sorted(sources)})")
    try:
        max_sessions = int(data.get("max_sessions_per_repo", 3))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"max_sessions_per_repo must be an integer: {exc}") from exc
    return Config(
        sources=sources,
        default_source=default_source,
        max_sessions_per_repo=max_sessions,
        control_path=data.get("control_path", "~/.ssh/cm-%r@%h:%p"),
    )
