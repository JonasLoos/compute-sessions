from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from compute_sessions.errors import ConfigError


DEFAULT_CONFIG_PATH = "~/.config/compute-sessions/config.toml"

# Partition / gpu-type / node tokens flow into sbatch shell commands. They come from the trusted config file, but constrain the charset anyway (defense-in-depth alongside the shlex.quote at use sites).
_TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class Config:
    """The SLURM cluster (`~/.config/compute-sessions/config.toml`, flat keys)."""
    host: str                          # ssh alias of the login node (connection details live in ~/.ssh/config)
    image: str                         # Apptainer .sif path, relative to remote_base unless absolute
    partitions: tuple[str, ...]
    default_partition: str
    gpu_types: tuple[str, ...] = ()    # sbatch --constraint values; empty = --gpu-type unsupported
    default_mem: int = 42              # GB
    exclude_nodes: tuple[str, ...] = ()   # nodes to never schedule on (sbatch --exclude), e.g. temporarily broken ones
    login_host: str = ""               # login-node alias as resolvable FROM compute nodes; empty = same as `host`
    remote_base: str = "~/.compute-sessions"   # session-state base dir on the cluster
    container_user: str = ""           # in-container sshd login; empty resolves to the remote `whoami`
    max_sessions_per_repo: int = 3     # active+pending cap per repo; 0 = unlimited
    control_path: str = "~/.ssh/cm-%r@%h:%p"   # OpenSSH ControlMaster socket path template

    @property
    def session_prefix(self) -> str:
        """Session ids are `<prefix>_<hex>`; the prefix is the host alias reduced to lowercase alphanumerics, so ids read as "<host>_8ec5…" and the CLI can tell a session token from a command word."""
        return re.sub(r"[^a-z0-9]", "", self.host.lower()) or "cs"


# Every key load_config reads. A key outside this set is rejected rather than ignored, so a half-migrated file (leftover `default_source`, `type`, `description`) or a typo can't silently do nothing.
_KEYS = frozenset(Config.__dataclass_fields__)


def config_path() -> Path:
    return Path(os.environ.get("COMPUTE_SESSIONS_CONFIG", DEFAULT_CONFIG_PATH)).expanduser()


def load_config(path: Path | None = None) -> Config:
    """Load and validate the config file. No env-var fallbacks — the file is the single source of truth (the `setup-cluster` skill writes it)."""
    path = path or config_path()
    if not path.is_file():
        raise ConfigError(
            f"config file not found at {path}. Create it (see examples/config.toml) or run the "
            f"setup-cluster skill to configure your cluster."
        )
    try:
        t = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc
    if "sources" in t:
        raise ConfigError(f"{path} uses the pre-0.5 multi-source layout ([sources.<name>] tables): move the slurm cluster's keys to the top level and drop default_source/type/description (see examples/config.toml)")
    unknown = sorted(set(t) - _KEYS)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {unknown}; valid keys: {sorted(_KEYS)}")
    for key in ("host", "image"):
        if not t.get(key):
            raise ConfigError(f"{path}: `{key}` is required")
    partitions = tuple(t.get("partitions", []))
    if not partitions:
        raise ConfigError(f"{path}: `partitions` must be a non-empty list")
    gpu_types = tuple(t.get("gpu_types", []))
    exclude_nodes = tuple(t.get("exclude_nodes", []))
    for tok in (*partitions, *gpu_types, *exclude_nodes):
        if not _TOKEN_RE.fullmatch(str(tok)):
            raise ConfigError(f"{path}: invalid partition/gpu_type/exclude_nodes token {tok!r}")
    default_partition = t.get("default_partition", partitions[0])
    if default_partition not in partitions:
        raise ConfigError(f"{path}: default_partition {default_partition!r} not in partitions")
    try:
        default_mem = int(t.get("default_mem", 42))
        max_sessions = int(t.get("max_sessions_per_repo", 3))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: default_mem and max_sessions_per_repo must be integers: {exc}") from exc
    return Config(
        host=t["host"],
        image=t["image"],
        partitions=partitions,
        default_partition=default_partition,
        gpu_types=gpu_types,
        default_mem=default_mem,
        exclude_nodes=exclude_nodes,
        login_host=t.get("login_host", ""),
        remote_base=t.get("remote_base", "~/.compute-sessions"),
        container_user=t.get("container_user", ""),
        max_sessions_per_repo=max_sessions,
        control_path=t.get("control_path", "~/.ssh/cm-%r@%h:%p"),
    )
