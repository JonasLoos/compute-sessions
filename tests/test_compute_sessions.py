"""High-level behavioral tests — no remote host or network required. They pin the contracts the cs CLI relies on (config parsing, validation, status semantics, sync planning), not implementation details. Run with `uv run pytest`."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from compute_sessions import vast_api
from compute_sessions.backends import DockerBackend, SlurmBackend, VastBackend, build_offer_filters
from compute_sessions.config import SourceConfig, load_config
from compute_sessions.errors import ComputeSessionsError
from compute_sessions.models import SessionInfo, SessionStatus
from compute_sessions.paths import InstancePaths, RemotePaths, validate_command_id, validate_session_id
from compute_sessions.project import project_id
from compute_sessions.run import _build_detached_launch, _parse_launch_status, _validate_signal
from compute_sessions.session import (
    _LIST_COMMANDS_PY,
    _PID_STALE_SECONDS,
    _classify_command_status,
    _parse_since,
    _resolve_local_inside_project,
    logs_with_access,
    source_of,
    wait_for_command,
)
from compute_sessions.ssh import LocalAccess
from compute_sessions.sync import plan_sync
from compute_sessions.vast_api import Ledger, parse_filter_clauses


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _slurm_backend() -> SlurmBackend:
    scfg = SourceConfig(
        name="hydra", type="slurm", host="hydra", image="images/default.sif",
        partitions=("cpu-2h", "gpu-2h", "gpu-test"), default_partition="gpu-2h",
        gpu_types=("h100", "80gb", "40gb"), default_mem=42,
    )
    return SlurmBackend(scfg, access=None)  # type: ignore[arg-type] — resolve_resources never touches access


def _docker_backend() -> DockerBackend:
    scfg = SourceConfig(name="gamingpc", type="docker", host="gamingpc", image="compute-sessions:latest")
    return DockerBackend(scfg, access=None)  # type: ignore[arg-type]


def _vast_backend() -> VastBackend:
    scfg = SourceConfig(
        name="vast", type="vast", image="me/compute-sessions-vast:latest",
        gpu_types=("RTX_4090", "RTX_5090", "H100_SXM"), default_gpu_type="RTX_4090",
        max_instance_price=1.5, max_hourly_spend=3.0, max_total_spend=100.0, max_session_hours=12,
    )
    return VastBackend(scfg, LocalAccess(scfg), control_path="~/.ssh/cm-%r@%h:%p")


# ---------- resource requests ----------

def test_valid_gpu_request_passes_through():
    r = _slurm_backend().resolve_resources(partition="gpu-2h", gpus=2, gpu_type="h100|80gb", mem=64)
    assert r == {"partition": "gpu-2h", "gpus": 2, "gpu_type": "h100|80gb", "mem": 64}


def test_slurm_defaults_come_from_source_config():
    r = _slurm_backend().resolve_resources(partition=None, gpus=None, gpu_type=None, mem=None)
    assert r == {"partition": "gpu-2h", "gpus": 1, "gpu_type": None, "mem": 42}


def test_cpu_partition_never_gets_gpus():
    r = _slurm_backend().resolve_resources(partition="cpu-2h", gpus=4, gpu_type=None, mem=8)
    assert r["gpus"] == 0


@pytest.mark.parametrize("override", [
    {"partition": "gpu-99h"},                  # unknown partition
    {"gpu_type": "h100; rm -rf /"},            # not on the allowlist (shell-injection shape)
    {"partition": "cpu-2h", "gpu_type": "h100"},  # gpu_type without GPUs
    {"gpus": 99},
    {"mem": 0},
])
def test_invalid_resource_requests_are_rejected(override):
    base = dict(partition="gpu-2h", gpus=1, gpu_type=None, mem=16)
    with pytest.raises(ComputeSessionsError):
        _slurm_backend().resolve_resources(**{**base, **override})


def test_docker_source_rejects_scheduler_args_and_takes_none():
    b = _docker_backend()
    assert b.resolve_resources(partition=None, gpus=None, gpu_type=None, mem=None) == {}
    with pytest.raises(ComputeSessionsError):
        b.resolve_resources(partition="gpu-2h", gpus=None, gpu_type=None, mem=None)
    with pytest.raises(ComputeSessionsError):
        b.resolve_resources(partition=None, gpus=1, gpu_type=None, mem=None)


def test_vast_resource_defaults_and_gpu_allowlist():
    b = _vast_backend()
    # default_gpu_type from the source config fills an unset gpu_type
    assert b.resolve_resources(partition=None, gpus=None, gpu_type=None, mem=None) == {"gpus": 1, "gpu_type": "RTX_4090", "mem": None}
    r = b.resolve_resources(partition=None, gpus=2, gpu_type="RTX_5090|H100_SXM", mem=64)
    assert r == {"gpus": 2, "gpu_type": "RTX_5090|H100_SXM", "mem": 64}
    for bad in (
        dict(partition="gpu-2h", gpus=None, gpu_type=None, mem=None),   # no partitions on vast
        dict(partition=None, gpus=0, gpu_type=None, mem=None),          # rented boxes always have ≥1 GPU
        dict(partition=None, gpus=None, gpu_type="A100; rm -rf /", mem=None),  # not on the allowlist
        dict(partition=None, gpus=None, gpu_type=None, mem=0),
    ):
        with pytest.raises(ComputeSessionsError):
            b.resolve_resources(**bad)


def test_vast_offer_filters_encode_request_caps_and_config_filter():
    b = _vast_backend()
    r = b.resolve_resources(partition=None, gpus=2, gpu_type="RTX_4090|H100_SXM", mem=32)
    f = build_offer_filters(b.source, r, max_price=1.25)
    assert f["num_gpus"] == {"eq": 2}
    assert f["gpu_name"] == {"in": ["RTX 4090", "H100 SXM"]}  # config underscores → vast's spaced names
    assert f["dph_total"] == {"lte": 1.25}
    assert f["cpu_ram"] == {"gte": 32 * 1024}  # vast reports MB
    assert f["disk_space"] == {"gte": b.source.disk}
    assert f["rentable"] == {"eq": True} and f["rented"] == {"eq": False}
    # the default offer_filter contributes the quality baseline
    assert f["verified"] == {"eq": True} and f["reliability"] == {"gt": 0.98}
    assert f["direct_port_count"] == {"gte": 2}


def test_offer_filter_clause_parsing():
    assert parse_filter_clauses("inet_down>=500 geolocation=DE static_ip=true dlperf>90 num<=4 cuda_vers!=11") == {
        "inet_down": {"gte": 500.0},
        "geolocation": {"eq": "DE"},
        "static_ip": {"eq": True},
        "dlperf": {"gt": 90.0},
        "num": {"lte": 4.0},
        "cuda_vers": {"neq": 11.0},
    }
    for bad in ("nonsense", "=5", "field="):
        with pytest.raises(ComputeSessionsError):
            parse_filter_clauses(bad)


def test_ledger_month_spend_open_stop_and_capped_reconcile(tmp_path, monkeypatch):
    # Freeze the clock mid-month so intervals never straddle the month boundary.
    fixed = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed if tz else fixed.replace(tzinfo=None)

    monkeypatch.setattr(vast_api, "datetime", _FrozenDatetime)
    monkeypatch.setattr(vast_api.time, "time", lambda: fixed.timestamp())
    now = fixed.timestamp()

    led = Ledger(tmp_path / "ledger.jsonl")
    led.start("1", "vast_a", dph=1.0)  # ts = now, zero accrual yet
    led._events()[0]  # sanity: parses
    # Rewrite instance 1's start to 2h ago: open instance accrues to now.
    led.path.write_text(json.dumps({"ts": now - 2 * 3600, "event": "start", "instance_id": "1", "session_id": "vast_a", "dph": 1.0}) + "\n")
    assert led.month_spend() == pytest.approx(2.0, abs=0.01)
    led.stop("1", "vast_a", ts=now - 3600)  # actually stopped after 1h
    assert led.month_spend() == pytest.approx(1.0, abs=0.01)
    assert not led.open_instances()

    # An instance that vanished without a stop: reconcile synthesizes one, capped at start + max_session_hours (the runner enforces that bound on-instance), not at now.
    led._append({"ts": now - 30 * 3600, "event": "start", "instance_id": "2", "session_id": "vast_b", "dph": 2.0})
    led.reconcile(live_ids=set(), max_session_hours=12)
    assert not led.open_instances()
    assert led.month_spend() == pytest.approx(1.0 + 12 * 2.0, abs=0.01)
    # A live instance stays open through reconcile.
    led._append({"ts": now - 1800, "event": "start", "instance_id": "3", "session_id": "vast_c", "dph": 4.0})
    led.reconcile(live_ids={"3"}, max_session_hours=12)
    assert set(led.open_instances()) == {"3"}


def test_instance_paths_map_to_flat_container_layout():
    p = InstancePaths()
    assert p.workdir("vast_abc") == "/workdir"
    assert p.logs_dir("vast_abc") == "/cs-logs"
    assert p.config_file("vast_abc") == "/cs-state/config.json"
    assert p.manifest("vast_abc") == "/cs-state/.sync-manifest"
    assert p.resolve_workdir_relative("vast_abc", "out/model.pt") == "/workdir/out/model.pt"
    for bad in ("../up", "/abs"):
        with pytest.raises(ComputeSessionsError):
            p.resolve_workdir_relative("vast_abc", bad)


# ---------- identifiers, source routing, and remote path scoping ----------

def test_session_id_routes_to_its_source():
    assert source_of("hydra_0a1b2c") == "hydra"
    assert source_of("gamingpc_ff00aa") == "gamingpc"
    assert source_of("vast_1234ab") == "vast"
    for bad in ("nodelimiter", "_leading", "has space"):
        with pytest.raises(ComputeSessionsError):
            source_of(bad)


def test_remote_paths_stay_inside_the_session():
    paths = RemotePaths("/base")
    assert paths.resolve_workdir_relative("hydra_abc", "sub/file.txt") == "/base/sessions/hydra_abc/workdir/sub/file.txt"
    for bad in ("../escape", "a/../../b", "/etc/passwd"):
        with pytest.raises(ComputeSessionsError):
            paths.resolve_workdir_relative("hydra_abc", bad)


def test_local_path_resolves_against_project_and_rejects_escapes(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    info = SimpleNamespace(project_path=str(proj), session_id="hydra_abc")
    # A relative local_path resolves under project_path (mirrors remote_path's workdir-relative contract).
    assert _resolve_local_inside_project(Path("out/x.jsonl"), info, "l") == (proj / "out/x.jsonl").resolve()
    # An absolute path inside the project is accepted as-is.
    assert _resolve_local_inside_project(proj / "a.bin", info, "l") == (proj / "a.bin").resolve()
    # Escapes are refused — traversal out of the project or an absolute path elsewhere.
    for bad in (Path("../escape"), Path("/etc/passwd"), Path("/tmp/other/x.py")):
        with pytest.raises(ComputeSessionsError):
            _resolve_local_inside_project(bad, info, "l")
    # Agent scratchpad roots are allowed despite being outside the project (path stays as given — /tmp may resolve to /private/tmp on macOS).
    scratch = Path("/tmp/claude-501/some-proj/some-session/scratchpad/helper.py")
    assert _resolve_local_inside_project(scratch, info, "l") == scratch.resolve()


def test_ids_must_be_plain_tokens():
    validate_session_id("hydra_0a1b2c")
    validate_command_id("cmd_deadbeef")
    for bad in ("", "sess/../x", "sess abc", "café"):
        with pytest.raises(ComputeSessionsError):
            validate_session_id(bad)


def test_signals_are_validated():
    for ok in ("TERM", "SIGKILL", "9"):
        _validate_signal(ok)
    for bad in ("FAKETERM", "99", "TERM; reboot"):
        with pytest.raises(ComputeSessionsError):
            _validate_signal(bad)


# ---------- command status semantics ----------

def test_command_status_classification():
    assert _classify_command_status("exited 0") == ("exited", 0)
    assert _classify_command_status("exited 137") == ("exited", 137)
    assert _classify_command_status("age 5")[0] == "running"    # fresh heartbeat
    assert _classify_command_status("age 999")[0] == "killed"   # stale heartbeat: died without recording an exit
    assert _classify_command_status("missing")[0] == "missing"


def test_launch_status_parsing():
    assert _parse_launch_status("DONE 0\n") == ("exited", 0)
    assert _parse_launch_status("login banner\nRUNNING\n") == ("running", None)
    assert _parse_launch_status("garbage") == ("", None)


def test_launcher_creates_start_sentinel():
    # The contract list_commands' started_at depends on: the launcher writes a `.start` file synchronously at launch (the `.pid` mtime is the heartbeat, not the launch time).
    script = _build_detached_launch("echo hi", "cmd_x")
    assert "cmd_x.start" in script


def test_launcher_records_exit_before_venv_snapshot_epilogue():
    # The hibernation epilogue must never delay the agent-visible result: the inner script writes `.exit` explicitly after the user command, and only then spawns the (detached, guarded) snapshot helper. The helper is bound only on slurm sources, so the epilogue must also be gated on its existence.
    script = _build_detached_launch("uv sync", "cmd_x")
    assert "-x /cs-helpers/venv-snapshot.sh" in script
    assert "setsid" in script
    explicit_exit_write = 'echo "$ec" >'
    assert script.index(explicit_exit_write) < script.index("venv-snapshot.sh")


# ---------- list_commands timestamps ----------

def test_list_commands_timestamps_come_from_launch_not_heartbeat(tmp_path):
    """started_at must reflect the `.start` sentinel (launch time), not the heartbeat-touched `.pid` mtime — a 14-min command previously reported a ~7s lifespan."""
    now = time.time()

    def mk(name: str, age_seconds: float, content: str = "") -> None:
        p = tmp_path / name
        p.write_text(content)
        os.utime(p, (now - age_seconds, now - age_seconds))

    mk("cmd_long.start", 900)
    mk("cmd_long.pid", 15, "1234")  # heartbeat kept touching it until just before exit
    mk("cmd_long.exit", 10, "0")
    mk("cmd_live.start", 120)
    mk("cmd_live.pid", 3, "5678")   # fresh heartbeat, no .exit → running
    mk("cmd_dead.pid", 500, "42")   # stale heartbeat, no .exit → killed; no .start → no started_at
    mk("cmd_venvsnap_1.start", 60)  # internal housekeeping — never listed
    mk("cmd_venvsnap_1.exit", 55, "0")

    res = subprocess.run(
        [sys.executable, "-", str(tmp_path), "0", "0", str(_PID_STALE_SECONDS)],
        input=_LIST_COMMANDS_PY, capture_output=True, text=True, check=True,
    )
    data = json.loads(res.stdout)
    rows = {r["command_id"]: r for r in data["commands"]}

    # Internal venvsnap commands are excluded entirely (rows and total).
    assert "cmd_venvsnap_1" not in rows
    assert data["total"] == 3

    fmt = "%Y-%m-%dT%H:%M:%SZ"
    long_row = rows["cmd_long"]
    duration = (datetime.strptime(long_row["exited_at"], fmt) - datetime.strptime(long_row["started_at"], fmt)).total_seconds()
    assert long_row["status"] == "exited" and long_row["exit_code"] == 0
    assert 880 <= duration <= 900  # ~890s; the .pid mtime would have given ~5s

    # Null fields (exit_code/exited_at while running, missing started_at) are omitted, not null.
    assert rows["cmd_live"]["status"] == "running"
    assert not {"exit_code", "exited_at"} & rows["cmd_live"].keys()
    assert rows["cmd_dead"]["status"] == "killed"
    assert "started_at" not in rows["cmd_dead"]

    # Newest-first ordering by launch time.
    assert [r["command_id"] for r in data["commands"]] == ["cmd_live", "cmd_long", "cmd_dead"]


# ---------- session serialization ----------

def test_session_info_round_trips_and_agent_view_hides_plumbing():
    info = SessionInfo(
        session_id="hydra_1", source="hydra", project_id="p", project_path="/tmp/p", created_at="2026-01-01T00:00:00Z",
        resources={"partition": "gpu-2h", "gpus": 1, "gpu_type": None, "mem": 42},
        idle_timeout_minutes=20, image="images/default.sif",
        status=SessionStatus.ACTIVE, job_id="123", sshd_port=2222,
    )
    assert SessionInfo.from_json(info.to_json()) == info
    agent = info.to_agent_json()
    assert agent["status"] == "active"
    assert agent["source"] == "hydra"
    # Backend resources are flattened into the row; null resource fields are omitted.
    assert agent["partition"] == "gpu-2h" and agent["mem"] == 42
    assert not {"job_id", "sshd_port", "project_id", "image", "resources", "gpu_type"} & agent.keys()
    # Null derived fields are omitted too (no idle heartbeat yet).
    assert "seconds_until_idle_deactivate" not in agent


def test_active_session_reports_idle_countdown():
    now = datetime.now(timezone.utc)
    info = SessionInfo(
        session_id="hydra_1", source="hydra", project_id="p", project_path="/p", created_at=_iso(now),
        idle_timeout_minutes=10, status=SessionStatus.ACTIVE,
        last_activated_at=_iso(now), last_activity_at=_iso(now - timedelta(minutes=4)),
    )
    remaining = info.to_enriched_json()["seconds_until_idle_deactivate"]
    assert 0 < remaining <= 6 * 60
    info.status = SessionStatus.INACTIVE
    assert info.to_enriched_json()["seconds_until_idle_deactivate"] is None


# ---------- sync planning and project identity ----------

def test_sync_plan_ships_non_ignored_files_only(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("data/\n")
    (tmp_path / "main.py").write_text("print('hi')\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "weights.bin").write_bytes(b"\x00")
    (tmp_path / "gone.py").write_text("x\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    (tmp_path / "untracked.txt").write_text("new\n")
    (tmp_path / "gone.py").unlink()  # tracked but deleted locally — must not be shipped

    files = plan_sync(tmp_path)
    assert {"main.py", ".gitignore", "untracked.txt"} <= set(files)
    assert not any(f.startswith("data/") for f in files)
    assert "gone.py" not in files


def test_sync_plan_outside_git_skips_junk_dirs(tmp_path):
    (tmp_path / "a.py").write_text("x\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.pyc").write_bytes(b"\x00")
    assert plan_sync(tmp_path) == ["a.py"]


def test_project_id_prefers_git_remote_and_falls_back_to_path(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "remote", "add", "origin", "git@github.com:owner/My-Repo.git"], cwd=repo, check=True)
    assert project_id(repo) == "owner-my-repo"
    plain = tmp_path / "plain"
    plain.mkdir()
    assert project_id(plain).startswith("plain-")


# ---------- config file ----------

_GOOD_CONFIG = """\
default_source = "hydra"
max_sessions_per_repo = 5

[sources.hydra]
type = "slurm"
host = "hydra"
image = "images/default.sif"
partitions = ["gpu-2h", "cpu-2h", "gpu-test"]
default_partition = "gpu-2h"
gpu_types = ["h100", "80gb"]
default_mem = 64

[sources.gamingpc]
type = "docker"
host = "gamingpc"
image = "compute-sessions:latest"
description = "1x RTX 4090 24GB, free but only one session at a time"

[sources.vast]
type = "vast"
image = "me/compute-sessions-vast:latest"
description = "rented GPUs - costly, ephemeral disk"
gpu_types = ["RTX_4090", "H100_SXM"]
default_gpu_type = "RTX_4090"
max_instance_price = 1.5
max_hourly_spend = 3.0
max_total_spend = 100.0
"""


def test_config_parses_sources(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(_GOOD_CONFIG)
    cfg = load_config(p)
    assert set(cfg.sources) == {"hydra", "gamingpc", "vast"}
    assert cfg.default_source == "hydra"
    assert cfg.max_sessions_per_repo == 5
    hydra = cfg.sources["hydra"]
    assert hydra.type == "slurm" and hydra.default_partition == "gpu-2h" and hydra.default_mem == 64
    pc = cfg.sources["gamingpc"]
    assert pc.type == "docker" and pc.description.startswith("1x RTX 4090")
    # remote_base defaults; docker sources need no partitions.
    assert pc.remote_base == "~/.compute-sessions" and pc.partitions == ()
    v = cfg.sources["vast"]
    # vast sources have no host; spend limits parsed; sizing defaults apply.
    assert v.host == "" and v.max_total_spend == 100.0 and v.max_session_hours == 12 and v.disk == 32


def test_config_env_var_selects_path(tmp_path, monkeypatch):
    p = tmp_path / "elsewhere.toml"
    p.write_text(_GOOD_CONFIG)
    monkeypatch.setenv("COMPUTE_SESSIONS_CONFIG", str(p))
    assert set(load_config().sources) == {"hydra", "gamingpc", "vast"}


@pytest.mark.parametrize("mutate", [
    lambda s: s.replace('type = "slurm"', 'type = "pbs"'),                     # unknown backend type
    lambda s: s.replace("[sources.hydra]", "[sources.my_cluster]"),           # underscore breaks session-id routing
    lambda s: s.replace('partitions = ["gpu-2h", "cpu-2h", "gpu-test"]\n', ""),  # slurm without partitions
    lambda s: s.replace('default_source = "hydra"', 'default_source = "nope"'),
    lambda s: s.replace('gpu_types = ["h100", "80gb"]', 'gpu_types = ["h100; rm -rf /"]'),  # injection-shaped token
    lambda s: s.replace('max_total_spend = 100.0\n', ""),                      # vast without a monthly cap
    lambda s: s.replace('default_gpu_type = "RTX_4090"', 'default_gpu_type = "B200"'),  # default not in gpu_types
])
def test_invalid_configs_are_rejected(tmp_path, mutate):
    p = tmp_path / "config.toml"
    p.write_text(mutate(_GOOD_CONFIG))
    with pytest.raises(ComputeSessionsError):
        load_config(p)


def test_missing_config_gives_actionable_error(tmp_path):
    with pytest.raises(ComputeSessionsError, match="setup-compute-source"):
        load_config(tmp_path / "nope.toml")


# ---------- incremental logs: cursor, early return, progress collapse ----------

def _local_slurm(tmp_path) -> SlurmBackend:
    """SlurmBackend whose access runs bash locally against tmp_path — real scripts, no ssh."""
    scfg = SourceConfig(
        name="hydra", type="slurm", host="hydra", image="i", remote_base=str(tmp_path),
        partitions=("gpu-2h",), default_partition="gpu-2h", default_mem=42,
    )
    return SlurmBackend(scfg, LocalAccess(scfg))


def _write_command_files(tmp_path, session_id: str, command_id: str, out: str = "", err: str = "") -> Path:
    logdir = tmp_path / "sessions" / session_id / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / f"{command_id}.out").write_text(out)
    (logdir / f"{command_id}.err").write_text(err)
    (logdir / f"{command_id}.pid").write_text("1")  # fresh mtime → running
    return logdir


def test_logs_cursor_returns_only_new_output(tmp_path):
    b = _local_slurm(tmp_path)
    logdir = _write_command_files(tmp_path, "hydra_1", "cmd_a", out="step 1\nstep 2\n")
    first = logs_with_access(b.access, "hydra_1", "cmd_a")
    assert first["status"] == "running"
    assert first["stdout"] == "step 1\nstep 2\n"
    cursor = first["cursor"]

    # No new output → empty delta, same cursor.
    again = logs_with_access(b.access, "hydra_1", "cmd_a", since=cursor)
    assert again["stdout"] == "" and again["stderr"] == ""
    assert again["cursor"] == cursor

    # New output + exit → delta only, exit recorded, no cursor on a settled command.
    with (logdir / "cmd_a.out").open("a") as fh:
        fh.write("step 3\n")
    (logdir / "cmd_a.exit").write_text("0")
    done = logs_with_access(b.access, "hydra_1", "cmd_a", since=cursor)
    assert done["status"] == "exited" and done["exit_code"] == 0
    assert done["stdout"] == "step 3\n"
    assert "cursor" not in done


def test_logs_cursor_covers_returned_bytes_under_concurrent_writes(tmp_path):
    # The cursor (wc -c) and the content read (tail) are separate steps of one remote script; the content read is capped to the measured sizes, else a burst-writing command pushes bytes past the reported cursor and the next since=cursor read re-emits them (observed as `cs run` printing uv's whole error block twice). The ±1 tolerance absorbs BSD sed's synthetic trailing newline on a mid-line cut.
    b = _local_slurm(tmp_path)
    logdir = _write_command_files(tmp_path, "hydra_1", "cmd_r")
    writer = subprocess.Popen([sys.executable, "-u", "-c", (
        "import sys, time\n"
        "f = open(sys.argv[1], 'ab', buffering=0)\n"
        "for _ in range(2000):\n"
        "    f.write(b'x' * 4095 + b'\\n')\n"
        "    time.sleep(0.001)\n"
    ), str(logdir / "cmd_r.out")])
    try:
        oo = 0
        for _ in range(12):
            res = logs_with_access(b.access, "hydra_1", "cmd_r", since=f"{oo}:0")
            assert res["status"] == "running"
            new_oo = int(res["cursor"].split(":")[0])
            assert len(res["stdout"]) <= new_oo - oo + 1
            oo = new_oo
    finally:
        writer.terminate()
        writer.wait()


def test_logs_collapses_progress_bars_and_counts_dropped_lines(tmp_path):
    b = _local_slurm(tmp_path)
    _write_command_files(
        tmp_path, "hydra_1", "cmd_b",
        out="loading: 10%\rloading: 50%\rloading: 100%\ndone\n",
        err="w1\nw2\nw3\n",
    )
    res = logs_with_access(b.access, "hydra_1", "cmd_b", max_lines=2)
    # \r-overwritten segments collapse to the final state; raw files stay untouched.
    assert res["stdout"] == "loading: 100%\ndone\n"
    assert res["stderr"] == "w2\nw3\n"
    assert res["stderr_truncated_lines"] == 1


def test_collapse_preserves_crlf_lines(tmp_path):
    # CRLF endings must survive the remote \r-collapse (the trailing \r is stripped, not the line).
    b = _local_slurm(tmp_path)
    _write_command_files(tmp_path, "hydra_1", "cmd_d", out="a\r\nb\n")
    res = logs_with_access(b.access, "hydra_1", "cmd_d")
    assert res["stdout"] == "a\nb\n"


def test_since_cursor_is_validated():
    assert _parse_since(None) == (0, 0)
    assert _parse_since("12:0") == (12, 0)
    for bad in ("12", "a:b", "-1:0", "1:2:3"):
        with pytest.raises(ComputeSessionsError):
            _parse_since(bad)


def test_wait_for_command_returns_early_on_new_output(tmp_path):
    b = _local_slurm(tmp_path)
    _write_command_files(tmp_path, "hydra_1", "cmd_c", out="already here\n")
    info = SessionInfo(
        session_id="hydra_1", source="hydra", project_id="p",
        project_path=str(tmp_path), created_at="2026-01-01T00:00:00Z",
    )
    sdir = tmp_path / "sessions" / "hydra_1"
    (sdir / "config.json").write_text(json.dumps(info.to_json()))
    t0 = time.monotonic()
    res = wait_for_command(b, "hydra_1", "cmd_c", timeout_seconds=10, since="0:0")
    # Output past the cursor already exists, so the wait must settle immediately instead of sitting out the window.
    assert time.monotonic() - t0 < 3
    assert res["status"] == "running" and res["stdout"] == "already here\n"


# ---------- clone: server-side workdir copy ----------

def test_copy_workdir_copies_contents_and_manifest(tmp_path):
    b = _local_slurm(tmp_path)
    src_wd = tmp_path / "sessions" / "hydra_1" / "workdir"
    (src_wd / "data").mkdir(parents=True)
    (src_wd / "data" / "weights.bin").write_text("blob")
    (src_wd / "train.py").write_text("code")
    (tmp_path / "sessions" / "hydra_1" / ".sync-manifest").write_text("train.py\n")
    (tmp_path / "sessions" / "hydra_2" / "workdir").mkdir(parents=True)

    b.copy_workdir("hydra_1", "hydra_2")
    dst = tmp_path / "sessions" / "hydra_2"
    assert (dst / "workdir" / "data" / "weights.bin").read_text() == "blob"
    assert (dst / "workdir" / "train.py").read_text() == "code"
    assert (dst / ".sync-manifest").read_text() == "train.py\n"


def test_vast_rejects_clone():
    with pytest.raises(ComputeSessionsError, match="not supported on vast"):
        _vast_backend().copy_workdir("vast_1", "vast_2")


# ---------- upload/download destination semantics ----------

class _TransferAccess(LocalAccess):
    """LocalAccess plus local-filesystem rsync, so download/upload run end-to-end against tmp dirs."""

    def rsync_from(self, remote_path, local_path, *, contents_only=False):
        src = f"{remote_path}/" if contents_only else str(remote_path)
        dst = f"{local_path}/" if contents_only else str(local_path)
        subprocess.run(["rsync", "-a", src, dst], check=True)

    rsync_to = rsync_from


def _transfer_backend(tmp_path) -> SlurmBackend:
    scfg = SourceConfig(
        name="hydra", type="slurm", host="hydra", image="i", remote_base=str(tmp_path),
        partitions=("gpu-2h",), default_partition="gpu-2h", default_mem=42,
    )
    b = SlurmBackend(scfg, _TransferAccess(scfg))
    proj = tmp_path / "proj"
    proj.mkdir()
    (tmp_path / "sessions" / "hydra_1" / "workdir").mkdir(parents=True)
    info = SessionInfo(
        session_id="hydra_1", source="hydra", project_id="p", project_path=str(proj),
        created_at="2026-01-01T00:00:00Z", status=SessionStatus.INACTIVE,
    )
    b.write_info(info)
    return b


def test_download_dest_conventions(tmp_path, monkeypatch):
    from compute_sessions.session import download
    b = _transfer_backend(tmp_path)
    proj = tmp_path / "proj"
    monkeypatch.chdir(proj)
    wd = tmp_path / "sessions" / "hydra_1" / "workdir"
    (wd / "runs").mkdir()
    (wd / "runs" / "log.csv").write_text("l")
    (wd / "runs" / "result.json").write_text("r")

    # A destination directory that doesn't exist is an error, not an implicit mkdir -p (typo'd-path regression).
    with pytest.raises(ComputeSessionsError, match="not an existing directory"):
        download(b, "hydra_1", "runs/log.csv", "runs/in_baseline/")
    with pytest.raises(ComputeSessionsError, match="parent directory"):
        download(b, "hydra_1", "runs/log.csv", "runs/in_baseline/log.csv")

    # Trailing-slash dest: files land INSIDE (three-files-into-one-file regression).
    (proj / "runs" / "in_baseline").mkdir(parents=True)
    for f in ("log.csv", "result.json"):
        out = download(b, "hydra_1", f"runs/{f}", "runs/in_baseline/")
        assert out["local_path"] == f"runs/in_baseline/{f}"
    assert (proj / "runs" / "in_baseline" / "log.csv").read_text() == "l"
    assert (proj / "runs" / "in_baseline" / "result.json").read_text() == "r"

    # A directory is placed AS a directory inside an existing dest — never merged over its contents (tracked-files-clobber regression).
    (proj / "results").mkdir()
    (proj / "results" / "rq1.json").write_text("keep")
    out = download(b, "hydra_1", "runs", "results/")
    assert out["local_path"] == "results/runs/"
    assert (proj / "results" / "rq1.json").read_text() == "keep"
    assert (proj / "results" / "runs" / "log.csv").exists()

    # Explicit contents-merge via source trailing slash; repeat to the same named dir stays idempotent (no results/runs/runs).
    download(b, "hydra_1", "runs/", "flat")
    assert (proj / "flat" / "log.csv").exists()
    download(b, "hydra_1", "runs", "results/runs")
    assert not (proj / "results" / "runs" / "runs").exists()

    # A file into an existing dir goes inside (cp semantics); a dir refuses a dest that exists as a file.
    out = download(b, "hydra_1", "runs/log.csv", "results/runs")
    assert out["local_path"] == "results/runs/log.csv"
    (proj / "clash").write_text("f")
    with pytest.raises(ComputeSessionsError, match="exists as a file"):
        download(b, "hydra_1", "runs", "clash")


def test_upload_dest_conventions(tmp_path, monkeypatch):
    from compute_sessions.session import upload
    b = _transfer_backend(tmp_path)
    proj = tmp_path / "proj"
    monkeypatch.chdir(proj)
    wd = tmp_path / "sessions" / "hydra_1" / "workdir"
    (proj / "ckpt.pt").write_text("w")
    (proj / "adapter").mkdir()
    (proj / "adapter" / "config.json").write_text("c")

    # A destination directory that doesn't exist is an error, not an implicit mkdir -p (typo'd-path regression).
    with pytest.raises(ComputeSessionsError, match="not an existing directory"):
        upload(b, "hydra_1", "ckpt.pt", "outputs/geometry/")
    with pytest.raises(ComputeSessionsError, match="parent directory"):
        upload(b, "hydra_1", "ckpt.pt", "outputs/geometry/ckpt.pt")

    # Trailing-slash remote: file lands INSIDE; a second file to the same dir doesn't clobber the first (file-named-like-the-dir regression).
    (wd / "outputs" / "geometry").mkdir(parents=True)
    out = upload(b, "hydra_1", "ckpt.pt", "outputs/geometry/")
    assert out["remote_path"] == "outputs/geometry/ckpt.pt"
    (proj / "eval.jsonl").write_text("e")
    upload(b, "hydra_1", "eval.jsonl", "outputs/geometry/")
    assert (wd / "outputs" / "geometry" / "ckpt.pt").read_text() == "w"
    assert (wd / "outputs" / "geometry" / "eval.jsonl").read_text() == "e"

    # Directory into an existing dir: placed as a directory (adapter-spilled-one-level-up regression); repeat upload to its own name stays idempotent.
    out = upload(b, "hydra_1", "adapter", "outputs/geometry/")
    assert out["remote_path"] == "outputs/geometry/adapter/"
    assert (wd / "outputs" / "geometry" / "adapter" / "config.json").exists()
    upload(b, "hydra_1", "adapter", "outputs/geometry/adapter")
    assert not (wd / "outputs" / "geometry" / "adapter" / "adapter").exists()

    # No remote → basename; existing remote FILE refuses a directory source.
    out = upload(b, "hydra_1", "adapter", "")
    assert out["remote_path"] == "adapter/"
    upload(b, "hydra_1", "ckpt.pt", "clash")
    with pytest.raises(ComputeSessionsError, match="exists as a file"):
        upload(b, "hydra_1", "adapter", "clash")


def test_cli_walltime_death_cause():
    from compute_sessions.cli import _res_summary, _walltime_hit
    base = {"resources": {"partition": "gpu-5h", "gpus": 1, "mem": 42},
            "last_activated_at": "2026-07-23T11:21:09Z", "last_deactivated_at": "2026-07-23T16:21:25Z"}
    assert "wall-clock limit" in _walltime_hit(base)
    # Died well before the window, unparseable partition, or missing timestamps → not a wall-clock kill.
    assert _walltime_hit({**base, "last_deactivated_at": "2026-07-23T13:00:00Z"}) is None
    assert _walltime_hit({**base, "resources": {"partition": "gpu-test"}}) is None
    assert _walltime_hit({**base, "last_activated_at": None}) is None
    assert _res_summary({"partition": "gpu-2d", "gpus": 1, "gpu_type": None, "mem": 96}) == "gpu-2d, 1 gpu, 96G"
    assert _res_summary({}) == ""


def test_cli_multi_source_split():
    from compute_sessions.cli import _split_sources_dest
    assert _split_sources_dest(["a"], "u") == (["a"], None)
    assert _split_sources_dest(["a", "b"], "u") == (["a"], "b")
    assert _split_sources_dest(["a", "b", "dir/"], "u") == (["a", "b"], "dir/")
    with pytest.raises(SystemExit):
        _split_sources_dest(["a", "b", "c"], "u")


def test_cli_token_normalization():
    # Copy-pasted ids arrive with stray whitespace and uppercase hex; the source name keeps its case (it's a config key).
    from compute_sessions.cli import _norm_token
    assert _norm_token(" 92A0 ") == "92a0"
    assert _norm_token("hydra_8EC5") == "hydra_8ec5"
    assert _norm_token("   ") == ""


# ---------- allocated-gpu reporting ----------

def test_gpu_field_round_trips_and_shows_in_agent_view():
    info = SessionInfo(
        session_id="hydra_1", source="hydra", project_id="p", project_path="/p",
        created_at="2026-01-01T00:00:00Z", status=SessionStatus.ACTIVE,
        gpu="NVIDIA A100 80GB PCIe",
    )
    assert SessionInfo.from_json(info.to_json()) == info
    assert info.to_agent_json()["gpu"] == "NVIDIA A100 80GB PCIe"
    info.gpu = None
    assert "gpu" not in info.to_agent_json()


def test_runner_detects_gpu_models(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cs_runner", Path(__file__).resolve().parent.parent / "remote" / "runner.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def fake_smi(out):
        return lambda *a, **k: SimpleNamespace(stdout=out)

    monkeypatch.setattr(runner.subprocess, "run", fake_smi("GPU 0: NVIDIA A100 80GB PCIe (UUID: GPU-1)\n"))
    assert runner.detect_gpu_models() == "NVIDIA A100 80GB PCIe"
    monkeypatch.setattr(runner.subprocess, "run", fake_smi(
        "GPU 0: NVIDIA H100 (UUID: GPU-1)\nGPU 1: NVIDIA H100 (UUID: GPU-2)\n"))
    assert runner.detect_gpu_models() == "2x NVIDIA H100"
    monkeypatch.setattr(runner.subprocess, "run", fake_smi(""))
    assert runner.detect_gpu_models() == ""

    # MIG node: parent GPUs stay visible to a 1-slice job — the allocated MIG device must win over the "8x A100" parent listing.
    mig_out = "".join(
        f"GPU {i}: NVIDIA A100 80GB PCIe (UUID: GPU-{i})\n" for i in range(8)
    ) + "  MIG 3g.40gb     Device  0: (UUID: MIG-abc)\n  MIG 3g.40gb     Device  1: (UUID: MIG-def)\n"
    monkeypatch.setattr(runner.subprocess, "run", fake_smi(mig_out))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-abc")
    assert runner.detect_gpu_models() == "MIG 3g.40gb (NVIDIA A100 80GB PCIe)"
    # No scheduler filter → all MIG devices, still never the parent count.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert runner.detect_gpu_models() == "2x MIG 3g.40gb (NVIDIA A100 80GB PCIe)"


# ---------- uncapped core waits ----------

def test_wait_for_command_takes_long_timeouts(tmp_path):
    # The poll-loop waits take arbitrary timeouts (a settled command still returns immediately).
    b = _local_slurm(tmp_path)
    logdir = _write_command_files(tmp_path, "hydra_1", "cmd_e", out="done\n")
    (logdir / "cmd_e.exit").write_text("0")
    info = SessionInfo(
        session_id="hydra_1", source="hydra", project_id="p",
        project_path=str(tmp_path), created_at="2026-01-01T00:00:00Z",
    )
    (tmp_path / "sessions" / "hydra_1" / "config.json").write_text(json.dumps(info.to_json()))
    t0 = time.monotonic()
    res = wait_for_command(b, "hydra_1", "cmd_e", timeout_seconds=600)
    assert time.monotonic() - t0 < 3
    assert res["status"] == "exited" and res["exit_code"] == 0


# ---------- cs CLI: session resolution, arg splitting, formatting ----------

def _cli_info(sid: str, project: str, status: SessionStatus) -> SessionInfo:
    return SessionInfo(
        session_id=sid, source=sid.partition("_")[0], project_id=project,
        project_path="/tmp/x", created_at="2026-01-01T00:00:00Z", status=status,
    )


class _FakeCliBackend:
    def __init__(self, infos):
        self._infos = infos

    def list_infos(self, pid=None):
        return [i for i in self._infos if pid is None or i.project_id == pid]

    def read_info(self, sid):
        for i in self._infos:
            if i.session_id == sid:
                return i
        raise ComputeSessionsError("not found")


def test_cli_session_resolution(monkeypatch):
    from compute_sessions import cli
    infos = [
        _cli_info("hy_aaa111", "p1", SessionStatus.ACTIVE),
        _cli_info("hy_bbb222", "p2", SessionStatus.INACTIVE),
        _cli_info("hy_bbb333", "p2", SessionStatus.ACTIVE),
    ]
    monkeypatch.setattr(cli, "_CFG", object())
    monkeypatch.setattr(cli, "_BACKENDS", {"hy": _FakeCliBackend(infos)})
    assert cli._resolve_session("hy_aaa111") == "hy_aaa111"       # exact id: fast path
    assert cli._resolve_session("hy_a") == "hy_aaa111"            # unique prefix
    assert cli._resolve_session("aaa1") == "hy_aaa111"            # bare hex tail prefix
    with pytest.raises(ComputeSessionsError, match="ambiguous"):
        cli._resolve_session("hy_bbb")
    with pytest.raises(ComputeSessionsError, match="no session matches"):
        cli._resolve_session("hy_zzz")
    # No token: the cwd project's only session, or its only live one.
    monkeypatch.setattr(cli, "derive_project_id", lambda p: "p1")
    assert cli._resolve_session(None) == "hy_aaa111"
    monkeypatch.setattr(cli, "derive_project_id", lambda p: "p2")
    assert cli._resolve_session(None) == "hy_bbb333"
    monkeypatch.setattr(cli, "derive_project_id", lambda p: "p9")
    with pytest.raises(ComputeSessionsError, match="no sessions for project"):
        cli._resolve_session(None)
    # Several sessions, none live: the error leads with that and lists most-recent-first, capped.
    dead = [_cli_info(f"hy_ddd{i}00", "p3", SessionStatus.INACTIVE) for i in range(6)]
    monkeypatch.setattr(cli, "_BACKENDS", {"hy": _FakeCliBackend(dead)})
    monkeypatch.setattr(cli, "derive_project_id", lambda p: "p3")
    with pytest.raises(ComputeSessionsError, match=r"none is live.*2 more via"):
        cli._resolve_session(None)


def test_cli_bare_hex_leading_token(monkeypatch):
    from compute_sessions import cli
    infos = [_cli_info("hy_aaa111", "p1", SessionStatus.ACTIVE)]
    monkeypatch.setattr(cli, "_BACKENDS", {"hy": _FakeCliBackend(infos)})
    # A hex token claims the session slot only when a session matches; anything else stays part of the command.
    assert cli._split_leading_session(("aaa1", "echo", "hi")) == ("aaa1", ["echo", "hi"])
    assert cli._split_leading_session(("beef", "echo", "hi")) == (None, ["beef", "echo", "hi"])
    assert cli._split_leading_session(("date", "+%s")) == (None, ["date", "+%s"])


def test_cli_leading_session_token_split(monkeypatch):
    from compute_sessions import cli
    monkeypatch.setattr(cli, "_BACKENDS", {"hy": object()})
    # <configured-source>_… up front is a session token; anything else belongs to the command/path args.
    assert cli._split_leading_session(("hy_abc", "echo", "hi")) == ("hy_abc", ["echo", "hi"])
    assert cli._split_leading_session(("echo", "hy_abc")) == (None, ["echo", "hy_abc"])
    assert cli._split_leading_session(("other_abc", "x")) == (None, ["other_abc", "x"])
    assert cli._split_leading_session(()) == (None, [])


def test_cli_run_arg_parsing(monkeypatch):
    """`cs run`'s option parsing must stop at the first positional (a `-d` inside the user command stays in the command), and multi-arg commands are quoted exec-style so args with spaces survive."""
    from click.testing import CliRunner
    from compute_sessions import cli
    seen = []
    monkeypatch.setattr(cli.run, "callback", lambda args, detach, tail_n: seen.append((args, detach, tail_n)))
    runner = CliRunner()
    for argv in (["run", "python", "train.py", "-d"], ["run", "-d", "python", "train.py"], ["run", "python", "-c", 'print("hi there")'], ["run", "--tail", "5", "sh", "x.sh", "--tail", "9"]):
        assert runner.invoke(cli.cli, argv).exit_code == 0
    assert seen == [
        (("python", "train.py", "-d"), False, None),  # -d after the command belongs to the command
        (("python", "train.py"), True, None),         # -d before the command is cs's detach flag
        (("python", "-c", 'print("hi there")'), False, None),
        (("sh", "x.sh", "--tail", "9"), False, 5),    # --tail after the command belongs to the command
    ]


def test_cli_version():
    from click.testing import CliRunner
    from compute_sessions import cli
    res = CliRunner().invoke(cli.cli, ["--version"])
    assert res.exit_code == 0 and res.output.startswith("cs, version ")


def test_cli_tail_emitter(capsys):
    from compute_sessions.cli import _TailEmitter
    t = _TailEmitter(2)
    t({"stdout": "a\nb\n", "stderr": "e1\n"})
    t({"stdout": "c\nd\ne\n"})
    assert capsys.readouterr() == ("", "")  # nothing until flush
    t.flush()
    got = capsys.readouterr()
    assert got.out == "d\ne\n" and got.err == "e1\n"
    t.flush()  # idempotent — the finally-path double flush must not re-emit
    assert capsys.readouterr() == ("", "")


def test_cli_duration_formatting():
    from compute_sessions import cli
    assert cli._dur(None) == "-"
    assert cli._dur(42) == "42s"
    assert cli._dur(75) == "1m15s"
    assert cli._dur(3660) == "1h01m"
    assert cli._dur(90000) == "1d1h"


def test_cli_help_renders_and_config_errors_surface_at_run_time(monkeypatch):
    from click.testing import CliRunner
    from compute_sessions import cli
    runner = CliRunner()
    assert runner.invoke(cli.cli, ["--help"]).exit_code == 0
    # A broken config must not break --help, but must fail commands with the tool-failure code.
    monkeypatch.setattr(cli, "_CFG", None)
    monkeypatch.setattr(cli, "_CFG_ERROR", RuntimeError("boom"))
    res = runner.invoke(cli.cli, ["list"])
    assert res.exit_code == cli.EXIT_TOOL_FAILURE
