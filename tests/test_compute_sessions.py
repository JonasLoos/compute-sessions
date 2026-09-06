"""High-level behavioral tests — no cluster or network required. They pin the contracts the cs CLI relies on (config parsing, validation, status semantics, sync planning), not implementation details. Run with `uv run pytest`."""
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

from compute_sessions.cluster import Cluster
from compute_sessions.config import Config, load_config
from compute_sessions.errors import ComputeSessionsError, RemoteError
from compute_sessions.models import SessionInfo, SessionStatus
from compute_sessions.paths import RemotePaths, validate_command_id, validate_session_id
from compute_sessions.project import project_id
from compute_sessions.run import _build_detached_launch, _parse_launch_status, _validate_signal
from compute_sessions.session import (
    _LIST_COMMANDS_PY,
    _PID_STALE_SECONDS,
    _classify_command_status,
    _parse_since,
    _resolve_local_inside_project,
    logs_with_access,
    wait_for_command,
)
from compute_sessions.ssh import CompletedRemote
from compute_sessions.sync import plan_sync


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _cfg(**overrides) -> Config:
    base = dict(
        host="hydra", image="images/default.sif",
        partitions=("cpu-2h", "gpu-2h", "gpu-test"), default_partition="gpu-2h",
        gpu_types=("h100", "80gb", "40gb"), default_mem=42,
    )
    return Config(**{**base, **overrides})


class LocalAccess:
    """HostAccess look-alike that runs the same shell scripts on this machine against a tmp base dir — no ssh."""

    def __init__(self, base: Path):
        self.paths = RemotePaths(str(base))

    def run(self, cmd: str, *, check: bool = True, input_text: str | None = None) -> CompletedRemote:
        proc = subprocess.run(["bash", "-c", cmd], input=input_text, capture_output=True, text=True, errors="replace")
        if check and proc.returncode != 0:
            raise RemoteError("local command failed", command=cmd, stderr=proc.stderr, exit_code=proc.returncode)
        return CompletedRemote(proc.returncode, proc.stdout, proc.stderr)

    def close_tunnel(self, session_id: str) -> None:
        pass


def _local_cluster(tmp_path) -> Cluster:
    return Cluster(_cfg(remote_base=str(tmp_path)), LocalAccess(tmp_path))


# ---------- resource requests ----------

def test_resource_requests_resolve_with_config_defaults():
    resolve = Cluster(_cfg()).resolve_resources
    assert resolve(partition="gpu-2h", gpus=2, gpu_type="h100|80gb", mem=64) == {"partition": "gpu-2h", "gpus": 2, "gpu_type": "h100|80gb", "mem": 64}
    assert resolve(partition=None, gpus=None, gpu_type=None, mem=None) == {"partition": "gpu-2h", "gpus": 1, "gpu_type": None, "mem": 42}
    assert resolve(partition="cpu-2h", gpus=4, gpu_type=None, mem=8)["gpus"] == 0  # cpu partitions never get GPUs


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
        Cluster(_cfg()).resolve_resources(**{**base, **override})


# ---------- identifiers and remote path scoping ----------

def test_session_prefix_derives_from_host_alias():
    assert _cfg(host="hydra").session_prefix == "hydra"
    assert _cfg(host="TU-Hydra.login").session_prefix == "tuhydralogin"


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


def test_launcher_script_contract():
    script = _build_detached_launch("uv sync", "cmd_x")
    # `.start` is written at launch (list_commands' started_at; the `.pid` mtime is the heartbeat, not the launch time).
    assert "cmd_x.start" in script
    # `.exit` is written before the detached venv-snapshot epilogue, so the hibernation never delays the agent-visible result.
    assert "setsid" in script and "-x /cs-helpers/venv-snapshot.sh" in script
    assert script.index('echo "$ec" >') < script.index("venv-snapshot.sh")


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
        session_id="hydra_1", project_id="p", project_path="/tmp/p", created_at="2026-01-01T00:00:00Z",
        resources={"partition": "gpu-2h", "gpus": 1, "gpu_type": None, "mem": 42},
        idle_timeout_minutes=20, image="images/default.sif",
        status=SessionStatus.ACTIVE, job_id="123", sshd_port=2222, gpu="NVIDIA A100 80GB PCIe",
    )
    assert SessionInfo.from_json(info.to_json()) == info
    agent = info.to_agent_json()
    assert agent["status"] == "active" and agent["gpu"] == "NVIDIA A100 80GB PCIe"
    # Resources are flattened into the row; null resource fields are omitted.
    assert agent["partition"] == "gpu-2h" and agent["mem"] == 42
    assert not {"job_id", "sshd_port", "project_id", "image", "resources", "gpu_type"} & agent.keys()
    # Null derived fields are omitted too (no idle heartbeat yet).
    assert "seconds_until_idle_deactivate" not in agent


def test_active_session_reports_idle_countdown():
    now = datetime.now(timezone.utc)
    info = SessionInfo(
        session_id="hydra_1", project_id="p", project_path="/p", created_at=_iso(now),
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
host = "hydra"
image = "images/default.sif"
partitions = ["gpu-2h", "cpu-2h", "gpu-test"]
default_partition = "gpu-2h"
gpu_types = ["h100", "80gb"]
default_mem = 64
exclude_nodes = ["head025"]
max_sessions_per_repo = 5
"""


def test_config_parses(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(_GOOD_CONFIG)
    cfg = load_config(p)
    assert cfg.host == "hydra" and cfg.default_partition == "gpu-2h" and cfg.default_mem == 64
    assert cfg.exclude_nodes == ("head025",) and cfg.max_sessions_per_repo == 5
    # Defaults apply for the optional keys.
    assert cfg.remote_base == "~/.compute-sessions" and cfg.login_host == "" and cfg.container_user == ""


def test_config_env_var_selects_path(tmp_path, monkeypatch):
    p = tmp_path / "elsewhere.toml"
    p.write_text(_GOOD_CONFIG)
    monkeypatch.setenv("COMPUTE_SESSIONS_CONFIG", str(p))
    assert load_config().host == "hydra"


@pytest.mark.parametrize("mutate", [
    lambda s: s.replace('host = "hydra"\n', ""),                                 # no host
    lambda s: s.replace('partitions = ["gpu-2h", "cpu-2h", "gpu-test"]\n', ""),  # no partitions
    lambda s: s.replace('default_partition = "gpu-2h"', 'default_partition = "nope"'),
    lambda s: s.replace('gpu_types = ["h100", "80gb"]', 'gpu_types = ["h100; rm -rf /"]'),  # injection-shaped token
    lambda s: s.replace('default_mem = 64', 'default_mem = "lots"'),
    lambda s: "[sources.hydra]\n" + s,                                            # the pre-0.5 multi-source layout
    lambda s: s + 'description = "free but slow"\n',                             # leftover pre-0.5 key: rejected, not silently ignored
    lambda s: s.replace("default_mem", "default_memory"),                        # typo'd key
])
def test_invalid_configs_are_rejected(tmp_path, mutate):
    p = tmp_path / "config.toml"
    p.write_text(mutate(_GOOD_CONFIG))
    with pytest.raises(ComputeSessionsError):
        load_config(p)


def test_missing_config_gives_actionable_error(tmp_path):
    with pytest.raises(ComputeSessionsError, match="setup-cluster"):
        load_config(tmp_path / "nope.toml")


# ---------- incremental logs: cursor, early return, progress collapse ----------

def _write_command_files(tmp_path, session_id: str, command_id: str, out: str = "", err: str = "") -> Path:
    logdir = tmp_path / "sessions" / session_id / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / f"{command_id}.out").write_text(out)
    (logdir / f"{command_id}.err").write_text(err)
    (logdir / f"{command_id}.pid").write_text("1")  # fresh mtime → running
    return logdir


def _write_session_record(tmp_path, session_id: str, **fields) -> None:
    info = SessionInfo(session_id=session_id, project_id="p", project_path=str(tmp_path), created_at="2026-01-01T00:00:00Z", **fields)
    sdir = tmp_path / "sessions" / session_id
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "config.json").write_text(json.dumps(info.to_json()))


def test_logs_cursor_returns_only_new_output(tmp_path):
    c = _local_cluster(tmp_path)
    logdir = _write_command_files(tmp_path, "hydra_1", "cmd_a", out="step 1\nstep 2\n")
    first = logs_with_access(c.access, "hydra_1", "cmd_a")
    assert first["status"] == "running"
    assert first["stdout"] == "step 1\nstep 2\n"
    cursor = first["cursor"]

    # No new output → empty delta, same cursor.
    again = logs_with_access(c.access, "hydra_1", "cmd_a", since=cursor)
    assert again["stdout"] == "" and again["stderr"] == ""
    assert again["cursor"] == cursor

    # New output + exit → delta only, exit recorded, no cursor on a settled command.
    with (logdir / "cmd_a.out").open("a") as fh:
        fh.write("step 3\n")
    (logdir / "cmd_a.exit").write_text("0")
    done = logs_with_access(c.access, "hydra_1", "cmd_a", since=cursor)
    assert done["status"] == "exited" and done["exit_code"] == 0
    assert done["stdout"] == "step 3\n"
    assert "cursor" not in done


def test_logs_cursor_covers_returned_bytes_under_concurrent_writes(tmp_path):
    # The cursor (wc -c) and the content read (tail) are separate steps of one remote script; the content read is capped to the measured sizes, else a burst-writing command pushes bytes past the reported cursor and the next since=cursor read re-emits them (observed as `cs run` printing uv's whole error block twice). The ±1 tolerance absorbs BSD sed's synthetic trailing newline on a mid-line cut.
    c = _local_cluster(tmp_path)
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
            res = logs_with_access(c.access, "hydra_1", "cmd_r", since=f"{oo}:0")
            assert res["status"] == "running"
            new_oo = int(res["cursor"].split(":")[0])
            assert len(res["stdout"]) <= new_oo - oo + 1
            oo = new_oo
    finally:
        writer.terminate()
        writer.wait()


def test_logs_collapses_progress_bars_and_counts_dropped_lines(tmp_path):
    c = _local_cluster(tmp_path)
    _write_command_files(
        tmp_path, "hydra_1", "cmd_b",
        out="loading: 10%\rloading: 50%\rloading: 100%\ndone\n",
        err="w1\nw2\nw3\n",
    )
    res = logs_with_access(c.access, "hydra_1", "cmd_b", max_lines=2)
    # \r-overwritten segments collapse to the final state; raw files stay untouched.
    assert res["stdout"] == "loading: 100%\ndone\n"
    assert res["stderr"] == "w2\nw3\n"
    assert res["stderr_truncated_lines"] == 1


def test_collapse_preserves_crlf_lines(tmp_path):
    # CRLF endings must survive the remote \r-collapse (the trailing \r is stripped, not the line).
    c = _local_cluster(tmp_path)
    _write_command_files(tmp_path, "hydra_1", "cmd_d", out="a\r\nb\n")
    res = logs_with_access(c.access, "hydra_1", "cmd_d")
    assert res["stdout"] == "a\nb\n"


def test_since_cursor_is_validated():
    assert _parse_since(None) == (0, 0)
    assert _parse_since("12:0") == (12, 0)
    for bad in ("12", "a:b", "-1:0", "1:2:3"):
        with pytest.raises(ComputeSessionsError):
            _parse_since(bad)


def test_wait_for_command_returns_at_once_when_there_is_something_to_report(tmp_path):
    c = _local_cluster(tmp_path)
    _write_session_record(tmp_path, "hydra_1")
    _write_command_files(tmp_path, "hydra_1", "cmd_c", out="already here\n")
    logdir = _write_command_files(tmp_path, "hydra_1", "cmd_e", out="done\n")
    (logdir / "cmd_e.exit").write_text("0")
    t0 = time.monotonic()
    # Output past the cursor, or a settled command: neither sits out the (arbitrarily long) window.
    res = wait_for_command(c, "hydra_1", "cmd_c", timeout_seconds=600, since="0:0")
    assert res["status"] == "running" and res["stdout"] == "already here\n"
    res = wait_for_command(c, "hydra_1", "cmd_e", timeout_seconds=600)
    assert res["status"] == "exited" and res["exit_code"] == 0
    assert time.monotonic() - t0 < 3


def test_read_info_distinguishes_missing_session(tmp_path):
    c = _local_cluster(tmp_path)
    _write_session_record(tmp_path, "hydra_1")
    assert c.read_info("hydra_1").session_id == "hydra_1"
    with pytest.raises(ComputeSessionsError, match="not found"):
        c.read_info("hydra_nope")


# ---------- sbatch submission ----------

class _RecordingAccess(LocalAccess):
    """Records commands instead of running them; answers like `sbatch --parsable`."""

    def __init__(self, base: Path):
        super().__init__(base)
        self.cmds: list[str] = []

    def run(self, cmd: str, *, check: bool = True, input_text: str | None = None) -> CompletedRemote:
        self.cmds.append(cmd)
        return CompletedRemote(0, "4711;cluster\n", "")


def test_submit_renders_sbatch_flags(tmp_path):
    acc = _RecordingAccess(tmp_path)
    c = Cluster(_cfg(remote_base=str(tmp_path), exclude_nodes=("head025",), login_host="hydra-internal"), acc)
    info = SessionInfo(
        session_id="hydra_1", project_id="p", project_path="/p", created_at="2026-01-01T00:00:00Z",
        resources={"partition": "gpu-2h", "gpus": 2, "gpu_type": "h100|80gb", "mem": 64},
    )
    assert c.submit(info) == "4711"
    cmd = acc.cmds[-1]
    assert cmd.startswith("sbatch --export=ALL --parsable ")
    for flag in ("--job-name=cs-hydra_1", "--partition=gpu-2h", "--exclude=head025", "--gpus-per-node=2", "'--constraint=h100|80gb'", "--mem=64G", f"--output={tmp_path}/sessions/hydra_1/logs/runner-%j.out"):
        assert flag in cmd
    assert cmd.endswith(f"{tmp_path}/runner.py --base {tmp_path} --session-id hydra_1 --login hydra-internal")
    # A CPU activation carries no GPU flags.
    info.resources = {"partition": "cpu-2h", "gpus": 0, "gpu_type": None, "mem": 8}
    c.submit(info)
    assert "--gpus-per-node" not in acc.cmds[-1] and "--constraint" not in acc.cmds[-1] and "--mem=8G" in acc.cmds[-1]


# ---------- clone: server-side workdir copy ----------

def test_copy_workdir_copies_contents_and_manifest(tmp_path):
    c = _local_cluster(tmp_path)
    src_wd = tmp_path / "sessions" / "hydra_1" / "workdir"
    (src_wd / "data").mkdir(parents=True)
    (src_wd / "data" / "weights.bin").write_text("blob")
    (src_wd / "train.py").write_text("code")
    (tmp_path / "sessions" / "hydra_1" / ".sync-manifest").write_text("train.py\n")
    (tmp_path / "sessions" / "hydra_2" / "workdir").mkdir(parents=True)

    c.copy_workdir("hydra_1", "hydra_2")
    dst = tmp_path / "sessions" / "hydra_2"
    assert (dst / "workdir" / "data" / "weights.bin").read_text() == "blob"
    assert (dst / "workdir" / "train.py").read_text() == "code"
    assert (dst / ".sync-manifest").read_text() == "train.py\n"


# ---------- upload/download destination semantics ----------

class _TransferAccess(LocalAccess):
    """LocalAccess plus local-filesystem rsync, so download/upload run end-to-end against tmp dirs."""

    def rsync_from(self, remote_path, local_path, *, contents_only=False):
        src = f"{remote_path}/" if contents_only else str(remote_path)
        dst = f"{local_path}/" if contents_only else str(local_path)
        subprocess.run(["rsync", "-a", src, dst], check=True)

    rsync_to = rsync_from


def _transfer_cluster(tmp_path) -> Cluster:
    c = Cluster(_cfg(remote_base=str(tmp_path)), _TransferAccess(tmp_path))
    proj = tmp_path / "proj"
    proj.mkdir()
    (tmp_path / "sessions" / "hydra_1" / "workdir").mkdir(parents=True)
    info = SessionInfo(
        session_id="hydra_1", project_id="p", project_path=str(proj),
        created_at="2026-01-01T00:00:00Z", status=SessionStatus.INACTIVE,
    )
    c.write_info(info)
    return c


def test_download_dest_conventions(tmp_path, monkeypatch):
    from compute_sessions.session import download
    c = _transfer_cluster(tmp_path)
    proj = tmp_path / "proj"
    monkeypatch.chdir(proj)
    wd = tmp_path / "sessions" / "hydra_1" / "workdir"
    (wd / "runs").mkdir()
    (wd / "runs" / "log.csv").write_text("l")
    (wd / "runs" / "result.json").write_text("r")

    # A destination directory that doesn't exist is an error, not an implicit mkdir -p (typo'd-path regression).
    with pytest.raises(ComputeSessionsError, match="not an existing directory"):
        download(c, "hydra_1", "runs/log.csv", "runs/in_baseline/")
    with pytest.raises(ComputeSessionsError, match="parent directory"):
        download(c, "hydra_1", "runs/log.csv", "runs/in_baseline/log.csv")

    # Trailing-slash dest: files land INSIDE (three-files-into-one-file regression).
    (proj / "runs" / "in_baseline").mkdir(parents=True)
    for f in ("log.csv", "result.json"):
        out = download(c, "hydra_1", f"runs/{f}", "runs/in_baseline/")
        assert out["local_path"] == f"runs/in_baseline/{f}"
    assert (proj / "runs" / "in_baseline" / "log.csv").read_text() == "l"
    assert (proj / "runs" / "in_baseline" / "result.json").read_text() == "r"

    # A directory is placed AS a directory inside an existing dest — never merged over its contents (tracked-files-clobber regression).
    (proj / "results").mkdir()
    (proj / "results" / "rq1.json").write_text("keep")
    out = download(c, "hydra_1", "runs", "results/")
    assert out["local_path"] == "results/runs/"
    assert (proj / "results" / "rq1.json").read_text() == "keep"
    assert (proj / "results" / "runs" / "log.csv").exists()

    # Explicit contents-merge via source trailing slash; repeat to the same named dir stays idempotent (no results/runs/runs).
    download(c, "hydra_1", "runs/", "flat")
    assert (proj / "flat" / "log.csv").exists()
    download(c, "hydra_1", "runs", "results/runs")
    assert not (proj / "results" / "runs" / "runs").exists()

    # A file into an existing dir goes inside (cp semantics); a dir refuses a dest that exists as a file.
    out = download(c, "hydra_1", "runs/log.csv", "results/runs")
    assert out["local_path"] == "results/runs/log.csv"
    (proj / "clash").write_text("f")
    with pytest.raises(ComputeSessionsError, match="exists as a file"):
        download(c, "hydra_1", "runs", "clash")


def test_upload_dest_conventions(tmp_path, monkeypatch):
    from compute_sessions.session import upload
    c = _transfer_cluster(tmp_path)
    proj = tmp_path / "proj"
    monkeypatch.chdir(proj)
    wd = tmp_path / "sessions" / "hydra_1" / "workdir"
    (proj / "ckpt.pt").write_text("w")
    (proj / "adapter").mkdir()
    (proj / "adapter" / "config.json").write_text("c")

    # A destination directory that doesn't exist is an error, not an implicit mkdir -p (typo'd-path regression).
    with pytest.raises(ComputeSessionsError, match="not an existing directory"):
        upload(c, "hydra_1", "ckpt.pt", "outputs/geometry/")
    with pytest.raises(ComputeSessionsError, match="parent directory"):
        upload(c, "hydra_1", "ckpt.pt", "outputs/geometry/ckpt.pt")

    # Trailing-slash remote: file lands INSIDE; a second file to the same dir doesn't clobber the first (file-named-like-the-dir regression).
    (wd / "outputs" / "geometry").mkdir(parents=True)
    out = upload(c, "hydra_1", "ckpt.pt", "outputs/geometry/")
    assert out["remote_path"] == "outputs/geometry/ckpt.pt"
    (proj / "eval.jsonl").write_text("e")
    upload(c, "hydra_1", "eval.jsonl", "outputs/geometry/")
    assert (wd / "outputs" / "geometry" / "ckpt.pt").read_text() == "w"
    assert (wd / "outputs" / "geometry" / "eval.jsonl").read_text() == "e"

    # Directory into an existing dir: placed as a directory (adapter-spilled-one-level-up regression); repeat upload to its own name stays idempotent.
    out = upload(c, "hydra_1", "adapter", "outputs/geometry/")
    assert out["remote_path"] == "outputs/geometry/adapter/"
    assert (wd / "outputs" / "geometry" / "adapter" / "config.json").exists()
    upload(c, "hydra_1", "adapter", "outputs/geometry/adapter")
    assert not (wd / "outputs" / "geometry" / "adapter" / "adapter").exists()

    # No remote → basename; existing remote FILE refuses a directory source.
    out = upload(c, "hydra_1", "adapter", "")
    assert out["remote_path"] == "adapter/"
    upload(c, "hydra_1", "ckpt.pt", "clash")
    with pytest.raises(ComputeSessionsError, match="exists as a file"):
        upload(c, "hydra_1", "adapter", "clash")


# ---------- runner ----------

def _load_runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cs_runner", Path(__file__).resolve().parent.parent / "remote" / "runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def test_runner_detects_gpu_models(monkeypatch):
    runner = _load_runner()
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


# ---------- cs CLI: session resolution, arg splitting, formatting ----------

def _cli_info(sid: str, project: str, status: SessionStatus) -> SessionInfo:
    return SessionInfo(session_id=sid, project_id=project, project_path="/tmp/x", created_at="2026-01-01T00:00:00Z", status=status)


class _FakeCluster:
    cfg = _cfg(host="hy")

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
        _cli_info("old_ccc999", "p1", SessionStatus.INACTIVE),   # created under a former host alias / id prefix
    ]
    monkeypatch.setattr(cli, "_CLUSTER", _FakeCluster(infos))
    assert cli._resolve_session("hy_aaa111") == "hy_aaa111"       # exact id: fast path
    assert cli._resolve_session(" HY_A ") == "hy_aaa111"          # unique prefix; copy-pasted ids arrive with whitespace and uppercase hex
    assert cli._resolve_session("aaa1") == "hy_aaa111"            # bare hex tail prefix
    assert cli._resolve_session("old_ccc") == "old_ccc999"        # older prefix still resolves via the listing
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
    monkeypatch.setattr(cli, "_CLUSTER", _FakeCluster(dead))
    monkeypatch.setattr(cli, "derive_project_id", lambda p: "p3")
    with pytest.raises(ComputeSessionsError, match=r"none is live.*2 more via"):
        cli._resolve_session(None)


def test_cli_leading_session_token_split(monkeypatch):
    from compute_sessions import cli
    monkeypatch.setattr(cli, "_CLUSTER", _FakeCluster([_cli_info("hy_aaa111", "p1", SessionStatus.ACTIVE), _cli_info("old_ccc999", "p1", SessionStatus.INACTIVE)]))
    # `<prefix>_…` up front is a session token; a bare hex token or an older-prefix id claims the slot only when a session matches; anything else belongs to the command/path args.
    assert cli._split_leading_session(("hy_abc", "echo", "hi")) == ("hy_abc", ["echo", "hi"])
    assert cli._split_leading_session(("aaa1", "echo", "hi")) == ("aaa1", ["echo", "hi"])
    assert cli._split_leading_session(("beef", "echo", "hi")) == (None, ["beef", "echo", "hi"])
    assert cli._split_leading_session(("old_ccc999", "echo", "hi")) == ("old_ccc999", ["echo", "hi"])
    assert cli._split_leading_session(("old_ccc9", "echo", "hi")) == ("old_ccc9", ["echo", "hi"])
    assert cli._split_leading_session(("old_dead1", "echo", "hi")) == (None, ["old_dead1", "echo", "hi"])
    assert cli._split_leading_session(("echo", "hy_abc")) == (None, ["echo", "hy_abc"])
    assert cli._split_leading_session(("other_abc", "x")) == (None, ["other_abc", "x"])
    assert cli._split_leading_session(("date", "+%s")) == (None, ["date", "+%s"])
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


def test_cli_transfer_args_split():
    from compute_sessions.cli import _split_sources_dest
    assert _split_sources_dest(["a"], "u") == (["a"], None)
    assert _split_sources_dest(["a", "b"], "u") == (["a"], "b")
    assert _split_sources_dest(["a", "b", "dir/"], "u") == (["a", "b"], "dir/")
    with pytest.raises(SystemExit):
        _split_sources_dest(["a", "b", "c"], "u")


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
    assert runner.invoke(cli.cli, ["--version"]).output.startswith("cs, version ")
    # A broken config must not break --help, but must fail commands with the tool-failure code.
    monkeypatch.setattr(cli, "_CLUSTER", None)
    monkeypatch.setattr(cli, "_CFG_ERROR", RuntimeError("boom"))
    res = runner.invoke(cli.cli, ["list"])
    assert res.exit_code == cli.EXIT_TOOL_FAILURE


# ---------- create/activate/deactivate: no zombie records ----------

def _session_records(tmp_path) -> list[SessionInfo]:
    root = tmp_path / "sessions"
    return [SessionInfo.from_json(json.loads(p.read_text())) for p in sorted(root.glob("*/config.json"))] if root.exists() else []


def _project_dir(tmp_path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    (proj / "train.py").write_text("print(1)")
    return proj


def test_create_validates_the_request_before_writing_anything(tmp_path, monkeypatch):
    from compute_sessions import session as sess
    c = _local_cluster(tmp_path)
    monkeypatch.setattr(sess, "sync_session", lambda *a, **k: pytest.fail("sync must not run for an invalid request"))
    with pytest.raises(ComputeSessionsError, match="gpu_type"):
        sess.create(c, sess.ResourceRequest(gpu_type="h200"), _project_dir(tmp_path))
    with pytest.raises(ComputeSessionsError, match="idle_timeout"):
        sess.create(c, sess.ResourceRequest(idle_timeout_minutes=0), _project_dir(tmp_path))
    assert _session_records(tmp_path) == []


@pytest.mark.parametrize("step, failure", [("sync", RemoteError("rsync failed")), ("sync", KeyboardInterrupt()), ("submit", RemoteError("sbatch: error"))])
def test_create_that_does_not_complete_leaves_a_failed_record_not_a_pending_one(tmp_path, monkeypatch, step, failure):
    from compute_sessions import session as sess
    c = _local_cluster(tmp_path)
    def boom(*a, **k):
        raise failure
    monkeypatch.setattr(sess, "sync_session", boom if step == "sync" else lambda *a, **k: None)
    if step == "submit":
        monkeypatch.setattr(Cluster, "submit", boom)
    with pytest.raises(type(failure)):
        sess.create(c, sess.ResourceRequest(), _project_dir(tmp_path))
    (rec,) = _session_records(tmp_path)
    assert rec.status == SessionStatus.FAILED and rec.job_id is None and rec.last_deactivated_at


def test_activate_submit_failure_restores_the_previous_status(tmp_path, monkeypatch):
    from compute_sessions import session as sess
    c = _local_cluster(tmp_path)
    _write_session_record(tmp_path, "hydra_1", status=SessionStatus.INACTIVE)
    monkeypatch.setattr(Cluster, "submit", lambda self, info: (_ for _ in ()).throw(RemoteError("sbatch down")))
    with pytest.raises(RemoteError):
        sess.activate(c, "hydra_1", sess.ResourceRequest())
    assert c.read_info("hydra_1").status == SessionStatus.INACTIVE
    # The happy path still ends PENDING with the job recorded.
    monkeypatch.setattr(Cluster, "submit", lambda self, info: "4711")
    info = sess.activate(c, "hydra_1", sess.ResourceRequest())
    assert (info.status, info.job_id) == (SessionStatus.PENDING, "4711")


def test_deactivate_clears_a_jobless_pending_record(tmp_path):
    from compute_sessions import session as sess
    c = _local_cluster(tmp_path)
    _write_session_record(tmp_path, "hydra_zombie")  # status pending, no job_id — a create that died before submitting
    assert c.read_info("hydra_zombie").status == SessionStatus.PENDING
    out = sess.deactivate(c, "hydra_zombie")
    assert out.status == SessionStatus.INACTIVE and out.last_deactivated_at
    assert c.read_info("hydra_zombie").status == SessionStatus.INACTIVE


# ---------- rsync: transient failures resume instead of restarting ----------

def _rsync_access(monkeypatch, returncodes: list[int]):
    from compute_sessions import ssh as ssh_mod
    acc = ssh_mod.HostAccess(_cfg())
    monkeypatch.setattr(acc, "open", lambda: None)
    calls: list[list[str]] = []
    sleeps: list[int] = []
    def fake_run(args, **kwargs):
        calls.append(args)
        rc = returncodes[len(calls) - 1]
        return subprocess.CompletedProcess(args, rc, stdout="", stderr="" if rc == 0 else f"rsync error: something (code {rc})\n")
    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(ssh_mod.time, "sleep", sleeps.append)
    return acc, calls, sleeps


@pytest.mark.parametrize("returncodes, pauses, final_error", [
    ([30, 12, 0], [10, 30], None),   # transient codes: paused and resumed until it passes
    ([23], [], 23),                  # a real error is not retried
    ([30, 30, 30], [10, 30], 30),    # three strikes
])
def test_rsync_retries_only_transient_failures(monkeypatch, returncodes, pauses, final_error):
    acc, calls, sleeps = _rsync_access(monkeypatch, returncodes)
    if final_error is None:
        acc._rsync(["src", "host:dst"])
        assert "--partial-dir=.rsync-partial" in calls[0] and calls[0][-2:] == ["src", "host:dst"]
    else:
        with pytest.raises(RemoteError) as exc:
            acc._rsync(["src", "host:dst"])
        assert exc.value.exit_code == final_error
    assert len(calls) == len(returncodes) and sleeps == pauses


# ---------- listing, scheduler state, record races, symlink escapes, sync ordering ----------

def test_list_infos_survives_a_session_dir_without_config(tmp_path):
    # A `for` loop exits with its last iteration's status: a config-less session dir sorting last must not fail the whole listing.
    c = _local_cluster(tmp_path)
    _write_session_record(tmp_path, "hydra_a")
    _write_session_record(tmp_path, "hydra_b")
    (tmp_path / "sessions" / "hydra_zz_stray" / "workdir").mkdir(parents=True)
    assert sorted(i.session_id for i in c.list_infos()) == ["hydra_a", "hydra_b"]


def _fake_slurm_bin(tmp_path, monkeypatch) -> None:
    """Puts fake `squeue`/`timeout` on PATH; behaviour set per case via SQUEUE_STDOUT / SQUEUE_STDERR / SQUEUE_RC / TIMEOUT_RC."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "squeue").write_text('#!/bin/bash\n[ -n "$SQUEUE_STDERR" ] && printf "%s\\n" "$SQUEUE_STDERR" >&2\n[ -n "$SQUEUE_STDOUT" ] && printf "%s\\n" "$SQUEUE_STDOUT"\nexit "${SQUEUE_RC:-0}"\n')
    (bin_dir / "timeout").write_text('#!/bin/bash\nshift\nif [ -n "$TIMEOUT_RC" ]; then exit "$TIMEOUT_RC"; fi\nexec "$@"\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    for var in ("SQUEUE_STDOUT", "SQUEUE_STDERR", "SQUEUE_RC", "TIMEOUT_RC"):
        monkeypatch.delenv(var, raising=False)


def test_job_state_distinguishes_gone_from_scheduler_failure(tmp_path, monkeypatch):
    _fake_slurm_bin(tmp_path, monkeypatch)
    c = _local_cluster(tmp_path)
    info = SessionInfo(session_id="hydra_1", project_id="p", project_path="/p", created_at="2026-01-01T00:00:00Z", job_id="4711")

    monkeypatch.setenv("SQUEUE_STDOUT", "RUNNING|1:59:00|head042")
    assert c.job_state(info) == {"state": "running", "reason": None, "node": "head042", "time_left": "1:59:00"}
    monkeypatch.setenv("SQUEUE_STDOUT", "PENDING|2:00:00|Priority")
    assert c.job_state(info)["state"] == "pending" and c.job_state(info)["reason"] == "Priority"
    # Terminal states hold no allocation → gone (with the state kept as the reason); an empty result → gone.
    monkeypatch.setenv("SQUEUE_STDOUT", "COMPLETING|0:00|head042")
    assert c.job_state(info) == {"state": "gone", "reason": "COMPLETING: head042", "node": None}
    monkeypatch.setenv("SQUEUE_STDOUT", "")
    assert c.job_state(info)["state"] == "gone"
    # A purged job id is the one non-zero exit that means gone …
    monkeypatch.setenv("SQUEUE_STDERR", "slurm_load_jobs error: Invalid job id specified")
    monkeypatch.setenv("SQUEUE_RC", "1")
    assert c.job_state(info)["state"] == "gone"
    # … an unreachable controller is not: that used to read as an empty (gone) result and mark live sessions failed/inactive.
    monkeypatch.setenv("SQUEUE_STDERR", "slurm_load_jobs error: Unable to contact slurm controller (connect failure)")
    st = c.job_state(info)
    assert st["state"] == "unknown" and "Unable to contact slurm controller" in st["reason"]
    monkeypatch.setenv("SQUEUE_STDERR", "")
    monkeypatch.setenv("SQUEUE_RC", "0")
    monkeypatch.setenv("TIMEOUT_RC", "124")
    assert c.job_state(info) == {"state": "unknown", "reason": "scheduler query timed out", "node": None}
    # Allocation-holding non-running states stay "pending" so activate() refuses to double-submit.
    monkeypatch.delenv("TIMEOUT_RC")
    monkeypatch.setenv("SQUEUE_STDOUT", "SUSPENDED|1:00:00|head042")
    assert c.job_state(info)["state"] == "pending"


def test_patch_info_is_a_compare_and_set(tmp_path):
    c = _local_cluster(tmp_path)
    _write_session_record(tmp_path, "hydra_1")
    assert c.patch_info("hydra_1", {"job_id": "1"}) is True
    assert c.read_info("hydra_1").job_id == "1"
    # Only the named fields change; a stale expectation writes nothing.
    assert c.patch_info("hydra_1", {"status": "inactive"}, expect={"job_id": "1"}) is True
    assert c.patch_info("hydra_1", {"status": "active"}, expect={"job_id": "2"}) is False
    rec = c.read_info("hydra_1")
    assert (rec.status, rec.job_id, rec.project_path) == (SessionStatus.INACTIVE, "1", str(tmp_path))


def test_activate_clears_the_stale_job_id_and_does_not_clobber_a_fast_runner(tmp_path, monkeypatch):
    from compute_sessions import session as sess
    c = _local_cluster(tmp_path)
    _write_session_record(tmp_path, "hydra_1", status=SessionStatus.INACTIVE, job_id="1", node="old-node", sshd_port=1)
    monkeypatch.setattr(Cluster, "job_state", lambda self, info: {"state": "gone", "reason": None, "node": None})

    def fake_submit(self, info):
        rec = self.read_info("hydra_1")
        # The pending record written before sbatch must not carry the previous job — show() would find it gone and flip the fresh session to failed.
        assert (rec.status, rec.job_id, rec.node) == (SessionStatus.PENDING, None, None)
        # The runner beats us to the post-submit write: a fast allocation flips the record to active.
        self.patch_info("hydra_1", {"status": "active", "job_id": "4711", "node": "head042", "sshd_port": 2222})
        return "4711"
    monkeypatch.setattr(Cluster, "submit", fake_submit)
    out = sess.activate(c, "hydra_1", sess.ResourceRequest())
    assert out.job_id == "4711"
    rec = c.read_info("hydra_1")
    # Previously a full overwrite of the pre-submit object here erased the runner's state and left the session reading `pending` while it ran.
    assert (rec.status, rec.job_id, rec.node, rec.sshd_port) == (SessionStatus.ACTIVE, "4711", "head042", 2222)


def test_activate_refuses_when_the_previous_job_state_is_unknown(tmp_path, monkeypatch):
    from compute_sessions import session as sess
    c = _local_cluster(tmp_path)
    _write_session_record(tmp_path, "hydra_1", status=SessionStatus.ACTIVE, job_id="1")
    monkeypatch.setattr(Cluster, "job_state", lambda self, info: {"state": "unknown", "reason": "scheduler query failed: connect failure", "node": None})
    monkeypatch.setattr(Cluster, "submit", lambda self, info: pytest.fail("must not submit on top of a possibly-live job"))
    with pytest.raises(ComputeSessionsError, match="cannot tell whether job 1"):
        sess.activate(c, "hydra_1", sess.ResourceRequest())
    assert c.read_info("hydra_1").status == SessionStatus.ACTIVE


def test_show_reconciles_a_gone_job_with_compare_and_set(tmp_path, monkeypatch):
    from compute_sessions import session as sess
    c = _local_cluster(tmp_path)
    _write_session_record(tmp_path, "hydra_1", status=SessionStatus.ACTIVE, job_id="1", last_activated_at="2026-01-01T00:00:00Z")
    monkeypatch.setattr(Cluster, "job_state", lambda self, info: {"state": "gone", "reason": None, "node": None})
    out = sess.show(c, "hydra_1")
    assert out["info"]["status"] == "inactive" and out["info"]["last_deactivated_at"]  # the anchor for IN-STATUS used to stay unset here
    assert c.read_info("hydra_1").status == SessionStatus.INACTIVE

    # A re-activation that lands while show() is looking the old job up owns the record: the reconciliation must not overwrite it.
    _write_session_record(tmp_path, "hydra_1", status=SessionStatus.ACTIVE, job_id="1")
    def gone_but_reactivated(self, info):
        self.patch_info("hydra_1", {"status": "pending", "job_id": "2"})
        return {"state": "gone", "reason": None, "node": None}
    monkeypatch.setattr(Cluster, "job_state", gone_but_reactivated)
    out = sess.show(c, "hydra_1")
    rec = c.read_info("hydra_1")
    assert (rec.status, rec.job_id) == (SessionStatus.PENDING, "2") and out["info"]["status"] == "pending"


def test_workdir_tools_refuse_symlink_escapes(tmp_path, monkeypatch):
    # `[ -d ]` and rsync follow a symlinked directory component and --safe-links only guards links inside the transfer — a link planted in the workdir from inside the container used to turn download into an arbitrary read and upload into an arbitrary write of the cluster user's files.
    from compute_sessions.session import download, ls, upload
    c = _transfer_cluster(tmp_path)
    proj = tmp_path / "proj"
    monkeypatch.chdir(proj)
    wd = tmp_path / "sessions" / "hydra_1" / "workdir"
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "id_rsa").write_text("PRIVATE")
    (wd / "evil").symlink_to(secret)
    (wd / "runs").mkdir()
    (wd / "runs" / "log.csv").write_text("l")
    (wd / "latest").symlink_to(wd / "runs")   # a link that stays inside the workdir is fine
    (wd / "okdir").mkdir()
    (wd / "okdir" / "adapter").symlink_to(secret)
    (proj / "out").mkdir()
    (proj / "ak").write_text("attacker key")
    (proj / "adapter").mkdir()
    (proj / "adapter" / "config.json").write_text("c")

    for rel in ("evil/", "evil/id_rsa", "evil"):
        with pytest.raises(ComputeSessionsError, match="outside the session workdir"):
            download(c, "hydra_1", rel, "out/")
    assert list((proj / "out").iterdir()) == []
    for local, rel in (("ak", "evil/authorized_keys"), ("ak", "evil/"), ("adapter", "okdir/")):
        with pytest.raises(ComputeSessionsError, match="outside the session workdir"):
            upload(c, "hydra_1", local, rel)
    assert sorted(p.name for p in secret.iterdir()) == ["id_rsa"]
    with pytest.raises(ComputeSessionsError, match="outside the session workdir"):
        ls(c, "hydra_1", "evil")

    out = download(c, "hydra_1", "latest/", "out/")
    assert (proj / "out" / "log.csv").read_text() == "l" and out["remote_path"] == "latest"
    assert "log.csv" in ls(c, "hydra_1", "latest")
    upload(c, "hydra_1", "ak", "latest/")
    assert (wd / "runs" / "ak").read_text() == "attacker key"


class _SyncAccess(LocalAccess):
    """LocalAccess plus a local-filesystem rsync_files, so sync_session runs end-to-end against tmp dirs."""

    def rsync_files(self, local_root, rel_files, remote_dir, *, follow_symlinks=False):
        files = list(rel_files)
        proc = subprocess.run(
            ["rsync", "-a", "--itemize-changes", "--stats", "--files-from=-", f"{local_root}/", f"{remote_dir}/"],
            input="\n".join(files) + "\n", capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RemoteError("rsync failed", stderr=proc.stderr, exit_code=proc.returncode)
        return proc.stdout + proc.stderr


def test_sync_unwinds_type_changes_before_the_transfer(tmp_path):
    # rsync refuses to write a file over a non-empty directory; with the manifest deletions running after the transfer, a local `foo/` → file `foo` change failed every sync until the workdir was fixed by hand.
    from compute_sessions.sync import sync_session
    acc = _SyncAccess(tmp_path)
    proj = tmp_path / "proj"
    (proj / "foo").mkdir(parents=True)
    (proj / "foo" / "bar.txt").write_text("in dir")
    wd = tmp_path / "sessions" / "hydra_1" / "workdir"
    res = sync_session(acc, "hydra_1", proj)
    assert res.added == ["foo/bar.txt"] and (wd / "foo" / "bar.txt").read_text() == "in dir"

    import shutil
    shutil.rmtree(proj / "foo")
    (proj / "foo").write_text("now a file")
    res = sync_session(acc, "hydra_1", proj)
    assert res.deleted_outdated == ["foo/bar.txt"] and res.cleanup_errors == []
    assert (wd / "foo").is_file() and (wd / "foo").read_text() == "now a file"

    (proj / "foo").unlink()
    (proj / "foo").mkdir()
    (proj / "foo" / "bar.txt").write_text("dir again")
    res = sync_session(acc, "hydra_1", proj)
    assert res.deleted_outdated == ["foo"] and (wd / "foo" / "bar.txt").read_text() == "dir again"


def test_runner_final_status_write_is_a_compare_and_set(tmp_path):
    runner = _load_runner()
    sess_dir = tmp_path / "sessions" / "s1"
    sess_dir.mkdir(parents=True)
    s = runner.Session(tmp_path, "s1")
    own = {"job_id": ("", None, "1")}
    s.config_path.write_text(json.dumps({"status": "active", "job_id": "1"}))
    assert s.config_update(own, status="inactive") is True and s.config()["status"] == "inactive"
    s.config_path.write_text(json.dumps({"status": "pending"}))   # job_id not stamped yet (very early failure)
    assert s.config_update(own, status="failed") is True and s.config()["status"] == "failed"
    # Re-activated meanwhile (a newer job owns the record): the teardown of job 1 must leave it alone.
    s.config_path.write_text(json.dumps({"status": "pending", "job_id": "2"}))
    assert s.config_update(own, status="inactive") is False and s.config() == {"status": "pending", "job_id": "2"}


def test_ssh_subprocesses_do_not_drain_inherited_stdin():
    # ssh forwards whatever stdin it gets, so `… | while read s; do cs deactivate "$s"; done` used to act on the first line only.
    from compute_sessions.ssh import _run_capture
    r, w = os.pipe()
    os.write(w, b"next-session-id\n")
    os.close(w)
    saved = os.dup(0)
    os.dup2(r, 0)
    try:
        assert _run_capture(["cat"], timeout=5).stdout == ""
        assert _run_capture(["cat"], timeout=5, input_text="body").stdout == "body"
        assert os.read(0, 100) == b"next-session-id\n"
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        os.close(r)


def test_cli_follow_rounds_are_spaced(monkeypatch):
    from compute_sessions import cli
    rounds = iter([
        {"status": "running", "stdout": "a\n", "stderr": "", "cursor": "2:0"},
        {"status": "running", "stdout": "b\n", "stderr": "", "cursor": "4:0"},
        {"status": "exited", "exit_code": 0, "stdout": "c\n", "stderr": ""},
    ])
    monkeypatch.setattr(cli, "_CLUSTER", object())
    monkeypatch.setattr(cli.sess, "wait_for_command", lambda *a, **k: next(rounds))
    sleeps: list[float] = []
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)
    assert cli._follow("hy_1", "cmd_1", "0:0", emit=lambda r: None) == ("exited", 0)
    # A chatty command makes every round return at once; without spacing that was a tight loop of login-node shell invocations.
    assert len(sleeps) == 3 and sleeps[0] == 0.0 and all(s > 0.9 for s in sleeps[1:])


def test_cli_logs_tail_follow_caps_the_whole_stream(monkeypatch):
    from click.testing import CliRunner
    from compute_sessions import cli
    monkeypatch.setattr(cli, "_CLUSTER", _FakeCluster([_cli_info("hy_aaa111", "p1", SessionStatus.ACTIVE)]))
    monkeypatch.setattr(cli, "derive_project_id", lambda p: "p1")
    monkeypatch.setattr(cli.sess, "logs", lambda *a, **k: {"status": "running", "stdout": "a\nb\n", "stderr": "", "cursor": "4:0"})
    monkeypatch.setattr(cli.sess, "wait_for_command", lambda *a, **k: {"status": "exited", "exit_code": 3, "stdout": "c\nd\ne\n", "stderr": "w1\nw2\nw3\n"})
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    res = CliRunner().invoke(cli.cli, ["logs", "-f", "--tail", "2", "hy_aaa111", "cmd_0123456789ab"])
    # The cap used to apply to the first read only; the follow streamed everything after it.
    assert res.exit_code == 3 and res.stdout == "d\ne\n"
