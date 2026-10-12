"""Behavioral tests — no cluster or network required. The cluster side runs for real but locally: the login-node helper as a `python3 -I` subprocess against a temp base dir (with fake sbatch/squeue/scancel on PATH), rsync between local dirs, and the "container" as a local bash running the real launcher/follower script. Run with `uv run pytest`."""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from compute_sessions import cli
from compute_sessions import run as run_mod
from compute_sessions import session as sess
from compute_sessions import ssh as ssh_mod
from compute_sessions.cluster import Cluster
from compute_sessions.config import Config, load_config
from compute_sessions.errors import ComputeSessionsError, RemoteError, SessionError, UsageError
from compute_sessions.models import SessionInfo, SessionStatus, fmt_duration
from compute_sessions.project import project_id
from compute_sessions.run import Frames, Output, Target, collapse_cr, container_script, validate_signal
from compute_sessions.session import _resolve_local_inside_project
from compute_sessions.ssh import HostAccess
from compute_sessions.sync import plan_sync

_real_sleep = time.sleep


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _cfg(**overrides) -> Config:
    base = dict(
        host="hydra", image="images/default.sif",
        partitions=("cpu-2h", "gpu-2h", "gpu-test"), default_partition="gpu-2h",
        gpu_types=("h100", "80gb", "40gb"), default_mem=42,
    )
    return Config(**{**base, **overrides})


# ---------- local stand-in for the cluster ----------

class LocalAccess(HostAccess):
    """HostAccess with only the transport swapped for local processes — requests, replies, the helper, rsync and the follower protocol are the real code."""

    def __init__(self, cfg: Config, home: Path):
        super().__init__(cfg)
        self.home = home
        self.container_calls = 0

    def open(self) -> None:
        pass

    def _helper_argv(self, nsrc: int) -> list[str]:
        return ["env", f"HOME={self.home}", os.environ.get("CS_TEST_HELPER_PYTHON", "python3"), "-I", "-c", ssh_mod._BOOT, str(nsrc)]

    def _remote(self, path: str) -> str:
        return path

    def _rsync_transport(self) -> list[str]:
        return []

    def container_argv(self, session_id: str, socket: str, remote_cmd: str) -> list[str]:
        self.container_calls += 1
        return ["env", f"HOME={self.home}", "bash", "-c", remote_cmd]

    def close_container(self, session_id: str) -> None:
        pass


_FAKE_SLURM = {
    # Records argv; SBATCH_HOOK runs first (e.g. a fast runner flipping the record), SBATCH_SLEEP delays the answer.
    "sbatch": 'echo "$@" >> "$FAKE_SLURM/sbatch.log"\n[ -n "$SBATCH_HOOK" ] && eval "$SBATCH_HOOK"\nsleep "${SBATCH_SLEEP:-0}"\n[ -n "$SBATCH_STDERR" ] && echo "$SBATCH_STDERR" >&2\necho "${SBATCH_OUT:-4711;cluster}"\nexit "${SBATCH_RC:-0}"\n',
    # By-name lookups (`-n cs-<sid>`) see the job once sbatch ran, unless SQUEUE_BYNAME says otherwise; job-id queries answer SQUEUE_STDOUT.
    "squeue": '[ -n "$SQUEUE_HOOK" ] && eval "$SQUEUE_HOOK"\n[ -n "$SQUEUE_STDERR" ] && printf "%s\\n" "$SQUEUE_STDERR" >&2\nif [[ " $* " == *" -n "* ]]; then\n  if [ -n "${SQUEUE_BYNAME+x}" ]; then printf "%s" "$SQUEUE_BYNAME"; elif [ -s "$FAKE_SLURM/sbatch.log" ]; then echo "4711|PENDING"; fi\nelse\n  [ -n "$SQUEUE_STDOUT" ] && printf "%s\\n" "$SQUEUE_STDOUT"\nfi\nexit "${SQUEUE_RC:-0}"\n',
    "scancel": 'echo "$@" >> "$FAKE_SLURM/scancel.log"\n',
    # The helper bounds scheduler calls with coreutils `timeout`, which macOS lacks; this stand-in only drops the duration.
    "timeout": 'shift\nexec "$@"\n',
}


class Env(SimpleNamespace):
    def record(self, sid: str, **fields) -> SessionInfo:
        fields.setdefault("project_id", self.pid)
        fields.setdefault("project_path", str(self.proj))
        fields.setdefault("resources", {"partition": "gpu-2h", "gpus": 1, "gpu_type": None, "mem": 42})
        info = SessionInfo(session_id=sid, created_at="2026-01-01T00:00:00Z", **fields)
        d = self.base / "sessions" / sid
        (d / "workdir").mkdir(parents=True, exist_ok=True)
        (d / "logs").mkdir(exist_ok=True)
        (d / "config.json").write_text(json.dumps(info.to_json()))
        return info

    def read(self, sid: str) -> SessionInfo:
        return SessionInfo.from_json(json.loads((self.base / "sessions" / sid / "config.json").read_text()))

    def records(self) -> list[SessionInfo]:
        return [SessionInfo.from_json(json.loads(p.read_text())) for p in sorted((self.base / "sessions").glob("*/config.json"))]

    def slurm_log(self, tool: str) -> list[str]:
        p = self.slurm / f"{tool}.log"
        return p.read_text().splitlines() if p.exists() else []

    def active(self, sid: str = "hydra_aaa111") -> Target:
        """An active session whose "container" is local: the follower and launcher work on this session's real logs dir and workdir. The scheduler reports its job as running."""
        self.record(sid, status=SessionStatus.ACTIVE, job_id="1", sshd_port=1)
        self.mp.setenv("SQUEUE_STDOUT", "RUNNING|1:00:00|head042")
        self.mp.setattr(run_mod, "_LOGS", str(self.logs_dir(sid)))
        self.mp.setattr(run_mod, "_WORKDIR", str(self.base / "sessions" / sid / "workdir"))
        return Target(sid, str(self.base / "sessions" / sid / "session.sock"))

    def logs_dir(self, sid: str = "hydra_aaa111") -> Path:
        return self.base / "sessions" / sid / "logs"


@pytest.fixture
def env(tmp_path, monkeypatch) -> Env:
    base, home, proj, slurm = (tmp_path / n for n in ("base", "home", "proj", "slurm"))
    for d in (base, home, proj, slurm):
        d.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in _FAKE_SLURM.items():
        (bin_dir / name).write_text("#!/bin/bash\n" + body)
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM", str(slurm))
    for var in ("SBATCH_HOOK", "SBATCH_SLEEP", "SBATCH_OUT", "SBATCH_RC", "SBATCH_STDERR", "SQUEUE_HOOK", "SQUEUE_STDOUT", "SQUEUE_STDERR", "SQUEUE_RC", "SQUEUE_BYNAME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(proj)
    cfg = _cfg(remote_base=str(base))
    cluster = Cluster(cfg, LocalAccess(cfg, home))
    monkeypatch.setattr(cli, "_CLUSTER", cluster)
    monkeypatch.setattr(cli, "_CFG", cfg)
    return Env(base=base, home=home, proj=proj, slurm=slurm, cluster=cluster, access=cluster.access, pid=cluster.project()[0], mp=monkeypatch)


def _fast_sleep(monkeypatch) -> None:
    """Backoff and FS-delay pauses shrink to nothing; real time still passes for the follower."""
    monkeypatch.setattr(time, "sleep", lambda s: _real_sleep(min(s, 0.02)))


def _cs(*argv: str) -> SimpleNamespace:
    """Invoke the CLI like main() does: cs errors become `cs: …` on stderr and exit 125 (2 for usage errors)."""
    from click.testing import CliRunner
    res = CliRunner().invoke(cli.cli, list(argv))
    out = SimpleNamespace(exit_code=res.exit_code, stdout=res.stdout, stderr=res.stderr)
    if isinstance(res.exception, ComputeSessionsError):
        out.exit_code = 2 if isinstance(res.exception, UsageError) else cli.EXIT_TOOL_FAILURE
        out.stderr += f"cs: {res.exception}\n"
    elif res.exception and not isinstance(res.exception, SystemExit):
        raise res.exception
    return out


# ---------- resource requests, ids, config ----------

def test_resource_requests_resolve_with_config_defaults():
    resolve = Cluster(_cfg()).resolve_resources
    assert resolve(partition="gpu-2h", gpus=2, gpu_type="h100|80gb", mem=64) == {"partition": "gpu-2h", "gpus": 2, "gpu_type": "h100|80gb", "mem": 64}
    assert resolve(partition=None, gpus=None, gpu_type=None, mem=None) == {"partition": "gpu-2h", "gpus": 1, "gpu_type": None, "mem": 42}
    assert resolve(partition="cpu-2h", gpus=4, gpu_type=None, mem=8)["gpus"] == 0  # cpu partitions never get GPUs
    assert resolve(partition="cpu-2h", gpus=None, gpu_type=None, mem=None, cpus=8)["cpus"] == 8


@pytest.mark.parametrize("override", [
    {"partition": "gpu-99h"},                  # unknown partition
    {"gpu_type": "h100; rm -rf /"},            # not on the allowlist (shell-injection shape)
    {"partition": "cpu-2h", "gpu_type": "h100"},  # gpu_type without GPUs
    {"gpus": 99},
    {"partition": "cpu-2h", "gpus": -1},       # invalid even where GPUs are dropped
    {"partition": ""},                         # empty is not "default"
    {"mem": 0},
    {"cpus": 0},
])
def test_invalid_resource_requests_are_rejected(override):
    base = dict(partition="gpu-2h", gpus=1, gpu_type=None, mem=16)
    with pytest.raises(ComputeSessionsError):
        Cluster(_cfg()).resolve_resources(**{**base, **override})


def test_session_prefix_derives_from_host_alias():
    assert _cfg(host="hydra").session_prefix == "hydra"
    assert _cfg(host="TU-Hydra.login").session_prefix == "tuhydralogin"


def test_local_path_resolves_against_project_and_rejects_escapes():
    proj = Path.home() / ".cs-test-never-created" / "proj"  # outside any agent scratchpad root
    info = SimpleNamespace(project_path=str(proj), session_id="hydra_abc")
    assert _resolve_local_inside_project(Path("out/x.jsonl"), info, "l") == (proj / "out/x.jsonl").resolve()
    assert _resolve_local_inside_project(proj / "a.bin", info, "l") == (proj / "a.bin").resolve()
    for bad in (Path("../escape"), Path("/etc/passwd"), Path("/tmp/other/x.py")):
        with pytest.raises(ComputeSessionsError):
            _resolve_local_inside_project(bad, info, "l")
    # Agent scratchpad roots are allowed despite being outside the project.
    scratch = Path("/tmp/claude-501/some-proj/some-session/scratchpad/helper.py")
    assert _resolve_local_inside_project(scratch, info, "l") == scratch.resolve()


def test_signals_are_validated():
    for ok in ("TERM", "SIGKILL", "9"):
        validate_signal(ok)
    for bad in ("FAKETERM", "99", "TERM; reboot"):
        with pytest.raises(ComputeSessionsError):
            validate_signal(bad)


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


def test_config_parses(tmp_path, monkeypatch):
    p = tmp_path / "config.toml"
    p.write_text(_GOOD_CONFIG)
    cfg = load_config(p)
    assert cfg.host == "hydra" and cfg.default_partition == "gpu-2h" and cfg.default_mem == 64
    assert cfg.exclude_nodes == ("head025",) and cfg.max_sessions_per_repo == 5
    assert cfg.remote_base == "~/.compute-sessions" and cfg.login_host == "" and cfg.container_user == ""
    monkeypatch.setenv("COMPUTE_SESSIONS_CONFIG", str(p))
    assert load_config().host == "hydra"


@pytest.mark.parametrize("mutate", [
    lambda s: s.replace('host = "hydra"\n', ""),                                 # no host
    lambda s: s.replace('partitions = ["gpu-2h", "cpu-2h", "gpu-test"]\n', ""),  # no partitions
    lambda s: s.replace('default_partition = "gpu-2h"', 'default_partition = "nope"'),
    lambda s: s.replace('gpu_types = ["h100", "80gb"]', 'gpu_types = ["h100; rm -rf /"]'),  # injection-shaped token
    lambda s: s.replace('default_mem = 64', 'default_mem = "lots"'),
    lambda s: s.replace("default_mem", "default_memory"),                        # typo'd key: rejected, not silently ignored
])
def test_invalid_configs_are_rejected(tmp_path, mutate):
    p = tmp_path / "config.toml"
    p.write_text(mutate(_GOOD_CONFIG))
    with pytest.raises(ComputeSessionsError):
        load_config(p)


def test_missing_config_gives_actionable_error(tmp_path):
    with pytest.raises(ComputeSessionsError, match="setup-cluster"):
        load_config(tmp_path / "nope.toml")


# ---------- session serialization, sync planning, project identity ----------

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
    assert agent["partition"] == "gpu-2h" and agent["mem"] == 42
    assert not {"job_id", "sshd_port", "project_id", "image", "resources", "gpu_type", "seconds_until_idle_deactivate"} & agent.keys()


def test_a_busy_session_shows_no_idle_countdown():
    now = datetime.now(timezone.utc)
    info = SessionInfo(session_id="hydra_x", project_id="p", project_path="/p", created_at="2026-01-01T00:00:00Z", status=SessionStatus.ACTIVE, idle_timeout_minutes=20, last_activity_at=_iso(now - timedelta(seconds=50)), busy=True)
    assert info.to_agent_json()["busy"] is True and "seconds_until_idle_deactivate" not in info.to_agent_json()
    info.status = SessionStatus.INACTIVE
    assert "busy" not in info.to_agent_json()


def test_active_session_reports_idle_countdown():
    now = datetime.now(timezone.utc)
    info = SessionInfo(
        session_id="hydra_1", project_id="p", project_path="/p", created_at=_iso(now),
        idle_timeout_minutes=10, status=SessionStatus.ACTIVE,
        last_activated_at=_iso(now), last_activity_at=_iso(now - timedelta(minutes=4)),
    )
    assert 0 < info.to_enriched_json()["seconds_until_idle_deactivate"] <= 6 * 60
    info.status = SessionStatus.INACTIVE
    assert info.to_enriched_json()["seconds_until_idle_deactivate"] is None


def test_sync_plan_ships_non_ignored_files_only(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / ".gitignore").write_text("data/\n")
    (repo / "main.py").write_text("print('hi')\n")
    (repo / "data").mkdir()
    (repo / "data" / "weights.bin").write_bytes(b"\x00")
    (repo / "gone.py").write_text("x\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    (repo / "untracked.txt").write_text("new\n")
    (repo / "gone.py").unlink()  # tracked but deleted locally — must not be shipped
    files = plan_sync(repo)
    assert {"main.py", ".gitignore", "untracked.txt"} <= set(files)
    assert not any(f.startswith("data/") for f in files) and "gone.py" not in files
    # Outside git: junk dirs are skipped.
    plain = tmp_path / "plain"
    (plain / "__pycache__").mkdir(parents=True)
    (plain / "a.py").write_text("x\n")
    (plain / "__pycache__" / "a.pyc").write_bytes(b"\x00")
    assert plan_sync(plain) == ["a.py"]


def test_project_id_prefers_git_remote_and_falls_back_to_path(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "remote", "add", "origin", "git@github.com:owner/My-Repo.git"], cwd=repo, check=True)
    assert project_id(repo) == "owner-my-repo"
    plain = tmp_path / "plain"
    plain.mkdir()
    assert project_id(plain).startswith("plain-")


# ---------- the login-node helper: session selection ----------

def _sel(env: Env, args, maybe=False) -> tuple[str, list[str]]:
    r = env.cluster.call("session", args, maybe=maybe)
    return r["sid"], r["rest"]


def test_session_selection(env):
    env.record("hydra_aaa111", status=SessionStatus.ACTIVE)
    env.record("hydra_bbbb22", project_id="p2", status=SessionStatus.INACTIVE)
    env.record("hydra_bbbb33", project_id="p2", status=SessionStatus.ACTIVE)
    notes = []
    env.cluster.on_note = lambda sid, where: notes.append(sid)
    assert _sel(env, ["hydra_aaa111"]) == ("hydra_aaa111", [])   # exact id
    assert _sel(env, [" HYDRA_A "]) == ("hydra_aaa111", [])      # unique prefix; copy-pasted ids arrive with whitespace and uppercase
    assert _sel(env, ["aaa1"]) == ("hydra_aaa111", [])           # bare hex tail
    assert notes == []
    assert _sel(env, ["hydra_bbbb33"])[0] == "hydra_bbbb33" and notes == ["hydra_bbbb33"]  # another project's session: legal, noted
    with pytest.raises(SessionError, match="ambiguous"):
        _sel(env, ["hydra_bbbb"])
    with pytest.raises(SessionError, match="no session matches"):
        _sel(env, ["hydra_zzzz"])
    # A prefix shorter than 4 characters matches only the cwd project's sessions.
    with pytest.raises(UsageError, match="only matches the cwd project's sessions"):
        _sel(env, ["b"])
    # No token: the cwd project's only session, or its only live one.
    assert _sel(env, []) == ("hydra_aaa111", [])
    env.record("hydra_ddd444", status=SessionStatus.INACTIVE)
    assert _sel(env, [])[0] == "hydra_aaa111"
    env.record("hydra_eee555", status=SessionStatus.PENDING)
    with pytest.raises(SessionError, match="2 live sessions — pass one"):
        _sel(env, [])
    # A prefix ambiguous globally but unique within the cwd project means the project's own session.
    env.record("hydra_fff666", project_id="p2")
    env.record("hydra_fff777")
    assert _sel(env, ["hydra_fff"])[0] == "hydra_fff777"


def test_a_subdirectory_points_at_the_sessions_of_the_directory_above(env, monkeypatch):
    env.record("hydra_aaa111", status=SessionStatus.ACTIVE)
    sub = env.proj / "src"
    sub.mkdir()
    monkeypatch.chdir(sub)
    cluster = Cluster(env.cluster.cfg, env.access)
    with pytest.raises(UsageError, match=f"{re.escape(str(env.proj))} has sessions: run cs there"):
        cluster.call("session", [])
    notes = []
    cluster.on_note = lambda sid, where: notes.append(sid)
    assert cluster.call("session", ["hydra_aaa111"])["sid"] == "hydra_aaa111" and notes == []  # the parent's session: no "another project" note


def test_session_inference_without_a_live_session_lists_most_recent_first(env):
    now = datetime.now(timezone.utc)
    for i in range(6):
        env.record(f"hydra_ddd{i}00", status=SessionStatus.INACTIVE, last_deactivated_at=_iso(now - timedelta(hours=i)))
    with pytest.raises(SessionError, match=r"none is live — pass one, most recent first: hydra_ddd000 .*2 more via"):
        _sel(env, [])
    (env.base / "sessions").rename(env.base / "elsewhere")
    with pytest.raises(SessionError, match="no sessions for project"):
        _sel(env, [])


def test_maybe_token_splits_off_only_session_shaped_args(env):
    """Commands with trailing free-form args (run/upload/ls/…): `<prefix>_…` up front is always the session (a typo'd id must not run as a shell command); a bare hex token only when a session matches."""
    env.record("hydra_aaa111", status=SessionStatus.ACTIVE)
    split = lambda *a: _sel(env, list(a), maybe=True)[1]  # noqa: E731
    assert split("aaa1", "echo", "hi") == ["echo", "hi"]
    assert split("beef", "echo", "hi") == ["beef", "echo", "hi"]
    assert split("echo", "hydra_abc") == ["echo", "hydra_abc"]
    assert split("date", "+%s") == ["date", "+%s"]
    assert split() == []
    with pytest.raises(SessionError, match="no session matches"):
        split("hydra_abcd", "echo", "hi")


def test_helper_ignores_a_stray_module_in_the_working_dir(env, tmp_path):
    # -I: neither the cwd nor PYTHONPATH may shadow a stdlib module the helper imports.
    trap = tmp_path / "trap"
    trap.mkdir()
    (trap / "json.py").write_text("raise SystemExit('shadowed stdlib json')\n")
    env.record("hydra_aaa111")
    env.mp.chdir(trap)
    env.mp.setenv("PYTHONPATH", str(trap))
    assert env.access.call("session", args=["hydra_aaa111"])["sid"] == "hydra_aaa111"


def test_helper_reply_survives_shell_banners(env, monkeypatch):
    noisy = env.access._helper_argv
    monkeypatch.setattr(env.access, "_helper_argv", lambda n: ["bash", "-c", 'trap "echo bye" EXIT; echo "Welcome! {banner}"; echo noise >&2; "$@"', "x", *noisy(n)])
    env.record("hydra_aaa111")
    assert env.access.call("session", args=["hydra_aaa111"])["sid"] == "hydra_aaa111"


def test_list_keeps_the_projects_and_same_directory_sessions(env):
    env.record("hydra_aaa111")
    env.record("hydra_bbb222", project_id="former-id")              # same dir, old project id
    env.record("hydra_ccc333", project_id="other", project_path="/elsewhere")
    (env.base / "sessions" / "hydra_zz_stray" / "workdir").mkdir(parents=True)  # a create that died before its record
    assert sorted(d["session_id"] for d in env.cluster.call("list")["sessions"]) == ["hydra_aaa111", "hydra_bbb222"]
    assert len(env.cluster.call("list", all=True)["sessions"]) == 3
    res = _cs("list")
    assert res.exit_code == 0 and "hydra_aaa111" in res.stdout and "hydra_bbb222" not in res.stdout
    assert "former project id 'former-id'" in res.stderr


def test_list_shows_live_sessions_and_the_20_most_recently_stopped(env, monkeypatch):
    for n in range(25):
        env.record(f"hydra_{n:06d}", status=SessionStatus.INACTIVE, last_deactivated_at=f"2026-01-01T00:{n:02d}:00Z")
    env.record("hydra_live01", status=SessionStatus.ACTIVE, last_activated_at="2026-01-01T00:00:00Z", busy=True)
    res = _cs("list", "--json")
    ids = [r["session_id"] for r in json.loads(res.stdout)]
    assert ids == ["hydra_live01"] + [f"hydra_{n:06d}" for n in range(24, 4, -1)]
    assert "5 older inactive/failed session(s) not shown" in res.stderr
    assert re.search(r"hydra_live01 +active .* busy", _cs("list").stdout)


def test_list_shows_the_time_left_of_an_active_session(env):
    started = _iso(datetime.now(timezone.utc) - timedelta(minutes=30))
    env.record("hydra_aaa111", status=SessionStatus.ACTIVE, job_started_at=started, last_activated_at=started)
    row = json.loads(_cs("list", "--json").stdout)[0]
    assert 5395 <= row["time_left_seconds"] <= 5400  # gpu-2h, 30 min in
    assert re.search(r"hydra_aaa111 .* 1h(29|30)m\s*$", _cs("list").stdout, re.M)
    env.record("hydra_bbb222", status=SessionStatus.ACTIVE, last_activated_at=started, resources={"partition": "gpu-test", "gpus": 1, "mem": 42})
    assert "time_left_seconds" not in json.loads(_cs("list", "--json").stdout)[1]  # no limit in the partition name


def test_a_pending_session_shows_its_wait_and_the_schedulers_start_forecast(env, monkeypatch):
    submitted = _iso(datetime.now(timezone.utc) - timedelta(minutes=12))
    env.record("hydra_aaa111", status=SessionStatus.PENDING, job_id="4711", submitted_at=submitted)
    today = datetime.now().strftime("%Y-%m-%d")
    monkeypatch.setenv("SQUEUE_STDOUT", f"4711|{today}T15:30:00")
    row = json.loads(_cs("list", "--json").stdout)[0]
    assert row["estimated_start"] == f"{today}T15:30:00" and 715 <= row["time_in_status_seconds"] <= 725
    assert re.search(r"hydra_aaa111 +pending +12m0\ds .* ~start 15:30$", _cs("list").stdout, re.M)
    # cs show's job block carries the same forecast (from the job's own query); N/A means none.
    monkeypatch.setenv("SQUEUE_STDOUT", f"PENDING|2:00:00|(Priority)|{today}T15:30:00")
    assert sess.show(env.cluster, "hydra_aaa111", probe=False)["job"]["estimated_start"] == f"{today}T15:30:00"
    monkeypatch.setenv("SQUEUE_STDOUT", "PENDING|2:00:00|(Priority)|N/A")
    assert sess.show(env.cluster, "hydra_aaa111", probe=False)["job"]["estimated_start"] is None
    assert cli._start_label("2030-01-02T03:04:05") == "~start 01-02 03:04"


def test_an_argument_with_spaces_gets_a_word_splitting_hint(env):
    env.record("hydra_aaa111")
    for arg in ("hydra_aaa111 cmd_1234", "aaa1 cmd_1234"):
        res = _cs("logs", arg)
        assert res.exit_code == 2 and "one argument containing spaces" in res.stderr


# ---------- scheduler: job state, submit, deactivate, records under a lock ----------

def test_job_state_distinguishes_gone_from_scheduler_failure(env, monkeypatch):
    env.record("hydra_aaa111", status=SessionStatus.PENDING, job_id="4711")
    job = lambda: sess.show(env.cluster, "hydra_aaa111", probe=False).get("job")  # noqa: E731
    monkeypatch.setenv("SQUEUE_STDOUT", "RUNNING|1:59:00|head042")
    assert job() == {"state": "running", "reason": None, "node": "head042", "time_left": "1:59:00"}
    monkeypatch.setenv("SQUEUE_STDOUT", "PENDING|2:00:00|(Priority)")
    assert job()["reason"] == "Priority"
    monkeypatch.setenv("SQUEUE_STDOUT", "PENDING|2:00:00|(None)")
    assert job()["reason"] is None
    monkeypatch.setenv("SQUEUE_STDOUT", "SUSPENDED|1:00:00|head042")  # still holds an allocation: never "gone"
    assert job()["state"] == "pending"
    monkeypatch.setenv("SQUEUE_RC", "124")  # `timeout` stopped it
    assert job() == {"state": "unknown", "reason": "scheduler query timed out", "node": None}
    # An unreachable controller is not "gone" (that would fail a queued session and orphan its allocation).
    monkeypatch.setenv("SQUEUE_STDOUT", "")
    monkeypatch.setenv("SQUEUE_STDERR", "slurm_load_jobs error: Unable to contact slurm controller (connect failure)")
    monkeypatch.setenv("SQUEUE_RC", "1")
    st = job()
    assert st["state"] == "unknown" and "Unable to contact slurm controller" in st["reason"]
    assert env.read("hydra_aaa111").status == SessionStatus.PENDING
    # A purged job id is the one failure that means gone; the pending session it left is reconciled to failed.
    monkeypatch.setenv("SQUEUE_STDERR", "slurm_load_jobs error: Invalid job id specified")
    assert job()["state"] == "gone"
    assert env.read("hydra_aaa111").status == SessionStatus.FAILED


def test_show_reconciles_a_gone_job_with_compare_and_set(env, monkeypatch):
    env.record("hydra_aaa111", status=SessionStatus.ACTIVE, job_id="1", last_activated_at="2026-01-01T00:00:00Z")
    out = sess.show(env.cluster, "hydra_aaa111", probe=False)
    assert out["info"]["status"] == "inactive" and out["info"]["last_deactivated_at"]
    # A re-activation that lands while the old job is looked up owns the record: no overwrite.
    env.record("hydra_aaa111", status=SessionStatus.ACTIVE, job_id="1")
    cfgp = env.base / "sessions" / "hydra_aaa111" / "config.json"
    monkeypatch.setenv("SQUEUE_HOOK", f"python3 -c \"import json; p='{cfgp}'; d=json.load(open(p)); d.update(status='pending', job_id='2'); json.dump(d, open(p, 'w'))\"")
    out = sess.show(env.cluster, "hydra_aaa111", probe=False)
    assert (env.read("hydra_aaa111").status, env.read("hydra_aaa111").job_id) == (SessionStatus.PENDING, "2")


def test_submit_renders_sbatch_flags(env):
    cluster = Cluster(_cfg(remote_base=str(env.base), exclude_nodes=("head025",), login_host="hydra-internal"), env.access)
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    info = sess.activate(cluster, "hydra_aaa111", sess.ResourceRequest(partition="gpu-2h", gpus=2, gpu_type="h100|80gb", mem=64))
    assert (info.status, info.job_id) == (SessionStatus.PENDING, "4711")
    (argv,) = env.slurm_log("sbatch")
    logs = env.base / "sessions" / "hydra_aaa111" / "logs"
    assert argv == f"--export=ALL --parsable --job-name=cs-hydra_aaa111 --partition=gpu-2h --exclude=head025 --gpus-per-node=2 --constraint=h100|80gb --mem=64G --output={logs}/runner-%j.out {env.base}/runner.py --base {env.base} --session-id hydra_aaa111 --login hydra-internal"
    rec = env.read("hydra_aaa111")
    assert rec.resources == {"partition": "gpu-2h", "gpus": 2, "gpu_type": "h100|80gb", "mem": 64} and rec.image == "images/default.sif"


def test_gpu_activations_avoid_nodes_with_a_failed_gpu_check(env):
    now = time.time()
    (env.base / "bad-nodes.json").write_text(json.dumps({"head023": {"at": now - 60}, "head074": {"at": now - 2 * 86400}, "bad;name": {"at": now}}))
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    sess.activate(env.cluster, "hydra_aaa111", sess.ResourceRequest(partition="gpu-2h"))
    env.record("hydra_bbb222", status=SessionStatus.INACTIVE)
    env.mp.setenv("SQUEUE_BYNAME", "")
    sess.activate(env.cluster, "hydra_bbb222", sess.ResourceRequest(partition="cpu-2h"))
    gpu, cpu = env.slurm_log("sbatch")
    assert "--exclude=head023 " in gpu and "--exclude" not in cpu  # only a recent entry, and only for GPU sessions


def test_activate_clears_stale_runtime_state_and_does_not_clobber_a_fast_runner(env, monkeypatch):
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE, job_id="1", node="old-node", sshd_port=1, gpu="old")
    cfgp = env.base / "sessions" / "hydra_aaa111" / "config.json"
    # Inside sbatch: the pending record must carry no previous job/node (show() would find that job gone and fail the fresh session); then the runner beats us to the post-submit write.
    monkeypatch.setenv("SBATCH_HOOK", f"python3 -c \"import json,sys; p='{cfgp}'; d=json.load(open(p)); assert (d['status'], d['job_id'], d['node'], d['gpu']) == ('pending', None, None, None), d; d.update(status='active', job_id='4711', node='head042', sshd_port=2222); json.dump(d, open(p, 'w'))\" || echo HOOK-FAILED >> $FAKE_SLURM/sbatch.log")
    sess.activate(env.cluster, "hydra_aaa111", sess.ResourceRequest())
    assert "HOOK-FAILED" not in env.slurm_log("sbatch")
    rec = env.read("hydra_aaa111")
    assert (rec.status, rec.job_id, rec.node, rec.sshd_port) == (SessionStatus.ACTIVE, "4711", "head042", 2222)


def test_activate_and_clone_reuse_previous_resources_without_flags(env):
    env.mp.setenv("SQUEUE_BYNAME", "")  # no session has a job yet
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE, resources={"partition": "cpu-2h", "gpus": 0, "gpu_type": None, "mem": 8}, idle_timeout_minutes=45)
    info = sess.activate(env.cluster, "hydra_aaa111", sess.ResourceRequest())
    assert info.resources == {"partition": "cpu-2h", "gpus": 0, "gpu_type": None, "mem": 8} and info.idle_timeout_minutes == 45
    assert "--partition=cpu-2h" in env.slurm_log("sbatch")[-1] and "--gpus-per-node" not in env.slurm_log("sbatch")[-1]
    clone = sess.clone(env.cluster, "hydra_aaa111", sess.ResourceRequest())
    assert clone.resources["partition"] == "cpu-2h" and clone.idle_timeout_minutes == 45
    # Any resource flag starts from the config defaults instead.
    env.record("hydra_bbb222", status=SessionStatus.INACTIVE, resources={"partition": "cpu-2h", "gpus": 0, "gpu_type": None, "mem": 8})
    info = sess.activate(env.cluster, "hydra_bbb222", sess.ResourceRequest(gpus=2))
    assert info.resources == {"partition": "gpu-2h", "gpus": 2, "gpu_type": None, "mem": 42} and info.idle_timeout_minutes == 20
    # --cpus is a resource flag like the others and reaches the scheduler.
    env.record("hydra_ccc333", status=SessionStatus.INACTIVE)
    info = sess.activate(env.cluster, "hydra_ccc333", sess.ResourceRequest(partition="cpu-2h", cpus=8))
    assert info.resources["cpus"] == 8 and "--cpus-per-task=8" in env.slurm_log("sbatch")[-1]


@pytest.mark.parametrize("job_id, squeue, match", [
    ("1", {"SQUEUE_BYNAME": "1|RUNNING"}, "already active"),
    (None, {"SQUEUE_BYNAME": "77|PENDING"}, r"already active \(job_id=77\)"),  # a job the record lost track of counts too
    ("1", {"SQUEUE_STDERR": "Unable to contact slurm controller", "SQUEUE_RC": "1"}, "cannot tell whether session hydra_aaa111 still has a job"),
])
def test_activate_refuses_on_a_live_or_unknown_previous_job(env, monkeypatch, job_id, squeue, match):
    env.record("hydra_aaa111", status=SessionStatus.ACTIVE, job_id=job_id)
    for k, v in squeue.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(SessionError, match=match):
        sess.activate(env.cluster, "hydra_aaa111", sess.ResourceRequest())
    assert env.slurm_log("sbatch") == [] and env.read("hydra_aaa111").status == SessionStatus.ACTIVE


def test_parallel_activations_submit_one_job(env, monkeypatch):
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    monkeypatch.setenv("SBATCH_SLEEP", "0.5")
    errors = []

    def go():
        try:
            sess.activate(env.cluster, "hydra_aaa111", sess.ResourceRequest())
        except SessionError as exc:
            errors.append(str(exc))
    threads = [threading.Thread(target=go) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(env.slurm_log("sbatch")) == 1 and len(errors) == 2 and all("already active" in e for e in errors)


def test_abort_waits_for_an_in_flight_submit_and_cancels_its_job(env, monkeypatch):
    env.record("hydra_aaa111")
    monkeypatch.setenv("SBATCH_SLEEP", "1")
    t = threading.Thread(target=lambda: sess.activate(env.cluster, "hydra_aaa111", sess.ResourceRequest()))
    t.start()
    _real_sleep(0.4)  # the submit holds the session lock
    env.access.call("abort", sid="hydra_aaa111")
    t.join()
    assert env.slurm_log("scancel") == [f"-u {_user()} -n cs-hydra_aaa111"]
    assert env.read("hydra_aaa111").status == SessionStatus.FAILED


def _user() -> str:
    import pwd
    return pwd.getpwuid(os.getuid()).pw_name


def test_a_submit_without_an_answer_is_settled_by_the_jobs_name(env, monkeypatch):
    import compute_sessions.login_helper as lh
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    answers = {"squeue": [(0, "", ""), (0, "4712|PENDING\n", "")], "sbatch": [(None, "", "timed out after 120s")]}
    monkeypatch.setattr(lh, "slurm", lambda argv, timeout: answers[argv[0]].pop(0))
    req = {"base": str(env.base), "args": ["hydra_aaa111"], "prefix": "hydra", "project_id": env.pid, "resources": {"partition": "gpu-2h", "gpus": 1, "gpu_type": None, "mem": 8}, "idle_timeout_minutes": 20, "image": "x.sif", "login": "hydra"}
    assert lh.op_submit(lh.Ctx(req))["info"]["job_id"] == "4712"
    assert (env.read("hydra_aaa111").status, env.read("hydra_aaa111").job_id) == (SessionStatus.PENDING, "4712")


def test_activate_submit_failure_restores_the_previous_status(env, monkeypatch):
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    monkeypatch.setenv("SBATCH_RC", "1")
    monkeypatch.setenv("SBATCH_STDERR", "sbatch: error: invalid partition")
    with pytest.raises(RemoteError, match="invalid partition"):
        sess.activate(env.cluster, "hydra_aaa111", sess.ResourceRequest())
    assert env.read("hydra_aaa111").status == SessionStatus.INACTIVE


def test_create_validates_the_request_before_writing_anything(env, monkeypatch):
    monkeypatch.setattr(sess, "plan_sync", lambda *a: pytest.fail("nothing may run for an invalid request"))
    with pytest.raises(ComputeSessionsError, match="gpu_type"):
        sess.create(env.cluster, sess.ResourceRequest(gpu_type="h200"), env.proj)
    with pytest.raises(ComputeSessionsError, match="idle_timeout"):
        sess.create(env.cluster, sess.ResourceRequest(idle_timeout_minutes=0), env.proj)
    assert not (env.base / "sessions").exists()


def test_create_syncs_and_submits(env):
    (env.proj / "train.py").write_text("print(1)")
    info = sess.create(env.cluster, sess.ResourceRequest(partition="cpu-2h"), env.proj)
    assert (info.status, info.job_id, info.project_path) == (SessionStatus.PENDING, "4711", str(env.proj))
    sdir = env.base / "sessions" / info.session_id
    assert (sdir / "workdir" / "train.py").read_text() == "print(1)" and (sdir / ".sync-manifest").read_text() == "train.py\n"
    # The session cap counts live sessions of the project.
    cluster = Cluster(_cfg(remote_base=str(env.base), max_sessions_per_repo=1), env.access)
    with pytest.raises(SessionError, match="limit is 1"):
        sess.create(cluster, sess.ResourceRequest(partition="cpu-2h"), env.proj)


@pytest.mark.parametrize("step", ["rsync", "interrupt", "sbatch"])
def test_create_that_does_not_complete_leaves_a_failed_record(env, monkeypatch, step):
    (env.proj / "train.py").write_text("print(1)")
    if step == "sbatch":
        monkeypatch.setenv("SBATCH_RC", "1")
    else:
        failure = RemoteError("rsync failed") if step == "rsync" else KeyboardInterrupt()
        monkeypatch.setattr(env.access, "rsync_files", lambda *a, **k: (_ for _ in ()).throw(failure))
    with pytest.raises((RemoteError, KeyboardInterrupt)):
        sess.create(env.cluster, sess.ResourceRequest(), env.proj)
    (rec,) = env.records()
    assert rec.status == SessionStatus.FAILED and rec.job_id is None and rec.last_deactivated_at


def test_an_interrupted_first_request_leaves_a_failed_record(env, monkeypatch):
    (env.proj / "train.py").write_text("print(1)")
    real = HostAccess.call

    def interrupted(self, op, **kw):
        if op == "create":
            real(self, op, **kw)          # the request ran on the login node …
            raise KeyboardInterrupt       # … but the client was interrupted before the reply
        return real(self, op, **kw)
    monkeypatch.setattr(LocalAccess, "call", interrupted)
    with pytest.raises(KeyboardInterrupt):
        sess.create(env.cluster, sess.ResourceRequest(), env.proj)
    (rec,) = env.records()
    assert rec.status == SessionStatus.FAILED
    # The abort can also arrive before the request it aborts: its failed record makes the late request fail.
    record = sess._new_record(env.cluster, env.pid, str(env.proj))
    env.access.call("abort", sid=record["session_id"], record=record)
    with pytest.raises(SessionError, match="aborted"):
        env.access.call("create", record=record)
    assert env.read(record["session_id"]).status == SessionStatus.FAILED
    # A refused create (session cap) wrote nothing, so there is nothing to abort.
    monkeypatch.setattr(LocalAccess, "call", real)
    cluster = Cluster(_cfg(remote_base=str(env.base), max_sessions_per_repo=1), env.access)
    env.record("hydra_live01", status=SessionStatus.ACTIVE)
    with pytest.raises(SessionError, match="limit is 1"):
        sess.create(cluster, sess.ResourceRequest(), env.proj)
    assert len(env.records()) == 3


def test_clone_copies_workdir_and_manifest_server_side(env):
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    src = env.base / "sessions" / "hydra_aaa111"
    (src / "workdir" / "data").mkdir()
    (src / "workdir" / "data" / "weights.bin").write_text("blob")
    (src / ".sync-manifest").write_text("train.py\n")
    (src / "logs" / "cmd_x.out").write_text("history")
    info = sess.clone(env.cluster, "hydra_aaa111", sess.ResourceRequest())
    dst = env.base / "sessions" / info.session_id
    assert info.project_id == env.pid and (dst / "workdir" / "data" / "weights.bin").read_text() == "blob"
    assert (dst / ".sync-manifest").read_text() == "train.py\n" and list((dst / "logs").iterdir()) == []


def test_deactivate(env, monkeypatch):
    _fast_sleep(monkeypatch)
    # A jobless pending record (a create that died before submitting) is cleared at once.
    env.record("hydra_aaa111")
    assert sess.deactivate(env.cluster, "hydra_aaa111").status == SessionStatus.INACTIVE
    # A queued job: cancelled and marked inactive at once — no runner ever started, so no teardown will write it.
    env.record("hydra_ccc333", status=SessionStatus.PENDING, job_id="8")
    ops = _count_calls(env, monkeypatch)
    assert sess.deactivate(env.cluster, "hydra_ccc333").status == SessionStatus.INACTIVE and ops == ["deactivate"]
    # An active job: cancelled, then marked inactive by compare-and-set when no runner teardown shows up.
    env.record("hydra_bbb222", status=SessionStatus.ACTIVE, job_id="7")
    info = sess.deactivate(env.cluster, "hydra_bbb222")
    assert env.slurm_log("scancel") == [f"-u {_user()} -n cs-hydra_{s}" for s in ("aaa111", "ccc333", "bbb222")]
    assert info.status == SessionStatus.INACTIVE and info.last_deactivated_at
    # cs deactivate records itself as the reason (before the cancel, so the runner's teardown keeps it); `cs show` names it.
    assert info.stop_reason == "cs deactivate"
    out = _cs("show", "hydra_bbb222").stdout
    assert "stop_reason: cs deactivate" in out and "job_id" not in out and "image" not in out  # plumbing stays out of cs show


def test_deactivate_names_the_commands_it_stops(env, monkeypatch):
    _fast_sleep(monkeypatch)
    env.active()
    (env.base / "sessions" / "hydra_aaa111" / ".sshd").mkdir()
    (env.base / "sessions" / "hydra_aaa111" / ".sshd" / "sshd.pid").write_text("1")
    os.utime(env.base / "sessions" / "hydra_aaa111" / ".sshd" / "sshd.pid", (time.time() - 60, time.time() - 60))
    _write_cmd(env, "cmd_live", pid=_NO_PID)
    (env.logs_dir() / "cmd_live.start").touch()
    (env.logs_dir() / "cmd_live.cmd").write_text("python train.py --epochs 3")
    _write_cmd(env, "cmd_done", pid=_NO_PID, exit_code="0")
    res = _cs("deactivate", "hydra_aaa111")
    assert res.exit_code == 0 and "stopping running command(s): cmd_live (python train.py --epochs 3)" in res.stderr and "cmd_done" not in res.stderr


def test_transfers_refuse_two_sources_of_one_name(env):
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    res = _cs("download", "hydra_aaa111", "a/x.json", "b/x.json", "out/")
    assert res.exit_code == 2 and "several sources are named 'x.json'" in res.stderr


# ---------- live output: the in-container follower ----------

def _follow_raw(env: Env, cid: str, *, command: str | None = None, offsets=(0, 0), round_seconds=300, keepalive=15, follow=True, timeout=20) -> list[tuple]:
    """Run one follower connection locally and return its parsed frames."""
    marker = "__CS_TEST_MARK__"
    script = container_script(cid, marker, command=command, offsets=offsets, round_seconds=round_seconds, keepalive=keepalive, follow=follow)
    p = subprocess.run(env.access.container_argv("s", "sock", "bash -l -s"), input=script, capture_output=True, timeout=timeout)
    return Frames(marker).feed(p.stdout)


# Above every pid_max, so no process has it: a fake command's `.pid` must never name a real process group.
_NO_PID = "4999999"


def _write_cmd(env: Env, cid: str, out: bytes = b"", err: bytes = b"", pid: str | None = "", exit_code: str | None = None, pid_age: float = 0) -> Path:
    d = env.logs_dir()
    (d / f"{cid}.out").write_bytes(out)
    (d / f"{cid}.err").write_bytes(err)
    if pid is not None:
        (d / f"{cid}.pid").write_text(pid)
        t = time.time() - pid_age
        os.utime(d / f"{cid}.pid", (t, t))
    if exit_code is not None:
        (d / f"{cid}.exit").write_text(exit_code)
    return d


def _payload(frames, tag) -> bytes:
    return b"".join(f[2] for f in frames if f[0] == tag)


def test_follower_streams_both_streams_binary_safe_and_reports_exit(env):
    env.active()
    frames = _follow_raw(env, "cmd_a", command="printf 'a\\0\\377b\\n'; echo oops >&2; exit 3")
    assert frames[0] == ("S", "") and frames[-1] == ("X", "3")
    assert _payload(frames, "O") == b"a\x00\xffb\n" and _payload(frames, "E") == b"oops\n"
    # The launch recorded the command like the shared-FS readers expect.
    d = env.logs_dir()
    assert (d / "cmd_a.exit").read_text().strip() == "3" and (d / "cmd_a.start").exists()


def test_follower_resumes_from_offsets_and_chunks_large_output(env):
    env.active()
    big = os.urandom(150_000)
    _write_cmd(env, "cmd_b", out=big, exit_code="0\n")
    frames = _follow_raw(env, "cmd_b", offsets=(1000, 0))
    outs = [f for f in frames if f[0] == "O"]
    assert outs[0][1] == 1000 and all(len(f[2]) <= 65536 for f in outs)
    assert _payload(frames, "O") == big[1000:] and frames[-1] == ("X", "0")


@pytest.mark.parametrize("case, expect", [
    ("missing", "M"),                 # no .pid: no such command
    ("stale_pid_reused", "K"),        # an earlier activation's command: its PID now belongs to another live process, but the heartbeat is old
    ("dead_pid", "K"),                # the process is gone without an exit (SIGKILL/OOM)
    ("garbage_exit", "X"),            # unreadable .exit of a dead command: exited, code unknown
])
def test_follower_liveness(env, case, expect):
    env.active()
    if case == "stale_pid_reused":
        _write_cmd(env, "cmd_c", pid=str(os.getpid()), pid_age=120)
    elif case == "dead_pid":
        p = subprocess.Popen(["true"])
        p.wait()
        _write_cmd(env, "cmd_c", pid=str(p.pid))
    elif case == "garbage_exit":
        _write_cmd(env, "cmd_c", pid=_NO_PID, pid_age=120, exit_code="")
    frames = _follow_raw(env, "cmd_c")
    assert frames[-1][0] == expect and frames[-1][1] == ""


def test_follower_waits_through_a_starting_command_and_an_unsettled_exit(env):
    env.active()
    d = _write_cmd(env, "cmd_d", pid="")  # .pid exists but empty: the launcher's inner bash has not written its PID yet

    def later():
        _real_sleep(0.5)
        (d / "cmd_d.exit").write_text("")      # truncate-then-write: an empty .exit is not a result yet
        (d / "cmd_d.out").write_bytes(b"late\n")
        _real_sleep(0.5)
        (d / "cmd_d.exit").write_text("7\n")
    t = threading.Thread(target=later)
    t.start()
    frames = _follow_raw(env, "cmd_d")
    t.join()
    assert _payload(frames, "O") == b"late\n" and frames[-1] == ("X", "7")


def test_follower_round_ends_and_keepalives(env):
    env.active()
    _write_cmd(env, "cmd_e", out=b"x", pid=str(os.getpid()))
    frames = _follow_raw(env, "cmd_e", round_seconds=2, keepalive=1)
    assert ("H", "") in frames and frames[-1] == ("T", "")


def test_follower_stops_when_the_reader_is_gone(env):
    env.active()
    _write_cmd(env, "cmd_f", pid=str(os.getpid()))
    script = container_script("cmd_f", "MARK", keepalive=1)
    p = subprocess.Popen(env.access.container_argv("s", "sock", "bash -l -s"), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.stdin.write(script)
    p.stdin.close()
    assert p.stdout.readline() == b"MARK\n"
    p.stdout.close()
    assert p.wait(timeout=10) is not None  # the next keepalive write fails


def test_frames_parse_incrementally_and_reject_garbage():
    data = b"login banner\n" + b"MARK\nS\nO 0 5\nhel\nlE 0 2\nerX 0\n"
    for cut in range(len(data)):
        f = Frames("MARK")
        assert f.feed(data[:cut]) + f.feed(data[cut:]) == [("S", ""), ("O", 0, b"hel\nl"), ("E", 0, b"er"), ("X", "0")]
    with pytest.raises(run_mod.FrameError):
        Frames("MARK").feed(b"MARK\nO x 5\n")
    with pytest.raises(run_mod.FrameError):
        Frames("MARK").feed(b"MARK\nwhat\n")


def test_output_collapses_progress_on_whole_lines(capsys):
    assert collapse_cr(b"10%\r50%\r100%") == b"100%" and collapse_cr(b"a\r") == b"a"
    out = Output(raw=False)
    for chunk in (b"loading: 10%\rload", b"ing: 50%\rloading: 100%\ndone\r\nhalf", b" line"):
        out.write("O", chunk)
    # The first completed progress state shows as a line at once; the partial line is held …
    assert capsys.readouterr().out == "loading: 10%\nloading: 100%\ndone\n"
    out.flush()
    assert capsys.readouterr().out == "half line"           # … until the end
    for i in range(1000):
        out.write("O", b"%d%%\r" % i)
    assert len(out.sinks["O"].partial) < 10                  # \r-only progress is not accumulated
    out.write("O", b"\n")
    assert capsys.readouterr().out == "0%\n999%\n"
    # A terminal gets the bytes as they are; --tail keeps the last N collapsed lines of each stream.
    raw = Output(raw=True)
    raw.write("O", b"a\rb\n")
    assert capsys.readouterr().out == "a\rb\n"
    tail = Output(tail=2)
    tail.write("O", b"1\n2\n3\r33\n")
    tail.write("E", b"e1\n")
    assert capsys.readouterr() == ("", "")
    tail.flush()
    assert capsys.readouterr() == ("2\n33\n", run_mod.STDERR_SEPARATOR.decode() + "e1\n")  # stderr printed after stdout is marked as such
    tail.flush()  # idempotent
    assert capsys.readouterr() == ("", "")


def test_output_shows_held_progress_at_most_every_30s(capsys, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(run_mod.time, "monotonic", lambda: clock[0])
    out = Output(raw=False)
    for step in range(1, 101):  # a tqdm-style bar: "\r" + state, one update per second
        clock[0] += 1
        out.write("E", b"\rtrain: %d/100" % step)
    # Only states a \r has completed, at most one per 30 s: steps 1, 31, 61, 91.
    assert capsys.readouterr().err == "".join(f"train: {n}/100\n" for n in (1, 31, 61, 91))
    out.write("E", b"\n")
    assert capsys.readouterr().err == "train: 100/100\n"
    out.write("E", b"\rsave: 1/2\rsave: 2/2\r")  # ends in the state just shown: not repeated by flush
    out.flush()
    assert capsys.readouterr().err == "save: 2/2\n"


def test_output_passes_binary_data_unchanged(capsysbinary):
    # Also when the first invalid byte arrives mid-stream: lines before it were text.
    data = b"text\rline\n" + bytes(range(256)) * 4 + b"\r\n\xff"
    out = Output(raw=False)
    out.write("O", data[:10])
    out.write("O", data[10:])
    out.flush()
    assert capsysbinary.readouterr().out == b"line\n" + data[10:]


# ---------- cs run / logs -f / kill end to end (local "container") ----------

def test_cs_run_streams_and_exits_with_the_commands_code(env):
    env.active()
    res = _cs("run", "printf 'step 1\\n'; echo warn >&2; exit 4")
    assert res.exit_code == 4 and res.stdout == "step 1\n" and "warn" in res.stderr and re.search(r"cs: cmd_\w+ on hydra_aaa111", res.stderr)
    # Exec-style args, a leading session token, and --tail.
    res = _cs("run", "--tail", "1", "hydra_aaa111", "python3", "-c", "print('a'); print('b c')")
    assert res.exit_code == 0 and res.stdout == "b c\n"
    assert _cs("run", "aaa1").exit_code == 2  # a session but no command
    # The command is parsed on its own: comments and heredocs work, a syntax error is reported at once.
    res = _cs("run", "echo a # trailing comment")
    assert res.exit_code == 0 and res.stdout == "a\n"
    res = _cs("run", "cat <<EOF\nheredoc\nEOF")
    assert res.exit_code == 0 and res.stdout == "heredoc\n"
    res = _cs("run", "echo (")
    assert res.exit_code in (1, 2) and "syntax error" in res.stderr  # bash 3.2 (macOS) says 1, bash 5 says 2
    res = _cs("run", "echo '@OO@ @INNER@'")
    assert res.stdout == "@OO@ @INNER@\n"
    # `exec`, the command's own EXIT trap and `exit` end only the command's subshell: the wrapper still records the code.
    assert _cs("run", "echo hi; exec sh -c 'exit 3'").exit_code == 3
    res = _cs("run", "trap 'echo cleanup' EXIT; exit 4")
    assert res.exit_code == 4 and res.stdout == "cleanup\n"
    # The command text reaches the container only as the `.cmd` file's content — never as an argument of cs's wrapper, so a `pgrep -f` of it inside the command cannot match the wrapper.
    assert container_script("cmd_x", "M", command="zzq-marker").count(b"zzq-marker") == 1


def test_cs_run_detach_then_logs_follow(env):
    env.active()
    res = _cs("run", "-d", "sleep 1; echo done; exit 5")
    cid = res.stdout.strip()
    assert res.exit_code == 0 and re.fullmatch(r"cmd_[0-9a-f]{16}", cid)
    res = _cs("logs", "-f")  # defaults to the newest running command
    assert res.exit_code == 5 and res.stdout == "done\n" and f"cs: {cid} (running)" in res.stderr
    res = _cs("logs", cid)
    assert res.exit_code == 5 and res.stdout == "done\n"
    res = _cs("commands", "--json")
    row = json.loads(res.stdout)["commands"][0]
    assert row == {"command_id": cid, "status": "exited", "exit_code": 5, "started_at": row["started_at"], "exited_at": row["exited_at"], "command": "sleep 1; echo done; exit 5"}
    res = _cs("logs", cid[:9])  # a unique prefix
    assert res.exit_code == 5 and res.stdout == "done\n"
    # stdout and stderr of a finished command come one after the other — the start of stderr is marked.
    cid = _cs("run", "-d", "echo out; echo err >&2").stdout.strip()
    _cs("logs", "-f", cid)
    assert _cs("logs", cid).stderr == run_mod.STDERR_SEPARATOR.decode() + "err\n"
    assert "sleep 1; echo done; exit 5" in _cs("commands").stdout


def test_a_detached_run_on_a_pending_session_is_queued_and_started_by_the_runner(env, monkeypatch):
    target = env.active()
    rec = env.read("hydra_aaa111")
    env.record("hydra_aaa111", status=SessionStatus.PENDING, job_id="1")
    res = _cs("run", "-d", "--deactivate-after", "echo queued-ran; exit 3")
    cid = res.stdout.strip()
    assert res.exit_code == 0 and f"{cid} queued on hydra_aaa111" in res.stderr
    row = json.loads(_cs("commands", "--json").stdout)["commands"][0]
    assert row["command_id"] == cid and row["status"] == "queued" and row["command"] == "echo queued-ran; exit 3"
    assert "is queued" in _cs("logs", cid).stderr
    # The runner launches the queue once active (its login to the container is a local shell here).
    runner = _load_runner()
    monkeypatch.setattr(runner, "container_login", lambda sess, port: ["env", f"HOME={env.home}", "bash", "-s"])
    rsess = runner.Session(env.base, "hydra_aaa111")
    env.record("hydra_aaa111", status=SessionStatus.ACTIVE, job_id="1", sshd_port=1)
    runner.launch_queued(rsess, 1)
    assert not list(rsess.queue.glob("cmd_*"))
    res = _cs("logs", "-f", cid)
    assert res.exit_code == 3 and res.stdout == "queued-ran\n"
    # --deactivate-after: the runner ends the session once the command's exit has settled and nothing else runs.
    assert (env.logs_dir() / f"{cid}.release").exists() and runner.release_reason(rsess) is None  # .exit too fresh
    old = time.time() - 30
    os.utime(env.logs_dir() / f"{cid}.exit", (old, old))
    assert runner.release_reason(rsess) == f"{cid} ended (cs run --deactivate-after)"
    _write_cmd(env, "cmd_other", pid=_NO_PID, exit_code="0")  # another command that just ended: its epilogue may still start
    assert runner.release_reason(rsess) is None
    os.utime(env.logs_dir() / "cmd_other.exit", (old, old))
    assert runner.release_reason(rsess) is not None
    _write_cmd(env, "cmd_live", pid=_NO_PID)  # a live command holds the session
    assert runner.release_reason(rsess) is None
    # A launch the runner was interrupted in (signal) ends as a failed command, never stuck as queued.
    rsess.queue.mkdir(exist_ok=True)
    (rsess.queue / "cmd_cut").write_text("")
    (env.logs_dir() / "cmd_cut.queued").touch()
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(SystemExit(143)))
    with pytest.raises(SystemExit):
        runner.launch_queued(rsess, 1)
    assert (env.logs_dir() / "cmd_cut.exit").read_text() == "125\n" and not (env.logs_dir() / "cmd_cut.queued").exists() and not list(rsess.queue.iterdir())


def test_queued_commands_can_be_dropped(env, monkeypatch):
    _fast_sleep(monkeypatch)
    env.record("hydra_aaa111", status=SessionStatus.PENDING, job_id="1")
    a, b = (_cs("run", "-d", f"echo {n}").stdout.strip() for n in "ab")
    res = _cs("kill", a)
    assert res.exit_code == 0 and res.stdout == f"{a}: removed from the queue\n"
    assert [r["command_id"] for r in json.loads(_cs("commands", "--json").stdout)["commands"]] == [b]
    res = _cs("deactivate")
    assert f"dropped queued command(s): {b}" in res.stderr
    assert not list((env.base / "sessions" / "hydra_aaa111" / "queue").glob("cmd_*"))
    # A failed launch is the command's result.
    runner = _load_runner()
    monkeypatch.setattr(runner, "container_login", lambda sess, port: ["false"])
    env.record("hydra_aaa111", status=SessionStatus.PENDING, job_id="1")
    c = _cs("run", "-d", "echo c").stdout.strip()
    runner.launch_queued(runner.Session(env.base, "hydra_aaa111"), 1)
    res = _cs("logs", c)
    assert res.exit_code == 125 and "could not be started" in res.stderr


def test_cs_run_on_a_session_that_cannot_run(env):
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    res = _cs("run", "echo hi")
    assert res.exit_code == 2 and "inactive — `cs activate hydra_aaa111` first" in res.stderr


def test_cs_kill_stops_a_running_command(env):
    env.active()
    res = _cs("run", "-d", "sleep 30")
    cid = res.stdout.strip()
    res = _cs("kill", cid)
    assert res.exit_code == 0 and res.stdout == f"{cid}: signalled — exited\n"
    res = _cs("logs", cid)
    assert res.exit_code == 143
    assert _cs("kill").stdout == "nothing to kill — no running commands\n"


def test_cs_kill_delivers_any_signal_to_the_command(env):
    env.active()
    # USR1 (a checkpoint request) reaches the command's handler instead of ending the wrapper with a false "exited 0".
    cid = _cs("run", "-d", "trap 'echo got; exit 5' USR1; sleep 30 & wait").stdout.strip()
    _real_sleep(0.5)
    assert _cs("kill", "-s", "USR1", cid).stdout == f"{cid}: signalled — exited\n"
    res = _cs("logs", cid)
    assert res.exit_code == 5 and res.stdout == "got\n"
    cid = _cs("run", "-d", "sleep 30").stdout.strip()
    _real_sleep(0.5)
    assert _cs("kill", "-s", "HUP", cid).stdout == f"{cid}: signalled — exited\n"
    assert _cs("logs", cid).exit_code == 129


def test_cs_kill_leaves_other_process_groups_alone(env, monkeypatch, tmp_path):
    env.active()
    victim = subprocess.Popen(["sleep", "30"], start_new_session=True)
    d = env.logs_dir()
    # A command from before this container started, whose recorded PID now belongs to an unrelated live process group.
    _write_cmd(env, "cmd_old", pid=str(victim.pid))
    (d / "cmd_old.start").write_text("")
    old = time.time() - 600
    os.utime(d / "cmd_old.start", (old, old))
    sshd_pid = tmp_path / "sshd.pid"
    sshd_pid.write_text("1")
    monkeypatch.setattr(run_mod, "_SSHD_PID", str(sshd_pid))
    # A command of this activation that died without an exit (stale heartbeat) — its PID was reused since.
    _write_cmd(env, "cmd_dead", pid=str(victim.pid), pid_age=120)
    (d / "cmd_dead.start").write_text("")
    # The internal venv snapshot, hidden from `cs commands`, is not the user's to kill.
    _write_cmd(env, "cmd_venvsnap_1", pid=str(victim.pid))
    (d / "cmd_venvsnap_1.start").write_text("")
    try:
        assert _cs("kill").stdout == "nothing to kill — no running commands\n"
        assert _cs("kill", "cmd_old").stdout == "cmd_old: ran in an earlier activation of the session — no longer running\n"
        assert _cs("kill", "cmd_dead").stdout == "cmd_dead: died without recording an exit — nothing to signal\n"
        assert victim.poll() is None
    finally:
        victim.kill()


def test_kill_never_signals_pid_0_or_1(env, tmp_path):
    """A garbage `.pid` must never become `kill -- -1` (every process of the user) or `kill -- -0` (the killer's own group). `kill` is stubbed, so this test cannot signal anything even if the guard breaks."""
    env.active()
    calls = tmp_path / "kill-calls"
    for i, pid in enumerate(["1", "0", "01", "00001", "-1", "4294967297", "18446744073709551617"]):
        _write_cmd(env, f"cmd_x{i}", pid=pid, exit_code="0\n")  # exited — a populated group would mean leftover processes
        _write_cmd(env, f"cmd_y{i}", pid=pid)                    # fresh heartbeat, no exit — "live"
    # The stub reports every process group as populated and every leader as gone — the state in which a real kill would be sent.
    stub = f'kill() {{ echo "$*" >> {calls}; case "$*" in "-0 -- -"*) return 0 ;; "-0 "*) return 1 ;; esac; return 0; }}\n'
    for cid in ["", "cmd_x0", "cmd_y0", "cmd_x2", "cmd_y3", "cmd_y5", "cmd_x6"]:
        subprocess.run(["bash", "-c", stub + run_mod.kill_script(cid, "TERM")], capture_output=True, check=True)
    assert not calls.exists(), calls.read_text()


def test_cs_run_reports_a_silent_death(env):
    env.active()
    res = _cs("run", "-d", "sleep 30")
    cid = res.stdout.strip()
    d = env.logs_dir()
    for _ in range(100):
        if (d / f"{cid}.pid").read_text().strip():
            break
        _real_sleep(0.05)
    os.killpg(int((d / f"{cid}.pid").read_text()), 9)  # SIGKILL: no .exit written
    res = _cs("logs", "-f", cid)
    assert res.exit_code == 137 and "still active, so most likely the OOM-killer" in res.stderr


def test_cs_run_options_after_the_session_and_commands_starting_with_a_dash(env):
    env.active()
    res = _cs("run", "hydra_aaa111", "--", "echo", "hi")
    assert res.exit_code == 0 and res.stdout == "hi\n"
    res = _cs("run", "hydra_aaa111", "-d", "echo hi")
    assert res.exit_code == 0 and re.fullmatch(r"cmd_[0-9a-f]{16}\n", res.stdout)
    res = _cs("run", "hydra_aaa111", "--tail", "1", "printf 'a\\nb\\n'")
    assert res.exit_code == 0 and res.stdout == "b\n"
    res = _cs("run", "hydra_aaa111", "--detatch", "echo hi")
    assert res.exit_code == 2 and "no such option: --detatch" in res.stderr
    assert _cs("run", "-d", "--tail", "2", "echo hi").exit_code == 2
    # The wrapper's eval takes the command as data even when it starts with '-'.
    assert _follow_raw(env, "cmd_dash", command="-weird")[-1] == ("X", "127")


def test_cs_kill_lets_a_graceful_shutdown_finish_before_the_exit_is_recorded(env):
    env.active()
    script = "import signal, sys, time\nsignal.signal(signal.SIGTERM, lambda *a: (time.sleep(4), print('saved', flush=True), sys.exit(0)))\nprint('ready', flush=True)\ntime.sleep(60)\n"
    (env.base / "sessions" / "hydra_aaa111" / "workdir" / "train.py").write_text(script)
    cid = _cs("run", "-d", "python3 train.py").stdout.strip()
    out = env.logs_dir() / f"{cid}.out"
    for _ in range(200):  # the handler is in place once it printed
        if out.exists() and out.read_text():
            break
        _real_sleep(0.05)
    assert "still running after 3s" in _cs("kill", cid).stdout
    res = _cs("logs", "-f", cid)
    assert res.exit_code == 143 and res.stdout == "ready\nsaved\n"
    # A shell loop still stops after the current step.
    cid = _cs("run", "-d", "for i in 1 2 3 4 5 6 7 8 9; do sleep 1; echo $i; done").stdout.strip()
    _real_sleep(1.5)
    assert _cs("kill", cid).stdout == f"{cid}: signalled — exited\n"
    assert _cs("logs", cid).exit_code == 143


def test_cs_kill_reports_processes_that_survive_the_signal(env):
    env.active()
    # TERM ends the command's shell, but a child that ignores TERM lives on.
    cid = _cs("run", "-d", "bash -c 'trap \"\" TERM; sleep 30' & wait").stdout.strip()
    try:
        _real_sleep(0.5)
        assert "its shell exited, but processes of it are still running" in _cs("kill", cid).stdout
        # A follow-up stops them at once (a unique prefix works too).
        assert _cs("kill", "-s", "KILL", cid[:10]).stdout == f"{cid}: signalled the processes it left running after it exited\n"
        _real_sleep(0.3)
        assert _cs("kill", cid).stdout == f"{cid}: already exited — nothing to signal\n"
    finally:
        _cs("kill", "-s", "KILL")
    res = _cs("kill", "cmd_nope")
    assert res.exit_code == 2 and "no command 'cmd_nope'" in res.stderr
    assert _cs("logs", "cmd_nope").exit_code == 2


class _FlakyAccess(LocalAccess):
    """The first container connections fail in a chosen way, then the real thing runs."""

    def __init__(self, cfg, home, failures: list[str]):
        super().__init__(cfg, home)
        self.failures = list(failures)

    def container_argv(self, session_id, socket, remote_cmd):
        argv = super().container_argv(session_id, socket, remote_cmd)
        mode = self.failures.pop(0) if self.failures else None
        if mode == "before_login":  # never reached sshd: nothing ran
            return ["bash", "-c", "cat >/dev/null; echo 'kex_exchange_identification: Connection closed by remote host' >&2; exit 255"]
        if mode == "dropped":  # reached the container, ran the script, the connection died before any reply arrived
            return ["bash", "-c", '"$@" >/dev/null; echo "Connection to cs-x closed by remote host." >&2; exit 255', "x", *argv]
        if mode == "reset_before_running":  # after login, but the script never ran
            return ["bash", "-c", "cat >/dev/null; echo 'Connection reset by peer' >&2; exit 255"]
        return argv


def _flaky(env: Env, failures: list[str]) -> _FlakyAccess:
    acc = _FlakyAccess(env.cluster.cfg, env.home, failures)
    env.cluster.access = acc
    return acc


def test_a_launch_is_repeated_only_after_a_before_login_failure(env, monkeypatch):
    _fast_sleep(monkeypatch)
    env.active()
    count = env.base / "launches"
    _flaky(env, ["before_login", "before_login"])
    res = _cs("run", f"echo x >> {count}; echo hi")
    assert res.exit_code == 0 and res.stdout == "hi\n" and count.read_text() == "x\n"
    assert res.stderr.count("connection hiccup") == 2


def test_a_dropped_launch_is_followed_not_repeated(env, monkeypatch):
    _fast_sleep(monkeypatch)
    env.active()
    count = env.base / "launches"
    _flaky(env, ["dropped"])
    res = _cs("run", f"echo x >> {count}; sleep 0.5; echo hi")
    assert res.exit_code == 0 and count.read_text() == "x\n" and "hi\n" in res.stdout


def test_a_dropped_detached_launch_prints_only_the_command_id(env, monkeypatch):
    _fast_sleep(monkeypatch)
    env.active()
    _flaky(env, ["dropped"])
    res = _cs("run", "-d", "echo hello")
    assert res.exit_code == 0 and re.fullmatch(r"cmd_[0-9a-f]{16}\n", res.stdout)


@pytest.mark.parametrize("rc, out, err, never", [
    (255, False, "kex_exchange_identification: Connection closed by remote host\r\nConnection closed by UNKNOWN port 65535\r\n", True),
    (255, False, "Connection closed by UNKNOWN port 65535\r\n", True),
    # A refused shared-connection session falls back to a direct connection, which may run the command before it drops.
    (255, False, "mux_client_request_session: session request failed: Session open refused by peer\r\nAuthenticated to cs-x (via proxy) using \"publickey\".\r\nConnection closed by UNKNOWN port 65535\r\n", False),
    (255, False, "mux_client_request_session: session request failed: Session open refused by peer\r\n", False),
    (255, True, "kex_exchange_identification: read: Connection reset by peer\r\n", False),
    (1, False, "kex_exchange_identification: Connection closed by remote host\r\n", False),
])
def test_only_a_failed_handshake_counts_as_never_reaching_sshd(rc, out, err, never):
    assert run_mod._never_reached_sshd(rc, out, err) is never


def test_an_unconfirmed_launch_is_reported_never_repeated(env, monkeypatch):
    _fast_sleep(monkeypatch)
    env.active()
    _flaky(env, ["reset_before_running"])
    res = _cs("run", "echo hi")
    assert res.exit_code == 125 and "launch not confirmed" in res.stderr
    assert not list(env.logs_dir().glob("*.start"))


def test_stream_resumes_after_drops_without_duplicating_output(env, monkeypatch):
    _fast_sleep(monkeypatch)
    target = env.active()
    cid = "cmd_g"
    d = _write_cmd(env, cid, out=b"line 1\nprogress 1\rprog", pid=str(os.getpid()))

    def writer():
        _real_sleep(0.4)
        with (d / f"{cid}.out").open("ab") as fh:
            fh.write(b"ress 2\nline 2\n")
        _real_sleep(0.4)
        (d / f"{cid}.exit").write_text("0\n")
    _flaky(env, [])
    real_argv = LocalAccess.container_argv
    calls = []

    def argv(self, sid, sock, cmd):
        calls.append(1)
        a = real_argv(self, sid, sock, cmd)
        # The first connection is cut off after the first frame.
        return ["bash", "-c", '"$@" | head -c 40; exit 255', "x", *a] if len(calls) == 1 else a
    monkeypatch.setattr(_FlakyAccess, "container_argv", argv)
    t = threading.Thread(target=writer)
    t.start()
    out = Output(raw=False)
    from io import BytesIO, TextIOWrapper
    buf = BytesIO()
    monkeypatch.setattr(run_mod.sys, "stdout", TextIOWrapper(buf, write_through=True))
    status = run_mod.stream(env.cluster, target, cid, out)
    out.flush()
    t.join()
    assert status == ("exited", 0) and len(calls) >= 2
    # "progress 1" shows as a progress snapshot when a read came before the writer's second chunk.
    assert buf.getvalue() in (b"line 1\nprogress 2\nline 2\n", b"line 1\nprogress 1\nprogress 2\nline 2\n")


def test_session_gone_mid_stream_gives_the_shared_fs_result(env, monkeypatch):
    _fast_sleep(monkeypatch)
    target = env.active()
    _write_cmd(env, "cmd_h", out=b"partial\n", pid=_NO_PID, exit_code="0\n")
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    _flaky(env, ["before_login"] * 3)
    from io import BytesIO, TextIOWrapper
    buf = BytesIO()
    monkeypatch.setattr(run_mod.sys, "stdout", TextIOWrapper(buf, write_through=True))
    out = Output(raw=False)
    assert run_mod.stream(env.cluster, target, "cmd_h", out) == ("exited", 0)
    out.flush()
    assert buf.getvalue() == b"partial\n"


def test_a_run_on_a_session_whose_job_is_gone_fails_at_once(env, monkeypatch):
    _fast_sleep(monkeypatch)
    env.active()
    monkeypatch.setenv("SQUEUE_STDOUT", "")  # the job is gone; the record never caught up
    _flaky(env, ["before_login"] * 20)
    res = _cs("run", "echo hi")
    assert res.exit_code == 125 and "is inactive — the command was not started" in res.stderr
    assert res.stderr.count("connection hiccup") == 0 and env.read("hydra_aaa111").status == SessionStatus.INACTIVE


def test_giving_up_flushes_tail_output_and_keeps_the_command(env, monkeypatch):
    _fast_sleep(monkeypatch)
    env.active()
    _write_cmd(env, "cmd_i", out=b"a\nb\nc\n", pid=str(os.getpid()))
    _flaky(env, ["before_login"] * 20)
    res = _cs("logs", "-f", "--tail", "2", "cmd_i")
    assert res.exit_code == 125 and res.stdout == "b\nc\n"
    assert "giving up" in res.stderr and "reattach with: cs logs -f hydra_aaa111 cmd_i" in res.stderr


# ---------- logs and commands from the shared FS ----------

def test_logs_reads_from_offsets_and_tails(env):
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    _write_cmd(env, "cmd_a", out=b"l1\nl2\nl3\nl4", err=b"e1\ne2\n", pid=_NO_PID, exit_code="0\n")
    r = sess.logs(env.cluster, ["cmd_a"])
    assert (r["out"], r["err"], r["status"], r["exit_code"], r["offsets"]) == (b"l1\nl2\nl3\nl4", b"e1\ne2\n", "exited", 0, [11, 6])
    assert sess.logs(env.cluster, ["cmd_a"], since=(6, 3))["out"] == b"l3\nl4"
    r = sess.logs(env.cluster, ["hydra_aaa111", "cmd_a"], max_lines=2)
    assert (r["out"], r["err"]) == (b"l3\nl4", b"e1\ne2\n")
    # Offsets never move back: the shared FS can show less than the container already delivered.
    assert sess.logs(env.cluster, ["cmd_a"], since=(100, 100))["offsets"] == [100, 100]
    assert _cs("logs", "a", "b", "c").exit_code == 2


def test_large_logs_arrive_in_bounded_requests(env, monkeypatch):
    import compute_sessions.login_helper as lh
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    big = os.urandom(lh.READ_CAP * 2 + 5)
    progress = b"".join(b"step %d\r" % i for i in range(lh.READ_CAP // 6)) + b"done"  # tqdm-style: no line breaks
    _write_cmd(env, "cmd_big", out=big, err=progress, pid=_NO_PID, exit_code="0\n")
    ops = _count_calls(env, monkeypatch)
    r = sess.logs(env.cluster, ["cmd_big"])
    assert r["out"] == big and r["err"] == progress and r["offsets"] == [len(big), len(progress)] and ops == ["logs"] * 3
    r = sess.logs(env.cluster, ["cmd_big"], max_lines=1)
    assert len(r["err"]) <= lh.READ_CAP and r["err"].endswith(b"done") and r["offsets"] == [len(big), len(progress)]


def test_log_reads_do_not_follow_symlinks_planted_in_the_container(env):
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    secret = env.base / "id_rsa"
    secret.write_text("PRIVATE KEY")
    d = env.logs_dir()
    (d / "cmd_leak.out").symlink_to(secret)
    (d / "cmd_leak.exit").symlink_to(secret)
    (d / "cmd_leak.pid").write_text("1")
    os.mkfifo(d / "cmd_leak.err")  # would block a plain open()
    r = sess.logs(env.cluster, ["cmd_leak"])
    assert r["out"] == b"" and r["err"] == b"" and "PRIVATE" not in json.dumps(sess.list_commands(env.cluster, None))
    (d / "runner-1.out").symlink_to(secret)
    env.record("hydra_aaa111", status=SessionStatus.FAILED)
    assert "failure_log_tail" not in sess.show(env.cluster, "hydra_aaa111", squeue=False, probe=False)


def test_command_states_from_log_files(env):
    env.record("hydra_aaa111", status=SessionStatus.ACTIVE)
    d = env.logs_dir()
    now = time.time()

    def mk(name: str, age: float, content: str = "") -> None:
        (d / name).write_text(content)
        os.utime(d / name, (now - age, now - age))
    (d.parent / ".sshd").mkdir()
    mk("../.sshd/sshd.pid", 200)    # the current container started 200 s ago
    mk("cmd_long.start", 900)
    mk("cmd_long.pid", 15, "1234")   # the heartbeat kept touching it until just before exit
    mk("cmd_long.exit", 10, "0")
    mk("cmd_live.start", 120)
    mk("cmd_live.pid", 3, "5678")    # fresh heartbeat, no .exit → running
    mk("cmd_dead.pid", 500, "42")    # stale heartbeat, no .exit → killed
    mk("cmd_settling.start", 30)
    mk("cmd_settling.pid", 2, "9")
    mk("cmd_settling.exit", 1, "")   # truncated but not yet rewritten → still running
    mk("cmd_venvsnap_1.start", 60)   # internal housekeeping — never listed
    mk("cmd_venvsnap_1.exit", 55, "0")
    data = sess.list_commands(env.cluster, None)
    rows = {r["command_id"]: r for r in data["commands"]}
    assert data["total"] == 4 and "cmd_venvsnap_1" not in rows
    long = rows["cmd_long"]
    dur = (datetime.strptime(long["exited_at"], "%Y-%m-%dT%H:%M:%SZ") - datetime.strptime(long["started_at"], "%Y-%m-%dT%H:%M:%SZ")).total_seconds()
    assert long["status"] == "exited" and long["exit_code"] == 0 and 880 <= dur <= 900  # started_at from .start, not the heartbeat
    assert rows["cmd_live"]["status"] == "running" and not {"exit_code", "exited_at"} & rows["cmd_live"].keys()
    assert rows["cmd_dead"]["status"] == "killed" and "started_at" not in rows["cmd_dead"]
    assert rows["cmd_settling"]["status"] == "running"
    assert [r["command_id"] for r in data["commands"]] == ["cmd_settling", "cmd_live", "cmd_long", "cmd_dead"]
    assert sess.logs(env.cluster, ["cmd_settling"])["status"] == "running"
    # The default command is the newest running one, else the newest.
    assert sess.logs(env.cluster, [])["cid"] == "cmd_settling"
    # A command of an earlier activation cannot run, even while its heartbeat still looks fresh.
    mk("cmd_prev.start", 400)
    mk("cmd_prev.pid", 5, "77")
    r = sess.logs(env.cluster, ["cmd_prev"])
    assert r["status"] == "killed" and r["earlier_activation"]
    assert {c["command_id"]: c["status"] for c in sess.list_commands(env.cluster, None)["commands"]}["cmd_prev"] == "killed"
    # Nor can any command of a session that is no longer active.
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    assert sess.logs(env.cluster, ["cmd_live"])["status"] == "killed"
    assert {c["command_id"]: c["status"] for c in sess.list_commands(env.cluster, None)["commands"]}["cmd_live"] == "killed"


# ---------- workdir transfers ----------

def _transfer_session(env: Env) -> tuple[Path, SessionInfo]:
    info = env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    return env.base / "sessions" / "hydra_aaa111" / "workdir", info


def test_download_dest_conventions(env):
    wd, info = _transfer_session(env)
    proj = env.proj
    (wd / "runs").mkdir()
    (wd / "runs" / "log.csv").write_text("l")
    (wd / "runs" / "result.json").write_text("r")
    dl = lambda src, dst: sess.download(env.cluster, "hydra_aaa111", info, src, dst)  # noqa: E731
    # A destination directory that doesn't exist is an error, not an implicit mkdir -p.
    with pytest.raises(ComputeSessionsError, match="not an existing directory"):
        dl("runs/log.csv", "runs/in_baseline/")
    with pytest.raises(ComputeSessionsError, match="parent directory"):
        dl("runs/log.csv", "runs/in_baseline/log.csv")
    with pytest.raises(ComputeSessionsError, match="does not exist"):
        dl("nope.txt", "x")
    (proj / "runs" / "in_baseline").mkdir(parents=True)
    for f in ("log.csv", "result.json"):
        assert dl(f"runs/{f}", "runs/in_baseline/")["local_path"] == f"runs/in_baseline/{f}"
    assert (proj / "runs" / "in_baseline" / "log.csv").read_text() == "l"
    # A directory is placed AS a directory inside an existing dest — never merged over its contents.
    (proj / "results").mkdir()
    (proj / "results" / "rq1.json").write_text("keep")
    assert dl("runs", "results/")["local_path"] == "results/runs/"
    assert (proj / "results" / "rq1.json").read_text() == "keep" and (proj / "results" / "runs" / "log.csv").exists()
    # Explicit contents-merge via a trailing slash; repeating to the same named dir stays idempotent.
    dl("runs/", "flat")
    assert (proj / "flat" / "log.csv").exists()
    dl("runs", "results/runs")
    assert not (proj / "results" / "runs" / "runs").exists()
    assert dl("runs/log.csv", "results/runs")["local_path"] == "results/runs/log.csv"
    (proj / "clash").write_text("f")
    with pytest.raises(ComputeSessionsError, match="exists as a file"):
        dl("runs", "clash")


def test_cs_cp_copies_between_session_workdirs(env):
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    env.record("hydra_bbb222", status=SessionStatus.INACTIVE)
    src = env.base / "sessions" / "hydra_aaa111" / "workdir"
    dst = env.base / "sessions" / "hydra_bbb222" / "workdir"
    (src / "ckpt" / "adapter").mkdir(parents=True)
    (src / "ckpt" / "adapter" / "w.bin").write_text("w")
    (src / "res.json").write_text("r")
    res = _cs("cp", "hydra_aaa111:ckpt/adapter", "hydra_bbb222:outputs/loras/")
    assert res.exit_code == 0 and res.stdout == "hydra_aaa111:ckpt/adapter -> hydra_bbb222:outputs/loras/adapter/\n"
    assert (dst / "outputs" / "loras" / "adapter" / "w.bin").read_text() == "w"
    assert _cs("cp", "aaa1:res.json", "bbb2:").stdout == "hydra_aaa111:res.json -> hydra_bbb222:res.json\n"
    assert _cs("cp", "hydra_aaa111:ckpt", "hydra_aaa111:ckpt/inner/").exit_code == 2  # into itself
    assert _cs("cp", "hydra_aaa111:nope", "hydra_bbb222:x").exit_code == 2
    assert _cs("cp", "res.json", "hydra_bbb222:x").exit_code == 2  # no session named
    # A link planted in the destination (from inside its container) is replaced, never written through.
    outside = env.base / "outside.txt"
    outside.write_text("keep")
    (dst / "ckpt2").mkdir()
    (dst / "ckpt2" / "w.bin").symlink_to(outside)
    (src / "ckpt2").mkdir()
    (src / "ckpt2" / "w.bin").write_text("new")
    assert _cs("cp", "hydra_aaa111:ckpt2/", "hydra_bbb222:ckpt2").exit_code == 0
    assert outside.read_text() == "keep" and (dst / "ckpt2" / "w.bin").read_text() == "new" and not (dst / "ckpt2" / "w.bin").is_symlink()


def test_cs_delete_moves_stopped_sessions_to_the_trash(env):
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    env.record("hydra_bbb222", status=SessionStatus.ACTIVE, job_id="7")
    old = env.base / "trash" / f"hydra_ccc333.{int(time.time()) - 8 * 86400}"
    old.mkdir(parents=True)
    res = _cs("delete", "hydra_aaa111")
    assert res.exit_code == 0 and "hydra_aaa111 deleted" in res.stdout
    assert not (env.base / "sessions" / "hydra_aaa111").exists() and list((env.base / "trash").glob("hydra_aaa111.*"))
    assert _cs("delete", "hydra_bbb222").exit_code == 2  # live sessions are refused
    assert _cs("delete", "aaa1").exit_code != 0  # never a prefix
    for _ in range(50):
        if not old.exists():
            break
        _real_sleep(0.1)
    assert not old.exists()  # trash entries older than a week are removed


def test_upload_dest_conventions(env):
    wd, info = _transfer_session(env)
    proj = env.proj
    (proj / "ckpt.pt").write_text("w")
    (proj / "adapter").mkdir()
    (proj / "adapter" / "config.json").write_text("c")
    up = lambda src, dst: sess.upload(env.cluster, "hydra_aaa111", info, src, dst)  # noqa: E731
    # Missing destination directories are created: a dir named with '/', else the parent of the named result.
    assert up("ckpt.pt", "outputs/a/b/")["remote_path"] == "outputs/a/b/ckpt.pt" and (wd / "outputs" / "a" / "b" / "ckpt.pt").read_text() == "w"
    assert up("ckpt.pt", "outputs/c/renamed.pt")["remote_path"] == "outputs/c/renamed.pt" and (wd / "outputs" / "c" / "renamed.pt").is_file()
    assert up("ckpt.pt", "outputs/geometry/")["remote_path"] == "outputs/geometry/ckpt.pt"
    (proj / "eval.jsonl").write_text("e")
    up("eval.jsonl", "outputs/geometry/")
    assert (wd / "outputs" / "geometry" / "ckpt.pt").read_text() == "w" and (wd / "outputs" / "geometry" / "eval.jsonl").read_text() == "e"
    # Directory into an existing dir: placed as a directory; a repeat to its own name stays idempotent; a missing last component is created.
    assert up("adapter", "outputs/geometry/")["remote_path"] == "outputs/geometry/adapter/"
    assert (wd / "outputs" / "geometry" / "adapter" / "config.json").exists()
    up("adapter", "outputs/geometry/adapter")
    assert not (wd / "outputs" / "geometry" / "adapter" / "adapter").exists()
    assert up("adapter/", "outputs/merged")["remote_path"] == "outputs/merged/" and (wd / "outputs" / "merged" / "config.json").exists()
    assert up("adapter", "")["remote_path"] == "adapter/"
    assert up("ckpt.pt", "./")["remote_path"] == "ckpt.pt"
    up("ckpt.pt", "clash")
    with pytest.raises(ComputeSessionsError, match="exists as a file"):
        up("adapter", "clash")
    # Through the CLI: several sources need a directory destination.
    res = _cs("upload", "ckpt.pt", "eval.jsonl", "outputs/")
    assert res.exit_code == 0 and res.stdout == "ckpt.pt -> outputs/ckpt.pt\neval.jsonl -> outputs/eval.jsonl\n"
    assert _cs("upload", "ckpt.pt", "eval.jsonl", "outputs").exit_code == 2
    # A symlink is uploaded under its own name, like cp — never over a remote file that happens to share its target's name.
    (wd / "two.txt").write_text("remote only")
    (proj / "nested").mkdir()
    (proj / "nested" / "two.txt").write_text("d2")
    (proj / "alias.txt").symlink_to("nested/two.txt")
    assert up("alias.txt", "")["remote_path"] == "alias.txt"
    assert (wd / "alias.txt").read_text() == "d2" and (wd / "two.txt").read_text() == "remote only"


def test_ls_shows_workdir_relative_names(env):
    wd, _ = _transfer_session(env)
    (wd / "notes.txt").write_text("n")
    res = _cs("ls", "notes.txt")
    assert res.exit_code == 0 and "notes.txt" in res.stdout and str(wd) not in res.stdout
    res = _cs("ls", "*.txt")
    assert res.exit_code == 2 and "wildcards are not expanded" in res.stderr


def test_workdir_tools_refuse_symlink_escapes(env):
    # `[ -d ]`, ls and rsync follow a symlinked directory component; a link planted in the workdir from inside the container must not turn file ops into arbitrary reads/writes of the cluster user's files.
    wd, info = _transfer_session(env)
    proj = env.proj
    secret = env.base / "secret"
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
            sess.download(env.cluster, "hydra_aaa111", info, rel, "out/")
    assert list((proj / "out").iterdir()) == []
    for local, rel in (("ak", "evil/authorized_keys"), ("ak", "evil/"), ("adapter", "okdir/")):
        with pytest.raises(ComputeSessionsError, match="outside the session workdir"):
            sess.upload(env.cluster, "hydra_aaa111", info, local, rel)
    assert sorted(p.name for p in secret.iterdir()) == ["id_rsa"]
    with pytest.raises(ComputeSessionsError, match="outside the session workdir"):
        sess.ls(env.cluster, ["evil"])
    with pytest.raises(ComputeSessionsError, match="not found"):
        sess.ls(env.cluster, ["nope"])
    with pytest.raises(ComputeSessionsError, match="traversal"):
        sess.ls(env.cluster, ["runs/../../x"])
    out = sess.download(env.cluster, "hydra_aaa111", info, "latest/", "out/")
    assert (proj / "out" / "log.csv").read_text() == "l" and out["remote_path"] == "latest"
    assert "log.csv" in sess.ls(env.cluster, ["hydra_aaa111", "latest"])
    sess.upload(env.cluster, "hydra_aaa111", info, "ak", "latest/")
    assert (wd / "runs" / "ak").read_text() == "attacker key"


# ---------- sync ----------

def _count_calls(env: Env, monkeypatch) -> list[str]:
    ops: list[str] = []
    real = HostAccess.call
    monkeypatch.setattr(LocalAccess, "call", lambda self, op, **kw: ops.append(op) or real(self, op, **kw))
    return ops


def test_sync_unwinds_type_changes_and_costs_one_request_when_nothing_was_added_or_removed(env, monkeypatch):
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    wd = env.base / "sessions" / "hydra_aaa111" / "workdir"
    proj = env.proj
    (proj / "foo").mkdir()
    (proj / "foo" / "bar.txt").write_text("in dir")
    _, res = sess.sync(env.cluster, None)
    assert res.added == ["foo/bar.txt"] and (wd / "foo" / "bar.txt").read_text() == "in dir"
    ops = _count_calls(env, monkeypatch)
    (proj / "foo" / "bar.txt").write_text("edited, longer")
    _, res = sess.sync(env.cluster, None)
    assert ops == ["sync_prepare"] and res.updated == ["foo/bar.txt"]
    # rsync refuses to write a file over a non-empty directory: the deletion must come first.
    import shutil
    shutil.rmtree(proj / "foo")
    (proj / "foo").write_text("now a file")
    _, res = sess.sync(env.cluster, None)
    assert res.deleted_outdated == ["foo/bar.txt"] and res.cleanup_errors == []
    assert (wd / "foo").read_text() == "now a file"
    (proj / "foo").unlink()
    (proj / "foo").mkdir()
    (proj / "foo" / "bar.txt").write_text("dir again")
    _, res = sess.sync(env.cluster, None)
    assert res.deleted_outdated == ["foo"] and (wd / "foo" / "bar.txt").read_text() == "dir again"


def test_a_failed_sync_still_records_what_it_may_have_placed(env):
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    wd = env.base / "sessions" / "hydra_aaa111" / "workdir"
    proj = env.proj
    (proj / "ok.txt").write_text("ok")
    (proj / "locked.txt").write_text("x")
    (proj / "locked.txt").chmod(0)
    with pytest.raises(RemoteError):
        sess.sync(env.cluster, None)
    assert (wd / "ok.txt").exists()
    # Removed locally after the failed sync: the next sync must remove it from the workdir too.
    (proj / "locked.txt").unlink()
    (proj / "ok.txt").unlink()
    _, res = sess.sync(env.cluster, None)
    assert res.deleted_outdated == ["locked.txt", "ok.txt"] and not (wd / "ok.txt").exists()
    # A directory sync never placed (an upload) stands where the project now has a file: a clear message, not an rsync dump.
    (wd / "blk" / "sub").mkdir(parents=True)
    (wd / "blk" / "sub" / "a.txt").write_text("uploaded")
    (proj / "blk").write_text("file now")
    with pytest.raises(SessionError, match=r"sync stopped: the workdir has a directory where the project now has a file \(blk\)"):
        sess.sync(env.cluster, None)
    assert (wd / "blk" / "sub" / "a.txt").exists()


def test_sync_delete_never_follows_a_symlink_out_of_the_workdir(env):
    env.record("hydra_aaa111", status=SessionStatus.INACTIVE)
    wd = env.base / "sessions" / "hydra_aaa111" / "workdir"
    proj = env.proj
    for f in ("a/x.txt", "keep/y.txt", "b/z.txt", "c/d/w.txt"):
        (proj / f).parent.mkdir(parents=True, exist_ok=True)
        (proj / f).write_text("v")
    sess.sync(env.cluster, None)
    # From inside the container: dir `a` replaced by a link to a victim dir holding an `x.txt`; an empty output dir the job created.
    victim = env.base / "victim"
    victim.mkdir()
    (victim / "x.txt").write_text("precious")
    import shutil
    shutil.rmtree(wd / "a")
    (wd / "a").symlink_to(victim)
    (wd / "outputs").mkdir()
    for f in ("a/x.txt", "b/z.txt", "c/d/w.txt"):
        (proj / f).unlink()
    _, res = sess.sync(env.cluster, None)
    assert (victim / "x.txt").read_text() == "precious"
    assert res.cleanup_errors and "symlink" in res.cleanup_errors[0]
    assert not (wd / "b").exists() and not (wd / "c").exists()  # emptied by the deletions: pruned
    assert (wd / "outputs").is_dir() and (wd / "keep" / "y.txt").exists()


# ---------- waiting for an allocation ----------

def test_wait_polls_often_but_asks_the_scheduler_rarely(env, monkeypatch):
    env.record("hydra_aaa111", status=SessionStatus.PENDING, job_id="4711")
    clock = [0.0]
    monkeypatch.setattr(sess.time, "monotonic", lambda: clock[0])
    sleeps: list[float] = []

    def sleep(s):
        sleeps.append(s)
        clock[0] += s
        if clock[0] >= 200:
            env.record("hydra_aaa111", status=SessionStatus.ACTIVE, job_id="4711", sshd_port=1)
    squeues: list[float] = []
    real_show = sess.show
    monkeypatch.setattr(sess, "show", lambda c, t, squeue=True, probe=True: (squeues.append(clock[0]) if squeue else None) or real_show(c, t, squeue=squeue, probe=probe))
    monkeypatch.setenv("SQUEUE_STDOUT", "PENDING|2:00:00|Priority")
    polls: list[tuple] = []
    detail = sess.wait_until_settled(env.cluster, "hydra_aaa111", lambda d, t, fresh: polls.append((t, fresh)), sleep=sleep)
    assert detail["info"]["status"] == "active"
    assert set(sleeps[:60]) == {2} and set(sleeps[60:]) == {5}
    assert all(b - a >= 30 for a, b in zip(squeues, squeues[1:], strict=False)) and len(squeues) == 7


def test_wait_reports_queue_state_then_only_changes(env, monkeypatch):
    printed: list[str] = []
    monkeypatch.setattr(cli.click, "echo", lambda msg="", err=False, **k: printed.append(msg))
    reasons = iter(["Priority"] * 6 + ["Resources"] * 4 + [None])
    t = [0.0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: t[0])

    def fake_wait(cluster, sid, on_poll, sleep=None):
        for r in reasons:
            if r is None:
                return {"info": {"status": "active", "gpu": "H100"}}
            t[0] += 30
            on_poll({"job": {"reason": r}}, t[0], True)
    monkeypatch.setattr(cli.sess, "wait_until_settled", fake_wait)
    cli._wait_active("hydra_aaa111")
    assert [m.split(" — ")[-1] for m in printed] == ["Priority"] * 3 + ["Resources", "H100"]


# ---------- CLI surface ----------

def test_cli_run_arg_parsing(monkeypatch):
    """`cs run`'s option parsing stops at the first positional (a `-d` inside the user command stays in the command)."""
    from click.testing import CliRunner
    seen = []
    monkeypatch.setattr(cli.run, "callback", lambda args, detach, tail_n, release: seen.append((args, detach, tail_n)))
    for argv in (["run", "python", "train.py", "-d"], ["run", "-d", "python", "train.py"], ["run", "python", "-c", 'print("hi there")'], ["run", "--tail", "5", "sh", "x.sh", "--tail", "9"]):
        assert CliRunner().invoke(cli.cli, argv).exit_code == 0
    assert seen == [
        (("python", "train.py", "-d"), False, None),
        (("python", "train.py"), True, None),
        (("python", "-c", 'print("hi there")'), False, None),
        (("sh", "x.sh", "--tail", "9"), False, 5),
    ]


def test_cli_helpers():
    # Why a session stopped: the runner's reason, or for an unnamed scheduler signal the limit its partition name encodes (m/h/d).
    ran = lambda p, a, b, reason=None: {"resources": {"partition": p}, "last_activated_at": a, "last_deactivated_at": b, "stop_reason": reason}  # noqa: E731
    assert cli._stop_cause(ran("gpu-5h", "2026-07-23T11:21:09Z", "2026-07-23T16:21:25Z")) == "the gpu-5h partition's time limit"
    assert cli._stop_cause(ran("cpu-9m", "2026-10-10T21:52:18Z", "2026-10-10T22:01:33Z", "signalled by the scheduler (time limit, preemption or scancel)")) == "the cpu-9m partition's time limit"
    assert cli._stop_cause(ran("gpu-test", "2026-07-23T11:00:00Z", "2026-07-23T11:30:00Z", "the partition's time limit")) == "the gpu-test partition's time limit"
    assert cli._stop_cause(ran("gpu-5h", "2026-07-23T11:21:09Z", "2026-07-23T13:00:00Z", "cs deactivate")) == "cs deactivate"
    assert cli._stop_cause(ran("gpu-5h", "2026-07-23T11:21:09Z", "2026-07-23T16:21:25Z", "idle timeout (20 min without activity)")) == "idle timeout (20 min without activity)"
    assert cli._stop_cause(ran("gpu-test", None, None)) is None
    assert cli._res_summary({"partition": "gpu-2d", "gpus": 1, "gpu_type": None, "mem": 96}) == "gpu-2d, 1 gpu, 96G"
    assert cli._split_sources_dest(["a", "b", "dir/"], "u") == (["a", "b"], "dir/")
    assert [fmt_duration(x) for x in (None, 42, 75, 3660, 90000)] == ["-", "42s", "1m15s", "1h01m", "1d1h"]


def test_cli_help_renders_and_config_errors_surface_at_run_time(monkeypatch):
    from click.testing import CliRunner
    runner = CliRunner()
    assert runner.invoke(cli.cli, ["--help"]).exit_code == 0
    assert runner.invoke(cli.cli, ["--version"]).output.startswith("cs, version ")
    monkeypatch.setattr(cli, "_CLUSTER", None)
    monkeypatch.setattr(cli, "_CFG_ERROR", RuntimeError("boom"))
    assert runner.invoke(cli.cli, ["list"]).exit_code == cli.EXIT_TOOL_FAILURE


def test_one_login_node_request_per_read_command(env, monkeypatch):
    env.active()
    _write_cmd(env, "cmd_a", out=b"x\n", pid=_NO_PID, exit_code="0\n")
    for argv in (["list"], ["show"], ["logs", "cmd_a"], ["commands"], ["ls"], ["kill", "cmd_a"]):
        ops = _count_calls(env, monkeypatch)
        _cs(*argv)
        assert len(ops) == 1, (argv, ops)


# ---------- transport ----------

def _rsync_access(monkeypatch, returncodes: list[int]):
    acc = HostAccess(_cfg())
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


def test_ssh_subprocesses_do_not_drain_inherited_stdin():
    # ssh forwards whatever stdin it gets: `… | while read s; do cs deactivate "$s"; done` must act on every line.
    r, w = os.pipe()
    os.write(w, b"next-session-id\n")
    os.close(w)
    saved = os.dup(0)
    os.dup2(r, 0)
    try:
        assert ssh_mod._run(["cat"], timeout=5).stdout == b""
        assert ssh_mod._run(["cat"], timeout=5, input=b"body").stdout == b"body"
        assert os.read(0, 100) == b"next-session-id\n"
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        os.close(r)


def test_container_ssh_escapes_the_login_control_path_in_the_proxy_command(monkeypatch):
    acc = HostAccess(_cfg(control_path="~/.ssh/cm-%r@%h:%p"))
    acc.user = "jonas"
    argv = acc.container_argv("hydra_aaa111", "/home/j/.cs/sessions/hydra_aaa111/session.sock", "bash -l -s")
    proxy = next(a for a in argv if a.startswith("ProxyCommand="))
    assert "cm-%%r@%%h:%%p" in proxy and "-W /home/j/.cs/sessions/hydra_aaa111/session.sock hydra" in proxy and "ControlMaster=no" in proxy
    assert argv[-4:] == ["-l", "jonas", "cs-hydra_aaa111", "bash -l -s"]
    monkeypatch.setattr(ssh_mod, "_ssh_version", lambda: (9, 3))
    with pytest.raises(RemoteError, match="needs OpenSSH 9.4 or newer"):
        acc.container_argv("hydra_aaa111", "/x.sock", "bash -l -s")


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
    monkeypatch.setattr(runner.subprocess, "run", fake_smi("GPU 0: NVIDIA H100 (UUID: GPU-1)\nGPU 1: NVIDIA H100 (UUID: GPU-2)\n"))
    assert runner.detect_gpu_models() == "2x NVIDIA H100"
    monkeypatch.setattr(runner.subprocess, "run", fake_smi(""))
    assert runner.detect_gpu_models() == ""
    # MIG node: parent GPUs stay visible to a 1-slice job — the allocated MIG device must win over the "8x A100" parent listing.
    mig_out = "".join(f"GPU {i}: NVIDIA A100 80GB PCIe (UUID: GPU-{i})\n" for i in range(8)) + "  MIG 3g.40gb     Device  0: (UUID: MIG-abc)\n  MIG 3g.40gb     Device  1: (UUID: MIG-def)\n"
    monkeypatch.setattr(runner.subprocess, "run", fake_smi(mig_out))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-abc")
    assert runner.detect_gpu_models() == "MIG 3g.40gb (NVIDIA A100 80GB PCIe)"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert runner.detect_gpu_models() == "2x MIG 3g.40gb (NVIDIA A100 80GB PCIe)"


def test_runner_detects_cuda_version(monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(runner.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    for header, version in [
        ("| NVIDIA-SMI 595.71.05    Driver Version: 595.71.05    CUDA Version: 13.2     |", "13.2"),
        ("| NVIDIA-SMI 610.43.02    Driver Version: 610.43.02    CUDA UMD Version: 13.3     |", "13.3"),  # newer drivers
        ("", ""),
    ]:
        monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=header + "\n"))
        assert runner.detect_cuda_version() == version


def test_runner_busy_means_a_running_command(tmp_path):
    runner = _load_runner()
    sess = runner.Session(tmp_path, "s1")
    sess.logdir.mkdir(parents=True)
    assert not runner.command_live(sess) and not runner.recent_file_ops(sess)
    sess.activity.touch()  # a file op holds the idle timer but is not "busy"
    assert runner.recent_file_ops(sess) and not runner.command_live(sess)
    (sess.logdir / "cmd_a.pid").touch()
    assert runner.command_live(sess)
    (sess.logdir / "cmd_a.exit").write_text("0")
    assert not runner.command_live(sess)


def test_runner_gpu_check_passes_without_a_cuda_driver(tmp_path):
    runner = _load_runner()
    assert runner.gpu_check() is None  # no libcuda here: nothing to check
    sess = runner.Session(tmp_path, "s1")
    runner.record_bad_node(sess, "head023", "cuInit failed: CUDA_ERROR_NO_DEVICE")
    assert json.loads((tmp_path / "bad-nodes.json").read_text())["head023"]["reason"] == "cuInit failed: CUDA_ERROR_NO_DEVICE"


def test_runner_final_status_write_is_a_compare_and_set(tmp_path):
    runner = _load_runner()
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    s = runner.Session(tmp_path, "s1")
    own = {"job_id": ("", None, "1")}
    s.config_path.write_text(json.dumps({"status": "active", "job_id": "1"}))
    assert s.config_update(own, status="inactive") is True and s.config()["status"] == "inactive"
    s.config_path.write_text(json.dumps({"status": "pending"}))   # job_id not stamped yet (very early failure)
    assert s.config_update(own, status="failed") is True and s.config()["status"] == "failed"
    # Re-activated meanwhile (a newer job owns the record): the teardown of job 1 must leave it alone.
    s.config_path.write_text(json.dumps({"status": "pending", "job_id": "2"}))
    assert s.config_update(own, status="inactive") is False and s.config() == {"status": "pending", "job_id": "2"}


# ---------- runner: venv snapshot helper ----------

def _snapshot_env(tmp_path, name: str, cache: Path, freeze: str = "torch==2.0\n", venv: bool = True) -> tuple[str, dict, Path]:
    """One session's view for the snapshot script: its own venv/logs/workdir/lock, the project's shared cache, and fake `uv`/`tar` on PATH."""
    root = tmp_path / name
    for d in ("venv", "logs", "workdir"):
        (root / d).mkdir(parents=True)
    if venv:
        (root / "venv" / "venv").mkdir()
        (root / "venv" / "venv" / "pyvenv.cfg").write_text("version = 3.12.3\n")
    (root / "freeze").write_text(freeze)
    bin_dir = tmp_path / "bin"
    if not bin_dir.exists():
        bin_dir.mkdir()
        (bin_dir / "uv").write_text('#!/bin/bash\ncat "$FAKE_FREEZE"\n')
        (bin_dir / "tar").write_text('#!/bin/bash\nwhile [ $# -gt 0 ]; do [ "$1" = -cf ] && out=$2; shift; done\nsleep "${FAKE_TAR_SLEEP:-0}"\necho archive > "$out"\n')
        for f in bin_dir.iterdir():
            f.chmod(0o755)
    script = _load_runner().VENV_SNAPSHOT_SH
    for var, path in (("VENV", root / "venv"), ("CACHE", cache), ("LOGS", root / "logs"), ("WORKDIR", root / "workdir"), ("LOCKDIR", root / "lock")):
        script = re.sub(rf"^{var}=.*$", f"{var}={path}", script, count=1, flags=re.M)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_FREEZE": str(root / "freeze")}
    return script, env, root


def _run_snapshot(script: str, env: dict, **popen) -> subprocess.Popen:
    return subprocess.Popen(["bash", "-c", script], env=env, **popen)


def test_venv_snapshot_publishes_once_and_adopts_existing_states(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    # No venv: nothing is published (uv would otherwise freeze some other interpreter).
    script, env, root = _snapshot_env(tmp_path, "s0", cache, venv=False)
    assert _run_snapshot(script, env).wait() == 0
    assert list(cache.iterdir()) == []

    script, env, root = _snapshot_env(tmp_path, "s1", cache)
    assert _run_snapshot(script, env).wait() == 0
    (snap,) = cache.glob("*.tar.zst")
    marker = (root / "venv" / ".cs-freeze-hash").read_text().strip()
    assert snap.name.startswith(f"nolock.{marker}.")
    assert [p.read_text().strip() for p in (root / "logs").glob("cmd_venvsnap_*.exit")] == ["0"]

    # Another session with the same packages adopts the published archive (touched to newest) instead of writing a second one.
    old = time.time() - 600
    os.utime(snap, (old, old))
    script, env, root2 = _snapshot_env(tmp_path, "s2", cache)
    assert _run_snapshot(script, env).wait() == 0
    assert list(cache.glob("*.tar.zst")) == [snap] and snap.stat().st_mtime > old + 300
    assert (root2 / "venv" / ".cs-freeze-hash").read_text().strip() == marker
    assert list((root2 / "logs").iterdir()) == []  # adopting is not a tracked command


def test_venv_snapshot_parallel_sessions_archive_a_state_once(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    runs = [_snapshot_env(tmp_path, f"s{i}", cache) for i in range(4)]
    procs = [_run_snapshot(script, {**env, "FAKE_TAR_SLEEP": "1"}) for script, env, _ in runs]
    assert [p.wait() for p in procs] == [0] * 4
    assert len(list(cache.glob("*.tar.zst"))) == 1
    assert not list(cache.glob(".claim.*"))
    # The sessions that found the claim taken have no marker yet; their next command's epilogue adopts the published archive.
    script, env, root = runs[0]
    for script, env, root in runs:
        assert _run_snapshot(script, env).wait() == 0
        assert (root / "venv" / ".cs-freeze-hash").exists()
    assert len(list(cache.glob("*.tar.zst"))) == 1


def test_venv_snapshot_claims_go_stale_when_the_owner_dies(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    script, env, root = _snapshot_env(tmp_path, "s1", cache)
    # Owner killed mid-archive (session teardown: the EXIT trap never runs) — its claim stays behind.
    p = _run_snapshot(script, {**env, "FAKE_TAR_SLEEP": "30"}, start_new_session=True)
    for _ in range(100):
        if list(cache.glob(".claim.*")):
            break
        time.sleep(0.05)
    os.killpg(p.pid, 9)
    p.wait()
    (claim,) = cache.glob(".claim.*")
    assert (root / "lock").is_dir()  # the dead owner's session lock is left behind too
    script2, env2, root2 = _snapshot_env(tmp_path, "s2", cache)
    # A freshly touched claim means a live owner: back off without a marker.
    assert _run_snapshot(script2, env2).wait() == 0
    assert not list(cache.glob("*.tar.zst")) and not (root2 / "venv" / ".cs-freeze-hash").exists()
    # Untouched for over 2 minutes: the killed session itself takes over its dead lock and the stale claim, archives, releases both.
    old = time.time() - 180
    os.utime(claim, (old, old))
    assert _run_snapshot(script, env).wait() == 0
    assert len(list(cache.glob("*.tar.zst"))) == 1 and not claim.exists() and not (root / "lock").exists()
    assert _run_snapshot(script2, env2).wait() == 0  # the other session adopts it
    assert (root2 / "venv" / ".cs-freeze-hash").exists() and len(list(cache.glob("*.tar.zst"))) == 1


def test_runner_sshd_config_allows_parallel_streams_and_drops_dead_clients(tmp_path, monkeypatch):
    runner = _load_runner()
    sess = runner.Session(tmp_path, "s1")
    monkeypatch.setattr(runner.Path, "home", lambda: tmp_path)
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "authorized_keys").write_text("ssh-ed25519 AAAA test\n")
    runner.prepare_sshd(sess, 2222, {"CUDA_VISIBLE_DEVICES": "0", "CS_CUDA_VERSION": ""})
    # The runner's own key (for launching queued commands) may log in from the node itself only; it lives outside the dirs bound into the container.
    keys = (sess.sshd_dir / "authorized_keys").read_text().splitlines()
    assert keys[0] == "ssh-ed25519 AAAA test" and keys[-1].startswith('from="127.0.0.1" ssh-ed25519 ') and sess.runner_key.parent == sess.dir
    lines = (sess.sshd_dir / "sshd_config").read_text().splitlines()
    assert {"MaxSessions 100", "ClientAliveInterval 15", "ClientAliveCountMax 3", "SetEnv CUDA_VISIBLE_DEVICES=0"} <= set(lines)
