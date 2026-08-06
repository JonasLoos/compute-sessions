# compute-sessions

> [!WARNING]
> Experimental. The whole tool is young, and the docker and slurm backends especially are lightly tested outside the author's own setups — expect rough edges and breaking changes.

Safe, project-scoped compute sessions on your own compute sources — a SLURM cluster, a gaming PC, rented cloud GPUs — driven by the `cs` CLI: for humans, shell pipelines, and agents driving it through their shell tool.

A **source** is compute the client reaches over plain SSH; nothing runs as a service on it — only ssh invocations the client makes plus the sessions themselves. A **session** is a container running `sshd` on a source, with the project mirrored into its workdir:

- **slurm** sources: the session is a SLURM job running an Apptainer container, reached through a reverse-SSH tunnel over a Unix socket on the login node's shared FS.
- **docker** sources: the session is a Docker container on an always-on host, its sshd published on the host's `127.0.0.1` and reached over the same SSH connection.
- **vast** sources: each activation rents a fresh on-demand [vast.ai](https://vast.ai) instance (the cheapest offer matching the request, within configured spend caps) running the session image directly; deactivation **destroys** it — the only thing that stops vast billing. Session records and a spend ledger live on the local machine; the workdir is ephemeral, so results must be `download`ed before deactivating.

Sessions are pinned to their source (the `session_id` prefix says which); move results between sources with `download` + `upload`.

## Setup

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv). Install the CLI and the agent skills:

```
uv tool install compute-sessions
cs install-skills
```

(`cs install-skills` copies the skills into `~/.agents/skills` — the cross-agent skills directory read by Codex, Gemini CLI, Cursor, Copilot, opencode, Amp, Goose, Windsurf, and others — plus `~/.claude/skills` for Claude Code; `--dir` for anywhere else. From a dev checkout, install with `uv tool install --force --from . compute-sessions` instead.)

Then let an agent do the rest: ask it to *"set up my cluster / gaming PC / vast.ai as a compute source"* — the [`setup-compute-source` skill](.claude/skills/setup-compute-source/SKILL.md) probes the host, prepares it (dirs, keys, `runner.py`, container image), writes the config file, and smoke-tests a session. Everything setup needs on sources ships with the install (`cs assets` prints the directory) — a checkout of this repo is only needed for development. The skill doubles as the manual setup reference.

### Config

One TOML file at `~/.config/compute-sessions/config.toml` (override the path with the `COMPUTE_SESSIONS_CONFIG` env var):

```toml
default_source = "mycluster"     # used by `create` when no source is given
max_sessions_per_repo = 3        # active+pending cap per repo, per source; 0 = unlimited
# control_path = "~/.ssh/cm-%r@%h:%p"   # OpenSSH ControlMaster socket template

[sources.mycluster]
type = "slurm"
host = "mycluster"               # ssh alias (connection details live in ~/.ssh/config)
image = "images/default.sif"     # relative to remote_base
partitions = ["cpu-2h", "gpu-2h", "gpu-test"]
default_partition = "gpu-2h"
gpu_types = ["h100", "80gb", "40gb"]   # sbatch --constraint values
default_mem = 42                 # GB
# remote_base = "~/.compute-sessions"
# login_host = "mycluster"      # alias as seen from compute nodes, if different
# container_user = ""            # in-container login; empty = remote whoami

[sources.gamingpc]
type = "docker"
host = "gamingpc"
image = "compute-sessions:latest"      # local image tag on the host
description = "1x RTX 4090 24GB, free but one session at a time"   # shown in --help

[sources.vast]
type = "vast"                          # no host — instances are rented per activation
image = "docker.io/you/compute-sessions-vast:latest"   # public registry; runner baked in (remote/Dockerfile.vast)
description = "rented GPUs - costly; prefer the cluster when it's up"
gpu_types = ["RTX_4090", "RTX_5090", "H100_SXM"]   # vast gpu_name values, underscored
default_gpu_type = ""                  # empty = cheapest offer with any GPU
disk = 32                              # GB
max_instance_price = 1.5               # $/hr per instance — required
max_hourly_spend = 3.0                 # $/hr across running instances — required
max_total_spend = 100.0                # $/calendar month (local ledger estimate) — required
max_session_hours = 12                 # hard self-destruct age
# offer_filter = "verified=true reliability>0.98"   # extra offer-search clauses
# api_key_file = "~/.config/vastai/vast_api_key"
```

Known-good configs for specific clusters live in [examples/](examples/). Source names must be lowercase alphanumeric (no separators) — they prefix session ids. Per-source vocabularies (partitions, GPU types, spend limits) and the free-text `description` are rendered into `cs create`/`cs activate` `--help` — the description is how you steer agents between sources ("free but slow", "costly, use sparingly").

## CLI (`cs`)

On PATH via the `uv tool install` above (or `uv run cs …` from a checkout).

- **Streaming**: `cs run [SESSION] CMD...` streams output live until the command exits and **exits with its code**. Ctrl-C detaches (the command keeps running; `cs logs -f` reattaches, `cs kill` stops it). `-d` launches detached and prints the `command_id`.
- **Session inference**: the session argument is optional everywhere — a full id, a unique prefix (`cs show mycluster_db`), or nothing at all (the cwd project's only / only-live session is used).
- **Composability**: ids print to stdout, progress to stderr (`cs create` waits for activation by default; `--no-wait` for scripting); `--json` on `list`/`show`/`commands`. Exit codes: the remote command's own; `137` = died without recording an exit (OOM/kill); `125` = cs itself failed.

```bash
cs create --partition gpu-2h --gpus 1     # waits until active, prints the session id
cs run 'uv sync'                          # streams; session inferred from the cwd project
cs run -d 'uv run python train.py'        # detached; prints command_id
cs logs -f                                # follow the most recent command
cs download results out/ && cs deactivate
```

`cs --help` / `cs <command> --help` carry the full per-source resource vocabulary, rendered from your config.

## Usual Workflow

Inside a git project with a `pyproject.toml`:

1. `cs create --partition gpu-test` — or `cs create --source gamingpc` / `cs create --source vast --gpu-type RTX_4090` — starts the session and waits until it's active (on vast, the first contact populates the fresh workdir).
2. `cs run 'uv sync'` — materializes the deps into the venv. On slurm sources the venv lives on node-local SSD and is snapshotted (tar+zstd) to the shared FS after any command that changes it, so the install cost is paid once per dependency change, not once per activation; on docker sources it simply persists on local disk; on vast it lives and dies with the instance.
3. `cs sync && cs run 'uv run python train.py'` — resync local edits before running.
4. `cs deactivate` — stops the session. `workdir/` and `logs/` persist on slurm/docker sources; on vast the instance is destroyed (billing stops) — `cs download` results first.

## Commands

| Command | Purpose |
| --- | --- |
| `cs list` | Compact rows for the cwd project's sessions (`--all` for every project, `--json` for machines). |
| `cs show` | Full detail for one session: resources + actually-allocated GPU + live job state + sshd probe; `-w` blocks while pending. |
| `cs create` | Create a session on a source and start it (workdir + config.json + initial sync + activate). |
| `cs clone` | New session seeded with a server-side copy of another session's workdir (same source; artifact reuse / fan-out without re-uploading). |
| `cs activate` | (Re)start the session. Sets this run's resources — on slurm sources pass `--partition`/`--gpus`/`--gpu-type`/`--mem`; `--idle-timeout` works everywhere. |
| `cs deactivate` | Stop the session (workdir + logs persist). |
| `cs run` | Run a command in the container; streams until it exits and exits with its code. |
| `cs logs` | Read a command's output/status; `-f` follows until it exits. |
| `cs kill` | Signal a running command (or all of them). |
| `cs commands` | Enumerate a session's commands (for recovering `command_id`s). |
| `cs sync` | Resync the project's non-ignored files (everything git doesn't ignore — tracked *and* untracked) into the session's `workdir/`. |
| `cs upload` / `cs download` | Copy files to / from the session `workdir/`, bypassing `.gitignore`. |
| `cs ls` | Directory listing inside the session `workdir/`. |

## Deploying changes

The installed `cs` should run from a pinned snapshot, not live from this repo — `uv tool install --force --from <this repo> compute-sessions` installs/refreshes it (binary at `~/.local/bin/cs`; an installed `cs` warns when it drifts from its source repo). Changes to `remote/runner*.py` additionally need a re-upload to each source's `remote_base` (the setup skill does this; a plain `scp` works too — live sessions are unaffected, they run from a spool copy).

## Spend limits (vast)

Rented GPUs bill by the second, so vast sources require explicit caps (config parsing fails without them). Three gates run before every activation: `max_instance_price` bounds the offer search, `max_hourly_spend` bounds the summed rate of running instances (checked against the live vast API, so leaked instances count), and `max_total_spend` bounds the calendar month, estimated from an append-only ledger at `remote_base/<name>-ledger.jsonl` that is reconciled against the API on every rental. On the instance itself, the runner self-destructs on idle timeout **and** at `max_session_hours` — using the vast-injected per-instance API key, so cost stays bounded even if your machine is offline. Worst case for a forgotten session: `max_instance_price × max_session_hours`.

## Isolation

Sessions run with **no host `$HOME`, no host env** (Apptainer `--containall --nv` on slurm; a non-root, user-mapped container on docker sources; a root container on vast — the whole box is rented and ephemeral). What's visible inside slurm/docker sessions:

- `/workdir` — the synced project (writable).
- `/cs-venv` — the project venv at `/cs-venv/venv` (`UV_PROJECT_ENVIRONMENT`; one level below the mount so uv can delete/recreate the env). On slurm sources this is a plain directory on node-local `/tmp`, restored from the project's newest snapshot at activation (preferring one matching the current `uv.lock`) and re-archived to `venv-cache/` on the shared FS after any command that changes its installed packages. On docker sources it's a persistent per-project directory under `remote_base/venvs/` — no snapshot machinery needed.
- Per-session ephemeral home at `$HOME`, backed by `sessions/$ID/home/`. Wiped when the session dir is deleted.
- A **constant** list of cache/config bind mounts from your real home: `~/.cache/{huggingface,uv}`, `~/.local/share/{huggingface,uv,wandb}`, `~/.config/wandb` (plus node-local `/tmp` on slurm). Binds whose source doesn't exist are created first so caches persist from the first session on.

Only GPU context is forwarded into the environment (`CUDA_VISIBLE_DEVICES` on slurm, `CS_CUDA_VERSION` everywhere) — never credentials; tools read those from their on-disk config (which is bound in). This means `huggingface-cli login` / `wandb login` on the source, *once*, is enough; every session inherits those tokens via the mounted cache dirs, and nothing sensitive ever sits in shell env.

Vast sessions have no bind mounts at all — nothing from your machine reaches the instance except the project files you sync/upload and your ssh public keys. Consequently there are no inherited HF/wandb tokens either: jobs needing them must bring them explicitly (e.g. `upload` a `.env`), which is deliberate on hardware operated by unknown third parties.

**Changing the isolation surface** requires editing the `ISOLATION_BINDS` list in [`remote/runner.py`](remote/runner.py) and re-uploading — not a field in any config file an agent routinely writes. This is deliberate: it means an agent creating sessions cannot widen its own sandbox.

## Notes

- Compute resources are chosen **per activation**, not fixed at create. To restart a session with different resources, `deactivate` then `activate(session_id, …)` — the `workdir/`, logs, and command history are preserved (`activate` refuses while the session is still active). On slurm sources the wall-clock limit is encoded in the partition name; omitting `gpu_type` yields any available GPU (faster allocation).
- Sync is one-way (local → source). Use `download` to pull anything back.
- Sync never deletes files the session created — only files it previously placed, per a per-session `.sync-manifest`.
- Concurrent `run` calls on the same session are supported.
- Session state is fully self-contained under `remote_base/sessions/$ID/` on its source (for vast: on the local machine, holding the record plus logs salvaged at deactivate). `rm -rf` that directory to clean up.
- Vast file tools while inactive: `logs`/`list_commands` read the salvaged copies; `ls`/`download`/`upload`/`sync` need a live instance (the workdir died with the last one).
