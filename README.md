> [!WARNING]
> This tool is experimental. Expect rough edges and breaking changes.


# compute-sessions

Safe, project-scoped compute sessions on your SLURM cluster, driven by the `cs` CLI: for humans, shell pipelines, and agents driving it through their shell tool.

Nothing runs as a service on the cluster — only the ssh invocations the client makes plus the sessions themselves. A **session** is a SLURM job running an Apptainer container with `sshd`, with the project mirrored into its workdir; the client reaches it through a reverse-SSH tunnel over a Unix socket on the login node's shared filesystem. Each `cs` command does its login-node work in one short request (a stdlib Python helper sent along with it, so the login node needs `python3` ≥ 3.8), and `cs run` / `cs logs -f` stream output live through one ssh connection into the container.


## Setup

Requires Python 3.11+, [uv](https://github.com/astral-sh/uv), and OpenSSH 9.4+ on the client. Install the CLI and the agent skills:

```
uv tool install compute-sessions
cs install-skills
```

(`cs install-skills` copies the skills into `~/.agents/skills` — the cross-agent skills directory read by most coding agents — plus `~/.claude/skills` for Claude Code; `--dir` for anywhere else.)

Then let an agent do the rest: ask it to *"set up my cluster for compute-sessions"* — the [`setup-cluster` skill](skills/setup-cluster/SKILL.md) probes the cluster, prepares it (dirs, internal key, `runner.py`, Apptainer image), writes the config file, and smoke-tests a session. Everything setup needs on the cluster ships with the install (`cs assets` prints the directory). The skill doubles as the manual setup reference.

Configuration is one TOML file at `~/.config/compute-sessions/config.toml` — see the [annotated example](examples/config.toml), and [examples/](examples/) for known-good configs for specific clusters. The partitions and GPU types are rendered into `cs create`/`cs activate` `--help`.


## Usage

Inside a git project with a `pyproject.toml`:

```bash
cs create --partition gpu-2h --gpus 1     # start a session; waits until active, prints its id
cs run 'uv sync'                          # install deps into the session's persistent venv
cs run 'uv run python train.py'           # streams output live, exits with the command's code
cs sync                                   # re-mirror local edits before the next run
cs download results out/ && cs deactivate # pull results, stop the session
```

- **Streaming**: `cs run` streams output until the command exits and **exits with its code**. Ctrl-C detaches (the command keeps running; `cs logs -f` reattaches, `cs kill` stops it); `-d` launches detached and prints the `command_id` — on a pending session it queues the command, which starts on its own once the session is active. `--deactivate-after` ends the session once the command is done, so a finished job never holds a GPU until the idle timeout.
- **Session inference**: the session argument is optional everywhere — a full id, a unique prefix (`cs show <host>_8e`, or just the hex tail `8ec5`), or nothing at all (the cwd project's only / only-live session is used).
- **Composability**: ids print to stdout, progress to stderr (`cs create --no-wait` for scripting); `--json` on `list`/`show`/`commands`. Exit codes: the remote command's own; `137` = died without recording an exit (OOM/kill); `2` = invalid request (bad arguments, unknown session/command/path, session in the wrong state); `125` = cs itself failed.
- **Resources are chosen per activation**, not fixed at create: `deactivate`, then `activate --gpus 2 …` restarts the same session — workdir, logs, and command history are preserved. `activate` without resource flags reuses the previous activation's resources.
- **Sync is one-way** (local → cluster) and never deletes files the session created — only files it previously placed. Use `download` to pull anything back.

`cs --help` / `cs <command> --help` carry the full resource vocabulary, rendered from your config.

| Command | Purpose |
| --- | --- |
| `cs list` | Compact rows for the cwd project's sessions: all live ones (with idle countdown and wall-clock time left, or the scheduler's start estimate while pending) and the 20 most recently stopped (`--all` for every project, `--json` for machines). |
| `cs show` | Full detail for one session: resources + actually-allocated GPU + live job state + sshd probe; `-w` blocks while pending. |
| `cs create` | Create a session for the cwd project and start it (workdir + initial sync + activate). |
| `cs clone` | New session seeded with a server-side copy of another session's workdir (artifact reuse / fan-out without re-uploading). |
| `cs activate` | (Re)start the session with this run's resources (`--partition`/`--gpus`/`--gpu-type`/`--mem`/`--cpus`/`--idle-timeout`; none given = the previous ones). |
| `cs deactivate` | Stop the session (workdir + logs persist). |
| `cs run` | Run a command in the container; streams until it exits and exits with its code. |
| `cs logs` | Read a command's output/status; `-f` follows until it exits. |
| `cs kill` | Signal a running command (or all of them); drops a queued one. |
| `cs commands` | Enumerate a session's commands with their command text (for recovering `command_id`s; a unique id prefix like `cmd_7c71` works). |
| `cs sync` | Resync the project's non-ignored files (everything git doesn't ignore — tracked *and* untracked) into the session's `workdir/`. |
| `cs upload` / `cs download` | Copy files to / from the session `workdir/`, bypassing `.gitignore`. |
| `cs cp` | Copy between two sessions' workdirs on the cluster (`cs cp A:ckpt/ B:outputs/`). |
| `cs delete` | Delete stopped sessions (workdir + logs); kept in `remote_base/trash/` for 7 days. |
| `cs ls` | Directory listing inside the session `workdir/`. |

Session state lives under `remote_base/sessions/$ID/` on the cluster — `cs delete` moves it into `remote_base/trash/`. GPU sessions start with a short CUDA check (context, allocation, memset); a node that fails it is recorded in `remote_base/bad-nodes.json` and left out of new GPU activations for 24 h. The project's venv snapshots (the newest three, each potentially GBs) live under `remote_base/venv-cache/<project_id>/`; parallel sessions of a project share them, and a venv state is archived only once.


## Isolation

Sessions run with **no host `$HOME`, no host env** (Apptainer `--containall --nv`). What's visible inside:

- `/workdir` — the synced project (writable).
- `/cs-venv` — the project venv at `/cs-venv/venv` (`UV_PROJECT_ENVIRONMENT`). It lives on node-local SSD and is snapshotted (tar+zstd) to the shared FS after any command that changes its installed packages, so the install cost is paid once per dependency change, not once per activation.
- Per-session ephemeral home at `$HOME`, backed by `sessions/$ID/home/`.
- A **constant** list of cache/config bind mounts from your real home: `~/.cache/{huggingface,uv}`, `~/.local/share/{huggingface,uv,wandb}`, `~/.config/wandb` — plus node-local `/tmp`.

Only GPU context is forwarded into the environment (`CUDA_VISIBLE_DEVICES`, `CS_CUDA_VERSION`) — never credentials; tools read those from their on-disk config (which is bound in). This means `huggingface-cli login` / `wandb login` on the cluster, *once*, is enough; every session inherits those tokens via the mounted cache dirs, and nothing sensitive ever sits in shell env.

**Changing the isolation surface** requires editing the `ISOLATION_BINDS` list in [`remote/runner.py`](remote/runner.py) and re-uploading — not a field in any config file an agent routinely writes. This is deliberate: it means an agent creating sessions cannot widen its own sandbox.


## Development

From a checkout: `uv tool install --force --from . compute-sessions` installs a pinned snapshot (`cs --version` warns when it drifts from its source repo). Changes to `remote/runner.py` additionally need a re-upload to the cluster's `remote_base` — live sessions are unaffected, they run from a spool copy.
