---
name: compute-sessions
description: Run compute-heavy work (model training, GPU experiments, large data jobs) on the SLURM cluster via the `cs` CLI — creates a container session on the cluster, syncs the project into it, and streams commands with real exit codes. Use when a task needs a GPU or more compute than the local machine, or when the user mentions compute sessions, the cluster, or `cs`.
---

# Compute sessions (`cs`)

`cs` runs commands in a **session**: a container job on the cluster, with the project mirrored into its `/workdir`. `cs --help` and `cs <cmd> --help` are authoritative and render the live config (partitions, GPU types) — consult them for flags, valid values, transfer path rules, and exit codes.

## Workflow

From the project root:

```bash
cs create                        # allocate a session, sync the project, print the session id
cs run uv run python train.py    # streams output; exits with the command's exit code
cs deactivate                    # release when done (workdir persists; `cs activate` re-enters)
```

- The session argument is inferred from the cwd project; with several sessions, pass an id (a unique prefix works).
- `cs sync` re-mirrors edited project files before a re-run; .gitignored data (datasets, weights, .env) needs `cs upload`.
- Resources are set per activation — pass them to `cs create` / `cs activate`; to change them, `cs deactivate` then `cs activate --gpus 2 ...`. Omit `--gpu-type` unless the task truly needs a specific GPU — unconstrained requests allocate much faster. The partition suffix is a hard wall-clock limit that kills the whole session mid-run — size it to the job (the default partition is short).
- Allocation takes seconds to hours: background the blocking `cs create`/`cs activate`, or use `--no-wait` and block later with `cs show -w`.

## Running and monitoring

- Default: run the **streaming** `cs run` as a background task (your harness's background-Bash facility) — completion is pushed to you with the real exit code. For jobs that might outlive the local cs process (laptop sleep, harness restarts): `cs run -d`, then `cs logs -f <cmd_id>` as the background task. After a lost stream, `cs commands` lists ids, `cs logs -f` re-attaches, `cs kill` stops one.
- **Never pipe `cs run`'s own output** through `| tail` / `| grep`: the pipe replaces the command's exit code (a killed job then reads as success) and an early-closing filter kills the stream. Cap output with `cs run --tail N` instead.
- Each `cs run` is a recorded command and resets the idle timer — monitor via logs (reading them never resets it), not by re-running probes in a loop.
- For anything multi-line or with tricky quoting: write a script, `cs sync`, `cs run bash script.sh`.

## Container environment

- No system python — use `uv run` / `uv sync`. The project venv lives at `/cs-venv/venv` (uv targets it automatically; no `uv venv` needed) and persists across deactivate/activate.
- Torch: the driver runs wheels up to CUDA `$CS_CUDA_VERSION` (env var set in every GPU session) — pin torch's index to match (e.g. `[tool.uv.sources]` → https://download.pytorch.org/whl/cu128); the default wheels target newer CUDA and won't run. Guard the pin with a `sys_platform == 'linux'` marker so local (mac) `uv run` keeps working.
- HF downloads go to the shared cache at `~/.cache/huggingface` — leave HF_HOME and cache_dir alone unless the project needs its own cache.
- Don't put secrets (API tokens) in `cs run` command strings — they are recorded verbatim in the session's command log. `cs upload` an env file and source it instead.

## Semantics worth knowing

- Deactivate sessions when finished — idle sessions hold hardware; there is an idle auto-shutdown, but don't rely on it.
