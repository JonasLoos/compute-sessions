---
name: setup-cluster
description: Set up a SLURM cluster for compute-sessions - probes it over SSH, prepares it (dirs, internal key, runner, Apptainer image), writes the config file, installs the client interface (cs CLI + skill), and smoke-tests a session. Use when the user wants to configure their cluster for compute-sessions or do first-time setup of this project.
---

# Set up the cluster

You are configuring a SLURM cluster for compute-sessions. Sessions are sbatch jobs running an Apptainer container with sshd, reached via a reverse-SSH tunnel socket on the login node's shared FS.

Everything on the cluster lives under `remote_base` (default `~/.compute-sessions`); deleting that directory wipes all state. The local config is one TOML file at `~/.config/compute-sessions/config.toml`.

Work through the steps below. Ask the user for anything you can't determine (ssh alias, cluster docs for partition names).

## 1. Client interface (skip if `cs` is already installed)

```bash
uv tool install compute-sessions      # from PyPI; add --force --from <checkout> when working from a dev checkout of the repo
cs install-skills                     # copies this skill + the compute-sessions usage skill into the standard skills dirs
```

Everything setup needs on the cluster (runner, image recipe) ships with the install — `cs assets` prints the directory; no repo checkout is required. After upgrading compute-sessions, re-run `cs install-skills` and re-upload/rebuild from the new `cs assets`.

Agents learn the `cs` CLI from the installed compute-sessions skill: `cs install-skills` writes to `~/.agents/skills` (read by Codex, Gemini CLI, Cursor, Copilot, opencode, Amp, Goose, Windsurf, and most others) and `~/.claude/skills` (Claude Code, which does not read `~/.agents`). The skill defers partitions/GPU types to `cs <cmd> --help`, which renders live from the config, so config changes need no skill change. For Claude Code, suggest allowlisting `cs` in Bash permissions (`Bash(cs *)`) so sessions don't prompt on every call. An agent without skills support can be pointed at `cs --help` in its instructions file (`AGENTS.md` or equivalent).

## 2. SSH reachability

Key-based `ssh <alias>` to the login node must work non-interactively. If there is no alias yet, add one to local `~/.ssh/config`:

```
Host <alias>
    HostName <fqdn-or-ip>
    User <username>
```

Verify: `ssh <alias> 'echo ok && python3 --version'` (python3 ≥ 3.8 is required on the cluster). If it prompts for a password, guide the user through `ssh-copy-id`.

The in-container sshd reuses the cluster's `~/.ssh/authorized_keys`, so the same key must be present there (it already is if `ssh <alias>` works with a key).

## 3. Base directory + runner

```bash
ssh <alias> 'mkdir -p ~/.compute-sessions/sessions'
scp "$(cs assets)"/runner.py <alias>:~/.compute-sessions/
ssh <alias> 'chmod +x ~/.compute-sessions/runner.py'
```

Re-upload `runner.py` after upgrading compute-sessions (live sessions are unaffected — they run from a spool copy).

## 4. Cluster-side pieces

**Cluster-side SSH alias.** Compute nodes open the reverse tunnel to the login node using an alias (the `login_host` config field, defaults to `host`). Ensure `~/.ssh/config` ON the cluster has a matching `Host` block for it (HostName can be the login node's internal name).

**Internal keypair** — used only inside jobs to open the reverse tunnel:

```bash
ssh <alias> 'ssh-keygen -t ed25519 -f ~/.compute-sessions/cluster-internal -N ""'
```

Then append to `~/.ssh/authorized_keys` on the cluster, restricting the key to forwarding only:

```
command="/bin/false",no-pty,no-X11-forwarding,no-agent-forwarding,permitopen="127.0.0.1:*" <contents of cluster-internal.pub>
```

Note: some clusters' PAM denies command sessions for such keys with an "account expired" banner — harmless; the `-N` tunnel is exempt. Never test this key with a plain `ssh -i cluster-internal <alias> <cmd>`.

**Discover the resource vocabulary** (goes into the config file):
- Partitions: `ssh <alias> 'sinfo -h -o "%P %l"'` (name + wall-clock limit). Pick a sensible `default_partition` (short GPU partition).
- GPU type constraints: `ssh <alias> 'sinfo -h -o "%N %f" | tr , "\n" | sort -u'` (node feature tags usable with `sbatch --constraint`), and/or the cluster docs. Only include tags that select GPU models.

**Apptainer image** (must ship `/usr/sbin/sshd`; the provided recipe is Ubuntu 24.04 + CUDA + uv + openssh-server + zstd — no project deps are baked in, one image serves every project):

```bash
ssh <alias> 'mkdir -p ~/.compute-sessions/images'
scp "$(cs assets)"/Apptainer.def <alias>:~/.compute-sessions/Apptainer.def
ssh <alias> 'srun --partition=<a-cpu-partition> apptainer build ~/.compute-sessions/images/default.sif ~/.compute-sessions/Apptainer.def'
```

Building takes several minutes; run it via a compute node (login nodes usually forbid heavy builds). Rebuild only when the `.def` changes.

## 5. Write the config file

Create `~/.config/compute-sessions/config.toml`:

```toml
host = "<alias>"
image = "images/default.sif"
partitions = ["cpu-2h", "gpu-2h", "gpu-test", ...]
default_partition = "gpu-2h"
gpu_types = ["h100", "80gb", ...]
default_mem = 42
max_sessions_per_repo = 3        # active+pending cap per repo; 0 = unlimited
# login_host = "<alias-as-seen-from-compute-nodes>"   # only if it differs from host
```

Validate: `cs list` (any directory — config errors surface here; the CLI picks up config changes immediately).

## 6. Smoke test

Run end-to-end via the CLI, from a scratch project directory (sessions are project-scoped to the cwd):

```bash
mkdir -p /tmp/cs-smoke-test && cd /tmp/cs-smoke-test
cs create --partition <a-fast-test-partition>    # waits for activation
cs run 'nvidia-smi -L; uv --version'
cs deactivate
```

Expect: `create` returns once the session is active, `run` lists the GPU(s) (or a clean no-GPU message on a CPU partition) and exits 0, `deactivate` reports `inactive`. On failure, `cs show` includes the failure log tail — read it, fix, retry. Common causes: missing `~/.ssh/authorized_keys` on the cluster, image missing `/usr/sbin/sshd`, wrong partition names, tunnel key not authorized.

## 7. Wrap up

Tell the user what was configured (host, partitions, defaults).
