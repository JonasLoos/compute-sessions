---
name: setup-compute-source
description: Set up a new compute source (SLURM cluster, always-on Docker host like a gaming PC, or vast.ai GPU rentals) for compute-sessions - probes the host over SSH, prepares it (dirs, keys, runner, container image), writes the config file, installs the client interface (cs CLI + skill), and smoke-tests a session. Use when the user wants to add/configure a cluster, gaming PC, rented GPUs, or other compute as a compute source, or do first-time setup of this project.
---

# Set up a compute source

You are configuring a compute source for compute-sessions. A source is one of:
- **slurm** — a SLURM cluster. Sessions are sbatch jobs running an Apptainer container with sshd, reached via a reverse-SSH tunnel socket on the login node.
- **docker** — an always-on SSH host (gaming PC with WSL, workstation). Sessions are Docker containers with sshd published on the host's 127.0.0.1.
- **vast** — GPU rentals on vast.ai. Every activation rents a fresh on-demand instance (cheapest offer matching the request, within configured spend caps); deactivation destroys it — that is the only thing that stops vast billing. There is no host to prepare: skip steps 2-3 and go to 4c.

For slurm/docker, everything on the source lives under `remote_base` (default `~/.compute-sessions`); deleting that directory wipes all state. For vast, the same directory ON THIS MACHINE holds session records and the spend ledger (instances are ephemeral). The local config is one TOML file at `~/.config/compute-sessions/config.toml`.

Work through the steps for the chosen type. Ask the user for anything you can't determine (source name, host alias, cluster docs for partition names, spend limits). Source names must be lowercase alphanumeric with no separators (`mycluster`, `gamingpc`, `vast`) — they prefix session ids.

## 1. Client interface (skip if `cs` is already installed)

```bash
uv tool install compute-sessions      # from PyPI; add --force --from <checkout> when working from a dev checkout of the repo
cs install-skills                     # copies this skill + the compute-sessions usage skill into ~/.claude/skills
```

Everything setup needs on sources (runner, image recipes) ships with the install — `cs assets` prints the directory; no repo checkout is required. After upgrading compute-sessions, re-run `cs install-skills` and re-upload/rebuild from the new `cs assets`.

**Claude Code** learns the `cs` CLI from the installed compute-sessions skill. New sources need no skill change — the skill defers sources/partitions/GPU types to `cs <cmd> --help`, which renders live from the config. Suggest allowlisting `cs` in Bash permissions (`Bash(cs *)`) so sessions don't prompt on every call. Other agents drive `cs` through their shell tool — point them at `cs --help` (or an equivalent of the skill doc) in their instructions file.

## 2. SSH reachability (slurm + docker)

Key-based `ssh <alias>` must work non-interactively. If there is no alias yet, add one to local `~/.ssh/config`:

```
Host <alias>
    HostName <fqdn-or-ip>
    User <username>
```

Verify: `ssh <alias> 'echo ok && python3 --version'` (python3 ≥ 3.8 is required on the host). If it prompts for a password, guide the user through `ssh-copy-id`. For a WSL host, sshd must run inside WSL and the Windows side must forward the port (or use tailscale etc.) — that is the user's setup to confirm; just test that `ssh <alias>` lands in a Linux shell with `docker` available.

The in-container sshd reuses the source's `~/.ssh/authorized_keys`, so the same key must be present there (it already is if `ssh <alias>` works with a key).

## 3. Base directory + runner (slurm + docker)

```bash
ssh <alias> 'mkdir -p ~/.compute-sessions/sessions'
scp "$(cs assets)"/runner.py "$(cs assets)"/runner_slurm.py "$(cs assets)"/runner_docker.py <alias>:~/.compute-sessions/
ssh <alias> 'chmod +x ~/.compute-sessions/runner.py'
```

`runner.py` is the shared entry point; it imports the matching `runner_<type>.py` from the same directory. Re-upload them after upgrading compute-sessions.

## 4a. SLURM-specific setup

**Cluster-side SSH alias.** Compute nodes open the reverse tunnel to the login node using an alias (the `login_host` config field, defaults to the source's `host` value). Ensure `~/.ssh/config` ON the cluster has a matching `Host` block for it (HostName can be the login node's internal name).

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

**Config entry:**

```toml
[sources.<name>]
type = "slurm"
host = "<alias>"
image = "images/default.sif"
partitions = ["cpu-2h", "gpu-2h", "gpu-test", ...]
default_partition = "gpu-2h"
gpu_types = ["h100", "80gb", ...]
default_mem = 42
# login_host = "<alias-as-seen-from-compute-nodes>"   # only if it differs from host
```

## 4b. Docker-specific setup

**Check Docker + GPU runtime** on the host:

```bash
ssh <alias> 'docker info >/dev/null && echo docker-ok'
ssh <alias> 'docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi'
```

If the GPU test fails, the NVIDIA container toolkit is missing — help the user install it (on WSL: install `nvidia-container-toolkit` inside the distro, or enable GPU support in Docker Desktop). A CPU-only host is also fine; the runner skips `--gpus` when `nvidia-smi` is absent.

**Build the session image ON the host** (bakes a passwd entry matching the host account — sshd runs as that non-root user):

```bash
scp "$(cs assets)"/Dockerfile <alias>:~/.compute-sessions/Dockerfile
ssh <alias> 'cd ~/.compute-sessions && docker build -t compute-sessions:latest --build-arg USERNAME=$(whoami) --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -f Dockerfile .'
```

Rebuild when the Dockerfile changes or the host account changes.

**Config entry:**

```toml
[sources.<name>]
type = "docker"
host = "<alias>"
image = "compute-sessions:latest"
gpu_desc = "<e.g. 1x RTX 4090 24GB>"   # informational, shown in tool docs
```

Known limitation: the idle monitor counts established sshd connections via `/proc/net/tcp`, which misses connections when Docker Desktop proxies ports outside the WSL distro. Command activity and file-tool activity are still tracked, so idle shutdown works; only long-lived *interactive* ssh sessions might not count as activity there. Docker Engine installed inside WSL avoids this.

## 4c. Vast-specific setup

**API key.** The user creates one at https://cloud.vast.ai → Keys (full-access key; the backend rents/destroys instances with it). Store it where the vast CLI would:

```bash
mkdir -p ~/.config/vastai && touch ~/.config/vastai/vast_api_key && chmod 600 ~/.config/vastai/vast_api_key
# then paste the key into that file (one line)
```

Verify: `curl -s -H "Authorization: Bearer $(cat ~/.config/vastai/vast_api_key)" https://console.vast.ai/api/v0/instances/ | head -c 200` — expect JSON, not an auth error. The account also needs billing credit.

**Session image.** The vast image bakes the runner in (an instance has no reachable machine to copy it from at boot), so it must live in a public registry the user controls. Build on any amd64-capable docker (a gaming-PC docker source works well; Apple Silicon needs `--platform linux/amd64` emulation and is slow):

```bash
docker build --platform linux/amd64 -f "$(cs assets)"/Dockerfile.vast -t docker.io/<dockerhub-user>/compute-sessions-vast:latest "$(cs assets)"
docker push docker.io/<dockerhub-user>/compute-sessions-vast:latest
```

ghcr.io works too (make the package public). Rebuild and push after upgrading compute-sessions (the image bakes in `Dockerfile.vast`, `runner.py`, and `runner_vast.py`).

**Spend limits** — ask the user, don't guess; the first three are mandatory (config parsing fails without them):
- `max_instance_price` ($/hr per instance — also caps the offer search)
- `max_hourly_spend` ($/hr summed over all running instances)
- `max_total_spend` ($ per calendar month, estimated from a local ledger at `~/.compute-sessions/<name>-ledger.jsonl`)
- `max_session_hours` (default 12): instances hard self-destruct at this age regardless of activity — the worst case for one forgotten session is `max_instance_price × max_session_hours`.

**GPU allowlist.** `gpu_types` are vast `gpu_name` values with underscores for spaces (`RTX_4090`, `RTX_5090`, `H100_SXM`, `A100_SXM4`). Check current market names and prices at https://cloud.vast.ai/create/. An empty `default_gpu_type` means "cheapest offer with any GPU" — usually right.

**Config entry:**

```toml
[sources.<name>]
type = "vast"
image = "docker.io/<dockerhub-user>/compute-sessions-vast:latest"
description = "rented GPUs - costly; <note when to prefer other sources>"
gpu_types = ["RTX_4090", "RTX_5090", "H100_SXM"]
default_gpu_type = ""              # empty = cheapest available GPU
disk = 32                          # GB
max_instance_price = 1.5
max_hourly_spend = 3.0
max_total_spend = 100.0
max_session_hours = 12
# offer_filter = "verified=true reliability>0.98"   # the default; add e.g. inet_down>500 geolocation=EU
```

No `host`, no runner upload, no internal keys. Make sure the user understands the model: **deactivate destroys the instance** — workdir, venv, and uploads are gone (command logs are salvaged locally); results must be `download`ed (or pushed to HF/wandb by the job itself) before deactivating. The idle monitor and the `max_session_hours` cap both self-destruct the instance from inside via the vast-injected per-instance API key, so cost stays bounded even if this machine goes offline.

## 5. Write the config file

Create or extend `~/.config/compute-sessions/config.toml`. Global keys (top of file):

```toml
default_source = "<name>"        # which source `create` uses when none is given
max_sessions_per_repo = 3        # per source; 0 = unlimited
```

Append the `[sources.<name>]` table from step 4. Validate: `cs list` (any directory — config errors surface here; the CLI picks up config changes immediately).

## 6. Smoke test

Run end-to-end via the CLI, from a scratch project directory (sessions are project-scoped to the cwd):

```bash
mkdir -p /tmp/cs-smoke-test && cd /tmp/cs-smoke-test
cs create --source <name>        # slurm: add --partition <a-fast-test-partition>; waits for activation
cs run 'nvidia-smi -L; uv --version'
cs deactivate
```

Expect: `create` returns once the session is active, `run` lists the GPU(s) (or a clean no-GPU message on CPU hosts) and exits 0, `deactivate` reports `inactive`. On failure, `cs show` includes the failure log tail — read it, fix, retry. Common causes: missing `~/.ssh/authorized_keys` on the source, image missing `/usr/sbin/sshd`, wrong partition names, slurm tunnel key not authorized.

Vast notes: the smoke test RENTS A REAL INSTANCE (cents — it activates in ~1-3 min with a cached image, runs one command, and the deactivate destroys it); warn the user before running. If it fails while pending, `deactivate` — never leave the session dangling — and check https://cloud.vast.ai/instances/ afterwards to confirm nothing is still running. Common vast failures: no offers match (caps too tight — the error says which constraint), image not public / not amd64, empty account balance.

## 7. Wrap up

Tell the user what was configured (source name, type, defaults). Note that sessions are pinned to their source — data moves between sources via `cs download` + `cs upload`.
