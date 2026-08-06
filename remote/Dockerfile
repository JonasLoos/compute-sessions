# syntax=docker/dockerfile:1
# compute-sessions session image for docker sources (e.g. a gaming PC's WSL).
# Built ON the source host at setup time:
#   docker build -t compute-sessions:latest \
#     --build-arg USERNAME=$(whoami) --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) \
#     -f Dockerfile .
#
# Mirrors remote/Apptainer.def: sshd + Python tooling, nothing project-specific. Each project brings its own deps (and, if needed, its own Python version) via `uv` into /cs-venv/venv (UV_PROJECT_ENVIRONMENT) — bound from a persistent per-project directory on the host, so no snapshot machinery is needed here.
#
# The venv sits one level BELOW the /cs-venv bind so uv owns a regular directory: uv recreates an env by deleting it (python-version change, invalid-env recovery), which fails with EBUSY on a mountpoint.
#
# The container runs as the invoking host user (`docker run --user uid:gid`), with sshd as that same non-root user — which restricts logins to exactly that user. sshd needs a passwd entry for the login name, so the user is baked in at build time to match the host account.

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates curl wget \
        build-essential pkg-config ninja-build \
        git git-lfs \
        openssh-client openssh-server \
        iproute2 \
        jq htop tree \
        ffmpeg libgl1 libglib2.0-0 \
        zip unzip tar zstd; \
    rm -rf /var/lib/apt/lists/*; \
    git lfs install --system; \
    # Bind mountpoints, pre-created so mounts attach cleanly; /run/sshd for sshd's startup check.
    mkdir -p /workdir /cs-venv /cs-logs /sshd /run/sshd

# Match the host account (ubuntu24.04 ships an "ubuntu" user at uid 1000 — rename/reuse on collision instead of failing).
RUN set -eux; \
    if getent group ${USER_GID} >/dev/null; then groupmod -n ${USERNAME} "$(getent group ${USER_GID} | cut -d: -f1)"; \
    else groupadd -g ${USER_GID} ${USERNAME}; fi; \
    if getent passwd ${USER_UID} >/dev/null; then usermod -l ${USERNAME} -d /home/${USERNAME} -m -s /bin/bash -g ${USER_GID} "$(getent passwd ${USER_UID} | cut -d: -f1)"; \
    else useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USERNAME}; fi

# Install uv into a system-wide location. uv itself is a single static binary — no writable state needed at runtime.
RUN UV_INSTALL_DIR=/opt/uv sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'

# Env for `bash -l` (sshd spawns a login shell for each `run` connection). Written to /etc/profile.d/ — same contract as the Apptainer image.
RUN cat > /etc/profile.d/00-cs-env.sh <<'EOF'
export PATH="/opt/uv:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export UV_CACHE_DIR="$HOME/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$HOME/.local/share/uv/python"
export UV_PROJECT_ENVIRONMENT="/cs-venv/venv"
export VIRTUAL_ENV="/cs-venv/venv"
export UV_LINK_MODE="copy"
export HF_HOME="$HOME/.cache/huggingface"
# CUDA_HOME: the devel image ships nvcc but sets no CUDA_HOME, which source builds (flash-attn, deepspeed, …) need. PYTHONUNBUFFERED: so `run`/`logs` capture stdout as it's produced instead of after a buffer flush.
export CUDA_HOME="/usr/local/cuda"
export CUDA_PATH="/usr/local/cuda"
export PYTHONUNBUFFERED="1"
EOF
