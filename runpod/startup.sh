#!/usr/bin/env bash
# rs-nodes pod-side bootstrap
# -----------------------------------------------------------------------------
# Goal: bring a freshly-booted RunPod container to a working ComfyUI on
# port 8188 with rs-nodes installed, idempotently. State lives on the
# network volume mounted at /workspace, so subsequent boots are fast
# (just a git pull + a venv check) and don't redownload models.
#
# Wire this script in as the pod template's "Container Start Command":
#   bash /workspace/startup.sh
# Place a copy at /workspace/startup.sh once during initial volume setup
# (see runpod/README.md for the one-time provisioning steps).

set -euo pipefail

WORKSPACE=/workspace
COMFY_DIR="$WORKSPACE/ComfyUI"
RS_NODES_DIR="$COMFY_DIR/custom_nodes/rs-nodes"
VENV="$WORKSPACE/.venv"
PORT="${COMFY_PORT:-8188}"
LOG_FILE="${COMFY_LOG:-/workspace/comfyui.log}"

# Ubuntu 24.04 (PEP 668) marks the system Python as externally-managed.
# We do all installs into a venv on the network volume instead, so
# state persists across container resets (no re-installing torch every
# boot) AND we sidestep PEP 668 entirely. The flag stays set as a
# fallback for any pip that escapes the venv (e.g. via sudo).
export PIP_BREAK_SYSTEM_PACKAGES=1

# Tee everything to a log file so external shells (e.g. launch.bat
# tailing from Windows) can see startup + ComfyUI output even when
# the script runs as the container's start command (no attached TTY).
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { printf '[startup] %s\n' "$*"; }

log "rs-nodes pod startup beginning at $(date -Is)"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

# -----------------------------------------------------------------------------
# 0. DNS guard — some RunPod regions (observed: EU-CZ-1) ship containers
#    with an empty /etc/resolv.conf, so even `git clone github.com` fails.
#    /etc/resolv.conf lives on the container disk, not the volume, so this
#    has to run on every boot. We only inject if the existing file is empty
#    or missing; pods with working DNS are left alone.
# -----------------------------------------------------------------------------
if ! getent hosts github.com >/dev/null 2>&1; then
    log "DNS resolution broken; injecting public resolvers into /etc/resolv.conf"
    {
        echo "nameserver 8.8.8.8"
        echo "nameserver 1.1.1.1"
    } > /etc/resolv.conf || log "WARN: could not write /etc/resolv.conf"
fi

# -----------------------------------------------------------------------------
# 0.4. Ensure sshd has host keys and is running.
#      Container Start Command replaces RunPod's stock init, which is
#      what would normally generate /etc/ssh/ssh_host_*_key on first
#      boot. Without those keys sshd refuses to start ("no hostkeys
#      available -- exiting"). We persist the host keys to the volume
#      so SSH clients don't get spammed with "host key changed"
#      warnings on every container reset.
# -----------------------------------------------------------------------------
mkdir -p /workspace/.ssh/host_keys /etc/ssh
if compgen -G "/workspace/.ssh/host_keys/ssh_host_*" > /dev/null; then
    log "Restoring sshd host keys from /workspace/.ssh/host_keys"
    cp -f /workspace/.ssh/host_keys/ssh_host_* /etc/ssh/
    chmod 600 /etc/ssh/ssh_host_*_key 2>/dev/null || true
    chmod 644 /etc/ssh/ssh_host_*_key.pub 2>/dev/null || true
elif compgen -G "/etc/ssh/ssh_host_*" > /dev/null; then
    log "Persisting existing sshd host keys to /workspace/.ssh/host_keys"
    cp -f /etc/ssh/ssh_host_* /workspace/.ssh/host_keys/
else
    log "No sshd host keys anywhere; generating fresh set with ssh-keygen -A"
    ssh-keygen -A
    cp -f /etc/ssh/ssh_host_* /workspace/.ssh/host_keys/
fi

# Start (or restart) sshd. service may or may not exist depending on
# the base image; fall back to direct sshd invocation.
if command -v service >/dev/null 2>&1; then
    service ssh restart 2>&1 | sed 's/^/[sshd] /' || true
elif [ -x /usr/sbin/sshd ]; then
    pkill -f /usr/sbin/sshd 2>/dev/null || true
    /usr/sbin/sshd
    log "sshd started directly"
else
    log "WARN: no sshd binary found; SSH will be unavailable on this container"
fi

# -----------------------------------------------------------------------------
# 0.5. Persist SSH authorized_keys on the network volume.
#      RunPod's stock auto-injection only fires on first container
#      creation; template changes / re-deploys produce a fresh
#      container with empty /root/.ssh and force a manual re-add.
#      We mirror authorized_keys to /workspace/.ssh/ so subsequent
#      boots always have the key, regardless of what spawned the
#      container. First-boot-after-manual-add direction is also
#      covered: if /root/.ssh has a key but the volume doesn't, we
#      persist UP to the volume.
# -----------------------------------------------------------------------------
mkdir -p /workspace/.ssh /root/.ssh
chmod 700 /workspace/.ssh /root/.ssh
if [ -s /workspace/.ssh/authorized_keys ]; then
    log "Restoring SSH authorized_keys from /workspace/.ssh"
    cp /workspace/.ssh/authorized_keys /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
elif [ -s /root/.ssh/authorized_keys ]; then
    log "First-time persist of SSH authorized_keys -> /workspace/.ssh"
    cp /root/.ssh/authorized_keys /workspace/.ssh/authorized_keys
    chmod 600 /workspace/.ssh/authorized_keys
else
    log "WARN: no authorized_keys found anywhere — SSH key auth will fail"
fi

# -----------------------------------------------------------------------------
# 1. ComfyUI install (one-time clone, every-boot pull)
# -----------------------------------------------------------------------------
if [ ! -d "$COMFY_DIR/.git" ]; then
    log "Cloning ComfyUI into $COMFY_DIR ..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
else
    log "ComfyUI already present; pulling latest..."
    git -C "$COMFY_DIR" pull --ff-only || \
        log "WARN: git pull failed; continuing with current revision"
fi

# -----------------------------------------------------------------------------
# 2. rs-nodes install (one-time clone, every-boot pull)
# -----------------------------------------------------------------------------
if [ ! -d "$RS_NODES_DIR/.git" ]; then
    log "Cloning rs-nodes into $RS_NODES_DIR ..."
    git clone https://github.com/richservo/rs-nodes.git "$RS_NODES_DIR"
else
    log "rs-nodes already present; pulling latest..."
    git -C "$RS_NODES_DIR" pull --ff-only || \
        log "WARN: rs-nodes pull failed; continuing with current revision"
fi
log "Updating rs-nodes submodules..."
git -C "$RS_NODES_DIR" submodule update --init --recursive || \
    log "WARN: submodule update failed"

# Mirror pod-side helper scripts from rs-nodes/runpod/ to /workspace/
# so Container Start Command + manual runs find them at the documented
# paths even on a fresh container disk. git pull is the delivery channel
# (works around wedged sshd / scp).
if [ -d "$RS_NODES_DIR/runpod" ]; then
    log "Mirroring rs-nodes/runpod scripts to /workspace/"
    cp -f "$RS_NODES_DIR/runpod/"*.sh /workspace/ 2>/dev/null || true
    chmod +x /workspace/*.sh 2>/dev/null || true
fi

# -----------------------------------------------------------------------------
# 2.6. Persistent venv on the network volume
#      Without this, every container reset re-downloads + reinstalls
#      every pip package (torch alone is ~3 GB). The venv lives on
#      /workspace so it survives Stop+Start / migrate / terminate.
#      --system-site-packages lets us inherit the base image's CUDA /
#      system libs while letting us override Python packages (torch,
#      sageattention, etc.) with newer versions installed into the venv.
# -----------------------------------------------------------------------------
if [ ! -f "$VENV/bin/python" ]; then
    log "Creating venv at $VENV (one-time, ~30s)"
    python3 -m venv --system-site-packages "$VENV"
fi
log "Activating venv at $VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
log "Python: $(which python)  ($(python --version 2>&1))"
log "Pip:    $(which pip)"

# -----------------------------------------------------------------------------
# 3. Python deps (idempotent — pip is a no-op on already-installed pkgs;
#    only first boot or version changes cause real downloads).
# -----------------------------------------------------------------------------
log "Installing ComfyUI Python deps..."
pip install --no-cache-dir -r "$COMFY_DIR/requirements.txt" || \
    log "WARN: ComfyUI deps install failed"

log "Installing rs-nodes Python deps..."
if [ -f "$RS_NODES_DIR/requirements.txt" ]; then
    pip install --no-cache-dir -r "$RS_NODES_DIR/requirements.txt" || \
        log "WARN: rs-nodes deps install failed"
fi

# ROSE optimizer — published as rose-opt, imported as rose_opt
log "Ensuring ROSE optimizer..."
pip install --no-cache-dir rose-opt || \
    log "WARN: ROSE install failed (only needed for training)"

# Performance stack:
#   * Upgrade PyTorch to cu130 wheels — Blackwell (RTX 50xx / PRO 6000)
#     hardware fp8 paths and comfy-kitchen's cuda+triton backends require
#     it. Stock RunPod pytorch image ships cu128 which leaves these
#     disabled (only the slow eager backend runs). Idempotent: pip
#     skips when already at the right version.
#   * SageAttention — 30-50% faster attention than vanilla PyTorch SDPA.
#     ComfyUI auto-detects and uses it.
log "Ensuring PyTorch cu130 wheels (Blackwell perf path)..."
pip install --upgrade --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu130 \
    torch torchvision torchaudio || \
    log "WARN: PyTorch cu130 upgrade failed; running on stock cu128"

# cu130 PyTorch needs the matching NVRTC + companion runtime libs
# for JIT-compiled kernels (e.g. torchaudio spectrogram's complex abs).
# Without these, runs crash with "failed to open libnvrtc-builtins.so.13.0".
# NVIDIA deprecated the -cu13 suffixed packages; use the unsuffixed
# names which now ship the cu13-compatible builds.
log "Ensuring NVIDIA CUDA runtime libraries..."
pip install --no-cache-dir \
    nvidia-cuda-nvrtc \
    nvidia-cuda-runtime \
    nvidia-cublas \
    nvidia-cudnn || \
    log "WARN: CUDA runtime libs install failed; JIT kernels may crash"

# pip-installed NVIDIA libs land in venv site-packages/nvidia/*/lib/.
# PyTorch's nvrtc dlopen() searches LD_LIBRARY_PATH at runtime, so
# every nvidia/* lib dir has to be visible. Compute and persist it
# into the venv's activate script so any shell that activates the
# venv (including the exec at the end of this script) inherits it.
NV_LIB_PATHS=$(python -c "
import nvidia, os
r = os.path.dirname(nvidia.__file__)
paths = []
for d in sorted(os.listdir(r)):
    lib_dir = os.path.join(r, d, 'lib')
    if os.path.isdir(lib_dir):
        paths.append(lib_dir)
print(':'.join(paths))
" 2>/dev/null || echo "")
if [ -n "$NV_LIB_PATHS" ]; then
    log "NVIDIA lib paths: $NV_LIB_PATHS"
    export LD_LIBRARY_PATH="$NV_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

log "Ensuring SageAttention..."
pip install --no-cache-dir sageattention || \
    log "WARN: SageAttention install failed; using stock PyTorch attention"

# -----------------------------------------------------------------------------
# 4. Optional InsightFace pre-download (gated by env var so the dispatch
#    path that doesn't need it doesn't pay the bandwidth)
# -----------------------------------------------------------------------------
if [ "${RS_PREFETCH_INSIGHTFACE:-0}" = "1" ]; then
    log "Pre-downloading InsightFace antelopev2..."
    python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])" || \
        log "WARN: InsightFace prefetch failed (will retry on first use)"
fi

# -----------------------------------------------------------------------------
# 5. Ollama install + serve (default ON — RSPromptFormatter requires it,
#    so it's load-bearing for the standard workflows). Set RS_INSTALL_OLLAMA=0
#    in pod env vars to skip if you ever want a slimmer boot.
# -----------------------------------------------------------------------------
if [ "${RS_INSTALL_OLLAMA:-1}" = "1" ]; then
    if ! command -v ollama >/dev/null 2>&1; then
        log "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh || \
            log "WARN: Ollama install failed"
    fi
    # Models live on the network volume so they survive pod terminate.
    # Without this, every fresh container would re-pull them.
    export OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/.ollama/models}"
    export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
    mkdir -p "$OLLAMA_MODELS"
    log "Starting Ollama (OLLAMA_MODELS=$OLLAMA_MODELS) in the background..."
    nohup env OLLAMA_MODELS="$OLLAMA_MODELS" OLLAMA_HOST="$OLLAMA_HOST" \
        ollama serve >/workspace/ollama.log 2>&1 &

    # Pull required models (no-op if already on volume).
    OLLAMA_MODEL_LIST="${OLLAMA_MODEL:-gemma4:31b gemma4:26b}"
    # Wait briefly for server to come up before pulling.
    for i in $(seq 1 15); do
        curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1 && break
        sleep 2
    done
    for model in $OLLAMA_MODEL_LIST; do
        if ! ollama list | awk 'NR>1 {print $1}' | grep -Fxq "$model"; then
            log "Pulling Ollama model: $model"
            ollama pull "$model" || log "WARN: pull failed for $model"
        fi
    done
fi

# -----------------------------------------------------------------------------
# 6. Launch ComfyUI on the public port
# -----------------------------------------------------------------------------
# --highvram keeps loaded weights resident on the GPU instead of
# offloading to CPU between operations. On a 96 GB Blackwell card
# the LTX-2.3 22B fp8 (~29 GB) + gemma fp4 text encoder (~9 GB) +
# audio_vae easily fit; eliminating the CPU↔GPU ping-pong can be
# 2-5x faster after the first warmup. Override via COMFY_EXTRA_ARGS
# env var (e.g. "" to disable, or "--gpu-only" for max-aggressive).
COMFY_EXTRA_ARGS="${COMFY_EXTRA_ARGS:---highvram}"
log "Launching ComfyUI on 0.0.0.0:${PORT}  (venv: $VENV, args: $COMFY_EXTRA_ARGS)"
cd "$COMFY_DIR"
exec "$VENV/bin/python" main.py --listen 0.0.0.0 --port "$PORT" $COMFY_EXTRA_ARGS "$@"
