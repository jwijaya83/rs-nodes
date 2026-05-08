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
PORT="${COMFY_PORT:-8188}"

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

# -----------------------------------------------------------------------------
# 3. Python deps (idempotent — pip is a no-op on already-installed pkgs)
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
# 5. Optional Ollama install (gated; only needed for caption-generation
#    workflows on the pod). Off by default to keep idle pods slim.
# -----------------------------------------------------------------------------
if [ "${RS_INSTALL_OLLAMA:-0}" = "1" ]; then
    if ! command -v ollama >/dev/null 2>&1; then
        log "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh || \
            log "WARN: Ollama install failed"
    fi
    # Models live on the network volume so they survive pod terminate.
    # Without this, every fresh container would re-pull them.
    export OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/.ollama/models}"
    mkdir -p "$OLLAMA_MODELS"
    log "Starting Ollama (OLLAMA_MODELS=$OLLAMA_MODELS) in the background..."
    nohup env OLLAMA_MODELS="$OLLAMA_MODELS" \
        ollama serve >/workspace/ollama.log 2>&1 &
fi

# -----------------------------------------------------------------------------
# 6. Launch ComfyUI on the public port
# -----------------------------------------------------------------------------
log "Launching ComfyUI on 0.0.0.0:${PORT}"
cd "$COMFY_DIR"
exec python main.py --listen 0.0.0.0 --port "$PORT" "$@"
