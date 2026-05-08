#!/usr/bin/env bash
# rs-nodes one-shot pod entry-point.
#
# Wired in as the pod template's Container Start Command via:
#   bash -c "(getent hosts github.com >/dev/null 2>&1 || { echo nameserver 8.8.8.8 > /etc/resolv.conf; echo nameserver 1.1.1.1 >> /etc/resolv.conf; }) && curl -fsSL https://raw.githubusercontent.com/richservo/rs-nodes/master/runpod/init.sh | bash"
#
# What it does:
#   * First boot on a fresh /workspace volume — fetch bootstrap.sh and
#     run the full provisioning (clone, deps, models, Ollama, launch).
#   * Every subsequent boot — fall through to /workspace/startup.sh
#     which runs idempotent re-checks and launches ComfyUI in seconds.
#
# Env vars (set in the pod template):
#   HF_TOKEN          — HuggingFace token for model downloads
#   OLLAMA_MODEL      — Ollama model(s), space-separated (default: gemma4:31b gemma4:26b)
#   RS_INSTALL_OLLAMA — "0" to skip Ollama (default "1")
#   RS_NODE_PACKS     — custom-node pack keys (default: full set)

set -e

WORKSPACE=/workspace
BOOTSTRAP_URL="https://raw.githubusercontent.com/richservo/rs-nodes/master/runpod/bootstrap.sh"

mkdir -p "$WORKSPACE"

# Subsequent-boot fast path: startup.sh exists on the volume already
# (mirrored there by an earlier bootstrap.sh run).
if [ -x "$WORKSPACE/startup.sh" ] && [ -d "$WORKSPACE/ComfyUI/custom_nodes/rs-nodes" ]; then
    echo "[init] /workspace/startup.sh present — running it"
    exec bash "$WORKSPACE/startup.sh"
fi

# First-boot path: fetch and exec bootstrap.sh.
echo "[init] First boot detected — fetching bootstrap.sh"
curl -fsSL "$BOOTSTRAP_URL" -o "$WORKSPACE/bootstrap.sh"
chmod +x "$WORKSPACE/bootstrap.sh"
exec bash "$WORKSPACE/bootstrap.sh"
