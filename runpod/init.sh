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
DONE_MARKER="$WORKSPACE/.bootstrap_done"

mkdir -p "$WORKSPACE"

# Honor RunPod's PUBLIC_KEY env-var convention on every boot. RunPod's
# stock init copies $PUBLIC_KEY into /root/.ssh/authorized_keys, but
# our init.sh replaces the stock start command so that hook never
# fires — meaning users who set the pubkey via env var (instead of
# account-level Settings -> SSH Public Keys) lose SSH access entirely.
# Re-implement the hook here, before the fast-path branch, so it runs
# whether we go straight to startup.sh or fall through to bootstrap.sh.
if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh /workspace/.ssh
    chmod 700 /root/.ssh /workspace/.ssh
    touch /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    if ! grep -qxF "$PUBLIC_KEY" /root/.ssh/authorized_keys 2>/dev/null; then
        echo "[init] Installing PUBLIC_KEY env-var pubkey into authorized_keys"
        echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    fi
    # Mirror to the volume so bootstrap.sh Phase 0.5 picks it up as the
    # canonical source instead of overwriting it.
    cp /root/.ssh/authorized_keys /workspace/.ssh/authorized_keys
    chmod 600 /workspace/.ssh/authorized_keys
fi

# Subsequent-boot fast path: bootstrap fully completed at least once
# (marker written by bootstrap.sh at the end of Phase 6).
#
# We require the marker rather than just file-existence checks: a
# partial bootstrap interrupted mid-run (e.g. user edited pod config
# and triggered a silent reboot) leaves startup.sh and the rs-nodes
# clone in place but skips Phase 4 (model weights). The marker tells
# us provisioning actually finished, so it's safe to take the fast
# path.
if [ -f "$DONE_MARKER" ] && [ -x "$WORKSPACE/startup.sh" ]; then
    echo "[init] $DONE_MARKER present — running startup.sh (fast path)"
    exec bash "$WORKSPACE/startup.sh"
fi

# First-boot OR incomplete-bootstrap path: fetch and exec bootstrap.sh.
# bootstrap.sh is idempotent; re-running on a partial volume picks up
# from where the last run was killed.
if [ -x "$WORKSPACE/startup.sh" ]; then
    echo "[init] startup.sh present but bootstrap marker missing — re-running bootstrap.sh"
else
    echo "[init] First boot detected — fetching bootstrap.sh"
fi
curl -fsSL "$BOOTSTRAP_URL" -o "$WORKSPACE/bootstrap.sh"
chmod +x "$WORKSPACE/bootstrap.sh"
exec bash "$WORKSPACE/bootstrap.sh"
