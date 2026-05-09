#!/usr/bin/env bash
# Pod resource monitor. Continuously displays GPU/RAM/disk/processes
# in a single frame that redraws in place. Press Ctrl+C to exit.
#
# Run on the pod via:
#   bash /workspace/monitor.sh
# or via E:\runpod\monitor.bat from Windows (opens an SSH session
# running this).
#
# Uses only stock tools (nvidia-smi, free, df, ps) — no apt installs.

set -u

INTERVAL="${INTERVAL:-2}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found — is this actually a GPU pod?"
    exit 1
fi

# Hide cursor for cleaner redraw, restore on exit.
tput civis 2>/dev/null
trap 'tput cnorm 2>/dev/null; clear; exit 0' INT TERM EXIT

clear
while true; do
    tput cup 0 0

    printf '\033[1;36m=================== POD RESOURCE MONITOR ====================\033[0m\n'
    printf '  %s   (%ds refresh, Ctrl+C to exit)\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$INTERVAL"
    echo

    # ---- GPU ----
    printf '\033[1;33mGPU\033[0m\n'
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' '{
            mem_pct = ($3 / $4) * 100;
            printf "  %s\n", $1;
            printf "    util:   %3d%%   |   mem: %5d / %5d MB (%.0f%%)\n", $2, $3, $4, mem_pct;
            printf "    temp:   %3d C  |   power: %s W\n", $5, $6;
        }'
    echo

    # ---- RAM ----
    printf '\033[1;33mRAM\033[0m\n'
    free -m | awk 'NR==2 {
        used_pct = ($3 / $2) * 100;
        printf "    %d / %d MB used (%.0f%%)\n", $3, $2, used_pct
    }'
    echo

    # ---- Disk on /workspace (network volume) ----
    printf '\033[1;33mDisk (/workspace)\033[0m\n'
    df -h /workspace 2>/dev/null | awk 'NR==2 {
        printf "    %s used / %s total (%s full)\n", $3, $2, $5
    }'
    echo

    # ---- Disk on container ----
    printf '\033[1;33mDisk (container /)\033[0m\n'
    df -h / 2>/dev/null | awk 'NR==2 {
        printf "    %s used / %s total (%s full)\n", $3, $2, $5
    }'
    echo

    # ---- Top GPU processes ----
    printf '\033[1;33mGPU processes\033[0m\n'
    nvidia-smi --query-compute-apps=pid,used_memory,process_name \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' 'NF >= 3 { printf "    PID %-7s  %5s MB   %s\n", $1, $2, $3 }'
    if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q .; then
        echo "    (no GPU processes)"
    fi
    echo

    # ---- Top CPU processes ----
    printf '\033[1;33mTop CPU processes\033[0m\n'
    ps -eo pid,pcpu,pmem,comm --sort=-pcpu --no-headers | head -5 | \
        awk '{ printf "    PID %-7s  %4s%% CPU  %4s%% MEM  %s\n", $1, $2, $3, $4 }'

    # Clear any leftover lines from a previous taller frame.
    printf '\033[J'

    sleep "$INTERVAL"
done
