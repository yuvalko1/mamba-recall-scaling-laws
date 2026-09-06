#!/usr/bin/env bash
# Launches several grid runs in parallel, one GPU each, detached with setsid+nohup so
# they survive SSH disconnects AND the launching terminal/session being torn down.
# Logs go to logs/ (not shown automatically). Each run's CPU affinity is capped at
# CPU_CAP_PERCENT of nproc so a many-worker grid can't starve SSH/system responsiveness.
#
# Usage: bash scripts/run_multi_grids_across_gpus.sh

set -euo pipefail

CPU_CAP_PERCENT=90

MAX_CORE=$(( $(nproc) * CPU_CAP_PERCENT / 100 - 1 ))

# ==== EDIT ME: (gpu, config) pairs ====

RUNS=(
  'cuda0  config/MQAR__D_N__linear_trained__example.json5'    # small example grid, linear Mamba model
  'cuda1  config/MQAR__D_N__full_trained__example.json5'      # small example grid, full Mamba model
#  'cuda2,3  config/MQAR__D_N__full_trained__regime_0.json5'  # actual paper grid; long run. see more under config/*
)

# ======================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
LOGS_DIR="$ROOT_DIR/logs"

mkdir -p "$LOGS_DIR"
cd "$ROOT_DIR"

LOG_FILES=()
PIDS=()

for run in "${RUNS[@]}"; do
  read -r gpu config <<< "$run"
  cuda="${gpu#cuda}"  # "cuda3" -> "3"
  if [ "$gpu" = '""' ]; then  # "" -> CPU-only (CUDA_VISIBLE_DEVICES=""): hides all GPUs
    cuda=""
    gpu="cpu"
  fi
  gpu="${gpu//,/-}"  # log-name friendly: "cuda3,7" -> "cuda3-7"

  if [ ! -f "$config" ]; then
    echo "ERROR: config not found: $config" >&2
    exit 1
  fi

  config_stem="$(basename "$config" .json5)"
  # per-run timestamp + 5s stagger, so every launch (incl. same-config reruns) gets a unique
  # log file and results/wandb naming
  timestamp="$(date +%Y%m%d_%H%M%S)"
  log_file="$LOGS_DIR/${config_stem}__${gpu}__${timestamp}.log"

  CUDA_VISIBLE_DEVICES="$cuda" setsid taskset -c "0-$MAX_CORE" nohup "$PYTHON" src/main.py -c "$config" > "$log_file" 2>&1 < /dev/null &
  pid=$!
  disown "$pid"

  LOG_FILES+=("$log_file")
  PIDS+=("$pid")
  echo "launched: $gpu | $config_stem | pid $pid | cores 0-$MAX_CORE | $(basename "$log_file")"
  echo "to view logs, run:"
  echo "tail -f $log_file"
  echo
  if [ "${#LOG_FILES[@]}" -lt "${#RUNS[@]}" ]; then  # skip the stagger after the last launch
    echo "sleeping 5s before the next launch..."
    sleep 5
  fi
done

echo
echo "${#PIDS[@]} grid runs launched (detached; they survive SSH disconnects)."
echo
echo "to stop a run (parent + all its workers):"
for k in "${!PIDS[@]}"; do
  echo "pkill -TERM -P ${PIDS[$k]}; kill ${PIDS[$k]}    # $(basename "${LOG_FILES[$k]}")"
done
