#!/usr/bin/env bash
# === Parameters (edit here or export before running) ===
KERNELS="${KERNELS:-3 5 7 9}"
OFFSETS="${OFFSETS:-0 2 4 6}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
MLPS_PER_GPU="${MLPS_PER_GPU:-4096}"
# ======================================================
# Train N_MLPS_PER_KERNEL masked MLPs per (kernel, offset) across the GPUs.
#
# Choose which physical cards to use:
#     GPU_IDS="2 3 4 5" bash scripts/02_train.sh
#   or (default: first four cards):
#     bash scripts/02_train.sh
#
# Each card trains 1/num_gpus of the MLPs for every (kernel, offset) pair;
# (kernel, offset) pairs are processed sequentially.
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_GPUS=$(echo "$GPU_IDS" | wc -w)

echo "Using ${NUM_GPUS} GPUs: ${GPU_IDS}"

for K in $KERNELS; do
  for S in $OFFSETS; do
    echo "=== kernel $K offset $S ==="
    PIDS=()
    local_idx=0
    for gpu in $GPU_IDS; do
      CUDA_VISIBLE_DEVICES="$gpu" python models/train.py \
        --kernel "$K" --offset "$S" --gpu_id "$local_idx" --num_gpus "$NUM_GPUS" \
        --train_steps "$TRAIN_STEPS" --mlps_per_gpu "$MLPS_PER_GPU" &
      PIDS+=($!)
      local_idx=$((local_idx + 1))
    done
    for pid in "${PIDS[@]}"; do
      wait "$pid" || exit 1
    done
    echo "=== kernel $K offset $S done ==="
  done
done
echo "All (kernel, offset) pairs trained."