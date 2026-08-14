#!/usr/bin/env bash
# Train N_MLPS_PER_PATTERN masked MLPs per pattern across GPUs.
# Usage:
#   GPU_IDS="0 1 2 3" bash scripts/02_train.sh
#   PATTERNS="0000 1111" GPU_IDS="0" bash scripts/02_train.sh
PATTERNS="${PATTERNS:-0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
MLPS_PER_GPU="${MLPS_PER_GPU:-2000}"
set -euo pipefail
cd "$(dirname "$0")/.."
NUM_GPUS=$(echo "$GPU_IDS" | wc -w)
echo "Using ${NUM_GPUS} GPUs: ${GPU_IDS}"
for PAT in $PATTERNS; do
  echo "=== pattern ${PAT} ==="
  PIDS=()
  local_idx=0
  for gpu in $GPU_IDS; do
    CUDA_VISIBLE_DEVICES="$gpu" python models/train.py \
      --pattern "$PAT" --gpu_id "$local_idx" --num_gpus "$NUM_GPUS" \
      --train_steps "$TRAIN_STEPS" --mlps_per_gpu "$MLPS_PER_GPU" &
    PIDS+=($!)
    local_idx=$((local_idx + 1))
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" || exit 1
  done
  echo "=== pattern ${PAT} done ==="
done
echo "All patterns trained."
