#!/usr/bin/env bash
# Evaluate generated masks on the pattern task across GPUs, then merge.
# Usage:
#   GPU_IDS="0 1 2 3" bash scripts/05_eval.sh
GPU_IDS="${GPU_IDS:-0 1 2 3}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
EVAL_N_MASKS="${EVAL_N_MASKS:-64}"
set -euo pipefail
cd "$(dirname "$0")/.."
NUM_GPUS=$(echo "$GPU_IDS" | wc -w)
echo "eval on ${NUM_GPUS} GPU(s): ${GPU_IDS}"
PIDS=()
gpu_idx=0
for gpu in $GPU_IDS; do
    CUDA_VISIBLE_DEVICES="$gpu" python evaluation/eval_generated_masks.py \
        --gpu_id "$gpu_idx" --num_gpus "$NUM_GPUS" \
        --steps "$EVAL_STEPS" --n_masks "$EVAL_N_MASKS" &
    PIDS+=($!)
    gpu_idx=$((gpu_idx + 1))
done
for pid in "${PIDS[@]}"; do
    wait "$pid" || exit 1
done
python evaluation/merge_eval.py
echo "eval done -> outputs/eval/"
