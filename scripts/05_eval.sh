#!/usr/bin/env bash
# === Parameters ===
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
EVAL_STEPS="${EVAL_STEPS:-5000}"
EVAL_N_MASKS="${EVAL_N_MASKS:-128}"
EVAL_PAIRS="${EVAL_PAIRS:-3,0 3,2 3,4 3,6 5,0 5,2 5,4 5,6 7,0 7,2 7,4 7,6 9,0 9,2 9,4 9,6}"
# ===================
# Evaluate generated masks on the MA task, optionally across multiple GPUs.
# Evaluates all 16 kernel/stride pairs (8 train + 8 test) from the CVAE experiment.
#
# Single GPU:                     bash scripts/05_eval.sh
# 4 GPUs in parallel:   GPU_IDS="0 1 2 3" bash scripts/05_eval.sh
#
# Extra arguments are forwarded verbatim to the eval script. Each GPU processes
# a disjoint subset of kernel/stride pairs; results are merged into
# outputs/eval/eval_results.json + eval_mse.png
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_GPUS=$(echo "$GPU_IDS" | wc -w)
EVAL_ARGS="--steps $EVAL_STEPS --n_masks $EVAL_N_MASKS $@"

echo "eval on ${NUM_GPUS} GPU(s): ${GPU_IDS}"

PIDS=()
gpu_idx=0
for gpu in $GPU_IDS; do
    CUDA_VISIBLE_DEVICES="$gpu" python evaluation/eval_generated_masks.py \
        --gpu_id "$gpu_idx" --num_gpus "$NUM_GPUS" --eval_pairs "$EVAL_PAIRS" $EVAL_ARGS &
    PIDS+=($!)
    gpu_idx=$((gpu_idx + 1))
done
for pid in "${PIDS[@]}"; do
    wait "$pid" || exit 1
done

python evaluation/merge_eval.py
echo "eval done -> outputs/eval/"