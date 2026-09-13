#!/usr/bin/env bash
# Robust hard-mask sampling on 32 saved VAE pairs (64 frozen decoders).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras
cd "$PROJECT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1

OUT_DIR="${OUT_DIR:-pattern/outputs/z_star_reachable/r4_hard_sampling_robust_pairs32_20260913}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a GPUS <<< "$GPU_IDS"

python -m pattern.evaluation.z_star_sampling_r4_pairs32 --stage prepare --out "$OUT_DIR"

seeds=($(seq 186 249))
for ((offset=0; offset<${#seeds[@]}; offset+=${#GPUS[@]})); do
  pids=()
  for ((slot=0; slot<${#GPUS[@]} && offset+slot<${#seeds[@]}; slot++)); do
    seed="${seeds[$((offset+slot))]}"
    CUDA_VISIBLE_DEVICES="${GPUS[$slot]}" \
      python -m pattern.evaluation.z_star_sampling_r4_pairs32 \
        --stage run --out "$OUT_DIR" --model-seed "$seed" --device cuda \
        > "$OUT_DIR/logs/seed_${seed}.log" 2>&1 &
    pids+=("$!")
    echo "START seed=$seed gpu=${GPUS[$slot]} pid=$!"
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "A worker failed; inspect $OUT_DIR/logs" >&2
    exit 1
  fi
  echo "COMPLETE $((offset + ${#pids[@]}))/${#seeds[@]} decoders"
done

python -m pattern.evaluation.z_star_sampling_r4_pairs32 --stage report --out "$OUT_DIR"
echo "Finished: $OUT_DIR/RESULTS.md"
