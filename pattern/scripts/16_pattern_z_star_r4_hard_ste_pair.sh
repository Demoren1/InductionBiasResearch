#!/usr/bin/env bash
# Two-seed R=4 pilot with hard top-k forward and soft-top-k STE backward.
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

OUT_DIR="${OUT_DIR:-pattern/outputs/z_star_reachable/r4_hard_ste_seeds186_188_20260912}"
SEEDS=(186 188)
GPUS=(3 6)

python pattern/evaluation/z_star_reachable_r4_hard_ste.py --stage prepare --out "$OUT_DIR"
pids=()
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  gpu="${GPUS[$index]}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python pattern/evaluation/z_star_reachable_r4_hard_ste.py \
      --stage run --out "$OUT_DIR" --model-seed "$seed" --device cuda \
      >"$OUT_DIR/logs/seed_${seed}.log" 2>&1 &
  pids+=("$!")
  echo "Started seed $seed on GPU $gpu"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Finished seed ${SEEDS[$index]}"
  else
    echo "FAILED seed ${SEEDS[$index]}" >&2
    failed=1
  fi
done
if (( failed )); then exit 1; fi
python pattern/evaluation/z_star_reachable_r4_hard_ste.py --stage report --out "$OUT_DIR" --device cpu
