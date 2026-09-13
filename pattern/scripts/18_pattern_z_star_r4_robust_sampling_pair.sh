#!/usr/bin/env bash
# Two-seed positive-control pilot for replicated 2000-step hard sampling.
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
OUT_DIR="${OUT_DIR:-pattern/outputs/z_star_reachable/r4_hard_sampling_robust_pair_20260913}"
python pattern/evaluation/z_star_sampling_r4_robust.py --stage prepare --scope pair --out "$OUT_DIR"
pids=()
for item in "186:3" "188:6"; do
  seed="${item%%:*}"; gpu="${item##*:}"
  CUDA_VISIBLE_DEVICES="$gpu" python pattern/evaluation/z_star_sampling_r4_robust.py \
    --stage run --scope pair --out "$OUT_DIR" --model-seed "$seed" --device cuda \
    >"$OUT_DIR/logs/seed_${seed}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
python pattern/evaluation/z_star_sampling_r4_robust.py --stage report --scope pair --out "$OUT_DIR" --device cpu
