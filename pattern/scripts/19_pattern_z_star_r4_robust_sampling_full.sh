#!/usr/bin/env bash
# Eight-seed replicated 2000-step hard-mask sampling.
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
OUT_DIR="${OUT_DIR:-pattern/outputs/z_star_reachable/r4_hard_sampling_robust_multiseed8_20260913}"
SEEDS=(186 188 190 192 194 196 198 200)
python pattern/evaluation/z_star_sampling_r4_robust.py --stage prepare --scope full --out "$OUT_DIR"
pids=()
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  CUDA_VISIBLE_DEVICES="$index" python pattern/evaluation/z_star_sampling_r4_robust.py \
    --stage run --scope full --out "$OUT_DIR" --model-seed "$seed" --device cuda \
    >"$OUT_DIR/logs/seed_${seed}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then exit 1; fi
python pattern/evaluation/z_star_sampling_r4_robust.py --stage report --scope full --out "$OUT_DIR" --device cpu
