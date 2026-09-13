#!/usr/bin/env bash
# Eight-seed binary-mask evolutionary latent search on GPUs 0..7.
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

OUT_DIR="${OUT_DIR:-pattern/outputs/z_star_reachable/r4_hard_sampling_multiseed8_20260912}"
SEEDS=(186 188 190 192 194 196 198 200)
GPUS=(0 1 2 3 4 5 6 7)

python pattern/evaluation/z_star_sampling_r4.py --stage prepare --out "$OUT_DIR"
pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup INT TERM

for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  gpu="${GPUS[$index]}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python pattern/evaluation/z_star_sampling_r4.py \
      --stage run --out "$OUT_DIR" --model-seed "$seed" --device cuda \
      >"$OUT_DIR/logs/seed_${seed}.log" 2>&1 &
  pids+=("$!")
  echo "Started seed $seed on GPU $gpu (pid ${pids[-1]})"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Finished seed ${SEEDS[$index]}"
  else
    echo "FAILED seed ${SEEDS[$index]}; see log" >&2
    failed=1
  fi
done
trap - INT TERM
if (( failed )); then exit 1; fi
python pattern/evaluation/z_star_sampling_r4.py --stage report --out "$OUT_DIR" --device cpu
