#!/usr/bin/env bash
# Oracle z* plus controlled latent-noise recovery on eight frozen pattern VAEs.
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

OUT_DIR="${OUT_DIR:-pattern/outputs/z_star_noise/multiseed8_20260912}"
PARENT_DIR="${PARENT_DIR:-pattern/outputs/decoder_agreement/multiseed32_tuned_20260912}"
SEEDS=(186 188 190 192 194 196 198 200)
GPU_IDS=(0 1 2 3 4 5 6 7)

python pattern/evaluation/z_star_noise.py \
  --out "$OUT_DIR" --parent "$PARENT_DIR" --stage prepare --device cpu

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  gpu="${GPU_IDS[$index]}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python pattern/evaluation/z_star_noise.py \
      --out "$OUT_DIR" --parent "$PARENT_DIR" --stage run \
      --model-seed "$seed" --device cuda \
      >"$OUT_DIR/logs/seed_${seed}.log" 2>&1 &
  pids+=("$!")
  echo "Started VAE seed $seed on GPU $gpu (pid ${pids[-1]})"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Finished VAE seed ${SEEDS[$index]}"
  else
    echo "FAILED VAE seed ${SEEDS[$index]}; see $OUT_DIR/logs/seed_${SEEDS[$index]}.log" >&2
    failed=1
  fi
done
trap - INT TERM
if (( failed )); then
  exit 1
fi

python pattern/evaluation/z_star_noise.py \
  --out "$OUT_DIR" --parent "$PARENT_DIR" --stage report --device cpu
echo "Report: $OUT_DIR/RESULTS.md"

