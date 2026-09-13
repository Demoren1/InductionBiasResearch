#!/usr/bin/env bash
# Pair-specific latent adapters for 32 frozen VAE pairs, sharded over 8 GPUs.
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

PAIR_ROOT="${PAIR_ROOT:-pattern/outputs/decoder_agreement/multiseed32_tuned_20260912}"
OUT_DIR="${OUT_DIR:-pattern/outputs/latent_adapter/multiseed32_20260913}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a GPUS <<< "$GPU_IDS"
mkdir -p "$OUT_DIR/logs"

pairs=()
for first in $(seq 186 2 248); do
  pairs+=("${first}_$((first + 1))")
done

run_queue() {
  local gpu="$1"
  local slot="$2"
  local pair first second
  for ((index=slot; index<${#pairs[@]}; index+=${#GPUS[@]})); do
    pair="${pairs[$index]}"
    first="${pair%_*}"
    second="${pair#*_}"
    echo "START pair=$pair gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python -m pattern.evaluation.run_latent_adapter \
      --pair-root "$PAIR_ROOT/pair_${pair}" \
      --out "$OUT_DIR/pair_${pair}" --device cuda \
      > "$OUT_DIR/logs/pair_${pair}.log" 2>&1
    echo "DONE pair=$first/$second gpu=$gpu"
  done
}

pids=()
for ((slot=0; slot<${#GPUS[@]}; slot++)); do
  run_queue "${GPUS[$slot]}" "$slot" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" -ne 0 ]]; then
  echo "A worker failed; inspect $OUT_DIR/logs" >&2
  exit 1
fi

python -m pattern.evaluation.report_latent_adapter --root "$OUT_DIR"
echo "Finished: $OUT_DIR/RESULTS.md"
