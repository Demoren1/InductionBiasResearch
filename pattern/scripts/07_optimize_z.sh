#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${GPU_IDS:-0}"
CVAE_CKPT="${CVAE_CKPT:-outputs/cvae/cvae_best.pt}"
MODES="${MODES:-z z0 free}"
OUTER_STEPS="${OUTER_STEPS:-30}"
WARMUP_STEPS="${WARMUP_STEPS:-300}"
GRAD_STEPS="${GRAD_STEPS:-100}"
FINAL_STEPS="${FINAL_STEPS:-2000}"
N_REPEATS="${N_REPEATS:-8}"
Z_SEED="${Z_SEED:-0}"
OUT_JSON="${OUT_JSON:-outputs/eval/zopt_results.json}"
MASK_OUT="${MASK_OUT:-outputs/eval/zopt_masks.pt}"
PLOT_DIR="${PLOT_DIR:-outputs/plots/z_analysis}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-outputs/.matplotlib}"

cd "$(dirname "$0")/.."

if [ ! -f "$CVAE_CKPT" ]; then
  echo "Missing frozen VAE checkpoint: $CVAE_CKPT" >&2
  echo "Run scripts/04_train_cvae.sh first or set CVAE_CKPT." >&2
  exit 1
fi

for MODE in $MODES; do
  CUDA_VISIBLE_DEVICES="$GPU_IDS" python evaluation/optimize_z.py \
    --ckpt "$CVAE_CKPT" --mode "$MODE" \
    --outer_steps "$OUTER_STEPS" --warmup_steps "$WARMUP_STEPS" \
    --grad_steps "$GRAD_STEPS" --final_steps "$FINAL_STEPS" \
    --n_repeats "$N_REPEATS" --seed "$Z_SEED" \
    --out_json "$OUT_JSON" --mask_out "$MASK_OUT" --plot_dir "$PLOT_DIR"
done

echo "z optimization done -> $OUT_JSON"
