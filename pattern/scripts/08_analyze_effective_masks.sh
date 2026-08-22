#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${GPU_IDS:-0}"
MODES="${MODES:-z z0 free}"
FINAL_STEPS="${FINAL_STEPS:-2000}"
N_REPEATS="${N_REPEATS:-8}"
MASKS="${MASKS:-outputs/eval/zopt_masks.pt}"
OUT_JSON="${OUT_JSON:-outputs/eval/effective_mask_analysis.json}"
OUT_TENSOR="${OUT_TENSOR:-outputs/eval/effective_mask_analysis.pt}"
PLOT_DIR="${PLOT_DIR:-outputs/plots/z_analysis}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-outputs/.matplotlib}"

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="$GPU_IDS" python evaluation/analyze_effective_masks.py \
  --masks "$MASKS" --modes $MODES --n_repeats "$N_REPEATS" \
  --final_steps "$FINAL_STEPS" --out_json "$OUT_JSON" \
  --out_tensor "$OUT_TENSOR" --plot_dir "$PLOT_DIR"
