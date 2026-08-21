#!/usr/bin/env bash
# Loss/beta sweep on raw importance maps: train a CVAE variant and evaluate
# its generated masks downstream, to test whether the VAE collapse is caused
# by a KL/recon cost mismatch (lower beta) vs permutation inference (BCE-sum).
#
# Usage:
#   GPU_IDS="0" bash scripts/06_loss_sweep.sh
#   EPOCHS=120 EVAL_N_MASKS=64 bash scripts/06_loss_sweep.sh
#
# Requires: outputs/checkpoints/pattern_{pat}/importance.pt and
#           outputs/eval/det_reg.pt
#           (run 03_select.sh / 04_train_cvae.sh first).
# Expected reference numbers (FINDINGS.md): baseline VAE ~0.79, random ~0.89,
#           ideal ~0.93 downstream accuracy.
#
# Writes only to outputs/cvae_sweep/ and outputs/eval/eval_results_sweep_<name>.json
# (does not overwrite outputs/cvae or outputs/eval/eval_results.json).
GPU_IDS="${GPU_IDS:-0}"
EPOCHS="${EPOCHS:-80}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
EVAL_N_MASKS="${EVAL_N_MASKS:-32}"
EVAL_PATTERNS="${EVAL_PATTERNS:-0000 0110 1001 1111}"
TRAIN_PATTERNS="${TRAIN_PATTERNS:-0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111}"
set -euo pipefail
cd "$(dirname "$0")/.."

# name|loss|reduction|beta (override the whole list with VARIANTS="..." env var)
VARIANTS="${VARIANTS:-mse_b1.0|mse|mean|1.0
mse_b0.1|mse|mean|0.1
mse_b0.016|mse|mean|0.016
mse_b0.001|mse|mean|0.001
bce_sum_b1.0|bce|sum|1.0}"

echo "Loss/beta sweep on raw importance maps (GPU_IDS=${GPU_IDS}, epochs=${EPOCHS})"
EVAL_JSONS=()
while IFS='|' read -r NAME LOSS REDUCTION BETA; do
  [ -n "$NAME" ] || continue
  echo "=== variant ${NAME} (loss=${LOSS} reduction=${REDUCTION} beta=${BETA}) ==="

  echo "--- training CVAE -> outputs/cvae_sweep/${NAME}"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" python models/train_cvae.py --mode train \
    --epochs "$EPOCHS" --beta "$BETA" --loss "$LOSS" --reduction "$REDUCTION" \
    --patterns $TRAIN_PATTERNS --importance_maps \
    --importance_name importance.pt --out_dir "outputs/cvae_sweep/${NAME}"

  echo "--- evaluating generated masks"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" python evaluation/eval_generated_masks.py \
    --cvae_ckpt "outputs/cvae_sweep/${NAME}/cvae_best.pt" \
    --patterns $EVAL_PATTERNS --steps "$EVAL_STEPS" --n_masks "$EVAL_N_MASKS" \
    --out_suffix "_sweep_${NAME}"

  EVAL_JSONS+=("outputs/eval/eval_results_sweep_${NAME}.json")
  echo "=== variant ${NAME} done ==="
done <<< "$VARIANTS"

echo
echo "Loss/beta sweep finished. Results:"
printf '  %s\n' "${EVAL_JSONS[@]}"