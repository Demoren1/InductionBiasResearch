#!/usr/bin/env bash
# === Parameters ===
# Default train patterns = all 16 (unconditional VAE on |W| maps from every pattern)
TRAIN_PATTERNS="${TRAIN_PATTERNS:-0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111}"
CVAE_EPOCHS="${CVAE_EPOCHS:-80}"
CVAE_BETA="${CVAE_BETA:-1.0}"
CVAE_MODE="${CVAE_MODE:-importance_maps}"   # or "selected" for binary masks
CVAE_K_ACTIVE="${CVAE_K_ACTIVE:-}"
CVAE_IMPORTANCE="${CVAE_IMPORTANCE:-importance.pt}"   # raw importance maps (no alignment)
CVAE_LOSS="${CVAE_LOSS:-mse}"                          # importance loss: mse|bce
CVAE_REDUCTION="${CVAE_REDUCTION:-mean}"               # per-pixel reduction: mean|sum
GPU_IDS="${GPU_IDS:-0}"
# ===================
# Train an unconditional VAE on |W| importance maps from all patterns. The
# mask is shared across patterns (config.CVAE_COND_DIM == 0), so every pattern
# contributes to the same shared prior.
#
# Single GPU training — specify which card:
#     GPU_IDS="2" bash scripts/04_train_cvae.sh
#   or (default: first card):
#     bash scripts/04_train_cvae.sh
#
# Raw importance maps by default (alignment was removed from the pipeline; run
# align_importance.py manually if you want aligned maps as CVAE_IMPORTANCE).
#
# Requires: outputs/checkpoints/pattern_{pat}/best10pct.pt and
#           outputs/checkpoints/pattern_{pat}/importance.pt
#           (run 03_select.sh / importance first).
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA=""
if [ "$CVAE_MODE" = "importance_maps" ]; then EXTRA="--importance_maps"; fi
if [ "$CVAE_MODE" = "ideal_masks" ]; then EXTRA="--ideal_masks"; fi

echo "Using GPU: ${GPU_IDS}"
echo "Train patterns: ${TRAIN_PATTERNS}"

# 1) train
CUDA_VISIBLE_DEVICES="$GPU_IDS" python models/train_cvae.py --mode train \
  --epochs "$CVAE_EPOCHS" --beta "$CVAE_BETA" \
  --loss "$CVAE_LOSS" --reduction "$CVAE_REDUCTION" \
  --patterns $TRAIN_PATTERNS $EXTRA \
  --importance_name "$CVAE_IMPORTANCE"

# 2) sample (deterministic top-k if k_active set, else Bernoulli)
SAMPLE_EXTRA=""
if [ -n "$CVAE_K_ACTIVE" ]; then SAMPLE_EXTRA="--k_active $CVAE_K_ACTIVE"; fi
CUDA_VISIBLE_DEVICES="$GPU_IDS" python models/train_cvae.py --mode sample \
  --patterns $TRAIN_PATTERNS $SAMPLE_EXTRA

echo "CVAE done -> outputs/cvae/"
echo "--- training deterministic regressor baseline ---"
CUDA_VISIBLE_DEVICES="$GPU_IDS" python evaluation/baselines.py