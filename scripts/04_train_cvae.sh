#!/usr/bin/env bash
# === Parameters ===
TRAIN_KERNELS="3 3 5 5 7 7 9 9"
TRAIN_OFFSETS="0 6 2 4 0 2 4 6"
CVAE_EPOCHS="${CVAE_EPOCHS:-80}"
CVAE_BETA="${CVAE_BETA:-1.0}"
CVAE_MODE="${CVAE_MODE:-importance_maps}"   # or "selected" for binary masks
CVAE_K_ACTIVE="${CVAE_K_ACTIVE:-102}"
GPU_IDS="${GPU_IDS:-0}"
# ===================
# Train the CVAE on the selected (best 10%) masks, conditioned on kernel size
# and offset, using 8 random training pairs (each kernel x 2 offsets, balanced).
#
# Single GPU training — specify which card:
#     GPU_IDS="2" bash scripts/04_train_cvae.sh
#   or (default: first card):
#     bash scripts/04_train_cvae.sh
#
# Requires: outputs/checkpoints/kernel_{k}/offset_{s}/best10pct.pt
#           (run 03_select.sh first).
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA=""
if [ "$CVAE_MODE" = "importance_maps" ]; then EXTRA="--importance_maps"; fi
if [ "$CVAE_MODE" = "ideal_masks" ]; then EXTRA="--ideal_masks"; fi

echo "Using GPU: ${GPU_IDS}"

CUDA_VISIBLE_DEVICES="$GPU_IDS" python models/train_cvae.py --mode train --epochs "$CVAE_EPOCHS" --beta "$CVAE_BETA" \
  --kernels $TRAIN_KERNELS --offsets $TRAIN_OFFSETS $EXTRA
CUDA_VISIBLE_DEVICES="$GPU_IDS" python models/train_cvae.py --mode sample --k_active "$CVAE_K_ACTIVE" \
  --kernels $TRAIN_KERNELS --offsets $TRAIN_OFFSETS
echo "CVAE done -> outputs/cvae/"
echo "--- training deterministic regressor baseline ---"
CUDA_VISIBLE_DEVICES="$GPU_IDS" python evaluation/baselines.py