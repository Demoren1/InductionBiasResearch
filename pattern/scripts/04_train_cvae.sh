#!/usr/bin/env bash
# === Parameters ===
# Default train patterns = all 16 (unconditional VAE on |W| maps from every pattern)
TRAIN_PATTERNS="${TRAIN_PATTERNS:-0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111}"
CVAE_EPOCHS="${CVAE_EPOCHS:-80}"
CVAE_BETA="${CVAE_BETA:-0.1}"
CVAE_MODE="${CVAE_MODE:-importance_maps}"
CVAE_K_ACTIVE="${CVAE_K_ACTIVE:-32}"
CVAE_IMPORTANCE="${CVAE_IMPORTANCE:-importance.pt}"
CVAE_LOSS="${CVAE_LOSS:-bce}"
CVAE_REDUCTION="${CVAE_REDUCTION:-sum}"
CVAE_TOP_FRAC="${CVAE_TOP_FRAC:-0.1}"
CVAE_OUT_DIR="${CVAE_OUT_DIR:-outputs/cvae}"
CVAE_SEED="${CVAE_SEED:-42}"
GPU_IDS="${GPU_IDS:-0}"
# ===================
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA=""
if [ "$CVAE_MODE" = "importance_maps" ]; then EXTRA="--importance_maps"; fi
if [ "$CVAE_MODE" = "ideal_masks" ]; then EXTRA="--ideal_masks"; fi

echo "Using GPU: ${GPU_IDS}"
echo "CVAE: loss=${CVAE_LOSS}/${CVAE_REDUCTION} beta=${CVAE_BETA} top_frac=${CVAE_TOP_FRAC}"

# 1) train
CUDA_VISIBLE_DEVICES="$GPU_IDS" python models/train_cvae.py --mode train \
  --epochs "$CVAE_EPOCHS" --beta "$CVAE_BETA" \
  --loss "$CVAE_LOSS" --reduction "$CVAE_REDUCTION" \
  --patterns $TRAIN_PATTERNS $EXTRA \
  --importance_name "$CVAE_IMPORTANCE" --top_frac "$CVAE_TOP_FRAC" \
  --out_dir "$CVAE_OUT_DIR" --seed "$CVAE_SEED"

# 2) sample (deterministic top-k if k_active set, else Bernoulli)
SAMPLE_EXTRA=""
if [ -n "$CVAE_K_ACTIVE" ]; then SAMPLE_EXTRA="--k_active $CVAE_K_ACTIVE"; fi
CUDA_VISIBLE_DEVICES="$GPU_IDS" python models/train_cvae.py --mode sample \
  --patterns $TRAIN_PATTERNS $SAMPLE_EXTRA \
  --out_dir "$CVAE_OUT_DIR" --seed "$CVAE_SEED"

echo "CVAE done -> ${CVAE_OUT_DIR}/"
echo "--- training deterministic regressor baseline ---"
CUDA_VISIBLE_DEVICES="$GPU_IDS" python evaluation/baselines.py
