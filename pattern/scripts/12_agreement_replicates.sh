#!/usr/bin/env bash
# Tuned label-free decoder agreement over 32 independent VAE pairs.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
OUT_DIR="${OUT_DIR:-outputs/decoder_agreement/multiseed32_tuned_20260912}"
python evaluation/run_agreement_replicates.py \
  --out_dir "$OUT_DIR" \
  --gpus 0 1 2 3 4 5 6 7 \
  --stage "${STAGE:-train-search}" \
  --device cuda
