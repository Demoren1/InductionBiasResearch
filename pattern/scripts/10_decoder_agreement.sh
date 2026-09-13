#!/usr/bin/env bash
# Run in the ras environment with GPU access (outside the Codex sandbox).
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${GPU_ID:-2}"
OUT_DIR="${OUT_DIR:-outputs/decoder_agreement/seed_20260906}"
# This launcher reproduces the immutable 2026-09-06 baseline protocol.
python evaluation/train_agreement_vaes.py --out_dir "$OUT_DIR" --epochs 80 --device cuda
python evaluation/run_decoder_agreement.py --out_dir "$OUT_DIR" --device cuda --stage all
python evaluation/report_decoder_agreement.py --out_dir "$OUT_DIR"
