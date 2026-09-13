#!/usr/bin/env bash
# Build the seq_len=32, pattern_len=5 bank and train 32 VAE pairs.
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

OUT_DIR="${OUT_DIR:-pattern/outputs/fixed_k5_agreement/sampling_32pairs_20260913}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a GPUS <<< "$GPU_IDS"

python -m pattern.fixed_k5_agreement.run --out "$OUT_DIR" --stage prepare \
  --vae-pairs 32 --gpus "${GPUS[@]}"
python -m pattern.fixed_k5_agreement.run --out "$OUT_DIR" --stage bank \
  --vae-pairs 32 --gpus "${GPUS[@]}"
python -m pattern.fixed_k5_agreement.run --out "$OUT_DIR" --stage vae \
  --vae-pairs 32 --gpus "${GPUS[@]}"

echo "Prepared 32 VAE pairs: $OUT_DIR"
