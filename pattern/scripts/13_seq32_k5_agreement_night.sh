#!/usr/bin/env bash
# Full overnight run: seq_len=32, pattern_len=5, 64 independent VAE pairs.
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

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-pattern/outputs/fixed_k5_agreement/night_${RUN_TAG}}"
STAGE="${STAGE:-all}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_ARRAY <<< "$GPU_IDS"

echo "Output: $OUT_DIR"
echo "Stage:  $STAGE"
echo "GPUs:   ${GPU_ARRAY[*]}"

python -m pattern.fixed_k5_agreement.run \
  --out "$OUT_DIR" \
  --stage "$STAGE" \
  --gpus "${GPU_ARRAY[@]}"

echo "Finished. Preliminary report: $OUT_DIR/RESULTS.md"

