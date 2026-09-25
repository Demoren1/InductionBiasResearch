#!/usr/bin/env bash
set -euo pipefail

# Default: one nested group of 1–10 VAE, four seeds, with bank/VAE reuse for
# 1100 and 1101. STAGE=bank|vae|controls|agreement|evaluate|report resumes a stage.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras

OUT_DIR="${OUT_DIR:-pattern/outputs/length11_multi_bce_night}"
REUSE_PAIR_ROOT="${REUSE_PAIR_ROOT:-pattern/outputs/length11_pair1100_1101_20260924}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$OUT_DIR"
  exec 9>"$OUT_DIR/.night.lock"
  if ! flock -n 9; then
    echo "Another run is using $OUT_DIR" >&2
    exit 1
  fi
fi

args=(
  --out "$OUT_DIR"
  --reuse-pair "$REUSE_PAIR_ROOT"
  --gpu-ids "$GPU_IDS"
  --stage "$STAGE"
)
if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

python -m pattern.length11.night "${args[@]}"
