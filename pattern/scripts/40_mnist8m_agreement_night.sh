#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras

OUT_DIR="${OUT_DIR:-pattern/outputs/mnist8m_night_20260927}"
REUSE_PAIR38="${REUSE_PAIR38:-pattern/outputs/mnist8m_raw_mlp_bce/pair38}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
ANCHORS_PER_TASK="${ANCHORS_PER_TASK:-4}"
SEARCH_STEPS="${SEARCH_STEPS:-24000}"
EVALUATION_STEPS="${EVALUATION_STEPS:-30000}"
TRANSFER_STEPS="${TRANSFER_STEPS:-20000}"

read -r -a gpu_ids_array <<< "$GPU_IDS"
if (( ${#gpu_ids_array[@]} == 0 )); then
  echo "GPU_IDS is empty" >&2
  exit 1
fi

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
  --reuse-pair38 "$REUSE_PAIR38"
  --gpu-ids "${gpu_ids_array[@]}"
  --stage "$STAGE"
  --anchors-per-task "$ANCHORS_PER_TASK"
  --search-steps "$SEARCH_STEPS"
  --evaluation-steps "$EVALUATION_STEPS"
  --transfer-steps "$TRANSFER_STEPS"
)
if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

python -m pattern.mnist8m_night "${args[@]}"
