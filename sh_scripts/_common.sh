#!/usr/bin/env bash
# Shared environment setup; source this file from a launch script.
set -euo pipefail

GE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$GE_PROJECT_ROOT"

GE_CONDA_ROOT="${GE_CONDA_ROOT:-/home/udeneev-av/miniconda3}"
source "$GE_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate ras

read -r -a GE_GPU_IDS <<< "${GPU_IDS//,/ }"
if (( ${#GE_GPU_IDS[@]} == 0 )); then
  printf 'GPU_IDS must contain at least one GPU number.\n' >&2
  exit 1
fi
for GE_GPU_ID in "${GE_GPU_IDS[@]}"; do
  if [[ ! "$GE_GPU_ID" =~ ^[0-9]+$ ]]; then
    printf 'Invalid GPU number: %s\n' "$GE_GPU_ID" >&2
    exit 1
  fi
done
GE_VISIBLE_GPUS="$(IFS=,; printf '%s' "${GE_GPU_IDS[*]}")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GE_GPU_UUID:-$GE_VISIBLE_GPUS}"
export PYTHONUNBUFFERED=1
export GENERATOR_EVALUATOR_PROGRESS=1

GE_SEED="${GE_SEED:-4100}"
GE_STAMP="$(TZ=Europe/Moscow date +%Y%m%d_%H%M%S)"

ge_run() {
  printf '\nRunning:'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}
