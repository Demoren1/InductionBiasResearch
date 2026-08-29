#!/usr/bin/env bash
# Select top-10% candidates and extract raw masked-weight importance maps.
set -euo pipefail
cd "$(dirname "$0")/.."

SPLIT_JSON="${SPLIT_JSON:-outputs/split.json}"
POSTPROC_GPU="${POSTPROC_GPU:-${GPU_ID:-0}}"
CKPT_ROOT="${CKPT_ROOT:-outputs/checkpoints}"
TOP_FRAC="${TOP_FRAC:-0.1}"
export CUDA_VISIBLE_DEVICES="$POSTPROC_GPU"
python selection/select_best.py --split_json "$SPLIT_JSON" --device cuda \
  --top_fraction "$TOP_FRAC" --ckpt_root "$CKPT_ROOT"
python evaluation/importance.py --split_json "$SPLIT_JSON" --device cuda --ckpt_root "$CKPT_ROOT"
