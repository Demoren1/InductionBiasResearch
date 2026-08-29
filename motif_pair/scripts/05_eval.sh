#!/usr/bin/env bash
# Evaluate only split.json test_tasks using freshly trained downstream MLPs.
set -euo pipefail
cd "$(dirname "$0")/.."
SPLIT_JSON="${SPLIT_JSON:-outputs/split.json}"
GEN_ROOT="${GEN_ROOT:-outputs/generative}"
OUT_DIR="${OUT_DIR:-outputs/eval}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
EVAL_N_MASKS="${EVAL_N_MASKS:-64}"
EVAL_SEED="${EVAL_SEED:-42}"
CKPT_ROOT="${CKPT_ROOT:-outputs/checkpoints}"
IMPORTANCE_NAME="${IMPORTANCE_NAME:-importance.pt}"
TOP_FRAC="${TOP_FRAC:-0.1}"
EVAL_GPU="${EVAL_GPU:-${GPU_ID:-0}}"
export CUDA_VISIBLE_DEVICES="$EVAL_GPU"

python evaluation/eval_generated_masks.py --split "$SPLIT_JSON" \
  --cvae_ckpt "$GEN_ROOT/cvae/best.pt" --vae_ckpt "$GEN_ROOT/vae/best.pt" \
  --out_dir "$OUT_DIR" --steps "$EVAL_STEPS" --n_masks "$EVAL_N_MASKS" \
  --seed "$EVAL_SEED" --device cuda --ckpt_root "$CKPT_ROOT" \
  --importance_name "$IMPORTANCE_NAME" --top_frac "$TOP_FRAC"
python evaluation/summarize.py --results "$OUT_DIR/eval_results.json" --out "$OUT_DIR/summary.json"
