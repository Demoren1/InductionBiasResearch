#!/usr/bin/env bash
# Full pattern pipeline: data -> train -> select -> CVAE -> eval.
export GPU_IDS="${GPU_IDS:-0 1 2 3}"
export TRAIN_STEPS="${TRAIN_STEPS:-2000}"
export CVAE_EPOCHS="${CVAE_EPOCHS:-80}"
export EVAL_STEPS="${EVAL_STEPS:-2000}"
set -euo pipefail
cd "$(dirname "$0")/.."
echo "==> 1/5 Generating data + plots"
bash scripts/01_generate_data.sh
echo "==> 2/5 Training masked MLPs (GPU_IDS=${GPU_IDS})"
bash scripts/02_train.sh
echo "==> 3/5 Selecting best 10% + importance maps"
bash scripts/03_select.sh
echo "==> 4/5 Training CVAE on importance maps + det_reg baseline"
GPU_IDS="${GPU_IDS%% *}" bash scripts/04_train_cvae.sh
echo "==> 5/5 Evaluating generated masks"
bash scripts/05_eval.sh
echo
echo "Pipeline finished."
echo "  plots     : outputs/plots/"
echo "  weights   : outputs/checkpoints/"
echo "  cvae      : outputs/cvae/"
echo "  eval      : outputs/eval/"
