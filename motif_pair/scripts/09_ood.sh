#!/usr/bin/env bash
# Full leakage-free OOD run.  Split creation, raw-bank selection, generative
# prior training and downstream evaluation are explicit separate stages.
set -euo pipefail
cd "$(dirname "$0")/.."

SPLIT_SEED="${SPLIT_SEED:-42}"
OUT_ROOT="${OUT_ROOT:-outputs/ood/split_seed_${SPLIT_SEED}}"
SPLIT_JSON="${SPLIT_JSON:-$OUT_ROOT/split.json}"
RUN_ROOT="${RUN_ROOT:-$OUT_ROOT}"
DATA_DIR="${DATA_DIR:-$OUT_ROOT/data}"
SPLIT_KIND="${SPLIT_KIND:-pair_disjoint}"
HELDOUT_GAPS="${HELDOUT_GAPS:-}"
PAIR_POLICY="${PAIR_POLICY:-shared}"
CONDITION_ENCODING="${CONDITION_ENCODING:-one_hot}"
if [[ "$SPLIT_KIND" == "gap_heldout" && "$CONDITION_ENCODING" != "scalar" ]]; then
  echo "SPLIT_KIND=gap_heldout requires CONDITION_ENCODING=scalar; use scripts/10_gap_ood.sh" >&2
  exit 2
fi
GPU_ID="${GPU_ID:-0}"
DATA_GPU="${DATA_GPU:-$GPU_ID}"
POSTPROC_GPU="${POSTPROC_GPU:-$GPU_ID}"
GEN_GPU="${GEN_GPU:-$GPU_ID}"
EVAL_GPU="${EVAL_GPU:-$GPU_ID}"
CKPT_ROOT="${CKPT_ROOT:-$OUT_ROOT/checkpoints}"
export SPLIT_SEED SPLIT_JSON OUT_ROOT RUN_ROOT DATA_DIR SPLIT_KIND HELDOUT_GAPS PAIR_POLICY CONDITION_ENCODING
export GPU_ID DATA_GPU POSTPROC_GPU GEN_GPU EVAL_GPU CKPT_ROOT
bash scripts/01_generate_data.sh
bash scripts/02_train.sh
bash scripts/03_select.sh
OUT_ROOT="$OUT_ROOT/generative" bash scripts/04_train_generative.sh
GEN_ROOT="$OUT_ROOT/generative" OUT_DIR="$OUT_ROOT/eval" bash scripts/05_eval.sh
PLOT_PDF="${PLOT_PDF:-1}" PLOT_REQUIRE_COMPLETE=1 RUN_ROOT="$RUN_ROOT" DATA_DIR="$DATA_DIR" \
  PLOT_DATA_DIR="$DATA_DIR" PLOT_DEVICE=cuda bash scripts/06_plot.sh
echo "OOD results -> $OUT_ROOT/eval/summary.json"
