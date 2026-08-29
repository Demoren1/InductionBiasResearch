#!/usr/bin/env bash
# Read-only diagnostic plotting for a completed or partially completed OOD run.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ROOT="${RUN_ROOT:-outputs/ood/pair_disjoint_aligned_seed_42}"
PLOTS_DIR="${PLOTS_DIR:-$RUN_ROOT/plots}"
PLOT_DATA_DIR="${PLOT_DATA_DIR:-${DATA_DIR:-}}"
PLOT_DEVICE="${PLOT_DEVICE:-auto}"
PLOT_MAX_TASKS="${PLOT_MAX_TASKS:-16}"
PLOT_IMPORTANCE_NAME="${PLOT_IMPORTANCE_NAME:-${IMPORTANCE_NAME:-importance.pt}}"
PLOT_TOP_FRAC="${PLOT_TOP_FRAC:-${TOP_FRAC:-0.1}}"
EXTRA=()
if [[ "${PLOT_PDF:-0}" == "1" ]]; then EXTRA+=(--pdf); fi
if [[ "${PLOT_NO_GENERATOR:-0}" == "1" ]]; then EXTRA+=(--no-generator); fi
if [[ "${PLOT_REQUIRE_COMPLETE:-0}" == "1" ]]; then EXTRA+=(--require-complete); fi
if [[ -n "$PLOT_DATA_DIR" ]]; then EXTRA+=(--data-dir "$PLOT_DATA_DIR"); fi

python evaluation/plot_pipeline.py --run-root "$RUN_ROOT" --plots-dir "$PLOTS_DIR" \
  --device "$PLOT_DEVICE" --max-tasks "$PLOT_MAX_TASKS" \
  --importance-name "$PLOT_IMPORTANCE_NAME" --top-frac "$PLOT_TOP_FRAC" "${EXTRA[@]}"
