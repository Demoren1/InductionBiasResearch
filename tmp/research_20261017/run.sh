#!/usr/bin/env bash
# Existing banks only; all new code, logs and artifacts stay in this directory.
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-mps}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$EXPERIMENT_DIR/.cache/matplotlib"
mkdir -p "$EXPERIMENT_DIR/logs" "$MPLCONFIGDIR"
STAGE="${1:-all}"
run() { "$@" 2>&1 | tee -a "$EXPERIMENT_DIR/logs/$STAGE.log"; }
case "$STAGE" in
  train)
    run "$PYTHON" -u "$EXPERIMENT_DIR/experiment.py" train --device "$DEVICE"
    run "$PYTHON" -u "$EXPERIMENT_DIR/experiment.py" train --device "$DEVICE" \
      --kinds shared --seeds 4214 --patience 100 --suffix _no_stop
    ;;
  probes)
    run "$PYTHON" -u "$EXPERIMENT_DIR/pattern_probe.py"
    run "$PYTHON" -u "$EXPERIMENT_DIR/latent_probe.py"
    ;;
  transfer)
    run "$PYTHON" -u "$EXPERIMENT_DIR/experiment.py" transfer --device "$DEVICE" \
      --seeds 4214 4215
    run "$PYTHON" -u "$EXPERIMENT_DIR/experiment.py" transfer --device "$DEVICE" \
      --kinds reverse shared cvae cvae_elbo --seeds 4213 --optimize
    run "$PYTHON" -u "$EXPERIMENT_DIR/experiment.py" transfer --device "$DEVICE" \
      --kinds reverse --routing prior
    ;;
  reports)
    run "$PYTHON" -u "$EXPERIMENT_DIR/report.py"
    ;;
  agop)
    run "$PYTHON" -u "$EXPERIMENT_DIR/agop.py" train --device "$DEVICE"
    run "$PYTHON" -u "$EXPERIMENT_DIR/agop.py" pattern --device "$DEVICE" \
      --representations functional branch_agop branch_functional branch_zero
    run "$PYTHON" -u "$EXPERIMENT_DIR/agop.py" transfer --optimize --device "$DEVICE"
    run "$PYTHON" -u "$EXPERIMENT_DIR/agop.py" transfer --optimize \
      --z-method evolution --device "$DEVICE"
    run "$PYTHON" -u "$EXPERIMENT_DIR/agop_report.py"
    ;;
  all)
    for part in train probes transfer reports; do "$0" "$part"; done
    ;;
  *) echo "Usage: $0 [train|probes|transfer|reports|agop|all]" >&2; exit 2 ;;
esac
