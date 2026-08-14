#!/usr/bin/env bash
# === Parameters (edit here or export before running) ===
KERNELS="${KERNELS:-3 5 7 9}"
N_VAL_SAMPLES="${N_VAL_SAMPLES:-10000}"
# ======================================================
set -euo pipefail
cd "$(dirname "$0")/.."
python data/generate.py --n_val "$N_VAL_SAMPLES"
python data/plot.py