#!/usr/bin/env bash
# === Parameters ===
KERNELS="${KERNELS:-3 5 7 9}"
OFFSETS="${OFFSETS:-0 2 4 6}"
# ===================
# Select the best 10% of MLPs per (kernel, offset), save weights+masks
# together, and plot the selected masks overlaid with their trained weights.
set -euo pipefail
cd "$(dirname "$0")/.."

for K in $KERNELS; do
  for S in $OFFSETS; do
    python selection/select_best.py --kernel "$K" --offset "$S"
  done
done

python selection/plot_selected.py

echo "--- extracting importance maps ---"
python evaluation/importance.py
echo "--- plotting importance profiles ---"
python evaluation/plot_importance.py