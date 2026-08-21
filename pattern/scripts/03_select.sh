#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python selection/select_best.py
python selection/plot_selected.py
python evaluation/importance.py
python evaluation/plot_importance.py
# Alignment removed from the default pipeline (raw importance.pt is used).
# Manual: python evaluation/align_importance.py --method window|refmatch|selfalign|gold