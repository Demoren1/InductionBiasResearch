#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python selection/select_best.py
python selection/plot_selected.py
python evaluation/importance.py
python evaluation/plot_importance.py
python evaluation/align_importance.py --method window
python evaluation/align_importance.py --method refmatch