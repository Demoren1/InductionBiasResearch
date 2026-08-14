#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python data/generate.py
python data/plot.py
echo "data + plots -> outputs/data  outputs/plots/data"