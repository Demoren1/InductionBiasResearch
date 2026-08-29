#!/usr/bin/env bash
# Create a fresh leakage-free 48/16 split, then materialize its task data.
set -euo pipefail
cd "$(dirname "$0")/.."

SPLIT_SEED="${SPLIT_SEED:-42}"
SPLIT_JSON="${SPLIT_JSON:-outputs/split.json}"
DATA_DIR="${DATA_DIR:-outputs/data}"
DATA_GPU="${DATA_GPU:-${GPU_ID:-0}}"
AUDIT_JSON="${AUDIT_JSON:-$(dirname "$SPLIT_JSON")/shortcut_audit.json}"
export CUDA_VISIBLE_DEVICES="$DATA_GPU"
export MOTIF_PAIR_DEVICE=cuda
mkdir -p "$(dirname "$SPLIT_JSON")"

python evaluation/task_split.py --seed "$SPLIT_SEED" --out "$SPLIT_JSON" --device cuda
python data/generate.py --split "$SPLIT_JSON" --data-dir "$DATA_DIR" --device cuda
python data/audit.py --split "$SPLIT_JSON" --out "$AUDIT_JSON" --device cuda
python - "$AUDIT_JSON" <<'PY'
import json, sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
if result["mean_accuracy"] > .54 or result["max_accuracy"] > .58:
    raise SystemExit(
        f"shortcut audit failed: mean={result['mean_accuracy']:.4f}, "
        f"max={result['max_accuracy']:.4f}"
    )
PY
echo "split, datasets ($DATA_DIR), and shortcut audit written using CUDA GPU $DATA_GPU"
