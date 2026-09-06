#!/usr/bin/env bash
# Create a fresh leakage-free 48/16 split, then materialize its task data.
set -euo pipefail
cd "$(dirname "$0")/.."

SPLIT_SEED="${SPLIT_SEED:-42}"
SPLIT_JSON="${SPLIT_JSON:-outputs/split.json}"
DATA_DIR="${DATA_DIR:-outputs/data}"
DATA_GPU="${DATA_GPU:-${GPU_ID:-0}}"
AUDIT_JSON="${AUDIT_JSON:-$(dirname "$SPLIT_JSON")/shortcut_audit.json}"
SPLIT_KIND="${SPLIT_KIND:-pair_disjoint}"
HELDOUT_GAPS="${HELDOUT_GAPS:-}"
PAIR_POLICY="${PAIR_POLICY:-shared}"
CONDITION_ENCODING="${CONDITION_ENCODING:-one_hot}"
export CUDA_VISIBLE_DEVICES="$DATA_GPU"
export MOTIF_PAIR_DEVICE=cuda
mkdir -p "$(dirname "$SPLIT_JSON")"

SPLIT_ARGS=()
case "$SPLIT_KIND" in
  pair_disjoint)
    if [[ -n "$HELDOUT_GAPS" ]]; then
      echo "HELDOUT_GAPS is only valid with SPLIT_KIND=gap_heldout" >&2
      exit 2
    fi
    ;;
  gap_heldout)
    read -r -a HELDOUT_GAP_ARRAY <<< "$HELDOUT_GAPS"
    if (( ${#HELDOUT_GAP_ARRAY[@]} != 2 )); then
      echo "gap_heldout requires exactly two space-separated HELDOUT_GAPS (for example: '5 8')" >&2
      exit 2
    fi
    SPLIT_ARGS=(--split-kind gap_heldout --heldout-gaps "${HELDOUT_GAP_ARRAY[@]}" --pair-policy "$PAIR_POLICY")
    ;;
  *)
    echo "unknown SPLIT_KIND=$SPLIT_KIND (expected pair_disjoint or gap_heldout)" >&2
    exit 2
    ;;
esac

python evaluation/task_split.py --seed "$SPLIT_SEED" --out "$SPLIT_JSON" --device cuda "${SPLIT_ARGS[@]}"
python data/generate.py --split "$SPLIT_JSON" --data-dir "$DATA_DIR" --device cuda \
  --condition-encoding "$CONDITION_ENCODING"
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
echo "$SPLIT_KIND split, datasets ($DATA_DIR), and shortcut audit written using CUDA GPU $DATA_GPU"
