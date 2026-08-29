#!/usr/bin/env bash
# Train both the gap-conditioned CVAE and an otherwise matched VAE ablation.
set -euo pipefail
cd "$(dirname "$0")/.."

SPLIT_JSON="${SPLIT_JSON:-outputs/split.json}"
OUT_ROOT="${OUT_ROOT:-outputs/generative}"
EPOCHS="${CVAE_EPOCHS:-80}"
BETA="${CVAE_BETA:-0.1}"
SEED="${CVAE_SEED:-42}"
TOP_FRAC="${TOP_FRAC:-0.1}"
IMPORTANCE_NAME="${IMPORTANCE_NAME:-importance.pt}"
CKPT_ROOT="${CKPT_ROOT:-outputs/checkpoints}"
GEN_GPU="${GEN_GPU:-${GPU_ID:-0}}"
export CUDA_VISIBLE_DEVICES="$GEN_GPU"

mapfile -t TRAIN_TASKS < <(python - "$SPLIT_JSON" <<'PY'
import json, sys
print(*json.load(open(sys.argv[1]))['train_tasks'], sep='\n')
PY
)
for task in "${TRAIN_TASKS[@]}"; do
  test -f "$CKPT_ROOT/task_${task}/${IMPORTANCE_NAME}" || {
    echo "missing continuous importance maps for $task; run scripts/02_train.sh and 03_select.sh" >&2; exit 1; }
done
python models/train_cvae.py --tasks "${TRAIN_TASKS[@]}" --out_dir "$OUT_ROOT/cvae" \
  --split "$SPLIT_JSON" --variant cvae --epochs "$EPOCHS" --beta "$BETA" --top_frac "$TOP_FRAC" \
  --importance_name "$IMPORTANCE_NAME" --seed "$SEED" --device cuda --ckpt_root "$CKPT_ROOT"
python models/train_cvae.py --tasks "${TRAIN_TASKS[@]}" --out_dir "$OUT_ROOT/vae" \
  --split "$SPLIT_JSON" --variant vae --epochs "$EPOCHS" --beta "$BETA" --top_frac "$TOP_FRAC" \
  --importance_name "$IMPORTANCE_NAME" --seed "$SEED" --device cuda --ckpt_root "$CKPT_ROOT"
