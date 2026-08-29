#!/usr/bin/env bash
# Zero-shot task-OOD experiment: learn the structural prior on 12 patterns and
# evaluate freshly trained masked MLPs on four held-out patterns.
SPLIT_SEED="${SPLIT_SEED:-42}"
GPU_ID="${GPU_ID:-0}"
CVAE_EPOCHS="${CVAE_EPOCHS:-80}"
CVAE_BETA="${CVAE_BETA:-0.1}"
CVAE_SEED="${CVAE_SEED:-42}"
CVAE_TOP_FRAC="${CVAE_TOP_FRAC:-0.1}"
CVAE_IMPORTANCE="${CVAE_IMPORTANCE:-importance.pt}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
EVAL_N_MASKS="${EVAL_N_MASKS:-64}"
EVAL_SEED="${EVAL_SEED:-42}"
OUT_ROOT="${OUT_ROOT:-outputs/ood/split_seed_${SPLIT_SEED}}"
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p "$OUT_ROOT/cvae" "$OUT_ROOT/eval"
python evaluation/ood_split.py --seed "$SPLIT_SEED" \
  --out "$OUT_ROOT/split.json" --shell_out "$OUT_ROOT/split.env"
# shellcheck source=/dev/null
source "$OUT_ROOT/split.env"

for pattern in $TRAIN_PATTERNS; do
  importance_path="outputs/checkpoints/pattern_${pattern}/${CVAE_IMPORTANCE}"
  if [ ! -f "$importance_path" ]; then
    echo "Missing meta-train importance maps: $importance_path" >&2
    echo "Run scripts/01_generate_data.sh, scripts/02_train.sh, and scripts/03_select.sh first." >&2
    exit 1
  fi
done

echo "==> OOD split seed=${SPLIT_SEED}"
echo "    meta-train: ${TRAIN_PATTERNS}"
echo "    meta-test : ${TEST_PATTERNS}"
echo "==> Training VAE on meta-train tasks only"
CUDA_VISIBLE_DEVICES="$GPU_ID" python models/train_cvae.py --mode train \
  --epochs "$CVAE_EPOCHS" --beta "$CVAE_BETA" \
  --loss bce --reduction sum --importance_maps \
  --patterns $TRAIN_PATTERNS --importance_name "$CVAE_IMPORTANCE" \
  --top_frac "$CVAE_TOP_FRAC" --out_dir "$OUT_ROOT/cvae" \
  --seed "$CVAE_SEED"

echo "==> Training deterministic baseline on meta-train tasks only"
CUDA_VISIBLE_DEVICES="$GPU_ID" python evaluation/baselines.py \
  --patterns $TRAIN_PATTERNS \
  --checkpoint_path "$OUT_ROOT/det_reg.pt" \
  --importance_name "$CVAE_IMPORTANCE" --top_frac "$CVAE_TOP_FRAC" \
  --seed "$CVAE_SEED"

echo "==> Evaluating zero-shot structural priors on held-out tasks"
CUDA_VISIBLE_DEVICES="$GPU_ID" python evaluation/eval_generated_masks.py \
  --patterns $TEST_PATTERNS \
  --methods random random_exact32 ideal cvae mean_imp det_reg \
  --prior_patterns $TRAIN_PATTERNS --train_patterns $TRAIN_PATTERNS \
  --cvae_ckpt "$OUT_ROOT/cvae/cvae_best.pt" \
  --det_reg_checkpoint "$OUT_ROOT/det_reg.pt" \
  --importance_name "$CVAE_IMPORTANCE" --top_frac "$CVAE_TOP_FRAC" \
  --steps "$EVAL_STEPS" --n_masks "$EVAL_N_MASKS" --seed "$EVAL_SEED" \
  --out_dir "$OUT_ROOT/eval"

python evaluation/summarize_ood.py \
  --results "$OUT_ROOT/eval/eval_results.json" \
  --split "$OUT_ROOT/split.json" --out "$OUT_ROOT/summary.json"

echo "OOD experiment finished -> ${OUT_ROOT}"
