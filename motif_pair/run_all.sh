#!/usr/bin/env bash
# Reproduce one pair-disjoint motif-pair OOD run end to end.
# Heavy numerical stages are CUDA-only; CPU is used only for manifests/plots.
set -euo pipefail

cd "$(dirname "$0")"

CONDA_SH="${CONDA_SH:-/home/udeneev-av/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-ras}"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "Missing conda initialization script: $CONDA_SH" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

SPLIT_SEED="${SPLIT_SEED:-42}"
RUN_ROOT="${RUN_ROOT:-outputs/ood/pair_disjoint_seed_${SPLIT_SEED}}"
SPLIT_JSON="${SPLIT_JSON:-$RUN_ROOT/split.json}"
CKPT_ROOT="${CKPT_ROOT:-$RUN_ROOT/checkpoints}"
DATA_DIR="${DATA_DIR:-$RUN_ROOT/data}"

# Physical GPU ids. Candidate banks and beta candidates use all listed GPUs;
# lighter CUDA stages use the first id unless overridden.
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_ARRAY <<< "$GPU_IDS"
if (( ${#GPU_ARRAY[@]} == 0 )); then
  echo "GPU_IDS must contain at least one physical GPU id" >&2
  exit 1
fi
DATA_GPU="${DATA_GPU:-${GPU_ARRAY[0]}}"
POSTPROC_GPU="${POSTPROC_GPU:-${GPU_ARRAY[0]}}"
EVAL_GPU="${EVAL_GPU:-${GPU_ARRAY[0]}}"
PLOT_GPU="${PLOT_GPU:-$EVAL_GPU}"
BETA_GPU_IDS="${BETA_GPU_IDS:-$GPU_IDS}"
read -r -a BETA_GPU_ARRAY <<< "$BETA_GPU_IDS"
if (( ${#BETA_GPU_ARRAY[@]} == 0 )); then
  echo "BETA_GPU_IDS must contain at least one physical GPU id" >&2
  exit 1
fi

N_MLPS_PER_TASK="${N_MLPS_PER_TASK:-512}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
CVAE_EPOCHS="${CVAE_EPOCHS:-80}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
EVAL_N_MASKS="${EVAL_N_MASKS:-64}"
TOP_FRAC="${TOP_FRAC:-0.1}"
IMPORTANCE_NAME="${IMPORTANCE_NAME:-importance.pt}"
BETAS="${BETAS:-1.0 0.3 0.1 0.03 0.01 0.003 0.001 0.0003}"
read -r -a BETA_ARRAY <<< "$BETAS"

export SPLIT_SEED SPLIT_JSON CKPT_ROOT DATA_DIR GPU_IDS DATA_GPU POSTPROC_GPU EVAL_GPU
export N_MLPS_PER_TASK TRAIN_STEPS TOP_FRAC IMPORTANCE_NAME
mkdir -p "$RUN_ROOT"

plot_available() {
  CUDA_VISIBLE_DEVICES="$PLOT_GPU" RUN_ROOT="$RUN_ROOT" \
    PLOT_DATA_DIR="$DATA_DIR" PLOT_DEVICE=cuda \
    PLOT_TOP_FRAC="$TOP_FRAC" PLOT_IMPORTANCE_NAME="$IMPORTANCE_NAME" \
    PLOT_PDF="${PLOT_PDF:-1}" PLOT_REQUIRE_COMPLETE="${PLOT_REQUIRE_COMPLETE:-0}" \
    bash scripts/06_plot.sh
}

echo "[1/6] Pair-disjoint split, datasets, and shortcut audit on GPU $DATA_GPU"
bash scripts/01_generate_data.sh
plot_available

echo "[2/6] Candidate MLP banks on GPUs: $GPU_IDS"
bash scripts/02_train.sh

echo "[3/6] Candidate selection (top fraction $TOP_FRAC) and continuous importance maps on GPU $POSTPROC_GPU"
bash scripts/03_select.sh
plot_available

mapfile -t TRAIN_TASKS < <(python - "$SPLIT_JSON" <<'PY'
import json
import sys
tasks = json.load(open(sys.argv[1], encoding="utf-8")).get("train_tasks")
if not isinstance(tasks, list) or not tasks:
    raise SystemExit("split JSON needs nonempty train_tasks")
print(*tasks, sep="\n")
PY
)

echo "[4/6] CVAE/VAE beta sweep on GPUs: $BETA_GPU_IDS"
python models/sweep_beta.py \
  --tasks "${TRAIN_TASKS[@]}" \
  --out_dir "$RUN_ROOT/generative" \
  --variants cvae vae \
  --betas "${BETA_ARRAY[@]}" \
  --gpu_ids "${BETA_GPU_ARRAY[@]}" \
  --epochs "$CVAE_EPOCHS" \
  --top_frac "$TOP_FRAC" \
  --importance_name "$IMPORTANCE_NAME" \
  --split "$SPLIT_JSON" \
  --ckpt_root "$CKPT_ROOT"
plot_available

echo "[5/6] Held-out downstream evaluation on GPU $EVAL_GPU"
GEN_ROOT="$RUN_ROOT/generative" OUT_DIR="$RUN_ROOT/eval" \
  EVAL_STEPS="$EVAL_STEPS" EVAL_N_MASKS="$EVAL_N_MASKS" \
  bash scripts/05_eval.sh

echo "[6/6] Final stage-wise plots and artifact manifest"
PLOT_REQUIRE_COMPLETE=1 plot_available

echo "Completed: $RUN_ROOT"
echo "Summary:   $RUN_ROOT/eval/summary.json"
echo "Plots:     $RUN_ROOT/plots/manifest.json"
