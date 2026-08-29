#!/usr/bin/env bash
# Train a 512-candidate bank for every task explicitly listed in train_tasks.
# Example: GPU_IDS="0 1" SPLIT_JSON=outputs/split.json bash scripts/02_train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SPLIT_JSON="${SPLIT_JSON:-outputs/split.json}"
GPU_IDS="${GPU_IDS:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
N_MLPS_PER_TASK="${N_MLPS_PER_TASK:-512}"
MLPS_PER_GPU="${MLPS_PER_GPU:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
DATA_DIR="${DATA_DIR:-outputs/data}"
CKPT_ROOT="${CKPT_ROOT:-outputs/checkpoints}"
# A full candidate-bank rerun must not merge shards made with a previous GPU
# layout or candidate count.  Set to 0 only for an intentional manual resume.
CLEAN_CANDIDATES="${CLEAN_CANDIDATES:-1}"

if [[ ! -f "$SPLIT_JSON" ]]; then
  echo "Missing split: $SPLIT_JSON. Run scripts/01_generate_data.sh first." >&2
  exit 1
fi

mapfile -t TRAIN_TASKS < <(python - "$SPLIT_JSON" <<'PY'
import json
import sys

tasks = json.load(open(sys.argv[1], encoding="utf-8")).get("train_tasks")
if not isinstance(tasks, list) or not tasks or not all(isinstance(t, str) for t in tasks):
    raise SystemExit("split JSON needs a nonempty string train_tasks list")
print(*tasks, sep="\n")
PY
)
read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"
if (( NUM_GPUS == 0 )); then
  echo "GPU_IDS must contain at least one physical GPU id" >&2
  exit 1
fi

echo "Training ${#TRAIN_TASKS[@]} meta-train tasks across logical workers 0..$((NUM_GPUS - 1))"
for task in "${TRAIN_TASKS[@]}"; do
  echo "=== task $task ==="
  task_dir="$CKPT_ROOT/task_$task"
  if [[ "$CLEAN_CANDIDATES" == "1" && -d "$task_dir" ]]; then
    shopt -s nullglob
    stale_shards=("$task_dir"/gpu[0-9]*_round[0-9]*.pt)
    if (( ${#stale_shards[@]} )); then
      rm -- "${stale_shards[@]}"
      echo "Removed ${#stale_shards[@]} stale candidate shard(s) from $task_dir"
    fi
    shopt -u nullglob
  fi
  pids=()
  for worker in "${!GPU_ARRAY[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$worker]}" python models/train.py \
      --task "$task" --split_json "$SPLIT_JSON" \
      --gpu_id "$worker" --num_gpus "$NUM_GPUS" \
      --n_mlps "$N_MLPS_PER_TASK" \
      --train_steps "$TRAIN_STEPS" --mlps_per_gpu "$MLPS_PER_GPU" \
      --batch_size "$TRAIN_BATCH_SIZE" --data_dir "$DATA_DIR" --ckpt_root "$CKPT_ROOT" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done
echo "All meta-train candidate banks completed."
