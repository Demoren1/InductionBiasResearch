#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

OUT_ROOT="${OUT_ROOT:-pattern/outputs/bilevel_mask/length32_pattern4_retrain_optimizer_20260915}"
MAX_JOBS="${MAX_JOBS:-16}"
mkdir -p "$OUT_ROOT"

# name weight_lr z_lr weight_steps
configs=(
  "best 0.003 0.3 10"
  "cheap 0.003 0.3 2"
  "stable 0.001 0.03 2"
)

pids=()
job_index=0
for seed in 42 43 44; do
  for spec in "${configs[@]}"; do
    read -r name weight_lr z_lr weight_steps <<<"$spec"
    out="$OUT_ROOT/${name}_seed${seed}"
    gpu="$((job_index % 8))"
    CUDA_VISIBLE_DEVICES="$gpu" python -m pattern.bilevel_mask.tune_length32 \
      --mode outer --device cuda --output "$out" --seed "$seed" \
      --latent-dim 4 --generator-width 16 --weight-lr "$weight_lr" \
      --weight-steps-per-z "$weight_steps" --z-lr "$z_lr" \
      --temperature-end 0.05 --generator-lr 0.01 --z-radius 8 \
      --outer-steps 100 --binary-penalty 0.01 --eval-weight-steps 1000 \
      >"$out.log" 2>&1 &
    pids+=("$!")
    job_index="$((job_index + 1))"
    if (( ${#pids[@]} >= MAX_JOBS )); then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
    fi
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done
