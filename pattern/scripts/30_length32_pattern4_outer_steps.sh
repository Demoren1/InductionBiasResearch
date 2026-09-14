#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

OUT_ROOT="${OUT_ROOT:-pattern/outputs/bilevel_mask/length32_pattern4_outer_steps_20260915}"
MAX_JOBS="${MAX_JOBS:-16}"
mkdir -p "$OUT_ROOT"

# name generator_lr radius binary_penalty
configs=(
  "fast 0.01 8 0.01"
  "stable 0.003 12 0"
)

pids=()
job_index=0
for seed in 42 43 44; do
  for spec in "${configs[@]}"; do
    read -r name generator_lr radius binary_penalty <<<"$spec"
    for outer_steps in 30 100; do
      out="$OUT_ROOT/${name}_o${outer_steps}_seed${seed}"
      gpu="$((job_index % 8))"
      CUDA_VISIBLE_DEVICES="$gpu" python -m pattern.bilevel_mask.tune_length32 \
        --mode outer --device cuda --output "$out" --seed "$seed" \
        --latent-dim 4 --generator-width 16 \
        --weight-steps-per-z 5 --z-lr 0.1 --temperature-end 0.05 \
        --generator-lr "$generator_lr" --z-radius "$radius" \
        --outer-steps "$outer_steps" --binary-penalty "$binary_penalty" \
        --eval-weight-steps 1000 >"$out.log" 2>&1 &
      pids+=("$!")
      job_index="$((job_index + 1))"
      if (( ${#pids[@]} >= MAX_JOBS )); then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
      fi
    done
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done
