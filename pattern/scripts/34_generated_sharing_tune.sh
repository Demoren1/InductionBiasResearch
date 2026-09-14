#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

OUT_ROOT="${OUT_ROOT:-pattern/outputs/bilevel_mask/generated_sharing_tuning_20260915}"
MAX_JOBS="${MAX_JOBS:-16}"
mkdir -p "$OUT_ROOT"

pids=()
job_index=0
for inner_lr in 0.03 0.1; do
  for inner_steps in 5 10 20; do
    for generator_lr in 0.001 0.003 0.01; do
      tag="lr${inner_lr}_steps${inner_steps}_glr${generator_lr}"
      gpu="$((job_index % 8))"
      CUDA_VISIBLE_DEVICES="$gpu" python -m pattern.bilevel_mask.tune_generated_sharing \
        --output "$OUT_ROOT/$tag" --device cuda --seed 42 \
        --inner-lr "$inner_lr" --inner-steps "$inner_steps" \
        --generator-lr "$generator_lr" >"$OUT_ROOT/$tag.log" 2>&1 &
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
