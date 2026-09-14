#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

OUT_ROOT="${OUT_ROOT:-pattern/outputs/bilevel_mask/length32_pattern4_tuning_20260915}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-pattern/outputs/bilevel_mask/length32_pattern4_joint_converged_20260915}"
mkdir -p "$OUT_ROOT/inner"

max_jobs="${MAX_JOBS:-16}"
job_index=0
pids=()
for seed in 42 43 44; do
  for weight_steps in 1 2 5 10; do
    for z_lr in 0.01 0.03 0.1; do
      for temperature in 0.05 0.15 0.3; do
        gpu="$((job_index % 8))"
        tag="seed${seed}_w${weight_steps}_z${z_lr}_t${temperature}"
        CUDA_VISIBLE_DEVICES="$gpu" python -m pattern.bilevel_mask.tune_length32 \
          --mode inner --device cuda \
          --checkpoint "$CHECKPOINT_ROOT/seed${seed}/training.pt" \
          --output "$OUT_ROOT/inner/$tag" --seed "$seed" \
          --weight-steps-per-z "$weight_steps" --z-lr "$z_lr" \
          --temperature-end "$temperature" --eval-weight-steps 1000 \
          >"$OUT_ROOT/inner/$tag.log" 2>&1 &
        pids+=("$!")
        job_index="$((job_index + 1))"
        if (( ${#pids[@]} >= max_jobs )); then
          wait "${pids[0]}"
          pids=("${pids[@]:1}")
        fi
      done
    done
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done
