#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

OUT_ROOT="${OUT_ROOT:-pattern/outputs/bilevel_mask/length32_pattern4_tuning_confirm_20260915}"
MAX_JOBS="${MAX_JOBS:-16}"
mkdir -p "$OUT_ROOT"

# name latent width generator_lr radius outer_steps binary_penalty
configs=(
  "a_bin0      8 64 0.001 4  60 0"
  "a_bin0001   8 64 0.001 4  60 0.001"
  "a_bin001    8 64 0.001 4  60 0.01"
  "a_bin005    8 64 0.001 4  60 0.05"
  "a_bin01     8 64 0.001 4  60 0.1"
  "b_r12       8 64 0.003 12 60 0.01"
  "c_d16      16 64 0.003 4  60 0.01"
  "d_compact   4 16 0.003 4  60 0.01"
)

pids=()
job_index=0
for seed in 42 43 44; do
  for spec in "${configs[@]}"; do
    read -r name latent width generator_lr radius outer_steps binary_penalty <<<"$spec"
    out="$OUT_ROOT/${name}_seed${seed}"
    if [[ -f "$out/result.json" ]]; then
      continue
    fi
    gpu="$((job_index % 8))"
    CUDA_VISIBLE_DEVICES="$gpu" python -m pattern.bilevel_mask.tune_length32 \
      --mode outer --device cuda --output "$out" --seed "$seed" \
      --latent-dim "$latent" --generator-width "$width" \
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
for pid in "${pids[@]}"; do wait "$pid"; done
