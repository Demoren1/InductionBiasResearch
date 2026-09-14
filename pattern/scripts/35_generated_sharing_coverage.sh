#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

OUT_ROOT="${OUT_ROOT:-pattern/outputs/bilevel_mask/generated_sharing_coverage_20260915}"
MAX_JOBS="${MAX_JOBS:-16}"
mkdir -p "$OUT_ROOT"

pids=()
job_index=0
for seed in 42 43 44; do
  for coverage in train10 train12 all16; do
    out="$OUT_ROOT/${coverage}_seed${seed}"
    gpu="$((job_index % 8))"
    CUDA_VISIBLE_DEVICES="$gpu" python -m pattern.bilevel_mask.tune_generated_sharing \
      --output "$out" --device cuda --seed "$seed" --pattern-coverage "$coverage" \
      --inner-lr 0.1 --inner-steps 20 --generator-lr 0.003 \
      --outer-steps 100 --train-restarts 16 --eval-restarts 32 \
      --latent-search-steps 80 --latent-patience 20 --final-refit-steps 1000 \
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
