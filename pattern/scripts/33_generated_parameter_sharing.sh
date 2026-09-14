#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

OUT_ROOT="${OUT_ROOT:-pattern/outputs/bilevel_mask/generated_sharing_20260915}"
SEEDS="${SEEDS:-42 43}"
PATTERN_COVERAGE="${PATTERN_COVERAGE:-all16}"
MAX_JOBS="${MAX_JOBS:-16}"
read -r -a SEED_ARRAY <<<"$SEEDS"
mkdir -p "$OUT_ROOT"

pids=()
job_index=0
for seed in "${SEED_ARRAY[@]}"; do
  gpu="$((job_index % 8))"
  CUDA_VISIBLE_DEVICES="$gpu" python -m pattern.bilevel_mask.run_generated_sharing \
    --output "$OUT_ROOT/seed${seed}" --device cuda --seed "$seed" \
    --pattern-coverage "$PATTERN_COVERAGE" \
    >"$OUT_ROOT/seed${seed}.log" 2>&1 &
  pids+=("$!")
  job_index="$((job_index + 1))"
  if (( ${#pids[@]} >= MAX_JOBS )); then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done
