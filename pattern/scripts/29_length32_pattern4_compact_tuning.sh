#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

OUT_ROOT="${OUT_ROOT:-pattern/outputs/bilevel_mask/length32_pattern4_compact_tuning_20260915}"
MAX_JOBS="${MAX_JOBS:-16}"
mkdir -p "$OUT_ROOT/grid" "$OUT_ROOT/architecture"

launch() {
  local out="$1" seed="$2" latent="$3" width="$4" generator_lr="$5"
  local radius="$6" binary_penalty="$7" gpu="$8"
  if [[ -f "$out/result.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python -m pattern.bilevel_mask.tune_length32 \
    --mode outer --device cuda --output "$out" --seed "$seed" \
    --latent-dim "$latent" --generator-width "$width" \
    --weight-steps-per-z 5 --z-lr 0.1 --temperature-end 0.05 \
    --generator-lr "$generator_lr" --z-radius "$radius" \
    --outer-steps 60 --binary-penalty "$binary_penalty" \
    --eval-weight-steps 1000 >"$out.log" 2>&1 &
  pids+=("$!")
  job_index="$((job_index + 1))"
  if (( ${#pids[@]} >= MAX_JOBS )); then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
}

pids=()
job_index=0
for seed in 42 43 44; do
  for generator_lr in 0.001 0.003 0.01; do
    for radius in 2 4 8 12; do
      for binary_penalty in 0 0.01; do
        tag="g${generator_lr}_r${radius}_b${binary_penalty}_seed${seed}"
        launch "$OUT_ROOT/grid/$tag" "$seed" 4 16 "$generator_lr" "$radius" \
          "$binary_penalty" "$((job_index % 8))"
      done
    done
  done

  # Neighboring small architectures under the current compact-model settings.
  launch "$OUT_ROOT/architecture/d2_w32_seed${seed}" "$seed" 2 32 0.003 4 0.01 \
    "$((job_index % 8))"
  launch "$OUT_ROOT/architecture/d8_w16_seed${seed}" "$seed" 8 16 0.003 4 0.01 \
    "$((job_index % 8))"
done
for pid in "${pids[@]}"; do wait "$pid"; done
