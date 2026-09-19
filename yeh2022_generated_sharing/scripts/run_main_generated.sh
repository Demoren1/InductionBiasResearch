#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras
cd "$ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

OUT_ROOT="${OUT_ROOT:-yeh2022_generated_sharing/outputs/main_generated_20260915}"
GPUS="${GPUS:-0 1 2 4 5 6 7}"
CPUS="${CPUS:-192 193 194 196 197 198 199}"
read -r -a GPU_ARRAY <<<"$GPUS"
read -r -a CPU_ARRAY <<<"$CPUS"
mkdir -p "$OUT_ROOT/logs"

pids=()
names=()
job=0
launch() {
  local name="$1"
  shift
  local slot=$((job % ${#GPU_ARRAY[@]}))
  if [[ -n "${pids[$slot]:-}" ]]; then
    wait "${pids[$slot]}"
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$slot]}" \
    taskset --cpu-list "${CPU_ARRAY[$slot]}" \
    "$@" >"$OUT_ROOT/logs/$name.log" 2>&1 &
  pids[$slot]="$!"
  names[$slot]="$name"
  job=$((job + 1))
}

for dimensions in 2 4 6; do
  launch "gaussian_k${dimensions}" \
    python -m yeh2022_generated_sharing.run_gaussian --device cuda \
    --dimensions "$dimensions" --true-rank 1 --optimizer rmsprop \
    --lower-solver release_normalized \
    --output-dir "$OUT_ROOT/gaussian/k${dimensions}"
done

for generator_seed in 0 1 2 3; do
  for latent_mode in per_task global; do
    launch "cross_a6_${latent_mode}_${generator_seed}" \
      python -m yeh2022_generated_sharing.run_multitask_linear --device cuda \
      --benchmark cross_correlation --input-length 3 --kernel-length 2 \
      --latent-mode "$latent_mode" --assignment-mode ste \
      --generator-seed "$generator_seed" --outer-lr 0.0003 --ridge 0.01 \
      --entropy-weight 0 --nuclear-weight 0 --lower-solver exact_constrained \
      --output "$OUT_ROOT/cross_a6/${latent_mode}_seed${generator_seed}"
    launch "cross_a15_${latent_mode}_${generator_seed}" \
      python -m yeh2022_generated_sharing.run_multitask_linear --device cuda \
      --benchmark cross_correlation --input-length 5 --kernel-length 3 \
      --latent-mode "$latent_mode" --assignment-mode ste \
      --generator-seed "$generator_seed" --outer-lr 0.0003 --ridge 0.01 \
      --entropy-weight 0 --nuclear-weight 0 --lower-solver exact_constrained \
      --output "$OUT_ROOT/cross_a15/${latent_mode}_seed${generator_seed}"
    launch "denoise_${latent_mode}_${generator_seed}" \
      python -m yeh2022_generated_sharing.run_multitask_linear --device cuda \
      --benchmark denoising --signal-length 8 --noise-std 3.1622776602 \
      --latent-mode "$latent_mode" --assignment-mode ste \
      --generator-seed "$generator_seed" --outer-lr 0.01 --ridge 0.1 \
      --entropy-weight 0 --nuclear-weight 0 --lower-solver exact_constrained \
      --output "$OUT_ROOT/denoising/${latent_mode}_seed${generator_seed}"
  done
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

python -m yeh2022_generated_sharing.report \
  --results-root "$OUT_ROOT" --output "$OUT_ROOT/FINAL_REPORT.md"
