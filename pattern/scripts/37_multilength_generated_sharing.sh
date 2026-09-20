#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

OUT_ROOT="${OUT_ROOT:-pattern/outputs/bilevel_mask/multilength_sharing_357_20260915}"
SEEDS="${SEEDS:-42 43 44 45}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
CPUS="${CPUS:-192 193 194 195 196 197 198 199}"
read -r -a SEED_ARRAY <<<"$SEEDS"
read -r -a GPU_ARRAY <<<"$GPUS"
read -r -a CPU_ARRAY <<<"$CPUS"
if (( ${#SEED_ARRAY[@]} != 4 || ${#GPU_ARRAY[@]} != 8 || ${#CPU_ARRAY[@]} != 8 )); then
  echo "Expected four seeds, eight GPUs, and eight CPU cores" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT"

pids=()
job=0
for variant in global length_latent; do
  for seed in "${SEED_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$job]}"
    cpu="${CPU_ARRAY[$job]}"
    output="$OUT_ROOT/${variant}_seed${seed}"
    CUDA_VISIBLE_DEVICES="$gpu" taskset --cpu-list "$cpu" \
      python -m pattern.bilevel_mask.run_multilength_sharing \
      --output "$output" --device cuda --variant "$variant" --seed "$seed" \
      >"$OUT_ROOT/${variant}_seed${seed}.log" 2>&1 &
    pids+=("$!")
    job="$((job + 1))"
  done
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
exit "$failed"
