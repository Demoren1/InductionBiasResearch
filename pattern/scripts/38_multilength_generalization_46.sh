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

TRAIN_ROOT="${TRAIN_ROOT:-pattern/outputs/bilevel_mask/multilength_sharing_357_converged_20260915}"
OUT_ROOT="${OUT_ROOT:-pattern/outputs/bilevel_mask/multilength_generalization_46_20260915}"
GPUS="${GPUS:-7 7 0 1 2 4 5 6}"
CPUS="${CPUS:-198 198 192 193 194 195 196 197}"
read -r -a GPU_ARRAY <<<"$GPUS"
read -r -a CPU_ARRAY <<<"$CPUS"
if (( ${#GPU_ARRAY[@]} != 8 || ${#CPU_ARRAY[@]} != 8 )); then
  echo "Expected eight GPU slots and eight CPU entries" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT"

variants=(global global global global length_latent length_latent length_latent length_latent)
seeds=(42 43 44 45 42 43 44 45)
pids=()
for job in 0 1 2 3 4 5 6 7; do
  variant="${variants[$job]}"
  seed="${seeds[$job]}"
  suffix=""
  if [[ "$seed" == 45 ]]; then suffix="_extended"; fi
  checkpoint="$TRAIN_ROOT/${variant}_seed${seed}${suffix}/training.pt"
  name="${variant}_seed${seed}"
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$job]}" taskset --cpu-list "${CPU_ARRAY[$job]}" \
    python -m pattern.bilevel_mask.evaluate_multilength_generalization \
      --checkpoint "$checkpoint" --output "$OUT_ROOT/$name.json" --device cuda \
      --unseen-lengths 4,6 --latent-steps 500 \
      >"$OUT_ROOT/$name.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
exit "$failed"
