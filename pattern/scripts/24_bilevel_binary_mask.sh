#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-pattern/outputs/bilevel_mask/direct_20260914}"
OUTER_STEPS="${OUTER_STEPS:-400}"
INNER_STEPS="${INNER_STEPS:-50}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49}"

mkdir -p "$OUT_ROOT"
read -r -a SEED_ARRAY <<< "$SEEDS"
pids=()
for index in "${!SEED_ARRAY[@]}"; do
  seed="${SEED_ARRAY[$index]}"
  gpu="$((index % 8))"
  for relaxation in soft ste; do
    output="$OUT_ROOT/${relaxation}_seed${seed}"
    log="$OUT_ROOT/${relaxation}_seed${seed}.log"
    CUDA_VISIBLE_DEVICES="$gpu" python -m pattern.bilevel_mask.run \
      --output "$output" --device cuda --seed "$seed" --relaxation "$relaxation" \
      --outer-steps "$OUTER_STEPS" --inner-steps "$INNER_STEPS" --eval-steps "$EVAL_STEPS" \
      >"$log" 2>&1 &
    pids+=("$!")
  done
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
