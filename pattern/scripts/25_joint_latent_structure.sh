#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-pattern/outputs/bilevel_mask/latent_joint_20260915}"
SEEDS="${SEEDS:-42}"
RESTARTS="${RESTARTS:-64}"

mkdir -p "$OUT_ROOT"
read -r -a SEED_ARRAY <<< "$SEEDS"
pids=()
for index in "${!SEED_ARRAY[@]}"; do
  seed="${SEED_ARRAY[$index]}"
  gpu="$((index % 8))"
  output="$OUT_ROOT/soft_seed${seed}"
  log="$OUT_ROOT/soft_seed${seed}.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m pattern.bilevel_mask.run_latent_joint \
    --output "$output" --device cuda --seed "$seed" --restarts "$RESTARTS" \
    >"$log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
