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

OUT_ROOT="${OUT_ROOT:-yeh2022_generated_sharing/outputs/paper_suite_20260915}"
GPUS="${GPUS:-0 1 2 4 5 6 7}"
CPUS="${CPUS:-192 193 194 196 197 198 199}"
SEEDS="${SEEDS:-0 1 2 3 4}"
SUITES="${SUITES:-gaussian sum crosscorr denoising}"
read -r -a GPU_ARRAY <<<"$GPUS"
read -r -a CPU_ARRAY <<<"$CPUS"
read -r -a SEED_ARRAY <<<"$SEEDS"
if (( ${#GPU_ARRAY[@]} == 0 || ${#CPU_ARRAY[@]} < ${#GPU_ARRAY[@]} )); then
  echo "Need at least one CPU core per listed GPU" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT/logs"

contains_suite() {
  [[ " $SUITES " == *" $1 "* ]]
}

pids=()
names=()
job=0
failed=0
launch() {
  local name="$1"
  shift
  local slot=$((job % ${#GPU_ARRAY[@]}))
  if [[ -n "${pids[$slot]:-}" ]]; then
    if ! wait "${pids[$slot]}"; then
      echo "FAILED: ${names[$slot]}" >&2
      failed=1
    fi
  fi
  local gpu="${GPU_ARRAY[$slot]}"
  local cpu="${CPU_ARRAY[$slot]}"
  CUDA_VISIBLE_DEVICES="$gpu" taskset --cpu-list "$cpu" "$@" \
    >"$OUT_ROOT/logs/$name.log" 2>&1 &
  pids[$slot]="$!"
  names[$slot]="$name"
  job=$((job + 1))
}

if contains_suite gaussian; then
  for dimensions in 2 3 4 5 6; do
    launch "gaussian_fig2_k${dimensions}" \
      python -m yeh2022_generated_sharing.run_gaussian --device cuda \
      --dimensions "$dimensions" --true-rank 1 \
      --output-dir "$OUT_ROOT/gaussian/fig2_k${dimensions}"
  done
  for train_size in 10 20 30 40 50 60 70 80 90; do
    launch "gaussian_fig3_t${train_size}" \
      python -m yeh2022_generated_sharing.run_gaussian --device cuda \
      --dimensions 5 --true-rank 1 --num-train "$train_size" \
      --output-dir "$OUT_ROOT/gaussian/fig3_t${train_size}"
  done
  for rank in 1 2 3 4 5; do
    launch "gaussian_fig4_r${rank}" \
      python -m yeh2022_generated_sharing.run_gaussian --device cuda \
      --dimensions 5 --true-rank "$rank" \
      --output-dir "$OUT_ROOT/gaussian/fig4_r${rank}"
  done
  for dimensions in 10 30 50 70 90; do
    launch "gaussian_fig5_k${dimensions}" \
      python -m yeh2022_generated_sharing.run_gaussian --device cuda \
      --dimensions "$dimensions" --true-rank 1 \
      --output-dir "$OUT_ROOT/gaussian/fig5_k${dimensions}"
  done
fi

if contains_suite sum; then
  for seed in "${SEED_ARRAY[@]}"; do
    for length in 2 4 6 8 10; do
      launch "sum_k${length}_seed${seed}" \
        python -m yeh2022_generated_sharing.run_sum_numbers --device cuda \
        --target both --seed "$seed" --sequence-length "$length" \
        --output-dir "$OUT_ROOT/sum/k${length}_seed${seed}"
    done
  done
fi

if contains_suite crosscorr; then
  # These valid-correlation shapes reproduce the assignment item counts
  # 6, 15, 35, and 80 shown in Fig. 8.  The omitted author code prevents a
  # stronger claim about their unpublished input/kernel convention.
  for seed in "${SEED_ARRAY[@]}"; do
    for dimensions in "3 2" "5 3" "7 3" "10 3"; do
      read -r input_length kernel_length <<<"$dimensions"
      item_count=$(((input_length - kernel_length + 1) * input_length))
      launch "crosscorr_a${item_count}_seed${seed}" \
        python -m yeh2022_generated_sharing.run_linear --device cuda \
        --benchmark cross_correlation --seed "$seed" \
        --input-length "$input_length" --kernel-length "$kernel_length" \
        --output "$OUT_ROOT/crosscorr/a${item_count}_seed${seed}"
    done
  done
fi

if contains_suite denoising; then
  # Figure A1 reports noise variances 5, 10, 15, and 20.
  for seed in "${SEED_ARRAY[@]}"; do
    for variance_and_std in "5 2.2360679775" "10 3.1622776602" "15 3.8729833462" "20 4.4721359550"; do
      read -r variance noise_std <<<"$variance_and_std"
      launch "denoise_var${variance}_seed${seed}" \
        python -m yeh2022_generated_sharing.run_linear --device cuda \
        --benchmark denoising --seed "$seed" --signal-length 8 \
        --noise-std "$noise_std" \
        --output "$OUT_ROOT/denoising/var${variance}_seed${seed}"
    done
  done
fi

for slot in "${!pids[@]}"; do
  if ! wait "${pids[$slot]}"; then
    echo "FAILED: ${names[$slot]}" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit "$failed"
fi

python -m yeh2022_generated_sharing.report \
  --results-root "$OUT_ROOT" --output "$OUT_ROOT/FINAL_REPORT.md"
echo "Final report: $OUT_ROOT/FINAL_REPORT.md"
