#!/usr/bin/env bash
GPU_IDS="${GPU_IDS:-0 1 2 4 5}"  # Например: "1 2 3".

# Build the initial functional banks, then search with own-task quality,
# aligned hard-mask agreement and aligned distillation to the real-mask archive.
# EVALUATOR_EPOCHS trains once from bank-origin cross-fits; evaluator stays frozen.
# The search refresh cadence adds measurements and feedback, not evaluator updates.
# Keep the best common mask across refreshes, using independent selection queries.
# Bootstrap trains the evaluator only by default; BOOTSTRAP_GENERATORS=1 opts into generator exploration.
# WARM_START_FROM=/path/to/completed/run skips bank/evaluator preparation.
# TRAIN_PATTERNS and TEST_PATTERNS select the same roles for both phases.
# The defaults use 12 training patterns and four held-out patterns.
# GE_OUT, GE_SEED, CHILD_STEPS, GENERATOR_EPOCHS, UPDATES_PER_EPOCH,
# AGREEMENT_WEIGHT and ELITE_DISTILLATION_WEIGHT override defaults.
# Trailing arguments apply to the final search.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

GE_OUT="${GE_OUT:-outputs/generator_evaluator/${GE_STAMP}_pattern_seed${GE_SEED}}"
GE_CHILD_STEPS="${CHILD_STEPS:-1000}"
GE_SOURCE="${WARM_START_FROM:-}"
read -r -a GE_TRAIN_PATTERNS <<< "${TRAIN_PATTERNS:-0000 0001 0011 0101 0110 0111 1000 1001 1010 1100 1110 1111}"
read -r -a GE_TEST_PATTERNS <<< "${TEST_PATTERNS:-0010 0100 1011 1101}"
GE_PATTERN_ROLES=(--train-patterns "${GE_TRAIN_PATTERNS[@]}")
if (( ${#GE_TEST_PATTERNS[@]} == 1 )); then
  GE_PATTERN_ROLES+=(--test-pattern "${GE_TEST_PATTERNS[0]}")
else
  GE_PATTERN_ROLES+=(--test-patterns "${GE_TEST_PATTERNS[@]}")
fi
GE_DATA_ARGS=(
  --support-count "${SUPPORT_COUNT:-192}" --query-count "${QUERY_COUNT:-256}"
  --selection-count "${SELECTION_COUNT:-128}" --probe-count "${PROBE_COUNT:-128}"
  --lr "${CHILD_LR:-0.03}" --l2 "${CHILD_L2:-0.003}"
  --training-mode joint
  --quality-objective "${QUALITY_OBJECTIVE:-worst}"
  --agreement-weight "${AGREEMENT_WEIGHT:-0.1}"
  --elite-distillation-weight "${ELITE_DISTILLATION_WEIGHT:-0.1}"
  --generator-pretrain-epochs "${GENERATOR_PRETRAIN_EPOCHS:-0}"
  --width "${GENERATOR_WIDTH:-32}" --heads "${GENERATOR_HEADS:-4}"
  --layers 1 --noise-dim 4
)
GE_DEVICE_ARGS=(
  --device cuda:0 --generator-devices auto --measurement-devices auto
  --measurement-batch-size "${CHILD_MASK_BATCH:-128}"
)
GE_BANK_ARGS=(
  --bank-candidates "${BANK_CANDIDATES:-1000}" --teachers "${TEACHERS:-100}"
  --bank-capacity "${BANK_CAPACITY:-100}" --bank-steps "${BANK_STEPS:-1000}"
  --teacher-batch-size "${CHILD_MASK_BATCH:-128}"
)
GE_BOOTSTRAP_ARGS=()
if [[ "${BOOTSTRAP_GENERATORS:-0}" == "1" ]]; then
  GE_BOOTSTRAP_ARGS+=(--bootstrap-generators)
fi

if [[ -z "$GE_SOURCE" ]]; then
  GE_SOURCE="$GE_OUT/bootstrap"
  ge_run python -u -m generator_evaluator.cooperative_run \
    --preset pattern-small --bootstrap-only \
    "${GE_BOOTSTRAP_ARGS[@]}" "${GE_PATTERN_ROLES[@]}" "${GE_DATA_ARGS[@]}" \
    "${GE_BANK_ARGS[@]}" \
    --steps "$GE_CHILD_STEPS" --replicas "${REPLICAS:-4}" \
    --generator-epochs "${BOOTSTRAP_EPOCHS:-20}" \
    --updates-per-epoch "${BOOTSTRAP_UPDATES:-10}" --refresh-every 2 \
    --evaluator-epochs "${EVALUATOR_EPOCHS:-30}" --acquisition-budget 6 --candidates 24 \
    --initial-random "${INITIAL_RANDOM:-8}" \
    --seed "$GE_SEED" "${GE_DEVICE_ARGS[@]}" --progress \
    --out "$GE_SOURCE"
fi

ge_run python -u -m generator_evaluator.cooperative_run \
  --preset pattern-small "${GE_BOOTSTRAP_ARGS[@]}" "${GE_PATTERN_ROLES[@]}" "${GE_DATA_ARGS[@]}" \
  --warm-start-from "$GE_SOURCE" \
  --elite-limit "${ELITE_LIMIT:-8}" \
  --steps "$GE_CHILD_STEPS" --replicas "${REPLICAS:-4}" "${GE_BANK_ARGS[@]}" \
  --generator-epochs "${GENERATOR_EPOCHS:-20}" \
  --updates-per-epoch "${UPDATES_PER_EPOCH:-10}" --refresh-every 2 \
  --evaluator-epochs "${EVALUATOR_EPOCHS:-30}" --acquisition-budget 6 --candidates 24 \
  --seed "$GE_SEED" "${GE_DEVICE_ARGS[@]}" --progress \
  --out "$GE_OUT/search" "$@"

printf '\nMasks and heatmaps: %s/search/figures/\n' "$GE_OUT"
