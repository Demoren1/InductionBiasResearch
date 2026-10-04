#!/usr/bin/env bash
GPU_IDS="${GPU_IDS:-0 1 2 4 5}"  # Например: "1 2 3".

# The search invocation prepares functional banks and the initial evaluator,
# then optimizes own-task quality, agreement and distillation to measured masks.
# EVALUATOR_EPOCHS trains the initial evaluator from bank-origin cross-fits; it stays frozen.
# Search refreshes add measurements and feedback, not evaluator updates.
# Keep the best common mask across refreshes, using independent selection queries.
# BOOTSTRAP_GENERATORS=1 is retained for CLI compatibility.
# WARM_START_FROM accepts a full legacy run; evaluator-only compact output lacks banks.
# TRAIN_PATTERNS and TEST_PATTERNS set the search run's task roles.
# The defaults use 12 training patterns and four held-out patterns.
# GE_OUT, GE_SEED, CHILD_STEPS, GENERATOR_EPOCHS, UPDATES_PER_EPOCH,
# AGREEMENT_WEIGHT and ELITE_DISTILLATION_WEIGHT override defaults.
# Trailing arguments apply to the final search.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

GE_OUT="${GE_OUT:-outputs/generator_evaluator/${GE_STAMP}_pattern_seed${GE_SEED}}"
GE_CHILD_STEPS="${CHILD_STEPS:-1000}"
GE_WARM_START_ARGS=()
if [[ -n "${WARM_START_FROM:-}" ]]; then
  GE_WARM_START_ARGS+=(--warm-start-from "$WARM_START_FROM")
fi
read -r -a GE_TRAIN_PATTERNS <<< "${TRAIN_PATTERNS:-0000 0001 0011 0101 0110 0111 1000 1001 1010 1100 1110 1111}"
read -r -a GE_TEST_PATTERNS <<< "${TEST_PATTERNS:-0010 0100 1011 1101}"
GE_PATTERN_ROLES=(--train-patterns "${GE_TRAIN_PATTERNS[@]}")
if (( ${#GE_TEST_PATTERNS[@]} == 1 )); then
  GE_PATTERN_ROLES+=(--test-pattern "${GE_TEST_PATTERNS[0]}")
else
  GE_PATTERN_ROLES+=(--test-patterns "${GE_TEST_PATTERNS[@]}")
fi
GE_DATA_ARGS=(
  --support-count "${SUPPORT_COUNT:-208}" --query-count "${QUERY_COUNT:-256}"
  --selection-count "${SELECTION_COUNT:-128}" --probe-count "${PROBE_COUNT:-128}"
  --lr "${CHILD_LR:-0.03}" --l2 "${CHILD_L2:-0.003}"
  --training-mode joint
  --quality-objective "${QUALITY_OBJECTIVE:-average}"
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
if [[ "${BOOTSTRAP_GENERATORS:-0}" == "1" ]]; then
  GE_BOOTSTRAP_ARGS=(--bootstrap-generators)
else
  GE_BOOTSTRAP_ARGS=()
fi

ge_run python -u -m generator_evaluator.cooperative_run \
  --preset pattern-small "${GE_BOOTSTRAP_ARGS[@]}" "${GE_PATTERN_ROLES[@]}" "${GE_DATA_ARGS[@]}" \
  "${GE_WARM_START_ARGS[@]}" \
  --elite-limit "${ELITE_LIMIT:-8}" \
  --steps "$GE_CHILD_STEPS" --replicas "${REPLICAS:-4}" "${GE_BANK_ARGS[@]}" \
  --generator-epochs "${GENERATOR_EPOCHS:-20}" \
  --updates-per-epoch "${UPDATES_PER_EPOCH:-10}" --refresh-every 2 \
  --evaluator-epochs "${EVALUATOR_EPOCHS:-30}" --acquisition-budget 6 --candidates 24 \
  --seed "$GE_SEED" "${GE_DEVICE_ARGS[@]}" --progress \
  --out "$GE_OUT/search" "$@"

printf '\nMasks and heatmaps: %s/search/figures/\n' "$GE_OUT"

printf "Final report: %s/search/final_report.md\n" "$GE_OUT"
