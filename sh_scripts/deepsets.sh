#!/usr/bin/env bash
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"  # Например: "1 2 3".

# The search invocation prepares functional banks and the initial evaluator,
# then jointly optimizes own-task quality, agreement and archive distillation.
# EVALUATOR_EPOCHS trains the initial evaluator from bank-origin cross-fits; it stays frozen.
# REFRESH_EVERY schedules measurement rounds and feedback, not evaluator updates.
# Keep the best common mask across refreshes, using independent selection queries.
# BOOTSTRAP_GENERATORS=1 is retained for CLI compatibility.
# RESTART_FROM reuses saved binary banks and evaluator with fresh generators.
# WARM_START_FROM accepts a full legacy checkpoint.
# GPU_IDS selects cards for task generators, bank creation, and batched child fits.
# TRAIN_TASKS and TEST_TASKS set the search run's task counts.
# FIXED_TEST_FROM keeps prior sealed test costs and pools when expanding train.
# Budget overrides: BANK_CANDIDATES, TEACHERS, BANK_STEPS, CHILD_STEPS,
# GENERATOR_EPOCHS, UPDATES_PER_EPOCH and CHILD_MASK_BATCH.
# Trailing CLI arguments apply to search.
# ELITE_LIMIT bounds the archive of measured real masks used for distillation.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

GE_OUT="${GE_OUT:-outputs/generator_evaluator/${GE_STAMP}_deepsets_seed${GE_SEED}}"
GE_DATA="${DATA_ROOT:-datasets/mnist8m}"
read -r -a GE_EVALUATOR_DEVICES <<< "${EVALUATOR_DEVICES:-auto}"
GE_WARM_START_ARGS=()
if [[ -n "${WARM_START_FROM:-}" ]]; then
  GE_WARM_START_ARGS+=(--warm-start-from "$WARM_START_FROM")
fi
if [[ -n "${RESTART_FROM:-}" ]]; then
  GE_WARM_START_ARGS+=(--restart-from "$RESTART_FROM")
fi
GE_COMMON_ARGS=(
  --domain deepsets --preset deepsets --training-mode joint --data-root "$GE_DATA"
  --train-task-count "${TRAIN_TASKS:-6}" --test-task-count "${TEST_TASKS:-4}"
  --bank-candidates "${BANK_CANDIDATES:-4096}" --teachers "${TEACHERS:-1024}"
  --bank-steps "${BANK_STEPS:-4000}" --teacher-batch-size "${CHILD_MASK_BATCH:-64}"
  --bank-capacity "${BANK_CAPACITY:-${TEACHERS:-1024}}"
  --probe-count "${PROBE_COUNT:-32}"
  --steps "${CHILD_STEPS:-2000}" --replicas "${REPLICAS:-4}"
  --lr "${CHILD_LR:-0.005}" --l2 "${CHILD_L2:-0.0001}"
  --width "${TRANSFORMER_WIDTH:-64}" --heads "${TRANSFORMER_HEADS:-4}" --layers "${TRANSFORMER_LAYERS:-2}" --noise-dim 8 --ensemble-members 2
  --refresh-every "${REFRESH_EVERY:-2}" --minimum-refresh-every "${MINIMUM_REFRESH_EVERY:-2}"
  --evaluator-epochs "${EVALUATOR_EPOCHS:-100}"
  --evaluator-batch-size "${EVALUATOR_BATCH_SIZE:-64}" --evaluator-lr "${EVALUATOR_LR:-0.0003}"
  --evaluator-devices "${GE_EVALUATOR_DEVICES[@]}"
  --acquisition-budget 6 --candidates 24 --initial-random 8
  --auxiliary-budget "${AUXILIARY_BUDGET:-0}" --feedback-masks 2
  --agreement-weight "${AGREEMENT_WEIGHT:-0.1}"
  --elite-distillation-weight "${ELITE_DISTILLATION_WEIGHT:-0.1}"
  --generator-pretrain-epochs "${GENERATOR_PRETRAIN_EPOCHS:-0}"
  --quality-objective "${QUALITY_OBJECTIVE:-average}"
  --seed "$GE_SEED" --device cuda:0 --generator-devices auto --measurement-devices auto
  --measurement-batch-size "${CHILD_MASK_BATCH:-64}" --progress
)
if [[ -n "${QUERY_COUNT:-}" ]]; then
  GE_COMMON_ARGS+=(--query-count "$QUERY_COUNT")
fi
if [[ -n "${SELECTION_COUNT:-}" ]]; then
  GE_COMMON_ARGS+=(--selection-count "$SELECTION_COUNT")
fi
if [[ -n "${SUPPORT_COUNT:-}" ]]; then
  GE_COMMON_ARGS+=(--support-count "$SUPPORT_COUNT")
fi
if [[ -n "${BANK_SUPPORT_COUNT:-}" ]]; then
  GE_COMMON_ARGS+=(--bank-support-count "$BANK_SUPPORT_COUNT")
fi
if [[ -n "${BANK_QUERY_COUNT:-}" ]]; then
  GE_COMMON_ARGS+=(--bank-query-count "$BANK_QUERY_COUNT")
fi
if [[ "${BOOTSTRAP_GENERATORS:-0}" == "1" ]]; then
  GE_COMMON_ARGS+=(--bootstrap-generators)
fi
if [[ -n "${FIXED_TEST_FROM:-}" ]]; then
  GE_COMMON_ARGS+=(--fixed-test-from "$FIXED_TEST_FROM")
fi

ge_run python -u -m generator_evaluator.cooperative_run "${GE_COMMON_ARGS[@]}" \
  "${GE_WARM_START_ARGS[@]}" \
  --generator-epochs "${GENERATOR_EPOCHS:-20}" \
  --updates-per-epoch "${UPDATES_PER_EPOCH:-10}" \
  --elite-limit "${ELITE_LIMIT:-8}" \
  --out "$GE_OUT/search" "$@"

printf '\nResults: %s/search/summary.json\nMasks and heatmaps: %s/search/figures/\n' "$GE_OUT" "$GE_OUT"

printf "Final report: %s/search/final_report.md\n" "$GE_OUT"
