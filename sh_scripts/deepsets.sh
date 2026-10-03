#!/usr/bin/env bash
GPU_IDS="${GPU_IDS:-0 1 2 4 5}"  # Например: "1 2 3".

# Reconstruct masks, then interleave own-task quality, agreement, reconstruction,
# measured-target feedback, and critic refresh in one joint search loop.
# Keep the best common mask across refreshes, using independent selection queries.
# WARM_START_FROM=/path/to/cooperative/bootstrap skips bank/critic preparation.
# GPU_IDS selects cards for task generators, bank creation, and batched child fits.
# TRAIN_TASKS and TEST_TASKS set task counts for both bootstrap and search.
# FIXED_TEST_FROM keeps prior sealed test costs and pools when expanding train.
# Budget overrides: BANK_CANDIDATES, TEACHERS, BANK_STEPS, CHILD_STEPS,
# BOOTSTRAP_EPOCHS, GENERATOR_EPOCHS, UPDATES_PER_EPOCH and CHILD_MASK_BATCH.
# Trailing CLI arguments apply to search; solver changes must match bootstrap.
# ELITE_LIMIT bounds the archive of measured targets sampled with random latents.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

GE_OUT="${GE_OUT:-outputs/generator_evaluator/${GE_STAMP}_deepsets_seed${GE_SEED}}"
GE_DATA="${DATA_ROOT:-datasets/mnist8m}"
GE_SOURCE="${WARM_START_FROM:-}"
GE_COMMON_ARGS=(
  --domain deepsets --preset deepsets --training-mode joint --data-root "$GE_DATA"
  --train-task-count "${TRAIN_TASKS:-2}" --test-task-count "${TEST_TASKS:-2}"
  --bank-candidates "${BANK_CANDIDATES:-1000}" --teachers "${TEACHERS:-100}"
  --bank-steps "${BANK_STEPS:-200}" --teacher-batch-size "${CHILD_MASK_BATCH:-8}"
  --bank-capacity "${BANK_CAPACITY:-100}"
  --support-count "${SUPPORT_COUNT:-205}" --query-count "${QUERY_COUNT:-51}"
  --selection-count "${SELECTION_COUNT:-51}" --probe-count "${PROBE_COUNT:-32}"
  --steps "${CHILD_STEPS:-1000}" --replicas "${REPLICAS:-2}"
  --lr "${CHILD_LR:-0.005}" --l2 "${CHILD_L2:-0.0001}"
  --width 32 --heads 4 --layers 1 --noise-dim 8 --ensemble-members 2
  --refresh-every "${REFRESH_EVERY:-2}" --minimum-refresh-every "${MINIMUM_REFRESH_EVERY:-2}"
  --evaluator-epochs "${EVALUATOR_EPOCHS:-30}"
  --evaluator-batch-size "${EVALUATOR_BATCH_SIZE:-32}" --acquisition-budget 6 --candidates 24 --initial-random 8
  --auxiliary-budget "${AUXILIARY_BUDGET:-0}" --feedback-masks 2 --agreement-weight "${AGREEMENT_WEIGHT:-0.1}"
  --quality-objective "${QUALITY_OBJECTIVE:-worst}"
  --generator-pretrain-epochs "${GENERATOR_PRETRAIN_EPOCHS:-5}"
  --pretrain-updates-per-epoch "${PRETRAIN_UPDATES:-20}"
  --reconstruction-batch-size "${RECONSTRUCTION_BATCH_SIZE:-8}"
  --reconstruction-weight "${RECONSTRUCTION_WEIGHT:-0.1}"
  --elite-distillation-weight "${ELITE_DISTILLATION_WEIGHT:-0.1}"
  --agreement-ramp-epochs "${AGREEMENT_RAMP_EPOCHS:-5}"
  --seed "$GE_SEED" --device cuda:0 --generator-devices auto --measurement-devices auto
  --measurement-batch-size "${CHILD_MASK_BATCH:-8}" --progress
)
if [[ -n "${FIXED_TEST_FROM:-}" ]]; then
  GE_COMMON_ARGS+=(--fixed-test-from "$FIXED_TEST_FROM")
fi

if [[ -z "$GE_SOURCE" ]]; then
  GE_SOURCE="$GE_OUT/bootstrap"
  ge_run python -u -m generator_evaluator.cooperative_run "${GE_COMMON_ARGS[@]}" \
    --bootstrap-only --generator-pretrain-epochs 0 --generator-epochs "${BOOTSTRAP_EPOCHS:-10}" \
    --updates-per-epoch "${BOOTSTRAP_UPDATES:-10}" --out "$GE_SOURCE"
fi

ge_run python -u -m generator_evaluator.cooperative_run "${GE_COMMON_ARGS[@]}" \
  --warm-start-from "$GE_SOURCE" \
  --generator-epochs "${GENERATOR_EPOCHS:-20}" \
  --updates-per-epoch "${UPDATES_PER_EPOCH:-10}" \
  --elite-limit "${ELITE_LIMIT:-8}" \
  --out "$GE_OUT/search" "$@"

printf '\nResults: %s/search/summary.json\nMasks and heatmaps: %s/search/figures/\n' "$GE_OUT" "$GE_OUT"
