#!/usr/bin/env bash
# Full scalar-conditioned gap-OOD run with a leakage-free gap-held-out split.
# Examples:
#   bash scripts/10_gap_ood.sh
#   GAP_OOD_PROFILE=extrapolation bash scripts/10_gap_ood.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SPLIT_SEED="${SPLIT_SEED:-42}"
GAP_OOD_PROFILE="${GAP_OOD_PROFILE:-interpolation}"
case "$GAP_OOD_PROFILE" in
  interpolation)
    EXPECTED_HELDOUT_GAPS="5 8"
    HELDOUT_GAPS="${HELDOUT_GAPS:-$EXPECTED_HELDOUT_GAPS}"
    PROFILE_LABEL="interp"
    ;;
  extrapolation)
    # Hold out two lower-end gaps.  Gap 10 is hidden-column-permutation
    # equivalent to gap 6, so {3, 10} would not be a clean geometry OOD test.
    EXPECTED_HELDOUT_GAPS="3 4"
    HELDOUT_GAPS="${HELDOUT_GAPS:-$EXPECTED_HELDOUT_GAPS}"
    PROFILE_LABEL="extrap"
    ;;
  *)
    echo "unknown GAP_OOD_PROFILE=$GAP_OOD_PROFILE (expected interpolation or extrapolation)" >&2
    exit 2
    ;;
esac
if [[ "$HELDOUT_GAPS" != "$EXPECTED_HELDOUT_GAPS" ]]; then
  echo "$GAP_OOD_PROFILE profile requires HELDOUT_GAPS='$EXPECTED_HELDOUT_GAPS'; use run_all.sh for custom splits" >&2
  exit 2
fi
read -r -a HELDOUT_GAP_ARRAY <<< "$HELDOUT_GAPS"
if (( ${#HELDOUT_GAP_ARRAY[@]} != 2 )); then
  echo "HELDOUT_GAPS must contain exactly two values, e.g. '5 8'" >&2
  exit 2
fi
printf -v GAP_LABEL 'g%02d_g%02d' "${HELDOUT_GAP_ARRAY[0]}" "${HELDOUT_GAP_ARRAY[1]}"

RUN_ROOT="${RUN_ROOT:-outputs/ood/gap_${PROFILE_LABEL}_${GAP_LABEL}_seed_${SPLIT_SEED}}"
SPLIT_JSON="${SPLIT_JSON:-$RUN_ROOT/split.json}"
CKPT_ROOT="${CKPT_ROOT:-$RUN_ROOT/checkpoints}"
DATA_DIR="${DATA_DIR:-$RUN_ROOT/data}"

export SPLIT_SEED RUN_ROOT SPLIT_JSON CKPT_ROOT DATA_DIR
if [[ "${PAIR_POLICY:-shared}" != "shared" ]]; then
  echo "the gap-only OOD profiles require PAIR_POLICY=shared" >&2
  exit 2
fi
export SPLIT_KIND=gap_heldout HELDOUT_GAPS PAIR_POLICY=shared
if [[ "${CONDITION_ENCODING:-scalar}" != "scalar" ]]; then
  echo "gap-OOD interpolation/extrapolation requires CONDITION_ENCODING=scalar" >&2
  exit 2
fi
export CONDITION_ENCODING=scalar
# run_all emits this immediately after the split exists, before training.
export PLOT_GAP_OOD_DESIGN=1

echo "gap-OOD profile=$GAP_OOD_PROFILE heldout_gaps=${HELDOUT_GAPS} condition=$CONDITION_ENCODING"
echo "run root: $RUN_ROOT"
bash run_all.sh
