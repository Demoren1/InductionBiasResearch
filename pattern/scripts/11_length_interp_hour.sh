#!/usr/bin/env bash
# Complete pattern-32 interpolation pipeline, bounded to one hour by default.
set -Eeuo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$TASK_ROOT"
source /home/udeneev-av/miniconda3/etc/profile.d/conda.sh
conda activate ras

TASK_STAMP="$(date +%Y%m%d_%H%M%S)"
TASK_OUT="${OUT_DIR:-$TASK_ROOT/pattern/outputs/length_interp/hour_$TASK_STAMP}"
TASK_LIMIT="${TIME_LIMIT:-120m}"
TASK_BANK_MLPS="${BANK_MLPS:-1000}"
TASK_BANK_STEPS="${BANK_STEPS:-2000}"
read -r -a TASK_GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"
mkdir -p -- "$(dirname -- "$TASK_OUT")"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

echo "Output: $TASK_OUT"
echo "GPUs: ${TASK_GPUS[*]}; time limit: $TASK_LIMIT"
echo "Bank: $TASK_BANK_MLPS MLPs/task, $TASK_BANK_STEPS steps, keep top 10%"
echo "CVAE: continuous length; train 3,4,6,8; interpolation test 5,7"

set +e
timeout --signal=TERM --kill-after=30s "$TASK_LIMIT" \
    python -u -m pattern.length_interp.run \
    --out "$TASK_OUT" --bank-mlps "$TASK_BANK_MLPS" --bank-steps "$TASK_BANK_STEPS" \
    --gpus "${TASK_GPUS[@]}" 2>&1 | tee -a "${TASK_OUT}.console.log"
TASK_STATUS=${PIPESTATUS[0]}
set -e

if [[ "$TASK_STATUS" -eq 0 ]]; then
    echo "Completed: $TASK_OUT/RESULTS.md"
elif [[ "$TASK_STATUS" -eq 124 || "$TASK_STATUS" -eq 137 ]]; then
    echo "Time limit reached; completed stage artifacts remain in $TASK_OUT"
    echo "The final report exists only if every stage completed. GPU workers were signalled to stop."
else
    echo "Run failed (exit $TASK_STATUS). Inspect ${TASK_OUT}.console.log and $TASK_OUT/logs"
fi
exit "$TASK_STATUS"
