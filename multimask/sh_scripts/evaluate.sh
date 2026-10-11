#!/usr/bin/env bash
# Two episodes: six source tasks -> two held-out tasks, frozen decoder, z only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-mps}"
THREADS="${THREADS:-1}"
RUN="${RUN:-multimask/runs/nf_vae_v1}"
PROGRESS="${PROGRESS:-1}"
LOG_DIR="${LOG_DIR:-multimask/runs/launch_logs}"
export PYTHONUNBUFFERED=1
if [[ $# -gt 0 && "$1" != --* ]]; then RUN="$1"; shift; fi
args=(--run "$RUN" --device "$DEVICE" --threads "$THREADS")
# All unspecified values come from the saved run config.
for option in Z_STARTS:z-starts Z_STEPS:z-steps Z_LR:z-lr INNER_STEPS:inner-steps \
              INNER_LR:inner-lr CHILD_STEPS:child-steps CHILD_LR:child-lr \
              SELECT_EVERY:select-every REPLICAS:replicas LOG_EVERY:log-every EVALUATION_ID:evaluation-id; do
  variable="${option%%:*}"; value="${!variable:-}"
  [[ -z "$value" ]] || args+=(--"${option#*:}" "$value")
done
[[ "$PROGRESS" != 0 ]] || args+=(--no-progress)
mkdir -p "$LOG_DIR"
LOG_FILE="$(mktemp "$LOG_DIR/evaluate_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
echo "Console log: $LOG_FILE"
"$PYTHON" -u -m multimask.evaluate "${args[@]}" "$@" 2>&1 | tee "$LOG_FILE"
