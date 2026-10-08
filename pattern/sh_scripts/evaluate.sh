#!/usr/bin/env bash
# RUN=pattern/runs/nf_vae_v1 CHILD_STEPS=500 REPLICAS=3 bash pattern/sh_scripts/evaluate.sh
# Run from any directory. Paths are resolved from the repository root.
# Edit defaults here, set environment variables, or append Python CLI options.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-auto}"  # auto: MPS (Mac) -> CUDA -> CPU; explicit mps/cpu/cuda:N accepted
THREADS="${THREADS:-1}"
PROGRESS="${PROGRESS:-1}"
LOG_DIR="${LOG_DIR:-pattern/runs/launch_logs}"
export PYTHONUNBUFFERED=1
RUN="${RUN:-${1:-}}"
EVALUATION_ID="${EVALUATION_ID:-}"
SPLIT="${SPLIT:-test}"
CHILD_STEPS="${CHILD_STEPS:-0}"
REPLICAS="${REPLICAS:-3}"
LOSS_LOG_EVERY="${LOSS_LOG_EVERY:-10}"

if [[ -z "$RUN" ]]; then
  echo "Set RUN or pass the training run directory as the first argument." >&2
  exit 2
fi
if [[ ${1:-} != --* && $# -gt 0 ]]; then shift; fi
args=(--run "$RUN" --split "$SPLIT" --device "$DEVICE" --threads "$THREADS"
      --child-steps "$CHILD_STEPS" --replicas "$REPLICAS" --loss-log-every "$LOSS_LOG_EVERY")
[[ -z "$EVALUATION_ID" ]] || args+=(--evaluation-id "$EVALUATION_ID")

[[ "$PROGRESS" != "0" ]] || args+=(--no-progress)
mkdir -p "$LOG_DIR"
LOG_FILE="$(mktemp "$LOG_DIR/evaluate_$(date -u +%Y%m%dT%H%M%SZ)_log_XXXXXX")"
echo "Console log: $LOG_FILE"
"$PYTHON" -u -m pattern.evaluate "${args[@]}" "$@" 2>&1 | tee "$LOG_FILE"
