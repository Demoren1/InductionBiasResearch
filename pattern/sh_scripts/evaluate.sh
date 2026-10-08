#!/usr/bin/env bash
# M4 Pro / 48 GB: source-only latent selection, 12 source tasks -> 4 held-out tasks.
# RUN=pattern/runs/nf_vae_v1 bash pattern/sh_scripts/evaluate.sh
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
MODE="${MODE:-latent}"  # latent: transfer; reconstruction: legacy bank-map diagnostic
CHILD_STEPS="${CHILD_STEPS:-500}"
REPLICAS="${REPLICAS:-3}"
LOSS_LOG_EVERY="${LOSS_LOG_EVERY:-10}"
Z_STARTS="${Z_STARTS:-8}"
Z_STEPS="${Z_STEPS:-100}"
Z_LR="${Z_LR:-0.05}"
INNER_STEPS="${INNER_STEPS:-64}"
INNER_LR="${INNER_LR:-0.2}"
SUPPORT_COUNT="${SUPPORT_COUNT:-256}"
QUERY_COUNT="${QUERY_COUNT:-64}"
CHILD_LR="${CHILD_LR:-0.03}"

if [[ -z "$RUN" ]]; then
  echo "Set RUN or pass the training run directory as the first argument." >&2
  exit 2
fi
if [[ ${1:-} != --* && $# -gt 0 ]]; then shift; fi
args=(--run "$RUN" --mode "$MODE" --split "$SPLIT" --device "$DEVICE" --threads "$THREADS"
      --child-steps "$CHILD_STEPS" --replicas "$REPLICAS" --loss-log-every "$LOSS_LOG_EVERY")
if [[ "$MODE" == "latent" ]]; then
  args+=(--z-starts "$Z_STARTS" --z-steps "$Z_STEPS" --z-lr "$Z_LR"
         --inner-steps "$INNER_STEPS" --inner-lr "$INNER_LR"
         --support-count "$SUPPORT_COUNT" --query-count "$QUERY_COUNT" --child-lr "$CHILD_LR")
fi
[[ -z "$EVALUATION_ID" ]] || args+=(--evaluation-id "$EVALUATION_ID")

[[ "$PROGRESS" != "0" ]] || args+=(--no-progress)
mkdir -p "$LOG_DIR"
LOG_FILE="$(mktemp "$LOG_DIR/evaluate_$(date -u +%Y%m%dT%H%M%SZ)_log_XXXXXX")"
echo "Console log: $LOG_FILE"
"$PYTHON" -u -m pattern.evaluate "${args[@]}" "$@" 2>&1 | tee "$LOG_FILE"
