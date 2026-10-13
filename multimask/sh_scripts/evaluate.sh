#!/usr/bin/env bash
# Two episodes: six source tasks -> two held-out tasks, frozen decoder, z only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-mps}"
THREADS="${THREADS:-1}"
RUN="${RUN:-multimask/runs/nf_vae_v3}"
Z_METHOD="${Z_METHOD:-gumbel}"  # Adam on z through Gumbel/RelaxedTopK.
Z_CANDIDATES="${Z_CANDIDATES:-4}"  # Two Gaussian directions and their opposites.
Z_RADIUS="${Z_RADIUS:-0.3}"
Z_LR="${Z_LR:-0.05}"
Z_STEPS="${Z_STEPS:-80}"
GUMBEL_TAU="${GUMBEL_TAU:-2.0}"
GUMBEL_TAU_END="${GUMBEL_TAU_END:-1.0}"
GUMBEL_NOISE="${GUMBEL_NOISE:-0.25}"
GUMBEL_NOISE_END="${GUMBEL_NOISE_END:-0.05}"
GUMBEL_SAMPLES="${GUMBEL_SAMPLES:-2}"
PROGRESS="${PROGRESS:-1}"
LOG_DIR="${LOG_DIR:-multimask/runs/launch_logs}"
export PYTHONUNBUFFERED=1
if [[ $# -gt 0 && "$1" != --* ]]; then RUN="$1"; shift; fi
args=(--run "$RUN" --device "$DEVICE" --threads "$THREADS")
# Inner/child settings come from the saved run config; Gumbel uses defaults above.
for option in Z_METHOD:z-method Z_CANDIDATES:z-candidates Z_RADIUS:z-radius \
              Z_STARTS:z-starts Z_STEPS:z-steps Z_LR:z-lr INNER_STEPS:inner-steps \
              GUMBEL_TAU:gumbel-tau GUMBEL_TAU_END:gumbel-tau-end \
              GUMBEL_NOISE:gumbel-noise GUMBEL_NOISE_END:gumbel-noise-end \
              GUMBEL_SAMPLES:gumbel-samples \
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
