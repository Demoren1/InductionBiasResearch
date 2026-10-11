#!/usr/bin/env bash
# M4 / 48 GB. Edit these defaults, set environment variables, or append CLI flags.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-mps}"  # Explicit auto/cpu/cuda:N also supported.
THREADS="${THREADS:-1}"
CONFIG="${CONFIG:-multimask/config.json}"
BANK="${BANK:-data/multimask/banks/imp32_1024_v1}"
RUN_ID="${RUN_ID:-nf_vae_v1}"
MAPS_PER_TASK="${MAPS_PER_TASK:-1024}"
BANK_BATCH="${BANK_BATCH:-64}"
PROGRESS="${PROGRESS:-1}"
LOG_DIR="${LOG_DIR:-multimask/runs/launch_logs}"
export PYTHONUNBUFFERED=1
if [[ $# -gt 0 && "$1" != --* ]]; then BANK="$1"; shift; fi
args=(--config "$CONFIG" --bank "$BANK" --run-id "$RUN_ID" --device "$DEVICE"
      --threads "$THREADS" --maps-per-task "$MAPS_PER_TASK" --bank-batch "$BANK_BATCH")
# Optional env overrides; otherwise Python uses CONFIG's training/IMP values.
for option in BANK_STEPS:bank-steps EPOCHS:epochs BATCH_SIZE:batch-size LR:lr BETA:beta; do
  variable="${option%%:*}"; value="${!variable:-}"
  [[ -z "$value" ]] || args+=(--"${option#*:}" "$value")
done
[[ "$PROGRESS" != 0 ]] || args+=(--no-progress)
mkdir -p "$LOG_DIR"
LOG_FILE="$(mktemp "$LOG_DIR/train_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
echo "Console log: $LOG_FILE"
"$PYTHON" -u -m multimask.train "${args[@]}" "$@" 2>&1 | tee "$LOG_FILE"
