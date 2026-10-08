#!/usr/bin/env bash
# BANK=data/pattern/banks/imp32_v1 EPOCHS=300 BATCH_SIZE=64 bash pattern/sh_scripts/train.sh
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
CONFIG="${CONFIG:-pattern/configs/experiment.json}"
BANK="${BANK:-${1:-}}"
RUN_ID="${RUN_ID:-}"
EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
BETA="${BETA:-}"
KL_WARMUP_EPOCHS="${KL_WARMUP_EPOCHS:-}"
HARD_LOSS_WEIGHT="${HARD_LOSS_WEIGHT:-}"
NF_CHANNELS="${NF_CHANNELS:-}"
LATENT_DIM="${LATENT_DIM:-}"
ENCODER_WIDTH="${ENCODER_WIDTH:-}"
DECODER_WIDTH="${DECODER_WIDTH:-}"

if [[ -z "$BANK" ]]; then
  echo "Set BANK or pass the bank directory as the first argument." >&2
  exit 2
fi
if [[ ${1:-} != --* && $# -gt 0 ]]; then shift; fi
args=(--config "$CONFIG" --bank "$BANK" --device "$DEVICE" --threads "$THREADS")
[[ -z "$RUN_ID" ]] || args+=(--run-id "$RUN_ID")
[[ -z "$EPOCHS" ]] || args+=(--epochs "$EPOCHS")
[[ -z "$BATCH_SIZE" ]] || args+=(--batch-size "$BATCH_SIZE")
[[ -z "$LR" ]] || args+=(--lr "$LR")
[[ -z "$BETA" ]] || args+=(--beta "$BETA")
[[ -z "$KL_WARMUP_EPOCHS" ]] || args+=(--kl-warmup-epochs "$KL_WARMUP_EPOCHS")
[[ -z "$HARD_LOSS_WEIGHT" ]] || args+=(--hard-loss-weight "$HARD_LOSS_WEIGHT")
[[ -z "$NF_CHANNELS" ]] || args+=(--nf-channels "$NF_CHANNELS")
[[ -z "$LATENT_DIM" ]] || args+=(--latent-dim "$LATENT_DIM")
[[ -z "$ENCODER_WIDTH" ]] || args+=(--encoder-width "$ENCODER_WIDTH")
[[ -z "$DECODER_WIDTH" ]] || args+=(--decoder-width "$DECODER_WIDTH")

[[ "$PROGRESS" != "0" ]] || args+=(--no-progress)
mkdir -p "$LOG_DIR"
LOG_FILE="$(mktemp "$LOG_DIR/train_$(date -u +%Y%m%dT%H%M%SZ)_log_XXXXXX")"
echo "Console log: $LOG_FILE"
"$PYTHON" -u -m pattern.train "${args[@]}" "$@" 2>&1 | tee "$LOG_FILE"
