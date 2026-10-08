#!/usr/bin/env bash
# M4 Pro / 48 GB: full source-task batches, 600 optimizer updates, MPS preferred.
# Missing banks are collected automatically before training.
# BANK=data/pattern/banks/imp32_wgf_v2 bash pattern/sh_scripts/train.sh
# Run from any directory. Paths are resolved from the repository root.
# Edit defaults here, set environment variables, or append Python CLI options.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-auto}"  # auto: MPS (Mac) -> CUDA -> CPU; explicit mps/cpu/cuda:N accepted
THREADS="${THREADS:-1}"
BANK_BATCH_SIZE="${BANK_BATCH_SIZE:-128}"  # Independent MLPs per IMP batch.
PROGRESS="${PROGRESS:-1}"
LOG_DIR="${LOG_DIR:-pattern/runs/launch_logs}"
export PYTHONUNBUFFERED=1
CONFIG="${CONFIG:-pattern/configs/experiment.json}"
for option in "$@"; do
  if [[ "$option" == --help || "$option" == -h ]]; then
    "$PYTHON" -m pattern.train --help
    exit 0
  fi
done
if [[ $# -gt 0 && "$1" != --* ]]; then
  BANK="${BANK:-$1}"
  shift
fi
BANK="${BANK:-data/pattern/banks/imp32_wgf_v2}"
RUN_ID="${RUN_ID:-}"
# Custom CONFIG keeps its own defaults; the M4 defaults below apply to the main config.
if [[ "$CONFIG" == "pattern/configs/experiment.json" ]]; then
  EPOCHS="${EPOCHS:-600}"
  BATCH_SIZE="${BATCH_SIZE:-96}"
  LR="${LR:-0.001}"
  BETA="${BETA:-0.001}"
  KL_WARMUP_EPOCHS="${KL_WARMUP_EPOCHS:-90}"
  HARD_LOSS_WEIGHT="${HARD_LOSS_WEIGHT:-0.2}"
  NF_CHANNELS="${NF_CHANNELS:-16}"
  LATENT_DIM="${LATENT_DIM:-16}"
  ENCODER_WIDTH="${ENCODER_WIDTH:-128}"
  DECODER_WIDTH="${DECODER_WIDTH:-128}"
fi

args=(--config "$CONFIG" --bank "$BANK" --device "$DEVICE" --threads "$THREADS")
[[ -z "$RUN_ID" ]] || args+=(--run-id "$RUN_ID")
[[ -z "${EPOCHS:-}" ]] || args+=(--epochs "$EPOCHS")
[[ -z "${BATCH_SIZE:-}" ]] || args+=(--batch-size "$BATCH_SIZE")
[[ -z "${LR:-}" ]] || args+=(--lr "$LR")
[[ -z "${BETA:-}" ]] || args+=(--beta "$BETA")
[[ -z "${KL_WARMUP_EPOCHS:-}" ]] || args+=(--kl-warmup-epochs "$KL_WARMUP_EPOCHS")
[[ -z "${HARD_LOSS_WEIGHT:-}" ]] || args+=(--hard-loss-weight "$HARD_LOSS_WEIGHT")
[[ -z "${NF_CHANNELS:-}" ]] || args+=(--nf-channels "$NF_CHANNELS")
[[ -z "${LATENT_DIM:-}" ]] || args+=(--latent-dim "$LATENT_DIM")
[[ -z "${ENCODER_WIDTH:-}" ]] || args+=(--encoder-width "$ENCODER_WIDTH")
[[ -z "${DECODER_WIDTH:-}" ]] || args+=(--decoder-width "$DECODER_WIDTH")

[[ "$PROGRESS" != "0" ]] || args+=(--no-progress)
collection_args=(--config "$CONFIG" --parent "$(dirname "$BANK")" --bank-id "$(basename "$BANK")"
                 --device "$DEVICE" --threads "$THREADS" --network-batch-size "$BANK_BATCH_SIZE")
[[ "$PROGRESS" != "0" ]] || collection_args+=(--no-progress)
mkdir -p "$LOG_DIR"
LOG_FILE="$(mktemp "$LOG_DIR/train_$(date -u +%Y%m%dT%H%M%SZ)_log_XXXXXX")"
echo "Console log: $LOG_FILE"
{
  if [[ ! -e "$BANK" ]]; then
    echo "Bank not found; collecting weight/gradient/functional maps: $BANK"
    "$PYTHON" -u data/collect_pattern_maps.py "${collection_args[@]}"
  else
    echo "Using existing bank: $BANK"
  fi
  "$PYTHON" -u -m pattern.train "${args[@]}" "$@"
} 2>&1 | tee "$LOG_FILE"
