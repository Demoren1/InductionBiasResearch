#!/usr/bin/env bash
# M4 / 48 GB: 12 x 1000 maps; NF8/encoder64/z16/decoder128; MPS preferred.
# Missing banks are collected automatically before training.
# Five epochs preserve the shared Toeplitz structure; best checkpoint uses aligned IoU.
# Run from any directory. Paths are resolved from the repository root.
# Edit defaults here, set environment variables, or append Python CLI options.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-auto}"  # auto: MPS (Mac) -> CUDA -> CPU; explicit mps/cpu/cuda:N accepted
THREADS="${THREADS:-1}"
BANK_BATCH_SIZE="${BANK_BATCH_SIZE:-256}"  # Independent MLPs per IMP batch.
PROGRESS="${PROGRESS:-1}"
LOG_DIR="${LOG_DIR:-pattern/runs/launch_logs}"
export PYTHONUNBUFFERED=1
CONFIG="${CONFIG:-pattern/configs/tuned.json}"
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
BANK="${BANK:-data/pattern/banks/imp32_fullsupport_1000_20261009}"
RUN_ID="${RUN_ID:-nf_vae_v2}"
# Custom CONFIG keeps its defaults; override the tuned configuration here or via env.
if [[ "$CONFIG" == "pattern/configs/tuned.json" ]]; then
  EPOCHS="${EPOCHS:-5}"
  BATCH_SIZE="${BATCH_SIZE:-128}"
  LR="${LR:-0.001}"
  BETA="${BETA:-0.005}"
  KL_WARMUP_EPOCHS="${KL_WARMUP_EPOCHS:-30}"
  HARD_LOSS_WEIGHT="${HARD_LOSS_WEIGHT:-0.2}"
  NF_CHANNELS="${NF_CHANNELS:-8}"
  LATENT_DIM="${LATENT_DIM:-16}"
  ENCODER_WIDTH="${ENCODER_WIDTH:-64}"
  DECODER_WIDTH="${DECODER_WIDTH:-128}"
  OPTIMIZER="${OPTIMIZER:-adamw}"
  WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
  INITIALIZATION_SEED="${INITIALIZATION_SEED:-4113}"
  SELECTION_METRIC="${SELECTION_METRIC:-aligned_iou}"
  PATIENCE="${PATIENCE:-60}"
fi

args=(--config "$CONFIG" --bank "$BANK" --device "$DEVICE" --threads "$THREADS")
[[ -z "$RUN_ID" ]] || args+=(--run-id "$RUN_ID")
for option in EPOCHS:epochs BATCH_SIZE:batch-size LR:lr BETA:beta \
              KL_WARMUP_EPOCHS:kl-warmup-epochs HARD_LOSS_WEIGHT:hard-loss-weight \
              NF_CHANNELS:nf-channels LATENT_DIM:latent-dim ENCODER_WIDTH:encoder-width \
              DECODER_WIDTH:decoder-width OPTIMIZER:optimizer WEIGHT_DECAY:weight-decay \
              INITIALIZATION_SEED:initialization-seed SELECTION_METRIC:selection-metric PATIENCE:patience; do
  variable="${option%%:*}"
  value="${!variable:-}"
  [[ -z "$value" ]] || args+=(--"${option#*:}" "$value")
done

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
