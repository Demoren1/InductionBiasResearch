#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m sine_mask_vae run "$@"
