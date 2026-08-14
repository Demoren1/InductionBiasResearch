"""Generate the MovingAverage datasets.

For each (kernel K, offset S) in KERNELS x OFFSETS:
  sample   = L i.i.d. values drawn from N(0, 1)
  target   = mean of the K values of the sample ending at position L - S

A fixed validation set per (k, s) pair is saved once so every MLP is
scored against identical data. Training data is generated on the fly
during training (infinite stream of i.i.d. windows).
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def make_dataset(kernel: int, offset: int, n_samples: int, seed: int) -> dict:
    """Return {"x": (n, L), "y": (n,) } for one (kernel, offset) pair."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, config.L, generator=g)
    y = x[:, config.L - offset - kernel: config.L - offset].mean(dim=1)
    return {"x": x, "y": y}


def preallocate_data_dir() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_data() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for kernel in config.KERNELS:
        for offset in config.OFFSETS:
            val = make_dataset(kernel, offset, config.N_VAL_SAMPLES,
                               seed=1000 + kernel * 100 + offset)
            path = config.DATA_DIR / f"val_kernel_{kernel}_offset_{offset}.pt"
            torch.save(val, path)
            print(f"[data] saved validation set kernel={kernel} offset={offset} "
                  f"n={config.N_VAL_SAMPLES} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MA validation datasets.")
    parser.add_argument("--n_val", type=int, default=config.N_VAL_SAMPLES)
    args = parser.parse_args()
    config.N_VAL_SAMPLES = args.n_val
    generate_data()


if __name__ == "__main__":
    main()