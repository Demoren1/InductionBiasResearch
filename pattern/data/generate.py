"""Generate validation datasets for the pattern-in-sequence experiment.

For each of the 16 length-4 bit patterns:
  - draw n random 0/1 sequences of length SEQ_LEN,
  - label each sequence 1 if it contains the pattern as a contiguous
    substring, else 0,
  - rebalance toward pos_fraction by injecting the pattern at a random
    window offset into randomly chosen negative sequences,
  - convert the 0/1 sequences to +/-1 and save a fixed validation set.

The fixed validation set per pattern is saved once so every MLP is scored
against identical data. Training data is generated on the fly during
training (infinite stream of i.i.d. windows).
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def contains_pattern(x01: torch.Tensor, pat_bits: torch.Tensor) -> torch.Tensor:
    """Return whether each binary sequence contains the pattern."""
    k = config.PATTERN_LEN
    windows = x01.unfold(1, k, 1)
    pat = pat_bits.view(1, 1, k).to(x01.dtype)
    match = (windows == pat).all(dim=2)
    return match.any(dim=1)


def inject_positive_examples(x01: torch.Tensor, y: torch.Tensor,
                             pat_bits: torch.Tensor, target_fraction: float,
                             generator: torch.Generator) -> torch.Tensor:
    """Inject the pattern into negatives until the target positive fraction."""
    target = round(target_fraction * x01.size(0))
    missing = max(0, target - int(y.sum().item()))
    negatives = torch.nonzero(y == 0).squeeze(1)
    selected = negatives[torch.randperm(negatives.numel(), generator=generator)[:missing]]
    starts = torch.randint(config.N_WINDOWS, (selected.numel(),), generator=generator)
    for index, start in zip(selected.tolist(), starts.tolist()):
        x01[index, start:start + config.PATTERN_LEN] = pat_bits
    return contains_pattern(x01, pat_bits).float()


def make_dataset(pat: str, n_samples: int, seed: int,
                 pos_fraction: float = 0.5) -> dict:
    """Create a seeded, balanced pattern-classification dataset."""
    if not 0.0 <= pos_fraction <= 1.0:
        raise ValueError("pos_fraction must be in [0, 1]")
    g = torch.Generator().manual_seed(seed)
    pat_bits = config.pattern_to_bits(pat)

    x01 = torch.randint(0, 2, (n_samples, config.SEQ_LEN), generator=g).to(torch.float)
    y = contains_pattern(x01, pat_bits).to(torch.float)
    y = inject_positive_examples(x01, y, pat_bits, pos_fraction, g)
    x = 2.0 * x01 - 1.0
    return {
        "x": x,
        "y": y,
        "pattern": pat,
        "pattern_pm1": config.pattern_to_pm1(pat),
    }


def gold_first_layer(pat: str) -> torch.Tensor:
    """Return the matched-filter weights for every input window."""
    W = torch.zeros(config.N_WINDOWS, config.SEQ_LEN)
    pm1 = config.pattern_to_pm1(pat)
    for i in range(config.N_WINDOWS):
        W[i, i:i + config.PATTERN_LEN] = pm1
    return W


def ideal_mask(hidden: int | None = None) -> torch.Tensor:
    """Return the pattern-independent Toeplitz support mask."""
    if hidden is None:
        hidden = config.H
    m = torch.zeros(config.SEQ_LEN, hidden, dtype=torch.long)
    for h in range(hidden):
        w = h % config.N_WINDOWS
        m[w:w + config.PATTERN_LEN, h] = 1
    return m


def generate_data() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for i, pat in enumerate(config.PATTERNS):
        val = make_dataset(pat, config.N_VAL_SAMPLES, seed=2000 + i,
                           pos_fraction=config.POS_FRACTION)
        path = config.val_path(pat)
        torch.save(val, path)
        pos = float(val["y"].mean().item())
        print(f"[data] pattern={pat} n={config.N_VAL_SAMPLES} "
              f"pos_fraction={pos:.3f} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pattern validation datasets.")
    parser.add_argument("--n_val", type=int, default=config.N_VAL_SAMPLES)
    parser.add_argument("--pos_fraction", type=float, default=config.POS_FRACTION)
    args = parser.parse_args()
    config.N_VAL_SAMPLES = args.n_val
    config.POS_FRACTION = args.pos_fraction
    generate_data()


if __name__ == "__main__":
    main()
