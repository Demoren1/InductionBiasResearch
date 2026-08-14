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
    """(n,) bool -- True where a row of x01 contains pat_bits as a substring.

    x01: (n, L) 0/1
    pat_bits: (PATTERN_LEN,) 0/1
    """
    k = config.PATTERN_LEN
    windows = x01.unfold(1, k, 1)                 # (n, L-k+1, k)
    pat = pat_bits.view(1, 1, k).to(x01.dtype)
    match = (windows == pat).all(dim=2)           # (n, L-k+1)
    return match.any(dim=1)


def make_dataset(pat: str, n_samples: int, seed: int,
                 pos_fraction: float = 0.5) -> dict:
    """Return {"x": (n, L) +/-1, "y": (n,) 0/1, "pattern": pat,
               "pattern_pm1": (PATTERN_LEN,) +/-1}."""
    g = torch.Generator().manual_seed(seed)
    pat_bits = config.pattern_to_bits(pat)

    x01 = torch.randint(0, 2, (n_samples, config.SEQ_LEN), generator=g).to(torch.float)
    y = contains_pattern(x01, pat_bits).to(torch.float)

    # Rebalance toward pos_fraction by injecting the pattern at a random
    # window offset into randomly chosen negative sequences.
    n_pos = int(y.sum().item())
    target_pos = round(pos_fraction * n_samples)
    need = target_pos - n_pos
    if need > 0:
        neg_idx = torch.nonzero(y == 0).squeeze(1)
        if neg_idx.numel() > 0:
            perm = torch.randperm(neg_idx.numel(), generator=g)[:need]
            inject_idx = neg_idx[perm]
            starts = torch.randint(0, config.N_WINDOWS, (inject_idx.numel(),),
                                   generator=g)
            for r, (idx, start) in enumerate(zip(inject_idx.tolist(),
                                                 starts.tolist())):
                x01[idx, start:start + config.PATTERN_LEN] = pat_bits
                y[idx] = 1.0

    x = 2.0 * x01 - 1.0  # 0 -> -1, 1 -> +1
    return {
        "x": x,
        "y": y,
        "pattern": pat,
        "pattern_pm1": config.pattern_to_pm1(pat),
    }


def gold_first_layer(pat: str) -> torch.Tensor:
    """(N_WINDOWS, SEQ_LEN) = (5, 8).

    W[i, i:i+PATTERN_LEN] = pattern_to_pm1(pat); zeros elsewhere.
    """
    W = torch.zeros(config.N_WINDOWS, config.SEQ_LEN)
    pm1 = config.pattern_to_pm1(pat)
    for i in range(config.N_WINDOWS):
        W[i, i:i + config.PATTERN_LEN] = pm1
    return W


def ideal_mask(hidden: int | None = None) -> torch.Tensor:
    """(SEQ_LEN, H) binary.

    Hidden unit h is assigned window (h % N_WINDOWS) and has ones on rows
    [w : w+PATTERN_LEN] of column h. This is the Toeplitz / sliding-window
    support (pattern-independent).
    """
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