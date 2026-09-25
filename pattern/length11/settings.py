"""Fixed dimensions and output paths for the length-11 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "length11_agreement_20260924"
PATTERNS = tuple(format(i, "04b") for i in range(16))
SEQ_LEN = 11
PATTERN_LEN = 4
HIDDEN = 8
WINDOWS = SEQ_LEN - PATTERN_LEN + 1
EDGES = WINDOWS * PATTERN_LEN
MASK_DIM = SEQ_LEN * HIDDEN
assert (WINDOWS, EDGES, MASK_DIM) == (8, 32, 88)


@dataclass(frozen=True)
class BankConfig:
    seq_len: int = SEQ_LEN
    hidden: int = HIDDEN
    seed: int = 20260924
    bank_mlps: int = 4096
    bank_steps: int = 4000
    bank_batch: int = 128
    bank_lr: float = 1e-3
    bank_val_size: int = 2048
    support_pool_size: int = 8192
    top_fraction: float = .1
    input_split_seed: int = 1729
