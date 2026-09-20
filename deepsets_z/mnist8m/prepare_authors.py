"""Prepare digit sets following the authors' released image_sum.ipynb protocol.

Uses the downloaded MNIST8m image order, split into eight contiguous packs.
Pack 0 is shuffled once for training; each test length draws a pack 1..7.
Zero digits are omitted because the original notebook reserves index 0 for
padding and skips image labels equal to zero.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


LENGTHS = tuple(range(5, 51, 5))
NOTEBOOK = "https://github.com/manzilzaheer/DeepSets/blob/master/DigitSum/image_sum.ipynb"


def write_sets(path: Path, indices: np.ndarray, labels: np.ndarray) -> None:
    mask = indices >= 0
    targets = np.where(mask, labels[np.maximum(indices, 0)], 0).sum(1).astype(np.float32)
    lengths = mask.sum(1).astype(np.int16)
    np.savez(path, indices=indices.astype(np.int32), targets=targets, lengths=lengths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mnist8m"))
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    labels = np.load(args.data_dir / "labels.npy", mmap_mode="r")
    if len(labels) != 8_100_000:
        raise ValueError(f"Unexpected number of labels: {len(labels)}")
    out = args.data_dir / "sets_authors"
    out.mkdir(exist_ok=True)
    pack_size = len(labels) // 8
    rng = np.random.default_rng(args.seed)

    # The author's notebook samples 1..9 actual digits, pads to ten slots,
    # then gives the final 1.23456789% of sets to Keras validation_split.
    lengths = rng.integers(1, 10, size=150_000, dtype=np.int16)
    order = rng.permutation(pack_size)
    usable = order[labels[order] != 0]
    required = int(lengths.sum())
    if required > len(usable):
        raise ValueError("Pack 0 contains too few nonzero digits")
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    train_all = np.full((150_000, 10), -1, dtype=np.int32)
    for index, length in enumerate(lengths):
        train_all[index, -length:] = usable[offsets[index]:offsets[index + 1]]
    # Keras uses ceil() for the validation boundary: 148148 / 1852.
    n_validation = math.ceil(150_000 * 0.0123456789)
    n_train = 150_000 - n_validation
    write_sets(out / "train.npz", train_all[:n_train], labels)
    write_sets(out / "validation.npz", train_all[n_train:], labels)

    packs = {}
    for length in LENGTHS:
        pack = int(rng.integers(1, 8))
        begin = pack * pack_size
        end = (pack + 1) * pack_size
        candidates = np.arange(begin, end)
        usable = candidates[labels[begin:end] != 0]
        required = 10_000 * length
        if required > len(usable):
            raise ValueError(f"Pack {pack} has too few nonzero digits")
        images = usable[:required].reshape(10_000, length)
        write_sets(out / f"test_{length}.npz", images, labels)
        packs[str(length)] = pack

    manifest = {
        "source_protocol": NOTEBOOK,
        "description": "Released notebook set construction; 8 equal contiguous packs from the downloaded MNIST8m mirror",
        "seed": args.seed, "pack_size": pack_size,
        "train_sets": n_train, "validation_sets": n_validation,
        "test_sets_per_length": 10_000, "train_lengths": [1, 9],
        "test_lengths": LENGTHS, "test_packs": packs,
        "skips_zero_digits": True, "individual_labels_used_for_training": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
