"""Create and validate a reproducible meta-train/meta-test pattern split."""

import argparse
import json
import random
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def make_split(seed: int, n_train: int = 12) -> dict:
    patterns = list(config.PATTERNS)
    rng = random.Random(seed)
    rng.shuffle(patterns)
    train_patterns = patterns[:n_train]
    test_patterns = patterns[n_train:]
    split = {
        "split_seed": seed,
        "train_patterns": train_patterns,
        "test_patterns": test_patterns,
    }
    validate_split(split, n_train=n_train)
    return split


def validate_split(split: dict, n_train: int = 12) -> None:
    train_patterns = split["train_patterns"]
    test_patterns = split["test_patterns"]
    expected = set(config.PATTERNS)
    if len(train_patterns) != n_train:
        raise ValueError(f"expected {n_train} train patterns, got {len(train_patterns)}")
    if len(test_patterns) != len(expected) - n_train:
        raise ValueError(
            f"expected {len(expected) - n_train} test patterns, "
            f"got {len(test_patterns)}")
    if len(set(train_patterns)) != len(train_patterns):
        raise ValueError("train_patterns contains duplicates")
    if len(set(test_patterns)) != len(test_patterns):
        raise ValueError("test_patterns contains duplicates")
    if set(train_patterns) & set(test_patterns):
        raise ValueError("train and test patterns overlap")
    if set(train_patterns) | set(test_patterns) != expected:
        raise ValueError("train/test patterns do not partition config.PATTERNS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shell_out", type=Path, default=None,
                        help="optional shell file exporting the validated lists")
    args = parser.parse_args()

    split = make_split(args.seed, args.n_train)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split, indent=2) + "\n")
    if args.shell_out is not None:
        args.shell_out.parent.mkdir(parents=True, exist_ok=True)
        train_value = " ".join(split["train_patterns"])
        test_value = " ".join(split["test_patterns"])
        args.shell_out.write_text(
            f"TRAIN_PATTERNS={shlex.quote(train_value)}\n"
            f"TEST_PATTERNS={shlex.quote(test_value)}\n")
    print("TRAIN_PATTERNS=" + " ".join(split["train_patterns"]))
    print("TEST_PATTERNS=" + " ".join(split["test_patterns"]))
    print(f"split -> {args.out}")


if __name__ == "__main__":
    main()
