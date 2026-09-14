"""CLI for direct bilevel mask learning."""

from __future__ import annotations

import argparse
from pathlib import Path

from .core import BilevelMaskConfig, train_structure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--relaxation", choices=("soft", "ste"), default="soft")
    for name in (
        "seed", "outer_steps", "inner_steps", "tasks_per_step", "validate_every",
        "validation_restarts", "eval_steps", "eval_restarts", "random_masks", "k_active",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(BilevelMaskConfig(), name))
    for name in (
        "inner_lr", "outer_lr", "temperature_start", "temperature_end", "binary_penalty",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(BilevelMaskConfig(), name))
    arguments = vars(parser.parse_args())
    output = arguments.pop("output")
    device = arguments.pop("device")
    train_structure(BilevelMaskConfig(**arguments), output, device)


if __name__ == "__main__":
    main()
