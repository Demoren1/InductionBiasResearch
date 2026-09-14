"""CLI for length-32 joint structure/weight search."""

from __future__ import annotations

import argparse
from pathlib import Path

from .length32_joint import Length32Config, run


def main() -> None:
    defaults = Length32Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    integer_names = (
        "seed", "latent_dim", "generator_width", "restarts", "outer_steps", "inner_max_steps", "eval_inner_max_steps",
        "inner_patience", "weight_steps_per_z", "frozen_pretrain_steps", "eval_weight_steps",
        "support_per_class", "validation_per_class", "query_per_class",
    )
    float_names = ("inner_min_delta", "weight_lr", "z_lr", "generator_lr", "z_radius",
                   "temperature_start", "temperature_end", "binary_penalty")
    for name in integer_names:
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in float_names:
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    arguments = vars(parser.parse_args())
    output, device = arguments.pop("output"), arguments.pop("device")
    run(Length32Config(**arguments), output, device)


if __name__ == "__main__":
    main()
