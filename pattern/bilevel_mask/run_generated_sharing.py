"""CLI for compact generated parameter-sharing experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .generated_sharing import SharingConfig, run


def main() -> None:
    defaults = SharingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pattern-coverage", choices=("train10", "all16"), default="all16")
    integer_names = (
        "seed", "filter_dim", "latent_dim", "generator_width", "k_active",
        "train_restarts", "eval_restarts", "outer_steps", "inner_steps", "latent_search_steps",
        "latent_patience", "final_refit_steps", "refit_validate_every", "support_per_class", "validation_per_class",
        "query_per_class",
    )
    float_names = (
        "latent_min_delta", "inner_lr", "final_refit_lr", "latent_lr", "generator_lr", "z_radius",
        "temperature_start", "temperature_end", "binary_penalty", "category_entropy_penalty",
        "category_balance_penalty",
    )
    for name in integer_names:
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in float_names:
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    arguments = vars(parser.parse_args())
    output, device = arguments.pop("output"), arguments.pop("device")
    pattern_coverage = arguments.pop("pattern_coverage")
    run(SharingConfig(**arguments), output, device, pattern_coverage)


if __name__ == "__main__":
    main()
