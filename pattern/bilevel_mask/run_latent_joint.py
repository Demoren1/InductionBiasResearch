"""CLI for vectorized joint latent-generator and masked-MLP training."""

from __future__ import annotations

import argparse
from pathlib import Path

from .latent_joint import LatentJointConfig, train_latent_generator


def main() -> None:
    defaults = LatentJointConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--relaxation", choices=("soft", "ste"), default=defaults.relaxation)
    parser.add_argument("--exact-population-loss", action="store_true", dest="exact_population_loss",
                        help="use all 2^8 inputs for every optimization role")
    parser.add_argument("--without-replacement", action="store_false", dest="sample_with_replacement")
    parser.set_defaults(exact_population_loss=defaults.exact_population_loss)
    parser.set_defaults(sample_with_replacement=defaults.sample_with_replacement)
    integer_names = (
        "seed", "latent_dim", "generator_width", "restarts", "outer_steps", "weight_steps",
        "eval_weight_steps", "eval_rounds", "z_max_steps", "z_patience", "support_positive",
        "support_negative", "validation_positive", "validation_negative", "query_positive",
        "query_negative", "k_active",
    )
    float_names = (
        "z_min_delta", "weight_lr", "z_lr", "generator_lr", "z_radius",
        "temperature_start", "temperature_end", "binary_penalty",
    )
    for name in integer_names:
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in float_names:
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    arguments = vars(parser.parse_args())
    output, device = arguments.pop("output"), arguments.pop("device")
    train_latent_generator(LatentJointConfig(**arguments), output, device)


if __name__ == "__main__":
    main()
