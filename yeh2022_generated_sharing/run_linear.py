"""Run the generated-sharing cross-correlation or denoising benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .linear import LinearExperimentConfig, run_linear_experiment
from .tasks import (
    CrossCorrelationSpec,
    UnitStepDenoisingSpec,
    make_cross_correlation,
    make_unit_step_denoising,
)


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("cross_correlation", "denoising"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-length", type=int, default=15)
    parser.add_argument("--kernel-length", type=int, default=5)
    parser.add_argument("--signal-length", type=int, default=15)
    parser.add_argument("--train-size", type=int, default=50)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=10_000)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--outer-steps", type=int, default=1_000)
    parser.add_argument("--outer-lr", type=float, default=None)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--generator-width", type=int, default=64)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--nuclear-weight", type=float, default=0.01)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument(
        "--lower-solver",
        choices=("paper_projection", "exact_constrained"),
        default="paper_projection",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=1_000)
    parser.add_argument("--restarts", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.benchmark == "cross_correlation":
        benchmark = make_cross_correlation(
            CrossCorrelationSpec(
                input_length=args.input_length,
                kernel_length=args.kernel_length,
                train_size=args.train_size,
                validation_size=args.validation_size,
                test_size=args.test_size,
                noise_std=0.1 if args.noise_std is None else args.noise_std,
            ),
            seed=args.seed,
            device=device,
        )
        default_lr = 0.1
    else:
        benchmark = make_unit_step_denoising(
            UnitStepDenoisingSpec(
                signal_length=args.signal_length,
                train_size=args.train_size,
                validation_size=args.validation_size,
                test_size=args.test_size,
                noise_std=1.0 if args.noise_std is None else args.noise_std,
            ),
            seed=args.seed,
            device=device,
        )
        default_lr = 0.2
    config = LinearExperimentConfig(
        seed=args.seed,
        outer_steps=args.outer_steps,
        outer_lr=default_lr if args.outer_lr is None else args.outer_lr,
        latent_dim=args.latent_dim,
        generator_width=args.generator_width,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        entropy_weight=args.entropy_weight,
        nuclear_weight=args.nuclear_weight,
        ridge=args.ridge,
        lower_solver=args.lower_solver,
        checkpoint_every=args.checkpoint_every,
        patience=args.patience,
        restarts=args.restarts,
    )
    run_linear_experiment(benchmark, config, args.output)


if __name__ == "__main__":
    main()
