"""Run one shared generator over several synthetic linear tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .linear import LinearExperimentConfig
from .multitask_linear import run_multitask_generated_linear
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
    parser.add_argument("--generator-seed", type=int, default=0)
    parser.add_argument("--latent-mode", choices=("per_task", "global"), default="per_task")
    parser.add_argument("--assignment-mode", choices=("soft", "ste"), default="soft")
    parser.add_argument("--task-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--input-length", type=int, default=5)
    parser.add_argument("--kernel-length", type=int, default=3)
    parser.add_argument("--signal-length", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--outer-steps", type=int, default=1_000)
    parser.add_argument("--outer-lr", type=float, default=None)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--generator-width", type=int, default=64)
    parser.add_argument(
        "--lower-solver",
        choices=("paper_projection", "exact_constrained"),
        default="exact_constrained",
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--nuclear-weight", type=float, default=0.01)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=1.0)
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.benchmark == "cross_correlation":
        benchmarks = [
            make_cross_correlation(
                CrossCorrelationSpec(
                    input_length=args.input_length,
                    kernel_length=args.kernel_length,
                    noise_std=0.1 if args.noise_std is None else args.noise_std,
                ),
                seed=seed,
                device=device,
            )
            for seed in args.task_seeds
        ]
        default_lr = 0.1
    else:
        benchmarks = [
            make_unit_step_denoising(
                UnitStepDenoisingSpec(
                    signal_length=args.signal_length,
                    noise_std=3.1622776602 if args.noise_std is None else args.noise_std,
                ),
                seed=seed,
                device=device,
            )
            for seed in args.task_seeds
        ]
        default_lr = 0.2
    config = LinearExperimentConfig(
        seed=args.generator_seed,
        outer_steps=args.outer_steps,
        outer_lr=default_lr if args.outer_lr is None else args.outer_lr,
        latent_dim=args.latent_dim,
        generator_width=args.generator_width,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        entropy_weight=args.entropy_weight,
        nuclear_weight=args.nuclear_weight,
        lower_solver=args.lower_solver,
        ridge=args.ridge,
        restarts=1,
    )
    run_multitask_generated_linear(
        benchmarks,
        config,
        args.output,
        shared_latent=args.latent_mode == "global",
        assignment_mode=args.assignment_mode,
    )


if __name__ == "__main__":
    main()
