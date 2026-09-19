"""CLI for multi-length generated parameter-sharing experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .multilength_sharing import MultiLengthConfig, run


def main() -> None:
    # Each process owns one GPU and is intentionally limited to one CPU core.
    # The launcher also enforces this with taskset; these settings protect
    # direct CLI invocations and prevent BLAS/OpenMP oversubscription.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    defaults = MultiLengthConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant", choices=("global", "length_latent"), required=True)
    parser.add_argument("--pattern-lengths", default="3,5,7")
    integer_names = (
        "seed", "hidden", "filter_dim", "latent_dim", "generator_width", "train_restarts",
        "eval_restarts", "tasks_per_length", "outer_steps", "inner_steps",
        "temperature_anneal_steps", "checkpoint_every", "outer_patience",
        "checkpoint_task_chunk", "final_refit_steps", "refit_validate_every", "eval_task_chunk",
        "support_per_class", "validation_per_class", "query_per_class",
        "evaluation_support_per_class", "evaluation_validation_per_class",
        "evaluation_query_per_class",
    )
    float_names = (
        "inner_lr", "final_refit_lr", "latent_lr", "generator_lr", "z_radius",
        "temperature_start", "temperature_end", "binary_penalty",
        "category_entropy_penalty", "category_balance_penalty", "outer_min_delta",
    )
    for name in integer_names:
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in float_names:
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    arguments = vars(parser.parse_args())
    output = arguments.pop("output")
    device = arguments.pop("device")
    variant = arguments.pop("variant")
    arguments["pattern_lengths"] = tuple(int(value) for value in arguments.pop("pattern_lengths").split(","))
    run(MultiLengthConfig(**arguments), variant, output, device)


if __name__ == "__main__":
    main()
