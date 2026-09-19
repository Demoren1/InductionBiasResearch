"""Run Yeh et al. (2022) sum-of-numbers with direct and generated A."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

import torch

from .sum_numbers import (
    SumNumbersConfig,
    markdown_report,
    run_sum_numbers_benchmark,
    save_sum_numbers_outputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("yeh2022_generated_sharing/outputs/sum_numbers")
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="small complete CPU/GPU smoke configuration")
    parser.add_argument("--target", choices=("standard", "alternating", "both"), default="both")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--validation-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--outer-steps", type=int, default=None)
    parser.add_argument("--inner-steps", type=int, default=None)
    parser.add_argument("--refit-steps", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--generator-width", type=int, default=None)
    parser.add_argument("--lower-lr", type=float, default=None)
    parser.add_argument("--outer-lr", type=float, default=None)
    parser.add_argument("--neumann-iterations", type=int, default=None)
    parser.add_argument("--neumann-alpha", type=float, default=None)
    parser.add_argument("--test-batch-size", type=int, default=None)
    return parser


def _config_from_args(args: argparse.Namespace) -> SumNumbersConfig:
    config = SumNumbersConfig.quick() if args.quick else SumNumbersConfig()
    values = {
        key: value
        for key, value in {
            "seed": args.seed,
            "sequence_length": args.sequence_length,
            "outer_steps": args.outer_steps,
            "inner_steps": args.inner_steps,
            "refit_steps": args.refit_steps,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "generator_width": args.generator_width,
            "lower_lr": args.lower_lr,
            "outer_lr": args.outer_lr,
            "neumann_iterations": args.neumann_iterations,
            "neumann_alpha": args.neumann_alpha,
            "test_batch_size": args.test_batch_size,
        }.items()
        if value is not None
    }
    return replace(config, **values)


def main() -> None:
    args = _parser().parse_args()
    # One CPU thread per GPU/process: all expensive work is vectorised Torch.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = _config_from_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    if args.quick:
        # Preserve a full protocol smoke run while avoiding the 100k test set.
        train_size = 12 if args.train_size is None else args.train_size
        validation_size = 16 if args.validation_size is None else args.validation_size
        test_size = 256 if args.test_size is None else args.test_size
    else:
        train_size = 100 if args.train_size is None else args.train_size
        validation_size = 150 if args.validation_size is None else args.validation_size
        test_size = 100_000 if args.test_size is None else args.test_size

    targets = ("standard", "alternating") if args.target == "both" else (args.target,)
    summaries: dict[str, dict[str, object]] = {}
    artifacts: dict[str, dict[str, object]] = {}
    started = time.perf_counter()
    for target in targets:
        summary, artifact = run_sum_numbers_benchmark(
            config,
            device,
            alternating=target == "alternating",
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
        )
        summaries[target] = summary
        artifacts[target] = artifact
        print(markdown_report(summary), end="")
    save_sum_numbers_outputs(args.output_dir, summaries, artifacts)
    print(f"runtime_seconds: {time.perf_counter() - started:.3f}")
    print(f"saved: {args.output_dir / 'sum_numbers_results.json'}")
    print(f"saved: {args.output_dir / 'sum_numbers_artifacts.pt'}")
    print(f"saved: {args.output_dir / 'SUM_NUMBERS_RESULTS.md'}")


if __name__ == "__main__":
    main()
