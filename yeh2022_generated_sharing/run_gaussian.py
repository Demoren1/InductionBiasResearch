"""CLI for the batched Yeh et al. Gaussian generated-sharing benchmark."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from .gaussian import GaussianConfig, markdown_report, run_gaussian_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("yeh2022_generated_sharing/outputs/gaussian"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="small CPU/GPU smoke configuration")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--dimensions", type=int, default=None)
    parser.add_argument("--true-rank", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--mean-spacing", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--optimizer", choices=("adam", "rmsprop"), default=None)
    parser.add_argument(
        "--lower-solver",
        choices=("paper_pinv", "release_normalized"),
        default=None,
    )
    parser.add_argument("--entropy-weight", type=float, default=None)
    parser.add_argument("--nuclear-weight", type=float, default=None)
    parser.add_argument("--generator-width", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    return parser


def _config_from_args(args: argparse.Namespace) -> GaussianConfig:
    config = GaussianConfig.quick() if args.quick else GaussianConfig()
    changes = {
        key: value
        for key, value in {
            "seed": args.seed,
            "runs": args.runs,
            "dimensions": args.dimensions,
            "true_rank": args.true_rank,
            "num_samples": args.num_samples,
            "num_train": args.num_train,
            "noise_std": args.noise_std,
            "mean_spacing": args.mean_spacing,
            "epochs": args.epochs,
            "restarts": args.restarts,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
            "lower_solver": args.lower_solver,
            "entropy_weight": args.entropy_weight,
            "nuclear_weight": args.nuclear_weight,
            "generator_width": args.generator_width,
            "latent_dim": args.latent_dim,
        }.items()
        if value is not None
    }
    return replace(config, **changes)


def main() -> None:
    args = _parser().parse_args()
    # The repository's GPU policy is one CPU worker per GPU.  This benchmark
    # uses batched tensor operations, so extra BLAS workers only add contention.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = _config_from_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    started = time.perf_counter()
    summary, artifacts = run_gaussian_benchmark(config, device)
    summary["runtime_seconds"] = time.perf_counter() - started
    summary["device"] = str(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "gaussian_results.json"
    tensor_path = args.output_dir / "gaussian_artifacts.pt"
    report_path = args.output_dir / "gaussian_report.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save(artifacts, tensor_path)
    report_path.write_text(markdown_report(summary), encoding="utf-8")
    print(markdown_report(summary), end="")
    print(f"saved: {json_path}")
    print(f"saved: {tensor_path}")
    print(f"saved: {report_path}")


if __name__ == "__main__":
    main()
