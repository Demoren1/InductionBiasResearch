"""CLI for the frozen-decoder latent-adapter experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .latent_adapter import run_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--n-train", type=int, default=16_384)
    parser.add_argument("--n-val", type=int, default=4_096)
    parser.add_argument("--n-test", type=int, default=4_096)
    parser.add_argument("--n-adam-test", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adam-steps", type=int, default=2_000)
    parser.add_argument("--adam-lr", type=float, default=.03)
    parser.add_argument("--radius", type=float, default=12.)
    parser.add_argument("--temperature", type=float, default=.5)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if min(args.n_train, args.n_val, args.n_test, args.n_adam_test,
           args.steps, args.batch_size, args.eval_every, args.adam_steps) <= 0:
        parser.error("sizes and step counts must be positive")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    settings = {
        "n_train": args.n_train, "n_val": args.n_val, "n_test": args.n_test,
        "n_adam_test": args.n_adam_test, "steps": args.steps,
        "batch_size": args.batch_size, "eval_every": args.eval_every,
        "lr": args.lr, "adam_steps": args.adam_steps, "adam_lr": args.adam_lr,
        "radius": args.radius, "temperature": args.temperature, "k": args.k,
        "seed": args.seed,
    }
    run_pair(args.pair_root.resolve(), args.out.resolve(), device, settings)


if __name__ == "__main__":
    main()
