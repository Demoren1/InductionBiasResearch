"""Validation-only hyperparameter runner for generated parameter sharing."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from .generated_sharing import ALL_PATTERNS, SharingConfig, _fit_shared, _fixed_assignment, train_generator
from .length32_joint import TEST_PATTERNS, TRAIN_PATTERNS, VALIDATION_PATTERNS, make_data


def main() -> None:
    defaults = SharingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--inner-lr", type=float, default=defaults.inner_lr)
    parser.add_argument("--inner-steps", type=int, default=defaults.inner_steps)
    parser.add_argument("--generator-lr", type=float, default=defaults.generator_lr)
    parser.add_argument("--outer-steps", type=int, default=30)
    parser.add_argument("--train-restarts", type=int, default=8)
    parser.add_argument("--eval-restarts", type=int, default=16)
    parser.add_argument("--latent-search-steps", type=int, default=40)
    parser.add_argument("--latent-patience", type=int, default=10)
    parser.add_argument("--final-refit-steps", type=int, default=800)
    parser.add_argument("--pattern-coverage", choices=("train10", "train12", "all16"), default="train10")
    args = parser.parse_args()
    config = SharingConfig(
        seed=args.seed, inner_lr=args.inner_lr, inner_steps=args.inner_steps,
        generator_lr=args.generator_lr, outer_steps=args.outer_steps,
        train_restarts=args.train_restarts, eval_restarts=args.eval_restarts,
        latent_search_steps=args.latent_search_steps, latent_patience=args.latent_patience,
        final_refit_steps=args.final_refit_steps,
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    started = time.monotonic()
    if args.pattern_coverage == "train10":
        training_patterns, evaluation_patterns, evaluation_config = TRAIN_PATTERNS, VALIDATION_PATTERNS, config
    elif args.pattern_coverage == "train12":
        training_patterns = TRAIN_PATTERNS + VALIDATION_PATTERNS
        evaluation_patterns, evaluation_config = TEST_PATTERNS, config
    else:
        training_patterns, evaluation_patterns = ALL_PATTERNS, ALL_PATTERNS
        evaluation_config = replace(config, seed=config.seed + 10_000)
    generator, shared_z, history = train_generator(config, device, training_patterns)
    data = make_data(evaluation_patterns, evaluation_config, device)
    candidate = _fixed_assignment(generator, shared_z, len(evaluation_patterns), config.eval_restarts, config)
    evaluation = _fit_shared(candidate, data, evaluation_config, device, "generated sharing")
    convergence = {"mode": "fixed shared latent bank", "candidates": config.train_restarts}
    result = {"config": config.to_dict(), "pattern_coverage": args.pattern_coverage,
              "training_patterns": list(training_patterns), "evaluation_patterns": list(evaluation_patterns),
              "evaluation_data_seed": evaluation_config.seed,
              "history": history, "convergence": convergence,
              "evaluation": evaluation,
              "generator_parameters": sum(parameter.numel() for parameter in generator.parameters()),
              "seconds": time.monotonic() - started}
    torch.save({"config": config.to_dict(), "generator": generator.state_dict(), "shared_z": shared_z,
                "history": history}, output / "training.pt")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
