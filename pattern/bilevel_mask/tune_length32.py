"""Validation-only runners for tuning the length-32 joint search."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from .length32_joint import (
    Generator, Length32Config, TRAIN_PATTERNS, VALIDATION_PATTERNS, Weights,
    _new_state, fit_and_evaluate, joint_search, make_data, masks, train_generator,
)


def tune_inner(checkpoint: Path, output: Path, device: torch.device, overrides: dict) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = replace(Length32Config(**payload["config"]), **overrides)
    generator = Generator(config).to(device)
    generator.load_state_dict(payload["generator"])
    data = make_data(VALIDATION_PATTERNS, config, device)
    z, weights = _new_state(len(VALIDATION_PATTERNS), config, device, "inner-sweep-common")
    started = time.monotonic()
    convergence = joint_search(generator, z, weights, data, config, config.temperature_end)
    with torch.no_grad():
        candidate = masks(generator, z, config, config.temperature_end, "hard")
    evaluation = fit_and_evaluate(candidate, data, config, device, "joint")
    result = {"mode": "inner", "checkpoint": str(checkpoint), "config": config.to_dict(),
              "convergence": convergence, "evaluation": evaluation, "seconds": time.monotonic() - started}
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def tune_outer(output: Path, device: torch.device, config: Length32Config) -> None:
    started = time.monotonic()
    generator, history = train_generator(config, device)
    data = make_data(VALIDATION_PATTERNS, config, device)
    z, weights = _new_state(len(VALIDATION_PATTERNS), config, device, "outer-sweep-common")
    convergence = joint_search(generator, z, weights, data, config, config.temperature_end)
    with torch.no_grad():
        candidate = masks(generator, z, config, config.temperature_end, "hard")
    evaluation = fit_and_evaluate(candidate, data, config, device, "joint")
    result = {"mode": "outer", "config": config.to_dict(), "history": history,
              "convergence": convergence, "evaluation": evaluation,
              "generator_parameters": sum(p.numel() for p in generator.parameters()),
              "seconds": time.monotonic() - started}
    output.mkdir(parents=True, exist_ok=False)
    torch.save({"config": config.to_dict(), "generator": generator.state_dict(), "history": history}, output / "training.pt")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    defaults = Length32Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inner", "outer"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--latent-dim", type=int, default=defaults.latent_dim)
    parser.add_argument("--generator-width", type=int, default=defaults.generator_width)
    parser.add_argument("--weight-steps-per-z", type=int, default=defaults.weight_steps_per_z)
    parser.add_argument("--weight-lr", type=float, default=defaults.weight_lr)
    parser.add_argument("--z-lr", type=float, default=defaults.z_lr)
    parser.add_argument("--temperature-end", type=float, default=defaults.temperature_end)
    parser.add_argument("--z-radius", type=float, default=defaults.z_radius)
    parser.add_argument("--generator-lr", type=float, default=defaults.generator_lr)
    parser.add_argument("--binary-penalty", type=float, default=defaults.binary_penalty)
    parser.add_argument("--outer-steps", type=int, default=defaults.outer_steps)
    parser.add_argument("--inner-max-steps", type=int, default=defaults.inner_max_steps)
    parser.add_argument("--eval-inner-max-steps", type=int, default=defaults.eval_inner_max_steps)
    parser.add_argument("--eval-weight-steps", type=int, default=1000)
    args = parser.parse_args()
    overrides = {"seed": args.seed, "latent_dim": args.latent_dim, "generator_width": args.generator_width,
                 "weight_steps_per_z": args.weight_steps_per_z,
                 "weight_lr": args.weight_lr, "z_lr": args.z_lr,
                 "temperature_end": args.temperature_end,
                 "z_radius": args.z_radius, "generator_lr": args.generator_lr,
                 "binary_penalty": args.binary_penalty,
                 "outer_steps": args.outer_steps, "inner_max_steps": args.inner_max_steps,
                 "eval_inner_max_steps": args.eval_inner_max_steps,
                 "eval_weight_steps": args.eval_weight_steps}
    if args.mode == "inner":
        if args.checkpoint is None:
            parser.error("--checkpoint is required in inner mode")
        tune_inner(args.checkpoint, args.output, torch.device(args.device), overrides)
    else:
        tune_outer(args.output, torch.device(args.device), replace(defaults, **overrides))


if __name__ == "__main__":
    main()
