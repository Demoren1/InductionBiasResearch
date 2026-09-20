"""Train only the multi-length generator for fast outer-loop diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .generated_sharing import assignments
from .multilength_sharing import (
    MultiLengthConfig,
    _active_iou,
    cyclic_assignment,
    train_generator,
)


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant", choices=("global", "length_latent"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-steps", type=int, default=400)
    parser.add_argument("--anneal-steps", type=int, default=200)
    parser.add_argument("--outer-patience", type=int, default=12)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--tasks-per-length", type=int, default=8)
    parser.add_argument("--inner-steps", type=int, default=80)
    parser.add_argument("--support-per-class", type=int, default=256)
    parser.add_argument("--validation-per-class", type=int, default=128)
    parser.add_argument("--query-per-class", type=int, default=256)
    parser.add_argument("--generator-lr", type=float, default=0.003)
    parser.add_argument("--latent-lr", type=float, default=0.03)
    parser.add_argument("--temperature-end", type=float, default=0.1)
    parser.add_argument("--binary-penalty", type=float, default=0.01)
    parser.add_argument("--category-entropy-penalty", type=float, default=0.001)
    parser.add_argument("--category-balance-penalty", type=float, default=0.01)
    args = parser.parse_args()

    config = MultiLengthConfig(
        seed=args.seed,
        outer_steps=args.outer_steps,
        temperature_anneal_steps=args.anneal_steps,
        outer_patience=args.outer_patience,
        checkpoint_every=args.checkpoint_every,
        tasks_per_length=args.tasks_per_length,
        inner_steps=args.inner_steps,
        support_per_class=args.support_per_class,
        validation_per_class=args.validation_per_class,
        query_per_class=args.query_per_class,
        generator_lr=args.generator_lr,
        latent_lr=args.latent_lr,
        temperature_end=args.temperature_end,
        binary_penalty=args.binary_penalty,
        category_entropy_penalty=args.category_entropy_penalty,
        category_balance_penalty=args.category_balance_penalty,
    )
    device = torch.device(args.device)
    torch.manual_seed(config.seed)
    generator, z, history = train_generator(config, args.variant, device)
    with torch.no_grad():
        learned = assignments(
            generator, z, config, config.temperature_end, "hard",
        ).squeeze(1).sum(-1)
    gold = cyclic_assignment(config, device).sum(-1)
    result = {
        "config": config.to_dict(),
        "variant": args.variant,
        "history": history,
        "active_iou": [_active_iou(item, gold) for item in learned],
        "generator_parameters": sum(parameter.numel() for parameter in generator.parameters()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    torch.save({
        "config": config.to_dict(),
        "variant": args.variant,
        "generator_state": {name: value.cpu() for name, value in generator.state_dict().items()},
        "z": z.cpu(),
    }, args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
