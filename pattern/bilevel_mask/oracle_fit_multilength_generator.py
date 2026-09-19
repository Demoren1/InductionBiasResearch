"""Test whether the coordinate generator can represent the analytic cyclic U."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .generated_sharing import SharingGenerator, _project, assignments
from .multilength_sharing import MultiLengthConfig, cyclic_assignment


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()
    device = torch.device(args.device)
    config = MultiLengthConfig(seed=args.seed, generator_width=args.width)
    generator = SharingGenerator(config).to(device)  # type: ignore[arg-type]
    rng = torch.Generator(device="cpu").manual_seed(args.seed + 104729)
    z = torch.nn.Parameter(torch.randn(1, 1, config.latent_dim, generator=rng).to(device))
    _project(z, config.z_radius)
    optimizer = torch.optim.Adam([z, *generator.parameters()], lr=args.lr)
    target = cyclic_assignment(config, device)
    target_active = target.sum(-1)
    target_category = target.argmax(-1)
    history = []
    for step in range(1, args.steps + 1):
        logits = generator(z).squeeze(0).squeeze(0)
        active_loss = F.binary_cross_entropy_with_logits(logits[..., 0], target_active)
        category_loss = F.cross_entropy(
            logits[..., 1:][target_active.bool()], target_category[target_active.bool()],
        )
        loss = active_loss + category_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _project(z, config.z_radius)
        if step == 1 or step % 100 == 0 or step == args.steps:
            with torch.no_grad():
                hard = assignments(generator, z, config, 0.1, "hard").squeeze(0).squeeze(0)  # type: ignore[arg-type]
                active_iou = float(
                    ((hard.sum(-1).bool() & target_active.bool()).sum()) /
                    ((hard.sum(-1).bool() | target_active.bool()).sum())
                )
                assignment_accuracy = float((hard == target).float().mean())
                exact = bool(torch.equal(hard, target))
            history.append({
                "step": step, "loss": float(loss.detach()), "active_iou": active_iou,
                "assignment_accuracy": assignment_accuracy, "exact": exact,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "seed": args.seed, "width": args.width, "steps": args.steps, "lr": args.lr,
        "parameters": sum(parameter.numel() for parameter in generator.parameters()),
        "final": history[-1], "history": history,
    }, indent=2))


if __name__ == "__main__":
    main()
