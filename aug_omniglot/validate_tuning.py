"""Evaluate saved tuning candidates on a fresh, fixed validation episode set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .run import MetaConv, augment_queries, episode, load_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument("--offset", type=int, default=10_000)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    data, classes, _ = load_data(device)
    results = {}
    for folder in args.folders:
        config = json.loads((folder / "config.json").read_text())["config"]
        model = MetaConv(
            config["kind"], config["ways"], config.get("generator_width", 64),
            config.get("generator_scale", .1), config.get("inner_lr_mode", "clamp"),
            config.get("inner_init", .4),
        ).to(device)
        checkpoint = torch.load(folder / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        u = {key: value.detach() for key, value in model.u.matrices().items()}
        accuracy, loss = [], []
        for index in range(args.tasks):
            seed = config["seed"] * 100_000_000 + (1_000_000 if args.split == "val" else 2_000_000) + args.offset + index
            sx, sy, qx, qy = episode(data["images"], classes[args.split], config["ways"], config["shots"], config["queries"], seed)
            qx = augment_queries(qx, seed + 9187)
            v = model.adapt(sx, sy, u, config["eval_inner_steps"], second_order=False)
            with torch.no_grad():
                logits = model(qx, v, u)
                accuracy.append((logits.argmax(-1) == qy).float().mean().item())
                loss.append(F.cross_entropy(logits, qy).item())
        name = folder.name
        results[name] = {
            "best_training_step": checkpoint["step"],
            "selection_accuracy": checkpoint["val"]["accuracy"],
            "confirm_accuracy": float(np.mean(accuracy)),
            "confirm_accuracy_se": float(np.std(accuracy, ddof=1) / math.sqrt(args.tasks)),
            "confirm_loss": float(np.mean(loss)),
            "accuracy_by_task": accuracy,
        }
        print(name, json.dumps({key: value for key, value in results[name].items() if key != "accuracy_by_task"}), flush=True)
    payload = {"split": args.split, "tasks": args.tasks, "offset": args.offset, "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload))


if __name__ == "__main__":
    main()
