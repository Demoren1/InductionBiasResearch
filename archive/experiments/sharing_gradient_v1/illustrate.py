"""Actual dense Adam control on the first frozen target, for weight heatmaps."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads((args.root / "protocol.json").read_text())
    folder = args.root / "hidden" / f"family_{plan['confirmation_families'][0]}"
    with np.load(folder / "target_after_freeze.npz") as archive:
        x = torch.tensor(archive["support_x"][0, :4], device=args.device)
        y = torch.tensor(archive["support_y"][0, :4], device=args.device)
        truth = torch.tensor(
            archive["teacher_weights_audit_only"][0], device=args.device
        )
    weights = torch.zeros_like(truth, requires_grad=True)
    optimizer = torch.optim.Adam([weights], lr=0.03)
    for _ in range(2000):
        loss = (x @ weights.T - y).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train = float((x @ weights.T - y).square().mean())
        population = float((weights - truth).square().sum() / weights.shape[0])
    path = folder / "audit" / "dense_adam.npz"
    np.savez_compressed(path, weights_support_4=weights.detach().cpu().numpy())
    record = {
        "steps": 2000,
        "lr": 0.03,
        "initialization": "zero",
        "target_task": 0,
        "support": 4,
        "support_training_mse": train,
        "population_mse": population,
        "models_frozen_sha256": digest(args.root / "models_frozen.json"),
        "target_sha256": digest(folder / "target_after_freeze.npz"),
        "scope": "Single fixed Adam illustration, no query or test selection, not an averaged competitor result.",
    }
    save_json(path.with_suffix(".json"), record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
