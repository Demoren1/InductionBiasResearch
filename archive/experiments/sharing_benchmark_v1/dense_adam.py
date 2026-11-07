"""Actual Adam weights for the requested numerical-reference heatmaps."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    for profile, family in (("hidden", 201), ("native", 301)):
        folder = args.root / profile / f"family_{family}"
        with np.load(folder / "target_after_freeze.npz") as archive:
            x = torch.tensor(archive["support_x"][0, :4], device="cuda:0")
            y = torch.tensor(archive["support_y"][0, :4], device="cuda:0")
        weights = torch.zeros(
            y.shape[-1],
            x.shape[-1],
            dtype=x.dtype,
            device=x.device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([weights], lr=0.03)
        for _ in range(2000):
            loss = (x @ weights.T - y).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            residual = float((x @ weights.T - y).square().mean())
        path = folder / "audit/dense_adam_seed_0.npz"
        np.savez_compressed(path, weights_support_4=weights.detach().cpu().numpy())
        save_json(
            path.with_suffix(".json"),
            {
                "lr": 0.03,
                "steps": 2000,
                "initialization": "zero",
                "support_size": 4,
                "target_task": 0,
                "support_training_mse": residual,
                "benchmark_frozen_sha256": digest(args.root / "benchmark_frozen.json"),
                "target_sha256": digest(folder / "target_after_freeze.npz"),
                "scope": "Requested actual-Adam heatmap for the first new task. Fixed numerical settings, no query/test selection. Added after the main audit as an illustration; primary competitor table retains exact LS/ridge controls.",
                "code_sha256": digest(Path(__file__)),
            },
        )
        print(json.dumps({"profile": profile, "support_training_mse": residual}))


if __name__ == "__main__":
    main()
