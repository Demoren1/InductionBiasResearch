"""Summarize paired direct-U versus fixed-random-U MNIST8m meta runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARMS = ("learned", "random")
SEEDS = (42, 43, 44)
LENGTHS = (1, 3, 5, 10, 20)
HORIZONS = (0, 5, 20)


def load_runs(root: Path) -> dict[tuple[str, int], dict]:
    runs = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = root / f"{arm}_seed{seed}.json"
            data = json.loads(path.read_text())
            if "test" not in data or data["model"]["rank"] != 16:
                raise ValueError(f"Incomplete or unexpected result: {path}")
            if data["history"][-1]["step"] != 10000:
                raise ValueError(f"Expected 10000 steps: {path}")
            runs[(arm, seed)] = data
    for seed in SEEDS:
        learned = runs[("learned", seed)]["config"]
        random = runs[("random", seed)]["config"]
        for key in ("seed", "inner_steps", "inner_lr", "outer_lr",
                    "tasks_per_step", "support", "query", "test_tasks",
                    "train_images_per_digit", "eval_images_per_digit"):
            if learned[key] != random[key]:
                raise ValueError(f"Mismatched {key} for seed {seed}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("deepsets_z/mnist8m/meta_u_results"))
    parser.add_argument("--out", type=Path,
                        default=Path("deepsets_z/mnist8m"))
    args = parser.parse_args()
    runs = load_runs(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"source": str(args.root), "seeds": list(SEEDS),
               "test_tasks_per_seed": runs[("learned", 42)]["config"]["test_tasks"],
               "test": {}, "per_seed": {}}
    for length in LENGTHS:
        summary["test"][str(length)] = {}
        for horizon in HORIZONS:
            result = {}
            for arm in ARMS:
                values = np.array([runs[(arm, seed)]["test"][str(length)][
                    str(horizon)]["mae"] for seed in SEEDS])
                result[arm] = {"mean_mae": float(values.mean()),
                               "sd_across_seeds": float(values.std(ddof=1)),
                               "by_seed": values.tolist()}
            delta = (np.array(result["learned"]["by_seed"]) -
                     np.array(result["random"]["by_seed"]))
            result["learned_minus_random_mae"] = {
                "mean": float(delta.mean()), "by_seed": delta.tolist()}
            summary["test"][str(length)][str(horizon)] = result
    for seed in SEEDS:
        summary["per_seed"][str(seed)] = {
            arm: {"best_step": runs[(arm, seed)]["best_step"],
                  "best_validation_score": runs[(arm, seed)][
                      "best_validation_score"],
                  "elapsed_seconds": runs[(arm, seed)]["history"][-1][
                      "elapsed_seconds"]}
            for arm in ARMS}
    (args.out / "meta_u_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
    colors = {"learned": "#1769aa", "random": "#dc6b23"}
    for axis, seed in zip(axes, SEEDS):
        for arm in ARMS:
            history = runs[(arm, seed)]["history"]
            axis.plot([row["step"] for row in history],
                      [row["validation_score"] for row in history],
                      color=colors[arm], label=arm, linewidth=1.7)
        axis.set_title(f"seed {seed}")
        axis.set_xlabel("Meta-training step")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Validation MSE / set length ↓")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "meta_u_convergence.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=False)
    for axis, length in zip(axes, (5, 10, 20)):
        for arm in ARMS:
            per_seed = np.array([[runs[(arm, seed)]["test"][str(length)][
                str(horizon)]["mae"] for horizon in HORIZONS]
                for seed in SEEDS])
            for curve in per_seed:
                axis.plot(HORIZONS, curve, color=colors[arm], alpha=0.2,
                          linewidth=0.8)
            axis.plot(HORIZONS, per_seed.mean(axis=0), color=colors[arm],
                      marker="o", label=arm, linewidth=2)
        axis.set_title(f"Set length {length}")
        axis.set_xlabel("Adaptation steps for v")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Test MAE ↓")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "meta_u_adaptation.png", dpi=180)
    plt.close(fig)
    print(json.dumps({length: summary["test"][length]["5"]
                      for length in ("5", "10", "20")}, indent=2))


if __name__ == "__main__":
    main()
