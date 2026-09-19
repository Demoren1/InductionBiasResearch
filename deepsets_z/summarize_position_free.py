"""Summarize the position-free digit-sum comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .run import ARMS, TEST_LENGTHS


LABELS = {
    "learned_z": "Learned z (16)",
    "fixed_z": "Fixed z (16)",
    "no_z_one": "No z (1)",
    "no_z_16": "No z (16)",
    "oracle": "Identity U",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path("deepsets_z/results_position_free"))
    parser.add_argument("--output", type=Path,
                        default=Path("deepsets_z/summary_position_free.json"))
    args = parser.parse_args()
    config = json.loads((args.results / "config.json").read_text())
    summary = {"config": config, "arms": {}}
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for arm in ARMS:
        runs = [json.loads((args.results / f"{arm}_seed{seed}.json").read_text())
                for seed in config["seeds"]]
        data = {}
        for length in TEST_LENGTHS:
            metrics = [run["metrics"][f"test_{length}"] for run in runs]
            data[str(length)] = {
                name: {"mean": float(np.mean(values)),
                       "std": float(np.std(values, ddof=1)),
                       "per_seed": values}
                for name in ("mae", "rmse", "exact_round_accuracy")
                for values in [[float(metric[name]) for metric in metrics]]
            }
        summary["arms"][arm] = {
            "training_seconds_mean": float(np.mean([run["training_seconds"] for run in runs])),
            "lengths": data,
        }
        ax.plot(TEST_LENGTHS, [data[str(n)]["mae"]["mean"] for n in TEST_LENGTHS],
                marker="o", linewidth=1.7, label=LABELS[arm])
    ax.set(xlabel="Number of digits", ylabel="Mean absolute error of sum",
           xticks=TEST_LENGTHS, title="One shared U for all set elements")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    plt.close(fig)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    for arm in ARMS:
        row = summary["arms"][arm]["lengths"]["100"]
        print(f"{arm:12s} MAE={row['mae']['mean']:.6f} "
              f"exact={row['exact_round_accuracy']['mean']:.1%}")


if __name__ == "__main__":
    main()
