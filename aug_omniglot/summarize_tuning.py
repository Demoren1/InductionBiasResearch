"""Summarize the validation-only hyperparameter search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .run import OUTPUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT / "tuning")
    args = parser.parse_args()
    runs = {}
    for folder in sorted(args.root.iterdir()):
        if folder.name.startswith("smoke") or not folder.is_dir() or not (folder / "history.jsonl").exists():
            continue
        history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
        if not history:
            continue
        config = json.loads((folder / "config.json").read_text())["config"]
        result = json.loads((folder / "result.json").read_text()) if (folder / "result.json").exists() else None
        runs[folder.name] = {"config": config, "history": history, "result": result}
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), constrained_layout=True)
    for name, item in runs.items():
        group = {"direct": 0, "generated": 1, "identity": 2, "random": 3}[item["config"]["kind"]]
        steps = [row["step"] for row in item["history"]]
        axes[0, group].plot(steps, [100 * row["val"]["accuracy"] for row in item["history"]], label=name)
        axes[1, group].plot(steps, [row["val"]["loss"] for row in item["history"]], label=name)
    for group, kind in enumerate(("Direct U", "Generated U", "Fixed identity U", "Fixed random U")):
        axes[0, group].set(title=f"{kind}: validation accuracy", xlabel="Outer step", ylabel="Accuracy, %")
        axes[1, group].set(title=f"{kind}: validation cross-entropy", xlabel="Outer step", ylabel="Cross-entropy")
        for row in range(2):
            axes[row, group].grid(alpha=.2)
            axes[row, group].legend(fontsize=7, frameon=False)
    figure = args.root / "validation_curves.png"
    fig.savefig(figure, dpi=160)
    ranking = sorted(
        ((name, item["result"]["validation"]["accuracy"], item["result"]["validation"]["loss"], item["result"]["best_step"])
         for name, item in runs.items() if item["result"] is not None),
        key=lambda row: row[1], reverse=True,
    )
    payload = {"ranking": [{"name": name, "validation_accuracy": acc, "validation_loss": loss, "best_step": step} for name, acc, loss, step in ranking]}
    (args.root / "screen_ranking.json").write_text(json.dumps(payload, indent=2))
    for name, accuracy, loss, step in ranking:
        print(f"{name:32s} accuracy={accuracy:.4f} loss={loss:.4f} step={step}")
    print(f"Figure: {figure}")


if __name__ == "__main__":
    main()
