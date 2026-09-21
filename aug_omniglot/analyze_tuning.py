"""Plot selected tuned models and paired accuracy differences on held-out tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
LABELS = {
    "generated": "Генератор U",
    "direct": "Прямая U",
    "random": "Случайная U",
    "identity": "Единичная U",
}
COLORS = {
    "generated": "#176692",
    "direct": "#ba5a3b",
    "random": "#8558a5",
    "identity": "#3d454c",
}


def paired_interval(first: np.ndarray, second: np.ndarray, rng: np.random.Generator) -> dict:
    differences = 100 * (first - second)
    samples = rng.choice(differences, size=(5000, len(differences)), replace=True).mean(axis=1)
    return {
        "mean_percentage_points": float(differences.mean()),
        "ci95_percentage_points": [float(x) for x in np.quantile(samples, [.025, .975])],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    args = parser.parse_args()
    test = json.loads(args.test_json.read_text())
    if test["split"] != "test":
        raise ValueError("Expected held-out test episodes")
    folders = {kind: getattr(args, kind) for kind in LABELS}
    arms, arrays = {}, {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for kind, folder in folders.items():
        name = folder.name
        row = test["results"][name]
        config = json.loads((folder / "config.json").read_text())
        history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
        steps = np.array([point["step"] for point in history])
        task_count = steps * config["config"]["meta_batch"]
        axes[0].plot(task_count / 1000, [100 * point["val"]["accuracy"] for point in history],
                     label=LABELS[kind], color=COLORS[kind], linewidth=1.8)
        array = np.asarray(row["accuracy_by_task"])
        arrays[kind] = array
        arms[kind] = {
            "name": name,
            "best_step": row["best_training_step"],
            "training_tasks_at_best": row["best_training_step"] * config["config"]["meta_batch"],
            "parameters_total": config["parameters"],
            "parameters_u": json.loads((folder / "result.json").read_text())["parameters_u"],
            "test_accuracy": row["confirm_accuracy"],
            "test_accuracy_se": row["confirm_accuracy_se"],
            "test_loss": row["confirm_loss"],
        }
    order = list(LABELS)
    positions = np.arange(len(order))
    axes[1].bar(positions, [100 * arms[kind]["test_accuracy"] for kind in order],
                yerr=[1.96 * 100 * arms[kind]["test_accuracy_se"] for kind in order],
                color=[COLORS[kind] for kind in order], capsize=4)
    axes[1].set_xticks(positions, [LABELS[kind].replace(" U", "") for kind in order], rotation=15)
    rng = np.random.default_rng(12345)
    comparisons = {
        kind: paired_interval(arrays["generated"], arrays[kind], rng)
        for kind in ("direct", "random", "identity")
    }
    for position, kind in enumerate(("direct", "random", "identity")):
        comparison = comparisons[kind]
        mean = comparison["mean_percentage_points"]
        low, high = comparison["ci95_percentage_points"]
        axes[2].errorbar(mean, position, xerr=[[mean - low], [high - mean]],
                         fmt="o", color=COLORS[kind], capsize=4, markersize=7)
    axes[2].set_yticks(range(3), [f"против {LABELS[kind]}" for kind in ("direct", "random", "identity")])
    axes[2].axvline(0, color="black", linewidth=1, alpha=.6)
    axes[0].set(title="Проверка во время обучения", xlabel="Обучающих задач, тыс.", ylabel="Точность, %")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set(title=f"Тест: {test['tasks']} задач", ylabel="Точность, %")
    axes[2].set(title="Разница: генератор минус контроль", xlabel="Процентные пункты (95% парный интервал)")
    for axis in axes:
        axis.grid(alpha=.2)
    figure = ROOT / "figures" / "tuned_comparison.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180)
    summary = {
        "seed": 42,
        "test_tasks": test["tasks"],
        "test_episode_offset": test["offset"],
        "arms": arms,
        "generated_minus": comparisons,
    }
    result = ROOT / "results" / "tuned_comparison.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Figure: {figure}")


if __name__ == "__main__":
    main()
