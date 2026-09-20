"""Plot the recorded latent-importance experiments for the Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    })


def condition_figure(stage1: dict, stage2: dict, output: Path) -> None:
    blue, orange = "#2563a6", "#dc6b28"
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6), layout="constrained")
    ax = axes[0]
    seeds = stage1["protocol"]["seeds"]
    rows = [row for row in stage1["rows"] if row["steps"] == 50]
    for length, offset, marker, color in ((5, -0.12, "o", blue), (7, 0.12, "s", orange)):
        values = [next(row["conditional_minus_unconditional_bce"] for row in rows
                       if row["seed"] == seed and row["length"] == length)
                  for seed in seeds]
        ax.scatter(values, np.arange(len(seeds)) + offset, s=60, marker=marker,
                   color=color, label=f"цель {length}", zorder=3)
        ax.scatter([mean(values)], [len(seeds) + (offset * 2)], s=95,
                   marker="D", color=color, edgecolor="white", linewidth=0.8, zorder=4)
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_yticks([*range(len(seeds)), len(seeds)], [*[str(seed) for seed in seeds], "среднее"])
    ax.set_ylim(len(seeds) + 0.6, -0.55)
    ax.set_xlim(-0.016, 0.019)
    ax.set_xlabel("BCE: условная − постоянная")
    ax.set_title("A. Полная оценка, 4 seeds")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    line_colors = ["#9aabbb", "#6d91b4", "#3c78a5", "#1b4e79"]
    all_deltas = []
    for target, ax, color in ((5, axes[1], blue), (7, axes[2], orange)):
        by_seed = []
        for seed, line_color in zip(stage1["protocol"]["seeds"], line_colors):
            record = next(row for row in stage2["rows"]
                          if row["seed"] == seed and row["target_length"] == target)
            bce = record["condition_bce"]
            deltas = [bce[str(condition)] - bce[str(target)] for condition in range(3, 9)]
            by_seed.append(deltas)
            ax.plot(range(3, 9), deltas, "o-", color=line_color, linewidth=1.2,
                    markersize=3.5, alpha=0.65)
            all_deltas.extend(deltas)
        averaged = np.mean(by_seed, axis=0)
        ax.plot(range(3, 9), averaged, "o-", color=color, linewidth=2.8,
                markersize=5.5, label="среднее")
        ax.axhline(0, color="#444444", linewidth=1)
        ax.axvline(target, color=color, linestyle=":", linewidth=1)
        ax.set_xticks(range(3, 9))
        ax.set_xlabel("Поданная длина")
        ax.set_title(f"{'B' if target == 5 else 'C'}. Подстановки, цель {target}")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, loc="upper right", fontsize=9)
    margin = max(0.006, max(abs(value) for value in all_deltas) * 1.1)
    for ax in axes[1:]:
        ax.set_ylim(-margin, margin)
        ax.set_ylabel("BCE: подмена − верная длина")
    fig.suptitle("Зависимость от входной длины: знак и величина эффекта", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def latent_figure(stage3: dict, stage4: dict, output: Path) -> None:
    blue, orange, green, gray = "#2563a6", "#dc6b28", "#278569", "#637589"
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6), layout="constrained")
    rows = stage3["held_rows"]
    variants = [
        ("code_midpoint", "z: середина", blue),
        ("code_adapted", "z: адаптация", green),
        ("scalar_correct", "длина k", orange),
        ("scalar_adapted", "k: адаптация", gray),
    ]
    ax = axes[0]
    offsets = np.linspace(-0.11, 0.11, len(rows))
    for index, (variant, label, color) in enumerate(variants):
        values = [row["variants"][variant]["bce"] -
                  row["variants"]["scalar_unconditional"]["bce"] for row in rows]
        ax.scatter(index + offsets, values, s=48, color=color, alpha=0.72)
        ax.scatter(index, mean(values), marker="D", s=100, color=color,
                   edgecolor="white", linewidth=0.8, zorder=4)
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_xticks(range(len(variants)), [item[1] for item in variants], rotation=25, ha="right")
    ax.set_ylabel("ΔBCE к постоянному входу")
    ax.set_title("A. Латент против постоянной U")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    for index, row in enumerate(rows):
        value = row["adapted_code_minus_midpoint_bce"]
        ax.scatter(index, value, s=90, color=green if value < 0 else orange,
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(f"{value:+.4f}", (index, value), xytext=(0, 8 if value >= 0 else -15),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_xticks(range(len(rows)), [f"{row['seed']}, k={row['length']}" for row in rows],
                  rotation=25, ha="right")
    ax.set_ylabel("BCE: адаптация z − середина")
    ax.set_ylim(-0.0016, 0.0016)
    ax.set_title("B. Польза подбора нового z")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[2]
    cases = [
        (False, False, "одинаковые\nодин код", blue),
        (False, True, "одинаковые\nдва кода", green),
        (True, False, "ортогональные\nодин код", orange),
        (True, True, "ортогональные\nдва кода", green),
    ]
    for index, (varying, variable, label, color) in enumerate(cases):
        values = [row["test_excess_mse"] for row in stage4["rows"]
                  if row["varying_truth"] == varying and row["variable_code"] == variable]
        ax.scatter(index + np.linspace(-0.12, 0.12, len(values)), values,
                   s=40, color=color, alpha=0.7)
        ax.scatter(index, median(values), marker="D", s=100, color=color,
                   edgecolor="white", linewidth=0.8, zorder=4)
    ax.axhline(2, linestyle="--", color="#555555", linewidth=1, label="теория: MSE = 2")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 8)
    ax.set_xticks(range(len(cases)), [item[2] for item in cases], fontsize=8)
    ax.set_ylabel("Избыточный test MSE (лог. шкала)")
    ax.set_title("C. Когда менять U необходимо")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle("Роль латента: короткий тест и контроль с известным ответом", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("mds/data/2026-09-19"))
    parser.add_argument("--out-dir", type=Path, default=Path("mds/assets/2026-09-19"))
    args = parser.parse_args()
    style()
    stage = {index: read(args.data_dir / f"latent_importance_stage{index}.json")
             for index in (1, 2, 3, 4)}
    condition_figure(stage[1], stage[2], args.out_dir / "latent_importance_condition.png")
    latent_figure(stage[3], stage[4], args.out_dir / "latent_importance_latent_control.png")


if __name__ == "__main__":
    main()
