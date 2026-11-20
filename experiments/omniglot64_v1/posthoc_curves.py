"""Plot support and held-out development curves for the selected models."""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .posthoc_source import PLAN, ROOT, WIDTHS


def plot_groups():
    fig, axes = plt.subplots(2, 2, figsize=(5.8, 5.8), layout="constrained")
    perm = np.random.default_rng(390001).permutation(784)
    with np.load(ROOT / "bank.npz") as bank:
        for row, width in enumerate(WIDTHS):
            ids = bank[f"w{width}_b4"][0]
            count = int(ids.max()) + 1
            colors = plt.cm.turbo(np.linspace(0.02, 0.98, count))[
                np.random.default_rng(width).permutation(count)
            ]
            for col, assigned in enumerate((ids, ids[perm])):
                ax = axes[row, col]
                ax.imshow(colors[assigned.reshape(28, 28)], interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if not col:
                    ax.set_ylabel(f"{count} групп")
                if not row:
                    ax.set_title(
                        ("Группы aSRR", "Те же группы после\nперестановки пикселей")[
                            col
                        ],
                        fontsize=11,
                    )
    fig.savefig(ROOT / "pixel_groups.png", dpi=170)
    plt.close(fig)


def plot():
    teacher = json.loads((ROOT / "teacher_curves.json").read_text())
    policies = json.loads((ROOT / "policy_curves.json").read_text())
    choices = json.loads((ROOT / "policy_frozen.json").read_text())["choices"]
    transfer_folder = ROOT / "source_dense"
    transfer_curves = json.loads((transfer_folder / "curves.json").read_text())
    transfer_choices = json.loads((transfer_folder / "policy_frozen.json").read_text())[
        "choices"
    ]
    columns = [
        ("train_ce", "Support · cross-entropy"),
        ("ce", "Validation · cross-entropy"),
        ("train_recall", "Support · macro recall, %"),
        ("accuracy", "Validation · macro recall, %"),
    ]
    methods = [
        ("teacher", "Dense 160", "#5e646c"),
        ("narrow", "Dense, с нуля", "#2a60a4"),
        ("prune", "Dense, из учителя", "#bd681b"),
        ("transfer", "Dense, с исходных задач", "#7545bd"),
        ("asrr", "aSRR, сжатие", "#007459"),
        ("fresh", "aSRR, с нуля", "#973e70"),
    ]
    for shot in map(str, PLAN["target_shots"]):
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.7), layout="constrained")
        for row, width in enumerate(WIDTHS):
            plan = choices[shot][str(width)]
            for key, label, color in methods:
                if key == "teacher":
                    spec, data = choices[shot]["teacher"], teacher[shot]
                elif key == "transfer":
                    spec, data = (
                        transfer_choices[shot][str(width)],
                        transfer_curves[shot],
                    )
                else:
                    spec, data = plan[key], policies[shot]
                selected = int(spec.rsplit(".", 1)[1])
                prefix = spec.rsplit(".", 1)[0] + "."
                points = sorted(
                    (int(k.rsplit(".", 1)[1]), v)
                    for k, v in data.items()
                    if k.startswith(prefix)
                )
                for (metric, title), ax in zip(columns, axes[row]):
                    factor = 100 if metric in ("accuracy", "train_recall") else 1
                    xx = [s for s, _ in points]
                    yy = [max(1e-6, factor * v[metric]) for _, v in points]
                    ax.plot(xx, yy, color=color, lw=1.65, alpha=0.9, label=label)
                    ax.plot(selected, yy[xx.index(selected)], "o", color=color, ms=4)
                    ax.set_xscale("symlog", linthresh=25)
                    ax.set_xticks(
                        [0, 25, 100, 300, 600], ["0", "25", "100", "300", "600"]
                    )
                    ax.grid(True, alpha=0.18)
                    if row == 0:
                        ax.set_title(title, fontsize=11)
                    if row == 1:
                        ax.set_xlabel("Обновления Adam")
                    if metric == "train_ce":
                        ax.set_yscale("log")
                        ax.set_ylabel(
                            f"{width * 790 + 5:,} параметров".replace(",", " ")
                        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="outside lower center", ncol=len(labels), frameon=False
        )
        fig.suptitle(
            f"Omniglot · 5 классов · {shot} примеров на класс\n"
            "Точками показаны режимы, выбранные по валидации",
            fontsize=14,
        )
        fig.savefig(ROOT / f"learning_curves_{shot}shot.png", dpi=155)
        plt.close(fig)


if __name__ == "__main__":
    plot()
    plot_groups()
