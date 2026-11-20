"""Render development learning curves for selected classification methods."""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .classification import DATASETS, SETTINGS, key_for
from .groups import ROOT

METHODS = (
    ("partial", "Согласованные группы", "#006e57"),
    ("ward", "Ward aSRR", "#335ca7"),
    ("dense", "Свободные веса", "#404449"),
    ("temporal", "Общий временной фильтр", "#b36e13"),
)


def plot(dataset, *, root=ROOT, methods=METHODS):
    out = root / dataset
    curves = json.loads((out / "dev_curves.json").read_text())
    choices = json.loads((out / "choices.json").read_text())
    steps = [0, *SETTINGS["steps"]]
    for shots, rows in curves.items():
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.3), layout="constrained")
        for row, kind in enumerate(("support", "query")):
            for method, label, color in methods:
                if method not in choices[shots]:
                    continue
                spec = choices[shots][method]
                name, rate, stop = spec["name"], spec["rate"], spec["step"]
                values = np.asarray(
                    [rows[key_for(name, rate, step, kind)] for step in steps]
                )
                for col, ax in enumerate(axes[row]):
                    y = values[:, col] * (100 if col else 1)
                    ax.plot(steps, y, label=label, color=color)
                    ax.plot(stop, y[steps.index(stop)], "o", ms=4, color=color)
            partition = "Опорные примеры" if row == 0 else "Валидация"
            for ax, metric in zip(
                axes[row], ("кросс-энтропия", "macro recall, %"), strict=True
            ):
                ax.set_title(f"{partition} · {metric}")
                ax.grid(alpha=0.2)
                if row == 1:
                    ax.set_xlabel("Обновления Adam")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
        title = "Aug-Omniglot" if dataset == "aug_omniglot" else "WISDM"
        fig.suptitle(f"{title} · {shots} примеров на класс")
        fig.savefig(out / f"learning_{shots}shot.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    plot(parser.parse_args().dataset)
