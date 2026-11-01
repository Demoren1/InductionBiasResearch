"""Post-fit audit: compare integer indexators up to irrelevant label permutations.

This module never trains models or selects their hyperparameters. Hidden teacher
coordinates are used only to align the finished structures for inspection.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.optimize import linear_sum_assignment

from experiments.dynamic_signal_v1.report import LABELS, best_rows, path_for


def align(indices, oracle):
    support, truth = indices != 0, oracle != 0
    paired = support[:, ::2].astype(float) + support[:, 1::2]
    target = truth[:, ::2].astype(float) + truth[:, 1::2]
    old, new = linear_sum_assignment(-(paired.T @ target))
    order = np.empty(32, int)
    order[2 * new], order[2 * new + 1] = 2 * old, 2 * old + 1
    value = indices[:, order]
    roles = np.where(value > 0, (value - 1) % 8 + 1, 0)
    true_roles = np.where(oracle > 0, (oracle - 1) % 8 + 1, 0)
    overlap = (roles > 0) & (true_roles > 0)
    counts = np.zeros((8, 8), int)
    np.add.at(counts, (roles[overlap] - 1, true_roles[overlap] - 1), 1)
    old, new = linear_sum_assignment(-counts)
    mapping = np.zeros(9, int)
    mapping[old + 1] = new + 1
    value = np.where(value > 0, mapping[roles] + 8 * ((value - 1) // 8), 0)
    return value


def main():
    plan = json.loads(
        Path("experiments/dynamic_signal_v1/protocol_confirmation.json").read_text()
    )
    raw = Path(plan["run"])
    root = (
        raw
        if (raw / "target_fit").exists()
        else Path("experiments/dynamic_signal_v1/results/v2")
    )
    permutation = np.load(root / "target/teacher_audit.npz")["permutation"]
    oracle = np.zeros((128, 32), int)
    for group in range(16):
        coords = permutation[group * 8 : (group + 1) * 8]
        oracle[coords, 2 * group] = np.arange(1, 9)
        oracle[coords, 2 * group + 1] = np.arange(9, 17)
    results, example = [], {"oracle": oracle}
    for method in ["ar", "frequency", "consensus", "random_topology"]:
        arrays = np.load(path_for(root, method, 1, 2048))
        rows = best_rows(arrays)
        for task, row in enumerate(rows):
            indices = align(arrays["indices"][row], oracle)
            support, truth = indices != 0, oracle != 0
            intersection = (support & truth).sum()
            results.append(
                {
                    "method": method,
                    "task": task,
                    "support_precision": intersection / max(1, support.sum()),
                    "support_recall": intersection / truth.sum(),
                    "active_index_accuracy": ((indices == oracle) & truth).sum()
                    / truth.sum(),
                }
            )
            if task == 0:
                example[method] = indices
    output = Path("experiments/dynamic_signal_v1/results/v2")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "indexator_recovery.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    colors = ListedColormap(["white", *list(plt.get_cmap("tab20").colors[:16])])
    fig, axes = plt.subplots(1, 5, figsize=(14, 6), sharey=True)
    for ax, method in zip(axes, example):
        im = ax.imshow(
            example[method][permutation],
            aspect="auto",
            interpolation="nearest",
            cmap=colors,
            vmin=-0.5,
            vmax=16.5,
        )
        ax.set_title(LABELS[method], fontsize=10)
        ax.set_xlabel("Скрытый нейрон")
    axes[0].set_ylabel("Вход (истинный порядок только для аудита)")
    fig.subplots_adjust(wspace=0.15, right=0.93, top=0.9)
    bar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        fraction=0.02,
        pad=0.02,
        ticks=[0, 1, 4, 8, 9, 12, 16],
    )
    bar.set_label("Индекс кодбука; 0 — нет связи")
    fig.suptitle("Полные индексаторы: одинаковый цвет означает общий коэффициент")
    fig.savefig(
        "experiments/dynamic_signal_v1/figures/full_indexators.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    fig = plt.figure(figsize=(17, 6))
    grid = fig.add_gridspec(
        1,
        8,
        width_ratios=[1] * 7 + [0.07],
        left=0.05,
        right=0.97,
        bottom=0.12,
        top=0.88,
        wspace=0.2,
    )
    for number, method in enumerate(
        ["ar", "frequency", "mask_unshared", "transfer", "adam", "sgd", "wsr"]
    ):
        ax = fig.add_subplot(grid[0, number])
        arrays = np.load(path_for(root, method, 1, 2048))
        row = best_rows(arrays)[0]
        weights = (
            arrays["parameter_0"][row]
            if "parameter_0" in arrays
            else arrays["selected_parameter_0"][0]
        )
        magnitude = np.abs(weights)
        score = magnitude.T @ (oracle != 0).astype(float)
        score /= np.maximum(magnitude.sum(0)[:, None], 1e-9)
        old, new = linear_sum_assignment(-score)
        order = np.empty(32, int)
        order[new] = old
        im = ax.imshow(
            magnitude[permutation][:, order] / max(magnitude.max(), 1e-9),
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="magma",
            interpolation="nearest",
        )
        ax.set_title(LABELS[method], fontsize=9)
        ax.set_xlabel("Нейрон")
        if number == 0:
            ax.set_ylabel("Вход (истинный порядок только для рисунка)")
        else:
            ax.set_yticks([])
    fig.colorbar(im, cax=fig.add_subplot(grid[0, 7]))
    fig.suptitle("Найденные веса первого слоя: |W| / max|W| каждой модели")
    fig.savefig("experiments/dynamic_signal_v1/figures/weights.png", dpi=180)
    plt.close(fig)
    for method in example:
        values = [row for row in results if row["method"] == method]
        if values:
            print(
                method,
                {
                    key: float(np.mean([row[key] for row in values]))
                    for key in [
                        "support_precision",
                        "support_recall",
                        "active_index_accuracy",
                    ]
                },
            )


if __name__ == "__main__":
    main()
