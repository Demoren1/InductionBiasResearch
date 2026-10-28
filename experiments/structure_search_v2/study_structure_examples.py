"""Draw tasks and actual saved indexators; never fit or select by test error."""

import json
import textwrap
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from distractor_data import image_table
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.optimize import linear_sum_assignment
from study_results import ASSETS, LABELS, RUNS
from torchvision.datasets import MNIST, FashionMNIST

ROOT = Path(__file__).resolve().parent
MAIN_METHODS = [
    "frequency",
    "functional",
    "vae_sample",
    "vae_z",
    "random",
    "untrained_vae_z",
    "dense_adam",
    "dense_sgd",
    "pretrained_adam",
]
DENSE_METHODS = {
    "dense_adam",
    "dense_sgd",
    "pretrained_adam",
    "frozen_pretrained",
    "dense_adam_sweep",
    "dense_sgd_sweep",
    "pretrained_adam_sweep",
    "frozen_pretrained_sweep",
    "dense_transfer",
}


def aligned_permutation(found, reference):
    """Rename groups and shared roles for a diagnostic comparison only."""
    inverse = np.argsort(reference)
    groups = found.reshape(8, 8)
    true_positions = inverse[groups]
    counts = np.array([np.bincount(row // 8, minlength=8) for row in true_positions])
    old_group, new_group = linear_sum_assignment(-counts)
    group_map = np.empty(8, dtype=int)
    group_map[old_group] = new_group
    role_counts = np.zeros((8, 8), dtype=int)
    for group in range(8):
        for role in range(8):
            position = true_positions[group, role]
            if position // 8 == group_map[group]:
                role_counts[role, position % 8] += 1
    old_role, new_role = linear_sum_assignment(-role_counts)
    role_map = np.empty(8, dtype=int)
    role_map[old_role] = new_role
    canonical = np.empty((8, 8), dtype=int)
    for group in range(8):
        canonical[group_map[group], role_map] = groups[group]
    return canonical.flatten(), int(sum(counts[g, group_map[g]] for g in range(8)))


def index_matrix(permutation):
    indexator = np.zeros((64, 32), dtype=int)
    for group in range(8):
        for role in range(8):
            for feature in range(4):
                indexator[permutation[8 * group + role], 4 * group + feature] = (
                    8 * feature + role + 1
                )
    return indexator


def synthetic_examples(records, arrays):
    source = RUNS / "scrambled_pattern_source_pilot_v1"
    out = RUNS / "scrambled_transfer_v1"
    teacher = np.load(source / "hidden_teacher.npz")
    reference = teacher["permutation"]
    oracle = index_matrix(reference)
    arrays["synthetic_analytic_I"] = oracle
    records["synthetic"] = {
        "task": 0,
        "selection": "minimum query NMSE among first task's eight starts",
        "analytic_groups_one_based": (reference.reshape(8, 8) + 1).tolist(),
        "methods": {},
    }
    colors = ["white"]
    for color in ["Blues", "Oranges", "Greens", "Purples"]:
        colors.extend(plt.get_cmap(color)(np.linspace(0.35, 0.95, 8)))
    cmap = ListedColormap(colors)
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), layout="constrained")
    methods = ["analytic"] + json.loads((out / "calibration.json").read_text())[
        "methods"
    ]
    for axis, method in zip(axes.flat, methods):
        if method == "analytic":
            matrix, title = oracle[reference], "Аналитический индексатор I*"
            axis.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=32.5, aspect="auto")
            kind = "indices"
        else:
            data = np.load(out / (method + ".npz"))
            chosen = int(data["query_nmse"][:8].argmin())
            row = {
                "selected_state": chosen,
                "query_nmse": float(data["query_nmse"][chosen]),
                "test_nmse": float(data["test_nmse"][chosen]),
            }
            if method in DENSE_METHODS:
                weight = data["parameter_0"][chosen]
                scale = np.abs(weight).max()
                axis.imshow(
                    np.abs(weight[reference]) / scale,
                    cmap="magma",
                    vmin=0,
                    vmax=1,
                    aspect="auto",
                )
                arrays["synthetic_" + method + "_W"] = weight
                title = LABELS[method] + f"\n|W| / max|W|; NMSE {row['test_nmse']:.3g}"
                kind = "weights"
                row["representation"] = (
                    "dense numeric weights; no learned discrete indexator"
                )
            else:
                permutation = data["indices"][chosen]
                aligned, group_correct = aligned_permutation(permutation, reference)
                matrix = index_matrix(aligned)
                axis.imshow(
                    matrix[reference], cmap=cmap, vmin=-0.5, vmax=32.5, aspect="auto"
                )
                arrays["synthetic_" + method + "_I_display"] = matrix
                arrays["synthetic_" + method + "_permutation_raw"] = permutation
                row.update(
                    {
                        "raw_groups_one_based": (
                            permutation.reshape(8, 8) + 1
                        ).tolist(),
                        "display_groups_one_based": (
                            aligned.reshape(8, 8) + 1
                        ).tolist(),
                        "group_correct_coordinates": group_correct,
                        "group_and_role_correct_coordinates": int(
                            (aligned == reference).sum()
                        ),
                    }
                )
                title = (
                    LABELS[method]
                    + f"\nI; позиции после переименования: {(aligned == reference).sum()}/64"
                )
                kind = "indices"
            records["synthetic"]["methods"][method] = row
        axis.set_title(title, fontsize=10)
        axis.set_xticks([0, 7, 15, 23, 31], [1, 8, 16, 24, 32])
        axis.set_yticks([0, 15, 31, 47, 63], [1, 16, 32, 48, 64])
        axis.set_xlabel("32 скрытых нейрона", fontsize=8)
        axis.set_ylabel("Входы в истинном порядке π", fontsize=8)
        if kind == "indices":
            for boundary in range(8, 64, 8):
                axis.axhline(boundary - 0.5, color="gray", lw=0.4)
            for boundary in range(4, 32, 4):
                axis.axvline(boundary - 0.5, color="gray", lw=0.4)
    fig.suptitle(
        "Первая новая синтетическая задача: эталон и результаты всех 8 методов\nЦвета I: четыре фильтра, восемь оттенков — позиции; белый — индекс 0",
        fontsize=13,
    )
    fig.savefig(ASSETS / "synthetic_indexators.png", dpi=180)
    plt.close(fig)

    target = np.load(out / "target_teacher.npz")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        gridspec_kw={"width_ratios": [1.1, 1]},
        layout="constrained",
    )
    axis = axes[0]
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = [
        (0.84, "64 независимых входа"),
        (0.66, "Неизвестная перестановка π"),
        (0.48, "8 групп по 8 входов"),
        (
            0.28,
            "Каждая группа → те же 4 ReLU-фильтра\n32 числа кодбука повторяются 8 раз",
        ),
        (0.07, "Сумма 32 ответов / √32 → y"),
    ]
    for y, label in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (0.08, y),
                0.84,
                0.11,
                boxstyle="round,pad=0.015",
                facecolor="#edf5f7",
                edgecolor="#087f8c",
            )
        )
        axis.text(0.5, y + 0.055, label, ha="center", va="center", fontsize=11)
    for (y1, _), (y2, _) in pairwise(boxes):
        axis.add_patch(
            FancyArrowPatch(
                (0.5, y1 - 0.01),
                (0.5, y2 + 0.125),
                arrowstyle="-|>",
                mutation_scale=13,
                color="#455a64",
            )
        )
    axis.set_title("Что вычисляет синтетическая задача")
    grid = np.linspace(-4, 4, 121)
    xx, yy = np.meshgrid(grid, grid)
    c, b, a = target["codebook"][0], target["bias"][0], target["readout"][0]
    hidden = np.maximum(0, xx[..., None] * c[:, 0] + yy[..., None] * c[:, 1] + b)
    output = (hidden * a).sum(-1) / np.sqrt(32)
    plot = axes[1].pcolormesh(grid, grid, output, cmap="viridis", shading="auto")
    axes[1].set_xlabel(f"Вход x{int(reference[0]) + 1}")
    axes[1].set_ylabel(f"Вход x{int(reference[1]) + 1}")
    axes[1].set_title(
        "Точный ответ первой новой функции\nОстальные 62 входа равны нулю"
    )
    fig.colorbar(plot, ax=axes[1], label="Ответ y")
    fig.savefig(ASSETS / "synthetic_task.png", dpi=180)
    plt.close(fig)


def restored_strip(values, permutation):
    original = np.zeros(1568)
    original[permutation] = values
    return original.reshape(8, 14, 14).transpose(1, 0, 2).reshape(14, 112)


def image_examples(dataset_name, records, arrays):
    out = RUNS / ("distractor_" + dataset_name + "_v5")
    dataset_type = {"MNIST": MNIST, "FashionMNIST": FashionMNIST}[dataset_name]
    dataset = dataset_type(ROOT / "datasets", train=True)
    pixels = image_table(dataset).numpy()
    case = json.loads((out / "target_cases.json").read_text())[0]
    permutation = np.load(out / "hidden_layout.npz")["permutation"]
    informative = permutation < 196
    splits = np.load(out / "target_splits.npz")
    arrays[dataset_name + "_analytic_input_mask"] = informative
    records[dataset_name] = {
        "case_id": 0,
        "pair": case["pair"],
        "split_seed": case["seed"],
        "selection": "minimum query BCE among first case's four starts",
        "analytic_input_ids_one_based": (np.flatnonzero(informative) + 1).tolist(),
        "methods": {},
    }
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 5),
        gridspec_kw={"width_ratios": [4, 2, 1]},
        layout="constrained",
    )
    selected_ids = []
    for row, example in enumerate([0, 16]):
        ids = splits["support"][0, example]
        images = pixels[ids]
        original = images.reshape(8, 14, 14).transpose(1, 0, 2).reshape(14, 112)
        observed = images.flatten()[permutation]
        label = int(dataset.targets[ids[0]])
        axes[row, 0].imshow(original, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].add_patch(
            Rectangle((-0.5, -0.5), 14, 14, fill=False, edgecolor="#00b36b", lw=2.5)
        )
        axes[row, 0].set_xticks(
            np.arange(8) * 14 + 6.5, [str(int(dataset.targets[i])) for i in ids]
        )
        axes[row, 0].set_yticks([])
        axes[row, 0].set_xlabel("Класс каждой картинки; зелёная рамка — полезная")
        axes[row, 0].set_title(
            f"Восемь исходных картинок; ответ задачи — класс {label}"
        )
        axes[row, 1].imshow(observed.reshape(28, 56), cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(
            "Вход модели: 1568 перемешанных чисел\nПрямоугольник — только способ показа",
            fontsize=10,
        )
        axes[row, 1].axis("off")
        axes[row, 2].imshow(images[0].reshape(14, 14), cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("196 входов\nаналитической маски", fontsize=10)
        axes[row, 2].axis("off")
        selected_ids.append(ids.tolist())
    fig.suptitle(
        f"{dataset_name}: пример новой задачи {case['pair'][0]} / {case['pair'][1]}, набор 0",
        fontsize=14,
    )
    fig.savefig(ASSETS / (dataset_name.lower() + "_task.png"), dpi=180)
    plt.close(fig)
    records[dataset_name]["task_picture_train_image_ids"] = selected_ids
    methods = json.loads((out / "evaluation_protocol.json").read_text())["methods"]
    displays = {
        "analytic": (
            informative.astype(float),
            "mask",
            "196 полезных входов из 196\nЭталон, численные веса не заданы",
        )
    }
    for method in methods:
        data = np.load(out / (method + ".npz"))
        selected = int(data["query_loss"][:4].argmin())
        indices = data["indices"][selected]
        accuracy = float(data["test_accuracy"][selected])
        row = {
            "selected_state": selected,
            "query_bce": float(data["query_loss"][selected]),
            "test_accuracy": accuracy,
        }
        if method in DENSE_METHODS:
            weight = data["parameter_0"][selected]
            strengths = np.zeros(1568)
            strengths[indices] = np.sqrt(np.mean(weight**2, axis=1))
            energy = float(
                np.square(strengths[informative]).sum() / np.square(strengths).sum()
            )
            values = strengths / strengths.max()
            kind = "weights"
            extra = f"Энергия полезных весов: {energy:.1%}"
            row.update(
                representation="dense input-weight RMS; no discrete indexator",
                informative_weight_energy=energy,
            )
            arrays[dataset_name + "_" + method + "_input_weight_rms"] = strengths
        else:
            values = np.zeros(1568)
            values[indices] = 1
            kind = "mask"
            correct = int(informative[indices].sum())
            extra = f"Полезных входов: {correct}/{len(indices)}"
            row.update(
                input_ids_one_based=(indices + 1).tolist(),
                informative_inputs=correct,
                selected_inputs=len(indices),
            )
            arrays[dataset_name + "_" + method + "_input_mask"] = values
        records[dataset_name]["methods"][method] = row
        displays[method] = (values, kind, extra + f"\nTest accuracy: {accuracy:.2%}")
    for part, selected_methods in [
        ("main", MAIN_METHODS),
        ("controls", [m for m in methods if m not in MAIN_METHODS]),
    ]:
        rows = ["analytic"] + selected_methods
        fig = plt.figure(figsize=(16, 0.92 * len(rows) + 1.0), layout="constrained")
        grid = fig.add_gridspec(len(rows), 3, width_ratios=[2.8, 8, 2.5])
        for position, method in enumerate(rows):
            text_axis = fig.add_subplot(grid[position, 0])
            text_axis.axis("off")
            label = (
                "Аналитическая маска полезного изображения"
                if method == "analytic"
                else LABELS[method]
            )
            text_axis.text(
                0.02, 0.5, textwrap.fill(label, 32), va="center", fontsize=10
            )
            axis = fig.add_subplot(grid[position, 1])
            values, kind, extra = displays[method]
            axis.imshow(
                restored_strip(values, permutation),
                cmap="Blues" if kind == "mask" else "Oranges",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            axis.set_yticks([])
            axis.set_xticks([])
            for boundary in range(14, 112, 14):
                axis.axvline(boundary - 0.5, color="#999999", lw=0.6)
            axis.add_patch(
                Rectangle((-0.5, -0.5), 14, 14, fill=False, edgecolor="#00a060", lw=1.5)
            )
            if position == 0:
                axis.set_xticks(
                    np.arange(8) * 14 + 6.5,
                    ["Полезное"] + [f"Помеха {i}" for i in range(1, 8)],
                )
                axis.tick_params(
                    top=True, labeltop=True, bottom=False, labelbottom=False
                )
            metric_axis = fig.add_subplot(grid[position, 2])
            metric_axis.axis("off")
            metric_axis.text(0.01, 0.5, extra, fontsize=9, va="center")
        fig.suptitle(
            f"{dataset_name}: индексаторы и веса для задачи {case['pair'][0]} / {case['pair'][1]}, набор 0\nСиний — выбранный вход; оранжевый — RMS весов плотной сети, нормированная от 0 до 1",
            fontsize=13,
        )
        fig.savefig(
            ASSETS / (dataset_name.lower() + "_indexators_" + part + ".png"), dpi=180
        )
        plt.close(fig)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    records = {
        "display_rule": "First target task/case fixed before viewing methods; best restart by query only. Ground truth used solely to draw and align diagnostic references."
    }
    arrays = {}
    synthetic_examples(records, arrays)
    for dataset in ["MNIST", "FashionMNIST"]:
        image_examples(dataset, records, arrays)
    (RUNS / "structure_examples.json").write_text(json.dumps(records, indent=2))
    np.savez_compressed(RUNS / "structure_examples.npz", **arrays)
    print(
        json.dumps(
            {
                "synthetic": records["synthetic"],
                "images": {
                    d: {
                        m: {k: v for k, v in row.items() if k != "input_ids_one_based"}
                        for m, row in records[d]["methods"].items()
                    }
                    for d in ["MNIST", "FashionMNIST"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
