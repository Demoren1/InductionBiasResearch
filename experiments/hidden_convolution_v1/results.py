"""Build audited prediction and indexator comparisons after frozen evaluation."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from diagnostics import (
    align_to_teacher,
    novelty_diagnostics,
    source_diagnostics,
    structure_metrics,
)
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT.parent.parent / "mds/assets/2026-10-28/hidden_convolution"
LABELS = {
    "bank": "Случайная целая структура банка",
    "best_bank": "Лучшая целая исходная структура",
    "best_support": "Найденные связи; независимые веса",
    "column_sample": "Случайное сочетание столбцов",
    "column_search": "Обучаемый выбор столбцов",
    "entry_search": "Обучаемые индексы; частоты банка",
    "entry_uniform": "Обучаемые индексы; равномерный старт",
    "column_vae_sample": "VAE столбцов без обучения z",
    "column_vae_z": "VAE столбцов + z",
    "global_vae_z": "Обычный VAE + z",
    "untrained_column_vae_z": "Необученный VAE столбцов + z",
    "random_codes": "Случайные коды; найденные связи",
    "shuffled_support": "Перемешанные найденные связи",
    "dense_adam": "Плотный Adam",
    "dense_sgd": "Плотный SGD с моментом",
    "dense_transfer": "Градиентный перенос плотной сети",
    "oracle": "Правильный индексатор задан",
}


def summarize(out):
    if not (out / "refit_complete.json").exists():
        raise RuntimeError(f"Fixed-indexator refits incomplete: {out}")
    frozen = json.loads((out / "evaluation_frozen.json").read_text())
    for name, expected in frozen.items():
        actual = hashlib.sha256((out / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Frozen file changed: {out / name}")
    if (out / "fresh_target_manifest.json").exists():
        fresh = json.loads((out / "fresh_target_manifest.json").read_text())
        for filename, key in [
            ("target_data.npz", "data_sha256"),
            ("target_teacher.npz", "teacher_sha256"),
        ]:
            if hashlib.sha256((out / filename).read_bytes()).hexdigest() != fresh[key]:
                raise RuntimeError(f"Fresh target changed: {filename}")
    spec = json.loads((out / "evaluation_protocol.json").read_text())
    teacher = np.load(out / "target_teacher.npz")
    source_bank = np.load(out / "structure_bank.npz")
    training_indices = source_bank["indices"][np.load(out / "prior_split.npz")["train"]]

    def fingerprint(matrix):
        nonzero = matrix[matrix != 0]
        counts = np.unique(nonzero, return_counts=True)[1]
        return tuple(sorted(counts.tolist()))

    fingerprints = [fingerprint(matrix) for matrix in training_indices]
    summary = {
        "source": source_diagnostics(out),
        "novelty": novelty_diagnostics(out),
        "protocol": spec,
        "frozen_hashes_unchanged": True,
        "methods": {},
    }
    examples = {"teacher": teacher["indices"][0]}
    for method in spec["methods"]:
        data = np.load(out / f"{method}.npz")
        choices = data["query_nmse"].reshape(8, 4).argmin(1)
        selected_ids = np.arange(8) * 4 + choices
        errors = data["test_nmse"][selected_ids]
        dense = method.startswith("dense")
        row = {
            "label": LABELS[method],
            "mean_nmse": float(errors.mean()),
            "task_nmse": errors.tolist(),
            "median_nmse": float(np.median(errors)),
            "all_start_mean_nmse": float(data["test_nmse"].mean()),
            "numeric_parameters": int(data["values"].shape[1]),
        }
        train_z = method.endswith("_z") or method in [
            "column_search",
            "entry_search",
            "entry_uniform",
        ]
        row["trained_structural_variables"] = (
            int(np.prod(data["z"].shape[1:])) if train_z else 0
        )
        if not dense:
            structural = [
                structure_metrics(data["indices"][index], truth)
                for index, truth in zip(selected_ids, teacher["indices"])
            ]
            row["structure"] = {
                key: float(np.mean([s[key] for s in structural]))
                for key in structural[0]
            }
            row["per_task_structure"] = structural
            copied = 0
            for index in selected_ids:
                predicted = data["indices"][index]
                signature = fingerprint(predicted)
                for candidate, candidate_signature in zip(
                    training_indices, fingerprints
                ):
                    if signature == candidate_signature and np.array_equal(
                        align_to_teacher(predicted, candidate)[0], candidate
                    ):
                        copied += 1
                        break
            row["selected_indexators_exactly_in_training_bank"] = copied
            refit = np.load(out / f"refit_{method}.npz")
            refit_choices = refit["query_nmse"].reshape(8, 4).argmin(1)
            refit_errors = refit["test_nmse"].reshape(8, 4)[np.arange(8), refit_choices]
            row["fixed_indexator_refit_mean_nmse"] = float(refit_errors.mean())
            row["fixed_indexator_refit_task_nmse"] = refit_errors.tolist()
        index = int(selected_ids[0])
        examples[method] = (
            data["values"][index].reshape(64, 36)
            if dense
            else align_to_teacher(data["indices"][index], teacher["indices"][0])[0]
        )
        summary["methods"][method] = row
    (out / "study_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    return summary, examples


def draw_indices(axis, matrix, title, method):
    if method.startswith("dense"):
        limit = np.abs(matrix).max()
        axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    elif method == "best_support":
        axis.imshow(matrix != 0, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    else:
        colors = np.vstack(
            [np.ones((1, 4)), plt.get_cmap("turbo")(np.linspace(0, 1, 36))]
        )
        axis.imshow(
            matrix,
            cmap=ListedColormap(colors),
            vmin=0,
            vmax=36,
            aspect="auto",
            interpolation="nearest",
        )
    axis.set_title(title, fontsize=9)
    axis.set_xlabel("Скрытые нейроны")
    axis.set_ylabel("Входы")


def draw_comparison(summary, filename):
    methods = list(summary["methods"])
    rows = summary["methods"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 9), sharey=True)
    positions = np.arange(len(methods))
    error = [rows[m]["mean_nmse"] for m in methods]
    refit = [rows[m].get("fixed_indexator_refit_mean_nmse", np.nan) for m in methods]
    axes[0].barh(
        positions - 0.17,
        error,
        height=0.33,
        color="#537f9e",
        label="Совместное обучение",
    )
    axes[0].barh(
        positions + 0.17,
        refit,
        height=0.33,
        color="#28a88d",
        label="Фиксированный I, новый кодбук",
    )
    for i, value in enumerate(error):
        axes[0].text(value + 0.01, i - 0.17, f"{value:.3g}", va="center", fontsize=8)
    axes[0].set_xlim(0, max(error) * 1.3)
    axes[0].set_xlabel("Средняя тестовая NMSE ↓")
    axes[0].set_yticks(positions, [LABELS[m] for m in methods])
    axes[0].invert_yaxis()
    for offset, key, label, color in [
        (-0.17, "support_f1", "Связи: F1", "#537f9e"),
        (0.17, "sharing_f1", "Повторения весов: F1", "#28a88d"),
    ]:
        values = [rows[m].get("structure", {}).get(key, np.nan) for m in methods]
        axes[1].barh(positions + offset, values, height=0.33, color=color, label=label)
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Восстановление структуры ↑")
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / filename, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/pilot_v4")
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    summary, examples = summarize(args.out)
    (ASSETS / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    metrics = []
    for method, row in summary["methods"].items():
        metrics.append(
            {
                "method": method,
                "label": row["label"],
                "mean_nmse": row["mean_nmse"],
                "fixed_indexator_refit_nmse": row.get(
                    "fixed_indexator_refit_mean_nmse"
                ),
                "support_f1": row.get("structure", {}).get("support_f1"),
                "sharing_f1": row.get("structure", {}).get("sharing_f1"),
                "numeric_parameters": row["numeric_parameters"],
                "trained_structural_variables": row["trained_structural_variables"],
            }
        )
    with (ASSETS / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(metrics[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(metrics)
    draw_comparison(summary, "comparison.png")
    first = ROOT / "runs/pilot_v2"
    previous, _ = summarize(first)
    (ASSETS / "initial_pilot_summary.json").write_text(
        json.dumps(previous, indent=2, ensure_ascii=False)
    )
    draw_comparison(previous, "initial_pilot_comparison.png")

    methods = ["teacher"] + list(summary["methods"])
    rows = math.ceil(len(methods) / 4)
    fig, axes = plt.subplots(rows, 4, figsize=(16, rows * 3.7))
    for axis, method in zip(axes.flat, methods):
        draw_indices(
            axis,
            examples[method],
            "Правильный индексатор" if method == "teacher" else LABELS[method],
            method,
        )
    for axis in axes.flat[len(methods) :]:
        axis.set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "all_indexators.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    selected = [
        "teacher",
        "best_support",
        "column_search",
        "entry_search",
        "column_vae_z",
        "dense_adam",
    ]
    fig, axes = plt.subplots(1, len(selected), figsize=(17, 5))
    for axis, method in zip(axes, selected):
        draw_indices(
            axis,
            examples[method],
            "Правильный индексатор" if method == "teacher" else LABELS[method],
            method,
        )
    fig.tight_layout()
    fig.savefig(ASSETS / "indexators.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    teacher = np.load(args.out / "target_teacher.npz")
    data = np.load(args.out / "target_data.npz")
    permutation = teacher["permutation"]
    observed = data["x"][0]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(observed[permutation].reshape(8, 8), cmap="coolwarm")
    axes[0].add_patch(
        plt.Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor="black", linewidth=3)
    )
    axes[0].set_title("Учитель: вход 8×8, окно 3×3")
    axes[1].imshow(observed.reshape(8, 8), cmap="coolwarm")
    for index in teacher["windows"][0]:
        row, column = divmod(int(index), 8)
        axes[1].add_patch(
            plt.Rectangle(
                (column - 0.5, row - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=2,
            )
        )
    axes[1].set_title("MLP: те же числа после перестановки")
    axes[2].imshow(teacher["choices"][0].reshape(6, 6), cmap="tab10", vmin=0, vmax=9)
    axes[2].set_title("Выбор фильтра в каждом из 36 окон")
    axes[3].imshow(
        teacher["values"][0, 1:].reshape(4, 9),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    axes[3].set_title("Кодбук: четыре фильтра по 9 чисел")
    axes[3].set_xlabel("Коэффициенты фильтра")
    axes[3].set_ylabel("Фильтр")
    axes[3].set_yticks(np.arange(4), np.arange(1, 5))
    axes[3].set_xticks(np.arange(9), np.arange(1, 10))
    fig.tight_layout()
    fig.savefig(ASSETS / "task.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    truth = teacher["indices"][0] != 0
    first_mask = np.load(first / "learned_support.npz")["mask"]
    final_mask = np.load(args.out / "learned_support.npz")["mask"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))
    for axis, mask, label in zip(
        axes,
        [truth, first_mask, final_mask],
        [
            "Правильные связи",
            "Первоначальное усреднение",
            "После переноса найденных связей",
        ],
    ):
        aligned, _ = align_to_teacher(mask.astype(int), truth.astype(int))
        axis.imshow(aligned != 0, cmap="Greys", vmin=0, vmax=1, aspect="auto")
        axis.set_title(label, fontsize=10)
        axis.set_xlabel("Скрытые нейроны")
        axis.set_ylabel("Входы")
    fig.tight_layout()
    fig.savefig(ASSETS / "supports.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    for method, row in summary["methods"].items():
        print(
            method,
            row["mean_nmse"],
            row.get("fixed_indexator_refit_mean_nmse"),
            row.get("structure", {}).get("support_f1"),
            row.get("structure", {}).get("sharing_f1"),
        )
    print("Novelty", summary["novelty"])


if __name__ == "__main__":
    main()
