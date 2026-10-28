"""Report the image ablation and the independent shared-codebook MLP check."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent / "structure_search_v2"
MLP = ROOT.parent / "shared_codebook_mlp_v1" / "runs" / "v2"
ASSETS = ROOT.parents[1] / "mds/assets/2026-10-28/structure_only"
sys.path.insert(0, str(SHARED))
from study_structure_examples import restored_strip

NAMES = {
    "bank_sample": "Целая структура из банка",
    "frequency": "Частоты связей",
    "task_consensus": "Согласие исходных задач",
    "vae_mode": "VAE: без шума z",
    "vae_sample": "VAE без обучения z",
    "vae_z": "VAE + обучение z",
    "untrained_vae_z": "Необученный VAE + z",
    "shuffled_frequency": "Частоты перемешанных масок",
    "shuffled_vae_z": "VAE перемешанных масок + z",
    "dense_adam": "Плотный Adam",
    "dense_sgd": "Плотный SGD с моментом",
    "dense_transfer": "Градиентный перенос",
    "pretrained_adam": "Предобучение + Adam",
    "frozen_pretrained": "Предобучение: только выход",
    "random": "Случайная структура",
    "consensus": "Общий индексатор из банка",
    "shuffled_consensus": "Перемешанный общий индексатор",
}


def align_partition(found, reference):
    d, h = reference.shape
    a = (found[:, None] == found[None]).reshape(d * d, h)
    b = (reference[:, None] == reference[None]).reshape(d * d, h)
    old, new = linear_sum_assignment(np.not_equal(a[:, :, None], b[:, None]).sum(0))
    order = np.empty(h, dtype=int)
    order[new] = old
    found = found[:, order]
    votes = np.zeros((32, 32), dtype=int)
    np.add.at(votes, (found.flatten(), reference.flatten()), 1)
    old, new = linear_sum_assignment(-votes)
    mapping = np.empty(32, dtype=int)
    mapping[old] = new
    aligned = mapping[found]
    return aligned, int((aligned == reference).sum())


def main():
    torch.set_num_threads(4)
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.dpi": 180,
        }
    )
    summary = {"images": {}, "mlp": {}, "provenance": {}}
    for ds in ["MNIST", "FashionMNIST"]:
        out = ROOT / "runs" / ds
        parent = SHARED / "runs" / ("distractor_" + ds + "_v5")
        protocol = json.loads((out / "evaluation_protocol.json").read_text())
        informative = np.load(parent / "hidden_layout.npz")["informative"]
        metrics = {}
        for method in protocol["methods"] + protocol["reused_baselines"]:
            origin = parent if method in protocol["reused_baselines"] else out
            data = np.load(origin / (method + ".npz"))
            selected = np.arange(48) * 4 + data["query_loss"].reshape(48, 4).argmin(1)
            values = data["test_accuracy"][selected]
            row = {
                "test_accuracy": float(values.mean()),
                "test_bce": float(data["test_loss"][selected].mean()),
                "per_pair_accuracy": values.reshape(6, 8).mean(1).tolist(),
            }
            if data["indices"].shape[-1] == 64:
                useful = np.isin(data["indices"][selected], informative).sum(1)
                row.update(
                    useful_inputs_mean=float(useful.mean()),
                    useful_inputs_first=int(useful[0]),
                )
            metrics[method] = row
        summary["images"][ds] = metrics
        history = json.loads((out / "prior_history.json").read_text())
        state = torch.load(out / "prior.pt", weights_only=True, map_location="cpu")
        initial = torch.load(
            out / "untrained_prior.pt", weights_only=True, map_location="cpu"
        )
        frozen = json.loads((out / "evaluation_frozen.json").read_text())
        unchanged = {}
        for name, digest in frozen.items():
            path = (
                out / "evaluation_protocol.json" if name == "protocol" else ROOT / name
            )
            unchanged[name] = hashlib.sha256(path.read_bytes()).hexdigest() == digest
        summary["provenance"][ds] = {
            "prior_best_history_epoch": min(history, key=lambda r: r["validation_bce"])[
                "epoch"
            ],
            "prior_identical_to_initial": all(
                torch.equal(state[k], initial[k]) for k in state
            ),
            "frozen_files_unchanged": unchanged,
            "source_only_extractor": json.loads(
                (out / "prior_complete.json").read_text()
            ),
        }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained")
    image_methods = [
        "bank_sample",
        "frequency",
        "task_consensus",
        "vae_z",
        "untrained_vae_z",
        "shuffled_frequency",
        "shuffled_vae_z",
        "dense_sgd",
        "pretrained_adam",
    ]
    for axis, ds in zip(axes, ["MNIST", "FashionMNIST"]):
        values = [
            summary["images"][ds][m]["test_accuracy"] * 100 for m in image_methods
        ]
        axis.barh(
            range(len(values)),
            np.array(values) - 50,
            left=50,
            color=[
                "#087f8c" if m in ["frequency", "vae_z"] else "#91a6ad"
                for m in image_methods
            ],
        )
        axis.set_yticks(
            range(len(values)), [NAMES[m] for m in image_methods], fontsize=9
        )
        axis.invert_yaxis()
        axis.set_xlim(50, 101)
        axis.set_xlabel("Средняя test accuracy, %; шкала от 50%")
        axis.set_title(ds)
        for i, value in enumerate(values):
            axis.text(value + 0.4, i, f"{value:.1f}%", va="center", fontsize=9)
    fig.suptitle("Изображения: извлечение только из масок, без группировки входов")
    fig.savefig(ASSETS / "image_control.png")
    plt.close(fig)

    image_methods = [
        "bank_sample",
        "frequency",
        "task_consensus",
        "vae_mode",
        "vae_sample",
        "vae_z",
        "untrained_vae_z",
        "shuffled_frequency",
        "shuffled_vae_z",
    ]
    fig, axes = plt.subplots(10, 2, figsize=(14, 12), layout="constrained")
    for column, ds in enumerate(["MNIST", "FashionMNIST"]):
        parent = SHARED / "runs" / ("distractor_" + ds + "_v5")
        permutation = np.load(parent / "hidden_layout.npz")["permutation"]
        reference = permutation < 196
        for row, method in enumerate(["analytic"] + image_methods):
            axis = axes[row, column]
            if method == "analytic":
                mask = reference.astype(float)
                label = ds + ": аналитическая полезная область"
            else:
                data = np.load(ROOT / "runs" / ds / (method + ".npz"))
                index = int(data["query_loss"][:4].argmin())
                mask = np.zeros(1568)
                mask[data["indices"][index]] = 1
                label = NAMES[method] + f"; полезных {int(mask[reference].sum())}/64"
            axis.imshow(restored_strip(mask, permutation), cmap="Blues", vmin=0, vmax=1)
            for boundary in range(14, 112, 14):
                axis.axvline(boundary - 0.5, color="gray", lw=0.5)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(label, fontsize=10)
    fig.suptitle(
        "Первый случай 6/7: фактические выбранные входы; слева в каждой строке — полезная картинка"
    )
    fig.savefig(ASSETS / "image_masks.png")
    plt.close(fig)

    truth = np.load(MLP / "teacher.npz")["indexator"]
    protocol = json.loads((MLP / "evaluation_protocol.json").read_text())
    example_matrices = {}
    for method in protocol["methods"]:
        data = np.load(MLP / (method + ".npz"))
        selected = np.arange(8) * 8 + data["query_nmse"].reshape(8, 8).argmin(1)
        error = data["test_nmse"][selected]
        row = {
            "selected_test_nmse": float(error.mean()),
            "mean_start_test_nmse": float(data["test_nmse"].mean()),
            "successful_starts_of_64": int((data["test_nmse"] < 0.001).sum()),
            "successful_selected_tasks_of_8": int((error < 0.001).sum()),
            "selected_state_first": int(selected[0]),
        }
        if data["values"].shape[-1] == 32:
            counts = []
            cache = {}
            for state in range(64):
                raw = data["indices"][state].reshape(32, 16)
                key = raw.tobytes()
                if key not in cache:
                    cache[key] = align_partition(raw, truth)
                aligned, count = cache[key]
                counts.append(count)
            first = int(selected[0])
            aligned, _ = cache[data["indices"][first].reshape(32, 16).tobytes()]
            example_matrices[method] = aligned
            row.update(
                exact_indexator_starts_of_64=int((np.array(counts) == 512).sum()),
                selected_indexator_correct_positions=np.array(counts)[
                    selected
                ].tolist(),
            )
        else:
            example_matrices[method] = data["values"][selected[0]].reshape(32, 16)
        summary["mlp"][method] = row
    bank = np.load(MLP / "structure_bank.npz")
    consensus, correct = align_partition(bank["consensus"], truth)
    summary["provenance"]["mlp"] = {
        "consensus_correct_positions_of_512": correct,
        "true_first_four_rows_one_based": (truth[:4] + 1).tolist(),
        "frozen_files_unchanged": {
            name: hashlib.sha256((MLP / name).read_bytes()).hexdigest() == digest
            for name, digest in json.loads(
                (MLP / "evaluation_frozen.json").read_text()
            ).items()
        },
    }
    np.savez_compressed(
        ROOT / "runs" / "indexator_examples.npz",
        analytic=truth,
        source_consensus=consensus,
        **example_matrices,
    )
    fig, axis = plt.subplots(figsize=(10, 5.4), layout="constrained")
    methods = protocol["methods"]
    values = [summary["mlp"][m]["selected_test_nmse"] for m in methods]
    axis.barh(
        range(len(methods)),
        values,
        color=[
            "#087f8c" if m in ["vae_z", "consensus"] else "#91a6ad" for m in methods
        ],
    )
    axis.set_xscale("log")
    axis.set_xlim(1e-15, 10)
    axis.invert_yaxis()
    axis.set_yticks(range(len(methods)), [NAMES[m] for m in methods])
    axis.axvline(0.001, color="black", ls="--", lw=1)
    for i, value in enumerate(values):
        axis.text(value * 1.5, i, f"{value:.2g}", va="center", fontsize=9)
    axis.set_xlabel("Средняя test NMSE после выбора по query; меньше — лучше")
    axis.set_title("Новая MLP: 32 числа кодбука вместо 512 независимых весов")
    fig.savefig(ASSETS / "mlp_comparison.png")
    plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), layout="constrained")
    for axis, method in zip(axes.flat, ["analytic"] + methods):
        if method == "analytic":
            matrix = truth
            title = "Аналитический I*"
            cmap = "turbo"
            lo, hi = -0.5, 31.5
        else:
            matrix = example_matrices[method]
            if method.startswith("dense"):
                matrix = np.abs(matrix) / np.abs(matrix).max()
                title = NAMES[method] + "\nМодули численных весов"
                cmap = "magma"
                lo, hi = 0, 1
            else:
                matched = int((matrix == truth).sum())
                title = NAMES[method] + f"\nI: совпадение {matched}/512"
                cmap = "turbo"
                lo, hi = -0.5, 31.5
        axis.imshow(matrix, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("16 скрытых нейронов", fontsize=8)
        axis.set_ylabel("32 входа", fontsize=8)
        axis.set_xticks([0, 7, 15], [1, 8, 16])
        axis.set_yticks([0, 15, 31], [1, 16, 32])
    axes.flat[-1].axis("off")
    fig.suptitle(
        "Первая новая функция: аналитический индексатор и результаты всех методов\nЦвет индекса означает общее число кодбука; совпадающие подписи согласованы только для диагностики"
    )
    fig.savefig(ASSETS / "mlp_indexators.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    axis = axes[0]
    axis.axis("off")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    boxes = [
        (0.79, "32 независимых нормальных входа"),
        (0.55, "Матрица W: 32×16 связей\nКаждая выбирает одно из 32 общих чисел"),
        (0.31, "16 ReLU-нейронов\nСмещения 0, выходные веса фиксированы"),
        (0.07, "Сумма ответов / 4 → y"),
    ]
    for y, label in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (0.06, y),
                0.88,
                0.14,
                boxstyle="round,pad=.01",
                edgecolor="#087f8c",
                facecolor="#edf5f7",
            )
        )
        axis.text(0.5, y + 0.07, label, ha="center", va="center", fontsize=10)
    for index in range(3):
        axis.add_patch(
            FancyArrowPatch(
                (0.5, boxes[index][0] - 0.01),
                (0.5, boxes[index + 1][0] + 0.16),
                arrowstyle="-|>",
                mutation_scale=12,
            )
        )
    axis.set_title("Что вычисляет новая синтетическая MLP")
    axes[1].imshow(truth == 0, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title(
        "16 мест использования первого числа кодбука\nИх расположение неизвестно алгоритму"
    )
    axes[1].set_xlabel("16 скрытых нейронов")
    axes[1].set_ylabel("32 входа")
    axes[1].set_xticks([0, 7, 15], [1, 8, 16])
    axes[1].set_yticks([0, 15, 31], [1, 16, 32])
    fig.savefig(ASSETS / "mlp_task.png")
    plt.close(fig)
    (ROOT / "runs" / "summary.json").write_text(json.dumps(summary, indent=2))
    with (ROOT / "runs" / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "experiment",
                "method",
                "test_accuracy",
                "selected_test_nmse",
                "successful_starts",
            ]
        )
        for ds, metrics in summary["images"].items():
            for method, row in metrics.items():
                writer.writerow([ds, method, row["test_accuracy"], "", ""])
        for method, row in summary["mlp"].items():
            writer.writerow(
                [
                    "MLP",
                    method,
                    "",
                    row["selected_test_nmse"],
                    row["successful_starts_of_64"],
                ]
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
