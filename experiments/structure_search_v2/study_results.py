"""Summarize the frozen three-case study and render figures from saved results."""

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
ASSETS = ROOT.parents[1] / "mds/assets/2026-10-28/structure_transfer"
LABELS = {
    "fixed_proposal": "Извлечённый индексатор",
    "vae_sample": "NF+VAE без обучения z",
    "vae_mode": "NF+VAE: маска без шума",
    "vae_z": "NF+VAE + обучение z",
    "untrained_vae_z": "Необученный VAE + z",
    "random": "Случайная структура",
    "dense_adam": "Плотный Adam",
    "dense_sgd": "Плотный SGD с моментом",
    "dense_transfer": "Градиентный перенос",
    "pretrained_adam": "Предобучение + Adam",
    "frozen_pretrained": "Предобучение, только выход",
    "frequency": "Частоты связей в банке",
    "functional": "Полезность связей в банке",
    "supervised_selector": "Связь входов с меткой",
    "component_random": "Случайные 64 из найденной группы",
    "component_full": "Вся найденная группа",
    "oracle_image": "Правильные входы заданы",
    "direct_indices": "Прямое обучение индексов",
    "dense_adam_sweep": "Плотный Adam: 4 шага обучения",
    "dense_sgd_sweep": "Плотный SGD: 4 шага обучения",
    "pretrained_adam_sweep": "Предобучение + Adam: 4 шага",
    "frozen_pretrained_sweep": "Только выход: 4 шага",
}
COLORS = {
    "vae_z": "#087f8c",
    "vae_sample": "#39a9ad",
    "fixed_proposal": "#9a5aab",
    "dense_adam": "#bc5a36",
    "dense_sgd": "#d89b39",
    "dense_transfer": "#455a64",
    "pretrained_adam": "#455a64",
    "frozen_pretrained": "#8d758d",
    "random": "#bdbdbd",
    "untrained_vae_z": "#7986cb",
    "functional": "#438749",
    "component_full": "#7756a2",
}


def canonical(rows):
    return {tuple(row) for row in np.sort(rows, axis=-1)}


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "axes.axisbelow": True,
            "savefig.dpi": 180,
        }
    )
    summary = {"synthetic": {}, "images": {}, "audit": {}}
    synthetic = RUNS / "scrambled_transfer_v1"
    methods = json.loads((synthetic / "calibration.json").read_text())["methods"]
    for method in methods:
        data = np.load(synthetic / (method + ".npz"))
        query = data["query_nmse"].reshape(8, 8)
        errors = data["test_nmse"].reshape(8, 8)
        chosen = query.argmin(1)
        values = errors[np.arange(8), chosen]
        summary["synthetic"][method] = {
            "mean_start_test_nmse": float(errors.mean()),
            "selected_test_nmse": float(values.mean()),
            "successful_starts": int((errors < 0.01).sum()),
            "successful_selected_tasks": int((values < 0.01).sum()),
            "per_task": values.tolist(),
        }
    for dataset in ["MNIST", "FashionMNIST"]:
        out = RUNS / ("distractor_" + dataset + "_v5")
        cases = json.loads((out / "target_cases.json").read_text())
        protocol = json.loads((out / "evaluation_protocol.json").read_text())
        metrics = {}
        bank = np.load(out / "good_bank.npz")
        bank_masks = canonical(bank["indices"])
        train_masks = canonical(bank["indices"][bank["role"] == 0])
        raw_bank = np.load(RUNS / ("distractor_" + dataset + "_v3/good_bank.npz"))
        for method in protocol["methods"]:
            data = np.load(out / (method + ".npz"))
            choice = data["query_loss"].reshape(48, 4).argmin(1)
            selected = np.arange(48) * 4 + choice
            accuracy = data["test_accuracy"][selected]
            ce = data["test_loss"][selected]
            per_pair = []
            for pair_id in range(6):
                per_pair.append(float(accuracy[pair_id * 8 : (pair_id + 1) * 8].mean()))
            metrics[method] = {
                "selected_test_accuracy": float(accuracy.mean()),
                "selected_test_bce": float(ce.mean()),
                "case_sd_accuracy": float(accuracy.std(ddof=1)),
                "all_start_accuracy": float(data["test_accuracy"].mean()),
                "case_accuracy": accuracy.tolist(),
                "per_pair_accuracy": per_pair,
                "selected_steps_median": float(np.median(data["step"][selected])),
            }
            if data["indices"].shape[-1] == 64:
                masks = canonical(data["indices"])
                metrics[method].update(
                    {
                        "unique_target_masks": len(masks),
                        "new_vs_all_good_bank": len(masks - bank_masks),
                        "new_vs_prior_training_bank": len(masks - train_masks),
                    }
                )
        summary["images"][dataset] = metrics
        frozen = json.loads((out / "evaluation_frozen.json").read_text())
        splits = np.load(out / "target_splits.npz")
        intersection = [
            len(np.intersect1d(a, b))
            for a, b in zip(splits["support"], splits["query"])
        ]
        component = np.load(
            RUNS / ("distractor_" + dataset + "_v4/component_proposal.npz")
        )
        pool_size = len(component["pool"])
        raw_masks = np.load(
            RUNS / ("distractor_" + dataset + "_v3/raw_mask_scores.npz")
        )
        refined_masks = np.load(out / "raw_mask_scores.npz")
        summary["audit"][dataset] = {
            "frozen_files_unchanged": {
                name: hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
                for name, digest in frozen["sha256"].items()
            },
            "target_cases": len(cases),
            "starts_per_case": 4,
            "support_query_image_id_overlap_max": max(intersection),
            "raw_bank_states": len(raw_masks["indices"]),
            "raw_unique_masks": len(canonical(raw_masks["indices"])),
            "good_raw_networks": len(raw_bank["indices"]),
            "good_raw_accuracy_mean": float(raw_bank["accuracy"].mean()),
            "refined_bank_states": len(refined_masks["indices"]),
            "refined_unique_masks": len(canonical(refined_masks["indices"])),
            "good_refined_networks": len(bank["indices"]),
            "good_refined_unique_masks": len(bank_masks),
            "good_refined_train_unique_masks": len(train_masks),
            "good_refined_accuracy_mean": float(bank["accuracy"].mean()),
            "pool_size": pool_size,
            "possible_64_subsets_of_pool": str(math.comb(pool_size, 64)),
            "possible_64_subsets_of_all_inputs": str(math.comb(1568, 64)),
            "group_diagnostic": json.loads(
                (
                    RUNS / ("distractor_" + dataset + "_v4/component_diagnostic.json")
                ).read_text()
            ),
        }
    (RUNS / "study_summary.json").write_text(json.dumps(summary, indent=2))
    with (RUNS / "study_metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "case",
                "method",
                "selected_test_accuracy",
                "selected_test_bce",
                "selected_test_nmse",
                "mean_start_test_nmse",
                "successful_starts",
            ]
        )
        for method, row in summary["synthetic"].items():
            writer.writerow(
                [
                    "synthetic",
                    method,
                    "",
                    "",
                    row["selected_test_nmse"],
                    row["mean_start_test_nmse"],
                    row["successful_starts"],
                ]
            )
        for dataset, metrics in summary["images"].items():
            for method, row in metrics.items():
                writer.writerow(
                    [
                        dataset,
                        method,
                        row["selected_test_accuracy"],
                        row["selected_test_bce"],
                        "",
                        "",
                        "",
                    ]
                )
    # The selection unit is a task / support split, not a restart.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    methods = [
        "random",
        "untrained_vae_z",
        "dense_adam",
        "dense_sgd",
        "pretrained_adam",
        "frozen_pretrained",
        "vae_sample",
        "vae_z",
        "functional",
        "component_full",
        "oracle_image",
    ]
    for axis, dataset in zip(axes, ["MNIST", "FashionMNIST"]):
        values = [
            100 * summary["images"][dataset][m]["selected_test_accuracy"]
            for m in methods
        ]
        y = np.arange(len(methods))
        axis.barh(
            y,
            np.array(values) - 50,
            left=50,
            color=[COLORS.get(m, "#78909c") for m in methods],
        )
        axis.set_yticks(y, [LABELS[m] for m in methods], fontsize=9)
        axis.invert_yaxis()
        axis.set_xlim(50, 102)
        for position, value in zip(y, values):
            axis.text(value + 0.4, position, f"{value:.1f}%", va="center", fontsize=9)
        axis.set_title(dataset + ": новое различение классов 6–9")
        axis.set_xlabel("Точность на официальном test, %; шкала от 50%")
    fig.suptitle(
        "Одна полезная картинка и семь отвлекающих: перенос найденных входов",
        fontsize=13,
    )
    fig.savefig(ASSETS / "image_comparison.png")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5), layout="constrained")
    methods = [
        "random",
        "untrained_vae_z",
        "dense_adam",
        "dense_sgd",
        "dense_transfer",
        "fixed_proposal",
        "vae_sample",
        "vae_z",
    ]
    values = [summary["synthetic"][m]["selected_test_nmse"] for m in methods]
    axis.barh(
        np.arange(len(methods)),
        values,
        color=[COLORS.get(m, "#78909c") for m in methods],
    )
    axis.set_xscale("log")
    axis.set_xlim(1e-13, 10)
    axis.set_yticks(np.arange(len(methods)), [LABELS[m] for m in methods])
    axis.invert_yaxis()
    for i, value in enumerate(values):
        axis.text(value * 1.7, i, f"{value:.2g}", va="center")
    axis.axvline(0.01, ls="--", color="black", lw=1, label="Порог успеха: 0,01")
    axis.set_xlabel("Средняя test NMSE после выбора по query; меньше — лучше")
    axis.set_title("Синтетическая MLP: 8 новых функций, по 8 стартов")
    axis.legend(loc="lower right")
    fig.savefig(ASSETS / "synthetic_comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    configurations = [
        (
            synthetic,
            "Синтетическая MLP",
            "best_query_nmse",
            ["vae_z", "fixed_proposal", "dense_adam", "dense_sgd", "dense_transfer"],
        ),
        (
            RUNS / "distractor_MNIST_v5",
            "MNIST с отвлекающими входами",
            "query_bce",
            ["vae_z", "functional", "dense_adam", "dense_sgd", "pretrained_adam"],
        ),
        (
            RUNS / "distractor_FashionMNIST_v5",
            "Fashion-MNIST с отвлекающими входами",
            "query_bce",
            ["vae_z", "functional", "dense_adam", "dense_sgd", "pretrained_adam"],
        ),
    ]
    for axis, (directory, title, metric, methods) in zip(axes, configurations):
        for method in methods:
            history = json.loads((directory / (method + "_history.json")).read_text())
            values = [np.array(row[metric]).mean() for row in history]
            axis.plot(
                [row["step"] for row in history],
                values,
                label=LABELS[method],
                color=COLORS.get(method, "#438749"),
                lw=1.7,
            )
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Шаги обучения новой задачи")
        axis.set_ylabel(
            "Средняя лучшая query NMSE"
            if metric.endswith("nmse")
            else "Средняя лучшая query BCE"
        )
        axis.set_yscale("log")
        axis.legend(fontsize=7, loc="upper right")
    fig.savefig(ASSETS / "training.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    moments = np.load(RUNS / "scrambled_pattern_source_pilot_v1/moment_proposal.npz")
    order = moments["permutation"]
    axes[0].imshow(moments["affinity"][order][:, order], cmap="magma", aspect="equal")
    axes[0].set_title("Синтетика: зависимости после группировки", fontsize=10)
    axes[0].set_xlabel("64 координаты, порядок найден алгоритмом")
    axes[0].set_ylabel("Координаты")
    for axis, dataset in zip(axes[1:], ["MNIST", "FashionMNIST"]):
        out = RUNS / ("distractor_" + dataset + "_v4")
        component = np.load(out / "component_proposal.npz")
        permutation = np.load(out / "hidden_layout.npz")["permutation"]
        selected = permutation[component["pool"]]
        image = np.zeros((8, 196))
        image[selected // 196, selected % 196] = 1
        axis.imshow(
            image, cmap="Blues", aspect="auto", interpolation="nearest", vmin=0, vmax=1
        )
        axis.set_title(dataset + ": найденная полезная группа", fontsize=10)
        axis.set_yticks(
            range(8), ["Полезное"] + [f"Помеха {i}" for i in range(1, 8)], fontsize=8
        )
        axis.set_xlabel("Пиксель исходной картинки (0–195)")
    fig.suptitle(
        "Какая структура извлечена из исходных данных и частичных решений", fontsize=12
    )
    fig.savefig(ASSETS / "structure.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for axis, dataset in zip(axes, ["MNIST", "FashionMNIST"]):
        directory = RUNS / ("distractor_" + dataset + "_v5")
        cases = json.loads((directory / "target_cases.json").read_text())
        pairs = [" / ".join(map(str, cases[i * 8]["pair"])) for i in range(6)]
        for method in ["vae_z", "dense_sgd", "pretrained_adam", "component_full"]:
            values = 100 * np.array(
                summary["images"][dataset][method]["per_pair_accuracy"]
            )
            axis.plot(
                range(6), values, "o-", label=LABELS[method], color=COLORS[method]
            )
        axis.set_xticks(range(6), pairs)
        axis.set_title(dataset)
        axis.set_xlabel("Пара новых классов")
        axis.set_ylabel("Test accuracy, %; среднее по 8 наборам")
        axis.set_ylim(65, 101)
        axis.legend(fontsize=8)
    fig.savefig(ASSETS / "per_pair.png")
    plt.close(fig)

    # A concrete prediction example: held-out observations of target function 1.
    test = np.load(synthetic / "target_data.npz")
    x, teacher = test["x"][10240:10640], test["y"][0, 10240:10640]
    fig, axis = plt.subplots(figsize=(6, 5), layout="constrained")
    for method in ["vae_z", "dense_sgd"]:
        data = np.load(synthetic / (method + ".npz"))
        start = int(data["query_nmse"][:8].argmin())
        w, b, a, c = [data[f"parameter_{i}"][start] for i in range(4)]
        if method == "vae_z":
            grouped = x[:, data["indices"][start]].reshape(-1, 8, 8)
            hidden = np.maximum(0, np.einsum("bgr,kr->bgk", grouped, w) + b[None, None])
            predicted = (hidden * a[None, None]).sum((1, 2)) / np.sqrt(32) + c
        else:
            predicted = (np.maximum(0, x @ w + b) * a).sum(1) + c
        axis.scatter(
            teacher,
            predicted,
            s=9,
            alpha=0.5,
            label=LABELS[method],
            color=COLORS[method],
        )
    limits = [float(teacher.min()), float(teacher.max())]
    axis.plot(limits, limits, "k--", lw=1, label="Точное решение: предсказание = ответ")
    axis.set_xlabel("Ответ новой функции-учителя")
    axis.set_ylabel("Предсказание модели")
    axis.set_title("Пример: 400 независимых входов первой новой функции")
    axis.legend(fontsize=8)
    fig.savefig(ASSETS / "synthetic_example.png")
    plt.close(fig)

    print(
        json.dumps(
            {
                "synthetic": summary["synthetic"],
                "images": {
                    ds: {
                        m: round(100 * r["selected_test_accuracy"], 3)
                        for m, r in metrics.items()
                    }
                    for ds, metrics in summary["images"].items()
                },
                "audit": summary["audit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
