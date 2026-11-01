"""Summarize independently selected target runs and draw scientific figures."""

import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

LABELS = {
    "ar": "Авторегрессия",
    "frequency": "Частоты",
    "consensus": "Один индексатор",
    "adam": "Adam",
    "sgd": "SGD",
    "linear": "Линейная модель",
    "transfer": "Перенос весов",
    "mask_unshared": "Маска без общих весов",
    "random_topology": "Случайные входы",
    "wsr": "WSR: субградиент",
    "oracle": "Правильный индексатор",
    "adam_long": "Adam: 16384 шага",
    "sgd_long": "SGD: 16384 шага",
}


def path_for(root, method, noise, size, stage="target"):
    if method in ["oracle", "adam", "sgd", "linear"]:
        return (
            root
            / ("target_fit" if stage == "target" else "calibration_fit")
            / f"noise_{noise}_n{size}"
            / f"{method}.npz"
        )
    if method in ["ar", "frequency", "consensus"]:
        return (
            root
            / "priors/relations"
            / f"noise_{noise}"
            / f"{stage}_n{size}"
            / f"{method}.npz"
        )
    if method == "wsr":
        return (
            root / "weight_sharing" / f"noise_{noise}" / f"{stage}_n{size}" / "wsr.npz"
        )
    if method.endswith("_long"):
        return (
            root
            / "long_control"
            / f"noise_{noise}"
            / f"{stage}_n{size}"
            / f"{method[:-5]}.npz"
        )
    return (
        root / "competitors" / f"noise_{noise}" / f"{stage}_n{size}" / f"{method}.npz"
    )


def best_rows(arrays, config=None):
    chosen = []
    for task in np.unique(arrays["task_ids"]):
        ids = np.flatnonzero(arrays["task_ids"] == task)
        if config is not None:
            ids = ids[arrays["config_ids"][ids] == config]
        chosen.append(ids[arrays["query_loss"][ids].argmin()])
    return np.array(chosen)


def calibration_score(path):
    refined = path.parent / "refinement/selection.json"
    if path.name == "wsr.npz" and refined.exists():
        return json.loads(refined.read_text())["score"]
    arrays = np.load(path)
    scores = []
    for config in np.unique(arrays["config_ids"]):
        ids = best_rows(arrays, config)
        scores.append(float(arrays["query_loss"][ids].mean()))
    return min(scores)


def freeze_primary(root, plan):
    endpoint = plan["primary_endpoint"]
    rankings = {}
    for role, names in [
        ("structural", endpoint["structural_candidates"]),
        ("competitor", endpoint["competitors"]),
    ]:
        scores = {
            name: calibration_score(
                path_for(
                    root,
                    name,
                    endpoint["noise_id"],
                    endpoint["support_size"],
                    "calibrate",
                )
            )
            for name in names
        }
        rankings[role] = {"chosen": min(scores, key=scores.get), "scores": scores}
    (root / "primary_selection.json").write_text(json.dumps(rankings, indent=2) + "\n")
    return rankings


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
    output = Path("experiments/dynamic_signal_v1/figures")
    output.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    rng = np.random.default_rng(2100481)
    bootstrap = rng.integers(plan["target_tasks"], size=(10000, plan["target_tasks"]))
    rows, selected = [], {}
    for noise, sigma in enumerate(plan["noise"]):
        for size in plan["target_sizes"]:
            for method in LABELS:
                path = path_for(root, method, noise, size)
                if not path.exists():
                    if method.endswith("_long") and size < 2048:
                        continue
                    raise FileNotFoundError(path)
                arrays = np.load(path)
                ids = best_rows(arrays)
                accuracy = arrays["test_accuracy"][ids]
                low, high = np.quantile(accuracy[bootstrap].mean(-1), [0.025, 0.975])
                rows.append(
                    {
                        "sigma": sigma,
                        "support_size": size,
                        "method": method,
                        "accuracy": float(accuracy.mean()),
                        "accuracy_low": float(low),
                        "accuracy_high": float(high),
                        "test_bce": float(arrays["test_loss"][ids].mean()),
                        "recall_positive": float(
                            arrays["test_recall_positive"][ids].mean()
                        ),
                        "recall_negative": float(
                            arrays["test_recall_negative"][ids].mean()
                        ),
                    }
                )
                selected[(noise, size, method)] = {
                    "arrays": arrays,
                    "ids": ids,
                    "accuracy": accuracy,
                    "path": path,
                }
    results = Path("experiments/dynamic_signal_v1/results/v2")
    results.mkdir(parents=True, exist_ok=True)
    with (results / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    primary = json.loads((root / "primary_selection.json").read_text())
    endpoint = plan["primary_endpoint"]
    key = (endpoint["noise_id"], endpoint["support_size"])
    structural, competitor = (
        primary["structural"]["chosen"],
        primary["competitor"]["chosen"],
    )
    difference = (
        selected[(*key, structural)]["accuracy"]
        - selected[(*key, competitor)]["accuracy"]
    )
    interval = np.quantile(difference[bootstrap].mean(-1), [0.025, 0.975])
    evidence = {
        "primary_methods": primary,
        "difference_pp": float(100 * difference.mean()),
        "paired_bootstrap_95_pp": (100 * interval).tolist(),
        "per_task_difference_pp": (100 * difference).tolist(),
        "task_count": len(difference),
        "all_results": rows,
    }
    (results / "summary.json").write_text(json.dumps(evidence, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    chart_methods = [
        "ar",
        "frequency",
        "consensus",
        "adam",
        "sgd",
        "transfer",
        "mask_unshared",
        "wsr",
        "oracle",
    ]
    colors = plt.get_cmap("tab10").colors
    for noise, ax in enumerate(axes):
        for number, method in enumerate(chart_methods):
            data = [
                row
                for row in rows
                if row["sigma"] == plan["noise"][noise] and row["method"] == method
            ]
            ax.plot(
                [row["support_size"] for row in data],
                [100 * row["accuracy"] for row in data],
                "o-",
                label=LABELS[method],
                color=colors[number],
                alpha=0.9,
            )
        bayes = json.loads(
            (root / "target_fit" / f"noise_{noise}_n128/bayes_audit.json").read_text()
        )["mean"]
        ax.axhline(100 * bayes, ls=":", color="black", label="Байес")
        ax.set_xscale("log", base=2)
        ax.set_xticks(plan["target_sizes"], [str(n) for n in plan["target_sizes"]])
        ax.set_title(f"Шум σ = {plan['noise'][noise]:.2f}")
        ax.set_xlabel("Обучающих примеров (+1024 query)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Точность на новых задачах, %")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output / "comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    # Primary comparison includes the stronger, fourfold-longer optimization controls.
    fig, ax = plt.subplots(figsize=(11, 5))
    displayed = [
        "ar",
        "frequency",
        "consensus",
        "mask_unshared",
        "transfer",
        "wsr",
        "adam",
        "sgd",
        "adam_long",
        "sgd_long",
        "oracle",
    ]
    data = [
        next(
            row
            for row in rows
            if row["sigma"] == 0.25
            and row["support_size"] == 2048
            and row["method"] == method
        )
        for method in displayed
    ]
    means = np.array([row["accuracy"] for row in data]) * 100
    error = (
        np.array(
            [
                [row["accuracy"] - row["accuracy_low"] for row in data],
                [row["accuracy_high"] - row["accuracy"] for row in data],
            ]
        )
        * 100
    )
    ax.barh(
        np.arange(len(data)),
        means,
        xerr=error,
        color=[
            "#178c83" if method in ["ar", "frequency", "consensus"] else "#6d819e"
            for method in displayed
        ],
    )
    ax.set_yticks(np.arange(len(data)), [LABELS[m] for m in displayed])
    ax.invert_yaxis()
    ax.set_xlim(50, 100)
    ax.set_xlabel("Точность, %, 95% bootstrap по 16 задачам")
    ax.set_title("Заранее выбранная проверка: σ=0,25; 2048 примеров")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "primary.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for method in ["ar", "frequency", "mask_unshared", "transfer", "adam", "sgd"]:
        result = selected[(1, 2048, method)]
        path = result["path"]
        history_path = path.parent / (path.stem + "_history.json.gz")
        with gzip.open(history_path, "rt") as stream:
            history = json.load(stream)
        steps = [row["step"] for row in history]
        for ax, split, metric in [
            (axes[0], "train", "loss"),
            (axes[1], "query", "loss"),
            (axes[2], "query", "recall_positive"),
        ]:
            values = [
                np.array(row[split][metric])[result["ids"]].mean() for row in history
            ]
            ax.plot(steps, values, label=LABELS[method])
    for ax, title in zip(
        axes,
        [
            "Обучающая BCE (мини-батч)",
            "Проверочная BCE",
            "Recall положительного класса",
        ],
    ):
        ax.set_title(title)
        ax.set_xlabel("Шаги")
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "learning.png", dpi=180)
    plt.close(fig)
    # Task and structure visualization uses hidden teacher information only for audit.
    audit = np.load(root / "target/teacher_audit.npz")
    permutation = audit["permutation"]
    direction = audit["directions"][0]
    x = np.random.default_rng(401).normal(scale=0.25, size=(16, 8))
    x[5] += direction
    observed = np.empty(128)
    observed[permutation] = x.flatten()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].bar(np.arange(8), direction, color="#178c83")
    axes[0].set_title("Сигнал одной задачи")
    axes[0].set_xlabel("Координата внутри группы")
    im = axes[1].imshow(x, aspect="auto", cmap="RdBu_r")
    axes[1].set_title("Пример до перемешивания")
    axes[1].set_ylabel("Группа")
    axes[1].set_xlabel("Координата")
    fig.colorbar(im, ax=axes[1])
    axes[2].plot(observed, color="#354e75")
    axes[2].set_title("128 входов, доступных MLP")
    axes[2].set_xlabel("Наблюдаемая координата")
    fig.tight_layout()
    fig.savefig(output / "task.png", dpi=180)
    plt.close(fig)
    oracle = np.zeros((128, 32), int)
    for group in range(16):
        coords = permutation[group * 8 : (group + 1) * 8]
        oracle[coords, 2 * group] = np.arange(1, 9)
        oracle[coords, 2 * group + 1] = np.arange(9, 17)
    fig, axes = plt.subplots(1, 5, figsize=(14, 6), sharey=True)
    for ax, method in zip(
        axes, ["oracle", "consensus", "ar", "frequency", "random_topology"]
    ):
        if method == "oracle":
            indices = oracle
        else:
            result = selected[(1, 2048, method)]
            indices = (
                result["arrays"]["indices"][result["ids"][0]]
                if "indices" in result["arrays"]
                else oracle
            )
        support = (indices != 0).astype(int)
        target = (oracle != 0).astype(int)
        old, new = linear_sum_assignment(-(support.T @ target))
        order = np.empty(32, int)
        order[new] = old
        ax.imshow(
            support[permutation][:, order],
            aspect="auto",
            cmap="Greys",
            interpolation="nearest",
        )
        ax.set_title(LABELS[method], fontsize=10)
        ax.set_xlabel("Скрытый нейрон")
    axes[0].set_ylabel("Вход (истинный порядок только для рисунка)")
    fig.tight_layout()
    fig.savefig(output / "indexators.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 7, figsize=(17, 6), sharey=True)
    for ax, method in zip(
        axes, ["ar", "frequency", "mask_unshared", "transfer", "adam", "sgd", "wsr"]
    ):
        result = selected[(1, 2048, method)]
        arrays = result["arrays"]
        weights = (
            arrays["parameter_0"][result["ids"][0]]
            if "parameter_0" in arrays
            else arrays["selected_parameter_0"][0]
        )
        magnitude = np.abs(weights)
        score = magnitude.T @ (oracle != 0).astype(float)
        score /= np.maximum(magnitude.sum(0)[:, None], 1e-9)
        old, new = linear_sum_assignment(-score)
        order = np.empty(32, int)
        order[new] = old
        image = ax.imshow(
            magnitude[permutation][:, order] / max(magnitude.max(), 1e-9),
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="magma",
            interpolation="nearest",
        )
        ax.set_title(LABELS[method], fontsize=9)
        ax.set_xlabel("Нейрон")
    axes[0].set_ylabel("Вход (истинный порядок только для рисунка)")
    fig.suptitle("Найденные веса первого слоя: |W| / max|W| каждой модели")
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    fig.subplots_adjust(wspace=0.15, right=0.94, top=0.88)
    fig.savefig(output / "weights.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(
        json.dumps(
            {key: value for key, value in evidence.items() if key != "all_results"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
