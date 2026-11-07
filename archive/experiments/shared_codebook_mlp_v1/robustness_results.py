"""Summarize completed robustness runs; teacher access is diagnostic only."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
ASSETS = REPO / "mds/assets/2026-10-28/mlp_robustness"
RUNS = [
    ("tanh_continuous_v1", "MLP с tanh"),
    ("clean_relu_continuous_v1", "MLP с ReLU без шума"),
    ("noisy_relu_continuous_v1", "MLP с ReLU и шумом"),
]
LABELS = {
    "oracle": "Заданный истинный индексатор",
    "consensus": "Общий индексатор из банка",
    "bank_sample": "Целая структура из банка",
    "vae_sample": "VAE без обучения z",
    "vae_z": "VAE + z",
    "untrained_vae_z": "Необученный VAE + z",
    "random": "Случайный индексатор",
    "shuffled_consensus": "Перемешанный общий индексатор",
    "dense_adam": "Плотный Adam",
    "dense_sgd": "Плотный SGD с моментом",
    "dense_transfer": "Градиентный перенос",
}


def align(indices, reference):
    indices = indices.reshape(32, 16)
    reference = reference.reshape(32, 16)
    signatures = indices[:, None, :] == indices[None, :, :]
    target = reference[:, None, :] == reference[None, :, :]
    cost = np.not_equal(
        signatures.reshape(1024, 16)[:, :, None],
        target.reshape(1024, 16)[:, None, :],
    ).sum(0)
    old, new = linear_sum_assignment(cost)
    order = np.empty(16, dtype=int)
    order[new] = old
    indices = indices[:, order]
    votes = np.zeros((32, 32), dtype=int)
    np.add.at(votes, (indices.flatten(), reference.flatten()), 1)
    old, new = linear_sum_assignment(-votes)
    mapping = np.empty(32, dtype=int)
    mapping[old] = new
    return mapping[indices]


def predict(inputs, weights, activation):
    hidden = inputs @ weights
    hidden = np.tanh(hidden) if activation == "tanh" else np.maximum(hidden, 0)
    return hidden.sum(-1) / 4


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    summaries, metrics, examples = {}, [], {}
    for version, title in RUNS:
        out = ROOT / "runs" / version
        if not (out / "evaluate_complete.json").exists():
            raise RuntimeError(f"Run has not completed: {out}")
        protocol = json.loads((out / "evaluation_protocol.json").read_text())
        frozen = json.loads((out / "evaluation_frozen.json").read_text())
        for filename, expected in frozen.items():
            actual = hashlib.sha256((out / filename).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Frozen input changed: {out / filename}")
        source = np.load(out / "bank.npz")
        bank = np.load(out / "structure_bank.npz")
        teacher = np.load(out / "teacher.npz")["indexator"]
        aligned = align(bank["consensus"], teacher)
        summary = {
            "title": title,
            "protocol": protocol,
            "source_query_best_mean": float(
                source["quality"].reshape(6, 4).min(1).mean()
            ),
            "consensus_correct_of_512": int((aligned == teacher).sum()),
            "methods": {},
        }
        tasks, starts = protocol["tasks"], protocol["starts"]
        examples[version] = {"teacher": teacher, "consensus": aligned}
        for method in protocol["methods"]:
            data = np.load(out / f"{method}.npz")
            choices = data["query_nmse"].reshape(tasks, starts).argmin(1)
            selected_ids = np.arange(tasks) * starts + choices
            selected = data["test_nmse"][selected_ids]
            row = {
                "method": method,
                "label": LABELS[method],
                "selected_mean_nmse": float(selected.mean()),
                "selected_median_nmse": float(np.median(selected)),
                "selected_task_nmse": selected.tolist(),
                "selected_successes": int((selected < 0.001).sum()),
                "successful_starts": int((data["test_nmse"] < 0.001).sum()),
                "all_starts_mean_nmse": float(data["test_nmse"].mean()),
            }
            summary["methods"][method] = row
            metrics.append(
                {
                    "version": version,
                    **{
                        key: value
                        for key, value in row.items()
                        if not isinstance(value, list)
                    },
                }
            )
            index = int(selected_ids[0])
            indices = data["indices"][index]
            values = data["values"][index]
            weights = (
                values.reshape(32, 16)
                if method.startswith("dense")
                else values[indices].reshape(32, 16)
            )
            examples[version][method] = (
                weights if method.startswith("dense") else align(indices, teacher)
            )
            examples[version][method + "_weights"] = weights
        summaries[version] = summary
        (out / "robustness_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )

    with (ASSETS / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)
    (ASSETS / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False)
    )

    methods = list(LABELS)
    fig, axes = plt.subplots(1, len(RUNS), figsize=(18, 6), sharey=True)
    for axis, (version, title) in zip(axes, RUNS):
        rows = summaries[version]["methods"]
        errors = [max(rows[m]["selected_mean_nmse"], 1e-12) for m in methods]
        colors = [
            "#159e8c" if m in ["vae_z", "consensus"] else "#637f9b" for m in methods
        ]
        axis.barh(np.arange(len(methods)), errors, color=colors)
        axis.set_xscale("log")
        axis.set_xlim(1e-12, 3)
        axis.set_title(title)
        axis.set_xlabel("Средняя тестовая NMSE; меньше — лучше")
        axis.grid(axis="x", alpha=0.2)
        for i, error in enumerate(errors):
            axis.text(
                min(error * 1.5, 1.3),
                i,
                f"{rows[methods[i]]['selected_mean_nmse']:.3g}",
                va="center",
            )
    axes[0].set_yticks(np.arange(len(methods)), [LABELS[m] for m in methods])
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(ASSETS / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    shown = ["teacher", "consensus", "vae_z", "random", "dense_adam", "dense_sgd"]
    fig, axes = plt.subplots(len(RUNS), len(shown), figsize=(15, 12))
    for row, (version, title) in enumerate(RUNS):
        for column, method in enumerate(shown):
            axis = axes[row, column]
            matrix = examples[version][method]
            if method.startswith("dense"):
                limit = np.abs(matrix).max()
                axis.imshow(
                    matrix,
                    cmap="coolwarm",
                    vmin=-limit,
                    vmax=limit,
                    aspect="auto",
                    interpolation="nearest",
                )
            else:
                axis.imshow(
                    matrix,
                    cmap="turbo",
                    vmin=0,
                    vmax=31,
                    aspect="auto",
                    interpolation="nearest",
                )
            axis.set_title(
                "Истинный индексатор" if method == "teacher" else LABELS[method],
                fontsize=9,
            )
            if column == 0:
                axis.set_ylabel(title + "\nВходы")
            axis.set_xlabel("Скрытые нейроны")
    fig.tight_layout()
    fig.savefig(ASSETS / "indexators.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for version, title in RUNS:
        fig, axes = plt.subplots(3, 4, figsize=(12, 12))
        for axis, method in zip(axes.flat, ["teacher"] + methods):
            matrix = examples[version][method]
            if method.startswith("dense"):
                limit = np.abs(matrix).max()
                axis.imshow(
                    matrix,
                    cmap="coolwarm",
                    vmin=-limit,
                    vmax=limit,
                    aspect="auto",
                    interpolation="nearest",
                )
            else:
                axis.imshow(
                    matrix,
                    cmap="turbo",
                    vmin=0,
                    vmax=31,
                    aspect="auto",
                    interpolation="nearest",
                )
            axis.set_title(
                "Истинный индексатор" if method == "teacher" else LABELS[method],
                fontsize=9,
            )
            axis.set_xlabel("Скрытые нейроны")
            axis.set_ylabel("Входы")
        fig.suptitle(title + ": все методы, первая новая функция")
        fig.tight_layout()
        fig.savefig(ASSETS / f"indexators_{version}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    rng = np.random.default_rng(282050)
    origin = rng.normal(size=32)
    direction = rng.normal(size=32)
    direction /= np.linalg.norm(direction)
    coordinate = np.linspace(-5, 5, 400)
    inputs = origin + coordinate[:, None] * direction
    fig, axes = plt.subplots(1, len(RUNS), figsize=(17, 4))
    for axis, (version, title) in zip(axes, RUNS):
        protocol = summaries[version]["protocol"]
        saved = np.load(ROOT / "runs" / version / "target_data.npz")
        teacher_weights = saved["c"][0][examples[version]["teacher"]]
        axis.plot(
            coordinate,
            predict(inputs, teacher_weights, protocol["activation"]),
            color="black",
            linewidth=2.5,
            label="Истинная функция",
        )
        for method, color in [("vae_z", "#159e8c"), ("dense_adam", "#c26739")]:
            weights = examples[version][method + "_weights"]
            axis.plot(
                coordinate,
                predict(inputs, weights, protocol["activation"]),
                color=color,
                linestyle="--",
                label=LABELS[method],
            )
        axis.set_title(title + ": сечение первой новой функции")
        axis.set_xlabel("t в x = x₀ + tv")
        axis.set_ylabel("Выход сети")
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSETS / "tasks.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    for version, _ in RUNS:
        summary = summaries[version]
        print(version, "recovered", summary["consensus_correct_of_512"])
        for method, row in summary["methods"].items():
            print(method, row["selected_mean_nmse"], row["selected_successes"])


if __name__ == "__main__":
    main()
