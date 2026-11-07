"""Summarize the frozen sine experiment and export standalone figures."""

import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

BASE = Path(__file__).resolve().parent
RUN = BASE / "results/run_20261103"
ASSETS = BASE.parents[1] / "final_mds/assets/2026-11-03"
LABELS = {
    "dense_sgd": "Плотный SGD",
    "dense_adam": "Плотный Adam",
    "ar": "Текущий AR + SGD",
    "ar_budget2": "AR, ровно 2 связи + SGD",
    "frequency": "Независимые частоты + SGD",
    "uniform": "Случайные 2 связи + SGD",
    "replay": "Маска из банка + SGD",
    "oracle": "Правильная маска + SGD",
}
COLORS = {
    "dense_sgd": "#d59127",
    "dense_adam": "#b94c55",
    "ar": "#157e72",
    "ar_budget2": "#48a79a",
    "frequency": "#82a65c",
    "uniform": "#a5aab0",
    "replay": "#7564a8",
    "oracle": "#65758b",
}


def save_figure(fig, name):
    fig.savefig(ASSETS / f"{name}.svg", bbox_inches="tight")
    fig.savefig(ASSETS / f"{name}.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def interval(values, rng):
    task_means = values.mean(1)
    draws = rng.integers(len(task_means), size=(10000, len(task_means)))
    return np.quantile(task_means[draws].mean(1), [0.025, 0.975]).tolist()


def main():
    if not (RUN / "COMPLETE.json").exists():
        raise RuntimeError("The declared experiment has not completed")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "path",
        }
    )
    plan = json.loads((RUN / "protocol.json").read_text())
    with np.load(RUN / "bank.npz") as raw:
        bank = dict(raw)
    training = bank["masks"][~bank["validation"]]
    train_set = set(row.tobytes() for row in training.astype(np.uint8))
    rng = np.random.default_rng(2026110307)
    summary = {
        "bank": json.loads((RUN / "bank_summary.json").read_text()),
        "settings": json.loads((RUN / "selection_frozen.json").read_text()),
        "conditions": {},
        "inference": "Task-cluster bootstrap conditional on one bank.",
    }
    tables, curves, pairs, proposals = [], [], {}, {}
    for condition in ["novel", "familiar"]:
        tasks = json.loads((RUN / condition / "tasks.json").read_text())
        truth = np.zeros((len(tasks), 8, 2), dtype=np.uint8)
        for t, task in enumerate(tasks):
            truth[t, task["rows"], [0, 1]] = 1
        summary["conditions"][condition] = {
            "task_count": len(tasks),
            "methods": {},
        }
        success_arrays = {}
        for method in plan["methods"]:
            outputs, seen, all_masks, hist_train, hist_pop = [], [], [], [], []
            prefixes = {n: [] for n in [8, 16, 32, 40, 48, 64]}
            seconds = 0.0
            for index in range(3):
                path = RUN / condition / f"{method}_seed{index}.npz"
                with np.load(path) as raw:
                    data = dict(raw)
                outputs.append(data["selected_population"])
                proposal_count = (
                    64
                    if method in ["dense_sgd", "dense_adam", "oracle"]
                    else 32
                )
                masks = data["masks"][:, :, :proposal_count]
                seen.append(
                    (masks == truth[:, None, None]).all((-2, -1)).any(2)
                )
                all_masks.append(masks.reshape(-1, 16).astype(np.uint8))
                seconds += float(data["seconds"])
                ti, si = np.indices(data["selected"].shape)
                for n in prefixes:
                    choice = data["query"][:, :, :n].argmin(2)
                    prefixes[n].append(
                        data["population"][ti, si, choice] < 0.1
                    )
                with gzip.open(
                    path.with_name(path.stem + "_history.json.gz"), "rt"
                ) as stream:
                    history = json.load(stream)
                hist_train.append(
                    [
                        np.asarray(point["train"])[
                            ti, si, data["selected"]
                        ].mean()
                        for point in history
                    ]
                )
                hist_pop.append(
                    [
                        np.asarray(point["population"])[
                            ti, si, data["selected"]
                        ].mean()
                        for point in history
                    ]
                )
            values = np.concatenate(outputs, axis=1)
            success = values < 0.1
            success_arrays[method] = success
            masks = np.concatenate(all_masks)
            proposals[condition, method] = masks
            exact_seen = np.concatenate(seen, axis=1)
            counts = masks.sum(1)
            entry = {
                "success": int(success.sum()),
                "searches": success.size,
                "rate": float(success.mean()),
                "mean_mse": float(values.mean()),
                "task_cluster_95": interval(success, rng),
                "per_task_success_rate": success.mean(1).tolist(),
                "per_seed_success_rate": [
                    float((v < 0.1).mean()) for v in outputs
                ],
                "exact_target_mask_seen_rate": float(exact_seen.mean()),
                "mean_active_edges": float(counts.mean()),
                "two_edge_proposal_fraction": float((counts == 2).mean()),
                "source_mask_proposal_fraction": float(
                    np.mean([row.tobytes() in train_set for row in masks])
                ),
                "unique_proposals": len(np.unique(masks, axis=0)),
                "fit_seconds": seconds,
                "prefix_success": {
                    n: float(np.concatenate(v, axis=1).mean())
                    for n, v in prefixes.items()
                },
            }
            summary["conditions"][condition]["methods"][method] = entry
            tables.append(
                {
                    "condition": condition,
                    "method": method,
                    **{
                        key: entry[key]
                        for key in [
                            "success",
                            "searches",
                            "rate",
                            "mean_mse",
                            "exact_target_mask_seen_rate",
                            "mean_active_edges",
                            "two_edge_proposal_fraction",
                            "source_mask_proposal_fraction",
                        ]
                    },
                }
            )
            steps = [point["step"] for point in history]
            for metric, histories in [
                ("train", hist_train),
                ("population", hist_pop),
            ]:
                for step, value in zip(steps, np.mean(histories, axis=0)):
                    curves.append(
                        {
                            "condition": condition,
                            "method": method,
                            "metric": metric,
                            "step": step,
                            "value": float(value),
                        }
                    )
        difference = (
            success_arrays["ar"].astype(float) - success_arrays["dense_adam"]
        )
        pairs[condition] = {
            "ar_minus_adam": float(difference.mean()),
            "task_cluster_95": interval(difference, rng),
        }
    summary["paired_comparisons"] = pairs
    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 5.2), constrained_layout=True
    )
    for ax, condition, title in zip(
        axes,
        ["novel", "familiar"],
        [
            "Невиданные сочетания: 8 новых функций",
            "Знакомые структуры: 16 новых функций",
        ],
    ):
        for i, method in enumerate(plan["methods"]):
            entry = summary["conditions"][condition]["methods"][method]
            rate = 100 * entry["rate"]
            lo, hi = np.array(entry["task_cluster_95"]) * 100
            ax.barh(i, rate, color=COLORS[method], height=0.65)
            ax.errorbar(
                rate,
                i,
                xerr=[[rate - lo], [hi - rate]],
                color="#24364b",
                capsize=3,
            )
            ax.text(rate + 1.2, i + 0.27, f"{rate:.2f}%", fontsize=8)
        ax.set_yticks(
            range(len(plan["methods"])), [LABELS[m] for m in plan["methods"]]
        )
        ax.set(
            xlabel="Успешные поиски (population MSE < 0.1), %",
            xlim=(0, 112),
            title=title,
        )
        ax.invert_yaxis()
    save_figure(fig, "sine_current_results")

    fig, axes = plt.subplots(
        1, 3, figsize=(12.5, 4.4), constrained_layout=True
    )
    matrices = {}
    for ax, name, masks in zip(
        axes,
        ["Исходный банк", "Текущий AR: предложения", "Независимые частоты"],
        [training, proposals["novel", "ar"], proposals["novel", "frequency"]],
    ):
        matrix = np.zeros((4, 4))
        for a, b in itertools.product(range(4), repeat=2):
            mask = np.zeros((8, 2), dtype=np.uint8)
            mask[a, 0], mask[b + 4, 1] = 1, 1
            matrix[a, b] = 100 * np.all(masks == mask.ravel(), axis=1).mean()
        matrices[name] = matrix.tolist()
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=25)
        for a, b in itertools.product(range(4), repeat=2):
            ax.text(
                b,
                a,
                f"{matrix[a, b]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[a, b] > 13 else "#24364b",
                fontsize=9,
            )
            if (b - a) % 4 == 3:
                ax.add_patch(
                    Rectangle(
                        (b - 0.48, a - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#c65a59",
                        lw=2,
                    )
                )
        ax.set(
            title=name + f"\nМасса показанных 16 масок: {matrix.sum():.2f}%",
            xlabel="Координата второго выхода",
            ylabel="Координата первого выхода",
        )
        ax.set_xticks(range(4), range(5, 9))
        ax.set_yticks(range(4), range(1, 5))
    fig.colorbar(
        im, ax=axes, shrink=0.8, label="Доля среди всех предложений, %"
    )
    save_figure(fig, "sine_current_structures")
    summary["proposal_pair_matrices_percent"] = matrices

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for row, condition in enumerate(["novel", "familiar"]):
        for col, metric in enumerate(["train", "population"]):
            ax = axes[row, col]
            for method in ["dense_sgd", "dense_adam", "ar", "oracle"]:
                points = [
                    point
                    for point in curves
                    if point["condition"] == condition
                    and point["method"] == method
                    and point["metric"] == metric
                ]
                ax.plot(
                    [p["step"] for p in points],
                    [p["value"] for p in points],
                    color=COLORS[method],
                    label=LABELS[method],
                    lw=2,
                )
            prefix = (
                "Невиданные сочетания" if row == 0 else "Знакомые структуры"
            )
            ax.set(
                title=prefix
                + (": train MSE" if col == 0 else ": population MSE"),
                xlabel="Шаг выбранной численной модели",
                ylabel="MSE",
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    save_figure(fig, "sine_current_learning")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for index in range(3):
        history = json.loads((RUN / f"prior_{index}_history.json").read_text())
        for ax, metric in zip(axes, ["train_nll", "validation_nll"]):
            ax.plot(
                [h["step"] for h in history],
                [h[metric] for h in history],
                label=f"seed {plan['prior_seeds'][index]}",
                lw=1.5,
            )
            ax.set(xlabel="Шаг обучения MADE", ylabel="NLL на позицию")
            ax.grid(alpha=0.2)
    axes[0].set_title("Обучающие маски")
    axes[1].set_title("Отложенные исходные карточки")
    axes[0].legend(fontsize=8)
    save_figure(fig, "sine_current_prior")
    with (RUN / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
        writer.writeheader()
        writer.writerows(tables)
    with (ASSETS / "sine_current_curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    (RUN / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    hashes = {
        str(path.relative_to(BASE.parents[1])): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in RUN.rglob("*")
        if path.is_file()
    }
    (ASSETS / "sine_current_sources.json").write_text(
        json.dumps(hashes, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"bank": summary["bank"], "rows": tables, "paired": pairs},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
