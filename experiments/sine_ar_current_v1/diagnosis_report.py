"""Export saved threshold diagnostics and fresh-function comparisons."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

BASE = Path(__file__).resolve().parent
RUN = BASE / "results/diagnosis_20261103"
ASSETS = BASE.parents[1] / "mds/assets/2026-11-03/sine_ar_diagnosis"
LABELS = {
    "strict_t2": "Порог 0.1, AR T=2",
    "strict_retuned": "Порог 0.1, настроенная T",
    "relaxed_t2": "Мягкий порог, AR T=2",
    "relaxed_selected": "Мягкий порог, настроенная T",
    "relaxed_replay": "Маска из мягкого банка",
    "strict_frequency": "Частоты строгого банка",
    "relaxed_frequency": "Частоты мягкого банка",
    "uniform": "Случайные 2 связи",
    "dense_sgd": "Плотный SGD",
    "dense_adam": "Плотный Adam",
    "oracle": "Правильная маска + SGD",
}


def figure(fig, name):
    fig.savefig(ASSETS / f"{name}.svg", bbox_inches="tight")
    fig.savefig(ASSETS / f"{name}.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def main():
    if not (RUN / "COMPLETE.json").exists():
        raise RuntimeError("Evaluation has not completed")
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "path",
        }
    )
    audit = json.loads((RUN / "bank_audit.json").read_text())
    selection = json.loads((RUN / "selection_frozen.json").read_text())
    calibration = json.loads((RUN / "calibration.json").read_text())
    cases = json.loads((RUN / "evaluation_cases.json").read_text())
    distributions = []
    for path in sorted(RUN.glob("threshold_*/distribution_*.json")):
        distributions.extend(json.loads(path.read_text()))
    banks = {}
    for _, _, threshold, _ in cases:
        with np.load(RUN / f"threshold_{threshold:g}/bank.npz") as saved:
            masks = saved["masks"][~saved["validation"]]
        banks[threshold] = {row.tobytes() for row in masks.astype(np.uint8)}
    summary = {
        "selection": selection,
        "bank_audit": audit,
        "evaluation": {},
        "paired_differences": {},
        "scope": (
            "Development diagnosis; one source bank, three priors, fresh "
            "numerical functions of an already explored structural split."
        ),
    }
    tables = []
    rng = np.random.default_rng(261103991)
    for condition in ["novel", "familiar"]:
        scores = {}
        summary["evaluation"][condition] = {}
        for name, method, threshold, temperature in cases:
            seeds, seen, new_selected, new_success = [], [], [], []
            for index in range(3):
                with np.load(RUN / condition / f"{name}_seed{index}.npz") as r:
                    risk = r["selected_population"]
                    seeds.append(risk)
                    ti, si = np.indices(risk.shape)
                    masks = r["masks"][ti, si, r["selected"]]
                    absent = np.asarray(
                        [
                            row.tobytes() not in banks[threshold]
                            for row in masks.reshape(-1, 16).astype(np.uint8)
                        ]
                    ).reshape(risk.shape)
                    new_selected.append(absent)
                    new_success.append(absent & (risk < 0.1))
                    specs = json.loads(
                        (RUN / condition / "tasks.json").read_text()
                    )
                    truth = np.zeros((len(specs), 8, 2), dtype=np.uint8)
                    for t, spec in enumerate(specs):
                        truth[t, spec["rows"], [0, 1]] = 1
                    seen.append(
                        (r["masks"][:, :, :32] == truth[:, None, None])
                        .all((-2, -1))
                        .any(2)
                    )
            risk = np.concatenate(seeds, axis=1)
            success = risk < 0.1
            scores[name] = success.astype(float)
            row = {
                "condition": condition,
                "method": name,
                "threshold": threshold,
                "temperature": temperature,
                "successes": int(success.sum()),
                "searches": success.size,
                "success_rate": float(success.mean()),
                "mean_mse": float(risk.mean()),
                "exact_target_mask_in_32": float(
                    np.concatenate(seen, axis=1).mean()
                ),
                "selected_mask_absent_from_bank": float(
                    np.concatenate(new_selected, axis=1).mean()
                ),
                "success_with_mask_absent_from_bank": float(
                    np.concatenate(new_success, axis=1).mean()
                ),
                "per_task_success": success.mean(1).tolist(),
                "per_seed_success": [float((v < 0.1).mean()) for v in seeds],
            }
            summary["evaluation"][condition][name] = row
            tables.append(
                {k: v for k, v in row.items() if not isinstance(v, list)}
            )
        summary["paired_differences"][condition] = {}
        for control in [
            "strict_t2",
            "strict_retuned",
            "relaxed_replay",
            "dense_adam",
        ]:
            difference = (scores["relaxed_selected"] - scores[control]).mean(1)
            draws = rng.integers(
                len(difference), size=(10000, len(difference))
            )
            interval = np.quantile(difference[draws].mean(1), [0.025, 0.975])
            summary["paired_differences"][condition][control] = {
                "difference": float(difference.mean()),
                "task_bootstrap_95": interval.tolist(),
            }
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (RUN / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
        writer.writeheader()
        writer.writerows(tables)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    rows = audit["thresholds"]
    x = [r["threshold"] for r in rows]
    axes[0].plot(x, [r["accepted"] for r in rows], "o-", color="#157e72")
    axes[0].set(
        yscale="log", xlabel="Порог source query MSE", ylabel="Принято попыток"
    )
    axes[1].plot(
        x,
        [r["unique_train_masks"] for r in rows],
        "o-",
        color="#157e72",
        label="Разные маски",
    )
    axes[1].plot(
        x,
        [r["effective_masks"] for r in rows],
        "s--",
        color="#d59127",
        label="Эффективное число",
    )
    axes[1].set(
        xlabel="Порог source query MSE", ylabel="Разнообразие train-банка"
    )
    axes[1].legend()
    for ax in axes:
        ax.axvline(
            selection["relaxed"]["threshold"], color="#b94c55", linestyle=":"
        )
        ax.grid(alpha=0.2)
    figure(fig, "bank_threshold")

    thresholds = sorted({r["threshold"] for r in calibration})
    temperatures = sorted({r["temperature"] for r in calibration})
    matrix = np.asarray(
        [
            [
                next(
                    r["mean_query"]
                    for r in calibration
                    if r["threshold"] == q and r["temperature"] == t
                )
                for t in temperatures
            ]
            for q in thresholds
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    plot = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
    ax.set(
        xticks=range(len(temperatures)),
        xticklabels=temperatures,
        yticks=range(len(thresholds)),
        yticklabels=thresholds,
        xlabel="Температура AR",
        ylabel="Порог банка",
        title="Средняя query MSE после SGD на калибровочных задачах ↓",
    )
    for i, j in np.ndindex(matrix.shape):
        ax.text(
            j,
            i,
            f"{matrix[i, j]:.3f}",
            ha="center",
            va="center",
                color="white" if matrix[i, j] > 0.275 else "black",
            fontsize=9,
        )
    best = selection["overall"]
    ax.add_patch(
        Rectangle(
            (
                temperatures.index(best["temperature"]) - 0.5,
                thresholds.index(best["threshold"]) - 0.5,
            ),
            1,
            1,
            fill=False,
            edgecolor="#e94f37",
            linewidth=3,
        )
    )
    fig.colorbar(plot, ax=ax, shrink=0.8)
    figure(fig, "calibration")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for threshold, label, color in [
        (0.1, "Строгий банк", "#b94c55"),
        (selection["relaxed"]["threshold"], "Мягкий банк", "#157e72"),
    ]:
        for ax, metric, ylabel in zip(
            axes,
            ["two_edge_mass", "novel_exact_hit_32"],
            [
                "Вероятность ровно двух связей, %",
                "Точная новая пара хотя бы раз за 32 предложения, %",
            ],
        ):
            temps = sorted({r["temperature"] for r in distributions})
            values = [
                np.mean(
                    [
                        r[metric]
                        for r in distributions
                        if r["threshold"] == threshold
                        and r["temperature"] == t
                        and r["method"] == "ar"
                    ]
                )
                * 100
                for t in temps
            ]
            ax.plot(temps, values, "o-", label=label, color=color)
            ax.set(xscale="log", xlabel="Температура", ylabel=ylabel)
            ax.grid(alpha=0.2)
            ax.legend()
    figure(fig, "temperature")

    matrices, titles = [], []
    for threshold, temperature in [
        (0.1, 2),
        (
            selection["relaxed"]["threshold"],
            selection["relaxed"]["temperature"],
        ),
    ]:
        bank = next(
            r for r in audit["thresholds"] if r["threshold"] == threshold
        )
        matrices.append(np.asarray(bank["train_pair_counts"]) / bank["train"])
        titles.append(f"Банк, порог {threshold:g}")
        matrices.append(
            np.mean(
                [
                    r["pair_probabilities"]
                    for r in distributions
                    if r["threshold"] == threshold
                    and r["temperature"] == temperature
                    and r["method"] == "ar"
                ],
                axis=0,
            )
        )
        titles.append(f"AR: порог {threshold:g}, T={temperature:g}")
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5), layout="constrained")
    maximum = max(float(m.max()) for m in matrices) * 100
    for ax, matrix, title in zip(axes, matrices, titles):
        ax.imshow(matrix * 100, cmap="Blues", vmin=0, vmax=maximum)
        ax.set(
            xticks=range(4),
            yticks=range(4),
            xlabel="Вторая координата b",
            ylabel="Первая координата a",
            title=title,
        )
        for a, b in np.ndindex((4, 4)):
            value = matrix[a, b] * 100
            ax.text(
                b,
                a,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > maximum * 0.6 else "black",
                fontsize=9,
            )
            if (b - a) % 4 == 3:
                ax.add_patch(
                    Rectangle(
                        (b - 0.48, a - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#b94c55",
                        linewidth=2,
                    )
                )
    fig.suptitle(
        "Полные двухсвязные маски: % от всех масок банка / предложений AR\nКрасная рамка — сочетания, отложенные при выборе source-задач",
        fontsize=11,
    )
    figure(fig, "pair_probabilities")

    names = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), layout="constrained")
    for ax, condition, title in zip(
        axes,
        ["novel", "familiar"],
        ["Новые сочетания source-задач", "Знакомые сочетания, новые функции"],
    ):
        data = summary["evaluation"][condition]
        values = [data[name]["success_rate"] * 100 for name in names]
        colors = [
            "#157e72"
            if name.startswith("relaxed")
            else "#b94c55"
            if name.startswith("strict_t")
            else "#8c98a3"
            for name in names
        ]
        ax.barh(range(len(names)), values, color=colors)
        ax.set(
            yticks=range(len(names)),
            yticklabels=[LABELS[n] for n in names],
            xlim=(0, 120),
            xticks=[0, 25, 50, 75, 100],
            xlabel="Успешные поиски, % (MSE < 0.1)",
            title=title,
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        for i, name in enumerate(names):
            ax.text(
                values[i] + 1,
                i,
                f"{values[i]:.1f}% · {data[name]['successes']}",
                va="center",
                fontsize=9,
            )
    figure(fig, "fresh_functions")
    print(
        json.dumps(
            {
                "selection": selection,
                "comparison": tables,
                "paired": summary["paired_differences"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
