"""Build report figures from recorded runs; no model fitting or test suite."""

import csv
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PATTERN = ROOT / "pattern/runs/autoregressive_debug_20261101_v1"
ORACLE = ROOT / "pattern/runs/autoregressive_20261101_v2"
DSD = ROOT / "experiments/dynamic_signal_v1/results/bank_repeats"
SINE = ROOT / "archive/history/experiments/sine_mask_vae/results/v1"
SOURCES = {}
CURVES = []
COLORS = {
    "oracle": "#65758b",
    "sgd": "#d59127",
    "adam": "#b94c55",
    "ar": "#157e72",
    "mask_unshared": "#527db8",
    "consensus": "#7564a8",
    "frequency": "#82a65c",
}


def record(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    return path


def read_json(path):
    return json.loads(record(path).read_text())


def read_npz(path):
    with np.load(record(path)) as arrays:
        return dict(arrays)


def read_history(path):
    with gzip.open(record(path), "rt") as stream:
        return json.load(stream)


def save_figure(fig, name):
    fig.savefig(HERE / f"{name}.svg", bbox_inches="tight")
    fig.savefig(HERE / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def best_rows(arrays):
    return np.array(
        [
            ids[np.argmin(arrays["query_loss"][ids])]
            for task in np.unique(arrays["task_ids"])
            for ids in [np.flatnonzero(arrays["task_ids"] == task)]
        ]
    )


def pattern_figures():
    patterns = ["0010", "0100", "1011", "1101"]
    rows = np.arange(11)[:, None]
    columns = np.arange(8)[None, :]
    truth = np.where(
        (rows >= columns) & (rows < columns + 4), rows - columns + 1, 0
    )
    codes = np.array([0, -1, -1, 1, -1])
    template = np.vstack([codes[truth], np.full((1, 8), -3)])
    template = template / np.linalg.norm(template, axis=0)
    definitions = [
        ("oracle", ORACLE / "oracle.npz", "Оракул: заданный I"),
        ("sgd", PATTERN / "dense_sgd.npz", "Плотная сеть: SGD"),
        ("adam", PATTERN / "dense_adam.npz", "Плотная сеть: Adam"),
        ("ar", PATTERN / "ar_calibrated.npz", "Сгенерированный I"),
    ]
    fig, axes = plt.subplots(
        1, 4, figsize=(13.5, 4.9), constrained_layout=True
    )
    audit_fig, audit_axes = plt.subplots(
        2, 4, figsize=(13.5, 8), constrained_layout=True
    )
    loss_fig, loss_axes = plt.subplots(
        1, 2, figsize=(11, 4), constrained_layout=True
    )
    metrics, selected_indices, matrices = {}, {}, {}
    domain = (np.arange(2048)[:, None] >> np.arange(10, -1, -1)) & 1
    domain = domain * 2.0 - 1.0
    for col, (key, path, name) in enumerate(definitions):
        arrays = read_npz(path)
        chosen = np.array(
            [
                start + np.argmin(arrays["query_loss"][start : start + 32])
                for start in range(0, 128, 32)
            ]
        )
        row = chosen[0]
        w, bias, output, intercept = [
            arrays[f"parameter_{k}"][row] for k in range(4)
        ]
        incoming = np.vstack([w, bias[None]])
        normalized = incoming / np.maximum(
            np.linalg.norm(incoming, axis=0), 1e-12
        )
        original, target = linear_sum_assignment(-(normalized.T @ template))
        order = np.empty(8, dtype=int)
        order[target] = original
        before = np.maximum(domain @ w + bias, 0) @ output + intercept
        after = (
            np.maximum(domain @ w[:, order] + bias[order], 0) @ output[order]
            + intercept
        )
        scale = float(np.abs(w).max())
        matrices[key + "_original"] = w
        matrices[key + "_aligned"] = w[:, order]
        matrices[key + "_permutation"] = order
        selected_indices[key] = arrays["indices"][row][:, order]
        for ax, shown in [
            (axes[col], w[:, order]),
            (audit_axes[0, col], w),
            (audit_axes[1, col], w[:, order]),
        ]:
            heat = ax.imshow(shown / scale, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(np.arange(8), np.arange(1, 9))
            ax.set_yticks(np.arange(11), np.arange(1, 12))
            ax.set_xlabel("Скрытый нейрон")
            ax.set_ylabel("Координата входа" if col == 0 else "")
        axes[col].set_title(
            name
            + f"\nПолный домен: {100 * arrays['population_accuracy'][row]:.2f}%"
        )
        audit_axes[0, col].set_title(name)
        for ax in audit_axes[:, col]:
            ax.set_xlabel("Скрытый нейрон")
        metric = {
            "test_accuracy": float(arrays["test_accuracy"][chosen].mean()),
            "domain_accuracy": float(
                arrays["population_accuracy"][chosen].mean()
            ),
            "test_bce": float(arrays["test_loss"][chosen].mean()),
            "perfect_starts": int((arrays["population_accuracy"] == 1).sum()),
            "perfect_tasks": int(
                (arrays["population_accuracy"][chosen] == 1).sum()
            ),
            "selected_rows": chosen.tolist(),
            "heatmap_pattern": patterns[0],
            "heatmap_row": int(row),
            "column_order_zero_based": order.tolist(),
            "display_max_abs_weight": scale,
            "max_prediction_change_after_permutation": float(
                np.max(np.abs(before - after))
            ),
        }
        metrics[key] = metric
        history = read_history(path.with_name(path.stem + "_history.json.gz"))
        for ax, split in zip(loss_axes, ["train", "query"]):
            steps = [point["step"] for point in history]
            values = [
                np.array(point[split]["loss"])[chosen].mean()
                for point in history
            ]
            ax.plot(steps, values, label=name, color=COLORS[key], lw=2)
            for step, value in zip(steps, values):
                CURVES.append(["pattern", key, split, step, float(value)])
    axes[0].set_ylabel("Координата входа")
    fig.colorbar(
        axes[-1].images[0],
        ax=axes,
        shrink=0.8,
        label="Вес / max |вес| в соответствующей сети",
    )
    audit_axes[0, 0].set_ylabel("До выравнивания\nКоордината входа")
    audit_axes[1, 0].set_ylabel("После выравнивания\nКоордината входа")
    audit_fig.colorbar(
        heat, ax=audit_axes, shrink=0.8, label="Нормированный вес"
    )
    save_figure(fig, "pattern_weights")
    save_figure(audit_fig, "pattern_alignment")
    for ax, title in zip(loss_axes, ["Support: BCE", "Query: BCE"]):
        ax.set(title=title, xlabel="Шаг оптимизатора", ylabel="BCE")
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
    loss_axes[0].legend(fontsize=8)
    save_figure(loss_fig, "pattern_learning")
    index = selected_indices["ar"].copy()
    table = np.zeros((4, 4), dtype=int)
    for actual, expected in zip(index[truth > 0], truth[truth > 0]):
        if actual:
            table[actual - 1, expected - 1] += 1
    old, new = linear_sum_assignment(-table)
    mapping = np.arange(5)
    mapping[old + 1] = new + 1
    index = mapping[index]
    metrics["ar"]["aligned_indexator_exact"] = bool(
        np.array_equal(index, truth)
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.7), constrained_layout=True)
    cmap = ListedColormap(
        ["#f2f4f6", "#527db8", "#e7ae54", "#7d69a7", "#438c79"]
    )
    for ax, values, title in zip(
        axes,
        [truth, index],
        ["Аналитический индексатор", "Сгенерированный индексатор"],
    ):
        im = ax.imshow(values, cmap=cmap, vmin=-0.5, vmax=4.5)
        ax.set(title=title, xlabel="Нейрон", ylabel="Координата входа")
        ax.set_xticks(np.arange(8), np.arange(1, 9))
        ax.set_yticks(np.arange(11), np.arange(1, 12))
    fig.colorbar(
        im,
        ax=axes,
        ticks=np.arange(5),
        shrink=0.8,
        label="Номер общего коэффициента; 0 = нет связи",
    )
    save_figure(fig, "pattern_indexators")
    matrices["analytic_indices"] = truth
    matrices["generated_indices_aligned"] = index
    np.savez_compressed(HERE / "pattern_heatmap_data.npz", **matrices)
    metrics["debug_summary"] = read_json(PATTERN / "summary.json")
    return metrics


def sine_figures():
    snapshot = HERE / "sine_summary.json"
    provenance = HERE / "sine_summary_origin.json"
    if not snapshot.exists():
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        original = "paper_vae_solver/results/summary.json"
        content = subprocess.check_output(
            ["git", "show", f"{revision}:{original}"], cwd=ROOT
        )
        snapshot.write_bytes(content)
        provenance.write_text(
            json.dumps(
                {
                    "revision": revision,
                    "path": original,
                    "sha256": hashlib.sha256(content).hexdigest(),
                },
                indent=2,
            )
            + "\n"
        )
    source = read_json(snapshot)
    read_json(provenance)
    geometry = read_json(SINE / "geometry_population.json")
    task = read_json(SINE / "bank_manifest.json")[0]["task"]
    low, high, alpha = [
        np.array(task[key]) for key in ["low", "high", "small"]
    ]

    def kernel(a, b):
        return 0.5 * (
            np.exp(-0.5 * (a - b) ** 2) - np.exp(-0.5 * (a + b) ** 2)
        )

    freq = np.linspace(0.02, 6, 1200)
    covariance = alpha * kernel(freq[:, None], low) + kernel(
        freq[:, None], high
    )
    energy = (
        alpha**2 * kernel(low, low)
        + kernel(high, high)
        + 2 * alpha * kernel(low, high)
    )
    risk = energy - covariance**2 / kernel(freq[:, None], freq[:, None])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for j in range(2):
        axes[0].plot(
            freq,
            risk[:, j],
            lw=2,
            label=f"Выход {j + 1}: оптимальная амплитуда",
        )
    axes[0].axhline(0.1, color="#7b8794", ls=":", label="Порог MSE = 0.1")
    for key, label, color in [
        ("broad_low_frequency", "Плохой минимум", COLORS["adam"]),
        ("good_high_frequency", "Хороший минимум", COLORS["ar"]),
    ]:
        values = geometry[key]
        params = np.array(values["parameters"])
        w = params[:16].reshape(8, 2)
        t = w[task["rows"][0], 0]
        a = params[16]
        value = (
            a * a * kernel(t, t)
            - 2 * a * (alpha[0] * kernel(t, low[0]) + kernel(t, high[0]))
            + energy[0]
        )
        axes[0].scatter([t], [value], color=color, zorder=5, s=55)
        x = np.linspace(-3, 3, 1200)
        axes[1].plot(x, a * np.sin(t * x), color=color, label=label, lw=1.8)
    axes[1].plot(
        x,
        alpha[0] * np.sin(low[0] * x) + np.sin(high[0] * x),
        color="#24364b",
        alpha=0.8,
        ls="--",
        label="Учитель",
        lw=1.5,
    )
    axes[0].set(
        title="Два бассейна на правильной координате",
        xlabel="Частота t при w = t eᵣ",
        ylabel="Population MSE одного выхода",
    )
    axes[1].set(
        title="Учитель и стационарные решения: выход 1",
        xlabel="Значение активной координаты",
        ylabel="Предсказание",
    )
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    save_figure(fig, "sine_landscape")
    entries = {entry["method"]: entry for entry in source["matched_v2"]}
    definitions = [
        ("dense_sgd", "Плотный SGD", COLORS["sgd"]),
        ("dense_adam", "Плотный Adam", COLORS["adam"]),
        ("uniform_global", "Случайные 2 связи", "#a5aab0"),
        ("product", "Частоты банка + SGD", COLORS["frequency"]),
        ("vae", "Partwise VAE + SGD", COLORS["ar"]),
        ("uniform_roles", "Известные роли + SGD", COLORS["consensus"]),
        ("enumerate_roles", "Перебор 16 структур", "#577aa0"),
        ("oracle", "Правильная маска + SGD", COLORS["oracle"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for i, (key, label, color) in enumerate(definitions):
        entry = entries[key]
        rate = 100 * entry["rate"]
        lo, hi = np.array(entry["task_cluster_interval"]) * 100
        axes[0].barh(i, rate, color=color, height=0.65, alpha=0.9)
        axes[0].errorbar(
            rate,
            i,
            xerr=[[rate - lo], [hi - rate]],
            color="#24364b",
            capsize=3,
        )
        axes[0].text(rate + 1.4, i + 0.27, f"{rate:.2f}%", fontsize=8)
        if key in ["dense_sgd", "dense_adam", "vae", "product", "oracle"]:
            budget = np.array([int(k) for k in entry["prefix_success"]])
            values = 100 * np.array(list(entry["prefix_success"].values()))
            axes[1].plot(
                budget,
                values,
                "o-",
                color=color,
                label=label,
                lw=2,
                markersize=4,
            )
            for step, value in zip(budget, values):
                CURVES.append(
                    [
                        "sine_search",
                        key,
                        "success_percent",
                        int(step),
                        float(value),
                    ]
                )
    axes[0].set_yticks(
        range(len(definitions)), [item[1] for item in definitions]
    )
    axes[0].set(
        xlim=(0, 111),
        xlabel="Успешные поиски, %",
        title="384 поиска; одинаковый онлайн-бюджет",
    )
    axes[0].invert_yaxis()
    axes[1].axvline(32, color="#7b8794", ls=":", lw=1)
    axes[1].text(33, 8, "Переход к refit\nдля генераторов", fontsize=8)
    axes[1].set(
        xlabel="Число обученных кандидатов × 1024 шага",
        ylabel="Успешные поиски, %",
        ylim=(0, 105),
        title="Качество при расходовании бюджета",
    )
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(alpha=0.2)
    save_figure(fig, "sine_results")
    return {
        "matched_v2": source["matched_v2"],
        "geometry": geometry,
        "task_for_geometry": task,
        "matched_differences": source["matched_differences"],
        "replication_v2": source["replication_v2"],
    }


def dsd_path(bank, method):
    if method in ["sgd", "adam", "oracle"]:
        return bank / f"target_fit/noise_1_n2048/{method}.npz"
    if method in ["ar", "consensus", "frequency"]:
        return bank / f"priors/relations/noise_1/target_n2048/{method}.npz"
    return bank / f"competitors/noise_1/target_n2048/{method}.npz"


def dsd_figures():
    summary = read_json(DSD / "summary.json")
    methods = [
        ("sgd", "Плотный SGD"),
        ("adam", "Плотный Adam"),
        ("mask_unshared", "Маска без sharing"),
        ("consensus", "Та же маска + sharing"),
        ("ar", "AR-индексатор"),
        ("oracle", "Оракул"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for index, (key, label) in enumerate(methods):
        entry = summary["methods"][key]
        values = entry["per_bank_accuracy_percent"]
        axes[0].scatter(np.arange(1, 6), values, color=COLORS[key], s=35)
        axes[0].plot(
            np.arange(1, 6), values, color=COLORS[key], lw=1.5, label=label
        )
        axes[1].errorbar(
            entry["mean_accuracy_percent"],
            index,
            xerr=entry["between_bank_sd_pp"],
            color=COLORS[key],
            fmt="o",
            capsize=4,
        )
        axes[1].text(
            entry["mean_accuracy_percent"] + 0.45,
            index,
            f"{entry['mean_accuracy_percent']:.2f}%",
            va="center",
            fontsize=9,
        )
    axes[0].set(
        xlabel="Независимый банк",
        ylabel="Test accuracy, %",
        xticks=np.arange(1, 6),
        title="Пять банков × 16 новых задач",
    )
    axes[0].legend(fontsize=8, loc="center right")
    axes[0].grid(alpha=0.2)
    axes[1].set_yticks(range(len(methods)), [label for _, label in methods])
    axes[1].set(
        xlabel="Test accuracy, %",
        xlim=(81.5, 100.2),
        title="Среднее и SD между банками",
    )
    axes[1].invert_yaxis()
    save_figure(fig, "dsd_results")
    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.1), constrained_layout=True
    )
    metrics = {}
    for key, label in methods:
        histories, losses = [], []
        for replica in range(1, 6):
            path = dsd_path(DSD / f"bank_{replica}", key)
            arrays = read_npz(path)
            chosen = best_rows(arrays)
            losses.append(float(arrays["test_loss"][chosen].mean()))
            history = read_history(
                path.with_name(path.stem + "_history.json.gz")
            )
            histories.append(
                {
                    split: np.array(
                        [
                            np.array(point[split]["loss"])[chosen].mean()
                            for point in history
                        ]
                    )
                    for split in ["train", "query"]
                }
            )
        steps = [point["step"] for point in history]
        metrics[key] = {"mean_test_bce": float(np.mean(losses))}
        for ax, split in zip(axes, ["train", "query"]):
            values = np.array([history[split] for history in histories])
            mean, sd = values.mean(0), values.std(0, ddof=1)
            ax.plot(steps, mean, color=COLORS[key], label=label, lw=2)
            ax.fill_between(
                steps, mean - sd, mean + sd, color=COLORS[key], alpha=0.1
            )
            for step, value in zip(steps, mean):
                CURVES.append(["dsd", key, split, step, float(value)])
    for ax, title in zip(axes, ["Support: BCE текущего батча", "Query: BCE"]):
        ax.set(
            title=title, xlabel="Шаг оптимизатора", ylabel="BCE", ylim=(0, 0.8)
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    save_figure(fig, "dsd_learning")
    teacher = read_npz(DSD / "bank_1/target/teacher_audit.npz")
    rng = np.random.default_rng(20261103)
    signal = np.zeros((16, 8))
    signal[5] = teacher["directions"][0]
    sample = signal + 0.25 * rng.normal(size=(16, 8))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for row, values in enumerate([signal, sample]):
        permuted = np.empty(128)
        permuted[teacher["permutation"]] = values.ravel()
        for col, shown in enumerate([values, permuted.reshape(16, 8)]):
            ax = axes[row, col]
            im = ax.imshow(
                shown, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"
            )
            ax.set_xticks(range(8), range(1, 9))
            ax.set_yticks([0, 5, 10, 15], [1, 6, 11, 16])
            ax.set_xlabel(
                "Координата внутри блока" if col == 0 else "Условный столбец"
            )
            ax.set_ylabel("Истинный блок" if col == 0 else "Условная строка")
    axes[0, 0].set_title("Сигнал до перестановки: y = +1, блок 6")
    axes[0, 1].set_title("Сигнал после скрытой перестановки")
    axes[1, 0].set_title("Добавлен гауссов шум σ = 0.25")
    axes[1, 1].set_title("Вход MLP: 128 наблюдаемых координат")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Значение координаты")
    save_figure(fig, "dsd_example")
    summary["additional_metrics"] = metrics
    summary["illustration"] = {
        "seed": 20261103,
        "bank": 1,
        "task": 0,
        "block_zero_based": 5,
        "label": 1,
    }
    return summary


def main():
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
    metrics = {
        "pattern": pattern_figures(),
        "sine": sine_figures(),
        "dsd": dsd_figures(),
    }
    (HERE / "report_data.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )
    (HERE / "sources.json").write_text(
        json.dumps(
            {
                "created": "2026-11-03",
                "root": str(ROOT),
                "sha256": SOURCES,
                "notes": [
                    "All result tables reuse recorded runs; no fits were repeated.",
                    "Sine landscape is calculated from the recorded teacher and minima.",
                    "DSD input illustration is newly sampled, not a training observation.",
                    "Pattern column permutations preserve the selected predictors.",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    with (HERE / "learning_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["experiment", "method", "metric", "step_or_candidate", "value"]
        )
        writer.writerows(CURVES)
    print(
        json.dumps(
            {
                "figures": len(list(HERE.glob("*.svg"))),
                "source_files": len(SOURCES),
                "pattern_permutation_max_error": max(
                    value["max_prediction_change_after_permutation"]
                    for key, value in metrics["pattern"].items()
                    if key != "debug_summary"
                ),
                "generated_indexator_exact": metrics["pattern"]["ar"][
                    "aligned_indexator_exact"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
