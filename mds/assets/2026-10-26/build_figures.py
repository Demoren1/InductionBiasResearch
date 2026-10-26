"""Build the final report figures from saved summaries; run from any folder."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
RESULTS = ROOT / "paper_vae_solver" / "results"
COLORS = {
    "vae": "#6654b8",
    "simple": "#008578",
    "dense": "#d27835",
    "other": "#7a8999",
    "hard": "#b34656",
}
CHART_DATA = []
SOURCES = {}


def read_summary(path):
    """Record the exact input file alongside the plotted data."""
    content = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def save_figure(fig, name):
    fig.savefig(OUTPUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def success_panel(ax, title, rows, methods):
    labels, rates, colors = [], [], []
    for key, label, color in methods:
        row = rows[key]
        successes = row.get("success", row.get("successes"))
        searches = row.get("searches", row.get("n"))
        rate = 100 * successes / searches
        labels.append(label)
        rates.append(rate)
        colors.append(COLORS[color])
        CHART_DATA.append(
            {
                "panel": title,
                "method": key,
                "metric": "success_percent",
                "value": rate,
                "successes": successes,
                "searches": searches,
            }
        )
    bars = ax.barh(labels, rates, color=colors, height=0.65)
    ax.invert_yaxis()
    ax.set_xlim(0, 155)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Успешные поиски, % ↑")
    ax.set_title(title, loc="left", pad=15, fontweight="bold")
    for bar, rate, (key, _, _) in zip(bars, rates, methods):
        row = rows[key]
        successes = row.get("success", row.get("successes"))
        searches = row.get("searches", row.get("n"))
        ax.text(
            rate + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1f}% · {successes}/{searches}",
            va="center",
            fontsize=9,
        )
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)


def analytical_figures():
    """Illustrate explicitly defined teachers, rather than new training runs."""
    coordinate = np.linspace(-2.5, 2.5, 501)
    low, high, alpha = 0.7, 4.0, 0.4

    def covariance(a, b):
        return 0.5 * (
            np.exp(-0.5 * (a - b) ** 2) - np.exp(-0.5 * (a + b) ** 2)
        )

    amplitude = alpha * covariance(low, high) + covariance(high, high)
    amplitude /= covariance(high, high)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), layout="constrained")
    axes[0].plot(
        coordinate,
        alpha * np.sin(low * coordinate) + np.sin(high * coordinate),
        label="Учитель: две частоты",
        color="#172b4d",
        linewidth=2,
    )
    axes[0].plot(
        coordinate,
        amplitude * np.sin(high * coordinate),
        "--",
        label="Одна синусоида при w = 4",
        color=COLORS["simple"],
    )
    axes[0].set(
        title="Маски: учитель вне класса ученика",
        xlabel="Полезная координата xᵣ",
        ylabel="Выход",
    )
    axes[0].legend(fontsize=9, loc="upper left")
    axes[1].plot(
        coordinate,
        np.sin(high * coordinate),
        label="Цель и точное решение: sin(4t)",
        color=COLORS["simple"],
    )
    axes[1].plot(
        coordinate,
        np.sin(coordinate),
        "--",
        label="Источник: sin(t)",
        color=COLORS["other"],
    )
    axes[1].set(
        title="Codebook: точное решение известно",
        xlabel="x = tv, ‖v‖ = 1",
        ylabel="Выход",
    )
    axes[1].legend(fontsize=9, loc="upper left")
    parallel, orthogonal = np.meshgrid(
        np.linspace(-6, 6, 321), np.linspace(0, 6, 161)
    )
    norm2 = parallel**2 + orthogonal**2
    a_value = 0.5 * (-np.expm1(-2 * norm2))
    b_value = (
        0.5
        * np.exp(-(orthogonal**2) / 2)
        * (
            np.exp(-((parallel - high) ** 2) / 2)
            - np.exp(-((parallel + high) ** 2) / 2)
        )
    )
    risk = 0.5 * (-np.expm1(-2 * high**2)) - np.divide(
        b_value**2, a_value, out=np.zeros_like(a_value), where=a_value > 0
    )
    plot = axes[2].pcolormesh(
        parallel,
        orthogonal,
        risk,
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=0.5,
        rasterized=True,
    )
    axes[2].set(
        title="Точный риск после выбора амплитуды",
        xlabel="Проекция w на v",
        ylabel="‖w − (wᵀv)v‖",
    )
    fig.colorbar(plot, ax=axes[2], label="min по γ: population MSE")
    for ax in axes[:2]:
        ax.grid(alpha=0.15)
    save_figure(fig, "sine_analytic")

    mask = np.zeros((48, 8))
    for hidden in range(8):
        mask[
            [2 * hidden, 2 * hidden + 1, 16 + 2 * hidden, 17 + 2 * hidden],
            hidden,
        ] = 1
    x_first, x_second = np.meshgrid(
        np.linspace(-3, 3, 121), np.linspace(-3, 3, 121)
    )
    grid = np.zeros((x_first.size, 48))
    grid[:, 0], grid[:, 2] = x_first.ravel(), x_second.ravel()
    weights = 0.5 * mask
    thresholds = np.linspace(-0.3, 0.3, 8)
    output = np.array([(-1) ** h for h in range(8)]) / np.sqrt(8)
    logits = np.maximum(grid @ weights - thresholds, 0) @ output
    teacher = (1 / (1 + np.exp(-logits))).reshape(x_first.shape)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    axes[0].imshow(mask, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[0].axhline(15.5, color="#d27835", linestyle="--")
    axes[0].axhline(31.5, color="#d27835", linestyle="--")
    axes[0].set(
        title="Пример правильной маски: 32 / 384",
        xlabel="Скрытый нейрон",
        ylabel="Вход: полезные 0–31, шум 32–47",
    )
    heatmap = axes[1].pcolormesh(
        x_first,
        x_second,
        teacher,
        shading="auto",
        cmap="viridis",
        rasterized=True,
    )
    contours = axes[1].contour(
        x_first, x_second, teacher, levels=6, colors="white", alpha=0.7
    )
    axes[1].clabel(contours, fontsize=8, fmt="%.2f")
    axes[1].set(
        title="Точный учитель на двумерном срезе",
        xlabel="x₀, остальные x = 0 кроме x₂",
        ylabel="x₂",
    )
    fig.colorbar(heatmap, ax=axes[1], label="Вероятность учителя")
    save_figure(fig, "relu_analytic")

    residual = np.linspace(0, 2, 301)
    a_value = 16
    b_value = residual**2
    floor = 0.5 * (
        1
        - np.exp(-2 * (a_value + b_value))
        - np.exp(-b_value) * (1 - np.exp(-2 * a_value))
    )
    fig, ax = plt.subplots(figsize=(7.5, 4), layout="constrained")
    ax.plot(
        residual,
        floor,
        color=COLORS["hard"],
        linewidth=2.5,
        label="Лучший возможный риск внутри подпространства",
    )
    ax.axhline(0.1, color="#465466", linestyle="--", label="Порог успеха: 0.1")
    critical = np.sqrt(-np.log(0.8))
    ax.axvline(critical, color="#7a8999", linestyle=":")
    ax.text(critical + 0.04, 0.12, f"ε ≈ {critical:.3f}")
    ax.set(
        xlabel="Норма ортогонального остатка ε",
        ylabel="Population MSE",
        title="Предел точности жёсткого связывания · ‖u‖ = 4",
        ylim=(0, 0.53),
    )
    ax.grid(alpha=0.18)
    ax.legend(fontsize=9)
    save_figure(fig, "approximate_floor")


def solution_figure(example):
    """Show actual saved predictors from the first fixed task and trial."""
    target = np.array(example["target_weights"])
    direction = target / np.linalg.norm(target)
    methods = [
        ("dense_adam", "Dense Adam", "dense"),
        ("code_vae", "VAE", "vae"),
        ("code_components", "Компоненты", "simple"),
        ("spectral_components", "Спектральный", "other"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    order = np.concatenate(example["inferred_components"])
    patterns = np.vstack(
        [example["source_assignments"], example["target_bits"]]
    )
    axes[0, 0].imshow(patterns[:, order], aspect="auto", cmap="Blues")
    axes[0, 0].axhline(len(patterns) - 1.5, color="#d27835", linewidth=2)
    for boundary in np.cumsum(
        [len(g) for g in example["inferred_components"]]
    )[:-1]:
        axes[0, 0].axvline(boundary - 0.5, color="#d27835", linestyle="--")
    axes[0, 0].set(
        title="Исходные назначения и новая цель (снизу)",
        xlabel="Координаты сгруппированы для показа",
        ylabel="Source-решения",
    )
    weights = np.vstack(
        [target]
        + [example["methods"][key]["weights"] for key, _, _ in methods]
    )
    limit = np.max(np.abs(weights))
    weight_plot = axes[0, 1].imshow(
        weights, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit
    )
    fig.colorbar(weight_plot, ax=axes[0, 1], label="Значение веса")
    axes[0, 1].set_yticks(
        range(len(weights)), ["Учитель"] + [label for _, label, _ in methods]
    )
    axes[0, 1].set(
        title="Веса возвращённых моделей · первый поиск",
        xlabel="Исходный порядок скрыто переставленных координат",
    )
    coordinate = np.linspace(-2, 2, 501)
    axes[1, 0].plot(
        coordinate,
        np.sin(np.linalg.norm(target) * coordinate),
        color="#172b4d",
        linewidth=4,
        alpha=0.35,
        label="Точный учитель",
    )
    for method, label, color in methods:
        model = example["methods"][method]
        prediction = model["amplitude"] * np.sin(
            coordinate * (np.array(model["weights"]) @ direction)
        )
        axes[1, 0].plot(
            coordinate,
            prediction,
            label=label,
            color=COLORS[color],
            linestyle="--",
            linewidth=1.8,
        )
    axes[1, 0].set(
        title="Предсказания на срезе x = tv", xlabel="t", ylabel="Выход"
    )
    axes[1, 0].legend(fontsize=9, loc="upper left")
    for index, (method, _, color) in enumerate(methods):
        errors = example["methods"][method]["all_trial_population_mse"]
        jitter = np.linspace(-0.12, 0.12, len(errors))
        axes[1, 1].scatter(
            index + jitter, errors, s=30, color=COLORS[color], alpha=0.8
        )
    axes[1, 1].axhline(0.1, color="#465466", linestyle="--")
    axes[1, 1].set_xticks(
        range(len(methods)), [label for _, label, _ in methods]
    )
    axes[1, 1].set(
        title="Все возвращённые решения этого контекста",
        ylabel="Population MSE",
        ylim=(-0.025, 0.56),
    )
    axes[1, 1].text(0.05, 0.115, "Порог успеха")
    axes[1, 0].grid(alpha=0.18)
    axes[1, 1].grid(axis="y", alpha=0.18)
    save_figure(fig, "solution_example")


def codebook_coverage_figure(codebook):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for method, label, color in [
        ("code_vae", "VAE", "vae"),
        ("code_blocks", "Эмпирические блоки", "simple"),
        ("dense_adam", "Dense Adam", "dense"),
    ]:
        rows = sorted(
            [row for row in codebook["rows"] if row["method"] == method],
            key=lambda row: row["dimension"],
        )
        axes[0].plot(
            [row["dimension"] for row in rows],
            [row["rate"] * 100 for row in rows],
            "o-",
            color=COLORS[color],
            label=label,
            linewidth=2,
        )
        if method == "dense_adam":
            continue
        for metric, linestyle, suffix in [
            ("correct_assignment_seen", "--", "правильное I встретилось"),
            ("correct_assignment_chosen", "-", "правильное I выбрано"),
        ]:
            axes[1].plot(
                [row["dimension"] for row in rows],
                [row[metric] * 100 for row in rows],
                marker="o",
                linestyle=linestyle,
                color=COLORS[color],
                label=f"{label}: {suffix}",
            )
    for ax in axes:
        ax.set_xticks([16, 32, 48])
        ax.set(xlabel="Количество весов P", ylim=(-3, 105))
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8)
    axes[0].set(
        title="Успех при фиксированном бюджете", ylabel="Успешные поиски, %"
    )
    axes[1].set(title="Покрытие и отбор — разные этапы", ylabel="Поиски, %")
    save_figure(fig, "codebook_coverage")


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
        }
    )
    mask = read_summary(RESULTS / "summary.json")
    codebook = read_summary(RESULTS / "codebook_summary.json")
    hidden = read_summary(RESULTS / "hidden_structure_summary.json")
    robustness = read_summary(RESULTS / "robustness_twolayer_summary.json")
    composition = read_summary(
        ROOT / "mds/assets/2026-10-21" / "composition_full/test_summary.json"
    )["summary"]
    example = read_summary(OUTPUT / "solution_example.json")
    analytical_figures()
    solution_figure(example)
    codebook_coverage_figure(codebook)

    fig, axes = plt.subplots(1, 3, figsize=(19, 4.6), layout="constrained")
    success_panel(
        axes[0],
        "Маски · 8 координат",
        {row["method"]: row for row in mask["matched_v2"]},
        [
            ("enumerate_roles", "Перебор 16 масок", "simple"),
            ("uniform_roles", "Случайно по ролям", "simple"),
            ("vae", "Partwise VAE", "vae"),
            ("product", "Эмпирический prior", "simple"),
            ("dense_adam", "Dense Adam", "dense"),
            ("dense_sgd", "Dense SGD", "dense"),
        ],
    )
    success_panel(
        axes[1],
        "Codebook · 32 веса",
        {
            row["method"]: row
            for row in codebook["rows"]
            if row["dimension"] == 32
        },
        [
            ("code_vae", "VAE", "vae"),
            ("code_blocks", "Эмпирические блоки", "simple"),
            ("dense_adam", "Dense Adam", "dense"),
            ("code_transfer", "Старые значения", "other"),
            ("code_replay", "Готовые назначения", "other"),
        ],
    )
    success_panel(
        axes[2],
        "Скрытые группы · 32 веса",
        {
            row["method"]: row
            for row in hidden["rows"]
            if row["dimension"] == 32
            and row["source_size"] == "all"
            and row["condition"] == "shared_permutation"
        },
        [
            ("code_components", "Восстановленные группы", "simple"),
            ("code_vae", "VAE", "vae"),
            ("dense_adam", "Dense Adam", "dense"),
            ("code_replay", "Готовые назначения", "other"),
        ],
    )
    save_figure(fig, "structural_search")

    fig, axes = plt.subplots(1, 3, figsize=(19, 4.8), layout="constrained")
    residuals = [0, 0.1, 0.3, 0.6, 1, 2]
    title = "Неточное связывание весов"
    for method, label, color in [
        ("components_release", "Группы + освобождение", "simple"),
        ("components_hard", "Жёсткое связывание", "hard"),
        ("vae_release", "VAE + освобождение", "vae"),
        ("dense_adam", "Dense Adam", "dense"),
    ]:
        rates = []
        for residual in residuals:
            row = robustness["robustness"][f"noise_0/e{residual:g}"][method]
            rate = 100 * row["successes"] / row["n"]
            rates.append(rate)
            CHART_DATA.append(
                {
                    "panel": f"{title}: residual={residual}",
                    "method": method,
                    "metric": "success_percent",
                    "value": rate,
                    "successes": row["successes"],
                    "searches": row["n"],
                }
            )
        axes[0].plot(
            residuals,
            rates,
            "o-",
            label=label,
            color=COLORS[color],
            linewidth=2,
        )
    axes[0].set(
        title=title,
        xlabel="Норма отклонения от общей структуры",
        ylabel="Успешные поиски, % ↑",
        ylim=(-2, 60),
    )
    axes[0].grid(alpha=0.18)
    axes[0].legend(fontsize=9, loc="upper right")
    success_panel(
        axes[1],
        "Шум source-весов · σ = 0.5",
        robustness["robustness"]["noise_0.5/e0"],
        [
            ("vae_release", "VAE + освобождение", "vae"),
            ("components_release", "Группы + освобождение", "simple"),
            ("components_hard", "Жёсткое связывание", "hard"),
            ("dense_adam", "Dense Adam", "dense"),
        ],
    )
    success_panel(
        axes[2],
        "Двухслойная tanh-сеть",
        robustness["twolayer"]["two_layer_tanh"],
        [
            ("dense_adam", "Dense Adam", "dense"),
            ("numerical_transfer", "Перенос значений", "other"),
            ("components_release", "Группы + освобождение", "simple"),
            ("components", "Группы", "simple"),
            ("vae", "VAE", "vae"),
        ],
    )
    save_figure(fig, "limits_and_robustness")

    methods = [
        ("dense", "Плотная MLP", "dense"),
        ("imp", "IMP", "other"),
        ("evolution", "W/G evolution", "simple"),
        ("random_budget", "Случайные z", "vae"),
        ("gumbel", "W/G Gumbel", "hard"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for ax, metric, title, scale in [
        (
            axes[0],
            "test_excess_bce",
            "Ошибка предсказания · excess BCE ↓",
            1000,
        ),
        (axes[1], "iou", "Совпадение разреженных масок · IoU ↑", 1),
    ]:
        selected = methods if metric == "test_excess_bce" else methods[1:]
        values = [
            composition[key][metric]["mean"] * scale for key, _, _ in selected
        ]
        bars = ax.barh(
            [label for _, label, _ in selected],
            values,
            color=[COLORS[color] for _, _, color in selected],
        )
        ax.invert_yaxis()
        ax.set_title(title, loc="left", pad=15, fontweight="bold")
        ax.set_xlim(0, max(values) * 1.2)
        ax.set_xlabel("× 10⁻³" if scale == 1000 else "IoU")
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        for bar, value, (key, _, _) in zip(bars, values, selected):
            ax.text(
                value + max(values) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
            )
            CHART_DATA.append(
                {
                    "panel": "ReLU: новые сочетания частей",
                    "method": key,
                    "metric": metric,
                    "value": composition[key][metric]["mean"],
                }
            )
    save_figure(fig, "relu_composition")

    with (OUTPUT / "chart_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "panel",
                "method",
                "metric",
                "value",
                "successes",
                "searches",
            ],
        )
        writer.writeheader()
        writer.writerows(CHART_DATA)
    (OUTPUT / "sources.json").write_text(json.dumps(SOURCES, indent=2) + "\n")
    print(f"Saved 8 figures and {len(CHART_DATA)} summary values to {OUTPUT}")


if __name__ == "__main__":
    main()
