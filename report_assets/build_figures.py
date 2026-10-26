"""Build the final report figures from saved summaries; run from any folder."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
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
        CHART_DATA.append({
            "panel": title,
            "method": key,
            "metric": "success_percent",
            "value": rate,
            "successes": successes,
            "searches": searches,
        })
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
        ax.text(rate + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}% · {successes}/{searches}",
                va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
    })
    mask = read_summary(RESULTS / "summary.json")
    codebook = read_summary(RESULTS / "codebook_summary.json")
    hidden = read_summary(RESULTS / "hidden_structure_summary.json")
    robustness = read_summary(RESULTS / "robustness_twolayer_summary.json")
    composition = read_summary(
        ROOT / "archive/history/reports/assets/2026-10-21"
        / "composition_full/test_summary.json"
    )["summary"]

    fig, axes = plt.subplots(1, 3, figsize=(19, 4.6), layout="constrained")
    success_panel(axes[0], "Маски · 8 координат", {
        row["method"]: row for row in mask["matched_v2"]
    }, [
        ("enumerate_roles", "Перебор 16 масок", "simple"),
        ("uniform_roles", "Случайно по ролям", "simple"),
        ("vae", "Partwise VAE", "vae"),
        ("product", "Эмпирический prior", "simple"),
        ("dense_adam", "Dense Adam", "dense"),
        ("dense_sgd", "Dense SGD", "dense"),
    ])
    success_panel(axes[1], "Codebook · 32 веса", {
        row["method"]: row for row in codebook["rows"]
        if row["dimension"] == 32
    }, [
        ("code_vae", "VAE", "vae"),
        ("code_blocks", "Эмпирические блоки", "simple"),
        ("dense_adam", "Dense Adam", "dense"),
        ("code_transfer", "Старые значения", "other"),
        ("code_replay", "Готовые назначения", "other"),
    ])
    success_panel(axes[2], "Скрытые группы · 32 веса", {
        row["method"]: row for row in hidden["rows"]
        if row["dimension"] == 32 and row["source_size"] == "all"
        and row["condition"] == "shared_permutation"
    }, [
        ("code_components", "Восстановленные группы", "simple"),
        ("code_vae", "VAE", "vae"),
        ("dense_adam", "Dense Adam", "dense"),
        ("code_replay", "Готовые назначения", "other"),
    ])
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
            CHART_DATA.append({
                "panel": f"{title}: residual={residual}",
                "method": method,
                "metric": "success_percent",
                "value": rate,
                "successes": row["successes"],
                "searches": row["n"],
            })
        axes[0].plot(residuals, rates, "o-", label=label,
                     color=COLORS[color], linewidth=2)
    axes[0].set(title=title, xlabel="Норма отклонения от общей структуры",
                ylabel="Успешные поиски, % ↑", ylim=(-2, 60))
    axes[0].grid(alpha=0.18)
    axes[0].legend(fontsize=9, loc="upper right")
    success_panel(axes[1], "Шум source-весов · σ = 0.5",
                  robustness["robustness"]["noise_0.5/e0"], [
                      ("vae_release", "VAE + освобождение", "vae"),
                      ("components_release", "Группы + освобождение", "simple"),
                      ("components_hard", "Жёсткое связывание", "hard"),
                      ("dense_adam", "Dense Adam", "dense"),
                  ])
    success_panel(axes[2], "Двухслойная tanh-сеть",
                  robustness["twolayer"]["two_layer_tanh"], [
                      ("dense_adam", "Dense Adam", "dense"),
                      ("numerical_transfer", "Перенос значений", "other"),
                      ("components_release", "Группы + освобождение", "simple"),
                      ("components", "Группы", "simple"),
                      ("vae", "VAE", "vae"),
                  ])
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
        (axes[0], "test_excess_bce", "Ошибка предсказания · excess BCE ↓", 1000),
        (axes[1], "iou", "Совпадение разреженных масок · IoU ↑", 1),
    ]:
        selected = methods if metric == "test_excess_bce" else methods[1:]
        values = [composition[key][metric]["mean"] * scale
                  for key, _, _ in selected]
        bars = ax.barh([label for _, label, _ in selected], values,
                       color=[COLORS[color] for _, _, color in selected])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", pad=15, fontweight="bold")
        ax.set_xlim(0, max(values) * 1.2)
        ax.set_xlabel("× 10⁻³" if scale == 1000 else "IoU")
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        for bar, value, (key, _, _) in zip(bars, values, selected):
            ax.text(value + max(values) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}", va="center")
            CHART_DATA.append({
                "panel": "ReLU: новые сочетания частей",
                "method": key,
                "metric": metric,
                "value": composition[key][metric]["mean"],
            })
    save_figure(fig, "relu_composition")

    with (OUTPUT / "chart_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "panel", "method", "metric", "value", "successes", "searches",
        ])
        writer.writeheader()
        writer.writerows(CHART_DATA)
    (OUTPUT / "sources.json").write_text(
        json.dumps(SOURCES, indent=2) + "\n"
    )
    print(f"Saved 3 figures and {len(CHART_DATA)} plotted values to {OUTPUT}")


if __name__ == "__main__":
    main()
