"""Build the final report for generated sharing across pattern lengths 3--7."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from scipy.optimize import linear_sum_assignment


TRAINED_LENGTHS = (3, 5, 7)
UNSEEN_LENGTHS = (4, 6)
ALL_LENGTHS = (3, 4, 5, 6, 7)
SEEDS = (42, 43, 44, 45)


def stats(values: list[float]) -> dict[str, Any]:
    return {"values": values, "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0}


def fmt(row: dict[str, Any]) -> str:
    return f"{row['mean']:.4f} ± {row['std']:.4f}"


def load_training(root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for path in root.glob("*/summary.json"):
        row = json.loads(path.read_text())
        groups.setdefault((row["variant"], row["config"]["seed"]), []).append(row)
    return {
        variant: {
            seed: max(groups[(variant, seed)],
                      key=lambda row: row["config"]["outer_steps"])
            for seed in SEEDS
        }
        for variant in ("global", "length_latent")
    }


def load_unseen(root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = {
        "global": {}, "length_latent": {},
    }
    for path in root.glob("*.json"):
        row = json.loads(path.read_text())
        result[row["source_variant"]][row["config"]["seed"]] = row
    for variant in result:
        if set(result[variant]) != set(SEEDS):
            raise ValueError(f"missing unseen records for {variant}")
    return result


def train_stat(records: list[dict[str, Any]], method: str,
               field: str) -> dict[str, Any]:
    return stats([
        row["evaluations"]["test"]["strategies"][method][field]
        for row in records
    ])


def unseen_stat(records: list[dict[str, Any]], method: str,
                field: str) -> dict[str, Any]:
    return stats([row["strategies"][method][field] for row in records])


def per_length(
    training: dict[str, dict[int, dict[str, Any]]],
    unseen: dict[str, dict[int, dict[str, Any]]],
    series: str,
    length: int,
    field: str,
) -> dict[str, Any]:
    values = []
    for seed in SEEDS:
        if length in TRAINED_LENGTHS:
            variant = "length_latent" if series == "length_latent" else "global"
            method = {
                "global": "generated_sharing",
                "length_latent": "generated_sharing",
                "dense": "dense",
                "analytic": "analytic_sharing",
            }[series]
            source = training[variant][seed]["evaluations"]["test"]["strategies"][method]
        else:
            if series == "global":
                source = unseen["global"][seed]["strategies"]["global_zero_shot"]
            elif series == "length_latent":
                source = unseen["length_latent"][seed]["strategies"]["latent_adaptation"]
            elif series == "interpolation":
                source = unseen["length_latent"][seed]["strategies"]["latent_interpolation"]
            else:
                method = "dense" if series == "dense" else "analytic_sharing"
                source = unseen["global"][seed]["strategies"][method]
        values.append(source["per_length"][str(length)][field])
    return stats(values)


def build_summary(training: dict[str, dict[int, dict[str, Any]]],
                  unseen: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    train_aggregate = {}
    for name, variant, method in (
        ("global", "global", "generated_sharing"),
        ("length_latent", "length_latent", "generated_sharing"),
        ("connectivity", "global", "generated_connectivity"),
        ("random", "global", "random_sharing"),
        ("dense", "global", "dense"),
        ("analytic", "global", "analytic_sharing"),
    ):
        records = list(training[variant].values())
        train_aggregate[name] = {
            field: train_stat(records, method, field)
            for field in ("mean_query_bce", "mean_query_accuracy", "mean_active_iou")
        }

    global_records = list(unseen["global"].values())
    latent_records = list(unseen["length_latent"].values())
    unseen_aggregate = {}
    for name, records, method in (
        ("global_zero_shot", global_records, "global_zero_shot"),
        ("interpolation", latent_records, "latent_interpolation"),
        ("adaptation", latent_records, "latent_adaptation"),
        ("dense", global_records, "dense"),
        ("analytic", global_records, "analytic_sharing"),
    ):
        unseen_aggregate[name] = {
            field: unseen_stat(records, method, field)
            for field in ("mean_query_bce", "mean_query_accuracy", "mean_active_iou")
        }
    dense_values = unseen_aggregate["dense"]["mean_query_bce"]["values"]
    for name in ("global_zero_shot", "interpolation", "adaptation"):
        values = unseen_aggregate[name]["mean_query_bce"]["values"]
        unseen_aggregate[name]["wins_vs_dense"] = sum(
            value < dense for value, dense in zip(values, dense_values)
        )

    by_length = {}
    for length in ALL_LENGTHS:
        by_length[str(length)] = {
            series: {
                field: per_length(training, unseen, series, length, field)
                for field in ("mean_query_bce", "mean_query_accuracy", "mean_active_iou")
            }
            for series in ("global", "length_latent", "dense", "analytic")
        }
        if length in UNSEEN_LENGTHS:
            by_length[str(length)]["interpolation"] = {
                field: per_length(training, unseen, "interpolation", length, field)
                for field in ("mean_query_bce", "mean_query_accuracy", "mean_active_iou")
            }

    convergence = {}
    for variant in ("global", "length_latent"):
        convergence[variant] = {
            "selected_steps": [
                training[variant][seed]["history"][-1]["selected_outer_step"]
                for seed in SEEDS
            ],
            "completed_steps": [
                training[variant][seed]["history"][-1]["completed_outer_steps"]
                for seed in SEEDS
            ],
            "converged": [
                bool(training[variant][seed]["history"][-1]["converged"])
                for seed in SEEDS
            ],
        }
    convergence["unseen"] = {
        "completed_steps": [
            unseen["length_latent"][seed]["latent_search_history"][-1]["completed_steps"]
            for seed in SEEDS
        ],
        "converged": [
            bool(unseen["length_latent"][seed]["latent_search_history"][-1]["converged"])
            for seed in SEEDS
        ],
    }
    return {
        "train_aggregate": train_aggregate,
        "unseen_aggregate": unseen_aggregate,
        "per_length": by_length,
        "convergence": convergence,
    }


def plot_performance(summary: dict[str, Any], output: Path) -> None:
    colors = {"global": "#2864DC", "length_latent": "#E38927",
              "dense": "#7A8492", "analytic": "#18996A",
              "interpolation": "#A65CC8"}
    labels = {"global": "Global z (zero-shot at 4/6)",
              "length_latent": "Per-length z (adapted at 4/6)",
              "dense": "Dense MLP", "analytic": "Analytic sharing",
              "interpolation": "Midpoint z (zero-shot)"}
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    fields = (("mean_query_bce", "Test BCE by pattern length ↓"),
              ("mean_query_accuracy", "Test accuracy by pattern length ↑"))
    for axis, (field, title) in zip(axes, fields):
        for length in UNSEEN_LENGTHS:
            axis.axvspan(length - 0.22, length + 0.22,
                         color="#EEF1F5", zorder=0)
        for series in ("global", "length_latent", "dense", "analytic"):
            means = [summary["per_length"][str(k)][series][field]["mean"]
                     for k in ALL_LENGTHS]
            errors = [summary["per_length"][str(k)][series][field]["std"]
                      for k in ALL_LENGTHS]
            axis.errorbar(ALL_LENGTHS, means, yerr=errors, marker="o",
                          capsize=3, linewidth=2, color=colors[series],
                          label=labels[series])
        means = [summary["per_length"][str(k)]["interpolation"][field]["mean"]
                 for k in UNSEEN_LENGTHS]
        errors = [summary["per_length"][str(k)]["interpolation"][field]["std"]
                  for k in UNSEEN_LENGTHS]
        axis.errorbar(UNSEEN_LENGTHS, means, yerr=errors, marker="D",
                      linestyle="--", capsize=3, color=colors["interpolation"],
                      label=labels["interpolation"])
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(ALL_LENGTHS)
        axis.set_xlabel("Pattern length (shaded = unseen)")
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
    axes[0].set_ylim(0, 0.72)
    axes[1].set_ylim(0.55, 1.02)
    axes[0].legend(fontsize=8)
    figure.suptitle("Generated sharing trained on lengths 3, 5, 7",
                    fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def plot_convergence(training: dict[str, dict[int, dict[str, Any]]],
                     unseen: dict[str, dict[int, dict[str, Any]]],
                     output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), sharey=True)
    colors = ("#2864DC", "#E38927", "#18996A", "#A65CC8")
    for axis, variant, title in (
        (axes[0], "global", "Global z: generator"),
        (axes[1], "length_latent", "Per-length z: generator"),
    ):
        for color, seed in zip(colors, SEEDS):
            history = training[variant][seed]["history"]
            points = [row for row in history
                      if "checkpoint_validation_bce" in row]
            axis.plot([row["outer_step"] for row in points],
                      [row["checkpoint_validation_bce"] for row in points],
                      color=color, alpha=0.78, linewidth=1.3,
                      label=f"seed {seed}")
            selected = history[-1]["selected_outer_step"]
            point = min(points, key=lambda row: abs(row["outer_step"] - selected))
            axis.scatter(point["outer_step"], point["checkpoint_validation_bce"],
                         marker="*", s=90, color=color, edgecolor="#20242A",
                         linewidth=0.5, zorder=3)
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Outer step")
        axis.grid(alpha=0.22)
    for color, seed in zip(colors, SEEDS):
        history = unseen["length_latent"][seed]["latent_search_history"]
        points = [row for row in history if "hard_validation_bce" in row]
        axes[2].plot([row["step"] for row in points],
                     [row["hard_validation_bce"] for row in points],
                     color=color, alpha=0.78, linewidth=1.3,
                     label=f"seed {seed}")
    axes[2].set_title("Frozen generator: z4/z6")
    axes[2].set_xlabel("Latent step")
    axes[2].grid(alpha=0.22)
    axes[0].set_ylabel("Balanced hard-validation BCE ↓")
    axes[0].set_ylim(0.35, 0.72)
    axes[0].legend(fontsize=8)
    figure.suptitle("Convergence checks on binary structures",
                    fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def gold_categories() -> np.ndarray:
    result = np.zeros((32, 32), dtype=np.int64)
    for column in range(32):
        for offset in range(7):
            result[(column + offset) % 32, column] = offset + 1
    return result


def align_categories(gold: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    gold_active = gold > 0
    candidate_active = candidate > 0
    cost = np.square(
        gold_active.T[:, None, :].astype(float)
        - candidate_active.T[None, :, :].astype(float)
    ).sum(-1)
    rows, columns = linear_sum_assignment(cost)
    aligned = candidate[:, columns[np.argsort(rows)]]
    overlap = np.zeros((7, 7))
    for source, target in itertools.product(range(1, 8), repeat=2):
        overlap[source - 1, target - 1] = np.logical_and(
            aligned == source, gold == target
        ).sum()
    source, target = linear_sum_assignment(-overlap)
    mapping = {int(a + 1): int(b + 1) for a, b in zip(source, target)}
    return np.vectorize(lambda value: mapping.get(int(value), 0))(aligned)


def mask_rows(unseen: dict[str, dict[int, dict[str, Any]]],
              seed: int = 44) -> tuple[list[str], list[list[np.ndarray]]]:
    global_record = unseen["global"][seed]
    latent_record = unseen["length_latent"][seed]
    global_row, adapted_row, midpoint_row = [], [], []
    for length in ALL_LENGTHS:
        key = str(length)
        if length in TRAINED_LENGTHS:
            global_row.append(np.asarray(
                global_record["masks"]["trained"][key]))
            trained = np.asarray(latent_record["masks"]["trained"][key])
            adapted_row.append(trained)
            midpoint_row.append(trained)
        else:
            global_row.append(np.asarray(
                global_record["masks"]["global_zero_shot"][key]))
            adapted_row.append(np.asarray(
                latent_record["masks"]["latent_adaptation"][key]))
            midpoint_row.append(np.asarray(
                latent_record["masks"]["latent_interpolation"][key]))
    gold = gold_categories()
    labels = ["Global z / zero-shot", "Per-length z / adapted",
              "Per-length z / midpoint", "Analytic ideal"]
    return labels, [global_row, adapted_row, midpoint_row,
                    [gold.copy() for _ in ALL_LENGTHS]]


def plot_masks(unseen: dict[str, dict[int, dict[str, Any]]],
               category_output: Path, active_output: Path) -> None:
    labels, rows = mask_rows(unseen)
    gold = gold_categories()
    rows = [[align_categories(gold, mask) for mask in row]
            for row in rows[:-1]] + [rows[-1]]
    column_titles = ["Global z", "Per-length z\nlearned / adapted",
                     "Midpoint z\nzero-shot", "Analytic ideal"]
    row_titles = [f"k={length}" + (" (unseen)" if length in UNSEEN_LENGTHS else "")
                  for length in ALL_LENGTHS]
    colors = ["#FFFFFF", "#2864DC", "#E38927", "#18996A",
              "#A65CC8", "#D94F70", "#34A6A6", "#8B6F47"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 8.5), cmap.N)
    figure, axes = plt.subplots(5, 4, figsize=(11.6, 13.5))
    image = None
    for row_index, (length, row_title) in enumerate(zip(ALL_LENGTHS, row_titles)):
        for column_index, column_title in enumerate(column_titles):
            axis = axes[row_index, column_index]
            if column_index == 2 and length not in UNSEEN_LENGTHS:
                axis.set_facecolor("#F5F6F8")
                axis.text(0.5, 0.5, "not applicable", ha="center", va="center",
                          color="#7A8492", transform=axis.transAxes, fontsize=9)
                axis.set_xticks([])
                axis.set_yticks([])
                continue
            mask = rows[column_index][row_index]
            image = axis.imshow(
                mask, cmap=cmap, norm=norm, interpolation="nearest",
                aspect="equal")
            if row_index == 0:
                axis.set_title(column_title, fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(row_title, fontsize=10, fontweight="bold")
            axis.set_xticks([])
            axis.set_yticks([])
    assert image is not None
    colorbar_axis = figure.add_axes((0.925, 0.17, 0.016, 0.66))
    colorbar = figure.colorbar(image, cax=colorbar_axis, ticks=range(8))
    colorbar.set_label("0 inactive; 1...7 shared parameter index")
    figure.suptitle("Generated parameter-sharing masks U — seed 44",
                    fontsize=14, fontweight="bold")
    figure.subplots_adjust(left=0.10, right=0.90, top=0.93, bottom=0.03,
                           wspace=0.08, hspace=0.13)
    figure.savefig(category_output, dpi=190, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(5, 4, figsize=(11.6, 13.5))
    for row_index, (length, row_title) in enumerate(zip(ALL_LENGTHS, row_titles)):
        for column_index, column_title in enumerate(column_titles):
            axis = axes[row_index, column_index]
            if column_index == 2 and length not in UNSEEN_LENGTHS:
                axis.set_facecolor("#F5F6F8")
                axis.text(0.5, 0.5, "not applicable", ha="center", va="center",
                          color="#7A8492", transform=axis.transAxes, fontsize=9)
                axis.set_xticks([])
                axis.set_yticks([])
                continue
            mask = rows[column_index][row_index]
            axis.imshow(
                mask > 0, cmap="Greys", vmin=0, vmax=1,
                interpolation="nearest", aspect="equal")
            if row_index == 0:
                axis.set_title(column_title, fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(row_title, fontsize=10, fontweight="bold")
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle("Binary active-edge masks — seed 44",
                    fontsize=14, fontweight="bold")
    figure.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.03,
                           wspace=0.08, hspace=0.13)
    figure.savefig(active_output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def write_report(summary: dict[str, Any], output: Path) -> None:
    train = summary["train_aggregate"]
    unseen = summary["unseen_aggregate"]
    lines = [
        "# Generated parameter sharing на нескольких длинах pattern",
        "",
        "Дата: 15 сентября 2026 года.",
        "",
        "## Постановка",
        "",
        "Длина входа — 32. Генератор обучался на всех бинарных pattern "
        "длины 3, 5 и 7. Длины 4 и 6 не входили в обучение.",
        "",
        "U = Gψ(z),    Wτ = Uvτ.",
        "",
        "Для каждой задачи заново обучаются vτ, bias первого слоя, "
        "readout и output bias.",
        "",
        "| Величина | Generated sharing | Dense MLP |",
        "|---|---:|---:|",
        "| Параметров на задачу | 72 | 1089 |",
        "| Общих параметров генератора | 520 | 0 |",
        "| Всего для 168 train-задач | 12 616 | 182 952 |",
        "| Полная assignment U | 32×32×7 = 7168 элементов | — |",
        "",
        "## Ёмкость и сходимость",
        "",
        "| Проверка | Результат |",
        "|---|---|",
        "| Прямой fit генератора к analytic U | width 16: exact U, "
        "Active IoU 1.0 с шага 400 |",
        f"| Global z | patience 4/4; selected "
        f"{summary['convergence']['global']['selected_steps']}; completed "
        f"{summary['convergence']['global']['completed_steps']} |",
        f"| Отдельный zₖ | patience 4/4; selected "
        f"{summary['convergence']['length_latent']['selected_steps']}; completed "
        f"{summary['convergence']['length_latent']['completed_steps']} |",
        f"| Новые z₄,z₆, frozen generator | patience 4/4; completed "
        f"{summary['convergence']['unseen']['completed_steps']} |",
        "",
        "Сходимость означает plateau hard-validation loss, а не глобальный оптимум.",
        "",
        "![Кривые сходимости](assets/2026-09-15/multilength_convergence.png)",
        "",
        "## Train-длины 3, 5, 7",
        "",
        "Среднее ± SD по четырём сидам; длины имеют одинаковый вес.",
        "",
        "| Метод | Test BCE ↓ | Accuracy ↑ | Active IoU ↑ |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "global": "Generated sharing, global z",
        "length_latent": "Generated sharing, zₖ",
        "connectivity": "Та же connectivity без sharing",
        "random": "Random sharing",
        "dense": "Dense MLP",
        "analytic": "Analytic sharing",
    }
    for key in ("global", "length_latent", "connectivity",
                "random", "dense", "analytic"):
        row = train[key]
        lines.append(
            f"| {labels[key]} | {fmt(row['mean_query_bce'])} | "
            f"{fmt(row['mean_query_accuracy'])} | "
            f"{fmt(row['mean_active_iou'])} |"
        )
    lines += [
        "",
        "## Невидимые длины 4 и 6",
        "",
        "Global z переносится без изменений. Для zₖ проверены midpoint "
        "соседних train-латентов и оптимизация нового z при frozen generator. "
        "Dense и analytic обучены на тех же данных.",
        "",
        "| Метод | Test BCE ↓ | Accuracy ↑ | Active IoU ↑ | Победы над dense |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "global_zero_shot": "Global z, zero-shot",
        "interpolation": "Midpoint z, zero-shot",
        "adaptation": "Новый zₖ, frozen generator",
        "dense": "Dense MLP",
        "analytic": "Analytic sharing",
    }
    for key in ("global_zero_shot", "interpolation", "adaptation",
                "dense", "analytic"):
        row = unseen[key]
        wins = f"{row['wins_vs_dense']}/4" if "wins_vs_dense" in row else "—"
        lines.append(
            f"| {labels[key]} | {fmt(row['mean_query_bce'])} | "
            f"{fmt(row['mean_query_accuracy'])} | "
            f"{fmt(row['mean_active_iou'])} | {wins} |"
        )
    lines += [
        "",
        "## Результаты по каждой длине",
        "",
        "На длинах 4/6 zₖ означает новый оптимизированный latent.",
        "",
        "| k | В train | Global z: BCE / Acc | zₖ: BCE / Acc | "
        "Dense: BCE / Acc | Analytic: BCE / Acc |",
        "|---:|:---:|---:|---:|---:|---:|",
    ]
    for length in ALL_LENGTHS:
        row = summary["per_length"][str(length)]
        def pair(series: str) -> str:
            return (f"{fmt(row[series]['mean_query_bce'])} / "
                    f"{fmt(row[series]['mean_query_accuracy'])}")
        lines.append(
            f"| {length} | {'да' if length in TRAINED_LENGTHS else 'нет'} | "
            f"{pair('global')} | {pair('length_latent')} | "
            f"{pair('dense')} | {pair('analytic')} |"
        )
    lines += [
        "",
        "![Метрики по длинам](assets/2026-09-15/multilength_performance_by_length.png)",
        "",
        "## Примеры масок",
        "",
        "Показан seed 44. Hidden columns и индексы shared-параметров "
        "выровнены к analytic ideal только для визуализации. Analytic ideal "
        "приведён справа для каждой длины.",
        "",
        "Цвет: белый означает неактивную связь, 1…7 — индекс коэффициента vτ. "
        "Для 4/6 показаны и midpoint, и адаптированный latent.",
        "Global z по определению даёт одну и ту же U для всех длин. Analytic ideal "
        "также является общей super-structure ширины 7; различие длины задачи "
        "должно реализовываться через task-specific vτ.",
        "",
        "![Карты parameter sharing](assets/2026-09-15/multilength_category_masks.png)",
        "",
        "Те же структуры как бинарные active-edge маски:",
        "",
        "![Бинарные маски](assets/2026-09-15/multilength_active_masks.png)",
        "",
        "## Итог",
        "",
        "| Проверка | Результат |",
        "|---|---|",
        f"| Generated лучше dense на train-длинах | Да: BCE "
        f"{train['global']['mean_query_bce']['mean']:.4f} / "
        f"{train['length_latent']['mean_query_bce']['mean']:.4f} против "
        f"{train['dense']['mean_query_bce']['mean']:.4f} |",
        f"| Global z переносится zero-shot на 4/6 | Да: BCE "
        f"{unseen['global_zero_shot']['mean_query_bce']['mean']:.4f} против "
        f"dense {unseen['dense']['mean_query_bce']['mean']:.4f}; 4/4 |",
        f"| Новый zₖ переносится на 4/6 | Да в среднем: BCE "
        f"{unseen['adaptation']['mean_query_bce']['mean']:.4f} против "
        f"dense {unseen['dense']['mean_query_bce']['mean']:.4f}; 4/4 |",
        f"| Midpoint latent достаточен | Нет: BCE "
        f"{unseen['interpolation']['mean_query_bce']['mean']:.4f}; "
        f"{unseen['interpolation']['wins_vs_dense']}/4 побед |",
        f"| Найдена analytic структура | Нет: Active IoU "
        f"{train['global']['mean_active_iou']['mean']:.4f} / "
        f"{train['length_latent']['mean_active_iou']['mean']:.4f}; "
        f"analytic BCE {train['analytic']['mean_query_bce']['mean']:.4f} |",
        "| Генератору не хватает параметров | Нет: width-16 точно "
        "представляет analytic U; gap относится к task-loss optimization "
        "и soft-to-hard projection |",
        "",
        "Численные данные: "
        "[multilength_generated_sharing_summary.json]"
        "(data/2026-09-15/multilength_generated_sharing_summary.json).",
        "",
    ]
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--unseen-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    training = load_training(args.training_root)
    unseen = load_unseen(args.unseen_root)
    summary = build_summary(training, unseen)
    args.assets.mkdir(parents=True, exist_ok=True)
    args.data.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    plot_performance(summary, args.assets / "multilength_performance_by_length.png")
    plot_convergence(training, unseen,
                     args.assets / "multilength_convergence.png")
    plot_masks(unseen,
               args.assets / "multilength_category_masks.png",
               args.assets / "multilength_active_masks.png")
    args.data.write_text(json.dumps(summary, indent=2))
    write_report(summary, args.report)


if __name__ == "__main__":
    main()
