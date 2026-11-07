"""Plot analytical and recorded DSD indexators for the November report."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from experiments.dynamic_signal_v1.indexator_audit import align
from experiments.dynamic_signal_v1.report import best_rows, path_for

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/dynamic_signal_v1/results/bank_repeats"
OUTPUT = ROOT / "final_mds/assets/2026-11-03"
METHODS = ("consensus", "ar")
ERROR_COLORS = ("white", "#dce2e9", "#c74747", "#286cb0", "#dd961c")


def indexator_errors(value, truth):
    expected = truth > 0
    present = value > 0
    missing = expected & ~present
    extra = present & ~expected
    wrong_code = expected & present & (value != truth)
    errors = np.zeros_like(truth)
    errors[expected] = 1
    errors[missing] = 2
    errors[extra] = 3
    errors[wrong_code] = 4
    correct = int((expected & (value == truth)).sum())
    return errors, {
        "missing_connections": int(missing.sum()),
        "extra_connections": int(extra.sum()),
        "wrong_codes": int(wrong_code.sum()),
        "correct_active_indices": correct,
        "active_connections": int(present.sum()),
        "active_index_accuracy_percent": 100 * correct / expected.sum(),
    }


def draw_indexator(ax, values, palette, norm, grouped=True):
    image = ax.imshow(
        values,
        cmap=palette,
        norm=norm,
        interpolation="nearest",
        aspect="auto",
    )
    pairs = np.array([0, 3, 7, 11, 15])
    ax.set_xticks(2 * pairs + 0.5, pairs + 1)
    ax.set_xticks(np.arange(-0.5, 32, 2), minor=True)
    ax.set_xlabel("Пара скрытых нейронов")
    if grouped:
        ax.set_yticks(8 * np.arange(16) + 3.5, np.arange(1, 17))
        ax.set_yticks(np.arange(-0.5, 128, 8), minor=True)
        ax.set_ylabel("Истинный блок: 8 входов")
    else:
        ax.set_yticks([0, 31, 63, 95, 127], [1, 32, 64, 96, 128])
        ax.set_ylabel("Наблюдаемая координата входа")
    ax.grid(which="minor", color="#afb8c4", linewidth=0.35, alpha=0.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", labelsize=8)
    return image


def analytical_figure(observed, canonical, palette, norm):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11, 7.6),
        gridspec_kw={"width_ratios": [1, 1, 0.6]},
        constrained_layout=True,
    )
    draw_indexator(axes[0], observed, palette, norm, grouped=False)
    image = draw_indexator(axes[1], canonical, palette, norm)
    axes[0].set_title("В порядке входов MLP\nБанк 1: матрица 128 × 32")
    axes[1].set_title("После группировки входов\n16 одинаковых блоков 8 × 2")
    block = canonical[:8, :2]
    axes[2].imshow(block, cmap=palette, norm=norm, aspect="auto")
    axes[2].set(
        title="Один блок крупно",
        xlabel="Нейрон внутри пары",
        ylabel="Координата внутри блока",
    )
    axes[2].set_xticks([0, 1], [1, 2])
    axes[2].set_yticks(range(8), range(1, 9))
    axes[2].set_xticks(np.arange(-0.5, 2, 1), minor=True)
    axes[2].set_yticks(np.arange(-0.5, 8, 1), minor=True)
    axes[2].grid(which="minor", color="white", linewidth=1)
    axes[2].tick_params(which="minor", length=0)
    for row in range(8):
        for column in range(2):
            axes[2].text(
                column,
                row,
                str(block[row, column]),
                ha="center",
                va="center",
                color="white" if column == 0 else "#172233",
                fontsize=15,
                weight="bold",
            )
    fig.suptitle(
        "Аналитический индексатор Dynamic Signal", fontsize=15, weight="bold"
    )
    bar = fig.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        ticks=np.arange(17),
        shrink=0.9,
        aspect=45,
        pad=0.06,
    )
    bar.set_label(
        "Индекс общего коэффициента: 0 — нет связи; "
        "1–8 — первый фильтр; 9–16 — второй"
    )
    for extension in ("svg", "png"):
        fig.savefig(
            OUTPUT / f"dsd_indexator_analytic.{extension}",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(fig)


def obtained_figure(aligned, canonical, palette, norm):
    fig = plt.figure(figsize=(16, 12.8), constrained_layout=True)
    grid = fig.add_gridspec(
        5, 5, height_ratios=[1, 1, 0.055, 1, 0.11]
    )
    error_palette = ListedColormap(ERROR_COLORS)
    error_norm = BoundaryNorm(np.arange(-0.5, 5.5), error_palette.N)
    for bank, values in enumerate(aligned):
        for method_id, label in enumerate(("Опорный I", "AR")):
            ax = fig.add_subplot(grid[method_id, bank])
            image = draw_indexator(
                ax, values[method_id, 0], palette, norm
            )
            _, metrics = indexator_errors(values[method_id, 0], canonical)
            correct = metrics["correct_active_indices"]
            title = f"{label}: {correct}/256 верных индексов"
            if method_id == 0:
                title = f"Банк {bank + 1}\n{title}"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("")
            if bank:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
        errors, metrics = indexator_errors(values[1, 0], canonical)
        ax = fig.add_subplot(grid[3, bank])
        draw_indexator(ax, errors, error_palette, error_norm)
        ax.set_title(
            f"Нет: {metrics['missing_connections']} · "
            f"лишних: {metrics['extra_connections']} · "
            f"код: {metrics['wrong_codes']}",
            fontsize=9,
        )
        if bank:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Ошибки AR: истинный блок")
    bar = fig.colorbar(
        image,
        cax=fig.add_subplot(grid[2, :]),
        orientation="horizontal",
        ticks=np.arange(17),
    )
    bar.set_label("Номер общего коэффициента; белый — отсутствие связи")
    legend_ax = fig.add_subplot(grid[4, :])
    legend_ax.axis("off")
    labels = (
        "Нет связи в обоих I",
        "Правильная связь и код",
        "Пропущенная связь",
        "Лишняя связь",
        "Связь верна, код неверен",
    )
    legend_ax.legend(
        handles=[
            Patch(facecolor=color, edgecolor="#a3aab5", label=label)
            for color, label in zip(ERROR_COLORS, labels)
        ],
        loc="center",
        ncol=5,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Полученные индексаторы: опорный I и AR в пяти банках\n"
        "Общий порядок истинных блоков; согласованы нейроны и имена кодов",
        fontsize=15,
        weight="bold",
    )
    for extension in ("svg", "png"):
        fig.savefig(
            OUTPUT / f"dsd_indexators_obtained.{extension}",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "svg.fonttype": "path",
        }
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab20").colors
    palette = ListedColormap(
        ["white", *colors[:16:2], *colors[1:16:2]]
    )
    norm = BoundaryNorm(np.arange(-0.5, 17.5), palette.N)
    canonical = np.zeros((128, 32), dtype=np.int64)
    inputs = np.arange(128)
    canonical[inputs, 2 * (inputs // 8)] = inputs % 8 + 1
    canonical[inputs, 2 * (inputs // 8) + 1] = inputs % 8 + 9
    sources = [
        Path(__file__).resolve(),
        ROOT / "experiments/dynamic_signal_v1/indexator_audit.py",
        ROOT / "experiments/dynamic_signal_v1/report.py",
    ]
    observed_truth, permutations, raw_values, aligned_values = [], [], [], []
    selected_rows, records, bank_summaries = [], [], []
    for bank in range(1, 6):
        directory = RESULTS / f"bank_{bank}"
        path = directory / "target/teacher_audit.npz"
        sources.append(path)
        with np.load(path) as teacher:
            permutation = teacher["permutation"]
        observed = np.empty_like(canonical)
        observed[permutation] = canonical
        observed_truth.append(observed)
        permutations.append(permutation)
        bank_raw, bank_aligned, bank_rows = [], [], []
        summary = {"bank": bank}
        for method in METHODS:
            path = path_for(directory, method, 1, 2048)
            sources.append(path)
            with np.load(path) as arrays:
                rows = best_rows(arrays)
                raw = arrays["indices"][rows].astype(np.int64)
                task_ids = arrays["task_ids"][rows]
                restarts = arrays["restart_ids"][rows]
            aligned = np.stack(
                [align(value, observed)[permutation] for value in raw]
            )
            bank_raw.append(raw)
            bank_aligned.append(aligned)
            bank_rows.append(rows)
            summary[f"{method}_unique_selected"] = len(
                np.unique(aligned, axis=0)
            )
            for task, restart, row, value in zip(
                task_ids, restarts, rows, aligned
            ):
                _, metrics = indexator_errors(value, canonical)
                records.append(
                    {
                        "bank": bank,
                        "method": method,
                        "task_zero_based": int(task),
                        "restart_zero_based": int(restart),
                        "selected_row": int(row),
                        **metrics,
                    }
                )
        raw_values.append(bank_raw)
        aligned_values.append(bank_aligned)
        selected_rows.append(bank_rows)
        bank_summaries.append(summary)
    aligned_values = np.asarray(aligned_values)
    np.savez_compressed(
        OUTPUT / "dsd_indexators_data.npz",
        methods=np.array(METHODS),
        analytical_grouped=canonical,
        analytical_observed=np.asarray(observed_truth),
        permutations=np.asarray(permutations),
        selected_rows=np.asarray(selected_rows),
        recorded_observed=np.asarray(raw_values),
        grouped_aligned=aligned_values,
    )
    with (OUTPUT / "dsd_indexators_recovery.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    analytical_figure(observed_truth[0], canonical, palette, norm)
    obtained_figure(aligned_values, canonical, palette, norm)
    manifest = {
        "created": "2026-11-03",
        "selection": "Lowest saved query loss per task, same as the report.",
        "displayed_task_zero_based": 0,
        "alignment": (
            "Existing audit: permute hidden pairs and globally rename roles; "
            "then reorder inputs by the teacher permutation for display only."
        ),
        "banks": bank_summaries,
        "sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sources
        },
    }
    (OUTPUT / "dsd_indexators_sources.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(bank_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
