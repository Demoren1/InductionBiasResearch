"""Plot frozen single-source results without fitting or selecting models."""

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from experiments.paper_structure_benchmark_v1.reporting.evaluate import (
    align_index,
)

COLORS = {"ratio": "#2563eb", "g34": "#8b5cf6", "dense": "#ea580c"}
CASES = ("case_04", "case_05")
BUDGETS = (64, 256, 1024)


def save(figure, folder, name):
    for extension in ("png", "pdf"):
        figure.savefig(
            folder / f"{name}.{extension}", dpi=180, bbox_inches="tight"
        )
    plt.close(figure)


def bank_plot(audit, folder):
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
    figure.subplots_adjust(
        left=0.09, right=0.98, top=0.73, bottom=0.22, wspace=0.34
    )
    positions, width = np.arange(2), 0.32
    for offset, method, label in (
        (-width / 2, "ratio", "Прямые отношения"),
        (width / 2, "g34", "G34"),
    ):
        for column, key in enumerate(
            ("evaluated_bank_size", "bank_and_selection_seconds")
        ):
            values = [item["methods"][method][key] for item in audit["cases"]]
            bars = axes[column].bar(
                positions + offset,
                values,
                width,
                color=COLORS[method],
                label=label,
                zorder=3,
            )
            labels = [
                str(int(value))
                if column == 0
                else f"{value:.1f}".replace(".", ",")
                for value in values
            ]
            axes[column].bar_label(bars, labels=labels, padding=5)
    for axis in axes:
        axis.set_xticks(positions, CASES)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
        axis.set_ylim(0, axis.get_ylim()[1] * 1.17)
    axes[0].set(title="Размер банка", ylabel="Кандидатов после отсечения")
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    axes[1].set(title="Поиск, подгонка и отбор", ylabel="Секунды")
    figure.suptitle("Извлечение структуры по одной задаче", y=0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.09,
        0.075,
        "Оба метода восстановили точный индексатор и маску в 2/2 случаях.\n"
        "Время исключает общую dense-подготовку (12,6 и 12,9 с); "
        "по одному измерению на случай.",
        fontsize=9,
        linespacing=1.6,
    )
    save(figure, folder, "bank_and_time")


def transfer_plot(results, plan, folder):
    lookup = {
        (row["case"], row["method"], row["budget"]): row
        for row in results["rows"]
    }
    shared = all(
        item["aliases"].get("g34") == item["aliases"].get("ratio")
        for item in plan["cases"]
    )
    methods = [("ratio", "Прямые отношения = G34")]
    if not shared:
        methods = [("ratio", "Прямые отношения"), ("g34", "G34")]
    methods.append(("dense", "Dense"))
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 5.8), sharey=True)
    figure.subplots_adjust(
        left=0.085, right=0.98, top=0.76, bottom=0.22, wspace=0.15
    )
    smallest = min(
        value
        for row in results["rows"]
        for value in row["per_task_mse"]
        if value > 0
    )
    for axis, case in zip(axes, CASES):
        for method, label in methods:
            means = [lookup[case, method, n]["mean_mse"] for n in BUDGETS]
            for n in BUDGETS:
                values = lookup[case, method, n]["per_task_mse"]
                x = n * np.exp(np.linspace(-0.045, 0.045, len(values)))
                axis.scatter(
                    x,
                    values,
                    s=24,
                    color=COLORS[method],
                    alpha=0.35,
                    edgecolors="none",
                    zorder=3,
                )
            axis.plot(
                BUDGETS,
                means,
                "o-",
                color=COLORS[method],
                label=label,
                linewidth=2.2,
                markersize=6,
                zorder=4,
            )
        axis.set(
            xscale="log",
            yscale="log",
            title=case,
            xlabel="Обучающих примеров на новой задаче",
            xlim=(52, 1250),
            ylim=(10 ** np.floor(np.log10(smallest)) / 2, 20),
        )
        axis.set_xticks(BUDGETS)
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_locator(plt.NullLocator())
        axis.set_yticks(10.0 ** np.arange(-15, 2, 3))
        axis.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("MSE на новых примерах, логарифмическая шкала")
    outlier = max(lookup["case_05", "ratio", 64]["per_task_mse"])
    axes[1].annotate(
        "Одна неудачная подгонка\nиз четырёх: MSE = 0,402",
        xy=(64, outlier),
        xytext=(0.24, 0.71),
        textcoords="axes fraction",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#64748b"},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    figure.suptitle(
        "Перенос на новые задачи: все числа обучаются с нуля", y=0.98
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(methods),
        frameon=False,
    )
    figure.text(
        0.085,
        0.065,
        "Линии — среднее по 4 новым задачам; "
        "полупрозрачные точки — отдельные задачи.\n"
        "Прямые отношения и G34 выбрали один индексатор; "
        "для них использована общая численная подгонка.",
        fontsize=9,
        linespacing=1.6,
    )
    save(figure, folder, "transfer_mse")


def structure_plot(root, folder):
    structures = dict(np.load(root / "recovery_structures.npz"))
    descriptors = dict(np.load(root / "single_source_descriptors.npz"))
    palette = ListedColormap(
        [
            "#5b2c83",
            "#225ea8",
            "#d6604d",
            "#238b45",
            "#ffffff",
            "#a1d99b",
            "#f4a582",
            "#9ecae1",
            "#c2a5cf",
        ]
    )
    norm = BoundaryNorm(np.arange(-4.5, 5.5), palette.N)
    figure, axes = plt.subplots(2, 4, figsize=(13.5, 8.2))
    figure.subplots_adjust(
        left=0.075,
        right=0.97,
        top=0.85,
        bottom=0.25,
        wspace=0.32,
        hspace=0.40,
    )
    categorical = None
    for row, case in enumerate(CASES):
        truth = structures[f"{case}_truth"]
        ratio, ratio_metrics = align_index(
            structures[f"{case}_ratio_selected"], truth, 4
        )
        g34, g34_metrics = align_index(
            structures[f"{case}_g34_selected"], truth, 4
        )
        matrices = (truth, ratio, g34)
        titles = (
            "Истинный индексатор\n ",
            f"Прямые отношения\n{ratio_metrics['signed_entry_errors']} ошибок",
            f"G34\n{g34_metrics['signed_entry_errors']} ошибок",
        )
        for column, (matrix, title) in enumerate(zip(matrices, titles)):
            categorical = axes[row, column].imshow(
                matrix, cmap=palette, norm=norm, interpolation="nearest"
            )
            if row == 0:
                axes[row, column].set_title(title, fontsize=11, pad=9)
        weight = descriptors[f"{case}_dense_parameters"][:256].reshape(16, 16)
        limit = float(np.abs(weight).max())
        dense_image = axes[row, 3].imshow(
            weight,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        colorbar = figure.colorbar(
            dense_image, ax=axes[row, 3], fraction=0.047, pad=0.04
        )
        colorbar.ax.tick_params(labelsize=8)
        colorbar.set_label("Вес", fontsize=9)
        if row == 0:
            axes[row, 3].set_title(
                "Веса source dense\nисходный порядок нейронов",
                fontsize=11,
                pad=9,
            )
        for column, axis in enumerate(axes[row]):
            axis.set_xticks((0, 4, 8, 12, 15), (1, 5, 9, 13, 16))
            axis.set_yticks((0, 4, 8, 12, 15), (1, 5, 9, 13, 16))
            axis.tick_params(labelsize=8)
            if row == 1:
                axis.set_xlabel("Нейрон", fontsize=10)
            if column == 0:
                axis.set_ylabel(f"{case}\nВходная координата", fontsize=10)
            else:
                axis.tick_params(labelleft=False)
    color_axis = figure.add_axes((0.10, 0.125, 0.57, 0.024))
    colorbar = figure.colorbar(
        categorical,
        cax=color_axis,
        orientation="horizontal",
        ticks=range(-4, 5),
    )
    colorbar.set_label(
        "Знаковая метка группы; 0 — нет соединения", fontsize=10
    )
    colorbar.ax.tick_params(labelsize=9)
    figure.suptitle(
        "Индексаторы восстановлены вместе с маской и знаками", y=0.98
    )
    figure.text(
        0.075,
        0.017,
        "Первые три столбца: нумерация групп и нейронов "
        "согласована с истиной только после фиксации результатов.\n"
        "Справа — фактические веса обученной dense-модели "
        "с отдельной числовой шкалой.",
        fontsize=9,
        linespacing=1.6,
    )
    save(figure, folder, "recovered_indexators")


def render(root):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "figure.titlesize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    audit = json.loads((root / "structural_audit.json").read_text())
    results = json.loads((root / "transfer_results.json").read_text())
    plan = json.loads((root / "transfer_plan.json").read_text())
    bank_plot(audit, folder)
    transfer_plot(results, plan, folder)
    structure_plot(root, folder)
    gallery = [
        "# Графики восстановления по одной задаче",
        "",
        "Графики построены по зафиксированным результатам; "
        "обучение и выбор моделей не повторялись.",
        "",
    ]
    for name, caption in (
        ("bank_and_time", "Размер банка и время поиска с source-отбором"),
        ("transfer_mse", "Перенос: средние и отдельные новые задачи"),
        ("recovered_indexators", "Истинные и восстановленные индексаторы"),
    ):
        gallery.extend(
            [
                f"## {caption}",
                "",
                f"![{caption}]({name}.png)",
                "",
                f"[PDF]({name}.pdf)",
                "",
            ]
        )
    (folder / "GALLERY.md").write_text("\n".join(gallery))
    snapshot = root / "code_snapshot/presentation"
    snapshot.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), snapshot / "plots.py")
    inputs = (
        "structural_audit.json",
        "transfer_results.json",
        "transfer_plan.json",
        "recovery_structures.npz",
        "single_source_descriptors.npz",
    )
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "input_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in inputs
        },
        "plot_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "dense_weights": "Raw source W1 in native hidden-unit order",
        "indexator_alignment": "Reference labels/neurons after source freeze",
        "new_fitting_or_model_selection": False,
    }
    (folder / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    readme = root / "README.md"
    content = readme.read_text()
    link = "[Графики PNG/PDF](figures/GALLERY.md)."
    if link not in content:
        readme.write_text(content + "\n" + link + "\n")
    manifest_path = root / "compact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = [*folder.iterdir(), snapshot / "plots.py", readme]
    for path in files:
        manifest["files"][str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(folder, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    render(args.root.resolve())


if __name__ == "__main__":
    main()
