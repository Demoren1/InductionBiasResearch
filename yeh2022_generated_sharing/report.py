"""Aggregate completed Yeh-style runs into one paper-vs-generator report."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt


METHODS = ("no_sharing", "oracle", "direct", "generated")


@dataclass
class Row:
    key: str
    label: str
    mse: dict[str, tuple[float, float]]
    pd: dict[str, tuple[float, float]]
    paper_mse: tuple[float, float] | None = None
    paper_pd: tuple[float, float] | None = None
    paper_note: str = ""


def _ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return mean(values), 0.0 if len(values) == 1 else 1.96 * stdev(values) / math.sqrt(len(values))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _gaussian_rows(root: Path, reference: dict[str, Any]) -> list[Row]:
    paper = {
        int(item["dimensions"]): item
        for item in reference["gaussian_ablation_table_1"]
        if item["penalties"] == "entropy+nuclear"
    }
    rows = []
    seen: set[str] = set()
    for path in sorted(root.rglob("gaussian_results.json")):
        record = _load(path)
        config = record["config"]
        dimension, rank = int(config["dimensions"]), int(config["true_rank"])
        train = int(config["num_train"])
        mse = {
            method: (
                float(record["methods"][method]["mse_sum_mean"]),
                float(record["methods"][method]["mse_sum_ci95"]),
            )
            for method in METHODS
        }
        pd = {
            method: (
                float(record["methods"][method]["partition_distance_mean"]),
                float(record["methods"][method]["partition_distance_ci95"]),
            )
            for method in METHODS
        }
        release_match = (
            config.get("lower_solver") == "release_normalized"
            and config.get("optimizer") == "rmsprop"
            and int(config.get("num_samples", 0)) == 100
            and float(config.get("noise_std", -1)) == 1.0
            and int(config.get("epochs", 0)) == 1000
            and int(config.get("restarts", 0)) == 3
        )
        paper_row = paper.get(dimension) if rank == 1 and train == 30 and release_match else None
        key = f"gaussian-k{dimension}-r{rank}-t{train}"
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            Row(
                key=key,
                label=f"Gaussian K={dimension}, r={rank}, T={train}",
                mse=mse,
                pd=pd,
                paper_mse=None
                if paper_row is None
                else (float(paper_row["mse_mean"]), float(paper_row["mse_ci95"])),
                paper_pd=None
                if paper_row is None
                else (float(paper_row["pd_mean"]), float(paper_row["pd_ci95"])),
                paper_note="Table 1, entropy+nuclear" if paper_row is not None else "figure only",
            )
        )
    return rows


def _linear_rows(root: Path, reference: dict[str, Any]) -> list[Row]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in root.rglob("summary.json"):
        record = _load(path)
        if record.get("benchmark") not in {"valid_cross_correlation", "unit_step_denoising"}:
            continue
        metadata = json.dumps(record.get("benchmark_metadata", {}), sort_keys=True)
        grouped[(record["benchmark"], metadata)].append(record)
    exact_pd = {
        int(item["assignment_items"]): float(item["partition_distance"])
        for item in reference["cross_correlation_exact_statements"]
    }
    rows = []
    for (benchmark, metadata_json), records in sorted(grouped.items()):
        metadata = json.loads(metadata_json)
        if benchmark == "valid_cross_correlation":
            items = int(records[0]["dimensions"]["assignment_items"])
            label = (
                f"Cross-corr {metadata.get('output_length')}×{metadata.get('input_length')} "
                f"(A:{items})"
            )
            paper_pd = (exact_pd[items], 0.0) if items in exact_pd else None
            note = "exact statement in Sec. 5.3" if paper_pd else "figure only"
        else:
            variance = float(metadata.get("noise_std", float("nan"))) ** 2
            label = f"Denoising K={metadata.get('signal_length')}, noise var={variance:g}"
            paper_pd, note = None, "Figure A1 only"
        mse = {
            method: _ci([float(record["methods"][method]["mse"]) for record in records])
            for method in METHODS
        }
        pd = {
            method: _ci(
                [float(record["methods"][method]["partition_distance"]) for record in records]
            )
            for method in METHODS
        }
        rows.append(
            Row(
                key=f"{benchmark}-{metadata_json}",
                label=label,
                mse=mse,
                pd=pd,
                paper_pd=paper_pd,
                paper_note=note,
            )
        )
    return rows


def _sum_rows(root: Path, reference: dict[str, Any]) -> list[Row]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for path in root.rglob("sum_numbers_results.json"):
        record = _load(path)
        for target_name, target in record["targets"].items():
            length = int(target["config"]["sequence_length"])
            grouped[(target_name, length)].append(target)
    paper_direct = next(
        item
        for item in reference["sum_numbers_table_a1"]
        if item["method"] == "Yeh direct A"
    )
    rows = []
    for (target, length), records in sorted(grouped.items()):
        mse = {
            method: _ci([float(record["methods"][method]["test_mse"]) for record in records])
            for method in METHODS
        }
        pd = {
            method: _ci(
                [float(record["methods"][method]["partition_distance"]) for record in records]
            )
            for method in METHODS
        }
        is_reference = target == "standard" and length == 10
        rows.append(
            Row(
                key=f"sum-{target}-k{length}",
                label=f"{'Sum' if target == 'standard' else 'Alternating sum'} K={length}",
                mse=mse,
                pd=pd,
                paper_mse=(float(paper_direct["l2_mean"]), float(paper_direct["l2_ci95"]))
                if is_reference
                else None,
                paper_note="Table A1" if is_reference else "Figure 7 only",
            )
        )
    return rows


def _multitask_rows(root: Path, reference: dict[str, Any]) -> list[Row]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in root.rglob("multitask_summary.json"):
        record = _load(path)
        metadata_json = json.dumps(record.get("benchmark_metadata", {}), sort_keys=True)
        grouped[(record["benchmark"], metadata_json, record["latent_mode"])].append(record)
    exact_pd = {
        int(item["assignment_items"]): float(item["partition_distance"])
        for item in reference["cross_correlation_exact_statements"]
    }
    rows: list[Row] = []
    for (benchmark, metadata_json, latent_mode), records in sorted(grouped.items()):
        selected = min(records, key=lambda record: float(record["best_hard_validation_mse"]))
        metadata = json.loads(metadata_json)
        if benchmark == "valid_cross_correlation":
            items = int(metadata["input_length"] * metadata["output_length"])
            label = f"MT Cross-corr A={items}, {latent_mode} z (best val/{len(records)})"
            paper_pd = (exact_pd[items], 0.0) if items in exact_pd else None
            note = "Sec. 5.3" if paper_pd else "Figure 8 only"
        else:
            variance = float(metadata["noise_std"]) ** 2
            label = (
                f"MT Denoising K={metadata['signal_length']}, var={variance:g}, "
                f"{latent_mode} z (best val/{len(records)})"
            )
            paper_pd, note = None, "Figure A1 only"
        first = records[0]["methods"]
        mse = {
            "generated": (
                float(selected["methods"]["generated"]["mse_mean"]),
                float(selected["methods"]["generated"]["mse_ci95"]),
            ),
            "no_sharing": (float(first["no_sharing"]["mse_mean"]), float(first["no_sharing"]["mse_ci95"])),
            "oracle": (float(first["oracle"]["mse_mean"]), float(first["oracle"]["mse_ci95"])),
        }
        pd = {
            "generated": (
                float(selected["methods"]["generated"]["partition_distance_mean"]),
                float(selected["methods"]["generated"]["partition_distance_ci95"]),
            ),
            "no_sharing": (
                float(first["no_sharing"]["partition_distance_mean"]),
                float(first["no_sharing"]["partition_distance_ci95"]),
            ),
            "oracle": (
                float(first["oracle"]["partition_distance_mean"]),
                float(first["oracle"]["partition_distance_ci95"]),
            ),
        }
        rows.append(
            Row(
                key=f"multitask-{benchmark}-{latent_mode}-{metadata_json}",
                label=label,
                mse=mse,
                pd=pd,
                paper_pd=paper_pd,
                paper_note=note,
            )
        )
    return rows


def collect_rows(root: Path, reference_path: Path) -> list[Row]:
    reference = _load(reference_path)
    multitask = _multitask_rows(root, reference)
    linear = [] if multitask else _linear_rows(root, reference)
    return _gaussian_rows(root, reference) + _sum_rows(root, reference) + multitask + linear


def _format(value: tuple[float, float] | None) -> str:
    return "—" if value is None else f"{value[0]:.5g} ± {value[1]:.2g}"


def _table(rows: list[Row], metric: str) -> list[str]:
    lines = [
        f"| Benchmark | Paper direct A | Our direct A | Generated A | No sharing | Oracle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = row.mse if metric == "mse" else row.pd
        paper = row.paper_mse if metric == "mse" else row.paper_pd
        paper_cell = _format(paper)
        if paper is None:
            note = row.paper_note
            if metric == "mse" and row.key.startswith("valid_cross_correlation"):
                note = "Figure 8 only"
            paper_cell = f"— ({note})"
        lines.append(
            f"| {row.label} | {paper_cell} | {_format(values.get('direct'))} | "
            f"{_format(values.get('generated'))} | {_format(values.get('no_sharing'))} | "
            f"{_format(values.get('oracle'))} |"
        )
    return lines


def _plot(rows: list[Row], output: Path) -> None:
    plotted = [row for row in rows if row.mse]
    if not plotted:
        return
    methods = ("no_sharing", "direct", "generated", "oracle")
    colors = ("#7A8492", "#2864DC", "#D94F70", "#18996A")
    width = 0.19
    figure, axis = plt.subplots(figsize=(max(9, len(plotted) * 1.1), 5.4))
    locations = list(range(len(plotted)))
    for offset, (method, color) in enumerate(zip(methods, colors)):
        values = [row.mse.get(method, (float("nan"), 0.0))[0] for row in plotted]
        errors = [row.mse.get(method, (float("nan"), 0.0))[1] for row in plotted]
        x = [position + (offset - 1.5) * width for position in locations]
        axis.bar(x, values, width, yerr=errors, label=method, color=color, capsize=2)
    axis.set_yscale("log")
    axis.set_ylabel("Test MSE (log scale)")
    axis.set_xticks(locations, [row.label for row in plotted], rotation=32, ha="right")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=4)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(rows: list[Row], output: Path, plot_path: Path) -> None:
    lines = [
        "# Yeh et al. (2022): direct versus generated parameter sharing",
        "",
        "Общая таблица сопоставляет опубликованные числа, нашу реализацию прямой "
        "оптимизации assignment-матрицы и параметризацию `A = Gψ(z)`.",
        "",
        "## Test loss",
        "",
    ]
    if rows:
        lines.extend(_table(rows, "mse"))
        lines.extend(["", f"![Сравнение test MSE]({plot_path.name})", "", "## Partition distance", ""])
        lines.extend(_table(rows, "pd"))
    else:
        lines.append("Завершённые запуски пока не найдены.")
    lines.extend(
        [
            "",
            "## Границы сравнения",
            "",
            "- Числа paper указаны только там, где они напечатаны в таблице или явно "
            "сформулированы в тексте; значения с графиков не выдаются за точные.",
            "- Официальный release не содержит `projects/ConvSharing`, поэтому наши "
            "cross-correlation и denoising являются документированной реализацией постановки, "
            "а не побитовой репликацией кода авторов.",
            "- Gaussian: paper пишет Adam, опубликованный код запускает RMSprop; конкретный "
            "optimizer сохраняется в каждой строке результата.",
            "- † В текущем Gaussian-протоколе один `Gψ` совместно обучается на всех "
            "оцениваемых Monte Carlo задачах. Это transductive multi-task результат, а не "
            "проверка переноса на новые задачи.",
            "- Multi-task linear: один `Gψ` обучается на пяти задачах; показаны варианты "
            "с общим и task-specific `z`. Используется exact constrained lower loss и "
            "бинарный forward со straight-through gradient.",
            "- Настройки выбраны только по validation на минимальном benchmark: cross-correlation "
            "`lr=3e-4, ridge=1e-2`; denoising `lr=1e-2, ridge=1e-1`. Для результата выбирается "
            "лучший по validation из четырёх рестартов.",
            "- Sum-of-numbers включает математически необходимый множитель `alpha` в "
            "Neumann-ряде; helper официального release его опускает.",
            "",
            "Источники: [статья](https://proceedings.mlr.press/v151/yeh22b/yeh22b.pdf), "
            "[официальный код](https://github.com/raymondyeh07/equivariance_discovery).",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).with_name("paper_reference.json"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot = args.output.with_name(args.output.stem + "_mse.png")
    rows = collect_rows(args.results_root, args.reference)
    _plot(rows, plot)
    write_report(rows, args.output, plot)
    print(f"wrote {args.output} ({len(rows)} configurations)")


if __name__ == "__main__":
    main()
