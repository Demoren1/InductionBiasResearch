"""Aggregate and plot deterministic AE source-map diagnostics by seed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t

from .followup_common import FOLLOWUP, SEEDS
from .run import write_json


LABELS = {
    "ae_flat_consensus": "Flat AE: consensus + BCE",
    "ae_flat_hungarian": "Flat AE: Hungarian BCE",
    "ae_set_hungarian": "Set AE: Hungarian BCE",
    "saved_vae": "Сохранённый VAE",
    "constant_task_mean": "Средняя карта задачи",
    "constant_global_consensus": "Общее среднее",
    "constant_medoid": "Медоид банка",
}
AE_METHODS = ("ae_flat_consensus", "ae_flat_hungarian", "ae_set_hungarian")
COLORS = {
    "ae_flat_consensus": "#4477AA", "ae_flat_hungarian": "#EE7733",
    "ae_set_hungarian": "#228833", "saved_vae": "#AA3377",
    "constant_task_mean": "#66CCEE", "constant_global_consensus": "#BBBBBB",
    "constant_medoid": "#CCBB44",
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _interval(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n_seeds": 0, "mean": None, "sd": None, "ci95_low": None, "ci95_high": None,
                "per_seed": []}
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if array.size > 1 else 0.0
    half = float(t.ppf(.975, array.size - 1) * sd / np.sqrt(array.size)) if array.size > 1 else 0.0
    return {"n_seeds": int(array.size), "mean": mean, "sd": sd,
            "ci95_low": mean - half, "ci95_high": mean + half,
            "per_seed": array.tolist()}


def _mean_nested_rows(rows: list[dict], path: tuple[str, ...]) -> float:
    value: Any = rows
    for key in path:
        value = [row[key] for row in value]
    return float(np.mean(value))


def _seed_method_metric(data: dict[str, Any], method: str, key: str) -> float:
    return float(data["summary"][method][key])


def _agreement_seed_metrics(value: dict[str, Any]) -> dict[str, float]:
    initial_dist = value["initial_agreement_mse_by_start"]
    final_dist = value["final_agreement_mse_by_start"]
    raw = value["final_raw_unmatched_soft_mse_by_start"]
    iou = value["final_hard_hungarian_column_iou_across_tasks_by_start"]
    null_iou = value["random_exact_k_hard_hungarian_column_iou_null_by_start"]
    anchors = np.asarray(value["final_anchor_bce_per_entry_by_task_start"], dtype=np.float64)
    softness = np.asarray(value["final_softness_by_task_start"], dtype=np.float64)
    distances = np.asarray(value["final_nearest_training_code_distance_by_task_start"], dtype=np.float64)
    radii = np.asarray(value["code_support_radius_by_task"], dtype=np.float64)
    normalized_distance = distances / np.maximum(radii[:, None], 1e-12)
    direct_radii = np.asarray(value["direct_half_median_train_nn_distance_by_task"], dtype=np.float64)
    configured_radius_fraction = radii / np.maximum(direct_radii, 1e-12)
    return {
        "initial_matched_soft_mse": float(np.mean(initial_dist)),
        "final_matched_soft_mse": float(np.mean(final_dist)),
        "final_raw_unmatched_soft_mse": float(np.mean(raw)),
        "final_hard_matched_column_iou": float(np.mean(iou)),
        "random_exact_k_hard_matched_column_iou_null": float(np.mean(null_iou)),
        "final_raw_bank_anchor_bce_per_entry": float(anchors.mean()),
        "final_softness": float(softness.mean()),
        "uniform_soft_null_softness": float(value["uniform_soft_consensus_null"]["softness_p_times_one_minus_p"]),
        "final_nearest_code_distance": float(distances.mean()),
        "final_nearest_code_distance_over_radius": float(normalized_distance.mean()),
        "maximum_nearest_code_distance_over_radius": float(normalized_distance.max()),
        "zero_radius_task_count": float((radii <= 1e-12).sum()),
        "configured_radius_over_direct_half_median_nn": float(configured_radius_fraction.mean()),
    }


def _audit_code_support(folder: Path) -> dict[str, Any]:
    """Recompute latent distances by float64 direct residuals."""
    audit_path = folder / "support_distance_audit.json"
    diagnostic_path = folder / "mask_diagnostics.json"
    diagnostics = _read(diagnostic_path)
    if audit_path.exists() and _read(audit_path).get("audit_version") == 3:
        audit = _read(audit_path)
    else:
        codes_path = folder / "agreement_codes.pt"
        if codes_path.exists():
            code_bundle = torch.load(codes_path, map_location="cpu", weights_only=False)
        else:
            full_artifacts = torch.load(folder / "ae_artifacts.pt", map_location="cpu", weights_only=False)
            code_bundle = full_artifacts["agreement_codes"]
            del full_artifacts
        audit = {"audit_version": 3,
                 "distance_metric": "direct float64 residual L2 norm to nearest encoded training code",
                 "methods": {}}
        for method, method_codes in code_bundle.items():
            old = diagnostics["agreement"][method]
            radii = [float(value) for value in old["code_support_radius_by_task"]]
            initial_rows = []
            final_rows = []
            ratios = []
            direct_half_medians = []
            for task, support in enumerate(method_codes["training_codes_by_task"]):
                support64 = support.to(torch.float64)
                support_distances = (support64[:, None, :] - support64[None, :, :]).square().sum(-1).sqrt()
                support_distances.fill_diagonal_(float("inf"))
                direct_half_median = float(support_distances.min(dim=1).values.median()) * .5
                direct_half_medians.append(direct_half_median)
                initial = method_codes["initial_codes"][task].to(torch.float64)
                final = method_codes["final_codes"][task].to(torch.float64)
                initial_distance = (initial[:, None, :] - support64[None, :, :]).square().sum(-1).sqrt().min(1).values
                final_distance = (final[:, None, :] - support64[None, :, :]).square().sum(-1).sqrt().min(1).values
                initial_rows.append(initial_distance.tolist())
                final_rows.append(final_distance.tolist())
                ratios.append(float(final_distance.max()) / max(radii[task], 1e-12))
            method_audit = {
                "code_support_radius_by_task": radii,
                "zero_radius_task_count": int(sum(radius <= 1e-12 for radius in radii)),
                "direct_half_median_train_nn_distance_by_task": direct_half_medians,
                "legacy_float32_cdist_radius_by_task": old.get("legacy_float32_cdist_radius_by_task", []),
                "configured_radius_over_direct_half_median_nn_by_task": [
                    radii[task] / max(direct_half_medians[task], 1e-12)
                    for task in range(len(radii))],
                "initial_nearest_training_code_distance_by_task_start": initial_rows,
                "final_nearest_training_code_distance_by_task_start": final_rows,
                "maximum_final_distance_over_radius_by_task": ratios,
            }
            audit["methods"][method] = method_audit
            old.update(method_audit)
            old["code_support_distance_metric"] = audit["distance_metric"]
            old["code_support_rule"] = (
                "the repaired run uses exactly 0.5 times the median nearest-neighbor distance among train codes; "
                "float64 direct residual norms are used for radii, nearest centers, projection, and final distances")
            old["code_support_caveat"] = (
                "the first cdist attempt is archived under cdist_attempt; its masks and target checkpoints are compared bitwise below")
        write_json(audit_path, audit)
        write_json(diagnostic_path, diagnostics)
    return audit


def _plot_reconstruction(metrics: dict[str, Any], out: Path) -> None:
    methods = [method for method in LABELS if method in metrics]
    labels = [LABELS[method] for method in methods]
    positions = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.3))
    panels = [
        ("matched_bce_per_entry", "Hungarian BCE на элемент (меньше лучше)"),
        ("topk_matched_iou", "IoU между matched top-K масками (больше лучше)"),
    ]
    for axis, (metric, title) in zip(axes, panels):
        means, lows, highs = [], [], []
        for method in methods:
            result = metrics[method][metric]
            means.append(result["mean"])
            lows.append(result["mean"] - result["ci95_low"])
            highs.append(result["ci95_high"] - result["mean"])
        axis.bar(positions, means, color=[COLORS[method] for method in methods], alpha=.86)
        axis.errorbar(positions, means, yerr=[lows, highs], fmt="none", ecolor="#222222",
                      capsize=4, linewidth=1.2)
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=32, ha="right")
        axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Среднее по seed; интервалы t-распределения 95%")
    fig.suptitle("Реконструкция 24 отложенных карт на seed, перестановочно-инвариантные метрики")
    fig.tight_layout()
    fig.savefig(out / "reconstruction_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "reconstruction_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_diversity(metrics: dict[str, Any], out: Path) -> None:
    methods = [method for method in LABELS if method in metrics]
    labels = [LABELS[method] for method in methods]
    positions = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.3))
    for axis, key, title in (
        (axes[0], "within_task_diversity_ratio_prediction_over_target",
         "Разнообразие внутри задачи / разнообразие целей"),
        (axes[1], "cross_task_diversity_ratio_prediction_over_target",
         "Разнообразие между задачами / разнообразие целей"),
    ):
        means, lows, highs = [], [], []
        for method in methods:
            value = metrics[method][key]
            means.append(value["mean"])
            lows.append(value["mean"] - value["ci95_low"])
            highs.append(value["ci95_high"] - value["mean"])
        axis.bar(positions, means, color=[COLORS[method] for method in methods], alpha=.86)
        axis.errorbar(positions, means, yerr=[lows, highs], fmt="none", ecolor="#222222",
                      capsize=4, linewidth=1.2)
        axis.axhline(1., color="#555555", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=32, ha="right")
        axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Среднее по seed; интервалы t-распределения 95%")
    fig.suptitle("Разнообразие оценивается отдельно внутри и между четырьмя source-задачами")
    fig.tight_layout()
    fig.savefig(out / "reconstruction_diversity.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "reconstruction_diversity.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_agreement(agreement: dict[str, dict[str, dict[str, Any]]], out: Path) -> None:
    methods = [method for method in AE_METHODS if method in agreement]
    positions = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    panels = [
        ("final_matched_soft_mse", "Согласие SoftTopK после matching (меньше лучше)"),
        ("final_hard_matched_column_iou", "Hungarian IoU hard top-K"),
        ("final_raw_bank_anchor_bce_per_entry", "BCE к ближайшей raw карте банка"),
    ]
    for axis, (metric, title) in zip(axes, panels):
        values = [agreement[method][metric] for method in methods]
        means = [value["mean"] for value in values]
        lows = [value["mean"] - value["ci95_low"] for value in values]
        highs = [value["ci95_high"] - value["mean"] for value in values]
        axis.bar(positions, means, color=[COLORS[method] for method in methods], alpha=.86)
        axis.errorbar(positions, means, yerr=[lows, highs], fmt="none", ecolor="#222222",
                      capsize=4, linewidth=1.2)
        axis.set_title(title)
        axis.set_xticks(positions, [LABELS[method] for method in methods], rotation=28, ha="right")
        axis.grid(axis="y", alpha=.25)
    fig.suptitle("Поиск масок: soft agreement, перестановочно-matched hard IoU и привязка к raw банку")
    fig.tight_layout()
    fig.savefig(out / "agreement_diagnostics.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "agreement_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)


def build_report(out: Path) -> dict[str, Any]:
    reconstruction_by_method: dict[str, dict[str, list[float]]] = {}
    agreement_by_method: dict[str, dict[str, list[float]]] = {}
    parameter_counts: dict[str, list[float]] = {method: [] for method in AE_METHODS}
    repair_reports: dict[int, dict[str, Any]] = {}
    per_seed: dict[str, Any] = {}
    missing: list[int] = []
    for seed in SEEDS:
        folder = out / f"seed_{seed}"
        reconstruction_path = folder / "reconstruction_metrics.json"
        agreement_path = folder / "mask_diagnostics.json"
        history_path = folder / "training_histories.json"
        if not (reconstruction_path.exists() and agreement_path.exists() and history_path.exists()):
            missing.append(seed)
            continue
        recon = _read(reconstruction_path)
        diagnostics = _read(agreement_path)
        histories = _read(history_path)
        support_audit = _audit_code_support(folder)
        diagnostics = _read(agreement_path)
        seed_row: dict[str, Any] = {"reconstruction": recon["summary"], "agreement": {}}
        repair_path = folder / "mask_repair_comparison.json"
        if repair_path.exists():
            repair_reports[seed] = _read(repair_path)
        for method, values in recon["summary"].items():
            if method not in LABELS:
                continue
            reconstruction_by_method.setdefault(method, {})
            for key in ("matched_bce_per_entry", "topk_matched_iou",
                        "within_task_diversity_ratio_prediction_over_target",
                        "cross_task_diversity_ratio_prediction_over_target",
                        "within_task_target_pairwise_matched_l1_per_entry",
                        "within_task_prediction_pairwise_matched_l1_per_entry"):
                reconstruction_by_method[method].setdefault(key, []).append(float(values[key]))
        for method in AE_METHODS:
            if method not in diagnostics.get("agreement", {}):
                continue
            method_row = _agreement_seed_metrics(diagnostics["agreement"][method])
            seed_row["agreement"][method] = method_row
            agreement_by_method.setdefault(method, {})
            for key, value in method_row.items():
                agreement_by_method[method].setdefault(key, []).append(float(value))
            task_counts = [record["parameter_count"] for record in histories["models"][method]]
            parameter_counts[method].append(float(np.mean(task_counts)))
        per_seed[str(seed)] = seed_row
    if not per_seed:
        raise FileNotFoundError(f"No completed AE source diagnostics found under {out}")
    aggregate = {
        "completed_seeds": [int(seed) for seed in per_seed],
        "missing_seeds": missing,
        "uncertainty_unit": "seed; each source-seed first aggregates its 24 held-out maps and 4 source tasks",
        "reconstruction": {
            method: {key: _interval(values) for key, values in metrics.items()}
            for method, metrics in reconstruction_by_method.items()},
        "agreement": {
            method: {key: _interval(values) for key, values in metrics.items()}
            for method, metrics in agreement_by_method.items()},
        "parameter_count": {method: _interval(values) for method, values in parameter_counts.items() if values},
        "mask_repair": _aggregate_mask_repairs(repair_reports),
        "per_seed": per_seed,
    }
    write_json(out / "aggregate_metrics.json", aggregate)
    _plot_reconstruction(aggregate["reconstruction"], out)
    _plot_diversity(aggregate["reconstruction"], out)
    _plot_agreement(aggregate["agreement"], out)
    report = _markdown_report(aggregate)
    (out / "REPORT.md").write_text(report)
    return aggregate


def _aggregate_mask_repairs(reports: dict[int, dict[str, Any]]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method in (*AE_METHODS, "ae_medoid"):
        rows = [(seed, report["masks"][method]) for seed, report in sorted(reports.items())]
        radius_old: list[float] = []
        radius_new: list[float] = []
        old_radius_zeros = 0
        new_radius_zeros = 0
        for _, report in sorted(reports.items()):
            method_row = report.get("legacy_radius_summary_by_method", {}).get(method)
            if method_row:
                radius_old.extend(float(value) for value in method_row["legacy_cdist_radius_by_task"])
                radius_new.extend(float(value) for value in method_row["correct_direct_radius_by_task"])
        methods[method] = {
            "seeds_compared": len(rows),
            "seeds_with_any_mask_change": sum(not row["bitwise_equal"] for _, row in rows),
            "changed_replicas": sum(row["changed_replicas"] for _, row in rows),
            "replicas_compared": sum(row["replicas"] for _, row in rows),
            "flipped_edges_total": sum(row["flipped_edges_total"] for _, row in rows),
            "legacy_cdist_radius_mean_over_seed_tasks": float(np.mean(radius_old)) if radius_old else None,
            "correct_direct_radius_mean_over_seed_tasks": float(np.mean(radius_new)) if radius_new else None,
            "legacy_cdist_zero_radii": sum(value <= 1e-12 for value in radius_old),
            "correct_direct_zero_radii": sum(value <= 1e-12 for value in radius_new),
            "per_seed": {str(seed): row for seed, row in rows},
        }
    return {
        "seeds_compared": len(reports),
        "seeds_requiring_target_refit": sorted(
            seed for seed, report in reports.items() if report["target_refit_required"]),
        "methods": methods,
    }


def _fmt(metric: dict[str, Any], digits: int = 4) -> str:
    if metric["n_seeds"] == 0:
        return "н/д"
    return f"{metric['mean']:.{digits}f} [{metric['ci95_low']:.{digits}f}; {metric['ci95_high']:.{digits}f}]"


def _markdown_report(aggregate: dict[str, Any]) -> str:
    reconstruction = aggregate["reconstruction"]
    repair = aggregate.get("mask_repair", {})
    lines = [
        "# Детерминированные AE для карт DeepSets",
        "",
        f"Завершено seed: {len(aggregate['completed_seeds'])}/8 — {aggregate['completed_seeds']}. "
        f"Ожидают завершения: {aggregate['missing_seeds'] or 'нет'}.",
        "Все интервалы и средние сначала агрегированы внутри seed по 24 отложенным картам и четырём source-задачам. "
        "Интервал 95% рассчитан по восьми (или имеющимся) seed, а не по картам как независимым повторам.",
        "",
        "## Реконструкция отложенных карт",
        "",
        "BCE и exact top-K IoU вычислены одной перестановочно-инвариантной метрикой для трёх AE, сохранённого VAE, "
        "средних карт и медоида. Для каждой пары карт Hungarian выбирает соответствие скрытых колонок отдельно "
        "по BCE и IoU. Ни метрика, ни target-задачи не выбирали маски.",
        "",
        "| Метод | Hungarian BCE на элемент, ниже лучше | Matched top-K IoU, выше лучше | Разнообразие внутри задачи / цели |",
        "|---|---:|---:|---:|",
    ]
    for method, label in LABELS.items():
        if method not in reconstruction:
            continue
        metrics = reconstruction[method]
        lines.append(f"| {label} | {_fmt(metrics['matched_bce_per_entry'])} | "
                     f"{_fmt(metrics['topk_matched_iou'])} | "
                     f"{_fmt(metrics['within_task_diversity_ratio_prediction_over_target'])} |")
    lines += [
        "",
        "## Разнообразие и поиск согласия",
        "",
        "Внутри-task diversity сравнивает шесть отложенных карт одной задачи и помогает обнаружить схлопывание "
        "в постоянное предсказание. Межзадачное значение приведено отдельно и не подменяет эту проверку. "
        "Agreement использует SoftTopK-карты после Hungarian matching; hard IoU также сопоставляет колонки по IoU. "
        "В качестве нуля показаны независимые случайные exact-K маски, а равномерная soft-карта имеет нулевое "
        "согласие по построению и высокую softness.",
        "",
        "| Метод | Soft agreement MSE | Hard matched IoU | Случайный exact-K null | Raw bank-anchor BCE/entry | Softness | Код: mean/max dist/radius; среднее r=0 задач; used/direct radius |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in AE_METHODS:
        if method not in aggregate["agreement"]:
            continue
        metrics = aggregate["agreement"][method]
        support = (f"{_fmt(metrics['final_nearest_code_distance_over_radius'])} / "
                   f"{_fmt(metrics['maximum_nearest_code_distance_over_radius'])}; "
                   f"{_fmt(metrics['zero_radius_task_count'], 2)}/4; "
                   f"radius/direct NN={_fmt(metrics['configured_radius_over_direct_half_median_nn'])}")
        lines.append(
            f"| {LABELS[method]} | {_fmt(metrics['final_matched_soft_mse'])} | "
            f"{_fmt(metrics['final_hard_matched_column_iou'])} | "
            f"{_fmt(metrics['random_exact_k_hard_matched_column_iou_null'])} | "
            f"{_fmt(metrics['final_raw_bank_anchor_bce_per_entry'])} | "
            f"{_fmt(metrics['final_softness'])} | {support} |")
    repair_labels = {**{method: LABELS[method] for method in AE_METHODS},
                     "ae_medoid": "Exact-K медоид"}
    lines += [
        "",
        "## Исправление cdist и побитовая проверка масок",
        "",
        f"Исправление выполнено для {repair.get('seeds_compared', 0)} seed без переобучения AE. "
        f"Изменённые маски требуют повторного target fit для seed {repair.get('seeds_requiring_target_refit', [])}; "
        "его статус сохранён в `target_refit_status.json`. Старые маски и состояния сохранены в `cdist_attempt/`.",
        "",
        "| Метод | Старый cdist radius, среднее | Прямой radius, среднее | Нулей старый → новый | Seed с изменением маски | Изменённые replica | Flipped edges |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in (*AE_METHODS, "ae_medoid"):
        metrics = repair.get("methods", {}).get(method)
        if not metrics:
            continue
        old_radius = metrics.get("legacy_cdist_radius_mean_over_seed_tasks")
        new_radius = metrics.get("correct_direct_radius_mean_over_seed_tasks")
        old_text = f"{old_radius:.6g}" if old_radius is not None else "—"
        new_text = f"{new_radius:.6g}" if new_radius is not None else "—"
        zero_text = (f"{metrics['legacy_cdist_zero_radii']} → {metrics['correct_direct_zero_radii']}"
                     if old_radius is not None and new_radius is not None else "—")
        lines.append(
            f"| {repair_labels[method]} | {old_text} | {new_text} | {zero_text} | "
            f"{metrics['seeds_with_any_mask_change']} | {metrics['changed_replicas']}/{metrics['replicas_compared']} | "
            f"{metrics['flipped_edges_total']} |")
    lines += [
        "",
        "## Параметры и ограничения",
        "",
        f"Среднее число параметров flat AE: {_fmt(aggregate['parameter_count'].get('ae_flat_consensus', {'n_seeds': 0}), 0)}; "
        f"set AE: {_fmt(aggregate['parameter_count'].get('ae_set_hungarian', {'n_seeds': 0}), 0)}. "
        "Flat fixed-loss и flat Hungarian-loss используют одинаковые начальные веса, одну train-only consensus-перестановку "
        "и одинаковые карты; это изолирует loss в пределах этой модели. Set encoder имеет другую ёмкость.",
        "Сохранённый VAE использует исторический checkpoint, выбранный по BCE плюс KL на validation; здесь его "
        "переоценивают общей matched BCE и IoU, но не переотбирают checkpoint. Короткий source-банк содержит 26 "
        "training-карт на задачу. Даже хорошая реконструкция или decoder agreement не гарантирует перенос на новые задачи, "
        "функциональное совпадение скрытых нейронов либо полезность exact-K маски.",
        "Исправленный agreement-search начинается с кодов реальных training-карт. Радиус каждого локального шара равен "
        "0.5 медианы расстояний до ближайшего другого train-code; расстояния до соседей, центров, при проекции и при "
        "финальном аудите вычисляются прямыми float64 residual-нормами. Старые float32 torch.cdist радиусы, маски "
        "и target-состояния архивированы и приведены рядом для аудита. Эта train-only локальная поддержка ограничивает "
        "область поиска, но не доказывает, что она представляет весь кодовый manifold. Raw BCE anchor использует raw "
        "importance до SoftTopK; target labels и held-out source-карты не выбирают коды или маски. Нулевой радиус остаётся "
        "когда медиана прямых расстояний до ближайшего соседа равна нулю.",
        "",
        "![Реконструкция](reconstruction_comparison.png)",
        "Слева — matched BCE на элемент, меньше лучше; справа — совпадение top-K масок после сопоставления колонок, "
        "больше лучше. Средние карты дают контроль постоянного предсказания. Ошибки — 95% t-интервалы по восьми seeds.",
        "",
        "![Разнообразие](reconstruction_diversity.png)",
        "Ось Y — отношение разнообразия реконструкций к разнообразию исходных карт, отдельно внутри одной задачи "
        "и между задачами. Ноль соответствует постоянной реконструкции, единица — исходному уровню разнообразия. "
        "Все AE сохраняют лишь небольшую часть разнообразия карт.",
        "",
        "![Поиск согласия](agreement_diagnostics.png)",
        "Три панели показывают расстояние между согласованными soft-картами (меньше лучше), IoU бинарных масок "
        "(больше лучше) и BCE до ближайшей raw-карты банка (меньше лучше). Высокое внутреннее согласие само по себе "
        "не подтверждает перенос: downstream-кривые и сравнение с random приведены в общем отчёте.",
        "",
        "PDF-копии имеют те же имена. Полные значения по seed сохранены в `aggregate_metrics.json`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FOLLOWUP / "ae")
    args = parser.parse_args()
    aggregate = build_report(args.out.resolve())
    print(json.dumps({"completed_seeds": aggregate["completed_seeds"],
                      "missing_seeds": aggregate["missing_seeds"],
                      "report": str(args.out.resolve() / "REPORT.md")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
