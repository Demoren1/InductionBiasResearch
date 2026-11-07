"""Report the registered G25 confirmation without changing any fitted model."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle
from scipy.stats import beta

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .shared_structure_numeric import canonical_index

LABELS = {
    "bootstrap": "Frozen bootstrap",
    "bootstrap_single_fixed": "Calibrated single/fixed",
    "bootstrap_joint_fixed": "Calibrated joint/fixed",
}


def write_report(root):
    plan = json.loads((root / "protocol.json").read_text())
    source = json.loads((root / "common/audit.json").read_text())
    targets = json.loads((root / "transfer/audit.json").read_text())
    if source["selection_freeze_sha256"] != digest(
        root / "common/selection_frozen.json"
    ):
        raise RuntimeError("Source audit does not match frozen I")
    if targets["target_fit_freeze_sha256"] != digest(root / "transfer/fit_frozen.json"):
        raise RuntimeError("Target audit does not match the complete target fit freeze")
    records = source["records"]
    report = root / "report"
    report.mkdir(exist_ok=False)
    rows = []
    for noise in plan["noise"]:
        for method in LABELS:
            cases = [
                r for r in records if r["noise"] == noise and r["method"] == method
            ]
            adapted = [
                r
                for r in targets["records"]
                if r["noise"] == noise and r["method"] == method
            ]
            if len(cases) != len(plan["families"]) or len(adapted) != len(cases):
                raise RuntimeError("Incomplete confirmation source/target audit")
            successes = sum(r["exact_I"] for r in cases)
            n = len(cases)
            complete = [r for r in adapted if r["complete"]]
            rows.append(
                {
                    "noise": noise,
                    "method": method,
                    "cases": n,
                    "exact_I": successes,
                    "exact_mask": sum(r["exact_mask"] for r in cases),
                    "exact_unsigned_I": sum(r["exact_unsigned_I"] for r in cases),
                    "success_CI95_low": float(
                        beta.ppf(0.025, successes, n - successes + 1)
                    )
                    if successes
                    else 0.0,
                    "success_CI95_high": float(
                        beta.ppf(0.975, successes + 1, n - successes)
                    )
                    if successes < n
                    else 1.0,
                    "no_candidate": sum(not r["complete"] for r in cases),
                    "bank_contains_true_I": sum(
                        r["bank_contains_true_I"] for r in cases
                    ),
                    "selection_failure_with_true_I": sum(
                        r["bank_contains_true_I"] and not r["exact_I"] for r in cases
                    ),
                    "completed_target_families": len(complete),
                    "completed_target_tasks": len(complete) * plan["targets"],
                    "mean_clean_target_MSE_completed_only": float(
                        np.mean([r["mean_target_clean_mse"] for r in complete])
                    )
                    if complete
                    else None,
                    "mean_excess_MSE_completed_only": float(
                        np.mean([r["mean_excess_mse_over_true_I"] for r in complete])
                    )
                    if complete
                    else None,
                }
            )
    with (report / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        report / "family_metrics.json",
        {"source": records, "target": targets["records"]},
    )

    strata = []
    for noise in plan["noise"]:
        for method in LABELS:
            for variant in ("local", "quartet_mix", "random"):
                subset = [
                    r
                    for r in records
                    if r["noise"] == noise
                    and r["method"] == method
                    and r["family"]["variant"] == variant
                ]
                successes = sum(r["exact_I"] for r in subset)
                n = len(subset)
                strata.append(
                    {
                        "noise": noise,
                        "method": method,
                        "variant": variant,
                        "cases": n,
                        "exact_I": successes,
                        "CI95_low": float(beta.ppf(0.025, successes, n - successes + 1))
                        if successes
                        else 0.0,
                        "CI95_high": float(
                            beta.ppf(0.975, successes + 1, n - successes)
                        )
                        if successes < n
                        else 1.0,
                    }
                )
    save_json(report / "stratum_intervals.json", {"strata": strata})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, noise in zip(axes, plan["noise"], strict=True):
        group = [r for r in rows if r["noise"] == noise]
        x = np.arange(len(group))
        rates = np.array([r["exact_I"] / r["cases"] for r in group])
        errors = np.array(
            [
                [rate - r["success_CI95_low"], r["success_CI95_high"] - rate]
                for rate, r in zip(rates, group, strict=True)
            ]
        ).T
        bars = axis.bar(x, rates, color=["#8b9298", "#2e8799", "#76649e"])
        axis.errorbar(x, rates, yerr=errors, fmt="none", color="black", capsize=5)
        axis.bar_label(
            bars, labels=[f"{r['exact_I']}/{r['cases']}" for r in group], padding=5
        )
        axis.set_ylim(0, 1.2)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.set_xticks(
            x, [LABELS[r["method"]].replace(" ", "\n", 1) for r in group], fontsize=10
        )
        axis.set_title(f"SD={noise:g}; exact signed I")
        axis.set_ylabel("Fraction of independent structures")
    fig.suptitle(
        "G25: frozen G24 settings, 12 new structures\nBinomial reference intervals; fixed strata and paired noise panels"
    )
    fig.savefig(report / "confirmation_structure.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for axis, noise in zip(axes, plan["noise"], strict=True):
        names = [*LABELS, "oracle_true_I"]
        for position, method in enumerate(names):
            risks = [
                r["mean_target_clean_mse"]
                for r in targets["records"]
                if r["noise"] == noise and r["method"] == method and r["complete"]
            ]
            axis.scatter(
                np.full(len(risks), position),
                np.maximum(risks, 1e-16),
                facecolors="none",
                edgecolors="C0",
                alpha=0.7,
            )
            if risks:
                axis.scatter(
                    position,
                    max(float(np.mean(risks)), 1e-16),
                    marker="D",
                    color="black",
                    zorder=3,
                )
        axis.set_yscale("log")
        axis.set_xticks(
            range(len(names)),
            [
                LABELS.get(m, "True-I\nC-fit control").replace(" ", "\n", 1)
                for m in names
            ],
            fontsize=9,
        )
        axis.set_title(f"Fresh target C, SD={noise:g}")
        axis.set_ylabel("Clean target MSE (8 new tasks / structure)")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Fixed I, independently reset C; circles are structure means\nAll target predictors frozen before any test-label audit"
    )
    fig.savefig(report / "confirmation_transfer.png", dpi=170)
    plt.close(fig)

    with np.load(root / "common/audit_only.npz") as truth:
        true_indices = [
            truth[f"family_{i}__truth_I"] for i in range(len(plan["families"]))
        ]
    canonical_truth = [
        canonical_index(index, plan["codes"]).tobytes() for index in true_indices
    ]
    old_path = Path(plan["pilot"]) / "common/audit_only.npz"
    with np.load(old_path) as old:
        old_signatures = {
            canonical_index(old[key], plan["codes"]).tobytes() for key in old.files
        }
    geometry_audit = {
        "declared_families": len(true_indices),
        "distinct_new_I_modulo_gauge": len(set(canonical_truth)),
        "overlap_with_old_six_I": sum(s in old_signatures for s in canonical_truth),
        "audit_only_no_replacement": True,
    }
    save_json(report / "geometry_independence_audit.json", geometry_audit)
    worst = max(
        (r for r in records if r["method"] == plan["primary_method"]),
        key=lambda r: (
            not r["complete"],
            r.get("categorical_errors", 0),
            r["noise"],
            r["family_id"],
        ),
    )
    for noise in plan["noise"]:
        fig, axes = plt.subplots(
            len(true_indices),
            4,
            figsize=(13, 3 * len(true_indices)),
            constrained_layout=True,
        )
        colormap = plt.get_cmap("RdBu_r", 9)
        normalization = BoundaryNorm(np.arange(-4.5, 5.5), 9)
        compact = None
        for family, expected in enumerate(true_indices):
            case = family * len(plan["noise"]) + plan["noise"].index(noise)
            panels = [(expected, "True I")]
            with np.load(root / "common/selected" / f"job_{case:03d}.npz") as selected:
                for method, label in LABELS.items():
                    record = next(
                        r
                        for r in records
                        if r["family_id"] == family
                        and r["noise"] == noise
                        and r["method"] == method
                    )
                    if record["complete"]:
                        _, aligned, _ = align_indices(
                            selected[f"{method}__common4__index"],
                            expected,
                            plan["codes"],
                        )
                        panels.append(
                            (aligned, f"{label}\n{record['categorical_errors']} errors")
                        )
                    else:
                        panels.append((None, label + "\nNo complete candidate"))
            if family == worst["family_id"] and noise == worst["noise"]:
                compact = panels
            for axis, (value, label) in zip(axes[family], panels, strict=True):
                if value is not None:
                    rendered = axis.imshow(value, cmap=colormap, norm=normalization)
                    for r, c in np.argwhere(value != expected):
                        axis.add_patch(
                            Rectangle(
                                (c - 0.5, r - 0.5),
                                1,
                                1,
                                fill=False,
                                edgecolor="black",
                                linewidth=1.3,
                            )
                        )
                else:
                    axis.text(
                        0.5,
                        0.5,
                        "No model",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                    )
                axis.set_title(label, fontsize=9)
                axis.set_xticks((0, 15))
                axis.set_yticks((0, 15))
            axes[family, 0].set_ylabel(
                f"{plan['families'][family]['variant']}\n{plan['families'][family]['family']}"
            )
        fig.colorbar(
            rendered,
            ax=axes.ravel().tolist(),
            ticks=np.arange(-4, 5),
            shrink=0.3,
            label="Signed index",
        )
        fig.suptitle(
            f"G25 independent confirmation, SD={noise:g}; matching only after frozen-I audit"
        )
        fig.savefig(report / f"confirmation_indexators_sd{noise:g}.png", dpi=130)
        plt.close(fig)
        if compact is not None:
            fig, axes = plt.subplots(1, 4, figsize=(12, 3.7), constrained_layout=True)
            expected = true_indices[worst["family_id"]]
            for axis, (value, label) in zip(axes, compact, strict=True):
                if value is not None:
                    rendered = axis.imshow(value, cmap=colormap, norm=normalization)
                    for r, c in np.argwhere(value != expected):
                        axis.add_patch(
                            Rectangle(
                                (c - 0.5, r - 0.5),
                                1,
                                1,
                                fill=False,
                                edgecolor="black",
                                linewidth=1.5,
                            )
                        )
                axis.set_title(label, fontsize=10)
                axis.set_xticks((0, 15))
                axis.set_yticks((0, 15))
            fig.colorbar(
                rendered,
                ax=axes.tolist(),
                ticks=np.arange(-4, 5),
                shrink=0.65,
                label="Signed index",
            )
            fig.suptitle(
                f"G25: selected diagnostic case {worst['family']['family']}, SD={noise:g}"
            )
            fig.savefig(report / "confirmation_hard_case.png", dpi=170)
            plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    curve_counts = []
    for axis, noise in zip(axes, plan["noise"], strict=True):
        points = {}
        count = 0
        for row in targets["records"]:
            if (
                row["method"] != plan["primary_method"]
                or row["noise"] != noise
                or not row["complete"]
            ):
                continue
            bank = json.loads(
                (root / "transfer/banks" / f"job_{row['job_id']:03d}.json").read_text()
            )
            signature = bank["candidates"][row["candidate"]]["signature"]
            fitted = json.loads(
                (
                    root
                    / "transfer/fits"
                    / f"job_{row['job_id']:03d}"
                    / f"{signature}.json"
                ).read_text()
            )
            for history in fitted["numeric_fits"]:
                count += 1
                start = history["selected_start"]
                for point in history["history"]:
                    point_values = points.setdefault(point["step"], [[], []])
                    for destination, key in zip(
                        point_values, ("train_mse", "validation_mse"), strict=True
                    ):
                        destination.append(
                            float(np.asarray(point[key]).reshape(-1)[start])
                        )
        curve_counts.append(count)
        steps = sorted(points)
        for component, label in enumerate(("Support loss", "Numeric-validation loss")):
            axis.plot(
                steps,
                [np.mean(points[step][component]) for step in steps],
                marker="o",
                markersize=3,
                label=label,
            )
        axis.set_yscale("log")
        axis.set_xlabel("Gauss-Newton step")
        axis.set_ylabel("Observed MSE, selected signed start")
        axis.set_title(f"Primary target C fitting, SD={noise:g}; {count} tasks")
        axis.legend()
    fig.suptitle("Actual saved optimization steps; no synthetic or extrapolated points")
    fig.savefig(report / "confirmation_target_loss.png", dpi=170)
    plt.close(fig)

    primary = [r for r in rows if r["method"] == plan["primary_method"]]
    replicated = all(r["exact_I"] == r["cases"] for r in primary)
    source_seconds = json.loads(
        (Path(plan["parent"]) / "source_execution.json").read_text()
    )
    target_seconds = json.loads(
        (root / "transfer/execution_measurement.json").read_text()
    )
    summary = {
        "comparison": rows,
        "strict_primary_replication_succeeded": replicated,
        "geometry_audit": geometry_audit,
        "independent_structure_count": len(set(canonical_truth)),
        "paired_noise": True,
        "new_HPO_or_algorithm_change": False,
        "primary_target_curve_tasks": curve_counts,
        "source_execution": source_seconds,
        "target_execution": target_seconds,
        "protocol_sha256": digest(root / "protocol.json"),
        "selection_freeze_sha256": source["selection_freeze_sha256"],
        "target_fit_freeze_sha256": targets["target_fit_freeze_sha256"],
        "no_general_novelty_or_competitor_superiority_claim": True,
    }
    save_json(report / "summary.json", summary)
    lines = [
        "# G25: независимый контроль фиксированного метода G24",
        "",
        "До получения новых данных зафиксированы настройки G24, 12 семейств (по четыре local/quartet_mix/random), четыре source-C и восемь новых target-C на I. Гиперпараметры, query-сетки, повторы, integer-solvers и численные старты не менялись. Основной метод — calibrated single/fixed, вторичные — bootstrap и calibrated joint/fixed. Отказы не исключались; неудачные семейства не заменялись.",
        "",
        "|SD|Метод|Exact mask|Exact unsigned I|Exact signed I|Справочный binomial95% CI|Нет кандидата|",
        "|---:|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"|{row['noise']:g}|{LABELS[row['method']]}|{row['exact_mask']}/{row['cases']}|{row['exact_unsigned_I']}/{row['cases']}|{row['exact_I']}/{row['cases']}|[{row['success_CI95_low']:.3f}, {row['success_CI95_high']:.3f}]|{row['no_candidate']}|"
        )
    lines.extend(
        [
            "",
            "![Точная структура](confirmation_structure.png)",
            "",
            f"Строгий заранее заданный критерий 12/12 основного метода на каждом шуме: {'выполнен' if replicated else 'не выполнен'}. Это результат одной независимой контрольной серии; повторного HPO по её исходам не было.",
            "",
            f"После позднего аудита различны {geometry_audit['distinct_new_I_modulo_gauge']}/12 новых I, совпадений с прежними шестью G24: {geometry_audit['overlap_with_old_six_I']}. Два уровня шума парные, это не 24 независимые геометрии. Объединённый binomial-интервал справочный: одинаковая вероятность успеха трёх классов не предполагается. Отдельные Clopper–Pearson интервалы внутри каждого класса сохранены в [stratum_intervals.json](stratum_intervals.json). Эти оценки не доказывают успех на всех структурах.",
            "",
            "|SD|Метод|Завершённые target-задачи|Clean target MSE|Excess к true-I fit|",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        risk, excess = (
            row["mean_clean_target_MSE_completed_only"],
            row["mean_excess_MSE_completed_only"],
        )
        lines.append(
            f"|{row['noise']:g}|{LABELS[row['method']]}|{row['completed_target_tasks']}/96|{risk:.6g}|{excess:.6g}|"
            if risk is not None
            else f"|{row['noise']:g}|{LABELS[row['method']]}|0/96|—|—|"
        )
    lines.extend(
        [
            "",
            "MSE сначала усреднена по восьми новым функциям одной геометрии, затем по геометриям. При отказе численная MSE приводится только для завершённых случаев, их число указано явно; отказ не считается нулевой ошибкой. Все target-C обучены с нуля при фиксированном I. Истинные C не используются даже в true-I численном контроле. Все target-предсказатели заморожены до первого target-test аудита.",
            "",
            "![Перенос](confirmation_transfer.png)",
            "",
            "![Настоящие loss-кривые](confirmation_target_loss.png)",
            "",
            "Кривые показывают фактически сохранённые шаги выбранных по numeric-validation стартов. Для остановившихся fit не дорисовывались будущие значения; на поздних шагах усредняются только имеющиеся наблюдения.",
            "",
            "![Выбранный диагностический случай](confirmation_hard_case.png)",
            "",
            "Полные индексаторы: [SD0.2](confirmation_indexators_sd0.2.png), [SD2](confirmation_indexators_sd2.png). Цвет — целый индекс; сопоставление имён/знаков групп и hidden columns выполнено только после фиксации всех решений.",
            "",
            "Обучающие данные и численные бюджеты унаследованы без изменений. Включая предварительный банк и новые G24-панели, цена составляет 275381223424 физических шумных ответов на source-функцию и условие, хотя Gaussian-средние симулируются аналитически. На target-функцию support4096 и validation1024, по64повтора; test8192. Это прежний known-common-bias/readout класс, не изображения, не свободные neuron-bias и не новое соревнование семи методов.",
            "",
            f"Парное измерение source C-fit: serial{source_seconds['serial_seconds']:.2f}с, batch{source_seconds['batch_seconds']:.2f}с; ускорение{source_seconds['speedup']:.2f}×. Target C-fit: {target_seconds['speedup']:.2f}×. Это отдельные фазы с проверенным численным соответствием. CPU использован для prefix-поиска, GPU только0–3 для пакетных численных fits; GPU6/7 и чужие процессы не затрагивались.",
            "",
            "Исходные данные: [comparison.csv](comparison.csv), [семейственные метрики](family_metrics.json), [summary/provenance](summary.json), [протокол](../protocol.json). Новый контроль завершён; дальнейшее расширение метода в него не входит.",
            "",
        ]
    )
    (report / "REPORT.md").write_text("\n".join(lines))
