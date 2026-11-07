"""Post-freeze competitive tables, fresh-C transfer and actual search traces."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.evaluate import alignment
from experiments.sharing_benchmark_v1.report import save_figure

from .diagnostic_report import categorical_map

LABELS = {
    "yeh__native_or_decoded": "Yeh: native MAP",
    "yeh__continuous": "Yeh: soft basis",
    "msr_exact__native_or_decoded": "MSR exact-adapt + readout",
    "msr_exact__continuous": "MSR exact-adapt: raw U",
    "lc__native_or_decoded": "LC: native I",
    "lutq__native_or_decoded": "LUT-Q: native I",
    "wsr_prox": "WSR: независимые fit + пересечение равенств",
    "oscar_prox": "OSCAR: независимые fit + пересечение равенств",
    "ols_profiles": "OLS: полные профили",
    "ols_pca": "OLS + PCA-базис",
    "ols_pca_scores": "OLS + PCA с амплитудами",
    "corrected_pca": "Noise-corrected PCA",
    "noise_whitened_pca": "Noise-whitened PCA",
    "signed_contrast_graph": "Полный F-граф",
    "codebook_crossfit": "Cross-fit кодбуков + граф",
    "whitened_pca_direction_crossfit": "PCA-направления + граф",
    "dense_empirical_bayes_ridge": "Плотная ridge-регрессия",
    "oracle_audit": "Истинный I: аудит",
}
ORDER = tuple(LABELS)


def collect(root):
    spec = json.loads((root / "confirmation_registration.json").read_text())
    cases = []
    for job_id in range(len(spec["jobs"])):
        path = root / "confirmation" / f"case_{job_id:03d}" / "audit.json"
        audit = json.loads(path.read_text())
        if audit["frozen_sha256"] != digest(root / "confirmation_models_frozen.json"):
            raise RuntimeError("Audits have different model freeze")
        cases.append(audit)
    rows = []
    for job_id, case in enumerate(cases):
        for record in case["records"]:
            rows.append(
                {
                    "case": job_id,
                    "family": case["job"]["family"]["id"],
                    "kind": case["job"]["family"]["kind"],
                    "regime": case["job"]["name"],
                    **record,
                }
            )
    return spec, cases, rows


def validation_restart_choices(root, spec, output):
    """Select numerical seeds from frozen source-validation scores only."""
    choices = json.loads((root / "hpo_frozen.json").read_text())["choices"]
    selected, provenance = [], {}
    for case_id, job in enumerate(spec["jobs"]):
        folder = root / "confirmation" / f"case_{case_id:03d}"
        for n in job["train_at"]:
            for method in ("yeh", "msr_exact", "lc", "lutq"):
                for representation in ("native_or_decoded", "continuous"):
                    key = f"{method}__{representation}"
                    if key not in choices:
                        continue
                    scores = []
                    for seed in spec["seeds"]:
                        path = folder / f"{method}_seed{seed}_n{n}.npz"
                        provenance[str(path.relative_to(root))] = digest(path)
                        with np.load(path) as archive:
                            row = (
                                archive["selected_original_rows"]
                                .tolist()
                                .index(choices[key]["row"])
                            )
                            score = float(
                                archive[f"{representation}__selection_mse"][row]
                            )
                        scores.append((score, seed))
                    score, seed = min(scores)
                    selected.append(
                        {
                            "case": case_id,
                            "n": n,
                            "method": key,
                            "seed": seed,
                            "source_validation_mse": score,
                        }
                    )
    save_json(
        output / "validation_restart_choices.json",
        {
            "rule": "Minimize frozen source-validation MSE over three numerical seeds; break ties by seed. No structural audit or target values read by this selector.",
            "status": "Additional post-hoc fairness analysis, not a newly preregistered confirmation. All source models and scores had already been frozen before any target evaluation.",
            "frozen_sha256": digest(root / "confirmation_models_frozen.json"),
            "source_model_sha256": provenance,
            "choices": selected,
        },
    )
    return {(r["case"], r["n"], r["method"]): r["seed"] for r in selected}


def selected_restart_rows(rows, choices):
    result = []
    for row in rows:
        if row["seed"] is None:
            result.append(row)
        elif row["seed"] == choices[row["case"], row["n"], row["method"]]:
            result.append({**row, "selected_seed": row["seed"], "seed": None})
    return result


def summary(rows, expected_families, seeds):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["regime"], row["n"], row["method"]].append(row)
    result = []
    for (regime, n, method), values in sorted(grouped.items()):
        structural = [row for row in values if row.get("recovery") is not None]
        families = expected_families // 2 if method == "wsr_prox" else expected_families
        stochastic = any(row["seed"] is not None for row in values)
        expected = families * (len(seeds) if stochastic else 1)
        by_family = defaultdict(list)
        for row in structural:
            by_family[row["family"]].append(row["recovery"]["exact_I"])
        row = {
            "regime": regime,
            "n": n,
            "method": method,
            "expected_runs": expected,
            "observed_runs": len(values),
            "exact": sum(v["recovery"]["exact_I"] for v in structural),
            "all_seed_exact_families": sum(
                len(flags) == (len(seeds) if stochastic else 1) and all(flags)
                for flags in by_family.values()
            ),
            "families": families,
        }
        if structural:
            row["mean_I_errors"] = float(
                np.mean([v["recovery"]["categorical_errors"] for v in structural])
            )
            row["mask_f1"] = float(
                np.mean([v["recovery"]["mask_f1"] for v in structural])
            )
        for support in (1, 4, 16, 64):
            values_with_transfer = [
                v["transfer"][str(support)]["mean_mse"]
                for v in values
                if "transfer" in v
            ]
            if values_with_transfer:
                row[f"mse_{support}"] = float(np.mean(values_with_transfer))
        if any("subspace_error" in v for v in values):
            row["subspace_error"] = float(
                np.mean([v["subspace_error"] for v in values])
            )
        result.append(row)
    return result


def markdown_table(summaries, regime, n):
    selected = {
        row["method"]: row
        for row in summaries
        if row["regime"] == regime and row["n"] == n
    }
    lines = [
        "| Метод | Exact I / запусков | Все seed точны / геометрий | F1 маски | MSE, support4 | MSE, support16 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ORDER:
        if method not in selected:
            if method in (
                "signed_contrast_graph",
                "codebook_crossfit",
                "whitened_pca_direction_crossfit",
            ):
                lines.append(f"|{LABELS[method]}|0/12, I не выдан|0/12|—|—|—|")
            continue
        row = selected[method]
        if method not in (
            "yeh__continuous",
            "msr_exact__continuous",
            "dense_empirical_bayes_ridge",
            "oracle_audit",
        ):
            exact = f"{row['exact']}/{row['expected_runs']}"
            stable = f"{row['all_seed_exact_families']}/{row['families']}"
            f1 = f"{row['mask_f1']:.4f}" if "mask_f1" in row else "—"
        else:
            exact, stable, f1 = "не выдаёт I", "—", "—"
        mse4 = f"{row['mse_4']:.6g}" if "mse_4" in row else "—"
        mse16 = f"{row['mse_16']:.6g}" if "mse_16" in row else "—"
        lines.append(f"|{LABELS[method]}|{exact}|{stable}|{f1}|{mse4}|{mse16}|")
    return "\n".join(lines)


def paired_transfer_intervals(rows):
    """Resample geometries, averaging optimizer seeds within each geometry."""
    values, kinds = defaultdict(list), {}
    for row in rows:
        if row["regime"] == "primary" and row["n"] == 131072 and "transfer" in row:
            values[row["method"], row["family"]].append(
                row["transfer"]["4"]["mean_mse"]
            )
            kinds[row["family"]] = row["kind"]
    reference = "codebook_crossfit"
    rng = np.random.default_rng(261105991)
    result = []
    for method in ORDER:
        if method == reference:
            continue
        families = sorted(
            family
            for family in kinds
            if (method, family) in values and (reference, family) in values
        )
        if not families:
            continue
        differences = np.array(
            [
                np.mean(values[method, family]) - np.mean(values[reference, family])
                for family in families
            ]
        )
        differences[np.abs(differences) < 1e-12] = 0.0
        bootstraps = np.zeros(10000)
        for kind in ("conv", "signed"):
            positions = np.array(
                [i for i, family in enumerate(families) if kinds[family] == kind]
            )
            if len(positions):
                sampled = rng.choice(
                    positions, size=(10000, len(positions)), replace=True
                )
                bootstraps += differences[sampled].sum(-1) / len(families)
        result.append(
            {
                "baseline": method,
                "families": len(families),
                "baseline_minus_codebook_mse4": float(differences.mean()),
                "bootstrap_lower95": float(np.quantile(bootstraps, 0.025)),
                "bootstrap_upper95": float(np.quantile(bootstraps, 0.975)),
            }
        )
    return result


def plots(root, output, spec, cases, rows, summaries, selected_summaries):
    displayed = (
        "yeh__native_or_decoded",
        "lc__native_or_decoded",
        "lutq__native_or_decoded",
        "msr_exact__native_or_decoded",
        "ols_profiles",
        "noise_whitened_pca",
        "codebook_crossfit",
        "whitened_pca_direction_crossfit",
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    for axis, n in zip(axes, (8192, 131072)):
        values = {
            r["method"]: r
            for r in summaries
            if r["regime"] == "primary" and r["n"] == n
        }
        ratios = [
            values.get(m, {}).get("exact", 0)
            / values.get(m, {}).get("expected_runs", 12)
            for m in displayed
        ]
        selected_values = {
            r["method"]: r
            for r in selected_summaries
            if r["regime"] == "primary" and r["n"] == n
        }
        selected_ratios = [
            selected_values.get(m, {}).get("exact", 0)
            / selected_values.get(m, {}).get("expected_runs", 12)
            for m in displayed
        ]
        positions = np.arange(len(displayed))
        axis.barh(positions - 0.18, ratios, height=0.34, label="Все отдельные старты")
        axis.barh(
            positions + 0.18,
            selected_ratios,
            height=0.34,
            label="Старт выбран по валидации",
        )
        axis.set_yticks(np.arange(len(displayed)), [LABELS[m] for m in displayed])
        axis.invert_yaxis()
        axis.set(
            xlim=(0, 1.04),
            xlabel="Доля точных I среди всех запусков",
            title=f"Примеров на source-функцию: {n}",
        )
        axis.grid(axis="x", alpha=0.2)
    axes[1].legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.suptitle("12 новых геометрий; выбор старта использует только source-validation")
    save_figure(fig, output, "confirmation_recovery")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True)
    transfer_methods = (
        *displayed[:4],
        "noise_whitened_pca",
        "codebook_crossfit",
        "oracle_audit",
        "dense_empirical_bayes_ridge",
    )
    for axis, n in zip(axes, (8192, 131072)):
        for method in transfer_methods:
            selected = [
                r
                for r in selected_summaries
                if r["regime"] == "primary"
                and r["n"] == n
                and r["method"] == method
                and "mse_4" in r
            ]
            if selected:
                row = selected[0]
                axis.plot(
                    (1, 4, 16, 64),
                    [row[f"mse_{s}"] for s in (1, 4, 16, 64)],
                    marker="o",
                    label=LABELS[method],
                )
        axis.set(
            xscale="log",
            yscale="log",
            xlabel="Наблюдений support новой функции",
            ylabel="Population clean prediction MSE",
            title=f"Источник: n = {n}",
        )
        axis.set_xticks((1, 4, 16, 64), ("1", "4", "16", "64"))
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Фиксированная структура; старт выбран по validation; новые C на 100 новых функциях"
    )
    save_figure(fig, output, "confirmation_transfer")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True)
    for method in (
        "signed_contrast_graph",
        "codebook_crossfit",
        "whitened_pca_direction_crossfit",
        "noise_whitened_pca",
    ):
        for axis, kind in zip(axes, ("conv", "signed")):
            chosen = [
                r
                for r in rows
                if r["regime"] == "primary"
                and r["kind"] == kind
                and r["method"] == method
            ]
            values = [
                sum(
                    r.get("recovery", {}).get("exact_I", False)
                    for r in chosen
                    if r["n"] == n
                )
                / 6
                for n in spec["observations"]
            ]
            axis.plot(spec["observations"], values, marker="o", label=LABELS[method])
            axis.set(
                xscale="log",
                ylim=(-0.03, 1.03),
                xlabel="Примеров на функцию",
                ylabel="Exact I / 6 геометрий",
                title=kind,
            )
            axis.set_xticks(
                spec["observations"],
                [str(n) for n in spec["observations"]],
                rotation=25,
            )
            axis.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    save_figure(fig, output, "confirmation_data_and_coverage")

    # Prespecified first arbitrary geometry, first numerical seed; no best-case selection.
    case_id = next(
        i
        for i, case in enumerate(cases)
        if case["job"]["family"]["id"] == 7201 and case["job"]["name"] == "primary"
    )
    folder = root / "confirmation" / f"case_{case_id:03d}"
    choices = json.loads((root / "hpo_frozen.json").read_text())["choices"]
    with np.load(folder / "teacher_audit_only.npz") as archive:
        truth = archive["indices"]
    matrices, labels = [truth], ["Истинный I"]
    for method in ("yeh", "lc", "lutq", "msr_exact"):
        with np.load(folder / f"{method}_seed31_n131072.npz") as archive:
            selected = (
                archive["selected_original_rows"]
                .tolist()
                .index(choices[f"{method}__native_or_decoded"]["row"])
            )
            index = archive["native_or_decoded__indices"][selected]
        metrics, aligned = alignment(index, truth, 3)
        matrices.append(aligned)
        label = "MSR + readout" if method == "msr_exact" else method
        labels.append(f"{label}\nошибок I: {metrics['categorical_errors']}")
    with np.load(folder / "statistical_indices.npz") as archive:
        for method in ("ols_profiles", "noise_whitened_pca", "codebook_crossfit"):
            key = f"{method}__n131072"
            if key in archive.files:
                metrics, aligned = alignment(archive[key], truth, 3)
                matrices.append(aligned)
                labels.append(
                    f"{LABELS[method]}\nошибок I: {metrics['categorical_errors']}"
                )
    cmap, norm = categorical_map()
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for axis, matrix, label in zip(axes.flat, matrices, labels):
        artist = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        axis.set(title=label, xlabel="Вход", ylabel="Выход")
    for axis in list(axes.flat)[len(matrices) :]:
        axis.set_visible(False)
    fig.colorbar(
        artist,
        ax=list(axes.flat),
        ticks=np.arange(-3, 4),
        shrink=0.75,
        label="Код после допустимого переименования",
    )
    fig.suptitle(
        "Геометрия 7201, n=131072, фиксированный seed31; MSR показан с отдельным readout"
    )
    save_figure(fig, output, "confirmation_indexators")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for method in ("yeh", "lc", "lutq", "msr_exact"):
        path = folder / f"{method}_seed31_n131072.npz"
        with np.load(path) as archive:
            selected = (
                archive["selected_original_rows"]
                .tolist()
                .index(choices[f"{method}__native_or_decoded"]["row"])
            )
            index_trace = archive["native_or_decoded__trace_indices"][:, selected]
        history = json.loads(path.with_suffix(".json").read_text())[
            "native_or_decoded"
        ]["history"]
        steps = [row["step"] for row in history]
        axes[0].plot(
            steps, [row["source_mse"][selected] for row in history], label=method
        )
        axes[1].plot(
            steps, [row["validation_mse"][selected] for row in history], label=method
        )
        metrics = [alignment(index, truth, 3)[0] for index in index_trace]
        axes[2].plot(
            steps, [row["categorical_errors"] for row in metrics], label=method
        )
    for axis in axes:
        axis.set(xlabel="Реальные шаги алгоритма")
        axis.grid(alpha=0.2)
    axes[0].set(
        ylabel="Source forward MSE", yscale="log", title="У Yeh здесь мягкий forward"
    )
    axes[1].set(
        ylabel="Validation MSE hard I + новый C",
        yscale="log",
        title="Именно эта кривая выбирает checkpoint",
    )
    axes[2].set(ylabel="Ошибок I", title="Teacher используется только после freeze")
    axes[2].legend(fontsize=9)
    save_figure(fig, output, "confirmation_search_curves")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "report"
    output.mkdir(exist_ok=True)
    spec = json.loads((args.root / "confirmation_registration.json").read_text())
    choices = validation_restart_choices(args.root, spec, output)
    spec, cases, rows = collect(args.root)
    summaries = summary(rows, len(spec["families"]), spec["seeds"])
    selected_rows = selected_restart_rows(rows, choices)
    selected_summaries = summary(selected_rows, len(spec["families"]), [0])
    save_json(
        output / "summary.json",
        {
            "summary": summaries,
            "validation_selected_restart_summary": selected_summaries,
            "frozen_sha256": digest(args.root / "confirmation_models_frozen.json"),
            "cases": len(cases),
        },
    )
    intervals = paired_transfer_intervals(selected_rows)
    save_json(
        output / "paired_transfer_intervals.json",
        {
            "method": "10000 paired geometry bootstrap replicates, stratified6conv/6signed where available; source-validation-selected restart per geometry; descriptive uncorrected intervals, not a simultaneous multiple-comparison test.",
            "records": intervals,
            "all_individual_seeds_averaged_control": paired_transfer_intervals(rows),
        },
    )
    columns = sorted(set().union(*(row.keys() for row in summaries)))
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)
    with (output / "validation_selected_comparison.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(selected_summaries)
    plots(args.root, output, spec, cases, rows, summaries, selected_summaries)
    sections = []
    for regime, n in (
        ("primary", 8192),
        ("primary", 131072),
        ("low_noise", 131072),
        ("high_noise", 131072),
    ):
        sections.append(
            f"## {regime}, n={n}\n\n### Старт выбран по source-validation\n\n"
            + markdown_table(selected_summaries, regime, n)
            + "\n\n### Все отдельные старты: устойчивость оптимизации\n\n"
            + markdown_table(summaries, regime, n)
        )
    stop_rows = []
    for case in cases:
        if case["job"]["name"] == "primary":
            stop_rows.append(
                "|"
                + "|".join(
                    [
                        str(case["job"]["family"]["id"]),
                        *[
                            str(case["first_return"].get(method, "не выдан"))
                            for method in (
                                "signed_contrast_graph",
                                "codebook_crossfit",
                                "whitened_pca_direction_crossfit",
                            )
                        ],
                    ]
                )
                + "|"
            )
    interval_table = [
        "|База|Геометрий|MSE4 базы − MSE4 cross-fit кодбуков|95% bootstrap interval|",
        "|---|---:|---:|---:|",
    ]
    for row in intervals:
        interval_table.append(
            f"|{LABELS[row['baseline']]}|{row['families']}|{row['baseline_minus_codebook_mse4']:.6g}|[{row['bootstrap_lower95']:.6g}, {row['bootstrap_upper95']:.6g}]|"
        )
    report = (
        """# Независимое сравнение общей структуры: G4

Все модели заморожены до структурного аудита и генерации новых C. Это12 новых геометрий ×3 уровня Gaussian шума. На каждую source-функцию128 задач и указанный бюджет наблюдений; covariance condition100; std кодов1/0.3/0.03. Размеры групп и позиции скрыты. Обучаемые методы имеют3 независимых numerical seed; статистические декодеры детерминированы заданным seed.

## Как читать таблицы

Exact I учитывает глобальную перестановку названий групп и общий знак группы. «Все seed» требует3/3 для одной геометрии. Непрерывный U не выдаёт дискретный I: его subspace и перенос сохранены отдельно. MSR с дополнительным readout — отдельная строка. У Yeh MAP, LC, LUT-Q отображаются их native назначения, без последующей исправляющей кластеризации.

Основное прикладное сравнение выбирает лучший из трёх стартов по замороженному source-validation MSE; для него в колонке «Все seed» один выбранный старт. Это дополнительный post-hoc контроль честности сравнения: правило использует только сохранённые source-оценки, но оно не было отдельно зарегистрировано до открытия аудита. Таблицы отдельных seed сохраняют информацию об устойчивости. Файл validation_restart_choices.json содержит правило, выбранные seed и хеши исходных source-моделей. Целевые ошибки и true I при выборе не используются.

**Вывод с учётом выбора старта:** в основном режиме n=131072 Yeh, LC, LUT-Q и MSR с readout, как и предложенные графы и статистические контроли, точны на12/12 геометрий. LC и MSR с readout достигают12/12 уже при n=8192, когда граф ещё воздерживается. Noise-whitened PCA также точен12/12 при8192. Поэтому преимущества нашего метода по точности/переносу над лучшими контролями здесь нет. Преимущество перед неудачными отдельными стартами нельзя выдавать за превосходство над настроенным конкурентом. Граф добавляет контроль ошибочной выдачи при явно ограниченной Gaussian модели.

Модели конкурентов — явно описанные multi-task адаптации, не неизменённые author scripts. Exact-adapt MSR использует более сильное аналитическое внутреннее обучение. Отдельные32 source-validation функции выбирают HPO/checkpoint; это дополнительная информация о структуре, одинаково доступная методам. Статистические методы не используют эти labels для подбора. Эти таблицы нельзя называть информационным пределом по всем данным.

Числа C полностью сброшены. Сто новых функций имеют независимые C стандартного масштаба, в том числе сильный третий код. После support-fit показана точная clean prediction MSE при isotropic Gaussian query: сумма квадратов ошибки весов делится на число выходов, **не** на число весов. Query labels/teacher C не участвуют в выборе. Плотная регрессия имеет observation-only empirical-Bayes ridge, замороженную до target. Истинный I служит только аудитом.

Граф может воздержаться. Невыданный I не засчитывается точным. F1 и MSE графа с частичной выдачей относятся лишь к выданным I; покрытие видно по Exact/всем запусков. Alpha0.05 относится к одному заранее заданному запуску, учитывая все7 looks и семейства контрастов, а не ко всему benchmark из36 случаев и нескольких разных алгоритмов.

"""
        + "\n\n".join(sections)
        + "\n\n## Парная неопределённость переноса\n\nПоложительная разница означает большую ошибку базы. Resampling выполняется по геометриям, отдельно внутри двух классов; numerical seed выбран по source-validation. Это описательные интервалы без одновременной поправки на все сравнения. При неполной выдаче берутся только общие геометрии; их число указано. В JSON отдельно сохранён контроль с усреднением всех стартов.\n\n"
        + "\n".join(interval_table)
        + """

## Независимые графики и индексаторы

![Точность структуры](confirmation_recovery.png)

![Фиксированный I, новые C](confirmation_transfer.png)

![Бюджет данных и выдача](confirmation_data_and_coverage.png)

![Выбранные индексаторы](confirmation_indexators.png)

![Настоящие кривые поиска](confirmation_search_curves.png)

## Первый достаточный бюджет

|Геометрия|Полный F-граф|Cross-fit кодбуков|Whitening-PCA направления|
|---|---:|---:|---:|
"""
        + "\n".join(stop_rows)
        + """

## Ограничения интерпретации

WSR/OSCAR в этом опыте обучались независимо по source-функциям; общий I получался пересечением точных native равенств профилей. Это явно ограниченная адаптация: пересечение фрагментирует группы, а сильная регуляризация может их слить. Их неуспех не доказывает неуспех опубликованного метода или полноценной совместной векторной fusion-оптимизации. Для такого вывода нужен отдельный сильный joint/adaptive fusion контроль.

Это линейная identifiable задача; результат не переносится автоматически на hidden-neuron MLP, original nonlinear DeepSets, Pattern или сумму синусоид. Достаточные статистики Gaussian данных симулировались точно; сырые миллионы x не сохранялись. Алгоритмы получают sample Grams и наблюдаемые cross-products, не teacher W/градиенты учителя.

Whitening-PCA является сильным известным контролем. Cross-fit/проекционные тесты/fixed-confidence clustering известны. Специфическая польза кодбуков требует преимущества перед PCA-направлениями, а не только перед произвольной проекцией. Успех общего компонента и приоритет его комбинации нужно оценивать отдельно; этот отчёт сам по себе не доказывает новую универсально лучшую схему. Полный контекст, отрицательные результаты и литература находятся в ../../../../REPORT.md и LITERATURE.md.
"""
    )
    path = output / "REPORT.md"
    path.write_text(report)
    save_json(
        output / "report_manifest.json",
        {
            "report_sha256": digest(path),
            "models_frozen_sha256": digest(
                args.root / "confirmation_models_frozen.json"
            ),
            "audits_sha256": {
                str(folder.relative_to(args.root)): digest(folder)
                for folder in sorted(
                    (args.root / "confirmation").glob("case_*/audit.json")
                )
            },
        },
    )


if __name__ == "__main__":
    main()
