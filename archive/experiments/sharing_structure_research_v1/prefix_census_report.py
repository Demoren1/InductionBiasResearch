"""Report the frozen G19 candidate and its separate factorial diagnostics."""

import argparse
import csv
import json
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

BRANCHES = {
    "primary": "Estimated C + noisy slices (primary)",
    "oracle_codes_observed": "True C + noisy slices (oracle diagnosis)",
    "estimated_codes_clean": "Estimated C + clean slices (oracle diagnosis)",
    "oracle_codes_clean": "True C + clean slices (oracle diagnosis)",
}


def exact(result):
    return result["complete"] and result["categorical_errors"] == 0


def generate(root):
    plan = json.loads((root / "protocol.json").read_text())
    frozen = json.loads((root / "models_frozen.json").read_text())
    for relative, expected in frozen["sha256"].items():
        if digest(root / relative) != expected:
            raise RuntimeError("Frozen G19 primary changed")
    cases = []
    for job_id, job in enumerate(plan["jobs"]):
        path = root / "models" / f"job_{job_id:03d}"
        meta = json.loads(path.with_suffix(".json").read_text())
        audit = json.loads(path.parent.joinpath(path.name + "_audit.json").read_text())
        if audit["models_frozen_sha256"] != digest(root / "models_frozen.json"):
            raise RuntimeError("G19 audit belongs to another freeze")
        with np.load(path.with_suffix(".npz")) as data:
            arrays = dict(data)
        with np.load(path.parent / (path.name + "_audit_only.npz")) as data:
            oracle = dict(data)
        cases.append(
            {
                "job": job,
                "meta": meta,
                "audit": audit,
                "arrays": arrays,
                "oracle": oracle,
            }
        )
    report = root / "report"
    report.mkdir(exist_ok=False)
    rows = []
    points = (
        2 * plan["inputs"] * plan["axis_points"] + plan["inputs"] * plan["slice_points"]
    )
    for variant in ("local", "quartet_mix", "random"):
        selected = [case for case in cases if case["job"]["variant"] == variant]
        for condition_id, condition in enumerate(plan["conditions"]):
            for branch in BRANCHES:
                results = [case["audit"]["records"][condition_id] for case in selected]
                rows.append(
                    {
                        "variant": variant,
                        **condition,
                        "branch": branch,
                        "cases": len(results),
                        "complete": sum(row[branch]["complete"] for row in results),
                        "exact": sum(exact(row[branch]) for row in results),
                        "median_max_C_error": float(
                            np.median([row["max_abs_C_error"] for row in results])
                        ),
                        "full_available_source_labels": points
                        * condition["replicates"],
                        "validation_labels": plan["validation"],
                    }
                )
    with (report / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aliasing = []
    for case in cases:
        c = np.sort(np.abs(case["oracle"]["true_C"]))
        knots = plan["bias"] / c[::-1]
        t = case["arrays"]["axis_t"]
        internal_counts = [
            int(((t > left) & (t < right)).sum())
            for left, right in pairwise(knots)
        ]
        aliasing.append(
            {
                **case["job"],
                "true_magnitudes": c.tolist(),
                "knot_bins": np.searchsorted(t, knots).tolist(),
                "points_in_internal_regions": internal_counts,
                "noiseless_C_error": case["audit"]["records"][0]["max_abs_C_error"],
                "noiseless_axis_mse": case["meta"]["records"][0]["axis"][
                    "axis_even_mse"
                ],
                "noiseless_exact_I": exact(case["audit"]["records"][0]["primary"]),
            }
        )
    save_json(
        report / "summary.json",
        {
            "comparisons": rows,
            "aliasing_diagnosis": aliasing,
            "models_frozen_sha256": digest(root / "models_frozen.json"),
            "full_available_source_points": points,
            "scope": "Primary estimated-C is noise-free15/18 and noisy0/108. Exact/clean oracle is18/18 unique jobs; repeated saved oracle branches are not new geometries. Known bias/readout, one source, no competitor win or transfer.",
        },
    )
    plot_recovery(report, rows)
    plot_prefixes(report, cases, plan)
    plot_indices(report, cases, plan)
    plot_aliasing(report, cases, plan)
    write_report(report, rows, aliasing, points, plan)


def plot_recovery(report, rows):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    colors = ("tab:blue", "tab:orange", "tab:green")
    for line, sigma in enumerate((0.2, 2.0)):
        for col, variant in enumerate(("local", "quartet_mix", "random")):
            for branch, color in zip(list(BRANCHES)[:3], colors, strict=True):
                selected = [
                    row
                    for row in rows
                    if row["variant"] == variant
                    and row["noise"] == sigma
                    and row["branch"] == branch
                ]
                axes[line, col].plot(
                    [r["full_available_source_labels"] for r in selected],
                    [r["exact"] / r["cases"] for r in selected],
                    marker="o",
                    color=color,
                    label=BRANCHES[branch],
                )
            axes[line, col].set_xscale("log")
            axes[line, col].set_ylim(-0.06, 1.06)
            axes[line, col].set_title(f"{variant}, noise SD {sigma}")
            axes[line, col].set_xlabel(
                "Full available source labels (axis + all 16 slices)"
            )
            axes[line, col].set_ylabel("Exact full weight indexator / 6")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "G19: primary failure and factorial oracle diagnosis\nKnown bias/readout; one source function per I; no competitor comparison"
    )
    fig.savefig(report / "census_recovery.png", dpi=180)
    plt.close(fig)


def first_case(cases, variant):
    return next(case for case in cases if case["job"]["variant"] == variant)


def plot_prefixes(report, cases, plan):
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
    condition_id = next(
        i
        for i, c in enumerate(plan["conditions"])
        if c == {"noise": 0.2, "replicates": 4096}
    )
    for line, variant in enumerate(("local", "quartet_mix", "random")):
        case = first_case(cases, variant)
        record = case["meta"]["records"][condition_id]
        audit = case["audit"]["records"][condition_id]
        for branch, color in (
            ("primary", "tab:blue"),
            ("oracle_codes_observed", "tab:orange"),
        ):
            trace = record["trace"] if branch == "primary" else audit[branch]["trace"]
            usable = [row for row in trace if "slice_mse_continuous" in row]
            x = [row["prefix_length"] for row in usable]
            axes[line, 0].plot(
                x,
                [max(row["slice_mse_continuous"], 1e-30) for row in usable],
                marker="o",
                color=color,
                label=BRANCHES[branch],
            )
            usable_integer = [row for row in trace if "slice_mse_integer" in row]
            axes[line, 1].plot(
                [row["prefix_length"] for row in usable_integer],
                [max(row["slice_mse_integer"], 1e-30) for row in usable_integer],
                marker="o",
                color=color,
            )
            key = record["key"] + (
                "__indices" if branch == "primary" else "__" + branch + "__indices"
            )
            source = case["arrays"] if branch == "primary" else case["oracle"]
            recall = []
            for step, index in enumerate(source[key], 1):
                truth = case["oracle"]["truth_I"].copy()
                truth[case["arrays"]["order"][step:]] = 0
                metric, _, _ = align_indices(index, truth, plan["codes"])
                recall.append(metric["mask_recall"])
            axes[line, 2].plot(
                range(1, len(recall) + 1), recall, marker="o", color=color
            )
        for col in (0, 1):
            axes[line, col].set_yscale("log")
            axes[line, col].axhline(
                0.2**2 / 4096, ls="--", color="gray", label="Mean-label noise variance"
            )
        for ax in axes[line]:
            ax.set_xlabel("Processed input coordinates (not optimizer steps)")
            ax.set_xticks([1, 4, 8, 12, 16])
        axes[line, 0].set_ylabel(variant + "\nContinuous slice LS residual MSE")
        axes[line, 1].set_ylabel("Integer-prefix slice-fit MSE")
        axes[line, 2].set_ylabel("Mask recall on processed rows (audit only)")
        axes[line, 2].set_ylim(-0.04, 1.04)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "G19 measured slice-fit residual and recall, noise0.2 / 4096 repeated labels\nFirst registered family of each type; stopped primary prefixes are not full I"
    )
    fig.savefig(report / "census_slice_loss_recall.png", dpi=180)
    plt.close(fig)


def plot_indices(report, cases, plan):
    selected = [case for case in cases if case["job"]["variant"] == "random"]
    condition_id = next(
        i
        for i, c in enumerate(plan["conditions"])
        if c == {"noise": 0.2, "replicates": 4096}
    )
    fig, axes = plt.subplots(6, 4, figsize=(13, 16), constrained_layout=True)
    for line, case in enumerate(selected):
        truth = case["oracle"]["truth_I"]
        keys = [
            (case["arrays"], "condition_0__indices", "Primary: no noise"),
            (
                case["arrays"],
                f"condition_{condition_id}__indices",
                "Primary: SD0.2 / r4096",
            ),
            (
                case["oracle"],
                f"condition_{condition_id}__oracle_codes_observed__indices",
                "Oracle C: SD0.2 / r4096",
            ),
        ]
        images, titles = [truth], ["True signed I"]
        for data, key, title in keys:
            trace = data[key]
            if len(trace):
                _, aligned, _ = align_indices(trace[-1], truth, plan["codes"])
            else:
                aligned = np.zeros_like(truth)
            images.append(aligned)
            titles.append(
                title
                + f"\n{len(trace)}/16 coordinates"
                + ("; no full I" if len(trace) < 16 else "")
            )
        for col, (image, title) in enumerate(zip(images, titles, strict=True)):
            im = axes[line, col].imshow(image, cmap="RdBu_r", vmin=-4, vmax=4)
            axes[line, col].set_title(title, fontsize=9)
            axes[line, col].set_xticks([0, 7, 15])
            axes[line, col].set_yticks([0, 7, 15])
            if col == 0:
                axes[line, col].set_ylabel(
                    f"Random {case['job']['family']}\nInput coordinate"
                )
            if line == len(selected) - 1:
                axes[line, col].set_xlabel("Hidden neuron (audit aligned)")
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.6,
        label="Signed code ID; 0 = absent or not yet processed",
    )
    fig.suptitle(
        "G19: all 6 new unrestricted random indexators\nPartial prefixes labelled explicitly; oracle C is diagnosis only"
    )
    fig.savefig(report / "census_indexators.png", dpi=180)
    plt.close(fig)


def plot_aliasing(report, cases, plan):
    case = next(
        case for case in cases if case["job"] == {"family": 8903, "variant": "random"}
    )
    source = case["meta"]["records"][0]
    t = case["arrays"]["axis_t"]
    observed = (
        case["arrays"]["condition_0__axis_y"].sum((0, 1))
        - 2 * plan["inputs"] * np.sqrt(plan["hidden"]) * plan["bias"]
    )
    estimate = np.asarray(source["codebook"])
    truth = np.sort(np.abs(case["oracle"]["true_C"]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    grid = np.linspace(0.085, 0.106, 1000)
    curve = (
        np.maximum(grid[:, None] * estimate - plan["bias"], 0)
        @ source["axis"]["amplitude"]
    )
    inside = (t >= grid[0]) & (t <= grid[-1])
    axes[0].plot(grid, curve, label="Fitted even-axis model")
    axes[0].scatter(
        t[inside],
        observed[inside],
        color="black",
        zorder=3,
        label="Observed noise-free points",
    )
    for knot in plan["bias"] / truth[-2:]:
        axes[0].axvline(knot, color="tab:green", ls="--", alpha=0.8)
    axes[0].set_xlabel("Axis query magnitude t (post-hoc zoom)")
    axes[0].set_ylabel("Summed even-axis response minus known baseline")
    axes[0].legend(fontsize=8)
    x = np.arange(plan["codes"])
    axes[1].bar(x - 0.18, truth, width=0.36, label="True magnitudes (audit)")
    axes[1].bar(x + 0.18, estimate, width=0.36, label="Estimated magnitudes")
    axes[1].set_xlabel("Sorted code (arbitrary group names)")
    axes[1].set_ylabel("Code magnitude")
    axes[1].legend(fontsize=8)
    errors = []
    labels = []
    for variant in ("local", "quartet_mix", "random"):
        errors.append(
            [
                row["audit"]["records"][4]["max_abs_C_error"]
                for row in cases
                if row["job"]["variant"] == variant
            ]
        )
        labels.append(variant)
    axes[2].boxplot(errors, tick_labels=labels, showfliers=True)
    axes[2].set_ylabel("Max absolute code error: SD2 / r1")
    axes[2].set_yscale("log")
    fig.suptitle(
        "G19 codebook diagnosis: near-zero axis-fit error does not ensure correct C\nRandom8903 noise-free, fixed coarse grid; oracle illustration after freeze"
    )
    fig.savefig(report / "census_codebook_diagnosis.png", dpi=180)
    plt.close(fig)


def write_report(report, rows, aliasing, points, plan):
    primary = [row for row in rows if row["branch"] == "primary"]
    table = "\n".join(
        "|{}|{}|{}|{}/6|{:.2e}|{:,}|".format(
            row["variant"],
            row["noise"],
            row["replicates"],
            row["exact"],
            row["median_max_C_error"],
            row["full_available_source_labels"],
        )
        for row in primary
    )
    diagnosis = []
    for variant in ("local", "quartet_mix", "random"):
        condition = [
            row
            for row in rows
            if row["variant"] == variant
            and row["noise"] == 0.2
            and row["replicates"] == 4096
        ]
        value = {row["branch"]: row["exact"] for row in condition}
        diagnosis.append(
            f"|{variant}|{value['primary']}/6|{value['oracle_codes_observed']}/6|{value['estimated_codes_clean']}/6|{value['oracle_codes_clean']}/6|"
        )
    diagnostic_table = "\n".join(diagnosis)
    failures = [row for row in aliasing if not row["noiseless_exact_I"]]
    failure_table = "\n".join(
        "|{}|{}|{:.2e}|{:.2e}|".format(
            row["family"],
            ", ".join(map(str, row["points_in_internal_regions"])),
            row["noiseless_C_error"],
            row["noiseless_axis_mse"],
        )
        for row in failures
    )
    text = f"""# G19: прямой подсчёт шаблонов индексатора

5 ноября 2026. **Без шума primary exact I15/18; с шумом0/108. Кандидат не готов.** Все18 source-fit и18 factorial-audit завершены на шести свободных V100. Ни конкурентной победы, ни полного multi-source/transfer результата нет.

## Механизм и условия

После G18 (одинаковые низкие моменты у разных I) THEORY.md §26 до запуска задаёт иной observable: значения функции на координатных срезах. Из even-axis queries оцениваются неизвестные модули четырёх кодов. Затем для каждого learned prefix нейронного вектора рассматриваются 0/±4 продолжения. Глобальная линейная LS оценивает **количества** нейронов каждого продолжения; integer rounding создаёт следующий multiset. После16 координат это полный signed I, без реконструкции плотной W и последующей её кластеризации. При нарушении неотрицательности/parent count/rank следует abstention.

Известны D=H16,K4, common bias0.15, положительный readout1/sqrt(H), offset0. Это дополнительное ограничение pilot. Маска, число активных весов, роли координат, размеры истинных модулей, численные C и teacher autograd learner недоступны. Переданы лишь запросы/шумные средние. Порядок координат независим от teacher. Шесть открытых G18 перестановок local/mixed имеют новые Gaussian C; ещё шесть новых arbitrary random signed I не имеют шаблона DeepSets. Одна функция на I, не общий I нескольких source-функций. Все18 fit/126 noise-budget состояний frozen до открытия teacher; все бюджеты сохранены, выбора по I нет.

При точном C и истинном предыдущем multiset full-column-rank design даёт точные количества без шума. Разные common-positive-bias ridge-atoms имеют разные kink-гиперплоскости; rich queries различают их. Это условная noiseless/stability мотивация, не гарантия noisy estimated-C восстановления. Generic reconstruction известен у [Rolnick–Kording2020](https://proceedings.mlr.press/v119/rolnick20a/rolnick20a.pdf), [Milli et al.2018](https://arxiv.org/pdf/1807.05185), [Daniely–Granot2023](https://arxiv.org/pdf/2105.09673). Segmented LS, variable projection, finite dictionary LS и integer rounding сами известны. Приоритет prefix-count комбинации пока не установлен, новый метод не объявлен доказанно лучшим.

## Primary: кодбук только из наблюдений

|Структура|Noise SD|Labels/query r|Точный полный I|Median max code error|Все доступные source labels|
|---|---:|---:|---:|---:|---:|
{table}

Без шума local6/6, mixed6/6, random3/6. Все15 выданных полных I точны. В noisy состояниях полных выдач нет: раннее округление нарушает допустимость или словарь теряет ранг. Это0/108 зависимых состояний, не108 независимых геометрий. Abstention не считать точным результатом. Хороший prefix или маска не подменяют полный I.

![Полный I и отдельные oracle-контроли](census_recovery.png)

## Факторный диагноз после freeze

Меняем только C в словаре и/или шум slice labels. Primary/estimated-C, noisy slices неизменны. Oracle-ветви не используются для выбора/выдачи метода. При noise0.2/r4096:

|Структура|Estimated C + noisy slices|True C + noisy slices|Estimated C + clean slices|True C + clean slices|
|---|---:|---:|---:|---:|
{diagnostic_table}

Known-C clean восстановил18/18 уникальных структур; этот одинаковый контроль сохранён для разных noise-состояний, **не считать его126 независимыми успехами**. True-C noisy при SD0.2/r4096 даёт16/18, при SD2/r4096 —0/18. Estimated-C clean остаётся0/18 в том же SD0.2/r4096 состоянии. Значит, C-stage уже достаточен для срыва, а при большом шуме отдельно ненадёжен и LS-rounding. Это не доказательство единственной причины во всех режимах.

![Настоящий slice-fit residual и recall по числу обработанных координат](census_slice_loss_recall.png)

## Почему нельзя исправить всё только увеличением повторов

Even-axis grid имеет257 точек t=0 и geomspace(.001,1024). У близких кодов короткая линейная область может содержать слишком мало точек, а непрерывные неизвестные amplitudes позволяют эквивалентные подгонки. Кроме этого numerical segmented-LS/variable-projection может оставить ненулевой остаток. В трёх noiseless random failure:

|Family|Число sampled точек между истинными соседними изломами|Max code error|Even-axis fit MSE|
|---|---|---:|---:|
{failure_table}

У8903 кодовая ошибка0.0565 при MSE около3.01e-24: почти нулевой fit на фиксированных точках не определяет кодбук. Между близкими true-kinks только одна точка; в continuous-amplitude модели одни и те же samples совместимы с разными близкими split. Истинные integer multiplicities в этом этапе ещё не использованы. В8901/8905 остаток ненулевой: solver/initial grid также требуют исправления. Не объявлять все три случая строгой информационной невозможностью или одним optimizer failure. Всё это post-hoc diagnosis после заморозки.

![Недостаточность coarse axis-fit как прокси кодбука](census_codebook_diagnosis.png)

## Индексаторы и отсутствие численного baseline в этой серии

![Все новые random I и явно помеченные partial prefixes](census_indexators.png)

Нулём на неполном prefix показаны также ещё не обработанные координаты, поэтому он явно подписан **no full I**. Колонки и signed code names выровнены к teacher только для картинки/метрики. Здесь нет обученного dense SGD/Adam/GN baseline: его результаты из G17 нельзя переносить на эти панели или придумывать numerical W. Signed I плюс сохранённый codebook определяют реальные численные prefix-W, но при abstention это не выбранная полная модель.

## Стоимость и воспроизводимость

Root `results/20261105/prefix_census_g19`, код prefix_census.py/prefix_census_report.py; protocol, theory_at_registration, executed_code_snapshot, models_frozen.json. models/job_*.json/.npz содержат наблюдаемые axis/slice средние и prefixes; отдельные _audit.json/_audit_only.npz содержат teacher и factorial-контроли. Freeze-хеши проверены до отчёта. Сохранены source timings/axis-solver evaluations/rank/conditioning/reasons. Software tests не создавались/не запускались; минимальный Ruff, math/provenance аудит и просмотр четырёх фигур.

Полный panel содержит {points:,} query-точек (2×16×257 axis +16×2048 slices). Gaussian mean r независимых labels симулирован точно как f+sigma/sqrt(r)Z; стоимость **r labels**, не один. На r4096 полный доступный panel стоит {points * 4096:,} labels плюс8192 validation. Режимы paired по unit noise. Raw поле `primary_label_budget` в job.json считает только axis и фактически обработанные префиксы до abstention; оно **не включает** оставшиеся срезы, которые заранее сгенерированы и сохранены в NPZ. В comparison.csv/кривой recovery используем полную стоимость всех доступных средних, чтобы не скрывать эти данные. Prefix-trace budgets остаются счётчиком реально использованных последовательных шагов. Не смешивать эти две величины.

## Следующее изменение должно следовать из этого диагноза

Сначала исправить codebook stage: снять coarse-grid ambiguity, отделить её от недостатка SNR и локального numerical fit. Проверить более плотные/активные axis-query и ограничения целых multiplicities относительно прежнего непрерывного fit; теория/протокол до нового запуска. Затем отдельно проверить noise amplification LS-rounding и exact/feasible integer-count решения. Это предложения, не уже выполненные улучшения.

Для общей цели всё ещё нужны свободные bias/readout, собственный компонент с самостоятельным эффектом, общий I разных source-C, сильные native LC/Yeh/LUT-Q на тех же query/label budget, misspecified noise, новые независимые структуры и перенос только I на свежие C. Данные не ограничены намеренно, но большой счётчик виртуальных labels сам по себе не является конкурентной победой. Цель остаётся active.
"""
    (report / "REPORT.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    generate(args.root)


if __name__ == "__main__":
    main()
