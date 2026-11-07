"""Post-freeze diagnostics of nonlinear moment reconstruction and dense Adam."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices
from experiments.sharing_benchmark_v1.report import save_figure

from .nonlinear_moments import reconstruct, weight_error


def collect(root):
    plan = json.loads((root / "protocol.json").read_text())
    cases = []
    for family in plan["families"]:
        for function in plan["functions"]:
            folder = root / f"family_{family}" / function
            audit = json.loads((folder / "audit.json").read_text())
            if audit["frozen_sha256"] != digest(folder / "learner_frozen.json"):
                raise RuntimeError("Nonlinear model freeze changed")
            cases.append((folder, audit))
    return plan, cases


def hybrid_oracle_diagnosis(cases, output):
    """Separate moment errors with explicit teacher-only oracle substitutions."""
    rows = []
    for folder, case in cases:
        with np.load(folder / "learner_outputs.npz") as archive:
            observed = dict(archive)
        with np.load(folder / "audit_only.npz") as archive:
            truth = dict(archive)
        generator = torch.Generator().manual_seed(2611059917)
        sketches = torch.randn((16, 8), generator=generator, dtype=torch.float64)
        for noise in (0.0, 0.2):
            key = f"affine_residual__noise{noise}__n8388608"
            for name, second_key, third_key in (
                ("oracle_M2_observed_M3", "population_second", f"{key}__third"),
                ("observed_M2_oracle_M3", f"{key}__second", "population_third"),
            ):
                second = truth.get(second_key, observed.get(second_key))
                third = truth.get(third_key, observed.get(third_key))
                weights, info = reconstruct(
                    torch.tensor(second), torch.tensor(third), sketches
                )
                row = {
                    "family": case["family"],
                    "function": case["function"],
                    "noise": noise,
                    "oracle_substitution": name,
                    **info,
                }
                if weights is not None:
                    row["weight_relative_squared_error"] = weight_error(
                        weights.numpy(), truth["truth_weights"] / 0.15
                    )
                rows.append(row)
    save_json(
        output / "hybrid_oracle_diagnosis.json",
        {
            "status": "Post-hoc oracle-only diagnosis, never a learning input or comparator result",
            "records": rows,
        },
    )
    return rows


def heatmap(folder, case, output):
    noise, n = 0.0, 1048576
    key = f"dense_adam__noise{noise}__n{n}"
    selected = next(r for r in case["records"] if r["key"] == key)["selected"]
    with np.load(folder / "learner_outputs.npz") as archive:
        dense = archive[f"{key}__weights"][selected]
        numeric = archive[f"{key}__numeric"][selected]
        index = archive[f"{key}__indices"]
        quantized = archive[f"{key}__quantized"]
    with np.load(folder / "audit_only.npz") as archive:
        truth, true_weights = archive["truth_indices"], archive["truth_weights"]
        moment_index = archive["affine_residual__noise0.0__n8388608__aligned_indices"]
    metrics, aligned, columns = align_indices(index, truth, 4)
    weights = [
        true_weights,
        dense[:, columns] * numeric[1],
        quantized[:, columns] * numeric[1],
    ]
    titles = [
        "Истинные W",
        "Настоящие веса плотного Adam",
        "После точной scalar-DP проекции",
    ]
    limit = max(np.abs(w).max() for w in weights)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, value, title in zip(axes[0], weights, titles):
        artist = axis.imshow(
            value, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto"
        )
        axis.set(title=title, xlabel="Скрытый нейрон", ylabel="Вход")
    fig.colorbar(artist, ax=list(axes[0]), shrink=0.8, label="Вес × общий amplitude")
    cmap = ListedColormap(
        [
            "#7f004f",
            "#a50045",
            "#f46d43",
            "#fee08b",
            "#f6f6f6",
            "#abdda4",
            "#66c2a5",
            "#3288bd",
            "#4a2284",
        ]
    )
    norm = BoundaryNorm(np.arange(-4.5, 5.0), 9)
    for axis, value, title in zip(
        axes[1],
        (truth, aligned, moment_index),
        (
            "Истинный I",
            f"Adam + scalar DP: {metrics['categorical_errors']} ошибок I",
            f"Наблюдаемые моменты: {int((moment_index != truth).sum())} ошибок I",
        ),
    ):
        artist = axis.imshow(value, cmap=cmap, norm=norm, aspect="auto")
        axis.set(title=title, xlabel="Скрытый нейрон", ylabel="Вход")
    fig.colorbar(
        artist, ax=list(axes[1]), ticks=np.arange(-4, 5), shrink=0.8, label="Код"
    )
    fig.suptitle(
        f"Нелинейный DeepSets, {case['family']}, {case['function']}, noise0; Adam n=1млн, моменты n=8млн"
    )
    save_figure(fig, output, f"nonlinear_indexators_{case['function']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan, cases = collect(args.root)
    output = args.root / "report"
    output.mkdir(exist_ok=True)
    rows = []
    for _, case in cases:
        for record in case["records"]:
            rows.append(
                {"family": case["family"], "function": case["function"], **record}
            )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"], row["noise"], row["n"], row["function"]].append(row)
    summaries = []
    for (method, noise, n, function), values in sorted(grouped.items()):
        issued = [v for v in values if "recovery" in v]
        item = {
            "method": method,
            "noise": noise,
            "n": n,
            "function": function,
            "total": len(values),
            "issued": len(issued),
            "exact": sum(v["recovery"]["categorical_errors"] == 0 for v in issued),
            "mean_errors_when_issued": float(
                np.mean([v["recovery"]["categorical_errors"] for v in issued])
            )
            if issued
            else None,
        }
        for metric in (
            "relative_second_error",
            "relative_third_error",
            "clean_validation_nmse_audit",
        ):
            if metric in values[0]:
                item[metric] = float(np.mean([v[metric] for v in values]))
        summaries.append(item)
    save_json(output / "summary.json", {"summary": summaries})
    fields = sorted(set().union(*(row.keys() for row in summaries)))
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    hybrids = hybrid_oracle_diagnosis(cases, output)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for column, function in enumerate(plan["functions"]):
        for row_id, metric in enumerate(
            ("relative_second_error", "relative_third_error")
        ):
            for method in plan["moment_variants"]:
                chosen = sorted(
                    [
                        s
                        for s in summaries
                        if s["function"] == function
                        and s["noise"] == 0.2
                        and s["method"] == method
                    ],
                    key=lambda s: s["n"],
                )
                axes[row_id, column].plot(
                    [s["n"] for s in chosen],
                    [s[metric] for s in chosen],
                    marker="o",
                    label=method,
                )
            axes[row_id, column].set(
                xscale="log",
                yscale="log",
                xlabel="Примеров",
                ylabel=f"Относительная ошибка M{row_id + 2}",
                title=function,
            )
            axes[row_id, column].grid(alpha=0.2)
    axes[0, 1].legend()
    fig.suptitle("Оценка наблюдаемых моментов; Gaussian noise0.2; среднее4 геометрий")
    save_figure(fig, output, "nonlinear_moment_errors")
    for folder, case in cases:
        if case["family"] == 8401:
            heatmap(folder, case, output)

    table = [
        "|Функция|noise|Метод|n|Выдано /4|Exact I /4|Среднее ошибок выданного I|",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        if row["n"] == (
            plan["dense"]["n"]
            if row["method"] == "dense_adam"
            else max(plan["observations"])
        ):
            errors = (
                "—"
                if row["mean_errors_when_issued"] is None
                else f"{row['mean_errors_when_issued']:.2f}"
            )
            table.append(
                f"|{row['function']}|{row['noise']}|{row['method']}|{row['n']}|{row['issued']}/4|{row['exact']}/4|{errors}|"
            )
    oracle_error = max(
        case["population_oracle_only"]["weight_relative_squared_error"]
        for _, case in cases
    )
    report = f"""# G6: нелинейная диагностика наблюдаемости

Исходная архитектура synthetic DeepSets сохранена:16 входов,16 скрытых ReLU, общий bias и amplitude, сумма выходов. Четыре перестановки входов; два набора коэффициентов — прежние0.8/−1.1/1.3/0.45 и новый Gaussian C каждой геометрии; два уровня шума. Это8 функций ×2 шума, exploratory-диагностика, не финальный competitive benchmark. Алгоритмы не получают истинные координатные роли, маску, W, C или autograd учителя.

**Диагноз:** population M2/M3 восстанавливают8/8 индексаторов, максимальная относительная квадратичная ошибка W/b={oracle_error:.3g}. Поэтому при данных допущениях извлечение в принципе возможно. Но оценки из8млн обычных Gaussian примеров не дают ни одного точного I. После affine-control-variate ошибка M3 на исходных коэффициентах всё ещё около0.24; на свежих около0.51 даже без label-noise. Третий момент пропорционален малому bias0.15. Измеренный шум статистики и conditioning усиливаются whitening/разложением.

Gaussian Hermite/whitening/tensor reconstruction — **известный контроль**, связанный с NN-LIFT. Общий bias позволяет восстановить W/b из масштаба третьего момента. Это явно описанная специализация, не неизменённая реализация NN-LIFT и не заявленная новизна. Сквозная матричная диагонализация выбирает один из8 random sketches по наблюдаемому off-diagonal tensor residual. Если M2 не positive definite, контроль не выдаёт W.

Плотный Adam обучен реально,9 стартов/скоростей на каждый шум,3000 шагов и1млн фиксированных source-примеров; отдельная noisy validation выбирает checkpoint и старт. После этого применяется **дополнительная** точная scalar-DP проекция в0/±4 центров; она не является native индексатором Adam. На исходных C точны7/8 случаев, на новых C1/8. Это результаты конкретного numerical-бюджета; они не доказывают невозможности более долгой или другой настройки Adam. Бюджеты Adam и максимального moment-контроля различаются и указаны в таблице.

Новый Gaussian C в двух случаях имеет минимальный разрыв модулей около0.025; ещё один случай даёт condition(W)≈19.7. Низкая ошибка предсказаний поэтому ещё слабее гарантирует правильное выделение групп. Учитель открывается только после freeze learner_outputs.npz/learner_records.json каждого случая; oracle-моменты и hybrid-подстановки записаны отдельно.

## Точные результаты

""" + "\n".join(table)
    report += """

## Диагностические графики

![Ошибка наблюдаемых моментов](nonlinear_moment_errors.png)

![Исходные коэффициенты: реальные веса Adam и I](nonlinear_indexators_original_codes.png)

![Свежие коэффициенты: реальные веса Adam и I](nonlinear_indexators_fresh_gaussian_codes.png)

Хитмапы используют заранее фиксированную первую геометрию8401, noise0. На ней нельзя оценивать средний результат. Скрытые столбцы/названия кодов выравниваются только при аудите. Весовые карты показывают настоящие численные веса × amplitude и их проекцию, не истинные W под названием Adam.

hybrid_oracle_diagnosis.json отдельно заменяет только M2 или только M3 population-значением. Это post-hoc диагноз, а не конкурент и не допустимый вход обучения. Данные позволили отделить сбой оценки статистики от принципиальной неидентифицируемости; алгоритмическое исправление требует отдельной теории и нового опыта. Успех линейного F-графа пока не заявляется для этих нелинейных параметров.
"""
    path = output / "REPORT.md"
    path.write_text(report)
    save_json(
        output / "report_manifest.json",
        {
            "report_sha256": digest(path),
            "protocol_sha256": digest(args.root / "protocol.json"),
            "audits_sha256": {
                str(folder.relative_to(args.root)): digest(folder / "audit.json")
                for folder, _ in cases
            },
            "hybrid_oracle_substitutions": len(hybrids),
        },
    )


if __name__ == "__main__":
    main()
