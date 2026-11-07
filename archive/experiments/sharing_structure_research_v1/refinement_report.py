"""G8/G9 reports: numerical controls, actual heatmaps and labelled ablations."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.report import save_figure

LABELS = {
    "cold": "Случайный старт",
    "moment_pairs8388608": "Моменты:16.78млн labels",
    "moment_pairs67108864": "Моменты:134.22млн labels",
    "moment_pairs536870912": "Моменты:1.074млрд labels",
}
DECODERS = {
    "scalar_dp": "Scalar DP",
    "wald_full": "Wald: полная covariance",
    "wald_diagonal": "Wald: диагональная covariance",
    "zero_wald_then_dp": "Обнуление по Wald + DP",
}


def collect(root):
    plan = json.loads((root / "protocol.json").read_text())
    cases = []
    for family in plan["families"]:
        folder = root / f"family_{family}"
        case = json.loads((folder / "audit.json").read_text())
        if case["frozen_sha256"] != digest(folder / "learner_frozen.json"):
            raise RuntimeError("Numerical experiment freeze changed")
        cases.append((folder, case))
    return plan, cases


def write_csv(path, rows):
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def g8_report(root, plan, cases):
    output = root / "report"
    output.mkdir(exist_ok=True)
    grouped = defaultdict(list)
    for _, case in cases:
        for row in case["records"]:
            grouped[row["noise"], row["method"]].append(row)
    summary = []
    table = [
        "|noise|Инициализация|Exact I /4|Среднее ошибок I|Dense validation MSE|Hard validation MSE|",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for (noise, method), rows in sorted(grouped.items()):
        item = {
            "noise": noise,
            "method": method,
            "exact": sum(r["recovery"]["categorical_errors"] == 0 for r in rows),
            "total": len(rows),
            "mean_I_errors": float(
                np.mean([r["recovery"]["categorical_errors"] for r in rows])
            ),
            "validation_mse": float(np.mean([r["validation_mse"] for r in rows])),
            "projected_validation_mse": float(
                np.mean([r["projected_validation_mse"] for r in rows])
            ),
        }
        summary.append(item)
        table.append(
            f"|{noise}|{LABELS[method]}|{item['exact']}/4|{item['mean_I_errors']:.2f}|{item['validation_mse']:.7g}|{item['projected_validation_mse']:.7g}|"
        )
    save_json(output / "summary.json", {"summary": summary})
    write_csv(output / "comparison.csv", summary)
    folder, case = next((f, c) for f, c in cases if c["family"] == 8404)
    source = json.loads((folder / "learner_records.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for axis, noise in zip(axes, plan["noise"]):
        for method, label in LABELS.items():
            row = next(
                r
                for r in case["records"]
                if r["noise"] == noise and r["method"] == method
            )["chosen_row"]
            axis.plot(
                [h["step"] for h in source["history"]],
                [h["validation_mse"][row] for h in source["history"]],
                label=label,
            )
        axis.set(
            yscale="log",
            xlabel="Реальные шаги numerical refinement",
            ylabel="Независимая noisy validation MSE",
            title=f"Геометрия8404, noise={noise}",
        )
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Одинаковый cooled Adam/variable projection; source-selection без true I"
    )
    save_figure(fig, output, "refinement_validation_curves")
    report = """# G8: одинаковое numerical refinement из cold и moment-start

Четыре уже открытые fresh-C геометрии8401–8404, noise0/0.2. Это диагностика, **не новое подтверждение и не равное по total-labels сравнение**. На каждый вариант9 стартов/скоростей,6000 шагов, одинаковые262144 независимых refinement-наблюдения,8192 readout-support и8192 validation. Bias1 — gauge положительной ReLU-однородности; amplitude>=0/offset обучаются аналитически, все256 весов свободны. LR масштабируется одинаковым правилом initial-RMS и cosine-cooldown. Моментные варианты дополнительно использовали указанный G7 источник и pilot16384labels.

Checkpoint и старт выбираются по dense noisy validation MSE. После этого используется **дополнительная** exact scalar-DP в0/±4 центров; исходная маска не фиксируется. Hard-validation здесь оценивает проецированную матрицу с новым наблюдаемым readout-fit, без последующего C-fine-tuning; target-перенос данный опыт не выполняет.

""" + "\n".join(table)
    report += """

![Настоящие кривые](refinement_validation_curves.png)

## Что именно диагностировано

Самый дорогой moment-start после refinement дал8/8 точных I. Но cold-Adam0/8 — результат этого режима, а не предел более сильной оптимизации: следующий G9 уже исправляет все четыре noiseless cold-случая полной кривизной. Поэтому headline о неизбежной неудаче плотных градиентных методов был бы неверен.

В8404noise0.2 средний бюджет даёт dense weight relative squared error0.000352, но scalar-DP131 ошибку I. Центры0.0353/1.437/4.822/8.320 показывают механизм: первый ненулевой код тратится на шум нулевых позиций, а два настоящих близких кода объединяются. Dense validation≈0.039267 после проекции становится≈0.039364. Это не объясняется только плохим source-fit и не устраняется одним требованием сделать softmax острее.

Все варианты заморожены до структурного аудита. Обоснование до запуска в THEORY.md§15, численные массивы/история выбора и хеши наблюдений сохранены. Улучшение после moment-start не доказывает новизну известных spectral-initialization/variable-projection/quantization компонентов или победу над matched-data native-конкурентами.
"""
    path = output / "REPORT.md"
    path.write_text(report)
    save_json(
        output / "report_manifest.json",
        {
            "report_sha256": digest(path),
            "protocol_sha256": digest(root / "protocol.json"),
            "audit_sha256": {
                str(f.relative_to(root)): digest(f / "audit.json") for f, _ in cases
            },
        },
    )


def g9_report(root, plan, cases, parent):
    output = root / "report"
    output.mkdir(exist_ok=True)
    grouped = defaultdict(list)
    for _, case in cases:
        for row in case["records"]:
            for decoder in plan["methods"]:
                grouped[row["noise"], row["method"], decoder].append(
                    row["recovery"].get(decoder)
                )
    summary = []
    table = [
        "|noise|Начальный numerical fit|Декодер|Exact I /4|Выдано /4|",
        "|---:|---|---|---:|---:|",
    ]
    for (noise, method, decoder), rows in sorted(grouped.items()):
        item = {
            "noise": noise,
            "initialization": method,
            "decoder": decoder,
            "exact": sum(r is not None and r["categorical_errors"] == 0 for r in rows),
            "issued": sum(r is not None for r in rows),
            "total": len(rows),
        }
        summary.append(item)
        table.append(
            f"|{noise}|{LABELS[method]}|{DECODERS[decoder]}|{item['exact']}/4|{item['issued']}/4|"
        )
    save_json(output / "summary.json", {"summary": summary})
    write_csv(output / "comparison.csv", summary)
    ablation = json.loads((root / "stationarity_guard_ablation.json").read_text())
    ablation_groups = defaultdict(list)
    for row in ablation["records"]:
        for decoder, value in row["decoders"].items():
            ablation_groups[row["initialization"], decoder].append(value)
    ablation_table = [
        "|Инициализация|Декодер, без критерия сходимости|Exact I /4|Выдано /4|",
        "|---|---|---:|---:|",
    ]
    ablation_summary = []
    for (method, decoder), rows in sorted(ablation_groups.items()):
        exact = sum(r.get("recovery", {}).get("categorical_errors") == 0 for r in rows)
        issued = sum(r["returned_index"] for r in rows)
        ablation_summary.append(
            {
                "initialization": method,
                "decoder": decoder,
                "exact": exact,
                "issued": issued,
                "total": len(rows),
            }
        )
        ablation_table.append(
            f"|{LABELS[method]}|{DECODERS[decoder]}|{exact}/4|{issued}/4|"
        )
    save_json(
        output / "posthoc_guard_ablation_summary.json",
        {"status": ablation["status"], "summary": ablation_summary},
    )

    family, key = 8404, "moment_pairs67108864__noise0.2"
    with np.load(parent / f"family_{family}" / "audit_only.npz") as archive:
        before = dict(archive)
    with np.load(root / f"family_{family}" / "audit_only.npz") as archive:
        after = dict(archive)
    with np.load(root / "stationarity_guard_ablation_indices.npz") as archive:
        diagnostic = dict(archive)
    truth = after["truth_indices"]
    panels = [
        (truth, "Истинный I"),
        (before[key + "__aligned_indices"], "До Gauss–Newton: scalar DP"),
        (after[key + "__scalar_dp__aligned"], "После Gauss–Newton: scalar DP"),
        (None, "Основной Wald-граф: I не выдан\nКритерий численной сходимости"),
        (None, "Wald-граф без критерия: I не выдан\n7 компонент вместо9; post-hoc"),
        (
            diagnostic[f"{family}__{key}__zero_wald_then_dp"],
            "Обнуление по Wald + DP\nДополнительный post-hoc контроль",
        ),
    ]
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
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis, (matrix, label) in zip(axes.flat, panels):
        if matrix is None:
            axis.text(0.5, 0.5, label, ha="center", va="center", fontsize=12)
            axis.set_axis_off()
        else:
            errors = int((matrix != truth).sum())
            artist = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            axis.set(
                title=f"{label}\nошибок I: {errors}",
                xlabel="Скрытый нейрон",
                ylabel="Вход",
            )
    fig.colorbar(
        artist, ax=list(axes.flat), ticks=np.arange(-4, 5), shrink=0.8, label="Код"
    )
    fig.suptitle("8404noise0.2: точность весов, нули и шэринг — разные критерии")
    save_figure(fig, output, "curvature_and_zero_indexators")
    report = """# G9: полная кривизна и статистические отношения весов

Это exploratory-диагностика на тех же четырёх свежих coefficient-функциях G8. Все G8 source/readout/validation наблюдения восстановлены побитово и проверены по сохранённому hash; новых labels нет. Положительный amplitude поглощён в U и общий beta. Для фиксированных ReLU-gates модель линейна по258 параметрам, поэтому внутри области Hessian точно2JᵀJ/n. Gauss–Newton с line-search уменьшает настоящую source-loss, не surrogate-норму весов. До запуска THEORY.md§16 описывает параметры, локальную идентифицируемость, ограничения NLS-Wald и контроли.

**Численный результат:** noiseless cold-start после полной кривизны даёт4/4 точных I, как и все moment-starts. Это устраняет предыдущее0/4 cold-Adam и показывает, что сильный numerical-контроль обязателен. В noisy случаях scalar-DP после Gauss–Newton всё ещё ошибается: каждый warm-вариант2/4, cold0/4. Более точное численное решение не гарантирует правильной квантованной структуры; оно может лучше подогнать шум нулей.

Основной nonlinear Wald-граф включает полный nuisance-profile covariance и заранее заданный критерий numerical-stationarity. На noisy данных этот критерий не выполнен ни в одном из16 fit-view: squared standardized decrement примерно0.017–5.20, вместо требуемого<=0.0001. Следовательно **основной метод воздержался во всех16 случаях**. Нельзя после просмотра результатов молча ослабить критерий и объявить подтверждённый успех.

Nonlinear Wald интерпретируется только локально/асимптотически, при consistency и nonsingular Fisher; прежняя exact finite-sample linear F-гарантия здесь не действует. В noiseless-данных float32 rounding учтён observation-scaled variance-floor; это numerical safeguard, а не Gaussian-coverage theorem. Alpha0.05 делится между8 views одной семьи и E² заранее заданными отношениями; не по всему benchmark из нескольких семейств/вариантов.

## Зарегистрированные результаты

""" + "\n".join(table)
    report += """

## Диагностика сходимости и роли нулей

Отдельный **post-hoc** контроль применяет те же декодеры к тем же frozen оценкам, отключив numerical-stationarity критерий. Новое обучение и подбор alpha не выполнялись. Это диагноз причины воздержания, **не результат зарегистрированной процедуры и не доказанная confidence-выдача**.

""" + "\n".join(ablation_table)
    report += """

Даже после отключения этого условия full-Wald и diagonal-Wald совпадают: из moment-start2/4 точных I, в двух других случаях5/7 компонент вместо9. Следовательно один строгий numerical-критерий не объясняет весь неуспех. В то же время простое обнуление statistically-nonsignificant весов и повторный scalar-DP даёт4/4 из каждого moment-start, cold2/4. Успех этого известного двухэтапного контроля показывает, что для данных ошибок достаточно убрать шум нулей; отдельная польза полной covariance для группирования здесь не продемонстрирована.

![Выбранные индексаторы и воздержание](curvature_and_zero_indexators.png)

В8404 noisy средний moment-start до GN имел131 ошибку, после GN обычный DP130, а zero-cleanup+DP0. Это не скрытая новая teacher-маска: нули определяются по наблюдаемой variance, но процедура является явным mask-then-quantization контролем. Предпочитаемая пользователем прямая схема пока не подтвердила здесь преимущество.

## Следующий обоснованный вопрос

В noisy ReLU-loss Newton-обновления меняют gates возле отдельных наблюдений; source-loss почти не меняется, а точный smooth-score критерий продолжает колебаться. Следующий кандидат — независимая linearization/one-step оценка с отдельным анализом её Taylor-bias и covariance, либо pooled-сравнение групп вместо требования разделить все отдельные пары. Эти варианты ещё не реализованы/проверены в G9. Критично не переносить exact linear-Gaussian guarantee на данные с ненулевым nonlinear remainder.

Никакой новой победы над Yeh/LC/LUT-Q или опубликованной новизны этот отчёт не заявляет. Нужны новое matched-data confirmation, сильные native numerical controls, перенос только I со свежими C и дополнительный литературный аудит. Полная история исследования сохранена в основном REPORT.md.
"""
    path = output / "REPORT.md"
    path.write_text(report)
    save_json(
        output / "report_manifest.json",
        {
            "report_sha256": digest(path),
            "protocol_sha256": digest(root / "protocol.json"),
            "audit_sha256": {
                str(f.relative_to(root)): digest(f / "audit.json") for f, _ in cases
            },
            "posthoc_ablation_sha256": digest(
                root / "stationarity_guard_ablation.json"
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement", type=Path, required=True)
    parser.add_argument("--curvature", type=Path, required=True)
    args = parser.parse_args()
    plan, cases = collect(args.refinement)
    g8_report(args.refinement, plan, cases)
    plan, cases = collect(args.curvature)
    g9_report(args.curvature, plan, cases, args.refinement)


if __name__ == "__main__":
    main()
