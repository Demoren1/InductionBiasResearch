"""Report G7 factorial controls, label budgets and failed structural cases."""

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
    "iid_affine__sketch": "IID + matrix sketch",
    "iid_affine__joint": "IID + joint diagonalization",
    "antithetic_affine__sketch": "Пары x/−x + matrix sketch",
    "antithetic_affine__joint": "Пары x/−x + joint diagonalization",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root / "protocol.json").read_text())
    cases, grouped = [], defaultdict(list)
    for family in plan["families"]:
        for function in plan["functions"]:
            folder = root / f"family_{family}" / function
            case = json.loads((folder / "audit.json").read_text())
            if case["frozen_sha256"] != digest(folder / "learner_frozen.json"):
                raise RuntimeError("G7 freeze changed")
            cases.append((folder, case))
            for row in case["records"]:
                grouped[function, row["noise"], row["pairs"], row["method"]].append(row)
    summaries = []
    for (function, noise, pairs, method), values in sorted(grouped.items()):
        issued = [v for v in values if "recovery" in v]
        summaries.append(
            {
                "function": function,
                "noise": noise,
                "pairs": pairs,
                "query_labels": 2 * pairs,
                "pilot_labels": 2 * plan["affine_pilot_pairs"],
                "method": method,
                "exact": sum(v["recovery"]["categorical_errors"] == 0 for v in issued),
                "issued": len(issued),
                "total": len(values),
                "mean_errors_when_issued": float(
                    np.mean([v["recovery"]["categorical_errors"] for v in issued])
                )
                if issued
                else None,
                "relative_second_error": float(
                    np.mean([v["relative_second_error"] for v in values])
                ),
                "relative_third_error": float(
                    np.mean([v["relative_third_error"] for v in values])
                ),
            }
        )
    output = root / "report"
    output.mkdir(exist_ok=True)
    save_json(output / "summary.json", {"summary": summaries, "cases": len(cases)})
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for row_id, noise in enumerate(plan["noise"]):
        for column, function in enumerate(plan["functions"]):
            axis = axes[row_id, column]
            for method, label in LABELS.items():
                chosen = sorted(
                    [
                        s
                        for s in summaries
                        if s["function"] == function
                        and s["noise"] == noise
                        and s["method"] == method
                    ],
                    key=lambda s: s["query_labels"],
                )
                axis.plot(
                    [s["query_labels"] for s in chosen],
                    [s["exact"] / s["total"] for s in chosen],
                    marker="o",
                    label=label,
                )
            axis.set(
                xscale="log",
                ylim=(-0.04, 1.04),
                xlabel="Labels на source-функцию, pilot +16384 отдельно",
                ylabel="Точный I /4 геометрии",
                title=f"{function}, noise={noise}",
            )
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "G7: равное число labels; фиксированные бюджеты; невыданный I считается неуспехом"
    )
    save_figure(fig, output, "parity_recovery")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, function in zip(axes, plan["functions"]):
        for view, label in (("iid_affine", "IID"), ("antithetic_affine", "Пары x/−x")):
            selected = sorted(
                [
                    s
                    for s in summaries
                    if s["function"] == function
                    and s["noise"] == 0.0
                    and s["method"] == view + "__joint"
                ],
                key=lambda s: s["query_labels"],
            )
            axis.plot(
                [s["query_labels"] for s in selected],
                [s["relative_third_error"] for s in selected],
                marker="o",
                label=label,
            )
        axis.set(
            xscale="log",
            yscale="log",
            xlabel="Labels на функцию",
            ylabel="Относительная ошибка M3",
            title=function,
        )
        axis.grid(alpha=0.2)
    axes[1].legend()
    fig.suptitle(
        "Устранение чётной variance; показан noise0, без изменения M3 в population"
    )
    save_figure(fig, output, "parity_moment_variance")

    folder = root / "family_8401" / "fresh_gaussian_codes"
    with np.load(folder / "audit_only.npz") as archive:
        arrays = dict(archive)
    pairs, noise = max(plan["pairs"]), 0.2
    selections = (
        "iid_affine__joint",
        "antithetic_affine__sketch",
        "antithetic_affine__joint",
    )
    matrices, titles = [arrays["truth_indices"]], ["Истинный I"]
    case = next(
        c
        for _, c in cases
        if c["family"] == 8401 and c["function"] == "fresh_gaussian_codes"
    )
    for method in selections:
        row = next(
            r
            for r in case["records"]
            if r["method"] == method and r["noise"] == noise and r["pairs"] == pairs
        )
        matrices.append(arrays[f"{row['key']}__aligned_indices"])
        titles.append(
            f"{LABELS[method]}\nошибок I: {row['recovery']['categorical_errors']}"
        )
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for axis, matrix, title in zip(axes.flat, matrices, titles):
        artist = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        axis.set(title=title, xlabel="Скрытый нейрон", ylabel="Вход")
    fig.colorbar(
        artist, ax=list(axes.flat), ticks=np.arange(-4, 5), shrink=0.75, label="Код"
    )
    fig.suptitle(
        "Фиксированная геометрия8401: оставшийся сбой при noise0.2 и1.074млрд labels"
    )
    save_figure(fig, output, "parity_failed_indexators")

    tables = []
    for function in plan["functions"]:
        for noise in plan["noise"]:
            lines = [
                f"## {function}, noise={noise}",
                "",
                "|Метод|" + "|".join(f"{2 * n:,}" for n in plan["pairs"]) + "|",
                "|---|" + "---:|" * len(plan["pairs"]),
            ]
            for method, label in LABELS.items():
                chosen = {
                    s["pairs"]: s
                    for s in summaries
                    if s["function"] == function
                    and s["noise"] == noise
                    and s["method"] == method
                }
                lines.append(
                    "|"
                    + label
                    + "|"
                    + "|".join(f"{chosen[n]['exact']}/4" for n in plan["pairs"])
                    + "|"
                )
            tables.append("\n".join(lines))
    report = """# G7: parity и совместная диагонализация

Диагностика G6 указала на ошибку третьего момента. THEORY.md§14 до запуска обосновывает две отдельные поправки: parity-сокращение чётной части функции и использование всех срезов тензора вместо одного matrix sketch. Обе техники известны; factor2×2 абляция проверяет их вклад. Восемь функций и четыре геометрии уже открыты в G6: это **exploratory**, не новые confirmation-семейства.

На каждый view выделено одинаковое число labels:2n IID наблюдений против n пар x/−x. Пары стоят двух labels, шум у них независим. Дополнительно одинаковый независимый pilot16384 labels оценивает affine-control-variate. Исходные роли координат, истинные W/C, численные bias/amplitude не подаются. Не используется autograd учителя. После извлечения W/b применяется явно дополнительная signed scalar-DP проекция с фиксированным нулём и4 центрами. Все prespecified бюджеты выполнены; teacher не управляет остановкой/выбором настроек.

**Что получилось:** на исходных C совместная диагонализация даёт4/4 при16.78млн labels даже с noise0.2; при этом обычный matrix sketch даёт0/4. Здесь joint-контроль IID тоже успешен: польза parity не должна приписываться всей комбинации без абляции. На свежих C без шума parity+joint достигает4/4 при134.22млн, IID+joint1/4. При noise0.2 максимальный бюджет1.074млрд даёт3/4 против2/4. Точное универсальное восстановление пока не достигнуто.

Источник statistical-улучшения измерен отдельно: у исходных C без шума при16.78млн labels относительная ошибка M3 снижается примерно0.167→0.017. Measurement-noise не сокращается таким же образом; в самом трудном почти коллинеарном семействе8401 одна ошибка остаётся даже при большом бюджете. Порядок величины числа labels — существенная цена. Это не практический high-efficiency результат и не matched-data победа над Adam/Yeh/LC/LUT-Q: таких новых конкурентов этот опыт не запускал.

Числа в таблицах — **точные I / четыре геометрии**. Бюджеты в заголовках — labels на одну функцию, без одинакового дополнительного pilot.

""" + "\n\n".join(tables)
    report += """

## Кривые и выбранные индексаторы

![Точное извлечение](parity_recovery.png)

![Измеренное уменьшение variance](parity_moment_variance.png)

![Оставшийся сбой](parity_failed_indexators.png)

Хитмапа показывает заранее фиксированную первую fresh-геометрию8401. Это оставшийся трудный случай, не лучший отобранный пример. При максимальном n IID+joint имеет104 ошибки, parity+matrix64, parity+joint23. Поэтому низкую общую weight-MSE нельзя подменять точностью маски и групп.

## Статистика или локальный numerical-start

После freeze отдельно выполнен **oracle-only** контроль oracle_joint_initialization_diagnosis.json: совместная диагонализация observed-тензора инициализирована teacher-derived ортогональным базисом. Во всех восьми noisy fresh/view случаях она приходит к тем же значениям критерия и числу ошибок, что и обычный старт, в пределах округления. Для8401 parity+joint обе инициализации дают23 ошибки и objective≈0.84691649. Это не доказательство глобального оптимума, но конкретная проверка не поддерживает объяснение только плохим стартом solver.

Необходим следующий диагноз: почему оставшаяся неоднородная/коррелированная погрешность W/b нарушает zero/weak-code разделение при scalar-DP, и помогает ли observation-only numerical refinement перед hard projection. Нельзя напрямую приписывать Gaussian F-гарантию линейного этапа этим нелинейно оценённым W. Возможная комбинация метода имеет явные отличия от LUT-Q/DARTS, но её приоритет и превосходство над сильными конкурентами ещё не установлены.
"""
    path = output / "REPORT.md"
    path.write_text(report)
    save_json(
        output / "report_manifest.json",
        {
            "report_sha256": digest(path),
            "protocol_sha256": digest(root / "protocol.json"),
            "audits_sha256": {
                str(folder.relative_to(root)): digest(folder / "audit.json")
                for folder, _ in cases
            },
            "oracle_initialization_diagnosis_sha256": digest(
                root / "oracle_joint_initialization_diagnosis.json"
            ),
        },
    )


if __name__ == "__main__":
    main()
