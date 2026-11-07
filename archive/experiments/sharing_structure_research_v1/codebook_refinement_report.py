"""Report G20 without repeating frozen source fits or oracle audits."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

LABELS = {
    "continuous": "Continuous",
    "continuous_dp_start": "Continuous + DP start",
    "integer_mass": "Fixed integer masses",
}
COLORS = {
    "continuous": "tab:blue",
    "continuous_dp_start": "tab:orange",
    "integer_mass": "tab:green",
}


def generate(root):
    plan = json.loads((root / "protocol.json").read_text())
    frozen = json.loads((root / "models_frozen.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    if audit["models_frozen_sha256"] != digest(root / "models_frozen.json"):
        raise RuntimeError("Audit does not match G20 freeze")
    for relative, expected in frozen["sha256"].items():
        if digest(root / relative) != expected:
            raise RuntimeError("Frozen G20 source changed")
    cases = []
    for job_id, job in enumerate(plan["jobs"]):
        path = root / "models" / f"job_{job_id:03d}"
        meta = json.loads(path.with_suffix(".json").read_text())
        with np.load(path.with_suffix(".npz")) as data:
            arrays = dict(data)
        cases.append({"job_id": job_id, "job": job, "meta": meta, "arrays": arrays})
    with np.load(root / "audit_only.npz") as data:
        truth = dict(data)
    rows = []
    for panel in plan["panels"]:
        for variant in ("all", "local", "quartet_mix", "random"):
            for method in LABELS:
                selected = [
                    row
                    for row in audit["records"]
                    if row["panel"] == panel
                    and row["method"] == method
                    and (variant == "all" or row["job"]["variant"] == variant)
                ]
                rows.append(
                    {
                        "variant": variant,
                        **panel,
                        "method": method,
                        "cases": len(selected),
                        "complete": sum(row["complete"] for row in selected),
                        "exact": sum(row["exact_I"] for row in selected),
                        "median_code_max_error": float(
                            np.median([row["code_max_error"] for row in selected])
                        ),
                        "correct_DP_masses": sum(
                            row["dp_masses_exact"] for row in selected
                        ),
                        "full_available_source_labels": selected[0][
                            "full_available_source_labels"
                        ],
                    }
                )
    report = root / "report"
    report.mkdir(exist_ok=False)
    with (report / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    panels = []
    for panel in plan["panels"]:
        source = [
            row
            for case in cases
            for row in case["meta"]["records"]
            if row["panel"] == panel and row["method"] == "integer_mass"
        ]
        panels.append(
            {
                **panel,
                "cases": len(source),
                "raw_DP_physical": sum(row["dp"]["raw_physical"] for row in source),
                "max_path_numeric_difference": max(
                    abs(row["dp"]["path_numeric_difference"]) for row in source
                ),
                "axis_replicates": source[0]["axis_replicates"],
                "axis_train_and_validation_labels": source[0][
                    "axis_train_and_validation_labels"
                ],
                "slice_labels": source[0]["slice_labels"],
                "full_available_source_labels": source[0][
                    "full_available_source_labels"
                ],
            }
        )
    save_json(
        report / "summary.json",
        {
            "comparisons": rows,
            "panels": panels,
            "models_frozen_sha256": digest(root / "models_frozen.json"),
            "wrong_complete": sum(
                row["complete"] and not row["exact_I"] for row in audit["records"]
            ),
            "scope": "Opened18 G19 networks; known bias/readout, one source each. Density/SNR/count constraint ablations; no native competitors, independent confirmation, new source-C transfer or novelty claim.",
        },
    )
    plot_recovery(report, rows)
    plot_prefixes(report, cases, truth, plan)
    plot_indexators(report, cases, truth, plan)
    plot_weights(report, cases, truth, plan)
    write_report(report, rows, panels, audit)


def selected_record(case, method, noise=0.2, grid=1025, budget=16777216):
    return next(
        row
        for row in case["meta"]["records"]
        if row["method"] == method
        and row["panel"]
        == {"noise": noise, "grid": grid, "coarse_equivalent_repeats": budget}
    )


def plot_recovery(report, rows):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for col, variant in enumerate(("local", "quartet_mix", "random")):
        for method, color in COLORS.items():
            for grid, style in ((257, "--"), (1025, "-")):
                selected = [
                    row
                    for row in rows
                    if row["variant"] == variant
                    and row["method"] == method
                    and row["noise"] == 0.2
                    and row["grid"] == grid
                ]
                x = [row["full_available_source_labels"] for row in selected]
                label = f"{LABELS[method]}, grid{grid}"
                axes[0, col].plot(
                    x,
                    [row["exact"] / row["cases"] for row in selected],
                    marker="o",
                    color=color,
                    ls=style,
                    label=label,
                )
                axes[1, col].plot(
                    x,
                    [row["median_code_max_error"] for row in selected],
                    marker="o",
                    color=color,
                    ls=style,
                )
        axes[0, col].set_title(variant)
        axes[0, col].set_ylim(-0.04, 1.04)
        axes[0, col].set_ylabel("Exact full I / 6")
        axes[1, col].set_ylabel("Median max code error (audit)")
        axes[1, col].set_yscale("log")
        for ax in axes[:, col]:
            ax.set_xscale("log")
            ax.set_xlabel(
                "All available axis + slice labels (includes axis validation)"
            )
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "G20: density, axis-SNR and integer-mass ablations at noiseSD0.2\nAxes up to 275 billion labels per condition; slices unchanged; opened18 networks"
    )
    fig.savefig(report / "codebook_recovery.png", dpi=180)
    plt.close(fig)


def plot_prefixes(report, cases, truth, plan):
    chosen = [
        next(case for case in cases if case["job"] == job)
        for job in (
            {"family": 8801, "variant": "local"},
            {"family": 8801, "variant": "quartet_mix"},
            {"family": 8905, "variant": "random"},
        )
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
    for line, case in enumerate(chosen):
        expected = truth[f"job_{case['job_id']}__truth_I"]
        for method, color in COLORS.items():
            row = selected_record(case, method)
            trace = row["trace"]
            continuous = [point for point in trace if "slice_mse_continuous" in point]
            integer = [point for point in trace if "slice_mse_integer" in point]
            axes[line, 0].plot(
                [p["prefix_length"] for p in continuous],
                [p["slice_mse_continuous"] for p in continuous],
                marker="o",
                color=color,
                label=LABELS[method],
            )
            axes[line, 1].plot(
                [p["prefix_length"] for p in integer],
                [p["slice_mse_integer"] for p in integer],
                marker="o",
                color=color,
            )
            indices = case["arrays"][row["key"] + "__indices"]
            recalls = []
            for step, index in enumerate(indices, 1):
                partial_truth = expected.copy()
                partial_truth[case["arrays"]["order"][step:]] = 0
                metric, _, _ = align_indices(index, partial_truth, plan["codes"])
                recalls.append(metric["mask_recall"])
            axes[line, 2].plot(
                range(1, len(recalls) + 1), recalls, marker="o", color=color
            )
        for col in (0, 1):
            axes[line, col].set_yscale("log")
            axes[line, col].axhline(0.2**2 / 4096, color="gray", ls="--")
        axes[line, 2].set_ylim(-0.04, 1.04)
        for ax in axes[line]:
            ax.set_xlabel("Processed coordinates (not optimizer epochs)")
            ax.set_xticks([1, 4, 8, 12, 16])
        axes[line, 0].set_ylabel(
            f"{case['job']['variant']} {case['job']['family']}\nContinuous slice-fit MSE"
        )
        axes[line, 1].set_ylabel("Integer prefix slice-fit MSE")
        axes[line, 2].set_ylabel("Mask recall on processed rows (audit only)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "G20 measured loss/recall at dense grid + high axis precision, SD0.2\nFirst local/mixed family and random8905 regression case; abstentions have short curves"
    )
    fig.savefig(report / "codebook_slice_loss_recall.png", dpi=180)
    plt.close(fig)


def plot_indexators(report, cases, truth, plan):
    selected = [case for case in cases if case["job"]["variant"] == "random"]
    fig, axes = plt.subplots(6, 4, figsize=(13, 16), constrained_layout=True)
    for line, case in enumerate(selected):
        expected = truth[f"job_{case['job_id']}__truth_I"]
        images, titles = [expected], ["True signed I"]
        for method, label in LABELS.items():
            record = selected_record(case, method)
            indices = case["arrays"][record["key"] + "__indices"]
            if len(indices):
                _, aligned, _ = align_indices(indices[-1], expected, plan["codes"])
            else:
                aligned = np.zeros_like(expected)
            images.append(aligned)
            titles.append(
                f"{label}\n{len(indices)}/16"
                + ("; no full I" if len(indices) < 16 else "")
            )
        for col, (data, title) in enumerate(zip(images, titles, strict=True)):
            im = axes[line, col].imshow(data, cmap="RdBu_r", vmin=-4, vmax=4)
            axes[line, col].set_title(title, fontsize=9)
            axes[line, col].set_xticks([0, 7, 15])
            axes[line, col].set_yticks([0, 7, 15])
            if col == 0:
                axes[line, col].set_ylabel(
                    f"Random {case['job']['family']}\nInput coordinate"
                )
            if line == 5:
                axes[line, col].set_xlabel("Hidden neuron (audit aligned)")
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.6,
        label="Signed code ID;0 = absent/unprocessed",
    )
    fig.suptitle(
        "G20 selected I and partial prefixes, SD0.2 / dense grid / high axis budget\nAll six arbitrary random networks; no teacher selection"
    )
    fig.savefig(report / "codebook_indexators.png", dpi=180)
    plt.close(fig)


def plot_weights(report, cases, truth, plan):
    chosen = [
        next(case for case in cases if case["job"] == job)
        for job in (
            {"family": 8801, "variant": "local"},
            {"family": 8905, "variant": "random"},
        )
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for line, case in enumerate(chosen):
        expected = truth[f"job_{case['job_id']}__truth_I"]
        center = truth[f"job_{case['job_id']}__true_C"]
        actual = np.r_[0.0, center][np.abs(expected)] * np.sign(expected)
        images, titles = [actual], ["True numerical W"]
        for method, label in LABELS.items():
            record = selected_record(case, method)
            indices = case["arrays"][record["key"] + "__indices"]
            index = indices[-1] if len(indices) else np.zeros_like(expected)
            _, _, columns = align_indices(index, expected, plan["codes"])
            weights = np.r_[0.0, record["codebook"]][np.abs(index)] * np.sign(index)
            images.append(weights[:, columns])
            titles.append(
                label
                + (
                    "\nFull W"
                    if record["complete"]
                    else f"\nPrefix W {len(indices)}/16; no full model"
                )
            )
        limit = max(float(np.abs(value).max()) for value in images)
        for col, (data, title) in enumerate(zip(images, titles, strict=True)):
            im = axes[line, col].imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit)
            axes[line, col].set_title(title, fontsize=9)
            axes[line, col].set_xticks([0, 7, 15])
            axes[line, col].set_yticks([0, 7, 15])
            axes[line, col].set_xlabel("Hidden neuron")
        axes[line, 0].set_ylabel(
            f"{case['job']['variant']} {case['job']['family']}\nInput coordinate"
        )
        fig.colorbar(
            im, ax=axes[line].tolist(), shrink=0.7, label="Actual numerical weight"
        )
    fig.suptitle(
        "G20 actual W from learned I and C; hidden columns aligned for audit\nOne integer-mass improvement and one regression at the same registered panel"
    )
    fig.savefig(report / "codebook_weights.png", dpi=180)
    plt.close(fig)


def write_report(report, rows, panels, audit):
    aggregate = [row for row in rows if row["variant"] == "all"]
    table = "\n".join(
        "|{}|{}|{}|{}|{}/18|{:.2e}|".format(
            row["noise"],
            row["grid"],
            row["coarse_equivalent_repeats"],
            row["method"],
            row["exact"],
            row["median_code_max_error"],
        )
        for row in aggregate
    )
    costs = "\n".join(
        "|{}|{}|{}|{:,}|{:,}|{:,}|{}/18|{:.2e}|".format(
            row["noise"],
            row["grid"],
            row["coarse_equivalent_repeats"],
            row["axis_replicates"],
            row["axis_train_and_validation_labels"],
            row["full_available_source_labels"],
            row["raw_DP_physical"],
            row["max_path_numeric_difference"],
        )
        for row in panels
    )
    wrong = sum(row["complete"] and not row["exact_I"] for row in audit["records"])
    text = f"""# G20: исправление кодбука через сетку, SNR и целые количества

5 ноября 2026. **На плотной сетке без шума все три варианта восстановили точный I в 18/18 случаях. При шуме 0.2 и большом осевом бюджете целые количества дали 15/18 против 14/18 у непрерывного контроля. На случайных структурах результат ухудшился с 4/6 до 3/6. Общего превосходства нет.** Все 18 запусков, 540 состояний методов, заморожены до аудита. Использованы пять свободных V100: 1, 2, 3, 6, 7. GPU 0 занял другой процесс до старта; занятые 0, 4, 5 не затрагивались.

## Почему проверено именно это

G19 дал 15/18 точных I без шума и 0/108 в шумных состояниях. Факторный контроль при шуме 0.2 и 4096 повторах дал 16/18 с истинным C и шумными срезами, но 0/18 с оценённым C и чистыми срезами. У random8903 почти нулевая ошибка на редкой сетке не означала правильного C. THEORY.md §27 до G20 вывел ограничение интерсептов: на каждом линейном участке осевого отклика свободный член кратен b/sqrt(H), а множитель — целое число уже активированных весов. Приращения M положительны, их сумма не превосходит DH. Истинное число активных связей не раскрывается.

Реализована DP по целым интерсептам со свободными наклонами. Она оптимизирует собственную релаксацию с возможными разрывами; это не автоматическое решение непрерывной физической модели. Допустимость исходного пути, выбранные количества, прямой SSE и численная разница сохранены. Инициализация через DP сравнивается отдельно от подгонки C с фиксированными целыми M. [CPOP](https://arxiv.org/abs/1701.01672) и [Troeng–Fält 2018](https://arxiv.org/abs/1806.11041) уже решают задачи непрерывных сплайнов иначе. Их алгоритмы здесь не воспроизводятся буквально. Приоритет нашей специализации не установлен.

## Протокол

Те же 18 открытых сетей G19: шесть local, шесть quartet_mix с парными перестановками входов, шесть произвольных случайных индексаторов. C и I прежние: это парная диагностика, не новое независимое подтверждение. Одна функция на I; общий bias 0.15 и положительная нормировка выхода известны, offset равен нулю. Новые осевые запросы используют 257 или 1025 точек на прежнем геометрическом диапазоне .001–1024. Срезы и порядок входов точно совпадают с G19: без шума одна метка, при шуме 0.2 и 2 — по 4096 меток. Совпадение сохранённых массивов проверено до обучения. K=4 и размеры известны; маска, истинные C, M и роли координат обучению недоступны.

Два общих осевых бюджета соответствуют 4096 и 16777216 повторам на редкой сетке. На плотной сетке повторов на точку примерно в четыре раза меньше. Обучающие осевые запросы и независимые запросы в промежуточных точках для валидации входят в бюджет: repeat=floor((2*257−1)*coarseR/(2*grid−1)). Одинаковые бесшумные бюджеты не повторялись. Все методы видят одинаковые наблюдения на каждом panel. Gaussian среднее r меток точно симулировано через sigma/sqrt(r)Z; шум парный между режимами. Повторы меток не являются независимыми геометриями.

У трёх методов по семь численных стартов и одинаковый предел итераций. **Continuous** использует исходную цель со свободными амплитудами. В G20 все методы выбирают старт по независимой осевой валидации; в G19 выбор был по обучающей ошибке. **Continuous+DPstart** заменяет только первый старт кодбуком из DP, остальные шесть совпадают. **Integer_mass** использует те же семь стартов, но фиксирует амплитуды M/sqrt(H) из DP. Дополнительного поиска M или подстановки истинных количеств нет. Выбранный C передаётся в неизменный поиск префиксов G19. Сохраняются все I и отказы; сетка и бюджет по истине не выбираются.

## Все итоги по18 структурам

|Noise SD|Axis grid|Coarse-equivalentR|Метод|Exact full I|Median max-C error|
|---|---:|---:|---|---:|---:|
{table}

Без шума на редкой сетке одна инициализация через DP улучшает 15/18 до 17/18; фиксация целых количеств дополнительных успехов не даёт. Плотная сетка даёт 18/18 уже непрерывному контролю. Нельзя приписывать весь выигрыш целочисленному компоненту. При умеренном шуме и большом бюджете плотная сетка даёт 14/18 у continuous и 15/18 у integer: local 4→6, mixed 6→6, random 4→3. Random8905 решён непрерывным вариантом, но неправильные фиксированные M приводят integer к отказу. На редкой сетке при большом бюджете seeded/integer дают 14/18 против 13/18 у continuous. При малом осевом бюджете все дают 0/18. При шуме 2 все также дают 0/18; неизменный поиск префиксов уже давал там 0/18 даже с истинным C.

Всего ошибочных полных выдач: {wrong}. Неуспех здесь означает отказ от полного I. Это не доверительная теорема: допустимость округлённых количеств не является сертификатом точности под шумом. Все состояния зависимы; local/mixed используют парные геометрии.

![Codebook error и точный I по полной стоимости данных](codebook_recovery.png)

## Измеренные loss, recall, выбранные I и numerical W

![Настоящие slice-fit residual и recall](codebook_slice_loss_recall.png)

Здесь X — число обработанных входных координат. Показаны измеренные остатки LS и подгонки дискретного префикса. Нелинейный этап C сохраняет результаты семи стартов, но не траекторию SGD; отсутствующие кривые не выдумываются. Recall неполного префикса считается только на обработанных строках и не означает успех полного I.

![Все шесть arbitrary random индексаторов и partial prefixes](codebook_indexators.png)

![Настоящие numerical W при тех же данных](codebook_weights.png)

Численные W построены из реальных исходных I и C. Для сравнения меняется только перестановка скрытых нейронов. Смена имён и знаков кодовых групп применяется лишь к категориальным картинкам I. При отказе префикс W явно помечен; неполная матрица не выдаётся за полную модель. Native LC, Yeh, LUT-Q и плотный Adam на этих данных не обучались; прежние числа G17 нельзя переносить в эту таблицу.

## Стоимость и контроль релаксации

|Noise|Grid|CoarseR|Фактическийr/query|Axis train+validationlabels|Все source labels|RawDPphysical|Max прямой-SSE минусDP-SSE, abs|
|---|---:|---:|---:|---:|---:|---:|---:|
{costs}

Высокий режим стоит около **275.55 миллиарда меток на условие**, плюс 8192 старых IID-меток валидации для отчёта. Это очень дорого. Сырые r шумных меток не материализованы, но их статистическая стоимость учтена. Новые осевые метки прибавлены ко всем доступным срезам G19, даже если процедура остановилась раньше. Из-за целого floor разница осевых бюджетов между сетками мала; точные числа приведены в таблице. Равенство FLOPs или времени не заявляется: integer DP добавляет вычисления, фактические времена сохранены.

Исходный DP-путь не всегда задаёт допустимую непрерывную кривую. Численная подгонка C с фиксированными M после DP не получает автоматической глобальной гарантии. Совпадение прямого SSE и DP проверяет вычисление релаксации, но не истинность M. При шуме 0.2 и большом бюджете истинные M получены в 17/18 случаях на редкой сетке и 16/18 на плотной; при шуме 2 — в 12/18 и 9/18. Фиксация неверного M может ухудшить результат. Точность I и ошибки C измерены только после заморозки.

## Артефакты и следующий содержательный шаг

Root `results/20261105/codebook_refinement_g20`; код codebook_counts.py, codebook_refinement.py, codebook_refinement_report.py, неизменный prefix_census.py. Сохранены protocol, theory_at_registration, executed_code_snapshot, models_frozen.json. Исходные job JSON/NPZ отделены от audit.json и audit_only.npz. В отчёте comparison.csv, summary.json и четыре просмотренные фигуры. Все обучения, аудит и отчёт завершены. Программные тесты не писали и не запускали; выполнены минимальный Ruff, необходимый аудит математики и происхождения данных, просмотр фигур.

G20 исправил бесшумный сбой с помощью более информативной сетки, но улучшение SNR получено огромным числом меток. Целые M помогли локально, ухудшили один случайный индексатор и не дали устойчивого общего выигрыша. Следующий шаг должен отвечать измеренной проблеме: допустимое назначение целых количеств по ошибке функции вместо свободного LS с округлением; совместное уточнение C по уже наблюдаемым префиксным данным. До изменения нужны теория и отдельные абляции. Пока эти улучшения не выполнены.

Большая цель сохраняется: полный общий I разных source-C, свободные bias и readout, независимые геометрии и шумовые условия, сильные конкуренты при согласованных данных и вычислениях, отдельный перенос I со свежими C и приоритет собственного компонента. Общая конкурентная победа, новизна и завершение цели пока не установлены.
"""
    (report / "REPORT.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    generate(args.root)


if __name__ == "__main__":
    main()
