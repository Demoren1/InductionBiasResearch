"""Diagnostic evidence, frozen correction comparison and scientific figures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from experiments.darts_structured_tasks_v1.data import save_json
from experiments.sharing_benchmark_v1.report import aggregate, number, save_figure

from .run import METHODS

LABELS = {
    "bank_dp": "Прежний банк + DP",
    "bank_dp_late": "Банк + плавный штраф",
    "bank_dp_64": "Банк: 64 предложения",
    "bank_dp_long": "Банк: 320 итераций",
    "joint_constant": "Совместный I, 64 функции",
    "joint_late": "Совместный I + плавный штраф",
    "joint_projection": "Одна общая проекция, 64 функции",
    "joint_8": "Совместный I, 8 функций",
    "joint_projection_8": "Одна общая проекция, 8 функций",
    "bank_query_edits": "Банк + правки I по общему query",
    "oracle": "Истинный I",
}


def diagnostics_figure(root, output, retrospective):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for axis, family in zip(axes.flat, retrospective):
        points = family["timeline"]
        steps = [point["step"] for point in points]
        axis.plot(
            steps,
            [point["active_mean"] for point in points],
            color="#1f77b4",
            label="Ненулевых связей, среднее",
        )
        axis.axhline(
            42, color="black", linestyle="--", label="Истинные 42, только аудит"
        )
        other = axis.twinx()
        other.plot(
            steps,
            [point["best_I_errors"] for point in points],
            color="#d62728",
            label="Ошибок лучшего предложения",
        )
        axis.set(
            title=f"Старое семейство {family['family']}",
            xlabel="Итерация",
            ylabel="Число активных связей",
            ylim=(0, 230),
        )
        other.set(ylabel="Ошибок I, минимум по 8 предложениям", ylim=(0, 230))
        if family["family"] == 201:
            axis.legend(fontsize=7, loc="upper left")
            other.legend(fontsize=7, loc="upper right")
        axis.grid(alpha=0.2)
    fig.suptitle("Прежний поиск: все восемь предложений сначала схлопываются в ноль")
    save_figure(fig, output, "old_search_diagnostics")

    positions = np.arange(len(retrospective))
    fig, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    axis.bar(
        positions - 0.18,
        [row["selected_source_mse"] for row in retrospective],
        width=0.36,
        label="Выбранный I",
    )
    axis.bar(
        positions + 0.18,
        [row["oracle_source_mse"] for row in retrospective],
        width=0.36,
        label="Если предложить истинный I",
    )
    axis.set(
        yscale="log",
        ylabel="Source-query MSE после нового fit C",
        xlabel="Старое семейство",
        title="Критерий отбора предпочитает истинную структуру; проблема в её обнаружении",
    )
    axis.set_xticks(positions, [row["family"] for row in retrospective])
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    save_figure(fig, output, "oracle_insertion")


def correction_figures(root, output, summary):
    methods = [*METHODS, "oracle"]
    positions = np.arange(len(methods))
    colors = [
        "#247ba0" if method != "bank_query_edits" else "#2a9d55" for method in methods
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    axes[0].barh(
        positions,
        [summary[method]["exact_all_seeds_families"] for method in methods],
        color=colors,
    )
    axes[0].set_yticks(positions, [LABELS[method] for method in methods])
    axes[0].invert_yaxis()
    axes[0].set(
        xlim=(0, 6.7),
        xlabel="Семейств с точным I во всех трёх запусках",
        title="Шесть новых скрытых геометрий",
    )
    for position, method in enumerate(methods):
        row = summary[method]
        axes[0].text(
            row["exact_all_seeds_families"] + 0.08,
            position,
            f"{row['exact_cells']}/{row['replicas']}",
            va="center",
            fontsize=8,
        )
    errors = [summary[method]["transfer"]["4"]["population_mse"] for method in methods]
    axes[1].barh(positions, errors, color=colors)
    axes[1].set_yticks(positions, [LABELS[method] for method in methods])
    axes[1].invert_yaxis()
    axes[1].set(
        xscale="log",
        xlabel="Population MSE, 4 новых support-примера",
        title="Независимые новые коэффициенты, свежий C",
    )
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
    save_figure(fig, output, "correction_comparison")

    folder = root / "hidden/family_1101"
    chosen = (
        "oracle",
        "bank_dp",
        "bank_dp_late",
        "bank_dp_64",
        "joint_constant",
        "bank_query_edits",
    )
    colors = [
        "#9e0142",
        "#f46d43",
        "#fee08b",
        "#f7f7f7",
        "#abdda4",
        "#66c2a5",
        "#3288bd",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-3.5, 4), cmap.N)
    records = json.loads((folder / "audit/summary.json").read_text())["records"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    raw = {}
    for axis, method in zip(axes.flat, chosen):
        seed = 0 if method == "oracle" else 31
        with np.load(folder / "audit" / f"{method}_seed_{seed}.npz") as archive:
            index = archive["aligned_indices"]
            raw[method] = {
                "raw_indices": archive["indices"].tolist(),
                "aligned_indices": index.tolist(),
            }
        record = next(
            row for row in records if row["method"] == method and row["seed"] == seed
        )
        image = axis.imshow(
            index, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto"
        )
        axis.set_title(
            f"{LABELS[method]}\nошибок I: {record['recovery']['categorical_errors']}",
            fontsize=10,
        )
        axis.set_xlabel("Вход")
        axis.set_ylabel("Выход")
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.8,
        ticks=np.arange(-3, 4),
        label="Согласованный индекс, 0 — отключённая связь",
    )
    fig.suptitle(
        "Первое новое семейство 1101, seed 31: маска и шэринг ищутся совместно"
    )
    save_figure(fig, output, "corrected_indexators")
    save_json(output / "corrected_indexators.json", raw)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    with np.load(folder / "target_after_freeze.npz") as archive:
        teacher = archive["teacher_weights_audit_only"][0]
    limit = np.abs(teacher).max()
    for axis, method in zip(axes.flat, chosen):
        seed = 0 if method == "oracle" else 31
        with np.load(folder / "audit" / f"{method}_seed_{seed}.npz") as archive:
            weights = archive["weights_support_4"]
        error = np.square(weights - teacher).sum() / teacher.shape[0]
        image = axis.imshow(
            weights,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            aspect="auto",
        )
        axis.set_title(
            f"{LABELS[method]}\nMSE первой новой задачи: {error:.3g}", fontsize=10
        )
        axis.set_xlabel("Вход")
        axis.set_ylabel("Выход")
    fig.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.8, label="Вес после fresh fit C"
    )
    fig.suptitle("Новая задача: исходные кодбуки отброшены, 4 support-примера")
    save_figure(fig, output, "corrected_weights")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, family in zip(axes.flat, range(1101, 1107)):
        path = (
            root
            / "hidden"
            / f"family_{family}"
            / "confirmation/bank_query_edits_seed_31.json"
        )
        history = json.loads(path.read_text())["history"]
        for key, label in (
            ("source_query_mse", "Source query"),
            ("selection_mse", "Новые validation-задачи"),
        ):
            axis.plot(
                [row["step"] for row in history],
                [row[key][0] for row in history],
                "o-",
                markersize=3,
                label=label,
            )
        axis.set(
            title=f"Новое семейство {family}",
            xlabel="Принятых изменений одной позиции I",
            ylabel="MSE",
            yscale="log",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(
        "Прямые правки индексатора: кодбуки точно fit-ятся заново после каждой правки"
    )
    save_figure(fig, output, "query_edit_curves")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    records = json.loads((root / "audit_summary.json").read_text())["records"]
    summary = aggregate(records)
    retrospective = json.loads((root / "retrospective/summary.json").read_text())[
        "families"
    ]
    output = root / "report"
    output.mkdir(exist_ok=True)
    save_json(output / "summary.json", summary)
    lines = [
        "# Почему прежний поиск не находил общий шэринг",
        "",
        "Диагностика 5 ноября 2026. Старые шесть семейств использованы только для объяснения сбоя. Все исправления подбирались на новых pilot-семействах 1001/1002 с новым random seed. Затем настройки заморожены и проверены на шести ещё не виденных геометриях 1101–1106, по три запуска и 100 новых коэффициентных задач на семейство. Исходные размеры/шум: W14×16, K3, 64 source-функции по 4 support + 4 query, σ=0.1. Новый test сгенерирован после фиксации всех моделей.",
        "",
        "## Проверенные причины",
        "",
        "1. **Сбой обнаружения, а не фильтрации.** В прежних пяти неудачных семействах истинный I не появился ни в одном из 8 предложений на любой из 81 итерации. Если предложить истинный I, его source-query ошибка около 0.01 против 0.25–1.59 у выбранных неправильных I. Тот же критерий его предпочёл бы. Повтор точно воспроизвёл сохранённый I во всех 6 семействах.",
        "2. **Штраф обнуляет даже правильное начало.** При прежних λ=0.03 и ρ₀=0.01 стоимость активного выбора в проекции 2λ/ρ=6. Во всех семействах все восемь начальных предложений стали полностью нулевыми. Даже при инициализации истинным I с оптимальными по данным числами возвращённый validation-checkpoint оказался неправильным: 0/6. При той же инициализации, том же ρ и плавном включении λ выбран правильный checkpoint в 6/6. Отбор включает начальный шаг; это не утверждение о сохранении I на каждой последующей итерации. Oracle-инициализация использовалась только в диагностике, не в исправлениях.",
        "3. **Плотный fit отдельной функции не определяет её веса.** Дизайн каждой source-функции имеет rank8 при 16 входах, nullity8. Полный W имеет 14×8=112 не наблюдаемых по её обучающим данным направлений. Шэринг с малым кодбуком потенциально идентифицируем; проблема в том, что промежуточный свободный Z остаётся неоднозначным. Евклидова проекция Z в кодбук может сохранять эту неоднозначность и выбирать структуру, плохо работающую при новых коэффициентах.",
        "4. **Не обнаружена ошибка скалярного DP.** На сохранённых Z+U его penalized projection-objective не хуже проекции на истинные группы с оптимальными численными центрами (расхождение вниз только порядка 2.4e−16). Это проверка конкретных состояний, не новое доказательство глобальной сходимости. Глобально решается проекция; не весь поиск I по ошибке задачи.",
        "5. **Кодбук не является объяснением плохого переноса.** Итоговые числа C при фиксированном I найдены точным МНК. После правок самого I тот же численный решатель достигает уровня истинной структуры.",
        "",
        "## Исправления на новых семействах",
        "",
        "Каждому основному варианту — та же заранее заданная сетка из 12 настроек, выбор по независимым coefficient-validation задачам двух новых pilot-семейств. Увеличение банка и числа итераций используют ту же сетку, но могут выбрать другие настройки: это сравнение полных validation-процедур, а не изоляция одного численного параметра при неизменном выбранном ρ/λ. Три дополнительных причинных контроля общей проекции зарегистрированы после pilot, до новых семейств и test; они наследуют одну фиксированную настройку joint_constant без дополнительного подбора.",
        "",
        "| Вариант | Точный I: все сиды семейства | Точных запусков | F1 маски | MSE4 переноса |",
        "|---|---:|---:|---:|---:|",
    ]
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["method", "exact_families", "exact_runs", "replicas", "mask_f1", "mse4"]
        )
        for method in (*METHODS, "oracle"):
            row = summary[method]
            error = row["transfer"]["4"]["population_mse"]
            lines.append(
                f"| {LABELS[method]} | {row['exact_all_seeds_families']}/6 | {row['exact_cells']}/{row['replicas']} | {row['mask_f1']:.3f} | {number(error)} |"
            )
            writer.writerow(
                [
                    method,
                    row["exact_all_seeds_families"],
                    row["exact_cells"],
                    row["replicas"],
                    row["mask_f1"],
                    error,
                ]
            )
    lines.extend(
        [
            "",
            "MSE аналитически усреднена по Gaussian input, затем по 100 новым coefficient-tasks и seed внутри семейства, затем по шести геометриям. Четыре support-примера, шум только в support, test-query чистый. Все source-кодбуки отброшены. Повторы детерминированных bank/coordinate вариантов не являются новыми независимыми геометриями; основная единица устойчивости — семейство.",
            "",
            "## Рабочее исправление без дополнительной кластеризации",
            "",
            "Вариант bank_query_edits продолжает прежний поиск по дискретному I. Он не фиксирует маску: любая позиция может стать нулём, присоединиться к любой группе или изменить знак. При каждой кандидатной правке заново точно fit-ятся отдельные C_t на support каждой source-функции. Выбирается правка с наименьшей средней query-ошибкой по всем 64 функциям. Независимая validation выбирает checkpoint, но не саму правку. Лимит 64 правки задан до новых семейств; фактически понадобилось 0–23 принятых правки. Начальная гипотеза — прежний bank_dp, никакие истинные позиции или группы не передаются.",
            "",
            "```text",
            "I ← прежний поиск банка, выбранный по validation",
            "повторить не более 64 раз:",
            "    для каждой позиции и каждого альтернативного индекса 0, ±1, …, ±K:",
            "        I_candidate ← изменить одну позицию I",
            "        для каждой source-функции t:",
            "            C_t ← точный fit на support при I_candidate",
            "        score ← средняя query-ошибка всех source-функций",
            "    принять лучшую правку, если она улучшает score",
            "    сохранить checkpoint при улучшении независимой validation",
            "вернуть validation-checkpoint I",
            "на каждой новой задаче: зафиксировать I и обучить свежий C",
            "```",
            "",
            "Скоринг перебора реализован через точные rank-two обновления Gram-матриц; выбранная правка дополнительно сверяется с явным новым fit. Это ускорение для линейного benchmark. Здесь нет MSR/Yeh warm-start, поиска только внутри готовой маски или кластеризации чужого U.",
            "",
            "## Что дал общий поиск",
            "",
            "Совместный вариант хранит Z_t и отдельные C_t, но один I для всех исходных функций. Его локальная векторная проекция назначений/центров выполняется внутри каждого consensus-шагa; точного многомерного DP не заявляется. При 64 source-функциях общий поиск дал 17/18, а одноразовая общая проекция — тоже 17/18. Поэтому при таком объёме данных выигрыш нельзя приписывать итерациям consensus: основную пользу уже даёт объединение информации разных функций. При восьми функциях полный consensus улучшает 3/18 до 15/18 относительно одной проекции, но всё ещё не устойчив во всех запусках. Плавный штраф при выбранном λ=0 для общего поиска ничего не меняет — отдельная строка это показывает.",
            "",
            "## Практический вывод и границы",
            "",
            "Основной дефект — поиск Euclidean-структур для отдельных недоопределённых Z с последующим выбором одной готовой гипотезы. Правильный I иногда вообще не предлагается, а сильный ранний штраф дополнительно разрушает полезные состояния. Рабочее исправление в этой проверке — после инициализации продолжать совместный поиск I по качеству адаптации кодбуков на нескольких задачах. Плавный штраф полезен как защита, но недостаточен сам по себе.",
            "",
            "Результат 18/18 относится к шести новым линейным скрытым свёрткам, а не к произвольным нейросетям. Однопозиционный поиск может попадать в локальный минимум и требует O(EK) кандидатных оценок на правку. Быстрый точный fit и Gram-обновления специфичны линейному случаю. Для нелинейного DeepSets понадобится отдельно проверить ограниченные inner-обновления C и стоимость такого поиска. Это диагностика и исправление, не заявление алгоритмической новизны или глобальной гарантии.",
            "",
            "## Рисунки и воспроизводимость",
            "",
            "- [Динамика старых предложений](old_search_diagnostics.png), [контроль добавления истинного I](oracle_insertion.png).",
            "- [Сводное сравнение исправлений](correction_comparison.png), [выбранные индексаторы](corrected_indexators.png), [raw/aligned I](corrected_indexators.json).",
            "- [Новые численные веса](corrected_weights.png), [source/validation при правках I](query_edit_curves.png).",
            "- [Таблица CSV](comparison.csv), [полные метрики](summary.json). Для всех рисунков есть SVG.",
            "",
            "protocol.json/registration.json фиксируют исходную серию, controls_frozen.json — дополнительные контроли до confirmation. hpo_frozen.json и models_frozen.json созданы до доступа к новым teacher/test. registered_code — исходники диагностики/pilot; confirmation_code — фактические исходники confirmation/evaluation. Никаких software-тестов, commit/push или изменений прежних результатов.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    diagnostics_figure(root, output, retrospective)
    correction_figures(root, output, summary)
    print(output / "REPORT.md")
    print(
        "\n".join(
            lines[
                lines.index(
                    "| Вариант | Точный I: все сиды семейства | Точных запусков | F1 маски | MSE4 переноса |"
                ) :
            ][:13]
        )
    )


if __name__ == "__main__":
    main()
