"""Export measured outcomes and plots, without consulting an analytic indexator."""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "dense": "Плотный Adam",
    "best": "Одна найденная структура",
    "bank_sample": "Выбор из банка",
    "ar": "Авторегрессия",
    "frequency": "Независимые частоты",
    "random": "Свежая случайная структура",
    "shuffled": "Перестановка входов",
    "mask_only": "Та же маска без sharing",
}


def save_figure(figure, path):
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="experiments/random_structure_search_v1/protocol.json"
    )
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    root = Path(plan["output"])
    export = Path("experiments/random_structure_search_v1/results/search_20261103")
    export.mkdir(parents=True, exist_ok=True)
    assets = Path("final_mds/assets/2026-11-03/random_structure_search")
    assets.mkdir(parents=True, exist_ok=True)
    paths = [
        "protocol.json",
        "started.json",
        "frozen.json",
        "search/history.json",
        "search/complete.json",
        "search/shortlist.json",
        "search/shortlist.npz",
        "search/all_scores.json",
        "selection/selection.json",
        "selection/bank.npz",
        "prior/training.json",
        "prior/selection.json",
        "benchmark/manifest.json",
        "audit/summary.json",
        "audit/comparison.csv",
        "target/summary.json",
        "target/comparison.csv",
        "target/selected_indexators.npz",
    ]
    hashes = {}
    for relative in paths:
        source = root / relative
        destination = export / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    (export / "manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    history = json.loads((root / "search/history.json").read_text())
    summary = json.loads((root / "target/summary.json").read_text())
    selection = json.loads((root / "selection/selection.json").read_text())
    methods = list(summary["methods"])
    with (root / "target/comparison.csv").open() as stream:
        records = list(csv.DictReader(stream))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(13, 4.7), gridspec_kw={"width_ratios": [1, 1.6]}
    )
    for kind, label in [("mask", "Случайные маски"), ("coded", "Случайные общие коды")]:
        axes[0].plot(
            [row["generation"] for row in history],
            [row["kinds"][kind]["best"]["fitness"] for row in history],
            label=label,
        )
    axes[0].set(
        xlabel="Поколение",
        ylabel="Лучшая source-query BCE",
        title="Поиск по предсказательной ошибке",
    )
    axes[0].legend(frameon=False)
    order = [
        "dense",
        "random",
        "shuffled",
        "best",
        "bank_sample",
        "frequency",
        "ar",
        "mask_only",
    ]
    positions = np.arange(len(order))
    means = [100 * summary["methods"][method]["test_accuracy"] for method in order]
    axes[1].barh(
        positions,
        means,
        color=["#087e8b" if method == "ar" else "#91a9bd" for method in order],
    )
    for position, method in enumerate(order):
        values = [
            100 * float(row["test_accuracy"])
            for row in records
            if row["method"] == method
        ]
        axes[1].scatter(
            values, np.full(len(values), position), color="#26394a", alpha=0.5, s=11
        )
    axes[1].set(
        yticks=positions,
        yticklabels=[LABELS[m] for m in order],
        xlabel="Test accuracy, %",
        title=f"{summary['tasks']} новых задач; один банк",
    )
    axes[1].invert_yaxis()
    axes[1].set_xlim(min(means) - 3, max(means) + 2)
    figure.tight_layout()
    save_figure(figure, assets / "search_and_transfer")
    bank = np.load(root / "selection/bank.npz")
    selected = np.load(root / "target/selected_indexators.npz")
    ar_id = methods.index("ar")
    random_id = methods.index("random")
    ar = selected["indices"][
        np.flatnonzero((selected["method_ids"] == ar_id) & (selected["task_ids"] == 0))[
            0
        ]
    ]
    random = selected["indices"][
        np.flatnonzero(
            (selected["method_ids"] == random_id) & (selected["task_ids"] == 0)
        )[0]
    ]
    figure, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=True)
    matrices = [bank["best"] > 0, (bank["tokens"] > 0).mean(0), ar > 0, random > 0]
    titles = [
        "Выбранная структура",
        "Частота связи в банке",
        "AR: первая новая задача",
        "Случайный контроль",
    ]
    for axis, matrix, title in zip(axes, matrices, titles):
        axis.imshow(
            matrix, cmap="Greys", vmin=0, vmax=1, interpolation="nearest", aspect="auto"
        )
        axis.set(title=title, xlabel="Скрытый нейрон")
    axes[0].set_ylabel("Вход в наблюдаемом порядке")
    figure.tight_layout()
    save_figure(figure, assets / "obtained_masks")
    with gzip.open(root / "target/fit_history.json.gz", "rt") as stream:
        raw_curves = json.load(stream)
    rows = json.loads((root / "target/fit.json").read_text())["rows"]
    selected_rows = selected["fit_rows"]
    curves = {}
    for method in methods:
        ids = [int(i) for i in selected_rows if rows[int(i)]["method"] == method]
        curves[method] = [
            {
                "step": row["step"],
                "train_loss": float(np.array(row["train_loss"])[ids].mean()),
                "query_loss": float(np.array(row["query_loss"])[ids].mean()),
            }
            for row in raw_curves
        ]
    (export / "target/learning_curves.json").write_text(
        json.dumps(curves, indent=2) + "\n"
    )
    table = "\n".join(
        f"| {LABELS[method]} | {summary['methods'][method]['test_accuracy'] * 100:.2f} | {summary['methods'][method]['test_bce']:.4f} | {summary['methods'][method]['numerical_parameters']:.0f} |"
        for method in order
    )
    kind_label = (
        "маски с независимыми активными весами"
        if selection["kind"] == "mask"
        else "индексаторы с общими весами"
    )
    ar_result = summary["methods"]["ar"]
    delta_dense = 100 * (
        ar_result["test_accuracy"] - summary["methods"]["dense"]["test_accuracy"]
    )
    delta_random = 100 * (
        ar_result["test_accuracy"] - summary["methods"]["random"]["test_accuracy"]
    )
    unique = history[-1]["unique_evaluated"]
    best = bank["best"]
    report = f"""# Случайный поиск структуры без подсказок решателя — 3 ноября 2026

Проверяется получение полезного банка из случайных масок и индексаторов.
Начальная структура и мутации не используют веса, градиенты, группы входов,
роли координат, число фильтров или аналитическое решение. Градиентный метод
обучает только численные коэффициенты внутри предложенной структуры.

## Протокол и происхождение банка

Две популяции по 64 кандидата проходят 64 поколения после начальной
популяции. Случайно меняются связи и группы равенств весов; отбор видит
только query BCE после нового обучения чисел. Всего оценено **{unique}**
уникальных структур на восьми исходных задачах. Каждая оценка использует
1024 обновления Adam на задачу. Блоки и общие фильтры не конструируются.

Общие предположения процедуры — ReLU MLP с 64 скрытыми нейронами,
конечные диапазоны сложности и выбранные операторы поиска. Поэтому здесь
нет знания устройства конкретной задачи, но нет и утверждения об
отсутствии любых индуктивных предположений.

Две отдельные исходные задачи выбирают структуру, численные настройки и
температуру генерации. Выбраны **{kind_label}**; банк содержит
**{selection["bank_records"]} записей и {selection["bank_unique"]} уникальных структур**.
В одной выбранной структуре {np.count_nonzero(best)} активных связей из {best.size}.
MADE обучается на этом банке. Его validation отделяет кандидатов,
но родственные потомки эволюционного поиска могут находиться в обеих частях.

До создания 16 новых целевых задач зафиксированы SHA-256 банка,
генератора, настроек и исходного кода. На каждой новой задаче все числа
обучаются с нуля: 2048 support-примеров, восемь стартов по 4096 обновлений,
1024 query-примера и 16 384 независимых test-примера. Query выбирает старт
и checkpoint; test не выбирает структуру или настройки.

## Результат на новых задачах

| Метод | Test accuracy, % | Test BCE | Численных параметров, среднее |
|---|---:|---:|---:|
{table}

AR отличается от плотного Adam на **{delta_dense:+.2f} п.п.**, а от свежих
случайных структур той же сложности — на **{delta_random:+.2f} п.п.**
Случайный контроль сохраняет число связей и частоты кодов выбранной
структуры. Перестановка входов сохраняет её топологию и sharing,
но меняет соответствие наблюдаемым координатам.

![Поиск и перенос](assets/2026-11-03/random_structure_search/search_and_transfer.svg)

Слева показана ошибка, по которой шёл поиск; её снижение не является
независимым подтверждением. Справа — окончательный test на новых задачах.
Точки обозначают отдельные задачи. Это **один исходный банк**: 16 задач
не заменяют независимые повторы создания банка.

## Полученные структуры

![Найденные и сгенерированные маски](assets/2026-11-03/random_structure_search/obtained_masks.svg)

Входы показаны в наблюдаемом порядке, без перестановки по истинным блокам.
Чёрный цвет обозначает связь, серый на второй панели — её частоту в банке.
Для AR и случайного контроля показана первая целевая задача, без выбора
по качеству картинки. Полные индексаторы всех выбранных решений сохранены.
График показывает подключения; для масочной популяции каждый активный вес
независим, и одинаковый чёрный цвет не означает равенства коэффициентов.

## Ограничения и воспроизведение

Это ограниченный случайный поиск в большом пространстве. Его неудача не
доказывает невозможность обнаружения структуры; небольшой выигрыш не
доказывает недостижимость решения другими настройками градиентного метода.
Целевой численный бюджет уравнен. Поиск банка и обучение AR требуют
дополнительных исходных вычислений, не включённых в бюджет плотного контроля.

Предыдущее извлечение из весов остановлено после уточнения пользователя и
не используется для предложений этого опыта. Результаты специализированного
извлечения DSD с заданными блоками также не являются результатами этого поиска.

[Код и порядок запуска](../experiments/random_structure_search_v1/README.md),
[полные численные итоги](../experiments/random_structure_search_v1/results/search_20261103/target/summary.json),
[построчные результаты](../experiments/random_structure_search_v1/results/search_20261103/target/comparison.csv),
[банк](../experiments/random_structure_search_v1/results/search_20261103/selection/bank.npz),
[индексаторы целевых решений](../experiments/random_structure_search_v1/results/search_20261103/target/selected_indexators.npz),
[замороженные входы](../experiments/random_structure_search_v1/results/search_20261103/frozen.json).
"""
    Path("final_mds/RANDOM_STRUCTURE_SEARCH_2026-11-03.md").write_text(report)
    print(
        json.dumps(
            {
                "ar_minus_dense_pp": delta_dense,
                "ar_minus_random_pp": delta_random,
                "report": "final_mds/RANDOM_STRUCTURE_SEARCH_2026-11-03.md",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
