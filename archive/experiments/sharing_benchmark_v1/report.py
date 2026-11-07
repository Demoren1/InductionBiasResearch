"""Tables and scientific figures from frozen post-search audit artifacts."""

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

from experiments.darts_structured_tasks_v1.data import save_json

LABELS = {
    "consensus_dp": "Наш: consensus + exact DP",
    "lc": "LC, Lloyd",
    "lc_matched": "Наш график + Lloyd (абляция)",
    "lutq": "LUT-Q",
    "yeh": "Yeh: банк, argmax",
    "yeh_soft": "Yeh: банк, soft",
    "yeh_shared_soft": "Yeh: общий P, soft",
    "yeh_shared_map": "Yeh: общий P, argmax",
    "msr": "MSR, непрерывный U",
    "mtsr": "MTSR, непрерывный U",
    "msr_cluster": "MSR + кластеризация U",
    "mtsr_cluster": "MTSR + кластеризация U",
    "yeh_shared_soft_cluster": "Yeh soft + кластеризация U",
    "posthoc_dp": "Dense → exact DP",
    "posthoc_lloyd": "Dense → Lloyd",
    "shuffle_sharing": "Перемешать группы в найденной маске",
    "shuffle_all": "Перемешать весь найденный I",
    "oracle": "Истинный I (контроль)",
    "dense_ols": "Плотный МНК",
    "dense_ridge": "Плотный ridge",
}
ORDER = list(LABELS)
LABELS["dense_adam"] = "Плотный Adam (иллюстрация)"


def number(value):
    return f"{value:.3g}"


def aggregate(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["method"]].append(row)
    summaries = {}
    for method, rows in grouped.items():
        families = sorted({row["family"] for row in rows})
        result = {"method": method, "replicas": len(rows), "families": len(families)}
        if "recovery" in rows[0]:
            result["exact_cells"] = sum(row["recovery"]["exact_I"] for row in rows)
            result["exact_all_seeds_families"] = sum(
                all(
                    row["recovery"]["exact_I"]
                    for row in rows
                    if row["family"] == family
                )
                for family in families
            )
            for key in ("mask_f1", "categorical_errors", "mask_errors"):
                result[key] = float(np.mean([row["recovery"][key] for row in rows]))
        if "subspace_error" in rows[0]:
            result["subspace_error"] = float(
                np.mean([row["subspace_error"] for row in rows])
            )
            result["representation_floor"] = float(
                np.mean([row["transfer"]["representation_floor"] for row in rows])
            )
        result["transfer"] = {}
        for size in ("1", "4", "16"):
            values = [
                np.mean(
                    [
                        row["transfer"][size]["population_mse"]
                        for row in rows
                        if row["family"] == family
                    ]
                )
                for family in families
            ]
            result["transfer"][size] = {
                "population_mse": float(np.mean(values)),
                "family_standard_error": float(
                    np.std(values, ddof=1) / np.sqrt(len(values))
                )
                if len(values) > 1
                else None,
                "test_mse": float(
                    np.mean([row["transfer"][size]["test_mse"] for row in rows])
                ),
                "by_family": dict(zip(map(str, families), map(float, values))),
            }
        summaries[method] = result
    return summaries


def table(summary):
    lines = [
        "| Метод | Точный I: все сиды семейства | Точных запусков | F1 маски | MSE, 1 пример | MSE, 4 примера | MSE, 16 примеров |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ORDER:
        if method not in summary:
            continue
        row = summary[method]
        exact = (
            f"{row['exact_all_seeds_families']}/{row['families']}"
            if "exact_cells" in row
            else "—"
        )
        cells = (
            f"{row['exact_cells']}/{row['replicas']}" if "exact_cells" in row else "—"
        )
        f1 = number(row["mask_f1"]) if "mask_f1" in row else "—"
        errors = " | ".join(
            number(row["transfer"][size]["population_mse"]) for size in ("1", "4", "16")
        )
        lines.append(f"| {LABELS[method]} | {exact} | {cells} | {f1} | {errors} |")
    return "\n".join(lines)


def save_figure(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def structure_figures(root, output, profile, family):
    folder = root / profile / f"family_{family}"
    spec = json.loads((folder / "specification.json").read_text())
    with np.load(folder / "teacher_audit_only.npz") as archive:
        truth = archive["indices"]
    methods = [
        "oracle",
        "consensus_dp",
        "lc_matched",
        "lutq",
        "yeh_shared_map",
        "msr_cluster",
        "mtsr_cluster",
        "yeh_shared_soft_cluster",
    ]
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
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    raw = {"truth": truth.tolist()}
    records = json.loads((folder / "audit/summary.json").read_text())["records"]
    for axis, method in zip(axes.flat, methods):
        seed = 0 if method == "oracle" else 11
        with np.load(folder / "audit" / f"{method}_seed_{seed}.npz") as archive:
            index = archive["aligned_indices"]
            raw[method] = {
                "indices": archive["indices"].tolist(),
                "aligned_indices": index.tolist(),
            }
        row = next(
            row for row in records if row["method"] == method and row["seed"] == seed
        )
        image = axis.imshow(
            index, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto"
        )
        axis.set_title(
            f"{LABELS[method]}\nошибок I: {row['recovery']['categorical_errors']}",
            fontsize=10,
        )
        axis.set_xlabel("Входная координата")
        axis.set_ylabel("Выходная координата")
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        ticks=np.arange(-3, 4),
        shrink=0.8,
        label="Индекс после допустимого переименования групп",
    )
    fig.suptitle(
        f"{profile}, family {family}, seed 11: выбранные индексаторы", fontsize=13
    )
    save_figure(fig, output, f"{profile}_indexators")
    save_json(output / f"{profile}_indexators.json", raw)
    methods = [
        "oracle",
        "consensus_dp",
        "dense_adam",
        "lutq",
        "yeh_shared_map",
        "msr",
        "mtsr",
        "dense_ols",
    ]
    with np.load(folder / "target_after_freeze.npz") as archive:
        actual = archive["teacher_weights_audit_only"][0]
    limit = np.abs(actual).max()
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for axis, method in zip(axes.flat, methods):
        seed = 0 if method in ("oracle", "dense_ols", "dense_adam") else 11
        with np.load(folder / "audit" / f"{method}_seed_{seed}.npz") as archive:
            weights = archive["weights_support_4"]
        error = np.square(weights - actual).sum() / spec["outputs"]
        image = axis.imshow(
            weights,
            vmin=-limit,
            vmax=limit,
            cmap="RdBu_r",
            aspect="auto",
            interpolation="nearest",
        )
        axis.set_title(f"{LABELS[method]}\nMSE матрицы: {error:.3g}", fontsize=10)
        axis.set_xlabel("Входная координата")
        axis.set_ylabel("Выходная координата")
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.8,
        label="Вес после обучения новых чисел",
    )
    fig.suptitle(
        f"{profile}: первая новая задача, 4 support-примера; числа source-задач отброшены"
    )
    save_figure(fig, output, f"{profile}_weights")


def transfer_figure(summaries, output):
    methods = [
        "consensus_dp",
        "lc",
        "lc_matched",
        "lutq",
        "yeh_shared_map",
        "msr",
        "mtsr",
        "msr_cluster",
        "mtsr_cluster",
        "oracle",
        "dense_ols",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, profile in zip(axes, ("hidden", "native")):
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#000000",
        ]
        for method, color in zip(methods, colors):
            row = summaries[profile][method]
            axis.plot(
                [1, 4, 16],
                [
                    max(row["transfer"][size]["population_mse"], 1e-15)
                    for size in ("1", "4", "16")
                ],
                "o",
                label=LABELS[method],
                markersize=4,
                color=color,
                linestyle="--" if method == "dense_ols" else "-",
            )
        axis.set(
            xscale="log",
            yscale="log",
            xlabel="Число support-примеров новой задачи",
            ylabel="Population MSE",
            title=profile,
        )
        axis.set_xticks([1, 4, 16], [1, 4, 16])
        axis.grid(alpha=0.25)
        if profile == "native":
            axis.text(
                0.04,
                0.08,
                "MSE < 1e−15 показана на уровне 1e−15",
                transform=axis.transAxes,
                fontsize=8,
            )
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, output, "transfer")


def selection_figure(root, output):
    folder = root / "hidden/family_201"
    methods = ["consensus_dp", "lc_matched", "lutq", "msr", "mtsr", "yeh_shared_soft"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    choices = json.loads((root / "hidden_shared_models_frozen.json").read_text())
    for axis, method in zip(axes.flat, methods):
        shared = method.startswith("yeh_shared")
        path = (
            folder
            / ("shared_confirmation" if shared else "confirmation")
            / f"{method}_seed_11.json"
        )
        state = json.loads(path.read_text())
        row = choices["row_by_method"][method] if shared else 0
        history = state["history"]
        x = [point["step"] for point in history]
        for key, label in (
            ("source_query_mse", "Source query"),
            ("selection_mse", "Новые validation-задачи"),
        ):
            axis.plot(
                x,
                [max(point[key][row], 1e-10) for point in history],
                "o-",
                markersize=3,
                label=label,
            )
        axis.set(
            title=LABELS[method],
            yscale="log",
            xlabel="Шаг соответствующего метода",
            ylabel="MSE",
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(
        "Контроль выбора checkpoint: source и независимые задачи с новыми коэффициентами"
    )
    save_figure(fig, output, "selection_curves")


def costs(root, output):
    result = defaultdict(lambda: {"pilot_seconds": 0.0, "confirmation_seconds": 0.0})
    archived = 0.0
    for profile in ("hidden", "native"):
        for path in (root / profile).glob("family_*/*/*.json"):
            state = json.loads(path.read_text())
            if "seconds" not in state or "inputs" not in state:
                continue
            phase = path.parent.name
            if not any(
                word in phase for word in ("pilot", "calibration", "confirmation")
            ):
                continue
            method = path.stem.split("_seed_")[0]
            key = "confirmation_seconds" if "confirmation" in phase else "pilot_seconds"
            result[method][key] += state["seconds"]
    for archive in ("superseded_hard_only", "shared_sparse_initialization"):
        for path in (root / archive).rglob("*.json"):
            state = json.loads(path.read_text())
            if "seconds" in state and "inputs" in state:
                archived += state["seconds"]
    save_json(
        output / "costs.json",
        {"methods": dict(result), "archived_completed_seconds": archived},
    )
    lines = [
        "## Фактическая стоимость запуска",
        "",
        "Сумма измеренного wall-time обучающих вызовов, секунды. Конфигурации в pilot обучались батчами; это не GPU-hours из счётчика устройства и не нормировка по FLOPs. Мягкий и MAP результат одного общего обучения учитываются один раз. Разные методы могли работать одновременно на разных GPU.",
        "",
        "| Метод | Pilot/calibration, с | Confirmation, с |",
        "|---|---:|---:|",
    ]
    for method in ORDER:
        if method in result:
            row = result[method]
            lines.append(
                f"| {LABELS[method]} | {row['pilot_seconds']:.1f} | {row['confirmation_seconds']:.1f} |"
            )
    lines.extend(
        [
            "",
            f"Дополнительно завершённые архивные попытки: {archived:.1f} с. Незавершённые прерванные попытки не имеют полного таймера; журнал сохранён. Время загрузки, подготовки данных, аудита и построения графиков сюда не включено.",
        ]
    )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "protocol.json").read_text())
    records, summaries = [], {}
    for profile, families in (
        ("hidden", plan["confirmation_families"]),
        ("native", [301]),
    ):
        rows = []
        for family in families:
            rows.extend(
                json.loads(
                    (root / profile / f"family_{family}/audit/summary.json").read_text()
                )["records"]
            )
        summaries[profile] = aggregate(rows)
        records.extend(rows)
    output = root / "report"
    output.mkdir(exist_ok=True)
    save_json(output / "summary.json", summaries)
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "profile",
                "method",
                "exact_cells",
                "replicas",
                "exact_all_seeds_families",
                "families",
                "mask_f1",
                "subspace_error",
                "mse_n1",
                "mse_n4",
                "mse_n16",
            ]
        )
        for profile, summary in summaries.items():
            for method in ORDER:
                row = summary[method]
                writer.writerow(
                    [
                        profile,
                        method,
                        row.get("exact_cells"),
                        row["replicas"],
                        row.get("exact_all_seeds_families"),
                        row["families"],
                        row.get("mask_f1"),
                        row.get("subspace_error"),
                        *[
                            row["transfer"][size]["population_mse"]
                            for size in ("1", "4", "16")
                        ],
                    ]
                )
    lines = [
        "# Проверка восстановления шэринга: 5 ноября 2026",
        "",
        f"Главный результат: прежний поиск через банк consensus+DP не является лидером. В сложном hidden-режиме он точно восстановил I в {summaries['hidden']['consensus_dp']['exact_all_seeds_families']}/6 семействах, общий Yeh с argmax и MSR с кластеризацией — в {summaries['hidden']['msr_cluster']['exact_all_seeds_families']}/6. В более обеспеченной данными native-постановке наш метод восстановил I в {summaries['native']['consensus_dp']['exact_cells']}/3 запусках, но тот же результат дают MSR/MTSR с кластеризацией. Превосходство и алгоритмическая новизна нашего варианта этими данными не подтверждены.",
        "",
        "Это результаты локальных реализаций и явно описанных адаптаций методов к единому линейному benchmark. Это не воспроизведение опубликованных чисел статей. Гиперпараметры и checkpoint выбирались по отдельным задачам с новыми коэффициентами; test-задачи созданы после фиксации всех моделей. Истинные маски, группы и перестановки при обучении не использовались.",
        "",
        "## Скрытая свёртка с шумом",
        "",
        "Шесть новых независимых перестановок входов/выходов, по три seed оптимизации. Матрица 14×16, три обучаемых числа, 64 source-задачи по 4 support + 4 query, шум σ=0.1. Для выбора checkpoint — ещё 24 независимые задачи, для итогового переноса — 100 новых задач в каждом семействе. При переносе все source-числа и состояния оптимизатора отбрасываются; новые числа находятся точным МНК. MSE измерена аналитически по гауссовскому входу и усреднена сначала по seed, затем по семействам. Эмпирическая test-MSE и ошибки по семействам сохранены в summary.json.",
        "",
        table(summaries["hidden"]),
        "",
        "## Постановка rank1 из MSR",
        "",
        "Размеры и распределение из авторского генератора: 70→68, ширина ядра 3, 400 source-задач, по 20 наблюдений, без шума. Одна геометрия, три seed; это не три независимых семейства. Наше разбиение support/query и выбор гиперпараметров зафиксированы отдельно от оригинального кода.",
        "",
        table(summaries["native"]),
        "",
        "Причина различий, согласующаяся с результатами: в hidden-режиме отдельная source-функция даёт только восемь наблюдений на шестнадцать входных координат. Поиск предложений сначала решает недоопределённую задачу для отдельной функции и может закрепить лишние подключения. Методы общего U/P используют все независимо меняющиеся коэффициенты для обучения одной структуры. Это объяснение является интерпретацией, а не отдельной причинной абляцией: native и hidden различаются также размерностью и шумом.",
        "",
        "Native Yeh-argmax имеет F1 маски = 1, но неверный полный I и высокую ошибку переноса. Его мягкое представление и последующая кластеризация строк работают хорошо. Значит, здесь наблюдается именно ошибка извлечения групп из мягкого представления; одной правильной маски недостаточно. На hidden после validation-подобранного охлаждения argmax, напротив, восстанавливает все группы. Это не универсальный запрет мягкого поиска.",
        "",
        "Большая ошибка плотного МНК при 16 шумных наблюдениях и 16 входах связана с плохо обусловленной интерполяцией: обращение почти вырожденной квадратной Gaussian design-матрицы усиливает шум. Это точное выпуклое решение, а не локальный минимум Adam. Ridge даёт меньшую ошибку; его единственный коэффициент выбран на pilot-selection support данного профиля, без отдельного test-подбора для каждого support-size.",
        "",
        "## Что считается правильной схемой",
        "",
        "Точное совпадение каждой позиции индексатора после единого допустимого переименования групп и переворота знака каждого кода. Ноль фиксирован. F1 маски проверяется отдельно: правильная маска ещё не означает правильный шэринг. Для непрерывных MSR/MTSR проверяется расстояние между подпространствами матриц, поскольку произвольный обратимый поворот латентных координат не меняет представляемое семейство. Их U не интерпретируется как готовый дискретный I. Отдельные строки с кластеризацией действительно возвращают I и проходят ту же точную проверку.",
        "",
        "Колонка «все сиды семейства» требует точного I во всех трёх запусках данной геометрии. Линейная версия нашего exact-DP метода детерминирована при фиксированных наблюдениях: его одинаковые повторы не являются независимыми подтверждениями. Для оценки устойчивости основная единица — семейство.",
        "",
        "## Сопоставимость реализаций",
        "",
        "- Наш метод: непрерывная Z, точное квадратичное обновление Z, scaled dual, проекция на фиксированный ноль и K значений с привязанными отрицательными двойниками через точное скалярное DP. Z между шагами сохраняется. Это линейная адаптация ранее зафиксированного варианта, а не новый запуск исходной нелинейной DeepSets.",
        "- LC: та же словарная область и сохранение непрерывной Z, но Lloyd-проекция и собственный подбор графика штрафа. «Наш график + Lloyd» использует ровно нашу сетку штрафов/разреженности и заменяет только проекцию.",
        "- LUT-Q: hard forward, STE для непрерывной матрицы, один assignment/center шаг за итерацию; метки остаются прикреплены к обновлённым центрам. Фиксированный ноль и привязанные знаки — наши явно заданные ограничения.",
        "- Все алгоритмы банка формируют восемь предложений из отдельных source-функций и выбирают один общий I по всему source-набору. MSR/MTSR и «Yeh: общий P» сразу обучают общее представление по всем source-функциям. Наблюдения доступны одинаковые, но способ их использования и вычислительные затраты различаются; равенство FLOPs не заявляется.",
        "- Yeh: категориальная матрица A, дифференцируемый inner-fit чисел, query-loss, entropy/nuclear penalties. Сохранены и мягкое представление, и argmax. Сильный вариант обучает один общий P на всех source-задачах. По валидации до открытия teacher/test его сетка расширена с 12 до 36 настроек: uniform/zero-biased initialization и охлаждение температуры. Первые попытки сохранены в shared_sparse_initialization и учитываются как дополнительная стоимость подбора.",
        "- MSR: общий вещественный U, три шага SGD во внутреннем обучении, обучаемая инициализация и learning rate; MTSR: общий U и отдельные числа source-задач. Основная таблица намеренно использует одинаковый fresh exact fit чисел для всех, а девять SGD-шагов из learned initialization для MSR сохранены отдельной диагностикой в audit/summary.json.",
        "- MSR/MTSR/Yeh-soft + кластеризация: из найденного U берётся polar factor для устранения произвольного выбора латентных координат, затем signed row-vector Lloyd с фиксированным нулём, K центрами и восемью стартами. Старт выбирается по source-query, штраф — по независимым pilot-задачам. Это отдельный контроль постобработки, не утверждение об алгоритме оригинальной статьи.",
        "- Dense → DP/Lloyd — постобработка обычного плотного решения, без consensus-оптимизации. Плотные МНК/ridge — численные контроли без поиска шэринга; они решают выпуклую задачу и исключают объяснение результата неудачей Adam.",
        "- По прежнему запросу пользователя в weight-heatmap добавлены реальные веса плотного Adam на первой новой задаче: нулевая инициализация, lr=0.03, 2000 шагов на четырёх support-примерах. Это иллюстрация после основного аудита с фиксированными настройками, без выбора по query/test; она не включена в таблицу конкурентов. Численный контроль в таблице — точный МНК/ridge.",
        "",
        "## Ограничения вывода",
        "",
        "Точное восстановление в этих семействах не доказывает глобальную сходимость на произвольных нейросетях. На большом benchmark известный teacher даёт проверяемый эталон структуры; строгое утверждение о глобальном минимуме делается только для отдельно полностью перебранного tiny-набора. Проверка изображений, нелинейного DeepSets и суммы синусоид в это сравнение не входит. По этим результатам нельзя объявлять новизну consensus/ADMM, ближайшего кода, скалярного DP или самого переноса общего представления.",
        "",
        "## Артефакты",
        "",
        "- [Выбранные индексаторы, hidden](hidden_indexators.png), [исходные метки и допустимое выравнивание](hidden_indexators.json).",
        "- [Матрицы после fresh fit, hidden](hidden_weights.png).",
        "- [Выбранные индексаторы, native](native_indexators.png), [матрицы native](native_weights.png).",
        "- [Перенос в зависимости от числа примеров](transfer.png), [source и validation кривые](selection_curves.png).",
        "- [Машиночитаемая таблица](comparison.csv), [полная агрегация](summary.json).",
        "",
        "## Первоисточники",
        "",
        "[Yeh et al., 2022](https://proceedings.mlr.press/v151/yeh22b/yeh22b.pdf); [MSR/MTSR](https://arxiv.org/abs/2007.02933), [авторский код](https://github.com/AllanYangZhou/metalearning-symmetries); [LC](https://arxiv.org/abs/1707.04319); [LUT-Q](https://arxiv.org/abs/1911.04951). Неизменённые исходники, использованные для проверки механики, лежат в references/ рядом с кодом benchmark.",
    ]
    tiny_path = root / "tiny/family_401/global_enumeration.json"
    if tiny_path.exists():
        tiny = json.loads(tiny_path.read_text())
        lines.extend(
            [
                "",
                "## Полный перебор tiny-задачи",
                "",
                f"Матрица 2×4, не более двух групп и фиксированный ноль; перебраны все {tiny['assignments']} канонических signed-индексаторов. Глобальный минимум validation bilevel-objective: {tiny['global_objective']:.6g}. Все настройки перенесены из hidden-подбора, до перебора ничего не перенастраивалось.",
                "",
                "| Метод | Gap к дискретному глобальному минимуму | Совпадает I глобального решения |",
                "|---|---:|---:|",
            ]
        )
        for method in ORDER:
            if method not in tiny["results"]:
                continue
            row = tiny["results"][method]
            if row["discrete"]:
                exact = "да" if row["agreement_with_global"]["exact_I"] else "нет"
                lines.append(
                    f"| {LABELS[method]} | {row['gap_to_discrete_global']:.3g} | {exact} |"
                )
        lines.extend(
            [
                "",
                "Сертификат относится к указанному конечному множеству структур и точному inner-fit на данной tiny-задаче. Непрерывные U в эту дискретную область не входят.",
            ]
        )
    lines.extend(["", *costs(root, output)])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    structure_figures(root, output, "hidden", 201)
    structure_figures(root, output, "native", 301)
    transfer_figure(summaries, output)
    selection_figure(root, output)
    print(output / "REPORT.md", flush=True)
    print(table(summaries["hidden"]), flush=True)


if __name__ == "__main__":
    main()
