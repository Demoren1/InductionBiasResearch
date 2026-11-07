"""Figures and tables for the frozen G27 DeepSets cohort."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.optimize import linear_sum_assignment

from experiments.darts_indexator_v1.data import digest, load_archive, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .deepsets_prefix import METHODS
from .pattern_prefix import timestamp

LABELS = {
    "bootstrap": "Первичный prefix",
    "single_fixed": "Калибровка + prefix",
    "oracle": "Истинный I",
    "dense_adam": "Плотный Adam",
    "random": "Случайный I",
}


def aggregate(plan, structure, rows):
    result = []
    for condition, sigma in enumerate(plan["noise"]):
        for method in (*METHODS, "oracle", "dense_adam", "random"):
            values = [
                row
                for row in rows
                if row["condition"] == condition and row["method"] == method
            ]
            if not values:
                continue
            exact = [
                row
                for row in structure
                if row.get("method") == method and row["condition"] == condition
            ]
            reference = {
                (row["family"], row["task"]): row["mse"]
                for row in rows
                if row["condition"] == condition and row["method"] == "oracle"
            }
            result.append(
                {
                    "condition": condition,
                    "noise": sigma,
                    "method": method,
                    "tasks": len(values),
                    "exact_geometries": sum(row["exact"] for row in exact)
                    if exact
                    else None,
                    "mean_mse": float(np.mean([row["mse"] for row in values])),
                    "mean_nmse": float(np.mean([row["nmse"] for row in values])),
                    "median_nmse": float(np.median([row["nmse"] for row in values])),
                    "worst_nmse": max(row["nmse"] for row in values),
                    "mean_excess_mse_to_true_I": float(
                        np.mean(
                            [
                                row["mse"] - reference[(row["family"], row["task"])]
                                for row in values
                            ]
                        )
                    ),
                    "mean_permutation_normalized_rms": float(
                        np.mean([row["permutation_normalized_rms"] for row in values])
                    ),
                    "max_permutation_rms": max(
                        row["permutation_rms"] for row in values
                    ),
                }
            )
    return result


def heatmaps(root, plan):
    aligned = load_archive(root / "aligned_indices.npz")
    condition = len(plan["noise"]) - 1
    key = f"0_{condition}_single_fixed"
    if key not in aligned:
        return
    teacher = load_archive(root / "private/teacher_0.npz")
    private = load_archive(root / "private/target_0.npz")
    fit = load_archive(root / "target/family_0" / f"codes_{condition}.npz")
    metadata = json.loads(
        (root / "target/family_0" / f"codes_{condition}.json").read_text()
    )
    number = metadata["origins"]["single_fixed"]
    index, codes = fit["indices"][number], fit["codes"][number, 0]
    truth = teacher["index"]
    _, _, columns = align_indices(index, truth, 4)
    numeric = np.r_[0.0, codes][np.abs(index)] * np.sign(index)
    numeric = numeric[:, columns]
    dense = load_archive(root / "target/family_0/dense.npz")
    winner = dense["winners"][condition * plan["targets"]]
    dense_weight = dense["weights"][winner].astype(float)
    true_weight = np.r_[0.0, private["codes"][0]][np.abs(truth)] * np.sign(truth)
    cost = ((dense_weight[:, :, None] - true_weight[:, None, :]) ** 2).sum(0)
    source, target = linear_sum_assignment(cost)
    dense_weight = dense_weight[:, source[np.argsort(target)]]
    # The input order is restored ONLY for this post-freeze visualization.
    rows = teacher["permutation"]
    col_order = sorted(
        range(16),
        key=lambda j: (int(np.flatnonzero(truth[rows, j])[0]), tuple(truth[rows, j])),
    )
    values = [
        truth[rows][:, col_order],
        aligned[key][rows][:, col_order],
        numeric[rows][:, col_order],
        dense_weight[rows][:, col_order],
    ]
    titles = [
        "Истинный I",
        f"Найденный I: сырой шум {plan['noise'][condition]:g}*",
        "Свежий кодбук: W",
        "Плотный Adam: W",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 6.3), layout="constrained")
    cmap = ListedColormap(
        [
            "#1e466b",
            "#427a9b",
            "#89b5c8",
            "#bcd5de",
            "#f6f7f8",
            "#e9d5b7",
            "#dfb27b",
            "#cb8547",
            "#9f5129",
        ]
    )
    norm = BoundaryNorm(np.arange(-4.5, 5), cmap.N)
    limit = max(float(np.abs(values[2]).max()), float(np.abs(values[3]).max()))
    for number, (ax, value, title) in enumerate(zip(axes, values, titles, strict=True)):
        artist = ax.imshow(
            value,
            cmap=cmap if number < 2 else "RdBu_r",
            norm=norm if number < 2 else None,
            vmin=None if number < 2 else -limit,
            vmax=None if number < 2 else limit,
        )
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(16), range(1, 17), fontsize=7)
        ax.set_yticks(range(16), range(1, 17), fontsize=7)
        ax.set_xlabel("Скрытый нейрон")
        if number == 0:
            ax.set_ylabel("Координата точки")
        for coordinate in np.arange(1.5, 16, 2):
            ax.axhline(coordinate, color="#757575", linewidth=0.6)
            ax.axvline(coordinate, color="#757575", linewidth=0.6)
        for i in range(16):
            for j in range(16):
                if number < 2 and value[i, j] == 0:
                    continue
                label = str(int(value[i, j])) if number < 2 else f"{value[i, j]:.1f}"
                dark = abs(value[i, j]) >= (3.5 if number < 2 else 0.65 * limit)
                ax.text(
                    j,
                    i,
                    label,
                    color="white" if dark else "#20252c",
                    ha="center",
                    va="center",
                    fontsize=5.6,
                )
        if number in (1, 3):
            fig.colorbar(
                artist,
                ax=axes[:2] if number == 1 else axes[2:],
                shrink=0.75,
                label="Индекс" if number == 1 else "Значение веса",
                ticks=list(range(-4, 5)) if number == 1 else None,
            )
    fig.suptitle(
        "DeepSets: общие фильтры для восьми точек восстановлены из полной матрицы 16×16",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.005,
        "* Шум усреднён. Порядок строк и столбцов раскрыт только для рисунка; в поиск он не передавался.",
        ha="center",
        fontsize=10,
    )
    output = root / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "indexators_and_weights.png", dpi=190, bbox_inches="tight")
    fig.savefig(output / "indexators_and_weights.pdf", bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(
        output / "heatmap_values.npz",
        truth=values[0],
        selected=values[1],
        fresh_weights=values[2],
        dense_weights=values[3],
        display_input_order=rows,
        display_column_order=np.array(col_order),
    )


def plots(root, plan, rows):
    output = root / "figures"
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    rng = np.random.default_rng(27017)
    colors = {"single_fixed": "#176b87", "dense_adam": "#d08335", "random": "#7e8791"}
    for method, color in colors.items():
        for column, condition in enumerate((1, 2)):
            values = [
                row["nmse"]
                for row in rows
                if row["method"] == method and row["condition"] == condition
            ]
            location = column * 4 + list(colors).index(method)
            axes[0].scatter(
                location + rng.uniform(-0.15, 0.15, len(values)),
                values,
                color=color,
                s=15,
                alpha=0.6,
                label=LABELS[method] if column == 0 else None,
            )
            axes[0].scatter(
                [location],
                [np.mean(values)],
                marker="_",
                color="black",
                s=160,
                zorder=5,
            )
    axes[0].set_xticks([1, 5], ["Сырой шум 0,2*", "Сырой шум 2*"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Clean target NMSE")
    axes[0].set_title("32 новые функции на условие; чёрная черта — среднее")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.2)
    dense_values = [
        row["permutation_normalized_rms"]
        for row in rows
        if row["method"] == "dense_adam" and row["condition"] == 2
    ]
    prefix_values = [
        row["permutation_normalized_rms"]
        for row in rows
        if row["method"] == "single_fixed" and row["condition"] == 2
    ]
    axes[1].bar(
        [0, 1],
        [np.mean(prefix_values), np.mean(dense_values)],
        color=["#176b87", "#d08335"],
    )
    for position, value in enumerate((np.mean(prefix_values), np.mean(dense_values))):
        axes[1].text(position, value, f"{value:.2e}", ha="center", va="bottom")
    axes[1].set_xticks([0, 1], ["Найденный I", "Плотный Adam"])
    axes[1].set_ylabel("RMS по перестановкам / target std")
    axes[1].set_title("Перестановка целых точек; шум 2*")
    fig.savefig(output / "transfer_and_invariance.png", dpi=175, bbox_inches="tight")
    fig.savefig(output / "transfer_and_invariance.pdf", bbox_inches="tight")
    plt.close(fig)

    condition = len(plan["noise"]) - 1
    folder = root / "target/family_0"
    record = json.loads((folder / f"codes_{condition}.json").read_text())
    if "single_fixed" not in record["origins"]:
        return
    history = record["histories"][record["origins"]["single_fixed"]][0]
    start = history["selected_start"]
    dense = load_archive(folder / "dense.npz")
    winner = int(dense["winners"][condition * plan["targets"]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].plot(
        [h["step"] for h in history["history"]],
        [h["train_mse"][start] for h in history["history"]],
        label="Support",
    )
    axes[0].plot(
        [h["step"] for h in history["history"]],
        [h["validation_mse"][start] for h in history["history"]],
        linestyle="--",
        label="Validation",
    )
    axes[0].set_xlabel("Шаг Gauss–Newton по четырём C")
    axes[0].set_title("Найденный I, свежий кодбук")
    axes[1].plot(
        dense["history_steps"],
        dense["train_history"][:, winner],
        label="Support (1024 входа)",
    )
    axes[1].plot(
        dense["history_steps"],
        dense["validation_history"][:, winner],
        linestyle="--",
        label="Validation",
    )
    axes[1].set_xlabel("Шаг Adam по 256 весам")
    axes[1].set_title("Плотный контроль, выбранный по validation запуск")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel("Наблюдаемая MSE, включая шум среднего")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(
        "Одна и та же новая функция: family 0, target 0, сырой шум 2 с усреднением"
    )
    fig.savefig(output / "target_learning.png", dpi=175, bbox_inches="tight")
    fig.savefig(output / "target_learning.pdf", bbox_inches="tight")
    plt.close(fig)


def failure_figure(root):
    if not (root / "failure_diagnosis.json").exists():
        return
    aligned = load_archive(root / "aligned_indices.npz")
    teacher = load_archive(root / "private/teacher_2.npz")
    truth = teacher["index"]
    order = teacher["permutation"]
    columns = sorted(
        range(16),
        key=lambda j: (
            int(np.flatnonzero(truth[order, j])[0]),
            tuple(truth[order, j]),
        ),
    )
    values = [
        truth[order][:, columns],
        aligned["2_2_bootstrap"][order][:, columns],
        aligned["2_2_single_fixed"][order][:, columns],
    ]
    cmap = ListedColormap(
        [
            "#1e466b",
            "#427a9b",
            "#89b5c8",
            "#bcd5de",
            "#f6f7f8",
            "#e9d5b7",
            "#dfb27b",
            "#cb8547",
            "#9f5129",
        ]
    )
    norm = BoundaryNorm(np.arange(-4.5, 5), cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.6), layout="constrained")
    titles = [
        "Истинный I: четыре группы",
        "Bootstrap: три группы, 8 ошибок",
        "После калибровки: три группы, 11 ошибок",
    ]
    for ax, value, title in zip(axes, values, titles, strict=True):
        artist = ax.imshow(value, cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Скрытый нейрон")
        ax.set_xticks(range(16), range(1, 17), fontsize=7)
        ax.set_yticks(range(16), range(1, 17), fontsize=7)
        for i in range(16):
            for j in range(16):
                if value[i, j] == 0:
                    continue
                ax.text(
                    j, i, str(int(value[i, j])), ha="center", va="center", fontsize=7
                )
        for coordinate in np.arange(1.5, 16, 2):
            ax.axhline(coordinate, color="#757575", linewidth=0.6)
            ax.axvline(coordinate, color="#757575", linewidth=0.6)
    axes[0].set_ylabel("Координата точки")
    fig.colorbar(artist, ax=axes, shrink=0.8, ticks=range(-4, 5), label="Индекс")
    fig.suptitle(
        "Неудачный случай: family 2, сырой шум 2 с усреднением; маска остаётся точной"
    )
    fig.savefig(root / "figures/failure_sharing.png", dpi=180, bbox_inches="tight")
    fig.savefig(root / "figures/failure_sharing.pdf", bbox_inches="tight")
    plt.close(fig)


def report(root, plan, structure, evaluation, aggregate_rows):
    with (root / "target_metrics.csv").open("w", newline="") as output:
        fields = sorted({key for row in evaluation["rows"] for key in row})
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(evaluation["rows"])
    with (root / "comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    recovery = []
    for method in METHODS:
        counts = [
            sum(
                row.get("method") == method
                and row["condition"] == condition
                and row["exact"]
                for row in structure
            )
            for condition in range(3)
        ]
        recovery.append(
            f"| {LABELS[method]} | {counts[0]}/4 | {counts[1]}/4 | {counts[2]}/4 |"
        )
    comparison = []
    for row in aggregate_rows:
        comparison.append(
            f"| {row['noise']:g} | {LABELS[row['method']]} | {row['tasks']}/32 | {row['mean_mse']:.6g} | {row['mean_nmse']:.6g} | {row['worst_nmse']:.6g} | {row['mean_permutation_normalized_rms']:.3g} |"
        )
    trajectories = {}
    for stage, filename in (
        ("bootstrap", "bootstrap_search.json"),
        ("repeat", "repeat_search.json"),
    ):
        records = [
            r
            for p in (root / "source").glob(f"case_*/{filename}")
            for r in json.loads(p.read_text())
        ]
        trajectories[stage] = {
            "total": len(records),
            "complete": sum(r["complete"] for r in records),
        }
    axis = 2 * plan["inputs"] * (2 * plan["axis_points"] - 1)
    prefix = 2 * plan["inputs"] * (plan["slice_points"] + plan["prefix_validation"])
    source_budget = (
        axis * plan["axis_repeats"]
        + prefix * plan["slice_repeats"]
        + (2 * plan["fit_validation"] + plan["meta_points"]) * plan["fit_repeats"]
        + plan["zero_repeats"]
    )
    target_budget = (plan["target_support"] + plan["target_validation"]) * plan[
        "target_repeats"
    ] + plan["target_zero_repeats"]
    summary = {
        "completed_utc": timestamp(),
        "geometries": plan["families"],
        "distinct_target_functions": evaluation["target_functions"],
        "conditions_paired": True,
        "comparison": aggregate_rows,
        "trajectories": trajectories,
        "noisy_labels_per_source_function_condition": source_budget,
        "noisy_labels_per_target_function_condition": target_budget,
    }
    save_json(root / "summary.json", summary)
    document = r"""# DeepSets: прямое восстановление индексатора и новые кодбуки

Эксперимент G27. Проверяется прежний синтетический DeepSets: восемь двумерных точек, общий блок 2→2 с ReLU, суммирование по точкам и двум фильтрам. Learner видит плоские 16 координат с неизвестной перестановкой; поиск идёт по полной матрице 16×16. Пары координат, общие фильтры и число активных связей не подаются в алгоритм.

## Модель и определения

В каждом из четырёх случаев скрыты новая перестановка входных координат и порядок скрытых нейронов. Внутри точки четыре произвольных коэффициента задают два ReLU-фильтра. Один общий индексатор используется с разными кодбуками.

$$
\begin{aligned}
\varphi_c(u,v)&=[b+c_1u+c_3v]_+ + [b+c_2u+c_4v]_+,\\
f_c(X)&=\frac1{\sqrt{16}}\sum_{i=1}^{8}\varphi_c(u_i,v_i)+d_c,\\
W_{dh}&=\operatorname{sgn}(I_{dh})c_{|I_{dh}|},\qquad I_{dh}\in\{0,\pm1,\ldots,\pm4\}.
\end{aligned}
$$

b=0.15 и readout известны, как в выбранном prefix-методе. Четыре C свободны по знаку; удвоенный словарь содержит девять значений с нулём. Выход центрируется в соответствии с прежней задачей, но неизвестный d не раскрывается из teacher: он оценивается по независимому наблюдению f(0)−sqrt(H)b. Этот наблюдаемый сдвиг одинаково используется всеми методами.

## Поиск и разделение информации

На геометрию есть четыре source-функции: прежний кодбук [0.8,−1.1,1.3,0.45] и три новых Gaussian-кодбука. Все численные значения, истинный I и роли координат скрыты от поиска. Доступны только вещественные выходы в выбранных точках; teacher-градиенты и активации не используются.

Алгоритм прежний: осевой fit численного словаря → достраивание мультимножества префиксов с целыми количествами продолжений по полной функциональной MSE → банк кандидатов → свежий fit четырёх C для каждого I → предварительный общий I → калибровка C → повторные prefix-поиски с пустых префиксов при фиксированных C → независимый meta-выбор. Обучающие labels и meta-labels разделены; все предсказатели фиксируются до meta, все I — до структурного аудита. Решатели QP/counts и гиперпараметры взяты из G25. Добавлена наблюдаемая калибровка output-offset.

Здесь сохранена исходная combined axis+prefix training loss при калибровке C. В отличие от G26 не используется сокращённая Gaussian-only калибровка. Основной метод — single/fixed; первичный bootstrap — отдельный контроль. Истинные I и схема point-pooling не используются для отбора. Данные для разных уровней шума спарены.

| Метод | Точный I, без шума | Точный I, сырой σ=0.2* | Точный I, сырой σ=2* |
|---|---:|---:|---:|
"""
    document += "\n".join(recovery)
    document += f"\n\nИсходных prefix-траекторий завершено {trajectories['bootstrap']['complete']}/{trajectories['bootstrap']['total']}, повторных {trajectories['repeat']['complete']}/{trajectories['repeat']['total']}. Все следы и отказы сохранены. Точный I означает совпадение нулей, групп и согласованных знаков с учётом только глобального переименования/переворота групп и перестановки скрытых нейронов; перестановка входов в структурном аудите запрещена.\n\n"
    document += "## Новые функции и плотный контроль\n\nПосле фиксации всех source-I созданы восемь новых Gaussian-кодбуков для каждой из четырёх геометрий: всего 32 новые функции, одинаковые в трёх условиях шума. Переносится только I. Все C и состояния оптимизатора инициализируются заново; 32 знаковых старта и до 24 шагов Gauss–Newton, выбор старта/checkpoint по validation. Support4096/validation1024, test8192. Совпадающие канонические I переиспользуют один одинаковый численный fit, включая oracle-контроль; численные значения между разными функциями не переносятся.\n\n"
    document += "Плотный контроль обучает 256 независимых весов при тех же известных b/readout и том же наблюдаемом output-offset. Используется настоящий torch.optim.Adam: два заранее заданных LR0.003/0.01 ×16 инициализаций,4096шагов,batch256; validation выбирает запуск/checkpoint. Все 24 target-предсказателя одной геометрии обучаются пакетом. Все четыре геометрии фиксируют модели до открытия любого clean-test. Кластеризации плотных W нет; у этого контроля нет категориального I для оценки exact-sharing.\n\n"
    document += "| Сырой σ* | Метод | Оценено функций | Mean clean MSE | Mean clean NMSE | Worst NMSE | Mean permutation RMS / std |\n|---|---|---:|---:|---:|---:|---:|\n"
    document += "\n".join(comparison)
    document += "\n\nNMSE делит MSE каждой функции на дисперсию её clean-test ответов, затем усредняется. В таблице 32 функции на условие; это не 96 независимых новых кодбуков. Стоимость поиска I дополнительная, поэтому таблица не является сравнением равных общих бюджетов. Четыре оптимизируемых C сравниваются с 256 весами после одинаковой калибровки d. Свободный поиск неизвестных b/readout здесь не проверяется.\n\n"
    document += "GN использует float64, плотный Adam — float32. Без шума значения около машинной точности не следует интерпретировать как практический выигрыш на десятки порядков. При сильном шуме средняя NMSE основного метода меньше dense-контроля, но worst-case NMSE хуже: ошибки одной структуры затрагивают восемь новых функций. Равномерного превосходства по всем новым задачам нет.\n\n"
    if (root / "failure_diagnosis.json").exists():
        document += "## Диагностика неудачного случая\n\nПри сыром шуме2 в family2 бинарная маска по-прежнему точна,32связи. Bootstrap объединил роли в три группы вместо четырёх:8/256ошибок; после калибровки выбраны три группы и11/256ошибок. Правильного I нет ни в исходном, ни в итоговом банке, поэтому это не ошибка выбора среди банка с правильным решением.\n\n"
        document += "На первой source-функции истинные модули кодов0.45/0.8/1.1/1.3 оценены как0.44846/0.82172/1.20721/16. Последний слот у границы не восстанавливает отдельный настоящий код. Остальные три source-оценки также неточны. Ошибка нулевого якоря source0 составляет−0.00020456 и создаёт общий сдвиг осевой суммы; его возможный вклад не отделён дополнительной факторной абляцией.\n\n"
        document += "После фиксации основного эксперимента зарегистрирован один диагностический oracle-контроль: тот же noisy source0, тот же наблюдаемый output-offset, те же prefix-панели и unit/QP-решатель, но истинные численные C. Он дал точный I,0/256ошибок. Этот контроль не включён в primary, не заменяет ошибочный выбранный I и не использует target-test для изменения метода. Он локализует данный сбой в оценке/калибровке словаря: count-поиск с правильными числами на тех же данных справляется. Артефакты: failure_diagnostic_registration.json/failure_diagnosis.json/failure_oracle_C_control.npz.\n\n"
        document += "![Ошибка шэринга при точной маске](figures/failure_sharing.png)\n\nОшибочный bootstrap остаётся инвариантным к перестановке точек, поскольку повторяет один и тот же неправильный локальный блок. Поэтому одна инвариантность не подтверждает правильность всех групп шэринга. После повторного поиска часть назначений различается между точками; инвариантность тоже нарушается.\n\n"
    document += "## Инвариантность и рисунки\n\nДля каждой новой функции используются 64 новых набора точек и 16 перестановок целых точек на набор. Измеряется RMS отклонения предсказаний от их среднего по перестановкам, нормированный на target std. Это новые входы после фиксации моделей. Переставляются истинные целые точки, а не отдельные признаки. Истинное разбиение используется только для этой итоговой диагностики.\n\n"
    document += "![Индексаторы и реальные веса](figures/indexators_and_weights.png)\n\nДля читаемости рисунка скрытая перестановка входов и столбцов раскрыта только после выбора. В вычислении exact-I входы не переставляются. Численные веса относятся к одной и той же новой target-функции при шуме2; плотные веса показаны непосредственно. Все значения и порядки отображения сохранены в figures/heatmap_values.npz.\n\n"
    document += "![Перенос и инвариантность](figures/transfer_and_invariance.png)\n\n![Динамика свежего обучения](figures/target_learning.png)\n\n"
    document += "## Шум, стоимость и ограничения\n\n"
    document += f"*Шум сырого ответа Gaussianσ0.2/2 усредняется. Axis R={plan['axis_repeats']:,}, prefix R={plan['slice_repeats']:,}, fit/meta и target R={plan['target_repeats']}. Нулевой source-якорь использует2D×Raxis, чтобы его вклад в дисперсию осевой суммы был сопоставим с шумом самих осевых запросов. На target R0=Nsupport×Rtarget. Все средние симулируются аналитически; физически миллиарды повторов не выполняются.\n\n"
    document += f"Номинальная цена одного шумного source-условия — {source_budget:,} независимых ответов на функцию; target-адаптация — {target_budget:,}. Это сильное ограничение практической применимости текущего результата. Он не означает устойчивость к неусреднённомуσ2 и не устанавливает извлечение из произвольного пассивного датасета.\n\n"
    document += "Результат относится к известному однослойному ReLU-классу с суммированием, фиксированным числом точек и известными общим bias/readout. MNIST/FashionMNIST, обучаемый нелинейный rho после pooling и неизвестные индивидуальные bias здесь не запускались. Предыдущие G23–G26 не изменялись.\n\n"
    document += "## Воспроизводимость\n\nПротокол/хеши исходников: protocol.json/code_snapshot. source_frozen.json фиксирует все I до аудита; target/cohort_frozen.json фиксирует всю target-когорту до test/permutation. structure_audit.json, target/evaluation.json и target_metrics.csv содержат отдельные измерения, comparison.csv/summary.json — сводку. Численные fits, исходные наблюдения, точки перестановок и реальные W сохранены. Программные тесты не создавались и не запускались. Карты6/7 не использовались.\n"
    (root / "REPORT.md").write_text(document)
    save_json(
        root / "report_provenance.json",
        {
            "created_utc": timestamp(),
            "report_sha256": digest(root / "REPORT.md"),
            "report_code_sha256": digest(Path(__file__)),
            "source_freeze_sha256": digest(root / "source_frozen.json"),
            "target_freeze_sha256": digest(root / "target/cohort_frozen.json"),
        },
    )
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.root / "protocol.json").read_text())
    structure = json.loads((args.root / "structure_audit.json").read_text())["rows"]
    evaluation = json.loads((args.root / "target/evaluation.json").read_text())
    aggregated = aggregate(plan, structure, evaluation["rows"])
    heatmaps(args.root, plan)
    plots(args.root, plan, evaluation["rows"])
    failure_figure(args.root)
    report(args.root, plan, structure, evaluation, aggregated)


if __name__ == "__main__":
    main()
