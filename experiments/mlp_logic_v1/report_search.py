"""Write the source-search report, figures, tables, and discrete indices."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT / "mds/assets/2026-10-28/mlp_structure_search"
REPORT = ROOT / "mds/MLP_STRUCTURE_SEARCH_PILOT_2026-10-28.md"
LABELS = {
    "random_indices": "Случайный поиск + индексы",
    "evolution_indices": "Эволюция + индексы",
    "random_weights": "Случайный поиск + веса",
    "evolution_weights": "Эволюция + веса",
}
COLORS = ["#2875b5", "#8053a4", "#c77827", "#148b79"]


def load(path):
    return json.loads(path.read_text())


def save_figure(fig, name):
    fig.savefig(ASSETS / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(ASSETS / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def export_indices(plan, source, refit=False):
    for name in ["random_indices", "evolution_indices"]:
        models = np.load(
            source
            / name
            / ("fitted_models.npz" if refit else "winner_models.npz")
        )
        structures = np.load(source / name / "structures.npz")
        winners = load(
            source
            / name
            / (
                "winners_evaluation.json"
                if refit
                else "winners_selection.json"
            )
        )
        base = np.asarray(plan["model"]["codebook_base"], dtype=np.float32)
        arrays = {}
        for layer in range(4):
            masks, scales = [], []
            for winner in winners:
                if refit:
                    masks.append(structures[f"mask_{layer}"][winner["task"]])
                    scales.append(structures["scales"][winner["task"], layer])
                else:
                    index = (
                        winner["task"] * plan["search"]["population"]
                        + winner["candidate"]
                    )
                    masks.append(
                        structures[f"mask_{layer}"][
                            winner["generation"], index
                        ]
                    )
                    scales.append(
                        structures["scales"][
                            winner["generation"], index, layer
                        ]
                    )
            masks, scales = (
                np.stack(masks),
                np.stack(scales).astype(np.float32),
            )
            codebooks = [
                base[None]
                * scales[:, :1]
                / np.sqrt(masks.sum(1).mean(1).astype(np.float32))[:, None],
                base[None]
                * scales[:, 1:]
                * plan["model"]["bias_codebook_step"],
            ]
            for offset, kind in [(0, "weights"), (1, "biases")]:
                values = models[f"parameter_{2 * layer + offset}"]
                if refit:
                    values = values[[w["model_index"] for w in winners]]
                indices = (
                    np.abs(
                        values[..., None] - codebooks[offset][:, None, None]
                    )
                    .argmin(-1)
                    .astype(np.int8)
                )
                # Preserve exact stored float32 values for every used code.
                for task in range(len(winners)):
                    for code in np.unique(indices[task]):
                        used = values[task][indices[task] == code]
                        if np.ptp(used) > 1e-6:
                            raise ValueError(
                                "A saved index refers to different values"
                            )
                        codebooks[offset][task, code] = used[0]
                arrays[f"{kind}_indices_{layer}"] = indices
                arrays[f"{kind}_codebook_{layer}"] = codebooks[offset]
            arrays[f"mask_{layer}"] = masks
        np.savez_compressed(source / name / "winner_indices.npz", **arrays)


def plot_progress(plan, source):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1), layout="constrained")
    for color, method in zip(COLORS, plan["methods"]):
        name = method["name"]
        progress = load(source / name / "progress.json")
        budget = [row["candidates_per_task"] for row in progress]
        loss = [
            np.mean([m["loss"] for m in row["incumbent_selection"]])
            for row in progress
        ]
        balanced = [
            np.mean(
                [
                    (m["recall_positive"] + m["recall_negative"]) / 2
                    for m in row["incumbent_selection"]
                ]
            )
            for row in progress
        ]
        axes[0].plot(budget, loss, label=LABELS[name], color=color)
        axes[1].plot(budget, np.array(balanced) * 100, color=color)
    axes[0].set_ylabel("Средняя BCE выбранных кандидатов ↓")
    axes[1].set_ylabel("Средний recall двух классов, % ↑")
    axes[0].set_title("Ход поиска: проверочная ошибка")
    axes[1].set_title("Ход поиска: balanced accuracy")
    axes[1].axhline(50, color="#777777", linewidth=1, linestyle=":")
    for axis in axes:
        axis.set_xlabel("Обученных кандидатов на задачу")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    save_figure(fig, "search_progress")


def plot_inner(plan, refit):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    for color, method in zip(COLORS, plan["methods"]):
        rows = load(refit / method["name"] / "curves.json")
        steps = [r["step"] for r in rows]
        validation = [np.mean(r["loss"]) for r in rows]
        training = [r for r in rows if "train_loss" in r]
        axes[0].plot(
            steps, validation, color=color, label=LABELS[method["name"]]
        )
        axes[0].plot(
            [r["step"] for r in training],
            [np.mean(r["train_loss"]) for r in training],
            color=color,
            linestyle="--",
            alpha=0.65,
        )
        axes[1].plot(
            steps,
            [np.mean(r["recall_positive"]) * 100 for r in rows],
            color=color,
        )
        axes[2].plot(
            steps,
            [np.mean(r["recall_negative"]) * 100 for r in rows],
            color=color,
        )
    axes[0].set_title("BCE: train -- / validation —")
    axes[1].set_title("Recall класса 1")
    axes[2].set_title("Recall класса 0")
    axes[0].set_ylabel("Ошибка ↓")
    axes[1].set_ylabel("Полнота, % ↑")
    axes[2].set_ylabel("Полнота, % ↑")
    axes[0].legend(fontsize=7)
    for axis in axes:
        axis.set_xlabel("Шагов SGD внутри выбранной структуры")
        axis.grid(alpha=0.2)
    save_figure(fig, "inner_training")


def plot_comparison(plan, short_rows, long_rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    names = [m["name"] for m in plan["methods"]]
    y = np.arange(len(names))
    for axis, field, title in [
        (axes[0], "mean_accuracy", "Доля правильных ответов"),
        (axes[1], "mean_balanced_accuracy", "Средний recall двух классов"),
    ]:
        short = [r[field] * 100 for r in short_rows]
        long = [r[field] * 100 for r in long_rows]
        axis.barh(
            y + 0.16,
            short,
            height=0.3,
            color="#a8b7c6",
            label="Поиск: 512 шагов на кандидата",
        )
        axis.barh(
            y - 0.16,
            long,
            height=0.3,
            color="#187f81",
            label="Дообучение: 16 384 шага",
        )
        axis.set_yticks(
            y, [LABELS[n] for n in names] if axis is axes[0] else []
        )
        axis.invert_yaxis()
        axis.set_xlim(0, 105)
        axis.set_xlabel("На независимых примерах, %")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
        for position, value in zip(y - 0.16, long):
            axis.text(
                value + 1, position, f"{value:.1f}", va="center", fontsize=9
            )
    axes[1].axvline(50, color="#777777", linestyle=":", linewidth=1)
    axes[0].legend(loc="lower left", fontsize=8)
    save_figure(fig, "comparison")


def plot_calibration():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for optimizer, color in [("adam", COLORS[0]), ("sgd", COLORS[2])]:
        rows = load(
            HERE / "results/calibration_v1" / f"{optimizer}_curves.json"
        )
        steps = [row["step"] for row in rows]
        axes[0].plot(
            steps,
            [np.mean(row["loss"]) for row in rows],
            color=color,
            label=optimizer.upper(),
        )
        trained = [row for row in rows if "train_loss" in row]
        axes[0].plot(
            [row["step"] for row in trained],
            [np.mean(row["train_loss"]) for row in trained],
            color=color,
            linestyle="--",
        )
        axes[1].plot(
            steps,
            [np.mean(row["recall_positive"]) * 100 for row in rows],
            color=color,
        )
        axes[2].plot(
            steps,
            [np.mean(row["recall_negative"]) * 100 for row in rows],
            color=color,
        )
    for axis, title in zip(
        axes,
        ["BCE: train -- / validation —", "Recall класса 1", "Recall класса 0"],
    ):
        axis.set_title(title)
        axis.set_xlabel("Шагов обучения плотной MLP")
        axis.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("Архивная калибровка: среднее по всем настройкам и стартам")
    save_figure(fig, "calibration_training")


def weak_bank_section(path):
    summary = load(path / "summary.json")["methods"]
    diagnostics = load(path / "structural_diagnostics.json")
    weights = load(path / "source_quality_weights.json")
    priors = np.load(path / "learned_prior.npz")
    truth_groups = np.load(HERE / "results/search_v1/source_tasks.npz")[
        "groups_diagnostic"
    ]
    by_name = {row["method"]: row for row in diagnostics}
    labels = {
        "weak_values": "Группы из частичных решений",
        "weak_masks": "Группы из одних масок",
        "shuffled": "Перемешанные группы",
        "oracle": "Правильные группы заданы",
        "dense_adam": "Плотный Adam",
        "dense_sgd": "Плотный SGD",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    precision = [
        7 / 63,
        by_name["pooled_masks"]["group_pair_precision"],
        by_name["pooled_values"]["group_pair_precision"],
    ]
    axes[0].barh(
        ["Случайная группировка", "По маскам", "По обученным значениям"],
        np.array(precision) * 100,
        color=["#a8b7c6", COLORS[2], COLORS[3]],
    )
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Доля правильно объединённых пар, %")
    axes[0].set_title("Информация о скрытых группах")
    axes[0].invert_yaxis()
    for index, value in enumerate(precision):
        axes[0].text(value * 100 + 1, index, f"{value:.1%}", va="center")
    balanced = [row["mean_balanced_accuracy"] * 100 for row in summary]
    axes[1].barh(
        [labels[row["method"]] for row in summary],
        balanced,
        color=[
            COLORS[3],
            COLORS[2],
            "#a8b7c6",
            "#44685d",
            COLORS[0],
            COLORS[1],
        ],
    )
    axes[1].set_xlim(0, 100)
    axes[1].invert_yaxis()
    axes[1].axvline(50, color="#777777", linestyle=":", linewidth=1)
    axes[1].set_title("Перенос на 8 новых задач")
    axes[1].set_xlabel("Средний recall двух классов, %")
    for index, value in enumerate(balanced):
        axes[1].text(value + 1, index, f"{value:.1f}", va="center", fontsize=9)
    save_figure(fig, "weak_information_transfer")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), layout="constrained")
    for axis, key, title in zip(
        axes,
        ["weak_values_groups", "weak_masks_groups"],
        ["Группы по обученным значениям", "Группы по одним маскам"],
    ):
        matrix = np.array(
            [
                [len(set(group) & set(actual)) for actual in truth_groups]
                for group in priors[key]
            ]
        )
        order = np.argsort(matrix.argmax(1), kind="stable")
        matrix = matrix[order]
        shown = axis.imshow(matrix, vmin=0, vmax=8, cmap="YlGnBu")
        axis.set_xticks(range(8), range(1, 9))
        axis.set_yticks(range(8), range(1, 9))
        axis.set_xlabel("Истинная группа")
        axis.set_ylabel("Найденная группа")
        axis.set_title(title)
        for i in range(8):
            for j in range(8):
                axis.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] >= 5 else "black",
                )
    fig.colorbar(shown, ax=axes, label="Совпавших входов из восьми")
    save_figure(fig, "weak_group_overlap")
    evaluation = path / "evaluation"
    curves = load(evaluation / "curves.json")
    plan = load(path / "protocol.json")
    per_method = (
        plan["transfer"]["evaluation_tasks"]
        * plan["transfer"]["evaluation_restarts"]
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), layout="constrained")
    for index, row in enumerate(summary):
        color = plt.get_cmap("tab10")(index)
        selected = slice(index * per_method, (index + 1) * per_method)
        steps = [point["step"] for point in curves]
        axes[0].plot(
            steps,
            [np.mean(point["loss"][selected]) for point in curves],
            color=color,
            label=labels[row["method"]],
        )
        trained = [point for point in curves if "train_loss" in point]
        axes[0].plot(
            [point["step"] for point in trained],
            [np.mean(point["train_loss"][selected]) for point in trained],
            linestyle="--",
            color=color,
            alpha=0.6,
        )
        for axis, key in zip(axes[1:], ["recall_positive", "recall_negative"]):
            axis.plot(
                steps,
                [np.mean(point[key][selected]) * 100 for point in curves],
                color=color,
            )
    for axis, title in zip(
        axes,
        ["BCE: train -- / validation —", "Recall класса 1", "Recall класса 0"],
    ):
        axis.set_title(title)
        axis.set_xlabel("Шагов на новой задаче")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    save_figure(fig, "weak_transfer_training")
    table = [
        "| Представление новой задачи | Средняя BCE ↓ | Accuracy | Средний recall двух классов |",
        "| :-- | --: | --: | --: |",
    ]
    for row in summary:
        table.append(
            f"| {labels[row['method']]} | {row['mean_bce']:.4f} | {row['mean_accuracy']:.1%} | {row['mean_balanced_accuracy']:.1%} |"
        )
    supported = [row for row in weights if row["quality_weight"] > 0]
    support_tasks = sorted({row["task"] + 1 for row in supported})
    detail = load(evaluation / "winners.json")
    value_losses = np.array(
        [
            row["evaluation"]["loss"]
            for row in detail
            if row["method"] == "weak_values"
        ]
    )
    rng = np.random.default_rng(281067)
    paired = []
    for method in ["shuffled", "dense_adam"]:
        comparator = np.array(
            [
                row["evaluation"]["loss"]
                for row in detail
                if row["method"] == method
            ]
        )
        difference = value_losses - comparator
        bootstrap = difference[
            rng.integers(len(difference), size=(4096, len(difference)))
        ].mean(1)
        interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        paired.append(
            {
                "comparator": method,
                "mean_bce_difference": float(difference.mean()),
                "paired_bootstrap_95_interval": interval,
                "tasks_with_lower_bce": int((difference < 0).sum()),
            }
        )
    (ASSETS / "weak_transfer.csv").write_text(
        "method,mean_bce,mean_accuracy,mean_balanced_accuracy\n"
        + "".join(
            f"{row['method']},{row['mean_bce']},{row['mean_accuracy']},{row['mean_balanced_accuracy']}\n"
            for row in summary
        )
    )
    (ASSETS / "weak_paired_differences.json").write_text(
        json.dumps(paired, ensure_ascii=False, indent=2) + "\n"
    )
    conclusion = "Частичные решения содержат информацию о скрытых связях. В этом пилоте извлечённая группировка не дала убедительного преимущества над плотным обучением на новых задачах."
    text = rf"""## Можно ли извлечь структуру без точных решений

По уточнению пользователя сняли требование точного решения. Рассмотрели
все 96 моделей длинного этапа и взвесили их по улучшению проверочной BCE
относительно постоянного прогноза частоты класса. Оставлен запас 0.005,
чтобы не придавать вес совсем малым улучшениям. Итоговая оценочная выборка
для этого отбора не использовалась.

Положительный вес получили **{len(supported)} модели из 96**, все — на исходной
задаче {", ".join(map(str, support_tasks))}. Таким образом, банк содержит разные
частичные решения фактически одной функции, а не многих разнообразных задач.
Это существенное ограничение текущего опыта.

По совместному участию входов в обученных нейронах восстановили восемь групп
по восемь координат и пары групп следующего уровня. Число и размеры групп
заданы постановкой; их состав извлекается из моделей. Сравнили с извлечением
из одних масок, без численных значений весов.

**Среди пар, объединённых по обученным значениям, {precision[2]:.1%} действительно
принадлежат одной скрытой группе.** У случайной группировки ожидается
{precision[0]:.1%}; по одним маскам получилось {precision[1]:.1%}. Ни одна из восьми
групп не восстановлена полностью. Это частичная информация о структуре,
а не готовое точное решение.

![Частичная информация и перенос](assets/2026-10-28/mlp_structure_search/weak_information_transfer.png)

![Пересечение найденных и истинных групп](assets/2026-10-28/mlp_structure_search/weak_group_overlap.png)

Группировка заморожена до генерации новых задач. Проверили две отдельные
задачи для выбора шага обучения и **восемь новых полных задач той же семьи**.
Для каждого представления настройка выбиралась только на двух первых задачах.
На новой задаче — четыре старта по 8192 шага, batch 128; все численные веса
обучаются заново. Правильная группировка дана только отдельному oracle-контролю.
Перемешанная группировка сохраняет размеры и число связей.

{chr(10).join(table)}

Извлечение из обученных значений немного лучше перемешанной группировки,
но BCE практически совпадает с плотным Adam. Плотный SGD имеет немного
больший средний recall классов. По восьми задачам нельзя уверенно утверждать
преимущество извлечённой структуры. Разности ошибок и предварительные
парные bootstrap-интервалы сохранены рядом с таблицей.

![Обучение на новых задачах](assets/2026-10-28/mlp_structure_search/weak_transfer_training.png)

**Итог: точность исходных решений не является обязательным условием
извлечения структуры.** Здесь информация нашлась, но её пока недостаточно
для убедительного выигрыша на новых задачах. Логичное продолжение — банк
частичных решений разных полных функций, чтобы общий рисунок связей
восстанавливался по нескольким независимым исходным задачам.

"""
    return text, conclusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search", type=Path, default=HERE / "results/search_v1"
    )
    parser.add_argument(
        "--refit", type=Path, default=HERE / "results/refit_v1"
    )
    parser.add_argument(
        "--weak", type=Path, default=HERE / "results/weak_bank_v1"
    )
    args = parser.parse_args()
    plan = load(args.search / "protocol.json")
    extension = load(args.refit / "extension_protocol.json")
    short = load(args.search / "summary.json")["methods"]
    long = load(args.refit / "summary.json")["methods"]
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    export_indices(plan, args.search)
    export_indices(plan, args.refit, refit=True)
    plot_progress(plan, args.search)
    plot_inner(plan, args.refit)
    plot_comparison(plan, short, long)
    plot_calibration()
    table = [
        "| Поиск структуры и внутреннее обучение | После поиска: accuracy | После дообучения: accuracy | После дообучения: средний recall двух классов | Решено |",
        "| :-- | --: | --: | --: | --: |",
    ]
    aggregates = []
    for initial, final in zip(short, long):
        table.append(
            f"| {LABELS[initial['method']]} | {initial['mean_accuracy']:.1%} | {final['mean_accuracy']:.1%} | {final['mean_balanced_accuracy']:.1%} | {final['successes']}/4 |"
        )
        aggregates.append(
            {
                "method": initial["method"],
                "short_accuracy": initial["mean_accuracy"],
                "short_balanced_accuracy": initial["mean_balanced_accuracy"],
                "refit_accuracy": final["mean_accuracy"],
                "refit_balanced_accuracy": final["mean_balanced_accuracy"],
                "refit_successes": final["successes"],
            }
        )
    with (ASSETS / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    per_task = [
        "| Задача | Случайный + индексы | Эволюция + индексы | Случайный + веса | Эволюция + веса | Частый класс |",
        "| :-- | --: | --: | --: | --: | --: |",
    ]
    details = {
        m["name"]: load(args.refit / m["name"] / "winners_evaluation.json")
        for m in plan["methods"]
    }
    for task in range(plan["tasks"]):
        values = [
            details[m["name"]][task]["evaluation"]["accuracy"]
            for m in plan["methods"]
        ]
        rate = details[plan["methods"][0]["name"]][task]["evaluation"][
            "positive_rate"
        ]
        per_task.append(
            f"| {task + 1} | "
            + " | ".join(f"{v:.1%}" for v in values)
            + f" | {max(rate, 1 - rate):.1%} |"
        )
    source_tasks = np.load(args.search / "source_tasks.npz")
    example = source_tasks["parents"][0, :2] + 1
    example_text = "; ".join(
        ", ".join(str(int(i)) for i in row) for row in example
    )
    per_search = plan["search"]["population"] * plan["search"]["generations"]
    short_examples = (
        per_search
        * plan["search"]["candidate_steps"]
        * plan["search"]["batch_size"]
    )
    extra_examples = (
        extension["restarts"]
        * extension["steps"]
        * plan["search"]["batch_size"]
    )
    success_total = sum(m["successes"] for m in long)
    conclusion = (
        "Успешный банк не получен ни одним из четырёх вариантов. Поэтому генератор структур и перенос на новые задачи не обучались и не оценивались."
        if success_total == 0
        else "Часть выбранных структур прошла критерий качества. Это результат сбора исходных решений; перенос на новые задачи в этой серии ещё не оценивался."
    )
    weak_text = ""
    if (args.weak / "summary.json").exists():
        weak_text, conclusion = weak_bank_section(args.weak)
    text = rf"""# Поиск структур MLP: случайный выбор и эволюция

28 октября 2026. Сбор полных и частичных решений, извлечение структуры и проверка переноса.

{conclusion}

## Задача и её точное решение

На вход поступают 64 независимых равновероятных бита. Ответ вычисляет скрытая
схема: 16 блоков первого уровня, четыре второго и один выходной. Каждый блок
получает четыре входа и вычисляет XOR либо порог «не менее двух/трёх единиц».
У задач общая скрытая группировка координат, но разные связи и операции.
Ученику дают только вход и конечный ответ всей схемы.

При известной схеме решение записывается точно через ReLU. Обозначим через
ReLU положительную часть числа, а через s — сумму четырёх входящих битов:

$$
\begin{{aligned}}
\operatorname{{XOR}}_4(s) &= \operatorname{{ReLU}}(s)
-2\operatorname{{ReLU}}(s-1)+2\operatorname{{ReLU}}(s-2)
-2\operatorname{{ReLU}}(s-3),\\
T_2(s) &= \operatorname{{ReLU}}(s-1)-\operatorname{{ReLU}}(s-2),\\
T_3(s) &= \operatorname{{ReLU}}(s-2)-\operatorname{{ReLU}}(s-3).
\end{{aligned}}
$$

На целых s от 0 до 4 эти формулы дают нужные нули и единицы. Следовательно,
все задачи точно представимы ReLU-MLP 64–64–16–4–1. Ученики имеют именно эти
размеры и обучаемые смещения. Это доказывает существование точного решения
для свободных численных весов; произвольный фиксированный codebook
не обязан содержать значения для такого решения.

Например, для четырёх битов (1, 0, 1, 0) сумма равна двум:
XOR возвращает 0, порог двух единиц возвращает 1, а порог трёх — 0.

Например, в первой задаче два первых блока берут координаты **{example_text}**.
Их ответы затем участвуют в операции следующего уровня. Эти координаты и
промежуточные ответы используются только генератором данных, а не поиском.

## Что выбираем и что обучаем

Кандидат задаёт маски связей и codebook каждого слоя: девять общих чисел
для весов и ещё девять для смещений. Codebook строится из целых значений
от −4 до 4 с выбранным масштабом; маски и масштабы меняются между кандидатами.
Число разрешённых входов нейрона выбирается из нескольких плотностей,
включая полный слой. Первый кандидат каждой задачи — плотная сеть.
Правильную группировку входов поиску не сообщаем.

**Обучение индексов.** Для каждого разрешённого веса обучается вещественное
вспомогательное число. Индекс выбирает ближайший элемент фиксированного
codebook; в прямом проходе используется именно это дискретное значение.
В обратном проходе применяем приближение: производную выбора заменяем
единицей. Это straight-through SGD, а не точный градиент дискретного индекса.
Числа самого codebook остаются фиксированными. Смещения обучаются аналогично.

Для одного слоя это можно записать так:

$$
W_{{ij}}=M_{{ij}}c_{{I_{{ij}}}},
\qquad I_{{ij}}=\arg\min_{{k\in\{{1,\ldots,9\}}}}|u_{{ij}}-c_k|.
$$

Здесь M — предложенная маска связей, c — девять фиксированных чисел,
u — обучаемые вспомогательные числа, I — полученные дискретные индексы.

**Обучение весов.** При той же фиксированной маске обычный SGD обучает
свободные численные веса и смещения. Индексов общего codebook в этом варианте нет.

**Случайный поиск** независимо предлагает новые маски и масштабы.
**Эволюция** выбирает четыре лучших кандидата по проверочной BCE,
скрещивает их связи на уровне нейронов, меняет часть связей и масштабы.
В каждой популяции из 16 есть четыре новых случайных кандидата. Численные
веса и вспомогательные числа индексов для каждого кандидата обучаются заново:
наследуется структура. Отбор не использует правильные связи учителя.

## Данные и бюджет

Взяты те же четыре полные задачи, которые уже просмотрены при предыдущей
проверке плотных SGD/Adam. Это исходные задачи для разработки способа сбора
банка, **не независимые новые задачи переноса**. Части по отдельности не обучались.

Каждый метод проверил {per_search} кандидатов на задачу: 16 популяций по 16,
по 512 шагов SGD и 128 новых примеров на шаг. Это {short_examples:,} предъявлений
обучающих примеров на задачу и метод. Сравниваются три шага обучения,
используется SGD с моментом 0.9. Начальная популяция, данные и случайные
старты согласованы между вариантами. Число предъявлений одинаково;
стоимость квантования означает, что это не равенство FLOPs.

Кандидат и чекпойнт выбираются по BCE на отдельных 4096 примерах.
Все четыре поиска завершились до оценки на 16 384 независимых примерах.
Успех требует accuracy ≥97% и recall каждого класса ≥95%.

После просмотра хода поиска добавлен диагностический этап: для выбранной
структуры каждой задачи по шесть новых стартов, по 16 384 шага. Это ещё
{extra_examples:,} предъявлений на задачу и метод. Настройки для всех методов
одинаковы; итог оценивается на новой независимой выборке. Этот этап проверяет
недостаток времени внутреннего обучения, но не заменяет длинную оценку всех
ранее отброшенных структур.

## Результаты

{chr(10).join(table)}

Accuracy — доля правильных ответов. Средний recall двух классов
(balanced accuracy) не позволяет выдать постоянное предсказание частого
класса за решение: у такого предсказания показатель равен 50%.

![Сравнение вариантов](assets/2026-10-28/mlp_structure_search/comparison.png)

Детали после длинного обучения выбранных структур:

{chr(10).join(per_task)}

Этапы поиска и дообучения оценены на разных независимых выборках.
Столбец «Частый класс» — доля более частого класса в той же итоговой выборке;
это ориентир для оценки accuracy, а не отдельный алгоритм поиска.

![Ход поиска](assets/2026-10-28/mlp_structure_search/search_progress.png)

На графике поиска показано среднее по четырём задачам для текущего кандидата,
выбранного по минимальной проверочной BCE. Это проверочные, а не итоговые метрики.

![Loss и recall при длинном внутреннем обучении](assets/2026-10-28/mlp_structure_search/inner_training.png)

Кривые усреднены по всем шести стартам каждой задачи, а не только по лучшим.
Пунктир показывает среднюю обучающую BCE между отметками; сплошные линии —
ошибку и recall на проверочной выборке текущих параметров.

{weak_text}## Что можно заключить

{conclusion}

Пилот проверяет конкретные маски, codebook и операторы эволюции при указанном
бюджете. Он не доказывает невозможность эволюционного решения или отсутствие
полезной структуры. По одному запуску на четырёх исходных задачах также
нельзя уверенно ранжировать небольшие различия между методами.

Для продолжения важнее разнообразие исходных функций: сейчас информативные
частичные решения относятся к одной задаче. Следует отдельно оценивать
восстановление общей структуры и улучшение качества новых задач; порог
точности 97% не является условием допуска модели к извлечению структуры.

## Воспроизведение

Код и протоколы: [experiments/mlp_logic_v1](../experiments/mlp_logic_v1/).
Исходные результаты поиска: [search_v1](../experiments/mlp_logic_v1/results/search_v1/summary.json),
длинного обучения: [refit_v1](../experiments/mlp_logic_v1/results/refit_v1/summary.json).
Сохранены все структуры, качества кандидатов, loss/recall, выбранные модели,
дискретные индексы лучших вариантов и SHA256 замороженных исходников.
Лучшие неуспешные кандидаты не называются успешным банком.
Уточнение с частичными решениями и переносом:
[weak_bank_v1](../experiments/mlp_logic_v1/results/weak_bank_v1/summary.json).
Кривые предыдущего этапа:
[калибровка плотных методов](assets/2026-10-28/mlp_structure_search/calibration_training.png).
"""
    REPORT.write_text(text)
    print(REPORT)


if __name__ == "__main__":
    main()
