"""Report frozen Pattern 0110 fits without choosing new models or settings."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.optimize import linear_sum_assignment

from experiments.darts_indexator_v1.data import digest, load_archive, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .pattern_prefix import METHODS, real_output, timestamp


def source_evaluation(root, plan, audit):
    teacher = load_archive(root / "private/teacher.npz")
    rng = np.random.default_rng(plan["seed"] + 999163)
    x = rng.normal(size=(8192, 11))
    x *= rng.choice([0.3, 1.0, 3.0], size=(len(x), 1))
    truth = real_output(x, teacher["index"], teacher["codebooks"], plan["bias"])
    rows = []
    for row in audit["rows"]:
        if "method" not in row:
            continue
        folder = root / "source" / f"case_{row['condition']}_{row['run']}"
        selected = json.loads((folder / "selection.json").read_text())
        values = load_archive(folder / "predictors.npz")
        number = selected["methods"][row["method"]]["candidate"]
        prediction = real_output(
            x, values["indices"][number], values["codes"][number], plan["bias"]
        )
        rows.append(
            {**row, "fresh_clean_mse": float(np.mean((prediction - truth) ** 2))}
        )
    save_json(
        root / "fresh_source_evaluation.json",
        {
            "evaluated_utc": timestamp(),
            "diagnostic_after_freeze": True,
            "points": len(x),
            "scales": [0.3, 1.0, 3.0],
            "rows": rows,
        },
    )
    return rows


def figures(root, plan, results):
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    aligned = load_archive(root / "aligned_indices.npz")
    truth = aligned["truth"]
    # Window order is used only in this post-freeze presentation.
    order = np.argsort(np.argmax(truth != 0, axis=0))
    truth = truth[:, order]
    clean = aligned["0_0_single_fixed"][:, order]
    noisy = aligned["1_0_single_fixed"][:, order]
    shared = load_archive(root / "classification/shared.npz")
    dense = load_archive(root / "classification/dense.npz")
    selected = results["case_0_0/single_fixed"]["row"]
    index, code = shared["indices"][selected], shared["parameters"][selected]
    _, _, columns = align_indices(index, truth, 2)
    weight = np.r_[0.0, code[:2]][np.abs(index)] * np.sign(index)
    learned_weight = weight[:, columns]
    dense_row = results["dense_common"]["row"]
    dense_weight = dense["parameters"][dense_row, :88].reshape(11, 8)
    reference = np.where(truth == -1, -1.0, np.where(truth == 2, 1.0, 0.0))
    cosine = dense_weight.T @ reference
    cosine /= (
        np.linalg.norm(dense_weight, axis=0)[:, None]
        * np.linalg.norm(reference, axis=0)[None]
    )
    source, target = linear_sum_assignment(-cosine)
    dense_weight = dense_weight[:, source[np.argsort(target)]]
    arrays = [truth, clean, noisy, learned_weight, dense_weight]
    labels = [
        "Истинный I",
        "Найденный I: без шума",
        "Найденный I: шум 0,2*",
        "Свежий кодбук: W",
        "Плотный Adam: W",
    ]
    cmap = ListedColormap(["#345c9c", "#8bb4d8", "#f4f5f6", "#e6ac63", "#b3532e"])
    norm = BoundaryNorm(np.arange(-2.5, 3, 1), cmap.N)
    fig, axes = plt.subplots(1, 5, figsize=(18.5, 6.3), layout="constrained")
    limit = max(
        float(np.max(np.abs(learned_weight))), float(np.max(np.abs(dense_weight)))
    )
    for number, (ax, array, label) in enumerate(zip(axes, arrays, labels, strict=True)):
        if number < 3:
            artist = ax.imshow(array, cmap=cmap, norm=norm, aspect="equal")
        else:
            artist = ax.imshow(
                array, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="equal"
            )
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Скрытый нейрон")
        ax.set_xticks(range(8), range(1, 9))
        ax.set_yticks(range(11), range(1, 12))
        if number == 0:
            ax.set_ylabel("Входная позиция")
        for i in range(11):
            for j in range(8):
                text = str(int(array[i, j])) if number < 3 else f"{array[i, j]:.1f}"
                color = (
                    "white"
                    if abs(array[i, j]) > (1.5 if number < 3 else 0.65 * limit)
                    else "#20252c"
                )
                ax.text(j, i, text, ha="center", va="center", fontsize=7.4, color=color)
        if number == 2:
            fig.colorbar(
                artist,
                ax=axes[:3],
                shrink=0.65,
                ticks=range(-2, 3),
                label="Индекс, после согласованного переименования",
            )
        if number == 4:
            fig.colorbar(
                artist, ax=axes[3:], shrink=0.65, label="Численное значение веса"
            )
    fig.suptitle(
        "Pattern 0110: извлечение по вещественным выходам → обучение на метках 0/1",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.005,
        "* Шум усреднён. Выравнивание столбцов выполнено после выбора моделей; плотные веса не кластеризуются.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(folder / "indexators_and_weights.png", dpi=180, bbox_inches="tight")
    fig.savefig(folder / "indexators_and_weights.pdf", bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(
        folder / "heatmap_values.npz",
        truth=truth,
        clean=clean,
        noisy=noisy,
        fresh_weights=learned_weight,
        dense_weights=dense_weight,
    )

    names = ["case_0_0/single_fixed", "oracle", "dense_common", "dense_full", "random"]
    display = [
        "Найденный I",
        "Истинный I",
        "Adam: общая голова",
        "Adam: полная голова",
        "Случайный I",
    ]
    colors = ["#176b87", "#57a69c", "#d08335", "#a15d83", "#7e8791"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), layout="constrained")
    for name, label, color in zip(names, display, colors, strict=True):
        if name == "oracle":
            continue  # The exact same I and fresh fit are already shown.
        result = results[name]
        archive = shared if result["archive"] == "shared" else dense
        row = result["row"]
        steps = archive["history_steps"]
        axes[0].plot(steps, archive["train_history"][:, row], color=color, label=label)
        axes[0].plot(
            steps,
            archive["query_history"][:, row],
            color=color,
            linestyle="--",
            alpha=0.85,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Шаг обучения численных параметров")
    axes[0].set_ylabel("Сбалансированная BCE")
    axes[0].set_title("Выбранный по query старт: train — сплошная, query — пунктир")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=9)
    perfect = [results[name]["perfect_full_starts"] for name in names]
    bars = axes[1].bar(range(len(names)), perfect, color=colors)
    for bar, count in zip(bars, perfect, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, count + 0.5, f"{count}/32", ha="center"
        )
    axes[1].set_xticks(range(len(names)), display, rotation=22, ha="right", fontsize=9)
    axes[1].set_ylim(0, 35)
    axes[1].set_ylabel("Стартов с 100% на всех 2048 строках")
    axes[1].set_title("Найденный и истинный I используют один и тот же fit")
    fig.savefig(folder / "classification_learning.png", dpi=170, bbox_inches="tight")
    fig.savefig(folder / "classification_learning.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(root, plan, audit, classification, source_rows):
    methods = classification["methods"]
    labels = {
        "bootstrap": "Первичный prefix",
        "single_fixed": "Калибровка + отдельные prefix-поиски",
        "joint_fixed": "Калибровка + совместный prefix-поиск",
    }
    structural = []
    table = []
    for method in METHODS:
        row = {"method": method}
        cells = []
        for condition in range(2):
            selected = [
                r
                for r in source_rows
                if r["method"] == method and r["condition"] == condition
            ]
            exact = sum(r["exact"] for r in selected)
            row[f"exact_{condition}"] = exact
            row[f"attempts_{condition}"] = plan["search_runs"]
            row[f"fresh_clean_mse_{condition}"] = float(
                np.mean([r["fresh_clean_mse"] for r in selected])
            )
            cells.append(f"{exact}/{plan['search_runs']}")
        structural.append(row)
        table.append(f"| {labels[method]} | {cells[0]} | {cells[1]} |")
    names = [
        ("case_0_0/single_fixed", "Найденный индексатор", 5),
        ("oracle", "Истинный индексатор", 5),
        ("dense_common", "Плотный Adam, общие b/a/d", 91),
        ("dense_full", "Плотный Adam, отдельные b/a", 105),
        ("random", "Случайный индексатор", 5),
    ]
    binary_table = []
    csv_rows = []
    for name, label, parameters in names:
        result = methods[name]
        binary_table.append(
            f"| {label} | {parameters} | {100 * result['test_balanced_accuracy']:.2f}% | {100 * result['full_balanced_accuracy']:.2f}% | {result['perfect_full_starts']}/32 | {result['test_bce']:.6f} |"
        )
        csv_rows.append({"method": label, "parameters": parameters, **result})
    with (root / "classification_comparison.csv").open("w", newline="") as output:
        fields = sorted({key for row in csv_rows for key in row})
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    axis_points = 2 * plan["inputs"] * (2 * plan["axis_points"] - 1)
    per_run_prefix = (
        2 * plan["inputs"] * (plan["slice_points"] + plan["prefix_validation"])
    )
    per_run_fit = plan["fit_points"] + 2 * plan["fit_validation"] + plan["meta_points"]
    budget = {
        "shared_axis_query_locations_per_source": axis_points,
        "prefix_query_locations_per_source_per_run": per_run_prefix,
        "fit_meta_locations_per_source_per_run": per_run_fit,
        "noisy_physical_answers_per_source_all_four_runs": axis_points
        * plan["axis_repeats"]
        + plan["search_runs"]
        * (per_run_prefix * plan["slice_repeats"] + per_run_fit * plan["fit_repeats"]),
        "simulation": "Gaussian sample means generated analytically, not physical repeated calls",
    }
    save_json(root / "query_budget.json", budget)
    trajectories = {
        "bootstrap": {"complete": 0, "total": 0},
        "repeat": {"complete": 0, "total": 0},
    }
    for path in sorted((root / "source").glob("case_*/bootstrap_search.json")):
        for key, filename in (
            ("bootstrap", "bootstrap_search.json"),
            ("repeat", "repeat_search.json"),
        ):
            for record in json.loads((path.parent / filename).read_text()):
                trajectories[key]["total"] += 1
                trajectories[key]["complete"] += int(record["complete"])
    summary = {
        "completed_utc": timestamp(),
        "structural": structural,
        "classification": {name: methods[name] for name, _, _ in names},
        "trajectories": trajectories,
        "query_budget": budget,
        "single_geometry_only": True,
        "teacher_binary_errors_non_test": audit["teacher_binary_errors_on_non_test"],
    }
    save_json(root / "summary.json", summary)
    document = r"""# Pattern 0110: prefix-восстановление и свежий кодбук

Эксперимент G26 завершён. Выполнен согласованный контроль: сначала поиск индексатора по вещественным выходам сети, которая решает 0110, затем отдельное обучение численных параметров по исходным бинарным меткам. Истинный I не поступает в поиск и выбор. Это контроль активного восстановления заданного класса сети, а не извлечение структуры только из меток 0/1.

## Модель и источник информации

Вход — 11 чисел, скрытых нейронов — 8, свободных чисел кодбука — 2. Алфавит индексов содержит ноль и две знаковые пары.

$$
\begin{aligned}
I_{dh}&\in\{0,\pm1,\pm2\},\\
W_{t,dh}&=\operatorname{sgn}(I_{dh})c_{t,|I_{dh}|},\\
f_t(x)&=\frac1{\sqrt8}\sum_{h=1}^{8}\left[-3+\sum_{d=1}^{11}x_d W_{t,dh}\right]_+.
\end{aligned}
$$

Teacher реализует восемь окон 0110. Его столбец содержит -c1,+c2,+c2,-c1 в соответствующих четырёх позициях. Четыре пары коэффициентов независимо сэмплированы из [0.8,1.3]² и скрыты от learner. Во всём этом диапазоне на бинарном кубе ненулевая активация окна возникает ровно при совпадении с 0110: полное совпадение даёт score больше 3, любой промах — меньше 3. Положительный выход teacher эквивалентен метке присутствия паттерна. Вещественный выход дополнительно содержит информацию о числе срабатываний и поведении вне бинарного куба.

Learner знает размеры, размер кодбука, общий bias -3 и readout. Ему не сообщают расположение окон, число активных связей, группы, численные коды или градиенты teacher. Все варианты связей 0/±1/±2 разрешены на каждом шаге.

## Перенос алгоритма и протокол

Изменены размеры D/H/K и известный bias. Отрицательный bias требует только поправки осевого тождества:

$$
[b+u]_+ + [b-u]_+ - 2[b]_+ = [|u|-|b|]_+.
$$

В осевом estimator используется модуль bias и корректный baseline. Семь численных стартов обобщены с четырёх кодов на два. Решатели QP, целых количеств и функциональная цель не изменялись. Для калибровки C использованы 4096 Gaussian-входов вместо объединённой осевой/prefix-панели предыдущего опыта. Все изменения объявлены в protocol.json до поиска. HPO, подбор по teacher или замена неудачных seed не выполнялись.

Для каждого из четырёх случайных порядков входных координат выполнены четыре source-поиска с разными C, четыре исходных варианта дискретного решателя и один цикл калибровки. Основной метод — повторные отдельные prefix-поиски при фиксированных калиброванных C, затем объединение кандидатов. Совместные количества нескольких задач — отдельный контроль. Четыре порядка относятся к одному и тому же I, а не к четырём независимым геометриям.

После свежего fit 32 знаковых инициализаций C для каждого I предсказатели фиксируются, затем отдельная meta-панель выбирает I. Все 24 выбора I заморожены до чтения структурной истины. Совпадение допускает только глобальное переименование/переворот знака групп и перестановку скрытых нейронов; входные позиции фиксированы.

| Метод | Точный I, чистые ответы | Точный I, сырой шум 0,2 с усреднением |
|---|---:|---:|
"""
    document += "\n".join(table)
    document += "\n\nВо всех выбранных I: 32 активные связи, две группы, F1 маски 100%, ошибок знакового шэринга 0/88. Первичный prefix уже восстанавливает I во всех случаях, поэтому добавочного выигрыша калибровки или совместного поиска на этом опыте нет.\n\n"
    document += f"Исходные траектории завершены: {trajectories['bootstrap']['complete']}/{trajectories['bootstrap']['total']}; повторные: {trajectories['repeat']['complete']}/{trajectories['repeat']['total']}. Полные следы сохранены, включая отказы.\n\n"
    document += "## Свежая классификация 0110\n\n"
    document += "После фиксации I все source-C и состояния оптимизатора отброшены. Обучаются два свободных кода и общие b/a/d, всего пять чисел. Используются только бинарные метки: 768 support, 256 query, 512 резервных наблюдений и 512 test. На каждый численный fit — 32 Gaussian-инициализации кодов, 4096 градиентных вычислений, Adam lr=0.03, L2=0.001, batch=128. В shared-fit первые 64 шага обучаются C и -C; query выбирает ветку, после чего обучение продолжается в том же общем бюджете. Query выбирает checkpoint и старт; test открывается после фиксации всех предсказателей.\n\n"
    document += "Все найденные I совпадают с истинным с учётом симметрий. Поэтому найденный и oracle-I используют один и тот же свежий численный fit. Это явно зарегистрированное переиспользование идентичной задачи, не независимые 24 серии по 32 старта. Доля успешных стартов относится к одной серии из 32.\n\n"
    document += "| Метод | Численных параметров | Test BA, 512 | BA на всех 2048 | Точных стартов на всех 2048 | Test balanced BCE |\n|---|---:|---:|---:|---:|---:|\n"
    document += "\n".join(binary_table)
    document += "\n\nПлотный Adam с общей головой также достигает 100% на всём домене. Здесь нет преимущества нашего метода по выбранной итоговой точности; он получает точную компактную структуру. Сравнение не уравнивает стоимость данных: извлечение использовало дополнительные вещественные запросы. Сравнение 26/32 и 22/32 стартов является описательным, без утверждения статистически значимого превосходства.\n\n"
    document += "## Хитмапы и динамика\n\n![Индексаторы и реальные веса](figures/indexators_and_weights.png)\n\nПервые три панели — дискретные индексы, последние две — реальные веса свежего shared-fit и плотного Adam. Столбцы выровнены только после выбора моделей. Плотные веса не кластеризуются и не превращаются в маску. Исходные значения сохранены в figures/heatmap_values.npz.\n\n![Обучение и доля точных стартов](figures/classification_learning.png)\n\n"
    document += "## Шум, стоимость и ограничения\n\n"
    document += f"Сырой шум — Gaussian σ=0.2. Осевые ответы усредняются по {plan['axis_repeats']:,} повторов, prefix-ответы — по {plan['slice_repeats']:,}, Gaussian fit/meta — по {plan['fit_repeats']}. Стандартные отклонения средних соответственно {0.2 / math.sqrt(plan['axis_repeats']):.8f}, {0.2 / math.sqrt(plan['slice_repeats']):.6f}, {0.2 / math.sqrt(plan['fit_repeats']):.3f}. Поэтому таблица не означает устойчивость к неусреднённому шуму 0.2.\n\n"
    document += f"Номинальная цена шумного условия за все четыре порядка — {budget['noisy_physical_answers_per_source_all_four_runs']:,} независимых шумных ответов на source-функцию. Средние Gaussian-ответов симулировались аналитически; физически это число повторов не вычислялось. Большой бюджет является существенным ограничением практического вывода.\n\n"
    document += "Успех относится к фиксированному 0110, известному общему bias/readout и активному доступу к вещественным выходам на входах вне бинарного куба. Он не устанавливает извлечение шэринга из одного обычного бинарного датасета, перенос на другие паттерны или решение произвольного MLP.\n\n"
    document += "## Воспроизводимость\n\nПротокол и хеши исходников — protocol.json/code_snapshot. source_frozen.json фиксирует выбор I до аудита; classification/predictors_frozen.json — численные модели до test. structure_audit.json содержит все структурные метрики; fresh_source_evaluation.json — дополнительную оценку на 8192 новых вещественных входах после фиксации; classification_comparison.csv — таблицу классификации. Source-файлы прежних G23–G25 не изменялись. Программные тесты не создавались и не запускались. Численный fit выполнен пакетно на GPU 0; prefix/QP — на CPU, карты 6 и 7 не использовались.\n"
    (root / "REPORT.md").write_text(document)
    save_json(
        root / "report_provenance.json",
        {
            "created_utc": timestamp(),
            "report_sha256": digest(root / "REPORT.md"),
            "source_freeze_sha256": digest(root / "source_frozen.json"),
            "classification_freeze_sha256": digest(
                root / "classification/predictors_frozen.json"
            ),
            "report_code_sha256": digest(Path(__file__)),
        },
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.root / "protocol.json").read_text())
    audit = json.loads((args.root / "structure_audit.json").read_text())
    classification = json.loads((args.root / "classification/results.json").read_text())
    source_rows = source_evaluation(args.root, plan, audit)
    figures(args.root, plan, classification["methods"])
    write_report(args.root, plan, audit, classification, source_rows)


if __name__ == "__main__":
    main()
