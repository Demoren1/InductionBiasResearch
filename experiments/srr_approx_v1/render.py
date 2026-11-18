"""Figures and a readable report from already frozen, scored results."""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .real import DEST
from .report import half_interval, write_csv

COLORS = {"dense": "#637080", "g34": "#2869ad", "new": "#d27725", "hybrid": "#3f8a65"}
NAMES = {"omniglot": "Omniglot", "mnist": "MNIST"}


def pooled(first, confirmation):
    rows, pairs = [], []
    for dataset in NAMES:
        left = dict(np.load(DEST / f"{dataset}_episode_scores.npz"))
        right = dict(np.load(DEST / "confirmation" / f"{dataset}_episode_scores.npz"))
        values = {}
        for method in ("dense", "dense6", "g34", "new", "hybrid"):
            accuracy = np.r_[left[f"{method}_accuracy"], right[f"{method}_accuracy"]]
            ce = np.r_[left[f"{method}_ce"], right[f"{method}_ce"]]
            values[method] = accuracy, ce
            old = next(
                row
                for row in first["comparison"]
                if row["dataset"] == dataset and row["method"] == method
            )
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "name": old["name"],
                    "codes": old["codes"],
                    "episodes": len(accuracy),
                    "accuracy": float(accuracy.mean()),
                    "accuracy_half_ci": half_interval(accuracy),
                    "ce": float(ce.mean()),
                    "ce_half_ci": half_interval(ce),
                }
            )
        for method in ("new", "hybrid"):
            for baseline in ("dense", "g34"):
                accuracy = values[method][0] - values[baseline][0]
                ce = values[method][1] - values[baseline][1]
                pairs.append(
                    {
                        "dataset": dataset,
                        "first": method,
                        "second": baseline,
                        "accuracy_difference": float(accuracy.mean()),
                        "accuracy_half_ci": half_interval(accuracy),
                        "ce_difference": float(ce.mean()),
                        "ce_half_ci": half_interval(ce),
                    }
                )
    save_json(
        DEST / "pooled_frozen_policies.json",
        {
            "comparison": rows,
            "paired": pairs,
            "scope": "Only policies fixed before the first cohort; the later one-SE policy is not pooled retroactively",
        },
    )
    write_csv(DEST / "pooled_comparison.csv", rows)
    return rows, pairs


def save_figure(fig, name):
    for extension in ("png", "pdf"):
        fig.savefig(DEST / f"{name}.{extension}", dpi=170)
    plt.close(fig)


def figures(first, confirmation, summary, synthetic):
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, dataset in zip(axes, NAMES):
        lookup = {row["method"]: row for row in summary if row["dataset"] == dataset}
        methods = ("dense", "g34", "new")
        values = [100 * lookup[key]["accuracy"] for key in methods]
        errors = [100 * lookup[key]["accuracy_half_ci"] for key in methods]
        bars = ax.bar(
            range(3),
            values,
            yerr=errors,
            capsize=4,
            color=[COLORS[key] for key in methods],
        )
        for bar, key, error in zip(bars, methods, errors):
            row = lookup[key]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + error + 0.7,
                f"{100 * row['accuracy']:.2f}%",
                ha="center",
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                4,
                f"m={row['codes']}",
                ha="center",
                color="white",
                fontsize=10,
            )
        ax.set_xticks(range(3), ("Dense\n24 starts", "G34", "aSRR\nown bank"))
        ax.set_ylim(0, 105)
        ax.set_ylabel("Fresh-task query accuracy (%)")
        ax.set_title(f"{NAMES[dataset]}: 768 tasks")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Unchanged candidates across two fresh cohorts (256 + 512)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "real_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, dataset in zip(axes, NAMES):
        for offset, result, color, label in (
            (-0.07, first, "#d27725", "First 256 tasks"),
            (0.07, confirmation, "#2869ad", "Additional 512 tasks"),
        ):
            for position, baseline in enumerate(("dense", "g34")):
                row = next(
                    row
                    for row in result["paired"]
                    if row["dataset"] == dataset
                    and row["first"] == "new"
                    and row["second"] == baseline
                )
                ax.errorbar(
                    100 * row["accuracy_difference"],
                    position + offset,
                    xerr=100 * row["accuracy_half_ci"],
                    fmt="o",
                    capsize=4,
                    color=color,
                    label=label if position == 0 else None,
                )
        ax.axvline(0, color="#555", ls="--")
        ax.set_yticks((0, 1), ("aSRR − Dense24", "aSRR − G34"))
        ax.set_xlabel("Paired accuracy difference (percentage points)")
        ax.set_title(NAMES[dataset])
        ax.grid(axis="x", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Each cohort evaluated without changing the aSRR candidate")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "paired_confirmation")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for ax, dataset in zip(axes, NAMES):
        for family, color, label in (
            ("asrr_local", "#d27725", "aSRR local"),
            ("asrr_input", "#3f8a65", "aSRR input groups"),
            ("asrr_global", "#b15377", "aSRR global"),
            ("g34", "#2869ad", "G34"),
        ):
            curve = {}
            for row in first["choices"]["development"]:
                if (
                    row["dataset"] == dataset
                    and row["family"] == family
                    and (
                        row["codes"] not in curve
                        or row["ce"] < curve[row["codes"]]["ce"]
                    )
                ):
                    curve[row["codes"]] = row
            ordered = [curve[key] for key in sorted(curve)]
            ax.plot(
                [row["codes"] for row in ordered],
                [row["ce"] for row in ordered],
                "o-",
                color=color,
                label=label,
            )
        dense = first["choices"]["choices"][dataset]["dense"]
        ax.axhline(dense["ce"], color=COLORS["dense"], ls="--", label="Dense24")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Code budget m")
        ax.set_ylabel("Development query CE")
        ax.set_title(NAMES[dataset])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Source-only candidate hierarchies; budget selected on development")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "development_budgets")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    control_names = (
        "randomized",
        "randomized_signs",
        "randomized_both",
        "positive_signs",
    )
    labels = ("Shuffle groups", "Random signs", "Both randomized", "All signs positive")
    for ax, dataset in zip(axes, NAMES):
        values = [
            next(
                row
                for row in first["paired"]
                if row["dataset"] == dataset
                and row["first"] == "new"
                and row["second"] == name
            )
            for name in control_names
        ]
        ax.errorbar(
            [100 * row["accuracy_difference"] for row in values],
            range(4),
            xerr=[100 * row["accuracy_half_ci"] for row in values],
            fmt="o",
            capsize=4,
            color=COLORS["new"],
        )
        ax.axvline(0, color="#555", ls="--")
        ax.set_yticks(range(4), labels)
        ax.set_xlabel("aSRR minus control (percentage points)")
        ax.set_title(NAMES[dataset])
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("First 256 tasks: the learned grouping matters")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "structure_ablation")

    titles = {
        "balanced": "Balanced global groups",
        "unbalanced": "Unequal group frequencies",
        "approximate": "Strong task-specific deformation",
        "input_groups": "Common input groups",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, scenario in zip(axes.ravel(), titles):
        for method, color, label in (
            ("dense", COLORS["dense"], "Dense ridge"),
            ("g34", COLORS["g34"], "Practical G34 bank"),
            ("asrr", COLORS["new"], "aSRR"),
            ("asrr_a1", COLORS["hybrid"], "aSRR + A1 bank"),
        ):
            curve = [
                row
                for row in synthetic["aggregate"]
                if row["scenario"] == scenario and row["method"] == method
            ]
            ax.plot(
                [row["noise"] for row in curve],
                [row["mean_mse"] for row in curve],
                "o-",
                color=color,
                label=label,
            )
        ax.set_yscale("log")
        ax.set_xlabel("Source direction measurement noise")
        ax.set_ylabel("Fresh noiseless-function MSE")
        ax.set_title(titles[scenario])
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Controlled weights: ranks 2/5, eight structures per scenario")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "synthetic_recovery")


def markdown(first, confirmation, summary, pairs, synthetic):
    lines = [
        "# Приближённый SRR: теория, синтетика и свежий перенос",
        "",
        "15 ноября 2026. Рабочее название — **aSRR** (approximate Sign–Ratio Recovery). Рукопись не менялась.",
        "",
        "**Результат:** заранее выбранные aSRR-кандидаты обошли Dense и практический G34 по accuracy и CE на Omniglot и MNIST. Выигрыш подтвердился отдельно на 256 и 512 новых задачах каждого датасета. Основное изменение — приближённые локальные группы вместо требования полного точного восстановления; число кодов выбирается на development. Дополнительный выбор более простой структуры улучшил accuracy, но на Omniglot его преимущество по CE не установлено.",
        "",
        "## Что изменилось",
        "",
        "Вместо полного точного восстановления строится иерархия приближённых signed-групп. Сомнительный вес сохраняет отдельный код при большом бюджете; нулевая связь принудительно не вводится. Новый signed-профиль учитывает весь вектор весов по задачам. Локальное семейство объединяет позиции внутри нейрона, сохраняя независимые коды между нейронами; input-семейство изучает общие группы входов с разными кодами в разных нейронах. Обе конструкции проверяются свежим численным обучением.",
        "",
        "Основной теоретический вывод: корректное локальное разбиение является уточнением глобального. Пропущенное объединение увеличивает число параметров, но не вносит bias; ложное объединение может его внести. Для линейного Gaussian контроля выведен точный критерий, когда ненулевой bias компенсируется снижением дисперсии. Отдельно доказаны эквивалентность signed-расстояний с фактором sqrt(2), граница ReLU-функции через остаток весов и границы использования selection. [Выводы и доказательства](APPROXIMATE_SRR_THEORY_2026-11-15.md).",
        "",
        "## Фиксированные кандидаты на двух новых когортах",
        "",
        "Первая когорта содержит 256 задач каждого датасета, следующая — 512. Представленные здесь aSRR-кандидаты, m, численные правила и baseline-кандидаты были зафиксированы до первой когорты и между когортами не менялись. Объединённая оценка использует 768 эпизодов; интервалы условны фиксированным source/development данным и классам.",
        "",
        "| Датасет | Метод | m | Accuracy, % (95% CI) | Query CE |",
        "|---|---|---:|---:|---:|",
    ]
    names = {
        "dense": "Dense, 24 starts",
        "dense6": "Dense, 6 starts",
        "g34": "G34, включая дополнительное A1-уточнение",
        "new": "aSRR, собственный банк",
        "hybrid": "Общий банк: aSRR + A1 + G34 + Dense",
    }
    for row in summary:
        lines.append(
            f"| {NAMES[row['dataset']]} | {names[row['method']]} | {row['codes']} | {100 * row['accuracy']:.2f} ± {100 * row['accuracy_half_ci']:.2f} | {row['ce']:.5f} |"
        )
    lines += [
        "",
        "| Парное сравнение | Omniglot, п.п. (95% CI) | MNIST, п.п. (95% CI) |",
        "|---|---:|---:|",
    ]
    for baseline in ("dense", "g34"):
        cells = []
        for dataset in NAMES:
            row = next(
                row
                for row in pairs
                if row["dataset"] == dataset
                and row["first"] == "new"
                and row["second"] == baseline
            )
            cells.append(
                f"{100 * row['accuracy_difference']:+.2f} ± {100 * row['accuracy_half_ci']:.2f}"
            )
        lines.append(f"| aSRR − {names[baseline]} | {' | '.join(cells)} |")
    lines += [
        "",
        "![Качество](../../artifacts/srr_approx_real_2026-11-15/real_comparison.png)",
        "",
        "![Две когорты](../../artifacts/srr_approx_real_2026-11-15/paired_confirmation.png)",
        "",
        "На Omniglot выбран исходный локальный aSRR m2048; на MNIST — исходное input-разбиение m1024. Дополнительная проекция A1 не улучшила их development CE, поэтому исходные семена сохранены. A1 улучшил G34-контроль на MNIST до m8192. Таким образом, исследуется новый банк и инициализация для общего процесса, а не утверждается необходимость каждого шага A1 для выигрыша.",
        "",
        "Общий минимизирующий development CE банк выбрал aSRR на Omniglot, но G34 на MNIST. Его MNIST-результат поэтому совпадает с G34. Отдельный aSRR-кандидат — самостоятельная заранее выбранная процедура; его нельзя задним числом выдавать за ответ общего банка.",
        "",
        "## Дополнительное правило выбора сложности",
        "",
        "После первой test-когорты предложен paired one-SE выбор: допустить модели в пределах одной standard error парной development-разности от минимальной CE, затем выбрать минимальное m. То же правило применено к G34, включая Dense как no-sharing endpoint. Параметр one-SE не подбирался. Это эвристика с отдельной проверкой на 512 новых задачах; первая когорта не является её независимой оценкой.",
        "",
        "| Датасет | Правило | Выбранная модель | m | Accuracy, % (95% CI) | CE |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in confirmation["comparison"]:
        if row["method"] in ("simple_hybrid", "simple_g34"):
            label = (
                "Общий банк, paired one-SE"
                if row["method"] == "simple_hybrid"
                else "G34, paired one-SE"
            )
            lines.append(
                f"| {NAMES[row['dataset']]} | {label} | {row['name']} | {row['codes']} | {100 * row['accuracy']:.2f} ± {100 * row['accuracy_half_ci']:.2f} | {row['ce']:.5f} |"
            )
    lines += [
        "",
        "Эта проверка selection отделена от подтверждения неизменённых исходных кандидатов; новые правила не объединяются ретроспективно с первой когортой.",
        "",
        "На второй когорте исходный aSRR с m2048/1024 получил 81.13%/96.28%, Dense с 24 стартами — 80.27%/96.03%, G34 — 79.53%/96.03%. Для paired one-SE общий банк выбрал m512/1024: 81.45%/96.28%. На Omniglot разность CE этого более простого варианта с Dense равна −0.00160 ± 0.02386, с исходным сильным G34 — −0.01503 ± 0.02448 (парные 95% интервалы): преимущество по CE не установлено. Для исходного локального m2048 преимущество над обоими контролями подтверждается по обеим метрикам.",
        "",
        "## Что даёт именно извлечённая структура",
        "",
        "Контроли первой когорты сохраняют число кодов, размеры групп, их появления в нейронах и шаблон знаков. Случайно меняются только позиции групп; отдельные абляции меняют знаки или оба компонента. Они зафиксированы до открытия test-ответов.",
        "",
        "| Датасет | Вариант | Accuracy, % (95% CI) |",
        "|---|---|---:|",
    ]
    labels = {
        "new": "aSRR",
        "randomized": "Случайные позиции групп, прежние знаки",
        "randomized_signs": "Прежние группы, случайные знаки",
        "randomized_both": "Случайные группы и знаки",
        "positive_signs": "Прежние группы, все знаки положительные",
        "legacy": "Прежний SRR без фильтра",
        "msr": "Сохранённый MSR-FC базис",
    }
    for row in first["comparison"]:
        if row["method"] in labels:
            lines.append(
                f"| {NAMES[row['dataset']]} | {labels[row['method']]} | {100 * row['accuracy']:.2f} ± {100 * row['accuracy_half_ci']:.2f} |"
            )
    lines += [
        "",
        "Случайное перемешивание групп ухудшает качество при тех же бюджетах и сохранённых знаках. На Omniglot выигрыш learned signs относительно всех положительных знаков не установлен; произвольные случайные знаки вредят. На MNIST input-кандидат изначально положительный, поэтому positive-sign контроль совпадает с ним. Это не свидетельство полезности произвольного кодбука или только сокращения числа параметров.",
        "",
        "![Абляции](../../artifacts/srr_approx_real_2026-11-15/structure_ablation.png)",
        "",
        "## Контролируемая синтетика",
        "",
        "32 независимые структуры, четыре сценария, ранги2/5 и три парных уровня measurement noise дают96 условий. Входы восстановителя — контролируемые noisy dense-направления, а не новые SGD-обучения. G34 здесь — конечный практический full-support банк с четырьмя инициализациями и12 шагами projection на бюджет, не прежний exhaustive native алгоритм. Все методы имеют одинаковую сетку8/16/24/32/64/96 и новые численные функции. Dense получает ту же ridge-сетку, численный выбор на H; структура и m выбираются на отдельных development-функциях.",
        "",
        "| Сценарий | Шум | Dense MSE | G34 MSE | aSRR MSE | aSRR+A1 MSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in ("balanced", "unbalanced", "approximate", "input_groups"):
        for noise in (0, 0.01, 0.04):
            values = {
                row["method"]: row
                for row in synthetic["aggregate"]
                if row["scenario"] == scenario and row["noise"] == noise
            }
            lines.append(
                f"| {scenario} | {noise:g} | {values['dense']['mean_mse']:.5f} | {values['g34']['mean_mse']:.5f} | {values['asrr']['mean_mse']:.5f} | {values['asrr_a1']['mean_mse']:.5f} |"
            )
    lines += [
        "",
        "При сильной индивидуальной деформации sharing не обязан помогать: выбор обычно возвращается к Dense, иногда проигрывает из-за конечной selection-выборки. Это согласуется с bias–variance расчётом. В других сценариях новый банк устойчивее к measurement noise. Эти development-результаты не доказывают превосходства над любым бюджетом полного native G34.",
        "",
        "### Восстановление самих групп",
        "",
        "При measurement noise 0.04 можно проверить выбранные индексаторы по скрытым истинным группам. Precision — доля предложенных общих пар с правильной группой и относительным знаком; recall — доля истинных общих пар, найденных с правильным знаком. Ни истинный индексатор, ни этот аудит не используются для выбора. В каждой строке средние по восьми независимым структурам (ранги 2 и 5, по четыре структуры). Среднее m может быть нецелым; число кодов выбиралось отдельно для каждой структуры.",
        "",
        "| Сценарий | Метод | Среднее m | Signed precision, % | Signed recall, % |",
        "|---|---|---:|---:|---:|",
    ]
    for scenario in ("balanced", "unbalanced", "input_groups"):
        for method, label in (
            ("g34", "Практический G34"),
            ("asrr", "aSRR"),
            ("asrr_a1", "aSRR+A1, общий банк"),
        ):
            cases = [
                row
                for row in synthetic["cases"]
                if row["scenario"] == scenario
                and row["noise"] == 0.04
                and row["method"] == method
            ]
            count = np.mean([row["codes"] for row in cases])
            precision = np.mean([row["signed_pair_precision"] for row in cases])
            recall = np.mean([row["signed_pair_recall"] for row in cases])
            lines.append(
                f"| {scenario} | {label} | {count:g} | {100 * precision:.2f} | {100 * recall:.2f} |"
            )
    lines += [
        "",
        "В сбалансированном случае aSRR восстанавливает существенно больше общих пар при сопоставимой precision. При неравномерных группах остаётся компромисс precision/recall; полного доминирования нет. В input-сценарии новое семейство восстанавливает группы точно в этой серии. Для сильной task-specific деформации исходные группы уже не задают точные равенства весов: низкий recall выбранного Dense здесь не означает ошибку выбора.",
        "",
        "![Синтетика](../../artifacts/srr_approx_real_2026-11-15/synthetic_recovery.png)",
        "",
        "## Протокол, масштаб и ограничения",
        "",
        "Omniglot:784–16–5, пять классов, по10 train+5 numerical validation+5 query изображений на класс. MNIST:784–16–2, по100+50+100 на класс. Source/development/test классы не пересекаются. На MNIST source0–4, development5–6, test-пары7–9; на Omniglot Vinyals1028/172/423. Source dense модели взяты из сохранённых обучений: Omniglot16 лучших H-задач, MNIST10 задач по8000 обучающих изображений. Все численные параметры новых задач обучаются с нуля; реплики source не переносят значения target-кодов.",
        "",
        "Согласуются четыре source-реплики; их направления усредняются. Это практическая стабилизация, а не certified оценка одного истинного ReLU-параметра. Структурные семейства и весь диапазон m16–8192 задаются до test; no-sharing endpoint12544 представлен тем же Dense. Кодовые и знаковые gauges канонизируются, чтобы переименование не давало скрытых дополнительных стартов.",
        "",
        "Численное обучение: Adam200, LR.003/.01/.03, шесть стартов; checkpoints и старты выбираются только на numerical validation. Дополнительно Dense получает четыре таких пула с выбором по той же H —24 старта. По development основной Dense-контроль выбирает этот вариант. G34 сохраняет исходные банки, их capacity-refinements и дополнительное A1-уточнение на том же source-consensus. Расширение больших m не является полным новым native source-search. Сохранённый MSR-FC — только вторичный basis-only контроль первой когорты; новые MSR source/CNN-запуски не выполнялись.",
        "",
        "Локальная конструкция не требует глобальной матрицы расстояний по всем весам: последовательно строятся H иерархий из D позиций. При T source-профилях и фиксированной сетке бюджетов её порядок работы O(H D² T + H D²), память O(D² + DH T). Глобальный вариант требует O((DH)²) памяти; он остаётся исследовательским семейством, а не обязательным этапом для локального варианта. Время конкурентных GPU jobs не является сравнением скорости методов.",
        "",
        "Все среды классов уже изучались раньше; новые эпизоды дают условное подтверждение на этих средах, а не утверждение о любых датасетах. Пересечения исходных изображений между эпизодами возможны. Exact recovery на реальных данных неизвестен. Линейная теорема не даёт безусловной гарантии для ReLU; новое численное обучение и перенос проверены экспериментально.",
        "",
        "## Файлы",
        "",
        "Теория: APPROXIMATE_SRR_THEORY_2026-11-15.md. Код: experiments/srr_approx_v1. Контролируемые опыты: artifacts/srr_approx_synthetic_2026-11-15. Реальные данные: artifacts/srr_approx_real_2026-11-15 и confirmation/. Raw-панели, индексы и fitted predictions: artifacts/local/srr_approx_real_2026-11-15. Заморозки предшествуют открытию query-ответов; дополнительный one-SE выбор зарегистрирован отдельно после первой когорты. Протоколы и исходники с хэшами сохранены. Software tests не создавались и не запускались; выполнены исследовательские эксперименты и Ruff.",
        "",
    ]
    body = "\n".join(lines)
    Path("paper/theory/APPROXIMATE_SRR_RESULTS_2026-11-15.md").write_text(body)
    (DEST / "REPORT.md").write_text(
        body.replace("../../artifacts/srr_approx_real_2026-11-15/", "").replace(
            "](APPROXIMATE_SRR_THEORY_2026-11-15.md)",
            "](../../paper/theory/APPROXIMATE_SRR_THEORY_2026-11-15.md)",
        )
    )


def main():
    first = json.loads((DEST / "results.json").read_text())
    confirmation = json.loads((DEST / "confirmation/results.json").read_text())
    synthetic = json.loads(
        Path("artifacts/srr_approx_synthetic_2026-11-15/results.json").read_text()
    )
    summary, pairs = pooled(first, confirmation)
    figures(first, confirmation, summary, synthetic)
    markdown(first, confirmation, summary, pairs, synthetic)
    snapshot = DEST / "executed_source"
    snapshot.mkdir(exist_ok=True)
    paths = (
        list(Path(__file__).parent.glob("*.py"))
        + list(Path(__file__).parent.glob("*.md"))
        + [
            Path("experiments/omniglot_prior_v1/model.py"),
            Path("experiments/srr_real_v1/predictive.py"),
            Path("paper/theory/APPROXIMATE_SRR_THEORY_2026-11-15.md"),
        ]
    )
    for path in paths:
        shutil.copy2(path, snapshot / path.name)
    save_json(
        DEST / "runtime_sources.json", {str(path): digest(path) for path in paths}
    )
    print(json.dumps({"pooled": summary, "paired": pairs}, indent=2), flush=True)


if __name__ == "__main__":
    main()
