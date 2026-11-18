"""Freeze real predictions, score once, and render from saved metrics."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.stats import t

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.real import combine_dense
from experiments.srr_approx_v1.report import half_interval, write_csv

from .clustering import METHODS, VENDOR
from .real import ROOT
from .synthetic import DEST

LABELS = {
    "ward": "Ward",
    "single": "Single",
    "complete": "Complete",
    "lloyd": "Lloyd k-means",
    "signed": "Signed Lloyd",
    "emeans": "e-means",
    "ward_previous": "Previous aSRR",
    "g34": "G34 / Dense",
    "g34_compressed": "G34",
    "dense": "Dense",
    "dense6": "Dense, 6 starts",
    "ward_adaptive": "Adaptive-local Ward",
    "union": "Full bank, min CE",
    "simple_union": "Full bank, one-SE",
}
COLORS = dict(
    zip(METHODS, ("#cf7926", "#8673bd", "#a64b79", "#2c8b91", "#51843b", "#2a66ae"))
)
COLORS.update({"g34": "#536073", "dense": "#929ca8"})


def score():
    if (DEST / "real_results.json").exists():
        raise RuntimeError("Real query labels already scored")
    choices = json.loads((DEST / "real_choices_frozen.json").read_text())
    frozen = {}
    for dataset, policies in choices["policies"].items():
        combine_dense(dataset, "test", study_root=ROOT)
        for name in set(policies.values()):
            path = ROOT / dataset / f"fits/test/{name}.npz"
            frozen[f"{dataset}/{name}"] = digest(path)
    save_json(
        DEST / "real_predictions_frozen.json",
        {
            "predictions": frozen,
            "choices_sha256": digest(DEST / "real_choices_frozen.json"),
            "query_labels_read": False,
        },
    )
    rows, paired = [], []
    for dataset, policies in choices["policies"].items():
        truth = np.load(ROOT / dataset / "private/query_answers.npz")["y"]
        episodes, values = {}, {}
        for method, name in policies.items():
            path = ROOT / dataset / f"fits/test/{name}.npz"
            if digest(path) != frozen[f"{dataset}/{name}"]:
                raise RuntimeError("Prediction changed after freeze")
            prediction = np.load(path)["prediction"]
            accuracy = (prediction.argmax(-1) == truth.argmax(-1)).mean(-1)
            ce = (logsumexp(prediction, -1) - (prediction * truth).sum(-1)).mean(-1)
            episodes[f"{method}_accuracy"] = accuracy
            episodes[f"{method}_ce"] = ce
            record = choices["selected"][dataset][method]
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "name": name,
                    "codes": record["codes"],
                    "accuracy": float(accuracy.mean()),
                    "accuracy_half_ci": half_interval(accuracy),
                    "ce": float(ce.mean()),
                    "ce_half_ci": half_interval(ce),
                    "episodes": len(ce),
                }
            )
            values[method] = accuracy, ce
        for method in (
            *METHODS,
            *(f"raw_{method}" for method in METHODS),
            "union",
            "simple_union",
        ):
            for baseline in ("ward_previous", "g34", "g34_compressed", "dense"):
                acc = values[method][0] - values[baseline][0]
                ce = values[method][1] - values[baseline][1]
                paired.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "baseline": baseline,
                        "accuracy_difference": float(acc.mean()),
                        "accuracy_half_ci": half_interval(acc),
                        "ce_difference": float(ce.mean()),
                        "ce_half_ci": half_interval(ce),
                    }
                )
        np.savez_compressed(DEST / f"{dataset}_real_episode_scores.npz", **episodes)
    save_json(DEST / "real_results.json", {"comparison": rows, "paired": paired})
    write_csv(DEST / "real_comparison.csv", rows)
    write_csv(DEST / "real_paired.csv", paired)
    for row in rows:
        if not row["method"].startswith(("raw_", "simple_")):
            print(json.dumps(row), flush=True)


def figures(synthetic, real, choices):
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, scenario in zip(
        axes.flat, ("balanced", "unbalanced", "approximate", "input_groups")
    ):
        for method in (*METHODS, "g34", "dense"):
            group = [
                row
                for row in synthetic["aggregate"]
                if row["scenario"] == scenario and row["method"] == method
            ]
            axis.plot(
                [row["noise"] for row in group],
                [row["mse"] for row in group],
                marker="o",
                label=LABELS[method],
                color=COLORS.get(method),
                linestyle="--" if method in ("dense", "g34") else "-",
            )
        axis.set(
            title=scenario,
            xlabel="Source measurement noise",
            ylabel="Fresh clean-function MSE",
            yscale="log",
        )
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(
        "New synthetic structures, ranks 2/5; scope and m selected on development"
    )
    save_figure(fig, "synthetic_mse")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric in zip(axes, ("signed_recall", "mse")):
        for method in METHODS:
            subset = [
                row
                for row in synthetic["aggregate"]
                if row["scenario"] == "balanced" and row["method"] == method
            ]
            axis.plot(
                [row["noise"] for row in subset],
                [
                    row[metric] * (100 if metric == "signed_recall" else 1)
                    for row in subset
                ],
                marker="o",
                label=LABELS[method],
                color=COLORS[method],
            )
        axis.set(
            xlabel="Source measurement noise",
            ylabel="Signed sharing recall (%)"
            if metric == "signed_recall"
            else "Fresh clean-function MSE",
        )
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    save_figure(fig, "synthetic_recovery")
    if real is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    methods = (*METHODS, "ward_previous", "g34_compressed", "dense")
    for axis, dataset in zip(axes, ("omniglot", "mnist")):
        group = {
            row["method"]: row
            for row in real["comparison"]
            if row["dataset"] == dataset
        }
        axis.errorbar(
            np.arange(len(methods)),
            [100 * group[method]["accuracy"] for method in methods],
            yerr=[100 * group[method]["accuracy_half_ci"] for method in methods],
            fmt="o",
            capsize=3,
        )
        axis.set_xticks(
            np.arange(len(methods)),
            [LABELS[method] for method in methods],
            rotation=45,
            ha="right",
        )
        axis.set(title=dataset.title(), ylabel="Fresh-task accuracy (%)")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "512 new tasks per dataset; original seeds and Algorithm1 refinements retained"
    )
    save_figure(fig, "real_accuracy")
    for baseline, name in (
        ("ward_previous", "real_paired"),
        ("ward", "matched_ward_paired"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for axis, dataset in zip(axes, ("omniglot", "mnist")):
            for number, method in enumerate(METHODS):
                row = next(
                    row
                    for row in real["paired"]
                    if row["dataset"] == dataset
                    and row["method"] == method
                    and row["baseline"] == baseline
                )
                axis.errorbar(
                    100 * row["accuracy_difference"],
                    number,
                    xerr=100 * row["accuracy_half_ci"],
                    fmt="o",
                    capsize=3,
                    color=COLORS[method],
                )
            axis.axvline(0, color="grey", linestyle="--")
            axis.set_yticks(range(len(METHODS)), [LABELS[method] for method in METHODS])
            label = (
                "previous aSRR"
                if baseline == "ward_previous"
                else "Ward in matched candidate families"
            )
            axis.set(
                title=dataset.title(),
                xlabel=f"Accuracy difference (pp), paired 95% CI\nBaseline: {label}",
            )
            axis.grid(axis="x", alpha=0.2)
        save_figure(fig, name)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, dataset in enumerate(("omniglot", "mnist")):
        for method in METHODS:
            name = choices["policies"][dataset][method]
            if name == "dense24":
                name = "dense"
            path = ROOT / dataset / f"fits/test/{name}.trace.npz"
            if not path.exists():
                continue
            trace = np.load(path)
            step = trace["step"] if "step" in trace else trace["steps"]
            for row, part in enumerate(("train", "validation")):
                loss = trace[part] if part in trace else trace[f"{part}_history"]
                loss = loss.reshape(len(step), -1).mean(1)
                axes[row, column].plot(
                    step, loss, label=LABELS[method], color=COLORS[method]
                )
                axes[row, column].set(
                    title=f"{dataset}: {part}",
                    xlabel="Numerical optimizer step",
                    ylabel="Cross entropy",
                )
                axes[row, column].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    save_figure(fig, "real_training_losses")
    fig, axis = plt.subplots(figsize=(7, 4))
    for method in METHODS:
        name = choices["policies"]["omniglot"][method]
        if name == "dense24":
            name = "dense"
        trace = np.load(ROOT / "omniglot" / f"fits/test/{name}.trace.npz")
        axis.plot(
            trace["step"],
            100 * trace["recall"],
            label=LABELS[method],
            color=COLORS[method],
        )
    axis.set(
        xlabel="Numerical optimizer step",
        ylabel="Within-task H accuracy (%)",
        title="Omniglot: mean across tasks and six numerical starts",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, ncol=2)
    save_figure(fig, "real_training_recall")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, dataset in zip(axes, ("omniglot", "mnist")):
        for method in METHODS:
            group = [
                row
                for row in choices["development"]
                if row["dataset"] == dataset
                and row["method"] == method
                and row["stage"] == "raw"
            ]
            counts = sorted({row["codes"] for row in group})
            axis.plot(
                counts,
                [
                    min(row["ce"] for row in group if row["codes"] == count)
                    for count in counts
                ],
                marker=".",
                label=LABELS[method],
                color=COLORS[method],
            )
        axis.set(
            title=dataset.title(),
            xlabel="Code budget m",
            ylabel="Development CE (best of local/input)",
            xscale="log",
        )
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    save_figure(fig, "development_budgets")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    reference = None
    for axis, method in zip(axes, ("ward", "complete", "lloyd", "emeans")):
        index = np.load(ROOT / "omniglot" / f"models/{method}_input_m512.npz")["index"][
            :, 0
        ]
        _, labels = np.unique(index, return_inverse=True)
        if reference is None:
            reference = labels.copy()
        else:
            overlap = np.zeros((32, 32), dtype=int)
            np.add.at(overlap, (reference, labels), 1)
            left, right = linear_sum_assignment(-overlap)
            mapping = np.empty(32, dtype=int)
            mapping[right] = left
            labels = mapping[labels]
        axis.imshow(
            labels.reshape(28, 28),
            cmap="turbo",
            vmin=0,
            vmax=31,
            interpolation="nearest",
        )
        axis.set(title=LABELS[method], xticks=[], yticks=[])
    fig.suptitle(
        "Matched budget: 32 input groups x16 independent neuron codes; colors aligned for display"
    )
    save_figure(fig, "matched_input_groups")


def save_figure(fig, name):
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(DEST / f"{name}.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def markdown(synthetic, real, choices):
    lines = [
        "# Кластеризаторы внутри aSRR",
        "",
        "15 ноября 2026. Сначала новая синтетика, затем новые реальные эпизоды. Рукопись не менялась.",
        "",
        "## Синтетика: шум 0.04",
        "",
        "32 новых независимых структуры, ранги 2/5, три парных уровня шума. Каждая строка ниже — восемь структур сценария. Общий бюджет и численный fitter; выбор семейства и m только на отдельных development-функциях.",
        "",
        "| Сценарий | Метод | Fresh MSE | Signed recall, % | Среднее m |",
        "|---|---|---:|---:|---:|",
    ]
    if real is not None:
        headline = [
            "## Результат проверки",
            "",
            "Убедительного выигрыша по accuracy над прежним aSRR/Ward не установлено ни у одного нового кластеризатора. Обычный k-means получил на Omniglot 81.75% с m512 против 81.45% у прежнего локального aSRR с m2048; парная разность +0.30 ± 0.64 п.п. Для MNIST соответствующий k-means дал 96.24% с m4096 против 96.21% у aSRR с m1024; +0.03 ± 0.08 п.п. Это отдельные заранее зафиксированные процедуры, не победитель, выбранный по test.",
            "",
            "В новом сравнении с одинаковыми семействами кандидатов и правилом подбора m, где локальным нейронам выделяется m/H групп, k-means и e-means превосходят соответствующий Ward по Omniglot accuracy: +0.82 ± 0.63 и +0.72 ± 0.57 п.п. соответственно (отдельные парные 95% интервалы, без поправки на множественные сравнения). Их CE с этим Ward не различается убедительно. Сам matched Ward уступил прежнему адаптивному Ward, поэтому положительный результат против него нельзя выдавать за улучшение нашего прежнего полного aSRR. На MNIST выигрыша относительно Ward нет.",
            "",
            "E-means дал 81.65%/96.17%, с m1024/4096. Его преимущество на шумной сбалансированной синтетике не превратилось в подтверждённое улучшение относительно прежнего полного aSRR на реальных данных. Расширенный выбор по development CE предпочёл e-means на MNIST, хотя его новая accuracy оказалась немного ниже прежнего Ward; интервал включает ноль. One-SE выбор complete_m1024 на MNIST снизил accuracy относительно Ward на −0.14 ± 0.09 п.п. при сопоставимой CE. Автоматическое расширение банка не гарантирует лучший перенос.",
            "",
            "Все двенадцать A1-уточнений новых методов имели development CE хуже исходного выбранного кандидата; в test-политиках остались исходные индексаторы. Эти результаты не доказывают бесполезность A1 в других постановках.",
            "",
            "## Выбранная до test общая процедура",
            "",
        ]
        for dataset in ("omniglot", "mnist"):
            row = next(
                row
                for row in real["comparison"]
                if row["dataset"] == dataset and row["method"] == "union"
            )
            headline.append(
                f"{dataset}: полный банк по development CE выбрал {row['name']} (m={row['codes']}); на новой когорте accuracy {100 * row['accuracy']:.2f}%, CE {row['ce']:.5f}. Это заранее выбранная процедура, а не лучший test-результат среди кластеризаторов."
            )
        headline += [
            "",
            "| Парное сравнение общей процедуры | Omniglot, п.п. (95% CI) | MNIST, п.п. (95% CI) |",
            "|---|---:|---:|",
        ]
        for baseline in ("ward_previous", "g34_compressed", "dense"):
            cells = []
            for dataset in ("omniglot", "mnist"):
                pair = next(
                    row
                    for row in real["paired"]
                    if row["dataset"] == dataset
                    and row["method"] == "union"
                    and row["baseline"] == baseline
                )
                cells.append(
                    f"{100 * pair['accuracy_difference']:+.2f} ± {100 * pair['accuracy_half_ci']:.2f}"
                )
            headline.append(f"| Full bank − {LABELS[baseline]} | {' | '.join(cells)} |")
        headline += [""]
        lines[4:4] = headline
    for row in synthetic["aggregate"]:
        if row["noise"] == 0.04:
            lines.append(
                f"| {row['scenario']} | {LABELS.get(row['method'], row['method'])} | {row['mse']:.5f} | {100 * row['signed_recall']:.2f} | {row['mean_codes']:.1f} |"
            )
    lines += [
        "",
        "Для сбалансированного случая при шуме 0.04 e-means имеет меньшую среднюю MSE, но парная разность с Ward равна −0.00167 ± 0.00229 (95% t-интервал, восемь структур). Убедительного превосходства над Ward по этой небольшой серии не установлено. При сильной индивидуальной деформации Dense имеет лучшую среднюю MSE; при шуме 0.04 его выбрали все методы, а на меньшем шуме часть селекторов ошибочно предпочла sharing.",
        "",
        "![Синтетика](../../artifacts/srr_clustering_2026-11-15/synthetic_mse.png)",
        "",
        "![Восстановление](../../artifacts/srr_clustering_2026-11-15/synthetic_recovery.png)",
        "",
    ]
    if real is not None:
        lines += [
            "## Реальные задачи: 512 новых эпизодов каждого датасета",
            "",
            "Профили, m и правила выбора новых методов зафиксированы до генерации этой когорты. Дополнительно до любого test-fit и открытия ответов явно добавлен compressed G34 контроль, выбранный только по прежнему development банку; исходная G34/Dense процедура выбрала Dense на Omniglot. Этот дополнительный контроль и исходная заморозка сохранены отдельно. Для каждого нового метода сохраняются исходные семена и одинаковое A1-уточнение лучшего compressed-кандидата. Ниже выбор по development CE; raw-результаты без A1 и paired one-SE находятся отдельно в JSON/CSV.",
            "",
            "| Датасет | Метод | Выбранный I | m | Accuracy, % (95% CI) | CE |",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in real["comparison"]:
            if row["method"] in (
                *METHODS,
                "ward_previous",
                "g34",
                "g34_compressed",
                "dense",
                "simple_union",
                "union",
                "dense6",
            ):
                lines.append(
                    f"| {row['dataset']} | {LABELS.get(row['method'], row['method'])} | {row['name']} | {row['codes']} | {100 * row['accuracy']:.2f} ± {100 * row['accuracy_half_ci']:.2f} | {row['ce']:.5f} |"
                )
        lines += [
            "",
            "### Что изменил алгоритм 1 на development",
            "",
            "Выбор ниже выполнен до test. В колонке исходного варианта допускается общий Dense endpoint; A1 применён к лучшему compressed seed каждого метода. Test-таблицы raw и final также сохранены в CSV.",
            "",
            "| Датасет | Метод | Исходный CE | CE после A1 | Выбранный вариант |",
            "|---|---|---:|---:|---|",
        ]
        for dataset in ("omniglot", "mnist"):
            for method in METHODS:
                raw = choices["selected"][dataset][f"raw_{method}"]
                final = choices["selected"][dataset][method]
                refined = next(
                    row
                    for row in choices["development"]
                    if row["dataset"] == dataset
                    and row["method"] == method
                    and row["stage"] == "a1"
                )
                lines.append(
                    f"| {dataset} | {LABELS[method]} | {raw['ce']:.5f} | {refined['ce']:.5f} | {'A1' if final['stage'] == 'a1' else 'Исходный'} |"
                )
        lines += [
            "",
            "![Реальные задачи](../../artifacts/srr_clustering_2026-11-15/real_accuracy.png)",
            "",
            "![Парные различия](../../artifacts/srr_clustering_2026-11-15/real_paired.png)",
            "",
            "![Парные различия в новом общем наборе семейств](../../artifacts/srr_clustering_2026-11-15/matched_ward_paired.png)",
            "",
            "![Обучение](../../artifacts/srr_clustering_2026-11-15/real_training_losses.png)",
            "",
            "Кривые усредняются по задачам и всем шести численным стартам. Финальные предсказания в таблицах используют старт и checkpoint, выбранные по H; их CE не равна среднему всех стартов на кривой.",
            "",
            "![Бюджеты](../../artifacts/srr_clustering_2026-11-15/development_budgets.png)",
            "",
            "![Группы входов при одинаковом m](../../artifacts/srr_clustering_2026-11-15/matched_input_groups.png)",
            "",
            "В карте групп одинаковый цвет означает общий код внутри нейрона. Метки других методов переставлены для наибольшего совпадения с Ward только при отображении; обученные индексаторы не менялись. Все четыре карты используют m512:32общие группы входов,16независимых кодов на группу. Координаты пикселей использованы для визуализации, а не при построении групп.",
            "",
        ]
    lines += [
        "## Постановка и ограничения",
        "",
        "Все локальные методы в новом сравнении получают m/H кодов на нейрон. Previous aSRR сохраняет прежний полный банк, в том числе адаптивное распределение Ward; поэтому это более сильный практический контроль, а не чистая абляция одного linkage. На реальных задачах новые методы проверены в local/input-семействах — победителях прежней серии. На синтетике также проверены global и общие fixed synchronized профили.",
        "",
        "E-means — неизменённый авторский код Faiss v1.16.0, native PyTorch FP32 на V100. Для него проверены auto/5, quality/5, quality/25 проходов, по четыре общих случайных начала; выбирается минимальная source SSE. Lloyd и signed Lloyd имеют25шагов и те же четыре начала. У e-means сохранены last-epoch labels; постобработка пустых групп общая для centroid-методов и обеспечивает точный бюджет m. Это сравнение качества, не равного compute. Рецепты, источник, repairs и source histories сохранены.",
        "",
        "Источник e-means: [Bessa et al., 2026](https://arxiv.org/pdf/2611.09501v1), [Faiss v1.16.0](https://github.com/facebookresearch/faiss/tree/v1.16.0/contrib/e_means). Параметры взяты из этого релиза: авторский base.py отмечает отличия начальной массы и коэффициентов scaling laws от PDF v1. Это применение к нашим профилям, не воспроизведение embedding-датасетов или H100 speedup из статьи.",
        "",
        "Omniglot784–16–5:5way,10train+5H+5query на класс. MNIST784–16–2:100train+50H+100query на класс; source0–4/development5–6/test7–9. Source16/10задач, по4согласованные реплики, development48/24эпизода. Adam200, LR.003/.01/.03,6стартов, Dense24с дополнительным H-выбором четырёх пулов. G34 — прежний практический расширенный банк, не новый exhaustive native search на каждом m.",
        "",
        "Реальные CI условны фиксированным source/development данным и классам. Изображения могут повторяться между эпизодами. Подбор по development CE и one-SE не является гарантией меньшего test-риска. Истинного реального I не знаем. Новые software tests не создавались и не запускались.",
        "",
        "Код: experiments/srr_cluster_v1. Теория: paper/theory/ASRR_CLUSTERING_2026-11-15.md. Полный протокол, seed/hash registration, frozen predictions, source diagnostics и CSV находятся в artifacts/srr_clustering_2026-11-15; raw — artifacts/local/srr_clustering_2026-11-15.",
        "",
    ]
    if real is not None:
        lines += [
            "## Фактическая стоимость построения банка",
            "",
            "Время внутри source-конструктора для двух семейств и всей сетки m, в секундах. Прогоны выполнялись при общей нагрузке GPU; это не изолированный speed-бенчмарк. Ward переиспользует дерево для разных m, centroid-методы обучаются отдельно для каждого m. E-means здесь включает три рецепта по четыре начала, Lloyd — четыре начала; число проходов также различается.",
            "",
            "| Метод | Omniglot, с | MNIST, с |",
            "|---|---:|---:|",
        ]
        for method in METHODS:
            times = [
                json.loads(
                    (ROOT / dataset / f"construction/{method}.json").read_text()
                )["seconds"]
                for dataset in ("omniglot", "mnist")
            ]
            lines.append(f"| {LABELS[method]} | {times[0]:.2f} | {times[1]:.2f} |")
        lines += [""]
    body = "\n".join(lines)
    Path("paper/theory/ASRR_CLUSTERING_RESULTS_2026-11-15.md").write_text(body)
    (DEST / "REPORT.md").write_text(
        body.replace("../../artifacts/srr_clustering_2026-11-15/", "")
    )


def render():
    synthetic = json.loads((DEST / "synthetic_results.json").read_text())
    pairs, ranked = [], []
    for scenario in ("balanced", "unbalanced", "approximate", "input_groups"):
        for noise in (0, 0.01, 0.04):
            reference = {
                row["case"]: row["mse"]
                for row in synthetic["cases"]
                if row["scenario"] == scenario
                and row["noise"] == noise
                and row["method"] == "ward"
            }
            for method in METHODS:
                cases = [
                    row
                    for row in synthetic["cases"]
                    if row["scenario"] == scenario
                    and row["noise"] == noise
                    and row["method"] == method
                ]
                difference = np.array(
                    [row["mse"] - reference[row["case"]] for row in cases]
                )
                pairs.append(
                    {
                        "scenario": scenario,
                        "noise": noise,
                        "method": method,
                        "baseline": "ward",
                        "structures": len(cases),
                        "mse_difference": float(difference.mean()),
                        "half_ci_t95": float(
                            t.ppf(0.975, len(cases) - 1)
                            * difference.std(ddof=1)
                            / np.sqrt(len(cases))
                        ),
                    }
                )
                for rank in (2, 5):
                    subset = [row for row in cases if row["rank"] == rank]
                    ranked.append(
                        {
                            "scenario": scenario,
                            "noise": noise,
                            "method": method,
                            "rank": rank,
                            "structures": len(subset),
                            "mse": float(np.mean([row["mse"] for row in subset])),
                            "signed_recall": float(
                                np.mean([row["signed_recall"] for row in subset])
                            ),
                        }
                    )
    save_json(DEST / "synthetic_details.json", {"paired": pairs, "by_rank": ranked})
    write_csv(DEST / "synthetic_paired.csv", pairs)
    write_csv(DEST / "synthetic_by_rank.csv", ranked)
    write_csv(DEST / "synthetic_comparison.csv", synthetic["aggregate"])
    real = (
        json.loads((DEST / "real_results.json").read_text())
        if (DEST / "real_results.json").exists()
        else None
    )
    choices = (
        json.loads((DEST / "real_choices_frozen.json").read_text()) if real else None
    )
    if real is not None:
        matched = []
        for dataset in ("omniglot", "mnist"):
            episodes = np.load(DEST / f"{dataset}_real_episode_scores.npz")
            for method in (*METHODS, "union", "simple_union"):
                accuracy = episodes[f"{method}_accuracy"] - episodes["ward_accuracy"]
                ce = episodes[f"{method}_ce"] - episodes["ward_ce"]
                matched.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "baseline": "ward",
                        "accuracy_difference": float(accuracy.mean()),
                        "accuracy_half_ci": half_interval(accuracy),
                        "ce_difference": float(ce.mean()),
                        "ce_half_ci": half_interval(ce),
                    }
                )
        save_json(
            DEST / "matched_ward_paired.json",
            {"derived_from_saved_episode_scores": True, "paired": matched},
        )
        write_csv(DEST / "matched_ward_paired.csv", matched)
        real["paired"].extend(matched)
    figures(synthetic, real, choices)
    markdown(synthetic, real, choices)
    snapshot = DEST / "executed_source"
    snapshot.mkdir(exist_ok=True)
    paths = [
        *Path(__file__).parent.glob("*.py"),
        *Path(__file__).parent.glob("*.md"),
        Path("paper/theory/ASRR_CLUSTERING_2026-11-15.md"),
        Path("experiments/srr_approx_v1/core.py"),
        Path("experiments/srr_approx_v1/synthetic.py"),
        Path("experiments/srr_approx_v1/projection.py"),
        Path("experiments/srr_approx_v1/real.py"),
        Path("experiments/srr_real_v1/predictive.py"),
        Path("experiments/omniglot_prior_v1/model.py"),
        Path("experiments/paper_recovery_extension_v1/adapter.py"),
    ]
    for path in paths:
        relative = path.relative_to(Path.cwd()) if path.is_absolute() else path
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    save_json(
        DEST / "runtime_sources.json",
        {str(path): digest(path) for path in [*paths, *VENDOR.glob("*.py")]},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("score", "render"))
    {"score": score, "render": render}[parser.parse_args().stage]()


if __name__ == "__main__":
    main()
