"""Frozen benchmark tables, indexator heatmaps and proxy diagnostics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import NullFormatter

from experiments.darts_structured_tasks_v1.data import save_json
from experiments.sharing_benchmark_v1.report import aggregate, number, save_figure

from .run import METHODS, result_path

LABELS = {
    "z_euclidean": "Z: евклидова проекция",
    "z_hessian": "Z: полный гессиан",
    "z_taylor": "Z: градиент + гессиан",
    "meta_stop": "Hard meta: C без производной",
    "meta_full": "Hard meta: через адаптацию C",
    "projected_stop": "Градиент + line search, C остановлен",
    "projected_full": "Метаградиент + line search",
    "query_edits": "Дискретные правки, контроль",
    "oracle": "Истинный I",
}
COLORS = {
    "z_euclidean": "#0072b2",
    "z_hessian": "#e69f00",
    "z_taylor": "#009e73",
    "meta_stop": "#cc79a7",
    "meta_full": "#d55e00",
    "projected_stop": "#8c564b",
    "projected_full": "#9467bd",
    "query_edits": "#56b4e9",
    "oracle": "#222222",
}
PROXY_LABELS = {
    "euclidean_projected_codes": "Евклидова, проекционный C",
    "euclidean_support_codes": "Евклидова, support-fit C",
    "diagonal_hessian": "Диагональный гессиан",
    "full_hessian": "Полный гессиан",
    "taylor": "Градиент + полный гессиан",
}


def comparison_figures(output, summaries, budgets):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    methods = [*METHODS, "oracle"]
    primary = summaries["4"]
    positions = np.arange(len(methods))
    axes[0].barh(
        positions,
        [primary[m]["exact_cells"] / primary[m]["replicas"] for m in methods],
        color=[COLORS[m] for m in methods],
    )
    axes[0].set_yticks(positions, [LABELS[m] for m in methods])
    axes[0].invert_yaxis()
    axes[0].set(
        xlim=(0, 1.05),
        xlabel="Доля точных запусков",
        title="4 исходные функции; 6 новых геометрий",
    )
    for position, method in enumerate(methods):
        axes[0].text(
            0.025,
            position,
            f"{primary[method]['exact_cells']}/{primary[method]['replicas']}; "
            f"{primary[method]['exact_all_seeds_families']}/6 семейств",
            va="center",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2},
        )
    errors = [primary[m]["transfer"]["4"]["population_mse"] for m in methods]
    axes[1].scatter(errors, positions, color=[COLORS[m] for m in methods], s=55)
    for position, value in enumerate(errors):
        axes[1].text(value * 1.15, position, number(value), va="center", fontsize=8)
    axes[1].set_yticks(positions, [LABELS[m] for m in methods])
    axes[1].invert_yaxis()
    axes[1].set(
        xscale="log",
        xlim=(min(errors) * 0.6, max(errors) * 1.8),
        xlabel="MSE переноса, 4 новых support-примера",
        title="100 новых функций / геометрию; свежий C",
    )
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
    save_figure(fig, output, "primary_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for method in methods:
        axes[0].plot(
            budgets,
            [
                summaries[str(t)][method]["exact_cells"]
                / summaries[str(t)][method]["replicas"]
                for t in budgets
            ],
            marker="o",
            label=LABELS[method],
            color=COLORS[method],
            linestyle="--" if method == "oracle" else "-",
        )
        axes[1].plot(
            budgets,
            [
                summaries[str(t)][method]["transfer"]["4"]["population_mse"]
                for t in budgets
            ],
            marker="o",
            label=LABELS[method],
            color=COLORS[method],
            linestyle="--" if method == "oracle" else "-",
        )
    axes[0].set(ylabel="Доля точных I", ylim=(-0.03, 1.03))
    axes[1].set(yscale="log", ylabel="Population MSE, 4 target-support")
    for axis in axes:
        axis.set(xscale="log", xlabel="Число исходных функций")
        axis.set_xticks(budgets, list(map(str, budgets)))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("Одни настройки из 4-task pilot для всех source-бюджетов")
    save_figure(fig, output, "source_budget")


def model_figures(root, output, records, plan):
    family, seed, tasks = (
        plan["confirmation_families"][0],
        plan["confirmation_seeds"][0],
        4,
    )
    folder = root / "hidden" / f"family_{family}"
    methods = ["oracle", *METHODS]
    cmap = ListedColormap(
        ["#9e0142", "#f46d43", "#fee08b", "#f7f7f7", "#abdda4", "#66c2a5", "#3288bd"]
    )
    norm = BoundaryNorm(np.arange(-3.5, 4), cmap.N)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    raw = {}
    for axis, method in zip(axes.flat, methods):
        local_seed = 0 if method == "oracle" else seed
        with np.load(
            folder / "audit" / "tasks_4" / f"{method}_seed_{local_seed}.npz"
        ) as archive:
            aligned = archive["aligned_indices"]
            raw[method] = {
                "raw_indices": archive["indices"].tolist(),
                "aligned_indices": aligned.tolist(),
            }
        row = next(
            r
            for r in records
            if r["family"] == family
            and r["tasks"] == tasks
            and r["seed"] == local_seed
            and r["method"] == method
        )
        image = axis.imshow(
            aligned, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest"
        )
        axis.set(
            title=f"{LABELS[method]}\nошибок I: {row['recovery']['categorical_errors']}; маски: {row['recovery']['mask_errors']}",
            xlabel="Вход",
            ylabel="Выход",
        )
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.8,
        ticks=np.arange(-3, 4),
        label="Индекс; 0 — отключение",
    )
    fig.suptitle(
        f"Первая новая геометрия {family}, seed {seed}, 4 source-функции; только переименование кодов"
    )
    save_figure(fig, output, "indexators")
    save_json(output / "indexators.json", raw)

    with np.load(folder / "target_after_freeze.npz") as archive:
        limit = float(np.abs(archive["teacher_weights_audit_only"][0]).max())
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    for axis, method in zip(axes.flat, methods):
        local_seed = 0 if method == "oracle" else seed
        with np.load(
            folder / "audit" / "tasks_4" / f"{method}_seed_{local_seed}.npz"
        ) as archive:
            weights = archive["weights_support_4"]
        image = axis.imshow(
            weights,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
            interpolation="nearest",
        )
        axis.set(title=LABELS[method], xlabel="Вход", ylabel="Выход")
    fig.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.8, label="Вес после свежего fit C"
    )
    fig.suptitle(
        "Первая новая target-функция: индексатор заморожен, C обучен с нуля на 4 примерах"
    )
    save_figure(fig, output, "weights")

    with np.load(folder / "target_after_freeze.npz") as archive:
        teacher = archive["teacher_weights_audit_only"][0]
    with np.load(folder / "audit" / "dense_adam.npz") as archive:
        adam_weights = archive["weights_support_4"]
    illustrated = [("Истинная W", teacher)]
    for method in ("z_euclidean", "z_taylor", "projected_full"):
        with np.load(
            folder / "audit" / "tasks_4" / f"{method}_seed_{seed}.npz"
        ) as archive:
            illustrated.append((LABELS[method], archive["weights_support_4"]))
    illustrated.append(("Плотный Adam, 2000 шагов", adam_weights))
    limit = max(float(np.abs(weights).max()) for _, weights in illustrated)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4), constrained_layout=True)
    for axis, (label, weights) in zip(axes, illustrated):
        population = float(np.square(weights - teacher).sum() / teacher.shape[0])
        image = axis.imshow(
            weights,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
            interpolation="nearest",
        )
        axis.set(
            title=f"{label}\nMSE новой функции: {population:.3g}",
            xlabel="Вход",
            ylabel="Выход",
        )
    fig.colorbar(image, ax=axes.tolist(), shrink=0.8, label="Вес")
    fig.suptitle(
        "Одна новая функция, 4 support; Adam с фиксированными настройками, без выбора по query"
    )
    save_figure(fig, output, "dense_adam_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for method in METHODS:
        path = result_path(folder, "confirmation", 4, method, seed).with_suffix(".json")
        history = json.loads(path.read_text())["history"]
        for axis, key in zip(axes, ("source_query_mse", "best_selection_mse")):
            axis.plot(
                [h["seconds"] for h in history],
                [h[key][0] for h in history],
                label=LABELS[method],
                color=COLORS[method],
            )
    for axis in axes:
        axis.set(xlabel="Время поиска, с", yscale="log")
        axis.grid(alpha=0.2)
    axes[0].set(ylabel="Source-query MSE после fit C", title="Обучение структуры")
    axes[1].set(ylabel="Лучшая независимая validation MSE", title="Выбор checkpoint")
    axes[1].legend(fontsize=8)
    save_figure(fig, output, "learning_curves")


def diagnostic_figures(root, output, plan):
    family, seed = plan["pilot_families"][0], plan["pilot_seeds"][0]
    path = root / "diagnostics" / f"family_{family}_seed_{seed}"
    row = json.loads(path.with_suffix(".json").read_text())
    with np.load(path.with_suffix(".npz")) as archive:
        arrays = dict(archive)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    names = ["euclidean_projected_codes", "full_hessian", "taylor"]
    for i, state in enumerate(("initial", "final")):
        for j, name in enumerate(names):
            axis = axes[i, j]
            exact = arrays[f"{state}_adapted_query"]
            values = arrays[f"{state}_{name}"]
            correlation = row["states"][state]["metrics"][name][
                "spearman_with_adapted_query"
            ]
            axis.scatter(
                np.maximum(values, 1e-8), exact, s=6, alpha=0.35, color="#0072b2"
            )
            axis.set(
                xscale="log",
                yscale="log",
                xlabel=PROXY_LABELS[name],
                ylabel="Query MSE после support-fit C",
                title=f"{'Начальный' if state == 'initial' else 'Конечный'} Z; Spearman={correlation:.3f}",
            )
            axis.grid(alpha=0.2)
    fig.suptitle(
        "Один и тот же пул кандидатов; полный Taylor совпадает с MSE по определению"
    )
    save_figure(fig, output, "proxy_diagnostics")


def failure_figure(root, output, records):
    failures = [
        r
        for r in records
        if r["tasks"] == 4
        and r["method"] == "z_euclidean"
        and not r["recovery"]["exact_I"]
    ]
    selected = min(failures, key=lambda row: (row["family"], row["seed"]))
    family, seed = selected["family"], selected["seed"]
    folder = root / "hidden" / f"family_{family}" / "audit" / "tasks_4"
    cmap = ListedColormap(
        ["#9e0142", "#f46d43", "#fee08b", "#f7f7f7", "#abdda4", "#66c2a5", "#3288bd"]
    )
    norm = BoundaryNorm(np.arange(-3.5, 4), cmap.N)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    saved = {
        "family": family,
        "seed": seed,
        "selection_rule": "First Euclidean failure by family then seed, after freeze",
        "methods": {},
    }
    for axis, method in zip(axes.flat, ("oracle", *METHODS)):
        local_seed = 0 if method == "oracle" else seed
        with np.load(folder / f"{method}_seed_{local_seed}.npz") as archive:
            indices = archive["aligned_indices"]
            saved["methods"][method] = {
                "raw_indices": archive["indices"].tolist(),
                "aligned_indices": indices.tolist(),
            }
        row = next(
            r
            for r in records
            if r["family"] == family
            and r["seed"] == local_seed
            and r["tasks"] == 4
            and r["method"] == method
        )
        image = axis.imshow(
            indices, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest"
        )
        axis.set(
            title=f"{LABELS[method]}\nошибок I: {row['recovery']['categorical_errors']}; маски: {row['recovery']['mask_errors']}",
            xlabel="Вход",
            ylabel="Выход",
        )
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.8,
        ticks=np.arange(-3, 4),
        label="Индекс; 0 — отключение",
    )
    fig.suptitle(
        f"Первый сбой Euclidean по порядку family/seed: {family}, seed {seed}, 4 source-функции"
    )
    save_figure(fig, output, "failure_indexators")
    save_json(output / "failure_indexators.json", saved)


def findings(root, summaries):
    posthoc = json.loads((root / "posthoc_objectives.json").read_text())
    bad = [
        r
        for r in posthoc["records"]
        if r["tasks"] == 4
        and r["method"] == "z_euclidean"
        and r["oracle_beats_selected_source"]
    ]
    edits = [
        r
        for r in posthoc["records"]
        if r["tasks"] == 4
        and r["method"] == "query_edits"
        and r["oracle_beats_last_search_source"]
    ]
    return [
        "",
        "## Интерпретация результатов",
        "",
        "Ранний свободный Z действительно задаёт плохую геометрию отбора. На четырёх pilot-репликах из одного и того же пула евклидова проекция начального Z выбирает структуры со средней независимой validation MSE2.129; support-adaptation/query-критерий выбирает0.01067. На конечном Z эти выбранные ошибки уже совпадают. Нельзя считать евклидову метрику всегда плохой или весь эффект приписать градиенту.",
        "",
        "Полный гессиан дал небольшой прирост в основном4-task режиме:12/18 вместо11/18 точных I, MSE4 0.319 вместо0.333. Это один дополнительный запуск и одна дополнительная геометрия со всеми точными seed; шесть независимых геометрий не дают основания объявлять большой устойчивый выигрыш. Добавление линейного gradient-term в Z-projector ухудшило результат до3/18 иMSE0.984. Его локальный проектор уменьшает собственную квадратичную цель; равенство Taylor MSE проверено до4.44e-16. Это не доказательство глобального решения assignment-задачи.",
        "",
        "Градиент через support-fit C вычислен корректно (направленная ошибка1.1e-11), но STE Adam не добавил точных восстановлений:11/18, почти та же transfer MSE. В pilot большие шаги открывали множество лишних связей; validation сохраняла старт. Projected full со строгим source-query line search в confirmation4 не принял ни одного из предложенных шагов и полностью сохранил исходные I. Следовательно, непрерывное направление и его жёсткая проекция не обеспечили полезный дискретный переход.",
        "",
        f"Важная послепоисковая проверка: во всех{len(bad)} ошибочных Euclidean-запусках4-task истинный I имеет меньший source-query loss. Даже после исчерпания однопозиционных улучшений контрольного поиска он лучше последнего состояния в{len(edits)} случаях. Истина добавлялась только в этот audit после freeze. В этих случаях критерий предпочёл бы нужную структуру, однако алгоритмы до неё не добрались. Это поддерживает объяснение через трудность дискретного поиска; недостаточно объяснять сбой одним лишь неправильным прокси или Adam-кодбуком.",
        "",
        "При2 исходных функциях все варианты дали0/18. При8 Euclidean и Hessian уже18/18, fresh-C transfer MSE0.0005688 на уровне истинного I; Taylor6/18. При64 все варианты18/18. В данном benchmark использование нескольких функций с общим I оказалось сильнее, чем проверенные изменения градиента. Восемь функций не являются универсальным теоретическим порогом.",
        "",
        "![Первый сбой](failure_indexators.png)",
        "",
        "Новый figure выбран по правилу первого Euclidean-сбоя в порядке family/seed; основной figure indexators.png по-прежнему показывает заранее фиксированные family2101/seed41. Переименованы только номера/знаки кодов, координаты не переставляются.",
    ]


def report(root):
    output = root / "report"
    output.mkdir(exist_ok=True)
    plan = json.loads((root / "protocol.json").read_text())
    records = []
    for family in plan["confirmation_families"]:
        records.extend(
            json.loads(
                (
                    root / "hidden" / f"family_{family}" / "audit" / "summary.json"
                ).read_text()
            )["records"]
        )
    summaries = {
        str(tasks): aggregate([row for row in records if row["tasks"] == tasks])
        for tasks in plan["task_budgets"]
    }
    save_json(output / "summary.json", summaries)
    costs = {}
    for tasks in plan["task_budgets"]:
        for method in METHODS:
            values = []
            for family in plan["confirmation_families"]:
                for seed in plan["confirmation_seeds"]:
                    path = result_path(
                        root / "hidden" / f"family_{family}",
                        "confirmation",
                        tasks,
                        method,
                        seed,
                    ).with_suffix(".json")
                    values.append(json.loads(path.read_text())["seconds"])
            costs[f"{tasks}/{method}"] = {
                "mean_seconds": float(np.mean(values)),
                "total_seconds": float(np.sum(values)),
            }
    save_json(output / "costs.json", costs)
    with (output / "comparison.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "source_tasks",
                "method",
                "exact_families_all_seeds",
                "exact_cells",
                "replicas",
                "mask_f1",
                "I_errors",
                "mse1",
                "mse4",
                "mse16",
                "mean_seconds",
            )
        )
        for tasks in plan["task_budgets"]:
            for method in (*METHODS, "oracle"):
                row = summaries[str(tasks)][method]
                writer.writerow(
                    (
                        tasks,
                        method,
                        row["exact_all_seeds_families"],
                        row["exact_cells"],
                        row["replicas"],
                        row["mask_f1"],
                        row["categorical_errors"],
                        *[
                            row["transfer"][size]["population_mse"]
                            for size in ("1", "4", "16")
                        ],
                        costs.get(f"{tasks}/{method}", {}).get("mean_seconds", 0),
                    )
                )
    comparison_figures(output, summaries, plan["task_budgets"])
    model_figures(root, output, records, plan)
    failure_figure(root, output, records)
    diagnostic_figures(root, output, plan)
    lines = [
        "# Градиент, кривизна и адаптация кодбука",
        "",
        "Исследование 5 ноября 2026. Четыре свободные V100 0–3, четыре параллельных shard. GPU 4–7 заняты чужими процессами. Скрытая перемешанная свёртка W14×16, K3, fixed0 и связанные отрицания. Все позиции и группы доступны поиску.",
        "",
        "## Протокол",
        "",
        "Первый pilot8 дал практически одинаковую минимальную validation loss у всех вариантов и learning rate. До новых подтверждающих геометрий зарегистрирован окончательный pilot4. Две pilot-геометрии2001/2002, два seed; по шесть настроек у каждого обучаемого варианта. После freeze настройки не менялись: шесть новых геометрий2101–2106, три seed, бюджеты2/4/8/64. Истинные I и новые100 коэффициентных функций на геометрию открыты только после заморозки всех моделей. Каждая новая функция адаптирует только свежий C. Независимая единица — шесть геометрий, не18 запусков.",
        "",
        "Первоначальный8-task pilot сохранён отдельно и не используется как подтверждающий результат. Для2/8/64 нет отдельного HPO: это контроли с настройками, выбранными на4 функциях. Во всех случаях4 support +4 query на source-функцию; независимая validation24 функций по4+64; шум0.1.",
        "",
        "Перед meta-pilot зарегистрированы четыре слепые случайные правки найденного Euclidean I: его исходная validation уже была на уровне шума. Оба STE, оба projected-варианта и query-edit контроль на pilot получают одинаковое испорченное начало; в confirmation индексатор не портится. Этот подбор исследует исправление найденной структуры и не использует teacher. После неудачного STE-pilot до подтверждения зарегистрированы projected-контроли без дополнительного HPO; см. meta_calibration.json и gradient_controls.json.",
        "",
        "## Результаты",
    ]
    for tasks in plan["task_budgets"]:
        lines += [
            "",
            f"### {tasks} исходных функций",
            "",
            "| Метод | Все seed точны, семейств | Точных запусков | F1 маски | MSE4, свежий C |",
            "|---|---:|---:|---:|---:|",
        ]
        for method in (*METHODS, "oracle"):
            row = summaries[str(tasks)][method]
            lines.append(
                f"| {LABELS[method]} | {row['exact_all_seeds_families']}/6 | {row['exact_cells']}/{row['replicas']} | {row['mask_f1']:.3f} | {number(row['transfer']['4']['population_mse'])} |"
            )
    lines += [
        "",
        "## Что именно сравнивается",
        "",
        "Z-варианты имеют общий старт, одинаковый L-step, rho/dual-сетки, четыре старта и80 раундов; штраф разреженности везде нулевой. Hessian-вариант меняет только критерий проекции. Taylor добавляет линейный член градиента в точке Z+U к той же полной кривизне. Проектор локальный, с условными assignment/C проходами; глобальное решение полного гессиана не заявляется. Это диагностическая модификация, а не доказанная новая ADMM-схема.",
        "",
        "Hard meta стартует с выбранного Z-евклидова I. Оба варианта имеют один жёсткий forward, одинаковые шесть learning rate,800 шагов и4 малых возмущения. Разница — только производная support-fit C. Градиент через жёсткое назначение суррогатный STE. Непрерывный накопитель задаёт коэффициенты общего базиса A, что отличается от исходных скалярных Z. На каждом checkpoint кодбуки заново оцениваются одинаковым точным МНК без ridge. Дискретные правки — отдельный контроль, не часть градиентного поиска.",
        "",
        "Projected stop/full используют сырой градиент по непрерывному расширению A, нормированный по максимальному модулю. Каждый шаг предлагает шесть целиком жёстких матриц для смещений0.25/0.5/1/2/4/8; после точного support-fit C source-query выбирает строгое улучшение. При отсутствии улучшения алгоритм останавливается; предел80 шагов. Это глобальный line search по градиентному направлению, а не перебор всех позиций. Полный вариант учитывает изменение C, остановленный — только прямое влияние A. Оба сохраняют шаг0 и выбирают checkpoint независимой validation.",
        "",
        "## Диагностика критериев",
        "",
        "Для каждой из четырёх pilot-реплик один пул из сохранённых checkpoints и1024 фиксированных случайных возмущений. Teacher не используется. Сравниваются евклидова проекция с её C, та же норма после support-fit C, диагональная/полная эмпирическая кривизна и полный Taylor. В Taylor C уже получен на support; его равенство query-loss является алгебраическим тождеством для линейной модели, не эмпирическим открытием. У изотропных Gaussian inputs population-метрика предсказаний равна Frobenius, поэтому нельзя автоматически считать гессиан лучшим.",
        "",
        "| Состояние Z | Критерий | Средний Spearman с адаптационным query-loss |",
        "|---|---|---:|",
    ]
    diagnostics = [
        json.loads(path.read_text())
        for path in sorted((root / "diagnostics").glob("family_*.json"))
    ]
    for state in ("initial", "final"):
        for name, label in PROXY_LABELS.items():
            values = [
                r["states"][state]["metrics"][name]["spearman_with_adapted_query"]
                for r in diagnostics
            ]
            lines.append(f"| {state} | {label} | {np.mean(values):.4f} |")
    audit = json.loads((root / "hypergradient_audit.json").read_text())
    dense_path = (
        root
        / "hidden"
        / f"family_{plan['confirmation_families'][0]}"
        / "audit"
        / "dense_adam.json"
    )
    dense = json.loads(dense_path.read_text())
    lines += [
        "",
        f"Направленная производная inner-fit: относительная ошибка {audit['directional_relative_error']:.3g}; cosine полного и остановленного градиентов {audit['cosine_full_stop']:.3g}. Это проверка численной реализации на одном pilot-состоянии, не гарантия поиска дискретного I.",
        "",
        f"Плотный Adam для первого target:2000 шагов, lr0.03, нулевая инициализация,4 support. Training MSE {dense['support_training_mse']:.4g}, population MSE {dense['population_mse']:.4g}. Это отдельная иллюстрация, не среднее по100 функциям и не настроенный конкурент. Линейная плотная задача выпуклая; разница обучения/переноса не свидетельствует о локальном минимуме Adam.",
        "",
        "## Графики",
        "",
        "![Основное сравнение](primary_comparison.png)",
        "",
        "![Бюджеты](source_budget.png)",
        "",
        "![Индексаторы](indexators.png)",
        "",
        "![Свежие веса](weights.png)",
        "",
        "![Adam для сравнения](dense_adam_comparison.png)",
        "",
        "![Прокси](proxy_diagnostics.png)",
        "",
        "![Обучение](learning_curves.png)",
        "",
        "## Ограничения",
        "",
        "Это линейный benchmark с аналитическим численным inner-fit. Нелинейные Pattern/DeepSets/синусы этим запуском не проверяются. Метаградиентные варианты продолжают общий Z-поиск, без случайного старта A. Задачи query служат внешней обучающей целью; независимая validation выбирает checkpoint. Новые тестовые функции в поиске не участвуют. Не заявляются глобальная сходимость или алгоритмическая новизна. Исходные методы и артефакты предыдущих этапов не изменялись.",
    ]
    lines += findings(root, summaries)
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                tasks: {
                    method: (
                        value["exact_cells"],
                        value["transfer"]["4"]["population_mse"],
                    )
                    for method, value in summary.items()
                }
                for tasks, summary in summaries.items()
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report(args.root)


if __name__ == "__main__":
    main()
