"""Scientific tables, heatmaps and recorded optimization curves."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator
from scipy.optimize import linear_sum_assignment

from .data import BASE, save_json

plt.rcParams.update(
    {"font.size": 10, "axes.titlesize": 11, "figure.dpi": 140, "savefig.dpi": 170}
)


def save_figure(figure, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def categorical(axis, values, codes, title):
    colors = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, 2 * codes + 1))
    colors[codes] = [0.96, 0.96, 0.96, 1]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-codes - 0.5, codes + 1), len(colors))
    plot = axis.imshow(
        values, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm
    )
    axis.set_title(title)
    axis.set_xlabel("Скрытый нейрон")
    axis.set_ylabel("Координата входа")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    return plot


def matched_columns(weights, reference):
    numerator = weights.T @ reference
    denominator = (
        np.linalg.norm(weights, axis=0)[:, None]
        * np.linalg.norm(reference, axis=0)[None]
    )
    similarity = numerator / np.maximum(denominator, 1e-10)
    source, target = linear_sum_assignment(-similarity)
    return weights[:, source[np.argsort(target)]]


def synthetic(output):
    summary = json.loads((output / "summary.json").read_text())
    spec = summary["task"]
    with np.load(output / "selected_states.npz") as archive:
        states = dict(archive)
    with np.load(output / "teacher_audit_only.npz") as archive:
        truth = dict(archive)
    save_json(
        output / "selected_indexator.json",
        {
            "indexator": states["indices"].tolist(),
            "numeric": states["learned_inherited_numeric"].tolist(),
            "aligned_for_display": states["aligned_indices"].tolist(),
            "negative_label_selects_negated_base_code": True,
        },
    )
    indices = states["true_indices"]
    values = np.r_[0.0, truth["numeric"][: spec["codes"]]]
    true_weights = values[np.abs(indices)] * np.sign(indices)
    row_order = truth["permutation"]
    followup = output.parent / "deepsets_mask_then_sharing"
    panels = 4 if (followup / "summary.json").exists() else 3
    fig, axes = plt.subplots(
        1, panels, figsize=(4.2 * panels, 5), constrained_layout=True
    )
    p = categorical(axes[0], indices[row_order], spec["codes"], "Истинный индексатор")
    fig.colorbar(
        p, ax=axes[0], ticks=range(-spec["codes"], spec["codes"] + 1), shrink=0.75
    )
    p = categorical(
        axes[1],
        states["aligned_indices"][row_order],
        spec["codes"],
        "Прямой signed DARTS",
    )
    fig.colorbar(
        p, ax=axes[1], ticks=range(-spec["codes"], spec["codes"] + 1), shrink=0.75
    )
    if panels == 4:
        with np.load(followup / "selected_states.npz") as archive:
            postprocessed = archive["aligned_indices"]
        p = categorical(
            axes[2],
            postprocessed[row_order],
            spec["codes"],
            "Маска DARTS → поиск групп",
        )
        fig.colorbar(
            p, ax=axes[2], ticks=range(-spec["codes"], spec["codes"] + 1), shrink=0.75
        )
    dense = matched_columns(states["dense_shared_adam_weights"], true_weights)[
        row_order
    ]
    maximum = float(np.abs(dense).max())
    p = axes[-1].imshow(
        dense, cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="auto"
    )
    axes[-1].set_title("Плотный Adam: численные веса")
    axes[-1].set_xlabel("Скрытый нейрон")
    axes[-1].set_ylabel("Координата входа")
    fig.colorbar(p, ax=axes[-1], shrink=0.75)
    save_figure(fig, output / "figures/selected_indexators")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for axis, name, title in zip(
        axes[:2],
        ("raw_probabilities", "canonical_probabilities"),
        ("P(0) до переноса", "P(0) после переноса"),
    ):
        p = axis.imshow(
            states[name][row_order, :, 0], vmin=0, vmax=1, aspect="auto", cmap="viridis"
        )
        axis.set_title(title)
        fig.colorbar(p, ax=axis, shrink=0.8)
    categorical(
        axes[2], states["raw_indices"][row_order], spec["codes"], "Argmax исходного P"
    )
    categorical(
        axes[3], states["indices"][row_order], spec["codes"], "Argmax после переноса"
    )
    save_figure(fig, output / "figures/mass_transfer")

    selected = summary["selection"]["primary"]["row"]
    fit = np.load(output / "confirmation/hard_fits.npz")
    restart = int(fit["query_mse"][selected].argmin())
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.plot(
        fit["history_steps"],
        fit["history_train"][:, selected, restart] / spec["target_variance"],
        label="Наш I: train",
    )
    axis.plot(
        fit["history_steps"],
        fit["history_query"][:, selected, restart] / spec["target_variance"],
        label="Наш I: query",
    )
    controls = json.loads((output / "controls/selection.json").read_text())
    for variant, method, title in (
        ("dense_shared", "adam", "Плотный Adam: query"),
        ("oracle", "adam", "Истинный I: query"),
    ):
        choice = controls[variant][method]
        with np.load(output / "controls" / (variant + ".npz")) as archive:
            values = archive["history_query"][:, choice["row"], choice["restart"]]
            axis.plot(
                archive["history_steps"],
                np.maximum(values / spec["target_variance"], 1e-12),
                label=title,
            )
    axis.set_yscale("log")
    axis.set_xlabel("Шаги численного обучения после фиксации I")
    axis.set_ylabel("MSE / variance(y)")
    axis.legend()
    axis.grid(alpha=0.2)
    save_figure(fig, output / "figures/numeric_curves")
    return summary


def image_report(output):
    summary = json.loads((output / "summary.json").read_text())
    spec = summary["task"]
    with np.load(output / "selected_states.npz") as archive:
        states = dict(archive)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    p = categorical(
        axes[0, 0],
        states["reference_indices_0"],
        spec["codes"][0],
        "Эталонный DeepSets: W₁",
    )
    fig.colorbar(
        p,
        ax=axes[0, 0],
        ticks=np.linspace(-spec["codes"][0], spec["codes"][0], 5).astype(int),
        shrink=0.75,
    )
    p = categorical(
        axes[0, 1],
        states["aligned_indices_0"],
        spec["codes"][0],
        "Найденный индексатор W₁",
    )
    fig.colorbar(
        p,
        ax=axes[0, 1],
        ticks=np.linspace(-spec["codes"][0], spec["codes"][0], 5).astype(int),
        shrink=0.75,
    )
    reference = states["oracle_adam_matrix_0"]
    weights = matched_columns(states["dense_adam_matrix_0"], reference)
    maximum = np.abs(weights).max()
    p = axes[0, 2].imshow(
        weights, aspect="auto", cmap="RdBu_r", vmin=-maximum, vmax=maximum
    )
    axes[0, 2].set_title("Плотный Adam: W₁")
    fig.colorbar(p, ax=axes[0, 2], shrink=0.75)
    categorical(
        axes[1, 0], states["reference_indices_2"].T, 1, "Эталонный индексатор poolingᵀ"
    )
    categorical(
        axes[1, 1], states["aligned_indices_2"].T, 1, "Найденный индексатор poolingᵀ"
    )
    p = axes[1, 2].imshow(
        states["canonical_probability_0"][..., 0],
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[1, 2].set_title("P(0) после переноса массы, W₁")
    for axis in axes[1, :2]:
        axis.set_ylabel("Пулируемый признак")
    fig.colorbar(p, ax=axes[1, 2], shrink=0.75)
    save_figure(fig, output / "figures/selected_indexators")
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    primary = summary["selection"]
    selected = primary["row"]
    restart = summary["methods"]["learned_inherited"]["query_selected_restart"]
    history = json.loads((output / "confirmation_hard/history.json").read_text())
    for key, label in (("train", "Наш I: train"), ("query", "Наш I: query")):
        axis.plot(
            [row["step"] for row in history],
            [row[key][selected][restart][0] for row in history],
            label=label,
        )
    choices = json.loads((output / "control_selection.json").read_text())
    for variant, label in (
        ("oracle", "Эталонный DeepSets: query"),
        ("dense", "Плотный Adam: query"),
    ):
        choice = choices[variant]["adam"]
        history = json.loads((output / variant / "history.json").read_text())
        axis.plot(
            [row["step"] for row in history],
            [row["query"][choice["row"]][choice["restart"]][0] for row in history],
            label=label,
        )
    axis.set_xlabel("Шаги численного обучения")
    axis.set_ylabel("Cross-entropy")
    axis.legend()
    axis.grid(alpha=0.2)
    save_figure(fig, output / "figures/numeric_curves")
    return summary


def sine_report(output):
    summary = json.loads((output / "summary.json").read_text())
    spec = summary["task"]
    with np.load(output / "selected_states.npz") as archive:
        states = dict(archive)
    with np.load(output / "teacher_audit_only.npz") as archive:
        direction = archive["direction"]
    save_json(
        output / "selected_indexator.json",
        {
            "indexator": states["indices"].tolist(),
            "numeric": states["learned_inherited_numeric"].tolist(),
            "aligned_for_display": states["aligned_indices"].tolist(),
            "negative_label_selects_negated_base_code": True,
        },
    )
    names = [
        ("Истинные группы", states["true_indices"], True),
        ("Найденный I", states["aligned_indices"], True),
        ("Adam: веса", states["dense_shared_adam_weights"], False),
        ("Adam + моментный старт", states["dense_moment_adam_weights"], False),
    ]
    quantized_folder = output.parent / (output.name + "_quantized")
    quantized = None
    if (quantized_folder / "selected_states.npz").exists():
        with np.load(quantized_folder / "selected_states.npz") as archive:
            quantized = dict(archive)
        names.insert(2, ("Soft W → signed k-means", quantized["aligned_indices"], True))
    fig, axes = plt.subplots(
        1, len(names), figsize=(3 * len(names), 6), constrained_layout=True
    )
    for axis, (title, values, discrete) in zip(axes, names):
        if discrete:
            p = categorical(axis, values, spec["codes"], title)
        else:
            maximum = np.abs(values).max()
            p = axis.imshow(
                values, cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="auto"
            )
            axis.set_title(title)
        axis.set_xticks([])
        fig.colorbar(
            p,
            ax=axis,
            ticks=range(-spec["codes"], spec["codes"] + 1) if discrete else None,
            shrink=0.65,
        )
    save_figure(fig, output / "figures/selected_indexators")
    coordinate = np.linspace(-2.5, 2.5, 1200)
    target = spec["low_amplitude"] * np.sin(
        coordinate * spec["low_frequency"]
    ) + np.sin(coordinate * spec["frequency"])
    fig, axis = plt.subplots(figsize=(11, 4), constrained_layout=True)
    axis.plot(
        coordinate, target, color="black", linewidth=2, label="Сумма двух синусоид"
    )
    for name, label in (
        ("learned_inherited", "Наш фиксированный I"),
        ("dense_shared_adam", "Adam, случайный старт"),
        ("dense_moment_adam", "Adam, моментный старт"),
    ):
        weights, numeric = states[name + "_weights"], states[name + "_numeric"]
        values = numeric[spec["codes"] + 1] * np.sin(
            coordinate * float(direction @ weights[:, 0])
        )
        axis.plot(coordinate, values, label=label, alpha=0.85)
    if quantized is not None:
        values = quantized["numeric"][spec["codes"] + 1] * np.sin(
            coordinate * float(direction @ quantized["weights"][:, 0])
        )
        axis.plot(coordinate, values, label="Soft W → signed k-means", alpha=0.8)
    axis.set_xlabel("Сечение входа x = t · v; по горизонтали t")
    axis.set_ylabel("Предсказание")
    axis.legend(loc="upper right", ncol=2)
    axis.grid(alpha=0.2)
    save_figure(fig, output / "figures/function_slice")
    return summary


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def measured_costs(output):
    costs = {}
    for folder in sorted(path for path in output.iterdir() if path.is_dir()):
        item = {
            "numeric_model_updates": 0,
            "architecture_model_updates": 0,
            "timed_training_seconds": 0.0,
        }
        for path in folder.rglob("complete.json"):
            data = json.loads(path.read_text())
            if "numeric_updates_per_root" in data:
                item["numeric_model_updates"] += (
                    sum(data["numeric_updates_per_root"]) * data["numeric_restarts"]
                )
                item["architecture_model_updates"] += data["steps"] * data["roots"]
                item["timed_training_seconds"] += data["seconds"]
            elif "numeric_updates" in data:
                item["numeric_model_updates"] += (
                    sum(data["numeric_updates"]) * data["restarts"]
                )
                item["architecture_model_updates"] += (
                    data["architecture_updates"] * data["rows"]
                )
                item["timed_training_seconds"] += data["seconds"]
        for path in folder.rglob("*.npz"):
            if not path.with_suffix(".json").is_file():
                continue
            with np.load(path) as archive:
                if (
                    "history_steps" not in archive.files
                    or "query_mse" not in archive.files
                ):
                    continue
                shape = archive["query_mse"].shape
                item["numeric_model_updates"] += int(np.prod(shape)) * int(
                    archive["history_steps"][-1]
                )
                item["timed_training_seconds"] += float(archive["seconds"])
        if item["numeric_model_updates"]:
            costs[folder.name] = item
    save_json(output / "measured_costs.json", costs)
    return costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "results/20261104")
    args = parser.parse_args()
    output = args.output
    rows, notes, all_summaries = [], [], {}
    for name in ("deepsets", "deepsets_refinement"):
        folder = output / name
        if (folder / "summary.json").exists():
            summary = synthetic(folder)
            all_summaries[name] = summary
            values = summary["methods"]
            rows.append(
                [
                    name,
                    f"{values['learned_inherited']['test_nmse']:.6g}",
                    f"{values['learned_fresh']['test_nmse']:.6g}",
                    f"{values['dense_shared_adam']['test_nmse']:.6g}",
                    f"{values['oracle_adam']['test_nmse']:.3g}",
                ]
            )
    if (output / "deepsets_mask_then_sharing/summary.json").exists():
        summary = json.loads(
            (output / "deepsets_mask_then_sharing/summary.json").read_text()
        )
        all_summaries["deepsets_mask_then_sharing"] = summary
        notes.append(
            f"Двухэтапный follow-up: NMSE {summary['inherited_test_nmse']:.6g}; свежий C {summary['fresh_test_nmse']:.6g}; ошибок I {summary['recovery']['categorical_errors']}/{summary['recovery']['categorical_positions']}. Это отдельный exploratory опыт: маска DARTS, независимые веса внутри неё, signed k-means и дообучение C."
        )
    image_rows = []
    for dataset in ("MNIST", "FashionMNIST"):
        folder = output / ("images_" + dataset)
        if (folder / "summary.json").exists():
            summary = image_report(folder)
            all_summaries["images_" + dataset] = summary
            values = summary["methods"]
            image_rows.append(
                [
                    dataset,
                    f"{100 * values['learned_inherited']['test_accuracy']:.2f}%",
                    f"{100 * values['learned_fresh']['test_accuracy']:.2f}%",
                    f"{100 * values['dense_adam']['test_accuracy']:.2f}%",
                    f"{100 * values['dense_sgdm']['test_accuracy']:.2f}%",
                    f"{100 * values['oracle_adam']['test_accuracy']:.2f}%",
                ]
            )
    sine_rows, quantized_rows = [], []
    for frequency in (4, 6, 8):
        folder = output / ("sine_" + str(frequency))
        if (folder / "summary.json").exists():
            summary = sine_report(folder)
            all_summaries["sine_" + str(frequency)] = summary
            values = summary["methods"]
            sine_rows.append(
                [
                    frequency,
                    *[
                        f"{values[name]['population_mse']:.6g}"
                        for name in (
                            "learned_inherited",
                            "learned_fresh",
                            "dense_shared_adam",
                            "dense_shared_sgdm",
                            "dense_long_adam",
                            "dense_moment_adam",
                            "oracle_adam",
                        )
                    ],
                ]
            )
            quantized_folder = output / ("sine_" + str(frequency) + "_quantized")
            if (quantized_folder / "summary.json").exists():
                quantized = json.loads((quantized_folder / "summary.json").read_text())
                all_summaries[quantized_folder.name] = quantized
                quantized_rows.append(
                    [
                        frequency,
                        f"{values['learned_inherited']['population_mse']:.6g}",
                        f"{quantized['selected_population_mse']:.6g}",
                        f"{quantized['fresh_population_mse']:.6g}",
                        f"{quantized['recovery']['categorical_errors']}/{quantized['recovery']['categorical_positions']}",
                        f"{quantized['fresh_successes']}/{quantized['fresh_restarts']}",
                    ]
                )
    costs = measured_costs(output)
    moment_rows = []
    for frequency in (4, 6, 8):
        moment = json.loads(
            (
                output / f"sine_{frequency}/moment_codebook_control/summary.json"
            ).read_text()
        )
        moment_rows.append(
            [
                frequency,
                *[
                    f"{moment[name]['selected_population_mse']:.6g} ({moment[name]['successes']}/32)"
                    for name in ("direct", "quantized", "oracle")
                ],
            ]
        )
    capacity = json.loads((output / "sine_capacity_diagnosis.json").read_text())
    capacity_rows = []
    for name, values in capacity["results"].items():
        capacity_rows.append(
            [
                name,
                f"{values['source_function_best_population_mse']:.6g}",
                f"{values['new_codebooks_best_risk_mean']:.6g}",
                f"{values['new_codebooks_best_risk_under_point1']}/32",
            ]
        )
    report = [
        "# Signed DARTS: DeepSets и сумма двух синусоид",
        "",
        "Все основные структуры, численные состояния и гиперпараметры выбраны по support/query/selection до оценки. Query, по которой обновляется P, не считается независимой валидацией. После переноса взаимно гасящейся массы C/−C в ноль берётся argmax и переобучаются только численные параметры. Справочники содержат m свободных C и ровно связанные −C; размер словаря 2m+1.",
        "",
        "## Синтетический DeepSets",
        "",
        "Восемь двухмерных точек, два общих ReLU-фильтра на точку, четыре независимых численных кода. Входные координаты скрыто переставлены; поиску доступны 16 входов и 16 скрытых нейронов без блоков, ролей и числа активных связей. Support/query/selection/test: 1024/512/2048/8192. Функция — сумма всех ReLU с общими b/a/d. Метрика ниже — MSE, делённая на дисперсию обучающих ответов; меньше лучше.",
        "",
        table(
            ["Поиск", "Выбранный C", "Свежий C", "Плотный Adam", "Истинный I + Adam"],
            rows,
        ),
        "",
        *notes,
        "",
        "Прямой поиск, восстановление бинарной маски и восстановление групп оцениваются отдельно. Приближённое решение функции не доказывает точное восстановление равенств. Научная оценка одной синтетической структуры не устанавливает перенос на произвольные DeepSets.",
        "",
        "![Индексаторы и плотный Adam](deepsets_refinement/figures/selected_indexators.png)",
        "",
        "В рисунках синтетики исходная перестановка входных координат восстановлена только для отображения. Скрытые нейроны и глобальные имена/знаки групп согласованы после выбора модели. Плотные Adam/SGD показываются реальными численными весами; дискретные индексаторы им не приписываются.",
        "",
        "![Перенос массы](deepsets_refinement/figures/mass_transfer.png)",
        "",
        "## Перемешанные точки изображений",
        "",
        "Это компактный опыт: изображения усредняются до 4×4 точек с признаками row/column/intensity. Эталонный DeepSets: phi 3→8 ReLU, mean pooling, rho 8→32 ReLU→10. Поиск использует общий dense 48→128→8→32→10 и узнаёт индексаторы первого слоя, bias и pooling с 24/8/1 независимыми кодами. Финальный rho обучается численно. Это меньшая модель и разрешение, чем исторический пилот на 196 точках и width64; результаты не являются его воспроизведением.",
        "",
        "Все десять классов; 6000/1000/1000 отдельных training-изображений для support/query/selection. В каждом обучающем minibatch точки переставляются; official test 10000 открыт после freeze. У реальных меток нет известного истинного I: сравнение индексаторов с ручным DeepSets — диагностика архитектурного сходства. В колонке свежего C все численные параметры, включая rho, переинициализированы.",
        "",
        table(
            [
                "Датасет",
                "Наш I + C",
                "Наш I + свежий C",
                "Плотный Adam",
                "Плотный SGDM",
                "Ручной DeepSets + Adam",
            ],
            image_rows,
        ),
        "",
        "![MNIST: выбранные индексаторы](images_MNIST/figures/selected_indexators.png)",
        "",
        "![FashionMNIST: выбранные индексаторы](images_FashionMNIST/figures/selected_indexators.png)",
        "",
        "## Две синусоиды и локальные минимумы",
        "",
        "24 Gaussian-входа, один неизвестный трёхгрупповой unit-вектор v. Цель 0.4 sin(0.7 vᵀx) + sin(f vᵀx). Ученик — одна sin с обучаемой амплитудой, без phase/bias. Она приближает сильную высокую компоненту; слабая низкая компонента даёт ненулевой floor. Число sin-нейронов задано заранее одинаково для всех методов.",
        "",
        "Для каждого замороженного I заново проверяются четыре старта при каждой норме 0.5/2/6/8. Query выбирает C среди этих 16 стартов, selection — структуру. Плотные Adam/SGD/SGDM независимо подбирают норму и шаг. Long Adam получает два старта по 12000 вместо восьми по 3000 шагов. Моментный контроль берёт направление mean(x*y) только по support и дообучает все 24 веса; он использует информированный старт и показан отдельно от случайных стартов. Точный Gaussian-риск и истинный v не участвуют в выборе.",
        "",
        table(
            [
                "f",
                "Наш I + C",
                "Наш I + свежий C",
                "Adam случайный",
                "SGDM случайный",
                "Adam long",
                "Adam моментный",
                "Истинный I + Adam",
            ],
            sine_rows,
        ),
        "",
        "Отдельный заранее зарегистрированный контроль: после канонизации P кластеризовать абсолютные мягкие численные веса в m групп, знаки сохранить в I, затем обучить новые C. Все варианты порога нуля и численные настройки заданы до просмотра sine-test результатов; выбор только по query/selection, новый тест после freeze. Этот постпроцесс отличается от прямого argmax P.",
        "",
        table(
            [
                "f",
                "Прямой DARTS",
                "Soft W → k-means",
                "Свежий C",
                "Ошибок I",
                "Свежих успехов",
            ],
            quantized_rows,
        ),
        "",
        "Таблица показывает точную population MSE, меньше лучше. В каждом summary сохранены нормы population-градиента и минимальные собственные значения полных Hessian в низком и высоком стационарных решениях. Наличие строгого плохого минимума не означает, что любой градиентный метод с любой инициализацией обязан в нём остаться.",
        "",
        "Дополнительная диагностика оптимизатора при неизменных I: начальные C получены усреднением mean(x*y) внутри их подписанных групп, затем нормированы и зашумлены на 2%. Показаны точная MSE и число успехов среди 32 таких стартов. Все численные значения обучались; инициализатор использовал только support. Это отдельный post-freeze контроль с источниковым подбором параметров, а не первоначальная Gaussian-инициализация.",
        "",
        table(
            [
                "f",
                "Прямой I + moment-C",
                "Квантованный I + moment-C",
                "Истинный I + moment-C",
            ],
            moment_rows,
        ),
        "",
        "Диагноз: при f=4 и f=6 ошибки прямого I остаются даже с хорошей численной инициализацией. При f=8 тот же фиксированный I достигает почти минимальной MSE после исправления инициализации C: здесь отдельной причиной неудачи был численный поиск. Случайные старты и старты с оценкой направления по support следует сравнивать отдельно и у нашего метода, и у dense Adam.",
        "",
        "### Проверка структуры при смене численных значений",
        "",
        "Это один вектор v и одно истинное разбиение на группы при трёх частотах, а не три независимых семейства. Его различные координатные значения примерно −0.289599, −0.144232 и +0.142582: две группы почти образуют пару C/−C. Хорошее приближение этой функции допускает ложный шэринг. Поэтому небольшой риск не доказывает, что извлечённый I переносится на произвольные новые значения истинных кодов.",
        "",
        "После заморозки выполнена аналитическая диагностика: проектирование v в линейное подпространство фиксированного подписанного I и одномерная минимизация точного Gaussian-риска по норме веса с аналитическим выбором амплитуды. Затем в истинных группах с фиксированным seed сгенерированы 32 новых направления. Ниже приведён численный минимум риска, достижимый при идеальном подборе C, и число новых функций, для которых этот минимум меньше 0.1. Это проверка представимости, а не результаты обучения на новых задачах.",
        "",
        table(
            [
                "f / индексатор",
                "Минимум на исходной функции",
                "Средний минимум при новых C",
                "Представимо с MSE<0.1",
            ],
            capacity_rows,
        ),
        "",
        "Полные предположения, principal angles подпространств и все 32 значения — `sine_capacity_diagnosis.json`. Данная проверка показывает недостаток переноса группировок, который может быть скрыт хорошим решением одной функции.",
        "",
        "![Синусы f=6: индексаторы и Adam](sine_6/figures/selected_indexators.png)",
        "",
        "## Что сохранено и ограничения сравнения",
        "",
        "В каталогах задач: specification/protocol, исполнявшиеся source snapshots, все поисковые кривые и выбранные checkpoints, полные численные метрики, P до/после переноса, точные выбранные целочисленные I, frozen/heldout manifests, PNG/SVG. Никаких программных тестов не создавалось и не запускалось.",
        "",
        "Структурный поиск требует отдельного многократного обучения P и C, подбора и проверки кандидатов. Его стоимость не равна одному dense fit; метрики после фиксации I не учитывать как полную стоимость метода. Сравнивать также все старты и все подтверждающие seed в JSON, не только основной выбранный запуск. Гарантия успеха и универсальное преимущество над Adam не установлены.",
        "",
        "Измеренная стоимость ниже включает все сохранённые численные обучения и поисковые обновления в каждом каталоге. Один model update — один minibatch одного численного старта; P update относится к одной архитектуре. Время — сумма измеренных интервалов обучения с промежуточной оценкой, без запуска Python и части CPU-постобработки. Общие controls двух синтетических продолжений учитываются только в исходном каталоге.",
        "",
        table(
            ["Каталог", "Численных model updates", "P updates", "Измерено, мин"],
            [
                [
                    name,
                    value["numeric_model_updates"],
                    value["architecture_model_updates"],
                    f"{value['timed_training_seconds'] / 60:.2f}",
                ]
                for name, value in costs.items()
            ],
        ),
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    save_json(output / "comparison.json", all_summaries)
    print(output / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
