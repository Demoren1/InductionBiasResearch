"""Audit frozen hard-sampling runs and render their indexators."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import digest, save_json
from .report import categorical, matched_columns, save_figure
from .sharing_report import evaluate

LABELS = {
    "soft": "Мягкая смесь",
    "soft_commit": "Мягкая смесь + commitment",
    "hard_argmax": "Hard argmax + STE",
    "gumbel_1": "Hard Gumbel: 1 сэмпл",
    "gumbel_4": "Hard Gumbel: 4 сэмпла",
    "gumbel_1_mass": "Hard Gumbel: перенос массы",
}


def figures(output, reference):
    families = json.loads((output / "summary.json").read_text())["families"]
    with np.load(output / "aligned_states.npz") as archive:
        aligned = dict(archive)
    with np.load(output / "confirmation/hard_fits.npz") as archive:
        fitted = dict(archive)
    selected = {}
    for family, result in families.items():
        primary = result["primary"]
        number, restart = primary["row"], primary["query_selected_restart"]
        selected[family] = {
            "selection": primary,
            "raw_indexator": fitted["indices"][number].tolist(),
            "aligned_indexator": aligned[f"row_{number}_indices"].tolist(),
            "numeric": fitted["numeric"][number, restart].tolist(),
            "weights": fitted["weights"][number, restart].tolist(),
            "train_mse": float(fitted["train_mse"][number, restart]),
            "query_mse": float(fitted["query_mse"][number, restart]),
            "selection_mse": float(fitted["selection_mse"][number, restart]),
        }
    save_json(output / "selected_indexators.json", selected)
    with np.load(output / "teacher_audit_only.npz") as archive:
        true_codes = np.r_[0.0, archive["numeric"][:4]]
    truth = aligned["true_indices"]
    order = aligned["row_order"]
    true_weights = true_codes[np.abs(truth)] * np.sign(truth)
    with np.load(reference / "selected_states.npz") as archive:
        dense = matched_columns(archive["dense_shared_adam_weights"], true_weights)
    figure, axes = plt.subplots(2, 4, figsize=(17, 9), constrained_layout=True)
    image = categorical(axes.flat[0], truth[order], 4, "Истинный индексатор")
    for axis, (family, label) in zip(list(axes.flat)[1:7], LABELS.items()):
        result = families[family]
        number = result["primary"]["row"]
        errors = result["primary"]["recovery"]["categorical_errors"]
        image = categorical(
            axis,
            aligned[f"row_{number}_indices"][order],
            4,
            f"{label}\nошибок {errors}/256; точных {result['exact_indexators']}/3",
        )
    figure.colorbar(image, ax=list(axes.flat)[:7], ticks=range(-4, 5), shrink=0.65)
    maximum = np.abs(dense).max()
    image = axes.flat[7].imshow(
        dense[order],
        cmap="RdBu_r",
        vmin=-maximum,
        vmax=maximum,
        interpolation="nearest",
    )
    axes.flat[7].set_title(
        "Плотный Adam: численные веса\nисходный контроль NMSE 0.0323"
    )
    axes.flat[7].set_xlabel("Скрытый нейрон")
    axes.flat[7].set_ylabel("Координата входа")
    figure.colorbar(image, ax=axes.flat[7], shrink=0.65)
    title = "На общей найденной маске" if output.name == "masked" else "Полная матрица"
    figure.suptitle(f"DeepSets: {title}; hard sampling", fontsize=16)
    save_figure(figure, output / "figures/indexators")

    stochastic_family = min(
        (name for name in families if name.startswith("gumbel_")),
        key=lambda name: families[name]["selection_nmse_mean"],
    )
    chosen = families[stochastic_family]["primary"]
    rows = json.loads((output / "confirmation/hard_fits.json").read_text())["rows"]
    row = rows[chosen["row"]]
    with np.load(output / "confirmation/snapshots.npz") as archive:
        policy = archive["policy_probabilities"][row["snapshot"], row["root"]]
    columns = aligned[f"row_{chosen['row']}_columns"]
    matrix = policy[order][:, columns]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for axis, values, label, maximum in (
        (axes[0], matrix[..., 0], "Вероятность нуля", 1),
        (axes[1], matrix.max(-1), "Вероятность наиболее вероятного кода", 1),
        (
            axes[2],
            -(matrix * np.log(np.maximum(matrix, 1e-12))).sum(-1),
            "Энтропия распределения",
            np.log(matrix.shape[-1]),
        ),
    ):
        image = axis.imshow(values, vmin=0, vmax=maximum, cmap="viridis")
        axis.set_title(label)
        axis.set_xlabel("Скрытый нейрон")
        axis.set_ylabel("Координата входа")
        figure.colorbar(image, ax=axis, shrink=0.75)
    figure.suptitle(f"{title}: {LABELS[stochastic_family]}, seed {chosen['seed']}")
    save_figure(figure, output / "figures/probabilities")

    history = json.loads((output / "confirmation/history.json").read_text())
    roots = json.loads((output / "confirmation/rows.json").read_text())
    variance = json.loads((output / "specification.json").read_text())[
        "target_variance"
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    for family, label in LABELS.items():
        number = next(
            i
            for i, root in enumerate(roots)
            if root["family"] == family and root["seed"] == 203
        )
        steps = [record["step"] for record in history]
        for axis, key in zip(axes[:2], ("query_mse", "hard_query_mse")):
            axis.plot(
                steps,
                [np.mean(record[key][number]) / variance for record in history],
                label=label,
            )
            axis.set_yscale("log")
        axes[2].plot(
            steps, [record["entropy"][number] for record in history], label=label
        )
    for axis, label in zip(
        axes,
        ("Query NMSE при обучении", "Query NMSE MAP до refit", "Энтропия политики"),
    ):
        axis.set_title(label)
        axis.set_xlabel("Шаг")
        axis.grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    figure.suptitle(f"{title}: общий seed 203; Gumbel query усреднён по 8 сэмплам")
    save_figure(figure, output / "figures/curves")


def report(root, reference):
    rows = []
    for regime in ("masked", "full"):
        output = root / regime
        summary = json.loads((output / "summary.json").read_text())
        for family in LABELS:
            result = summary["families"][family]
            rows.append({"regime": regime, "family": family, **result})
        figures(output, reference)
    costs = {}
    for regime in ("masked", "full"):
        output = root / regime
        seconds, forwards = 0.0, 0
        for phase in ("pilot", "confirmation"):
            complete = json.loads((output / phase / "complete.json").read_text())
            seconds += complete["seconds"]
            forwards += complete["effective_structure_forwards"]
        for name in (
            "pilot/hard_fits.npz",
            "confirmation/hard_fits.npz",
            "fresh_calibration.npz",
            "fresh_evaluation.npz",
        ):
            with np.load(output / name) as archive:
                seconds += float(archive["seconds"])
        costs[regime] = {
            "training_minutes": seconds / 60,
            "structure_forwards": forwards,
        }
    save_json(root / "comparison.json", {"rows": rows, "costs": costs})
    text = [
        "# DeepSets: жёсткое сэмплирование индексаторов",
        "",
        "Сравнение на одном ранее изученном синтетическом учителе. Маски и численные параметры выбираются только по source-данным. Три новых seed не являются тремя новыми задачами.",
        "",
        "В hard-вариантах каждая позиция в каждом прямом проходе имеет ровно один код из [0,C1,…,C4,−C1,…,−C4]. Индексатор общий для minibatch; четыре сэмпла используют один общий кодбук. Backward использует смещённую straight-through оценку.",
        "",
        "| Область | Метод | Средняя selection NMSE | Точных I/3 | Точных масок/3 | Ошибок выбранного I | Test NMSE выбранного I | Средняя test NMSE 32 свежих C |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        text.append(
            f"| {'Выученная маска' if row['regime'] == 'masked' else 'Полная матрица'} "
            f"| {LABELS[row['family']]} | {row['selection_nmse_mean']:.6g} "
            f"| {row['exact_indexators']}/3 | {row['exact_masks']}/3 "
            f"| {row['primary']['recovery']['categorical_errors']}/256 "
            f"| {row['primary']['test_nmse']:.6g} | {row['fresh']['mean_test_nmse']:.6g} |"
        )
    masked = {row["family"]: row for row in rows if row["regime"] == "masked"}
    full = {row["family"]: row for row in rows if row["regime"] == "full"}
    best_masked = min(
        (name for name in LABELS if name.startswith("gumbel_")),
        key=lambda name: masked[name]["selection_nmse_mean"],
    )
    best_full = min(
        (name for name in LABELS if name.startswith("gumbel_")),
        key=lambda name: full[name]["selection_nmse_mean"],
    )
    text.extend(
        [
            "",
            "## Что получилось",
            "",
            f"На найденной маске лучший сэмплирующий вариант — {LABELS[best_masked]}: средняя selection NMSE {masked[best_masked]['selection_nmse_mean']:.6g}, точных индексаторов {masked[best_masked]['exact_indexators']}/3. Мягкий контроль: {masked['soft']['selection_nmse_mean']:.6g}; детерминированный hard STE: {masked['hard_argmax']['selection_nmse_mean']:.6g}. Три запуска дают предварительное сравнение; надёжное преимущество шума над детерминированным hard forward не установлено.",
            "",
            f"Полный поиск: лучший сэмплирующий вариант {LABELS[best_full]}, NMSE {full[best_full]['selection_nmse_mean']:.6g}, точных I {full[best_full]['exact_indexators']}/3. Эти результаты следует отделять от поиска шэринга внутри готовой найденной маски.",
            "",
            "Численные значения выбранных кодбуков, полные матрицы I/W и train/query/selection ошибки экспортированы в masked/selected_indexators.json и full/selected_indexators.json. Изображения групп согласованы с teacher только после выбора.",
        ]
    )
    text.extend(
        [
            "",
            "## Протокол",
            "",
            "По24 настройки в каждой области. Pilot201/202 выбирает скорость обучения и, для полного поиска, sparsity по hard-refit selection NMSE. Confirmation203/204/205 использует выбранные настройки. Каждый поиск12000 шагов, один численный старт. Контроль soft_commit добавляет нормированный commitment0.1. Точные настройки сохранены в registration.json и family_selection.json.",
            "",
            "После снимков0/4000/8000/12000 фиксируется MAP индексатор именно обучаемой политики, без дополнительного переноса массы. Кодбук дообучается3000 шагов с одного исходного и трёх свежих Gaussian-стартов; query выбирает численные checkpoints, средняя selection ошибка выбирает структуру. Диагностическая ошибка случайных структур не используется для выбора.",
            "",
            "Маска в masked взята из одного ранее выбранного DARTS запуска; вне неё связи запрещены, внутри разрешён нулевой код. Full начинает со всех256 допустимых позиций. Масса +/- переносится до sampling только в отдельной ветке gumbel_1_mass; это изменение распределения, а не эквивалентная канонизация стохастического loss.",
            "",
            "Температура backward плавно меняется1→0.3, вероятность sampling остаётся softmax(alpha). Во всех ветках одинаковый поздний entropy penalty0.02. Начальные alpha/C и support minibatch парные, Gumbel noise парный по seed, один сэмпл — префикс четырёх. Gumbel4 использует в4 раза больше стохастических оценок loss за шаг. Реализация пакетирует все ветки по максимуму4, поэтому отдельные wall-clock затраты методов из общего времени не выводятся.",
            "",
            "Fresh-C calibration использует8 стартов на сочетание rate/radius; затем32 новые Gaussian-инициализации при зафиксированных структуре и настройках. Это проверка оптимизатора на той же функции, не перенос на другой истинный кодбук.",
            "",
            "## Индексаторы и вероятности",
            "",
            "Имена групп и нейроны согласованы с teacher только для финального аудита и рисунков. Панель Adam показывает исходный численный контроль; он здесь повторно не обучался.",
            "",
            "![Индексаторы на маске](masked/figures/indexators.png)",
            "",
            "![Вероятности на маске](masked/figures/probabilities.png)",
            "",
            "![Полный поиск](full/figures/indexators.png)",
            "",
            "![Вероятности полного поиска](full/figures/probabilities.png)",
            "",
            "![Кривые на маске](masked/figures/curves.png)",
            "",
            "![Кривые полного поиска](full/figures/curves.png)",
            "",
            "## Затраты",
            "",
        ]
    )
    for regime, cost in costs.items():
        text.append(
            f"- {regime}: измеренное обучение {cost['training_minutes']:.2f} мин."
        )
    text.extend(
        [
            "",
            "Сохранены регистрации, исходники выполненных запусков, P, alpha, I, C, кривые и refit. Программные тесты не создавались и не запускались.",
            "",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(text))
    save_json(
        root / "complete.json",
        {
            "sha256": {
                str(path.relative_to(root)): digest(path)
                for path in root.rglob("*")
                if path.is_file() and path.name != "complete.json"
            }
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    reporting_snapshot = args.root / "reporting_code_snapshot.py"
    reporting_snapshot.write_bytes(Path(__file__).read_bytes())
    if not args.report_only:
        for regime in ("masked", "full"):
            evaluate(args.root / regime)
    report(args.root, args.reference)


if __name__ == "__main__":
    main()
