"""Consolidate the completed sharing interventions without rerunning fits."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from .data import digest, save_json
from .report import matched_columns


def run(root):
    names = (
        "deepsets_sharing_ablation",
        "deepsets_sharing_geometry",
        "deepsets_sharing_full",
    )
    summaries = {
        name: json.loads((root / name / "summary.json").read_text())["families"]
        for name in names
    }
    full_family = min(
        summaries[names[2]],
        key=lambda key: summaries[names[2]][key]["selection_nmse_mean"],
    )
    methods = [
        ("Полная матрица", "Исходный DARTS", names[0], "baseline"),
        ("Полная матрица", "DARTS + лучший commitment", names[0], "commit_constant"),
        ("Выученная маска", "Свободные логиты", names[1], "logits_w0"),
        ("Выученная маска", "Логиты + commitment", names[1], "logits_commit_w0"),
        ("Выученная маска", "Логиты после warmup1000", names[1], "logits_w1000"),
        (
            "Выученная маска",
            "Ближайший код + STE, без warmup",
            names[1],
            "nearest_ste_w0",
        ),
        ("Полная матрица", "Ближайший код + STE", names[2], full_family),
    ]
    rows = []
    for regime, label, folder, family in methods:
        result = summaries[folder][family]
        rows.append(
            {
                "regime": regime,
                "label": label,
                "folder": folder,
                "family": family,
                "result": result,
            }
        )
    costs = {}
    for name in names:
        directory = root / name
        seconds, search_updates, fit_updates = 0.0, 0, 0
        for phase in ("pilot", "confirmation"):
            complete = json.loads((directory / phase / "complete.json").read_text())
            seconds += complete["seconds"]
            search_updates += (
                complete["steps"] * complete["roots"] * complete["numeric_restarts"]
            )
        for path in (
            directory / "pilot/hard_fits.npz",
            directory / "confirmation/hard_fits.npz",
            directory / "fresh_calibration.npz",
            directory / "fresh_evaluation.npz",
        ):
            with np.load(path) as archive:
                seconds += float(archive["seconds"])
                fit_updates += int(np.prod(archive["numeric"].shape[:2])) * 3000
        costs[name] = {
            "measured_training_minutes": seconds / 60,
            "search_numeric_updates_excluding_original_warmup": search_updates,
            "hard_and_fresh_numeric_updates": fit_updates,
            "bytes": sum(
                path.stat().st_size for path in directory.rglob("*") if path.is_file()
            ),
        }
    save_json(root / "sharing_comparison.json", {"rows": rows, "costs": costs})
    with np.load(root / names[0] / "aligned_states.npz") as archive:
        truth = archive["true_indices"]
        row_order = archive["row_order"]
    with np.load(root / "deepsets_refinement/teacher_audit_only.npz") as archive:
        code_values = np.r_[0.0, archive["numeric"][:4]]
    true_weights = code_values[np.abs(truth)] * np.sign(truth)
    with np.load(root / "deepsets_refinement/selected_states.npz") as archive:
        dense = matched_columns(archive["dense_shared_adam_weights"], true_weights)
    panels = [("Истинный I", truth)]
    displayed = [rows[0], rows[1], rows[2], rows[5], rows[6]]
    for row in displayed:
        result = row["result"]
        number = result["primary"]["row"]
        with np.load(root / row["folder"] / "aligned_states.npz") as archive:
            matrix = archive[f"row_{number}_indices"]
        title = (
            f"{row['label']}\n{row['regime']}; "
            f"ошибок {result['primary']['recovery']['categorical_errors']}/256"
        )
        panels.append((title, matrix))
    colors = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, 9))
    colors[4] = [0.96, 0.96, 0.96, 1]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-4.5, 5), len(colors))
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 170})
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for axis, (title, matrix) in zip(axes.flat, panels):
        image = axis.imshow(
            matrix[row_order], cmap=cmap, norm=norm, interpolation="nearest"
        )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Скрытый нейрон")
        axis.set_ylabel("Координата входа")
    colorbar = figure.colorbar(
        image, ax=list(axes.flat)[:6], ticks=range(-4, 5), shrink=0.65
    )
    colorbar.set_label("Индекс группы; 0 = отключено")
    maximum = float(np.abs(dense).max())
    image = axes.flat[6].imshow(
        dense[row_order],
        cmap="RdBu_r",
        vmin=-maximum,
        vmax=maximum,
        interpolation="nearest",
    )
    axes.flat[6].set_title("Плотный Adam: реальные веса\nNMSE 0.0323", fontsize=10)
    axes.flat[6].set_xlabel("Скрытый нейрон")
    axes.flat[6].set_ylabel("Координата входа")
    figure.colorbar(image, ax=axes.flat[6], shrink=0.65)
    axes.flat[7].set_visible(False)
    figure.suptitle(
        "DeepSets: commitment, инициализация и дискретный прямой проход", fontsize=15
    )
    figures = root / "sharing_figures"
    figures.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(figures / f"comparison.{extension}", bbox_inches="tight")
    plt.close(figure)
    lines = [
        "# DeepSets: что исправляет поиск шэринга",
        "",
        "Проверены три отдельно зарегистрированные серии. Словарь во всех случаях [0,C1,…,C4,−C1,…,−C4], четыре независимых числа. Структура и настройки выбираются только по support/query/selection; teacher открывается после фиксации выбора. Это продолжение исследования одной ранее изученной численной функции, а не новый benchmark.",
        "",
        "| Область поиска | Вариант | Средняя selection NMSE по 3 seed | Точных I/3 | Ошибок выбранного I/256 | Test NMSE выбранного I |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        result = row["result"]
        primary = result["primary"]
        lines.append(
            f"| {row['regime']} | {row['label']} | {result['selection_nmse_mean']:.6g} | "
            f"{result['exact_indexators']}/3 | {primary['recovery']['categorical_errors']} | "
            f"{primary['test_nmse']:.6g} |"
        )
    lines += [
        "",
        "## Как читать результат",
        "",
        "Commitment при полном поиске улучшает функцию, но не восстанавливает истинные группы. На общей маске, ранее найденной DARTS, логиты с commitment дают2/3 точных I. Численный warmup1000 с инициализацией P и C по весам даёт3/3. Ближайший код в прямом проходе и identity straight-through дают3/3 без численного warmup. Это альтернативный способ поиска, не точный градиент через argmax и не обычный мягкий DARTS.",
        "",
        "Для nearest-STE каждая позиция имеет латентное число Z. В прямом проходе берётся ближайшее значение из подписанного кодбука. Градиент веса копируется на Z, а C получает сгруппированный градиент; добавлено слабое притяжение Z к ближайшему коду и кодов к назначенным Z. Центры изначально задаются квантилями случайных Z. Ни k-means, ни истинные связи, блоки или значения весов не используются.",
        "",
        "Это проверка условного поиска шэринга: одна и та же маска получена ранее из данных, но не переоткрывается на каждом новом seed. Статистика3/3 не означает3/3 успеха полного алгоритма с нуля. Полная матрица проверена отдельно и отражена в таблице. Свежие32 C — новые инициализации на той же функции, не смена истинной задачи.",
        "",
        "Инициализация по расстояниям и совместное обучение центров связаны с [Soft-to-Hard Vector Quantization](https://proceedings.neurips.cc/paper_files/paper/2017/file/86b122d4358357d834a87ce618a55de0-Paper.pdf). Прямой перенос градиента — приближённый [straight-through estimator](https://arxiv.org/abs/1308.3432). Здесь это собственная адаптация к знаковому кодбуку, не воспроизведение результатов этих работ.",
        "",
        "![Сравнение индексаторов и плотного Adam](sharing_figures/comparison.png)",
        "",
        "На дискретных панелях цвет обозначает имя группы, а не величину веса. Координаты, скрытые нейроны и имена групп согласованы с teacher только для отображения после выбора. Панель Adam показывает реальные численные веса.",
        "",
        "## Полные серии",
        "",
    ]
    for name in names:
        lines.append(f"- [{name}]({name}/REPORT.md)")
    lines += ["", "## Стоимость и сохранённые данные", ""]
    for name, cost in costs.items():
        lines.append(
            f"- {name}: измеренное обучение {cost['measured_training_minutes']:.2f} мин, артефакты {cost['bytes'] / 2**20:.1f} MiB."
        )
    lines += [
        "",
        "Время — сумма записанных интервалов поиска и численных fit, без части CPU-постобработки и запуска Python. Три серии имеют разные области поиска и бюджеты; это не сравнение равных FLOPs. В каждой серии сравнения парные по seed и minibatch. Сохранены все P/I, латентные веса геометрических методов, исходные численные состояния, кривые, свежие C, протоколы и executed source snapshots. Программные тесты не создавались и не запускались.",
    ]
    (root / "SHARING_ABLATIONS.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "status": "complete",
        "source_sha256": {
            path.name: digest(path)
            for path in Path(__file__).parent.glob("sharing_*.py")
        },
        "artifacts_sha256": {
            str(path.relative_to(root)): digest(path)
            for name in names
            for path in (root / name).rglob("*")
            if path.is_file()
        },
    }
    save_json(root / "sharing_complete.json", manifest)
    print(
        json.dumps(
            {"comparison": str(root / "SHARING_ABLATIONS.md"), "costs": costs}, indent=2
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
