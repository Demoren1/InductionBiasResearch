"""Inspect full-matrix consensus search after all source choices are frozen."""

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
    "quadratic": "Квадратичное согласование",
    "damped_admm": "ADMM, шаг dual0.25",
    "admm": "ADMM, шаг dual1",
}


def report(output, reference):
    plan = json.loads((output / "protocol.json").read_text())
    dynamic = plan["consensus"].get("projection", "lloyd") == "dynamic"
    families = json.loads((output / "summary.json").read_text())["families"]
    with np.load(output / "aligned_states.npz") as archive:
        aligned = dict(archive)
    with np.load(output / "confirmation/hard_fits.npz") as archive:
        fitted = dict(archive)
    selected = {}
    for name, row in families.items():
        primary = row["primary"]
        number, restart = primary["row"], primary["query_selected_restart"]
        selected[name] = {
            "source_selection": primary,
            "raw_indexator": fitted["indices"][number].tolist(),
            "aligned_indexator": aligned[f"row_{number}_indices"].tolist(),
            "numeric": fitted["numeric"][number, restart].tolist(),
            "weights": fitted["weights"][number, restart].tolist(),
        }
    save_json(output / "selected_indexators.json", selected)
    truth, order = aligned["true_indices"], aligned["row_order"]
    with np.load(output / "teacher_audit_only.npz") as archive:
        values = np.r_[0.0, archive["numeric"][:4]]
    true_weights = values[np.abs(truth)] * np.sign(truth)
    with np.load(reference / "selected_states.npz") as archive:
        dense = matched_columns(archive["dense_shared_adam_weights"], true_weights)
    figure, axes = plt.subplots(1, 5, figsize=(20, 5.8), constrained_layout=True)
    image = categorical(axes[0], truth[order], 4, "Истинный индексатор")
    for axis, (name, label) in zip(axes[1:4], LABELS.items()):
        primary = families[name]["primary"]
        image = categorical(
            axis,
            aligned[f"row_{primary['row']}_indices"][order],
            4,
            f"{label}\nошибок{primary['recovery']['categorical_errors']}/256; точных I{families[name]['exact_indexators']}/3",
        )
    figure.colorbar(
        image,
        ax=list(axes[:4]),
        location="bottom",
        ticks=range(-4, 5),
        shrink=0.55,
        fraction=0.045,
        pad=0.04,
    )
    maximum = np.abs(dense).max()
    image = axes[4].imshow(
        dense[order], cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="auto"
    )
    axes[4].set_title("Плотный Adam\nисходный контроль0.0323")
    axes[4].set_xlabel("Скрытый нейрон")
    figure.colorbar(image, ax=axes[4], shrink=0.65)
    figure.suptitle("DeepSets: непрерывная матрица и дискретное согласование")
    save_figure(figure, output / "figures/indexators")
    history = json.loads((output / "confirmation/history.json").read_text())
    roots = json.loads((output / "confirmation/rows.json").read_text())
    variance = json.loads((output / "specification.json").read_text())[
        "target_variance"
    ]
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)
    for name, label in LABELS.items():
        root = next(
            i
            for i, row in enumerate(roots)
            if row["family"] == name and row["seed"] == 283
        )
        steps = [row["step"] for row in history]
        for axis, field in zip(axes[:2], ("train_mse", "query_mse")):
            axis.plot(
                steps,
                [np.mean(row[field][root]) / variance for row in history],
                label=label,
            )
        axes[2].plot(
            steps,
            [np.mean(row["hard_query_mse"][root]) / variance for row in history],
            label=label,
        )
        axes[3].plot(steps, [row["active_edges"][root] for row in history], label=label)
    for axis, title in zip(
        axes,
        (
            "Train NMSE, Z",
            "Query NMSE, Z",
            "Query NMSE, C[I] до refit",
            "Число ненулевых связей",
        ),
    ):
        axis.set_title(title)
        axis.set_xlabel("Обновление")
        axis.grid(alpha=0.2)
    for axis in axes[:3]:
        axis.set_yscale("log")
    axes[3].legend(fontsize=8)
    save_figure(figure, output / "figures/curves")
    lines = [
        "# DeepSets: согласование непрерывной и дискретной матриц",
        "",
        "Полный поиск256 позиций без готовой маски, истинного числа связей, групп или чисел учителя. Сравниваются квадратичный штраф без dual и два шага dual-переменной. Никакая позиция не фиксируется до окончательного source-выбора.",
        "",
        "| Метод | Средняя selection NMSE | Средняя test NMSE,3seed | Точных масок/3 | Точных I/3 | Связей выбранного I | Средняя test NMSE32 свежих C |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in LABELS.items():
        row = families[name]
        lines.append(
            f"| {label} | {row['selection_nmse_mean']:.6g} | {np.mean([item['test_nmse'] for item in row['per_seed']]):.6g} | {row['exact_masks']}/3 | {row['exact_indexators']}/3 | {row['primary']['recovery']['active_edges']} | {row['fresh']['mean_test_nmse']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Алгоритм и границы",
            "",
            "Сначала 2000 непрерывных обновлений Z, затем 10000 обновлений со связью с проекцией Q=C[I]. На support оптимизируется L(Z)+rho/2*mean((Z-Q+U)^2). Каждые 50 шагов обновляются индексатор и четыре центра по Z+U: "
            + (
                "точной одномерной проекцией"
                if dynamic
                else "локальными Lloyd-итерациями"
            )
            + ", с фиксированным нулём и ценой ненулевого выбора. Затем U+=eta*(Z-Q). Сравниваются eta0, eta0.25 и eta1. Rho растёт геометрически от 0.01 до 1/100; U масштабируется обратно при изменении rho.",
            "",
            "Цена lambda выбирается из 0/0.003/0.03/0.3. Число активных связей неизвестно поиску. Внутренняя непрерывная задача не решается точно; гарантий глобальной сходимости нейросетевого поиска нет. Forward при поиске использует Z, итоговая сеть — дискретный C[I]. Ошибка Z отдельно показана на графиках; итоговая таблица оценивает только жёсткий индексатор после C-fit.",
            "",
            "24 настройки, pilot281/282; одна настройка на семью фиксируется до confirmation283/284/285. Снимки0/4000/8000/12000 отбираются только после3000 hard-C-fit шагов с1исходным+3свежими Gaussian стартами. Query выбирает численные checkpoints, средняя selection выбирает I/настройку. Новый heldout8192 и teacher-аудит после source/freshfreeze. Fresh32C на прежней функции — численный контроль, не перенос.",
            "",
            "Литературная основа — [ADMM для pruning и clustering](https://arxiv.org/abs/1811.01907). Общий механизм не заявляется новым; здесь исследуется teacher-blind восстановление полного I со signed-кодбуком. Adam на рисунке — прежний контроль, не повторное обучение.",
            "",
            "![Индексаторы](figures/indexators.png)",
            "",
            "![Кривые](figures/curves.png)",
            "",
            "Полные выбранные I/C/W: selected_indexators.json; исходники, регистрация, freeze и states сохранены. Программные тесты не создавались и не запускались.",
            "",
        ]
    )
    if dynamic:
        lines.extend(
            [
                "## Точная проекция вместо локального Lloyd",
                "",
                "Одномерный DP сортирует абсолютные латентные веса, выбирает нулевой префикс произвольной длины и до четырёх непустых положительных кластеров. Цена каждого ненулевого значения — 2*lambda/rho. Выбираются и число нулевых позиций, и центры, и назначения; исходные знаки восстанавливают signed-I. Интервальная SSE вычисляется в float64, сложность O(K*256^2). Оптимальна эта проекция; гарантия не распространяется на весь нейросетевой поиск.",
                "",
                "Общая идея — [оптимальный одномерный k-means](https://journal.r-project.org/articles/RJ-2011-015/); здесь добавлены фиксированный нулевой префикс, штраф активных позиций и signed-кодбук. Одинаковые seed281–285, начальные Z и minibatch позволяют сравнивать с локальной проекцией; это парное сравнение на прежней функции, не новые независимые задачи. Ни отсутствие пустых центров, ни их заполнение сами по себе не гарантируют истинный sharing.",
                "",
            ]
        )
    (output / "REPORT.md").write_text("\n".join(lines))
    save_json(
        output / "complete.json",
        {
            "sha256": {
                str(path.relative_to(output)): digest(path)
                for path in output.rglob("*")
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
    (args.root / "reporting_code_snapshot.py").write_bytes(Path(__file__).read_bytes())
    if not args.report_only:
        evaluate(args.root)
    report(args.root, args.reference)


if __name__ == "__main__":
    main()
