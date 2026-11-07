"""Audit full projection searches and render the learned sharing matrices."""

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
    "nearest": "Ближайший код + centering",
    "task_centers": "Ближайший код; C по задаче",
    "proximal": "Проекция с ценой связи",
    "proximal_warm": "Проекция + dense warmup",
}


def report(output, reference):
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
        true_codes = np.r_[0.0, archive["numeric"][:4]]
    true_weights = true_codes[np.abs(truth)] * np.sign(truth)
    with np.load(reference / "selected_states.npz") as archive:
        dense = matched_columns(archive["dense_shared_adam_weights"], true_weights)
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 150})
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 9.8), constrained_layout=True)
    image = categorical(axes.flat[0], truth[order], 4, "Истинный индексатор")
    for axis, (name, label) in zip(list(axes.flat)[1:5], LABELS.items()):
        row = families[name]
        primary = row["primary"]
        image = categorical(
            axis,
            aligned[f"row_{primary['row']}_indices"][order],
            4,
            f"{label}\nошибок {primary['recovery']['categorical_errors']}/256; "
            f"точных I {row['exact_indexators']}/3",
        )
    bar = figure.colorbar(
        image,
        ax=list(axes.flat),
        location="bottom",
        ticks=range(-4, 5),
        shrink=0.5,
        pad=0.04,
        fraction=0.035,
    )
    bar.set_label("Индекс кодбука")
    maximum = np.abs(dense).max()
    image = axes.flat[5].imshow(
        dense[order], cmap="RdBu_r", vmin=-maximum, vmax=maximum
    )
    axes.flat[5].set_title("Плотный Adam: численные веса\nисходный контроль NMSE0.0323")
    axes.flat[5].set_xlabel("Скрытый нейрон")
    axes.flat[5].set_ylabel("Координата входа")
    figure.colorbar(image, ax=axes.flat[5], shrink=0.65)
    figure.suptitle("DeepSets: полный поиск, нулевой код и шэринг", fontsize=15)
    save_figure(figure, output / "figures/indexators")
    history = json.loads((output / "confirmation/history.json").read_text())
    roots = json.loads((output / "confirmation/rows.json").read_text())
    variance = json.loads((output / "specification.json").read_text())[
        "target_variance"
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for name, label in LABELS.items():
        root = next(
            i
            for i, row in enumerate(roots)
            if row["family"] == name and row["seed"] == 263
        )
        steps = [row["step"] for row in history]
        axes[0].plot(
            steps,
            [np.mean(row["train_mse"][root]) / variance for row in history],
            label=label,
        )
        axes[1].plot(
            steps,
            [np.mean(row["query_mse"][root]) / variance for row in history],
            label=label,
        )
        axes[2].plot(steps, [row["active_edges"][root] for row in history], label=label)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_title("Train NMSE при поиске")
    axes[1].set_title("Query NMSE при поиске")
    axes[2].set_title("Число связей; целевое число не задаётся")
    for axis in axes:
        axis.set_xlabel("Обновление")
        axis.grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    save_figure(figure, output / "figures/curves")
    costs = 0.0
    for phase in ("pilot", "confirmation"):
        costs += json.loads((output / phase / "complete.json").read_text())["seconds"]
    for name in (
        "pilot/hard_fits.npz",
        "confirmation/hard_fits.npz",
        "fresh_calibration.npz",
        "fresh_evaluation.npz",
    ):
        with np.load(output / name) as archive:
            costs += float(archive["seconds"])
    text = [
        "# DeepSets: извлечение полного индексатора проекцией",
        "",
        "Один ранее изученный синтетический учитель, все256 позиции доступны. Никакой готовой маски, ролей координат или истинного числа32 связей в поиске нет. Новый hard-forward nearest-code контроль и модификации: C только по task-loss, цена ненулевого кода в проекции, полный dense warmup3000 шагов.",
        "",
        "| Метод | Средняя selection NMSE | Средняя test NMSE, 3 seed | Точных масок/3 | Точных I/3 | Связей выбранного I | Средняя test NMSE32 свежих C |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in LABELS.items():
        row = families[name]
        text.append(
            f"| {label} | {row['selection_nmse_mean']:.6g} | {np.mean([r['test_nmse'] for r in row['per_seed']]):.6g} | {row['exact_masks']}/3 | {row['exact_indexators']}/3 | {row['primary']['recovery']['active_edges']} | {row['fresh']['mean_test_nmse']:.6g} |"
        )
    text.extend(
        [
            "",
            "## Проекция",
            "",
            "Мягкие смеси после warmup не используются. Для каждой позиции берётся ровно один код; при сохранении фиксируется ровно этот индекс. Ниже s — stop-gradient среднего квадрата четырёх численных кодов; lambda постепенно нарастает.",
            "",
            "$$",
            "I_e=\\arg\\min_k\\left[\\frac{(Z_e-D_k(C))^2}{s}+\\lambda\\,\\mathbf{1}_{k\\ne0}\\right],\\qquad W_e=D_{I_e}(C).",
            "$$",
            "",
            "Ненулевой штраф меняет границы решения в пользу нуля. Это стоимость выбора дискретного кода, не несмещённый градиент task-loss с L0. Z получает identity STE и commitment0.01; C получает градиент задачи по выбранным кодам. У nearest-контроля дополнительно centering0.01, в остальных C-centering отключён. Численные L2/Adam/clip общие. При инференсе храним I и C, заново округлять по обычному nearest нельзя.",
            "",
            "Pilot261/262 выбирает одну настройку каждой семьи, confirmation263/264/265 использует выбранные настройки. Численные скорости0.003/0.01; цена ненулевого кода0.001/0.01/0.1/1.0. Поиск12000 шагов, включая optional warmup3000. Начальные Z и support minibatch парные. Dense warmup не отключает связи: вся матрица остаётся обучаемой.",
            "",
            "Каждый снимок0/4000/8000/12000 оценивается после3000 C-fit шагов с одного исходного и трёх свежих C. Query выбирает численные checkpoints, средняя selection ошибка — структуру и настройки. Teacher и новый heldout8192 открываются после source/fresh-freeze. Четыре семьи не являются независимыми задачами; три seed — также не новые функции.",
            "",
            "32 свежих C на той же функции показывают численную сходимость, а не перенос структуры. Перенос требует новых целевых функций с новыми истинными коэффициентами при фиксированном I и оценивается отдельной стадией.",
            "",
            "Основа: [LUT-Q](https://arxiv.org/abs/1911.04951), [оптимизация VQ/STE](https://arxiv.org/abs/2305.08842), [ProxQuant](https://arxiv.org/abs/1810.00861). Здесь проверяются конкретные модификации извлечения sharing; новизна механизма проекции сама по себе не заявляется.",
            "",
            "![Индексаторы](figures/indexators.png)",
            "",
            "![Кривые](figures/curves.png)",
            "",
            "Полные выбранные I/C/W: selected_indexators.json. Согласование с teacher используется только для аудита и рисунков. Adam — исходный численный контроль, здесь не переобучался.",
            "",
            f"Измеренное обучение {costs / 60:.2f} мин. Программные тесты не создавались и не запускались.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(text))
    save_json(
        output / "complete.json",
        {
            "sha256": {
                str(p.relative_to(output)): digest(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != "complete.json"
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
