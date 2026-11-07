"""Report frozen Pattern architecture search, hardening and fresh-codebook fits."""

import argparse
import csv
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.optimize import linear_sum_assignment

from pattern.datasets import input_table, labels_for

from .data import BASE, digest, load_archive, save_json
from .model import metrics
from .run import combined

LABELS = {
    "darts_primary": "DARTS selected I",
    "learned_binary": "Learned support, untied",
    "random_sharing_on_support": "Random groups, learned support",
    "random_same_code_counts": "Random positions, same group sizes",
    "initial_indexator": "Initial random I",
    "dense_shared": "Dense, shared head",
    "dense_full": "Dense MLP",
    "oracle_shared": "Oracle I",
    "oracle_binary": "Oracle support, untied",
    "pilot_best_reference": "Pilot winner (source-selected)",
}


def label(name):
    return LABELS.get(name, name.replace("_", " "))


def table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def save_figure(figure, output, name):
    figure.savefig(output / "figures" / (name + ".png"), dpi=150, bbox_inches="tight")
    figure.savefig(output / "figures" / (name + ".svg"), bbox_inches="tight")
    plt.close(figure)


def align(indices, truth):
    best = None
    for permutation in itertools.permutations(range(1, 5)):
        renamed = np.array((0, *permutation))[indices]
        cost = (renamed[:, :, None] != truth[:, None, :]).sum(0)
        source, target = linear_sum_assignment(cost)
        candidate = np.empty_like(indices)
        candidate[:, target] = renamed[:, source]
        errors = int((candidate != truth).sum())
        if best is None or errors < best[0]:
            best = (errors, candidate, permutation, source, target)
    binary_cost = -((indices > 0).astype(int).T @ (truth > 0).astype(int))
    source, target = linear_sum_assignment(binary_cost)
    binary = np.empty_like(indices, dtype=bool)
    binary[:, target] = indices[:, source] > 0
    expected = truth > 0
    true_positive = int((binary & expected).sum())
    false_positive = int((binary & ~expected).sum())
    false_negative = int((~binary & expected).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    summary = {
        "categorical_errors": best[0],
        "categorical_accuracy": 1 - best[0] / 88,
        "categorical_exact": best[0] == 0,
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "binary_exact": bool(np.array_equal(binary, expected)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "group_mapping": [0, *best[2]],
        "source_columns": best[3].tolist(),
        "target_columns": best[4].tolist(),
    }
    return best[1], summary


def indexator_figure(output, primary):
    rows, columns = np.arange(11)[:, None], np.arange(8)[None]
    truth = np.where((rows >= columns) & (rows < columns + 4), rows - columns + 1, 0)
    aligned, summary = align(primary, truth)
    figure, axes = plt.subplots(2, 2, figsize=(10, 10))
    code_colors = ["#edf0f2", "#4879b8", "#e49d48", "#65a58d", "#ac72b3"]
    cmap = ListedColormap(code_colors)
    norm = BoundaryNorm(np.arange(-0.5, 5.5), 5)
    for axis, values, title in (
        (axes[0, 0], truth, "True indexator (diagnostic only)"),
        (axes[0, 1], primary, "Selected I: raw canonical labels"),
        (
            axes[1, 0],
            aligned,
            f"Selected I: aligned after freezing\n{summary['categorical_errors']} / 88 categorical errors",
        ),
    ):
        axis.imshow(values, cmap=cmap, norm=norm, aspect="equal")
        axis.set_title(title)
    errors = np.zeros((11, 8), dtype=int)
    errors[(aligned > 0) & (truth > 0)] = 1
    errors[(aligned > 0) & (truth == 0)] = 2
    errors[(aligned == 0) & (truth > 0)] = 3
    errors[(aligned > 0) & (truth > 0) & (aligned != truth)] = 4
    axes[1, 1].imshow(
        errors,
        cmap=ListedColormap(["#edf0f2", "#65a58d", "#d76a73", "#efbd65", "#ac72b3"]),
        vmin=0,
        vmax=4,
    )
    axes[1, 1].set_title(
        f"Support and group errors\nBinary F1 {100 * summary['binary_f1']:.2f}%"
    )
    for axis in axes.ravel():
        axis.set_xticks(range(8))
        axis.set_yticks(range(11))
        axis.set_xlabel("Hidden unit")
        axis.set_ylabel("Observed input coordinate")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in code_colors]
    figure.legend(
        handles,
        ["0: off", "code 1", "code 2", "code 3", "code 4"],
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.035),
    )
    figure.text(
        0.5,
        0.012,
        "Error panel: green = correct; red = extra; yellow = missing; purple = wrong group",
        ha="center",
        fontsize=9,
    )
    figure.suptitle(
        "Pattern: selected indexator; teacher alignment is for display only", y=0.99
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.97))
    save_figure(figure, output, "selected_indexator")
    save_json(output / "structure_metrics.json", summary)
    save_json(
        output / "selected_indexator.json",
        {
            "raw": primary.tolist(),
            "aligned": aligned.tolist(),
            "true": truth.tolist(),
            **summary,
        },
    )
    return summary


def probability_figure(output, frozen):
    origin = frozen["primary"]["origin"]
    folder = output / origin["shard"]
    snapshots = load_archive(folder / "snapshots.npz")
    history = json.loads((folder / "history.json").read_text())
    record = next(row for row in history if row["step"] == origin["step"])
    temperature = record["temperature"][origin["root"]]
    alpha = snapshots["alpha"][origin["snapshot"], origin["root"]]
    numeric = snapshots["numeric"][origin["snapshot"], origin["root"]]
    probabilities = torch.softmax(torch.tensor(alpha) / temperature, -1).numpy()
    raw_indices = probabilities.argmax(-1)
    canonical = load_archive(output / "frozen_indexators.npz")["primary"]
    for permutation in itertools.permutations(range(1, 5)):
        mapping = np.array((0, *permutation))
        renamed = mapping[raw_indices]
        order = sorted(range(8), key=lambda column: tuple(renamed[:, column]))
        if np.array_equal(renamed[:, order], canonical):
            inverse = np.argsort(mapping)
            canonical_probabilities = probabilities[:, order][:, :, inverse]
            canonical_alpha = alpha[:, order][:, :, inverse]
            canonical_numeric = numeric.copy()
            canonical_numeric[..., :4] = numeric[..., inverse[1:] - 1]
            break
    else:
        raise ValueError(
            "Cannot couple saved probabilities to the frozen hard indexator"
        )
    np.savez_compressed(
        output / "selected_probabilities.npz",
        probabilities=canonical_probabilities,
        alpha=canonical_alpha,
        numeric=canonical_numeric,
        indices=canonical,
        temperature=temperature,
        raw_probabilities=probabilities,
        raw_numeric=numeric,
        raw_indices=raw_indices,
    )
    save_json(
        output / "selected_probabilities.json",
        {
            "probabilities": canonical_probabilities.tolist(),
            "indices": canonical.tolist(),
            "temperature": temperature,
            "sum_over_codes_max_error": float(
                np.abs(canonical_probabilities.sum(-1) - 1).max()
            ),
            "canonicalization_uses_teacher": False,
        },
    )
    figure, axes = plt.subplots(2, 3, figsize=(12, 10), layout="constrained")
    for code, axis in enumerate(axes.ravel()[:5]):
        plotted = axis.imshow(
            canonical_probabilities[..., code], vmin=0, vmax=1, cmap="viridis"
        )
        axis.set_title(
            f"P{code}"
            + (": fixed zero code" if code == 0 else f": coefficient of C{code}")
        )
    figure.colorbar(
        plotted, ax=list(axes.ravel()[:5]), label="Assignment probability", shrink=0.8
    )
    cmap = ListedColormap(["#edf0f2", "#4879b8", "#e49d48", "#65a58d", "#ac72b3"])
    norm = BoundaryNorm(np.arange(-0.5, 5.5), 5)
    hard = axes.ravel()[5].imshow(canonical, cmap=cmap, norm=norm)
    axes.ravel()[5].set_title("I = argmax over P0,...,P4")
    figure.colorbar(hard, ax=axes.ravel()[5], ticks=range(5), shrink=0.8)
    for row in range(11):
        for column in range(8):
            axes.ravel()[5].text(
                column,
                row,
                str(canonical[row, column]),
                ha="center",
                va="center",
                fontsize=8,
            )
    for axis in axes.ravel():
        axis.set_xticks(range(8))
        axis.set_yticks(range(11))
        axis.set_xlabel("Hidden unit")
        axis.set_ylabel("Observed input coordinate")
    figure.suptitle(
        "Five learned assignment matrices and the selected hard I\nCanonicalization by group renaming and column sorting; no teacher alignment"
    )
    save_figure(figure, output, "assignment_probabilities")


def comparison_figure(output):
    methods = json.loads((output / "audit_methods.json").read_text())
    matrices = load_archive(output / "audit_indexators.npz")["indices"]
    rows, columns = np.arange(11)[:, None], np.arange(8)[None]
    truth = np.where((rows >= columns) & (rows < columns + 4), rows - columns + 1, 0)
    names = ["darts_primary", "pilot_best_reference"] + [
        row["name"] for row in methods if row["name"].startswith("darts_seed_")
    ]
    records = {}
    for name in names:
        number = next(
            number for number, row in enumerate(methods) if row["name"] == name
        )
        raw = matrices[number]
        aligned, structural = align(raw, truth)
        records[name] = {"raw": raw.tolist(), "aligned": aligned.tolist(), **structural}
    selected = records["darts_primary"]
    panels = [
        (truth, "True I (diagnostic only)"),
        (np.array(selected["raw"]), "Primary I: raw canonical labels"),
        (
            np.array(selected["aligned"]),
            f"Primary I: aligned\nF1 {100 * selected['binary_f1']:.2f}%; {selected['categorical_errors']}/88 code errors",
        ),
    ]
    for name in names[1:]:
        row = records[name]
        panels.append(
            (
                np.array(row["aligned"]),
                f"{label(name)}: aligned\nF1 {100 * row['binary_f1']:.2f}%; {row['categorical_errors']}/88 code errors",
            )
        )
    figure, axes = plt.subplots(2, 3, figsize=(14, 11))
    cmap = ListedColormap(["#edf0f2", "#4879b8", "#e49d48", "#65a58d", "#ac72b3"])
    norm = BoundaryNorm(np.arange(-0.5, 5.5), 5)
    for axis, (values, title) in zip(axes.ravel(), panels):
        axis.imshow(values, cmap=cmap, norm=norm)
        axis.set_title(title, fontsize=11)
        axis.set_xticks(range(8))
        axis.set_yticks(range(11))
        axis.set_xlabel("Hidden unit")
        axis.set_ylabel("Observed input coordinate")
        for row in range(11):
            for column in range(8):
                axis.text(
                    column,
                    row,
                    str(values[row, column]),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    figure.suptitle(
        "Selected indexators from independent search seeds and the pilot\nAll teacher alignment is post-freeze diagnostic only"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(figure, output, "indexator_comparison")
    save_json(output / "selected_indexators.json", records)
    np.savez_compressed(
        output / "selected_indexators.npz",
        names=np.array(names),
        raw=np.stack([records[name]["raw"] for name in names]),
        aligned=np.stack([records[name]["aligned"] for name in names]),
        true=truth,
    )
    return records


def search_curves(output, frozen):
    origin = frozen["primary"]["origin"]
    folder = output / origin["shard"]
    history = json.loads((folder / "history.json").read_text())
    root = origin["root"]
    steps = np.array([row["step"] for row in history])
    train = np.array([row["train"][root] for row in history])
    query = np.array([row["query"][root] for row in history])
    hard = np.array([row["inherited_hard_query"][root] for row in history])
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(steps, train[:, 0], label="Support BCE")
    axes[0, 0].plot(steps, query[:, 0], label="Architecture-query BCE")
    axes[0, 0].set_title("Continuous model: measured losses")
    axes[0, 1].plot(steps, query[:, 0], label="Soft assignments")
    axes[0, 1].plot(steps, hard[:, 0], label="Argmax, inherited codebooks")
    axes[0, 1].set_title("Immediate discretization gap")
    axes[1, 0].plot(steps, query[:, 2] * 100, label="Soft recall")
    axes[1, 0].plot(steps, hard[:, 2] * 100, label="Argmax recall")
    axes[1, 0].set_ylabel("Positive-class recall, %")
    axes[1, 1].plot(
        steps,
        [row["entropy"][root] for row in history],
        label="Mean assignment entropy",
    )
    axes[1, 1].plot(
        steps,
        [row["maximum_probability"][root] for row in history],
        label="Mean maximum probability",
    )
    axes[1, 1].set_title("Assignment concentration")
    for axis in axes.ravel():
        axis.axvline(
            origin["step"], color="black", linestyle=":", label="Selected checkpoint"
        )
        axis.set_xlabel("Architecture updates")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Selected search seed; query observations update architecture")
    figure.tight_layout()
    save_figure(figure, output, "search_loss_recall")


def fitting_curves(output, filename, title, arrays, groups):
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, ids in groups:
        steps = arrays["history_steps"]
        axes[0].plot(steps, arrays["train_history"][:, ids].mean(1), label=name)
        axes[1].plot(steps, arrays["query_history"][:, ids].mean(1), label=name)
        axes[2].plot(steps, arrays["recall_history"][:, ids].mean(1) * 100, label=name)
    for axis, ylabel in zip(
        axes, ("Support balanced BCE", "Query balanced BCE", "Query positive recall, %")
    ):
        axis.set_xlabel("Numeric optimizer updates")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    save_figure(figure, output, filename)


def analytic_oracle(plan, output):
    _, inputs = input_table()
    records = []
    for pattern in plan["target_patterns"]:
        weights = torch.zeros(11, 8)
        bits = torch.tensor([int(bit) * 2 - 1 for bit in pattern])
        for hidden in range(8):
            weights[hidden : hidden + 4, hidden] = bits
        logits = (inputs @ weights - 3).relu().sum(-1) * 20 - 10
        score = metrics(logits[None], labels_for(inputs, pattern)[None])[0]
        records.append(
            {
                "pattern": pattern,
                "balanced_accuracy": float(score[1]),
                "recall": float(score[2]),
            }
        )
    save_json(
        output / "analytic_oracle.json",
        {"privileged": True, "all_2048_inputs": records},
    )


def report(output):
    plan = json.loads((output / "protocol.json").read_text())
    frozen = json.loads((output / "frozen.json").read_text())
    fine_tune = json.loads((output / "source_hard_fine_tune.json").read_text())
    summary = json.loads((output / "target_summary.json").read_text())
    confirmation = json.loads((output / "confirmation_selection.json").read_text())
    (output / "figures").mkdir(exist_ok=True)
    structural = indexator_figure(
        output, load_archive(output / "frozen_indexators.npz")["primary"]
    )
    probability_figure(output, frozen)
    selected_structures = comparison_figure(output)
    search_curves(output, frozen)
    warm = load_archive(output / "source_hard_fine_tune.npz")
    fitting_curves(
        output,
        "fine_tune_loss_recall",
        "Fixed I: continue inherited codebooks on discovery tasks",
        warm,
        [("Inherited codebooks, fixed I", np.arange(len(warm["parameters"])))],
    )
    rows, fits = combined(output / "target")
    main = [
        "darts_primary",
        "random_sharing_on_support",
        "learned_binary",
        "oracle_shared",
        "dense_full",
    ]
    methods = json.loads((output / "audit_methods.json").read_text())
    groups = [
        (
            label(name),
            [
                number
                for number, row in enumerate(rows)
                if methods[row["candidate"]]["name"] == name
            ],
        )
        for name in main
    ]
    fitting_curves(
        output,
        "transfer_loss_recall",
        "Four target functions, 32 fresh starts each; all starts averaged",
        fits,
        groups,
    )
    analytic_oracle(plan, output)
    numeric_rows = [
        [
            label(name),
            row["active_edges"],
            row["numeric_parameters"],
            f"{100 * row['mean_start_test_balanced_accuracy']:.2f}",
            f"{100 * row['selected_test_balanced_accuracy']:.2f}",
            f"{100 * row['selected_test_recall']:.2f}",
        ]
        for name, row in summary.items()
    ]
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "method",
                "active_edges",
                "numeric_parameters",
                "mean_start_test_BA_percent",
                "query_selected_test_BA_percent",
                "query_selected_test_recall_percent",
            ]
        )
        writer.writerows(numeric_rows)
    confirmation_rows = [
        [
            group["configuration"],
            group["order"],
            f"{1000 * group['mean_selection_bce']:.2f}",
            ", ".join(
                f"{100 * row['selection_balanced_accuracy']:.2f}"
                for row in group["seeds"]
            ),
        ]
        for group in confirmation["groups"]
    ]
    source_rows = [
        [name, f"{value[0]:.2f}", f"{100 * value[1]:.2f}", f"{100 * value[2]:.2f}"]
        for name, value in (
            ("Soft, selected checkpoint", fine_tune["inherited_soft_query"]),
            ("Argmax, same codebooks", fine_tune["inherited_hard_query"]),
            ("Fixed I, codebooks fine-tuned", fine_tune["after_fine_tune_query"]),
        )
    ]
    result = {
        "structural": structural,
        "selected_structures": selected_structures,
        "source_hardening": fine_tune,
        "confirmation": confirmation,
        "target": summary,
        "primary_search_step": frozen["primary"]["origin"]["step"],
        "software_tests_run": False,
    }
    save_json(output / "summary.json", result)
    text = (
        "\n\n".join(
            [
                "# Pattern: чередующийся DARTS-поиск индексатора и кодбуков",
                (
                    f"Схема обучения P/C и дообучения после argmax выполнена. Точный I "
                    f"не восстановлен: основной confirmation-selected индексатор даёт "
                    f"{100 * summary['darts_primary']['selected_test_balanced_accuracy']:.2f}% "
                    f"test BA, отдельный source-selected победитель пилота — "
                    f"{100 * summary['pilot_best_reference']['selected_test_balanced_accuracy']:.2f}%. "
                    f"Для той же основной бинарной маски снятие шэринга повышает BA до "
                    f"{100 * summary['learned_binary']['selected_test_balanced_accuracy']:.2f}%. "
                    f"С истинным I и заново обученным кодбуком — "
                    f"{100 * summary['oracle_shared']['selected_test_balanced_accuracy']:.2f}%. "
                    "Таким образом, в этом опыте найдена часть связности, но назначения "
                    "групп пока не дают качественного переноса."
                ),
                (
                    "Реализована согласованная схема W=sum(P_i*C_i), C_0=0: общий индексатор "
                    "11×8×5, отдельные кодбуки десяти исходных задач и четыре численных старта на "
                    "задачу. Исходный Pattern 11→8 ReLU→1 сохранён. Четыре числа C, общее скрытое "
                    "смещение, общий readout и offset дают 7 параметров. Геометрия окон, число "
                    "активных связей и размеры групп в поиск не передавались. После argmax "
                    "индексатор фиксируется; отдельно выполнены дообучение унаследованных чисел "
                    "и проверка переноса с совершенно новыми кодбуками."
                ),
                (
                    "Основа — [DARTS, §2–3](https://arxiv.org/pdf/1806.09055). Чередуются "
                    "три support-обновления чисел и одно query-обновление архитектуры. Unrolled-"
                    "контроль дифференцирует через один виртуальный Adam-шаг с текущими моментами. "
                    "Это приближение адаптации, не точный градиент полностью решённой внутренней "
                    "задачи. Архитектура оптимизируется по данным, GT-меток индексатора нет."
                ),
                "## Настройка на исходных функциях",
                (
                    "Пилот: 12 конфигураций × два search seed, 3000 outer-шагов после 512 numeric "
                    "warmup. Настраивались numeric_lr .03/.1, alpha_lr .003/.01/.03, soft или "
                    "annealed режим. Annealed снижает T до .1 и поздно вводит штраф энтропии .02. "
                    "72 различных жёстких I проверены на отдельных функциях 1110/1111 с шестью "
                    "численными режимами и четырьмя свежими стартами на режим/функцию. Каждый "
                    "старт выбирает checkpoint по query; средний BCE на третьем selection-пуле "
                    "выбирает индексатор/режим. Качество мягкой модели и близость к oracle не "
                    "участвуют в выборе. Лучшие конфигурации подтверждены на трёх новых search "
                    "seed с first-order и unrolled обновлениями, по восемь свежих численных стартов."
                ),
                table(
                    [
                        "Config",
                        "Architecture update",
                        "Mean selection BCE ×1000",
                        "Selection BA по seed, %",
                    ],
                    confirmation_rows,
                ),
                f"Выбран config {frozen['selected_group']['configuration']}, order {frozen['selected_group']['order']}; primary seed {frozen['primary']['origin']['search_seed']}, checkpoint {frozen['primary']['origin']['step']}. Числа: lr {frozen['numeric_setting']['lr']}, initial C norm {frozen['numeric_setting']['norm']}.",
                "![Фактические кривые поиска, recall и дискретизация](figures/search_loss_recall.png)",
                "![Пять матриц вероятностей и их argmax](figures/assignment_probabilities.png)",
                (
                    "Каждая P_i имеет размер 11×8; сумма пяти каналов в каждой позиции "
                    "равна единице. Канал 0 умножает фиксированный нулевой код. "
                    "Рисунок показывает канонические имена групп и порядок столбцов без "
                    "учителя; I на последней панели равен argmax этих же вероятностей. "
                    "Матрицы, logits, temperature и соответствующие source-кодбуки "
                    "сохранены в selected_probabilities.npz/json."
                ),
                "## Фиксируем I и дообучаем те же кодбуки",
                (
                    "Здесь используются десять исходных задач и те же четыре численных состояния "
                    "на задачу, что были у выбранного search checkpoint. Сохраняются значения "
                    "C/b/readout/offset, жёсткий I не меняется. Adam moments сброшены; ещё 4096 "
                    "support-шагов с исходным search numeric_lr. Query выбирает checkpoint каждого "
                    "старта. Таблица усредняет все задачи и старты; это обучение тех же задач, "
                    "отдельно от следующего переноса."
                ),
                table(
                    ["Этап", "Query BCE", "Query BA, %", "Query recall, %"], source_rows
                ),
                "![Фактическое дообучение чисел после фиксации I](figures/fine_tune_loss_recall.png)",
                "## Новые численные параметры на целевых функциях",
                (
                    "Архитектура и все численные настройки заморожены до создания целевых меток. "
                    "0010/0100/1011/1101 отсутствуют в текущем discovery/selection, но исторически "
                    "уже изучались: это повторная проверка на новом split/seed, не новый benchmark. "
                    "На функцию 768 support, 256 query и 1024 disjoint test; каждый метод получает "
                    "32 свежих старта и 4096 Adam-шагов, query выбирает checkpoint и старт. "
                    "Dense/untied численные настройки также выбраны только на source selection. "
                    "Средний старт и query-selected качество разделены."
                ),
                table(
                    [
                        "Структура",
                        "Связи",
                        "Численных параметров",
                        "Средний test BA, %",
                        "Query-selected test BA, %",
                        "Query-selected recall, %",
                    ],
                    numeric_rows,
                ),
                "![Фактические train/query loss и recall свежих кодбуков](figures/transfer_loss_recall.png)",
                "## Какой индексатор получен",
                f"После заморозки проверено соответствие аналитическому I: binary F1 {100 * structural['binary_f1']:.2f}%, TP/FP/FN {structural['true_positive']}/{structural['false_positive']}/{structural['false_negative']}; категориальных ошибок {structural['categorical_errors']}/88. Переименование групп и Hungarian-сопоставление скрытых столбцов используются только для диагностики и рисунка. До численного обучения применяется лишь канонизация без учителя.",
                "![Выбранный индексатор и post-freeze диагностическое выравнивание](figures/selected_indexator.png)",
                "![Индексаторы трёх подтверждающих seed и победителя пилота](figures/indexator_comparison.png)",
                (
                    "Основной выбор следует заранее заданному confirmation-критерию. "
                    "Победитель пилота добавлен как отдельный описательный контроль "
                    "после вариативных подтверждающих запусков, до открытия target: "
                    "pilot_reference_frozen.json фиксирует происхождение и время. "
                    "Он не подменяет основной результат. Его 100% на двух selection-"
                    "функциях не перенеслись на четыре целевые функции. Матрицы всех "
                    "четырёх выбранных I до/после диагностического выравнивания и ошибки "
                    "групп доступны в selected_indexators.json/npz."
                ),
                (
                    "Random groups сохраняет выбранную бинарную маску и размеры групп, перемешивая "
                    "только положительные назначения. Random positions сохраняет общие размеры "
                    "групп, перемешивая все 88 позиций. Это по одному фиксированному случайному "
                    "контролю, не распределение структур. Oracle использует истинный индексатор "
                    "как привилегированный ориентир; его кодбук обучается заново тем же режимом. "
                    "Аналитический oracle с истинными битами, bias=-3, readout=20 и offset=-10 "
                    "проверяет представимость на всех 2048 входах."
                ),
                (
                    "Ограничения: одно семейство с одним общим истинным I, лишь две функции для "
                    "выбора настроек, три подтверждающих search seed. Успех не доказывает вывод "
                    "произвольной архитектуры, превосходство над Adam или необходимость генератора "
                    "структур. Alpha здесь оптимизируется непосредственно; MLP/AR/diffusion-генераторы "
                    "не обучались. Все исходные и промежуточные состояния сохранены; программные "
                    "тесты не создавались и не запускались."
                ),
            ]
        )
        + "\n"
    )
    (output / "REPORT.md").write_text(text)
    snapshot = output / "analysis_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in BASE.iterdir():
        if path.suffix in (".py", ".md", ".json"):
            shutil.copy2(path, snapshot / path.name)
    artifacts = [
        path
        for path in output.rglob("*")
        if path.is_file()
        and path != output / "complete.json"
        and path.suffix in (".py", ".json", ".npz", ".md", ".pt", ".png", ".svg")
    ]
    save_json(
        output / "complete.json",
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "software_tests_run": False,
            "artifacts": {
                str(path.relative_to(output)): digest(path)
                for path in sorted(artifacts)
            },
        },
    )
    print(
        json.dumps(
            {
                "structural": structural,
                "target": {
                    name: row["selected_test_balanced_accuracy"]
                    for name, row in summary.items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BASE / "results/pattern_20261104"
    )
    args = parser.parse_args()
    report(args.output)


if __name__ == "__main__":
    main()
