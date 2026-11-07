"""Post-freeze structural diagnostics for the three-class Pattern 0110 search."""

import argparse
import csv
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
from .model import metrics, soft_predict
from .report import fitting_curves, save_figure, search_curves, table
from .single import combined

COLORS = ["#edf0f2", "#4879b8", "#e49d48"]
LABELS = {
    "primary": "DARTS: selected I",
    "pilot_reference": "Pilot: selected I",
    "oracle": "True I (diagnostic)",
    "random_same_counts": "Random I, same label counts",
    "initial_indexator": "Initial random I",
}


def label(name):
    return LABELS.get(name, name.replace("_", " "))


def align(indices, truth):
    best = None
    for mapping in (np.array([0, 1, 2]), np.array([0, 2, 1])):
        renamed = mapping[indices]
        cost = (renamed[:, :, None] != truth[:, None, :]).sum(0)
        source, target = linear_sum_assignment(cost)
        candidate = np.empty_like(indices)
        candidate[:, target] = renamed[:, source]
        errors = int((candidate != truth).sum())
        if best is None or errors < best[0]:
            best = errors, candidate, mapping, source, target
    source, target = linear_sum_assignment(
        -((indices > 0).astype(int).T @ (truth > 0).astype(int))
    )
    binary = np.empty_like(indices, dtype=bool)
    binary[:, target] = indices[:, source] > 0
    expected = truth > 0
    tp = int((binary & expected).sum())
    fp = int((binary & ~expected).sum())
    fn = int((~binary & expected).sum())
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return best[1], {
        "categorical_errors": best[0],
        "categorical_exact": best[0] == 0,
        "active_edges": int((indices > 0).sum()),
        "group_counts": np.bincount(indices.ravel(), minlength=3).tolist(),
        "ignored_input_coordinates": np.flatnonzero(~(indices > 0).any(1)).tolist(),
        "unused_hidden_columns": np.flatnonzero(~(indices > 0).any(0)).tolist(),
        "exact_aligned_hidden_columns": int((best[1] == truth).all(0).sum()),
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "binary_exact": bool(np.array_equal(binary, expected)),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "group_mapping": best[2].tolist(),
        "source_columns": best[3].tolist(),
        "target_columns": best[4].tolist(),
    }


def draw_index(axis, values, title):
    plotted = axis.imshow(
        values,
        cmap=ListedColormap(COLORS),
        norm=BoundaryNorm(np.arange(-0.5, 3.5), 3),
    )
    axis.set_title(title, fontsize=11)
    axis.set_xticks(range(8))
    axis.set_yticks(range(11))
    axis.set_xlabel("Hidden unit")
    axis.set_ylabel("Input coordinate")
    for row, column in np.ndindex(values.shape):
        axis.text(column, row, str(values[row, column]), ha="center", va="center")
    return plotted


def structure_figures(output, summary, truth):
    names = [
        name
        for name in summary["methods"]
        if name in ("primary", "pilot_reference") or name.startswith("search_seed_")
    ]
    records = {}
    for name in names:
        raw = np.array(summary["methods"][name]["indices"])
        aligned, structural = align(raw, truth)
        records[name] = {"raw": raw.tolist(), "aligned": aligned.tolist(), **structural}
    selected = records["primary"]
    panels = [
        (truth, "True I: 0110 windows (diagnostic only)"),
        (np.array(selected["raw"]), "Selected I: raw canonical labels"),
        (
            np.array(selected["aligned"]),
            f"Selected I: aligned; {selected['categorical_errors']}/88 errors",
        ),
    ]
    for name in names[1:]:
        row = records[name]
        panels.append(
            (
                np.array(row["aligned"]),
                f"{label(name)}: aligned; {row['categorical_errors']}/88 errors",
            )
        )
    figure, axes = plt.subplots(2, 3, figsize=(14, 11), layout="constrained")
    for axis, (values, title) in zip(axes.ravel(), panels):
        draw_index(axis, values, title)
    figure.suptitle(
        "Fixed Pattern 0110: selected indexators\n0 = off; 1 and 2 = shared learned values; alignment only after selection"
    )
    save_figure(figure, output, "indexator_comparison")
    figure, axes = plt.subplots(1, 3, figsize=(13, 6), layout="constrained")
    draw_index(axes[0], truth, "True indexator (diagnostic only)")
    aligned = np.array(selected["aligned"])
    draw_index(axes[1], aligned, "Selected indexator: aligned after freezing")
    errors = np.zeros_like(truth)
    errors[(aligned > 0) & (truth > 0)] = 1
    errors[(aligned > 0) & (truth == 0)] = 2
    errors[(aligned == 0) & (truth > 0)] = 3
    errors[(aligned > 0) & (truth > 0) & (aligned != truth)] = 4
    axes[2].imshow(
        errors,
        cmap=ListedColormap([COLORS[0], "#65a58d", "#d76a73", "#efbd65", "#ac72b3"]),
        vmin=0,
        vmax=4,
    )
    axes[2].set_title(
        f"Errors: {selected['categorical_errors']}/88; binary F1 {100 * selected['binary_f1']:.2f}%"
    )
    axes[2].set_xticks(range(8))
    axes[2].set_yticks(range(11))
    axes[2].set_xlabel("Hidden unit")
    axes[2].set_ylabel("Input coordinate")
    figure.suptitle(
        "Green: correct active; red: extra; yellow: missing; purple: wrong group"
    )
    save_figure(figure, output, "selected_indexator")
    save_json(output / "selected_indexators.json", {"true": truth.tolist(), **records})
    np.savez_compressed(
        output / "selected_indexators.npz",
        names=np.array(names),
        raw=np.array([records[name]["raw"] for name in names]),
        aligned=np.array([records[name]["aligned"] for name in names]),
        true=truth,
    )
    return records


def probabilities(output, frozen):
    origin = frozen["primary"]["origin"]
    folder = output / origin["folder"]
    snapshots = load_archive(folder / "snapshots.npz")
    history = json.loads((folder / "history.json").read_text())
    record = next(row for row in history if row["step"] == origin["step"])
    temperature = record["temperature"][origin["root"]]
    alpha = snapshots["alpha"][origin["snapshot"], origin["root"]]
    raw = torch.softmax(torch.tensor(alpha) / temperature, -1).numpy()
    inverse = np.argsort(origin["mapping"])
    values = raw[:, origin["columns"]][:, :, inverse]
    indices = load_archive(output / "frozen.npz")["indices"]
    if not np.array_equal(values.argmax(-1), indices):
        raise ValueError("Probability heatmaps disagree with the frozen indexator")
    np.savez_compressed(
        output / "selected_probabilities.npz",
        probabilities=values,
        indices=indices,
        temperature=temperature,
        raw_probabilities=raw,
    )
    save_json(
        output / "selected_probabilities.json",
        {
            "probabilities": values.tolist(),
            "indices": indices.tolist(),
            "temperature": temperature,
            "origin": origin,
            "sum_over_codes_max_error": float(np.abs(values.sum(-1) - 1).max()),
            "canonicalization_uses_teacher": False,
        },
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 11), layout="constrained")
    for code, axis in enumerate(axes.ravel()[:3]):
        plotted = axis.imshow(values[..., code], cmap="viridis", vmin=0, vmax=1)
        axis.set_title(
            f"P{code}: " + ("fixed zero" if code == 0 else f"coefficient of C{code}")
        )
        axis.set_xticks(range(8))
        axis.set_yticks(range(11))
        axis.set_xlabel("Hidden unit")
        axis.set_ylabel("Input coordinate")
        figure.colorbar(plotted, ax=axis, shrink=0.7, label="Probability")
    hard = draw_index(axes[1, 1], indices, "I = argmax(P0, P1, P2)")
    figure.colorbar(hard, ax=axes[1, 1], ticks=range(3), shrink=0.7)
    figure.suptitle(
        f"Selected three-class relaxation: temperature {temperature:.2f}\nCanonical labels and column order; no teacher alignment"
    )
    save_figure(figure, output, "assignment_probabilities")
    numeric = torch.tensor(
        snapshots["numeric"][origin["snapshot"], origin["root"]][None]
    )
    soft = torch.tensor(raw[None])
    hard = torch.nn.functional.one_hot(soft.argmax(-1), 3).float()
    observed = load_archive(output / "source.npz")
    heldout = load_archive(output / "heldout.npz")
    _, population_x = input_table()
    payloads = {
        "query": (
            torch.tensor(observed["query_x"]),
            torch.tensor(observed["query_y"][0]),
        ),
        "test": (torch.tensor(heldout["test_x"]), torch.tensor(heldout["test_y"][0])),
        "population": (population_x, labels_for(population_x, "0110")),
    }
    measured = {
        "scope": "Frozen search checkpoint, four inherited numerical replicas averaged; no further optimization"
    }
    with torch.no_grad():
        for name, probability in (("soft", soft), ("argmax", hard)):
            measured[name] = {
                split: metrics(
                    soft_predict(x, probability, numeric), y[None, None, None]
                )[0, 0]
                .mean(0)
                .tolist()
                for split, (x, y) in payloads.items()
            }
    save_json(output / "frozen_relaxation_metrics.json", measured)
    return measured


def snapshot_audit(output, truth):
    records = []
    for phase in ("pilot", "confirmation"):
        archive = load_archive(output / phase / "hard_candidates.npz")
        origins = json.loads((output / phase / "hard_candidates.json").read_text())
        for origin, indices in zip(origins, archive["indices"]):
            _, structural = align(indices, truth)
            records.append({"phase": phase, "origin": origin, **structural})
    save_json(
        output / "snapshot_structure_audit.json",
        {
            "diagnostic_only": True,
            "scope": "Registered saved snapshots only; not every optimization step",
            "count": len(records),
            "categorical_exact_count": sum(row["categorical_exact"] for row in records),
            "binary_exact_count": sum(row["binary_exact"] for row in records),
            "records": records,
        },
    )
    return records


def identifiability(output, indices):
    _, inputs = input_table()
    values = inputs.numpy()
    targets = labels_for(inputs, "0110").numpy().astype(bool)
    retained = (indices > 0).any(1)
    _, groups = np.unique(values[:, retained], axis=0, return_inverse=True)
    positive = np.bincount(groups, weights=targets)
    negative = np.bincount(groups, weights=~targets)
    ceiling = np.maximum(
        positive / (2 * targets.sum()), negative / (2 * (~targets).sum())
    ).sum()
    conflicting = np.flatnonzero((positive > 0) & (negative > 0))
    example = None
    if len(conflicting):
        members = np.flatnonzero(groups == conflicting[0])
        pos = members[targets[members]][0]
        neg = members[~targets[members]][0]
        example = {
            "positive": "".join(str(int(value > 0)) for value in values[pos]),
            "negative": "".join(str(int(value > 0)) for value in values[neg]),
            "differing_coordinates": np.flatnonzero(
                values[pos] != values[neg]
            ).tolist(),
        }
    result = {
        "diagnostic_only_all_2048_inputs": True,
        "ignored_input_coordinates": np.flatnonzero(~retained).tolist(),
        "conflicting_projected_inputs": len(conflicting),
        "balanced_accuracy_upper_bound_from_retained_inputs": float(ceiling),
        "counterexample": example,
        "bound_scope": "Any classifier using retained inputs, independent of codebook or MLP optimization",
    }
    save_json(output / "input_identifiability.json", result)
    return result


def report(output):
    plan = json.loads((output / "protocol.json").read_text())
    frozen = json.loads((output / "frozen.json").read_text())
    summary = json.loads((output / "evaluation_summary.json").read_text())
    truth = np.zeros((11, 8), dtype=int)
    for hidden in range(8):
        truth[hidden : hidden + 4, hidden] = [1, 2, 2, 1]
    (output / "figures").mkdir(exist_ok=True)
    structures = structure_figures(output, summary, truth)
    invariant = identifiability(output, np.array(structures["primary"]["raw"]))
    relaxation = probabilities(output, frozen)
    diagnostic = snapshot_audit(output, truth)
    origin = frozen["primary"]["origin"]
    search_curves(
        output, {"primary": {"origin": {**origin, "shard": origin["folder"]}}}
    )
    fits = combined(output / "confirmation")
    chosen = np.flatnonzero(fits["entry_ids"] == frozen["primary"]["entry"])
    fitting_curves(
        output,
        "fine_tune_loss_recall",
        "Fixed selected I: continue all four inherited codebooks",
        fits,
        [("Inherited C, fixed I", chosen)],
    )
    fresh = load_archive(output / "evaluation_fits.npz")
    restarts = plan["fit"]["evaluation_restarts"]
    groups = [
        (label(name), np.arange(number * restarts, (number + 1) * restarts))
        for number, name in enumerate(fresh["methods"])
        if name in ("primary", "pilot_reference", "oracle", "random_same_counts")
    ]
    fitting_curves(
        output,
        "fresh_codebook_loss_recall",
        "Fixed I: 32 fresh numerical starts each; all starts averaged",
        fresh,
        groups,
    )
    rows = []
    for name, row in summary["methods"].items():
        test = row["selected_test_metrics"]
        structural = align(np.array(row["indices"]), truth)[1]
        rows.append(
            [
                label(name),
                row["active_edges"],
                structural["categorical_errors"],
                f"{100 * structural['binary_f1']:.2f}",
                f"{100 * row['mean_test_metrics'][1]:.2f}",
                f"{100 * test[1]:.2f}",
                f"{100 * test[2]:.2f}",
                f"{100 * row['selected_population_metrics'][1]:.2f}",
            ]
        )
    headers = [
        "Метод",
        "Связей",
        "Ошибок I /88",
        "F1 маски, %",
        "BA: среднее 32 стартов, %",
        "BA: query-выбор, %",
        "Recall: query-выбор, %",
        "BA: все 2048, %",
    ]
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    warm_query = fits["query_metrics"][chosen].mean(0)
    bridge = [
        [
            "Soft, перед argmax",
            *[
                f"{x:.2f}"
                for x in (
                    origin["soft_query"][0],
                    100 * origin["soft_query"][1],
                    100 * origin["soft_query"][2],
                )
            ],
        ],
        [
            "Argmax, исходные C",
            *[
                f"{x:.2f}"
                for x in (
                    origin["hard_query"][0],
                    100 * origin["hard_query"][1],
                    100 * origin["hard_query"][2],
                )
            ],
        ],
        [
            "Argmax, дообученные C",
            *[
                f"{x:.2f}"
                for x in (warm_query[0], 100 * warm_query[1], 100 * warm_query[2])
            ],
        ],
    ]
    selected = structures["primary"]
    exact_count = sum(row["categorical_exact"] for row in diagnostic)
    status = (
        "восстановлен точно"
        if selected["categorical_exact"]
        else "не восстановлен точно"
    )
    inherited = summary["inherited_primary_test"]
    params = load_archive(output / "frozen.npz")["parameters"]
    paragraphs = [
        "# DARTS-like: один фиксированный Pattern 0110",
        f"Выбранный индексатор **{status}**: {selected['categorical_errors']}/88 категориальных ошибок после глобальной перестановки кодов и сопоставления скрытых нейронов. F1 бинарной маски — {100 * selected['binary_f1']:.2f}%. Из {len(diagnostic)} сохранённых снимков {exact_count} точно совпали с истинным индексатором; это диагностика после выбора, а не критерий выбора.",
        f"Точно совпадают {selected['exact_aligned_hidden_columns']}/8 выровненных скрытых столбцов. Полностью отключённые входы (нумерация с 0): {invariant['ignored_input_coordinates']}. Верхняя граница BA для любой функции оставшихся входов на всех 2048 наблюдениях — {100 * invariant['balanced_accuracy_upper_bound_from_retained_inputs']:.2f}%. Противоречащая пара: {json.dumps(invariant['counterexample'], ensure_ascii=False)}. У такой пары одинаковы все подключённые координаты и любой выход сети при любых числах кодбука, но метки разные. Это структурная потеря информации, которую дообучение C исправить не может.",
        "## Постановка и выбор",
        "Только распознавание подстроки 0110 в 11 бинарных координатах. 8 ReLU-нейронов. На каждую из 88 позиций — softmax по 3 индексам: C0=0 фиксирован, C1/C2 обучаются. Три дополнительных числа — общий скрытый bias, общий readout и выходной offset. Итого 5 численных параметров. Модель не заменялась маской с независимыми весами.",
        "768 support → численные обновления; 256 query → архитектурные обновления и выбор численных checkpoint; 512 selection → сравнение настроек и индексаторов; 512 test → только после заморозки. Все 2048 входов показаны дополнительно и включают обучающие наблюдения. Задача исторически изучалась в проекте; это новый разрез одной известной задачи.",
        "Пилот: 24 настройки × 2 seed; подтверждение: две выбранные настройки × 3 новых seed × first-order/один виртуальный Adam-шаг. По 6000 внешних шагов, по 3 численных обновления. C1/C2 случайно инициализированы разными знаками для согласования имён между 4 репликами; во время обучения их значения и знаки не ограничены. Разметка окон, позиции связей и число активных связей не сообщались поиску.",
        "В каждой точке 0/1500/3000/4500/6000 argmax фиксировал I, после чего исходные C дообучались 4096 шагов. Checkpoint по query BCE, архитектуры — по средней selection BCE четырёх реплик; настройки подтверждения — по среднему трёх seed. Основной seed выбран по selection. Истинный I и диагностические совпадения не использовались в выборе.",
        f"Выбранная настройка: `{json.dumps({key: origin[key] for key in ('id', 'numeric_lr', 'architecture_lr', 'schedule', 'sparsity_penalty', 'order', 'search_seed')}, ensure_ascii=False)}`, снимок {origin['step']}.",
        "## Результаты",
        f"С сохранённым и дообученным C основной индексатор: test BA {100 * inherited[1]:.2f}%, recall {100 * inherited[2]:.2f}%. Пять чисел [C1,C2,b,a,offset]: {', '.join(f'{value:.2f}' for value in params)}.",
        "Ниже все структуры переобучаются с 32 свежими C. Половина стартов меняет местами начальные C1/C2; аналитические значения не используются. Общие LR/масштаб выбраны по selection только основного I. Среднее всех стартов и результат старта, выбранного по query BCE, разделены. BA — balanced accuracy; test содержит 512 входов.",
        table(headers, rows),
        f"Аналитический контроль истинного I: C=(-1,+1), b=-3, a=20, offset=-10, BA на всех 2048 входах {100 * summary['analytic_oracle_population'][1]:.2f}%. Эти числа использованы только в контроле после заморозки.",
        "## Что меняется при дискретизации",
        f"У замороженного поискового checkpoint средняя soft BA четырёх реплик: query {100 * relaxation['soft']['query'][1]:.2f}%, test {100 * relaxation['soft']['test'][1]:.2f}%. Сразу после argmax с теми же C средняя test BA {100 * relaxation['argmax']['test'][1]:.2f}%. Это различает переобучение архитектурной query и потери дискретизации; отдельные soft/hard значения сохранены в frozen_relaxation_metrics.json.",
        "Средние query-метрики четырёх исходных численных реплик выбранного поиска. Query участвовала в поиске; это не test.",
        table(["Состояние", "BCE", "BA, %", "Recall, %"], bridge),
        "## Точные артефакты",
        "`selected_indexators.json/.npz`: raw/выровненные I всех выбранных seed, истинный I и преобразования. `selected_probabilities.json/.npz`: P0/P1/P2 выбранного checkpoint. `snapshot_structure_audit.json`: диагностика каждого сохранённого снимка. `frozen.json/.npz`: выбор без test. `evaluation_fits.npz`: все численные старты и реально измеренные кривые.",
    ]
    for name in (
        "indexator_comparison",
        "assignment_probabilities",
        "selected_indexator",
        "search_loss_recall",
        "fine_tune_loss_recall",
        "fresh_codebook_loss_recall",
    ):
        paragraphs.append(f"![{name}](figures/{name}.png)")
    paragraphs += [
        "Классификация и восстановление I — разные критерии. Неполное совпадение не доказывает невозможность другой реализации той же функции. Эксперимент проверяет данную релаксацию, бюджет и процедуру выбора на одном паттерне.",
        "Программные тесты не запускались; выполнены сами экспериментальные обучения и аналитический контроль. Никаких изменений алгоритма после открытия test.",
    ]
    (output / "REPORT.md").write_text("\n\n".join(paragraphs) + "\n")
    snapshot = output / "analysis_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in BASE.iterdir():
        if path.suffix in (".py", ".json", ".md"):
            shutil.copy2(path, snapshot / path.name)
    shutil.copy2(Path("pattern/datasets.py"), snapshot / "pattern_datasets.py")
    save_json(
        output / "complete.json",
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "selected_structure": selected,
            "saved_snapshots": len(diagnostic),
            "exact_saved_snapshots": exact_count,
            "software_tests_run": False,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "artifacts": {
                str(path.relative_to(output)): digest(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "complete.json"
            },
        },
    )
    print(
        json.dumps(
            {"selected_structure": selected, "exact_saved_snapshots": exact_count},
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BASE / "results/single_0110_20261104"
    )
    args = parser.parse_args()
    report(args.output)


if __name__ == "__main__":
    main()
