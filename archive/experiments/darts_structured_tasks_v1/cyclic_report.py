"""Report cyclic full-matrix search without supplying a connectivity mask."""

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
    "soft_P_soft_C": "Soft P / soft C",
    "soft_P_hard_C": "Soft P / argmax C",
    "hard_P_hard_C": "Hard P / argmax C",
}


def report(output, reference):
    families = json.loads((output / "summary.json").read_text())["families"]
    with np.load(output / "aligned_states.npz") as archive:
        aligned = dict(archive)
    with np.load(output / "confirmation/hard_fits.npz") as archive:
        fitted = dict(archive)
    selected = {}
    for name, result in families.items():
        primary = result["primary"]
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
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 9.7), constrained_layout=True)
    image = categorical(axes.flat[0], truth[order], 4, "Истинный индексатор")
    for axis, (name, label) in zip(list(axes.flat)[1:4], LABELS.items()):
        row = families[name]
        primary = row["primary"]
        image = categorical(
            axis,
            aligned[f"row_{primary['row']}_indices"][order],
            4,
            f"{label}\nошибок {primary['recovery']['categorical_errors']}/256; "
            f"масок {row['exact_masks']}/3; I {row['exact_indexators']}/3",
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
    proposal = families["soft_P_hard_C"]["primary"]
    proposal_weights = fitted["weights"][
        proposal["row"], proposal["query_selected_restart"]
    ]
    proposal_weights = proposal_weights[:, aligned[f"row_{proposal['row']}_columns"]]
    for axis, matrix, label in (
        (axes.flat[4], dense, "Плотный Adam\nисходный контроль NMSE0.0323"),
        (
            axes.flat[5],
            proposal_weights,
            f"Soft P / argmax C: веса\nвыбранный I test NMSE{proposal['test_nmse']:.4g}",
        ),
    ):
        maximum = np.abs(matrix).max()
        image = axis.imshow(matrix[order], cmap="RdBu_r", vmin=-maximum, vmax=maximum)
        axis.set_title(label)
        axis.set_xlabel("Скрытый нейрон")
        axis.set_ylabel("Координата входа")
        figure.colorbar(image, ax=axis, shrink=0.65)
    figure.suptitle("DeepSets: циклическое обучение P и C; полная матрица", fontsize=15)
    save_figure(figure, output / "figures/indexators")

    history = json.loads((output / "confirmation/history.json").read_text())
    roots = json.loads((output / "confirmation/rows.json").read_text())
    variance = json.loads((output / "specification.json").read_text())[
        "target_variance"
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for name, label in LABELS.items():
        number = next(
            i for i, r in enumerate(roots) if r["family"] == name and r["seed"] == 243
        )
        steps = [r["step"] for r in history]
        for axis, key in zip(axes[:2], ("soft_query_mse", "hard_query_mse")):
            axis.plot(
                steps,
                [np.mean(r[key][number]) / variance for r in history],
                label=label,
            )
            axis.set_yscale("log")
        axes[2].plot(steps, [r["MAP_edges"][number] for r in history], label=label)
    for axis, label in zip(
        axes, ("Мягкая query NMSE", "MAP query NMSE до refit", "Связей в текущем MAP")
    ):
        axis.set_title(label)
        axis.set_xlabel("Обновлений C")
        axis.grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    figure.suptitle("Общий seed243; каждые4 шага P следуют64 шага C")
    save_figure(figure, output / "figures/curves")

    diagnostics = {}
    fit_rows = json.loads((output / "confirmation/hard_fits.json").read_text())["rows"]
    for name, result in families.items():
        diagnostics[name] = []
        for candidate in result["per_seed"]:
            row = fit_rows[candidate["row"]]
            record = next(
                item for item in history if item["step"] == row["search_step"]
            )
            root = row["root"]
            diagnostics[name].append(
                {
                    "seed": row["seed"],
                    "C_updates": record["C_updates"],
                    "P_updates": record["P_updates"],
                    "soft_query_nmse_before_refit": float(
                        np.mean(record["soft_query_mse"][root]) / variance
                    ),
                    "hard_query_nmse_before_refit": float(
                        np.mean(record["hard_query_mse"][root]) / variance
                    ),
                    "MAP_changes_last_cycle": record["MAP_changes_last_cycle"][root],
                    "maximum_logged_MAP_changes_per_cycle": max(
                        item["MAP_changes_last_cycle"][root] for item in history
                    ),
                }
            )
    save_json(output / "search_diagnostics.json", diagnostics)
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
    interrupted_seconds = 0.0
    if (output / "interruption.json").exists():
        interrupted = json.loads(
            (output / "pilot_interrupted/history.json").read_text()
        )
        interrupted_seconds = interrupted[-1]["seconds"]
        costs += interrupted_seconds
    text = [
        "# DeepSets: повторяющиеся блоки P и C на полной матрице",
        "",
        "В каждом цикле сначала4 support-шага P при остановленном C, затем64 support-шага C при остановленном P. Индексатор заново вычисляется после каждой P-фазы. Логиты сохраняются; ни P, ни C не фиксируются навсегда.200 циклов,800 обновлений P,12800 обновлений C, четыре численные реплики.",
        "",
        "| Режим | Средняя selection NMSE | Средняя test NMSE, 3 seed | Точных масок/3 | Точных I/3 | Связей выбранного I | Средняя test NMSE32 свежих C |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in LABELS.items():
        r = families[name]
        text.append(
            f"| {label} | {r['selection_nmse_mean']:.6g} | {np.mean([s['test_nmse'] for s in r['per_seed']]):.6g} | {r['exact_masks']}/3 | {r['exact_indexators']}/3 | {r['primary']['recovery']['active_edges']} | {r['fresh']['mean_test_nmse']:.6g} |"
        )
    text.extend(
        [
            "",
            "## Мягкая и дискретная модели до refit",
            "",
            "Среднее по source-selected снимкам трёх confirmation seed при их текущих C. Эти ошибки относятся к модели до независимого C-refit из основной таблицы.",
            "",
            "| Режим | Soft query NMSE | Argmax query NMSE | Максимум изменений MAP за зарегистрированный цикл, 3 seed |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for name, label in LABELS.items():
        values = diagnostics[name]
        text.append(
            f"| {label} | {np.mean([r['soft_query_nmse_before_refit'] for r in values]):.6g} "
            f"| {np.mean([r['hard_query_nmse_before_refit'] for r in values]):.6g} "
            f"| {', '.join(str(r['maximum_logged_MAP_changes_per_cycle']) for r in values)} |"
        )
    text.extend(
        [
            "",
            "Изменения argmax между циклами подтверждают, что I не фиксируется навсегда. Циклический soft-P/hard-C проверяет предложение пользователя; один размер блока не исчерпывает возможные расписания.",
            "",
            "## Протокол",
            "",
            "Soft P/soft C — парный блочный контроль. Soft P/hard C — предложенная пользователем схема: архитектурная фаза оптимизирует смесь кодов, численная фаза обучается на текущем argmax I. Hard P/hard C использует argmax также в архитектурном forward, градиент P — через softmax STE. Это смещённая оценка; градиент через адаптацию C не вычисляется.",
            "",
            "P=softmax(alpha), без sampling и без охлаждения; словарь [0,C1,…,C4,−C1,…,−C4]. Raw MAP без переноса массы; sparsity0.03 и поздний entropy0.02 общие. Все256 позиций разрешены. Предварительная маска, истинные связи, перестановка входов и число истинных связей поиску не передаются.",
            "",
            "Шесть настроек на семью: C lr0.01/0.03, P lr0.01/0.03/0.1. Pilot241/242 выбирает одну настройку, confirmation243/244/245 проверяет её на новых seed. Начальные alpha/C и phase-minibatch совпадают. Numerical Adam moments сохраняются между циклами.",
            "",
            "Снимки после0/67/133/200 циклов фиксируют MAP; затем четыре исходных C проходят3000 шагов refit с новыми моментами оптимизатора. Query выбирает численные checkpoints, средняя selection ошибка выбирает снимки и настройки. Teacher открывается после source/fresh freeze. Fresh-C калибровка и32 новые Gaussian-инициализации, heldout8192 новых входов той же ранее изученной функции. Три seed не являются тремя независимыми задачами.",
            "",
            "Чередование замороженных весов и архитектурных параметров при hard-путях также применяется в [ProxylessNAS, §3.2](https://arxiv.org/abs/1812.00332). Здесь проверяется его конкретная адаптация к индексатору и общему численному кодбуку; новизна самого принципа не заявляется.",
            "",
            "![Индексаторы](figures/indexators.png)",
            "",
            "![Кривые](figures/curves.png)",
            "",
            "Полные выбранные I/C/W сохранены в selected_indexators.json; raw P/alpha/C — в snapshots.npz. Перестановки нейронов и названия групп согласованы с teacher только для рисунков/аудита. Adam — исходный численный контроль, не новое обучение.",
            "",
            f"Измеренное обучение {costs / 60:.2f} мин., включая {interrupted_seconds:.2f}с незавершённой попытки. Первая попытка получила SIGTERM; повторный пилот использовал тот же протокол и seed. Программные тесты не создавались и не запускались.",
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
