"""Report matched reset-cycle ablations and preserve their selected indexators."""

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
    "consensus_baseline": "Базовое согласование",
    "consensus_reset": "Согласование + сброс Z",
    "consensus_adapt_keep": "Согласование + C-fit, сохранить Z",
    "consensus_adapt_reset": "Согласование + C-fit + сброс Z",
    "plain_adapt_keep": "Z/C без согласования, сохранить Z",
    "plain_adapt_reset": "Z/C без согласования + сброс Z",
}
SHORT = {
    "consensus_baseline": "Baseline",
    "consensus_reset": "Consensus + reset",
    "consensus_adapt_keep": "Consensus + C-fit, keep Z",
    "consensus_adapt_reset": "Consensus + C-fit + reset",
    "plain_adapt_keep": "Plain C-fit, keep Z",
    "plain_adapt_reset": "Plain C-fit + reset",
}


def report(output, reference):
    families = json.loads((output / "summary.json").read_text())["families"]
    labels = {name: label for name, label in LABELS.items() if name in families}
    plan = json.loads((output / "protocol.json").read_text())
    long_cycles = plan["reset"].get("long_cycles", False)
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
    with np.load(reference / "selected_states.npz") as archive:
        dense = matched_columns(
            archive["dense_shared_adam_weights"], values[np.abs(truth)] * np.sign(truth)
        )
    figure, axes = plt.subplots(2, 4, figsize=(19, 10.7), constrained_layout=True)
    image = categorical(axes.flat[0], truth[order], 4, "Истинный индексатор")
    for axis, (name, label) in zip(list(axes.flat)[1:7], labels.items()):
        row = families[name]
        primary = row["primary"]
        image = categorical(
            axis,
            aligned[f"row_{primary['row']}_indices"][order],
            4,
            f"{label}\nошибок {primary['recovery']['categorical_errors']}/256; точных I {row['exact_indexators']}/3",
        )
    for axis in list(axes.flat)[1 + len(labels) : 7]:
        axis.set_visible(False)
    figure.colorbar(
        image,
        ax=list(axes.flat)[:7],
        location="bottom",
        ticks=range(-4, 5),
        shrink=0.5,
        fraction=0.035,
        pad=0.035,
    )
    maximum = np.abs(dense).max()
    image = axes.flat[7].imshow(
        dense[order], cmap="RdBu_r", vmin=-maximum, vmax=maximum
    )
    axes.flat[7].set_title("Плотный Adam: прежние численные веса")
    axes.flat[7].set_xlabel("Скрытый нейрон")
    figure.colorbar(image, ax=axes.flat[7], shrink=0.65)
    figure.suptitle("DeepSets: дообучение кодбука и циклический сброс Z", fontsize=16)
    save_figure(figure, output / "figures/indexators")
    history = json.loads((output / "confirmation/history.json").read_text())
    roots = json.loads((output / "confirmation/rows.json").read_text())
    complete = json.loads((output / "confirmation/complete.json").read_text())
    variance = json.loads((output / "specification.json").read_text())[
        "target_variance"
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    diagnostics = {}
    for name in labels:
        family_roots = [i for i, row in enumerate(roots) if row["family"] == name]
        root = family_roots[0]
        steps = [row["step"] for row in history]
        for axis, field in zip(
            list(axes.flat)[:3], ("train_mse", "query_mse", "hard_query_mse")
        ):
            axis.plot(
                steps,
                [
                    np.mean(np.asarray(row[field])[family_roots]) / variance
                    for row in history
                ],
                label=SHORT[name],
            )
        axes.flat[3].plot(
            steps,
            [np.mean(np.asarray(row["active_edges"])[family_roots]) for row in history],
            label=SHORT[name],
        )
        diagnostics[name] = {
            "settings": roots[root],
            "per_seed": [
                {
                    "seed": row["seed"],
                    "z_updates": complete["actual_z_updates"][i],
                    "c_updates": complete["actual_c_updates"][i],
                    "head_updates": complete["actual_head_updates"][i],
                    "resets": complete["actual_resets"][i],
                    "projections": complete["actual_projections"][i],
                    "distinct_final_columns": history[-1]["distinct_columns"][i],
                    "final_column_spread": history[-1]["column_spread"][i],
                    "index_changes": history[-1]["index_changes"][i],
                }
                for i, row in enumerate(roots)
                if row["family"] == name
            ],
        }
    for axis, title in zip(
        axes.flat,
        (
            "Mean train NMSE, all 3 search seeds",
            "Mean query NMSE, all 3 search seeds",
            "Mean hard C[I] query NMSE before final refit",
            "Mean number of active indexator positions",
        ),
    ):
        axis.set_title(title)
        axis.set_xlabel("Task-gradient update")
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[:3]:
        axis.set_yscale("log")
    axes.flat[3].legend(fontsize=8)
    save_figure(figure, output / "figures/curves")
    save_json(
        output / "cycle_diagnostics.json",
        {
            "families": diagnostics,
            "maximum_reset_error": complete["maximum_reset_error"],
        },
    )
    lines = [
        "# Циклическое дообучение C и присваивание Z=C[I]",
        "",
        "Успешный предыдущий вариант отдельно зафиксирован с исходниками, протоколом, состояниями и хешами: [deepsets_consensus_exact_v1](../../../frozen_methods/deepsets_consensus_exact_v1/README.md). Его результат3/3 не перезаписывается. Здесь новый парный контроль на других seed и предложенный пользователем цикл с task-обучением C и обратным копированием весов в Z.",
        "",
        "## Исходная функция",
        "",
        "| Метод | Z/C шагов за цикл | Adam Z после копирования | Selection NMSE | Mean test NMSE,3seed | Точных масок | Точных I | Связей выбранного I | Mean test32 fresh C |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in labels.items():
        row = families[name]
        config = row["settings"]
        state = (
            ("очистить" if config["clear_z_moments"] else "сохранить")
            if config["copy_z"]
            else "без копирования"
        )
        lines.append(
            f"| {label} | {config['z_steps']}/{config['c_steps']} | {state} | {row['selection_nmse_mean']:.6g} | {np.mean([r['test_nmse'] for r in row['per_seed']]):.6g} | {row['exact_masks']}/3 | {row['exact_indexators']}/3 | {row['primary']['recovery']['active_edges']} | {row['fresh']['mean_test_nmse']:.6g} |"
        )
    lines.extend(
        [
            "",
            "![Индексаторы и плотный Adam](figures/indexators.png)",
            "",
            "![Train/query и активные связи](figures/curves.png)",
            "",
            "## Что именно сравнивается",
            "",
            (
                "2000 общих warmup и10000 дальнейших task-gradient шагов:12000 суммарно. Базовый контроль сохраняет200циклов50/0, адаптируемые методы имеют10длинных циклов800/200,500/500 или200/800Z/C. Суммарное число Z/C обновлений совпадает с короткими блоками соответствующей пропорции; меняются длительность блока и частота проекции/сброса. Rho меняется каждые50общих шагов, dual — на границе цикла. "
                if long_cycles
                else "Во всех режимах2000 общих плотных warmup-обновлений, затем200 циклов по50 task-gradient шагов:12000 суммарно. В базовом варианте все50 шагов получает Z. При C-fit цикл делится на40/10,25/25 или10/40 шагов Z/C. "
            )
            + "Голова обновляется каждый раз. Это равенство количества градиентных проходов, не числа обучаемых координат или FLOPs; реальные счётчики сохранены в cycle_diagnostics.json.",
            "",
            "Цикл: обучить Z → точно спроецировать в I/C → обучить C и голову при фиксированном I → при необходимости присвоить Z=C[I]. Все256 позиций могут меняться в следующем цикле. Проекция не получает маску, истинные группы, численные коды или число32 связей.",
            "",
            "Согласование сохраняет прежние rho0.01→100,lambda0.3,dual0.25. В plain-варианте Z обучается только по task-loss и L2; нет ни dual, ни штрафа согласования, а проекция использует постоянную цену tau0.0006/0.006/0.06. Во всех вариантах точная одномерная проекция совместно определяет центры и нулевые позиции.",
            "",
            "При перезаписи Z отдельно проверяются сохранение и очистка его моментов Adam. Моменты C очищаются при каждой новой проекции, моменты головы сохраняются. В гибридном режиме dual обновляется по расхождению с адаптированным C[I] до перезаписи Z. Это модифицированный цикл, без утверждения стандартной ADMM-гарантии.",
            "",
            f"Базовые настройки не ретюнятся.{len(json.loads((output / 'registration.json').read_text())['candidates'])} настроек зарегистрированы до пилота341/342; одна настройка на семью фиксируется для confirmation343/344/345. Все старты Z и support-minibatch парные. Четыре снимка получают одинаковый hard-C-refit3000 шагов:1исходный+3свежих GaussianC. Query выбирает численныйcheckpoint, meanselection выбирает структуру и гиперпараметры. Query не участвует в task-градиентах или выборе состояния C внутри цикла.",
            "",
            "Кривые в точках завершения цикла показывают состояние после копирования, если оно включено. Ошибка Z до копирования и величина разрыва сохранены отдельно в history.json. Равенство Z=C[I] в эти моменты не означает правильной структуры. Максимальная фактическая ошибка присваивания сохранена в cycle_diagnostics.json.",
            "",
            "Новый source-heldout8192 и teacher-аудит открываются после source/freshfreeze. Fresh32 относится к той же функции, а не к переносу. Adam на хитмапе — прежний численный контроль. Всё исследование пока на синтетическом DeepSets одной скрытой структуры.",
            "",
        ]
    )
    transfer = output / "transfer_confirmation/summary.json"
    if transfer.exists():
        values = json.loads(transfer.read_text())["aggregate"]
        lines.extend(
            [
                "## Перенос на новые коэффициенты",
                "",
                "Шесть ранее не оценённых функций зарегистрированы до поиска (seed-offset15100000). Исходные I фиксированы; на каждой новой функции обучаются только заново инициализированные численные параметры. Query выбирает старт из8, selection — численную настройку из6. После этого создаётся отдельный test8192.",
                "",
                "| Метод | Mean test NMSE, query-selected C | Mean test NMSE, all8 starts |",
                "| --- | ---: | ---: |",
            ]
        )
        for name, label in labels.items():
            row = values[name + "/learned"]
            lines.append(
                f"| {label} | {row['mean_query_selected_test_nmse']:.6g} | {row['mean_all_restarts_test_nmse']:.6g} |"
            )
        for name in ("oracle", "dense_adam"):
            row = values[name]
            lines.append(
                f"| {name} | {row['mean_query_selected_test_nmse']:.6g} | {row['mean_all_restarts_test_nmse']:.6g} |"
            )
        lines.extend(
            [
                "",
                "![Перенос](transfer_confirmation/figures/transfer_errors.png)",
                "",
                "![Train/query при переносе](transfer_confirmation/figures/numerical_curves.png)",
                "",
                "Shuffle-sharing и shuffle-all контроли каждой семьи, все старты и отдельные функции сохранены в transfer_confirmation/REPORT.md и summary.json. Среднее всех8 стартов отдельно от выбранного по query результата позволяет видеть численные срывы.",
                "",
            ]
        )
    seconds = sum(
        json.loads((output / phase / "complete.json").read_text())["seconds"]
        for phase in ("pilot", "confirmation")
    )
    for name in (
        "pilot/hard_fits.npz",
        "confirmation/hard_fits.npz",
        "fresh_calibration.npz",
        "fresh_evaluation.npz",
    ):
        with np.load(output / name) as archive:
            seconds += float(archive["seconds"])
    lines.extend(
        [
            f"Измеренное source-обучение всех кандидатов и численных стартов: {seconds / 60:.2f} мин. Стоимость переноса указана отдельно в его отчёте. Программные тесты не создавались и не запускались; Ruff изменённых Python-файлов прошёл.",
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
