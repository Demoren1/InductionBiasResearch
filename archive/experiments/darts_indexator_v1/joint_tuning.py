"""Validation-selected learning-rate tuning of joint P/C training on 0110."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import single
from .data import BASE, digest, load_archive, save_json
from .report import fitting_curves, save_figure, table
from .search import train
from .single_report import align, draw_index, probabilities, structure_figures


def prepare(parent):
    output = parent / "joint_tuning"
    output.mkdir(exist_ok=False)
    plan = json.loads((parent / "protocol.json").read_text())
    plan["output"] = str(output)
    plan["search"].update(
        {
            "fixed_configuration_ids": [2, 3, 6, 7, 10, 11],
            "pilot_seeds": [5, 6, 7],
            "confirmation_seeds": [8, 9, 10],
            "confirmation_orders": ["first"],
            "confirmation_top_configs": 2,
            "update_mode": "joint_support",
            "numeric_steps_per_outer": 1,
            "paired_batch_streams": True,
        }
    )
    save_json(output / "protocol.json", plan)
    save_json(
        output / "tuning_registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "posthoc_after_initial_ablation_test": True,
            "reason": "Joint support had the lowest mean source hard-validation BCE (.2447); late soft validation loss increased. Compare lower alpha rates and confirm on new search seeds.",
            "numeric_lr": 0.03,
            "alpha_rates": [0.003, 0.01, 0.03],
            "sparsity_penalties": [0.0, 0.01],
            "schedule": "annealed",
            "selection": "Same query numeric checkpoint and mean hard+fine-tuned-C selection BCE; no test or true-indexator ranking.",
            "fresh_confirmation_search_seeds": [8, 9, 10],
            "reused_high_rate_search": str(
                parent / "update_ablation/joint_support/pilot/search/shard_0"
            ),
        },
    )
    for name in ("source.npz", "heldout.npz"):
        shutil.copy2(parent / name, output / name)
    search_folder = output / "pilot/search"
    search_folder.mkdir(parents=True)
    shutil.copytree(
        parent / "update_ablation/joint_support/pilot/search/shard_0",
        search_folder / "shard_high_rate",
    )
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for path in BASE.iterdir():
        if path.suffix in (".py", ".json", ".md"):
            shutil.copy2(path, snapshot / path.name)
    print("PREPARED", output, flush=True)


def run(parent, phase, device):
    output = parent / "joint_tuning"
    plan = json.loads((output / "protocol.json").read_text())
    search_plan = json.loads(json.dumps(plan))
    if phase == "pilot":
        search_plan["search"]["fixed_configuration_ids"] = [2, 3, 6, 7]
    train(search_plan, output, phase, device, 0, 1)
    single.collect(plan, output, phase)
    single.fine_tune(plan, output, phase, device, 0, 1)
    single.select(plan, output, phase)
    if phase == "pilot":
        individual_reference(parent)


def individual_reference(parent):
    output = parent / "joint_tuning"
    selected = json.loads((output / "pilot/ranking.json").read_text())[0]
    candidates = load_archive(output / "pilot/hard_candidates.npz")
    fits = single.combined(output / "pilot")
    ids = np.flatnonzero(fits["entry_ids"] == selected["entry"])
    chosen = ids[np.argmin(fits["query_metrics"][ids, 0])]
    np.savez_compressed(
        output / "individual_reference.npz",
        indices=candidates["indices"][selected["entry"]],
        parameters=fits["parameters"][chosen],
    )
    save_json(
        output / "individual_reference.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "selection": "Minimum mean selection BCE over four inherited C among all pilot checkpoints/configurations/search seeds; independent of test or true structure",
            "purpose": "Separate ability to discover at least one good I from robustness of configuration selection averaged over search seeds",
            "selected": selected,
            "state_sha256": digest(output / "individual_reference.npz"),
            "tuning_test_evaluated": False,
        },
    )
    print("INDIVIDUAL SOURCE REFERENCE", selected, flush=True)


def report(parent):
    output = parent / "joint_tuning"
    (output / "figures").mkdir(exist_ok=True)
    frozen = json.loads((output / "frozen.json").read_text())
    summary = json.loads((output / "evaluation_summary.json").read_text())
    truth = np.zeros((11, 8), dtype=int)
    for column in range(8):
        truth[column : column + 4, column] = [1, 2, 2, 1]
    records = structure_figures(output, summary, truth)
    individual = np.array(summary["methods"]["best_individual_source"]["indices"])
    individual_aligned, individual_metrics = align(individual, truth)
    records["best_individual_source"] = {
        "raw": individual.tolist(),
        "aligned": individual_aligned.tolist(),
        **individual_metrics,
    }
    save_json(output / "individual_indexator.json", records["best_individual_source"])
    figure, axes = plt.subplots(1, 3, figsize=(13, 6), layout="constrained")
    draw_index(axes[0], truth, "True I (diagnostic only)")
    draw_index(
        axes[1],
        np.array(records["primary"]["aligned"]),
        f"New-seed configuration winner: {records['primary']['categorical_errors']}/88 errors",
    )
    draw_index(
        axes[2],
        individual_aligned,
        f"Best pilot I by independent validation: {individual_metrics['categorical_errors']}/88 errors",
    )
    figure.suptitle(
        "Joint P/C training: configuration robustness and best individual structure\nBoth selections use source validation only; alignment after freezing"
    )
    save_figure(figure, output, "individual_indexator")
    relaxation = probabilities(output, frozen)
    selected = records["primary"]
    baseline = load_archive(parent / "adam_baseline/weight_heatmaps.npz")
    state = load_archive(output / "frozen.npz")
    raw = np.r_[0, state["parameters"][:2]][state["indices"]]
    aligned = np.empty_like(raw)
    aligned[:, selected["target_columns"]] = raw[:, selected["source_columns"]]
    reference_state = load_archive(output / "individual_reference.npz")
    reference_raw = np.r_[0, reference_state["parameters"][:2]][
        reference_state["indices"]
    ]
    reference_weights = np.empty_like(reference_raw)
    reference_weights[:, individual_metrics["target_columns"]] = reference_raw[
        :, individual_metrics["source_columns"]
    ]
    weights = [
        baseline["weights"][0],
        baseline["weights"][1],
        aligned,
        reference_weights,
        baseline["weights"][2],
        baseline["weights"][3],
    ]
    titles = [
        "True signed filters",
        "Original alternating DARTS",
        "Joint training: tuned, fixed I",
        "Joint: best pilot I by validation",
        "Dense Adam: shared head",
        "Dense Adam: full MLP",
    ]
    figure, axes = plt.subplots(2, 3, figsize=(14, 11), layout="constrained")
    bound = max(float(np.abs(weight).max()) for weight in weights)
    for axis, weight, title in zip(axes.ravel(), weights, titles):
        plotted = axis.imshow(weight, cmap="RdBu_r", vmin=-bound, vmax=bound)
        axis.set_title(title, fontsize=10)
        axis.set_xticks(range(8))
        axis.set_yticks(range(11))
        axis.set_xlabel("Hidden unit (matched for display)")
        axis.set_ylabel("Input coordinate")
    figure.colorbar(
        plotted,
        ax=list(axes.ravel()),
        shrink=0.8,
        label="Actual weight; common color scale",
    )
    figure.suptitle(
        "Pattern 0110: validation-tuned joint P/C training versus dense Adam\nNo normalization, sign flips or quantization; only column matching after selection"
    )
    save_figure(figure, output, "weight_comparison")
    np.savez_compressed(
        output / "weight_heatmaps.npz",
        weights=np.stack(weights),
        titles=np.array(titles),
    )
    fits = single.combined(output / "confirmation")
    ids = np.flatnonzero(fits["entry_ids"] == frozen["primary"]["entry"])
    fitting_curves(
        output,
        "fine_tune_loss_recall",
        "Tuned joint I: continue all four inherited C",
        fits,
        [("Inherited C, fixed I", ids)],
    )
    fresh = load_archive(output / "evaluation_fits.npz")
    fitting_curves(
        output,
        "fresh_codebook_loss_recall",
        "Tuned joint I versus oracle: 32 fresh starts each",
        fresh,
        [
            (str(name), np.arange(i * 32, (i + 1) * 32))
            for i, name in enumerate(fresh["methods"])
            if name in ("primary", "oracle", "random_same_counts")
        ],
    )
    source = frozen["primary"]["origin"]
    confirmation = json.loads((output / "confirmation_selection.json").read_text())
    warm = summary["inherited_primary_test"]
    rows = []
    for name, record in records.items():
        result = summary["methods"][name]
        rows.append(
            [
                name,
                record["categorical_errors"],
                f"{100 * record['binary_f1']:.2f}",
                f"{100 * result['selected_test_metrics'][1]:.2f}",
                f"{100 * result['selected_population_metrics'][1]:.2f}",
            ]
        )
    status = (
        "восстановлен точно"
        if selected["categorical_exact"]
        else "не восстановлен точно"
    )
    content = [
        "## Осторожный подбор гиперпараметров совместного обучения",
        "Дополнительный source-only подбор после первой абляции: numeric LR 0.03, alpha LR 0.003/0.01/0.03, annealing, sparsity0/0.01. Три парных pilot seed 5/6/7; уже обученные поиски с alpha0.03 переиспользованы. Две конфигурации выбраны по средней selection BCE после argmax и дообучения C, затем подтверждены на новых seed 8/9/10. Совместно P/C учатся только на support; query не получает архитектурных градиентов. 6000 шагов — максимум, итог выбирается из пяти сохранённых точек. Test и истинный I не выбирают настройки.",
        f"Индексатор **{status}**: {selected['categorical_errors']}/88 ошибок; F1 маски {100 * selected['binary_f1']:.2f}%. Выбраны numeric LR{source['numeric_lr']}, alpha LR{source['architecture_lr']}, sparsity{source['sparsity_penalty']}, search seed{source['search_seed']}, шаг{source['step']}. Mean selection BCE трёх новых seed: {confirmation['selected_group']['mean_selection_bce']:.2f}.",
        f"С исходным дообученным C: test BA {100 * warm[1]:.2f}%, recall {100 * warm[2]:.2f}%. Soft-модель выбранного checkpoint: query BA {100 * relaxation['soft']['query'][1]:.2f}%, test BA {100 * relaxation['soft']['test'][1]:.2f}%. Свежие C ниже выбираются по query из32стартов; их LR/масштаб отдельно откалиброваны по source selection.",
        table(
            ["I", "Ошибок /88", "F1, %", "Fresh C: test BA, %", "BA все2048, %"], rows
        ),
        f"Отдельный вопрос — нашёл ли поиск хотя бы один полезный I: до итогового test сохранён кандидат с минимальной средней selection BCE четырёх C среди всего пилота, без усреднения по search seed. У него {individual_metrics['categorical_errors']}/88 ошибок, test BA со свежим C {100 * summary['methods']['best_individual_source']['selected_test_metrics'][1]:.2f}%. Он не подменяет основной confirmation-based выбор. Протокол и исходное состояние — individual_reference.json/.npz.",
        "![Лучший индивидуальный source-кандидат](figures/individual_indexator.png)",
        "![Выбранный индексатор](figures/selected_indexator.png)",
        "![P0/P1/P2](figures/assignment_probabilities.png)",
        "![Сравнение весов с Adam](figures/weight_comparison.png)",
        "![Дообучение исходных C](figures/fine_tune_loss_recall.png)",
        "![Свежие C](figures/fresh_codebook_loss_recall.png)",
        "Это исследовательское продолжение на известном паттерне, не независимый новый benchmark. Новые search seed проверяют воспроизводимость поиска; дискретное совпадение с истинным I остаётся отдельной диагностикой после заморозки.",
    ]
    text = "\n\n".join(content) + "\n"
    (output / "REPORT.md").write_text(text)
    parent_report = parent / "REPORT.md"
    previous = parent_report.read_text().split(
        "\n\n## Осторожный подбор гиперпараметров совместного обучения"
    )[0]
    parent_report.write_text(
        previous.rstrip()
        + "\n\n"
        + text.replace("](figures/", "](joint_tuning/figures/")
    )
    ablation = json.loads(
        (parent / "update_ablation/selected_indexators.json").read_text()
    )
    comparison_rows = []
    for item in ablation:
        if item["primary"]:
            comparison_rows.append(
                [
                    item["label"],
                    5,
                    item["categorical_errors"],
                    f"{100 * item['inherited_test_metrics'][1]:.2f}",
                ]
            )
    comparison_rows.append(
        [
            "Joint support: tuned LR",
            5,
            selected["categorical_errors"],
            f"{100 * warm[1]:.2f}",
        ]
    )
    adam = json.loads((parent / "adam_baseline/summary.json").read_text())
    for item in adam.values():
        comparison_rows.append(
            [
                item["label"],
                item["parameters"],
                "—",
                f"{100 * item['selected_test_metrics'][1]:.2f}",
            ]
        )
    confirmation_records = [
        record
        for name, record in records.items()
        if name == "primary" or name.startswith("search_seed_")
    ]
    exact_count = sum(record["categorical_exact"] for record in confirmation_records)
    final_lines = [
        "# Pattern 0110: итог сравнения чередования и совместного обучения",
        f"Совместный поиск: выбранный на новых search seed индексатор {status}. Ошибок {selected['categorical_errors']}/88, F1 маски {100 * selected['binary_f1']:.2f}%, test BA после дообучения исходного C {100 * warm[1]:.2f}%. Точных I на трёх новых search seed: {exact_count}/3. Лучший индивидуальный pilot I, выбранный отдельно по validation, имеет {individual_metrics['categorical_errors']}/88 ошибок.",
        "Только фиксированное 0110, 11→8 ReLU→1, C0=0 и два свободных кода; вместе с общей головой 5 численных параметров. Во время поиска P являются softmax по 3 классам, после argmax I фиксируется и C дообучается 4096 шагов. Истинные окна/метки I не подаются поиску. Совпадение с GT проверяется только после выбора, с учётом переименования двух кодов и перестановок скрытых колонок.",
        table(
            ["Вариант", "Численных параметров", "Ошибок I/88", "Test BA, %"],
            comparison_rows,
        ),
        "Основные строки: source-selected индексаторы с продолжением исходного кодбука; Adam — плотные численные сети. Порядок обновлений сравнивался на парных seed 5/6/7 при одинаковых заранее выбранных LR. Дополнительный подбор joint LR отдельно подтверждён на новых seed 8/9/10, поэтому эта строка не является парной с первыми четырьмя. Чередование 3:1 получает втрое больше численных шагов, чем 1:1.",
        f"Подбор joint: numeric LR 0.03, alpha LR 0.003/0.01/0.03, sparsity 0/0.01; подтверждение выбрало alpha LR{source['architecture_lr']}, sparsity{source['sparsity_penalty']}, seed{source['search_seed']}, шаг{source['step']}. Основной I со свежими C: query-выбранный test BA {100 * summary['methods']['primary']['selected_test_metrics'][1]:.2f}%, среднее 32 стартов {100 * summary['methods']['primary']['mean_test_metrics'][1]:.2f}%. Способность обучить C и вероятность найти I оцениваются раздельно.",
        "Support 768 обучает C; в split-режимах query 256 обучает P. Joint использует один support-loss для P и C, query не получает архитектурных градиентов. Отдельная selection 512 без градиентов выбирает checkpoint и настройки по жёсткой модели после дообучения C, с усреднением по четырём текущим кодбукам и затем трём search seed. Test 512 не ранжирует кандидатов. Охлаждение и плохие локальные минимумы остаются источниками нестабильности; рост позднего soft-loss не всегда означает чистое переобучение.",
        "![Истинный и выбранные I](joint_tuning/figures/individual_indexator.png)",
        "![P0/P1/P2](joint_tuning/figures/assignment_probabilities.png)",
        "![Реальные веса и Adam](joint_tuning/figures/weight_comparison.png)",
        "![Validation абляции до открытия её test](update_ablation/figures/validation_loss.png)",
        "Все стадии и исходники сохранены: [полная история](REPORT.md), [парная абляция](update_ablation/REPORT.md), [подбор joint](joint_tuning/REPORT.md), [Adam](adam_baseline/REPORT.md). Это исследовательская серия на известном паттерне; расширения после первоначального test явно зарегистрированы. Новые seed — проверка инициализаций, а не новый независимый benchmark. Программные тесты не запускались; сохранены реальные обучающие кривые, состояния и хеши.",
    ]
    (parent / "FINAL_COMPARISON.md").write_text("\n\n".join(final_lines) + "\n")
    snapshot = output / "analysis_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in BASE.iterdir():
        if path.suffix in (".py", ".json", ".md"):
            shutil.copy2(path, snapshot / path.name)
    save_json(
        output / "complete.json",
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "selected_structure": selected,
            "software_tests_run": False,
        },
    )
    complete = json.loads((parent / "complete.json").read_text())
    complete["joint_tuning_added_utc"] = datetime.now(UTC).isoformat()
    complete["artifacts"] = {
        str(path.relative_to(parent)): digest(path)
        for path in sorted(parent.rglob("*"))
        if path.is_file() and path.name != "complete.json"
    }
    save_json(parent / "complete.json", complete)
    print(json.dumps({"structure": selected, "warm_test": warm}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "pilot", "confirmation", "reference", "finalize", "report"),
    )
    parser.add_argument(
        "--output", type=Path, default=BASE / "results/single_0110_20261104"
    )
    parser.add_argument("--device", default="cuda:6")
    args = parser.parse_args()
    torch.set_num_threads(2)
    child = args.output / "joint_tuning"
    device = torch.device(args.device)
    if args.stage == "prepare":
        prepare(args.output)
    elif args.stage in ("pilot", "confirmation"):
        run(args.output, args.stage, device)
    elif args.stage == "finalize":
        plan = json.loads((child / "protocol.json").read_text())
        single.freeze(plan, child)
        single.fresh_calibrate(plan, child, device)
        single.evaluate(plan, child, device)
    elif args.stage == "reference":
        individual_reference(args.output)
    else:
        report(args.output)


if __name__ == "__main__":
    main()
