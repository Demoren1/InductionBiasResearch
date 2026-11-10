"""Locate recovery failures using frozen banks and stored validation scores."""

import json
from collections import Counter

import numpy as np

from experiments.paper_structure_benchmark_v1.reporting.evaluate import (
    write_csv,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
    save_json,
)


def diagnose(base, output):
    """Use structural truth only after fitting to distinguish failure stages."""
    rows = json.loads((output / "summary.json").read_text())["rows"]
    diagnostics = []
    for row in rows:
        cohort = (
            "general" if row["family"].startswith("general_") else "standard"
        )
        root = base / cohort
        case = row["case"]
        codes = json.loads((root / "public" / f"{case}.json").read_text())[
            "codes"
        ]
        truth = np.load(root / "private" / f"{case}.npz")["index"]
        truth = canonical_index(truth, codes)
        condition = int(row["source_noise"] > 0)
        folder = (
            root / "methods" / row["method"] / "fits" / f"{case}_c{condition}"
        )
        proposed = np.load(folder / "joint_bank.npz")["indices"]
        fitted = np.load(folder / "joint_fitted.npz")["indices"]
        ranks = [
            i + 1
            for i, index in enumerate(proposed)
            if np.array_equal(index, truth)
        ]
        positions = [
            i for i, index in enumerate(fitted) if np.array_equal(index, truth)
        ]
        scores = json.loads((folder / "joint_selected.json").read_text())[
            "all_meta_mse"
        ]
        if row["exact"]:
            stage = "recovered"
        elif positions:
            stage = "numerical_fit_or_validation_selection"
        elif ranks and ranks[0] > 32:
            stage = "projection_rank_cutoff"
        else:
            stage = "not_proposed"
        diagnostics.append(
            {
                "family": row["family"],
                "case": case,
                "source_noise": row["source_noise"],
                "method": row["method"],
                "stage": stage,
                "true_projection_rank": ranks[0] if ranks else None,
                "true_in_comparison_bank": bool(positions),
                "true_structure_meta_mse": min(scores[i] for i in positions)
                if positions
                else None,
                "selected_meta_mse": row["source_meta_mse"],
            }
        )
    write_csv(output / "bank_diagnostics.csv", diagnostics)
    counts = {
        method: dict(
            Counter(
                row["stage"] for row in diagnostics if row["method"] == method
            )
        )
        for method in ("full", "anchor_fixed", "anchor_free")
    }
    save_json(
        output / "bank_diagnostics.json",
        {
            "posthoc_analysis": True,
            "learner_or_hyperparameter_changes": False,
            "uses_stored_predictions_and_scores": True,
            "counts": counts,
        },
    )
    report_path = output / "REPORT.md"
    title, body = report_path.read_text().split("\n", 1)
    total = len(rows) // 3
    exact = [
        counts[method].get("recovered", 0)
        for method in ("full", "anchor_fixed", "anchor_free")
    ]
    overview = [
        "",
        f"Точно восстановленные индексаторы: полный поиск {exact[0]}/{total}; один запуск с фиксированной связностью {exact[1]}/{total}; один запуск с разрешёнными нулями {exact[2]}/{total}.",
        "В проверенной конфигурации упрощение снижает качество, поэтому основной остаётся полная схема.",
        "",
        "| Набор | Полный поиск | Один запуск, фиксированная связность | Один запуск, разрешены нули |",
        "|---|---:|---:|---:|",
    ]
    for label, families in (
        ("Pattern", ("pattern",)),
        ("DeepSets", ("deepsets",)),
        (
            "Общая синтетика",
            ("general_local", "general_quartet_mix", "general_random"),
        ),
    ):
        entries = []
        for method in ("full", "anchor_fixed", "anchor_free"):
            group = [
                row
                for row in rows
                if row["family"] in families and row["method"] == method
            ]
            entries.append(
                f"{sum(row['exact'] for row in group)}/{len(group)}"
            )
        overview.append(f"| {label} | {' | '.join(entries)} |")
    overview.extend(
        [
            "",
            "В таблице — точный подписанный индексатор с учётом допустимых симметрий, а не точность предсказаний. Чистые и шумные source-условия объединены; подробная разбивка приведена ниже.",
            "Вызовов проекции стало в 132 раза меньше. Это сокращение числа вызовов исходного кода; обучение source-моделей и численный refit сохраняются. Одновременно убраны дополнительные старты и альтернативные support proposals, поэтому их отдельный вклад не измерен.",
            "",
            "Диагностика по уже сохранённым банкам (анализ после завершения, без изменения learners):",
            "",
            "| Метод | Восстановлен | Правильный кандидат не построен | Отсечён лимитом 32 | Проверен, но проиграл при численной подгонке/валидации |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method, stages in counts.items():
        values = [
            stages.get(stage, 0)
            for stage in (
                "recovered",
                "not_proposed",
                "projection_rank_cutoff",
                "numerical_fit_or_validation_selection",
            )
        ]
        overview.append(f"| {method} | {' | '.join(map(str, values))} |")
    for row in diagnostics:
        if (
            row["method"] == "full"
            and row["source_noise"] == 0
            and row["stage"] == "numerical_fit_or_validation_selection"
        ):
            overview.extend(
                [
                    "",
                    f"На чистом {row['family']} {row['case']} правильный индексатор получил source meta-MSE {row['true_structure_meta_mse']:.5g}, а выбранный неточный — {row['selected_meta_mse']:.5g}. Численная подгонка правильного кандидата тоже требует внимания: при точной генеративной структуре и отсутствии шума нулевая ошибка достижима.",
                ]
            )
    overview.extend(
        [
            "",
            "Численные состояния сохранены; большие покадровые JSON-истории refit исключены из компактного архива. Итоговые параметры, оценки кандидатов и source/target-кривые сохранены.",
            "",
        ]
    )
    report_path.write_text(title + "\n" + "\n".join(overview) + body)
    return counts
