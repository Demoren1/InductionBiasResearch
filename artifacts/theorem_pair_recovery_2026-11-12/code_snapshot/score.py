"""Audit frozen source selections; structural truth never enters extraction."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from experiments.paper_structure_benchmark_v1.reporting.evaluate import (
    adjusted_rand,
    align_index,
    write_csv,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
    digest,
    save_json,
)
from experiments.sharing_structure_research_v1.blind_joint_bank import align_source

from .recover import canonical_partition, contrast_rank


def graph_conditions(index, codes):
    """Check repeated-pair connectivity only in the post-freeze audit."""
    groups = [set(np.abs(column[column != 0]).tolist()) for column in index.T]
    counts = []
    for group in range(1, codes + 1):
        neurons = [h for h, values in enumerate(groups) if group in values]
        edges = np.eye(len(neurons), dtype=bool)
        for left, first in enumerate(neurons):
            for right, second in enumerate(neurons):
                edges[left, right] = bool((groups[first] & groups[second]) - {group})
        count, _ = connected_components(csr_matrix(edges), directed=False)
        counts.append(count)
    return counts


def anchor_space_errors(work, cohort, case, condition, truth, codes):
    """Measure subspace error where an anchor's unsigned groups are correct."""
    root = work / cohort
    key = f"{case}_c{condition}"
    base = dict(np.load(root / "fits" / key / "fitted.npz"))
    true_key = canonical_partition(np.abs(truth), codes).tobytes()
    tasks = base["parameters"].shape[1]
    sources = [
        dict(np.load(root / "source" / f"{key}_t{task}" / "learned.npz"))
        for task in range(tasks)
    ]
    result = []
    for number in base["validation_mse"].mean(1).argsort()[:4]:
        index = base["indices"][number]
        if canonical_partition(np.abs(index), codes).tobytes() != true_key:
            continue
        mappings = [
            align_source(
                index,
                base["parameters"][number, task, :codes],
                source["normal"],
                source["uncertainty"],
            )
            for task, source in enumerate(sources)
        ]
        normal = np.stack(
            [source["normal"][:, mapping] for source, mapping in zip(sources, mappings)]
        )
        rows, columns = np.where(index)
        labels = np.abs(index[rows, columns]) - 1
        incidence = np.eye(codes)[labels]
        logs = np.log(np.maximum(np.abs(normal[:, rows, columns]), 1e-300)).T
        for neuron in np.unique(columns):
            active = columns == neuron
            logs[active] -= logs[active].mean(0)
            incidence[active] -= incidence[active].mean(0)
        rank = contrast_rank(index, codes)
        true_basis = np.linalg.svd(incidence, full_matrices=False)[0][:, :rank]
        basis, singular, _ = np.linalg.svd(logs, full_matrices=False)
        estimated = basis[:, :rank]
        eta = float(
            np.linalg.norm(estimated @ estimated.T - true_basis @ true_basis.T, 2)
        )
        maximum_size = int(np.bincount(labels).max())
        limit = 1 / (8 * np.sqrt(maximum_size))
        result.append(
            {
                "anchor": int(number),
                "eta": eta,
                "sufficient_limit": float(limit),
                "stable_condition": bool(eta < limit),
                "rank": rank,
                "singular_values": singular.tolist(),
            }
        )
    return result


def score(bundle, output, work):
    for filename in ("proposal_freeze.json", "selection_freeze.json"):
        freeze = json.loads((output / filename).read_text())
        for name, expected in freeze["files"].items():
            if digest(output / name) != expected:
                raise RuntimeError(f"Frozen artifact changed: {name}")
    with np.load(bundle / "selected_structures.npz") as stored:
        references = dict(stored)
    sizes = json.loads((bundle / "case_sizes.json").read_text())
    rows, conditions, arrays = [], [], {}
    for case_key, case_sizes in sorted(sizes.items()):
        cohort, case = case_key.split("/")
        codes = case_sizes["codes"]
        number = int(case.rsplit("_", 1)[1])
        family = (
            ("pattern" if number < 3 else "deepsets")
            if cohort == "standard"
            else ("general_local", "general_quartet_mix", "general_random")[number // 2]
        )
        for condition in (0, 1):
            folder = output / cohort / f"{case}_c{condition}"
            truth = references[f"{cohort}_{case}_c{condition}_full_truth"]
            true_key = canonical_index(truth, codes).tobytes()
            unsigned_key = canonical_partition(np.abs(truth), codes).tobytes()
            candidates = dict(np.load(folder / "candidates.npz"))
            selected = dict(np.load(folder / "selection.npz"))
            functional = json.loads((folder / "functional.json").read_text())
            metadata = json.loads((folder / "candidates.json").read_text())
            signature_ranks = [
                i + 1
                for i, index in enumerate(candidates["indices"])
                if index.tobytes() == true_key
            ]
            unsigned_ranks = [
                i + 1
                for i, index in enumerate(candidates["unsigned_indices"])
                if index.tobytes() == unsigned_key
            ]
            connectivity = graph_conditions(truth, codes)
            space_errors = anchor_space_errors(
                work, cohort, case, condition, truth, codes
            )
            conditions.append(
                {
                    "cohort": cohort,
                    "family": family,
                    "case": case,
                    "condition": condition,
                    "pair_graphs_connected": all(count == 1 for count in connectivity),
                    "graph_components": connectivity,
                    "true_rank": contrast_rank(truth, codes),
                    "correct_unsigned_anchors": len(space_errors),
                    "anchor_space_errors": space_errors,
                    "any_stable_anchor": any(
                        item["stable_condition"] for item in space_errors
                    ),
                }
            )
            coverage = {
                "signature_unsigned_candidates": len(candidates["unsigned_indices"]),
                "signature_signed_candidates": len(candidates["indices"]),
                "true_unsigned_rank": unsigned_ranks[0] if unsigned_ranks else None,
                "true_signed_rank": signature_ranks[0] if signature_ranks else None,
                "true_unsigned_proposed": bool(unsigned_ranks),
                "true_signed_proposed": bool(signature_ranks),
                "true_signed_in_top32": bool(
                    signature_ranks and signature_ranks[0] <= 32
                ),
                "true_in_initial_bank": any(
                    index.tobytes() == true_key
                    for index in candidates["initial_indices"]
                ),
                "pair_graphs_connected": all(count == 1 for count in connectivity),
                "any_stable_anchor": any(
                    item["stable_condition"] for item in space_errors
                ),
                "proposal_seconds": metadata["seconds"],
            }
            for method in ("baseline_full", "signature_only", "augmented"):
                choice = functional["choices"].get(method)
                row = {
                    "cohort": cohort,
                    "family": family,
                    "case": case,
                    "condition": condition,
                    "method": method,
                    **coverage,
                }
                if choice is None:
                    row.update(
                        exact=False,
                        unsigned_exact=False,
                        mask_exact=False,
                        signed_entry_errors=None,
                        source_meta_mse=None,
                    )
                    rows.append(row)
                    continue
                index = selected[f"{method}_index"]
                aligned, metrics = align_index(index, truth, codes)
                overlap = (aligned != 0) & (truth != 0)
                row.update(metrics)
                row.update(
                    unsigned_exact=canonical_partition(np.abs(index), codes).tobytes()
                    == unsigned_key,
                    sharing_ari=adjusted_rand(
                        np.abs(aligned[overlap]), np.abs(truth[overlap])
                    ),
                    source_meta_mse=choice["source_meta_mse"],
                )
                rows.append(row)
                prefix = f"{cohort}_{case}_c{condition}_{method}"
                arrays[f"{prefix}_index"] = index
                arrays[f"{prefix}_aligned"] = aligned
                arrays[f"{prefix}_truth"] = truth
    write_csv(output / "recovery.csv", rows)
    np.savez_compressed(output / "selected_structures.npz", **arrays)
    save_json(
        output / "assumption_diagnostics.json",
        {"posthoc": True, "conditions": conditions},
    )
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    summary = {
        method: {
            "exact": sum(row["exact"] for row in group),
            "unsigned_exact": sum(row["unsigned_exact"] for row in group),
            "mask_exact": sum(row["mask_exact"] for row in group),
            "conditions": len(group),
        }
        for method, group in groups.items()
    }
    coverage_rows = groups["baseline_full"]
    summary["coverage"] = {
        key: sum(row[key] for row in coverage_rows)
        for key in (
            "true_unsigned_proposed",
            "true_signed_proposed",
            "true_signed_in_top32",
            "true_in_initial_bank",
            "pair_graphs_connected",
            "any_stable_anchor",
        )
    }
    save_json(output / "summary.json", summary)
    lines = [
        "# Восстановление индексаторов по теореме 1",
        "",
        "Ретроспективный эксперимент на сохранённых плотных моделях: 24 условия, 16 исходных задач в каждом. Все новые банки и source-выборы зафиксированы до открытия истинных структур. Гиперпараметры по истинным индексаторам не подбирались. Исторические результаты были известны заранее, поэтому это не новая независимая подтверждающая серия.",
        "",
        "| Метод | Точный знаковый индексатор | Точные группы без знаков | Точная маска |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "baseline_full": "G34, сохранённый полный поиск",
        "signature_only": "Только кандидаты по сигнатурам",
        "augmented": "G34 + кандидаты по сигнатурам",
    }
    for method, label in labels.items():
        item = summary[method]
        lines.append(
            f"| {label} | {item['exact']}/24 | {item['unsigned_exact']}/24 | {item['mask_exact']}/24 |"
        )
    lines.extend(
        [
            "",
            "Это результаты выбора по отдельной source-валидации, а не выбор лучшего по истинной структуре. Объединённый банк получает дополнительный численный бюджет; равенство стоимости не заявляется. Target-адаптация в этом эксперименте не измерялась.",
            "",
            "## Покрытие банка",
            "",
            f"Правильная группировка без знаков построена в {summary['coverage']['true_unsigned_proposed']}/24 условиях; правильный знаковый индексатор — в {summary['coverage']['true_signed_proposed']}/24, из них в первых 32 — в {summary['coverage']['true_signed_in_top32']}/24. Это диагностическое oracle-покрытие банка, не точность выбора.",
            "",
            "| Семейство | G34 | Только сигнатуры | Объединённый банк | Истинные группы есть среди предложенных |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for family in sorted({row["family"] for row in rows}):
        entries = [
            [row for row in groups[method] if row["family"] == family]
            for method in labels
        ]
        count = len(entries[0])
        values = [f"{sum(row['exact'] for row in group)}/{count}" for group in entries]
        present = sum(row["true_unsigned_proposed"] for row in entries[0])
        lines.append(f"| {family} | {' | '.join(values)} | {present}/{count} |")
    lines.extend(
        [
            "",
            "## Что именно реализовано",
            "",
            "Маски и сопоставление нейронов берутся из существующего source-only механизма якорей. Для каждой маски вычитаются средние логарифмов внутри нейрона, оценивается пространство через SVD, вычисляются сигнатуры пар и выполняются оба правила объединения теоремы. Частичные разбиения не дополняются k-means. Для шума используются заранее заданные ранги и пороги; ранги проверяются по размерности полученного разбиения. Относительные знаки оцениваются по профилям исходных задач. LS-калибровка оценивает уже построенный кандидат и не меняет его группы.",
            "",
            "Новые кандидаты получают 4 source-derived старта и 48 шагов Gauss–Newton на каждой из 16 задач. Численные состояния выбираются по numerical validation; структура — по отдельной source meta выборке. Совпадения с исходным банком используют его сохранённые параметры. Все 96 восстановленных публичных файлов наблюдений совпали с исходной серией по SHA-256.",
            "",
            "## Предпосылки",
            "",
            f"Графы повторяющихся пар связны во всех группах в {summary['coverage']['pair_graphs_connected']}/24 условиях. Среди правильных unsigned-якорей достаточный порог устойчивости оценённого пространства выполнен хотя бы для одного якоря в {summary['coverage']['any_stable_anchor']}/24. Последнее — диагностика после раскрытия истины, а не правило выбора. Полные значения ошибки проекторов, спектры и компоненты графов сохранены в assumption_diagnostics.json.",
            "",
            "Один набор не дал полного кандидата с заданным числом групп. Фиксированная сетка порогов, неизвестный истинный ранг, ошибки маски/выравнивания и ненулевые ошибки плотных моделей ограничивают практическую полноту ветки. Формальная гарантия теоремы не переносится автоматически на произвольные обученные dense-модели.",
            "",
            "Подробные выбранные структуры и метрики: recovery.csv и selected_structures.npz. Параметры, source-оценки, происхождение кандидатов, заморозки и журналы сохранены по каждому условию. Исходное ядро G34 и статья не изменялись.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, default=Path("artifacts/anchor_start_2026-11-10")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    score(args.bundle, args.output, args.work)


if __name__ == "__main__":
    main()
