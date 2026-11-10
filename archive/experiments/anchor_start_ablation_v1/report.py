"""Score only frozen outputs and preserve a compact, explicit comparison."""

import json
import shutil
import tarfile
from pathlib import Path

import numpy as np

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

from .diagnostics import diagnose
from .plots import heatmaps


def verify_completion(base, project):
    """Verify completed learners and immutable inputs before final scoring."""
    state = json.loads((base / "state.json").read_text())
    if (
        state["failed"]
        or state["live"]
        or len(state["done"]) != state["total"]
    ):
        raise RuntimeError("The paired experiment is incomplete")
    registration = json.loads((base / "registration.json").read_text())
    for name, expected in registration["code_sha256"].items():
        if name.startswith("experiments/anchor_start_ablation_v1/"):
            continue
        if digest(project / name) != expected:
            raise RuntimeError(f"Original learner code changed: {name}")
    commitments = json.loads((base / "data_commitments.json").read_text())
    for name, expected in commitments["files"].items():
        if digest(base / name) != expected:
            raise RuntimeError(f"Registered observation changed: {name}")
    receipts = list(base.glob("*/receipts/*.json"))
    for path in receipts:
        item = json.loads(path.read_text())
        if item["status"] != "complete" or item["violations"]:
            raise RuntimeError(f"Unsuccessful learner receipt: {path}")
    return {
        "complete_jobs": len(state["done"]),
        "data_commitments_checked": len(commitments["files"]),
        "unchanged_core_files": sum(
            not name.startswith("experiments/anchor_start_ablation_v1/")
            for name in registration["code_sha256"]
        ),
        "receipts": len(receipts),
        "violations": 0,
        "interim_scoring": (
            "User-requested source-only interim scoring for Pattern and DeepSets; remaining learners unchanged"
        ),
    }


def preserve_numerical_state(base, output):
    """Retain compact source solutions and fitted banks, excluding raw data."""
    files = []
    for cohort in ("standard", "general"):
        root = base / cohort
        for name in (
            "source",
            "dense",
            "fits",
            "metrics",
            "receipts",
            "target_cache",
        ):
            files.extend(p for p in (root / name).rglob("*") if p.is_file())
        files.extend(
            p
            for p in (root / "methods").glob("*/fits/*/*")
            if p.is_file()
            and p.name not in ("fitted.npz", "joint_fitted.json")
        )
        for path in (root / "methods").glob("*/fits/*/joint_fitted.json"):
            record = json.loads(path.read_text())
            history = record.pop("history")
            record["omitted_iteration_history_blocks"] = len(history)
            record["original_trace_json_sha256"] = digest(path)
            destination = (
                output
                / "records"
                / cohort
                / path.parent.name
                / path.parents[2].name
                / "joint_fitted_summary.json"
            )
            save_json(destination, record)
        files.extend((root / "public").glob("*.json"))
        files.append(root / "protocol.json")
    files = sorted(set(files))
    archive = output / "numerical_state.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=3) as stream:
        for path in files:
            stream.add(
                path, arcname=str(path.relative_to(base)), recursive=False
            )
    save_json(
        output / "numerical_state_manifest.json",
        {
            "archive_sha256": digest(archive),
            "bytes": archive.stat().st_size,
            "files": {str(p.relative_to(base)): digest(p) for p in files},
            "raw_observations_included": False,
            "large_joint_iteration_histories_included": False,
        },
    )


def report(base):
    frozen = json.loads((base / "frozen.json").read_text())
    for name, expected in frozen["files"].items():
        if digest(base / name) != expected:
            raise RuntimeError(f"Frozen output changed: {name}")
    project = Path(__file__).resolve().parents[2]
    audit = verify_completion(base, project)
    output = project / "artifacts" / "anchor_start_2026-11-10"
    output.mkdir(exist_ok=False)
    save_json(output / "verified_before_final_scoring.json", audit)
    mapping = json.loads((base / "target_map.json").read_text())
    if len(mapping) != 72:
        raise RuntimeError("Expected all 72 paired source selections")
    rows, target_rows, arrays, answers = [], [], {}, {}
    for item in mapping.values():
        root = base / item["cohort"]
        case, condition, method = [
            item[k] for k in ("case", "condition", "method")
        ]
        number = int(case.rsplit("_", 1)[1])
        family = (
            ("pattern" if number < 3 else "deepsets")
            if item["cohort"] == "standard"
            else ("general_local", "general_quartet_mix", "general_random")[
                number // 2
            ]
        )
        key = f"{case}_c{condition}"
        sizes = json.loads((root / "public" / f"{case}.json").read_text())
        codes = sizes["codes"]
        folder = root / "methods" / method / "fits" / key
        chosen = dict(np.load(folder / "joint_selected.npz"))
        index = chosen["joint_functional_index"]
        truth = np.load(root / "private" / f"{case}.npz")["index"]
        aligned, recovery = align_index(index, truth, codes)
        overlap = (truth != 0) & (aligned != 0)
        initial = np.load(folder / "fitted.npz")["indices"]
        proposals = np.load(folder / "joint_bank.npz")["indices"]
        truth_key = canonical_index(truth, codes).tobytes()
        metadata = json.loads((folder / "joint_bank.json").read_text())
        selected_metadata = json.loads(
            (folder / "joint_selected.json").read_text()
        )
        refit_metadata = json.loads((folder / "joint_fitted.json").read_text())
        cost = json.loads(
            (root / "metrics" / f"{key}_fit_{method}.json").read_text()
        )
        selected_meta = selected_metadata["methods"]["joint_functional"][
            "meta_mse"
        ]
        predictions = np.load(
            root / "target_cache" / item["cache"] / "predictions.npz"
        )
        private_targets = np.load(root / "private" / f"{case}_targets.npz")
        target = np.stack([private_targets[f"query_y_{t}"] for t in range(8)])
        answers[f"{item['cohort']}_{case}_query_y"] = target
        losses = ((predictions["prediction"] - target) ** 2).mean((1, 2))
        adam_losses = ((predictions["adam_prediction"] - target) ** 2).mean(
            (1, 2)
        )
        entry = {
            "family": family,
            "case": case,
            "replicate": number % (3 if item["cohort"] == "standard" else 2),
            "source_noise": condition
            * (0.1 if item["cohort"] == "standard" else 0.2),
            "method": method,
            **recovery,
            "true_active_connections": int((truth != 0).sum()),
            "sharing_ari_on_overlap": adjusted_rand(
                np.abs(aligned[overlap]), np.abs(truth[overlap])
            ),
            "selected_in_initial_bank": any(
                np.array_equal(index, a) for a in initial
            ),
            "true_in_initial_bank": any(
                a.tobytes() == truth_key for a in initial
            ),
            "true_in_projection_bank": any(
                a.tobytes() == truth_key for a in proposals
            ),
            "projection_calls": metadata["projection_calls"],
            "anchors": len(metadata["anchors"]),
            "projection_seconds": metadata["seconds"],
            "projection_candidates": len(proposals),
            "refitted_proposals": len(
                refit_metadata["new_numeric_candidates"]
            ),
            "refit_and_selection_wall_seconds": cost["wall_seconds"],
            "source_meta_mse": selected_meta,
            "target_mse_1024": float(losses.mean()),
            "target_adam_mse_1024": float(adam_losses.mean()),
            "numerical_parameters": codes
            + sizes["hidden"] * (1 + sizes["outputs"])
            + sizes["outputs"],
            "outputs": sizes["outputs"],
        }
        rows.append(entry)
        for task, (mse, adam) in enumerate(zip(losses, adam_losses)):
            target_rows.append(
                {
                    "family": family,
                    "case": case,
                    "condition": condition,
                    "method": method,
                    "task": task,
                    "mse": float(mse),
                    "adam_mse": float(adam),
                }
            )
        prefix = f"{item['cohort']}_{key}_{method}"
        arrays[prefix + "_index"] = index
        arrays[prefix + "_aligned"] = aligned
        arrays[prefix + "_truth"] = truth
        arrays[prefix + "_source_parameters"] = chosen[
            "joint_functional_parameters"
        ]
        for name in (
            "joint_selected.json",
            "joint_bank.json",
        ):
            destination = (
                output / "records" / item["cohort"] / key / method / name
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / name, destination)
    rows.sort(
        key=lambda r: (r["family"], r["case"], r["source_noise"], r["method"])
    )
    write_csv(output / "recovery.csv", rows)
    write_csv(output / "transfer_per_task.csv", target_rows)
    comparisons = []
    for row in rows:
        if row["method"] == "full":
            continue
        reference = next(
            item
            for item in rows
            if item["method"] == "full"
            and (item["family"], item["case"], item["source_noise"])
            == (row["family"], row["case"], row["source_noise"])
        )
        comparisons.append(
            {
                "family": row["family"],
                "case": row["case"],
                "source_noise": row["source_noise"],
                "method": row["method"],
                "exact_difference": int(row["exact"])
                - int(reference["exact"]),
                "entry_error_difference": row["signed_entry_errors"]
                - reference["signed_entry_errors"],
                "target_mse_difference": row["target_mse_1024"]
                - reference["target_mse_1024"],
                "projection_call_ratio": reference["projection_calls"]
                / row["projection_calls"],
            }
        )
    write_csv(output / "paired_comparison.csv", comparisons)
    np.savez_compressed(output / "selected_structures.npz", **arrays)
    np.savez_compressed(output / "scoring_answers.npz", **answers)
    for family in sorted({row["family"] for row in rows}):
        panels = []
        conditions = sorted(
            {
                (row["case"], row["source_noise"])
                for row in rows
                if row["family"] == family
            }
        )
        cohort = "general" if family.startswith("general_") else "standard"
        for case, noise in conditions:
            condition = int(noise > 0)
            prefix = f"{cohort}_{case}_c{condition}"
            panels.append(
                {
                    "label": f"{case}; noise {noise:g}",
                    "indices": [
                        arrays[prefix + "_full_truth"],
                        *(
                            arrays[f"{prefix}_{method}_aligned"]
                            for method in (
                                "full",
                                "anchor_fixed",
                                "anchor_free",
                            )
                        ),
                    ],
                }
            )
        codes = 2 if family == "pattern" else 4
        heatmaps(output / f"{family}_heatmaps.png", panels, codes, family)
    costs = []
    for cohort in ("standard", "general"):
        for path in (base / cohort / "metrics").glob("*.json"):
            record = json.loads(path.read_text())
            costs.append({"cohort": cohort, "file": path.name, **record})
    write_csv(output / "execution_costs.csv", costs)
    text = [
        "# Один запуск проекции из каждого якоря",
        "",
        "Свежая парная проверка; численные гиперпараметры не подбирались по результатам.",
        "16 source и 8 target-задач на структуру; все target-числа обучены заново.",
        "В этом контроле Pattern 0110 использует вещественные выходы сети; отдельная бинарная классификация не перезапускалась.",
        "Структура задаёт только первую матрицу. Используется удвоенный кодбук с нулём и парами противоположных значений: K=2 независимых числа для Pattern и K=4 для DeepSets и общей синтетики. Выходная матрица и bias обучаются свободно.",
        "На каждой source-задаче сначала обучается плотная модель. Из её весов строится начальный банк B0; несколько лучших по средней численной валидации индексаторов задают якоря и выравнивания source-моделей.",
        "Затем варианты поиска предлагают общие группы весов, чередуя калибровку кодбуков и дискретные назначения. Каждый предложенный индексатор проверяется обучением численных параметров на source-задачах; независимая структурная валидация выбирает итоговый индексатор.",
        "На target-задачах индексатор фиксирован, кодбук, скрытые bias и свободная выходная матрица обучаются с новой инициализации. Структурная истина используется только для итоговой оценки.",
        "",
        "| Семейство | Source noise | Метод | Exact I | Exact mask | Target MSE, N=1024 | Выбран из B0 | Вызовов проекции |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for family in sorted({r["family"] for r in rows}):
        for noise in sorted(
            {r["source_noise"] for r in rows if r["family"] == family}
        ):
            for method in ("full", "anchor_fixed", "anchor_free"):
                group = [
                    r
                    for r in rows
                    if (r["family"], r["source_noise"], r["method"])
                    == (family, noise, method)
                ]
                total = len(group)
                text.append(
                    f"| {family} | {noise:g} | {method} | "
                    f"{sum(r['exact'] for r in group)}/{total} | "
                    f"{sum(r['mask_exact'] for r in group)}/{total} | "
                    f"{np.mean([r['target_mse_1024'] for r in group]):.6g} | "
                    f"{sum(r['selected_in_initial_bank'] for r in group)}/{total} | "
                    f"{np.mean([r['projection_calls'] for r in group]):g} |"
                )
    text.extend(
        [
            "",
            "`full` — исходный G34; `anchor_fixed` — один вызов на anchor с его support;",
            "`anchor_free` — один вызов на anchor с разрешённым нулём на всех позициях.",
            "Во всех вариантах сохранены B0 и исходная обработка знаковых ориентаций.",
            "Уменьшаются и число стартов, и набор support proposals; их отдельный причинный вклад не разделён.",
            "Если выбран исходный B0, успех нельзя приписывать совместной проекции.",
            "General содержит по две заранее заданные страты на семейство: один/два выхода и общий/раздельные bias; это не одинаково распределённые повторы.",
            "Pattern-повторы используют одну архитектуру 0110 с новыми source-данными и допустимыми перестановками нейронов.",
            "ARI шэринга посчитан только на пересечении активных позиций; его нужно читать вместе с метриками маски.",
            "Wall time измерен при чужой нагрузке на GPU и не является выделенным hardware benchmark.",
            "Torch FLOPs в этой быстрой проверке не измерялись; CPU-геометрия также не включена в FLOPs.",
            "Сокращение вызовов относится к совместной проекции; обучение source-моделей, построение B0 и численный refit сохраняются. Это не равный коэффициент ускорения всего метода.",
            "Source-модели, численные состояния банков и target-предсказания сохранены в numerical_state.tar.gz; сырые наблюдения воспроизводятся генераторами из code_snapshot и seeds регистрации.",
            "По просьбе пользователя структурные результаты завершённых Pattern и DeepSets были раскрыты до общего окончания; это отражено в interim_pattern_amendment.json и interim_deepsets_amendment.json. После этого оставшиеся learners не менялись.",
            "Хитмапы показывают все условия; выравнивание с истинными индексаторами выполнено только при оценке.",
        ]
    )
    text.extend(["", "Хитмапы всех условий:", ""])
    for family in sorted({row["family"] for row in rows}):
        text.append(f"- [{family}]({family}_heatmaps.png)")
    (output / "REPORT.md").write_text("\n".join(text) + "\n")
    for name in (
        "registration.json",
        "PROTOCOL.md",
        "frozen.json",
        "state.json",
        "target_map.json",
        "data_commitments.json",
        "pretraining_formatting.json",
        "execution_batch_amendment.json",
        "resumable_batch_amendment.json",
        "repacking.json",
        "interim_pattern_amendment.json",
        "interim_deepsets_amendment.json",
        "environment.json",
    ):
        shutil.copy2(base / name, output / name)
    shutil.copytree(base / "code_snapshot", output / "code_snapshot")
    shutil.copytree(base / "code_versions", output / "code_versions")
    shutil.copytree(base / "logs", output / "logs")
    shutil.copytree(
        Path(__file__).parent,
        output / "analysis_code",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in (
        "archive/experiments/g34_theory_v1/general.py",
        "experiments/paper_structure_benchmark_v1/protocol/prepare.py",
        "experiments/paper_structure_benchmark_v1/reporting/evaluate.py",
    ):
        path = output / "code_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, path)
    preserve_numerical_state(base, output)
    save_json(output / "summary.json", {"rows": rows, "run_root": str(base)})
    diagnose(base, output)
    print(json.dumps({"report": str(output / "REPORT.md"), "rows": len(rows)}))
