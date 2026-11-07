"""Separate candidate-pool and meta-task effects using frozen G23 predictors."""

import argparse
import csv
import itertools
import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .shared_structure_common import structural_metrics
from .shared_structure_data import teacher
from .shared_structure_numeric import canonical_index
from .shared_structure_pipeline import BASE, check_code, pid_identity


def register(root):
    destination = root / "factor_ablation"
    if (root / "common/audit.json").exists():
        raise RuntimeError("This diagnostic must be registered before teacher audit")
    parent = json.loads((root / "performance_continuation.json").read_text())
    check_code(parent)
    if pid_identity(parent["supervisor"]["pid"]) != parent["supervisor"]:
        raise RuntimeError("Expected primary comparison supervisor is not live")
    destination.mkdir(exist_ok=False)
    snapshot = destination / "executed_code_snapshot"
    snapshot.mkdir()
    names = set(parent["code_sha256"]) | {Path(__file__).name}
    protocol = {
        "registered_utc": datetime.now(UTC).isoformat(),
        "primary_protocol_sha256": digest(root / "protocol.json"),
        "performance_protocol_sha256": digest(root / "performance_continuation.json"),
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": {name: digest(BASE / name) for name in sorted(names)},
        "primary_supervisor": parent["supervisor"],
        "supervisor": pid_identity(os.getpid()),
        "selection_rule": "2x2: candidate origins source0/all4 crossed with meta scoring task0/mean4. Reuse primary selected candidate for the two original cells; cross cells argmin stored task scores. No fit, I refinement, new labels or GPU use. Freeze every choice before this diagnostic's teacher audit.",
        "scope": "Original single-source G23 banks only; not joint bi-level supplement. Same geometries and meta panels, not independent confirmation or a new extractor.",
        "poll_seconds": 30,
    }
    for name in (Path(__file__).name, "shared_structure_common.py"):
        shutil.copy2(BASE / name, snapshot / name)
    shutil.copy2(BASE / "THEORY.md", destination / "theory_at_registration.md")
    save_json(destination / "protocol.json", protocol)
    return protocol


def select_factors(root, plan, protocol):
    """Use saved scores only; no call to teacher or evaluation alignment here."""
    common = root / "common"
    frozen = json.loads((common / "selection_frozen.json").read_text())
    fitted = json.loads((common / "fit_frozen.json").read_text())
    fit_freeze_hash = digest(common / "fit_frozen.json")
    records, selected, input_hashes = [], {}, {}
    for job_id in range(len(plan["families"]) * len(plan["noise"])):
        path = common / "selected" / f"job_{job_id:03d}"
        for suffix in (".json", ".npz"):
            source = path.with_suffix(suffix)
            expected = frozen["sha256"][str(source.relative_to(common))]
            if digest(source) != expected:
                raise RuntimeError("Frozen primary selections/predictors changed")
            input_hashes[str(source.relative_to(root))] = expected
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata["fit_freeze_sha256"] != fit_freeze_hash:
            raise RuntimeError("Primary selection refers to another numerical fit")
        bank_path = common / "banks" / f"job_{job_id:03d}.json"
        bank_hash = digest(bank_path)
        if bank_hash != fitted["sha256"][str(bank_path.relative_to(common))]:
            raise RuntimeError("Candidate pool changed after numerical fit freeze")
        bank = json.loads(bank_path.read_text())
        input_hashes[str(bank_path.relative_to(root))] = bank_hash
        primary = {(row["method"], row["scope"]): row for row in metadata["records"]}
        with np.load(path.with_suffix(".npz")) as values:
            indices = values["candidate_indices"]
            scores = values["task_scores"]
            for method, pool, criterion in itertools.product(
                plan["methods"], ("source0", "all4"), ("task0", "mean4")
            ):
                available = [
                    number
                    for number in bank["methods"][method]
                    if pool == "all4"
                    or any(
                        origin["method"] == method and origin["task"] == 0
                        for origin in bank["candidates"][number]["origins"]
                    )
                ]
                record = {
                    "job_id": job_id,
                    "family_id": metadata["family_id"],
                    "noise": metadata["noise"],
                    "method": method,
                    "pool": pool,
                    "criterion": criterion,
                    "available_candidates": available,
                    "complete": bool(available),
                }
                if available:
                    objective = (
                        scores[available, 0]
                        if criterion == "task0"
                        else scores[available].mean(1)
                    )
                    winner = available[int(objective.argmin())]
                    primary_scope = (
                        "source1"
                        if (pool, criterion) == ("source0", "task0")
                        else "common4"
                        if (pool, criterion) == ("all4", "mean4")
                        else None
                    )
                    recalculated = winner
                    if primary_scope is not None:
                        winner = primary[(method, primary_scope)]["selected_candidate"]
                    if winner not in available:
                        raise RuntimeError("Selected primary I outside declared pool")
                    key = f"job{job_id}_{method}_{pool}_{criterion}"
                    selected[key] = indices[winner]
                    sorted_score = np.sort(objective)
                    record.update(
                        {
                            "selected_candidate": winner,
                            "selected_index_key": key,
                            "task_meta_mse": scores[winner].tolist(),
                            "mean_meta_mse": float(scores[winner].mean()),
                            "recomputed_argmin": recalculated,
                            "primary_reused": primary_scope,
                            "primary_argmin_recomputed_matches": winner == recalculated,
                            "observed_best_runnerup_gap": None
                            if len(sorted_score) == 1
                            else float(sorted_score[1] - sorted_score[0]),
                        }
                    )
                records.append(record)
    destination = root / "factor_ablation"
    np.savez_compressed(destination / "selected_indices.npz", **selected)
    save_json(destination / "selected.json", {"records": records})
    save_json(
        destination / "selection_frozen.json",
        {
            "utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(destination / "protocol.json"),
            "primary_selection_freeze_sha256": digest(common / "selection_frozen.json"),
            "inputs_sha256": input_hashes,
            "sha256": {
                name: digest(destination / name)
                for name in ("selected_indices.npz", "selected.json")
            },
            "diagnostic_teacher_audit_opened": False,
            "selection_rule": protocol["selection_rule"],
        },
    )


def audit_factors(root, plan):
    destination = root / "factor_ablation"
    frozen = json.loads((destination / "selection_frozen.json").read_text())
    if digest(destination / "protocol.json") != frozen["protocol_sha256"]:
        raise RuntimeError("Diagnostic selection protocol changed")
    for name, expected in frozen["sha256"].items():
        if digest(destination / name) != expected:
            raise RuntimeError("Factor choices changed after freeze")
    for name, expected in frozen["inputs_sha256"].items():
        if digest(root / name) != expected:
            raise RuntimeError("Frozen inputs changed before factor audit")
    records = json.loads((destination / "selected.json").read_text())["records"]
    truth = [teacher(plan, family)[0] for family in plan["families"]]
    canonical = [canonical_index(index, plan["codes"]) for index in truth]
    cache = {}
    for job_id in range(len(plan["families"]) * len(plan["noise"])):
        with np.load(root / "common/selected" / f"job_{job_id:03d}.npz") as arrays:
            indices = arrays["candidate_indices"]
            for record in (row for row in records if row["job_id"] == job_id):
                family = record["family_id"]
                available = record["available_candidates"]
                present = any(
                    np.array_equal(indices[n], canonical[family]) for n in available
                )
                record["bank_contains_true_I"] = present
                if record["complete"]:
                    index = indices[record["selected_candidate"]]
                    signature = (family, index.tobytes())
                    if signature not in cache:
                        cache[signature] = structural_metrics(
                            index, truth[family], plan["codes"]
                        )
                    record.update(cache[signature])
                else:
                    record.update(
                        exact_I=False, exact_mask=False, exact_unsigned_I=False
                    )
                record["selection_failure_given_truth_in_bank"] = (
                    present and not record["exact_I"]
                )
    summaries = []
    for noise, method, pool, criterion in itertools.product(
        plan["noise"], plan["methods"], ("source0", "all4"), ("task0", "mean4")
    ):
        current = [
            row
            for row in records
            if (row["noise"], row["method"], row["pool"], row["criterion"])
            == (noise, method, pool, criterion)
        ]
        summaries.append(
            {
                "noise": noise,
                "method": method,
                "pool": pool,
                "criterion": criterion,
                "geometries": len(current),
                **{
                    key: sum(row[key] for row in current)
                    for key in (
                        "exact_I",
                        "exact_mask",
                        "exact_unsigned_I",
                        "bank_contains_true_I",
                        "selection_failure_given_truth_in_bank",
                    )
                },
            }
        )
    save_json(destination / "audit.json", {"records": records, "summary": summaries})
    with (destination / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        "# G23: отдельно эффект банка и общего критерия",
        "",
        "Диагностика зарегистрирована до первичного teacher-аудита. Те же frozen I/C и meta-панели, без нового обучения или labels. Это разложение двух эффектов, не независимое подтверждение и не новый extractor.",
        "",
        "| SD | Метод | Банк | Критерий | Точный I / 6 | Истина в банке / 6 | Ошибка выбора при наличии истины |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['noise']} | {row['method']} | {row['pool']} | {row['criterion']} | {row['exact_I']} | {row['bank_contains_true_I']} | {row['selection_failure_given_truth_in_bank']} |"
        )
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = register(root)
    while not (root / "structure_report/REPORT.md").exists():
        if (
            pid_identity(protocol["primary_supervisor"]["pid"])
            != protocol["primary_supervisor"]
        ):
            raise RuntimeError("Primary supervisor terminal without its report")
        save_json(
            root / "factor_ablation/state.json",
            {
                "stage": "waiting_for_primary_report",
                "primary_supervisor": protocol["primary_supervisor"],
            },
        )
        time.sleep(protocol["poll_seconds"])
    check_code(protocol)
    if digest(root / "protocol.json") != protocol["primary_protocol_sha256"]:
        raise RuntimeError("Primary source protocol changed")
    plan = json.loads((root / "protocol.json").read_text())
    select_factors(root, plan, protocol)
    audit_factors(root, plan)
    save_json(root / "factor_ablation/state.json", {"stage": "report_ready"})
    print(str(root / "factor_ablation/REPORT.md"), flush=True)


if __name__ == "__main__":
    main()
