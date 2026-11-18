"""Apply the frozen local SRR to retained real dense source models."""

import itertools
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.dense_recovery_probability_v1.srr_assignment_margin import (
    assignment_margin,
)
from experiments.dense_recovery_probability_v1.srr_local import (
    local_groups,
    recover_local,
    score_on_tasks,
)
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

ROOT = Path("artifacts/local/srr_real_2026-11-15")
DEST = Path("artifacts/srr_real_2026-11-15")
MULTIPLIERS = (1.5, 2.5, 4.0, 6.0)
GRID = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
PAIR_LIMIT = 1_000_000
_WEIGHTS = None
_UNCERTAINTY = None
SOURCES = {
    "omniglot": (
        Path(
            "artifacts/local/omniglot_single_task_2026-11-14/study/source/dense/models.npz"
        ),
        784,
    ),
    **{
        name: (
            Path(
                f"artifacts/local/paper_recovery_extension_2026-11-14/{name}/source/dense/models.npz"
            ),
            561 if name == "har" else 784,
        )
        for name in ("mnist", "fashion", "har")
    },
}


def descriptors(path, inputs, hidden=16):
    """Use the existing real-data direction-dispersion uncertainty rule."""
    with np.load(path) as data:
        raw = data["parameters"][:, : inputs * hidden].reshape(-1, inputs, hidden)
        ensemble = data["ensemble_parameters"][:, :, : inputs * hidden].reshape(
            len(raw), -1, inputs, hidden
        )
        ranking = data["numerical_cross_entropy"].min(0).argsort()
    normal = raw.astype(float) / np.maximum(
        np.linalg.norm(raw, axis=1, keepdims=True), 1e-12
    )
    uncertainty = []
    for reference, choices in zip(normal, ensemble):
        aligned = []
        for weight in choices:
            unit = weight / np.maximum(np.linalg.norm(weight, axis=0), 1e-12)
            left, right = linear_sum_assignment(-reference.T @ unit)
            aligned.append(unit[:, right[np.argsort(left)]])
        uncertainty.append(
            np.sqrt(np.mean((np.stack(aligned) - reference) ** 2, axis=(0, 1)))
        )
    return normal, np.maximum(uncertainty, 0.0003), ranking


def work_bound(counts):
    """Upper bound all subset comparisons in the unchanged pair matcher."""
    total = 0
    for first, second in itertools.combinations(counts, 2):
        # Vandermonde: sum_k C(a,k) C(b,k) = C(a+b,a).
        total += math.comb(int(first + second), int(first)) - 1 - first * second
        if total > PAIR_LIMIT:
            break
    return int(total)


def verify_candidate(item):
    """Verify one fixed candidate; fork workers share read-only observations."""
    codes, candidate = item
    score, checks = score_on_tasks(candidate["index"], _WEIGHTS, _UNCERTAINTY, codes)
    return {**candidate, "pooled_score": score, "checks": checks}


def run_dataset(name, path, inputs):
    global _WEIGHTS, _UNCERTAINTY
    started = time.perf_counter()
    folder = ROOT / name
    folder.mkdir(parents=True, exist_ok=True)
    weights, uncertainty, ranking = descriptors(path, inputs)
    _WEIGHTS, _UNCERTAINTY = weights, uncertainty
    np.savez_compressed(
        folder / "descriptors.npz",
        normal=weights,
        uncertainty=uncertainty,
        ranking=ranking,
    )
    budgets = (1, *(m for m in GRID if m < inputs * 16), inputs * 16)
    pool = {m: [] for m in budgets}
    diagnostics = []
    for task in ranking:
        for multiplier in MULTIPLIERS:
            local = local_groups(weights[task], uncertainty[task], multiplier, inputs)
            counts = np.bincount(local["neurons"].astype(int), minlength=16)
            minimum, maximum = int(counts.max()), int(counts.sum())
            bound = work_bound(counts)
            record = {
                "task": int(task),
                "multiplier": multiplier,
                "local_counts": counts.tolist(),
                "total_local_classes": maximum,
                "active_fraction": float(np.mean(local["memberships"] >= 0)),
                "pair_comparisons_bound": bound,
                "budgets": {},
            }
            for codes in budgets:
                if codes < minimum:
                    status = "too_many_local_groups"
                elif codes > maximum:
                    status = "too_few_total_local_groups"
                elif bound > PAIR_LIMIT:
                    status = "computational_limit"
                else:
                    candidates, trace = recover_local(
                        weights[task], uncertainty[task], codes, (multiplier,)
                    )
                    record["budgets"][str(codes)] = trace[0]
                    for candidate in candidates:
                        pool[codes].append({**candidate, "task": int(task)})
                    continue
                record["budgets"][str(codes)] = {"status": status}
            diagnostics.append(record)
    print(
        json.dumps(
            {
                "dataset": name,
                "stage": "constructed",
                "candidates": {str(m): len(rows) for m, rows in pool.items()},
            }
        ),
        flush=True,
    )
    choices, arrays = {}, {}
    for codes, candidates in pool.items():
        unique = {}
        for candidate in candidates:
            unique.setdefault(candidate["index"].tobytes(), candidate)
        with ProcessPoolExecutor(
            max_workers=8, mp_context=multiprocessing.get_context("fork")
        ) as executor:
            ranked = list(
                executor.map(
                    verify_candidate, ((codes, row) for row in unique.values())
                )
            )
        ranked.sort(key=lambda row: row["pooled_score"])
        chosen = ranked[0] if ranked else None
        row = {
            "raw_candidates": len(candidates),
            "unique_candidates": len(unique),
            "weight_accepted": bool(chosen and chosen["pooled_score"] <= 9),
            "margin_accepted": False,
        }
        if chosen is not None:
            arrays[f"m{codes}"] = chosen["index"]
            margin = assignment_margin(
                chosen["index"], weights, uncertainty, codes, radius=2.0
            )
            row.update(
                task=chosen["task"],
                multiplier=chosen["multiplier"],
                pooled_score=chosen["pooled_score"],
                own_score=chosen["score"],
                margin_accepted=bool(row["weight_accepted"] and margin["accepted"]),
                margin={
                    key: value
                    for key, value in margin.items()
                    if key != "ambiguous_mask"
                },
                checks=chosen["checks"],
            )
        choices[str(codes)] = row
        print(
            json.dumps(
                {
                    "dataset": name,
                    "stage": "verified",
                    "codes": codes,
                    "candidates": len(unique),
                    "weight_accepted": row["weight_accepted"],
                    "margin_accepted": row["margin_accepted"],
                }
            ),
            flush=True,
        )
    np.savez_compressed(folder / "candidate_bank.npz", **arrays)
    summary = {
        "dataset": name,
        "source_tasks": len(weights),
        "weights_per_task": inputs * 16,
        "uncertainty_median": float(np.median(uncertainty)),
        "typical_direction_coordinate": float(1 / np.sqrt(inputs)),
        "uncertainty_over_coordinate_scale": float(
            np.median(uncertainty) * np.sqrt(inputs)
        ),
        "budgets": list(budgets),
        "choices": choices,
        "seconds": time.perf_counter() - started,
        "descriptors_sha256": digest(folder / "descriptors.npz"),
        "candidate_bank_sha256": digest(folder / "candidate_bank.npz"),
    }
    save_json(DEST / name / "diagnostics.json", diagnostics)
    save_json(DEST / name / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        (DEST / name).mkdir(exist_ok=True)
    registration = DEST / "registered.json"
    current = {
        "protocol_sha256": digest(Path(__file__).with_name("PROTOCOL.md")),
        "extract_code_sha256": digest(Path(__file__)),
        "source_files": {name: digest(path) for name, (path, _) in SOURCES.items()},
        "multipliers": MULTIPLIERS,
        "grid": GRID,
        "pair_limit": PAIR_LIMIT,
        "test_query_labels_opened": False,
    }
    if registration.exists():
        previous = json.loads(registration.read_text())
        for key in ("protocol_sha256", "source_files", "grid", "pair_limit"):
            left = json.dumps(previous[key], sort_keys=True)
            right = json.dumps(current[key], sort_keys=True)
            if left != right:
                raise RuntimeError(f"Changed registered input: {key}")
        save_json(
            DEST / "technical_amendment.json",
            {
                "reason": "Cast empty local neuron IDs before bincount; parallelize identical source-only candidate checks after interrupting the incomplete serial calculation",
                "extract_code_sha256": digest(Path(__file__)),
                "previous_results_completed": False,
                "test_query_labels_opened": False,
            },
        )
    else:
        save_json(registration, current)
    summaries = [run_dataset(name, *settings) for name, settings in SOURCES.items()]
    save_json(DEST / "feasibility.json", summaries)


if __name__ == "__main__":
    run()
