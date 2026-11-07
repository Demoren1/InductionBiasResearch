"""Exploratory noise-scaled ridge ablation of unknown-readout prefix search."""

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.darts_indexator_v1.data import digest, load_archive, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .pattern_prefix import timestamp
from .unknown_readout import observations


def ridge_support(matrix, y, vmatrix, vy, hidden, variance, multiplier):
    count, outputs = y.shape
    gram = matrix.T @ matrix / count
    cross = matrix.T @ y / count
    eigenvalues, vectors = np.linalg.eigh(gram)
    penalty = max(multiplier * variance / count, eigenvalues[-1] * 1e-14)
    eigenvalues = np.maximum(eigenvalues, 0) + penalty
    inverse = (vectors / eigenvalues) @ vectors.T
    coefficient = inverse @ cross
    active = np.arange(len(gram))
    records = []
    while len(active):
        if len(active) <= hidden:
            # Refit small supports to avoid accumulated Schur-complement error.
            g = gram[np.ix_(active, active)] + penalty * np.eye(len(active))
            fitted = np.linalg.solve(g, cross[active])
            records.append(
                {
                    "active": active.copy(),
                    "coefficient": fitted,
                    "train_mse": float(np.mean((matrix[:, active] @ fitted - y) ** 2)),
                    "validation_mse": float(
                        np.mean((vmatrix[:, active] @ fitted - vy) ** 2)
                    ),
                }
            )
        if len(active) == 1:
            break
        increase = np.sum(coefficient**2, axis=1) / np.maximum(
            inverse.diagonal(), 1e-30
        )
        removed = int(increase.argmin())
        keep = np.arange(len(active)) != removed
        column, diagonal = inverse[keep, removed], inverse[removed, removed]
        coefficient = (
            coefficient[keep] - column[:, None] * coefficient[removed] / diagonal
        )
        inverse = inverse[np.ix_(keep, keep)] - np.outer(column, column) / diagonal
        active = active[keep]
    records.append(
        {
            "active": np.empty(0, dtype=int),
            "coefficient": np.empty((0, outputs)),
            "train_mse": float(np.mean(y**2)),
            "validation_mse": float(np.mean(vy**2)),
        }
    )
    best = min(row["validation_mse"] for row in records)
    tolerance = 2 * variance * math.sqrt(2 / (len(vy) * outputs)) + 1e-12 * max(
        float(np.var(vy)), 1
    )
    selected = min(
        (row for row in records if row["validation_mse"] <= best + tolerance),
        key=lambda row: len(row["active"]),
    )
    return selected["active"], {
        "penalty": penalty,
        "columns": len(gram),
        "regularized_condition": float(eigenvalues[-1] / eigenvalues[0]),
        "selected_atoms": len(selected["active"]),
        "validation_mse": selected["validation_mse"],
        "path": [
            {"atoms": len(row["active"]), "validation_mse": row["validation_mse"]}
            for row in records
        ],
    }


def extract_ridge(data, code, spec, task, sigma, multiplier):
    order, hidden = data["order"], spec["hidden"]
    choices = np.r_[
        0, np.arange(1, spec["codes"] + 1), -np.arange(1, spec["codes"] + 1)
    ]
    codebook = np.r_[0, code]
    parents = np.empty((1, 0), dtype=np.int16)
    trace = []
    index = np.zeros((spec["inputs"], hidden), dtype=np.int16)
    for step in range(1, spec["inputs"] + 1):
        candidates = np.c_[
            np.repeat(parents, len(choices), axis=0), np.tile(choices, len(parents))
        ].astype(np.int16)
        candidates = np.unique(candidates, axis=0)
        candidates = candidates[np.any(candidates != 0, axis=1)]
        weight = np.sign(candidates) * codebook[np.abs(candidates)]
        block = slice((step - 1) * spec["slice_points"], step * spec["slice_points"])
        vblock = slice(
            (step - 1) * spec["prefix_validation"], step * spec["prefix_validation"]
        )
        x = data["prefix_x_0"][block][:, order[:step]]
        vx = data["prefix_vx_0"][vblock][:, order[:step]]
        matrix = (
            np.maximum(x @ weight.T + spec["bias"], 0) - max(spec["bias"], 0)
        ) / math.sqrt(hidden)
        vmatrix = (
            np.maximum(vx @ weight.T + spec["bias"], 0) - max(spec["bias"], 0)
        ) / math.sqrt(hidden)
        selected, record = ridge_support(
            matrix,
            data["prefix_y_0"][task, block],
            vmatrix,
            data["prefix_vy_0"][task, vblock],
            hidden,
            sigma**2 / spec["slice_repeats"],
            multiplier,
        )
        trace.append({"step": step, **record})
        prefixes = candidates[selected]
        index = np.zeros((spec["inputs"], hidden), dtype=np.int16)
        index[np.ix_(order[:step], np.arange(len(prefixes)))] = prefixes.T
        parents = np.unique(
            np.r_[prefixes, np.zeros((1, step), dtype=np.int16)], axis=0
        )
    return index, trace


def run_case(job):
    root, case, condition, plan, multipliers = job
    torch.set_num_threads(2)
    data = observations(root, case, condition)
    code = load_archive(
        root / "source" / f"{case['name']}_{condition}" / "bootstrap_bank.npz"
    )["axis_codes"]
    rows, indices = [], {}
    # All searches finish before any teacher structure is loaded for this case.
    for multiplier in multipliers:
        for task in range(plan["sources"]):
            index, trace = extract_ridge(
                data,
                code[task],
                {**plan, **case},
                task,
                plan["noise"][condition],
                multiplier,
            )
            key = f"ridge_{multiplier}_{task}"
            indices[key] = index
            rows.append(
                {"key": key, "multiplier": multiplier, "task": task, "trace": trace}
            )
    output = root / "ridge_ablation"
    np.savez_compressed(output / f"{case['name']}_{condition}.npz", **indices)
    teacher = load_archive(root / "private" / f"{case['name']}.npz")
    for row in rows:
        metrics, _, _ = align_indices(
            indices[row["key"]], teacher["index"], case["codes"]
        )
        row.update(
            exact=metrics["categorical_errors"] == 0,
            categorical_errors=metrics["categorical_errors"],
            mask_f1=metrics["mask_f1"],
        )
    save_json(output / f"{case['name']}_{condition}.json", {"rows": rows})
    brief = {
        str(multiplier): sum(
            row["exact"] for row in rows if row["multiplier"] == multiplier
        )
        for multiplier in multipliers
    }
    print(
        json.dumps({"case": case["name"], "condition": condition, "exact": brief}),
        flush=True,
    )
    return {"case": case["name"], "condition": condition, "exact_tasks": brief}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "protocol.json").read_text())
    multipliers = [0.1, 1.0, 10.0]
    output = root / "ridge_ablation"
    output.mkdir(exist_ok=False)
    save_json(
        output / "registration.json",
        {
            "registered_utc": timestamp(),
            "code_sha256": digest(Path(__file__)),
            "parent_source_freeze_sha256": digest(root / "source_frozen.json"),
            "design": "Post-freeze exploratory ablation, all six G23 geometries and both noises; bootstrap queries, original estimated C, unknown W2. Penalized readout objective mean squared error + lambda*sum(B^2), lambda=m*noise_variance/N; m fixed to [.1,1,10]. Same backward pruning, at-most-H and validation tolerance. No source choices change, no target codebook runs. Exact-I audit after saved indexators; report all multipliers and candidate-bank presence, no oracle selection claim.",
            "multipliers": multipliers,
        },
    )
    jobs = [
        (root, case, condition, plan, multipliers)
        for case in plan["cases"]
        for condition in range(len(plan["noise"]))
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(run_case, jobs))
    save_json(output / "summary.json", {"completed_utc": timestamp(), "rows": rows})


if __name__ == "__main__":
    main()
