"""Native readout sharing with task gauges synchronized before selection."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.darts_indexator_v1.data import digest, load_archive, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .indexed_readout import code_gauges, design, head_metrics
from .pattern_prefix import timestamp
from .readout_symmetry import automorphisms, synchronized_path
from .shared_structure_numeric import canonical_index
from .two_layer_study import observed


def joint_head_metric(index, head, truth_index, truth_head, codes):
    metric, aligned, columns = align_indices(index, truth_index, codes)
    possibilities = (
        automorphisms(index, codes)
        if metric["categorical_errors"] == 0
        else [{"columns": np.arange(index.shape[1])}]
    )
    best = None
    for auto in possibilities:
        candidate = head[auto["columns"][columns]]
        result = head_metrics(candidate, truth_head)
        key = (result["mask_errors"] + result["pair_errors"], result["pair_errors"])
        if best is None or key < best[0]:
            best = key, result, candidate
    return metric, aligned, best[1], best[2]


def fit_job(job):
    root, case, condition, plan = job
    torch.set_num_threads(2)
    source = root / "source" / f"{case['name']}_{condition}"
    folder = root / "synchronized_head_ablation" / source.name
    folder.mkdir(exist_ok=False)
    initial, fitted = (
        load_archive(source / "bootstrap.npz"),
        load_archive(source / "fitted.npz"),
    )
    bootstrap = json.loads((source / "bootstrap.json").read_text())["rows"]
    metadata = json.loads((source / "fitted.json").read_text())
    banks = {mode: set() for mode in plan["modes"]}
    for row in bootstrap:
        if row["complete"]:
            banks[row["mode"]].add(
                canonical_index(initial[row["key"]], case["codes"]).tobytes()
            )
    grid = code_gauges(initial["axis_codes"])
    data = observed(root, case, condition)
    arrays, candidates, details = {}, [], []
    for detail in metadata["details"]:
        number = detail["candidate"]
        index = fitted[f"i{number}_index"]
        initial_code = grid[
            np.arange(plan["sources"]), detail["mapping"]["selected_mapping"]
        ]
        states = {"axis": initial_code, "polished": fitted[f"i{number}_codes"]}
        selected, free, trace = synchronized_path(index, states, data, {**plan, **case})
        arrays[f"i{number}_index"] = index
        modes = [mode for mode, bank in banks.items() if index.tobytes() in bank]
        for variant, row in (("indexed", selected), ("free", free)):
            key = f"i{number}_{variant}"
            (
                arrays[key + "_codes"],
                arrays[key + "_head_index"],
                arrays[key + "_head"],
            ) = row["codes"], row["index"], row["head"]
            # Check the saved parameterization on the actual meta panel.
            represented = np.einsum(
                "tnh,tho->tno",
                design(data["meta_x"], index, row["codes"], case["bias"]),
                row["head"],
            )
            direct_mse = float(np.mean((represented - data["meta_y"]) ** 2))
            if not np.isclose(direct_mse, row["meta_mse"], rtol=1e-8, atol=1e-10):
                raise ValueError(
                    "Saved symmetry-transformed predictor differs from fitted function"
                )
            candidates.append(
                {
                    "candidate": number,
                    "variant": variant,
                    "key": key,
                    "modes": modes,
                    "meta_mse": row["meta_mse"],
                    "groups": row["groups"],
                    "state": row["state"],
                    "gauge": row["gauge"],
                }
            )
        details.append({"candidate": number, **trace})
    selections = []
    for mode in banks:
        for variant in ("free", "indexed"):
            eligible = [
                row
                for row in candidates
                if mode in row["modes"] and row["variant"] == variant
            ]
            selections.append(
                {
                    "method": f"{mode}_{variant}",
                    "abstained": not eligible,
                    **(
                        min(eligible, key=lambda row: row["meta_mse"])
                        if eligible
                        else {}
                    ),
                }
            )
    np.savez_compressed(folder / "fitted.npz", **arrays)
    save_json(
        folder / "fitted.json",
        {
            "frozen_utc": timestamp(),
            "sha256": digest(folder / "fitted.npz"),
            "selections": selections,
            "details": details,
        },
    )
    print(
        json.dumps({"synchronized": case["name"], "condition": condition}), flush=True
    )


def audit(root, plan):
    output = root / "synchronized_head_ablation"
    paths = list(output.glob("*/fitted.json"))
    if len(paths) != len(plan["cases"]) * len(plan["noise"]):
        raise ValueError("Need all source fits before audit")
    save_json(
        output / "cohort_frozen.json",
        {
            "frozen_utc": timestamp(),
            "sha256": {str(p.relative_to(output)): digest(p) for p in paths},
        },
    )
    rows, aligned = [], {}
    for case in plan["cases"]:
        teacher = load_archive(root / "private" / f"{case['name']}.npz")
        for condition, sigma in enumerate(plan["noise"]):
            folder = output / f"{case['name']}_{condition}"
            fitted, metadata = (
                load_archive(folder / "fitted.npz"),
                json.loads((folder / "fitted.json").read_text()),
            )
            for row in metadata["selections"]:
                result = {
                    "case": case["name"],
                    "kind": case["kind"],
                    "style": case["style"],
                    "noise": sigma,
                    "method": row["method"],
                    "abstained": row["abstained"],
                }
                if not row["abstained"]:
                    metric, matrix, head, j = joint_head_metric(
                        fitted[f"i{row['candidate']}_index"],
                        fitted[row["key"] + "_head_index"],
                        teacher["index"],
                        teacher["head_index"],
                        case["codes"],
                    )
                    result.update(
                        metric,
                        exact=metric["categorical_errors"] == 0,
                        head=head,
                        groups=row["groups"],
                        meta_mse=row["meta_mse"],
                        state=row["state"],
                        gauge=row["gauge"],
                    )
                    key = f"{case['name']}_{condition}_{row['method']}"
                    aligned[key + "_I"], aligned[key + "_J"] = matrix, j
                rows.append(result)
    np.savez_compressed(output / "aligned_indices.npz", **aligned)
    save_json(output / "evaluation.json", {"evaluated_utc": timestamp(), "rows": rows})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve() / "confirmation"
    plan = json.loads((root / "protocol.json").read_text())
    output = root / "synchronized_head_ablation"
    output.mkdir(exist_ok=False)
    save_json(
        output / "registration.json",
        {
            "registered_utc": timestamp(),
            "parent_freeze_sha256": digest(root / "cohort_frozen.json"),
            "code_sha256": {
                p.name: digest(p)
                for p in [
                    Path(__file__),
                    Path(__file__).with_name("readout_symmetry.py"),
                ]
            },
            "design": "Post-freeze exploratory intervention on all44conditions. Preserve all original candidateI1. Per candidate, enumerate only its signed code/hidden-column automorphisms, task0 fixes global gauge. Joint H1 selects nativeJ2 and remaining task gauges, with data-axis and Gaussian-polished C states; at most4096 gauge combinations. No roles/teacherJ/K2 passed. Whole-cohort freeze before joint symmetry-aware structural audit. New independent cohort required for adapted-method confirmation.",
        },
    )
    jobs = [
        (root, case, condition, plan)
        for case in plan["cases"]
        for condition in range(len(plan["noise"]))
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(fit_job, jobs))
    audit(root, plan)


if __name__ == "__main__":
    main()
