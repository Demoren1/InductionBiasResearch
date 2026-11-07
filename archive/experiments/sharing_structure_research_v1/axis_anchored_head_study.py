"""Post-freeze intervention: retain precise axis codes during J2 discovery."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.darts_indexator_v1.data import digest, load_archive, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .indexed_readout import code_gauges, design, head_metrics, head_path
from .pattern_prefix import timestamp
from .shared_structure_numeric import canonical_index
from .two_layer_study import observed


def run_source(job):
    root, case, condition, plan = job
    torch.set_num_threads(2)
    output = root / "axis_anchor_ablation" / f"{case['name']}_{condition}"
    output.mkdir(exist_ok=False)
    source = root / "source" / output.name
    data = observed(root, case, condition)
    initial, fitted = (
        load_archive(source / "bootstrap.npz"),
        load_archive(source / "fitted.npz"),
    )
    trace = json.loads((source / "fitted.json").read_text())
    bootstrap = json.loads((source / "bootstrap.json").read_text())["rows"]
    banks = {mode: set() for mode in plan["modes"]}
    for row in bootstrap:
        if row["complete"]:
            banks[row["mode"]].add(
                canonical_index(initial[row["key"]], case["codes"]).tobytes()
            )
    grid = code_gauges(initial["axis_codes"])
    arrays, candidates = {}, []
    for detail in trace["details"]:
        number = detail["candidate"]
        index = fitted[f"i{number}_index"]
        modes = [mode for mode, bank in banks.items() if index.tobytes() in bank]
        mappings = detail["mapping"]["selected_mapping"]
        code = grid[np.arange(plan["sources"]), mappings]
        phi, vphi, mphi = (
            design(data[name + "_x"], index, code, case["bias"])
            for name in ("fit", "h1", "meta")
        )
        path, selected = head_path(phi, data["fit_y"], vphi, data["h1_y"])
        arrays[f"i{number}_index"], arrays[f"i{number}_code"] = index, code
        for variant, row in (("free", path[0]), ("indexed", selected)):
            key = f"i{number}_{variant}"
            arrays[key + "_head_index"], arrays[key + "_head"] = (
                row["index"],
                row["head"],
            )
            score = float(
                np.mean(
                    (np.einsum("tnh,tho->tno", mphi, row["head"]) - data["meta_y"]) ** 2
                )
            )
            candidates.append(
                {
                    "key": key,
                    "candidate": number,
                    "variant": variant,
                    "modes": modes,
                    "meta_mse": score,
                    "groups": row["groups"],
                    "path": [
                        {
                            "groups": p["groups"],
                            "train_mse": p["train_mse"],
                            "validation_mse": p["validation_mse"],
                        }
                        for p in path
                    ],
                }
            )
    selections = []
    for mode in plan["modes"]:
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
    np.savez_compressed(output / "fitted.npz", **arrays)
    save_json(
        output / "fitted.json",
        {
            "frozen_utc": timestamp(),
            "sha256": digest(output / "fitted.npz"),
            "selections": selections,
        },
    )
    print(json.dumps({"axis_source": case["name"], "condition": condition}), flush=True)


def audit_and_oracle(job):
    root, case, condition, plan = job
    torch.set_num_threads(2)
    folder = root / "axis_anchor_ablation" / f"{case['name']}_{condition}"
    fitted, frozen = (
        load_archive(folder / "fitted.npz"),
        json.loads((folder / "fitted.json").read_text()),
    )
    teacher = load_archive(root / "private" / f"{case['name']}.npz")
    data = observed(root, case, condition)
    rows = []
    for row in frozen["selections"]:
        result = {
            "case": case["name"],
            "kind": case["kind"],
            "style": case["style"],
            "noise": plan["noise"][condition],
            "method": row["method"],
            "abstained": row["abstained"],
        }
        if not row["abstained"]:
            index = fitted[f"i{row['candidate']}_index"]
            metric, _, columns = align_indices(index, teacher["index"], case["codes"])
            result.update(
                metric,
                exact=metric["categorical_errors"] == 0,
                head=head_metrics(
                    fitted[row["key"] + "_head_index"][columns], teacher["head_index"]
                ),
                meta_mse=row["meta_mse"],
            )
        rows.append(result)
    axis = load_archive(
        root / "source" / f"{case['name']}_{condition}" / "bootstrap.npz"
    )["axis_codes"]
    ranks = np.argsort(np.argsort(np.abs(teacher["codes"]), axis=1), axis=1)
    oracle_axis_code = np.sign(teacher["codes"]) * np.take_along_axis(
        axis, ranks, axis=1
    )
    controls = []
    for label, code in (
        ("trueI_trueC", teacher["codes"]),
        ("trueI_axisC_oracle_gauge", oracle_axis_code),
    ):
        phi, vphi = (
            design(data[name + "_x"], teacher["index"], code, case["bias"])
            for name in ("fit", "h1")
        )
        path, selected = head_path(phi, data["fit_y"], vphi, data["h1_y"])
        metric = head_metrics(selected["index"], teacher["head_index"])
        controls.append(
            {
                "case": case["name"],
                "kind": case["kind"],
                "style": case["style"],
                "noise": plan["noise"][condition],
                "control": label,
                **metric,
                "true_on_path": any(
                    head_metrics(row["index"], teacher["head_index"])["exact"]
                    for row in path
                ),
            }
        )
    save_json(folder / "audit.json", {"rows": rows, "controls": controls})
    return {"rows": rows, "controls": controls}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve() / "confirmation"
    plan = json.loads((root / "protocol.json").read_text())
    output = root / "axis_anchor_ablation"
    output.mkdir(exist_ok=False)
    save_json(
        output / "registration.json",
        {
            "registered_utc": timestamp(),
            "code_sha256": digest(Path(__file__)),
            "parent_cohort_sha256": digest(root / "cohort_frozen.json"),
            "design": "Open post-freeze intervention on all44conditions. Preserve data-derived signed/permuted axisC selected by existingH0, omit noisy Gaussian joint GN code drift, relearn nativeJ2 by unchanged functional merges andH1 paired selection. Reuse every original candidateI1, no teacher passed to learner. Freeze all new source fits before structural audit. Afterwards trueI1/trueC1 and trueI1/axisC1-with-oracle-gauge isolate head-search limitations; these are controls only. Not an independent confirmation of an adapted method.",
        },
    )
    jobs = [
        (root, case, condition, plan)
        for case in plan["cases"]
        for condition in range(len(plan["noise"]))
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(run_source, jobs))
    save_json(
        output / "cohort_frozen.json",
        {
            "frozen_utc": timestamp(),
            "sha256": {
                str(p.relative_to(output)): digest(p)
                for p in output.glob("*/fitted.json")
            },
        },
    )
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(audit_and_oracle, jobs))
    save_json(
        output / "evaluation.json",
        {
            "evaluated_utc": timestamp(),
            "rows": [row for result in results for row in result["rows"]],
            "controls": [row for result in results for row in result["controls"]],
        },
    )
    print(json.dumps({"completed_conditions": len(results)}), flush=True)


if __name__ == "__main__":
    main()
