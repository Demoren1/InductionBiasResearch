"""Replay source-only projection states to measure near-zero calibration."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.paper_structure_benchmark_v1.reporting.evaluate import (
    write_csv,
)
from experiments.sharing_structure_research_v1.blind_contract import save_json
from experiments.sharing_structure_research_v1.blind_joint_bank import (
    align_source,
    projection_score,
)

from . import bank, relative_bank


def run(base, wait):
    if wait:
        while not (base / "source_plan.finished.json").exists():
            time.sleep(5)
    state = json.loads(
        (base / "source_plan.finished.json").read_text()
    )
    if state["completed"] != state["total"] or state["failures"]:
        raise RuntimeError("Source comparison is not complete")
    protocol = json.loads((base / "guard/protocol.json").read_text())
    rows = []
    for method in ("guard", "irls", "relative"):
        module = relative_bank if method == "relative" else bank
        root = base / method
        for sizes in protocol["cases"]:
            case, codes = sizes["name"], sizes["codes"]
            for condition in protocol["conditions"]:
                key = f"{case}_c{condition}"
                folder = root / "fits" / key
                fitted = dict(np.load(folder / "fitted.npz"))
                candidates = dict(np.load(folder / "joint_bank.npz"))
                selected = np.load(
                    folder / "joint_selected.npz"
                )["joint_functional_index"]
                origins = json.loads(
                    (folder / "joint_bank.json").read_text()
                )["origins"]
                traces = json.loads(
                    (folder / "optimization.json").read_text()
                )["runs"]
                source = [
                    dict(np.load(
                        root / "source" / f"{key}_t{t}" / "learned.npz"
                    ))
                    for t in range(protocol["sources"])
                ]
                selected_ids = [
                    i for i, index in enumerate(candidates["indices"])
                    if np.array_equal(index, selected)
                ]
                ids = [0]
                if selected_ids and selected_ids[0] != 0:
                    ids.append(selected_ids[0])
                for rank in ids:
                    trace = traces[origins[rank]["trace"]]
                    anchor = fitted["indices"][trace["anchor"]]
                    mappings = [
                        align_source(
                            anchor,
                            fitted["parameters"][trace["anchor"], t, :codes],
                            observed["normal"], observed["uncertainty"],
                        )
                        for t, observed in enumerate(source)
                    ]
                    normal = np.stack([
                        observed["normal"][:, mapping]
                        for observed, mapping in zip(source, mappings)
                    ])
                    uncertainty = np.stack([
                        observed["uncertainty"][mapping]
                        for observed, mapping in zip(source, mappings)
                    ])
                    votes = trace["votes"]
                    evidence = np.abs(normal) > np.maximum(
                        3 * uncertainty[:, None], 0.002
                    )
                    mask = (
                        anchor != 0 if votes == 0
                        else evidence.sum(0) >= votes
                    )
                    seeds = [anchor]
                    for observed, mapping in zip(source, mappings):
                        for name in [
                            k for k in observed if k.startswith("index_")
                        ][:2]:
                            seeds.append(observed[name][:, mapping])
                    if trace["seed"] >= len(seeds):
                        raise RuntimeError("Replay needs a random seed")
                    result = module.joint_assign(
                        seeds[trace["seed"]], mask, normal, uncertainty,
                        codes, robust=method != "guard",
                    )
                    index, code, scales, score, _ = result
                    rows.append({
                        "case": case, "condition": condition,
                        "method": method, "geometry_rank": rank + 1,
                        "selected_by_functional_meta": (
                            bool(selected_ids) and rank == selected_ids[0]
                        ),
                        "selected_present_in_geometry_bank": bool(
                            selected_ids
                        ),
                        "projection_q": score,
                        "score_replay_error": abs(
                            score - float(candidates["score"][rank])
                        ),
                        "same_state_floor_q": float(bank.objective(
                            index, code, scales, normal, uncertainty
                        ).mean()),
                        "same_state_original_ranking_q": projection_score(
                            index, code, scales, normal, uncertainty
                        ),
                        "minimum_abs_scale": float(
                            np.min(np.abs(scales))
                        ),
                        "fraction_scales_below_0_01": float(
                            np.mean(np.abs(scales) < 0.01)
                        ),
                        "fraction_scales_below_1e_6": float(
                            np.mean(np.abs(scales) < 1e-6)
                        ),
                        "median_code_norm": float(np.median(
                            np.linalg.norm(code, axis=1)
                        )),
                    })
    output = base / "report"
    output.mkdir(exist_ok=True)
    write_csv(output / "calibration_states.csv", rows)
    save_json(output / "calibration_states.json", {
        "teacher_or_target_answer_access": False,
        "rows": rows,
    })
    print(json.dumps({
        "replayed": len(rows),
        "largest_score_replay_error": max(
            row["score_replay_error"] for row in rows
        ),
    }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    run(args.base.resolve(), args.wait)


if __name__ == "__main__":
    main()
