"""Independent output-query gate for accepting additional readout equalities."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.darts_indexator_v1.data import digest, load_archive, save_json

from .indexed_readout import design
from .pattern_prefix import timestamp
from .shared_structure_numeric import canonical_index
from .synchronized_head_study import joint_head_metric
from .two_layer_study import observed
from .unknown_readout import teacher_output


def register(folder, plan):
    output = folder / "head_gate"
    output.mkdir(exist_ok=False)
    for name in ("data", "predictors", "decisions"):
        (output / name).mkdir()
    started = bool(list((folder / "source").glob("*/bootstrap.json")))
    if plan["stage"].startswith("G32") and started:
        raise ValueError(
            "Independent-cohort gate must be registered before learner starts"
        )
    save_json(
        output / "registration.json",
        {
            "registered_utc": timestamp(),
            "code_sha256": digest(Path(__file__)),
            "parent_protocol_sha256": digest(folder / "protocol.json"),
            "primary": "keep2_gated",
            "before_any_new_cohort_bootstrap": not started,
            "gate_points": 4096,
            "gate_repeats": 64,
            "standard_errors": 2,
            "design": "Additional independent Gaussian output queries. Keep synchronized native J2 only when its loss improves over an independently fitted free head at the same I1/C1 by more than two paired standard errors. Otherwise return independent head. Evaluate every candidate I1; only afterwards meta chooses I1. All gate predictors freeze before opening gate labels; all decisions freeze before teacher structure audit. Gate never fits numerical parameters. No teacher groups/counts/roles passed. G31 is exploratory; G32 amendment is registered before any learner run and fixes the final algorithm without HPO on G32.",
        },
    )
    for number, case in enumerate(plan["cases"]):
        rng = np.random.default_rng(plan["seed"] + 60000037 + 100003 * number)
        teacher = load_archive(folder / "private" / f"{case['name']}.npz")
        x = rng.normal(size=(4096, case["inputs"]))
        clean = teacher_output(
            x,
            teacher["index"],
            teacher["codes"],
            teacher["readout"],
            teacher["offset"],
            case["bias"],
        )
        noise = rng.normal(size=clean.shape)
        arrays = {"x": x}
        for condition, sigma in enumerate(plan["noise"]):
            arrays[f"y_{condition}"] = clean + sigma * noise / 8
        np.savez_compressed(output / "data" / f"{case['name']}.npz", **arrays)
    save_json(
        output / "data_frozen.json",
        {
            "frozen_utc": timestamp(),
            "sha256": {p.name: digest(p) for p in (output / "data").glob("*.npz")},
        },
    )


def predictors(folder, case, condition):
    source = folder / "source" / f"{case['name']}_{condition}"
    synchronized = folder / "synchronized_head_ablation" / source.name
    fitted = load_archive(synchronized / "fitted.npz")
    bootstrap = json.loads((source / "bootstrap.json").read_text())["rows"]
    raw = load_archive(source / "bootstrap.npz")
    banks = {mode: set() for mode in ("old", "keep2")}
    for row in bootstrap:
        if row["complete"]:
            banks[row["mode"]].add(
                canonical_index(raw[row["key"]], case["codes"]).tobytes()
            )
    data = observed(folder, case, condition)
    arrays, rows = {}, []
    keys = sorted(
        key
        for key in fitted
        if key.startswith("i") and key.endswith("_index") and "head" not in key
    )
    for key in keys:
        prefix = key.removesuffix("_index")
        index = fitted[key]
        code = fitted[prefix + "_indexed_codes"]
        phi = design(data["fit_x"], index, code, case["bias"])
        free = np.stack(
            [
                np.linalg.lstsq(phi[t], data["fit_y"][t], rcond=1e-12)[0]
                for t in range(len(code))
            ]
        )
        arrays[prefix + "_index"], arrays[prefix + "_codes"] = index, code
        arrays[prefix + "_head_index"] = fitted[prefix + "_indexed_head_index"]
        arrays[prefix + "_head"] = fitted[prefix + "_indexed_head"]
        arrays[prefix + "_free_head"] = free
        rows.append(
            {
                "key": prefix,
                "modes": [
                    mode for mode, bank in banks.items() if index.tobytes() in bank
                ],
            }
        )
    output = folder / "head_gate/predictors" / source.name
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    save_json(
        output.with_suffix(".json"),
        {
            "frozen_utc": timestamp(),
            "sha256": digest(output.with_suffix(".npz")),
            "rows": rows,
        },
    )


def evaluate(folder, plan):
    output = folder / "head_gate"
    expected = len(plan["cases"]) * len(plan["noise"])
    for name in ("source", "synchronized_head_ablation"):
        if len(list((folder / name).glob("*/fitted.json"))) != expected:
            raise ValueError(
                "Need all baseline and synchronized predictors before the gate"
            )
    for case in plan["cases"]:
        for condition in range(len(plan["noise"])):
            predictors(folder, case, condition)
    paths = list((output / "predictors").glob("*.json"))
    save_json(
        output / "predictors_frozen.json",
        {"frozen_utc": timestamp(), "sha256": {p.name: digest(p) for p in paths}},
    )
    decisions = []
    for case in plan["cases"]:
        gate = load_archive(output / "data" / f"{case['name']}.npz")
        for condition, sigma in enumerate(plan["noise"]):
            name = f"{case['name']}_{condition}"
            source_labels = load_archive(folder / "data" / f"{case['name']}.npz")
            y = gate[f"y_{condition}"] - source_labels[f"zero_{condition}"][:, None]
            data = observed(folder, case, condition)
            fitted = load_archive(output / "predictors" / f"{name}.npz")
            records = json.loads((output / "predictors" / f"{name}.json").read_text())[
                "rows"
            ]
            arrays, candidates = {}, []
            for row in records:
                key = row["key"]
                index, code, j = (
                    fitted[key + "_index"],
                    fitted[key + "_codes"],
                    fitted[key + "_head_index"],
                )
                phi = design(gate["x"], index, code, case["bias"])
                shared = np.einsum("tnh,tho->tno", phi, fitted[key + "_head"])
                free = np.einsum("tnh,tho->tno", phi, fitted[key + "_free_head"])
                delta = (((free - y) ** 2 - (shared - y) ** 2).mean(axis=2)).ravel()
                gain, se = (
                    float(delta.mean()),
                    float(np.std(delta, ddof=1) / math.sqrt(len(delta))),
                )
                groups = len(np.unique(np.abs(j[j != 0])))
                accepted = groups < j.size and gain > 2 * se + 1e-12 * max(
                    float(np.var(y)), 1
                )
                chosen = fitted[key + ("_head" if accepted else "_free_head")]
                chosen_j = (
                    j
                    if accepted
                    else np.arange(1, j.size + 1, dtype=np.int16).reshape(j.shape)
                )
                meta = np.einsum(
                    "tnh,tho->tno",
                    design(data["meta_x"], index, code, case["bias"]),
                    chosen,
                )
                meta_mse = float(np.mean((meta - data["meta_y"]) ** 2))
                arrays[key + "_index"], arrays[key + "_codes"] = index, code
                arrays[key + "_head_index"], arrays[key + "_head"] = chosen_j, chosen
                candidates.append(
                    {
                        **row,
                        "accepted": accepted,
                        "groups": groups if accepted else j.size,
                        "gate_gain": gain,
                        "gate_standard_error": se,
                        "meta_mse": meta_mse,
                    }
                )
            selected = []
            for mode in ("old", "keep2"):
                eligible = [row for row in candidates if mode in row["modes"]]
                selected.append(
                    {
                        "method": mode + "_gated",
                        "abstained": not eligible,
                        **(
                            min(eligible, key=lambda row: row["meta_mse"])
                            if eligible
                            else {}
                        ),
                    }
                )
            np.savez_compressed(output / "decisions" / f"{name}.npz", **arrays)
            save_json(
                output / "decisions" / f"{name}.json",
                {
                    "case": case["name"],
                    "noise": sigma,
                    "selections": selected,
                    "sha256": digest(output / "decisions" / f"{name}.npz"),
                },
            )
            decisions.extend(selected)
    paths = list((output / "decisions").glob("*.json"))
    save_json(
        output / "decisions_frozen.json",
        {"frozen_utc": timestamp(), "sha256": {p.name: digest(p) for p in paths}},
    )
    rows, aligned = [], {}
    for case in plan["cases"]:
        teacher = load_archive(folder / "private" / f"{case['name']}.npz")
        for condition, sigma in enumerate(plan["noise"]):
            name = f"{case['name']}_{condition}"
            values = load_archive(output / "decisions" / f"{name}.npz")
            metadata = json.loads((output / "decisions" / f"{name}.json").read_text())
            for selection in metadata["selections"]:
                row = {
                    "case": case["name"],
                    "kind": case["kind"],
                    "style": case["style"],
                    "noise": sigma,
                    **selection,
                }
                if not selection["abstained"]:
                    key = selection["key"]
                    metric, matrix, head, j = joint_head_metric(
                        values[key + "_index"],
                        values[key + "_head_index"],
                        teacher["index"],
                        teacher["head_index"],
                        case["codes"],
                    )
                    row.update(
                        metric, exact=metric["categorical_errors"] == 0, head=head
                    )
                    (
                        aligned[f"{name}_{row['method']}_I"],
                        aligned[f"{name}_{row['method']}_J"],
                    ) = matrix, j
                rows.append(row)
    np.savez_compressed(output / "aligned_indices.npz", **aligned)
    save_json(output / "evaluation.json", {"evaluated_utc": timestamp(), "rows": rows})
    print(
        {"evaluated_conditions": expected, "gate_candidates": len(decisions)},
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("register", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    folder = args.root.resolve() / "confirmation"
    plan = json.loads((folder / "protocol.json").read_text())
    if args.action == "register":
        register(folder, plan)
    else:
        evaluate(folder, plan)


if __name__ == "__main__":
    main()
