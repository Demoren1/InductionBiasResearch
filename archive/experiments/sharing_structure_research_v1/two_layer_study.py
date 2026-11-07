"""Frozen generalization cohort for guarded prefixes and indexed readouts."""

import argparse
import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.darts_indexator_v1.data import digest, load_archive, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .guarded_prefix import extract
from .indexed_readout import (
    design,
    head_metrics,
    head_path,
    polish_codes,
    profile_codes,
)
from .pattern_prefix import prefix_inputs, timestamp
from .shared_structure_data import teacher as synthetic_teacher
from .shared_structure_numeric import canonical_index
from .unknown_readout import observations, teacher_output
from .unknown_readout_synthetic import wide_axis

BASE = Path(__file__).resolve().parent
OLD = BASE / "results/20261106/unknown_readout_synthetic_g30"


def prepare(root):
    folder = root / "confirmation"
    folder.mkdir(exist_ok=False)
    for name in ("panels", "data", "private", "source", "code_snapshot"):
        (folder / name).mkdir()
    plan = json.loads((OLD / "protocol.json").read_text())
    cases = []
    for i, variant in enumerate(
        ("local", "local", "quartet_mix", "quartet_mix", "random", "random")
    ):
        geometry = {
            "name": f"new_{i}",
            "kind": "synthetic",
            "family": 31901 + i,
            "variant": variant,
            "inputs": 16,
            "hidden": 16,
            "codes": 4,
            "outputs": 1 + i % 2,
            "bias": 0.15,
        }
        for style in ("free", "tied2", "tied4"):
            cases.append(
                {
                    **geometry,
                    "geometry": geometry["name"],
                    "name": f"new_{i}_{style}",
                    "style": style,
                }
            )
    for kind, dimensions, hidden, codes, outputs, bias in (
        ("pattern", 11, 8, 2, 1, -3.0),
        ("sets", 16, 16, 4, 2, 0.15),
    ):
        for style in ("free", f"tied{codes}"):
            cases.append(
                {
                    "name": f"{kind}_{style}",
                    "geometry": kind,
                    "kind": kind,
                    "family": 0,
                    "inputs": dimensions,
                    "hidden": hidden,
                    "codes": codes,
                    "outputs": outputs,
                    "bias": bias,
                    "style": style,
                }
            )
    plan.update(
        stage="G31 independent generalization: both layer indexators",
        seed=26110631037,
        registered_utc=timestamp(),
        cases=cases,
        modes=["old", "keep2"],
        primary="keep2_indexed",
        variants=["free", "indexed", "forced4"],
        numeric_polish_steps=16,
        known="D/H/K1/O/common nonzero b, output queries and noise variance. K2, teacherI1/J2, roles and edge counts hidden. Free readout (one code per edge) remains an eligible J2.",
        design="Six new geometries from all three families, scalar/vector outputs balanced; free/tied2/tied4 heads on each. Additional new Pattern0110 and DeepSets geometries, free/tied heads. Source C1 never rejected for separation. Four functions/case, paired noise .2/2, exact-I metrics after whole cohort freeze. No target-codebook runs.",
        search="Freeze keep2 after exploratory G30 replay: 2H intermediate candidates, H final, noise-scaled ridge. Old selection replay is the paired baseline. Single prefix pass, identical panels and axis estimation. No teacher audit in source search.",
        numeric="For each canonical I, enumerate signed/permuted data-derived axis C1; fit arbitrary W2 by linear LS; H0 picks mapping. Sixteen joint C1/W2 GN steps, H0 checkpoint. Native signed J2 starts with all independent edges, exact functional-cost group merges/zero removals; H1 paired two-SE selects complexity including free fallback. Independent meta selects common I/J.",
        pilot_root=str(root),
        pilot_summary_sha256=digest(root / "pilot_summary.json"),
        software_tests=False,
    )
    files = ("guarded_prefix.py", "indexed_readout.py", "two_layer_study.py")
    plan["code_sha256"] = {name: digest(BASE / name) for name in files}
    for name in files:
        shutil.copy2(BASE / name, folder / "code_snapshot" / name)
    save_json(folder / "protocol.json", plan)
    geometry_data = {}
    for number, case in enumerate(cases):
        key = case["geometry"]
        if key not in geometry_data:
            geometry_data[key] = make_geometry(folder, plan, case, len(geometry_data))
        make_case(folder, plan, case, number, geometry_data[key])
    save_json(
        folder / "data_frozen.json",
        {
            "frozen_utc": timestamp(),
            "sha256": {
                str(p.relative_to(folder)): digest(p)
                for name in ("panels", "data", "private")
                for p in (folder / name).glob("*.npz")
            },
        },
    )
    policy = json.loads((BASE / "GPU_POLICY.json").read_text())
    policy["active_continuation"] = str((folder / "protocol.json").relative_to(BASE))
    policy["current_execution"] = (
        "G31: paired prefix diagnostics, independent free/tied readout cohort; CPU prefix and packed numeric GPU0-3 only. Never6/7."
    )
    save_json(BASE / "GPU_POLICY.json", policy)
    print(
        json.dumps(
            {"prepared_cases": len(cases), "new_geometries": len(geometry_data)}
        ),
        flush=True,
    )


def make_geometry(folder, plan, case, number):
    rng = np.random.default_rng(plan["seed"] + 100003 * number)
    dimensions, hidden, codes = case["inputs"], case["hidden"], case["codes"]
    if case["kind"] == "synthetic":
        index, code = synthetic_teacher({**plan, **case}, case)
    else:
        index = np.zeros((dimensions, hidden), dtype=np.int16)
        if case["kind"] == "pattern":
            for h in range(hidden):
                index[h : h + 4, h] = [-1, 2, 2, -1]
            code = rng.uniform(0.8, 1.3, (4, codes))
        else:
            order = rng.permutation(dimensions)
            for h in range(8):
                index[np.ix_(order[2 * h : 2 * h + 2], [2 * h, 2 * h + 1])] = [
                    [1, 2],
                    [3, 4],
                ]
            code = rng.normal(size=(4, codes))
        index = index[:, rng.permutation(hidden)]
    order = rng.permutation(dimensions)
    positive = np.geomspace(plan["axis_min"], plan["axis_max"], plan["axis_points"])
    vpositive = np.sqrt(positive[:-1] * positive[1:])
    s, vs = np.r_[-positive[::-1], 0, positive], np.r_[-vpositive[::-1], 0, vpositive]
    arrays = {"order": order, "axis_s": s, "axis_vs": vs}
    for name, grid in (("axis", s), ("axis_v", vs)):
        x = np.zeros((dimensions, len(grid), dimensions))
        for d in range(dimensions):
            x[d, :, d] = grid
        arrays[name + "_x"] = x.reshape(-1, dimensions)
    arrays["prefix_x_0"] = prefix_inputs(
        rng, order, plan["slice_points"], {**plan, **case}
    )
    arrays["prefix_vx_0"] = prefix_inputs(
        rng, order, plan["prefix_validation"], {**plan, **case}
    )
    for name, count in (
        ("fit", plan["fit_points"]),
        ("h0", plan["fit_validation"]),
        ("h1", plan["fit_validation"]),
        ("meta", plan["meta_points"]),
    ):
        arrays[name + "_x"] = rng.normal(size=(count, dimensions))
    np.savez_compressed(folder / "panels" / f"{case['geometry']}.npz", **arrays)
    return index, code, arrays


def make_case(folder, plan, case, number, geometry):
    index, code, panels = geometry
    rng = np.random.default_rng(plan["seed"] + 7000003 + 100019 * number)
    hidden, outputs = case["hidden"], case["outputs"]
    if case["style"] == "free":
        head_index = np.arange(1, hidden * outputs + 1, dtype=np.int16).reshape(
            hidden, outputs
        )
        head_code = rng.uniform(0.5, 1.5, (4, hidden * outputs)) * rng.choice(
            (-1, 1), (4, hidden * outputs)
        )
    else:
        count = int(case["style"].removeprefix("tied"))
        labels = np.resize(np.arange(1, count + 1), hidden * outputs)
        rng.shuffle(labels)
        head_index = (
            (labels * rng.choice((-1, 1), len(labels)))
            .reshape(hidden, outputs)
            .astype(np.int16)
        )
        head_code = rng.uniform(0.4, 1.6, (4, count))
    readout = (
        np.sign(head_index)[None] * np.c_[np.zeros(4), head_code][:, np.abs(head_index)]
    )
    offset = rng.normal(0, 0.3, (4, outputs))
    np.savez_compressed(
        folder / "private" / f"{case['name']}.npz",
        index=index,
        codes=code,
        readout=readout,
        head_index=head_index,
        head_codes=head_code,
        offset=offset,
    )
    zero = teacher_output(
        np.zeros((1, case["inputs"])), index, code, readout, offset, case["bias"]
    )[:, 0]
    zero_noise = rng.normal(size=zero.shape)
    arrays = {}
    for condition, sigma in enumerate(plan["noise"]):
        arrays[f"zero_{condition}"] = zero + sigma * zero_noise / math.sqrt(
            2 * case["inputs"] * plan["axis_repeats"]
        )
    for name in ("axis", "axis_v", "prefix", "prefix_v", "fit", "h0", "h1", "meta"):
        xname = {"prefix": "prefix_x_0", "prefix_v": "prefix_vx_0"}.get(
            name, name + "_x"
        )
        y = teacher_output(panels[xname], index, code, readout, offset, case["bias"])
        noise = rng.normal(size=y.shape)
        repeats = (
            plan["axis_repeats"]
            if name.startswith("axis")
            else plan["slice_repeats"]
            if name.startswith("prefix")
            else plan["fit_repeats"]
        )
        for condition, sigma in enumerate(plan["noise"]):
            arrays[f"{name}_{condition}"] = y + sigma * noise / math.sqrt(repeats)
    np.savez_compressed(folder / "data" / f"{case['name']}.npz", **arrays)


def observed(folder, case, condition):
    panels = load_archive(folder / "panels" / f"{case['geometry']}.npz")
    labels = load_archive(folder / "data" / f"{case['name']}.npz")
    data = dict(panels)
    zero = labels[f"zero_{condition}"][:, None]
    for name in ("axis", "axis_v", "fit", "h0", "h1", "meta"):
        data[name + "_y"] = labels[f"{name}_{condition}"] - zero
    data["prefix_y_0"] = labels[f"prefix_{condition}"] - zero
    data["prefix_vy_0"] = labels[f"prefix_v_{condition}"] - zero
    return data


def bootstrap_job(job):
    folder, case, condition, plan = job
    torch.set_num_threads(2)
    output = folder / "source" / f"{case['name']}_{condition}"
    output.mkdir(exist_ok=False)
    data = observed(folder, case, condition)
    arrays, traces, codes = {}, [], []
    for task in range(plan["sources"]):
        y = (
            data["axis_y"][task]
            .reshape(case["inputs"], -1, case["outputs"])
            .transpose(1, 0, 2)
        )
        vy = (
            data["axis_v_y"][task]
            .reshape(case["inputs"], -1, case["outputs"])
            .transpose(1, 0, 2)
        )
        code, axis_trace = wide_axis(
            data["axis_s"],
            y,
            data["axis_vs"],
            vy,
            {**plan, **case},
            plan["seed"] + task,
        )
        codes.append(code)
        for mode in plan["modes"]:
            index, trace, prefixes = extract(
                data, code, {**plan, **case}, task, 0, plan["noise"][condition], mode
            )
            key = f"{mode}_{task}"
            arrays[key] = index
            for step, value in enumerate(prefixes, 1):
                arrays[f"{key}_prefix_{step}"] = value
            traces.append({"key": key, "task": task, "axis": axis_trace, **trace})
    np.savez_compressed(output / "bootstrap.npz", axis_codes=np.array(codes), **arrays)
    save_json(
        output / "bootstrap.json",
        {
            "frozen_utc": timestamp(),
            "sha256": digest(output / "bootstrap.npz"),
            "rows": traces,
        },
    )
    print(json.dumps({"bootstrap": case["name"], "condition": condition}), flush=True)


def fit_case(folder, case, condition, plan, device, pilot=False):
    if pilot:
        data = observations(OLD, case, condition)
        raw = load_archive(folder / "pilot" / f"{case['name']}_{condition}.npz")
        axis = load_archive(
            OLD / "source" / f"{case['name']}_{condition}" / "bootstrap_bank.npz"
        )["axis_codes"]
        records = json.loads(
            (folder / "pilot" / f"{case['name']}_{condition}.json").read_text()
        )["rows"]
        output = folder / "pilot_fits" / f"{case['name']}_{condition}"
    else:
        output = folder / "source" / f"{case['name']}_{condition}"
        data = observed(folder, case, condition)
        raw = load_archive(output / "bootstrap.npz")
        axis = raw["axis_codes"]
        records = json.loads((output / "bootstrap.json").read_text())["rows"]
    output.mkdir(exist_ok=True)
    banks = {mode: {} for mode in ("old", "keep2")}
    for record in records:
        mode = record["mode"]
        if mode not in banks or not record["complete"]:
            continue
        index = canonical_index(raw[record["key"]], case["codes"])
        banks[mode][index.tobytes()] = index
    combined = {key: value for bank in banks.values() for key, value in bank.items()}
    arrays, rows, details = {}, [], []
    for number, (key, index) in enumerate(combined.items()):
        spec = {**plan, **case}
        code, head, trace = profile_codes(index, axis, data, spec, device)
        code, head, history = polish_codes(index, code, head, data, spec, device)
        phi, vphi, mphi = (
            design(data[name + "_x"], index, code, case["bias"])
            for name in ("fit", "h1", "meta")
        )
        path, selected = head_path(phi, data["fit_y"], vphi, data["h1_y"])
        choices = {
            "free": path[0],
            "indexed": selected,
            "forced4": next((row for row in path if row["groups"] <= 4), path[-1]),
        }
        arrays[f"i{number}_index"] = index
        arrays[f"i{number}_codes"] = code
        arrays[f"i{number}_grid_train"] = trace.pop("grid_train")
        arrays[f"i{number}_grid_validation"] = trace.pop("grid_validation")
        for variant, fitted in choices.items():
            prefix = f"i{number}_{variant}"
            arrays[prefix + "_head_index"] = fitted["index"]
            arrays[prefix + "_head"] = fitted["head"]
            arrays[prefix + "_head_codes"] = fitted["centers"]
            error = float(
                np.mean(
                    (np.einsum("tnh,tho->tno", mphi, fitted["head"]) - data["meta_y"])
                    ** 2
                )
            )
            rows.append(
                {
                    "candidate": number,
                    "variant": variant,
                    "modes": [mode for mode, bank in banks.items() if key in bank],
                    "meta_mse": error,
                    "h1_mse": fitted["validation_mse"],
                    "head_groups": fitted["groups"],
                }
            )
        details.append(
            {
                "candidate": number,
                "mapping": trace,
                "polish": history,
                "head_path": [
                    {
                        "groups": row["groups"],
                        "train_mse": row["train_mse"],
                        "validation_mse": row["validation_mse"],
                    }
                    for row in path
                ],
            }
        )
    selections = []
    for mode in banks:
        for variant in ("free", "indexed", "forced4"):
            eligible = [
                row
                for row in rows
                if row["variant"] == variant and mode in row["modes"]
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
            "details": details,
        },
    )
    print(
        json.dumps(
            {
                "fit": case["name"],
                "condition": condition,
                "candidates": len(combined),
                "selections": selections,
            }
        ),
        flush=True,
    )


def audit(folder, plan, pilot=False):
    source = folder / ("pilot_fits" if pilot else "source")
    paths = [
        source / f"{case['name']}_{condition}" / "fitted.json"
        for case in plan["cases"]
        for condition in range(len(plan["noise"]))
    ]
    if not all(p.exists() for p in paths):
        raise ValueError("Need whole fitted cohort before structural audit")
    save_json(
        folder / ("pilot_frozen.json" if pilot else "cohort_frozen.json"),
        {
            "frozen_utc": timestamp(),
            "sha256": {str(p.relative_to(folder)): digest(p) for p in paths},
        },
    )
    rows, aligned = [], {}
    for case in plan["cases"]:
        teacher = load_archive(
            (OLD if pilot else folder) / "private" / f"{case['name']}.npz"
        )
        truth_head = teacher.get(
            "head_index",
            np.arange(1, case["hidden"] * case["outputs"] + 1).reshape(
                case["hidden"], case["outputs"]
            ),
        )
        for condition, sigma in enumerate(plan["noise"]):
            path = source / f"{case['name']}_{condition}"
            frozen, fitted = (
                json.loads((path / "fitted.json").read_text()),
                load_archive(path / "fitted.npz"),
            )
            if digest(path / "fitted.npz") != frozen["sha256"]:
                raise ValueError("Fitted cohort changed")
            for row in frozen["selections"]:
                result = {
                    "case": case["name"],
                    "geometry": case.get("geometry", case["name"]),
                    "kind": case["kind"],
                    "style": case.get("style", "free"),
                    "noise": sigma,
                    "method": row["method"],
                    "abstained": row["abstained"],
                }
                if not row["abstained"]:
                    prefix = f"i{row['candidate']}"
                    metric, matrix, columns = align_indices(
                        fitted[prefix + "_index"], teacher["index"], case["codes"]
                    )
                    j = fitted[prefix + "_" + row["variant"] + "_head_index"][columns]
                    result.update(
                        metric,
                        exact=metric["categorical_errors"] == 0,
                        head=head_metrics(j, truth_head),
                        meta_mse=row["meta_mse"],
                    )
                    aligned[f"{case['name']}_{condition}_{row['method']}_I"] = matrix
                    aligned[f"{case['name']}_{condition}_{row['method']}_J"] = j
                rows.append(result)
    name = "pilot_structure_audit" if pilot else "structure_audit"
    np.savez_compressed(folder / f"{name}.npz", **aligned)
    save_json(folder / f"{name}.json", {"evaluated_utc": timestamp(), "rows": rows})
    print(json.dumps({"audited_rows": len(rows)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("prepare", "bootstrap", "fit", "audit", "pilot_fit", "pilot_audit"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root)
        return
    pilot = args.action.startswith("pilot_")
    folder = root if pilot else root / "confirmation"
    plan = json.loads(
        (root / "protocol.json" if pilot else folder / "protocol.json").read_text()
    )
    if args.action in ("audit", "pilot_audit"):
        audit(folder, plan, pilot)
    elif args.action == "bootstrap":
        jobs = [
            (folder, case, condition, plan)
            for case in plan["cases"]
            for condition in range(len(plan["noise"]))
        ]
        with ProcessPoolExecutor(max_workers=4) as pool:
            list(pool.map(bootstrap_job, jobs))
    else:
        fit_case(
            folder, plan["cases"][args.case], args.condition, plan, args.device, pilot
        )


if __name__ == "__main__":
    main()
