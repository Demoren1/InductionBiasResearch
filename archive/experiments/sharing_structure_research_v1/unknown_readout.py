"""G29: output-only structure search with unknown freely fitted W2."""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch

from experiments.darts_indexator_v1.data import digest, load_archive, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .joint_readout_control import feature, head_groups
from .pattern_prefix import prefix_inputs, timestamp
from .shared_structure_numeric import canonical_index
from .weighted_prefix_search import estimate_codes, extract, fit_numeric, values

BASE = Path(__file__).resolve().parent


def teacher_output(x, index, codes, readout, offset, bias):
    codebook = np.c_[np.zeros(len(codes)), codes]
    weight = np.sign(index)[None] * codebook[:, np.abs(index)]
    activation = np.maximum(np.einsum("nd,tdh->tnh", x, weight) + bias, 0)
    return (
        np.einsum("tnh,tho->tno", activation, readout) / math.sqrt(index.shape[1])
        + offset[:, None]
    )


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    cases = []
    for number in range(2):
        cases.append(
            {
                "name": f"pattern_{number}",
                "kind": "pattern",
                "family": number,
                "inputs": 11,
                "hidden": 8,
                "codes": 2,
                "outputs": 1,
                "bias": -3.0,
            }
        )
    for number in range(4):
        cases.append(
            {
                "name": f"sets_{number}",
                "kind": "sets",
                "family": number,
                "inputs": 16,
                "hidden": 16,
                "codes": 4,
                "outputs": 2,
                "bias": 0.15,
            }
        )
    plan = {
        "stage": "G29 unknown-readout hard weighted-prefix extraction",
        "registered_utc": timestamp(),
        "seed": 26110629017,
        "cases": cases,
        "sources": 4,
        "noise": [0.0, 0.2],
        "axis_points": 257,
        "axis_min": 0.001,
        "axis_max": 128.0,
        "axis_repeats": 4194304,
        "slice_repeats": 4096,
        "slice_points": 2048,
        "prefix_validation": 512,
        "slice_scales": [0.05, 0.2, 0.8, 3.2, 12.8, 51.2],
        "fit_points": 4096,
        "fit_validation": 1024,
        "meta_points": 2048,
        "fit_repeats": 64,
        "numeric_starts": 64,
        "numeric_steps": 64,
        "target_functions": 8,
        "target_test": 8192,
        "primary": "calibrated_weighted_prefix",
        "software_tests": False,
        "known": "D,H,K,output dimension, ReLU, common nonzero hidden bias, active output queries. No I/C/W2/point roles/edge count supplied to learner. Zero-output anchor observed.",
        "axis": "Separate signed coordinate curves, variable-projection LS of signed amplitudes and bounded nonlinear LS of shared positive code magnitudes. Seven starts, independent interleaved-axis validation; no counts/positive-amplitude assumption.",
        "prefix": "All discrete child-prefixes plus persistent all-zero parent. Fit arbitrary signed vector readout. Backward deletion uses exact functional LS loss increase, not coefficient norm. At most H nonzero atoms, smallest support within validation noise/numerical tolerance. No weight clustering or softmax argmax. Generic noncancelled prefix responses and sufficient query rank required.",
        "calibration": "Bootstrap per-source candidates; canonicalize/deduplicate; fresh joint C/W2 fits for every task; select preliminary I on H0. Rerun prefixes from empty with calibrated C, readout profiled freely at each step. Final fresh joint fits use H1; meta-labels opened only after predictor freeze.",
        "source_teacher": "Pattern0110: eight sliding four-bit detectors, shuffled hidden columns, free signed per-neuron readout. Two numerical/source-query cases of one geometry. DeepSets: four new hidden input/column permutations, eight 2D points/two shared ReLU channels; unknown signed2x2 output matrix after pooling. Learner uses unconstrained2x16 W2. Source code magnitudes separated by >=.08 for Pattern/.10 for DeepSets, min .15 for DeepSets; a disclosed identifiable pilot.",
        "synthetic_followup": "After Pattern/DeepSets, repeat on the exact six G23 comparator geometries (local/quartet_mix/random), with unknown readout. Separate results; no equal-budget old-table claim.",
        "resource_policy": "CPU2threads for axis/prefix; batch GPU0-3 only for numerical C/W2 fits. Never GPU6/7; no foreign process termination.",
    }
    (root / "private").mkdir()
    (root / "data").mkdir()
    (root / "source").mkdir()
    (root / "code_snapshot").mkdir()
    for name in ["unknown_readout.py", "weighted_prefix_search.py"]:
        shutil.copy2(BASE / name, root / "code_snapshot" / name)
    plan["code_sha256"] = {
        name: digest(BASE / name)
        for name in ["unknown_readout.py", "weighted_prefix_search.py"]
    }
    save_json(root / "protocol.json", plan)
    for case_id, case in enumerate(cases):
        prepare_case(root, plan, case, case_id)
    save_json(
        root / "data_frozen.json",
        {
            "frozen_utc": timestamp(),
            "sha256": {p.name: digest(p) for p in (root / "data").glob("*.npz")},
            "private_sha256": {
                p.name: digest(p) for p in (root / "private").glob("*.npz")
            },
        },
    )
    policy = json.loads((BASE / "GPU_POLICY.json").read_text())
    policy["active_continuation"] = str(root.relative_to(BASE) / "protocol.json")
    policy["current_execution"] = (
        "G29 unknown-W2 full structure search: Pattern/DeepSets, followed by the requested old synthetic comparator geometries. CPU prefix; packed numeric GPU0-3 only."
    )
    save_json(BASE / "GPU_POLICY.json", policy)
    print(json.dumps({"prepared_cases": len(cases), "utc": timestamp()}), flush=True)


def prepare_case(root, plan, case, case_id):
    rng = np.random.default_rng(plan["seed"] + 100003 * case_id)
    dimensions, hidden, codes, outputs = [
        case[k] for k in ("inputs", "hidden", "codes", "outputs")
    ]
    index = np.zeros((dimensions, hidden), dtype=np.int16)
    if case["kind"] == "pattern":
        for h in range(hidden):
            index[h : h + 4, h] = [-1, 2, 2, -1]
        permutation = np.arange(dimensions)
    else:
        permutation = rng.permutation(dimensions)
        for point in range(8):
            index[
                np.ix_(
                    permutation[2 * point : 2 * point + 2], [2 * point, 2 * point + 1]
                )
            ] = [[1, 2], [3, 4]]
    index = index[:, rng.permutation(hidden)]
    code = []
    for _ in range(4):
        while True:
            candidate = (
                rng.uniform(0.8, 1.3, codes)
                if case["kind"] == "pattern"
                else rng.normal(size=codes)
            )
            sorted_abs = np.sort(np.abs(candidate))
            if sorted_abs[0] >= 0.15 and np.min(np.diff(sorted_abs)) >= (
                0.08 if codes == 2 else 0.10
            ):
                break
        code.append(candidate)
    code = np.array(code)
    if case["kind"] == "pattern":
        readout = rng.uniform(0.5, 1.5, (4, hidden, outputs)) * rng.choice(
            (-1, 1), (4, hidden, outputs)
        )
    else:
        pooled = rng.uniform(0.4, 1.5, (4, 2, outputs)) * rng.choice(
            (-1, 1), (4, 2, outputs)
        )
        readout = pooled[:, head_groups(index)]
    offset = rng.normal(0, 0.3, (4, outputs))
    np.savez_compressed(
        root / "private" / f"{case['name']}.npz",
        index=index,
        codes=code,
        readout=readout,
        offset=offset,
        permutation=permutation,
    )
    order = rng.permutation(dimensions)
    positive = np.geomspace(plan["axis_min"], plan["axis_max"], plan["axis_points"])
    vpositive = np.sqrt(positive[:-1] * positive[1:])
    s, vs = np.r_[-positive[::-1], 0, positive], np.r_[-vpositive[::-1], 0, vpositive]
    panels = {}
    for name, grid in [("axis", s), ("axis_v", vs)]:
        x = np.zeros((dimensions, len(grid), dimensions))
        for d in range(dimensions):
            x[d, :, d] = grid
        panels[name] = x.reshape(-1, dimensions)
    spec = {**plan, **case}
    for phase in range(2):
        panels[f"prefix_{phase}"] = prefix_inputs(
            rng, order, plan["slice_points"], spec
        )
        panels[f"prefix_v_{phase}"] = prefix_inputs(
            rng, order, plan["prefix_validation"], spec
        )
    for name, number in [
        ("fit", plan["fit_points"]),
        ("h0", plan["fit_validation"]),
        ("h1", plan["fit_validation"]),
        ("meta", plan["meta_points"]),
    ]:
        panels[name] = rng.normal(size=(number, dimensions))
    arrays = {"order": order, "axis_s": s, "axis_vs": vs}
    zero = teacher_output(
        np.zeros((1, dimensions)), index, code, readout, offset, case["bias"]
    )[:, 0]
    zero_noise = rng.normal(size=zero.shape)
    for condition, sigma in enumerate(plan["noise"]):
        arrays[f"zero_y_{condition}"] = zero + sigma * zero_noise / math.sqrt(
            2 * dimensions * plan["axis_repeats"]
        )
    for name, x in panels.items():
        arrays[name + "_x"] = x
        y = teacher_output(x, index, code, readout, offset, case["bias"])
        noise = rng.normal(size=y.shape)
        repeats = (
            plan["axis_repeats"]
            if name.startswith("axis")
            else plan["slice_repeats"]
            if name.startswith("prefix")
            else plan["fit_repeats"]
        )
        for condition, sigma in enumerate(plan["noise"]):
            arrays[f"{name}_y_{condition}"] = y + sigma * noise / math.sqrt(repeats)
    np.savez_compressed(root / "data" / f"{case['name']}.npz", **arrays)


def observations(root, case, condition):
    raw = load_archive(root / "data" / f"{case['name']}.npz")
    zero = raw[f"zero_y_{condition}"]
    data = {
        key: value
        for key, value in raw.items()
        if key.endswith("_x") or key in ("order", "axis_s", "axis_vs")
    }
    for name in [
        "axis",
        "axis_v",
        "prefix_0",
        "prefix_v_0",
        "prefix_1",
        "prefix_v_1",
        "fit",
        "h0",
        "h1",
        "meta",
    ]:
        data[name + "_y"] = raw[f"{name}_y_{condition}"] - zero[:, None]
    for phase in range(2):
        data[f"prefix_x_{phase}"] = data[f"prefix_{phase}_x"]
        data[f"prefix_vx_{phase}"] = data[f"prefix_v_{phase}_x"]
        data[f"prefix_y_{phase}"] = data[f"prefix_{phase}_y"]
        data[f"prefix_vy_{phase}"] = data[f"prefix_v_{phase}_y"]
    return data


def bootstrap(root, plan, case, condition):
    output = root / "source" / f"{case['name']}_{condition}"
    output.mkdir(exist_ok=False)
    data = observations(root, case, condition)
    spec = {**plan, **case}
    axis, traces, indices = [], [], []
    outputs, dimensions = case["outputs"], case["inputs"]
    for task in range(4):
        y = data["axis_y"][task].reshape(dimensions, -1, outputs).transpose(1, 0, 2)
        vy = data["axis_v_y"][task].reshape(dimensions, -1, outputs).transpose(1, 0, 2)
        code, axis_trace = estimate_codes(
            data["axis_s"],
            y,
            data["axis_vs"],
            vy,
            spec,
            plan["seed"] + condition * 73 + task,
        )
        index, trace = extract(data, code, spec, task, 0, plan["noise"][condition])
        axis.append(code)
        traces.append({"task": task, "axis": axis_trace, "search": trace})
        if trace["complete"]:
            indices.append(canonical_index(index, case["codes"]))
        print(
            json.dumps(
                {
                    "case": case["name"],
                    "condition": condition,
                    "task": task,
                    "phase": "bootstrap",
                    "complete": trace["complete"],
                    "code": code.tolist(),
                    "seconds": trace["seconds"],
                }
            ),
            flush=True,
        )
    unique = {index.tobytes(): index for index in indices}
    bank = (
        np.stack(list(unique.values()))
        if unique
        else np.empty((0, case["inputs"], case["hidden"]), dtype=np.int16)
    )
    np.savez_compressed(
        output / "bootstrap_bank.npz", indices=bank, axis_codes=np.stack(axis)
    )
    save_json(
        output / "bootstrap_trace.json", {"tasks": traces, "frozen_utc": timestamp()}
    )


def fit_bank(root, plan, case, condition, phase, device):
    output = root / "source" / f"{case['name']}_{condition}"
    bank_name = "bootstrap_bank.npz" if phase == 0 else "repeat_bank.npz"
    bank = load_archive(output / bank_name)["indices"]
    if not len(bank):
        save_json(
            output / f"fit_{phase}.json", {"abstained": True, "frozen_utc": timestamp()}
        )
        return
    data = observations(root, case, condition)
    spec = {**plan, **case}
    parameters, scores, histories = [], [], {}
    for candidate, index in enumerate(bank):
        fitted, trace, history = fit_numeric(
            index,
            data["fit_x"],
            data["fit_y"],
            data[f"h{phase}_x"],
            data[f"h{phase}_y"],
            spec,
            plan["seed"] + 1009 * candidate,
            device,
        )
        parameters.append(fitted)
        scores.append(trace["validation_mse"])
        histories.update({f"candidate_{candidate}_{k}": v for k, v in history.items()})
        print(
            json.dumps(
                {
                    "case": case["name"],
                    "condition": condition,
                    "phase": phase,
                    "candidate": candidate,
                    "fit": trace,
                }
            ),
            flush=True,
        )
    scores = np.array(scores)
    provisional = int(scores.mean(1).argmin())
    np.savez_compressed(
        output / f"fits_{phase}.npz",
        indices=bank,
        parameters=np.stack(parameters),
        validation=scores,
        **histories,
    )
    save_json(
        output / f"fit_{phase}.json",
        {
            "abstained": False,
            "provisional": provisional,
            "frozen_utc": timestamp(),
            "sha256": digest(output / f"fits_{phase}.npz"),
        },
    )


def repeat(root, plan, case, condition):
    output = root / "source" / f"{case['name']}_{condition}"
    frozen = json.loads((output / "fit_0.json").read_text())
    if frozen["abstained"]:
        np.savez_compressed(
            output / "repeat_bank.npz",
            indices=np.empty((0, case["inputs"], case["hidden"]), dtype=np.int16),
        )
        return
    fitted = load_archive(output / "fits_0.npz")
    initial = fitted["parameters"][frozen["provisional"], :, : case["codes"]]
    data = observations(root, case, condition)
    indices, traces = [], []
    for task in range(4):
        index, trace = extract(
            data, initial[task], {**plan, **case}, task, 1, plan["noise"][condition]
        )
        traces.append(trace)
        if trace["complete"]:
            indices.append(canonical_index(index, case["codes"]))
        print(
            json.dumps(
                {
                    "case": case["name"],
                    "condition": condition,
                    "phase": "repeat",
                    "task": task,
                    "complete": trace["complete"],
                    "seconds": trace["seconds"],
                }
            ),
            flush=True,
        )
    unique = {index.tobytes(): index for index in indices}
    bank = (
        np.stack(list(unique.values()))
        if unique
        else np.empty((0, case["inputs"], case["hidden"]), dtype=np.int16)
    )
    np.savez_compressed(output / "repeat_bank.npz", indices=bank)
    save_json(
        output / "repeat_trace.json", {"tasks": traces, "frozen_utc": timestamp()}
    )


@torch.no_grad()
def select(root, plan, case, condition, device):
    output = root / "source" / f"{case['name']}_{condition}"
    frozen = json.loads((output / "fit_1.json").read_text())
    if frozen["abstained"]:
        save_json(
            output / "selection.json", {"abstained": True, "frozen_utc": timestamp()}
        )
        return
    fitted = load_archive(output / "fits_1.npz")
    data = observations(root, case, condition)
    scores = []
    for index, parameters in zip(fitted["indices"], fitted["parameters"], strict=True):
        feat = feature(data["meta_x"], index, case["codes"], device, torch.float64)
        p = torch.tensor(parameters, device=device)
        y = torch.tensor(data["meta_y"], device=device)
        prediction = values(feat, p, {**plan, **case}, case["outputs"])
        scores.append(float((prediction - y).square().mean()))
    chosen = int(np.argmin(scores))
    np.savez_compressed(
        output / "selected.npz",
        index=fitted["indices"][chosen],
        parameters=fitted["parameters"][chosen],
    )
    save_json(
        output / "selection.json",
        {
            "abstained": False,
            "candidate": chosen,
            "meta_mse": scores[chosen],
            "scores": scores,
            "frozen_utc": timestamp(),
            "sha256": digest(output / "selected.npz"),
        },
    )


def audit(root, plan):
    paths = list((root / "source").glob("*/selection.json"))
    expected = len(plan["cases"]) * len(plan["noise"])
    if len(paths) != expected:
        raise ValueError(f"Need all {expected} selections before structural audit")
    save_json(
        root / "source_frozen.json",
        {
            "frozen_utc": timestamp(),
            "sha256": {str(p.relative_to(root)): digest(p) for p in paths},
        },
    )
    rows, aligned = [], {}
    for case in plan["cases"]:
        teacher = load_archive(root / "private" / f"{case['name']}.npz")
        for condition, noise in enumerate(plan["noise"]):
            folder = root / "source" / f"{case['name']}_{condition}"
            selection = json.loads((folder / "selection.json").read_text())
            row = {
                "case": case["name"],
                "kind": case["kind"],
                "noise": noise,
                "abstained": selection["abstained"],
            }
            if not selection["abstained"]:
                index = load_archive(folder / "selected.npz")["index"]
                metric, matrix, _ = align_indices(
                    index, teacher["index"], case["codes"]
                )
                row.update(
                    metric,
                    exact=metric["categorical_errors"] == 0,
                    meta_mse=selection["meta_mse"],
                )
                aligned[case["name"] + "_" + str(condition)] = matrix
            rows.append(row)
    np.savez_compressed(root / "aligned_indices.npz", **aligned)
    save_json(
        root / "structure_audit.json", {"evaluated_utc": timestamp(), "rows": rows}
    )
    print(json.dumps(rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "bootstrap",
            "fit_bootstrap",
            "repeat",
            "fit_final",
            "select",
            "audit",
        ],
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.root = args.root.resolve()
    torch.set_num_threads(2)
    if args.action == "prepare":
        prepare(args.root)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    if args.action == "audit":
        audit(args.root, plan)
        return
    case = plan["cases"][args.case]
    if args.action == "bootstrap":
        bootstrap(args.root, plan, case, args.condition)
    elif args.action == "fit_bootstrap":
        fit_bank(args.root, plan, case, args.condition, 0, args.device)
    elif args.action == "repeat":
        repeat(args.root, plan, case, args.condition)
    elif args.action == "fit_final":
        fit_bank(args.root, plan, case, args.condition, 1, args.device)
    else:
        select(args.root, plan, case, args.condition, args.device)


if __name__ == "__main__":
    main()
