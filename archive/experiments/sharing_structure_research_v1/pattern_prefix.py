"""Pattern 0110: output-only prefix extraction followed by fresh classification."""

import argparse
import itertools
import json
import math
import shutil
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares, nnls

from experiments.darts_indexator_v1 import fit as dense_fit
from experiments.darts_indexator_v1 import single
from experiments.darts_indexator_v1.data import (
    digest,
    load_archive,
    observations,
    save_json,
)
from experiments.darts_indexator_v1.model import metrics
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .codebook_counts import integer_segments
from .count_exchanges import search_counts
from .count_projection import census
from .count_qp_active import function_qp
from .joint_prefix_search import census as calibrated_census
from .shared_structure_numeric import canonical_index, code_prediction, features
from .shared_structure_numeric_batch import fit_codes

BASE = Path(__file__).resolve().parent
METHODS = ("bootstrap", "single_fixed", "joint_fixed")


def timestamp():
    return datetime.now(UTC).isoformat()


def register(root):
    parent = BASE / "results/20261105/joint_prefix_confirmation_g25/protocol.json"
    previous = json.loads(parent.read_text())
    keys = [
        "axis_points",
        "axis_min",
        "axis_max",
        "axis_repeats",
        "slice_repeats",
        "axis_code_min",
        "axis_code_max",
        "axis_starts",
        "axis_solver_evaluations",
        "dp_chunk",
        "slice_points",
        "prefix_validation",
        "slice_scales",
        "rank_relative",
        "qp_gap_absolute",
        "qp_primal_tolerance",
        "active_iterations",
        "active_zero_tolerance",
        "active_direction_tolerance",
        "integer_moves",
        "integer_decrease",
        "exchange_iterations",
        "composition_chunk",
        "code_refit_steps",
        "code_ridge_relative",
        "code_line_search",
        "numeric_chunk",
        "numeric_code_steps",
        "source_polish_steps",
        "numeric_ridge",
        "numeric_line_search",
    ]
    plan = {key: previous[key] for key in keys}
    plan.update(
        stage="G26 Pattern 0110, real-valued teacher outputs then binary labels",
        registered_utc=timestamp(),
        seed=26110526017,
        inputs=11,
        hidden=8,
        codes=2,
        sources=4,
        bias=-3.0,
        noise=[0.0, 0.2],
        search_runs=4,
        fit_points=4096,
        fit_validation=2048,
        meta_points=2048,
        numerical_starts=32,
        fit_repeats=64,
        support=768,
        query=256,
        selection=512,
        classification_restarts=32,
        classification_spec={
            "steps": 4096,
            "batch": 128,
            "numeric_l2": 0.001,
            "gradient_clip": 0.5,
            "checkpoint_every": 64,
            "paired_initialization_steps": 64,
        },
        classification_lr=0.03,
        parent_protocol_sha256=digest(parent),
        methods=list(METHODS),
        primary="single_fixed",
        oracle_interface=(
            "Real-valued sum of eight ReLUs, common known bias -3 and known "
            "readout 1/sqrt(8). Queries may be real and off the binary cube. "
            "Teacher I/C, activations, gradients, window positions and edge "
            "counts are absent from learner inputs."
        ),
        teacher_design=(
            "One fixed 0110 geometry; four numerical C sampled uniformly "
            "in [0.8,1.3]^2. Same function class and shared I. Hidden columns "
            "are shuffled. This is not reconstruction from binary labels."
        ),
        changes_from_g25=(
            "D/H/K and bias changed for Pattern. Axis estimator uses abs(b) "
            "and baseline 2D*sqrt(H)*max(b,0); starts generalized from K=4 "
            "to K=2. Fresh-C calibration uses 4096 standard Gaussian inputs "
            "rather than the old combined axis/prefix training panel. "
            "Count/QP/integer solvers unchanged. No HPO or teacher selection."
        ),
        interpretation=(
            "Four runs randomize learner coordinate order and query panels, "
            "not the underlying geometry. Binary evaluation is only 0110; "
            "it does not establish transfer to other patterns."
        ),
        classification_policy=(
            "Freeze all source I before teacher audit or binary fits. Reset "
            "C/b/a/d and optimizer, train only on binary support; query BCE "
            "selects checkpoint/start. Freeze every predictor before test. "
            "Identical canonical I shares one numerical fit, with provenance."
        ),
        resource_policy="CPU prefix/QP, one available GPU for batched C fits; never GPU6/7",
        software_tests=False,
    )
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root / "code_snapshot"
    snapshot.mkdir()
    paths = [
        Path(__file__),
        BASE / "codebook_counts.py",
        BASE / "prefix_census.py",
        BASE / "count_projection.py",
        BASE / "count_qp_active.py",
        BASE / "count_exchanges.py",
        BASE / "joint_prefix_search.py",
        BASE / "shared_structure_numeric.py",
        BASE / "shared_structure_numeric_batch.py",
        Path(single.__file__),
        Path(dense_fit.__file__),
    ]
    plan["code_sha256"] = {str(path): digest(path) for path in paths}
    for path in paths:
        name = path.name if path.parent == BASE else f"pattern_{path.name}"
        shutil.copy2(path, snapshot / name)
    save_json(root / "protocol.json", plan)
    return plan


def real_output(x, index, codebooks, bias):
    values = np.c_[np.zeros(len(codebooks)), codebooks]
    weight = values[:, np.abs(index)] * np.sign(index)
    activation = np.einsum("nd,tdh->tnh", x, weight) + bias
    return np.maximum(activation, 0).sum(-1) / math.sqrt(index.shape[1])


def axis_inputs(t, dimensions):
    result = np.zeros((2, dimensions, len(t), dimensions))
    for coordinate in range(dimensions):
        result[0, coordinate, :, coordinate] = t
        result[1, coordinate, :, coordinate] = -t
    return result.reshape(-1, dimensions)


def prefix_inputs(rng, order, count, plan):
    blocks = []
    for length in range(1, plan["inputs"] + 1):
        x = np.zeros((count, plan["inputs"]))
        scale = rng.choice(plan["slice_scales"], size=count)
        x[:, order[:length]] = rng.normal(size=(count, length)) * scale[:, None]
        blocks.append(x)
    return np.concatenate(blocks)


def prepare(root, plan):
    rng = np.random.default_rng(plan["seed"])
    index = np.zeros((11, 8), dtype=np.int16)
    for window in range(8):
        index[window : window + 4, window] = [-1, 2, 2, -1]
    index = index[:, rng.permutation(8)]
    codebooks = rng.uniform(0.8, 1.3, size=(plan["sources"], plan["codes"]))
    private = root / "private"
    private.mkdir()
    np.savez_compressed(private / "teacher.npz", index=index, codebooks=codebooks)
    public = root / "data"
    public.mkdir()
    for condition, sigma in enumerate(plan["noise"]):
        crng = np.random.default_rng(plan["seed"] + 100001 * condition + 41)
        t = np.geomspace(plan["axis_min"], plan["axis_max"], plan["axis_points"])
        vt = np.sqrt(t[:-1] * t[1:])
        x, vx = axis_inputs(t, 11), axis_inputs(vt, 11)
        y = real_output(x, index, codebooks, plan["bias"])
        vy = real_output(vx, index, codebooks, plan["bias"])
        y += crng.normal(size=y.shape) * sigma / math.sqrt(plan["axis_repeats"])
        vy += crng.normal(size=vy.shape) * sigma / math.sqrt(plan["axis_repeats"])
        np.savez_compressed(public / f"axis_{condition}.npz", t=t, vt=vt, y=y, vy=vy)
        for run in range(plan["search_runs"]):
            rrng = np.random.default_rng(plan["seed"] + 1009 * run + 701)
            order = rrng.permutation(11)
            arrays = {"order": order}
            for phase in ("boot", "repeat"):
                for kind, count in (
                    ("train", plan["slice_points"]),
                    ("validation", plan["prefix_validation"]),
                ):
                    x = prefix_inputs(rrng, order, count, plan)
                    y = real_output(x, index, codebooks, plan["bias"])
                    y += (
                        crng.normal(size=y.shape)
                        * sigma
                        / math.sqrt(plan["slice_repeats"])
                    )
                    arrays[f"{phase}_{kind}_x"] = x
                    arrays[f"{phase}_{kind}_y"] = y
            for kind, count in (
                ("fit_train", plan["fit_points"]),
                ("fit_validation", plan["fit_validation"]),
                ("final_validation", plan["fit_validation"]),
                ("meta", plan["meta_points"]),
            ):
                x = rrng.normal(size=(count, 11))
                y = real_output(x, index, codebooks, plan["bias"])
                y += crng.normal(size=y.shape) * sigma / math.sqrt(plan["fit_repeats"])
                if kind == "meta":
                    np.savez_compressed(
                        public / f"meta_{condition}_{run}.npz", x=x, y=y
                    )
                else:
                    arrays[f"{kind}_x"] = x
                    arrays[f"{kind}_y"] = y
            np.savez_compressed(public / f"source_{condition}_{run}.npz", **arrays)
    binary = observations(["0110"], plan["seed"] + 900001, plan)
    np.savez_compressed(
        public / "binary_train.npz",
        **{key: value for key, value in binary.items() if not key.startswith("test_")},
    )
    np.savez_compressed(
        public / "binary_test.npz",
        **{key: value for key, value in binary.items() if key.startswith("test_")},
    )
    save_json(
        root / "data_frozen.json",
        {
            "frozen_utc": timestamp(),
            "sha256": {str(p.relative_to(root)): digest(p) for p in public.iterdir()},
            "private_teacher_sha256": digest(private / "teacher.npz"),
        },
    )


def estimate_axis(data, task, plan):
    # The even-axis identity uses |b| for both signs of the common bias.
    axis_plan = {**plan, "bias": abs(plan["bias"])}
    t, vt = data["t"], data["vt"]
    offset = 2 * plan["inputs"] * math.sqrt(plan["hidden"]) * max(plan["bias"], 0)
    y = data["y"][task].reshape(2, 11, len(t)).sum((0, 1)) - offset
    vy = data["vy"][task].reshape(2, 11, len(vt)).sum((0, 1)) - offset
    initial, _, dp = integer_segments(t, y, axis_plan, "cpu")
    rng = np.random.default_rng(plan["seed"] + 1307 * task)
    starts = [initial, np.geomspace(0.15, 2.5, 2), np.geomspace(0.3, 1.7, 2)]
    while len(starts) < plan["axis_starts"]:
        starts.append(np.exp(rng.uniform(np.log(0.03), np.log(4), 2)))

    def residual(log_codes, return_amplitude=False):
        code = np.sort(np.exp(log_codes))
        matrix = np.maximum(t[:, None] * code - abs(plan["bias"]), 0)
        scale = np.linalg.norm(matrix, axis=0).clip(1e-30)
        amplitude = nnls(matrix / scale, y, maxiter=500)[0] / scale
        value = matrix @ amplitude - y
        return (value, amplitude) if return_amplitude else value

    attempts = []
    for start in starts:
        fitted = least_squares(
            residual,
            np.log(np.clip(start, plan["axis_code_min"], plan["axis_code_max"])),
            bounds=(math.log(plan["axis_code_min"]), math.log(plan["axis_code_max"])),
            max_nfev=plan["axis_solver_evaluations"],
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        code = np.sort(np.exp(fitted.x))
        error, amplitude = residual(fitted.x, True)
        prediction = np.maximum(vt[:, None] * code - abs(plan["bias"]), 0) @ amplitude
        attempts.append(
            {
                "code": code.tolist(),
                "amplitude": amplitude.tolist(),
                "train_mse": float(np.mean(error**2)),
                "validation_mse": float(np.mean((prediction - vy) ** 2)),
                "evaluations": fitted.nfev,
            }
        )
    selected = int(np.argmin([a["validation_mse"] for a in attempts]))
    return np.array(attempts[selected]["code"]), {
        "selected": selected,
        "attempts": attempts,
        "dp": dp,
    }


def numeric_initial(plan, run, device):
    rng = np.random.default_rng(plan["seed"] + 8011 * run + 400001)
    signs = np.repeat(np.array(list(itertools.product((-1, 1), repeat=2))), 8, axis=0)
    values = signs[None] * np.exp(rng.normal(-0.3, 0.7, size=(4, 32, 2)))
    return torch.tensor(values, dtype=torch.float64, device=device)


def fit_bank(indices, data, validation, plan, run, device):
    def tensor(key):
        return torch.tensor(data[key], dtype=torch.float64, device=device)

    return fit_codes(
        np.stack(indices),
        tensor("fit_train_x"),
        tensor("fit_train_y"),
        tensor(validation + "_x"),
        tensor(validation + "_y"),
        numeric_initial(plan, run, device),
        plan,
    )


def learn_case(root, plan, condition, run, axis_codes, device):
    destination = root / "source" / f"case_{condition}_{run}"
    destination.mkdir(parents=True, exist_ok=False)
    data = load_archive(root / "data" / f"source_{condition}_{run}.npz")
    boot_bank, boot_seen, records = [], set(), []
    for task in range(plan["sources"]):
        for setting in ("unit", "unit_refit", "parent_pair", "parent_pair_refit"):
            result = census(
                list(torch.tensor(data["boot_train_x"]).reshape(11, -1, 11)),
                torch.tensor(data["boot_train_y"][task]).reshape(11, -1),
                list(torch.tensor(data["boot_validation_x"]).reshape(11, -1, 11)),
                torch.tensor(data["boot_validation_y"][task]).reshape(11, -1),
                data["order"],
                axis_codes[task],
                "qp_integer_refit" if setting.endswith("refit") else "qp_integer",
                plan,
                qp_solver=function_qp,
                integer_solver=partial(search_counts, kind="parent_pair", device="cpu")
                if setting.startswith("parent")
                else None,
            )
            name = f"boot_{task}_{setting}"
            np.savez_compressed(
                destination / f"{name}.npz",
                indices=result["indices"],
                codebooks=result["codebooks"],
            )
            records.append(
                {"name": name, "complete": result["complete"], "trace": result["trace"]}
            )
            if result["complete"]:
                index = canonical_index(result["indices"][-1], plan["codes"])
                if index.tobytes() not in boot_seen:
                    boot_seen.add(index.tobytes())
                    boot_bank.append(index)
    save_json(destination / "bootstrap_search.json", records)
    if not boot_bank:
        result = {
            "condition": condition,
            "run": run,
            "abstained": True,
            "reason": "no_bootstrap_candidate",
        }
        save_json(destination / "selection.json", result)
        return result
    code, histories, _ = fit_bank(boot_bank, data, "fit_validation", plan, run, device)
    scores = np.array([[h["validation_mse"] for h in row] for row in histories]).mean(1)
    provisional = int(scores.argmin())
    bootstrap = boot_bank[provisional]
    calibrated = code[provisional].cpu().numpy()
    np.savez_compressed(
        destination / "calibration.npz",
        indices=np.stack(boot_bank),
        codes=code.cpu().numpy(),
    )
    save_json(
        destination / "calibration.json",
        {"selected": provisional, "histories": histories, "frozen_utc": timestamp()},
    )

    repeat_data = {
        "order": data["order"],
        "search_train_x": data["repeat_train_x"],
        "search_train_y": data["repeat_train_y"],
        "prefix_validation_x": data["repeat_validation_x"],
        "prefix_validation_y": data["repeat_validation_y"],
    }
    banks = {"bootstrap": [bootstrap], "single_fixed": [], "joint_fixed": []}
    repeat_records = []
    for method, tasks in [("single_fixed", [t]) for t in range(4)] + [
        ("joint_fixed", list(range(4)))
    ]:
        result = calibrated_census(
            repeat_data, calibrated, tasks, False, "unit", plan, "cpu"
        )
        name = f"{method}_{tasks[0]}"
        np.savez_compressed(
            destination / f"{name}.npz",
            indices=result["indices"],
            codebooks=result["codebooks"],
        )
        repeat_records.append(
            {"name": name, "complete": result["complete"], "trace": result["trace"]}
        )
        if result["complete"]:
            index = canonical_index(result["indices"][-1], 2)
            if not any(np.array_equal(index, other) for other in banks[method]):
                banks[method].append(index)
    save_json(destination / "repeat_search.json", repeat_records)
    all_indices, signatures, method_ids = [], {}, {}
    for method, bank in banks.items():
        method_ids[method] = []
        for index in bank:
            key = index.tobytes()
            if key not in signatures:
                signatures[key] = len(all_indices)
                all_indices.append(index)
            method_ids[method].append(signatures[key])
    codes, histories, _ = fit_bank(
        all_indices, data, "final_validation", plan, run, device
    )
    np.savez_compressed(
        destination / "predictors.npz",
        indices=np.stack(all_indices),
        codes=codes.cpu().numpy(),
    )
    save_json(
        destination / "predictors_frozen.json",
        {
            "frozen_utc": timestamp(),
            "sha256": digest(destination / "predictors.npz"),
            "histories": histories,
        },
    )
    meta = load_archive(root / "data" / f"meta_{condition}_{run}.npz")
    tx = torch.tensor(meta["x"], dtype=torch.float64, device=device)
    ty = torch.tensor(meta["y"], dtype=torch.float64, device=device)
    meta_scores = []
    for index, values in zip(all_indices, codes, strict=True):
        prediction = code_prediction(features(tx, index, 2), values, plan)
        meta_scores.append(float((prediction - ty).square().mean()))
    selected = {}
    for method, ids in method_ids.items():
        if ids:
            chosen = min(ids, key=lambda number: meta_scores[number])
            selected[method] = {
                "candidate": chosen,
                "meta_mse": meta_scores[chosen],
                "bank_size": len(ids),
            }
    result = {
        "condition": condition,
        "run": run,
        "abstained": False,
        "methods": selected,
        "meta_scores": meta_scores,
        "frozen_utc": timestamp(),
    }
    save_json(destination / "selection.json", result)
    print(
        json.dumps(
            {
                "source_case": [condition, run],
                "bootstrap_bank": len(boot_bank),
                "selected": selected,
            }
        ),
        flush=True,
    )
    return result


def learn(root, plan, device):
    results = []
    for condition, _ in enumerate(plan["noise"]):
        axis = load_archive(root / "data" / f"axis_{condition}.npz")
        estimates, records = [], []
        for task in range(plan["sources"]):
            estimate, record = estimate_axis(axis, task, plan)
            estimates.append(estimate)
            records.append(record)
        save_json(root / f"axis_fit_{condition}.json", records)
        print(
            json.dumps(
                {
                    "axis_condition": condition,
                    "estimated_codes": np.array(estimates).tolist(),
                }
            ),
            flush=True,
        )
        for run in range(plan["search_runs"]):
            results.append(learn_case(root, plan, condition, run, estimates, device))
    save_json(
        root / "source_frozen.json",
        {
            "frozen_utc": timestamp(),
            "results": results,
            "sha256": {
                str(p.relative_to(root)): digest(p)
                for p in (root / "source").rglob("selection.json")
            },
        },
    )


def audit(root, plan):
    frozen = json.loads((root / "source_frozen.json").read_text())
    teacher = load_archive(root / "private/teacher.npz")
    rows, aligned = [], {}
    for result in frozen["results"]:
        if result["abstained"]:
            rows.append(result)
            continue
        condition, run = result["condition"], result["run"]
        folder = root / "source" / f"case_{condition}_{run}"
        arrays = load_archive(folder / "predictors.npz")
        for method, selection in result["methods"].items():
            index = arrays["indices"][selection["candidate"]]
            metric, matched, _ = align_indices(index, teacher["index"], 2)
            rows.append(
                {
                    "condition": condition,
                    "noise": plan["noise"][condition],
                    "run": run,
                    "method": method,
                    "exact": metric["categorical_errors"] == 0,
                    **metric,
                }
            )
            aligned[f"{condition}_{run}_{method}"] = matched
    np.savez_compressed(root / "aligned_indices.npz", truth=teacher["index"], **aligned)
    binary = load_archive(root / "data/binary_train.npz")
    x = np.concatenate([binary[f"{s}_x"] for s in ("support", "query", "selection")])
    y = np.concatenate([binary[f"{s}_y"][0] for s in ("support", "query", "selection")])
    teacher_error = int(
        (
            (real_output(x, teacher["index"], teacher["codebooks"], plan["bias"]) > 0)
            != y
        ).sum()
    )
    result = {
        "audited_utc": timestamp(),
        "rows": rows,
        "teacher_binary_errors_on_non_test": teacher_error,
        "teacher_codebooks": teacher["codebooks"].tolist(),
    }
    save_json(root / "structure_audit.json", result)
    print(json.dumps({"structure_audit": rows}), flush=True)


def classification(root, plan, device):
    json.loads((root / "source_frozen.json").read_text())
    data = load_archive(root / "data/binary_train.npz")
    groups, seen, origins = [], {}, {}
    for path in sorted((root / "source").glob("case_*/selection.json")):
        selection = json.loads(path.read_text())
        if selection["abstained"]:
            continue
        arrays = load_archive(path.parent / "predictors.npz")
        for method, chosen in selection["methods"].items():
            index = arrays["indices"][chosen["candidate"]]
            key = canonical_index(index, 2).tobytes()
            if key not in seen:
                seen[key] = len(groups)
                groups.append(index)
            origins[f"{path.parent.name}/{method}"] = seen[key]
    teacher = load_archive(root / "private/teacher.npz")
    oracle = canonical_index(teacher["index"], 2)
    if oracle.tobytes() not in seen:
        seen[oracle.tobytes()] = len(groups)
        groups.append(oracle)
    origins["oracle"] = seen[oracle.tobytes()]
    rng = np.random.default_rng(plan["seed"] + 700001)
    random_index = rng.integers(1, 3, size=(11, 8)) * rng.choice((-1, 1), size=(11, 8))
    random_index[rng.random((11, 8)) > 0.36] = 0
    origins["random"] = len(groups)
    groups.append(random_index.astype(np.int16))
    count = plan["classification_restarts"]
    indices = np.repeat(np.stack(groups), count, axis=0)
    initial_rng = np.random.default_rng(plan["seed"] + 800001)
    initial_codes = initial_rng.normal(size=(count, 2))
    initial_codes *= 0.5 / np.linalg.norm(initial_codes, axis=1, keepdims=True)
    initial = np.c_[initial_codes, np.zeros(count), np.ones(count), np.zeros(count)]
    initial = np.tile(initial, (len(groups), 1))
    fitted = single.fit_fixed(
        data,
        indices,
        initial,
        [plan["classification_lr"]] * len(initial),
        plan["classification_spec"],
        plan["seed"] + 800003,
        device,
    )
    output = root / "classification"
    output.mkdir()
    np.savez_compressed(output / "shared.npz", **fitted)
    rows = [
        {
            "kind": kind,
            "task": 0,
            "restart": restart,
            "norm": 0.3,
            "lr": plan["classification_lr"],
        }
        for kind in ("dense_common", "dense_full")
        for restart in range(count)
    ]
    dense_indices = np.repeat(
        np.arange(1, 89, dtype=np.int16).reshape(1, 11, 8), len(rows), axis=0
    )
    dense_spec = {**plan["classification_spec"], "paired_initialization_steps": 0}
    dense = dense_fit.fit_fixed(
        data, dense_indices, rows, dense_spec, plan["seed"] + 800003, device
    )
    np.savez_compressed(output / "dense.npz", **dense)
    selections = {}
    for name, group in origins.items():
        ids = np.arange(group * count, (group + 1) * count)
        winner = int(ids[fitted["query_metrics"][ids, 0].argmin()])
        selections[name] = {"archive": "shared", "row": winner, "group": group}
    for number, kind in enumerate(("dense_common", "dense_full")):
        ids = np.arange(number * count, (number + 1) * count)
        winner = int(ids[dense["query_metrics"][ids, 0].argmin()])
        selections[kind] = {"archive": "dense", "row": winner}
    save_json(
        output / "predictors_frozen.json",
        {
            "frozen_utc": timestamp(),
            "selection": selections,
            "origins": origins,
            "unique_shared_fits": len(groups),
            "sha256": {
                name: digest(output / f"{name}.npz") for name in ("shared", "dense")
            },
        },
    )
    data.update(load_archive(root / "data/binary_test.npz"))
    tx, ty = (
        torch.tensor(data["test_x"], device=device),
        torch.tensor(data["test_y"], device=device),
    )
    ux = torch.tensor(
        np.concatenate(
            [data[f"{s}_x"] for s in ("support", "query", "selection", "test")]
        ),
        device=device,
    )
    uy = torch.tensor(
        np.concatenate(
            [data[f"{s}_y"] for s in ("support", "query", "selection", "test")], axis=1
        ),
        device=device,
    )
    with torch.no_grad():
        shared_index = torch.tensor(fitted["indices"], device=device).long()
        shared_code = torch.tensor(fitted["parameters"], device=device)
        test_shared = (
            metrics(single.predict(tx, shared_index, shared_code), ty).cpu().numpy()
        )
        full_shared = (
            metrics(single.predict(ux, shared_index, shared_code), uy).cpu().numpy()
        )
        dense_index = torch.tensor(dense["indices"], device=device).long()
        dense_code = torch.tensor(dense["parameters"], device=device)
        head = torch.tensor([r["kind"] == "dense_full" for r in rows], device=device)
        test_dense = (
            metrics(dense_fit.predict(tx, dense_index, dense_code, head), ty)
            .cpu()
            .numpy()
        )
        full_dense = (
            metrics(dense_fit.predict(ux, dense_index, dense_code, head), uy)
            .cpu()
            .numpy()
        )
    results = {}
    for name, selected in selections.items():
        arrays = fitted if selected["archive"] == "shared" else dense
        test = test_shared if selected["archive"] == "shared" else test_dense
        full = full_shared if selected["archive"] == "shared" else full_dense
        row = selected["row"]
        if selected["archive"] == "shared":
            group = selected["group"]
            ids = np.arange(group * count, (group + 1) * count)
        else:
            first = 0 if name == "dense_common" else count
            ids = np.arange(first, first + count)
        results[name] = {
            **selected,
            "query_bce": float(arrays["query_metrics"][row, 0]),
            "test_bce": float(test[row, 0]),
            "test_balanced_accuracy": float(test[row, 1]),
            "full_balanced_accuracy": float(full[row, 1]),
            "perfect_full_starts": int(
                np.isclose(full[ids, 1], 1, atol=1e-7, rtol=0).sum()
            ),
            "starts": count,
        }
    np.savez_compressed(
        output / "evaluation.npz",
        shared_test=test_shared,
        shared_full=full_shared,
        dense_test=test_dense,
        dense_full=full_dense,
    )
    save_json(
        output / "results.json",
        {
            "evaluated_utc": timestamp(),
            "methods": results,
            "test_count": len(tx),
            "full_count": len(ux),
        },
    )
    print(json.dumps({"classification": results}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("prepare", "learn", "audit", "classify", "all")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.monotonic()
    if args.action in ("prepare", "all"):
        plan = register(args.root)
        prepare(args.root, plan)
        print("PREPARED", str(args.root), flush=True)
    else:
        plan = json.loads((args.root / "protocol.json").read_text())
    if args.action in ("learn", "all"):
        learn(args.root, plan, args.device)
    if args.action in ("audit", "all"):
        audit(args.root, plan)
    if args.action in ("classify", "all"):
        classification(args.root, plan, args.device)
    print(
        json.dumps({"action": args.action, "seconds": time.monotonic() - started}),
        flush=True,
    )


if __name__ == "__main__":
    main()
