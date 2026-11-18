"""Compare learned sharing with fixed-budget random controls on synthetic tasks."""

import argparse
import itertools
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from experiments.asrr_anchor_init_v1.report import interval
from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
    digest,
    save_json,
)
from experiments.wsr_omniglot_v1.source import read_json

from .core import canonical, input_forest, signed_forest
from .synthetic import NOISE, SCENARIOS, cases, numerical
from .synthetic import prepare as generate

OUTPUT = Path("artifacts/asrr_synthetic_random_controls_2026-11-18")
ROOT = OUTPUT.parent / "local" / OUTPUT.name
LINEAR = ROOT / "linear"
RELU = ROOT / "relu"
RELU_REFERENCE = Path("artifacts/local/paper_asrr_2026-11-17/synthetic_4")
RELU_CASE = "case_04"
RELU_PANELS = RELU_REFERENCE / "targets" / f"{RELU_CASE}_c0"
RELU_TARGET = RELU / "targets" / f"{RELU_CASE}_c0"
RELU_TRUTH = Path("artifacts/single_task_recovery_2026-11-14/private/case_04.npz")
RELU_SCORES = Path("artifacts/compact_experiments_2026-11-18/synthetic/results.json")
SEED = 7611189600
SHUFFLES = 4
RELU_METHODS = ("random_asrr", "random_teacher")
LINEAR_METHODS = ("asrr", "random_matched", "shared_background", "dense", "oracle")


def shuffle_groups(index, generator, *, shared_inputs=False):
    """Preserve support, signs and group counts within each neuron."""
    if shared_inputs:
        return index[generator.permutation(len(index))]
    shuffled = index.copy()
    for neuron in range(index.shape[1]):
        for sign in (-1, 1):
            locations = np.sign(index[:, neuron]) == sign
            shuffled[locations, neuron] = generator.permutation(
                index[locations, neuron]
            )
    return shuffled


def shared_background(shape, codes, generator):
    inputs, hidden = shape
    base, extra = divmod(codes - 1, hidden)
    index = np.ones(shape, dtype=np.int16)
    next_code = 2
    for neuron in range(hidden):
        count = base + int(neuron < extra)
        rows = generator.choice(inputs, size=count, replace=False)
        index[rows, neuron] = np.arange(next_code, next_code + count)
        next_code += count
    return index


def fit_linear_case(item):
    name, scenario, rank, replica, _ = item
    with np.load(LINEAR / "public" / f"{name}.npz") as archive:
        data = dict(archive)
    weights = data["weights"]
    input_groups = scenario == "input_groups"
    if input_groups:
        codes = 24
        learned = input_forest(weights, (codes,))[codes]
    else:
        codes = 8
        learned = signed_forest(weights, (codes,))[0][codes]
    generator = np.random.default_rng(
        SEED + 1 + 100003 * SCENARIOS.index(scenario) + 1009 * rank + 101 * replica
    )
    dense = np.arange(1, learned.size + 1, dtype=np.int16).reshape(learned.shape)
    indices = {"asrr": learned, "dense": dense}
    for number in range(SHUFFLES):
        indices[f"random_matched_{number}"] = shuffle_groups(
            learned, generator, shared_inputs=input_groups
        )
        indices[f"shared_background_{number}"] = shared_background(
            learned.shape, codes, generator
        )
    values = {}
    for method, index in indices.items():
        prediction, _ = numerical(index, data, "test")
        values[f"{method}_index"] = index
        values[f"{method}_prediction"] = prediction
    path = LINEAR / "predictions" / f"{name}.npz"
    np.savez_compressed(path, **values)
    return name, digest(path)


def sharing_pairs(index):
    flat = index.ravel()
    equal = np.abs(flat)[:, None] == np.abs(flat)[None, :]
    return np.triu(equal * np.sign(flat)[:, None] * np.sign(flat)[None, :], k=1)


def audit_index(index, truth, prediction, answer):
    real = sharing_pairs(truth)
    proposed = sharing_pairs(index)
    hits = np.count_nonzero((real == proposed) & (real != 0))
    denominator = np.count_nonzero(real) + np.count_nonzero(proposed)
    return {
        "mse": float(np.mean(np.square(prediction - answer))),
        "signed_pair_f1": float(2 * hits / max(denominator, 1)),
        "exact": float(np.array_equal(canonical(index), canonical(truth))),
    }


def average(rows):
    return {key: float(np.mean([item[key] for item in rows])) for key in rows[0]}


def linear():
    folder = OUTPUT / "linear"
    folder.mkdir(parents=True)
    generate(destination=LINEAR, seed_base=SEED)
    with ProcessPoolExecutor(max_workers=8) as executor:
        predictions = dict(executor.map(fit_linear_case, cases()))
    save_json(folder / "predictions_frozen.json", predictions)
    rows = []
    for name, scenario, rank, replica, level in cases():
        path = LINEAR / "predictions" / f"{name}.npz"
        if digest(path) != predictions[name]:
            raise ValueError(f"Changed predictions: {name}")
        with (
            np.load(LINEAR / "private" / f"{name}.npz") as truth,
            np.load(LINEAR / "public" / f"{name}.npz") as public,
            np.load(path) as fitted,
        ):
            teacher, answer = truth["index"], truth["test_query_clean"]
            numbers, _ = numerical(teacher, public, "test")
            scores = {"oracle": audit_index(teacher, teacher, numbers, answer)}
            for method in ("asrr", "dense", "random_matched", "shared_background"):
                candidates = (
                    [method]
                    if method in ("asrr", "dense")
                    else [f"{method}_{i}" for i in range(SHUFFLES)]
                )
                scores[method] = average(
                    [
                        audit_index(
                            fitted[f"{key}_index"],
                            teacher,
                            fitted[f"{key}_prediction"],
                            answer,
                        )
                        for key in candidates
                    ]
                )
        rows.append(
            {
                "case": name,
                "scenario": scenario,
                "rank": rank,
                "replica": replica,
                "noise": NOISE[level],
                "methods": scores,
            }
        )
    summary = []
    for scenario, noise in itertools.product(SCENARIOS, NOISE):
        panel = [
            r["methods"]
            for r in rows
            if r["scenario"] == scenario and r["noise"] == noise
        ]
        delta = np.array([r["random_matched"]["mse"] - r["asrr"]["mse"] for r in panel])
        summary.append(
            {
                "scenario": scenario,
                "noise": noise,
                "structures": len(panel),
                "methods": {
                    method: average([r[method] for r in panel])
                    for method in LINEAR_METHODS
                },
                "matched_random_minus_asrr": {
                    "mse_difference": float(delta.mean()),
                    "ci95_halfwidth": interval(delta),
                },
            }
        )
    save_json(
        folder / "results.json",
        {
            "structures": 32,
            "new_targets_per_structure": 3,
            "random_masks_per_control": SHUFFLES,
            "seed": SEED,
            "summary": summary,
            "per_structure": rows,
        },
    )
    print(
        {"linear_conditions": len(rows), "results": str(folder / "results.json")},
        flush=True,
    )


def relu_prepare():
    folder = OUTPUT / "relu"
    folder.mkdir(parents=True)
    RELU_TARGET.mkdir(parents=True)
    panel_names = ("support.npz", "selection.npz", "query_x.npz")
    receipt = read_json(RELU_PANELS / "asrr/registered.json")
    for name in panel_names:
        if digest(RELU_PANELS / name) != receipt["input_sha256"][name]:
            raise ValueError(f"Changed ReLU observations: {name}")
    for name in (*panel_names, "sizes.json"):
        shutil.copyfile(RELU_PANELS / name, RELU_TARGET / name)
    source = RELU_REFERENCE / "fits/structures.npz"
    selected = read_json(
        Path("artifacts/paper_asrr_2026-11-17/synthetic_4/source_selected.json")
    )
    if digest(source) != selected["models_sha256"]:
        raise ValueError("Changed archived aSRR assignment")
    with np.load(source) as archive:
        asrr = archive["index"]
    with np.load(RELU_TRUTH) as archive:
        teacher = archive["index"]
    generator = np.random.default_rng(SEED + 1)
    indices = {
        f"{method}_{i}": shuffle_groups(index, generator)
        for method, index in zip(RELU_METHODS, (asrr, teacher), strict=True)
        for i in range(SHUFFLES)
    }
    basis = {
        method: np.eye(5)[np.abs(index)][..., 1:] * np.sign(index)[..., None]
        for method, index in indices.items()
    }
    for values in basis.values():
        values /= np.sqrt(np.square(values).sum(axis=-1).mean())
    np.savez_compressed(RELU_TARGET / "structures.npz", **basis)
    np.savez_compressed(folder / "fixed_indexators.npz", **indices)
    save_json(
        folder / "registration.json",
        {
            "case": RELU_CASE,
            "methods": list(indices),
            "panel_sha256": {name: digest(RELU_TARGET / name) for name in panel_names},
            "solver_sha256": {
                key: receipt[key]
                for key in ("code_sha256", "adapter_sha256", "numeric_sha256")
            },
            "assignment_sha256": digest(folder / "fixed_indexators.npz"),
            "seed": SEED + 1,
            "from_fitted_dense": "Random controls of the frozen aSRR groups",
            "oracle_control": "Random controls with the teacher's support and signs supplied",
        },
    )


def relu(gpus):
    folder = OUTPUT / "relu"
    registration = read_json(folder / "registration.json")
    methods = registration["methods"]
    if digest(folder / "fixed_indexators.npz") != registration["assignment_sha256"]:
        raise ValueError("Changed ReLU indexators")

    def worker(number):
        gpu = gpus[number % len(gpus)]
        for method in methods[number :: len(gpus)]:
            with (folder / f"{method}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "experiments.paper_recovery_extension_v1.regression_target",
                        "--root",
                        str(RELU),
                        "--case",
                        RELU_CASE,
                        "--method",
                        method,
                        "--device",
                        f"cuda:{gpu}",
                    ],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            print({"method": method, "gpu": gpu, "status": "complete"}, flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        list(executor.map(worker, range(len(gpus))))
    frozen = {}
    for method in methods:
        target = RELU_TARGET / method
        receipt = read_json(target / "registered.json")
        metrics = read_json(target / "metrics.json")
        prediction = target / "frozen_predictions.npz"
        sha = digest(prediction)
        if (
            any(
                receipt["input_sha256"][name] != expected
                for name, expected in registration["panel_sha256"].items()
            )
            or any(
                receipt[key] != value
                for key, value in registration["solver_sha256"].items()
            )
            or sha != metrics["predictions_sha256"]
            or read_json(target / "access_receipt.json")["violations"]
        ):
            raise ValueError(f"Unverified ReLU target fit: {method}")
        frozen[method] = sha
    save_json(folder / "predictions_frozen.json", frozen)
    with np.load(RELU_REFERENCE / "private/answers.npz") as data:
        answer = data["y"]
    with np.load(RELU_TRUTH) as data:
        teacher = canonical_index(data["index"], 4)
    with np.load(folder / "fixed_indexators.npz") as archive:
        scores = {}
        for method in methods:
            with np.load(RELU_TARGET / method / "frozen_predictions.npz") as data:
                prediction = data["prediction_256"]
            if prediction.shape != answer.shape or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid ReLU prediction: {method}")
            mse = np.square(prediction - answer).mean(axis=(1, 2))
            index = archive[method]
            scores[method] = {
                "mse": float(mse.mean()),
                "per_function_mse": mse.tolist(),
                "signed_exact": bool(
                    np.array_equal(canonical_index(index, 4), teacher)
                ),
                "active_connections": int(np.count_nonzero(index)),
            }
    baseline = read_json(RELU_SCORES)["methods"]
    save_json(
        folder / "results.json",
        {
            "target_functions": len(answer),
            "codes": 4,
            "controls": scores,
            "baseline": {key: baseline[key] for key in ("asrr", "g34", "dense")},
            "means": {
                method: float(
                    np.mean([scores[f"{method}_{i}"]["mse"] for i in range(SHUFFLES)])
                )
                for method in RELU_METHODS
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("linear", "prepare-relu", "relu"))
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    if args.action == "linear":
        linear()
    elif args.action == "prepare-relu":
        relu_prepare()
    else:
        relu(args.gpus)
