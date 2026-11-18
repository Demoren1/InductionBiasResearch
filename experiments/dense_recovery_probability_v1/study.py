"""A fixed CPU probability study with genuinely fitted dense linear layers."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components

from .bounds import (
    ols_error_bound,
    overlap_failure,
    sign_distinct_probability,
    uniform_gap_failure,
)
from .recover import recover_from_dense


def _save_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _signed_exact(predicted, truth):
    active = truth != 0
    if not np.array_equal(predicted != 0, active):
        return False
    first = np.abs(predicted[active])
    second = np.abs(truth[active])
    pairs = np.unique(np.c_[first, second], axis=0)
    if len(pairs) != len(np.unique(first)):
        return False
    if len(pairs) != len(np.unique(second)):
        return False
    relative_sign = np.sign(predicted[active] * truth[active])
    return all(
        np.all(
            relative_sign[second == code] == relative_sign[second == code][0]
        )
        for code in np.unique(second)
    )


def _geometry_connected(truth, codes):
    local = [set(np.abs(column)) - {0} for column in truth.T]
    for code in range(1, codes + 1):
        members = [group for group in local if code in group]
        if not members:
            return False
        graph = np.array(
            [
                [len(first & second) >= 2 for second in members]
                for first in members
            ],
            dtype=int,
        )
        if (
            connected_components(graph, directed=False, return_labels=False)
            != 1
        ):
            return False
    return True


def _relations(codes):
    found = set()
    for size in (1, 2):
        multisets = list(
            itertools.combinations_with_replacement(range(codes), size)
        )
        for first in multisets:
            for second in multisets:
                vector = np.bincount(first, minlength=codes) - np.bincount(
                    second, minlength=codes
                )
                nonzero = vector[vector != 0]
                if not len(nonzero) or math.gcd(*map(int, nonzero)) != 1:
                    continue
                if nonzero[0] < 0:
                    vector *= -1
                found.add(tuple(vector))
    return np.array(sorted(found), dtype=int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--sign-trials", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/dense_recovery_probability_2026-11-15"),
    )
    args = parser.parse_args()
    if args.trials < 1 or args.sign_trials < 1:
        raise ValueError("Trial counts must be positive.")
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    protocol = {
        "seed": 2611150927,
        "magnitude_trials": args.trials,
        "sign_trials": args.sign_trials,
        "codes": 4,
        "magnitude_inputs": 16,
        "neurons": 8,
        "active_per_neuron": 12,
        "magnitude_tasks": [1, 2, 4],
        "magnitude_samples": [32, 128],
        "label_noise": [0.0005, 0.003, 0.015],
        "sign_inputs": 8,
        "sign_tasks": [1, 2, 4, 8, 12],
        "sign_samples": 32,
        "sign_noise": 0.003,
        "log_code_length": 1.0,
        "log_scale_limit": 0.25,
        "beta": 0.01,
        "minimum_active": math.exp(-0.75),
        "design": "repeated orthogonal blocks, X.T X = N I",
        "dense_fit": "full ordinary least squares from generated labels",
        "selection": "fixed algorithm and theorem-prescribed tolerance",
        "source_files": {},
    }
    files = list(Path(__file__).parent.glob("*.py"))
    files.append(
        Path("experiments/constructive_dense_recovery_v1/recovery.py")
    )
    for path in files:
        protocol["source_files"][str(path)] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    _save_json(root / "protocol.json", protocol)
    snapshot = root / "executed"
    snapshot.mkdir()
    for path in files:
        destination = snapshot / path.name
        if destination.exists():
            destination = snapshot / (path.parent.name + "_" + path.name)
        destination.write_bytes(path.read_bytes())
    rng = np.random.default_rng(protocol["seed"])
    inputs, neurons, codes = 16, 8, 4
    maximum_tasks, maximum_samples = 4, 128
    orthogonal, _ = np.linalg.qr(rng.normal(size=(inputs, inputs)))
    design = np.tile(math.sqrt(inputs) * orthogonal, (8, 1))
    records, predictions, fitted_for_audit = [], [], []
    references, true_weights, code_logs = [], [], []
    source_weights = []
    for trial in range(args.trials):
        mask = np.zeros((inputs, neurons), dtype=bool)
        for neuron in range(neurons):
            mask[
                rng.choice(
                    inputs, protocol["active_per_neuron"], replace=False
                ),
                neuron,
            ] = True
        rows, columns = np.where(mask)
        truth = np.zeros(mask.shape, dtype=np.int16)
        truth[rows, columns] = rng.integers(
            1, codes + 1, len(rows)
        ) * rng.choice([-1, 1], len(rows))
        logs = rng.uniform(-0.5, 0.5, (maximum_tasks, codes))
        scales = rng.uniform(-0.25, 0.25, (maximum_tasks, neurons))
        weights = np.zeros((maximum_tasks, inputs, neurons))
        weights[:, rows, columns] = np.sign(truth[rows, columns]) * np.exp(
            logs[:, np.abs(truth[rows, columns]) - 1] - scales[:, columns]
        )
        noise = rng.normal(size=(maximum_tasks, maximum_samples, neurons))
        signal = np.einsum("nd,tdh->tnh", design, weights)
        references.append(truth)
        true_weights.append(weights)
        code_logs.append(logs)
        for samples in protocol["magnitude_samples"]:
            local_design = design[:samples]
            for level in protocol["label_noise"]:
                labels = signal[:, :samples] + level * noise[:, :samples]
                dense = (
                    np.einsum("nd,tnh->tdh", local_design, labels) / samples
                )
                source_weights.append(dense)
                for tasks in protocol["magnitude_tasks"]:
                    eta = ols_error_bound(
                        inputs,
                        neurons,
                        tasks,
                        samples,
                        level,
                        protocol["beta"],
                    )
                    result = recover_from_dense(
                        dense[:tasks], eta, protocol["minimum_active"], codes
                    )
                    predicted = result.pop(
                        "index", np.zeros((inputs, neurons), dtype=np.int16)
                    )
                    residual = (
                        np.einsum("nd,tdh->tnh", local_design, dense[:tasks])
                        - labels[:tasks]
                    )
                    epsilon = result["epsilon"]
                    gap_bound = uniform_gap_failure(
                        codes, tasks, 8 * epsilon, 1.0
                    )
                    geometry = overlap_failure(
                        codes, neurons, protocol["active_per_neuron"]
                    )
                    records.append(
                        {
                            "phase": "magnitude",
                            "trial": trial,
                            "samples": samples,
                            "tasks": tasks,
                            "noise": level,
                            "dense_train_mse": float(np.mean(residual**2)),
                            "gap_failure_bound": gap_bound,
                            "geometry_failure_bound": geometry,
                            "success_lower_bound": max(
                                0.0,
                                1 - protocol["beta"] - geometry - gap_bound,
                            ),
                            **result,
                        }
                    )
                    predictions.append(predicted)
                    fitted_for_audit.append(dense[:tasks].copy())
        if (trial + 1) % 20 == 0:
            print("Magnitude trials:", trial + 1, flush=True)

    sign_inputs, sign_tasks, sign_samples = 8, 12, 32
    sign_orthogonal, _ = np.linalg.qr(
        rng.normal(size=(sign_inputs, sign_inputs))
    )
    sign_design = np.tile(math.sqrt(sign_inputs) * sign_orthogonal, (4, 1))
    sign_references, sign_true_weights, sign_source_weights = [], [], []
    for trial in range(args.sign_trials):
        truth = np.zeros((sign_inputs, neurons), dtype=np.int16)
        rows = rng.integers(sign_inputs, size=neurons)
        groups = rng.permutation(np.tile(np.arange(1, codes + 1), 2))
        truth[rows, np.arange(neurons)] = groups * rng.choice([-1, 1], neurons)
        logs = rng.uniform(-0.5, 0.5, (sign_tasks, codes))
        scales = rng.uniform(-0.25, 0.25, (sign_tasks, neurons))
        signs = rng.choice([-1, 1], (sign_tasks, codes))
        weights = np.zeros((sign_tasks, sign_inputs, neurons))
        weights[:, rows, np.arange(neurons)] = (
            np.sign(truth[rows, np.arange(neurons)])
            * signs[:, groups - 1]
            * np.exp(logs[:, groups - 1] - scales)
        )
        labels = np.einsum("nd,tdh->tnh", sign_design, weights)
        labels += protocol["sign_noise"] * rng.normal(size=labels.shape)
        dense = np.einsum("nd,tnh->tdh", sign_design, labels) / sign_samples
        sign_references.append(truth)
        sign_true_weights.append(weights)
        sign_source_weights.append(dense)
        for tasks in protocol["sign_tasks"]:
            eta = ols_error_bound(
                sign_inputs,
                neurons,
                tasks,
                sign_samples,
                protocol["sign_noise"],
                protocol["beta"],
            )
            result = recover_from_dense(
                dense[:tasks], eta, protocol["minimum_active"], codes
            )
            predicted = result.pop(
                "index", np.zeros((sign_inputs, neurons), dtype=np.int16)
            )
            residual = (
                np.einsum("nd,tdh->tnh", sign_design, dense[:tasks])
                - labels[:tasks]
            )
            probability = sign_distinct_probability(codes, tasks)
            records.append(
                {
                    "phase": "sign",
                    "trial": trial,
                    "samples": sign_samples,
                    "tasks": tasks,
                    "noise": protocol["sign_noise"],
                    "dense_train_mse": float(np.mean(residual**2)),
                    "sign_probability": probability,
                    "success_lower_bound": max(
                        0.0, probability - protocol["beta"]
                    ),
                    **result,
                }
            )
            predictions.append(predicted)
            fitted_for_audit.append(dense[:tasks].copy())
        if (trial + 1) % 200 == 0:
            print("Sign trials:", trial + 1, flush=True)

    _save_json(root / "predictions_frozen.json", records)
    boundary = args.trials * 18
    np.savez_compressed(
        root / "predictions_frozen.npz",
        magnitude=np.stack(predictions[:boundary]),
        sign=np.stack(predictions[boundary:]),
    )
    np.savez_compressed(
        root / "dense_descriptors.npz",
        magnitude=np.stack(source_weights),
        sign=np.stack(sign_source_weights),
        magnitude_design=design,
        sign_design=sign_design,
    )
    np.savez_compressed(
        root / "references_for_audit.npz",
        magnitude=np.stack(references),
        magnitude_weights=np.stack(true_weights),
        magnitude_code_logs=np.stack(code_logs),
        sign=np.stack(sign_references),
        sign_weights=np.stack(sign_true_weights),
    )
    print("All predictions frozen; starting structural audit.", flush=True)
    relations = _relations(codes)
    audit = []
    geometry_cache = [
        _geometry_connected(truth, codes) for truth in references
    ]
    for record, predicted, dense in zip(
        records, predictions, fitted_for_audit
    ):
        trial, tasks = record["trial"], record["tasks"]
        if record["phase"] == "magnitude":
            truth = references[trial]
            weights = true_weights[trial][:tasks]
            gap = float(
                np.min(
                    np.max(
                        np.abs(relations @ code_logs[trial][:tasks].T), axis=1
                    )
                )
            )
            geometry = geometry_cache[trial]
            gap_event = gap > 8 * record["epsilon"]
        else:
            truth = sign_references[trial]
            weights = sign_true_weights[trial][:tasks]
            gap, geometry, gap_event = None, None, None
        maximum_error = float(np.max(np.abs(dense - weights)))
        channel = maximum_error <= record["absolute_error_bound"]
        audit.append(
            {
                **record,
                "signed_exact": _signed_exact(predicted, truth),
                "dense_max_weight_error": maximum_error,
                "dense_relative_population_mse": float(
                    np.sum((dense - weights) ** 2) / np.sum(weights**2)
                ),
                "dense_error_event": channel,
                "actual_delta_4": gap,
                "repeated_pair_graphs_connected": geometry,
                "delta_4_event": gap_event,
                "sufficient_events_hold": (
                    bool(channel and geometry and gap_event)
                    if record["phase"] == "magnitude"
                    else None
                ),
            }
        )
    _save_json(root / "audit.json", audit)
    print(
        "Failures on sufficient magnitude events:",
        sum(
            row["sufficient_events_hold"] and not row["signed_exact"]
            for row in audit
            if row["phase"] == "magnitude"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
