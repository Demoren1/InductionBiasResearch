"""CPU illustrations and replay for constructive dense recovery.

These are reproducible research examples, not a software test suite.
No fitting of neural networks or tuning against reference indexators occurs.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .recovery import recover_dense_structure


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _save(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_value)
        + "\n",
        encoding="utf-8",
    )


def _equivalent(first, second):
    first = np.asarray(first)
    second = np.asarray(second)
    return bool(
        np.array_equal(
            first[:, None] == first[None, :],
            second[:, None] == second[None, :],
        )
    )


def _illustrations():
    code_logs = np.array([0.0, 0.01, 0.1, 1.0])
    triple = [(0, 1, 2), (0, 3), (1, 3), (2, 3)]
    pairs = list(itertools.combinations(range(4), 2))
    cases = []
    for name, local_groups, values, tasks in [
        ("unique_without_repeated_pairs", triple, code_logs, 1),
        ("reflection_two_families", pairs, code_logs, 1),
        ("constant_neuron_four_choices", triple + [(2, 2)], code_logs, 1),
        ("resonant_inconclusive", pairs, np.array([0, 1, 4, 6]) / 6, 1),
        ("signs_remove_reflection", pairs, code_logs, 2),
    ]:
        truth = np.array([group for local in local_groups for group in local])
        neurons = np.repeat(
            np.arange(len(local_groups)),
            [len(local) for local in local_groups],
        )
        values = np.repeat(values[:, None], tasks, axis=1)
        if tasks == 2:
            values[:, 1] = [0.0, 0.023, 0.17, 0.81]
        scales = np.arange(len(local_groups))[:, None] * np.array(
            [0.037, -0.029][:tasks]
        )
        logs = values[truth] - scales[neurons]
        signs = None
        if tasks == 2:
            code_signs = np.array([[1, 1], [1, 1], [1, 1], [1, -1]])
            fixed_signs = np.where(np.arange(len(truth)) % 2, -1, 1)
            signs = fixed_signs[:, None] * code_signs[truth]
        cases.append(
            {
                "name": name,
                "logs": logs,
                "neurons": neurons,
                "epsilon": 1e-8,
                "signs": signs,
                "truth": truth,
            }
        )
    base = cases[0]
    epsilon = 0.0002
    cases.append(
        {
            "name": "bounded_noise_without_repeated_pairs",
            "logs": (
                base["logs"]
                + 0.9 * epsilon * np.sin(np.arange(len(base["logs"])))[:, None]
            ),
            "neurons": base["neurons"],
            "epsilon": epsilon,
            "signs": None,
            "truth": base["truth"],
        }
    )
    neurons = np.repeat(np.arange(4), [3, 2, 2, 2])
    cases.append(
        {
            "name": "unstructured_budget_four_excluded",
            "logs": 10.0 ** np.arange(9)[:, None] / 1e8,
            "neurons": neurons,
            "epsilon": 1e-12,
            "signs": None,
            "truth": None,
        }
    )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/constructive_dense_recovery_2026-11-14"),
    )
    parser.add_argument(
        "--descriptors",
        type=Path,
        default=Path(
            "artifacts/single_task_recovery_2026-11-14/"
            "single_source_descriptors.npz"
        ),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    examples = _illustrations()
    results = []
    for example in examples:
        result = recover_dense_structure(
            example["logs"],
            example["neurons"],
            example["epsilon"],
            max_codes=4,
            signs=example["signs"],
        )
        result["name"] = example["name"]
        results.append(result)
        print(
            example["name"],
            result["status"],
            result["reason"],
            len(result["families"]),
            flush=True,
        )
    _save(args.output / "illustrations_frozen.json", results)
    audit = []
    for example, result in zip(examples, results):
        reference = example["truth"]
        compatible = None
        if reference is not None:
            compatible = []
            for family in result["families"]:
                core = np.ones(len(reference), dtype=bool)
                for free in family["free_neurons"]:
                    core[free["positions"]] = False
                compatible.append(
                    _equivalent(
                        reference[core], np.asarray(family["indexator"])[core]
                    )
                )
        audit.append(
            {
                "name": example["name"],
                "core_matches_constructed_example": compatible,
                "compact_groupings_count": sum(
                    family["groupings_count"] for family in result["families"]
                ),
            }
        )
    _save(args.output / "illustrations_audit.json", audit)
    if not args.descriptors.is_file():
        print(
            "Saved dense descriptors unavailable; replay omitted.", flush=True
        )
        return
    replay = []
    epsilon_grid = [1e-9, 1e-6, 1e-4, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    with np.load(args.descriptors, allow_pickle=False) as descriptors:
        for case in ("case_04", "case_05"):
            normal = descriptors[f"{case}_normal"]
            uncertainty = descriptors[f"{case}_uncertainty"]
            mask = np.abs(normal) > np.maximum(
                3 * uncertainty[None, :], 0.002
            )
            rows, neurons = np.where(mask)
            logs = np.log(np.abs(normal[rows, neurons]))
            signs = np.sign(normal[rows, neurons]).astype(int)
            for epsilon in epsilon_grid:
                result = recover_dense_structure(
                    logs, neurons, epsilon, max_codes=4, signs=signs
                )
                result["case"] = case
                result["observed_positions"] = len(rows)
                result["rows"] = rows
                result["columns"] = neurons
                replay.append(result)
                print(
                    case,
                    epsilon,
                    result["status"],
                    result["reason"],
                    result["contrast_components"],
                    flush=True,
                )
    _save(
        args.output / "dense_replay_frozen.json",
        {
            "descriptor_file": str(args.descriptors),
            "mask_rule": (
                "abs(normal) > maximum(3 * uncertainty[None, :], 0.002)"
            ),
            "code_budget": 4,
            "epsilon_grid": epsilon_grid,
            "same_saved_cases_as_previous_experiment": True,
            "results": replay,
        },
    )
    selected = []
    for case in ("case_04", "case_05"):
        complete = [
            row for row in replay
            if row["case"] == case and row["status"] == "complete"
        ]
        if complete:
            selected.append(complete[0])
    _save(
        args.output / "dense_selected_frozen.json",
        {
            "rule": (
                "first complete result in the fixed ascending epsilon grid"
            ),
            "results": selected,
        },
    )


if __name__ == "__main__":
    main()
