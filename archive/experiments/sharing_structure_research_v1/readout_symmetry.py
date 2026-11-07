"""Synchronize task gauges using only automorphisms of the discovered I1."""

import itertools
import math
from collections import defaultdict, deque

import numpy as np

from .indexed_readout import design, head_path


def automorphisms(index, codes):
    hidden = index.shape[1]
    identity = {"columns": np.arange(hidden), "code_matrix": np.eye(codes)}
    found = {tuple(range(hidden)): identity}
    for permutation in itertools.permutations(range(1, codes + 1)):
        for signs in itertools.product((-1, 1), repeat=codes):
            labels = np.r_[0, np.array(permutation) * signs]
            renamed = labels[np.abs(index)] * np.sign(index)
            lookup = defaultdict(deque)
            for h, column in enumerate(renamed.T):
                lookup[tuple(column)].append(h)
            columns = []
            for column in index.T:
                available = lookup[tuple(column)]
                if not available:
                    break
                columns.append(available.popleft())
            if len(columns) != hidden:
                continue
            matrix = np.zeros((codes, codes))
            matrix[np.arange(codes), np.array(permutation) - 1] = signs
            found.setdefault(
                tuple(columns), {"columns": np.array(columns), "code_matrix": matrix}
            )
    return list(found.values())


def synchronized_path(index, states, data, spec, alignment_budget=4096):
    autos = automorphisms(index, spec["codes"])
    tasks = len(next(iter(states.values())))
    combinations = len(autos) ** (tasks - 1)
    exhausted = combinations > alignment_budget
    gauges = (
        [tuple([0] * tasks)]
        if exhausted
        else [
            (0,) + rest
            for rest in itertools.product(range(len(autos)), repeat=tasks - 1)
        ]
    )
    candidates = []
    designs = {}
    curves = []
    for state, codes in states.items():
        designs[state] = {
            name: design(data[name + "_x"], index, codes, spec["bias"])
            for name in ("fit", "h1", "meta")
        }
        for gauge in gauges:
            phi = np.stack(
                [
                    designs[state]["fit"][t][:, autos[gauge[t]]["columns"]]
                    for t in range(tasks)
                ]
            )
            vphi = np.stack(
                [
                    designs[state]["h1"][t][:, autos[gauge[t]]["columns"]]
                    for t in range(tasks)
                ]
            )
            path, _ = head_path(phi, data["fit_y"], vphi, data["h1_y"])
            curves.append(
                {
                    "state": state,
                    "gauge": gauge,
                    "path": [
                        {
                            "groups": row["groups"],
                            "train_mse": row["train_mse"],
                            "validation_mse": row["validation_mse"],
                        }
                        for row in path
                    ],
                }
            )
            for row in path:
                candidates.append(
                    {
                        "state": state,
                        "gauge": gauge,
                        "index": row["index"],
                        "head": row["head"],
                        "centers": row["centers"],
                        "groups": row["groups"],
                        "train_mse": row["train_mse"],
                        "validation_mse": row["validation_mse"],
                    }
                )

    def prediction(candidate, panel):
        matrix = designs[candidate["state"]][panel]
        return np.stack(
            [
                matrix[t][:, autos[candidate["gauge"][t]]["columns"]]
                @ candidate["head"][t]
                for t in range(tasks)
            ]
        )

    best = min(candidates, key=lambda row: row["validation_mse"])
    best_loss = ((prediction(best, "h1") - data["h1_y"]) ** 2).mean(axis=2).ravel()
    accepted = []
    floor = 1e-12 * max(float(np.var(data["h1_y"])), 1)
    for candidate in candidates:
        delta = ((prediction(candidate, "h1") - data["h1_y"]) ** 2).mean(
            axis=2
        ).ravel() - best_loss
        if delta.mean() <= 2 * np.std(delta, ddof=1) / math.sqrt(len(delta)) + floor:
            accepted.append(candidate)
    selected = min(accepted, key=lambda row: (row["groups"], row["validation_mse"]))
    free = min(
        (
            row
            for row in candidates
            if row["groups"] == index.shape[1] * spec["outputs"]
        ),
        key=lambda row: row["validation_mse"],
    )
    for candidate in (selected, free):
        original = states[candidate["state"]]
        candidate["codes"] = np.stack(
            [
                original[t] @ autos[candidate["gauge"][t]]["code_matrix"]
                for t in range(tasks)
            ]
        )
        candidate["meta_mse"] = float(
            np.mean((prediction(candidate, "meta") - data["meta_y"]) ** 2)
        )
    return (
        selected,
        free,
        {
            "automorphisms": len(autos),
            "alignments": len(gauges),
            "possible_alignments": combinations,
            "alignment_budget_exceeded": exhausted,
            "numeric_states": list(states),
            "curves": curves,
        },
    )
