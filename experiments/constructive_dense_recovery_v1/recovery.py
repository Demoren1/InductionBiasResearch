"""Recover code groups from within-neuron log-weight differences.

The finite-family criterion and completeness assumptions are documented in
paper/theory/CONSTRUCTIVE_DENSE_RECOVERY_2026-11-14.md. Computations use
float64; numerical residuals are not formal interval certificates.
"""

from __future__ import annotations

import math
import operator
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import (
    connected_components,
    maximum_bipartite_matching,
)
from scipy.spatial import cKDTree


def _contrast_components(logs, neurons, epsilon):
    """Cluster all local differences, retaining their ordered positions."""
    differences = []
    blocks = []
    offset = 0
    for neuron in range(int(neurons.max()) + 1):
        positions = np.flatnonzero(neurons == neuron)
        local = logs[positions]
        values = (local[:, None, :] - local[None, :, :]).reshape(
            -1, logs.shape[1]
        )
        differences.append(values)
        blocks.append((positions, offset))
        offset += len(values)
    differences = np.concatenate(differences)
    radius = 4 * epsilon
    if logs.shape[1] == 1:
        order = np.argsort(differences[:, 0], kind="stable")
        cuts = np.diff(differences[order, 0]) > radius
        ordered_labels = np.r_[0, np.cumsum(cuts)]
        labels = np.empty(len(order), dtype=int)
        labels[order] = ordered_labels
        count = int(ordered_labels[-1]) + 1
    else:
        edges = cKDTree(differences).query_pairs(
            radius, p=np.inf, output_type="ndarray"
        )
        rows = np.r_[edges[:, 0], edges[:, 1]]
        columns = np.r_[edges[:, 1], edges[:, 0]]
        graph = csr_matrix(
            (np.ones(len(rows)), (rows, columns)),
            shape=(len(differences), len(differences)),
        )
        count, labels = connected_components(graph, directed=False)
    zero_class = int(labels[0])
    _, first = np.unique(labels, return_index=True)
    representatives = differences[first]
    minima = np.full((count, logs.shape[1]), np.inf)
    maxima = np.full((count, logs.shape[1]), -np.inf)
    np.minimum.at(minima, labels, differences)
    np.maximum.at(maxima, labels, differences)
    diameters = np.max(maxima - minima, axis=1)
    zero_radius = float(
        np.max(np.abs(differences[labels == zero_class]))
    )
    nonzero = np.flatnonzero(np.arange(count) != zero_class)
    class_to_root = np.full(count, -1, dtype=int)
    class_to_root[nonzero] = np.arange(len(nonzero))
    local_classes = []
    for positions, start in blocks:
        size = len(positions)
        local_classes.append(
            (
                positions,
                class_to_root[
                    labels[start:start + size * size]
                ].reshape(size, size),
            )
        )
    return (
        representatives[nonzero],
        local_classes,
        diameters,
        zero_radius,
        len(differences),
    )


def _root_matching(points, representatives, epsilon):
    """Find a unique correspondence between ordered codes and root classes."""
    codes = len(points)
    left, right = np.where(~np.eye(codes, dtype=bool))
    values = points[left] - points[right]
    neighbours = cKDTree(representatives).query_ball_point(
        values, 6 * epsilon, p=np.inf
    )
    rows = np.repeat(np.arange(len(values)), [len(x) for x in neighbours])
    columns = np.array(
        [column for adjacent in neighbours for column in adjacent], dtype=int
    )
    graph = csr_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(len(values), len(representatives)),
    )
    matching = maximum_bipartite_matching(graph, perm_type="column")
    if np.any(matching < 0):
        return "absent", None
    inverse = np.empty_like(matching)
    inverse[matching] = np.arange(len(matching))
    unmatched = columns != matching[rows]
    alternating = csr_matrix(
        (
            np.ones(np.count_nonzero(unmatched)),
            (rows[unmatched], inverse[columns[unmatched]]),
        ),
        shape=(len(values), len(values)),
    )
    _, components = connected_components(
        alternating, directed=True, connection="strong"
    )
    if np.any(np.bincount(components) > 1):
        return "nonunique", None
    root_for_pair = np.full((codes, codes), -1, dtype=int)
    root_for_pair[left, right] = matching
    return "unique", root_for_pair


def _assign_groups(local_classes, root_for_pair, entries):
    """Propagate ordered-pair labels within each neuron."""
    first, second = np.where(root_for_pair >= 0)
    first_for_root = np.empty(len(first), dtype=int)
    second_for_root = np.empty(len(first), dtype=int)
    first_for_root[root_for_pair[first, second]] = first
    second_for_root[root_for_pair[first, second]] = second
    labels = np.full(entries, -1, dtype=int)
    free = []
    for neuron, (positions, classes) in enumerate(local_classes):
        if np.all(classes == -1):
            free.append((neuron, positions))
            continue
        different = np.flatnonzero(classes[:, 0] != -1)
        if not len(different):
            return None, None
        anchor_root = classes[different[0], 0]
        anchor_code = second_for_root[anchor_root]
        local_labels = np.full(len(positions), anchor_code, dtype=int)
        for index, root in enumerate(classes[:, 0]):
            if root < 0:
                continue
            if second_for_root[root] != anchor_code:
                return None, None
            local_labels[index] = first_for_root[root]
        if not np.array_equal(
            root_for_pair[np.ix_(local_labels, local_labels)], classes
        ):
            return None, None
        labels[positions] = local_labels
    return labels, free


def _fit_log_values(logs, neurons, labels, codes, epsilon):
    """Verify a primal fit, using Chebyshev linear programs when needed."""
    entries, tasks = logs.shape
    hidden = int(neurons.max()) + 1
    design = np.zeros((entries, codes + hidden))
    design[np.arange(entries), labels] = 1
    design[np.arange(entries), codes + neurons] = -1
    reduced = design[:, 1:]
    numerical = np.empty((codes + hidden, tasks))
    numerical[0] = 0
    numerical[1:] = np.linalg.lstsq(reduced, logs, rcond=None)[0]
    residual = float(np.max(np.abs(design @ numerical - logs)))
    if residual <= epsilon:
        return {
            "state": "feasible",
            "code_logs": numerical[:codes].T,
            "log_scales": numerical[codes:].T,
            "max_log_residual": residual,
            "fit_method": "float64_primal_least_squares",
        }

    objective = np.r_[np.zeros(codes + hidden), 1.0]
    constraints = np.r_[
        np.c_[design, -np.ones(entries)],
        np.c_[-design, -np.ones(entries)],
    ]
    bounds = [(0, 0)] + [(None, None)] * (codes + hidden - 1) + [(0, None)]
    optima = []
    for task in range(tasks):
        fit = linprog(
            objective,
            A_ub=constraints,
            b_ub=np.r_[logs[:, task], -logs[:, task]],
            bounds=bounds,
            method="highs",
            options={
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
            },
        )
        if not fit.success:
            return {"state": "numerically_unresolved", "solver": fit.message}
        numerical[:, task] = fit.x[:-1]
        optima.append(float(fit.fun))
    residual = float(np.max(np.abs(design @ numerical - logs)))
    if residual <= epsilon:
        state = "feasible"
    elif max(optima) > epsilon + 1e-8:
        state = "infeasible"
    else:
        state = "numerically_unresolved"
    return {
        "state": state,
        "code_logs": numerical[:codes].T,
        "log_scales": numerical[codes:].T,
        "max_log_residual": residual,
        "minimax_objective": max(optima),
        "fit_method": "float64_highs",
    }


def _complete_family(
    logs, neurons, labels, free_neurons, codes, epsilon, signs
):
    """Apply signs and materialize one member of a compact family."""
    tasks = logs.shape[1]
    code_profiles = np.ones((codes, tasks), dtype=int)
    profiles = None if signs is None else signs * signs[:, :1]
    if profiles is not None:
        for code in range(codes):
            members = profiles[labels == code]
            if not len(members) or np.any(members != members[0]):
                return None, "incompatible_sign_profiles"
            code_profiles[code] = members[0]
    choices = []
    indexator = labels.copy()
    for neuron, positions in free_neurons:
        if profiles is None:
            allowed = np.arange(codes)
        else:
            members = profiles[positions]
            if np.any(members != members[0]):
                return None, "incompatible_constant_neuron_signs"
            allowed = np.flatnonzero(
                np.all(code_profiles == members[0], axis=1)
            )
        if not len(allowed):
            return None, "no_sign_compatible_code"
        indexator[positions] = allowed[0]
        choices.append(
            {
                "neuron": neuron,
                "positions": positions.tolist(),
                "allowed_codes": allowed.tolist(),
            }
        )
    fit = _fit_log_values(logs, neurons, indexator, codes, epsilon)
    state = fit.pop("state")
    if state != "feasible":
        return None, state
    return {
        "indexator": indexator,
        "fixed_signs": (
            np.ones(len(labels), dtype=int) if signs is None else signs[:, 0]
        ),
        "code_sign_profiles": code_profiles,
        "free_neurons": choices,
        "groupings_count": math.prod(len(x["allowed_codes"]) for x in choices),
        **fit,
    }, None


def recover_dense_structure(
    log_weights,
    neuron_indices,
    epsilon,
    max_codes=None,
    signs=None,
):
    """Recover all regular, pair-complete explanations at a code budget.

    Parameters
    ----------
    log_weights : array, shape (entries,) or (entries, tasks)
        Finite log magnitudes, with the same positions across tasks.
    neuron_indices : integer array, shape (entries,)
        Receiving neuron for each position. Names need not be consecutive.
    epsilon : float
        Maximum allowed absolute error in each log-weight entry.
    max_codes : int or None
        Code budget. None infers a saturated budget from contrast components.
    signs : optional array with the same shape as log_weights
        Observed signs, restricted to -1 and +1.

    Returns
    -------
    dict
        A complete compact family when the observable regularity conditions
        hold. "inconclusive" is distinct from "infeasible_at_budget".
        Arithmetic and LP checks use float64, not interval verification.
    """
    started = time.perf_counter()
    logs = np.asarray(log_weights, dtype=float)
    if logs.ndim == 1:
        logs = logs[:, None]
    raw_neurons = np.asarray(neuron_indices)
    if (
        logs.ndim != 2
        or min(logs.shape) == 0
        or not np.isfinite(logs).all()
        or raw_neurons.shape != (len(logs),)
        or not np.issubdtype(raw_neurons.dtype, np.integer)
    ):
        raise ValueError(
            "Expected finite log profiles and integer neuron indices."
        )
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative.")
    if max_codes is not None:
        max_codes = operator.index(max_codes)
        if max_codes < 1:
            raise ValueError("max_codes must be positive.")
    if signs is not None:
        signs = np.asarray(signs)
        if signs.ndim == 1:
            signs = signs[:, None]
        if signs.shape != logs.shape or not np.isin(signs, (-1, 1)).all():
            raise ValueError(
                "Signs must match the log profiles and be -1 or +1."
            )
        signs = signs.astype(int, copy=False)
    neuron_names, neurons = np.unique(raw_neurons, return_inverse=True)
    representatives, local_classes, diameters, zero_radius, pairs = (
        _contrast_components(logs, neurons, epsilon)
    )
    nonzero = len(representatives)
    lower = (1 + math.isqrt(1 + 4 * nonzero)) // 2
    while lower * (lower - 1) < nonzero:
        lower += 1
    codes = lower if max_codes is None else max_codes
    result = {
        "status": "inconclusive",
        "complete_at_budget": False,
        "arithmetic": "float64_not_interval_certified",
        "code_budget": codes,
        "code_lower_bound": lower,
        "epsilon": float(epsilon),
        "contrast_observations": pairs,
        "contrast_components": nonzero + 1,
        "max_component_diameter": float(np.max(diameters)),
        "zero_component_radius": zero_radius,
        "neuron_names": neuron_names.tolist(),
        "families": [],
        "branch_diagnostics": [],
    }

    def finish(status, reason, complete=False):
        result["status"] = status
        result["reason"] = reason
        result["complete_at_budget"] = complete
        result["seconds"] = time.perf_counter() - started
        return result

    if nonzero > codes * (codes - 1):
        return finish(
            "infeasible_at_budget", "contrast_count_lower_bound", True
        )
    if nonzero != codes * (codes - 1):
        return finish("inconclusive", "contrast_count_not_saturated")
    if np.max(diameters) > 4 * epsilon or zero_radius > 2 * epsilon:
        return finish(
            "infeasible_at_budget", "saturated_component_too_wide", True
        )
    if codes == 1:
        labels = np.zeros(len(logs), dtype=int)
        family, failure = _complete_family(
            logs, neurons, labels, [], codes, epsilon, signs
        )
        if family is not None:
            result["families"].append(family)
            return finish("complete", "one_code_explanation", True)
        if failure == "numerically_unresolved":
            return finish("inconclusive", failure)
        return finish("infeasible_at_budget", failure, True)

    tree = cKDTree(representatives)
    anchor = representatives[0]
    if codes == 2:
        point_sets = [np.stack([np.zeros(logs.shape[1]), anchor])]
    else:
        possible = tree.query_ball_point(
            anchor - representatives, 6 * epsilon, p=np.inf
        )
        neighbours = np.array(
            [i for i, adjacent in enumerate(possible) if i != 0 and adjacent],
            dtype=int,
        )
        result["anchor_neighbours"] = len(neighbours)
        expected = 2 * (codes - 2)
        if len(neighbours) < expected:
            return finish(
                "infeasible_at_budget", "too_few_anchor_relations", True
            )
        if len(neighbours) != expected:
            return finish("inconclusive", "extra_anchor_relations")
        local = representatives[neighbours]
        graph = np.zeros((len(local), len(local)), dtype=bool)
        for i, value in enumerate(local):
            hits = tree.query_ball_point(
                value - local, 6 * epsilon, p=np.inf
            )
            graph[i] = [bool(adjacent) for adjacent in hits]
        graph |= graph.T
        np.fill_diagonal(graph, False)
        count, labels = connected_components(csr_matrix(graph), directed=False)
        sizes = np.bincount(labels)
        if (
            count != 2
            or not np.all(sizes == codes - 2)
            or not np.all(graph.sum(axis=1) == codes - 3)
        ):
            return finish("inconclusive", "anchor_graph_not_two_cliques")
        point_sets = [
            np.vstack(
                [
                    np.zeros(logs.shape[1]),
                    anchor,
                    local[labels == component],
                ]
            )
            for component in range(2)
        ]

    unresolved = False
    for branch, points in enumerate(point_sets):
        matching_state, root_for_pair = _root_matching(
            points, representatives, epsilon
        )
        diagnostic = {"branch": branch, "matching": matching_state}
        result["branch_diagnostics"].append(diagnostic)
        if matching_state == "nonunique":
            unresolved = True
            continue
        if matching_state == "absent":
            continue
        labels, free = _assign_groups(
            local_classes, root_for_pair, len(logs)
        )
        if labels is None:
            diagnostic["assignment"] = "inconsistent"
            continue
        family, failure = _complete_family(
            logs, neurons, labels, free, codes, epsilon, signs
        )
        if family is None:
            diagnostic["assignment"] = failure
            unresolved |= failure == "numerically_unresolved"
            continue
        family["branch"] = branch
        result["families"].append(family)
        diagnostic["assignment"] = "feasible"

    if unresolved:
        return finish("inconclusive", "unresolved_matching_or_numerics")
    if result["families"]:
        return finish("complete", "all_regular_branches_resolved", True)
    return finish(
        "infeasible_at_budget", "all_regular_branches_rejected", True
    )


def recover_repeated_pair_partition(log_weights, neuron_indices, epsilon):
    """Merge positions using matching local contrasts, without saturation.

    Under log error at most epsilon and delta_4 > 8 * epsilon, every
    returned block lies in one true code group. The blocks equal the
    connected components of that group's repeated-pair occurrence graph.
    This is a conditional recovery result, not a per-instance certificate.
    """
    logs = np.asarray(log_weights, dtype=float)
    if logs.ndim == 1:
        logs = logs[:, None]
    raw_neurons = np.asarray(neuron_indices)
    if (
        logs.ndim != 2
        or min(logs.shape) == 0
        or not np.isfinite(logs).all()
        or raw_neurons.shape != (len(logs),)
        or not np.issubdtype(raw_neurons.dtype, np.integer)
    ):
        raise ValueError(
            "Expected finite profiles and integer neuron indices."
        )
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative.")
    _, neurons = np.unique(raw_neurons, return_inverse=True)
    representatives, local_classes, diameters, zero_radius, pairs = (
        _contrast_components(logs, neurons, epsilon)
    )
    first, second, classes = [], [], []
    zero_first, zero_second = [], []
    for positions, local in local_classes:
        left, right = np.indices(local.shape).reshape(2, -1)
        local = local.ravel()
        nonzero = local >= 0
        first.append(positions[left[nonzero]])
        second.append(positions[right[nonzero]])
        classes.append(local[nonzero])
        zero_first.append(positions[left[~nonzero]])
        zero_second.append(positions[right[~nonzero]])
    first = np.concatenate(first)
    second = np.concatenate(second)
    classes = np.concatenate(classes)
    if len(classes):
        _, initial = np.unique(classes, return_index=True)
        first_anchor = first[initial]
        second_anchor = second[initial]
        rows = np.r_[
            first_anchor[classes],
            second_anchor[classes],
            np.concatenate(zero_first),
        ]
        columns = np.r_[
            first, second, np.concatenate(zero_second)
        ]
    else:
        rows = np.concatenate(zero_first)
        columns = np.concatenate(zero_second)
    graph = csr_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(len(logs), len(logs)),
    )
    count, partition = connected_components(graph, directed=False)
    return {
        "indexator": partition,
        "groups": count,
        "contrast_components": len(representatives) + 1,
        "contrast_observations": pairs,
        "max_component_diameter": float(np.max(diameters)),
        "zero_component_radius": zero_radius,
        "arithmetic": "float64_not_interval_certified",
    }
