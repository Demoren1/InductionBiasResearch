"""Recover a shared signed indexator from estimated dense weights."""

from __future__ import annotations

import operator
import time

import numpy as np

from experiments.constructive_dense_recovery_v1.recovery import (
    recover_dense_structure,
    recover_repeated_pair_partition,
)


def recover_from_dense(
    weights, absolute_error, minimum_active, codes, use_root_fallback=True
):
    """Recover from a stated absolute error bound in aligned dense weights.

    weights has shape (tasks, inputs, neurons). minimum_active is a
    deterministic lower bound on nonzero true magnitudes, not an estimate
    taken from hidden weights. The success probabilities and assumptions
    are in paper/theory/DENSE_RECOVERY_SUCCESS_2026-11-15.md.
    """
    started = time.perf_counter()
    weights = np.asarray(weights, dtype=float)
    codes = operator.index(codes)
    if (
        weights.ndim != 3
        or min(weights.shape) == 0
        or not np.isfinite(weights).all()
        or codes < 1
    ):
        raise ValueError("Expected finite task-by-input-by-neuron weights.")
    if (
        not np.isfinite(absolute_error)
        or absolute_error < 0
        or not np.isfinite(minimum_active)
        or minimum_active <= 2 * absolute_error
    ):
        raise ValueError("Require minimum_active > 2 * absolute_error >= 0.")
    epsilon = float(-np.log1p(-absolute_error / minimum_active))
    task_masks = np.abs(weights) > absolute_error
    result = {
        "status": "inconclusive",
        "epsilon": epsilon,
        "absolute_error_bound": float(absolute_error),
        "minimum_active_bound": float(minimum_active),
        "arithmetic": "float64_not_interval_certified",
    }

    def finish(reason, index=None, method=None):
        result["reason"] = reason
        result["seconds"] = time.perf_counter() - started
        if index is not None:
            result["status"] = "candidate"
            result["index"] = index
            result["method"] = method
        return result

    if np.any(task_masks != task_masks[:1]):
        return finish("support_disagrees_across_tasks")
    mask = task_masks[0]
    rows, neurons = np.where(mask)
    if not len(rows):
        return finish("empty_support")
    profiles = weights[:, rows, neurons].T
    signs = np.sign(profiles).astype(int)
    normalized = signs * signs[:, :1]
    sign_profiles, sign_groups = np.unique(
        normalized, axis=0, return_inverse=True
    )
    result["active_positions"] = len(rows)
    result["sign_profiles"] = len(sign_profiles)

    def signed_index(labels):
        index = np.zeros(mask.shape, dtype=np.int16)
        index[rows, neurons] = (labels + 1) * signs[:, 0]
        return index

    if len(sign_profiles) == codes:
        return finish(
            "all_sign_profiles_separated",
            signed_index(sign_groups),
            "sign_profiles",
        )
    logs = np.log(np.abs(profiles))
    pair = recover_repeated_pair_partition(logs, neurons, epsilon)
    labels = pair.pop("indexator")
    result["pair_diagnostics"] = pair
    if pair["groups"] == codes:
        consistent = all(
            np.all(normalized[labels == code] == normalized[labels == code][0])
            for code in range(codes)
        )
        if consistent:
            return finish(
                "pair_partition_at_budget", signed_index(labels), "pair_union"
            )
    if not use_root_fallback:
        return finish("pair_partition_incomplete_or_inconsistent")
    root = recover_dense_structure(
        logs, neurons, epsilon, max_codes=codes, signs=signs
    )
    result["root_status"] = root["status"]
    result["root_reason"] = root["reason"]
    if root["status"] == "complete":
        count = sum(family["groupings_count"] for family in root["families"])
        result["root_groupings_count"] = count
        if count == 1:
            labels = root["families"][0]["indexator"]
            return finish(
                "unique_root_family", signed_index(labels), "root_complete"
            )
        return finish("multiple_root_explanations")
    return finish("root_criterion_not_resolved")
