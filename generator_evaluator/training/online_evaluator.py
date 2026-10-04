"""Bounded replay views for online evaluator refreshes.

The view borrows a :class:`RealReplay`'s mask/context stores and replaces only
its row list.  Training rows are sampled into a balanced, fixed-size set while
validation rows retain their original replay split and topology grouping.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Sequence
from copy import copy
from typing import Any

import torch

from generator_evaluator.data.types import RealReplay


def _sample_rows(rows: Sequence[dict[str, Any]], count: int,
                 rng: torch.Generator) -> list[dict[str, Any]]:
    """Sample exactly ``count`` rows, using replacement only to fill a short pool."""
    if not rows or count < 1:
        return []
    if len(rows) >= count:
        indices = torch.randperm(len(rows), generator=rng)[:count].tolist()
    else:
        indices = torch.randint(len(rows), (count,), generator=rng).tolist()
    return [rows[index] for index in indices]


def _topology_groups(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["topology_id"])].append(row)
    return dict(grouped)


def build_online_evaluator_replay(
    replay: RealReplay,
    initial_topology_ids: Collection[str],
    task_ids: Collection[str],
    bank_rows: int = 512,
    seed: int = 0,
    active_edges: int | None = None,
) -> tuple[RealReplay | None, dict[str, Any]]:
    """Create a balanced old/new train view and topology-balanced validation view.

    Original training rows come from the initial bank. New training rows must
    have an ``acquisition:*`` origin, belong to an allowed train task, and have
    a non-initial topology. Both pools contribute exactly ``bank_rows`` rows;
    a pool shorter than that is sampled with replacement. No replay split is
    recalculated or mutated.

    For ``mask_validation``, all eligible original and acquired heldout
    topology groups are kept with all their eligible rows, sorted by topology
    ID. Original validation is therefore stable across refreshes, independent
    of the train sampler seed. The returned ``None`` signals that no acquired
    training rows or no original source rows were eligible, so the caller can
    skip this refresh.
    """
    if not isinstance(replay, RealReplay):
        raise TypeError("replay must be a RealReplay")
    if bank_rows < 1:
        raise ValueError("bank_rows must be positive")
    if active_edges is not None and active_edges < 1:
        raise ValueError("active_edges must be positive when provided")

    initial_ids = set(initial_topology_ids)
    allowed_tasks = set(task_ids)
    if any(not isinstance(value, str) or not value for value in initial_ids):
        raise ValueError("initial_topology_ids must contain nonempty strings")
    if any(not isinstance(value, str) or not value for value in allowed_tasks):
        raise ValueError("task_ids must contain nonempty strings")

    records = replay.records
    heldout_ids = {
        str(row["topology_id"])
        for row in records
        if row.get("split") in ("mask_validation", "joint_validation")
    }

    def allowed_train_task(row: dict[str, Any]) -> bool:
        return row.get("task_split") == "train" and row.get("task_id") in allowed_tasks

    original_train = [
        row for row in records
        if row.get("split") == "train"
        and row.get("topology_id") in initial_ids
        and row.get("topology_id") not in heldout_ids
        and not str(row.get("origin", "")).startswith("acquisition:")
        and allowed_train_task(row)
    ]
    acquired_train = [
        row for row in records
        if row.get("split") == "train"
        and str(row.get("origin", "")).startswith("acquisition:")
        and row.get("topology_id") not in initial_ids
        and row.get("topology_id") not in heldout_ids
        and allowed_train_task(row)
    ]

    metadata: dict[str, Any] = {
        "seed": int(seed),
        "bank_rows_per_source": int(bank_rows),
        "active_edges": active_edges,
        "initial_topology_count": len(initial_ids),
        "allowed_train_task_count": len(allowed_tasks),
        "eligible_original_train_rows": len(original_train),
        "eligible_acquired_train_rows": len(acquired_train),
        "eligible_original_train_origins": dict(Counter(str(row.get("origin", ""))
                                                         for row in original_train)),
        "eligible_acquired_train_origins": dict(Counter(str(row.get("origin", ""))
                                                         for row in acquired_train)),
    }
    if not acquired_train or not original_train:
        metadata.update(
            sampled_original_train_rows=0,
            sampled_acquired_train_rows=0,
            reason=("no_eligible_acquired_train_rows" if not acquired_train else
                    "no_eligible_original_train_rows"),
        )
        return None, metadata

    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    sampled_original = _sample_rows(original_train, bank_rows, rng)
    sampled_acquired = _sample_rows(acquired_train, bank_rows, rng)
    metadata.update(
        sampled_original_train_rows=len(sampled_original),
        sampled_acquired_train_rows=len(sampled_acquired),
        sampled_original_train_unique_rows=len({id(row) for row in sampled_original}),
        sampled_acquired_train_unique_rows=len({id(row) for row in sampled_acquired}),
    )

    original_mask_validation = [
        row for row in records
        if row.get("split") == "mask_validation"
        and row.get("topology_id") in initial_ids
        and allowed_train_task(row)
        and (active_edges is None or row.get("active_edges") == active_edges)
    ]
    acquired_mask_validation = [
        row for row in records
        if row.get("split") == "mask_validation"
        and str(row.get("origin", "")).startswith("acquisition:")
        and row.get("topology_id") not in initial_ids
        and allowed_train_task(row)
        and (active_edges is None or row.get("active_edges") == active_edges)
    ]
    # Keep every eligible topology exactly once and all its rows. Sorting keeps
    # the initial-bank validation pool identical on each online refresh.
    original_groups = _topology_groups(original_mask_validation)
    acquired_groups = _topology_groups(acquired_mask_validation)
    original_topologies = sorted(original_groups)
    acquired_topologies = sorted(acquired_groups)
    mask_validation_rows = [
        row
        for identity in original_topologies + acquired_topologies
        for row in (original_groups if identity in original_groups else acquired_groups)[identity]
    ]
    metadata.update(
        eligible_original_mask_validation_topologies=len(original_groups),
        eligible_acquired_mask_validation_topologies=len(acquired_groups),
        selected_original_mask_validation_topologies=len(original_topologies),
        selected_acquired_mask_validation_topologies=len(acquired_topologies),
        selected_original_mask_validation_rows=sum(len(original_groups[key]) for key in original_topologies),
        selected_acquired_mask_validation_rows=sum(len(acquired_groups[key]) for key in acquired_topologies),
        selected_mask_validation_rows=sum(len(original_groups[key]) for key in original_topologies)
                                     + sum(len(acquired_groups[key]) for key in acquired_topologies),
    )
    if not mask_validation_rows:
        metadata["reason"] = "no_eligible_mask_validation_rows"
        return None, metadata

    # These fixed validation partitions are report-only. Preserve their rows
    # only when they refer to an explicitly allowed task, and never add an
    # acquired row here: acquired measurements are restricted to train tasks.
    meta_validation = [
        row for row in records
        if row.get("split") == "meta_validation"
        and row.get("topology_id") in initial_ids
        and row.get("task_id") in allowed_tasks
    ]
    joint_validation = [
        row for row in records
        if row.get("split") == "joint_validation"
        and row.get("topology_id") in initial_ids
        and row.get("task_id") in allowed_tasks
    ]
    metadata["selected_meta_validation_rows"] = len(meta_validation)
    metadata["selected_joint_validation_rows"] = len(joint_validation)

    view = copy(replay)
    view.records = sampled_original + sampled_acquired + mask_validation_rows + meta_validation + joint_validation
    view.evaluator_validation_sources = {
        **{identity: "original" for identity in original_topologies},
        **{identity: "acquired" for identity in acquired_topologies},
    }
    return view, metadata
