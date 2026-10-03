"""Data boundaries for cooperative pattern-mask generators.

This module builds one functional bank per *training* pattern.  The banks
share a fixed, label-free probe, while the quality evaluator sees measurements
from both patterns through ordinary :class:`~generator_evaluator.data.TaskData`
objects.  The held-out pattern is deliberately represented by a small opaque
specification until final evaluation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .adapters import (
    FunctionalBank,
    _balanced_indices,
    _exact_topk,
    _fit_pattern,
    _pattern_labels,
    _pattern_table,
    _random_pattern_masks,
    _uniform_indices,
)
from .data import InnerProtocol, TaskData, support_context, tensor_hash
from .progress import progress


_FEATURES, _HIDDEN = 11, 8
_EDGE_COUNT = _FEATURES * _HIDDEN
_DENSITIES = (round(.10 * _EDGE_COUNT), "target", round(.50 * _EDGE_COUNT),
              round(.70 * _EDGE_COUNT), _EDGE_COUNT)


def _validate_pattern(pattern: str, name: str) -> None:
    if not isinstance(pattern, str) or len(pattern) != 4 or set(pattern) - {"0", "1"}:
        raise ValueError(f"{name} must be a four-bit binary pattern")


def _orbit(pattern: str) -> frozenset[str]:
    complement = "".join("1" if bit == "0" else "0" for bit in pattern)
    return frozenset((pattern, pattern[::-1], complement, complement[::-1]))


def _validate_roles(train_patterns: tuple[str, ...], test_pattern: str | tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(train_patterns, tuple) or len(train_patterns) < 2:
        raise ValueError("train_patterns must contain at least two patterns")
    for number, pattern in enumerate(train_patterns):
        _validate_pattern(pattern, f"train_patterns[{number}]")
    if len(set(train_patterns)) != len(train_patterns):
        raise ValueError("training patterns must be unique")
    # Preserve the original two-task requirement. Larger cooperative runs may
    # include multiple patterns from the same reversal/complement orbit.
    if len(train_patterns) == 2 and _orbit(train_patterns[0]) == _orbit(train_patterns[1]):
        raise ValueError("training patterns must belong to distinct reversal/complement orbits")
    if isinstance(test_pattern, str):
        test_patterns = (test_pattern,)
    elif isinstance(test_pattern, tuple) and len(test_pattern) >= 2:
        test_patterns = test_pattern
    else:
        raise ValueError("test_pattern must be a pattern string or a tuple of at least two patterns")
    for number, pattern in enumerate(test_patterns):
        _validate_pattern(pattern, f"test_patterns[{number}]")
    if len(set(test_patterns)) != len(test_patterns):
        raise ValueError("test patterns must be unique")
    train_orbits = {_orbit(pattern) for pattern in train_patterns}
    for number, pattern in enumerate(test_patterns):
        if _orbit(pattern) in train_orbits:
            raise ValueError("test_pattern must belong to an orbit distinct from every training pattern")
    return test_patterns


def _cooperative_partitions(seed: int) -> dict[str, Tensor]:
    """Frozen observation IDs, with a separate final-test pool.

    The pattern universe has 2048 observations.  The test pool contains both
    its support and query rows; the other pools are exclusively non-test.
    """
    ids, _ = _pattern_table()
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(int(seed) + 19_731))
    sizes = (256, 128, 128, 512, 384, 128, 256, 256)
    names = ("bank_support", "bank_query", "probe", "evaluator_support",
             "evaluator_query", "selection", "test_support", "test_query")
    result: dict[str, Tensor] = {}
    offset = 0
    for name, size in zip(names, sizes):
        result[name] = order[offset:offset + size]
        offset += size
    if offset != len(ids):  # Keep this assertion coupled to the exhaustive table.
        raise AssertionError("cooperative partitions must exhaust the pattern universe")
    return result


def _require_count(name: str, value: int, limit: int, *, minimum: int = 1) -> None:
    if not isinstance(value, int) or not minimum <= value <= limit:
        raise ValueError(f"{name} must be an integer in [{minimum}, {limit}], got {value!r}")


def _available_measurement_devices(device: str,
                                  measurement_devices: Sequence[str] | None) -> tuple[str, tuple[str, ...]]:
    requested = tuple(str(value) for value in measurement_devices) if measurement_devices else (str(device),)
    usable: list[str] = []
    for value in requested:
        try:
            target = torch.device(value)
        except (ValueError, RuntimeError):
            continue
        if target.type == "cuda" and (not torch.cuda.is_available() or
                                      (target.index is not None and target.index >= torch.cuda.device_count())):
            continue
        if target.type in ("cpu", "cuda"):
            normalized = str(target)
            if normalized not in usable:
                usable.append(normalized)
    if not usable:
        usable = ["cpu"]
    # The spawned path is deliberately CUDA-only. For mixed lists retain the
    # first viable target as the established one-device fallback.
    if any(value.startswith("cuda") for value in usable) and any(value == "cpu" for value in usable):
        usable = [usable[0]]
    return usable[0], tuple(usable)


def _make_train_task(pattern: str, parts: dict[str, Tensor], support_count: int,
                     query_count: int, selection_count: int, seed: int) -> tuple[TaskData, TaskData]:
    ids, x = _pattern_table()
    y = _pattern_labels(x, pattern)
    support = _balanced_indices(y, parts["evaluator_support"], support_count, seed + 1)
    # Support is balanced to make the fitted child stable.  Every scoring
    # query is a uniform draw from its reserved pool, so replay, selection,
    # and sealed-test quality have the same class-prior convention.
    query = _uniform_indices(parts["evaluator_query"], query_count, seed + 2)
    selection_query = _uniform_indices(parts["selection"], selection_count, seed + 3)
    context = support_context(x[support], y[support])
    common = dict(x_support=x[support], y_support=y[support], context=context,
                  support_ids=ids[support], provenance={"family": "pattern", "pattern": pattern,
                  "support_partition": "evaluator_support"})
    train = TaskData(task_id=f"pattern:{pattern}", split="train", x_query=x[query], y_query=y[query],
                     query_ids=ids[query], provenance={**common["provenance"],
                     "query_partition": "evaluator_query"}, **{key: value for key, value in common.items()
                                                                   if key != "provenance"})
    selection = TaskData(task_id=f"pattern:{pattern}:selection", split="validation",
                         x_query=x[selection_query], y_query=y[selection_query], query_ids=ids[selection_query],
                         provenance={**common["provenance"], "query_partition": "selection",
                                     "role": "selection"},
                         **{key: value for key, value in common.items() if key != "provenance"})
    return train, selection


def extract_pattern_tokens(state: dict[str, Tensor], mask: Tensor, probe_x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Make normalized generator tokens from a terminal pattern-child state.

    ``psi`` and all ``q_*`` entries in the returned raw profile are unscaled.
    The returned token concatenates normalized full probe activations, signed,
    absolute and RMS functional maps, and the binary mask.  Initial teachers
    and feedback always pass through this one function.
    """
    if not isinstance(state, dict) or any(name not in state for name in ("w", "b", "a", "c")):
        raise ValueError("pattern state must contain terminal w, b, a and c tensors")
    weight = torch.as_tensor(state["w"], dtype=torch.float32).detach().cpu()
    bias = torch.as_tensor(state["b"], dtype=torch.float32).detach().cpu()
    readout = torch.as_tensor(state["a"], dtype=torch.float32).detach().cpu()
    offset = torch.as_tensor(state["c"], dtype=torch.float32).detach().cpu()
    mask = torch.as_tensor(mask, dtype=torch.float32).detach().cpu()
    probe_x = torch.as_tensor(probe_x, dtype=torch.float32).detach().cpu()
    if weight.shape != (_FEATURES, _HIDDEN) or mask.shape != weight.shape:
        raise ValueError("pattern state and mask must have shape [11, 8]")
    if bias.shape != (_HIDDEN,) or readout.shape != (_HIDDEN,) or offset.numel() != 1:
        raise ValueError("pattern state biases/readout must have shape [8]")
    if probe_x.ndim != 2 or probe_x.shape[1] != _FEATURES or len(probe_x) < 1:
        raise ValueError("probe_x must have shape [probe_rows, 11]")
    if not all(torch.isfinite(item).all() for item in (weight, bias, readout, offset, mask, probe_x)):
        raise ValueError("functional profile inputs must be finite")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("pattern masks must be binary")
    effective = weight * mask
    preactivation = probe_x @ effective + bias
    psi = F.relu(preactivation) * readout
    q = probe_x[:, :, None] * effective[None] * readout[None, None] * (preactivation > 0)[:, None]
    signed, absolute, rms = q.mean(0), q.abs().mean(0), q.square().mean(0).sqrt()
    psi_scale = psi.square().mean().sqrt().clamp_min(1e-8)
    q_scale = rms.amax().clamp_min(1e-8)
    tokens = torch.cat((psi.div(psi_scale).T, signed.div(q_scale).T, absolute.div(q_scale).T,
                        rms.div(q_scale).T, mask.T), dim=1).contiguous()
    raw = {"psi": psi.contiguous(), "q_signed": signed.contiguous(), "q_abs": absolute.contiguous(),
           "q_rms": rms.contiguous(), "psi_scale": psi_scale.reshape(1), "q_scale": q_scale.reshape(1),
           "effective_weights": effective.contiguous()}
    return tokens, raw


def _state_row_hash(state: dict[str, Tensor], mask: Tensor) -> str:
    if not isinstance(state, dict) or any(name not in state for name in ("w", "b", "a", "c")):
        raise ValueError("terminal feedback state must contain w, b, a and c")
    digest = hashlib.sha256()
    for name in ("w", "b", "a", "c"):
        digest.update(tensor_hash(torch.as_tensor(state[name], dtype=torch.float32)).encode())
    digest.update(tensor_hash(torch.as_tensor(mask, dtype=torch.float32)).encode())
    return digest.hexdigest()


def _teacher_masks(count: int, k: int, seed: int) -> Tensor:
    if count < len(_DENSITIES):
        raise ValueError("teachers_per_pattern must be at least 5 to retain sparse, target, mid, and dense anchors")
    base = [k if density == "target" else int(density) for density in _DENSITIES]
    counts = [base[index % len(base)] for index in range(count)]
    return torch.cat([_random_pattern_masks(1, edges, seed + row) if edges < _EDGE_COUNT
                      else torch.ones(1, _FEATURES, _HIDDEN)
                      for row, edges in enumerate(counts)], dim=0)


def _candidate_density_buckets(k: int, retained: int) -> tuple[int, ...]:
    """Return the density strata used only by the large teacher search.

    The normal small-bank path retains its original five anchors.  Candidate
    search uses ten strata, with the requested target cardinality replacing
    the nearest nominal density when it is not already represented.
    """
    if retained < 10:
        return tuple(k if density == "target" else int(density) for density in _DENSITIES)
    buckets = [round(_EDGE_COUNT * rho) for rho in (.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.)]
    if k not in buckets:
        nearest = min(range(len(buckets)), key=lambda index: (abs(buckets[index] - k), index))
        buckets[nearest] = k
    return tuple(sorted(buckets))


def _stratum_counts(total: int, buckets: tuple[int, ...]) -> list[int]:
    """Allocate a fixed total as evenly as possible, deterministically."""
    quotient, remainder = divmod(total, len(buckets))
    return [quotient + int(index < remainder) for index in range(len(buckets))]


def _candidate_masks(count: int, k: int, seed: int, retained: int) -> tuple[Tensor, list[int], list[int]]:
    """Generate mixed-density candidates with stable IDs and stratum labels."""
    buckets = _candidate_density_buckets(k, retained)
    masks: list[Tensor] = []
    candidate_ids: list[int] = []
    strata: list[int] = []
    candidate_id = 0
    for edges, amount in zip(buckets, _stratum_counts(count, buckets)):
        for _ in range(amount):
            masks.append(_random_pattern_masks(1, edges, seed + candidate_id)
                         if edges < _EDGE_COUNT else torch.ones(1, _FEATURES, _HIDDEN))
            candidate_ids.append(candidate_id)
            strata.append(edges)
            candidate_id += 1
    return torch.cat(masks, dim=0), candidate_ids, strata


def _candidate_task(pattern: str, labels: Tensor, ids: Tensor, x: Tensor, parts: dict[str, Tensor], *,
                    seed: int, candidate_id: int, fixed_query: Tensor,
                    support_count: int) -> TaskData:
    """A candidate's support fit and a shared, bank-reserved ranking query."""
    support = _balanced_indices(labels, parts["bank_support"], support_count,
                                seed + 101 * candidate_id)
    return TaskData(f"bank:{pattern}:candidate:{candidate_id}", "train",
                    x[support], labels[support], x[fixed_query], labels[fixed_query],
                    support_context(x[support], labels[support]), ids[support], ids[fixed_query],
                    {"family": "pattern", "pattern": pattern, "role": "bank_teacher_candidate",
                     "query_partition": "bank_query"})


def _candidate_initialization_seed(seed: int, candidate_id: int) -> int:
    return int(seed) + 1_000_003 * (int(candidate_id) + 1)


def _balanced_query_budget(labels: Tensor, candidates: Tensor, requested: int = 64) -> int:
    """Use the largest balanced source query up to the established budget."""
    return _balanced_pool_budget(labels, candidates, requested, pool_name="bank query", sample_name="query")


def _balanced_pool_budget(labels: Tensor, candidates: Tensor, requested: int, *,
                          pool_name: str, sample_name: str) -> int:
    """Return the largest even class-balanced count available up to ``requested``."""
    values = labels[candidates]
    positives = int((values > .5).sum())
    negatives = len(values) - positives
    budget = min(requested, 2 * min(positives, negatives))
    if budget < 2:
        raise ValueError(f"{pool_name} pool cannot supply a balanced source {sample_name}")
    return budget


def _balanced_support_budget(labels: Tensor, candidates: Tensor) -> int:
    """Use the entire bank-support pool up to its largest balanced sample."""
    return _balanced_pool_budget(labels, candidates, min(256, len(candidates)),
                                 pool_name="bank support", sample_name="support")


def _centroid_align_pattern_columns(values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Canonicalize one absolute functional map by input-coordinate centroid.

    Pattern inputs are ordered sequences. Each hidden column is sorted by the
    centroid of its absolute response over those coordinates. Exact centroid
    ties are broken lexicographically by the functional column itself, making
    the result independent of the incoming hidden-column permutation. Equal
    columns are interchangeable because they produce identical aligned maps.
    """
    values = torch.as_tensor(values, dtype=torch.float32).detach().cpu()
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("functional alignment needs a non-empty [features, hidden] map")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("centroid alignment requires finite non-negative functional maps")

    work = values.to(torch.float64)
    coordinates = torch.arange(values.shape[0], dtype=torch.float64)[:, None]
    mass = work.sum(0)
    centroids = torch.where(mass > 0, (work * coordinates).sum(0) / mass,
                            torch.full_like(mass, float("inf")))
    order = sorted(range(values.shape[1]),
                   key=lambda column: (float(centroids[column]),
                                       tuple(float(value) for value in work[:, column])))
    order_tensor = torch.tensor(order, dtype=torch.long)
    return values[:, order_tensor].contiguous(), order_tensor, centroids


def _centroid_align_pattern_maps(q_abs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Align absolute maps to ordered input coordinates for their consensus."""
    maps = torch.as_tensor(q_abs, dtype=torch.float32).detach().cpu()
    if maps.ndim != 3 or maps.shape[0] < 1:
        raise ValueError("functional maps must have shape [source_maps, features, hidden]")
    aligned, orders, centroids = [], [], []
    for values in maps:
        canonical, order, center = _centroid_align_pattern_columns(values)
        aligned.append(canonical)
        orders.append(order)
        centroids.append(center)
    return torch.stack(aligned), torch.stack(orders), torch.stack(centroids)


def _preflight_bank_query_pools(train_patterns: tuple[str, ...], parts: dict[str, Tensor]) -> dict[str, int]:
    """Check every training source query before any bank teacher is fitted."""
    _, x = _pattern_table()
    budgets: dict[str, int] = {}
    for pattern in train_patterns:
        labels = _pattern_labels(x, pattern)
        budgets[pattern] = _balanced_query_budget(labels, parts["bank_query"])
    return budgets


def _preflight_bank_support_pools(train_patterns: tuple[str, ...], parts: dict[str, Tensor]) -> dict[str, int]:
    """Record each pattern's maximum balanced sample from reserved bank support."""
    _, x = _pattern_table()
    budgets: dict[str, int] = {}
    for pattern in train_patterns:
        labels = _pattern_labels(x, pattern)
        budgets[pattern] = _balanced_support_budget(labels, parts["bank_support"])
    return budgets


def _build_bank(pattern: str, parts: dict[str, Tensor], *, seed: int, bank_steps: int,
                teachers_per_pattern: int, k: int, probe_x: Tensor, probe_ids: Tensor,
                device: str, batch_teachers: bool = False, bank_candidates: int | None = None,
                teacher_batch_size: int = 128,
                measurement_devices: tuple[str, ...] | None = None) -> FunctionalBank:
    """Build a bank, optionally selecting retained teachers from a larger search."""
    if bank_candidates is not None:
        return _build_selected_bank(pattern, parts, seed=seed, bank_steps=bank_steps,
                                    teachers_per_pattern=teachers_per_pattern, bank_candidates=bank_candidates,
                                    k=k, probe_x=probe_x, probe_ids=probe_ids, device=device,
                                    teacher_batch_size=teacher_batch_size,
                                    measurement_devices=measurement_devices or (device,))
    ids, x = _pattern_table()
    labels = _pattern_labels(x, pattern)
    bank_query_count = _balanced_query_budget(labels, parts["bank_query"])
    bank_support_count = _balanced_support_budget(labels, parts["bank_support"])
    masks = _teacher_masks(teachers_per_pattern, k, seed)
    protocol = InnerProtocol(steps=bank_steps, replicas=1, lr=.03, l2=.001,
                             checkpoint_every=max(1, bank_steps // 4), seed=seed)
    tokens: list[Tensor] = []
    states: list[dict[str, Any]] = []
    tasks = []
    for row in range(teachers_per_pattern):
        support = _balanced_indices(labels, parts["bank_support"], bank_support_count, seed + 101 * row)
        query = _balanced_indices(labels, parts["bank_query"], bank_query_count, seed + 101 * row + 1)
        task = TaskData(f"bank:{pattern}:{row}", "train", x[support], labels[support], x[query], labels[query],
                        support_context(x[support], labels[support]), ids[support], ids[query],
                        {"family": "pattern", "pattern": pattern, "role": "bank_teacher"})
        tasks.append(task)
    if batch_teachers:
        from .pattern_batch import fit_pattern_batch
        fits = fit_pattern_batch(masks, tasks, protocol, device)
    else:
        fits = [_fit_pattern(masks[row], tasks[row], protocol, device)
                for row in progress(range(teachers_per_pattern), desc=f"Bank {pattern}", unit="teacher")]
    for row, fitted in enumerate(fits):
        state = fitted["state_dict"][0]
        token, raw = extract_pattern_tokens(state, masks[row], probe_x)
        row_hash = _state_row_hash(state, masks[row])
        tokens.append(token)
        states.append({**raw, "state_dict": {name: value.detach().cpu().clone() for name, value in state.items()},
                       "optimizer_state": deepcopy(fitted["optimizer_state"][0]),
                       "history": deepcopy(fitted["history"][0]),
                       "row_hash": row_hash, "source": {"kind": "initial_teacher", "pattern": pattern,
                                                           "teacher": row, "active_edges": int(masks[row].sum())}})
    q_abs = torch.stack([item["q_abs"] for item in states])
    aligned, column_orders, column_centroids = _centroid_align_pattern_maps(q_abs)
    baseline = _exact_topk(aligned.mean(0), k)
    density_counts = {int(edges): int((masks.sum((1, 2)) == edges).sum())
                      for edges in torch.unique(masks.sum((1, 2)).long()).tolist()}
    provenance = {"family": "cooperative_pattern", "pattern": pattern, "seed": seed,
                  "baseline_k": k, "probe_ids": probe_ids.tolist(), "probe_fingerprint": tensor_hash(probe_x),
                  "partitions": {name: ids[value].tolist() for name, value in parts.items()},
                  "bank_query_count": bank_query_count, "bank_support_count": bank_support_count,
                  "teacher_density_counts": density_counts, "quality_source": None,
                  "functional_alignment_method": "input_coordinate_centroid",
                  "functional_alignment_summary": "mean absolute probe response over ordered input coordinates",
                  "functional_alignment_tie_break": "lexicographic absolute functional column values",
                  "functional_alignment_scope": "baseline only; teacher tokens and masks keep their original pairing",
                  "accepted_feedback_task_ids": [f"pattern:{pattern}"], "feedback_hashes": []}
    return FunctionalBank(torch.stack(tokens)[None], None, masks, baseline, provenance, states=states,
                          diagnostics={"aligned_q_abs": aligned, "functional_column_orders": column_orders,
                                       "functional_column_centroids": column_centroids,
                                       "probe_x": probe_x.detach().cpu().clone()})


def _build_selected_bank(pattern: str, parts: dict[str, Tensor], *, seed: int, bank_steps: int,
                         teachers_per_pattern: int, bank_candidates: int, k: int, probe_x: Tensor,
                         probe_ids: Tensor, device: str, teacher_batch_size: int,
                         measurement_devices: tuple[str, ...]) -> FunctionalBank:
    """Fit many independent candidates and retain the best fixed-query maps per stratum."""
    from .parallel_measurements import _balanced_chunk_sizes, iter_pattern_candidate_batches

    ids, x = _pattern_table()
    labels = _pattern_labels(x, pattern)
    masks, candidate_ids, strata = _candidate_masks(bank_candidates, k, seed, teachers_per_pattern)
    # This is a single immutable query set for every candidate's rank.  It is
    # part of the bank partition and is unrelated to evaluator, selection, or
    # sealed test examples.
    bank_query_count = _balanced_query_budget(labels, parts["bank_query"])
    bank_support_count = _balanced_support_budget(labels, parts["bank_support"])
    fixed_query = _balanced_indices(labels, parts["bank_query"], bank_query_count, seed + 91_009)
    tasks = [_candidate_task(pattern, labels, ids, x, parts, seed=seed, candidate_id=candidate_id,
                             fixed_query=fixed_query, support_count=bank_support_count)
             for candidate_id in candidate_ids]
    protocol = InnerProtocol(steps=bank_steps, replicas=1, lr=.03, l2=.001,
                             checkpoint_every=max(1, bank_steps // 4), seed=seed)
    buckets = _candidate_density_buckets(k, teachers_per_pattern)
    quota = _stratum_counts(teachers_per_pattern, buckets)
    best_by_density: dict[int, list[tuple[float, int, dict[str, Any]]]] = {
        density: [] for density in buckets
    }
    candidate_seeds = [_candidate_initialization_seed(seed, candidate_id)
                        for candidate_id in candidate_ids]
    fit_batches = iter_pattern_candidate_batches(
        masks, tasks, protocol, devices=measurement_devices, batch_size=teacher_batch_size,
        initialization_seeds=candidate_seeds, device=device)
    worker_count = (len(measurement_devices)
                    if len(measurement_devices) > 1 and
                    all(value.startswith("cuda") for value in measurement_devices) else 1)
    total_batches = len(_balanced_chunk_sizes(bank_candidates, teacher_batch_size, worker_count))
    for start, fitted in progress(fit_batches, total=total_batches,
                                  desc=f"Bank candidates {pattern}", unit="batch"):
        for offset, result in enumerate(fitted):
            candidate_index = start + offset
            density = strata[candidate_index]
            score = float(result["replica_losses"][0])
            retained = best_by_density[density]
            # Rank with a light terminal-state record. The candidate's full
            # history and Adam state can be released as soon as its batch is
            # discarded.
            retained.append((score, candidate_ids[candidate_index], result["state_dict"][0]))
            retained.sort(key=lambda row: (row[0], row[1]))
            del retained[quota[buckets.index(density)]:]

    for edges, limit in zip(buckets, quota):
        if len(best_by_density[edges]) != limit:
            raise ValueError("bank_candidates must provide enough candidates for every density stratum")
    selected_rows = sorted(
        [(candidate_id, state, score)
         for rows in best_by_density.values() for score, candidate_id, state in rows],
        key=lambda row: row[0])
    selected_ids = [candidate_id for candidate_id, _, _ in selected_rows]
    selection_losses = {candidate_id: score for candidate_id, _, score in selected_rows}
    candidate_index_by_id = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    selected = [candidate_index_by_id[candidate_id] for candidate_id in selected_ids]
    selected_masks = masks[selected]
    tokens: list[Tensor] = []
    q_abs_values: list[Tensor] = []
    states: list[dict[str, Any]] = []
    result_by_id = {candidate_id: state for candidate_id, state, _ in selected_rows}
    del selected_rows
    best_by_density.clear()
    for index in selected:
        candidate_id = candidate_ids[index]
        state = result_by_id.pop(candidate_id)
        token, raw = extract_pattern_tokens(state, masks[index], probe_x)
        tokens.append(token)
        q_abs_values.append(raw["q_abs"])
        # Retained source cards need the terminal parameters and provenance;
        # optimizer moments and per-step histories are only needed while
        # ranking candidates and are dropped after the winner is materialized.
        states.append({"state_dict": {name: value.detach().cpu().clone() for name, value in state.items()},
                       "row_hash": _state_row_hash(state, masks[index]),
                       "source": {"kind": "initial_teacher", "pattern": pattern, "teacher": len(states),
                                  "candidate_id": candidate_id, "active_edges": int(masks[index].sum()),
                                  "density_stratum": strata[index],
                                  "initialization_seed": candidate_seeds[candidate_id]}})
    q_abs = torch.stack(q_abs_values)
    aligned, column_orders, column_centroids = _centroid_align_pattern_maps(q_abs)
    baseline = _exact_topk(aligned.mean(0), k)
    density_counts = {int(edges): int((selected_masks.sum((1, 2)) == edges).sum())
                      for edges in torch.unique(selected_masks.sum((1, 2)).long()).tolist()}
    candidate_density_counts = {int(edges): int(sum(stratum == edges for stratum in strata)) for edges in buckets}
    provenance = {"family": "cooperative_pattern", "pattern": pattern, "seed": seed,
                  "baseline_k": k, "probe_ids": probe_ids.tolist(), "probe_fingerprint": tensor_hash(probe_x),
                  "partitions": {name: ids[value].tolist() for name, value in parts.items()},
                  "bank_query_count": bank_query_count, "bank_support_count": bank_support_count,
                  "teacher_density_counts": density_counts, "candidate_count": bank_candidates,
                  "candidate_density_counts": candidate_density_counts,
                  "selected_candidate_ids": selected_ids,
                  "selected_candidate_query_losses": selection_losses,
                  "selected_initialization_seeds": [_candidate_initialization_seed(seed, candidate_ids[index])
                                                   for index in selected],
                  "candidate_seed_rule": "seed + 1000003 * (candidate_id + 1)",
                  "selection_rule": "lowest terminal BCE on shared bank_query, per density stratum; candidate ID breaks ties",
                  "selection_density_buckets": list(buckets), "selection_density_quotas": quota,
                  "functional_alignment_method": "input_coordinate_centroid",
                  "functional_alignment_summary": "mean absolute probe response over ordered input coordinates",
                  "functional_alignment_tie_break": "lexicographic absolute functional column values",
                  "functional_alignment_scope": "baseline only; teacher tokens and masks keep their original pairing",
                  "quality_source": None, "accepted_feedback_task_ids": [f"pattern:{pattern}"],
                  "feedback_hashes": []}
    return FunctionalBank(torch.stack(tokens)[None], None, selected_masks, baseline, provenance, states=states,
                          diagnostics={"aligned_q_abs": aligned, "functional_column_orders": column_orders,
                                       "functional_column_centroids": column_centroids,
                                       "probe_x": probe_x.detach().cpu().clone(),
                                       "bank_selection_query_ids": ids[fixed_query].tolist(),
                                       "bank_query_ids": ids[fixed_query].tolist()})


def build_cooperative_fixture(train_patterns: tuple[str, ...] = ("0001", "0011"),
                              test_pattern: str | tuple[str, ...] = "0101",
                              seed: int = 4100, bank_steps: int = 2000, teachers_per_pattern: int = 16,
                              support_count: int = 128, query_count: int = 128, selection_count: int = 64,
                              k: int = 32, device: str = "cpu", probe_count: int = 128,
                              batch_teachers: bool = False, bank_candidates: int | None = None,
                              teacher_batch_size: int = 128,
                              measurement_devices: Sequence[str] | None = None) -> tuple[dict[str, FunctionalBank], list[TaskData], list[TaskData], dict[str, Any]]:
    """Build two train-only banks, global-evaluator tasks, and a sealed test spec."""
    _validate_roles(train_patterns, test_pattern)
    _require_count("bank_steps", bank_steps, 10_000)
    _require_count("teachers_per_pattern", teachers_per_pattern, 256, minimum=5)
    if bank_candidates is not None:
        _require_count("bank_candidates", bank_candidates, 4096, minimum=teachers_per_pattern)
        _require_count("teacher_batch_size", teacher_batch_size, 4096)
    _require_count("support_count", support_count, 256, minimum=2)
    _require_count("query_count", query_count, 256, minimum=2)
    _require_count("selection_count", selection_count, 128, minimum=2)
    _require_count("k", k, _EDGE_COUNT)
    _require_count("probe_count", probe_count, 128, minimum=2)
    primary_device, devices = _available_measurement_devices(device, measurement_devices)
    # Check stratification feasibility before spending any teacher-fit budget.
    test_spec = make_cooperative_test_spec(train_patterns, test_pattern, seed=seed,
                                           support_count=support_count, query_count=query_count)
    parts = _cooperative_partitions(seed)
    _preflight_bank_query_pools(train_patterns, parts)
    _preflight_bank_support_pools(train_patterns, parts)
    ids, x = _pattern_table()
    probe_rows = parts["probe"][:probe_count]
    probe_x, probe_ids = x[probe_rows], ids[probe_rows]
    train_tasks, selection_tasks = [], []
    for number, pattern in enumerate(train_patterns):
        train, selection = _make_train_task(pattern, parts, support_count, query_count, selection_count,
                                            seed + 30_001 * number)
        train_tasks.append(train); selection_tasks.append(selection)
    banks = {pattern: _build_bank(pattern, parts, seed=seed + 10_003 * number, bank_steps=bank_steps,
                                  teachers_per_pattern=teachers_per_pattern, k=k, probe_x=probe_x,
                                  probe_ids=probe_ids, device=primary_device, batch_teachers=batch_teachers,
                                  bank_candidates=bank_candidates, teacher_batch_size=teacher_batch_size,
                                  measurement_devices=devices)
             for number, pattern in enumerate(train_patterns)}
    # This intentionally contains no features, labels, TaskData, or materialized test task.
    return banks, train_tasks, selection_tasks, test_spec


def _make_single_test_spec(train_patterns: tuple[str, ...], test_pattern: str, *, seed: int,
                           support_count: int, query_count: int) -> dict[str, Any]:
    parts = _cooperative_partitions(seed)
    ids, x = _pattern_table()
    support_ids, query_ids = ids[parts["test_support"]], ids[parts["test_query"]]
    support_labels = _pattern_labels(x[parts["test_support"]], test_pattern)
    positives = int((support_labels > .5).sum())
    negatives = len(support_labels) - positives
    if positives < support_count // 2 or negatives < support_count - support_count // 2:
        raise ValueError("test support_count cannot be balanced in its reserved support pool; use a smaller count")
    return {"family": "cooperative_pattern", "seed": seed, "test_pattern": test_pattern,
            # Keep the combined field for the sealed-boundary audit while the
            # disjoint fields bind support and query to independent pools.
            "test_ids": torch.cat((support_ids, query_ids)).tolist(),
            "test_support_ids": support_ids.tolist(), "test_query_ids": query_ids.tolist(),
            "support_count": support_count, "query_count": query_count,
            "train_patterns": list(train_patterns), "materialized": False}


def make_cooperative_test_spec(train_patterns: tuple[str, ...] = ("0001", "0011"),
                               test_pattern: str | tuple[str, ...] = "0101", *, seed: int = 4100,
                               support_count: int = 128, query_count: int = 128) -> dict[str, Any]:
    """Make an opaque single or composite spec without test features/labels.

    Every held-out pattern gets a support-only class-count preflight before
    the spec is returned. A composite shares one frozen pair of observation
    pools while preserving its ordered test-role list.
    """
    test_patterns = _validate_roles(train_patterns, test_pattern)
    _require_count("support_count", support_count, 256, minimum=2)
    _require_count("query_count", query_count, 256, minimum=2)
    if isinstance(test_pattern, str):
        return _make_single_test_spec(train_patterns, test_pattern, seed=seed,
                                      support_count=support_count, query_count=query_count)

    children = [_make_single_test_spec(train_patterns, pattern, seed=seed,
                                       support_count=support_count, query_count=query_count)
                for pattern in test_patterns]
    first = children[0]
    return {"family": "cooperative_pattern", "seed": seed,
            "test_pattern": test_patterns[0], "test_patterns": list(test_patterns),
            "test_specs": children,
            "test_ids": list(first["test_ids"]),
            "test_support_ids": list(first["test_support_ids"]),
            "test_query_ids": list(first["test_query_ids"]),
            "support_count": support_count, "query_count": query_count,
            "train_patterns": list(train_patterns), "materialized": False}


def _matches_expected_spec(actual: Any, expected: Any) -> bool:
    """Compare serialized spec data without coercing tensors or scalar types."""
    if isinstance(expected, dict):
        return (type(actual) is dict and actual.keys() == expected.keys() and
                all(_matches_expected_spec(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (type(actual) is list and len(actual) == len(expected) and
                all(_matches_expected_spec(a, e) for a, e in zip(actual, expected)))
    return type(actual) is type(expected) and actual == expected


def make_cooperative_test_task(spec: dict[str, Any]) -> TaskData:
    """Materialize one legacy-shaped sealed test spec after model selection."""
    if not isinstance(spec, dict) or spec.get("family") != "cooperative_pattern":
        raise ValueError("not a cooperative pattern test specification")
    if "test_specs" in spec or "test_patterns" in spec:
        raise ValueError("composite specifications require make_cooperative_test_tasks")
    pattern = spec.get("test_pattern")
    _validate_pattern(pattern, "test_pattern")
    train = tuple(spec.get("train_patterns", ()))
    _validate_roles(train, pattern)
    support_count, query_count = spec.get("support_count"), spec.get("query_count")
    _require_count("support_count", support_count, 256, minimum=2)
    _require_count("query_count", query_count, 256, minimum=2)
    expected = _make_single_test_spec(train, pattern, seed=spec.get("seed"),
                                      support_count=support_count, query_count=query_count)
    if not _matches_expected_spec(spec, expected):
        raise ValueError("test specification roles or observation IDs were altered")
    ids, x = _pattern_table()
    test_support_ids = torch.as_tensor(spec["test_support_ids"], dtype=torch.long)
    test_query_ids = torch.as_tensor(spec["test_query_ids"], dtype=torch.long)
    labels = _pattern_labels(x, pattern)
    seed = int(spec["seed"])
    support = _balanced_indices(labels, test_support_ids, support_count, seed + 91_001)
    query = _uniform_indices(test_query_ids, query_count, seed + 91_002)
    return TaskData(f"pattern:{pattern}:test", "test", x[support], labels[support], x[query], labels[query],
                    support_context(x[support], labels[support]), ids[support], ids[query],
                    {"family": "pattern", "pattern": pattern, "role": "sealed_test",
                     "support_partition": "test_support", "query_partition": "test_query"})


def make_cooperative_test_tasks(spec: dict[str, Any]) -> list[TaskData]:
    """Materialize all held-out tasks in the sealed spec's declared order."""
    if not isinstance(spec, dict) or spec.get("family") != "cooperative_pattern":
        raise ValueError("not a cooperative pattern test specification")
    if "test_specs" not in spec and "test_patterns" not in spec:
        return [make_cooperative_test_task(spec)]
    raw_patterns = spec.get("test_patterns")
    if type(raw_patterns) is not list or len(raw_patterns) < 2:
        raise ValueError("composite test specification has invalid test_patterns")
    train = tuple(spec.get("train_patterns", ()))
    patterns = tuple(raw_patterns)
    _validate_roles(train, patterns)
    expected = make_cooperative_test_spec(train, patterns, seed=spec.get("seed"),
                                          support_count=spec.get("support_count"),
                                          query_count=spec.get("query_count"))
    if not _matches_expected_spec(spec, expected):
        raise ValueError("composite test specification roles, order, child specs, or observation IDs were altered")
    return [make_cooperative_test_task(child) for child in spec["test_specs"]]


def _stratified_rows(bank: FunctionalBank, max_teachers: int) -> list[int]:
    """Keep one stable representative per density before filling remaining slots."""
    edges = [int(mask.sum()) for mask in bank.masks]
    groups: dict[int, list[int]] = {}
    for index, edge_count in enumerate(edges):
        groups.setdefault(edge_count, []).append(index)
    if max_teachers < len(groups):
        raise ValueError("max_teachers must retain an anchor for every input density")
    def is_feedback(index: int) -> bool:
        state = bank.states[index] if index < len(bank.states) else {}
        return isinstance(state, dict) and state.get("source", {}).get("kind") == "feedback"
    # Prefer the required end anchors and target K, then cover remaining density strata.
    target = int(bank.provenance.get("baseline_k", -1))
    required = []
    for edge_count in (min(groups), target, max(groups)):
        if edge_count in groups and edge_count not in required:
            required.append(edge_count)
    required.extend(edge_count for edge_count in sorted(groups) if edge_count not in required)
    # A bank's original sparse/target/dense teachers are durable anchors.
    # Everything after that makes room for the actual, most-recent feedback
    # replicas before older redundant teachers are retained.
    selected: list[int] = []
    for edge in required:
        anchor = next((index for index in groups[edge] if not is_feedback(index)), groups[edge][0])
        groups[edge].remove(anchor)
        selected.append(anchor)
    feedback = sorted((index for index in range(len(bank.masks))
                       if is_feedback(index) and index not in selected), reverse=True)
    feedback = feedback[:max(0, max_teachers - len(selected))]
    selected.extend(feedback)
    for index in feedback:
        groups[edges[index]].remove(index)
    while len(selected) < max_teachers:
        candidates = [edge for edge, rows in groups.items() if rows]
        if not candidates:
            break
        # Select the currently least represented density, deterministic on ties.
        counts = {edge: sum(edges[index] == edge for index in selected) for edge in candidates}
        edge = min(candidates, key=lambda item: (counts[item], item))
        selected.append(groups[edge].pop(0))
    return selected


def append_feedback(bank: FunctionalBank, mask: Tensor, measurement: dict[str, Any], probe_x: Tensor, *,
                    task_id: str, artifact_path: str | Path, max_teachers: int = 64,
                    eligible: bool = True) -> FunctionalBank:
    """Add terminal fresh-fit replica profiles without importing query quality.

    ``measurement`` is accepted only for a training task of this bank and must
    contain at least two fresh terminal replicas. Its query losses are
    intentionally neither read nor retained as bank channels.
    """
    if not isinstance(bank, FunctionalBank) or bank.provenance.get("family") != "cooperative_pattern":
        raise ValueError("feedback requires a cooperative pattern functional bank")
    if not eligible:
        raise ValueError("held-out topology measurements are not eligible for functional-bank feedback")
    if task_id not in bank.provenance.get("accepted_feedback_task_ids", ()):
        raise ValueError("feedback is eligible only for this bank's training task")
    if not isinstance(measurement, dict) or measurement.get("label_source") != "fresh_terminal_query" or not measurement.get("fixed_horizon"):
        raise ValueError("feedback requires a fresh fixed-horizon terminal measurement")
    if measurement.get("task_id") != task_id:
        raise ValueError("feedback measurement task_id does not match the declared training task")
    if not isinstance(measurement.get("protocol_id"), str) or not measurement["protocol_id"]:
        raise ValueError("feedback measurement must retain its frozen protocol identity")
    states = measurement.get("state_dict")
    if not isinstance(states, (list, tuple)) or len(states) < 2:
        raise ValueError("feedback requires terminal states from at least two replicas")
    replicas = len(states)
    if "replica_losses" not in measurement or len(measurement["replica_losses"]) != replicas:
        raise ValueError("feedback requires one protocol record per replica")
    for field in ("seeds", "plateau_flags"):
        if not isinstance(measurement.get(field), (list, tuple)) or len(measurement[field]) != replicas:
            raise ValueError(f"feedback requires one {field} record per replica")
    if not torch.isfinite(torch.as_tensor(measurement["replica_losses"], dtype=torch.float32)).all():
        raise ValueError("feedback replica losses must be finite")
    path = Path(artifact_path).resolve()
    if not path.is_file():
        raise ValueError("feedback artifact_path must name the saved real measurement")
    mask = torch.as_tensor(mask, dtype=torch.float32).detach().cpu()
    if mask.shape != bank.baseline_mask.shape or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("feedback mask must be a binary [11, 8] matrix")
    probe_x = torch.as_tensor(probe_x, dtype=torch.float32).detach().cpu()
    if tensor_hash(probe_x) != bank.provenance.get("probe_fingerprint"):
        raise ValueError("feedback must use this bank's fixed reserved probe")
    _require_count("max_teachers", max_teachers, 4096, minimum=3)
    new_tokens, new_masks, new_states = [], [], []
    existing = set(bank.provenance.get("feedback_hashes", ()))
    existing.update(item.get("row_hash") for item in bank.states if isinstance(item, dict) and item.get("row_hash"))
    for replica, state in enumerate(states):
        row_hash = _state_row_hash(state, mask)
        if row_hash in existing:
            continue
        token, raw = extract_pattern_tokens(state, mask, probe_x)
        new_tokens.append(token); new_masks.append(mask.clone()); existing.add(row_hash)
        new_states.append({**raw, "state_dict": {name: torch.as_tensor(value).detach().cpu().clone()
                                                   for name, value in state.items()}, "row_hash": row_hash,
                           "optimizer_state": deepcopy(measurement.get("optimizer_state", [None] * replicas)[replica]),
                           "history": deepcopy(measurement.get("history", [None] * replicas)[replica]),
                           "source": {"kind": "feedback", "task_id": task_id, "replica": replica,
                                      "artifact_path": str(path), "source_mask": mask.clone(),
                                      "protocol_id": measurement.get("protocol_id")}})
    if not new_tokens:
        return bank
    combined = FunctionalBank(torch.cat((bank.tokens, torch.stack(new_tokens)[None]), dim=1), None,
                              torch.cat((bank.masks, torch.stack(new_masks)), dim=0), bank.baseline_mask,
                              deepcopy(bank.provenance), states=[*bank.states, *new_states],
                              diagnostics=deepcopy(bank.diagnostics))
    keep = _stratified_rows(combined, max_teachers)
    provenance = deepcopy(combined.provenance)
    provenance["feedback_hashes"] = sorted(existing)
    provenance["teacher_density_counts"] = {int(edges): int((combined.masks[keep].sum((1, 2)) == edges).sum())
                                             for edges in torch.unique(combined.masks[keep].sum((1, 2)).long()).tolist()}
    diagnostics = deepcopy(combined.diagnostics)
    diagnostics["feedback_rows_added"] = int(diagnostics.get("feedback_rows_added", 0)) + len(new_tokens)
    return FunctionalBank(combined.tokens[:, keep], None, combined.masks[keep], combined.baseline_mask,
                          provenance, states=[combined.states[index] for index in keep], diagnostics=diagnostics)


def bank_input_fingerprint(bank: FunctionalBank) -> str:
    """Stable identity for generator inputs, including the fixed-probe binding."""
    if not isinstance(bank, FunctionalBank):
        raise TypeError("bank must be a FunctionalBank")
    digest = hashlib.sha256()
    for tensor in (bank.tokens, bank.masks, bank.baseline_mask):
        digest.update(tensor_hash(tensor).encode())
    digest.update(str(bank.provenance.get("probe_fingerprint", "")).encode())
    digest.update(str(bank.provenance.get("pattern", "")).encode())
    return digest.hexdigest()
