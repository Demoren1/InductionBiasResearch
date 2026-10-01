"""Leakage-safe scoring and validation helpers for task-quality runs.

This module contains evaluation-only functions.  It never updates model
parameters, selects a mask, or uses test labels to choose a checkpoint.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from meta_pattern.data import build_task_splits, partition_ids

from .core import common_probe
from .structure import exact_topk_mask, gold_toeplitz_mask


METHODS = (
    "transformer_mask",
    "free_mask",
    "functional_centroid_mean",
    "random_exact32",
    "dense",
    "oracle_mask",
)
EXPECTED_OUTER_SEEDS = (8100, 8101, 8102, 8103)
EXPECTED_TEST_TASK_IDS = ("k4:0010", "k4:0100", "k4:1011", "k4:1101")


@lru_cache(maxsize=1)
def protocol_task_ids() -> dict[str, tuple[str, ...]]:
    """Stable task IDs for the prespecified length-4 orbit-safe split."""
    splits = build_task_splits([4], seed=42)
    result = {name: tuple(task.task_id for task in tasks)
              for name, tasks in splits.items()}
    if result["test"] != EXPECTED_TEST_TASK_IDS:
        raise RuntimeError(f"task split changed unexpectedly: {result['test']}")
    return result


def analytic_labels(ids: np.ndarray | torch.Tensor, pattern: str, seq_len: int = 11) -> np.ndarray:
    """Analytically label integer IDs by whether their bit string contains a pattern."""
    values = ids.detach().cpu().numpy() if isinstance(ids, torch.Tensor) else np.asarray(ids)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("ids must be a one-dimensional integer array")
    if not pattern or any(bit not in "01" for bit in pattern):
        raise ValueError("pattern must be a non-empty binary string")
    if len(pattern) > seq_len:
        raise ValueError("pattern cannot be longer than the input sequence")
    if np.any(values < 0) or np.any(values >= 1 << seq_len):
        raise ValueError(f"ids must be in [0, 2**{seq_len})")
    values = values.astype(np.uint64, copy=False)
    shifts = np.arange(seq_len - 1, -1, -1, dtype=np.uint64)
    bits = ((values[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.uint8)
    width = len(pattern)
    target = np.fromiter((int(bit) for bit in pattern), dtype=np.uint8, count=width)
    found = np.zeros(values.shape[0], dtype=bool)
    for start in range(seq_len - width + 1):
        found |= np.all(bits[:, start:start + width] == target[None, :], axis=1)
    return found


def validate_finite_partition(split_ids: Mapping[str, np.ndarray | torch.Tensor],
                              seq_len: int = 11, split_seed: int = 1729) -> dict[str, int]:
    """Check complete, disjoint ID coverage against the global hash partition."""
    expected_names = {"support", "query", "test"}
    if set(split_ids) != expected_names:
        raise ValueError(f"split IDs must have exactly {sorted(expected_names)}")
    all_ids = np.arange(1 << seq_len, dtype=np.int64)
    expected_codes = partition_ids(all_ids, split_seed=split_seed).numpy()
    code_by_name = {"support": 0, "query": 1, "test": 2}
    result: dict[str, int] = {}
    observed = np.zeros(all_ids.shape, dtype=np.int8)
    for name, raw in split_ids.items():
        values = raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
        values = values.astype(np.int64, copy=False)
        if values.ndim != 1 or not np.array_equal(np.sort(values), np.flatnonzero(expected_codes == code_by_name[name])):
            raise ValueError(f"{name} must contain each ID in its global partition exactly once")
        observed[values] += 1
        result[name] = int(values.size)
    if not np.all(observed == 1):
        raise ValueError("finite ID partition is incomplete or overlaps")
    return result


def validate_sampled_pools(pools: Mapping[str, Mapping[str, Any]],
                           pattern: str, split_seed: int = 1729,
                           seq_len: int = 11) -> dict[str, int]:
    """Check support/query/test sampled pools for ID and label leakage."""
    if set(pools) != {"support", "query", "test"}:
        raise ValueError("pools must include support, query and test")
    seen: set[int] = set()
    counts: dict[str, int] = {}
    split_code = {"support": 0, "query": 1, "test": 2}
    for name in ("support", "query", "test"):
        pool = pools[name]
        ids = pool["ids"]
        y = pool["y"]
        ids_np = ids.detach().cpu().numpy() if isinstance(ids, torch.Tensor) else np.asarray(ids)
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        ids_np = ids_np.astype(np.int64, copy=False).reshape(-1)
        y_np = y_np.astype(bool, copy=False).reshape(-1)
        if ids_np.size != y_np.size:
            raise ValueError(f"{name} IDs and labels have different lengths")
        if len(np.unique(ids_np)) != ids_np.size:
            raise ValueError(f"{name} contains duplicate IDs")
        if seen.intersection(map(int, ids_np)):
            raise ValueError("sampled split pools share IDs")
        seen.update(map(int, ids_np))
        actual_codes = partition_ids(ids_np, split_seed=split_seed).numpy()
        if not np.all(actual_codes == split_code[name]):
            raise ValueError(f"{name} contains IDs from another global partition")
        if not np.array_equal(y_np, analytic_labels(ids_np, pattern, seq_len)):
            raise ValueError(f"{name} labels disagree with analytic labels for {pattern}")
        counts[name] = int(ids_np.size)
    return counts


def binary_metrics(logits: np.ndarray | torch.Tensor,
                   labels: np.ndarray | torch.Tensor) -> dict[str, float | int]:
    """Return deterministic threshold-zero accuracy, BCE, Brier score and counts."""
    logit_tensor = torch.as_tensor(logits).detach().float().reshape(-1).cpu()
    label_tensor = torch.as_tensor(labels).detach().float().reshape(-1).cpu()
    if logit_tensor.numel() != label_tensor.numel() or not logit_tensor.numel():
        raise ValueError("logits and labels must be non-empty vectors of equal length")
    if not torch.isfinite(logit_tensor).all() or not torch.isfinite(label_tensor).all():
        raise ValueError("logits and labels must be finite")
    if not torch.all((label_tensor == 0) | (label_tensor == 1)):
        raise ValueError("labels must be binary")
    probabilities = torch.sigmoid(logit_tensor)
    predictions = logit_tensor > 0
    truth = label_tensor >= 0.5
    per_example_bce = F.binary_cross_entropy_with_logits(
        logit_tensor, label_tensor, reduction="none")
    positive = truth
    negative = ~truth
    if not positive.any() or not negative.any():
        raise ValueError("balanced metrics require both positive and negative labels")
    balanced_bce = 0.5 * (per_example_bce[positive].mean() + per_example_bce[negative].mean())
    balanced_accuracy = 0.5 * (
        (predictions[positive] == truth[positive]).float().mean()
        + (predictions[negative] == truth[negative]).float().mean()
    )
    return {
        "n": int(label_tensor.numel()),
        "positive_n": int(truth.sum()),
        "negative_n": int((~truth).sum()),
        "accuracy": float((predictions == truth).float().mean()),
        "bce": float(F.binary_cross_entropy_with_logits(logit_tensor, label_tensor)),
        "balanced_accuracy": float(balanced_accuracy),
        "balanced_bce": float(balanced_bce),
        "brier": float(torch.square(probabilities - label_tensor).mean()),
    }


def score_child(
    params: Mapping[str, torch.Tensor],
    mask: np.ndarray | torch.Tensor,
    pool: Mapping[str, torch.Tensor],
    *,
    predictor: Callable[..., torch.Tensor] | None = None,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> dict[str, float | int]:
    """Score a frozen child on a pool without changing its parameters."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if predictor is None:
        from .meta import child_logits as predictor  # local import avoids a module cycle
    x = pool["x"].to(device)
    y = pool["y"].detach().float().reshape(-1).cpu()
    fixed_mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
    if x.size(0) != y.numel() or fixed_mask.ndim != 2:
        raise ValueError("pool tensors or mask have invalid shapes")
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            batch = x[start:start + batch_size]
            outputs.append(predictor(batch, fixed_mask, params).detach().reshape(-1).cpu())
    return binary_metrics(torch.cat(outputs), y)


def score_children_batched(
    params: Mapping[str, torch.Tensor],
    masks: np.ndarray | torch.Tensor,
    pool: Mapping[str, torch.Tensor],
    *,
    predictor: Callable[..., torch.Tensor] | None = None,
    device: str | torch.device = "cpu",
) -> list[dict[str, float | int]]:
    """Score a run batch on shared test vectors with one batched forward call.

    ``params`` has leading run dimension C (`w:[C,11,8]`, `b/a:[C,8]`,
    `c:[C]`), masks are ``[C,11,8]``, and the default test pool is shared
    ``x:[N,11], y:[N]``.  A per-run pool ``x:[C,N,11], y:[C,N]`` is also
    accepted.  The caller freezes all masks and query-selected parameters
    before passing a test pool.
    """
    if predictor is None:
        from .meta import child_logits_batch as predictor
    x = pool["x"].to(device)
    y = pool["y"].detach().float().to(device)
    mask_tensor = torch.as_tensor(masks, dtype=torch.float32, device=device)
    if mask_tensor.ndim != 3 or mask_tensor.shape[1:] != (11, 8):
        raise ValueError("masks must have shape [C,11,8]")
    if x.ndim not in (2, 3) or x.shape[-1] != 11:
        raise ValueError("test inputs must have shape [N,11] or [C,N,11]")
    if x.ndim == 3 and x.size(0) != mask_tensor.size(0):
        raise ValueError("per-run test input count must equal mask count")
    logits = predictor(x, mask_tensor, params).detach().float()
    if logits.ndim == 1 and mask_tensor.size(0) == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2 or logits.size(0) != mask_tensor.size(0):
        raise ValueError(f"batched predictor must return [C,N], got {tuple(logits.shape)}")
    if y.ndim == 1:
        if y.numel() != logits.size(1):
            raise ValueError("shared test labels do not match batched logits")
        y = y.unsqueeze(0).expand(logits.size(0), -1)
    elif y.ndim == 2:
        if y.shape != logits.shape:
            raise ValueError("per-run test labels do not match batched logits")
    else:
        raise ValueError("test labels must have shape [N] or [C,N]")
    if not torch.isfinite(logits).all() or not torch.isfinite(y).all():
        raise ValueError("batched test logits and labels must be finite")
    if not torch.all((y == 0) | (y == 1)):
        raise ValueError("test labels must be binary")
    probabilities = torch.sigmoid(logits)
    prediction = logits > 0
    truth = y >= 0.5
    per_example_bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    positive_n = truth.sum(dim=1)
    negative_n = (~truth).sum(dim=1)
    if torch.any(positive_n == 0) or torch.any(negative_n == 0):
        raise ValueError("balanced batched metrics require both classes in every test pool")
    accuracy = (prediction == truth).float().mean(dim=1)
    bce = per_example_bce.mean(dim=1)
    balanced_bce = 0.5 * (
        (per_example_bce * truth).sum(dim=1) / positive_n
        + (per_example_bce * (~truth)).sum(dim=1) / negative_n
    )
    balanced_accuracy = 0.5 * (
        ((prediction == truth) & truth).sum(dim=1) / positive_n
        + ((prediction == truth) & (~truth)).sum(dim=1) / negative_n
    )
    brier = torch.square(probabilities - y).mean(dim=1)
    result = []
    for candidate in range(mask_tensor.size(0)):
        result.append({
            "n": int(logits.size(1)),
            "positive_n": int(positive_n[candidate].item()),
            "negative_n": int(logits.size(1) - positive_n[candidate].item()),
            "accuracy": float(accuracy[candidate].item()),
            "bce": float(bce[candidate].item()),
            "balanced_accuracy": float(balanced_accuracy[candidate].item()),
            "balanced_bce": float(balanced_bce[candidate].item()),
            "brier": float(brier[candidate].item()),
        })
    return result


def validate_method_mask(method: str, mask: np.ndarray | torch.Tensor,
                         k: int = 32) -> dict[str, Any]:
    """Check exact cardinality rules and return immutable mask provenance."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    values = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    if values.ndim == 3:
        if values.shape[0] != 1:
            raise ValueError("method masks must describe one task and one fixed child")
        values = values[0]
    if values.ndim != 2 or values.shape != (11, 8):
        raise ValueError(f"every method mask must have shape [11, 8], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("mask contains non-finite values")
    if not np.all(np.isin(values, (0, 1))):
        raise ValueError("final child masks must be binary 0/1 values")
    binary = values != 0
    expected = 88 if method == "dense" else k
    active = int(binary.sum())
    if active != expected:
        raise ValueError(f"{method} has {active} active edges, expected {expected}")
    gold = gold_toeplitz_mask(11, 8, 4)
    if method == "oracle_mask" and not np.array_equal(binary, gold):
        raise ValueError("oracle_mask must equal the gold support and is evaluation-only")
    return {"method": method, "mask": binary.astype(np.uint8),
            "active_edges": active, "exact_k": active == expected,
            "oracle_reference_only": method == "oracle_mask"}


def functional_centroid_mean_mask(edge_q: np.ndarray | torch.Tensor, k: int = 32) -> dict[str, np.ndarray]:
    """Build the source-only centroid-aligned mean exact-K baseline.

    ``edge_q`` has shape ``[source_map, source_probe, 11, 8]``.  For each
    source map, columns are stably sorted by the input-coordinate centroid of
    mean absolute edge response over the source probe.  The aligned source
    maps are averaged and the top K edges are selected with stable row-major
    tie breaking.  No target or gold labels enter this construction.
    """
    values = edge_q.detach().cpu().numpy() if isinstance(edge_q, torch.Tensor) else np.asarray(edge_q)
    if values.ndim != 4 or values.shape[-2:] != (11, 8):
        raise ValueError("edge_q must have shape [source_map, probe, 11, 8]")
    if values.shape[0] < 1 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("edge_q must contain finite source maps and probe rows")
    mean_abs = np.abs(values.astype(np.float64, copy=False)).mean(axis=1)
    coordinate = np.arange(11, dtype=np.float64)[None, :, None]
    mass = mean_abs.sum(axis=1)
    centroid = np.divide(
        (mean_abs * coordinate).sum(axis=1), mass,
        out=np.full_like(mass, np.inf), where=mass > 0,
    )
    column_order = np.argsort(centroid, axis=1, kind="stable")
    aligned = np.take_along_axis(mean_abs, column_order[:, None, :], axis=2)
    scores = aligned.mean(axis=0)
    mask = exact_topk_mask(scores, k)
    return {
        "mask": mask.astype(np.uint8),
        "scores": scores.astype(np.float64),
        "column_order": column_order.astype(np.int64),
        "centroids": centroid.astype(np.float64),
    }


def random_exact32_mask(seed: int, k: int = 32) -> dict[str, np.ndarray]:
    """Generate a seeded exact-K random mask and save its nested random order."""
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(11 * 8)
    mask = np.zeros(11 * 8, dtype=bool)
    mask[order[:k]] = True
    return {"mask": mask.reshape(11, 8).astype(np.uint8),
            "nested_order": order.astype(np.int64)}


def build_comparison_masks(
    transformer_mask: np.ndarray | torch.Tensor,
    free_mask: np.ndarray | torch.Tensor,
    edge_q: np.ndarray | torch.Tensor,
    *,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Assemble the six masks once, before child fitting or test-label access."""
    learned = {}
    for method, raw in (("transformer_mask", transformer_mask), ("free_mask", free_mask)):
        learned[method] = validate_method_mask(method, raw)["mask"]
    functional = functional_centroid_mean_mask(edge_q)
    random = random_exact32_mask(random_seed)
    output = {
        **learned,
        "functional_centroid_mean": functional["mask"],
        "random_exact32": random["mask"],
        "dense": np.ones((11, 8), dtype=np.uint8),
        "oracle_mask": gold_toeplitz_mask(11, 8, 4).astype(np.uint8),
    }
    for method, mask in output.items():
        validate_method_mask(method, mask)
    return output


def test_checkpoint_metadata(metadata: Mapping[str, Any]) -> None:
    """Require explicit query-based selection and frozen status before test scoring."""
    selected_on = metadata.get("checkpoint_selected_on", metadata.get("selected_on"))
    if selected_on != "query":
        raise ValueError("test scoring requires a checkpoint selected on query data")
    if metadata.get("frozen_before_test") is not True:
        raise ValueError("test scoring requires an explicit frozen_before_test=true marker")
    if metadata.get("test_labels_used_for_selection", False):
        raise ValueError("test labels must not influence checkpoint or mask selection")


def evaluate_frozen_test(
    params: Mapping[str, torch.Tensor],
    mask: np.ndarray | torch.Tensor,
    test_pool: Mapping[str, torch.Tensor],
    checkpoint_metadata: Mapping[str, Any],
    *,
    method: str,
    predictor: Callable[..., torch.Tensor] | None = None,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Score the held-out test pool only after provenance checks succeed."""
    test_checkpoint_metadata(checkpoint_metadata)
    validated = validate_method_mask(method, mask)
    score = score_child(params, validated["mask"], test_pool, predictor=predictor,
                        batch_size=batch_size, device=device)
    return {"method": method, "test": score,
            "checkpoint_selected_on": "query", "frozen_before_test": True,
            "test_labels_used_for_selection": False}


def validate_eval_record(record: Mapping[str, Any]) -> None:
    """Validate one serialized run record before it enters aggregate reports."""
    required = {"method", "seed", "task_id", "task_index", "budget", "init_id", "mask",
                "weight", "bias", "readout",
                "checkpoint_selected_on", "frozen_before_test", "test_labels_used_for_selection",
                "support_ids", "query_ids", "test_ids", "lr_selection", "fit_status", "test"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"evaluation record missing required keys: {sorted(missing)}")
    if int(record["seed"]) not in EXPECTED_OUTER_SEEDS:
        raise ValueError(f"seed must be one of the prespecified outer seeds {EXPECTED_OUTER_SEEDS}")
    if record["method"] not in METHODS:
        raise ValueError(f"unknown method in record: {record['method']}")
    if record["test_labels_used_for_selection"] is not False:
        raise ValueError("test labels were marked as used for selection")
    test_checkpoint_metadata(record)
    validate_method_mask(record["method"], np.asarray(record["mask"]))
    for name, shape in (("weight", (11, 8)), ("bias", (8,)), ("readout", (8,))):
        value = np.asarray(record[name], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite with shape {shape}")
    if int(record["task_index"]) not in range(4):
        raise ValueError("task_index must identify one of the four held-out patterns")
    if int(record["budget"]) not in (32, 128):
        raise ValueError("budget must be 32 or 128")
    expected_test_tasks = protocol_task_ids()["test"]
    if str(record["task_id"]) != expected_test_tasks[int(record["task_index"])]:
        raise ValueError("task_index/task_id do not match the prespecified orbit-safe test split")
    seen: set[int] = set()
    recorded_ids: dict[str, np.ndarray] = {}
    expected_partition = {"support": 0, "query": 1, "test": 2}
    for name in ("support", "query", "test"):
        ids = np.asarray(record[f"{name}_ids"], dtype=np.int64).reshape(-1)
        if not ids.size or np.unique(ids).size != ids.size:
            raise ValueError(f"{name}_ids must be a non-empty list without duplicates")
        if seen.intersection(map(int, ids)):
            raise ValueError("support, query and test ID lists must be disjoint")
        seen.update(map(int, ids))
        recorded_ids[name] = ids
        codes = partition_ids(ids, split_seed=1729).numpy()
        if not np.all(codes == expected_partition[name]):
            raise ValueError(f"{name}_ids include examples from another global split")
    if np.asarray(record["support_ids"]).size != int(record["budget"]):
        raise ValueError("support ID count must equal the declared label budget")
    all_ids = np.arange(1 << 11, dtype=np.int64)
    all_codes = partition_ids(all_ids, split_seed=1729).numpy()
    expected_query_ids = np.flatnonzero(all_codes == 1)
    expected_test_ids = np.flatnonzero(all_codes == 2)
    if not np.array_equal(np.sort(recorded_ids["query"]), expected_query_ids):
        raise ValueError("query IDs must cover the complete global query partition")
    if not np.array_equal(np.sort(recorded_ids["test"]), expected_test_ids):
        raise ValueError("test IDs must cover the complete global test partition")
    pattern = str(record["task_id"]).split(":", 1)[1]
    support_labels = analytic_labels(recorded_ids["support"], pattern)
    if int(support_labels.sum()) != int(record["budget"]) // 2:
        raise ValueError("support examples must be class-balanced with an extra negative for odd budgets")
    probe_ids = common_probe(n_probe=128, seed=int(record["seed"]), split_seed=1729)["ids"].numpy()
    if np.intersect1d(recorded_ids["support"], probe_ids).size:
        raise ValueError("supervised support IDs must exclude the reserved unlabeled bank probe")
    lr_selection = record["lr_selection"]
    if not isinstance(lr_selection, Mapping):
        raise ValueError("lr_selection must be a provenance object")
    expected_lr_grid = np.asarray([0.001, 0.003, 0.01], dtype=np.float64)
    actual_lr_grid = np.asarray(lr_selection.get("grid", lr_selection.get("lr_grid", [])), dtype=np.float64)
    if actual_lr_grid.shape != expected_lr_grid.shape or not np.allclose(actual_lr_grid, expected_lr_grid):
        raise ValueError("LR selection must use the predeclared grid [0.001, 0.003, 0.01]")
    selected_lr = float(lr_selection.get("selected_lr", np.nan))
    if not np.isfinite(selected_lr) or not np.any(np.isclose(selected_lr, expected_lr_grid)):
        raise ValueError("selected_lr must be one of the predeclared rates")
    raw_scores = lr_selection.get("scores_by_lr")
    if isinstance(raw_scores, Mapping):
        score_values = np.asarray([
            raw_scores.get(str(rate), raw_scores.get(f"{rate:g}", np.nan))
            for rate in expected_lr_grid
        ], dtype=np.float64)
    else:
        score_values = np.asarray(raw_scores if raw_scores is not None else [], dtype=np.float64)
    if score_values.shape != expected_lr_grid.shape or not np.isfinite(score_values).all():
        raise ValueError("LR tuning must record finite mean validation query BCE for all three rates")
    if np.any(score_values < 0):
        raise ValueError("LR validation BCE values must be non-negative")
    expected_selected_lr = float(expected_lr_grid[int(np.argmin(score_values))])
    if not np.isclose(selected_lr, expected_selected_lr):
        raise ValueError("selected_lr must minimize mean meta-validation query BCE")
    selection_split = lr_selection.get("selection_split", lr_selection.get("split"))
    if selection_split not in ("meta_val_query", "val"):
        raise ValueError("learning rate must be selected only on meta-validation query data")
    used_test = lr_selection.get("used_test", lr_selection.get("test_labels_used_for_selection"))
    if used_test is not False:
        raise ValueError("held-out test tasks cannot be used for learning-rate selection")
    selection_budgets = lr_selection.get("budgets")
    lr_budget = lr_selection.get("budget")
    if (lr_budget is not None and int(lr_budget) != int(record["budget"])) or (
        selection_budgets is not None and int(record["budget"]) not in list(map(int, selection_budgets))
    ) or (lr_budget is None and selection_budgets is None):
        raise ValueError("learning-rate selection budget must match the evaluation budget")
    outer_seeds = lr_selection.get("outer_seeds")
    init_ids = lr_selection.get("init_ids")
    n_seeds = lr_selection.get("n_seeds", len(outer_seeds) if isinstance(outer_seeds, list) else -1)
    n_inits = lr_selection.get("n_inits", len(init_ids) if isinstance(init_ids, list) else -1)
    expected_seeds = list(EXPECTED_OUTER_SEEDS)
    if int(n_seeds) != 4 or int(n_inits) != 4:
        raise ValueError("learning-rate selection must aggregate four seeds and four child initializations")
    if outer_seeds is not None and sorted(map(int, outer_seeds)) != expected_seeds:
        raise ValueError(f"learning-rate selection seeds must be {expected_seeds}")
    if int(record["seed"]) not in expected_seeds:
        raise ValueError(f"outer seed must be one of {expected_seeds}")
    if init_ids is not None and sorted(map(int, init_ids)) != [0, 1, 2, 3]:
        raise ValueError("learning-rate selection must use initialization IDs [0,1,2,3]")
    if selection_budgets is not None and sorted(map(int, selection_budgets)) != [32, 128]:
        raise ValueError("learning-rate selection must cover support budgets 32 and 128")
    val_tasks = lr_selection.get("task_ids")
    if not isinstance(val_tasks, list) or set(map(str, val_tasks)) != set(protocol_task_ids()["val"]):
        raise ValueError("learning-rate selection must use the two length-4 meta-validation tasks")
    selection_score = str(lr_selection.get("score", "")).lower()
    if "query" not in selection_score or "bce" not in selection_score:
        raise ValueError("learning-rate selection score must be meta-validation query BCE")
    fit_status = record["fit_status"]
    if not isinstance(fit_status, Mapping):
        raise ValueError("fit_status must preserve child convergence and checkpoint-selection state")
    if not isinstance(fit_status.get("converged"), bool):
        raise ValueError("fit_status.converged must be a boolean")
    if fit_status.get("selection_complete") is not True:
        raise ValueError("a scored test record requires completed query checkpoint selection")
    steps, max_steps = int(fit_status.get("steps", 0)), int(fit_status.get("max_steps", 0))
    if steps < 1 or max_steps < 1 or steps > max_steps or max_steps > 48000:
        raise ValueError("fit_status steps/max_steps are invalid or exceed the 48000-step cap")
    if not isinstance(fit_status.get("stop_reason"), str) or not fit_status["stop_reason"]:
        raise ValueError("fit_status.stop_reason must identify convergence or the step cap")
    test = record["test"]
    metric_names = ("accuracy", "bce", "balanced_accuracy", "balanced_bce", "brier")
    if not isinstance(test, Mapping) or not {"n", *metric_names}.issubset(test):
        raise ValueError("test metrics must include natural/balanced accuracy and BCE, and Brier score")
    if int(test["n"]) < 1:
        raise ValueError("test metric count must be positive")
    expected_test_labels = analytic_labels(recorded_ids["test"], pattern)
    if int(test["n"]) != expected_test_ids.size:
        raise ValueError("test metric count must equal the exhaustive held-out ID partition")
    if int(test.get("positive_n", -1)) != int(expected_test_labels.sum()) or int(
        test.get("negative_n", -1)) != int((~expected_test_labels).sum()
    ):
        raise ValueError("test class counts disagree with analytic labels on the exhaustive test IDs")
    if not all(math.isfinite(float(test[key])) for key in metric_names):
        raise ValueError("test metrics must be finite")
    if any(not 0.0 <= float(test[key]) <= 1.0 for key in ("accuracy", "balanced_accuracy")):
        raise ValueError("test accuracy metrics must lie in [0, 1]")
    if any(float(test[key]) < 0 for key in ("bce", "balanced_bce", "brier")):
        raise ValueError("test loss metrics must be non-negative")


def validate_artifact_root(root: str | Path) -> dict[str, Any]:
    """Validate JSONL/JSON evaluation records and numeric summary arrays.

    The canonical aggregate inputs are ``records.jsonl`` and ``summary.npz``.
    The JSONL records carry per-run provenance; the NPZ is required to contain
    numeric arrays for the same six comparison methods.  This validator is
    CPU-only and does not open model checkpoints or recompute test metrics.
    """
    path = Path(root)
    records_path = path / "records.jsonl"
    summary_path = path / "summary.npz"
    if not records_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"expected {records_path.name} and {summary_path.name} under {path}")
    records: list[dict[str, Any]] = []
    with records_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_eval_record(record)
            except Exception as error:
                raise ValueError(f"invalid record at {records_path}:{line_number}: {error}") from error
            records.append(record)
    if not records:
        raise ValueError("records.jsonl contains no evaluation records")
    keys = [(r["method"], int(r["seed"]), r["task_id"], int(r["budget"]), int(r["init_id"]))
            for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("records.jsonl contains duplicate method/seed/task/budget/init keys")

    with np.load(summary_path, allow_pickle=False) as archive:
        required = {"method", "seed", "task_id", "task_index", "budget", "init_id",
                    "accuracy", "bce", "balanced_accuracy", "balanced_bce", "brier",
                    "selected_lr", "test_n", "child_steps",
                    "child_max_steps", "child_converged", "child_selection_complete", "child_stop_reason"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"summary.npz missing required numeric arrays: {sorted(missing)}")
        arrays = {name: archive[name] for name in required}
        lengths = {value.shape[0] for value in arrays.values() if value.ndim >= 1}
        if len(lengths) != 1 or any(value.ndim != 1 for value in arrays.values()):
            raise ValueError("summary.npz fields must be aligned one-dimensional arrays")
        if not all(np.isfinite(arrays[name].astype(float)).all() for name in
                   ("accuracy", "bce", "balanced_accuracy", "balanced_bce", "brier", "selected_lr")):
            raise ValueError("summary.npz contains non-finite metrics")
        if not np.all((arrays["accuracy"] >= 0) & (arrays["accuracy"] <= 1)):
            raise ValueError("summary accuracy values must lie in [0, 1]")
        if not np.all((arrays["balanced_accuracy"] >= 0) & (arrays["balanced_accuracy"] <= 1)):
            raise ValueError("summary balanced accuracy values must lie in [0, 1]")
        if not np.all(arrays["bce"] >= 0) or not np.all(arrays["balanced_bce"] >= 0):
            raise ValueError("summary BCE values must be non-negative")
        if not np.all(arrays["brier"] >= 0):
            raise ValueError("summary Brier values must be non-negative")
        if not np.all(np.isin(arrays["selected_lr"].astype(float), [0.001, 0.003, 0.01])):
            raise ValueError("summary selected learning rates must be from the predeclared grid")
        if not np.all(arrays["test_n"] > 0) or not np.all(arrays["child_steps"] > 0):
            raise ValueError("summary test counts and child steps must be positive")
        if np.any(arrays["child_steps"] > arrays["child_max_steps"]):
            raise ValueError("summary child steps exceed the configured cap")
        if not np.all(arrays["child_selection_complete"]):
            raise ValueError("summary contains a child without completed query checkpoint selection")
        methods = set(arrays["method"].astype(str).tolist())
        if methods != set(METHODS):
            raise ValueError(f"summary methods must equal {sorted(METHODS)}, got {sorted(methods)}")
        npz_keys = list(zip(arrays["method"].astype(str), arrays["seed"].astype(int),
                            arrays["task_id"].astype(str), arrays["budget"].astype(int),
                            arrays["init_id"].astype(int)))
        if len(set(npz_keys)) != len(npz_keys):
            raise ValueError("summary.npz contains duplicate run keys")
        if set(npz_keys) != set(keys):
            raise ValueError("summary.npz run keys do not match records.jsonl")
        return {"n_records": len(records), "n_methods": len(methods),
                "methods": sorted(methods), "summary_arrays": len(required)}


def paired_seed_interval(values: np.ndarray, confidence: float = 0.95) -> dict[str, float | int]:
    """Descriptive paired mean and Student-t interval over independent seeds."""
    from scipy.stats import t

    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("values must be a non-empty finite vector")
    mean = float(samples.mean())
    if samples.size < 2:
        return {"n": int(samples.size), "mean": mean, "ci_low": float("nan"),
                "ci_high": float("nan"), "confidence": float(confidence)}
    sem = float(samples.std(ddof=1) / np.sqrt(samples.size))
    critical = float(t.ppf(0.5 + confidence / 2, df=samples.size - 1))
    half_width = critical * sem
    return {"n": int(samples.size), "mean": mean, "ci_low": mean - half_width,
            "ci_high": mean + half_width, "confidence": float(confidence)}


def validate_evaluation_toy_cases() -> dict[str, Any]:
    """CPU self-checks for partition IDs, analytic labels, NaN rejection and t CI."""
    from scipy.stats import t

    ids = np.arange(1 << 11, dtype=np.int64)
    counts = partition_ids(ids, split_seed=1729).numpy()
    split = {name: ids[counts == code] for name, code in
             (("support", 0), ("query", 1), ("test", 2))}
    partition_counts = validate_finite_partition(split, seq_len=11, split_seed=1729)
    label_pattern = "0010"
    expected_labels = np.fromiter(
        (label_pattern in f"{int(value):011b}" for value in ids),
        dtype=bool,
        count=ids.size,
    )
    if not np.array_equal(analytic_labels(ids, label_pattern), expected_labels):
        raise AssertionError("analytic_labels disagrees with string matching")

    test_task = protocol_task_ids()["test"][0]
    probe_ids = common_probe(n_probe=128, seed=EXPECTED_OUTER_SEEDS[0], split_seed=1729)["ids"].numpy()
    support_candidates = split["support"][~np.isin(split["support"], probe_ids)]
    support_labels = analytic_labels(support_candidates, test_task.split(":", 1)[1])
    support_ids = np.concatenate((support_candidates[support_labels][:16],
                                  support_candidates[~support_labels][:16])).tolist()
    query_ids = split["query"].tolist()
    test_ids = split["test"].tolist()
    test_labels = analytic_labels(split["test"], test_task.split(":", 1)[1])
    record = {
        "method": "transformer_mask", "seed": EXPECTED_OUTER_SEEDS[0],
        "task_id": test_task, "task_index": 0, "budget": 32, "init_id": 0,
        "mask": gold_toeplitz_mask().astype(int).tolist(),
        "weight": np.ones((11, 8), dtype=float).tolist(),
        "bias": np.zeros(8, dtype=float).tolist(),
        "readout": np.ones(8, dtype=float).tolist(),
        "checkpoint_selected_on": "query", "frozen_before_test": True,
        "test_labels_used_for_selection": False,
        "support_ids": support_ids, "query_ids": query_ids, "test_ids": test_ids,
        "lr_selection": {
            "selected_lr": 0.003, "grid": [0.001, 0.003, 0.01],
            "scores_by_lr": {"0.001": 0.7, "0.003": 0.5, "0.01": 0.6},
            "selection_split": "val", "used_test": False, "budgets": [32, 128],
            "outer_seeds": list(EXPECTED_OUTER_SEEDS), "init_ids": [0, 1, 2, 3],
            "task_ids": list(protocol_task_ids()["val"]),
            "score": "mean query balanced BCE",
        },
        "fit_status": {"converged": True, "stop_reason": "converged",
                       "steps": 1000, "max_steps": 12000, "selection_complete": True},
        "test": {"n": len(test_ids), "positive_n": int(test_labels.sum()),
                 "negative_n": int((~test_labels).sum()),
                 "accuracy": 0.75, "balanced_accuracy": 0.75,
                 "bce": 0.5, "balanced_bce": 0.55, "brier": 0.18},
    }
    validate_eval_record(record)
    extended = {**record, "fit_status": {
        **record["fit_status"], "steps": 24000, "max_steps": 48000,
        "stop_reason": "converged",
    }}
    validate_eval_record(extended)
    invalid_seed = {**record, "seed": 8104}
    try:
        validate_eval_record(invalid_seed)
    except ValueError:
        seed_rejected = True
    else:
        raise AssertionError("artifact validator accepted an unregistered outer seed")
    invalid_mask = {**record, "mask": np.where(np.asarray(record["mask"]), 0.5, 0.0).tolist()}
    try:
        validate_eval_record(invalid_mask)
    except ValueError:
        nonbinary_rejected = True
    else:
        raise AssertionError("artifact validator accepted nonbinary mask values")
    invalid = {**record, "test": {**record["test"], "bce": float("nan")}}
    try:
        validate_eval_record(invalid)
    except ValueError:
        nan_rejected = True
    else:
        raise AssertionError("artifact validator accepted a NaN test BCE")

    samples = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    ci = paired_seed_interval(samples)
    expected_half_width = float(t.ppf(0.975, df=3) * samples.std(ddof=1) / np.sqrt(4))
    if ci["n"] != 4 or not np.isclose(ci["ci_high"], samples.mean() + expected_half_width):
        raise AssertionError("paired seed confidence interval does not use t with df=3")

    def fake_batched_predictor(x, masks, params):
        if x.ndim == 2:
            logits = x.sum(dim=-1).unsqueeze(0).expand(masks.size(0), -1)
        else:
            logits = x.sum(dim=-1)
        return logits

    shared_pool = {"x": torch.zeros((32, 11)),
                   "y": torch.cat((torch.zeros(16), torch.ones(16)))}
    batch_metrics = score_children_batched(
        params={}, masks=np.zeros((2, 11, 8), dtype=np.uint8), pool=shared_pool,
        predictor=fake_batched_predictor,
    )
    if len(batch_metrics) != 2 or any(item["n"] != 32 for item in batch_metrics):
        raise AssertionError("batched child scoring did not preserve the run dimension")
    return {
        "partition_counts": partition_counts,
        "analytic_labels_match_string_reference": True,
        "nan_test_metric_rejected": nan_rejected,
        "invalid_outer_seed_rejected": seed_rejected,
        "nonbinary_mask_rejected": nonbinary_rejected,
        "extended_48000_step_cap_accepted": True,
        "paired_interval_df": 3,
        "batched_candidate_count": len(batch_metrics),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="validate an output root with records.jsonl and summary.npz")
    parser.add_argument("--self-test", action="store_true", help="run CPU validation fixtures")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(validate_evaluation_toy_cases(), indent=2))
    elif args.root is not None:
        print(json.dumps(validate_artifact_root(args.root), indent=2))
    else:
        parser.error("provide --root or --self-test")
