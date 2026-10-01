"""Shared data and metrics for the length-11 pattern-quality experiment.

The task and input partitions come from :mod:`meta_pattern.data`.  This
module only materializes labels for train/validation pools; held-out test
labels are created separately by :func:`build_test_pool` for evaluation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from meta_pattern.data import PatternTask, build_task_splits, contains_pattern, partition_ids


SEQ_LEN = 11
PATTERN_LEN = 4
HIDDEN = 8
WINDOWS = SEQ_LEN - PATTERN_LEN + 1
EDGES = SEQ_LEN * HIDDEN
ACTIVE_EDGES = HIDDEN * PATTERN_LEN
SPLIT_SEED = 1729
TASK_SPLIT_SEED = 42
N_INPUTS = 1 << SEQ_LEN


def exhaustive_inputs(seq_len: int = SEQ_LEN) -> dict[str, torch.Tensor]:
    """Return all binary input IDs, their bits, and ±1 model inputs."""
    if seq_len < 1 or seq_len > 20:
        raise ValueError("seq_len must lie in [1, 20] for exhaustive enumeration")
    ids = torch.arange(1 << seq_len, dtype=torch.int64)
    shifts = torch.arange(seq_len - 1, -1, -1, dtype=torch.int64)
    bits = ((ids[:, None] >> shifts) & 1).to(torch.float32)
    return {"ids": ids, "bits": bits, "x": bits.mul(2).sub(1)}


def _pool(x: torch.Tensor, ids: torch.Tensor, pattern: str) -> dict[str, torch.Tensor]:
    pattern_bits = torch.tensor([int(bit) for bit in pattern], dtype=torch.float32)
    y = contains_pattern((x + 1.0) * 0.5, pattern_bits).to(torch.float32)
    return {"x": x.contiguous(), "y": y.contiguous(), "ids": ids.contiguous()}


def common_probe(seq_len: int = SEQ_LEN, split_seed: int = SPLIT_SEED,
                 n_probe: int = 128, seed: int = 8100) -> dict[str, torch.Tensor]:
    """Choose reserved, unlabeled probe IDs from the global support partition."""
    if n_probe < 1:
        raise ValueError("n_probe must be positive")
    table = exhaustive_inputs(seq_len)
    is_support = partition_ids(table["ids"], split_seed=split_seed) == 0
    candidates = table["ids"][is_support]
    if candidates.numel() < n_probe:
        raise ValueError("the support partition is too small for the requested probe")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    probe_ids = candidates[torch.randperm(candidates.numel(), generator=generator)[:n_probe]]
    return {"ids": probe_ids, "x": table["x"].index_select(0, probe_ids)}


def build_experiment_data(
    task_seed: int = TASK_SPLIT_SEED,
    split_seed: int = SPLIT_SEED,
    probe_size: int = 128,
    probe_seed: int = 8100,
) -> dict[str, Any]:
    """Build train/validation task pools and an unlabeled support probe.

    The exhaustive ID hash partition is global and task-independent.  The
    probe IDs are removed from supervised support pools.  Test task pools and
    test labels are intentionally absent from this payload.
    """
    splits = build_task_splits([PATTERN_LEN], seed=task_seed)
    probe = common_probe(n_probe=probe_size, seed=probe_seed, split_seed=split_seed)
    table = exhaustive_inputs()
    codes = partition_ids(table["ids"], split_seed=split_seed)
    probe_mask = torch.isin(table["ids"], probe["ids"])
    support_idx = torch.nonzero((codes == 0) & ~probe_mask, as_tuple=False).flatten()
    query_idx = torch.nonzero(codes == 1, as_tuple=False).flatten()

    pools: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for split_name in ("train", "val"):
        for task in splits[split_name]:
            pools[task.task_id] = {
                "support": _pool(table["x"].index_select(0, support_idx),
                                 table["ids"].index_select(0, support_idx), task.pattern),
                "query": _pool(table["x"].index_select(0, query_idx),
                               table["ids"].index_select(0, query_idx), task.pattern),
            }
    return {
        "task_seed": int(task_seed),
        "split_seed": int(split_seed),
        "splits": splits,
        "pools": pools,
        "probe": probe,
        "seq_len": SEQ_LEN,
        "pattern_len": PATTERN_LEN,
        "hidden": HIDDEN,
    }


def build_test_pool(
    task: PatternTask | str,
    split_seed: int = SPLIT_SEED,
) -> dict[str, torch.Tensor]:
    """Materialize labels only for one held-out task's global test IDs."""
    pattern = task.pattern if isinstance(task, PatternTask) else str(task)
    if len(pattern) != PATTERN_LEN or any(bit not in "01" for bit in pattern):
        raise ValueError(f"expected a {PATTERN_LEN}-bit binary pattern")
    table = exhaustive_inputs()
    test_idx = torch.nonzero(partition_ids(table["ids"], split_seed=split_seed) == 2,
                             as_tuple=False).flatten()
    return _pool(table["x"].index_select(0, test_idx),
                 table["ids"].index_select(0, test_idx), pattern)


def build_task_support_query(
    task: PatternTask | str,
    probe_ids: torch.Tensor,
    split_seed: int = SPLIT_SEED,
) -> dict[str, dict[str, torch.Tensor]]:
    """Create only support/query labels for an evaluation task.

    The common probe is excluded from supervised support.  Test labels are not
    computed here; evaluators create them after freezing the child checkpoint.
    """
    pattern = task.pattern if isinstance(task, PatternTask) else str(task)
    if len(pattern) != PATTERN_LEN or any(bit not in "01" for bit in pattern):
        raise ValueError(f"expected a {PATTERN_LEN}-bit binary pattern")
    table = exhaustive_inputs()
    ids = table["ids"]
    codes = partition_ids(ids, split_seed=split_seed)
    reserved = torch.isin(ids, probe_ids.detach().cpu().to(torch.int64))
    support_idx = torch.nonzero((codes == 0) & ~reserved, as_tuple=False).flatten()
    query_idx = torch.nonzero(codes == 1, as_tuple=False).flatten()
    return {
        "support": _pool(table["x"].index_select(0, support_idx),
                         ids.index_select(0, support_idx), pattern),
        "query": _pool(table["x"].index_select(0, query_idx),
                       ids.index_select(0, query_idx), pattern),
    }


def sample_balanced(
    pool: dict[str, torch.Tensor],
    n_samples: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Sample a class-balanced batch, without replacement when possible."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    y = pool["y"].detach().cpu()
    pos = torch.nonzero(y > 0.5, as_tuple=False).flatten()
    neg = torch.nonzero(y <= 0.5, as_tuple=False).flatten()
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos
    if not pos.numel() or not neg.numel():
        raise ValueError("balanced sampling requires both classes in the pool")

    def choose(indices: torch.Tensor, count: int) -> torch.Tensor:
        if count <= indices.numel():
            return indices[torch.randperm(indices.numel(), generator=generator)[:count]]
        return indices[torch.randint(indices.numel(), (count,), generator=generator)]

    chosen = torch.cat((choose(pos, n_pos), choose(neg, n_neg)))
    chosen = chosen[torch.randperm(chosen.numel(), generator=generator)]
    return {key: value.detach().cpu().index_select(0, chosen)
            for key, value in pool.items()}


def sample_uniform(
    pool: dict[str, torch.Tensor],
    n_samples: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Uniformly sample distinct IDs without conditioning on their labels."""
    if n_samples < 1 or n_samples > pool["ids"].numel():
        raise ValueError("n_samples must fit within the pool")
    chosen = torch.randperm(pool["ids"].numel(), generator=generator)[:n_samples]
    return {key: value.detach().cpu().index_select(0, chosen)
            for key, value in pool.items()}


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return natural BCE/accuracy and class-balanced BCE/accuracy."""
    logits = logits.reshape(-1)
    labels = labels.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    if logits.numel() != labels.numel() or not logits.numel():
        raise ValueError("logits and labels must be non-empty aligned vectors")
    per_example = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    predictions = logits > 0
    positive = labels > 0.5
    negative = ~positive
    natural_accuracy = (predictions == positive).to(logits.dtype).mean()
    if positive.any() and negative.any():
        balanced_bce = 0.5 * (per_example[positive].mean() + per_example[negative].mean())
        balanced_accuracy = 0.5 * (
            (predictions[positive] == positive[positive]).to(logits.dtype).mean()
            + (predictions[negative] == positive[negative]).to(logits.dtype).mean()
        )
    else:
        balanced_bce = per_example.mean()
        balanced_accuracy = natural_accuracy
    natural_bce = per_example.mean()
    return {
        "natural_bce": natural_bce,
        "bce": natural_bce,
        "balanced_bce": balanced_bce,
        "natural_accuracy": natural_accuracy,
        "accuracy": natural_accuracy,
        "balanced_accuracy": balanced_accuracy,
    }
