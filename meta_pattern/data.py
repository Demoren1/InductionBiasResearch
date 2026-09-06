"""Data for the length-32, variable-length pattern experiment.

Sequences are represented by their unsigned 32-bit integer ID: bit 31 is the
left-most element of ``x``.  The public sampler always draws IDs uniformly,
then rejects IDs outside a global support/query/test hash partition and (for a
balanced set) conditions on the label.  It never constructs examples by
inserting a pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

import numpy as np
import torch


SplitName = Literal["support", "query", "test"]
_SPLIT_TO_CODE: Mapping[str, int] = {"support": 0, "query": 1, "test": 2}
_UINT32_SPACE = 1 << 32


@dataclass(frozen=True, slots=True)
class PatternTask:
    """One binary substring-detection task."""

    pattern: str

    def __post_init__(self) -> None:
        if not self.pattern or any(bit not in "01" for bit in self.pattern):
            raise ValueError("pattern must be a non-empty string of '0' and '1'")

    @property
    def length(self) -> int:
        return len(self.pattern)

    @property
    def task_id(self) -> str:
        """Stable identifier that cannot conflate equal-looking task names."""
        return f"k{self.length}:{self.pattern}"


def _orbit(pattern: str) -> tuple[str, ...]:
    """Return the reversal/complement orbit of a pattern."""
    complement = "".join("1" if bit == "0" else "0" for bit in pattern)
    return tuple(sorted({pattern, pattern[::-1], complement, complement[::-1]}))


def _assign_orbits(orbit_sizes: list[int]) -> list[int]:
    """Assign shuffled orbits to 60/20/20 splits with each split non-empty.

    We enumerate the at most ``3**128`` theoretical assignments only after
    dynamic programming has compressed identical count states.  Pattern
    lengths 3--8 have at most 72 orbits, so the state space here remains tiny:
    each state is just its accumulated train and validation task counts.
    """
    # State maps (train count, val count, used-split mask) to the assignment
    # prefix.  Keeping one prefix per state is enough because the remaining
    # objective depends only on final counts and split usage.
    states: dict[tuple[int, int, int], tuple[int, ...]] = {(0, 0, 0): ()}
    total = sum(orbit_sizes)
    for size in orbit_sizes:
        next_states: dict[tuple[int, int, int], tuple[int, ...]] = {}
        for (n_train, n_val, used), prefix in states.items():
            for split in range(3):
                key = (
                    n_train + (size if split == 0 else 0),
                    n_val + (size if split == 1 else 0),
                    used | (1 << split),
                )
                candidate = prefix + (split,)
                # A deterministic tie break makes the result independent of
                # dictionary insertion details.
                if key not in next_states or candidate < next_states[key]:
                    next_states[key] = candidate
        states = next_states

    target = (0.60 * total, 0.20 * total, 0.20 * total)
    candidates = []
    for (n_train, n_val, used), assignment in states.items():
        if used != 0b111:
            continue
        n_test = total - n_train - n_val
        counts = (n_train, n_val, n_test)
        # Squared relative errors make a one-task discrepancy comparable
        # across the three intended proportions.
        score = sum(((count - wanted) / total) ** 2 for count, wanted in zip(counts, target))
        candidates.append((score, assignment, counts))
    if not candidates:
        raise ValueError("at least three equivalence orbits are required")
    _, assignment, _ = min(candidates, key=lambda item: (item[0], item[1]))
    return list(assignment)


def build_task_splits(
    lengths: Iterable[int] = (3, 4, 5, 6, 7, 8), seed: int = 42
) -> dict[str, list[PatternTask]]:
    """Split patterns by reversal/complement orbit for every requested length.

    Each length is independently partitioned near 60/20/20 by *number of
    patterns*, while every split receives at least one full orbit.  Length 3
    has three orbits of sizes 2, 4, and 2, so its closest orbit-safe split is
    necessarily 4/2/2 rather than exact 60/20/20.  No orbit is shared by two
    splits, avoiding trivial transfer between a pattern and its reversal or
    bit-complement.
    """
    lengths = tuple(lengths)
    if not lengths:
        raise ValueError("lengths must not be empty")
    if len(set(lengths)) != len(lengths):
        raise ValueError("lengths must be unique")
    if any(not isinstance(length, int) or length < 1 or length > 32 for length in lengths):
        raise ValueError("every pattern length must be an integer in [1, 32]")

    output: dict[str, list[PatternTask]] = {name: [] for name in ("train", "val", "test")}
    # SeedSequence makes changing the order of ``lengths`` immaterial.
    for length in sorted(lengths):
        patterns = (format(value, f"0{length}b") for value in range(1 << length))
        remaining = set(patterns)
        orbits: list[tuple[str, ...]] = []
        while remaining:
            representative = min(remaining)
            current = _orbit(representative)
            orbits.append(current)
            remaining.difference_update(current)

        rng = np.random.default_rng(np.random.SeedSequence([int(seed), length]))
        permutation = rng.permutation(len(orbits))
        shuffled = [orbits[index] for index in permutation]
        assignments = _assign_orbits([len(orbit) for orbit in shuffled])
        for orbit, split_code in zip(shuffled, assignments):
            split_name = ("train", "val", "test")[split_code]
            output[split_name].extend(PatternTask(pattern) for pattern in orbit)

    # Stable ordering helps experiment manifests and never alters membership.
    for tasks in output.values():
        tasks.sort(key=lambda task: (task.length, task.pattern))
    return output


def partition_ids(ids: torch.Tensor | np.ndarray | Iterable[int], split_seed: int = 1729) -> torch.Tensor:
    """Map uint32 IDs to global support/query/test codes 0/1/2.

    The mapping is a deterministic 64-bit SplitMix-style hash followed by ten
    buckets: 0--5 are support (60%), 6--7 query (20%), and 8--9 test (20%).
    It depends only on an input ID and ``split_seed``; it never depends on a
    task or sampling seed.  Returned codes are CPU ``torch.int64``.
    """
    if isinstance(ids, torch.Tensor):
        values = ids.detach().cpu().numpy()
    else:
        values = np.asarray(ids)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("ids must be integer-valued unsigned 32-bit IDs")
    if np.issubdtype(values.dtype, np.signedinteger) and np.any(values < 0):
        raise ValueError("ids must lie in [0, 2**32)")
    values_uint64 = values.astype(np.uint64, copy=False)
    if np.any(values_uint64 >= np.uint64(_UINT32_SPACE)):
        raise ValueError("ids must lie in [0, 2**32)")
    x = values_uint64
    seed_value = np.uint64(int(split_seed) & ((1 << 64) - 1))
    x = x + seed_value + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    buckets = (x % np.uint64(10)).astype(np.int64)
    codes = np.where(buckets < 6, 0, np.where(buckets < 8, 1, 2)).astype(np.int64)
    return torch.from_numpy(codes)


def contains_pattern(x01: torch.Tensor, pat_bits: torch.Tensor) -> torch.Tensor:
    """Return a bool vector: whether each binary row contains ``pat_bits``."""
    if x01.ndim != 2:
        raise ValueError("x01 must have shape [batch, sequence_length]")
    if pat_bits.ndim != 1 or pat_bits.numel() == 0:
        raise ValueError("pat_bits must be a non-empty vector")
    if pat_bits.numel() > x01.size(1):
        raise ValueError("pattern length cannot exceed sequence length")
    windows = x01.unfold(dimension=1, size=pat_bits.numel(), step=1)
    pattern = pat_bits.to(device=x01.device, dtype=x01.dtype).view(1, 1, -1)
    return (windows == pattern).all(dim=-1).any(dim=1)


def _contains_pattern_ids(ids: np.ndarray, pattern: str, seq_len: int) -> np.ndarray:
    """Vectorised substring labels without materialising a [batch, 32] tensor."""
    pattern_value = np.uint64(int(pattern, 2))
    width = len(pattern)
    mask = np.uint64((1 << width) - 1)
    found = np.zeros(ids.shape, dtype=bool)
    for start in range(seq_len - width + 1):
        shift = np.uint64(seq_len - width - start)
        found |= ((ids >> shift) & mask) == pattern_value
    return found


def _ids_to_bits(ids: np.ndarray, seq_len: int) -> np.ndarray:
    shifts = np.arange(seq_len - 1, -1, -1, dtype=np.uint64)
    return ((ids[:, None] >> shifts) & np.uint64(1)).astype(np.float32)


def sample_dataset(
    task: PatternTask,
    n_samples: int,
    seed: int,
    split: SplitName,
    split_seed: int = 1729,
    balanced: bool = True,
    seq_len: int = 32,
) -> dict[str, torch.Tensor]:
    """Sample a deterministic, globally disjoint dataset for one task.

    ``support``, ``query``, and ``test`` are selected before seeing a task
    label, so their ID sets are disjoint for every task and every random seed.
    With ``balanced=True``, the sampler draws exactly ``n_samples // 2``
    positives and the remaining samples as negatives (an odd total therefore
    has one extra negative).  With ``balanced=False`` it retains the natural
    label frequency inside the requested global partition.
    """
    if seq_len != 32:
        raise ValueError("this experiment represents sequences by unsigned 32-bit IDs; seq_len must be 32")
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if split not in _SPLIT_TO_CODE:
        raise ValueError("split must be one of 'support', 'query', or 'test'")
    if task.length > seq_len:
        raise ValueError("pattern length cannot exceed seq_len")

    rng = np.random.default_rng(seed)
    split_code = _SPLIT_TO_CODE[split]
    positive_target = n_samples // 2
    negative_target = n_samples - positive_target
    positive_ids: list[int] = []
    negative_ids: list[int] = []
    natural_ids: list[int] = []
    natural_labels: list[bool] = []
    seen: set[int] = set()

    # The rarest requested class is a k=3 non-occurrence.  Its probability is
    # still large enough for normal experiment sizes; this cap prevents an
    # accidental impossible request from running forever.
    max_draws = max(100_000, min(20_000_000, n_samples * 20_000))
    draws = 0
    while draws < max_draws:
        if balanced:
            needed = (positive_target - len(positive_ids)) + (negative_target - len(negative_ids))
            if needed <= 0:
                break
        else:
            needed = n_samples - len(natural_ids)
            if needed <= 0:
                break
        batch_size = min(65_536, max(4_096, needed * 128))
        batch_size = min(batch_size, max_draws - draws)
        candidates = rng.integers(0, _UINT32_SPACE, size=batch_size, dtype=np.uint64)
        draws += batch_size
        partition = partition_ids(candidates, split_seed=split_seed).numpy()
        candidates = candidates[partition == split_code]
        if not candidates.size:
            continue
        labels = _contains_pattern_ids(candidates, task.pattern, seq_len)
        for candidate, label in zip(candidates.tolist(), labels.tolist()):
            identifier = int(candidate)
            if identifier in seen:
                continue
            if balanced:
                if label and len(positive_ids) < positive_target:
                    positive_ids.append(identifier)
                    seen.add(identifier)
                elif not label and len(negative_ids) < negative_target:
                    negative_ids.append(identifier)
                    seen.add(identifier)
            elif len(natural_ids) < n_samples:
                natural_ids.append(identifier)
                natural_labels.append(bool(label))
                seen.add(identifier)

    if balanced:
        if len(positive_ids) != positive_target or len(negative_ids) != negative_target:
            raise RuntimeError(
                f"could not draw {positive_target} positives and {negative_target} negatives "
                f"for task {task.task_id} in split={split!r} after {draws} candidates; "
                "reduce n_samples or inspect the requested partition"
            )
        ids = np.asarray(positive_ids + negative_ids, dtype=np.uint64)
        labels = np.asarray([True] * len(positive_ids) + [False] * len(negative_ids), dtype=bool)
    else:
        if len(natural_ids) != n_samples:
            raise RuntimeError(
                f"could not draw {n_samples} distinct examples for split={split!r} after {draws} candidates"
            )
        ids = np.asarray(natural_ids, dtype=np.uint64)
        labels = np.asarray(natural_labels, dtype=bool)

    # Randomising final order keeps the label order from becoming a feature,
    # while preserving reproducibility under the same sampler seed.
    order = rng.permutation(n_samples)
    ids = ids[order]
    labels = labels[order]
    x01 = _ids_to_bits(ids, seq_len)
    return {
        "x": torch.from_numpy(2.0 * x01 - 1.0).to(torch.float32),
        "y": torch.from_numpy(labels.astype(np.float32)),
        "ids": torch.from_numpy(ids.astype(np.int64)),
    }
