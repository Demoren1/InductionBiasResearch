"""Exhaustive, leakage-free data for circular motif-pair tasks.

For a task ``(A, B, g)`` we enumerate every length-16 bit sequence.  A
candidate is retained only when it has exactly one circular occurrence of A,
exactly one of B, and their receptive fields do not overlap.  A positive has
``start(B) - start(A) == g (mod 16)``; candidates at every other allowed gap
are deliberately retained as hard negatives.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

try:  # Supports both ``python -m motif_pair...`` and running inside motif_pair.
    from motif_pair import config
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI invocation style
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config  # type: ignore


def compute_device() -> torch.device:
    """Return the explicitly requested tensor-compute device.

    Library/tests default to CPU, while every production runner sets
    ``MOTIF_PAIR_DEVICE=cuda``.  CUDA requests are fail-fast: silently
    continuing on CPU would invalidate runtime expectations for this study.
    """
    device = torch.device(os.environ.get("MOTIF_PAIR_DEVICE", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "MOTIF_PAIR_DEVICE requests CUDA, but torch.cuda.is_available() is false"
        )
    return device


def configure_compute_device(device: str) -> torch.device:
    """Set the process-wide device before any cached data are materialized."""
    requested = torch.device(device)
    if requested.type not in {"cpu", "cuda"}:
        raise ValueError("motif_pair data generation supports only cpu or cuda")
    previous = os.environ.get("MOTIF_PAIR_DEVICE", "cpu")
    os.environ["MOTIF_PAIR_DEVICE"] = str(requested)
    if previous != str(requested):
        # Device is part of the cache identity in practice.  Clearing here
        # also makes notebook/test processes safe when they deliberately
        # switch between CPU smoke checks and production CUDA execution.
        for cached in (
            exhaustive_sequence_bank, _all_circular_windows, _all_motif_matches,
            _motif_occurrence_summary, _pair_metadata, _pair_bank,
            _make_task_bank_cached, valid_pairs_for_gap,
        ):
            cached.cache_clear()
    return compute_device()


def circular_windows(x01: torch.Tensor, width: int = config.MOTIF_LEN) -> torch.Tensor:
    """Return all circular contiguous windows, shape ``[N, SEQ_LEN, width]``."""
    if x01.ndim != 2 or x01.shape[1] != config.SEQ_LEN:
        raise ValueError(f"expected [N, {config.SEQ_LEN}] binary sequences")
    if not 1 <= width <= config.SEQ_LEN:
        raise ValueError("window width must be in [1, SEQ_LEN]")
    padded = torch.cat((x01, x01[:, : width - 1]), dim=1)
    return padded.unfold(1, width, 1)


@lru_cache(maxsize=1)
def exhaustive_sequence_bank() -> torch.Tensor:
    """All ``2**16`` binary sequences in deterministic integer order."""
    device = compute_device()
    values = torch.arange(2 ** config.SEQ_LEN, dtype=torch.long, device=device)
    shifts = torch.arange(config.SEQ_LEN - 1, -1, -1, dtype=torch.long, device=device)
    return ((values[:, None] >> shifts) & 1).to(torch.float32)


def _motif_bits(motif: str, *, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor([int(bit) for bit in motif], dtype=torch.float32,
                        device=device or compute_device())


def motif_matches(x01: torch.Tensor, motif: str) -> torch.Tensor:
    """Boolean circular occurrence matrix, indexed by sequence and start."""
    if motif not in config.MOTIFS:
        raise ValueError(f"invalid motif {motif!r}")
    return (circular_windows(x01) == _motif_bits(motif, device=x01.device).view(1, 1, -1)).all(dim=-1)


@lru_cache(maxsize=1)
def _all_circular_windows() -> torch.Tensor:
    return circular_windows(exhaustive_sequence_bank())


@lru_cache(maxsize=None)
def _all_motif_matches(motif: str) -> torch.Tensor:
    """Occurrence matrix for one motif over the fixed exhaustive universe."""
    if motif not in config.MOTIFS:
        raise ValueError(f"invalid motif {motif!r}")
    windows = _all_circular_windows()
    return (windows == _motif_bits(motif, device=windows.device).view(1, 1, -1)).all(dim=-1)


@lru_cache(maxsize=None)
def _motif_occurrence_summary(motif: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (number of occurrences, first start) for all exhaustive inputs."""
    matches = _all_motif_matches(motif)
    return matches.sum(dim=1), matches.to(torch.long).argmax(dim=1)


def _non_overlapping(a_start: torch.Tensor, b_start: torch.Tensor) -> torch.Tensor:
    offsets = torch.arange(config.MOTIF_LEN, device=a_start.device)
    a_positions = (a_start[:, None] + offsets) % config.SEQ_LEN
    b_positions = (b_start[:, None] + offsets) % config.SEQ_LEN
    return ~(a_positions[:, :, None] == b_positions[:, None, :]).any(dim=(1, 2))


@lru_cache(maxsize=None)
def _pair_metadata(a: str, b: str) -> dict[str, torch.Tensor]:
    """Candidate indices/geometry shared by every gap for one motif pair."""
    a_count, a_first_start = _motif_occurrence_summary(a)
    b_count, b_first_start = _motif_occurrence_summary(b)
    exactly_once = (a_count == 1) & (b_count == 1)
    candidate_indices = torch.nonzero(exactly_once, as_tuple=False).squeeze(1)
    # ``first_start`` is meaningful for exactly-once candidates only.
    a_start = a_first_start[candidate_indices]
    b_start = b_first_start[candidate_indices]
    delta = (b_start - a_start) % config.SEQ_LEN
    allowed_gap = torch.zeros(config.SEQ_LEN, dtype=torch.bool, device=delta.device)
    allowed_gap[list(config.GAPS)] = True
    # Every allowed start offset is at least MOTIF_LEN and at most 10, so the
    # two circular three-bit receptive fields are necessarily disjoint.  This
    # algebraic check is equivalent to materialising the 3x3 overlap matrix.
    keep = allowed_gap[delta]
    selected_indices = candidate_indices[keep]
    return {
        "sequence_indices": selected_indices,
        "a_start": a_start[keep].to(torch.long),
        "b_start": b_start[keep].to(torch.long),
        "delta": delta[keep].to(torch.long),
    }


@lru_cache(maxsize=None)
def _pair_bank(a: str, b: str) -> dict[str, Any]:
    """Exhaustive candidate population shared by every gap for one A/B pair."""
    # Materialising sequences is delayed until a pair actually occurs in an
    # experiment.  The splitter only needs metadata and is much cheaper.
    x01 = exhaustive_sequence_bank()
    metadata = _pair_metadata(a, b)
    selected_indices = metadata["sequence_indices"]
    return {
        "x": 2.0 * x01[selected_indices] - 1.0,
        "ones_count": x01[selected_indices].sum(dim=1).to(torch.long),
        "a_start": metadata["a_start"],
        "b_start": metadata["b_start"],
        "delta": metadata["delta"],
    }


@lru_cache(maxsize=None)
def _make_task_bank_cached(task_id: str) -> dict[str, Any]:
    task = config.parse_task(task_id)
    pair = _pair_bank(task.a, task.b)
    y = (pair["delta"] == task.gap).to(torch.float32)
    if not bool((y == 1).any()) or not bool((y == 0).any()):
        raise RuntimeError(f"task {task.id} has no positive or hard-negative candidates")
    return {
        **pair,
        "y": y,
        "task": task.id,
        "task_condition": config.task_to_condition(task),
    }


def make_task_bank(task: config.Task | str) -> dict[str, Any]:
    """Return the exhaustive candidate bank for ``task``.

    Treat the returned tensors as read-only: they are cached so all datasets
    for a task use the identical finite population.
    """
    return _make_task_bank_cached(config.parse_task(task).id)


@lru_cache(maxsize=None)
def valid_pairs_for_gap(gap: int) -> tuple[tuple[str, str], ...]:
    """Ordered motif pairs with both positives and hard negatives at ``gap``."""
    if gap not in config.GAPS:
        raise ValueError(f"gap must be one of {config.GAPS}")
    valid: list[tuple[str, str]] = []
    for a in config.MOTIFS:
        for b in config.MOTIFS:
            if a == b:
                continue
            delta = _pair_metadata(a, b)["delta"]
            if bool((delta == gap).any()) and bool((delta != gap).any()):
                valid.append((a, b))
    return tuple(valid)


def is_task_feasible(task: config.Task | str) -> bool:
    """Whether a task has at least one positive and one allowed-gap negative."""
    parsed = config.parse_task(task)
    return (parsed.a, parsed.b) in valid_pairs_for_gap(parsed.gap)


def _draw(indices: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    if count == 0:
        return indices[:0]
    if indices.numel() == 0:
        raise ValueError("cannot draw from an empty class")
    if count <= indices.numel():
        return indices[torch.randperm(indices.numel(), generator=generator,
                                      device=indices.device)[:count]]
    # Replacement is necessary only for unusually large requested validation
    # sets; it remains seeded and class-balanced.
    return indices[torch.randint(indices.numel(), (count,), generator=generator,
                                 device=indices.device)]


def _draw_matched_positive_negative_pairs(bank: dict[str, Any], target_gap: int, count: int,
                                          generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw pairs with the same one-count and uniformly chosen negative gap.

    Matching every negative to a positive at the same total number of ones
    prevents a linear classifier from solving the task through bit density.
    Circular rotation leaves both the count and each chosen stratum invariant,
    so uniform draws within strata also preserve uniform start positions.
    """
    if count == 0:
        empty = torch.empty(0, dtype=torch.long, device=bank["delta"].device)
        return empty, empty
    delta, ones = bank["delta"], bank["ones_count"]
    positive_by_count = {
        int(one_count): torch.nonzero((delta == target_gap) & (ones == one_count), as_tuple=False).squeeze(1)
        for one_count in torch.unique(ones[delta == target_gap]).tolist()
    }
    alternatives: dict[int, list[int]] = {}
    for gap in config.GAPS:
        if gap == target_gap:
            continue
        compatible_counts = [
            one_count for one_count, positives in positive_by_count.items()
            if positives.numel() and bool(((delta == gap) & (ones == one_count)).any())
        ]
        if compatible_counts:
            alternatives[gap] = compatible_counts
    if not alternatives:
        raise ValueError("task has no one-count-matched hard negatives")

    alternative_gaps = tuple(alternatives)
    device = delta.device
    chosen_gaps = torch.randint(len(alternative_gaps), (count,), generator=generator,
                                device=device)
    positive_indices = torch.empty(count, dtype=torch.long, device=device)
    negative_indices = torch.empty(count, dtype=torch.long, device=device)
    # At most seven gaps and seventeen one-count strata: grouping removes a
    # per-example Python/CUDA synchronization from the sampler.
    for gap_index, gap in enumerate(alternative_gaps):
        gap_positions = torch.nonzero(chosen_gaps == gap_index, as_tuple=False).squeeze(1)
        if gap_positions.numel() == 0:
            continue
        compatible_counts = alternatives[gap]
        selected_counts = torch.randint(len(compatible_counts), (gap_positions.numel(),),
                                        generator=generator, device=device)
        for count_index, one_count in enumerate(compatible_counts):
            positions = gap_positions[selected_counts == count_index]
            if positions.numel() == 0:
                continue
            positives = positive_by_count[one_count]
            negatives = torch.nonzero((delta == gap) & (ones == one_count),
                                      as_tuple=False).squeeze(1)
            positive_indices[positions] = positives[torch.randint(
                positives.numel(), (positions.numel(),), generator=generator, device=device
            )]
            negative_indices[positions] = negatives[torch.randint(
                negatives.numel(), (positions.numel(),), generator=generator, device=device
            )]
    return positive_indices, negative_indices


def sample_balanced(task: config.Task | str, n_samples: int, seed: int) -> dict[str, Any]:
    """Seededly sample an exactly balanced (up to one item) task dataset."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    parsed = config.parse_task(task)
    bank = make_task_bank(parsed)
    device = bank["x"].device
    generator = torch.Generator(device=device).manual_seed(seed)
    n_pos = (n_samples + 1) // 2
    n_neg = n_samples // 2
    # Build matched pairs first.  For odd cardinalities the one extra positive
    # is drawn from the same compatible strata; class balance differs by one.
    paired_pos, negatives = _draw_matched_positive_negative_pairs(bank, parsed.gap, n_neg, generator)
    if n_pos > n_neg:
        extra_pos, _ = _draw_matched_positive_negative_pairs(bank, parsed.gap, 1, generator)
        positives = torch.cat((paired_pos, extra_pos))
    else:
        positives = paired_pos
    selected = torch.cat((positives, negatives))
    selected = selected[torch.randperm(selected.numel(), generator=generator,
                                       device=device)]

    result: dict[str, Any] = {
        key: value[selected].clone()
        for key, value in bank.items()
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == bank["y"].shape[0]
    }
    result.update({
        "task": parsed.id,
        "task_condition": config.task_to_condition(parsed),
        "seed": seed,
    })
    return result


def make_dataset(task: config.Task | str, n_samples: int, seed: int,
                 pos_fraction: float = config.POS_FRACTION) -> dict[str, Any]:
    """Compatibility wrapper for a balanced fixed task dataset.

    Only 0.5 is accepted: the experiment's comparison is intentionally based
    on equally many positives and hard negatives.
    """
    if pos_fraction != 0.5:
        raise ValueError("motif_pair uses a fixed balanced 0.5 positive fraction")
    return sample_balanced(task, n_samples=n_samples, seed=seed)


def gold_mask(task: config.Task | str, hidden: int | None = None) -> torch.Tensor:
    """Task-specific circular two-motif support, with six edges per column."""
    parsed = config.parse_task(task)
    hidden = config.H if hidden is None else hidden
    if hidden < 1:
        raise ValueError("hidden must be positive")
    mask = torch.zeros(config.SEQ_LEN, hidden, dtype=torch.long)
    offsets = torch.arange(config.MOTIF_LEN)
    for h in range(hidden):
        a_start = h % config.SEQ_LEN
        b_start = (a_start + parsed.gap) % config.SEQ_LEN
        mask[(a_start + offsets) % config.SEQ_LEN, h] = 1
        mask[(b_start + offsets) % config.SEQ_LEN, h] = 1
    expected = 2 * config.MOTIF_LEN * hidden
    if int(mask.sum()) != expected:
        raise AssertionError("gold receptive fields unexpectedly overlap")
    return mask


def ideal_mask(task: config.Task | str, hidden: int | None = None) -> torch.Tensor:
    """Alias used by evaluation code for the known structural oracle."""
    return gold_mask(task, hidden=hidden)


def gold_first_layer(task: config.Task | str, hidden: int | None = None) -> torch.Tensor:
    """Circular matched-filter weights whose support is :func:`gold_mask`."""
    parsed = config.parse_task(task)
    hidden = config.H if hidden is None else hidden
    weights = torch.zeros(config.SEQ_LEN, hidden, dtype=torch.float32)
    a = 2.0 * _motif_bits(parsed.a, device=weights.device) - 1.0
    b = 2.0 * _motif_bits(parsed.b, device=weights.device) - 1.0
    for h in range(hidden):
        a_start = h % config.SEQ_LEN
        b_start = (a_start + parsed.gap) % config.SEQ_LEN
        for offset in range(config.MOTIF_LEN):
            weights[(a_start + offset) % config.SEQ_LEN, h] = a[offset]
            weights[(b_start + offset) % config.SEQ_LEN, h] = b[offset]
    return weights


def _cpu_artifact(value: Any) -> Any:
    """Move tensors to CPU only at the persistence boundary."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_artifact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_artifact(item) for item in value)
    return value


def generate_data(split_path: Path, n_val: int = config.N_VAL_SAMPLES,
                  device: str | None = None, data_dir: Path | None = None,
                  condition_encoding: str | None = None) -> None:
    """Persist exhaustive banks and fixed validation sets for a split manifest.

    ``data_dir`` is deliberately an explicit persistence boundary.  A single
    global data directory lets a later split overwrite validation artifacts
    consumed by an earlier split, which makes split-specific OOD runs
    irreproducible.  Omitting it retains the historical ``config.DATA_DIR``
    layout for direct users of this module.
    """
    if device is not None:
        configure_compute_device(device)
    condition_encoding = config.normalize_condition_encoding(condition_encoding)
    payload = json.loads(split_path.read_text())
    task_ids = payload.get("train_tasks", []) + payload.get("test_tasks", [])
    if not task_ids:
        raise ValueError("split manifest has no train_tasks/test_tasks")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("split manifest repeats a task")
    artifact_dir = (data_dir or config.DATA_DIR).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    split_provenance = {
        "split_path": str(split_path.resolve()),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "split_train_tasks": list(payload.get("train_tasks", [])),
        **config.condition_metadata(condition_encoding),
    }
    for index, task_id in enumerate(task_ids):
        task = config.parse_task(task_id)
        bank = {
            **make_task_bank(task),
            "task_condition": config.task_to_condition(task, encoding=condition_encoding),
        }
        torch.save(_cpu_artifact({**bank, **split_provenance}),
                   artifact_dir / f"bank_{task.id}.pt")
        val = {
            **make_dataset(task, n_samples=n_val, seed=20_000 + index),
            "task_condition": config.task_to_condition(task, encoding=condition_encoding),
        }
        torch.save(_cpu_artifact({**val, **split_provenance}),
                   artifact_dir / f"val_{task.id}.pt")
        print(f"[data] {task.id}: bank={len(bank['y'])} val={len(val['y'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate motif-pair bank and validation artifacts.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR,
                        help="directory for split-specific bank_*.pt and val_*.pt artifacts")
    parser.add_argument("--n-val", type=int, default=config.N_VAL_SAMPLES)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--condition-encoding", choices=("one_hot", "scalar"),
                        default=config.DEFAULT_CONDITION_ENCODING,
                        help="condition metadata stored with split-specific artifacts")
    args = parser.parse_args()
    generate_data(args.split, n_val=args.n_val, device=args.device, data_dir=args.data_dir,
                  condition_encoding=args.condition_encoding)


if __name__ == "__main__":
    main()
