"""Durable real-measurement stores shared by all search runners.

The store owns only cache layout and replay bookkeeping.  Fitting is supplied
as an injected callable, so a runner can keep its domain-specific fitter (and
tests can replace it) without importing one runner from another.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

from .adapters import measure_mask as _default_measure_mask
from .artifacts import save_torch
from .data import RealReplay, TaskData, tensor_hash
from .pattern_batch import fit_pattern_batch as _default_batch_fit


MeasureFn = Callable[..., dict]
BatchFitFn = Callable[[torch.Tensor, Sequence[TaskData], object, str], list[dict]]


def measurement_digest(mask: torch.Tensor, task: TaskData, protocol) -> str:
    """Stable cache key for one non-fresh measurement."""
    value = tensor_hash(mask.detach().cpu().float()) + task.fingerprint + protocol.fingerprint
    return hashlib.sha256(value.encode()).hexdigest()


def _payload(mask: torch.Tensor, task: TaskData, protocol, result: dict) -> dict:
    # Inputs are sometimes views into stacked candidate/provenance batches.
    # Own compact rows so each artifact serializes only its logical payload.
    compact_mask = mask.detach().cpu().clone(memory_format=torch.contiguous_format)
    compact_support_ids = task.support_ids.detach().cpu().clone(memory_format=torch.contiguous_format)
    compact_query_ids = task.query_ids.detach().cpu().clone(memory_format=torch.contiguous_format)
    return dict(result=result, mask=compact_mask, mask_key=tensor_hash(mask), task_id=task.task_id,
                task_fingerprint=task.fingerprint, task_provenance=task.provenance,
                support_ids=compact_support_ids, query_ids=compact_query_ids,
                protocol=asdict(protocol))


class MeasurementStore:
    """Cache single real child fits and append each fit exactly once to replay."""

    def __init__(self, out: str | Path, replay: RealReplay, device: str,
                 measure_fn: MeasureFn | None = None, *, save_torch_fn: Callable | None = None):
        self.out = Path(out)
        self.replay = replay
        self.device = str(device)
        self.measure_fn = measure_fn or _default_measure_mask
        self._save_torch = save_torch_fn or save_torch

    def _path(self, mask: torch.Tensor, task: TaskData, *, fresh: bool) -> Path:
        digest = measurement_digest(mask, task, self.replay.protocol)
        if fresh:
            # Cache keys must not accidentally reuse random child initializations.
            digest += f"_draw_{len(self.replay.records)}"
        return self.out / "children" / f"{digest}.pt"

    @staticmethod
    def _read(path: Path, mask: torch.Tensor, task: TaskData) -> dict:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (payload.get("task_fingerprint") != task.fingerprint or
                payload.get("mask_key") != tensor_hash(mask)):
            raise ValueError("cached measurement provenance mismatch")
        if "result" not in payload:
            raise ValueError("cached measurement is missing its child result")
        return payload["result"]

    def _append(self, mask: torch.Tensor, task: TaskData, origin: str, path: Path, result: dict):
        resolved = str(path.resolve())
        existing = next((row for row in self.replay.records
                         if row["artifact_path"] == resolved), None)
        return existing if existing is not None else self.replay.append(
            mask, task, result, origin=origin, artifact_path=path)

    def measure(self, mask: torch.Tensor, task: TaskData, origin: str, *, fresh: bool = False):
        mask = mask.detach().cpu().float()
        path = self._path(mask, task, fresh=fresh)
        if path.exists():
            result = self._read(path, mask, task)
        else:
            initialization_seed = (self.replay.protocol.seed + 1_000_003 * (len(self.replay.records) + 1)
                                   if fresh else None)
            result = self.measure_fn(mask, task, self.replay.protocol, device=self.device,
                                     initialization_seed=initialization_seed)
            self._save_torch(path, _payload(mask, task, self.replay.protocol, result))
        return self._append(mask, task, origin, path, result), result

    def measure_many(self, entries: Iterable[tuple[torch.Tensor, TaskData, str]]):
        """Sequential fallback useful for non-vectorizable domains."""
        return [self.measure(mask, task, origin) for mask, task, origin in entries]


class BatchedMeasurementStore(MeasurementStore):
    """Pattern specialization that preserves cache and replay semantics.

    Cached and missing entries intentionally stay in the same numerical batch:
    after a partial artifact write, rerunning produces the exact same states
    for the missing artifacts as an uninterrupted batch would.
    """

    def __init__(self, out: str | Path, replay: RealReplay, device: str,
                 measure_fn: MeasureFn | None = None, batch_fit_fn: BatchFitFn | None = None,
                 *, save_torch_fn: Callable | None = None):
        super().__init__(out, replay, device, measure_fn=measure_fn, save_torch_fn=save_torch_fn)
        self.batch_fit_fn = batch_fit_fn or _default_batch_fit

    @staticmethod
    def _shape_group(task: TaskData) -> tuple:
        return (tuple(task.x_support.shape), tuple(task.x_query.shape),
                task.provenance.get("family"))

    def measure_many(self, entries: Iterable[tuple[torch.Tensor, TaskData, str]]):
        requested = list(entries)
        groups: dict[tuple, dict[str, tuple[torch.Tensor, TaskData, str, Path]]] = defaultdict(dict)
        for mask, task, origin in requested:
            clean = mask.detach().cpu().float()
            digest = measurement_digest(clean, task, self.replay.protocol)
            groups[self._shape_group(task)][digest] = (clean, task, origin, self._path(clean, task, fresh=False))

        for rows_by_digest in groups.values():
            rows = list(rows_by_digest.values())
            if all(path.exists() for _, _, _, path in rows):
                continue
            supports_full_batch = self.replay.protocol.batch_size is None or all(
                self.replay.protocol.batch_size >= len(task.x_support) for _, task, _, _ in rows)
            is_pattern = all(task.provenance.get("family") == "pattern" for _, task, _, _ in rows)
            if not supports_full_batch or not is_pattern:
                for mask, task, origin, _ in rows:
                    self.measure(mask, task, origin)
                continue
            results = self.batch_fit_fn(torch.stack([mask for mask, _, _, _ in rows]),
                                        [task for _, task, _, _ in rows], self.replay.protocol,
                                        self.device)
            if len(results) != len(rows):
                raise ValueError("batched fitter returned a result count different from its input batch")
            for (mask, task, _, path), result in zip(rows, results):
                if not path.exists():
                    self._save_torch(path, _payload(mask, task, self.replay.protocol, result))
        return [self.measure(mask, task, origin) for mask, task, origin in requested]
