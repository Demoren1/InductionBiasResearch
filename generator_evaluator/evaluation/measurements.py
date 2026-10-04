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

from generator_evaluator.data.adapters import measure_mask as _default_measure_mask
from generator_evaluator.storage.artifacts import save_torch
from generator_evaluator.data.types import RealReplay, TaskData, tensor_hash
from generator_evaluator.evaluation.pattern import fit_pattern_batch as _default_batch_fit


MeasureFn = Callable[..., dict]
BatchFitFn = Callable[[torch.Tensor, Sequence[TaskData], object, str], list[dict]]


_REPLAY_RESULT_FIELDS = ("label_source", "fixed_horizon", "protocol_id", "replica_losses",
                        "seeds", "plateau_flags")


def _compact_result(result: dict) -> dict:
    """Keep only the small terminal label needed to train/evaluate the surrogate."""
    compact = {key: result[key] for key in _REPLAY_RESULT_FIELDS}
    if isinstance(compact["replica_losses"], torch.Tensor):
        compact["replica_losses"] = compact["replica_losses"].detach().cpu().float().tolist()
    else:
        compact["replica_losses"] = [float(value) for value in compact["replica_losses"]]
    if isinstance(compact["plateau_flags"], torch.Tensor):
        compact["plateau_flags"] = compact["plateau_flags"].detach().cpu().bool().tolist()
    else:
        compact["plateau_flags"] = [bool(value) for value in compact["plateau_flags"]]
    compact["seeds"] = list(compact["seeds"])
    return compact


def _same_measurement_protocol(left: dict, right: dict) -> bool:
    """Check measurement identity while allowing subset refits to differ numerically."""
    first, second = _compact_result(left), _compact_result(right)
    losses = torch.tensor(second["replica_losses"], dtype=torch.float32)
    return (first["label_source"] == second["label_source"] and
            first["fixed_horizon"] == second["fixed_horizon"] and
            first["protocol_id"] == second["protocol_id"] and
            first["seeds"] == second["seeds"] and
            losses.ndim == 1 and len(losses) == len(first["replica_losses"]) and
            bool(torch.isfinite(losses).all()))


def _result_key(row: dict) -> tuple[str, str, str]:
    return row["mask_key"], row["task_id"], row["protocol_id"]


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
                 measure_fn: MeasureFn | None = None, *, save_torch_fn: Callable | None = None,
                 persist_artifacts: bool = True):
        self.out = Path(out)
        self.replay = replay
        self.device = str(device)
        self.measure_fn = measure_fn or _default_measure_mask
        self._save_torch = save_torch_fn or save_torch
        self.persist_artifacts = bool(persist_artifacts)
        self._result_cache: dict[str, dict] = {}
        self.last_results: dict[tuple[str, str, str], dict] = {}
        self._record_index: dict[str, dict] = {}
        self._indexed_records = None
        self._indexed_row_count = 0
        if not self.persist_artifacts:
            for row in replay.records:
                key = row.get("measurement_key")
                if row.get("artifact_ephemeral") is True and key:
                    self._result_cache[key] = {
                        "label_source": row["label_source"],
                        "fixed_horizon": row.get("fixed_horizon", True),
                        "protocol_id": row["protocol_id"],
                        "replica_losses": list(row["replica_losses"]),
                        "seeds": list(row["seeds"]),
                        "plateau_flags": list(row["plateau_flags"]),
                    }

    def get_result(self, row: dict) -> dict:
        """Return a full result retained by the immediately preceding opted-in call."""
        try:
            return self.last_results[_result_key(row)]
        except KeyError as error:
            raise KeyError("full measurement result was not retained for this replay row") from error

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

    def _append(self, mask: torch.Tensor, task: TaskData, origin: str, path: Path, result: dict,
                *, fresh: bool = False):
        records = self.replay.records
        if self._indexed_records is not records or self._indexed_row_count != len(records):
            self._record_index = {}
            for row in records:
                key = (row.get("artifact_path") if self.persist_artifacts else
                       row.get("measurement_key") if row.get("artifact_ephemeral") is True else None)
                if key is not None:
                    self._record_index.setdefault(key, row)
            self._indexed_records = records
            self._indexed_row_count = len(records)
        key = str(path.resolve()) if self.persist_artifacts else path.stem
        existing = None if fresh else self._record_index.get(key)
        if existing is not None:
            return existing
        row = self.replay.append(
            mask, task, result, origin=origin,
            artifact_path=path if self.persist_artifacts else None,
            measurement_key=None if self.persist_artifacts else key)
        self._record_index.setdefault(key, row)
        self._indexed_row_count = len(records)
        return row

    def measure(self, mask: torch.Tensor, task: TaskData, origin: str, *, fresh: bool = False,
                retain_results: bool = False):
        self.last_results = {}
        mask = mask.detach().cpu().float()
        path = self._path(mask, task, fresh=fresh)
        digest = path.stem
        full_result = None
        if not self.persist_artifacts and digest in self._result_cache and not retain_results:
            result = self._result_cache[digest]
        elif self.persist_artifacts and path.exists():
            result = self._read(path, mask, task)
        else:
            initialization_seed = (self.replay.protocol.seed + 1_000_003 * (len(self.replay.records) + 1)
                                   if fresh else None)
            full_result = self.measure_fn(mask, task, self.replay.protocol, device=self.device,
                                           initialization_seed=initialization_seed)
            if not self.persist_artifacts and digest in self._result_cache:
                if not _same_measurement_protocol(self._result_cache[digest], full_result):
                    raise ValueError("transient refit differs from the cached measurement protocol")
                result = full_result
            elif self.persist_artifacts:
                self._save_torch(path, _payload(mask, task, self.replay.protocol, full_result))
                result = full_result
            else:
                result = _compact_result(full_result)
                self._result_cache[digest] = result
                if retain_results:
                    result = full_result
        row = self._append(mask, task, origin, path, result, fresh=fresh)
        if retain_results and (full_result is not None or self.persist_artifacts):
            self.last_results[_result_key(row)] = result
        return row, result

    def measure_many(self, entries: Iterable[tuple[torch.Tensor, TaskData, str]], *,
                     retain_results: bool = False):
        """Sequential fallback useful for non-vectorizable domains."""
        self.last_results = {}
        output = []
        retained_results = {}
        transient_by_digest = {}
        for mask, task, origin in entries:
            digest = self._path(mask.detach().cpu().float(), task, fresh=False).stem
            if retain_results and digest in transient_by_digest:
                result = transient_by_digest[digest]
                row = self._append(mask.detach().cpu().float(), task, origin,
                                   self._path(mask.detach().cpu().float(), task, fresh=False), result)
                item = (row, result)
                self.last_results = {_result_key(row): result}
            else:
                # Scalar calls clear the recent result map; collect it after
                # each call so callers can retrieve rows from this batch.
                item = self.measure(mask, task, origin, retain_results=retain_results)
            if retain_results:
                # ``measure`` clears this mapping on each call, so merge its
                # full result into the batch's bounded result view.
                retained_results.update(self.last_results)
                if _result_key(item[0]) in self.last_results:
                    transient_by_digest[digest] = item[1]
            output.append(item)
        self.last_results = retained_results
        return output


class BatchedMeasurementStore(MeasurementStore):
    """Pattern specialization that preserves cache and replay semantics.

    Cached and missing entries intentionally stay in the same numerical batch:
    after a partial artifact write, rerunning produces the exact same states
    for the missing artifacts as an uninterrupted batch would.
    """

    def __init__(self, out: str | Path, replay: RealReplay, device: str,
                 measure_fn: MeasureFn | None = None, batch_fit_fn: BatchFitFn | None = None,
                 *, save_torch_fn: Callable | None = None, persist_artifacts: bool = True):
        super().__init__(out, replay, device, measure_fn=measure_fn,
                         save_torch_fn=save_torch_fn, persist_artifacts=persist_artifacts)
        self.batch_fit_fn = batch_fit_fn or _default_batch_fit

    @staticmethod
    def _shape_group(task: TaskData) -> tuple:
        return (tuple(task.x_support.shape), tuple(task.x_query.shape),
                task.provenance.get("family"))

    def measure_many(self, entries: Iterable[tuple[torch.Tensor, TaskData, str]], *,
                     retain_results: bool = False):
        self.last_results = {}
        requested = list(entries)
        groups: dict[tuple, dict[str, tuple[torch.Tensor, TaskData, str, Path]]] = defaultdict(dict)
        for mask, task, origin in requested:
            clean = mask.detach().cpu().float()
            digest = measurement_digest(clean, task, self.replay.protocol)
            groups[self._shape_group(task)][digest] = (clean, task, origin, self._path(clean, task, fresh=False))

        transient_results: dict[str, dict] = {}
        for rows_by_digest in groups.values():
            rows = list(rows_by_digest.values())
            if self.persist_artifacts:
                needs_fit = rows
                if all(path.exists() for _, _, _, path in rows):
                    continue
            else:
                needs_fit = rows
                if not retain_results and all(path.stem in self._result_cache
                                              for _, _, _, path in rows):
                    continue
            if not needs_fit:
                continue
            supports_full_batch = self.replay.protocol.batch_size is None or all(
                self.replay.protocol.batch_size >= len(task.x_support) for _, task, _, _ in needs_fit)
            is_pattern = all(task.provenance.get("family") == "pattern" for _, task, _, _ in needs_fit)
            if not supports_full_batch or not is_pattern:
                for mask, task, origin, _ in needs_fit:
                    row, result = self.measure(mask, task, origin, retain_results=retain_results)
                    if retain_results and _result_key(row) in self.last_results:
                        transient_results[measurement_digest(mask, task, self.replay.protocol)] = result
                continue
            results = self.batch_fit_fn(torch.stack([mask for mask, _, _, _ in needs_fit]),
                                        [task for _, task, _, _ in needs_fit], self.replay.protocol,
                                        self.device)
            if len(results) != len(needs_fit):
                raise ValueError("batched fitter returned a result count different from its input batch")
            for (mask, task, _, path), result in zip(needs_fit, results):
                digest = path.stem
                if self.persist_artifacts:
                    if not path.exists():
                        self._save_torch(path, _payload(mask, task, self.replay.protocol, result))
                else:
                    compact = _compact_result(result)
                    cached = self._result_cache.get(digest)
                    if cached is not None and not _same_measurement_protocol(cached, result):
                        raise ValueError("transient refit differs from the cached measurement protocol")
                    self._result_cache.setdefault(digest, compact)
                    if retain_results:
                        transient_results[digest] = result

        output = []
        retained_results = {}
        for mask, task, origin in requested:
            digest = measurement_digest(mask, task, self.replay.protocol)
            path = self._path(mask, task, fresh=False)
            if digest in transient_results:
                result = transient_results[digest]
                row = self._append(mask, task, origin, path, result)
                if retain_results:
                    retained_results[_result_key(row)] = result
                output.append((row, result))
            else:
                row, result = self.measure(mask, task, origin, retain_results=retain_results)
                if retain_results:
                    retained_results.update(self.last_results)
                output.append((row, result))
        self.last_results = retained_results
        return output
