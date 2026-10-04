"""Multi-device real-measurement store for independent pattern and DeepSets fits.

CUDA work is isolated in spawned workers.  Artifact creation and replay
mutation deliberately remain in the parent process so their ordering and
cache semantics are identical to :class:`MeasurementStore`.
"""
from __future__ import annotations

from generator_evaluator.storage.artifacts import save_torch
from generator_evaluator.evaluation.pattern import fit_pattern_batch

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from io import BytesIO
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

from generator_evaluator.data.types import TaskData
from generator_evaluator.evaluation.measurements import (
    MeasurementStore, _compact_result, _payload, _result_key, _same_measurement_protocol,
    measurement_digest,
)
from generator_evaluator.storage.progress import progress


BatchFitFn = Callable[[torch.Tensor, object, object, str], list[dict]]


def _cpu_value(value):
    """Return a pickle-safe result with no CUDA storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_value(item) for item in value)
    if isinstance(value, list):
        return [_cpu_value(item) for item in value]
    return value


def _cpu_task(task: TaskData) -> TaskData:
    """Do not serialize CUDA tensors to a child process."""
    return replace(task,
                   x_support=task.x_support.detach().cpu(),
                   y_support=task.y_support.detach().cpu(),
                   x_query=task.x_query.detach().cpu(),
                   y_query=task.y_query.detach().cpu(),
                   context=task.context.detach().cpu(),
                   support_ids=task.support_ids.detach().cpu(),
                   query_ids=task.query_ids.detach().cpu())


def _worker_initializer(device: str) -> None:
    """Keep a spawned CUDA worker small and bind it to its sole GPU."""
    torch.set_num_threads(1)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)


def _deepsets_worker(masks: torch.Tensor, task: TaskData, protocol, device: str,
                     initialization_seeds=None) -> list[dict]:
    """Spawn-safe worker entry point; imported here to avoid CUDA at parent import."""
    from generator_evaluator.evaluation.deepsets import fit_deepsets_batch

    results = fit_deepsets_batch(masks.detach().cpu(), _cpu_task(task), protocol, device,
                                initialization_seeds=initialization_seeds)
    if len(results) != len(masks):
        raise ValueError("batched fitter returned a result count different from its input batch")
    return [_cpu_value(result) for result in results]


def _pattern_worker(masks: torch.Tensor, tasks: Sequence[TaskData], protocol, device: str,
                    initialization_seeds=None) -> list[dict]:
    """Spawn-safe worker entry point for full-batch pattern fits."""
    from generator_evaluator.evaluation.pattern import fit_pattern_batch

    worker_tasks = [_cpu_task(task) for task in tasks]
    results = fit_pattern_batch(masks.detach().cpu(), worker_tasks, protocol, device,
                                initialization_seeds=initialization_seeds)
    if len(results) != len(masks):
        raise ValueError("batched fitter returned a result count different from its input batch")
    return [_cpu_value(result) for result in results]


def _pattern_batch_args(masks, tasks, protocol, device, initialization_seeds=None):
    args = (masks, [_cpu_task(task) for task in tasks], protocol, device)
    return args if initialization_seeds is None else args + (list(initialization_seeds),)


def _balanced_chunk_sizes(row_count: int, batch_size: int, device_count: int) -> list[int]:
    """Pack full batches, balancing smaller groups across every available device."""
    if row_count < 1:
        return []
    if device_count > 1 and row_count <= batch_size * device_count:
        chunk_count = min(row_count, device_count)
        quotient, remainder = divmod(row_count, chunk_count)
        return [quotient + int(index < remainder) for index in range(chunk_count)]
    return [min(batch_size, row_count - start)
            for start in range(0, row_count, batch_size)]


def _serialized_pattern_worker(payload: bytes) -> bytes:
    """Fit a pattern batch with self-contained byte-only process messages."""
    args = _decode_payload(payload)
    return _encode_payload(_pattern_worker(*args))


def iter_pattern_candidate_batches(masks: torch.Tensor, tasks: Sequence[TaskData], protocol,
                                  *, devices: Sequence[str], batch_size: int,
                                  initialization_seeds: Sequence[int], device: str | None = None):
    """Yield deterministic pattern candidate batches, bounded by worker count.

    The caller fixes candidate chunk boundaries with ``batch_size``. At most
    one chunk per CUDA worker is serialized and in flight; each wave is yielded
    in source order after all of its workers finish. CPU and one-device paths
    call the same batched fitter directly without process startup.
    """
    clean = torch.as_tensor(masks, dtype=torch.float32).detach().cpu().contiguous()
    if clean.ndim != 3 or len(clean) != len(tasks):
        raise ValueError("pattern candidate masks and tasks must have matching batch axes")
    seeds = list(map(int, initialization_seeds))
    if len(seeds) != len(clean):
        raise ValueError("one initialization seed is required per pattern candidate")
    if not devices:
        raise ValueError("devices must contain at least one device")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    targets = tuple(dict.fromkeys(str(value) for value in devices))
    use_workers = len(targets) > 1 and all(value.startswith("cuda") for value in targets)
    chunk_sizes = _balanced_chunk_sizes(len(clean), batch_size, len(targets) if use_workers else 1)
    if not use_workers:
        target = str(device) if device is not None else targets[0]
        from generator_evaluator.evaluation.pattern import fit_pattern_batch

        start = 0
        for chunk_size in chunk_sizes:
            stop = start + chunk_size
            fitted = fit_pattern_batch(clean[start:stop], tasks[start:stop], protocol, target,
                                       initialization_seeds=seeds[start:stop])
            yield start, fitted
            start = stop
        return

    executors: dict[str, ProcessPoolExecutor] = {}
    try:
        chunk_bounds = []
        start = 0
        for chunk_size in chunk_sizes:
            chunk_bounds.append((start, start + chunk_size))
            start += chunk_size
        for wave_start in range(0, len(chunk_bounds), len(targets)):
            wave = chunk_bounds[wave_start:wave_start + len(targets)]
            futures = {}
            for offset, (start, stop) in enumerate(wave):
                target = targets[offset]
                executor = executors.get(target)
                if executor is None:
                    executor = ProcessPoolExecutor(
                        max_workers=1, mp_context=mp.get_context("spawn"),
                        initializer=_worker_initializer, initargs=(target,))
                    executors[target] = executor
                args = _pattern_batch_args(clean[start:stop], tasks[start:stop], protocol,
                                           target, seeds[start:stop])
                future = executor.submit(_serialized_pattern_worker, _encode_payload(args))
                futures[future] = offset
            fitted_by_offset = {}
            for future in as_completed(futures):
                fitted_by_offset[futures[future]] = _decode_payload(future.result())
            for offset, (start, _) in enumerate(wave):
                yield start, fitted_by_offset[offset]
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)


def _encode_payload(value) -> bytes:
    """Inline tensor storage instead of sending multiprocessing FD handles."""
    buffer = BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _decode_payload(payload: bytes):
    return torch.load(BytesIO(payload), map_location="cpu", weights_only=False)


def _serialized_deepsets_worker(payload: bytes) -> bytes:
    """Only bytes cross the process boundary, including all child results.

    A fit contains many state/history/Adam tensors. Passing them directly
    through ProcessPoolExecutor invokes PyTorch's file-descriptor reducers,
    whose resource-sharer lifetime/descriptor limits can break the pool.
    The self-contained archive keeps tensor transport independent of them.
    """
    args = _decode_payload(payload)
    return _encode_payload(_deepsets_worker(*args))


class ParallelMeasurementStore(MeasurementStore):
    """Batch and distribute independent real measurements over CUDA devices.

    ``batch_fit_fn`` is intentionally an optional testing seam.  Supplying it
    keeps all work in the parent process, which also makes a CPU-only setup a
    normal, deterministic execution path.
    """

    def __init__(self, out: str | Path, replay, device: str, *, devices: Sequence[str],
                 batch_size: int, measure_fn=None, batch_fit_fn: BatchFitFn | None = None,
                 pattern_batch_fit_fn: BatchFitFn | None = None, save_torch_fn=None,
                 persist_artifacts: bool = True):
        super().__init__(out, replay, device, measure_fn=measure_fn,
                         save_torch_fn=save_torch_fn, persist_artifacts=persist_artifacts)
        if not devices:
            raise ValueError("devices must contain at least one device")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.devices = tuple(dict.fromkeys(str(item) for item in devices))
        self.batch_size = int(batch_size)
        self.batch_fit_fn = batch_fit_fn
        self.pattern_batch_fit_fn = pattern_batch_fit_fn
        self._executors: dict[str, ProcessPoolExecutor] = {}
        self._closed = False

    @staticmethod
    def _group_key(mask: torch.Tensor, task: TaskData) -> tuple:
        family = task.provenance.get("family")
        shape = (family, tuple(mask.shape), tuple(task.x_support.shape), tuple(task.x_query.shape))
        # PatternFitEngine accepts one task per candidate, so equal-sized
        # pattern tasks can share a batch. DeepSets packing requires a common
        # task because its task observations are uploaded once per batch.
        return shape if family == "pattern" else (task.fingerprint, *shape)

    def _is_parallel(self, rows) -> bool:
        family = rows[0][2].provenance.get("family")
        return (self.batch_fit_fn is None and self.pattern_batch_fit_fn is None and
                family in ("pattern", "deepsets") and
                len(self.devices) > 1 and all(device.startswith("cuda") for device in self.devices) and
                all(task.provenance.get("family") == family for _, _, task, _, _ in rows))

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("parallel measurement store is closed")

    def _executor(self, device: str) -> ProcessPoolExecutor:
        self._ensure_open()
        if device not in self._executors:
            # One single-worker executor per GPU pins its sole process to the
            # device passed into the domain worker, without ever forking CUDA.
            self._executors[device] = ProcessPoolExecutor(
                max_workers=1, mp_context=mp.get_context("spawn"),
                initializer=_worker_initializer, initargs=(device,))
        return self._executors[device]

    def _fit_direct(self, rows, initialization_seeds=None) -> list[dict]:
        masks = torch.stack([mask for _, mask, _, _, _ in rows])
        tasks = [row[2] for row in rows]
        task = tasks[0]
        if self.batch_fit_fn is not None:
            extras = {} if initialization_seeds is None else dict(initialization_seeds=initialization_seeds)
            fit_tasks = task if all(item.fingerprint == task.fingerprint for item in tasks) else tasks
            results = self.batch_fit_fn(masks, fit_tasks, self.replay.protocol, self.devices[0], **extras)
        elif task.provenance.get("family") == "pattern":
            if self.pattern_batch_fit_fn is not None:
                results = self.pattern_batch_fit_fn(masks, tasks, self.replay.protocol,
                                                    self.devices[0],
                                                    **({} if initialization_seeds is None else
                                                       dict(initialization_seeds=initialization_seeds)))
            else:
                results = _pattern_worker(masks, tasks, self.replay.protocol, self.devices[0],
                                          initialization_seeds)
        elif task.provenance.get("family") == "deepsets":
            # Even one device benefits from packing candidates on its model
            # axis; CPU takes this direct route too, which keeps tests and
            # small local runs free from process-spawn overhead.
            results = _deepsets_worker(masks, task, self.replay.protocol, self.devices[0], initialization_seeds)
        else:
            # A single CUDA device has no process-level parallelism to gain;
            # execute through the standard per-mask fitter, retaining fresh
            # initialization/cache behaviour for unsupported domains.
            results = [self.measure_fn(mask, task, self.replay.protocol, device=self.device,
                                       initialization_seed=None) for _, mask, task, _, _ in rows]
        if len(results) != len(rows):
            raise ValueError("batched fitter returned a result count different from its input batch")
        return [_cpu_value(result) for result in results]

    @staticmethod
    def _write_chunk(chunk, fitted, results, store, bar, *, retain_results: bool) -> None:
        if len(fitted) != len(chunk):
            raise ValueError("batched fitter returned a result count different from its input batch")
        written = 0
        for (digest, mask, task, _, path), result in zip(chunk, fitted):
            if digest in results:
                continue
            if store.persist_artifacts and path.exists():
                # A concurrent writer or duplicate path can materialize the
                # artifact after the initial cache scan; restore it here.
                results[digest] = store._read(path, mask, task)
            elif not store.persist_artifacts:
                compact = _compact_result(result)
                cached = store._result_cache.get(digest)
                if cached is not None and not _same_measurement_protocol(cached, result):
                    raise ValueError("transient refit differs from the cached measurement protocol")
                store._result_cache.setdefault(digest, compact)
                results[digest] = result if retain_results else compact
                written += 1
            else:
                results[digest] = result
                store._save_torch(path, _payload(mask, task, store.replay.protocol, result))
                written += 1
        bar.update(written)

    def measure(self, mask: torch.Tensor, task: TaskData, origin: str, *, fresh: bool = False,
                retain_results: bool = False):
        self._ensure_open()
        # Fresh initializations intentionally stay on the well-tested single
        # measurement path: their cache key depends on replay append order.
        if fresh:
            return super().measure(mask, task, origin, fresh=True, retain_results=retain_results)
        return self.measure_many([(mask, task, origin)], retain_results=retain_results)[0]

    def measure_many(self, entries: Iterable[tuple[torch.Tensor, TaskData, str]], *,
                     desc: str = "Real labels", initialization_seeds: Sequence[int] | None = None,
                     retain_results: bool = False):
        self._ensure_open()
        self.last_results = {}
        requested = list(entries)
        if not requested:
            return []
        seeds = ([None] * len(requested) if initialization_seeds is None
                 else list(map(int, initialization_seeds)))
        if len(seeds) != len(requested):
            raise ValueError("one initialization seed is required per measurement")

        # The same digest can occur many times in a request.  Fit it once,
        # then append/cache in original order below.
        unique: dict[str, tuple[torch.Tensor, TaskData, str, Path]] = {}
        ordered: list[tuple[torch.Tensor, TaskData, str, Path, str]] = []
        seeds_by_digest = {}
        for (mask, task, origin), seed in zip(requested, seeds):
            clean = mask.detach().cpu().float()
            digest = measurement_digest(clean, task, self.replay.protocol)
            path = self._path(clean, task, fresh=False)
            if seed is not None:
                # Teacher candidates with identical dense masks still have
                # independent fits; seed is part of their cache identity.
                digest += f"_init_{seed}"
                path = self.out / "children" / f"{digest}.pt"
            seeds_by_digest[digest] = seed
            unique.setdefault(digest, (clean, task, origin, path))
            ordered.append((clean, task, origin, path, digest))

        results: dict[str, dict] = {}
        grouped: dict[tuple, list[tuple[str, torch.Tensor, TaskData, str, Path]]] = defaultdict(list)
        for digest, (mask, task, origin, path) in unique.items():
            if self.persist_artifacts and path.exists():
                results[digest] = self._read(path, mask, task)
            elif (not self.persist_artifacts and not retain_results and
                  digest in self._result_cache):
                results[digest] = self._result_cache[digest]
            grouped[self._group_key(mask, task)].append((digest, mask, task, origin, path))

        # Chunk before filtering cache hits.  A partially completed run must
        # retain exactly the same packed model-axis shape as an uninterrupted
        # run, or a resumed deterministic fit can produce different states.
        all_chunks = []
        for rows in grouped.values():
            worker_count = len(self.devices) if self._is_parallel(rows) else 1
            chunk_sizes = _balanced_chunk_sizes(len(rows), self.batch_size, worker_count)
            start = 0
            for chunk_size in chunk_sizes:
                all_chunks.append(rows[start:start + chunk_size])
                start += chunk_size
        chunks = [chunk for chunk in all_chunks
                  if any(digest not in results for digest, _, _, _, _ in chunk)]
        if chunks:
            bar = progress(desc=desc, total=sum(
                digest not in results for chunk in chunks for digest, _, _, _, _ in chunk), unit="fit")
            try:
                parallel = all(self._is_parallel(chunk) for chunk in chunks)
                if parallel:
                    # A wave has at most one job per worker. Results are
                    # collected and artifacts committed in chunk order, so
                    # both in-flight tensor memory and parent-side write order
                    # stay bounded and deterministic.
                    for wave_start in range(0, len(chunks), len(self.devices)):
                        wave = chunks[wave_start:wave_start + len(self.devices)]
                        futures = {}
                        for offset, chunk in enumerate(wave):
                            device = self.devices[offset]
                            masks = torch.stack([row[1] for row in chunk])
                            family = chunk[0][2].provenance.get("family")
                            seed_values = (None if initialization_seeds is None else
                                           [seeds_by_digest[row[0]] for row in chunk])
                            if family == "pattern":
                                tasks = [row[2] for row in chunk]
                                args = _pattern_batch_args(masks, tasks, self.replay.protocol,
                                                           device, seed_values)
                                worker = _serialized_pattern_worker
                            else:
                                worker_task = _cpu_task(chunk[0][2])
                                args = (masks, worker_task, self.replay.protocol, device)
                                if seed_values is not None:
                                    args += (seed_values,)
                                worker = _serialized_deepsets_worker
                            future = self._executor(device).submit(worker, _encode_payload(args))
                            futures[future] = offset
                        fitted_by_offset = {}
                        for future in as_completed(futures):
                            fitted_by_offset[futures[future]] = _decode_payload(future.result())
                        for offset, chunk in enumerate(wave):
                            self._write_chunk(chunk, fitted_by_offset[offset], results, self, bar,
                                              retain_results=retain_results)
                        fitted_by_offset.clear()
                else:
                    for chunk in chunks:
                        seed_values = (None if initialization_seeds is None else
                                       [seeds_by_digest[row[0]] for row in chunk])
                        fitted = self._fit_direct(chunk, seed_values)
                        self._write_chunk(chunk, fitted, results, self, bar,
                                          retain_results=retain_results)
                        del fitted
            except BaseException:
                # Leave the store reusable after a partial artifact write.
                # Submitted workers only compute results; parent-side cache
                # writes are already ordered below, so a retry can safely
                # reuse completed artifacts and refit the original chunk.
                raise
            finally:
                bar.close()

        output = []
        retained_results = {}
        for mask, task, origin, path, digest in ordered:
            result = results[digest]
            row = self._append(mask, task, origin, path, result)
            output.append((row, result))
            if retain_results and (self.persist_artifacts or digest not in self._result_cache or
                                   result is not self._result_cache.get(digest)):
                retained_results[_result_key(row)] = result
        self.last_results = retained_results
        return output

    def close(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        for executor in self._executors.values():
            executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        self._executors.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(wait=exc_type is None, cancel_futures=exc_type is not None)
        return False


class CooperativeBatchedMeasurementStore(ParallelMeasurementStore):
    """Preserve the cooperative entry point with injectable fit/write hooks."""

    def __init__(self, out, replay, device, *, devices=None, batch_size=128,
                 persist_artifacts: bool = True):
        selected_devices = tuple(devices) if devices else (device,)

        def fit_pattern(masks, tasks, protocol, target, **kwargs):
            if not isinstance(tasks, (list, tuple)):
                tasks = [tasks] * len(masks)
            return fit_pattern_batch(masks, tasks, protocol, target, **kwargs)

        spawned_multi_gpu = (len(selected_devices) > 1 and
                             all(str(target).startswith("cuda") for target in selected_devices))
        super().__init__(out, replay, device, devices=selected_devices, batch_size=batch_size,
                         pattern_batch_fit_fn=None if spawned_multi_gpu else fit_pattern,
                         save_torch_fn=lambda *args, **kwargs: save_torch(*args, **kwargs),
                         persist_artifacts=persist_artifacts)
