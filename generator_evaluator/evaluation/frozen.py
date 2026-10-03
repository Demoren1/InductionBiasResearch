"""Post-run diagnostic for frozen DeepSets common and dense masks."""
from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ProcessPoolExecutor
import csv
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
from collections.abc import Mapping
from typing import Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from generator_evaluator.storage.artifacts import save_json
from generator_evaluator.data.types import InnerProtocol, TaskData, tensor_hash
from generator_evaluator.evaluation.parallel import (_cpu_task, _decode_payload, _encode_payload,
                                    _worker_initializer)


def _load(path: Path):
    """Mmap large run artifacts so the frozen model collection stays off heap."""
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def _masks(run: Path, *, include_generators: bool = False) -> tuple[dict[str, torch.Tensor], dict]:
    frozen_path = run / "frozen.pt"
    if not (run / "COMPLETE").is_file() or not frozen_path.is_file():
        raise ValueError(f"completed run with frozen.pt is required: {run}")
    frozen = _load(frozen_path)
    methods = frozen.get("methods", {})
    if "common" not in methods or "dense" not in methods:
        raise ValueError("frozen.pt must contain common and dense masks")
    result = {name: torch.as_tensor(methods[name], dtype=torch.float32).detach().cpu().contiguous()
              for name in ("common", "dense")}
    for name, mask in result.items():
        if (mask.ndim != 2 or mask.shape[0] != 784 or mask.shape[1] < 1 or
                not torch.isfinite(mask).all() or not bool(((mask == 0) | (mask == 1)).all())):
            raise ValueError(f"frozen {name} mask must be finite binary [784, hidden]")
    if result["common"].shape != result["dense"].shape or not bool((result["dense"] == 1).all()):
        raise ValueError("frozen common and dense masks must share an architecture with a dense control")
    if include_generators:
        proposal_path = run / "final_generator_proposals.pt"
        if not proposal_path.is_file():
            raise FileNotFoundError(f"current run has no final generator proposals: {proposal_path}")
        proposals = _load(proposal_path).get("masks")
        if not isinstance(proposals, dict) or not proposals:
            raise ValueError("final_generator_proposals.pt must contain a nonempty masks mapping")
        for name, value in proposals.items():
            mask = torch.as_tensor(value, dtype=torch.float32).detach().cpu().contiguous()
            if (not isinstance(name, str) or name in result or mask.shape != result["common"].shape or
                    not torch.isfinite(mask).all() or not bool(((mask == 0) | (mask == 1)).all())):
                raise ValueError(f"final generator proposal {name!r} must be a matching binary mask")
            result[name] = mask
    return result, frozen


def _candidate_masks(path: str | Path | None, architecture: torch.Size,
                     reserved_names: set[str]) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"candidate mask file does not exist: {source}")
    payload = _load(source)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("candidate mask file must contain a nonempty name-to-tensor mapping")
    result = {}
    for name, value in payload.items():
        if not isinstance(name, str) or not name:
            raise ValueError("candidate mask names must be nonempty strings")
        if name in reserved_names:
            raise ValueError(f"candidate mask name is reserved or already in use: {name}")
        if not torch.is_tensor(value):
            raise ValueError(f"candidate mask {name!r} must be a tensor")
        value = value.detach().cpu()
        if (value.is_complex() or value.shape != architecture or
                not torch.isfinite(value).all() or not bool(((value == 0) | (value == 1)).all())):
            raise ValueError(f"candidate mask {name!r} must be finite binary with shape {tuple(architecture)}")
        result[name] = value.float().contiguous()
    return result


def _task_manifest(tasks: Sequence[TaskData], partition: str) -> list[dict]:
    split, role = (("validation", "selection") if partition == "selection" else ("test", "sealed_test"))
    rows = []
    for index, task in enumerate(tasks):
        expected_id = f"deepsets:{index}:selection" if partition == "selection" else f"deepsets:test:{index}"
        if (not isinstance(task, TaskData) or task.split != split or
                task.provenance.get("family") != "deepsets" or task.provenance.get("domain") != "deepsets" or
                task.provenance.get("role") != role or task.x_support.ndim != 3 or task.x_support.shape[1:] != (5, 784)):
            raise ValueError(f"invalid saved DeepSets {partition} task at position {index}")
        if task.task_id != expected_id:
            raise ValueError(f"saved {partition} task order or role is invalid at position {index}")
        rows.append({"index": index, "task_id": task.task_id, "split": split, "role": role,
                     "fingerprint": task.fingerprint, "task_index": task.provenance.get("task_index"),
                     "heldout_condition": task.provenance.get("heldout_condition")})
    return rows


def _fit_batch(names, masks, task, protocol, device):
    """Keep only terminal NMSE and seed keys; never serialize child states."""
    from generator_evaluator.evaluation.deepsets import fit_deepsets_batch

    fitted = fit_deepsets_batch(masks, _cpu_task(task), protocol, device,
                                initialization_seeds=[protocol.seed] * len(masks))
    if len(fitted) != len(masks):
        raise ValueError("DeepSets fitter returned a result count different from its masks")
    expected = [f"{protocol.seed}:{i}" for i in range(protocol.replicas)]
    losses = {}
    for name, row in zip(names, fitted):
        values = np.asarray(row.get("replica_losses"), dtype=np.float64)
        if values.shape != (protocol.replicas,) or not np.isfinite(values).all():
            raise ValueError("DeepSets fit must return one finite NMSE per requested replica")
        if row.get("seeds") != expected:
            raise ValueError("frozen methods did not share the requested initialization seeds")
        losses[name] = values.tolist()
    return {"losses": losses, "seeds": expected}


def _serialized_worker(payload: bytes) -> bytes:
    names, masks, task, protocol, device = _decode_payload(payload)
    return _encode_payload(_fit_batch(names, masks, task, protocol, device))


def _paired_stats(deltas, unit: str) -> dict:
    from scipy.stats import t
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired NMSE deltas must be a nonempty finite vector")
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None
    return {"mean_paired_delta": mean, "ci95": None if half is None else [mean - half, mean + half],
            "paired_deltas": values.tolist(), "n": len(values), "uncertainty_unit": unit}


def _read_cache(path: Path, identity: dict, methods: Sequence[str], protocol: InnerProtocol):
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    expected = [f"{protocol.seed}:{i}" for i in range(protocol.replicas)]
    if saved.get("identity") != identity or saved.get("seeds") != expected:
        return None
    losses = saved.get("losses")
    if not isinstance(losses, dict) or set(losses) != set(methods):
        return None
    for values in losses.values():
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (protocol.replicas,) or not np.isfinite(values).all():
            return None
    return saved


def _plot(path: Path, rows: Sequence[dict], aggregates: dict, methods: Sequence[str]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parts = list(aggregates)
    labels = [row["task_id"] for row in rows] + [f"{part} aggregate" for part in parts]
    width = .8 / len(methods)
    fig, ax = plt.subplots(figsize=(max(7, 1.15 * len(labels) + 3), 4.8))
    for i, method in enumerate(methods):
        stats = [row["comparisons"][method] for row in rows] + [aggregates[p][method] for p in parts]
        means = np.asarray([row["mean_paired_delta"] for row in stats])
        intervals = np.asarray([row["ci95"] or [row["mean_paired_delta"]] * 2 for row in stats])
        errors = np.asarray([means - intervals[:, 0], intervals[:, 1] - means])
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(np.arange(len(labels)) + offset, means, width, yerr=errors, capsize=3, label=method)
    ax.axhline(0., color="black", linewidth=.8)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("Paired query NMSE delta vs dense (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_frozen(run: str | Path, *, replicas: int = 16,
                    devices: Sequence[str] = ("cpu",), partition: str = "test",
                    comparison_run: str | Path | None = None,
                    show_progress: bool = False, include_generators: bool = False,
                    candidate_masks: str | Path | None = None) -> dict:
    if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 2:
        raise ValueError("replicas must be an integer of at least two")
    if partition not in ("selection", "test", "both"):
        raise ValueError("partition must be selection, test or both")
    targets = tuple(dict.fromkeys(map(str, devices)))
    if not targets or any(d != "cpu" and not d.startswith("cuda") for d in targets):
        raise ValueError("devices must contain cpu or CUDA devices")
    cuda = [torch.device(d) for d in targets if d.startswith("cuda")]
    if cuda and (not torch.cuda.is_available() or any(d.index is not None and
            d.index >= torch.cuda.device_count() for d in cuda)):
        raise ValueError("a requested CUDA device is unavailable")

    run = Path(run).expanduser().resolve()
    if not (run / "COMPLETE").is_file():
        raise ValueError(f"run is incomplete: {run}")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary.get("domain") != "deepsets":
        raise ValueError("frozen evaluation supports DeepSets runs only")
    run_spec = json.loads((run / "run_spec.json").read_text(encoding="utf-8"))
    test_spec = run_spec.get("test_spec")
    if not isinstance(test_spec, dict):
        raise ValueError("run_spec.json does not retain the sealed test specification")
    test_spec_hash = _digest(test_spec)
    masks, frozen = _masks(run, include_generators=include_generators)
    prior_path = None
    prior_hash = None
    if comparison_run is not None:
        prior_path = Path(comparison_run).expanduser().resolve()
        if not (prior_path / "COMPLETE").is_file():
            raise ValueError(f"comparison run is incomplete: {prior_path}")
        old_masks, _ = _masks(prior_path)
        if old_masks["common"].shape != masks["common"].shape:
            raise ValueError("previous common mask architecture differs from current run")
        masks["previous_common"] = old_masks["common"]
        prior_hash = tensor_hash(old_masks["common"])
    masks.update(_candidate_masks(candidate_masks, masks["common"].shape, set(masks)))
    frozen_protocol = InnerProtocol(**{**frozen["protocol"], "replicas": replicas})
    if frozen_protocol.metric != "nmse":
        raise ValueError("frozen DeepSets protocol must use NMSE")
    methods = [name for name in masks if name != "dense"]
    mask_hashes = {name: tensor_hash(mask) for name, mask in masks.items()}

    task_parts = {}
    if partition in ("selection", "both"):
        inputs = _load(run / "inputs.pt")
        selection = list(inputs["selection_tasks"])
        selection_manifest = _task_manifest(selection, "selection")
        saved_task_hashes = run_spec.get("tasks", {})
        if any(saved_task_hashes.get(row["task_id"]) != row["fingerprint"] for row in selection_manifest):
            raise ValueError("saved selection tasks differ from run_spec.json")
        if any(task.provenance.get("task_index") != i for i, task in enumerate(selection)):
            raise ValueError("saved selection task order differs from its roles")
        task_parts["selection"] = (selection, selection_manifest)
    if partition in ("test", "both"):
        tests = list(_load(run / "test_tasks.pt"))
        test_manifest = _task_manifest(tests, "test")
        expected = int(test_spec.get("test_task_count", len(test_spec.get("costs", ()))))
        if not tests or len(tests) != expected or len(test_spec.get("costs", ())) != expected:
            raise ValueError("saved test tasks differ from the sealed test specification")
        if any(task.task_id != f"deepsets:test:{i}" or task.provenance.get("task_index") != i or
               task.provenance.get("heldout_condition") != str(i) for i, task in enumerate(tests)):
            raise ValueError("saved test task order differs from its sealed roles")
        task_parts["test"] = (tests, test_manifest)

    output = run / "frozen_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    rows, misses, hits = [], [], 0
    cache_entries = {}
    for part, (tasks, manifest) in task_parts.items():
        task_hash = _digest(manifest)
        for index, (task, role_row) in enumerate(zip(tasks, manifest)):
            identity = {"partition": part, "task_manifest": task_hash,
                        "task_spec": test_spec_hash, "protocol": frozen_protocol.fingerprint,
                        "mask_hashes": mask_hashes}
            cache = output / "cache" / f"{part}-{index:04d}.json"
            found = _read_cache(cache, identity, [*methods, "dense"], frozen_protocol) if cache.exists() else None
            entry = (part, task, role_row, identity, cache)
            cache_entries[(part, index)] = entry
            if found is None:
                misses.append((part, index, task))
            else:
                hits += 1
                rows.append(_task_result(part, task, role_row, found["losses"], found["seeds"]))

    executors, completed = {}, {}
    bar = tqdm(total=len(misses), desc="Frozen DeepSets fits", unit="task", disable=not show_progress)
    try:
        for start in range(0, len(misses), len(targets)):
            wave = misses[start:start + len(targets)]
            pending, wave_error = {}, None
            for offset, (part, index, task) in enumerate(wave):
                device = targets[offset]
                names = [*methods, "dense"]
                batch = torch.stack([masks[name] for name in names])
                if device.startswith("cuda"):
                    executor = executors.get(device)
                    if executor is None:
                        executor = ProcessPoolExecutor(max_workers=1,
                            mp_context=mp.get_context("spawn"), initializer=_worker_initializer,
                            initargs=(device,))
                        executors[device] = executor
                    args = (names, batch, _cpu_task(task), frozen_protocol, device)
                    pending[executor.submit(_serialized_worker, _encode_payload(args))] = (part, index)
                else:
                    try:
                        fit = _fit_batch(names, batch, task, frozen_protocol, device)
                        _commit_fit(cache_entries, completed, part, index, fit)
                        bar.update(1)
                    except Exception as error:
                        wave_error = wave_error or error
            for future in as_completed(pending):
                part, index = pending[future]
                try:
                    fit = _decode_payload(future.result())
                    _commit_fit(cache_entries, completed, part, index, fit)
                    bar.update(1)
                except Exception as error:
                    wave_error = wave_error or error
            if wave_error is not None:
                raise wave_error
    finally:
        bar.close()
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)

    for part, index, task in misses:
        _, _, role_row, _, _ = cache_entries[(part, index)]
        fit = completed[(part, index)]
        rows.append(_task_result(part, task, role_row, fit["losses"], fit["seeds"]))
    order = {(part, task.task_id): (part_index, index)
             for part_index, (part, (tasks, _)) in enumerate(task_parts.items())
             for index, task in enumerate(tasks)}
    rows.sort(key=lambda row: order[(row["partition"], row["task_id"])])

    partition_aggregate = {part: _aggregate([row for row in rows if row["partition"] == part], methods)
                           for part in task_parts}
    aggregate = next(iter(partition_aggregate.values())) if len(partition_aggregate) == 1 else None
    result = {"domain": "deepsets", "metric": "nmse", "partition": partition,
        "current_run": str(run), "replicas": replicas, "protocol": frozen_protocol.__dict__,
        "protocol_id": frozen_protocol.fingerprint, "mask_hashes": mask_hashes,
        "include_generators": include_generators,
        "candidate_mask_file": None if candidate_masks is None else str(Path(candidate_masks).expanduser().resolve()),
        "evaluated_tasks_from": "current run saved task artifacts",
        "comparison_run": None if prior_path is None else {
            "path": str(prior_path), "method": "previous_common",
            "mask_hash": prior_hash,
            "interpretation": "previous frozen common was evaluated on current tasks; old task data and metrics were not loaded"},
        "test_spec_hash": test_spec_hash, "task_results": rows,
        "aggregate": aggregate, "partition_aggregates": partition_aggregate,
        "cache": {"directory": str(output / "cache"), "reused_tasks": hits,
                  "newly_fitted_tasks": len(misses)},
        "aggregate_uncertainty_unit": "fresh initialization averaged over fixed current tasks"}
    save_json(output / "summary.json", result)
    _write_csv(output / "paired_deltas.csv", rows, partition_aggregate, methods)
    _plot(output / "paired_deltas.png", rows, partition_aggregate, methods)
    return result


def _task_result(part, task, manifest, losses, seeds):
    dense = np.asarray(losses["dense"], dtype=np.float64)
    return {"partition": part, "task_id": task.task_id, "role": manifest["role"],
            "task_fingerprint": manifest["fingerprint"], "replica_losses": losses,
            "seed_metadata": {"base_seed": int(seeds[0].split(":")[0]),
                              "replica_ids": list(range(len(seeds))), "shared_seeds": seeds},
            "comparisons": {name: _paired_stats(np.asarray(values) - dense,
                "fresh initialization on this fixed current task")
                for name, values in losses.items() if name != "dense"}}


def _commit_fit(cache_entries, completed, part, index, fit):
    _, _, _, identity, cache = cache_entries[(part, index)]
    save_json(cache, {"identity": identity, **fit})
    completed[(part, index)] = fit


def _aggregate(rows, methods):
    result = {}
    for method in methods:
        task_by_replica = np.asarray([row["comparisons"][method]["paired_deltas"] for row in rows])
        result[method] = _paired_stats(task_by_replica.mean(axis=0),
            "fresh initialization averaged over fixed current tasks")
        result[method]["n_tasks"] = len(task_by_replica)
    return result


def _write_csv(path, rows, aggregates, methods):
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ("partition", "task_id", "role", "method", "n", "n_tasks", "mean_paired_delta",
                  "ci95_low", "ci95_high", "uncertainty_unit")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for method in methods:
                stats = row["comparisons"][method]
                ci = stats["ci95"] or (None, None)
                writer.writerow({"partition": row["partition"], "task_id": row["task_id"],
                    "role": row["role"], "method": method, "n": stats["n"], "n_tasks": 1,
                    "mean_paired_delta": stats["mean_paired_delta"], "ci95_low": ci[0],
                    "ci95_high": ci[1], "uncertainty_unit": stats["uncertainty_unit"]})
        for part, rows in aggregates.items():
            for method in methods:
                stats = rows[method]
                ci = stats["ci95"] or (None, None)
                writer.writerow({"partition": part, "task_id": "__aggregate__", "role": "aggregate",
                    "method": method, "n": stats["n"], "n_tasks": stats["n_tasks"],
                    "mean_paired_delta": stats["mean_paired_delta"],
                    "ci95_low": ci[0], "ci95_high": ci[1], "uncertainty_unit": stats["uncertainty_unit"]})


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--replicas", type=int, default=16)
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--partition", choices=("selection", "test", "both"), default="test")
    parser.add_argument("--comparison-run", type=Path)
    parser.add_argument("--include-generators", action="store_true",
                        help="also evaluate current-run final generator proposals")
    parser.add_argument("--candidate-masks", type=Path,
                        help="torch file containing a name-to-binary-mask mapping")
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    result = evaluate_frozen(args.run, replicas=args.replicas, devices=args.devices,
        partition=args.partition, comparison_run=args.comparison_run,
        show_progress=args.progress, include_generators=args.include_generators,
        candidate_masks=args.candidate_masks)
    out = Path(args.run).expanduser().resolve() / "frozen_diagnostic"
    print(json.dumps({"output": str(out), "tasks": len(result["task_results"]),
                      "aggregate": result["aggregate"] or result["partition_aggregates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
