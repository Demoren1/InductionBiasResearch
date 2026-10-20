"""Collect resumable W/G cards using packed, independent CUDA MLPs."""

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..io import (
    BANK_SCHEMA,
    DEFAULT_BANK,
    PACKAGE,
    save_json,
    save_torch,
    snapshot,
    source_hashes,
)
from ..io import sha256 as digest
from ..masks import structural_metrics
from ..models.task import bce, entropy, fit, imp, initialization
from ..problem import families, observations, reference, teacher
from ..runtime import configure_cuda, cuda_devices
from .features import extract

DEVICE = None


def cpu_state(state):
    return {key: value.detach().cpu() for key, value in state.items()}


def initialize_worker(queue, memory_gib):
    global DEVICE
    torch.set_num_threads(1)
    DEVICE = configure_cuda(queue.get(), memory_gib)


def seed_for(config, family, function, index):
    return (
        config["seed"]
        + family["id"] * 100003
        + function * 1009
        + (index + 1) * 1000003
    )


def collect(job, prepared):
    started = time.monotonic()
    config = job["config"]
    family, function = job["family"], job["function"]
    destination = Path(job["bank"]) / family["split"] / job["task"]
    destination.mkdir(parents=True, exist_ok=True)
    target = teacher(config, family, function, DEVICE)
    data = observations(
        config, family, function, target, DEVICE, ["support", "query", "test"]
    )
    models = [prepared[seed] for seed in job["seeds"]]
    training_seconds = job["amortized_training_seconds"]
    state = {
        key: torch.stack([model["state"][key].to(DEVICE) for model in models])
        for key in models[0]["state"]
    }
    original = {
        key: torch.stack([model["initial"][key].cpu() for model in models])
        for key in models[0]["initial"]
    }
    masks = torch.stack([model["mask"].to(DEVICE) for model in models])
    feature_started = time.monotonic()
    payload = extract(state, masks, data["support"])
    metrics, orders = structural_metrics(masks, reference(family, DEVICE))
    for split, values in data.items():
        loss = bce(state, masks, values).cpu().tolist()
        minimum = float(entropy(values))
        for record, value in zip(metrics, loss):
            record[f"{split}_bce"] = value
            record[f"{split}_excess_bce"] = value - minimum
    payload.update(
        {
            "schema": BANK_SCHEMA,
            "task": job["task"],
            "family": family["id"],
            "function": function,
            "family_split": family["split"],
            "network_seeds": torch.tensor(job["seeds"], dtype=torch.long),
            "network_roles": job["roles"],
            "raw_state": cpu_state(state),
            "raw_masks": masks.bool().cpu(),
            "initial_state": original,
            "metrics": metrics,
            "diagnostic_reference_orders": orders,
            "origins": [model["source"] for model in models],
            "training_layouts": [model["layout"] for model in models],
        }
    )
    if not torch.all(payload["targets"].sum((1, 2)) == 32):
        raise RuntimeError("IMP cards do not have exactly 32 active edges")
    save_torch(destination / "cards.pt", payload)
    save_torch(
        destination / "observations.pt",
        {
            split: {"x": x.cpu(), "y": y.cpu()}
            for split, (x, y) in data.items()
        },
    )
    save_torch(
        destination / "teacher.pt",
        {"state": cpu_state(target), "reference": reference(family, "cpu")},
    )
    files = {
        name: digest(destination / name)
        for name in ["cards.pt", "observations.pt", "teacher.pt"]
    }
    summary = {
        "task": job["task"],
        "family": family["id"],
        "function": function,
        "split": family["split"],
        "count": len(models),
        "path": str(destination.relative_to(job["bank"])),
        "hashes": files,
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(DEVICE),
        "training_seconds": training_seconds,
        "group_models": job.get("group_models", len(models)),
        "group_slots": job.get("group_slots", len(models)),
        "feature_and_save_seconds": time.monotonic() - feature_started,
        "elapsed_seconds": time.monotonic() - started,
        "mean_iou": float(np.mean([row["iou"] for row in metrics])),
        "mean_variable_iou": float(
            np.mean([row["variable_iou"] for row in metrics])
        ),
        "mean_test_excess_bce": float(
            np.mean([row["test_excess_bce"] for row in metrics])
        ),
        "exact_masks": sum(row["exact_mask"] for row in metrics),
    }
    save_json(destination / "complete.json", summary)
    return summary


def collect_group(jobs):
    """Fit all assigned tasks in one CUDA batch with independent objectives."""
    pending = [(job, job["seeds"]) for job in jobs]
    prepared = {job["task"]: {} for job in jobs}
    if pending:
        width = max(len(seeds) for _, seeds in pending)
        states, observations_by_task = [], []
        active = torch.zeros(len(pending), width, device=DEVICE)
        for task_index, (job, seeds) in enumerate(pending):
            original = initialization(seeds, DEVICE, job["options"])
            padded = {}
            for key, value in original.items():
                zeros = value.new_zeros(
                    (width - len(seeds),) + value.shape[1:]
                )
                padded[key] = torch.cat((value, zeros))
            states.append(padded)
            target = teacher(
                job["config"], job["family"], job["function"], DEVICE
            )
            observations_by_task.append(
                observations(
                    job["config"],
                    job["family"],
                    job["function"],
                    target,
                    DEVICE,
                    ["support", "query"],
                )
            )
            active[task_index, : len(seeds)] = 1
        original = {
            key: torch.stack([state[key] for state in states])
            for key in states[0]
        }
        data = {
            split: tuple(
                torch.stack(
                    [item[split][index] for item in observations_by_task]
                )
                for index in range(2)
            )
            for split in ["support", "query"]
        }
        masks = torch.ones_like(original["w"])
        count = sum(len(seeds) for _, seeds in pending)
        slots = len(pending) * width
        print(
            f"{DEVICE}: grouped batch {len(pending)} tasks, "
            f"{count} real MLPs / {slots} slots",
            flush=True,
        )
        torch.cuda.synchronize(DEVICE)
        started = time.monotonic()
        first = pending[0][0]
        state, steps, history = fit(
            original, masks, data, first["options"], active_models=active
        )
        state, masks, rounds = imp(
            original,
            state,
            steps,
            history,
            data,
            first["options"],
            first["config"]["imp"],
            lambda text: None,
            active_models=active,
        )
        torch.cuda.synchronize(DEVICE)
        elapsed = time.monotonic() - started
        for task_index, (job, seeds) in enumerate(pending):
            job["amortized_training_seconds"] = elapsed * len(seeds) / count
            job["group_models"], job["group_slots"] = count, slots
            for index, seed in enumerate(seeds):
                prepared[job["task"]][seed] = {
                    "state": {
                        key: value[task_index, index]
                        for key, value in state.items()
                    },
                    "initial": {
                        key: value[task_index, index]
                        for key, value in original.items()
                    },
                    "mask": masks[task_index, index],
                    "source": "new grouped CUDA IMP",
                    "layout": "packed_grouped_tasks",
                }
            task_rounds = []
            for item in rounds:
                task_history = []
                for row in item["history"]:
                    task_history.append(
                        {
                            "step": row["step"],
                            "support_bce": row["support_bce"][task_index][
                                : len(seeds)
                            ],
                            "query_bce": row["query_bce"][task_index][
                                : len(seeds)
                            ],
                        }
                    )
                task_rounds.append(
                    {
                        "active_edges": item["active_edges"],
                        "selected_steps": item["selected_steps"][task_index][
                            : len(seeds)
                        ],
                        "query_bce": item["query_bce"][task_index][
                            : len(seeds)
                        ],
                        "history": task_history,
                    }
                )
            directory = (
                Path(job["bank"]) / job["family"]["split"] / job["task"]
            )
            save_json(directory / "pruning_history.json", task_rounds)
    results = []
    for job in jobs:
        result = collect(job, prepared=prepared[job["task"]])
        results.append(result)
        print(f"Saved {job['task']} on {DEVICE}", flush=True)
    return results


def group_jobs(jobs, devices, maximum):
    """Balance cards across GPUs, then cap padding slots per CUDA batch."""
    if maximum < max((len(job["seeds"]) for job in jobs), default=1):
        raise ValueError("models-per-gpu cannot be smaller than one task")
    buckets = [[] for _ in devices]
    loads = [0 for _ in devices]
    for job in sorted(jobs, key=lambda item: len(item["seeds"]), reverse=True):
        index = min(range(len(devices)), key=lambda item: loads[item])
        buckets[index].append(job)
        loads[index] += len(job["seeds"])
    groups = []
    for bucket in buckets:
        current, width = [], 0
        for job in bucket:
            next_width = max(width, len(job["seeds"]))
            if current and (len(current) + 1) * next_width > maximum:
                groups.append(current)
                current, width = [], 0
            current.append(job)
            width = max(width, len(job["seeds"]))
        if current:
            groups.append(current)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument(
        "--config", type=Path, default=PACKAGE / "configs/bank.json"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-task")
    parser.add_argument(
        "--devices", default=",".join(f"cuda:{i}" for i in range(8))
    )
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--validation-count", type=int, default=32)
    parser.add_argument(
        "--models-per-gpu",
        type=int,
        default=512,
        help="Maximum padded MLP slots per GPU batch; tasks are grouped together",
    )
    parser.add_argument(
        "--gpu-memory-gib",
        type=float,
        default=12,
        help="Per-GPU budget in GiB, including a 2 GiB reserve for CUDA",
    )
    args = parser.parse_args()
    if args.gpu_memory_gib <= 2:
        parser.error("gpu-memory-gib must exceed the 2 GiB CUDA reserve")
    torch.set_num_threads(1)
    devices = cuda_devices(args.devices)
    config = json.loads(args.config.read_text())
    options = config["training"]
    if args.train_count < 3 or args.validation_count < 1:
        parser.error(
            "Need at least three training cards and one validation-family card"
        )
    if config["functions_per_family"] != 4:
        parser.error(
            "Known-mask transfer reserves functions 4+; collect functions 0..3 only"
        )
    if (
        options["layout"] != "packed"
        or options["steps"] < 1
        or options["select_every"] < 1
        or options["lr"] <= 0
    ):
        parser.error("Invalid packed training configuration")
    if (
        not 0 <= options["warmup_steps"] < options["steps"]
        or options["l2"] < 0
    ):
        parser.error("Invalid warmup or regularization")
    if (
        config["imp"]["keep"] != 32
        or not 0 < config["imp"]["prune_fraction"] < 1
    ):
        parser.error(
            "IMP must retain 32 edges with pruning fraction in (0, 1)"
        )
    definitions, structure = families(config)
    bank = args.bank.resolve()
    manifest_path = bank / "manifest.json"
    counts = {"train": args.train_count, "validation": args.validation_count}
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["schema"] != BANK_SCHEMA
            or manifest["implementation_sha256"] != source_hashes()
            or manifest["config"] != config
            or manifest["counts_per_task"] != counts
            or manifest["options"] != options
        ):
            raise ValueError("Cannot resume a bank with a different protocol")
    else:
        bank.mkdir(parents=True, exist_ok=False)
        snapshot(bank / "source", args.config)
        manifest = {
            "schema": BANK_SCHEMA,
            "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "options": options,
            "counts_per_task": counts,
            "family_structure": structure,
            "completed_tasks": {},
            "implementation_sha256": source_hashes(),
            "feature_shape": [48, 8, 2],
            "channels": ["weights", "loss_gradient"],
            "gradient": "mean support BCE excluding L2, dL/dW in f(W*mask); pruned coordinates zero",
            "canonicalization": "hard mask column profiles; no reference alignment",
            "normalization": "raw W/G values plus per-channel max-abs scales",
            "selection": "query BCE only; no filtering by IoU, reference mask or test metrics",
            "training": "CUDA packed wide first layer; each eight-column block has independent readout, biases and objective",
            "test_families": "no MLP bank; preserved for later transfer",
            "family_definitions": definitions,
        }
    jobs = []
    for family in definitions:
        if family["split"] not in counts:
            continue
        for function in range(config["functions_per_family"]):
            task = f"family_{family['id']:02d}_function_{function}"
            count = counts[family["split"]]
            completion = bank / family["split"] / task / "complete.json"
            if completion.exists():
                saved = json.loads(completion.read_text())
                if saved["count"] != count:
                    raise ValueError(
                        "Completed task has a different number of models"
                    )
                manifest["completed_tasks"][task] = saved
                continue
            if args.only_task and task != args.only_task:
                continue
            roles = ["validation_family"] * count
            if family["split"] == "train":
                generator = torch.Generator().manual_seed(
                    config["seed"]
                    + family["id"] * 100003
                    + function * 1009
                    + 71
                )
                order = torch.randperm(count, generator=generator).tolist()
                holdout = max(1, count // 8)
                roles = ["train"] * count
                for rank, index in enumerate(order[: 2 * holdout]):
                    roles[index] = "test" if rank < holdout else "validation"
            jobs.append(
                {
                    "bank": str(bank),
                    "task": task,
                    "family": family,
                    "function": function,
                    "config": config,
                    "options": options,
                    "seeds": [
                        seed_for(config, family, function, index)
                        for index in range(count)
                    ],
                    "roles": roles,
                }
            )
    groups = group_jobs(jobs, devices, args.models_per_gpu)
    revision = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    source_directory = bank / "source_revisions" / revision
    source_directory.mkdir(parents=True, exist_ok=False)
    snapshot(source_directory, args.config)
    manifest.setdefault("executions", []).append(
        {
            "source": str(source_directory.relative_to(bank)),
            "grouped_tasks": True,
            "models_per_gpu_limit": args.models_per_gpu,
            "gpu_memory_budget_gib": args.gpu_memory_gib,
            "allocator_limit_gib": args.gpu_memory_gib - 2,
            "groups": [
                {
                    "tasks": len(group),
                    "models": sum(len(job["seeds"]) for job in group),
                    "slots": len(group)
                    * max(len(job["seeds"]) for job in group),
                }
                for group in groups
            ],
        }
    )
    manifest["status"] = "running"
    manifest["batching"] = (
        "Multiple tasks per GPU; wide first layer within each task; no coupling between models"
    )
    manifest["models_per_gpu_limit"] = args.models_per_gpu
    manifest["gpu_memory_budget_gib"] = args.gpu_memory_gib
    manifest["allocator_limit_gib"] = args.gpu_memory_gib - 2
    manifest["devices"] = devices
    manifest["expected_tasks"] = sum(
        config["families"][split] * config["functions_per_family"]
        for split in counts
    )
    manifest["expected_cards"] = sum(
        config["families"][split] * config["functions_per_family"] * count
        for split, count in counts.items()
    )
    save_json(manifest_path, manifest)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for device in devices:
        queue.put(device)
    started = time.monotonic()
    try:
        with ProcessPoolExecutor(
            max_workers=min(len(devices), max(1, len(groups))),
            mp_context=context,
            initializer=initialize_worker,
            initargs=(queue, args.gpu_memory_gib),
        ) as pool:
            futures = {
                pool.submit(collect_group, group): group for group in groups
            }
            for future in as_completed(futures):
                for result in future.result():
                    manifest["completed_tasks"][result["task"]] = result
                    manifest["completed_cards"] = sum(
                        row["count"]
                        for row in manifest["completed_tasks"].values()
                    )
                    save_json(manifest_path, manifest)
                    print(
                        f"{len(manifest['completed_tasks'])}/{manifest['expected_tasks']} tasks; "
                        f"{manifest['completed_cards']}/{manifest['expected_cards']} cards; "
                        f"{result['task']} {result['device']} {result['elapsed_seconds']:.1f}s; "
                        f"IoU={result['mean_iou']:.4f}",
                        flush=True,
                    )
    except BaseException as error:
        manifest["status"] = "interrupted"
        manifest["error"] = repr(error)
        save_json(manifest_path, manifest)
        raise
    manifest["completed_cards"] = sum(
        row["count"] for row in manifest["completed_tasks"].values()
    )
    manifest["status"] = (
        "complete"
        if len(manifest["completed_tasks"]) == manifest["expected_tasks"]
        else "partial"
    )
    manifest["last_invocation_seconds"] = time.monotonic() - started
    save_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "cards": manifest["completed_cards"],
                "elapsed_seconds": manifest["last_invocation_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
