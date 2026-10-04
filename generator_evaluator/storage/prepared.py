"""Reusable functional banks and terminal labels, without child checkpoints."""
from __future__ import annotations

from dataclasses import fields
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from generator_evaluator.data.types import InnerProtocol, RealReplay, TaskData
from generator_evaluator.storage.artifacts import _atomic_write, save_json
from generator_evaluator.storage.binary_banks import load_banks, save_banks


def _encode(value, root: Path, counter: list[int]):
    if torch.is_tensor(value):
        name = f"array_{counter[0]:04d}.npy"
        counter[0] += 1
        array = value.detach().cpu().contiguous().numpy()
        _atomic_write(root / name, lambda stream: np.save(stream, array, allow_pickle=False), binary=True)
        return {"__array__": name, "shape": list(array.shape), "dtype": str(array.dtype)}
    if isinstance(value, dict):
        return {str(key): _encode(item, root, counter) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item, root, counter) for item in value]
    return value


def _decode(value, root: Path):
    if isinstance(value, dict) and "__array__" in value:
        name = value["__array__"]
        if Path(name).name != name:
            raise ValueError("invalid prepared array filename")
        array = np.load(root / name, allow_pickle=False)
        if list(array.shape) != value["shape"] or str(array.dtype) != value["dtype"]:
            raise ValueError("prepared array shape or dtype mismatch")
        return torch.from_numpy(array)
    if isinstance(value, dict):
        return {key: _decode(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, root) for item in value]
    return value


def _link_or_copy(source, destination):
    try:
        return os.link(source, destination)
    except OSError:
        return shutil.copyfile(source, destination)


def save_prepared(out, banks, train_tasks, selection_tasks, test_spec, *,
                  config, protocol, build_settings, bank_source=None):
    root = Path(out) / "prepared"
    if (root / "manifest.json").is_file():
        return
    root.mkdir(parents=True, exist_ok=True)
    if bank_source is None:
        save_banks(root / "banks", banks)
    else:
        shutil.copytree(Path(bank_source) / "prepared" / "banks", root / "banks",
                        copy_function=_link_or_copy)
    tasks = [[{field.name: getattr(task, field.name) for field in fields(TaskData)}
              for task in group] for group in (train_tasks, selection_tasks)]
    payload = dict(schema=1, tasks=tasks, test_spec=test_spec, config=config,
                   requested_protocol=protocol, build_settings=build_settings)
    save_json(root / "manifest.json", _encode(payload, root, [0]))


def load_prepared(out):
    root = Path(out) / "prepared"
    payload = _decode(json.loads((root / "manifest.json").read_text()), root)
    if payload.get("schema") != 1:
        raise ValueError("unsupported prepared bank schema")
    payload["banks"] = load_banks(root / "banks")
    payload["train_tasks"], payload["selection_tasks"] = (
        [TaskData(**task) for task in group] for group in payload.pop("tasks"))
    return payload


def save_labels(out, replay: RealReplay):
    root = Path(out) / "prepared" / "labels"
    if (root / "manifest.json").is_file():
        return
    # Store each binary topology once; rows contain only terminal labels.
    keys = list(replay.masks)
    masks = torch.stack([replay.masks[key] for key in keys]).to(torch.uint8)
    records = [{key: value for key, value in row.items() if key != "provenance"}
               for row in replay.records]
    payload = dict(schema=1, protocol=replay.protocol.__dict__,
                   holdout_fraction=replay.holdout_fraction, split_seed=replay.split_seed,
                   records=records, mask_keys=keys, masks=masks, contexts=replay.contexts,
                   task_fingerprints=replay.task_fingerprints, task_splits=replay.task_splits,
                   mask_splits=replay.mask_splits)
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "manifest.json", _encode(payload, root, [0]))


def load_labels(out, tasks):
    root = Path(out) / "prepared" / "labels"
    if not (root / "manifest.json").is_file():
        return None
    payload = _decode(json.loads((root / "manifest.json").read_text()), root)
    if payload.get("schema") != 1:
        raise ValueError("unsupported prepared label schema")
    replay = RealReplay(InnerProtocol(**payload["protocol"]),
                        holdout_fraction=payload["holdout_fraction"], split_seed=payload["split_seed"])
    by_id = {task.task_id: task for task in tasks}
    fingerprints = {task_id: task.fingerprint for task_id, task in by_id.items()}
    replay.masks = {key: mask.float() for key, mask in zip(payload["mask_keys"], payload["masks"])}
    for name in ("contexts", "task_fingerprints", "task_splits", "mask_splits"):
        setattr(replay, name, payload[name])
    for row in payload["records"]:
        task = by_id[row["task_id"]]
        if row["task_fingerprint"] != fingerprints[task.task_id]:
            raise ValueError("prepared labels refer to different task data")
        row["provenance"] = task.provenance
        row["artifact_path"] = None
        row["artifact_ephemeral"] = True
        replay.records.append(row)
    replay.validate()
    return replay
