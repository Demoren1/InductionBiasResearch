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


def save_online_labels(out, replay: RealReplay, refresh: int):
    """Save the cumulative acquired train labels for one refresh."""
    if not isinstance(refresh, int) or isinstance(refresh, bool) or refresh < 0:
        raise ValueError("refresh must be a nonnegative integer")
    root = Path(out) / "prepared" / "online_labels" / f"refresh_{refresh:04d}"
    manifest = root / "manifest.json"
    if manifest.is_file():
        return
    records = [row for row in replay.records
               if str(row.get("origin", "")).startswith("acquisition:") and
               row.get("task_split") == "train" and
               row.get("split") not in ("test", "control", "selection")]
    if not records:
        return

    mask_keys = list(dict.fromkeys(row["mask_key"] for row in records))
    task_ids = list(dict.fromkeys(row["task_id"] for row in records))
    topology_ids = list(dict.fromkeys(row["topology_id"] for row in records))
    payload = dict(
        schema=1,
        protocol=replay.protocol.__dict__,
        holdout_fraction=replay.holdout_fraction,
        split_seed=replay.split_seed,
        records=[{key: value for key, value in row.items() if key != "provenance"}
                 for row in records],
        mask_keys=mask_keys,
        masks=torch.stack([replay.masks[key] for key in mask_keys]).to(torch.uint8),
        contexts={task_id: replay.contexts[task_id] for task_id in task_ids},
        task_fingerprints={task_id: replay.task_fingerprints[task_id] for task_id in task_ids},
        task_splits={task_id: replay.task_splits[task_id] for task_id in task_ids},
        mask_splits={identity: replay.mask_splits[identity] for identity in topology_ids},
    )
    root.mkdir(parents=True, exist_ok=True)
    save_json(manifest, _encode(payload, root, [0]))


def _latest_online_label_manifest(root: Path) -> Path | None:
    snapshots = []
    for manifest in root.glob("refresh_*/manifest.json"):
        suffix = manifest.parent.name.removeprefix("refresh_")
        if suffix.isdecimal():
            snapshots.append((int(suffix), manifest))
    return max(snapshots, key=lambda item: item[0])[1] if snapshots else None


def _merge_label_metadata(target: dict, source: dict, name: str):
    for key, value in source.items():
        if key in target:
            current = target[key]
            equal = torch.equal(current, value) if torch.is_tensor(current) else current == value
            if not equal:
                raise ValueError(f"prepared online label {name} conflicts with baseline")
        else:
            target[key] = value


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

    online_manifest = _latest_online_label_manifest(root.parent / "online_labels")
    if online_manifest is not None:
        online_root = online_manifest.parent
        online = _decode(json.loads(online_manifest.read_text()), online_root)
        if online.get("schema") != 1:
            raise ValueError("unsupported prepared online label schema")
        online_protocol = InnerProtocol(**online["protocol"])
        if (online_protocol != replay.protocol or
                online["holdout_fraction"] != replay.holdout_fraction or
                online["split_seed"] != replay.split_seed):
            raise ValueError("prepared online labels use a different protocol or split")

        online_replay = RealReplay(online_protocol,
                                   holdout_fraction=online["holdout_fraction"],
                                   split_seed=online["split_seed"])
        online_replay.masks = {
            key: mask.float() for key, mask in zip(online["mask_keys"], online["masks"])
        }
        for name in ("contexts", "task_fingerprints", "task_splits", "mask_splits"):
            setattr(online_replay, name, online[name])
        for row in online["records"]:
            if (not str(row.get("origin", "")).startswith("acquisition:") or
                    row.get("task_split") != "train" or
                    row.get("split") in ("test", "control", "selection")):
                raise ValueError("prepared online labels contain an ineligible measurement")
            task = by_id.get(row["task_id"])
            if task is None or row["task_fingerprint"] != fingerprints.get(row["task_id"]):
                raise ValueError("prepared online labels refer to different task data")
            if task.split != row["task_split"]:
                raise ValueError("prepared online labels refer to a different task split")
            row["provenance"] = task.provenance
            row["artifact_path"] = None
            row["artifact_ephemeral"] = True
            online_replay.records.append(row)
        online_replay.validate()

        _merge_label_metadata(replay.masks, online_replay.masks, "masks")
        for name in ("contexts", "task_fingerprints", "task_splits", "mask_splits"):
            _merge_label_metadata(getattr(replay, name), getattr(online_replay, name), name)
        known_measurements = {
            row["measurement_key"] for row in replay.records
            if row.get("measurement_key") is not None
        }
        for row in online_replay.records:
            key = row.get("measurement_key")
            if key is not None and key in known_measurements:
                continue
            replay.records.append(row)
            if key is not None:
                known_measurements.add(key)
    replay.validate()
    return replay
