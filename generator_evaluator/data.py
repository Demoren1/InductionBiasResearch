"""Auditable real-label replay and task/hidden-permutation-safe partitions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def tensor_hash(value: Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode()
                          + value.numpy().tobytes()).hexdigest()


def topology_id(mask: Tensor) -> str:
    """Column permutations share an identity; labelled input rows stay fixed."""
    mask = torch.as_tensor(mask).detach().cpu()
    if mask.ndim != 2 or min(mask.shape) < 1:
        raise ValueError("mask must be a nonempty [features,hidden] matrix")
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("real measurements require binary finite masks")
    columns = sorted(bytes(column.tolist()) for column in mask.T.to(torch.uint8))
    return hashlib.sha256(str(tuple(mask.shape)).encode() + b"".join(columns)).hexdigest()


@dataclass(frozen=True)
class InnerProtocol:
    steps: int = 2000
    replicas: int = 2
    lr: float = 0.01
    l2: float = 0.0
    batch_size: int | None = None
    checkpoint_every: int = 25
    plateau_tolerance: float = 0.01
    seed: int = 4100
    solver: str = "adam"
    metric: str = "bce"
    lr_decay_every: int = 200
    lr_floor: float = 1 / 64

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (self.lr, self.l2, self.plateau_tolerance, self.lr_floor)):
            raise ValueError("inner solver parameters must be finite")
        if min(self.steps, self.replicas, self.checkpoint_every, self.lr_decay_every) < 1:
            raise ValueError("inner step, replica and checkpoint budgets must be positive")
        if self.lr <= 0 or self.l2 < 0 or self.plateau_tolerance <= 0:
            raise ValueError("invalid inner solver parameters")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.solver != "adam" or self.metric not in ("bce", "nmse"):
            raise ValueError("supported protocols use adam and bce/nmse")
        if not 0 < self.lr_floor <= 1:
            raise ValueError("lr_floor must be in (0,1]")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass
class TaskData:
    task_id: str
    split: str
    x_support: Tensor
    y_support: Tensor
    x_query: Tensor
    y_query: Tensor
    context: Tensor
    support_ids: Tensor
    query_ids: Tensor
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.split not in ("train", "validation", "test"):
            raise ValueError("task split must be train, validation or test")
        if not self.task_id:
            raise ValueError("task_id cannot be empty")
        for x, y in ((self.x_support, self.y_support), (self.x_query, self.y_query)):
            if x.ndim not in (2, 3) or len(x) < 1 or y.shape != (len(x),):
                raise ValueError("task tensors must have matching nonempty data/label rows")
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError("task data must be finite")
        if self.x_support.shape[1:] != self.x_query.shape[1:]:
            raise ValueError("support/query input dimensions differ")
        if self.context.ndim != 1 or not torch.isfinite(self.context).all():
            raise ValueError("task context must be a finite vector computed from support")
        if self.support_ids.numel() == 0 or self.query_ids.numel() == 0:
            raise ValueError("data provenance needs observation IDs")
        if self.support_ids.ndim < 1 or self.query_ids.ndim < 1:
            raise ValueError("observation IDs need a leading data-row axis")
        if len(self.support_ids) != len(self.x_support) or len(self.query_ids) != len(self.x_query):
            raise ValueError("observation IDs must identify every measured data row")
        if self.x_support.ndim == 3 and (self.support_ids.shape != self.x_support.shape[:2] or
                                         self.query_ids.shape != self.x_query.shape[:2]):
            raise ValueError("set data need observation IDs for each member")
        if torch.isin(self.support_ids.cpu(), self.query_ids.cpu()).any():
            raise ValueError("support/query observation IDs must be disjoint")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256("".join(tensor_hash(t) for t in (
            self.x_support, self.y_support, self.x_query, self.y_query,
            self.context, self.support_ids, self.query_ids)).encode()).hexdigest()


def support_context(x: Tensor, y: Tensor) -> Tensor:
    """Label covariance and moments; never sees query or latent task labels."""
    x = x.float().mean(1) if x.ndim == 3 else x.float()
    y = y.float()
    cov = ((x - x.mean(0)) * (y - y.mean())[:, None]).mean(0)
    cov = cov / cov.square().mean().sqrt().clamp_min(1e-4)
    return torch.cat((x.mean(0), cov, y.mean()[None], y.std(unbiased=False)[None]))


class RealReplay:
    """Only terminal fresh-fit measurements are accepted as regression targets.

    Task and topology holdouts are crossed. Dense is an optimization anchor,
    excluded from regression holdouts/training to avoid a single shared
    topology crossing partitions. Full child artifacts live beside this file.
    """

    def __init__(self, protocol: InnerProtocol, *, holdout_fraction: float = 0.2,
                 split_seed: int = 42):
        if not 0 < holdout_fraction < 1:
            raise ValueError("holdout_fraction must be in (0,1)")
        self.protocol = protocol
        self.holdout_fraction = holdout_fraction
        self.split_seed = split_seed
        self.records: list[dict[str, Any]] = []
        self.masks: dict[str, Tensor] = {}
        self.contexts: dict[str, Tensor] = {}
        self.task_fingerprints: dict[str, str] = {}
        self.task_splits: dict[str, str] = {}
        self.mask_splits: dict[str, str] = {}

    def mask_split(self, mask: Tensor) -> str:
        identity = topology_id(mask)
        if bool((mask == 1).all()):
            return "control"
        digest = hashlib.sha256(f"{self.split_seed}:{identity}".encode()).digest()
        return "holdout" if int.from_bytes(digest[:8], "big") / 2**64 < self.holdout_fraction else "train"

    def append(self, mask: Tensor, task: TaskData, result: dict[str, Any], *,
               origin: str, artifact_path: str | Path) -> dict[str, Any]:
        identity = topology_id(mask)
        if result.get("label_source") != "fresh_terminal_query" or not result.get("fixed_horizon"):
            raise ValueError("replay accepts only fresh fixed-horizon terminal query measurements")
        if result.get("protocol_id") != self.protocol.fingerprint:
            raise ValueError("measurement protocol differs from replay protocol")
        losses = torch.as_tensor(result["replica_losses"]).detach().cpu().float()
        seeds = list(result["seeds"])
        if losses.shape != (self.protocol.replicas,) or len(seeds) != self.protocol.replicas:
            raise ValueError("measurement must contain the prescribed replica count")
        if not torch.isfinite(losses).all() or len(set(seeds)) != len(seeds):
            raise ValueError("replica losses must be finite and initializations independent")
        artifact_path = Path(artifact_path).resolve()
        if not artifact_path.is_file():
            raise ValueError("full child measurement artifact must exist before adding its label")
        fingerprint = task.fingerprint
        if task.task_id in self.task_fingerprints and self.task_fingerprints[task.task_id] != fingerprint:
            raise ValueError("task data changed within the replay")
        if task.task_id in self.task_splits and self.task_splits[task.task_id] != task.split:
            raise ValueError("task cannot cross partitions")
        mask_split = self.mask_split(mask)
        split = ("control" if mask_split == "control" else
                 "joint_validation" if task.split == "validation" and mask_split == "holdout" else
                 "meta_validation" if task.split == "validation" else
                 "test" if task.split == "test" else
                 "mask_validation" if mask_split == "holdout" else "train")
        # Store actual coordinate order too, although the split identity is canonical.
        mask_key = tensor_hash(mask.float())
        self.masks[mask_key] = mask.detach().cpu().float().clone()
        self.contexts[task.task_id] = task.context.detach().cpu().float().clone()
        self.task_fingerprints[task.task_id] = fingerprint
        self.task_splits[task.task_id] = task.split
        self.mask_splits[identity] = mask_split
        row = dict(topology_id=identity, mask_key=mask_key, task_id=task.task_id,
                   task_split=task.split, split=split, protocol_id=self.protocol.fingerprint,
                   task_fingerprint=fingerprint, origin=origin, label_source=result["label_source"],
                   quality=float(losses.mean()), replica_losses=losses.tolist(), seeds=seeds,
                   density=float(mask.float().mean()), active_edges=int(mask.sum()),
                   plateau_flags=torch.as_tensor(result["plateau_flags"]).bool().tolist(),
                   artifact_path=str(artifact_path), support_ids_hash=tensor_hash(task.support_ids),
                   query_ids_hash=tensor_hash(task.query_ids), provenance=task.provenance)
        self.records.append(row)
        return row

    def tensors(self, split: str = "train") -> tuple[Tensor, Tensor, Tensor]:
        rows = [row for row in self.records if row["split"] == split]
        if not rows:
            raise ValueError(f"no real measurements in {split} partition")
        return (torch.stack([self.masks[row["mask_key"]] for row in rows]),
                torch.stack([self.contexts[row["task_id"]] for row in rows]),
                torch.tensor([row["quality"] for row in rows], dtype=torch.float32))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(protocol=asdict(self.protocol), holdout_fraction=self.holdout_fraction,
                       split_seed=self.split_seed, records=self.records, masks=self.masks,
                       contexts=self.contexts, task_fingerprints=self.task_fingerprints,
                       task_splits=self.task_splits, mask_splits=self.mask_splits)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "RealReplay":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        replay = cls(InnerProtocol(**payload["protocol"]),
                     holdout_fraction=payload["holdout_fraction"], split_seed=payload["split_seed"])
        for name in ("records", "masks", "contexts", "task_fingerprints", "task_splits", "mask_splits"):
            setattr(replay, name, payload[name])
        replay.validate()
        return replay

    def validate(self) -> None:
        """Recompute identities/partitions and bind labels to saved real fits."""
        for row in self.records:
            if row["protocol_id"] != self.protocol.fingerprint or row["label_source"] != "fresh_terminal_query":
                raise ValueError("invalid replay label provenance")
            mask = self.masks[row["mask_key"]]
            identity = topology_id(mask)
            if row["mask_key"] != tensor_hash(mask) or row["topology_id"] != identity:
                raise ValueError("replay mask identity is corrupt")
            if self.mask_splits[identity] != self.mask_split(mask):
                raise ValueError("replay topology partition is corrupt")
            task_id = row["task_id"]
            if self.task_splits[task_id] != row["task_split"] or self.task_fingerprints[task_id] != row["task_fingerprint"]:
                raise ValueError("replay task identity/partition is corrupt")
            expected_split = ("control" if self.mask_split(mask) == "control" else
                              "joint_validation" if row["task_split"] == "validation" and self.mask_split(mask) == "holdout" else
                              "meta_validation" if row["task_split"] == "validation" else
                              "test" if row["task_split"] == "test" else
                              "mask_validation" if self.mask_split(mask) == "holdout" else "train")
            if expected_split != row["split"]:
                raise ValueError("replay row crossed partitions")
            context = self.contexts[task_id]
            if context.ndim != 1 or not torch.isfinite(context).all():
                raise ValueError("replay context is corrupt")
            path = Path(row["artifact_path"])
            if not path.is_file():
                raise ValueError(f"missing real measurement artifact: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            result = payload.get("result", payload)
            if (result.get("protocol_id") != self.protocol.fingerprint or
                result.get("label_source") != "fresh_terminal_query" or
                not result.get("fixed_horizon")):
                raise ValueError("child artifact is not a measurement of this protocol")
            losses = torch.as_tensor(result["replica_losses"]).float()
            if not torch.equal(losses, torch.tensor(row["replica_losses"]).float()):
                raise ValueError("replay label differs from its child measurement")
            if row["quality"] != float(losses.mean()) or row["seeds"] != list(result["seeds"]):
                raise ValueError("replay quality/initialization provenance is corrupt")
            if payload.get("mask_key", row["mask_key"]) != row["mask_key"]:
                raise ValueError("child artifact mask does not match replay")
            if payload.get("task_fingerprint", row["task_fingerprint"]) != row["task_fingerprint"]:
                raise ValueError("child artifact task does not match replay")
