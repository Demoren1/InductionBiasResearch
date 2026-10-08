#!/usr/bin/env python3
"""Collect DeepSets functional maps without generators, evaluators or VAEs.

Requires torch, numpy, scipy and tqdm. Supply MNIST8m images.npy [N,784]
(uint8 pixels) and labels.npy [N] (digit IDs). Uses the released eight-block
layout, source block 0 for fitting/probes and block 1 for ranking. All source
pools exclude repeated pixel arrays; tasks reserve disjoint support/query/probe
image IDs. Labels are sums of digit costs in sets of five images. Teachers are
ranked by terminal source-query NMSE within mask-density strata.

Outputs: compact maps/*.pt cards, bank.pt and manifest.json for each task.
No external dataset is bundled with this script.

Example:
    python data/collect_deepsets_maps.py --data-root /path/to/mnist8m \
        --out /tmp/deepsets_maps --devices cuda:0 cuda:1
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import hashlib
from io import BytesIO
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm

def tensor_hash(value: Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode() + value.numpy().tobytes()).hexdigest()

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
    solver: str = 'adam'
    metric: str = 'bce'
    lr_decay_every: int = 200
    lr_floor: float = 1 / 64

    def __post_init__(self):
        if not all((math.isfinite(value) for value in (self.lr, self.l2, self.plateau_tolerance, self.lr_floor))):
            raise ValueError('inner solver parameters must be finite')
        if min(self.steps, self.replicas, self.checkpoint_every, self.lr_decay_every) < 1:
            raise ValueError('inner step, replica and checkpoint budgets must be positive')
        if self.lr <= 0 or self.l2 < 0 or self.plateau_tolerance <= 0:
            raise ValueError('invalid inner solver parameters')
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError('batch_size must be positive')
        if self.solver != 'adam' or self.metric not in ('bce', 'nmse'):
            raise ValueError('supported protocols use adam and bce/nmse')
        if not 0 < self.lr_floor <= 1:
            raise ValueError('lr_floor must be in (0,1]')

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
        if self.split not in ('train', 'validation', 'test'):
            raise ValueError('task split must be train, validation or test')
        if not self.task_id:
            raise ValueError('task_id cannot be empty')
        for x, y in ((self.x_support, self.y_support), (self.x_query, self.y_query)):
            if x.ndim not in (2, 3) or len(x) < 1 or y.shape != (len(x),):
                raise ValueError('task tensors must have matching nonempty data/label rows')
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError('task data must be finite')
        if self.x_support.shape[1:] != self.x_query.shape[1:]:
            raise ValueError('support/query input dimensions differ')
        if self.context.ndim != 1 or not torch.isfinite(self.context).all():
            raise ValueError('task context must be a finite vector computed from support')
        if self.support_ids.numel() == 0 or self.query_ids.numel() == 0:
            raise ValueError('data provenance needs observation IDs')
        if self.support_ids.ndim < 1 or self.query_ids.ndim < 1:
            raise ValueError('observation IDs need a leading data-row axis')
        if len(self.support_ids) != len(self.x_support) or len(self.query_ids) != len(self.x_query):
            raise ValueError('observation IDs must identify every measured data row')
        if self.x_support.ndim == 3 and (self.support_ids.shape != self.x_support.shape[:2] or self.query_ids.shape != self.x_query.shape[:2]):
            raise ValueError('set data need observation IDs for each member')
        if torch.isin(self.support_ids.cpu(), self.query_ids.cpu()).any():
            raise ValueError('support/query observation IDs must be disjoint')

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(''.join((tensor_hash(t) for t in (self.x_support, self.y_support, self.x_query, self.y_query, self.context, self.support_ids, self.query_ids))).encode()).hexdigest()

def support_context(x: Tensor, y: Tensor) -> Tensor:
    """Label covariance and moments; never sees query or latent task labels."""
    x = x.float().mean(1) if x.ndim == 3 else x.float()
    y = y.float()
    cov = ((x - x.mean(0)) * (y - y.mean())[:, None]).mean(0)
    cov = cov / cov.square().mean().sqrt().clamp_min(0.0001)
    return torch.cat((x.mean(0), cov, y.mean()[None], y.std(unbiased=False)[None]))

@dataclass
class FunctionalBank:
    """Raw functional teacher profiles and fixed-cardinality candidate masks."""
    tokens: Tensor
    quality: Tensor | None
    masks: Tensor
    baseline_mask: Tensor
    provenance: dict[str, Any] = field(default_factory=dict)
    states: list[Any] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tokens = torch.as_tensor(self.tokens, dtype=torch.float32).cpu().contiguous()
        self.masks = torch.as_tensor(self.masks, dtype=torch.float32).cpu().contiguous()
        self.baseline_mask = torch.as_tensor(self.baseline_mask, dtype=torch.float32).cpu().contiguous()
        if self.tokens.ndim != 4 or self.tokens.shape[0] != 1 or min(self.tokens.shape[1:]) < 1:
            raise ValueError('tokens must have shape [1, teachers, hidden, token_dim]')
        if self.masks.ndim != 3 or self.masks.shape[0] != self.tokens.shape[1]:
            raise ValueError('masks must have one [features, hidden] row per teacher')
        if self.masks.shape[2] != self.tokens.shape[2] or self.baseline_mask.shape != self.masks.shape[1:]:
            raise ValueError('bank masks must agree with teacher and baseline dimensions')
        if not torch.isfinite(self.tokens).all() or not torch.isfinite(self.masks).all():
            raise ValueError('functional-bank tensors must be finite')
        if not ((self.masks == 0) | (self.masks == 1)).all() or not ((self.baseline_mask == 0) | (self.baseline_mask == 1)).all():
            raise ValueError('bank masks must be binary')
        if self.quality is not None:
            self.quality = torch.as_tensor(self.quality, dtype=torch.float32).cpu().contiguous()
            if self.quality.shape != (1, self.tokens.shape[1], 1) or not torch.isfinite(self.quality).all():
                raise ValueError('quality must have shape [1, teachers, 1]')

def _exact_topk(scores: Tensor, k: int) -> Tensor:
    scores = torch.as_tensor(scores, dtype=torch.float32).cpu()
    if scores.ndim != 2 or not 1 <= k <= scores.numel():
        raise ValueError('K must be within the number of mask edges')
    values = scores.flatten() + torch.arange(scores.numel(), dtype=torch.float32) * 1e-12
    result = torch.zeros_like(values)
    result[values.topk(k).indices] = 1.0
    return result.reshape_as(scores)

def _atomic_write(path: Path, write_payload, *, binary: bool) -> None:
    """Write through an owned, unique sibling file and replace the destination."""
    temporary = None
    descriptor = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
        temporary = Path(temporary_name)
        if binary:
            stream = os.fdopen(descriptor, 'wb')
        else:
            stream = os.fdopen(descriptor, 'w', encoding='utf-8')
        descriptor = None
        with stream:
            write_payload(stream)
            stream.flush()
        os.replace(temporary, path)
    except OSError as exc:
        temporary_context = f' (temporary file {temporary})' if temporary else ''
        reason = exc.strerror or str(exc)
        raise OSError(exc.errno, f'Failed to atomically write artifact {path}{temporary_context}: {reason}', exc.filename) from exc
    except RuntimeError as exc:
        raise RuntimeError(f'Failed to atomically write artifact {path}: {exc}') from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

def save_json(path: Path, payload) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    _atomic_write(path, lambda stream: stream.write(serialized), binary=False)

def save_torch(path: Path, payload) -> None:
    _atomic_write(Path(path), lambda stream: torch.save(payload, stream), binary=True)


def progress(iterable=None, *, desc, total=None, position=0, leave=True, unit="it"):
    disabled = os.environ.get("FUNCTIONAL_MAP_PROGRESS", "auto") == "0" or not sys.stderr.isatty()
    return tqdm(iterable, total=total, desc=desc, leave=leave, unit=unit,
                dynamic_ncols=True, mininterval=1.0, disable=disabled)


_SCHEMA = 'generator_evaluator.functional_map_card'

_SCHEMA_VERSION = 1

_FORBIDDEN_METADATA_KEY_PARTS = ('optimizer', 'adam', 'history')

_MAX_METADATA_TENSOR_ELEMENTS = 65536

def _compact_metadata(value: Any, path: str='metadata') -> Any:
    """Clone small metadata values and refuse training-state payloads."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            name = str(key)
            normalized = name.lower()
            if any((part in normalized for part in _FORBIDDEN_METADATA_KEY_PARTS)):
                raise ValueError(f'functional cards cannot contain optimizer or history field {path}.{name}')
            result[key] = _compact_metadata(item, f'{path}.{name}')
        return result
    if isinstance(value, list):
        return [_compact_metadata(item, f'{path}[]') for item in value]
    if isinstance(value, tuple):
        return tuple((_compact_metadata(item, f'{path}[]') for item in value))
    if isinstance(value, Tensor):
        if value.numel() > _MAX_METADATA_TENSOR_ELEMENTS:
            raise ValueError(f'functional card {path} tensor is too large for metadata')
        compact = value.detach().cpu().clone(memory_format=torch.contiguous_format)
        if compact.is_floating_point() and (not bool(torch.isfinite(compact).all())):
            raise ValueError(f'functional card {path} tensor must be finite')
        return compact
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f'functional card {path} number must be finite')
        return value
    raise TypeError(f'functional card {path} has unsupported metadata value {type(value).__name__}')

def _compact_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError('functional card state must be a nonempty tensor mapping')
    result: dict[str, Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError('functional card state must map string names to tensors')
        compact = value.detach().to(device='cpu', dtype=torch.float32).clone(memory_format=torch.contiguous_format)
        if not bool(torch.isfinite(compact).all()):
            raise ValueError(f'functional card state {name!r} must be finite')
        result[name] = compact
    return result

def write_functional_card(path: str | Path, *, token: Tensor, mask: Tensor, state: Mapping[str, Tensor], metadata: Mapping[str, Any], diagnostics: Mapping[str, Any] | None=None) -> Path:
    """Atomically write one compact functional map and its terminal state.

    State names are preserved so pattern (``w/b/v/c``) and DeepSets models
    (``weight/bias/readout/per_image_offset``) use the same card schema.
    Optimizer state and training histories are deliberately excluded.
    """
    if not isinstance(token, Tensor) or token.ndim != 2 or (not token.is_floating_point()):
        raise ValueError('functional card token must be a floating [hidden, channels] tensor')
    if not bool(torch.isfinite(token).all()):
        raise ValueError('functional card token must be finite')
    if not isinstance(mask, Tensor) or mask.ndim != 2:
        raise ValueError('functional card mask must be a [features, hidden] tensor')
    if mask.is_floating_point() and (not bool(torch.isfinite(mask).all())):
        raise ValueError('functional card mask must be finite')
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError('functional card mask must be binary')
    if token.shape[0] != mask.shape[1]:
        raise ValueError('functional card token and mask hidden dimensions differ')
    if not isinstance(metadata, Mapping):
        raise TypeError('functional card metadata must be a mapping')
    if diagnostics is not None and (not isinstance(diagnostics, Mapping)):
        raise TypeError('functional card diagnostics must be a mapping')
    payload = {'schema': _SCHEMA, 'schema_version': _SCHEMA_VERSION, 'mask': mask.detach().to(device='cpu', dtype=torch.bool).clone(memory_format=torch.contiguous_format), 'token': token.detach().to(device='cpu', dtype=torch.float32).clone(memory_format=torch.contiguous_format), 'state_dict': _compact_state(state), 'metadata': _compact_metadata(metadata)}
    if diagnostics is not None:
        payload['diagnostics'] = _compact_metadata(diagnostics, 'diagnostics')
    destination = Path(path)
    save_torch(destination, payload)
    return destination

def _fit_worker(masks, tasks, protocol, device, seeds):
    torch.set_num_threads(1)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    return _fit_batch(masks, tasks, protocol, device, seeds)


def iter_candidate_batches(masks, tasks, protocol, *, devices, batch_size,
                          initialization_seeds, device):
    """Fit ordered candidate chunks; keep at most one pending chunk per GPU."""
    chunks = [(start, min(start + batch_size, len(masks)))
              for start in range(0, len(masks), batch_size)]
    def arguments(start, stop, target):
        selected_tasks = tasks[start:stop] if isinstance(tasks, list) else tasks
        return (masks[start:stop], selected_tasks, protocol, target,
                initialization_seeds[start:stop])
    if len(devices) <= 1:
        for start, stop in chunks:
            yield start, _fit_batch(*arguments(start, stop, device))
        return
    context = mp.get_context("spawn")
    pools = [ProcessPoolExecutor(max_workers=1, mp_context=context) for _ in devices]
    pending = []
    next_chunk = 0
    try:
        for index, target in enumerate(devices):
            if next_chunk == len(chunks):
                break
            start, stop = chunks[next_chunk]
            pending.append((start, index, pools[index].submit(_fit_worker, *arguments(start, stop, target))))
            next_chunk += 1
        while pending:
            start, index, future = pending.pop(0)
            yield start, future.result()
            if next_chunk < len(chunks):
                begin, end = chunks[next_chunk]
                pending.append((begin, index, pools[index].submit(_fit_worker, *arguments(begin, end, devices[index]))))
                next_chunk += 1
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)


def validate_devices(args):
    devices = tuple(args.devices or [args.device])
    devices = tuple(("cuda:0" if torch.cuda.is_available() else "cpu") if d == "auto" else d
                    for d in devices)
    for value in devices:
        target = torch.device(value)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("supported devices: cpu and cuda:N")
        if target.type == "cuda" and (not torch.cuda.is_available() or
            (target.index or 0) >= torch.cuda.device_count()):
            raise ValueError(f"unavailable device: {value}")
    if len(set(devices)) != len(devices):
        raise ValueError("devices must be unique")
    if len(devices) > 1 and any(torch.device(d).type != "cuda" for d in devices):
        raise ValueError("multiple devices must all be CUDA GPUs")
    return devices


def save_bank(bank, destination, task_name, extract_profile):
    destination.mkdir(parents=True, exist_ok=False)
    manifest = []
    for number, (mask, token, row) in enumerate(zip(bank.masks, bank.tokens[0], bank.states)):
        source = row["source"]
        _, raw = extract_profile(row["state_dict"], mask, bank.diagnostics["probe_x"])
        metadata = {"task": task_name, "source": source, "row_hash": row["row_hash"],
                    "probe_ids": bank.provenance["probe_ids"],
                    "probe_fingerprint": bank.provenance["probe_fingerprint"]}
        path = destination / "maps" / f"candidate_{source['candidate_id']:06d}.pt"
        write_functional_card(path, token=token, mask=mask, state=row["state_dict"],
                              metadata=metadata, diagnostics=raw)
        manifest.append({"path": str(path.relative_to(destination)),
                         "candidate_id": source["candidate_id"],
                         "active_edges": int(mask.sum()), "row_hash": row["row_hash"]})
    # Portable tensors/dicts only: artifacts do not depend on Python module paths.
    save_torch(destination / "bank.pt", {
        "tokens": bank.tokens, "masks": bank.masks, "baseline_mask": bank.baseline_mask,
        "provenance": bank.provenance, "diagnostics": bank.diagnostics})
    save_json(destination / "manifest.json", {"schema": "functional_maps.bank.v1",
              "task": task_name, "provenance": bank.provenance, "maps": manifest})
    return len(manifest)


def base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--candidates", type=int, default=1000)
    parser.add_argument("--teachers", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda:N")
    parser.add_argument("--devices", nargs="+", help="CUDA devices for parallel candidate fits")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads per process")
    return parser


def preflight(args):
    for name in ("steps", "candidates", "teachers", "batch_size", "threads", "probe_count", "k"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.candidates < args.teachers:
        raise ValueError("candidates must be at least teachers")
    if args.out.exists():
        raise FileExistsError(f"output directory already exists: {args.out}")
    torch.set_num_threads(args.threads)
    return validate_devices(args)

INPUT_DIM = _FEATURES = 784
_SET_SIZE = 5
_DENSE_FRACTION_BUCKETS = (.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.)

class Split(NamedTuple):
    """A disjoint image pool; ``source_ids`` are original MNIST8m row ids."""
    features: Tensor
    digits: Tensor
    source_ids: Tensor

def _hash_ids(ids: Tensor) -> str:
    return hashlib.sha256(ids.detach().cpu().numpy().astype('<i8', copy=False).tobytes()).hexdigest()

def _read_split(images: np.ndarray, labels: np.ndarray, *, block_id: int, per_digit: int, seed: int, part: int, device: torch.device, used_rows: set[int], used_pixel_hashes: set[bytes]) -> tuple[Split, int]:
    """Choose a reproducible per-digit partition from one MNIST8m eighth."""
    block = len(labels) // 8
    lo, hi = (block_id * block, (block_id + 1) * block)
    local = np.asarray(labels[lo:hi])
    rng = np.random.default_rng(seed + 1009 * block_id)
    selected: list[np.ndarray] = []
    duplicate_rows_skipped = 0
    for digit in range(10):
        candidates = np.flatnonzero(local == digit)
        need = (part + 1) * per_digit
        if len(candidates) < need:
            raise ValueError(f'block {block_id} lacks images of digit {digit}')
        draw = rng.permutation(candidates)
        chosen: list[int] = []
        for local_id in draw[part * per_digit:]:
            row_id = int(local_id + lo)
            if row_id in used_rows:
                continue
            image_hash = hashlib.sha256(np.ascontiguousarray(images[row_id]).tobytes()).digest()
            if image_hash in used_pixel_hashes:
                duplicate_rows_skipped += 1
                continue
            used_rows.add(row_id)
            used_pixel_hashes.add(image_hash)
            chosen.append(row_id)
            if len(chosen) == per_digit:
                break
        if len(chosen) != per_digit:
            raise ValueError(f'could not form pixel-unique split in block {block_id}, digit {digit}')
        selected.append(np.asarray(chosen, dtype=np.int64))
    ids = np.concatenate(selected)
    np.random.default_rng(seed + 7919 * (block_id + 1) + part).shuffle(ids)
    pixels = torch.from_numpy(np.asarray(images[ids], dtype=np.uint8)).to(device)
    digits = torch.from_numpy(np.asarray(labels[ids], dtype=np.int64)).to(device)
    original_ids = torch.from_numpy(ids.astype(np.int64, copy=False)).to(device)
    return (Split(pixels.float().div_(255.0), digits, original_ids), duplicate_rows_skipped)

class MaskedDeepSets(nn.Module):
    """A vectorized batch of raw-pixel DeepSets regressors.

    Parameters have leading dimension ``models``.  A single model computes
    ``sum_i a^T tanh(x_i @ (W * M) + b) + c``.
    """

    def __init__(self, masks: Tensor, *, seed: int, initialization_reference_models: int | None=None) -> None:
        super().__init__()
        if masks.ndim != 3 or masks.shape[1] != INPUT_DIM:
            raise ValueError('masks must have shape [models, 784, hidden]')
        self.models, _, self.hidden = masks.shape
        self.register_buffer('masks', masks.float())
        gen = torch.Generator(device=masks.device).manual_seed(seed)
        bound = math.sqrt(6.0 / (INPUT_DIM + self.hidden))
        if initialization_reference_models is None:
            self.weight = nn.Parameter(torch.empty_like(masks).uniform_(-bound, bound, generator=gen))
            reference_readout = None
        else:
            if initialization_reference_models < 1:
                raise ValueError('initialization_reference_models must be positive')
            reference_weight = torch.empty(initialization_reference_models, INPUT_DIM, self.hidden, device=masks.device).uniform_(-bound, bound, generator=gen)
            reference_readout = torch.empty(initialization_reference_models, self.hidden, device=masks.device).uniform_(-1.0 / math.sqrt(self.hidden), 1.0 / math.sqrt(self.hidden), generator=gen)
            reference_indices = torch.arange(self.models, device=masks.device) % initialization_reference_models
            self.weight = nn.Parameter(reference_weight[reference_indices].clone())
        self.bias = nn.Parameter(torch.zeros(self.models, self.hidden, device=masks.device))
        if reference_readout is None:
            self.readout = nn.Parameter(torch.empty(self.models, self.hidden, device=masks.device).uniform_(-1.0 / math.sqrt(self.hidden), 1.0 / math.sqrt(self.hidden), generator=gen))
        else:
            self.readout = nn.Parameter(reference_readout[reference_indices].clone())
        self.per_image_offset = nn.Parameter(torch.zeros(self.models, device=masks.device))

    def forward(self, x: Tensor) -> Tensor:
        hidden = torch.tanh(torch.einsum('bsi,mih->mbsh', x, self.weight * self.masks) + self.bias[:, None, None, :])
        per_image = (hidden * self.readout[:, None, None, :]).sum(dim=-1)
        per_image = per_image + self.per_image_offset[:, None, None]
        return per_image.sum(dim=-1)

    def importance(self) -> Tensor:
        values = (self.weight.detach().abs() * self.masks).flatten(1)
        return (values / values.amax(dim=1, keepdim=True).clamp_min(1e-12)).reshape_as(self.masks)

class BatchedConditions(nn.Module):

    def __init__(self, masks, condition_seeds, replicas, reference_models=20, kernel_mode='reference'):
        super().__init__()
        self.kernel_mode = kernel_mode
        self.register_buffer('masks', masks[None].expand(len(condition_seeds), -1, -1, -1))
        states = []
        exemplar = {}
        for i, r in enumerate(replicas):
            exemplar.setdefault(r, i)
        source = torch.tensor([exemplar[r] for r in replicas], device=masks.device)
        for seed in condition_seeds:
            model = MaskedDeepSets(masks, seed=seed, initialization_reference_models=reference_models)
            states.append({name: p.detach()[source].clone() for name, p in model.named_parameters()})
        for name in states[0]:
            self.register_parameter(name, nn.Parameter(torch.stack([s[name] for s in states])))

    def forward(self, x):
        if self.kernel_mode == 'reference':
            values = []
            for j in range(len(x)):
                hidden = torch.tanh(torch.einsum('bsi,mih->mbsh', x[j], self.weight[j] * self.masks[j]) + self.bias[j, :, None, None, :])
                values.append(((hidden * self.readout[j, :, None, None, :]).sum(-1) + self.per_image_offset[j, :, None, None]).sum(-1))
            return torch.stack(values)
        c, b, s, f = x.shape
        _, m, _, h = self.weight.shape
        weight = (self.weight * self.masks).permute(0, 2, 1, 3).reshape(c, f, m * h)
        pre = torch.bmm(x.reshape(c, b * s, f), weight).reshape(c, b, s, m, h).permute(0, 3, 1, 2, 4)
        hidden = torch.tanh(pre + self.bias[:, :, None, None, :])
        return ((hidden * self.readout[:, :, None, None, :]).sum(-1) + self.per_image_offset[:, :, None, None]).sum(-1)

@torch.no_grad()
def losses(model, x, y, set_size, chunk=128):
    total = None
    for start in range(0, x.shape[1], chunk):
        values = (model(x[:, start:start + chunk]) - y[:, None, start:start + chunk]).square().sum(-1)
        total = values if total is None else total + values
    return total / x.shape[1] / set_size

def _exact_replica_initialization(model: BatchedConditions, masks: Tensor, condition_seeds: Sequence[int], replicas: Sequence[int], reference_models: int) -> None:
    """Use the numbered rows of the same reference initialization bank.

    ``BatchedConditions`` pairs replicas by first occurrence. That is useful
    for generic repeated labels, but these fits have stable reference IDs: a
    child labelled replica ``r`` must receive reference draw ``r`` even when
    another method's replica appeared earlier in the input list.
    """
    features, hidden = masks.shape[1:]
    reference_masks = torch.ones(reference_models, features, hidden, dtype=masks.dtype, device=masks.device)
    replica_index = torch.as_tensor(replicas, dtype=torch.long, device=masks.device)
    with torch.no_grad():
        for condition, seed in enumerate(condition_seeds):
            reference = MaskedDeepSets(reference_masks, seed=int(seed), initialization_reference_models=reference_models)
            model.weight[condition].copy_(reference.weight[replica_index])
            model.readout[condition].copy_(reference.readout[replica_index])

def _l2_penalty(model: BatchedConditions, l2: float) -> Tensor:
    """Per-child half-scaled sum of squares on effective parameters."""
    effective_weight = model.weight * model.masks
    return 0.5 * l2 * (effective_weight.square().sum(dim=(2, 3)) + model.bias.square().sum(dim=-1) + model.readout.square().sum(dim=-1) + model.per_image_offset.square())

def _streamed_losses(model: BatchedConditions, x: Tensor, y: Tensor, set_size: int, chunk_size: int, device: torch.device) -> Tensor:
    """Evaluate the shared batched loss while moving query chunks as needed."""
    total: Tensor | None = None
    count = x.shape[1]
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        x_chunk = x[:, start:stop].to(device=device, dtype=torch.float32)
        y_chunk = y[:, start:stop].to(device=device, dtype=torch.float32)
        chunk_mean = losses(model, x_chunk, y_chunk, set_size, chunk=chunk_size)
        weighted = chunk_mean * (stop - start)
        total = weighted if total is None else total + weighted
    assert total is not None
    return total / count

def _plateau_flags(history: list[Tensor], steps: list[int], checkpoint_every: int, *, tolerance: float=0.01) -> Tensor:
    """Support-objective stability audit over the last 100 and 50 updates.

    The 100-step endpoint change and the range across the trailing 50-step
    window must both be within 1% of the corresponding objective scale. With
    insufficient history the flag is false. This flag is descriptive only.
    """
    current = history[-1]
    n100 = max(1, int(round(100 / checkpoint_every)))
    n50 = max(1, int(round(50 / checkpoint_every)))
    if len(history) <= n100 or steps[-1] - steps[-1 - n100] < 100:
        return torch.zeros_like(current, dtype=torch.bool)
    before = history[-1 - n100]
    scale100 = torch.maximum(before.abs(), current.abs()).clamp_min(1e-08)
    stable100 = (current - before).abs() / scale100 <= tolerance
    recent = torch.stack(history[-(n50 + 1):])
    recent_mean = recent.mean(dim=0)
    scale50 = recent_mean.abs().clamp_min(1e-08)
    stable50 = (recent.amax(dim=0) - recent.amin(dim=0)) / scale50 <= tolerance
    return stable100 & stable50

def fit_children(masks: Tensor, x_support: Tensor, y_support: Tensor, x_query: Tensor, y_query: Tensor, condition_seeds: Sequence[int], replicas: Sequence[int], steps: int, lr: float, l2: float, device: str | torch.device, checkpoint_callback: Callable[[dict[str, Any]], None] | None=None, *, reference_models: int=20, chunk_size: int=32, batch_size: int | None=None, seed: int=0, lr_decay_every: int=200, lr_floor: float=1.0 / 64.0, support_sampler: Callable[[int], tuple[Tensor, Tensor]] | None=None, initial_state: dict[str, Tensor] | None=None, optimizer_state: dict[str, Any] | None=None, start_step: int=0, checkpoint_every: int=25, plateau_tolerance: float=0.01, metrics_callback: Callable[[dict[str, Any]], None] | None=None) -> dict[str, Any]:
    """Fit fresh children for paired task conditions and candidate masks.

    Args:
        masks: Candidate masks ``[M, 784, H]``.
        x_support/y_support: Paired support sets ``[C,N,5,784]`` and ``[C,N]``.
        x_query/y_query: Diagnostic query sets ``[C,Q,5,784]`` and ``[C,Q]``.
        condition_seeds: One deterministic child initialization seed per task.
        replicas: Reference-model initialization IDs, one per mask row.
        steps: Fixed Adam update count. No early stopping is performed.
        lr: Constant Adam learning rate.
        l2: Coefficient for ``0.5*l2*(||W*M||² + ||b||² + ||a||² + o²)``.
        device: Training device.
        checkpoint_callback: Optional one-argument callback receiving a CPU
            metrics record and the current trainable parameter tensors at each
            checkpoint. It does not control the optimizer.
        reference_models: Size of the shared initialization bank (default 20).
        chunk_size: Number of support/query sets per forward chunk. Training
            accumulates scaled gradients over every chunk before each step.
        batch_size: Optional support-only minibatch size. ``None`` (default)
            uses every support set at every step; smaller values sample rows
            independently per condition using the fixed ``seed`` and transfer
            only that minibatch.
        seed: CPU RNG seed for reproducible support minibatch indices.
        lr_decay_every: Halve the learning rate at this step interval.
        lr_floor: Minimum learning-rate multiplier relative to ``lr``.
        checkpoint_every: Diagnostic cadence; the terminal step is always saved.
        plateau_tolerance: Relative tolerance for the support-objective audit.
        metrics_callback: CPU scalar metrics at checkpoints, without copying weights.

    Returns CPU tensors. ``support_loss`` and ``query_loss`` are the per-child
    normalized MSE used by ``followup_batched_eval.losses``. The optimizer
    minimizes ``support_loss + l2_penalty``. Query metrics are diagnostics only.
    """
    device = torch.device(device)
    masks = torch.as_tensor(masks, dtype=torch.float32, device=device)
    x_support = torch.as_tensor(x_support, dtype=torch.float32)
    y_support = torch.as_tensor(y_support, dtype=torch.float32)
    condition_seeds = [int(seed) for seed in condition_seeds]
    replicas = [int(replica) for replica in replicas]
    if masks.ndim != 3 or masks.shape[1] != 784 or masks.shape[0] < 1 or (masks.shape[2] < 1):
        raise ValueError('masks must have shape [M, 784, H] with positive M and H')
    if x_support.ndim != 4 or x_support.shape[2:] != (5, 784):
        raise ValueError('x_support must have shape [C, N, 5, 784]')
    conditions, support_count = x_support.shape[:2]
    if support_count < 1 or y_support.shape != (conditions, support_count):
        raise ValueError('y_support must have shape [C, N] matching x_support')
    if x_query.ndim != 4 or x_query.shape[0] != conditions or x_query.shape[2:] != (5, 784):
        raise ValueError('x_query must have shape [C, Q, 5, 784]')
    query_count = x_query.shape[1]
    if query_count < 1 or y_query.shape != (conditions, query_count):
        raise ValueError('y_query must have shape [C, Q] matching x_query')
    if len(condition_seeds) != conditions or len(replicas) != masks.shape[0]:
        raise ValueError('condition_seeds must have length C and replicas length M')
    if min(steps, chunk_size, checkpoint_every, reference_models, lr_decay_every) < 1:
        raise ValueError('steps, chunk_size, checkpoint cadence, reference count, and lr decay must be positive')
    if batch_size is not None and batch_size < 1:
        raise ValueError('batch_size must be positive or None')
    if lr <= 0 or l2 < 0 or plateau_tolerance <= 0 or (not 0 < lr_floor <= 1):
        raise ValueError('lr, plateau_tolerance, and lr_floor must be positive; l2 must be nonnegative')
    if any((replica < 0 or replica >= reference_models for replica in replicas)):
        raise ValueError('replica IDs must index the reference initialization bank')
    x_query = torch.as_tensor(x_query)
    y_query = torch.as_tensor(y_query)
    model = BatchedConditions(masks, condition_seeds, replicas, reference_models=reference_models, kernel_mode='bmm').to(device)
    _exact_replica_initialization(model, masks, condition_seeds, replicas, reference_models)
    if initial_state is not None:
        model.load_state_dict(initial_state)
        if not torch.equal(model.masks, masks[None].expand_as(model.masks)):
            raise ValueError('resume masks differ')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    callback_masks = model.masks.detach().cpu().clone() if checkpoint_callback is not None else None
    step_history: list[int] = []
    support_history: list[Tensor] = []
    objective_history: list[Tensor] = []
    query_history: list[Tensor] = []
    penalty_history: list[Tensor] = []
    plateau_history: list[Tensor] = []
    minibatch_generators = [torch.Generator(device='cpu').manual_seed(int(seed) + 1009 * (condition + 1)) for condition in range(conditions)]
    learning_rate_history: list[float] = []
    full_support_slices = tuple((slice(start, min(support_count, start + chunk_size)) for start in range(0, support_count, chunk_size)))
    parameters = tuple(model.parameters())
    gradients_finite = torch.ones((), dtype=torch.bool, device=device) if device.type == 'cuda' else None

    @torch.no_grad()
    def record(step: int) -> None:
        support_nmse = _streamed_losses(model, x_support, y_support, 5, chunk_size, device)
        penalty = _l2_penalty(model, l2)
        objective = support_nmse + penalty
        query_nmse = _streamed_losses(model, x_query, y_query, 5, chunk_size, device)
        prospective_objectives = objective_history + [objective.detach().cpu().clone()]
        prospective_steps = step_history + [step]
        plateau = _plateau_flags(prospective_objectives, prospective_steps, checkpoint_every, tolerance=plateau_tolerance)
        step_history.append(step)
        support_history.append(support_nmse.detach().cpu().clone())
        objective_history.append(objective.detach().cpu().clone())
        query_history.append(query_nmse.detach().cpu().clone())
        penalty_history.append(penalty.detach().cpu().clone())
        plateau_history.append(plateau.detach().cpu().clone())
        learning_rate_history.append(float(optimizer.param_groups[0]['lr']))
        if metrics_callback is not None:
            metrics_callback({'step': step, 'support_nmse': float(support_history[-1].mean()), 'query_nmse': float(query_history[-1].mean()), 'lr': learning_rate_history[-1]})
        if checkpoint_callback is not None:
            callback_state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
            callback_state['masks'] = callback_masks
            checkpoint_callback({'step': step, 'support_loss': support_history[-1], 'support_objective': objective_history[-1], 'trainNMSE': support_history[-1], 'queryNMSE': query_history[-1], 'l2_penalty': penalty_history[-1], 'plateau_flags': plateau_history[-1], 'lr': learning_rate_history[-1], 'state_dict': callback_state, 'optimizer_state': optimizer.state_dict()})
    record(start_step)
    for step in range(start_step + 1, start_step + steps + 1):
        current_lr = lr * max(lr_floor, 0.5 ** ((step - 1) // lr_decay_every))
        for group in optimizer.param_groups:
            group['lr'] = current_lr
        optimizer.zero_grad(set_to_none=True)
        if support_sampler is not None:
            sampled_x, sampled_y = support_sampler(step)
            sampled_x = sampled_x.to(device)
            sampled_y = sampled_y.to(device)
            if sampled_x.shape[:1] != (conditions,) or sampled_x.shape[2:] != (5, 784):
                raise ValueError('support sampler must return [C,B,5,784] and [C,B]')
            if sampled_y.shape != sampled_x.shape[:2]:
                raise ValueError('support sampler labels must match batch dimensions')
            train_count = sampled_x.shape[1]
            chunk_indices = [None]
        elif batch_size is None or batch_size >= support_count:
            chunk_indices = full_support_slices
            train_count = support_count
        else:
            train_count = batch_size
            condition_indices = torch.stack([torch.randint(support_count, (train_count,), generator=generator) for generator in minibatch_generators])
            chunk_indices = [condition_indices]
        for indices_cpu in chunk_indices:
            if support_sampler is not None:
                x_chunk, y_chunk = (sampled_x, sampled_y)
            elif batch_size is not None and batch_size < support_count:
                indices = indices_cpu.to(x_support.device)
                x_chunk = torch.stack([x_support[condition].index_select(0, indices[condition]) for condition in range(conditions)]).to(device=device, dtype=torch.float32)
                y_chunk = torch.stack([y_support[condition].index_select(0, indices[condition]) for condition in range(conditions)]).to(device=device, dtype=torch.float32)
            else:
                x_chunk = x_support[:, indices_cpu].to(device=device, dtype=torch.float32)
                y_chunk = y_support[:, indices_cpu].to(device=device, dtype=torch.float32)
            prediction = model(x_chunk)
            residual = prediction - y_chunk[:, None, :]
            per_model_chunk_loss = residual.square().sum(dim=-1) / (train_count * 5 * reference_models)
            per_model_chunk_loss.sum().backward()
        (_l2_penalty(model, l2).sum() / reference_models).backward()
        for parameter in parameters:
            if parameter.grad is not None:
                finite = torch.isfinite(parameter.grad).all()
                if gradients_finite is None:
                    if not finite:
                        raise RuntimeError('nonfinite fresh-child gradient')
                else:
                    gradients_finite.logical_and_(finite)
        is_checkpoint = step % checkpoint_every == 0 or step == start_step + steps
        if gradients_finite is not None and (step % 64 == 0 or is_checkpoint):
            if not gradients_finite:
                raise RuntimeError('nonfinite fresh-child gradient')
        optimizer.step()
        if is_checkpoint:
            record(step)
    with torch.no_grad():
        final_support = _streamed_losses(model, x_support, y_support, 5, chunk_size, device)
        final_penalty = _l2_penalty(model, l2)
        final_query = _streamed_losses(model, x_query, y_query, 5, chunk_size, device)
        final_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history = {'steps': torch.tensor(step_history, dtype=torch.long), 'support_loss': torch.stack(support_history), 'support_objective': torch.stack(objective_history), 'trainNMSE': torch.stack(support_history), 'queryNMSE': torch.stack(query_history), 'l2_penalty': torch.stack(penalty_history), 'plateau_flags': torch.stack(plateau_history), 'learning_rate': torch.tensor(learning_rate_history, dtype=torch.float32)}
    return {'state_dict': final_state, 'support_loss': final_support.detach().cpu(), 'query_loss': final_query.detach().cpu(), 'support_objective': (final_support + final_penalty).detach().cpu(), 'l2_penalty': final_penalty.detach().cpu(), 'plateau_flags': history['plateau_flags'][-1].clone(), 'history': history, 'steps_run': steps, 'fixed_horizon': True, 'stopping_source': 'none; fixed step horizon', 'plateau_source': 'support objective only; diagnostic and non-stopping', 'l2_definition': '0.5*l2*(sum((weight*masks)^2)+sum(bias^2)+sum(readout^2)+sum(offset^2)) per child', 'gradient_normalizer': reference_models, 'chunk_size': chunk_size, 'batch_size': batch_size, 'minibatch_seed': int(seed) if batch_size is not None else None, 'minibatch_condition_seeds': [int(seed) + 1009 * (condition + 1) for condition in range(conditions)] if batch_size is not None else None, 'base_lr': lr, 'lr_decay_every': lr_decay_every, 'lr_floor': lr_floor, 'support_sampling': 'independent fresh source sets per update; fixed support monitor' if support_sampler is not None else 'fixed support dataset', 'optimizer_state': optimizer.state_dict(), 'terminal_step': start_step + steps}

def _slice_model_axis(value: Any, start: int, stop: int) -> Any:
    """Copy a packed ``[condition, model, ...]`` result slice to CPU."""
    if not torch.is_tensor(value):
        return deepcopy(value)
    value = value.detach().cpu()
    if value.ndim >= 2 and value.shape[0] == 1:
        return value[:, start:stop].clone()
    return value.clone()

def _slice_history(value: Any, start: int, stop: int) -> Any:
    """Copy one candidate from a ``[checkpoint, condition, model]`` trace."""
    if not torch.is_tensor(value):
        return deepcopy(value)
    value = value.detach().cpu()
    if value.ndim >= 3 and value.shape[1] == 1:
        return value[:, :, start:stop].clone()
    return value.clone()

def _slice_optimizer_state(state: dict[str, Any], start: int, stop: int) -> dict[str, Any]:
    """Keep Adam moments for one candidate's replica range."""
    result = {'state': {}, 'param_groups': deepcopy(state['param_groups'])}
    for parameter_id, values in state['state'].items():
        result['state'][parameter_id] = {name: _slice_model_axis(value, start, stop) for name, value in values.items()}
    return result

def _validate(masks: Tensor, task: TaskData, protocol: InnerProtocol) -> Tensor:
    masks = torch.as_tensor(masks, dtype=torch.float32).detach().cpu().contiguous()
    if protocol.metric != 'nmse':
        raise ValueError("the DeepSets batch adapter requires protocol.metric='nmse'")
    if masks.ndim != 3 or masks.shape[0] < 1 or masks.shape[1] != 784 or (masks.shape[2] < 1):
        raise ValueError('masks must have shape [count, 784, hidden]')
    if not torch.isfinite(masks).all() or not bool(((masks == 0) | (masks == 1)).all()):
        raise ValueError('DeepSets masks must be finite and binary')
    if task.x_support.ndim != 3 or task.x_support.shape[1:] != (5, 784):
        raise ValueError('DeepSets task inputs must have shape [sets, 5, 784]')
    return masks

def fit_deepsets_batch(masks: Tensor, task: TaskData, protocol: InnerProtocol, device: str='cpu', *, initialization_seeds: Sequence[int] | None=None) -> list[dict[str, Any]]:
    """Fit many masks on one task and return ordinary individual results.

    Each candidate keeps the same numbered initialization replicas as a
    scalar adapter call.  Support and query data are uploaded once when the
    target device is CUDA; the fixed-horizon solver then performs one full
    support forward per update rather than seven small 32-set chunks.
    """
    masks = _validate(masks, task, protocol)
    replicas = protocol.replicas
    count = len(masks)
    target = torch.device(device)
    x_support = task.x_support.detach().float().unsqueeze(0).to(target)
    y_support = task.y_support.detach().float().unsqueeze(0).to(target)
    x_query = task.x_query.detach().float().unsqueeze(0).to(target)
    y_query = task.y_query.detach().float().unsqueeze(0).to(target)
    packed_masks = masks[:, None].expand(-1, replicas, -1, -1).reshape(count * replicas, *masks.shape[1:])
    seeds = [protocol.seed] * count if initialization_seeds is None else list(map(int, initialization_seeds))
    if len(seeds) != count:
        raise ValueError('one initialization seed is required per mask')
    initial_state = None
    if initialization_seeds is not None:
        if protocol.batch_size is not None and protocol.batch_size < len(task.x_support):
            raise ValueError('per-candidate initialization seeds require full-support fits')
        reference_masks = torch.ones(max(20, replicas), *masks.shape[1:], device=target)
        rows = []
        for seed in seeds:
            reference = MaskedDeepSets(reference_masks, seed=seed, initialization_reference_models=max(20, replicas))
            rows.append({name: value.detach()[:replicas].clone() for name, value in reference.named_parameters()})
        initial_state = {name: torch.cat([row[name] for row in rows])[None] for name in rows[0]}
        initial_state['masks'] = packed_masks.to(target)[None]
    fitted = fit_children(packed_masks, x_support, y_support, x_query, y_query, [protocol.seed], list(range(replicas)) * count, protocol.steps, protocol.lr, protocol.l2, target, reference_models=max(20, replicas), chunk_size=len(task.x_support), batch_size=protocol.batch_size, lr_decay_every=protocol.lr_decay_every, lr_floor=protocol.lr_floor, checkpoint_every=protocol.checkpoint_every, plateau_tolerance=protocol.plateau_tolerance, initial_state=initial_state)
    state = fitted['state_dict']
    results: list[dict[str, Any]] = []
    for index, mask in enumerate(masks):
        actual_seed = seeds[index]
        start, stop = (index * replicas, (index + 1) * replicas)
        child_state = {name: _slice_model_axis(value, start, stop) for name, value in state.items()}
        history = {name: _slice_history(value, start, stop) for name, value in fitted['history'].items()}
        weights = (child_state['weight'] * mask[None, None]).squeeze(0)
        results.append({'label_source': 'fresh_terminal_query', 'fixed_horizon': True, 'protocol_id': protocol.fingerprint, 'task_id': task.task_id, 'replica_losses': fitted['query_loss'][0, start:stop].detach().cpu().tolist(), 'seeds': [f'{actual_seed}:{replica}' for replica in range(replicas)], 'initialization': {'base_seed': actual_seed, 'reference_replica_ids': list(range(replicas))}, 'actual_initialization_seed': actual_seed, 'solver_protocol_seed': protocol.seed, 'minibatch_seed_base': protocol.seed if protocol.batch_size is not None else None, 'plateau_flags': fitted['plateau_flags'][0, start:stop].detach().cpu().tolist(), 'state_dict': child_state, 'optimizer_state': _slice_optimizer_state(fitted['optimizer_state'], start, stop), 'history': history, 'effective_weights': weights.detach().cpu()})
    return results

def _fit_batch(masks, task, protocol, device, seeds):
    return fit_deepsets_batch(masks, task, protocol, device, initialization_seeds=seeds)

def _cost_vectors(seed: int, count: int) -> Tensor:
    gen = torch.Generator().manual_seed(int(seed))
    values = torch.randn(count, 10, generator=gen)
    return (values - values.mean(1, keepdim=True)) / values.std(1, keepdim=True, unbiased=False).clamp_min(1e-06)

def _deepsets_sets(split: Any, costs: Tensor, count: int, generator: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
    """The core sampler plus the actual source image IDs used in each set."""
    indices = torch.randint(len(split.features), (count, 5), generator=generator)
    x = split.features[indices]
    y = costs[split.digits[indices]].sum(dim=1)
    return (x, y, split.source_ids[indices].detach().cpu())

def _candidate_initialization_seed(seed: int, candidate_id: int) -> int:
    return int(seed) + 1000003 * (int(candidate_id) + 1)

def _density_buckets(k: int, teachers: int, edges: int) -> tuple[int, ...]:
    """Use ten mixed-density strata, retaining the smoke fixture's five anchors."""
    if teachers >= 10:
        buckets = [round(edges * fraction) for fraction in _DENSE_FRACTION_BUCKETS]
    else:
        buckets = [round(edges * 0.1), k, round(edges * 0.5), round(edges * 0.7), edges]
    if k not in buckets:
        nearest = min(range(len(buckets)), key=lambda index: (abs(buckets[index] - k), index))
        buckets[nearest] = k
    return tuple(sorted(set((int(value) for value in buckets))))

def _stratum_counts(total: int, count: int) -> list[int]:
    quotient, remainder = divmod(total, count)
    return [quotient + int(index < remainder) for index in range(count)]

def _candidate_masks(candidate_count: int, k: int, seed: int, features: int, hidden: int, buckets: tuple[int, ...]) -> tuple[Tensor, list[int], list[int]]:
    edges = features * hidden
    counts = _stratum_counts(candidate_count, len(buckets))
    masks: list[Tensor] = []
    strata: list[int] = []
    identifiers: list[int] = []
    for density, amount in zip(buckets, counts):
        for _ in range(amount):
            candidate_id = len(identifiers)
            generator = torch.Generator(device='cpu').manual_seed(int(seed) + candidate_id)
            mask = torch.zeros(edges, dtype=torch.float32)
            mask[torch.randperm(edges, generator=generator)[:density]] = 1.0
            masks.append(mask.reshape(features, hidden))
            strata.append(density)
            identifiers.append(candidate_id)
    return (torch.stack(masks), identifiers, strata)

def _split_rows(split: Any, rows: Tensor) -> Any:
    rows = torch.as_tensor(rows, dtype=torch.long, device=split.features.device)
    return type(split)(split.features.index_select(0, rows), split.digits.index_select(0, rows), split.source_ids.index_select(0, rows))

def _partition_split(split: Any, sizes: Sequence[int], seed: int) -> list[Any]:
    if any((int(size) < 1 for size in sizes)) or sum(map(int, sizes)) > len(split.features):
        raise ValueError('requested disjoint image pools do not fit the supplied DeepSets split')
    order = torch.randperm(len(split.features), generator=torch.Generator().manual_seed(int(seed)))
    result, offset = ([], 0)
    for size in sizes:
        size = int(size)
        result.append(_split_rows(split, order[offset:offset + size]))
        offset += size
    return result

def _cost(task: TaskData) -> Tensor:
    value = torch.as_tensor(task.provenance.get('costs'), dtype=torch.float32)
    if value.shape != (10,) or not torch.isfinite(value).all():
        raise ValueError(f'{task.task_id} lacks its ten-value task cost vector')
    return value

def _fixture_cost_vector(seed: int, role: str, index: int) -> Tensor:
    """Keep the legacy four costs while deriving stable vectors for extra roles."""
    if role == 'train' and index < 2:
        return _cost_vectors(seed, 6)[index].clone()
    if role == 'heldout' and index < 2:
        return _cost_vectors(seed, 6)[4 + index].clone()
    if role == 'train':
        derived_seed = int(seed) + 1000003 * (int(index) + 1)
    elif role == 'heldout':
        derived_seed = int(seed) + 2000003 * (int(index) + 1)
    else:
        raise ValueError(f'unknown DeepSets task-cost role: {role!r}')
    return _cost_vectors(derived_seed, 1)[0]

def _unwrap_model_state(state: dict[str, Tensor], *, row: int=0, rows: int=1) -> dict[str, Tensor]:
    """Select one candidate and replica from the packed child-state schema."""
    result: dict[str, Tensor] = {}
    for name, value in state.items():
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().float().cpu()
        if tensor.ndim >= 2 and tensor.shape[0] == 1 and (tensor.shape[1] == rows):
            tensor = tensor[0, row]
        elif tensor.ndim >= 1 and tensor.shape[0] == 1 and (rows == 1):
            tensor = tensor[0]
        result[name] = tensor.contiguous().clone()
    return result

def extract_deepsets_functional_token(state: dict[str, Tensor], mask: Tensor, probe_x: Tensor, *, device: str | torch.device | None=None) -> tuple[Tensor, dict[str, Tensor]]:
    """Build normalized tanh contributions and feature derivatives on images."""
    required = ('weight', 'bias', 'readout', 'per_image_offset')
    if not isinstance(state, dict) or any((name not in state for name in required)):
        raise ValueError('DeepSets child state lacks weight, bias, readout or per_image_offset')
    target_device = torch.device('cpu' if device is None else device)
    weight = torch.as_tensor(state['weight'], dtype=torch.float32).detach().to(target_device)
    bias = torch.as_tensor(state['bias'], dtype=torch.float32).detach().to(target_device)
    readout = torch.as_tensor(state['readout'], dtype=torch.float32).detach().to(target_device)
    offset = torch.as_tensor(state['per_image_offset'], dtype=torch.float32).detach().to(target_device)
    mask = torch.as_tensor(mask, dtype=torch.float32).detach().to(target_device)
    probe_x = torch.as_tensor(probe_x, dtype=torch.float32).detach().to(target_device)
    if weight.ndim != 2 or weight.shape[0] != _FEATURES or mask.shape != weight.shape:
        raise ValueError('DeepSets weight and mask must share shape [784, hidden]')
    if bias.shape != (weight.shape[1],) or readout.shape != bias.shape or offset.numel() != 1:
        raise ValueError('DeepSets bias/readout must match hidden width and offset must be scalar')
    if probe_x.ndim != 2 or probe_x.shape[1] != _FEATURES or len(probe_x) < 1:
        raise ValueError('probe_x must have shape [probe_rows, 784]')
    if not all((torch.isfinite(value).all() for value in (weight, bias, readout, offset, mask, probe_x))):
        raise ValueError('functional profile inputs must be finite')
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError('DeepSets masks must be binary')
    effective = weight * mask
    activation = torch.tanh(probe_x @ effective + bias)
    psi = activation * readout
    gain = (1.0 - activation.square()) * readout
    count = len(probe_x)
    signed = effective * torch.einsum('pf,ph->fh', probe_x, gain) / count
    absolute = effective.abs() * torch.einsum('pf,ph->fh', probe_x.abs(), gain.abs()) / count
    rms = effective.abs() * torch.einsum('pf,ph->fh', probe_x.square(), gain.square()).div(count).sqrt()
    psi_scale = psi.square().mean().sqrt().clamp_min(1e-08)
    q_scale = rms.amax().clamp_min(1e-08)
    tokens = torch.cat((psi.div(psi_scale).T, signed.div(q_scale).T, absolute.div(q_scale).T, rms.div(q_scale).T, mask.T), dim=1)
    tokens = tokens.contiguous().cpu().contiguous()
    raw = {'psi': psi.contiguous().cpu().contiguous(), 'q_signed_mean': signed.contiguous().cpu().contiguous(), 'q_abs_mean': absolute.contiguous().cpu().contiguous(), 'q_rms': rms.contiguous().cpu().contiguous(), 'psi_scale': psi_scale.reshape(1).cpu().contiguous(), 'q_scale': q_scale.reshape(1).cpu().contiguous(), 'effective_weights': effective.contiguous().cpu().contiguous()}
    return (tokens, raw)

def _state_hash(state: dict[str, Tensor], mask: Tensor) -> str:
    digest = hashlib.sha256()
    for name in ('weight', 'bias', 'readout', 'per_image_offset'):
        if name not in state:
            raise ValueError('terminal DeepSets state is incomplete')
        digest.update(tensor_hash(torch.as_tensor(state[name], dtype=torch.float32)).encode())
    digest.update(tensor_hash(torch.as_tensor(mask, dtype=torch.float32)).encode())
    return digest.hexdigest()

def _align_hidden_columns(reference: Tensor, values: Tensor) -> Tensor:
    """Align a raw feature-by-hidden map to a reference by cosine similarity."""
    from scipy.optimize import linear_sum_assignment
    if reference.shape != values.shape or reference.ndim != 2:
        raise ValueError('DeepSets functional alignment needs matching [784, hidden] maps')
    ref = reference / reference.square().sum(0).sqrt().clamp_min(1e-12)
    candidate = values / values.square().sum(0).sqrt().clamp_min(1e-12)
    rows, columns = linear_sum_assignment(-(ref.T @ candidate).detach().cpu().numpy())
    order = torch.empty(reference.shape[1], dtype=torch.long)
    order[torch.as_tensor(rows, dtype=torch.long)] = torch.as_tensor(columns, dtype=torch.long)
    return values.index_select(1, order)

def _build_bank(task_index: int, task: TaskData, support_split: Any, query_split: Any, probe_x: Tensor, probe_ids: Tensor, *, seed: int, bank_steps: int, bank_candidates: int, teachers_per_task: int, support_count: int, query_count: int, teacher_batch_size: int, k: int, hidden: int, device: str, measurement_devices: tuple[str, ...], bank_out: Path, persist_artifacts: bool=False) -> FunctionalBank:
    features = _FEATURES
    edge_count = features * hidden
    buckets = _density_buckets(k, teachers_per_task, edge_count)
    quotas = _stratum_counts(teachers_per_task, len(buckets))
    candidate_counts = _stratum_counts(bank_candidates, len(buckets))
    if any((available < keep for available, keep in zip(candidate_counts, quotas))):
        raise ValueError('bank_candidates must provide enough candidates in every density stratum')
    masks, candidate_ids, strata = _candidate_masks(bank_candidates, k, seed, features, hidden, buckets)
    cost = _cost(task)
    generator = torch.Generator(device='cpu').manual_seed(seed + 10007)
    x_support, y_support, support_ids = _deepsets_sets(support_split, cost, support_count, generator)
    x_query, y_query, query_ids = _deepsets_sets(query_split, cost, query_count, generator)
    if set(support_ids.reshape(-1).tolist()) & set(probe_ids.tolist()) or set(support_ids.reshape(-1).tolist()) & set(query_ids.reshape(-1).tolist()) or set(probe_ids.tolist()) & set(query_ids.reshape(-1).tolist()):
        raise ValueError('source teacher support/query/probe raw image IDs overlap')
    source_task = TaskData(f'deepsets:{task_index}:bank', 'train', x_support, y_support, x_query, y_query, support_context(x_support.mean(1), y_support), support_ids, query_ids, {'family': 'deepsets', 'domain': 'deepsets', 'role': 'bank_teacher', 'task_id': task.task_id, 'task_index': task_index, 'costs': cost.tolist(), 'support_pool': 'source_train', 'query_pool': 'source_validation', 'set_size': _SET_SIZE})
    protocol = InnerProtocol(steps=bank_steps, replicas=1, lr=0.03, l2=0.001, checkpoint_every=max(1, bank_steps // 4), seed=seed, metric='nmse')
    best_by_density: dict[int, list[tuple[float, int, Tensor, dict[str, Any], dict[str, Any]]]] = {density: [] for density in buckets}
    if persist_artifacts:
        bank_out.mkdir(parents=True, exist_ok=True)
    for start, fitted in progress(iter_candidate_batches(
            masks, source_task, protocol, devices=measurement_devices,
            batch_size=teacher_batch_size, device=device,
            initialization_seeds=[_candidate_initialization_seed(seed, i) for i in candidate_ids]),
            desc=f"DeepSets bank candidates {task_index}", unit="batch"):
        for offset, result in enumerate(fitted):
            candidate_id = start + offset
            density = strata[candidate_id]
            score = float(result["replica_losses"][0])
            retained = best_by_density[density]
            retained_result = {"state_dict": result["state_dict"],
                               "replica_losses": result["replica_losses"]}
            retained.append((score, candidate_id, masks[candidate_id], retained_result, {}))
            retained.sort(key=lambda row: (row[0], row[1]))
            del retained[quotas[buckets.index(density)]:]
    del fitted, result
    selected: list[tuple[float, int, Tensor, dict[str, Any], dict[str, Any]]] = []
    for density, quota in zip(buckets, quotas):
        rows = best_by_density[density]
        if len(rows) != quota:
            raise ValueError('source candidate selection did not retain the requested density quota')
        selected.extend(rows)
    selected.sort(key=lambda row: row[1])
    tokens: list[Tensor] = []
    selected_masks: list[Tensor] = []
    states: list[dict[str, Any]] = []
    for teacher_index, (score, candidate_id, mask, result, record) in enumerate(selected):
        state = _unwrap_model_state(result['state_dict'], row=0, rows=1)
        if 'masks' in state and (not torch.equal(state['masks'], mask)):
            raise ValueError('source child state mask differs from its candidate mask')
        token, raw = extract_deepsets_functional_token(state, mask, probe_x)
        row_hash = _state_hash(state, mask)
        initialization_seed = _candidate_initialization_seed(seed, candidate_id)
        card_path = bank_out / 'maps' / f'candidate_{candidate_id:06d}_init_{initialization_seed}.pt' if persist_artifacts else None
        card_state = {name: state[name] for name in ('weight', 'bias', 'readout', 'per_image_offset')}
        if persist_artifacts:
            write_functional_card(card_path, token=token, mask=mask, state=card_state, metadata={'kind': 'initial_teacher', 'candidate_id': int(candidate_id), 'initialization_seed': int(initialization_seed), 'score_name': 'source_query_nmse', 'score': float(score), 'row_hash': row_hash, 'task_source': {'task_id': task.task_id, 'task_index': int(task_index), 'bank_task_id': source_task.task_id, 'task_provenance': deepcopy(task.provenance), 'bank_task_provenance': deepcopy(source_task.provenance), 'probe_ids': probe_ids.detach().cpu().tolist(), 'probe_fingerprint': tensor_hash(probe_x)}, 'measurement': {'protocol_id': protocol.fingerprint, 'mask_key': record.get('mask_key'), 'active_edges': int(mask.sum()), 'density_stratum': int(strata[candidate_id]), 'candidate_seed_rule': 'seed + 1000003 * (candidate_id + 1)'}})
        tokens.append(token)
        selected_masks.append(mask.detach().cpu().clone())
        teacher_state = {**raw, 'state_dict': state, 'row_hash': row_hash, 'source': {'kind': 'initial_teacher', 'task_id': task.task_id, 'task_index': task_index, 'teacher': teacher_index, 'candidate_id': candidate_id, 'density_stratum': strata[candidate_id], 'active_edges': int(mask.sum()), 'source_query_nmse': score, 'artifact_path': str(card_path.resolve()) if card_path is not None else None, 'artifact_storage': 'file' if card_path is not None else 'memory', 'card_schema': 'generator_evaluator.functional_map_card:v1' if card_path is not None else None}}
        if persist_artifacts:
            teacher_state['optimizer_state'] = deepcopy(result['optimizer_state'])
            teacher_state['history'] = deepcopy(result['history'])
        states.append(teacher_state)
    selected_masks_tensor = torch.stack(selected_masks)
    selected_masks.clear()
    reference_q_abs = states[0]['q_abs_mean']
    aligned_q_abs = torch.stack([reference_q_abs] + [_align_hidden_columns(reference_q_abs, row['q_abs_mean']) for row in states[1:]])
    mean_q_abs = aligned_q_abs.mean(0)
    baseline = _exact_topk(mean_q_abs, k)
    density_counts = {int(density): int((selected_masks_tensor.sum((1, 2)) == density).sum()) for density in buckets}
    selected_ids = [int(row[1]) for row in selected]
    selected_scores = {str(row[1]): row[0] for row in selected}
    selected_initialization_seeds = [_candidate_initialization_seed(seed, candidate_id) for candidate_id in selected_ids]
    best_by_density.clear()
    selected.clear()
    del masks, mask, result, record
    bank_tokens = torch.stack(tokens)[None]
    tokens.clear()
    del token
    partitions = {'bank_support_ids': support_ids.reshape(-1).unique().tolist(), 'bank_query_ids': query_ids.reshape(-1).unique().tolist(), 'probe_ids': probe_ids.detach().cpu().tolist()}
    provenance = {'family': 'cooperative_deepsets', 'domain': 'deepsets', 'pattern': str(task_index), 'task_id': task.task_id, 'seed': int(seed), 'baseline_k': int(k), 'probe_ids': partitions['probe_ids'], 'probe_fingerprint': tensor_hash(probe_x), 'partitions': partitions, 'bank_task_id': source_task.task_id, 'teacher_density_counts': density_counts, 'candidate_count': int(bank_candidates), 'candidate_density_counts': {str(density): int(candidate_counts[index]) for index, density in enumerate(buckets)}, 'selected_candidate_ids': selected_ids, 'selected_initialization_seeds': selected_initialization_seeds, 'candidate_seed_rule': 'seed + 1000003 * (candidate_id + 1)', 'selected_candidate_query_nmse': selected_scores, 'selection_density_buckets': list(buckets), 'selection_density_quotas': quotas, 'baseline_alignment': 'label-free Hungarian cosine alignment of q_abs to teacher 0', 'selection_rule': 'lowest fixed-horizon source-query NMSE per density; candidate ID breaks ties', 'quality_source': None, 'bank_support_count': support_count, 'bank_query_count': query_count, 'accepted_feedback_task_ids': [task.task_id], 'feedback_hashes': [], 'persist_artifacts': bool(persist_artifacts), 'artifact_storage': 'file' if persist_artifacts else 'memory'}
    bank = FunctionalBank(bank_tokens, None, selected_masks_tensor, baseline, provenance, states=states, diagnostics={'probe_x': probe_x.detach().cpu().clone(), 'aligned_q_abs': aligned_q_abs, 'bank_support_ids': support_ids.detach().cpu().clone(), 'bank_query_ids': query_ids.detach().cpu().clone(), 'feedback_rows_added': 0})
    return bank

def load_source_pools(data_root, seed, per_digit_train, per_digit_query):
    images = np.load(data_root / "images.npy", mmap_mode="r")
    labels = np.load(data_root / "labels.npy", mmap_mode="r")
    if images.ndim != 2 or images.shape[1] != 784 or labels.shape != (len(images),):
        raise ValueError("expected images.npy [N,784] and labels.npy [N]")
    if images.dtype != np.uint8 or not np.isin(labels, np.arange(10)).all():
        raise ValueError("images must be uint8 pixels and labels must be digit IDs 0..9")
    if len(images) < 8:
        raise ValueError("need at least eight rows for MNIST8m blocks")
    used_rows, pixel_hashes = set(), set()
    support, _ = _read_split(images, labels, block_id=0, per_digit=per_digit_train,
        seed=seed, part=0, device=torch.device("cpu"),
        used_rows=used_rows, used_pixel_hashes=pixel_hashes)
    query, _ = _read_split(images, labels, block_id=1, per_digit=per_digit_query,
        seed=seed, part=0, device=torch.device("cpu"),
        used_rows=used_rows, used_pixel_hashes=pixel_hashes)
    return support, query


def main():
    parser = base_parser("Collect raw-pixel DeepSets functional maps")
    parser.set_defaults(steps=4000, candidates=4096, teachers=1024)
    parser.add_argument("--data-root", type=Path, required=True, help="MNIST8m directory with images.npy and labels.npy")
    parser.add_argument("--task-count", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--k", type=int, default=15053, help="baseline-mask active edges")
    parser.add_argument("--probe-count", type=int, default=32)
    parser.add_argument("--per-digit-train", type=int, default=1000)
    parser.add_argument("--per-digit-query", type=int, default=300)
    parser.add_argument("--support-count", type=int, help="sets per candidate; defaults to support image count /5")
    parser.add_argument("--query-count", type=int, help="sets per candidate; defaults to query image count /5")
    args = parser.parse_args()
    devices = preflight(args)
    if min(args.task_count, args.hidden, args.per_digit_train, args.per_digit_query) < 1:
        raise ValueError("task-count, hidden and per-digit counts must be positive")
    if args.k > 784 * args.hidden:
        raise ValueError("k exceeds the number of connections")
    if args.support_count is not None and args.support_count < 1:
        raise ValueError("support-count must be positive")
    if args.query_count is not None and args.query_count < 1:
        raise ValueError("query-count must be positive")
    support, query = load_source_pools(args.data_root, args.seed,
                                     args.per_digit_train, args.per_digit_query)
    support_pools = _partition_split(support, _stratum_counts(len(support.features), args.task_count), args.seed+31123)
    query_pools = _partition_split(query, _stratum_counts(len(query.features), args.task_count), args.seed+31127)
    if any(len(pool.features) <= args.probe_count+1 for pool in support_pools):
        raise ValueError("source training pool is too small for reserved probes")
    if any(len(pool.features) < 5 for pool in query_pools):
        raise ValueError("source query pool needs at least five images per task")
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out / "run.json", {"domain": "deepsets", "seed": args.seed,
              "data_root": str(args.data_root.resolve()), "task_count": args.task_count,
              "steps": args.steps, "candidates": args.candidates, "teachers": args.teachers,
              "devices": list(devices), "status": "running"})
    for index, (pool, query_pool) in enumerate(zip(support_pools, query_pools)):
        probe_rows = torch.randperm(len(pool.features), generator=torch.Generator().manual_seed(args.seed+60001+index))[:args.probe_count]
        remaining = torch.ones(len(pool.features), dtype=torch.bool)
        remaining[probe_rows] = False
        bank_support = _split_rows(pool, torch.nonzero(remaining).flatten())
        cost = _fixture_cost_vector(args.seed, "train", index)
        identity = SimpleNamespace(task_id=f"deepsets:{index}",
            provenance={"family": "deepsets", "task_index": index, "costs": cost.tolist()})
        bank = _build_bank(index, identity, bank_support, query_pool,
            pool.features[probe_rows], pool.source_ids[probe_rows],
            seed=args.seed+70003*(index+1), bank_steps=args.steps,
            bank_candidates=args.candidates, teachers_per_task=args.teachers,
            support_count=args.support_count or max(1, len(bank_support.features)//5),
            query_count=args.query_count or len(query_pool.features)//5,
            teacher_batch_size=args.batch_size, k=args.k, hidden=args.hidden,
            device=devices[0], measurement_devices=devices, bank_out=args.out / str(index))
        count = save_bank(bank, args.out / str(index), str(index), extract_deepsets_functional_token)
        print(f"DeepSets task {index}: saved {count} functional maps", flush=True)
        del bank
    save_json(args.out / "COMPLETE.json", {"domain": "deepsets", "tasks": args.task_count,
                                           "maps": args.task_count * args.teachers})


if __name__ == "__main__":
    main()
