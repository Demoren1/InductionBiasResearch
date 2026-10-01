"""Data and training primitives for the raw-pixel DeepSets VAAE pilot.

The bank uses labels of complete five-image sets only.  The target pools are
kept out of :func:`build_bank`, so downstream transfer cannot accidentally use
target labels while discovering a mask.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import torch
from torch import Tensor, nn


INPUT_DIM = 784


class Split(NamedTuple):
    """A disjoint image pool; ``source_ids`` are original MNIST8m row ids."""

    features: Tensor
    digits: Tensor
    source_ids: Tensor


def _hash_ids(ids: Tensor) -> str:
    return hashlib.sha256(ids.detach().cpu().numpy().astype("<i8", copy=False).tobytes()).hexdigest()


def _read_split(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    block_id: int,
    per_digit: int,
    seed: int,
    part: int,
    device: torch.device,
    used_rows: set[int],
    used_pixel_hashes: set[bytes],
) -> tuple[Split, int]:
    """Choose a reproducible per-digit partition from one MNIST8m eighth."""
    block = len(labels) // 8
    lo, hi = block_id * block, (block_id + 1) * block
    local = np.asarray(labels[lo:hi])
    rng = np.random.default_rng(seed + 1009 * block_id)
    selected: list[np.ndarray] = []
    duplicate_rows_skipped = 0
    # ``part`` uses a separate contiguous draw for each digit.  This makes
    # source/target partitions disjoint even when their order is shuffled.
    for digit in range(10):
        candidates = np.flatnonzero(local == digit)
        need = (part + 1) * per_digit
        if len(candidates) < need:
            raise ValueError(f"block {block_id} lacks images of digit {digit}")
        # Use a full random ordering: augmented MNIST8m may have identical
        # pixel rows.  Keep the first unseen raw image hashes globally.  This
        # guarantees pixel-disjoint pools, though it cannot establish that two
        # different augmentations came from different handwritten originals.
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
            raise ValueError(f"could not form pixel-unique split in block {block_id}, digit {digit}")
        selected.append(np.asarray(chosen, dtype=np.int64))
    ids = np.concatenate(selected)
    # A part-specific shuffle preserves the paired partition while avoiding
    # per-digit ordering in later set sampling.
    np.random.default_rng(seed + 7919 * (block_id + 1) + part).shuffle(ids)
    pixels = torch.from_numpy(np.asarray(images[ids], dtype=np.uint8)).to(device)
    digits = torch.from_numpy(np.asarray(labels[ids], dtype=np.int64)).to(device)
    original_ids = torch.from_numpy(ids.astype(np.int64, copy=False)).to(device)
    return Split(pixels.float().div_(255.0), digits, original_ids), duplicate_rows_skipped


def load_data(
    data_dir: str | Path,
    seed: int,
    device: str | torch.device,
    per_digit_train: int = 1000,
    per_digit_validation: int = 300,
    per_digit_test: int = 300,
) -> dict:
    """Load five mutually disjoint pools from the first three MNIST8m blocks.

    Block 0 is partitioned into source and target training pools.  Block 1 is
    partitioned in the same way for source and target validation.  Target test
    is independently selected from block 2.  Each returned split has exactly
    the requested number of images *per digit*.
    """
    if min(per_digit_train, per_digit_validation, per_digit_test) < 1:
        raise ValueError("all per-digit counts must be positive")
    data_dir = Path(data_dir)
    images = np.load(data_dir / "images.npy", mmap_mode="r")
    labels = np.load(data_dir / "labels.npy", mmap_mode="r")
    if images.ndim != 2 or images.shape[1] != INPUT_DIM or labels.shape != (len(images),):
        raise ValueError("expected images.npy [N, 784] and matching labels.npy [N]")
    # Retain the released MNIST8m block convention when the full corpus is
    # supplied, but permit compact fixtures in the smoke test.
    if len(labels) < 8:
        raise ValueError("need at least eight rows to form MNIST8m blocks")
    target_device = torch.device(device)
    used_rows: set[int] = set()
    used_pixel_hashes: set[bytes] = set()
    result: dict[str, Split | dict | int] = {}
    duplicate_counts: dict[str, int] = {}
    specs = (("source_train", 0, per_digit_train, 0), ("target_train", 0, per_digit_train, 1),
             ("source_validation", 1, per_digit_validation, 0),
             ("target_validation", 1, per_digit_validation, 1), ("target_test", 2, per_digit_test, 0))
    for name, block_id, count, part in specs:
        split, skipped = _read_split(images, labels, block_id=block_id, per_digit=count, seed=seed,
                                     part=part, device=target_device, used_rows=used_rows,
                                     used_pixel_hashes=used_pixel_hashes)
        result[name], duplicate_counts[name] = split, skipped
    split_names = tuple(name for name, *_ in specs)
    for left_index, left_name in enumerate(split_names):
        left = result[left_name]
        assert isinstance(left, Split)
        for right_name in split_names[left_index + 1:]:
            right = result[right_name]
            assert isinstance(right, Split)
            if torch.isin(left.source_ids, right.source_ids).any():
                raise AssertionError(f"row-id overlap between {left_name} and {right_name}")
    result["split_hashes"] = {name: _hash_ids(value.source_ids) for name, value in result.items()
                              if isinstance(value, Split)}
    result["exact_pixel_duplicates_excluded"] = duplicate_counts
    result["row_ids_pairwise_disjoint"] = True
    return result


def centred_costs(costs: Tensor | Sequence[float], device: torch.device) -> Tensor:
    """Return a 10-vector with zero mean and unit population standard deviation."""
    value = torch.as_tensor(costs, dtype=torch.float32, device=device).flatten()
    if value.numel() != 10:
        raise ValueError("each digit-cost task must contain exactly ten values")
    value = value - value.mean()
    scale = value.std(unbiased=False)
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("digit costs must not all be equal")
    return value / scale


def _sets(split: Split, costs: Tensor, n_sets: int, set_size: int,
          generator: torch.Generator) -> tuple[Tensor, Tensor]:
    if n_sets < 1 or set_size < 1:
        raise ValueError("n_sets and set_size must be positive")
    ids = torch.randint(len(split.features), (n_sets, set_size), device=split.features.device,
                        generator=generator)
    x = split.features[ids]
    y = costs[split.digits[ids]].sum(dim=1)
    return x, y


class MaskedDeepSets(nn.Module):
    """A vectorized batch of raw-pixel DeepSets regressors.

    Parameters have leading dimension ``models``.  A single model computes
    ``sum_i a^T tanh(x_i @ (W * M) + b) + c``.
    """

    def __init__(self, masks: Tensor, *, seed: int,
                 initialization_reference_models: int | None = None) -> None:
        super().__init__()
        if masks.ndim != 3 or masks.shape[1] != INPUT_DIM:
            raise ValueError("masks must have shape [models, 784, hidden]")
        self.models, _, self.hidden = masks.shape
        self.register_buffer("masks", masks.float())
        gen = torch.Generator(device=masks.device).manual_seed(seed)
        # Xavier scale for 784->H.  The same initializer is used in target
        # comparisons before the fixed masks are applied.
        bound = math.sqrt(6.0 / (INPUT_DIM + self.hidden))
        if initialization_reference_models is None:
            self.weight = nn.Parameter(torch.empty_like(masks).uniform_(-bound, bound, generator=gen))
            reference_readout = None
        else:
            if initialization_reference_models < 1:
                raise ValueError("initialization_reference_models must be positive")
            # Fix the RNG draw sizes when new methods are added. The original
            # pilot had 20 models; its readout draw follows all 20 weight draws.
            reference_weight = torch.empty(initialization_reference_models, INPUT_DIM,
                                           self.hidden, device=masks.device).uniform_(
                                               -bound, bound, generator=gen)
            reference_readout = torch.empty(initialization_reference_models, self.hidden,
                                            device=masks.device).uniform_(
                                                -1.0 / math.sqrt(self.hidden),
                                                1.0 / math.sqrt(self.hidden), generator=gen)
            reference_indices = torch.arange(self.models, device=masks.device) % initialization_reference_models
            self.weight = nn.Parameter(reference_weight[reference_indices].clone())
        self.bias = nn.Parameter(torch.zeros(self.models, self.hidden, device=masks.device))
        if reference_readout is None:
            self.readout = nn.Parameter(torch.empty(self.models, self.hidden, device=masks.device)
                                       .uniform_(-1.0 / math.sqrt(self.hidden),
                                                1.0 / math.sqrt(self.hidden), generator=gen))
        else:
            self.readout = nn.Parameter(reference_readout[reference_indices].clone())
        self.per_image_offset = nn.Parameter(torch.zeros(self.models, device=masks.device))

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, set_size, 784], output: [models, batch]
        hidden = torch.tanh(torch.einsum("bsi,mih->mbsh", x, self.weight * self.masks)
                            + self.bias[:, None, None, :])
        per_image = (hidden * self.readout[:, None, None, :]).sum(dim=-1)
        per_image = per_image + self.per_image_offset[:, None, None]
        return per_image.sum(dim=-1)

    def importance(self) -> Tensor:
        values = (self.weight.detach().abs() * self.masks).flatten(1)
        return (values / values.amax(dim=1, keepdim=True).clamp_min(1e-12)).reshape_as(self.masks)


def _exact_random_masks(models: int, hidden: int, density: float, *,
                        device: torch.device, generator: torch.Generator) -> Tensor:
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    edges = int(round(density * INPUT_DIM * hidden))
    edges = min(max(edges, 1), INPUT_DIM * hidden)
    scores = torch.rand(models, INPUT_DIM * hidden, device=device, generator=generator)
    masks = torch.zeros_like(scores)
    masks.scatter_(1, scores.topk(edges, dim=1).indices, 1.0)
    return masks.reshape(models, INPUT_DIM, hidden)


@torch.no_grad()
def _losses(model: MaskedDeepSets, x: Tensor, y: Tensor, set_size: int) -> Tensor:
    return (model(x).sub(y[None]).square().mean(dim=1) / set_size)


def build_bank(
    data: dict,
    costs: Tensor | Sequence[float],
    seed: int,
    device: str | torch.device,
    hidden: int = 32,
    density: float = 0.2,
    candidates: int = 128,
    keep: int = 32,
    steps: int = 800,
    batch_size: int = 32,
    set_size: int = 5,
) -> dict:
    """Train sparse source-task candidates and retain their validation winners.

    Candidates are independent models sharing only a vectorized execution.  A
    fixed, disjoint source-validation collection selects each candidate's best
    checkpoint; then the ``keep`` lowest validation losses form the bank.
    """
    if min(hidden, candidates, keep, steps, batch_size, set_size) < 1 or keep > candidates:
        raise ValueError("invalid positive bank sizes")
    target_device = torch.device(device)
    source_train, source_validation = data["source_train"], data["source_validation"]
    if not isinstance(source_train, Split):
        source_train = Split(*source_train)
    if not isinstance(source_validation, Split):
        source_validation = Split(*source_validation)
    if source_train.features.device != target_device:
        source_train = Split(*(value.to(target_device) for value in source_train))
        source_validation = Split(*(value.to(target_device) for value in source_validation))
    task_costs = centred_costs(costs, target_device)
    mask_generator = torch.Generator(device=target_device).manual_seed(seed + 31)
    data_generator = torch.Generator(device=target_device).manual_seed(seed + 32)
    masks = _exact_random_masks(candidates, hidden, density, device=target_device, generator=mask_generator)
    model = MaskedDeepSets(masks, seed=seed + 101).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    val_x, val_y = _sets(source_validation, task_costs, max(128, batch_size * 4), set_size, data_generator)
    best_loss = torch.full((candidates,), float("inf"), device=target_device)
    best_state = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    curves: list[dict] = []
    check_every = max(1, min(50, steps // 8))
    for step in range(1, steps + 1):
        model.train()
        x, y = _sets(source_train, task_costs, batch_size, set_size, data_generator)
        prediction = model(x)
        per_model = prediction.sub(y[None]).square().mean(dim=1) / set_size
        loss = per_model.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % check_every == 0 or step == steps:
            model.eval()
            validation = _losses(model, val_x, val_y, set_size)
            improved = validation < best_loss
            for name, parameter in model.named_parameters():
                best_state[name][improved] = parameter.detach()[improved]
            best_loss = torch.minimum(best_loss, validation)
            curves.append({"step": step, "train_normalized_mse": float(loss.item()),
                           "validation_mean_normalized_mse": float(validation.mean().item())})
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(best_state[name])
    chosen = best_loss.argsort()[:keep]
    maps = model.importance()[chosen].detach().cpu()
    result = {
        "maps": maps,
        "masks": masks[chosen].detach().cpu().bool(),
        "weights": best_state["weight"][chosen].detach().cpu(),
        "validation_losses": best_loss[chosen].detach().cpu().tolist(),
        "selected_candidates": chosen.detach().cpu().tolist(),
        "training_curves": curves,
        "best_validation_normalized_mse": best_loss.detach().cpu().tolist(),
        "edges_per_mask": int(masks[0].sum().item()),
        "source_split_hashes": {name: data.get("split_hashes", {}).get(name)
                                for name in ("source_train", "source_validation")},
    }
    return result


def _mask_items(masks: dict[str, Tensor], device: torch.device) -> tuple[list[str], Tensor, list[int]]:
    names, values, replica_counts = [], [], []
    hidden: int | None = None
    for name, value in masks.items():
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tensor.ndim != 3 or tensor.shape[1] != INPUT_DIM:
            raise ValueError(f"mask {name!r} must have shape [R, 784, H]")
        if hidden is None:
            hidden = tensor.shape[2]
        elif hidden != tensor.shape[2]:
            raise ValueError("all methods must use the same hidden width")
        names.extend([name] * len(tensor))
        replica_counts.append(len(tensor))
        values.append(tensor)
    if not values:
        raise ValueError("at least one mask method is required")
    return names, torch.cat(values), replica_counts


def _make_target_sets(split: Split, costs: Tensor, count: int, set_size: int,
                      generator: torch.Generator) -> tuple[Tensor, Tensor]:
    return _sets(split, costs, count, set_size, generator)


def evaluate_masks(
    data: dict,
    costs: Tensor | Sequence[Sequence[float]],
    masks: dict[str, Tensor],
    seed: int,
    device: str | torch.device,
    support_sizes: Sequence[int] = (32, 64, 128, 256),
    steps: int = 800,
    batch_size: int = 32,
    set_size: int = 5,
    validation_sets: int = 128,
    test_sets: int = 512,
    artifact_dir: str | Path | None = None,
    initialization_reference_models: int | None = None,
) -> list[dict]:
    """Fit every fixed mask on paired target supports and evaluate untouched tests.

    ``support_sizes`` are total labelled target-set budgets.  For budget ``N``,
    a nested target-training prefix of ``N - n_valid`` sets is fitted and a
    disjoint target-validation prefix of ``n_valid`` sets selects a checkpoint,
    where ``n_valid=min(validation_sets, max(1, round(.2*N)))``.  Thus the
    checkpoint split does not quietly add labels to a small-support condition.
    Within a fit, all methods and replicas receive identical minibatch indices
    and the same unmasked numerical initialization.  Test data are touched
    only after checkpoint selection.  If ``artifact_dir`` is supplied, the
    restored best checkpoint for every task/budget is saved as exactly one
    ``target_task{task}_budget{budget}.pt`` artifact.  This is intentionally
    optional so the original evaluation path has identical numerical work.
    """
    if not support_sizes or min(support_sizes) < 2 or min(steps, batch_size, set_size,
                                                           validation_sets, test_sets) < 1:
        raise ValueError("support sizes must be at least two and other sizes positive")
    target_device = torch.device(device)
    output_dir = None if artifact_dir is None else Path(artifact_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(costs, Tensor):
        task_values = costs.to(device=target_device, dtype=torch.float32)
    else:
        # ``torch.as_tensor`` cannot convert a list of CUDA/CPU tensors; this
        # convenience path is useful for callers constructing task vectors in
        # a loop.
        try:
            task_values = torch.as_tensor(costs, dtype=torch.float32, device=target_device)
        except (TypeError, ValueError):
            task_values = torch.stack([torch.as_tensor(row, dtype=torch.float32,
                                                        device=target_device) for row in costs])
    if task_values.ndim == 1:
        task_values = task_values[None]
    if task_values.ndim != 2 or task_values.shape[1] != 10:
        raise ValueError("costs must be [tasks, 10]")
    task_values = torch.stack([centred_costs(row, target_device) for row in task_values])
    method_names, stacked_masks, _ = _mask_items(masks, target_device)
    target_train, target_validation, target_test = (data[key] for key in
                                                     ("target_train", "target_validation", "target_test"))
    target_train = target_train if isinstance(target_train, Split) else Split(*target_train)
    target_validation = target_validation if isinstance(target_validation, Split) else Split(*target_validation)
    target_test = target_test if isinstance(target_test, Split) else Split(*target_test)
    target_train = Split(*(v.to(target_device) for v in target_train))
    target_validation = Split(*(v.to(target_device) for v in target_validation))
    target_test = Split(*(v.to(target_device) for v in target_test))
    largest_support = max(support_sizes)
    validation_by_budget = {int(total): min(validation_sets, max(1, round(.2 * total)))
                            for total in support_sizes}
    train_by_budget = {total: total - validation_by_budget[int(total)] for total in support_sizes}
    records: list[dict] = []
    check_every = max(1, min(50, steps // 8))
    for task_index, raw_costs in enumerate(task_values):
        data_generator = torch.Generator(device=target_device).manual_seed(seed + 10_007 * (task_index + 1))
        support_x, support_y = _make_target_sets(target_train, raw_costs, largest_support, set_size, data_generator)
        max_validation = max(validation_by_budget.values())
        validation_x, validation_y = _make_target_sets(target_validation, raw_costs, max_validation, set_size, data_generator)
        test_x, test_y = _make_target_sets(target_test, raw_costs, test_sets, set_size, data_generator)
        for support_size in support_sizes:
            train_sets = train_by_budget[int(support_size)]
            valid_sets = validation_by_budget[int(support_size)]
            print(f"deepsets target task={task_index} support={support_size} (train={train_sets}, val={valid_sets}): fitting {len(stacked_masks)} masks", flush=True)
            batch_generator = torch.Generator(device=target_device).manual_seed(
                seed + 70_001 * (task_index + 1) + support_size)
            # All masks start from exactly the same numerical weights for a
            # replica.  Masked entries cannot affect a prediction or gradient.
            model = MaskedDeepSets(stacked_masks, seed=seed + 3_001 * task_index + support_size,
                                  initialization_reference_models=initialization_reference_models).to(target_device)
            # A replica's numerical start is shared across every method.  For
            # example, ``agreement[2]`` and ``random[2]`` begin identically;
            # only their fixed connectivity differs.
            replica_for_model: list[int] = []
            seen_for_method: dict[str, int] = {}
            for method in method_names:
                replica_for_model.append(seen_for_method.get(method, 0))
                seen_for_method[method] = seen_for_method.get(method, 0) + 1
            exemplar: dict[int, int] = {}
            for index, replica in enumerate(replica_for_model):
                exemplar.setdefault(replica, index)
            with torch.no_grad():
                source_indices = torch.tensor([exemplar[replica] for replica in replica_for_model],
                                              device=target_device)
                for parameter in model.parameters():
                    parameter.copy_(parameter.detach()[source_indices])
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
            best_val = torch.full((len(stacked_masks),), float("inf"), device=target_device)
            best_state = {name: p.detach().clone() for name, p in model.named_parameters()}
            best_train = torch.full((len(stacked_masks),), float("nan"), device=target_device)
            best_step = torch.zeros(len(stacked_masks), dtype=torch.long, device=target_device)
            initial_train = initial_val = final_train = final_val = None
            for step in range(1, steps + 1):
                ids = torch.randint(train_sets, (batch_size,), device=target_device, generator=batch_generator)
                prediction = model(support_x[ids])
                loss_per_model = prediction.sub(support_y[ids][None]).square().mean(dim=1) / set_size
                optimizer.zero_grad(set_to_none=True)
                if initialization_reference_models is None:
                    loss_per_model.mean().backward()
                else:
                    # Keep each independent model's gradient scale (and Adam
                    # epsilon effect) identical to the original 20-model fit.
                    (loss_per_model.sum() / initialization_reference_models).backward()
                optimizer.step()
                if step == 1 or step % check_every == 0 or step == steps:
                    model.eval()
                    with torch.no_grad():
                        train_loss = _losses(model, support_x[:train_sets], support_y[:train_sets], set_size)
                        validation_loss = _losses(model, validation_x[:valid_sets], validation_y[:valid_sets], set_size)
                    if initial_train is None:
                        initial_train, initial_val = train_loss.detach().clone(), validation_loss.detach().clone()
                    improved = validation_loss < best_val
                    for name, parameter in model.named_parameters():
                        best_state[name][improved] = parameter.detach()[improved]
                    best_train[improved] = train_loss[improved]
                    best_step[improved] = step
                    best_val = torch.minimum(best_val, validation_loss)
                    final_train, final_val = train_loss, validation_loss
                    model.train()
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(best_state[name])
                test_prediction = model(test_x)
                test_mse = test_prediction.sub(test_y[None]).square().mean(dim=1) / set_size
                test_mae = test_prediction.sub(test_y[None]).abs().mean(dim=1)
            per_method_seen: dict[str, int] = {}
            checkpoint_records: list[dict] = []
            for index, method in enumerate(method_names):
                init = per_method_seen.get(method, 0)
                per_method_seen[method] = init + 1
                record = {
                    "task": int(task_index), "support_size": int(support_size), "method": method, "init": init,
                    "train_sets": int(train_sets), "validation_sets": int(valid_sets),
                    "total_labeled_sets": int(support_size),
                    "mse": float(test_mse[index].item()), "mae": float(test_mae[index].item()),
                    "best_step": int(best_step[index].item()),
                    "validation_mse": float(best_val[index].item()),
                    "best_train_mse": float(best_train[index].item()),
                    "final_train_mse": float(final_train[index].item()),
                    "final_validation_mse": float(final_val[index].item()),
                    "early_train_normalized_mse": float(initial_train[index].item()),
                    "early_validation_normalized_mse": float(initial_val[index].item()),
                    "best_validation_normalized_mse": float(best_val[index].item()),
                }
                records.append(record)
                checkpoint_records.append(record)
            if output_dir is not None:
                # Save after copying the selected checkpoint back into the
                # model: the tensors therefore reproduce test evaluation,
                # rather than the final optimizer step.
                state = {name: value.detach().cpu().clone()
                         for name, value in model.state_dict().items()}
                torch.save({
                    "schema": "deepsets_vaae.target_checkpoint.v1",
                    "task": int(task_index),
                    "support_size": int(support_size),
                    "set_size": int(set_size),
                    "method_names": [record["method"] for record in checkpoint_records],
                    "replica_indices": [record["init"] for record in checkpoint_records],
                    "methods": [{"model_index": index, "method": record["method"],
                                 "init": record["init"]}
                                for index, record in enumerate(checkpoint_records)],
                    "records": checkpoint_records,
                    "state_dict": state,
                    "masks": state["masks"],
                    "weight": state["weight"],
                    "effective_weight": state["weight"] * state["masks"],
                    "bias": state["bias"],
                    "readout": state["readout"],
                    "per_image_offset": state["per_image_offset"],
                    "offset": state["per_image_offset"],
                }, output_dir / f"target_task{task_index}_budget{support_size}.pt")
    return records


@torch.no_grad()
def permutation_invariance_selfcheck(device: str | torch.device = "cpu") -> float:
    """Return the maximum numerical deviation after independently permuting sets."""
    dev = torch.device(device)
    mask = _exact_random_masks(3, 7, 0.2, device=dev,
                               generator=torch.Generator(device=dev).manual_seed(9))
    model = MaskedDeepSets(mask, seed=10).to(dev).eval()
    x = torch.randn(11, 5, INPUT_DIM, device=dev)
    order = torch.stack([torch.randperm(5, device=dev) for _ in range(len(x))])
    permuted = x[torch.arange(len(x), device=dev)[:, None], order]
    error = (model(x) - model(permuted)).abs().max().item()
    if error > 2e-5:
        raise AssertionError(f"DeepSets permutation check failed: {error}")
    return float(error)


def tiny_smoke_test() -> dict:
    """CPU-only check for decreasing bank loss, exact masks, and ID separation."""
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    generator = torch.Generator().manual_seed(44)
    def synthetic(offset: int) -> Split:
        digits = torch.arange(10).repeat_interleave(8)
        features = torch.zeros(len(digits), INPUT_DIM)
        features[torch.arange(len(digits)), digits] = 1.0
        features += torch.randn(features.shape, generator=generator) * 0.01
        return Split(features, digits, torch.arange(offset, offset + len(digits)))
    data = {"source_train": synthetic(0), "target_train": synthetic(100),
            "source_validation": synthetic(200), "target_validation": synthetic(300),
            "target_test": synthetic(400)}
    bank = build_bank(data, torch.arange(10), seed=2, device="cpu", hidden=4, density=0.1,
                      candidates=4, keep=2, steps=30, batch_size=12, set_size=3)
    all_ids = [data[key].source_ids for key in data]
    disjoint = all(not bool(torch.isin(left, right).any())
                   for i, left in enumerate(all_ids) for right in all_ids[i + 1:])
    decreasing = (bank["training_curves"][-1]["validation_mean_normalized_mse"]
                  < bank["training_curves"][0]["validation_mean_normalized_mse"])
    if not disjoint or not decreasing or not all(int(mask.sum()) == bank["edges_per_mask"] for mask in bank["masks"]):
        raise AssertionError("tiny smoke test failed")
    return {"loss_decreased": decreasing, "disjoint_ids": disjoint,
            "edges_per_mask": bank["edges_per_mask"], "permutation_error": permutation_invariance_selfcheck()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the lightweight DeepSets VAAE core smoke test.")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("pass --smoke")
    print(tiny_smoke_test())
