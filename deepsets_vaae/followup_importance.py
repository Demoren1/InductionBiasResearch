"""Direction-4 importance representations and source-function ablations.

For each selected source-bank model, this script compares raw ``|W|*M``,
activity-weighted ``|W|*M*std(x)``, a function-gradient saliency, and signed
single-edge deletion changes in source-training loss. Source-validation
deletion effects provide a held-out check. Target labels are only touched by
the shared downstream evaluator after masks have been extracted.

Run one seed per worker (one worker per visible CUDA device):

    python -m deepsets_vaae.followup_importance --seed 4100 --out ...

After all eight workers finish, aggregate figures and the Russian report with
``--report --out <importance-root>``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr, t as student_t

from . import core, masks as mask_ops
from .followup_common import (
    FOLLOWUP, PILOT, configure, evaluate_followup, load_pilot_seed,
    save_provenance,
)
from .run import task_vectors, write_json


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = FOLLOWUP / "importance"
CHECKPOINT_ROOT = FOLLOWUP / "source_checkpoints"
FEATURES = 784
HIDDEN = 32
EDGES = int(round(.2 * FEATURES * HIDDEN))
SET_SIZE = 5
ANCHOR_SETS = 128
FUNCTION_CALIBRATION_IMAGES = 4096
FUNCTION_IMAGE_BATCH = 256
FUNCTION_EDGE_CHUNK = 512
DELTA_EDGE_CHUNK = 256
PRUNE_FRACTION = .02

REP_NAMES = (
    "raw_abs_weight",
    "activity_weighted",
    "function_gradient",
    "train_deletion_positive",
)


def _write_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _progress(out: Path, seed: int, started: float, stage: str, **details: Any) -> None:
    payload = {"seed": seed, "stage": stage,
               "elapsed_seconds": time.monotonic() - started, **_jsonable(details)}
    write_json(out / "status.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _wait_checkpoint(seed: int, out: Path, started: float, timeout: float) -> Path:
    folder = CHECKPOINT_ROOT / f"seed_{seed}"
    marker = folder / "COMPLETE"
    deadline = time.monotonic() + timeout
    while not marker.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"source replay checkpoint not ready after {timeout}s: {folder}")
        _progress(out, seed, started, "waiting_for_source_replay", checkpoint_dir=folder,
                  marker_exists=False, poll_seconds=20)
        time.sleep(20)
    return folder


def _normalize_nonnegative(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError("importance tensors must be [models, features, hidden]")
    value = value.clamp_min(0.)
    flat = value.flatten(1)
    maximum = flat.amax(dim=1, keepdim=True)
    return (flat / maximum.clamp_min(1e-12)).reshape_as(value)


def _select_state(checkpoint: dict, expected_bank: dict, task: int) -> dict[str, torch.Tensor]:
    if "state_dict" not in checkpoint:
        raise ValueError(f"source checkpoint bank_{task} lacks state_dict")
    state = checkpoint["state_dict"]
    required = ("weight", "masks", "bias", "readout", "per_image_offset")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"source checkpoint bank_{task} missing fields {missing}")
    values = {key: torch.as_tensor(state[key]).detach().float() for key in required}
    if tuple(values["weight"].shape) != (32, 784, 32):
        raise ValueError(f"unexpected checkpoint W shape: {values['weight'].shape}")
    if not torch.equal(values["masks"].bool(), expected_bank["masks"].cpu().bool()):
        raise ValueError(f"source checkpoint bank_{task} mask differs from legacy bank")
    if not torch.allclose(values["weight"], expected_bank["weights"].cpu(), atol=0., rtol=0.):
        raise ValueError(f"source checkpoint bank_{task} W differs from legacy bank")
    return values


def _prediction(weight: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor,
                readout: torch.Tensor, offset: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    effective = weight * mask
    hidden = torch.tanh(torch.einsum("bsi,mih->mbsh", x, effective)
                        + bias[:, None, None, :])
    per_item = (hidden * readout[:, None, None, :]).sum(-1) + offset[:, None, None]
    return per_item.sum(-1)


def _active_edge_indices(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    models, features, hidden = mask.shape
    flat = mask.reshape(models, -1).bool()
    counts = flat.sum(dim=1)
    if not torch.all(counts == counts[0]):
        raise ValueError("selected source models must have a common edge count")
    indexes = flat.nonzero(as_tuple=False)[:, 1].reshape(models, -1)
    return indexes.div(hidden, rounding_mode="floor"), indexes.remainder(hidden)


@torch.no_grad()
def _edge_deletion_deltas(
    state: dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor,
    *, edge_chunk: int = DELTA_EDGE_CHUNK,
) -> torch.Tensor:
    """Exact signed loss change for deleting each active edge.

    Returns ``loss(delete edge) - loss(original)`` in the same normalized
    source-task MSE units as the bank objective. Negative values mean that
    deleting the edge improves loss. The algebra removes one input contribution
    from its tanh neuron and therefore covers the nonlinear change exactly.
    """
    weight, mask = state["weight"], state["masks"]
    bias, readout, offset = state["bias"], state["readout"], state["per_image_offset"]
    models, _, hidden_width = weight.shape
    effective = weight * mask
    hidden_pre = torch.einsum("bsi,mih->mbsh", x, effective) + bias[:, None, None, :]
    hidden_value = torch.tanh(hidden_pre)
    prediction = ((hidden_value * readout[:, None, None, :]).sum(-1)
                  + offset[:, None, None]).sum(-1)
    residual = prediction - y[None, :]
    base_squared = residual.square()
    pix, hid = _active_edge_indices(mask)
    edge_count = pix.shape[1]
    result = torch.zeros(models, FEATURES * hidden_width, device=x.device, dtype=x.dtype)
    batch, set_size, _ = x.shape
    for start in range(0, edge_count, edge_chunk):
        stop = min(start + edge_chunk, edge_count)
        pixels, neurons = pix[:, start:stop], hid[:, start:stop]
        edge_weights = effective.reshape(models, -1).gather(
            1, pixels * hidden_width + neurons)
        gather_index = pixels[:, None, None, :].expand(models, batch, set_size, -1)
        edge_x = x[None].expand(models, -1, -1, -1).gather(3, gather_index)
        neuron_index = neurons[:, None, None, :].expand(models, batch, set_size, -1)
        pre = hidden_pre.gather(3, neuron_index)
        removed_pre = pre - edge_x * edge_weights[:, None, None, :]
        edge_readout = readout.gather(1, neurons)
        delta_prediction = ((torch.tanh(removed_pre) - torch.tanh(pre))
                            * edge_readout[:, None, None, :]).sum(dim=2)
        delta_loss = ((residual[:, :, None] + delta_prediction).square()
                      - base_squared[:, :, None]).mean(dim=1) / set_size
        flat_index = pixels * hidden_width + neurons
        result.scatter_(1, flat_index, delta_loss)
    return result.reshape(models, FEATURES, hidden_width)


@torch.no_grad()
def _function_gradient_scores(state: dict[str, torch.Tensor], images: torch.Tensor,
                              *, image_batch: int = FUNCTION_IMAGE_BATCH,
                              edge_chunk: int = FUNCTION_EDGE_CHUNK) -> torch.Tensor:
    """Mean absolute edge score ``|w*x*a*(1-tanh(u)^2)|`` on train images."""
    weight, mask = state["weight"], state["masks"]
    bias, readout = state["bias"], state["readout"]
    models, _, hidden_width = weight.shape
    effective = weight * mask
    pix, hid = _active_edge_indices(mask)
    edge_count = pix.shape[1]
    total = torch.zeros(models, FEATURES * hidden_width, device=images.device,
                        dtype=images.dtype)
    for image_start in range(0, len(images), image_batch):
        x = images[image_start:image_start + image_batch]
        pre = torch.einsum("bf,mfh->mbh", x, effective) + bias[:, None, :]
        derivative = readout[:, None, :] * (1. - torch.tanh(pre).square())
        for start in range(0, edge_count, edge_chunk):
            stop = min(start + edge_chunk, edge_count)
            pixels, neurons = pix[:, start:stop], hid[:, start:stop]
            edge_weights = effective.reshape(models, -1).gather(1, pixels * hidden_width + neurons)
            image_index = pixels[:, None, :].expand(models, len(x), -1)
            edge_x = x[None].expand(models, -1, -1).gather(2, image_index)
            derivative_index = neurons[:, None, :].expand(models, len(x), -1)
            edge_derivative = derivative.gather(2, derivative_index)
            scores = (edge_x * edge_weights[:, None, :] * edge_derivative).abs().sum(dim=1)
            total.scatter_add_(1, pixels * hidden_width + neurons, scores)
    total.div_(len(images))
    return total.reshape(models, FEATURES, hidden_width)


@torch.no_grad()
def _brute_predictions(state: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    return _prediction(state["weight"], state["masks"], state["bias"],
                       state["readout"], state["per_image_offset"], x)


def _self_check(device: torch.device) -> dict[str, float]:
    """Run the algebra comparison in strict FP32, even after TF32 training."""
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        return _self_check_impl(device)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _self_check_impl(device: torch.device) -> dict[str, float]:
    """Compare edge-delta algebra with brute forward deletion on edge cases."""
    generator = torch.Generator(device=device).manual_seed(97121)
    models, features, hidden = 3, 7, 4
    weight = torch.randn(models, features, hidden, device=device, generator=generator)
    base_mask = (torch.rand(features, hidden, device=device, generator=generator) > .55).float()
    mask = base_mask[None].expand(models, -1, -1).clone()
    # Include a structurally zero feature and strongly saturated neurons.
    mask[:, 0, :] = 1.
    weight[:, 0, :] = 0.
    bias = torch.zeros(models, hidden, device=device)
    bias[0, 0] = 35.
    bias[1, 1] = -35.
    readout = torch.randn(models, hidden, device=device, generator=generator)
    offset = torch.randn(models, device=device, generator=generator)
    x = torch.randn(11, 5, features, device=device, generator=generator)
    x[:, :, 0] = 0.
    y = torch.randn(11, device=device, generator=generator)
    state = {"weight": weight, "masks": mask, "bias": bias,
             "readout": readout, "per_image_offset": offset}
    exact = _edge_deletion_deltas(state, x, y, edge_chunk=5)
    base = _brute_predictions(state, x)
    base_loss = (base - y[None]).square().mean(dim=1) / x.shape[1]
    brute = torch.zeros_like(exact)
    effective = weight * mask
    with torch.no_grad():
        for model in range(models):
            for feature, neuron in mask[model].nonzero(as_tuple=False).tolist():
                if effective[model, feature, neuron] == 0:
                    # Structural zero edge: physically deleting it changes no output.
                    brute[model, feature, neuron] = 0.
                    continue
                altered = effective.clone()
                altered[model, feature, neuron] = 0.
                changed_hidden = torch.tanh(
                    torch.einsum("bsi,mih->mbsh", x, altered)
                    + bias[:, None, None, :])
                changed = ((changed_hidden * readout[:, None, None, :]).sum(-1)
                           + offset[:, None, None]).sum(-1)
                loss = (changed[model] - y).square().mean() / x.shape[1]
                brute[model, feature, neuron] = loss - base_loss[model]
    max_error = float((exact - brute).abs().max().cpu())
    zero_pixel_error = float(exact[:, 0].abs().max().cpu())
    if max_error > 2e-5 or zero_pixel_error != 0.:
        raise AssertionError(f"edge deletion formula self-check failed: {max_error=}, {zero_pixel_error=}")
    return {"max_absolute_delta_error": max_error,
            "zero_feature_max_absolute_delta": zero_pixel_error,
            "saturated_neuron_biases": [35., -35.],
            "checked_models": models, "checked_active_edges": int(mask.sum().item())}


def _calibration_and_validation(data: dict, task_cost: torch.Tensor, bank_seed: int,
                                device: torch.device) -> tuple[torch.Tensor, ...]:
    source_train = data["source_train"]
    source_validation = data["source_validation"]
    if not isinstance(source_train, core.Split):
        source_train = core.Split(*source_train)
    if not isinstance(source_validation, core.Split):
        source_validation = core.Split(*source_validation)
    source_train = core.Split(*(item.to(device) for item in source_train))
    source_validation = core.Split(*(item.to(device) for item in source_validation))
    # The original bank's fixed source-validation anchor sets are generated by
    # the first call to _sets using generator bank_seed+32.
    validation_rng = torch.Generator(device=device).manual_seed(bank_seed + 32)
    val_x, val_y = core._sets(source_validation, task_cost, ANCHOR_SETS,
                              SET_SIZE, validation_rng)
    # A distinct generator creates fixed source-training calibration sets.
    # Calibration draws come only from source_train; source_validation labels
    # remain held out for ranking/pruning checks below.
    train_rng_seed = bank_seed + 500_013
    train_rng = torch.Generator(device=device).manual_seed(train_rng_seed)
    train_x, train_y = core._sets(source_train, task_cost, ANCHOR_SETS,
                                  SET_SIZE, train_rng)
    image_rng = torch.Generator(device=device).manual_seed(bank_seed + 501_019)
    image_count = min(FUNCTION_CALIBRATION_IMAGES, len(source_train.features))
    image_index = torch.randperm(len(source_train.features), generator=image_rng,
                                 device=device)[:image_count]
    function_images = source_train.features[image_index]
    pixel_std = source_train.features.std(dim=0, unbiased=False)
    return train_x, train_y, val_x, val_y, function_images, pixel_std, train_rng_seed


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    value = spearmanr(left, right).statistic
    return None if not np.isfinite(value) else float(value)


def _rank_relevance(state: dict[str, torch.Tensor], maps: dict[str, torch.Tensor],
                    train_delta: torch.Tensor, val_delta: torch.Tensor) -> dict[str, Any]:
    active = state["masks"].bool()
    result: dict[str, Any] = {}
    definitions = {**maps,
                   "train_deletion_signed": train_delta,
                   "train_deletion_positive": train_delta.clamp_min(0.)}
    for name, score in definitions.items():
        signed_corr, positive_corr = [], []
        for model in range(len(active)):
            mask = active[model]
            score_values = score[model][mask].detach().cpu().numpy()
            heldout_signed = val_delta[model][mask].detach().cpu().numpy()
            heldout_positive = np.maximum(heldout_signed, 0.)
            c_signed = _spearman(score_values, heldout_signed)
            c_positive = _spearman(score_values, heldout_positive)
            if c_signed is not None:
                signed_corr.append(c_signed)
            if c_positive is not None:
                positive_corr.append(c_positive)
        result[name] = {
            "spearman_vs_heldout_signed_delta_mean": float(np.mean(signed_corr)) if signed_corr else None,
            "spearman_vs_heldout_positive_delta_mean": float(np.mean(positive_corr)) if positive_corr else None,
            "n_models_signed": len(signed_corr), "n_models_positive": len(positive_corr),
        }
    return result


@torch.no_grad()
def _validation_pruning(state: dict[str, torch.Tensor], scores: dict[str, torch.Tensor],
                        x: torch.Tensor, y: torch.Tensor, seed: int, task: int,
                        *, fraction: float = PRUNE_FRACTION) -> dict[str, Any]:
    """Measure high/low/random simultaneous deletions on held-out source sets."""
    masks = state["masks"].bool()
    weight = state["weight"] * state["masks"]
    readout = state["readout"]
    baseline = _prediction(weight, state["masks"], state["bias"], readout,
                           state["per_image_offset"], x)
    baseline_loss = (baseline - y[None]).square().mean(dim=1) / x.shape[1]
    count = max(1, int(round(fraction * EDGES)))
    generator = torch.Generator(device=x.device).manual_seed(seed + 800_000 + task)
    names = list(scores)
    rows: dict[str, list[float]] = {f"{name}_{which}": []
                                   for name in names for which in ("high", "low")}
    rows["random"] = []
    for model in range(len(masks)):
        active = masks[model].flatten().nonzero(as_tuple=False).flatten()
        random_order = torch.randperm(len(active), generator=generator, device=x.device)
        selected_by_name: list[tuple[str, torch.Tensor]] = []
        for name, score in scores.items():
            active_scores = score[model].flatten()[active]
            high = active[active_scores.topk(count).indices]
            low = active[torch.topk(active_scores, count, largest=False).indices]
            selected_by_name.extend(((f"{name}_high", high), (f"{name}_low", low)))
        selected_by_name.append(("random", active[random_order[:count]]))
        selected = torch.zeros(len(selected_by_name), FEATURES * HIDDEN,
                               dtype=torch.bool, device=x.device)
        for row, (_, indices) in enumerate(selected_by_name):
            selected[row, indices] = True
        pruned_weight = weight[model].flatten()[None].expand(len(selected_by_name), -1).clone()
        pruned_weight.masked_fill_(selected, 0.)
        pruned_weight = pruned_weight.reshape(-1, FEATURES, HIDDEN)
        hidden = torch.tanh(torch.einsum("bsi,mih->mbsh", x, pruned_weight)
                            + state["bias"][model][None, None, None, :])
        prediction = ((hidden * readout[model][None, None, None, :]).sum(-1)
                      + state["per_image_offset"][model]).sum(-1)
        delta = ((prediction - y[None]).square().mean(dim=1) / x.shape[1]
                 - baseline_loss[model])
        for row, (name, _) in enumerate(selected_by_name):
            rows[name].append(float(delta[row].cpu()))
    return {name: {"mean_delta_nmse": float(np.mean(values)),
                   "std_across_models": float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
                   "n_models": len(values), "deltas": values}
            for name, values in rows.items()}


def _raw_alignment_orders(train_raw: list[torch.Tensor]) -> list[torch.Tensor]:
    """Return the final raw-only assignments used by four consensus rounds."""
    consensus = torch.cat(train_raw, dim=0)[0]
    final_orders: list[torch.Tensor] = []
    for _ in range(4):
        aligned: list[torch.Tensor] = []
        final_orders = []
        for task in train_raw:
            ref = consensus.expand_as(task).detach().transpose(1, 2)
            other = task.detach().transpose(1, 2)
            costs = (ref[:, :, None] - other[:, None, :]).square().sum(-1).cpu().numpy()
            orders = []
            for cost in costs:
                row, col = mask_ops.linear_sum_assignment(cost)
                order = torch.empty(task.shape[-1], dtype=torch.long)
                order[torch.as_tensor(row)] = torch.as_tensor(col)
                orders.append(order)
            order_tensor = torch.stack(orders).to(task.device)
            final_orders.append(order_tensor)
            aligned.append(task.gather(2, order_tensor[:, None, :].expand_as(task)))
        consensus = torch.cat(aligned, dim=0).mean(0)
    return final_orders


def _training_row_indices(raw_banks: list[torch.Tensor], seed: int,
                          device: torch.device) -> tuple[list[torch.Tensor], list[list[int]]]:
    generator = torch.Generator(device=device).manual_seed(seed + 50_101)
    train_maps, train_original_rows = [], []
    for maps in raw_banks:
        unique = mask_ops._unique_rows(maps)
        flat_original = maps.detach().cpu().reshape(len(maps), -1)
        flat_unique = unique.detach().cpu().reshape(len(unique), -1)
        lookup: dict[bytes, int] = {}
        for index, row in enumerate(flat_original):
            lookup.setdefault(row.numpy().tobytes(), index)
        original_for_unique = torch.tensor(
            [lookup[row.numpy().tobytes()] for row in flat_unique],
            dtype=torch.long, device=device)
        if len(unique) == 1:
            train_maps.append(unique)
            train_original_rows.append(original_for_unique)
            continue
        order = torch.randperm(len(unique), generator=generator, device=device)
        n_val = max(1, int(round(.2 * len(unique))))
        n_val = min(n_val, len(unique) - 1)
        train_order = order[n_val:]
        train_maps.append(unique[train_order])
        train_original_rows.append(original_for_unique[train_order])
    return train_maps, [value.detach().cpu().tolist() for value in train_original_rows]


def _extract_aligned_mean_masks(
    representation_maps: dict[str, list[torch.Tensor]], raw_banks: list[torch.Tensor],
    original_train_rows: list[list[int]], seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    device = raw_banks[0].device
    raw_train_by_task = [bank[torch.tensor(rows, device=device)]
                         for bank, rows in zip(raw_banks, original_train_rows)]
    # The exact same training-only raw consensus assignments are applied to
    # every representation. No representation gets a fitted alignment.
    orders = _raw_alignment_orders(raw_train_by_task)
    masks: dict[str, torch.Tensor] = {}
    reports: dict[str, Any] = {"train_models_per_task": [len(x) for x in raw_train_by_task],
                               "alignment_source": "raw maps, training rows only; four Hungarian rounds",
                               "same_column_orders_for_all_representations": True}
    for name, task_maps in representation_maps.items():
        aligned = []
        for task, order in zip(task_maps, orders):
            task_rows = task[torch.tensor(
                original_train_rows[len(aligned)], device=device)]
            aligned.append(task_rows.gather(2, order[:, None, :].expand_as(task_rows)))
        consensus = torch.cat(aligned, dim=0).mean(0)
        mask = torch.zeros_like(consensus.flatten())
        mask[consensus.flatten().topk(EDGES).indices] = 1.
        masks[name] = mask.reshape(FEATURES, HIDDEN).detach().cpu()
        reports[name] = {"consensus_mean": float(consensus.mean().cpu()),
                         "consensus_max": float(consensus.max().cpu()),
                         "selected_edges": int(mask.sum().cpu()),
                         "training_rows_by_task": original_train_rows}
    return masks, reports


def _load_source_checkpoint(folder: Path, task: int, old_bank: dict,
                            device: torch.device) -> dict[str, torch.Tensor]:
    path = folder / f"bank_{task}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = _select_state(checkpoint, old_bank, task)
    return {name: value.to(device) for name, value in state.items()}


def _write_protocol(out: Path, seed: int, device: torch.device, original: dict,
                    checkpoints: list[dict], *, raw_mean_exact: bool,
                    target_tf32: bool) -> None:
    write_json(out / "protocol.json", {
        "experiment": "direction4 importance representations and function-aware deletion",
        "seed": seed, "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "source_replay_audits": checkpoints,
        "source_split_hashes": original["data"].get("split_hashes", {}),
        "selected_source_models": 32, "source_tasks": 4,
        "source_analysis_tf32": False, "target_evaluation_tf32": target_tf32,
        "raw_mean_mask_exactly_matches_original_control": raw_mean_exact,
        "representation_definitions": {
            "raw_abs_weight": "per-model max-normalized abs(weight)*mask",
            "activity_weighted": "per-model max-normalized raw score times population std of source_train pixels",
            "function_gradient": "mean abs(weight*input*readout*(1-tanh(preactivation)^2)) over 4096 source_train images, strict FP32",
            "actual_deletion": "exact signed normalized source_train loss change after deleting one edge, strict FP32; positive tail is used for top-K extraction",
        },
        "source_train_calibration_sets": ANCHOR_SETS,
        "source_validation_selection_anchor_sets": ANCHOR_SETS,
        "target_labels_for_mask_extraction": False,
        "source_labels_for_deletion_importance": True,
        "mask_extraction": f"top {EDGES} of mean maps after fixed raw-map train-only 4-round Hungarian assignments",
        "target_evaluation": {"support_sizes": [32, 64, 128, 256], "steps": 800,
                              "paired_initializations": 4, "all_target_states_saved": True},
    })


def _save_run_provenance(out: Path, seed: int) -> None:
    save_provenance(out, seed, {
        "objective": "direction4 importance representation and function-deletion followup",
        "seed": seed, "source_labels_for_importance_maps": True,
        "target_labels_for_mask_extraction": False,
        "source_analysis_tf32": False,
        "alignment": "same four-round raw train-only column assignments applied to all representations",
        "mask_rule": f"top {EDGES} scores of mean aligned source-train maps",
        "deletion_loss_sign": "loss after deletion minus baseline; negative means removal improves loss",
    }, [Path(__file__), Path(core.__file__), Path(mask_ops.__file__),
        ROOT / "deepsets_vaae/followup_common.py"])


def _run_seed(seed: int, out: Path, timeout: float, *, saliency_only: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        print(f"seed {seed}: results already exist; refusing overwrite", flush=True)
        return
    started = time.monotonic()
    configure(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("importance queue requires CUDA; bind one approved UUID per worker")
    _progress(out, seed, started, "load_original_inputs", device=str(device),
              cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    original = load_pilot_seed(seed, device)
    data, old_banks = original["data"], original["banks"]
    target_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    target_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    # Source-function scores and direct deletion forwards use consistent
    # strict FP32 arithmetic. The paired target evaluator restores the pilot's
    # original TF32 setting below.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint_dir = _wait_checkpoint(seed, out, started, timeout)
    replay_audit_path = checkpoint_dir / "replay_audit.json"
    replay_audit = json.loads(replay_audit_path.read_text())
    if replay_audit.get("status") != "passed" or len(replay_audit.get("tasks", [])) != 4:
        raise ValueError(f"source replay audit is incomplete or failed: {replay_audit_path}")
    if any(task.get("audit") != "passed" or task.get("max_abs_errors", {}).get("weights_max_abs") != 0.0
           or task.get("max_abs_errors", {}).get("maps_max_abs") != 0.0
           for task in replay_audit["tasks"]):
        raise ValueError(f"source replay task-level audit did not pass exactly: {replay_audit_path}")
    task_costs = torch.tensor(task_vectors()["source"], dtype=torch.float32, device=device)
    loaded_states: list[dict[str, torch.Tensor]] = []
    checkpoints = []
    for task, bank in enumerate(old_banks):
        checkpoint_path = checkpoint_dir / f"bank_{task}.pt"
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        loaded_states.append({name: tensor.to(device)
                              for name, tensor in _select_state(payload, bank, task).items()})
        checkpoints.append({"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path),
                            "replay_audit_path": str(replay_audit_path),
                            "replay_audit_sha256": _sha256(replay_audit_path),
                            "replay_audit_status": replay_audit["status"],
                            "task_audit": replay_audit["tasks"][task]})
    source_hashes = data.get("split_hashes", {})
    for split_name in ("source_train", "source_validation"):
        if any(old_banks[task].get("source_split_hashes", {}).get(split_name) !=
               source_hashes.get(split_name) for task in range(4)):
            raise ValueError(f"recreated {split_name} split hash differs from original bank")
        if replay_audit.get("data_split_hashes", {}).get(split_name) != source_hashes.get(split_name):
            raise ValueError(f"recreated {split_name} split hash differs from source replay audit")
    _progress(out, seed, started, "calibrate_representations",
              source_checkpoint_audits=checkpoints)

    representation_per_task: dict[str, list[torch.Tensor]] = {
        name: [] for name in REP_NAMES
    }
    signed_train: list[torch.Tensor] = []
    positive_train: list[torch.Tensor] = []
    signed_validation: list[torch.Tensor] = []
    positive_validation: list[torch.Tensor] = []
    relevance_by_task = []
    pruning_by_task = []
    task_calibrations = []
    for task, (bank, state) in enumerate(zip(old_banks, loaded_states)):
        raw = (state["weight"].abs() * state["masks"]).detach()
        legacy = torch.as_tensor(bank["maps"], device=device, dtype=torch.float32)
        if not torch.allclose(_normalize_nonnegative(raw), legacy, atol=2e-6, rtol=2e-6):
            raise ValueError(f"raw |W|*M does not reproduce legacy maps for seed={seed}, task={task}")
        train_x, train_y, val_x, val_y, function_images, pixel_std, train_rng_seed = \
            _calibration_and_validation(data, task_costs[task], seed + 1000 * task, device)
        activity = raw * pixel_std[None, :, None]
        function = _function_gradient_scores(state, function_images)
        delta_train = _edge_deletion_deltas(state, train_x, train_y)
        delta_val = _edge_deletion_deltas(state, val_x, val_y)
        train_positive = delta_train.clamp_min(0.)
        val_positive = delta_val.clamp_min(0.)
        representation_per_task["raw_abs_weight"].append(_normalize_nonnegative(raw))
        representation_per_task["activity_weighted"].append(_normalize_nonnegative(activity))
        representation_per_task["function_gradient"].append(_normalize_nonnegative(function))
        representation_per_task["train_deletion_positive"].append(
            _normalize_nonnegative(train_positive))
        signed_train.append(delta_train)
        positive_train.append(train_positive)
        signed_validation.append(delta_val)
        positive_validation.append(val_positive)
        maps = {"raw_abs_weight": raw, "activity_weighted": activity,
                "function_gradient": function}
        relevance_by_task.append(_rank_relevance(state, maps, delta_train, delta_val))
        pruning_scores = {name: representation_per_task[name][-1]
                          for name in REP_NAMES}
        pruning_by_task.append(_validation_pruning(state, pruning_scores, val_x, val_y,
                                                  seed, task))
        task_calibrations.append({
            "task": task, "bank_seed": seed + 1000 * task,
            "source_train_fixed_set_seed": train_rng_seed,
            "source_train_fixed_sets": ANCHOR_SETS,
            "source_validation_fixed_sets": ANCHOR_SETS,
            "function_gradient_images": len(function_images),
            "function_gradient_uses_source_labels": False,
            "deletion_map_uses_source_train_labels": True,
            "heldout_validation_uses_source_labels": True,
            "target_labels_used_for_extraction": False,
            "active_edges_per_model": int(state["masks"].sum((-1, -2)).min().item()),
        })
        _progress(out, seed, started, "task_complete", task=task,
                  mean_source_train_delta=float(delta_train.mean().cpu()),
                  positive_source_train_delta_fraction=float((delta_train > 0).float().mean().cpu()),
                  mean_source_validation_delta=float(delta_val.mean().cpu()))

    raw_bank_maps = [torch.as_tensor(bank["maps"], device=device, dtype=torch.float32)
                     for bank in old_banks]
    _, train_rows = _training_row_indices(raw_bank_maps, seed, device)
    new_mask_singles, alignment_details = _extract_aligned_mean_masks(
        representation_per_task, raw_bank_maps, train_rows, seed)
    new_masks = {f"importance_{name}": value[None].expand(4, -1, -1).clone()
                 for name, value in new_mask_singles.items()}
    original_mean = original["original_masks"]["mean"].detach().cpu().float()
    raw_mean_exact = torch.equal(new_masks["importance_raw_abs_weight"], original_mean)
    if not raw_mean_exact:
        mismatch = float((new_masks["importance_raw_abs_weight"] - original_mean).abs().max().cpu())
        raise AssertionError(f"fixed raw-importance control differs from original mean mask: {mismatch}")
    torch.save(new_masks, out / "transfer_masks.pt")

    # Every signed delta array is retained as-is. Positive maps are separately
    # stored for fixed-cardinality extraction; this keeps improvements from
    # edge removal visible instead of silently relabeling them as saliency.
    arrays: dict[str, np.ndarray] = {
        "source_train_pixel_std": data["source_train"].features.std(dim=0, unbiased=False).cpu().numpy(),
        "training_row_indices_by_task": np.asarray(train_rows, dtype=np.int64),
        "masks": np.stack([bank["masks"].detach().cpu().numpy() for bank in old_banks]),
        "raw_abs_weight": np.stack([x.cpu().numpy() for x in representation_per_task["raw_abs_weight"]]),
        "activity_weighted": np.stack([x.cpu().numpy() for x in representation_per_task["activity_weighted"]]),
        "function_gradient": np.stack([x.cpu().numpy() for x in representation_per_task["function_gradient"]]),
        "train_deletion_signed_delta": np.stack([x.cpu().numpy() for x in signed_train]),
        "train_deletion_positive_part": np.stack([x.cpu().numpy() for x in positive_train]),
        "validation_deletion_signed_delta": np.stack([x.cpu().numpy() for x in signed_validation]),
        "validation_deletion_positive_part": np.stack([x.cpu().numpy() for x in positive_validation]),
    }
    _write_npz(out / "importance_arrays.npz", **arrays)

    diagnostics_payload = {
        "seed": seed, "representation_definitions": {
            "raw_abs_weight": "per-model max-normalized |W|*M; exactly reproduces legacy bank maps",
            "activity_weighted": "per-model max-normalized |W|*M*population_std(source_train pixel)",
            "function_gradient": "per-model max-normalized mean |W_ij*x_i*a_j*(1-tanh(u_j)^2)| on source_train images; strict FP32 forward",
            "train_deletion_signed_delta": "exact strict-FP32 L_source_train(delete edge)-L_source_train(original), signed; negative means deletion improves the bank task loss",
            "train_deletion_positive": "positive part of signed source_train delta, normalized only for mask extraction",
            "validation_deletion_signed_delta": "exact strict-FP32 L_source_validation(delete edge)-L_source_validation(original), signed; these old anchors were also used for checkpoint selection",
        },
        "task_calibrations": task_calibrations,
        "source_split_hashes": source_hashes,
        "source_checkpoint_files": checkpoints,
        "rank_relevance_vs_heldout_validation": relevance_by_task,
        "heldout_validation_pruning_delta_nmse": pruning_by_task,
        "fixed_alignment_and_extraction": alignment_details,
        "raw_mean_mask_exactly_matches_original_control": raw_mean_exact,
        "self_check": _self_check(device),
        "source_analysis_tf32": False,
        "target_evaluation_tf32": bool(target_matmul_tf32),
        "deletion_delta_sign_convention": "positive = removing edge worsens heldout/source loss; negative = removal improves loss",
        "limitations": [
            "Source-train calibration sets are resampled from the same source_train pool used to train bank candidates; source_validation is a separate heldout split.",
            "The source_validation image split also selected the original bank checkpoints; its fixed 128-set deletion check is held out from saliency construction, but is not an independent source test.",
            "Importance masks are extracted from a training-only mean of aligned selected maps; no target labels enter mask selection.",
            "Single-edge deltas are exact one-at-a-time effects; deleting several edges together can interact, so grouped pruning was also measured directly by forward evaluation.",
            "Four transfer replicas repeat the same deterministic representation-derived mask; variation comes from paired target initializations, not mask extraction randomness.",
        ],
    }
    write_json(out / "importance_diagnostics.json", diagnostics_payload)
    if saliency_only:
        _write_protocol(out, seed, device, original, checkpoints,
                        raw_mean_exact=raw_mean_exact, target_tf32=bool(target_matmul_tf32))
        _save_run_provenance(out, seed)
        torch.backends.cuda.matmul.allow_tf32 = target_matmul_tf32
        torch.backends.cudnn.allow_tf32 = target_cudnn_tf32
        write_json(out / "status.json", {"seed": seed, "stage": "saliency_complete",
                  "source_analysis_tf32": False, "raw_mean_control_exact": raw_mean_exact})
        return

    _progress(out, seed, started, "paired_target_transfer",
              methods=list(new_masks), training_rows_by_task=[len(x) for x in train_rows])
    torch.backends.cuda.matmul.allow_tf32 = target_matmul_tf32
    torch.backends.cudnn.allow_tf32 = target_cudnn_tf32
    records = evaluate_followup(
        data, torch.tensor(task_vectors()["test"], device=device, dtype=torch.float32),
        new_masks, seed, device, out / "weights", original["original_masks"])
    write_json(out / "results.json", {
        "seed": seed, "records": records,
        "elapsed_seconds": time.monotonic() - started,
        "method_names": list(new_masks),
        "task_vectors_seed": task_vectors()["seed"],
    })
    _write_protocol(out, seed, device, original, checkpoints,
                    raw_mean_exact=raw_mean_exact, target_tf32=bool(target_matmul_tf32))
    _save_run_provenance(out, seed)
    write_json(out / "status.json", {"seed": seed, "stage": "complete",
              "elapsed_seconds": time.monotonic() - started,
              "records": len(records), "output": str(out)})
    _progress(out, seed, started, "complete", records=len(records))


@torch.no_grad()
def _independent_source_validation_recheck(out_root: Path) -> None:
    """Check train-derived ranks on fresh source-validation set draws.

    The underlying source_validation image split selected the original bank
    checkpoints. These fresh set draws avoid reusing the exact checkpoint
    selection anchors, but they are not an independent image-level test.
    """
    seeds = tuple(range(4100, 4108))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("independent source-validation recheck requires CUDA")
    for seed in seeds:
        out = out_root / f"seed_{seed}"
        if not (out / "results.json").exists():
            raise FileNotFoundError(f"seed {seed} target transfer is incomplete")
        configure(seed)
        original = load_pilot_seed(seed, device)
        target_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        target_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        data = original["data"]
        source_validation = data["source_validation"]
        if not isinstance(source_validation, core.Split):
            source_validation = core.Split(*source_validation)
        task_costs = torch.tensor(task_vectors()["source"], dtype=torch.float32, device=device)
        checkpoint_dir = CHECKPOINT_ROOT / f"seed_{seed}"
        audit_path = checkpoint_dir / "replay_audit.json"
        audit = json.loads(audit_path.read_text())
        if audit.get("status") != "passed" or len(audit.get("tasks", [])) != 4:
            raise ValueError(f"seed {seed}: source replay audit failed or incomplete")
        source_hashes = data.get("split_hashes", {})
        for split_name in ("source_train", "source_validation"):
            if audit.get("data_split_hashes", {}).get(split_name) != source_hashes.get(split_name):
                raise ValueError(f"seed {seed}: {split_name} hash differs from source replay audit")
            if any(original["banks"][task].get("source_split_hashes", {}).get(split_name) !=
                   source_hashes.get(split_name) for task in range(4)):
                raise ValueError(f"seed {seed}: {split_name} hash differs from old source bank")
        with np.load(out / "importance_arrays.npz") as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
            raw_train_maps = torch.as_tensor(archive["raw_abs_weight"], device=device)
            activity_maps = torch.as_tensor(archive["activity_weighted"], device=device)
            function_maps = torch.as_tensor(archive["function_gradient"], device=device)
            train_delta_maps = torch.as_tensor(archive["train_deletion_signed_delta"], device=device)
            train_positive_maps = torch.as_tensor(archive["train_deletion_positive_part"], device=device)

        independent_delta_by_task = []
        relevance_by_task = []
        pruning_by_task = []
        anchor_records = []
        for task in range(4):
            bank_seed = seed + 1000 * task
            selection_seed = bank_seed + 32
            selection_rng = torch.Generator(device=device).manual_seed(selection_seed)
            selection_ids = torch.randint(len(source_validation.features),
                                          (ANCHOR_SETS, SET_SIZE),
                                          generator=selection_rng, device=device)
            selection_keys = {tuple(sorted(row)) for row in selection_ids.cpu().tolist()}
            heldout_seed = None
            heldout_ids = None
            for increment in range(100):
                candidate_seed = bank_seed + 700_032 + increment
                candidate_rng = torch.Generator(device=device).manual_seed(candidate_seed)
                candidate_ids = torch.randint(len(source_validation.features),
                                              (ANCHOR_SETS, SET_SIZE),
                                              generator=candidate_rng, device=device)
                candidate_keys = {tuple(sorted(row)) for row in candidate_ids.cpu().tolist()}
                if not selection_keys.intersection(candidate_keys):
                    heldout_seed, heldout_ids = candidate_seed, candidate_ids
                    break
            if heldout_ids is None or heldout_seed is None:
                raise RuntimeError(f"no disjoint anchor draw for seed={seed}, task={task}")
            val_x = source_validation.features[heldout_ids]
            val_y = task_costs[task][source_validation.digits[heldout_ids]].sum(dim=1)
            checkpoint = torch.load(checkpoint_dir / f"bank_{task}.pt",
                                    map_location="cpu", weights_only=False)
            state = {key: value.to(device) for key, value in _select_state(
                checkpoint, original["banks"][task], task).items()}
            val_delta = _edge_deletion_deltas(state, val_x, val_y)
            independent_delta_by_task.append(val_delta.cpu().numpy())
            score_maps = {
                "raw_abs_weight": raw_train_maps[task],
                "activity_weighted": activity_maps[task],
                "function_gradient": function_maps[task],
            }
            relevance_by_task.append(_rank_relevance(
                state, score_maps, train_delta_maps[task], val_delta))
            pruning_scores = {
                **score_maps,
                "train_deletion_positive": _normalize_nonnegative(train_positive_maps[task]),
            }
            pruning_by_task.append(_validation_pruning(
                state, pruning_scores, val_x, val_y, seed, task))
            anchor_records.append({
                "task": task,
                "checkpoint_bank_seed": bank_seed,
                "source_validation_selection_anchor_seed": selection_seed,
                "independent_source_validation_anchor_seed": heldout_seed,
                "anchor_sets": ANCHOR_SETS,
                "set_size": SET_SIZE,
                "exact_anchor_multiset_overlap_with_selection": 0,
                "source_validation_split_hash": data.get("split_hashes", {}).get("source_validation"),
                "qualification": "Fresh set draws share the source_validation image pool used for checkpoint selection.",
            })

        arrays["independent_validation_deletion_signed_delta"] = np.stack(independent_delta_by_task)
        arrays["independent_validation_deletion_positive_part"] = np.maximum(
            arrays["independent_validation_deletion_signed_delta"], 0.)
        _write_npz(out / "importance_arrays.npz", **arrays)
        diagnostics_path = out / "importance_diagnostics.json"
        diagnostics = json.loads(diagnostics_path.read_text())
        diagnostics["independent_source_validation_resampled"] = {
            "anchor_protocol": anchor_records,
            "rank_relevance_vs_independent_source_validation": relevance_by_task,
            "heldout_validation_pruning_delta_nmse": pruning_by_task,
            "interpretation": "Fresh complete-set draws from the same source_validation image split; no exact set-multiset overlaps with checkpoint-selection anchors. The image split itself selected the source-bank checkpoints.",
        }
        write_json(diagnostics_path, diagnostics)
        write_json(out / "independent_validation_status.json", {
            "seed": seed, "stage": "complete", "tasks": 4,
            "anchor_set_overlap": 0, "source_validation_image_split_independent": False,
        })
        torch.backends.cuda.matmul.allow_tf32 = target_matmul_tf32
        torch.backends.cudnn.allow_tf32 = target_cudnn_tf32
        print(json.dumps({"seed": seed, "stage": "independent_source_validation_complete"}), flush=True)


def _aggregate(out_root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = tuple(range(4100, 4108))
    missing = [seed for seed in seeds if not (out_root / f"seed_{seed}" / "results.json").exists()
               or not (out_root / f"seed_{seed}" / "importance_diagnostics.json").exists()
               or not (out_root / f"seed_{seed}" / "independent_validation_status.json").exists()]
    if missing:
        raise FileNotFoundError(f"cannot aggregate; missing completed seeds: {missing}")
    result_by_seed = {seed: json.loads((out_root / f"seed_{seed}" / "results.json").read_text())
                      for seed in seeds}
    diag_by_seed = {seed: json.loads((out_root / f"seed_{seed}" / "importance_diagnostics.json").read_text())
                    for seed in seeds}

    # Each map figure shows the average normalized source-map mass by pixel;
    # the heldout signed deletion panel uses a diverging scale and is not
    # interpreted as a nonnegative importance heatmap.
    map_names = ["raw_abs_weight", "activity_weighted", "function_gradient",
                 "train_deletion_positive_part", "independent_validation_deletion_signed_delta"]
    map_titles = ["Raw |W|×M", "Activity weighted", "Function gradient",
                  "Positive train deletion Δloss", "Signed independent validation Δloss"]
    map_means = []
    for name in map_names:
        collected = []
        for seed in seeds:
            with np.load(out_root / f"seed_{seed}" / "importance_arrays.npz") as values:
                data = values[name]
                if name.endswith("signed_delta"):
                    # signed actual deletion deltas stay on their original
                    # scale; do not map them through a positive-only transform.
                    collected.append(data.mean(axis=(0, 1, 3)))
                else:
                    if name == "train_deletion_positive_part":
                        data = data.reshape(data.shape[0], data.shape[1], -1)
                        maxima = data.max(axis=-1, keepdims=True)
                        data = data / np.maximum(maxima, 1e-12)
                        data = data.reshape(4, 32, 784, 32)
                    collected.append(data.mean(axis=(0, 1, 3)))
        map_means.append(np.mean(collected, axis=0).reshape(28, 28))
    fig, axes = plt.subplots(1, len(map_means), figsize=(16, 3.8), constrained_layout=True)
    for i, (ax, mean, title) in enumerate(zip(axes, map_means, map_titles)):
        if i == len(map_means) - 1:
            vmax = float(np.max(np.abs(mean))) or 1.
            image = ax.imshow(mean, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        else:
            image = ax.imshow(mean, cmap="magma", vmin=0, vmax=max(float(mean.max()), 1e-12))
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.suptitle("Средние importance представления по 8 повторам", fontsize=12)
    fig.savefig(out_root / "importance_representations.png", dpi=180)
    fig.savefig(out_root / "importance_representations.pdf")
    plt.close(fig)

    # Source-validation grouped ablation deltas, aggregated first within each
    # seed across tasks/models, retaining repeats as the sampling unit.
    prune_keys = []
    for method in ("raw_abs_weight", "activity_weighted", "function_gradient",
                   "train_deletion_positive"):
        prune_keys.extend((f"{method}_high", f"{method}_low"))
    prune_keys.append("random")
    seed_deltas = {key: [] for key in prune_keys}
    for seed in seeds:
        task_rows = diag_by_seed[seed]["independent_source_validation_resampled"][
            "heldout_validation_pruning_delta_nmse"]
        for key in prune_keys:
            values = [row[key]["mean_delta_nmse"] for row in task_rows]
            seed_deltas[key].append(float(np.mean(values)))
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    labels = [key.replace("_", "\n", 1) for key in prune_keys]
    means = [float(np.mean(seed_deltas[key])) for key in prune_keys]
    errors = [float(np.std(seed_deltas[key], ddof=1) / np.sqrt(len(seeds))) for key in prune_keys]
    colors = ["#446688" if key.endswith("high") else
              "#7799aa" if key.endswith("low") else "#999999" for key in prune_keys]
    ax.bar(np.arange(len(prune_keys)), means, yerr=errors, color=colors, capsize=3)
    ax.axhline(0., color="black", linewidth=.8)
    ax.set_xticks(np.arange(len(prune_keys)), labels, fontsize=8)
    ax.set_ylabel("Δ NMSE source validation (after pruning − baseline)")
    ax.set_title("Проверка удаления связей на held-out source validation")
    fig.savefig(out_root / "heldout_pruning_validation.png", dpi=180)
    fig.savefig(out_root / "heldout_pruning_validation.pdf")
    plt.close(fig)

    # Paired target transfer curves, preserving repeat as the sampling unit.
    methods = ["importance_raw_abs_weight", "importance_activity_weighted",
               "importance_function_gradient", "importance_train_deletion_positive"]
    budgets = (32, 64, 128, 256)
    test_records = {}
    for seed in seeds:
        test_records[seed] = [record for record in result_by_seed[seed]["records"]
                              if record["method"] in methods]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    colors = ["#555555", "#e69f00", "#0072b2", "#cc79a7"]
    labels = ["raw", "activity weighted", "function gradient", "train deletion"]
    for method, color, label in zip(methods, colors, labels):
        per_seed = []
        for seed in seeds:
            vals_by_budget = []
            for budget in budgets:
                vals = [row["mse"] for row in test_records[seed]
                        if row["method"] == method and row["support_size"] == budget]
                vals_by_budget.append(float(np.mean(vals)))
            per_seed.append(vals_by_budget)
        values = np.asarray(per_seed)
        mean = values.mean(axis=0)
        se = values.std(axis=0, ddof=1) / np.sqrt(len(seeds))
        ci_half_width = float(student_t.ppf(.975, df=len(seeds) - 1)) * se
        ax.plot(budgets, mean, marker="o", color=color, label=label)
        ax.fill_between(budgets, mean - ci_half_width, mean + ci_half_width,
                        color=color, alpha=.16)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets, [str(value) for value in budgets])
    ax.set_xlabel("Число размеченных целевых наборов")
    ax.set_ylabel("Средняя тестовая MSE")
    ax.set_title("Перенос масок, извлечённых из разных importance карт")
    ax.legend(frameon=False)
    fig.savefig(out_root / "importance_transfer.png", dpi=180)
    fig.savefig(out_root / "importance_transfer.pdf")
    plt.close(fig)

    methods_report = {}
    for method in methods:
        rows = [record for seed in seeds for record in result_by_seed[seed]["records"]
                if record["method"] == method]
        methods_report[method] = {}
        for budget in budgets:
            per_seed = []
            for seed in seeds:
                vals = [record["mse"] for record in result_by_seed[seed]["records"]
                        if record["method"] == method and record["support_size"] == budget]
                per_seed.append(float(np.mean(vals)))
            values = np.asarray(per_seed)
            methods_report[method][str(budget)] = {
                "mean_test_mse": float(values.mean()),
                "std_across_seeds": float(values.std(ddof=1)),
                "n_seeds": len(values),
            }

    relevance_summary = {}
    relevance_names = ("raw_abs_weight", "activity_weighted", "function_gradient",
                       "train_deletion_signed", "train_deletion_positive")
    for name in relevance_names:
        vals_signed, vals_positive = [], []
        for seed in seeds:
            for task in diag_by_seed[seed]["independent_source_validation_resampled"][
                    "rank_relevance_vs_independent_source_validation"]:
                report = task[name]
                if report["spearman_vs_heldout_signed_delta_mean"] is not None:
                    vals_signed.append(report["spearman_vs_heldout_signed_delta_mean"])
                if report["spearman_vs_heldout_positive_delta_mean"] is not None:
                    vals_positive.append(report["spearman_vs_heldout_positive_delta_mean"])
        relevance_summary[name] = {
            "spearman_to_heldout_signed_delta_mean": float(np.mean(vals_signed)) if vals_signed else None,
            "spearman_to_heldout_positive_delta_mean": float(np.mean(vals_positive)) if vals_positive else None,
            "n_task_repeat_groups": len(vals_signed),
        }

    report = {
        "experiment": "direction4 source importance representation checks",
        "seeds": list(seeds), "source_tasks_per_seed": 4,
        "selected_models_per_task": 32, "methods": methods_report,
        "rank_relevance": relevance_summary,
        "heldout_pruning_delta_nmse_by_method": {
            key: {"mean_across_seeds": float(np.mean(values)),
                  "std_across_seeds": float(np.std(values, ddof=1)),
                  "seed_values": values}
            for key, values in seed_deltas.items()},
        "signed_delta_convention": "loss after edge deletion minus original loss; negative means deletion improves loss",
        "plots": {
            "importance_representations": "Mean spatial map by input pixel; first four are nonnegative normalized saliency, last panel is signed source-validation loss delta on fresh anchor draws, displayed with a diverging color scale.",
            "heldout_pruning_validation": "Grouped high-score, low-score, and random 2% edge removals measured directly on fresh source-validation set draws; error bars are standard errors across eight seeds. The underlying validation image split selected source-bank checkpoints.",
            "importance_transfer": "Test NMSE for new tasks; line is mean across seeds after averaging tasks and paired starts, band is a pointwise 95% Student-t confidence interval across eight seeds (df=7).",
        },
        "limitations": [
            "Source deletion importance uses source labels by design. Fresh source-validation set draws are held out from saliency construction and disjoint from the exact checkpoint-selection anchors, but the underlying image split selected the source-bank checkpoints.",
            "Importance maps are normalized per model for representation comparison and top-K extraction; signed raw deletion deltas are saved separately and never passed off as nonnegative scores.",
            "The four repeated extraction masks per method are identical; target initialization variation is paired through the shared evaluator.",
            "The source banks retain only selected top-quartile candidates, so conclusions apply to these selected banks.",
        ],
    }
    write_json(out_root / "importance_summary.json", report)
    (out_root / "REPORT.md").write_text(_russian_report(report), encoding="utf-8")


def _russian_report(report: dict[str, Any]) -> str:
    lines = [
        "# Проверка представлений importance и функционального вклада",
        "",
        "Проверены банки из восьми повторов, по четырём исходным задачам и 32 выбранным моделям в каждой задаче. Сравнены исходные карты `|W|×M`, карты с весом стандартного отклонения пикселя, функциональная оценка через производную `tanh` и точное изменение source-train loss после удаления одной связи.",
        "",
        "Для каждого ребра сохранено знаковое `Δloss = loss(после удаления) − loss(до удаления)`. Положительное значение означает, что удаление ухудшило модель; отрицательное — что удаление улучшило loss. Для извлечения бинарной маски использовалась только положительная часть source-train Δloss, отдельно от сохранённого знакового массива. Проверка на source-validation повторно генерирует наборы из отдельного RNG-потока, без совпадений полных наборов с anchors выбора checkpoint. Target labels не использовались для извлечения масок.",
        "",
        "## Перенос",
        "",
        "Одинаковые четыре consensus-перестановки, вычисленные по train-части исходных raw-карт, применены ко всем вариантам importance. Из среднего согласованного training-набора выбирались ровно 5018 связей. На каждой target-задаче сравнивались одинаковые paired initialization и бюджеты; исходные пять методов повторялись и служили контролем.",
        "",
        "Средняя тестовая MSE по 8 повторам (задачи и paired starts усреднены внутри повтора):",
        "",
        "| Представление | " + " | ".join(f"{budget}" for budget in (32, 64, 128, 256)) + " |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "importance_raw_abs_weight": "Raw `|W|×M`",
        "importance_activity_weighted": "Activity weighted",
        "importance_function_gradient": "Function gradient",
        "importance_train_deletion_positive": "Positive train deletion Δloss",
    }
    for method, label in labels.items():
        values = report["methods"][method]
        lines.append("| " + label + " | " + " | ".join(
            f"{values[str(budget)]['mean_test_mse']:.4f}" for budget in (32, 64, 128, 256)) + " |")
    lines.extend([
        "",
        "## Проверка функциональной значимости",
        "",
        "График `heldout_pruning_validation.png` показывает изменение source-validation NMSE на свежих наборах после совместного удаления верхних или нижних 2% активных связей по каждому train-derived score, а также случайных 2%. Это прямой пересчёт модели после удаления групп рёбер; он учитывает взаимодействия, которые сумма независимых edge-дельт не отражает. Наборы не совпадают с исходными anchors выбора модели, но используют тот же пул изображений source_validation.",
        "",
        "Средняя Spearman-корреляция рангов train score с Δloss на свежих source-validation anchors:",
        "",
        "| Train score | Signed heldout Δloss | Positive heldout Δloss |",
        "|---|---:|---:|",
    ])
    for name, values in report["rank_relevance"].items():
        lines.append(f"| {name} | {values['spearman_to_heldout_signed_delta_mean']!s} | {values['spearman_to_heldout_positive_delta_mean']!s} |")
    lines.extend([
        "",
        "![Importance-карты](importance_representations.png)",
        "",
        "**Пояснение.** Первые четыре панели показывают средние нормированные неотрицательные карты. Последняя панель показывает знаковое source-validation Δloss на свежих наборах в diverging шкале: отрицательные значения — связи, удаление которых улучшило loss.",
        "",
        "![Source-validation абляции](heldout_pruning_validation.png)",
        "",
        "**Пояснение.** Столбец показывает `NMSE после удаления − NMSE исходной модели`; положительное значение означает ухудшение. Ошибки — стандартная ошибка по восьми повторам. Повтор остаётся единицей агрегации.",
        "",
        "![Перенос importance-масок](importance_transfer.png)",
        "",
        "**Пояснение.** Тестовая MSE усреднена по восьми новым задачам, четырём paired starts и затем по восьми seed-повторам. Полоса — точечный 95%-й интервал по seed-повторам; target labels участвовали только в общих target fit/validation/test оценках.",
        "",
        "Ограничение: fixed calibration sets пересэмплируются из source-train пула, использованного при обучении исходного банка. Новые source-validation anchors независимы от выбора на уровне set draws, но тот же source_validation image split использовался для выбора checkpoint и 32 победителей; это не независимый image-level source test. Банки включают только выбранные верхние 32 кандидата из исходных 128. Четыре новых маски-реплики каждого способа совпадают; разброс downstream результата отражает paired target initialization, а не стохастичность извлечения.",
        "",
        "Машиночитаемые результаты, исходные signed arrays, provenance, checkpoints целевых моделей и protocol сохранены рядом с этим отчётом.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    parser.add_argument("--wait-seconds", type=float, default=7200.)
    parser.add_argument("--saliency-only", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--revalidate", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.report:
        _aggregate(args.out)
        return
    if args.revalidate:
        _independent_source_validation_recheck(args.out)
        return
    if args.self_check:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(json.dumps(_self_check(device), ensure_ascii=False), flush=True)
        return
    if args.seed not in range(4100, 4108):
        parser.error("--seed must be one of 4100..4107")
    _run_seed(args.seed, args.out, args.wait_seconds, saliency_only=args.saliency_only)


if __name__ == "__main__":
    main()
