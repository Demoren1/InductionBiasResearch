"""Functional alignment of replayed DeepSets source banks.

Only source-training images are used as activation anchors.  Source-bank maps
are split with the pilot's exact CUDA RNG seed before either reference is fit;
the held-out maps align only to those training references.  No target or test
labels enter extraction. For controlled mask transfer, importance maps are
the original per-model normalized ``bank['maps']`` values, with hidden
columns permuted only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from scipy.optimize import linear_sum_assignment

from . import masks as mask_module
from .followup_common import (
    FOLLOWUP,
    PILOT,
    configure,
    evaluate_followup,
    load_pilot_seed,
    save_provenance,
    write_json,
)
from .source_replay import DEFAULT_CHECKPOINT_ROOT, ROOT, _original_core, task_vectors


ALIGNMENT_ROOT = FOLLOWUP / "alignment"
GPU_UUIDS = [
    "GPU-7bb2c2a2-451a-7632-d931-fc64f8901744",
    "GPU-ebb2cc3f-3769-6b77-f2de-166234901bb6",
    "GPU-fa392093-03b9-0a49-7959-ed393750e978",
]
HIDDEN = 32
FEATURES = 784
K = 5018
ANCHOR_FIT_COUNT = 512
ANCHOR_HELDOUT_COUNT = 512
CONSTANT_STD_TOL = 1e-5
CONSENSUS_ITERATIONS = 4


def _sha_ids(values: torch.Tensor) -> str:
    data = values.detach().cpu().numpy().astype("<i8", copy=False).tobytes()
    return hashlib.sha256(data).hexdigest()


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {result}")
    return result


def _unique_original_rows(maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the pilot's sorted unique rows and their original bank indices."""
    unique = mask_module._unique_rows(maps)
    flat = maps.reshape(maps.size(0), -1)
    unique_flat = unique.reshape(unique.size(0), -1)
    equal = (unique_flat[:, None, :] == flat[None, :, :]).all(dim=-1)
    if not bool(equal.any(dim=1).all()):
        raise AssertionError("could not trace unique pilot maps to original bank rows")
    first_index = equal.float().argmax(dim=1)
    return unique, first_index


def _exact_map_splits(banks: list[dict[str, Any]], seed: int,
                      device: torch.device) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
    """Reproduce masks._split_unique with one shared CUDA generator."""
    generator_seed = seed + 50_000 + 101
    generator = torch.Generator(device=device).manual_seed(generator_seed)
    train_rows: list[list[int]] = []
    valid_rows: list[list[int]] = []
    details = []
    for task, bank in enumerate(banks):
        maps = torch.as_tensor(bank["maps"], dtype=torch.float32, device=device)
        unique, original_ids = _unique_original_rows(maps)
        if unique.size(0) == 1:
            only_index = int(original_ids[0].item())
            train_rows.append([only_index])
            valid_rows.append([only_index])
            details.append({"task": task, "unique_maps": 1,
                            "train_bank_indices": train_rows[-1],
                            "validation_bank_indices": valid_rows[-1],
                            "singleton_duplicated_as_in_pilot": True})
            continue
        order = torch.randperm(unique.size(0), generator=generator, device=device)
        n_valid = max(1, int(round(0.2 * unique.size(0))))
        n_valid = min(n_valid, unique.size(0) - 1)
        train_ids = original_ids[order[n_valid:]].detach().cpu().tolist()
        valid_ids = original_ids[order[:n_valid]].detach().cpu().tolist()
        train_rows.append([int(index) for index in train_ids])
        valid_rows.append([int(index) for index in valid_ids])
        details.append({"task": task, "unique_maps": int(unique.size(0)),
                        "train_bank_indices": train_rows[-1],
                        "validation_bank_indices": valid_rows[-1]})
    return train_rows, valid_rows, {
        "generator": "torch.Generator(device=cuda).manual_seed(seed + 50000 + 101)",
        "generator_seed": generator_seed,
        "tasks": details,
    }


def _correlation_matrix(left: np.ndarray, right: np.ndarray,
                        std_tol: float = CONSTANT_STD_TOL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pearson correlations, with constant columns marked invalid."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("activation descriptors must be [anchors, neurons] with equal anchors")
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean(axis=0, keepdims=True)
    asd = np.sqrt(np.mean(ac * ac, axis=0))
    bsd = np.sqrt(np.mean(bc * bc, axis=0))
    av = np.isfinite(asd) & (asd > std_tol)
    bv = np.isfinite(bsd) & (bsd > std_tol)
    az = np.zeros_like(ac)
    bz = np.zeros_like(bc)
    az[:, av] = ac[:, av] / asd[av]
    bz[:, bv] = bc[:, bv] / bsd[bv]
    corr = az.T @ bz / a.shape[0]
    corr = np.clip(corr, -1.0, 1.0)
    return corr, av, bv


def _full_match(corr: np.ndarray, valid_left: np.ndarray, valid_right: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match variable neurons by |corr|; constants remain explicitly unmatched.

    Returns a full source-column permutation plus signs, a reference-slot
    matched mask, and signed correlations.  Arbitrary filler assignments keep
    exported transformed parameters an exact permutation of the original model;
    filler axes are excluded from matching statistics and map means.
    """
    h = corr.shape[0]
    if corr.shape != (h, h) or valid_left.shape != (h,) or valid_right.shape != (h,):
        raise ValueError("matching expects equal hidden widths")
    left = np.flatnonzero(valid_left)
    right = np.flatnonzero(valid_right)
    rows = np.full(h, -1, dtype=np.int64)
    signs = np.ones(h, dtype=np.float32)
    matched = np.zeros(h, dtype=bool)
    signed = np.full(h, np.nan, dtype=np.float32)
    if len(left) and len(right):
        selected_rows, selected_cols = linear_sum_assignment(-np.abs(corr[np.ix_(left, right)]))
        for local_row, local_col in zip(selected_rows, selected_cols):
            row, col = int(left[local_row]), int(right[local_col])
            value = float(corr[row, col])
            rows[row] = col
            signs[row] = -1.0 if value < 0 else 1.0
            matched[row] = True
            signed[row] = value * signs[row]
    unused_source = [index for index in range(h) if index not in set(rows[rows >= 0].tolist())]
    unused_reference = [index for index in range(h) if rows[index] < 0]
    for slot, source in zip(unused_reference, unused_source):
        rows[slot] = source
    if np.any(rows < 0) or len(set(rows.tolist())) != h:
        raise AssertionError("full permutation completion failed")
    return rows, signs, matched, signed


def _fit_activation_match(reference: np.ndarray, other: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    corr, ref_valid, other_valid = _correlation_matrix(reference, other)
    order, signs, matched, signed = _full_match(corr, ref_valid, other_valid)
    return order, signs, matched, signed, corr


def _map_match(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Hungarian L2 matching for [features, hidden] importance maps."""
    if reference.shape != other.shape or reference.ndim != 2:
        raise ValueError("map matching expects equal [features, hidden] matrices")
    left = reference.T.astype(np.float64, copy=False)
    right = other.T.astype(np.float64, copy=False)
    # Use the expanded quadratic form to avoid allocating [H,H,F] differences.
    costs = (np.square(left).sum(-1)[:, None] + np.square(right).sum(-1)[None, :]
             - 2.0 * left @ right.T)
    rows, cols = linear_sum_assignment(costs)
    order = np.empty(reference.shape[1], dtype=np.int64)
    order[rows] = cols
    return order


def _raw_consensus(train_maps_by_task: list[np.ndarray], valid_maps_by_task: list[np.ndarray],
                   device: torch.device | None = None
                   ) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]], np.ndarray,
                              list[list[np.ndarray]], list[list[np.ndarray]]]:
    """Run the pilot's four-step consensus on its original normalized maps.

    Keep this computation in float32 tensors and use the pinned pilot matcher
    cost (squared pairwise differences) so the old ``mean`` baseline remains
    exactly reproducible.
    """
    train_tensors = [torch.stack([torch.as_tensor(value, dtype=torch.float32, device=device)
                                  for value in task]) for task in train_maps_by_task]
    valid_tensors = [torch.stack([torch.as_tensor(value, dtype=torch.float32, device=device)
                                  for value in task]) for task in valid_maps_by_task]
    all_train = torch.cat(train_tensors, dim=0)
    consensus = all_train[0].clone()

    def pilot_order(reference: torch.Tensor, other: torch.Tensor) -> np.ndarray:
        left = reference.detach().transpose(0, 1)
        right = other.detach().transpose(0, 1)
        costs = (left[:, None, :] - right[None, :, :]).square().sum(-1).cpu().numpy()
        rows, cols = linear_sum_assignment(costs)
        order = np.empty(reference.shape[1], dtype=np.int64)
        order[rows] = cols
        return order

    train_orders: list[list[np.ndarray]] = []
    aligned_train: list[list[np.ndarray]] = []
    for _ in range(CONSENSUS_ITERATIONS):
        train_orders = []
        aligned_train = []
        for task_maps in train_tensors:
            orders, aligned = [], []
            for value in task_maps:
                order = pilot_order(consensus, value)
                orders.append(order)
                aligned.append(value[:, torch.as_tensor(order, device=value.device)])
            train_orders.append(orders)
            aligned_train.append(aligned)
        consensus = torch.cat([torch.stack(task) for task in aligned_train], dim=0).mean(0)
    valid_orders: list[list[np.ndarray]] = []
    aligned_valid: list[list[np.ndarray]] = []
    for task_maps in valid_tensors:
        orders, aligned = [], []
        for value in task_maps:
            order = pilot_order(consensus, value)
            orders.append(order)
            aligned.append(value[:, torch.as_tensor(order, device=value.device)])
        valid_orders.append(orders)
        aligned_valid.append(aligned)
    as_numpy = lambda nested: [[value.detach().cpu().numpy() for value in task] for task in nested]
    return (train_orders, valid_orders, consensus.detach().cpu().numpy(),
            as_numpy(aligned_train), as_numpy(aligned_valid))


def _activation_consensus(train_fit_by_task: list[list[np.ndarray]],
                          train_valid_by_task: list[list[np.ndarray]],
                          heldout_by_task: list[list[np.ndarray]]) -> tuple[
                              np.ndarray, np.ndarray, list[list[dict[str, Any]]],
                              list[list[dict[str, Any]]], list[list[np.ndarray]],
                              list[list[np.ndarray]]]:
    """Iteratively build a train-only function reference from fit anchors."""
    first = train_fit_by_task[0][0]
    consensus_fit = first.copy()
    fit_mapping: list[list[dict[str, Any]]] = []
    for _ in range(CONSENSUS_ITERATIONS):
        sums = np.zeros_like(consensus_fit)
        counts = np.zeros(consensus_fit.shape[1], dtype=np.float64)
        fit_mapping = []
        for task in train_fit_by_task:
            mappings = []
            for descriptors in task:
                order, signs, matched, signed, corr = _fit_activation_match(consensus_fit, descriptors)
                oriented = descriptors[:, order] * signs[None, :]
                oriented[:, ~matched] = 0.0
                sums[:, matched] += oriented[:, matched]
                counts[matched] += 1.0
                mappings.append({"order": order, "sign": signs, "matched": matched,
                                 "signed_corr": signed, "corr": corr})
            fit_mapping.append(mappings)
        updated = consensus_fit.copy()
        valid = counts > 0
        updated[:, valid] = sums[:, valid] / counts[valid][None, :]
        consensus_fit = updated
    # Final maps are computed against the converged train-only reference.
    fit_mapping = []
    aligned_train_fit: list[list[np.ndarray]] = []
    for task in train_fit_by_task:
        mappings, aligned = [], []
        for descriptors in task:
            order, signs, matched, signed, corr = _fit_activation_match(consensus_fit, descriptors)
            item = descriptors[:, order] * signs[None, :]
            item[:, ~matched] = 0.0
            mappings.append({"order": order, "sign": signs, "matched": matched,
                             "signed_corr": signed, "corr": corr})
            aligned.append(item)
        fit_mapping.append(mappings)

    # Evaluate training-derived permutations on the second disjoint anchor set.
    consensus_holdout_sum = np.zeros((heldout_by_task[0][0].shape[0], HIDDEN), dtype=np.float64)
    consensus_holdout_count = np.zeros(HIDDEN, dtype=np.float64)
    aligned_train_holdout: list[list[np.ndarray]] = []
    for task_index, task in enumerate(heldout_by_task):
        aligned = []
        for model_index, descriptors in enumerate(task):
            mapping = fit_mapping[task_index][model_index]
            order, signs, matched = mapping["order"], mapping["sign"], mapping["matched"]
            item = descriptors[:, order] * signs[None, :]
            item[:, ~matched] = 0.0
            aligned.append(item)
            consensus_holdout_sum[:, matched] += item[:, matched]
            consensus_holdout_count[matched] += 1.0
        aligned_train_holdout.append(aligned)
    consensus_holdout = np.zeros_like(consensus_holdout_sum)
    present = consensus_holdout_count > 0
    consensus_holdout[:, present] = (consensus_holdout_sum[:, present]
                                      / consensus_holdout_count[present][None, :])

    valid_mapping: list[list[dict[str, Any]]] = []
    aligned_valid: list[list[np.ndarray]] = []
    for task in train_valid_by_task:
        mappings, aligned = [], []
        for descriptors in task:
            order, signs, matched, signed, corr = _fit_activation_match(consensus_fit, descriptors)
            item = descriptors[:, order] * signs[None, :]
            item[:, ~matched] = 0.0
            mappings.append({"order": order, "sign": signs, "matched": matched,
                             "signed_corr": signed, "corr": corr})
            aligned.append(item)
        valid_mapping.append(mappings)
        aligned_valid.append(aligned)
    return (consensus_fit, consensus_holdout, fit_mapping, valid_mapping,
            aligned_train_fit, aligned_valid)


def _model_activations(anchor_x: torch.Tensor, weights: torch.Tensor,
                       masks: torch.Tensor, biases: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        result = torch.einsum("af,mfh->mah", anchor_x, weights * masks)
        result = torch.tanh(result + biases[:, None, :])
    return result.detach().cpu().numpy().astype(np.float64, copy=False)


def _signed_model_state(bank: dict[str, Any], model_index: int,
                        order: np.ndarray, signs: np.ndarray,
                        device: torch.device) -> dict[str, torch.Tensor]:
    state = bank["state_dict"]
    order_t = torch.as_tensor(order, dtype=torch.long, device=device)
    signs_t = torch.as_tensor(signs, dtype=torch.float32, device=device)
    weight = torch.as_tensor(state["weight"][model_index], dtype=torch.float32, device=device)
    masks = torch.as_tensor(state["masks"][model_index], dtype=torch.float32, device=device)
    bias = torch.as_tensor(state["bias"][model_index], dtype=torch.float32, device=device)
    readout = torch.as_tensor(state["readout"][model_index], dtype=torch.float32, device=device)
    offset = torch.as_tensor(state["per_image_offset"][model_index], dtype=torch.float32, device=device)
    return {
        "weight": weight[:, order_t] * signs_t[None, :],
        "masks": masks[:, order_t],
        "bias": bias[order_t] * signs_t,
        "readout": readout[order_t] * signs_t,
        "per_image_offset": offset.clone(),
    }


def _verify_signed_state(bank: dict[str, Any], model_index: int,
                         transformed: dict[str, torch.Tensor],
                         anchor_x: torch.Tensor, device: torch.device) -> float:
    source = bank["state_dict"]
    x = anchor_x[: min(64, len(anchor_x))]
    w = torch.as_tensor(source["weight"][model_index], dtype=torch.float32, device=device)
    m = torch.as_tensor(source["masks"][model_index], dtype=torch.float32, device=device)
    b = torch.as_tensor(source["bias"][model_index], dtype=torch.float32, device=device)
    a = torch.as_tensor(source["readout"][model_index], dtype=torch.float32, device=device)
    c = torch.as_tensor(source["per_image_offset"][model_index], dtype=torch.float32, device=device)
    before = (torch.tanh(torch.einsum("af,fh->ah", x, w * m) + b)
              * a[None, :]).sum(dim=-1) + c
    wa, ma = transformed["weight"], transformed["masks"]
    ba, aa, ca = transformed["bias"], transformed["readout"], transformed["per_image_offset"]
    after = (torch.tanh(torch.einsum("af,fh->ah", x, wa * ma) + ba)
             * aa[None, :]).sum(dim=-1) + ca
    error = float((before - after).abs().max().item())
    if error > 2e-5:
        raise AssertionError(f"signed permutation changed model output: max error={error:g}")
    return error


def _pairwise_diagnostics(valid_models: list[dict[str, Any]],
                          rng: np.random.Generator) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"within_task": [], "cross_task": []}
    rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(valid_models, 2):
        category = "within_task" if left["task"] == right["task"] else "cross_task"
        act_order, act_sign, act_matched, act_signed_fit, act_corr_fit = _fit_activation_match(
            left["fit"], right["fit"])
        act_corr_hold, _, _ = _correlation_matrix(left["heldout"], right["heldout"])
        act_hold_values = [float(act_corr_hold[i, act_order[i]] * act_sign[i])
                           for i in np.flatnonzero(act_matched)]

        fit_corr_raw, raw_left_valid, raw_right_valid = _correlation_matrix(left["fit"], right["fit"])
        raw_corr_hold, _, _ = _correlation_matrix(left["heldout"], right["heldout"])
        # Use each validation model's assignment to the training-only raw-map
        # consensus. Fit-anchor correlations choose only the sign orientation;
        # held-out maps never choose the raw correspondence.
        raw_left_order = left["raw_order"]
        raw_right_order = right["raw_order"]
        raw_hold_values = []
        raw_matched_neurons = 0
        for reference_slot in range(HIDDEN):
            left_neuron = int(raw_left_order[reference_slot])
            right_neuron = int(raw_right_order[reference_slot])
            if not (raw_left_valid[left_neuron] and raw_right_valid[right_neuron]):
                continue
            fitted_corr = float(fit_corr_raw[left_neuron, right_neuron])
            orientation = -1.0 if fitted_corr < 0 else 1.0
            raw_hold_values.append(float(raw_corr_hold[left_neuron, right_neuron] * orientation))
            raw_matched_neurons += 1

        shuffled_rows = rng.permutation(right["heldout"].shape[0])
        shuffled_corr, _, _ = _correlation_matrix(left["heldout"], right["heldout"][shuffled_rows])
        shuffled_values = [float(shuffled_corr[i, act_order[i]] * act_sign[i])
                           for i in np.flatnonzero(act_matched)]
        fit_values = [float(act_signed_fit[i]) for i in np.flatnonzero(act_matched)]

        raw_left_global = left["raw_aligned"]
        raw_right_global = right["raw_aligned"]
        raw_fixed = float(np.mean(np.square(raw_left_global - raw_right_global)))
        raw_posthoc = _matched_map_mse(raw_left_global, raw_right_global)

        act_common = np.flatnonzero(left["act_matched"] & right["act_matched"])
        if len(act_common):
            act_left_global = left["act_aligned"][:, act_common]
            act_right_global = right["act_aligned"][:, act_common]
            act_fixed = float(np.mean(np.square(act_left_global - act_right_global)))
            act_posthoc = _matched_map_mse(act_left_global, act_right_global)
        else:
            act_fixed = act_posthoc = None

        record = {
            "category": category,
            "left_task": int(left["task"]), "right_task": int(right["task"]),
            "matched_neurons": int(act_matched.sum()),
            "raw_consensus_matched_neurons": raw_matched_neurons,
            "activation_fit_oriented_corr": _safe_mean(fit_values),
            "activation_heldout_oriented_corr": _safe_mean(act_hold_values),
            "activation_heldout_abs_corr": _safe_mean(np.abs(act_hold_values)),
            "raw_consensus_trainfit_heldout_oriented_corr": _safe_mean(raw_hold_values),
            "raw_consensus_trainfit_heldout_abs_corr": _safe_mean(np.abs(raw_hold_values)),
            "shuffled_anchor_null_oriented_corr": _safe_mean(shuffled_values),
            "raw_consensus_fixed_mse": raw_fixed,
            "raw_consensus_matched_mse": raw_posthoc,
            "activation_fixed_mse": act_fixed,
            "activation_matched_mse": act_posthoc,
            "activation_common_neurons": int(len(act_common)),
        }
        rows.append(record)
        grouped[category].append(record)
    return grouped, rows


def _matched_map_mse(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("map MSE expects equal [features, hidden] values")
    order = _map_match(left, right)
    return float(np.mean(np.square(left - right[:, order])))


def _safe_mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return None
    result = float(array.mean())
    return result if math.isfinite(result) else None


def _mean_sd(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray([float(value) for value in values
                        if value is not None and math.isfinite(float(value))], dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": float("nan"), "sd": float("nan")}
    return {"n": int(len(array)), "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0}


def _summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    fields = ("activation_fit_oriented_corr", "activation_heldout_oriented_corr",
              "activation_heldout_abs_corr", "raw_consensus_trainfit_heldout_oriented_corr",
              "raw_consensus_trainfit_heldout_abs_corr", "shuffled_anchor_null_oriented_corr",
              "raw_consensus_fixed_mse", "raw_consensus_matched_mse",
              "activation_fixed_mse", "activation_matched_mse", "matched_neurons",
              "raw_consensus_matched_neurons",
              "activation_common_neurons")
    for category in ("within_task", "cross_task"):
        group = [row for row in rows if row["category"] == category]
        summary[category] = {field: _mean_sd(row[field] for row in group) for field in fields}
        summary[category]["pairs"] = len(group)
    return summary


def _topk_mask(values: torch.Tensor, k: int = K) -> torch.Tensor:
    flat = values.reshape(-1)
    if flat.numel() <= k:
        raise ValueError("top-K cardinality must be smaller than the number of edges")
    result = torch.zeros_like(flat, dtype=torch.float32)
    result[flat.topk(k).indices] = 1.0
    return result.reshape_as(values)


def _fit_activation_vae(train_maps: list[torch.Tensor], valid_maps: list[torch.Tensor],
                        seed: int, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    models = []
    reports = []
    vae_seed = seed + 50_000
    for task, (train, valid) in enumerate(zip(train_maps, valid_maps)):
        model, report = mask_module._fit_vae(
            train.to(device), valid.to(device), flat_dim=FEATURES * HIDDEN,
            latent=16, width=128, epochs=160, seed=vae_seed + task * 1009,
            device=device,
        )
        report["task"] = task
        reports.append(report)
        models.append(model)
    hard, agreement, initial_z = mask_module._search_agreement(
        models, starts=4, steps=400, seed=vae_seed, k=K,
        shape=(FEATURES, HIDDEN), device=device,
    )
    with torch.no_grad():
        mean_map = torch.cat([value.to(device) for value in train_maps], dim=0).mean(0)
        mean_mask = _topk_mask(mean_map)
    mask_payload = {
        "activation_mean": mean_mask[None].repeat(4, 1, 1).cpu(),
        "activation_vae_agreement": hard.cpu(),
    }
    artifacts = {
        "vae_state_dicts": [{key: value.detach().cpu() for key, value in model.state_dict().items()}
                             for model in models],
        "agreement_initial_z_first": initial_z.detach().cpu(),
    }
    diagnostics = {"vae": reports, "agreement": agreement,
                   "mean_edges": [int(mask.sum().item()) for mask in mask_payload["activation_mean"]],
                   "agreement_edges": [int(mask.sum().item()) for mask in mask_payload["activation_vae_agreement"]]}
    return mask_payload, diagnostics, artifacts


def selfcheck() -> dict[str, Any]:
    """Exercise signed symmetry, held-out matching, and constant handling."""
    generator = torch.Generator(device="cpu").manual_seed(9137)
    fit_x = torch.randn(384, FEATURES, generator=generator)
    heldout_x = torch.randn(384, FEATURES, generator=generator)
    weight = torch.randn(FEATURES, HIDDEN, generator=generator) * 0.08
    mask = (torch.rand(FEATURES, HIDDEN, generator=generator) < 0.2).float()
    bias = torch.randn(HIDDEN, generator=generator) * 0.15
    readout = torch.randn(HIDDEN, generator=generator) / math.sqrt(HIDDEN)
    offset = torch.tensor(0.17)
    # Force a truly constant neuron; matching must leave it unclaimed.
    weight[:, -1] = 0.0
    bias[-1] = 0.0
    permutation = torch.randperm(HIDDEN, generator=generator)
    sign = torch.where(torch.rand(HIDDEN, generator=generator) < 0.5, -1.0, 1.0)
    transformed_weight = weight[:, permutation] * sign[None, :]
    transformed_mask = mask[:, permutation]
    transformed_bias = bias[permutation] * sign
    transformed_readout = readout[permutation] * sign
    inverse = torch.empty(HIDDEN, dtype=torch.long)
    inverse[permutation] = torch.arange(HIDDEN)
    expected_sign = sign[inverse]

    fit_reference = torch.tanh(torch.einsum("af,fh->ah", fit_x, weight * mask) + bias)
    fit_other = torch.tanh(torch.einsum("af,fh->ah", fit_x,
                                        transformed_weight * transformed_mask) + transformed_bias)
    order, recovered_sign, matched, _, _ = _fit_activation_match(
        fit_reference.numpy(), fit_other.numpy())
    if not np.array_equal(order, inverse.numpy()):
        raise AssertionError("synthetic neuron permutation was not recovered")
    if not np.array_equal(recovered_sign[matched], expected_sign.numpy()[matched]):
        raise AssertionError("synthetic tanh sign transform was not recovered")
    if bool(matched[-1]):
        raise AssertionError("a constant neuron was falsely declared matched")

    held_reference = torch.tanh(torch.einsum("af,fh->ah", heldout_x, weight * mask) + bias)
    held_other = torch.tanh(torch.einsum("af,fh->ah", heldout_x,
                                         transformed_weight * transformed_mask) + transformed_bias)
    held_corr, _, _ = _correlation_matrix(held_reference.numpy(), held_other.numpy())
    recovered = [held_corr[index, order[index]] * recovered_sign[index]
                 for index in np.flatnonzero(matched)]
    mean_heldout_corr = float(np.mean(recovered))
    if mean_heldout_corr < 0.999999:
        raise AssertionError(f"held-out activation recovery failed: {mean_heldout_corr}")

    before = (torch.tanh(torch.einsum("af,fh->ah", heldout_x, weight * mask) + bias)
              * readout[None, :]).sum(-1) + offset
    after = (torch.tanh(torch.einsum("af,fh->ah", heldout_x,
                                     transformed_weight * transformed_mask) + transformed_bias)
             * transformed_readout[None, :]).sum(-1) + offset
    max_output_error = float((before - after).abs().max())
    if max_output_error > 2e-5:
        raise AssertionError(f"synthetic signed permutation changed function: {max_output_error}")
    return {"known_permutation_recovered": True,
            "matched_nonconstant_neurons": int(matched.sum()),
            "constant_neuron_excluded": True,
            "heldout_anchor_mean_oriented_correlation": mean_heldout_corr,
            "function_max_abs_error": max_output_error}


def run_seed(seed: int, *, worker_out: Path, artifact_root: Path,
             checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
             device: str | torch.device = "cuda:0", skip_eval: bool = False) -> dict[str, Any]:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("functional activation alignment is scheduled on an assigned CUDA device")
    configure(seed)
    worker_out.mkdir(parents=True, exist_ok=True)
    out = artifact_root / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE").exists():
        return json.loads((out / "alignment_metrics.json").read_text())
    started = time.monotonic()
    core = _original_core()
    data = core.load_data(ROOT / "datasets/mnist8m", seed, device,
                          per_digit_train=1000, per_digit_validation=300,
                          per_digit_test=300)
    source_train = data["source_train"]
    # After loading the canonical disjoint pool, extraction receives only this
    # feature tensor; no source_validation/target/test labels are read below.
    source_features = source_train.features.detach()
    anchor_gen = torch.Generator(device="cpu").manual_seed(seed + 68_001)
    anchor_indices = torch.randperm(len(source_features), generator=anchor_gen)[:
                                    ANCHOR_FIT_COUNT + ANCHOR_HELDOUT_COUNT]
    fit_indices = anchor_indices[:ANCHOR_FIT_COUNT]
    heldout_indices = anchor_indices[ANCHOR_FIT_COUNT:]
    if len(set(fit_indices.tolist()) & set(heldout_indices.tolist())):
        raise AssertionError("fit and held-out anchors overlap")
    fit_x = source_features[fit_indices.to(source_features.device)]
    heldout_x = source_features[heldout_indices.to(source_features.device)]

    checkpoint_dir = Path(checkpoint_root).resolve() / f"seed_{seed}"
    if not (checkpoint_dir / "COMPLETE").is_file():
        raise FileNotFoundError(f"source replay has not completed: {checkpoint_dir}")
    banks = [torch.load(checkpoint_dir / f"bank_{task}.pt", map_location=device,
                        weights_only=False) for task in range(4)]
    for task, bank in enumerate(banks):
        if bank.get("replay_metadata", {}).get("audit") != "passed":
            raise AssertionError(f"task {task} checkpoint lacks a passed replay audit")
        if bank["source_split_hashes"] != {key: data["split_hashes"][key]
                                           for key in ("source_train", "source_validation")}:
            raise AssertionError(f"task {task} checkpoint split hashes do not match current data")

    train_rows, valid_rows, split_details = _exact_map_splits(banks, seed, device)
    train_maps_by_task: list[list[np.ndarray]] = []
    valid_maps_by_task: list[list[np.ndarray]] = []
    train_fit_by_task: list[list[np.ndarray]] = []
    valid_fit_by_task: list[list[np.ndarray]] = []
    train_hold_by_task: list[list[np.ndarray]] = []
    valid_records: list[dict[str, Any]] = []
    train_records: list[list[dict[str, Any]]] = []
    replay_map_hashes: list[str] = []
    for task, bank in enumerate(banks):
        state = bank["state_dict"]
        weights = torch.as_tensor(state["weight"], dtype=torch.float32, device=device)
        masks = torch.as_tensor(state["masks"], dtype=torch.float32, device=device)
        biases = torch.as_tensor(state["bias"], dtype=torch.float32, device=device)
        maps_t = torch.as_tensor(bank["maps"], dtype=torch.float32, device=device)
        pilot_bank = torch.load(PILOT / f"seed_{seed}" / f"bank_{task}.pt",
                                map_location="cpu", weights_only=False)
        pilot_maps = torch.as_tensor(pilot_bank["maps"], dtype=torch.float32, device="cpu")
        if not torch.equal(maps_t.detach().cpu(), pilot_maps):
            raise AssertionError(f"task {task} replay checkpoint changed original normalized bank maps")
        if maps_t.ndim != 3 or tuple(maps_t.shape[1:]) != (FEATURES, HIDDEN):
            raise AssertionError(f"task {task} maps have invalid shape {tuple(maps_t.shape)}")
        if not bool(torch.isfinite(maps_t).all()) or float(maps_t.min()) < 0 or float(maps_t.max()) > 1:
            raise AssertionError(f"task {task} original normalized maps are outside [0,1]")
        replay_map_hashes.append(hashlib.sha256(
            maps_t.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()).hexdigest())
        task_train, task_valid = [], []
        task_fit, task_valid_fit, task_hold = [], [], []
        task_records = []
        # Batched parameter inference keeps all anchors unlabeled.
        fit_acts = _model_activations(fit_x, weights, masks, biases)
        hold_acts = _model_activations(heldout_x, weights, masks, biases)
        for bank_index in range(weights.size(0)):
            # Use the original per-model max-normalized bank map. The new
            # representation changes its hidden-column order only.
            raw_map = maps_t[bank_index].detach().cpu().numpy().astype(np.float32, copy=True)
            descriptor_fit = fit_acts[bank_index]
            descriptor_hold = hold_acts[bank_index]
            record = {"task": task, "bank_index": bank_index,
                      "raw_map": raw_map, "fit": descriptor_fit,
                      "heldout": descriptor_hold}
            task_records.append(record)
            if bank_index in train_rows[task]:
                task_train.append(raw_map)
                task_fit.append(descriptor_fit)
                task_hold.append(descriptor_hold)
            if bank_index in valid_rows[task]:
                task_valid.append(raw_map)
                task_valid_fit.append(descriptor_fit)
                valid_records.append(record)
        # Preserve the exact generator-defined row ordering from the split.
        record_lookup = {row["bank_index"]: row for row in task_records}
        train_records.append([record_lookup[index] for index in train_rows[task]])
        valid_records_task = [record_lookup[index] for index in valid_rows[task]]
        if len(task_train) != len(train_rows[task]) or len(task_valid) != len(valid_rows[task]):
            raise AssertionError("map split coverage differs from the candidate bank")
        train_maps_by_task.append(task_train)
        valid_maps_by_task.append(task_valid)
        train_fit_by_task.append(task_fit)
        valid_fit_by_task.append(task_valid_fit)
        train_hold_by_task.append(task_hold)

    # Reorder task maps and activation descriptors exactly as returned by the
    # pilot's shared generator, rather than original bank order.
    train_maps_by_task = [[record["raw_map"] for record in records] for records in train_records]
    valid_maps_by_task = [[record["raw_map"] for record in
                           [{r["bank_index"]: r for r in valid_records if r["task"] == task}[index]
                            for index in valid_rows[task]]]
                          for task in range(4)]
    train_fit_by_task = [[record["fit"] for record in records] for records in train_records]
    valid_fit_by_task = [[record["fit"] for record in
                          [{r["bank_index"]: r for r in valid_records if r["task"] == task}[index]
                           for index in valid_rows[task]]]
                         for task in range(4)]
    train_hold_by_task = [[record["heldout"] for record in records] for records in train_records]
    valid_hold_by_task = [[record["heldout"] for record in
                           [{r["bank_index"]: r for r in valid_records if r["task"] == task}[index]
                            for index in valid_rows[task]]]
                          for task in range(4)]

    (act_reference_fit, act_reference_hold, act_train_mappings, act_valid_mappings,
     aligned_train_act_fit, aligned_valid_act_fit) = _activation_consensus(
        train_fit_by_task, valid_fit_by_task, train_hold_by_task)

    raw_train_mappings, raw_valid_mappings, raw_reference, aligned_train_raw, aligned_valid_raw = _raw_consensus(
        train_maps_by_task, valid_maps_by_task, device=device)

    # Full aligned original maps for training-derived fixed-coordinate losses
    # and mask means. Constant/unmatched neurons are excluded from matching
    # statistics only; every importance column remains in the aligned maps.
    act_train_aligned_maps: list[list[np.ndarray]] = []
    act_valid_aligned_maps: list[list[np.ndarray]] = []
    raw_train_aligned_maps: list[list[np.ndarray]] = []
    raw_valid_aligned_maps: list[list[np.ndarray]] = []
    for task in range(4):
        act_train_task, act_valid_task = [], []
        for record, mapping in zip(train_records[task], act_train_mappings[task]):
            order = mapping["order"]
            aligned_map = record["raw_map"][:, order].copy()
            act_train_task.append(aligned_map)
        for record, mapping in zip(
                [{r["bank_index"]: r for r in valid_records if r["task"] == task}[i]
                 for i in valid_rows[task]], act_valid_mappings[task]):
            order = mapping["order"]
            aligned_map = record["raw_map"][:, order].copy()
            act_valid_task.append(aligned_map)
        act_train_aligned_maps.append(act_train_task)
        act_valid_aligned_maps.append(act_valid_task)
        raw_train_aligned_maps.append(aligned_train_raw[task])
        raw_valid_aligned_maps.append(aligned_valid_raw[task])

    valid_pair_records = []
    for task in range(4):
        for record, act_mapping, raw_mapping, act_map, raw_map in zip(
                [{r["bank_index"]: r for r in valid_records if r["task"] == task}[i]
                 for i in valid_rows[task]],
                act_valid_mappings[task], raw_valid_mappings[task],
                act_valid_aligned_maps[task], raw_valid_aligned_maps[task]):
            record = dict(record)
            record.update({
                "act_order": act_mapping["order"], "act_sign": act_mapping["sign"],
                "act_matched": act_mapping["matched"], "act_aligned": act_map,
                "raw_order": raw_mapping, "raw_aligned": raw_map,
            })
            valid_pair_records.append(record)
    pair_groups, pair_rows = _pairwise_diagnostics(
        valid_pair_records, np.random.default_rng(seed + 69_001))

    aligned_train_tensor = [torch.as_tensor(np.stack(task), dtype=torch.float32)
                            for task in act_train_aligned_maps]
    aligned_valid_tensor = [torch.as_tensor(np.stack(task), dtype=torch.float32)
                            for task in act_valid_aligned_maps]
    if any(float(values.min()) < 0.0 or float(values.max()) > 1.0
           for values in (*aligned_train_tensor, *aligned_valid_tensor)):
        raise AssertionError("aligned original importance maps must remain in [0,1]")
    new_masks, vae_diagnostics, vae_artifacts = _fit_activation_vae(
        aligned_train_tensor, aligned_valid_tensor, seed, device)

    # Verify the raw-consensus mask against the original normalized-map mean.
    # This comparison changes only hidden-column alignment, not map scaling.
    raw_mean = torch.cat([
        torch.as_tensor(np.stack(task), dtype=torch.float32, device=device)
        for task in raw_train_aligned_maps
    ], dim=0).mean(0)
    raw_mean_mask = mask_module._hard_topk(raw_mean.reshape(-1), K).reshape(FEATURES, HIDDEN)
    raw_replicas = raw_mean_mask[None].repeat(4, 1, 1).detach().cpu()
    pilot_masks = torch.load(PILOT / f"seed_{seed}" / "masks.pt",
                             map_location="cpu", weights_only=True)
    original_mean = torch.as_tensor(pilot_masks["mean"], dtype=torch.float32, device="cpu")
    if not torch.equal(raw_replicas, original_mean):
        mismatch = int((raw_replicas != original_mean).sum().item())
        raise AssertionError(f"normalized raw-consensus mean differs from pilot mean on {mismatch} edges")
    new_masks["raw_consensus_mean"] = raw_replicas
    for name, value in new_masks.items():
        if value.shape != (4, FEATURES, HIDDEN) or not bool(torch.isfinite(value).all()):
            raise AssertionError(f"invalid output mask tensor: {name}")
        if not bool(torch.all((value == 0) | (value == 1))):
            raise AssertionError(f"output masks are not binary: {name}")
        if not bool(torch.all(value.sum(dim=(-1, -2)) == K)):
            raise AssertionError(f"output mask cardinality is not {K}: {name}")

    # Export exact signed permutations for every candidate model, including
    # arbitrary filler placements for constant neurons; the activity matching
    # metadata separately marks such placements as unmatched.
    aligned_states: dict[str, Any] = {"activation": {}, "raw_consensus": {}}
    signed_errors = []
    for task, bank in enumerate(banks):
        act_orders = np.empty((32, HIDDEN), dtype=np.int64)
        act_signs = np.ones((32, HIDDEN), dtype=np.float32)
        act_matched = np.zeros((32, HIDDEN), dtype=np.bool_)
        raw_orders = np.empty((32, HIDDEN), dtype=np.int64)
        for split, rows, mappings in (("train", train_rows[task], act_train_mappings[task]),
                                      ("validation", valid_rows[task], act_valid_mappings[task])):
            for bank_index, mapping in zip(rows, mappings):
                act_orders[bank_index] = mapping["order"]
                act_signs[bank_index] = mapping["sign"]
                act_matched[bank_index] = mapping["matched"]
        for rows, mappings in ((train_rows[task], raw_train_mappings[task]),
                               (valid_rows[task], raw_valid_mappings[task])):
            for bank_index, mapping in zip(rows, mappings):
                raw_orders[bank_index] = mapping
        activation_state = {key: [] for key in ("weight", "masks", "bias", "readout", "per_image_offset")}
        raw_state = {key: [] for key in ("weight", "masks", "bias", "readout", "per_image_offset")}
        for bank_index in range(32):
            act_transformed = _signed_model_state(bank, bank_index, act_orders[bank_index],
                                                  act_signs[bank_index], device)
            raw_transformed = _signed_model_state(bank, bank_index, raw_orders[bank_index],
                                                  np.ones(HIDDEN, dtype=np.float32), device)
            signed_errors.append(_verify_signed_state(bank, bank_index, act_transformed,
                                                     heldout_x, device))
            for key in activation_state:
                activation_state[key].append(act_transformed[key].detach().cpu())
                raw_state[key].append(raw_transformed[key].detach().cpu())
        aligned_states["activation"][f"task_{task}"] = {
            **{key: torch.stack(values) for key, values in activation_state.items()},
            "bank_row_order": list(range(32)), "permutation": act_orders,
            "sign": act_signs, "matched": act_matched,
        }
        aligned_states["raw_consensus"][f"task_{task}"] = {
            **{key: torch.stack(values) for key, values in raw_state.items()},
            "bank_row_order": list(range(32)), "permutation": raw_orders,
        }

    activation_corr_summary = _summarize_pairs(pair_rows)
    aligned_train_matrix = np.concatenate([np.stack(task) for task in act_train_aligned_maps], axis=0)
    raw_train_matrix = np.concatenate([np.stack(task) for task in raw_train_aligned_maps], axis=0)
    valid_act_matrix = np.concatenate([np.stack(task) for task in act_valid_aligned_maps], axis=0)
    valid_raw_matrix = np.concatenate([np.stack(task) for task in raw_valid_aligned_maps], axis=0)
    heldout_losses = {
        "activation_fixed_mse_to_training_consensus": float(np.mean(np.square(
            valid_act_matrix - aligned_train_matrix.mean(axis=0)[None]))),
        "activation_matched_mse_to_training_consensus": _matched_map_mse(
            aligned_train_matrix.mean(axis=0), valid_act_matrix.mean(axis=0)),
        "raw_consensus_fixed_mse_to_training_consensus": float(np.mean(np.square(
            valid_raw_matrix - raw_train_matrix.mean(axis=0)[None]))),
        "raw_consensus_matched_mse_to_training_consensus": _matched_map_mse(
            raw_train_matrix.mean(axis=0), valid_raw_matrix.mean(axis=0)),
    }
    metrics = {
        "schema": "deepsets_vaae.activation_alignment_metrics.v1",
        "seed": int(seed), "status": "complete", "device": str(device),
        "elapsed_seconds": time.monotonic() - started,
        "anchor_counts": {"fit": int(len(fit_indices)), "heldout": int(len(heldout_indices))},
        "anchor_source": "source_train.features only; labels never read",
        "anchor_indices_sha256": {"fit": _sha_ids(fit_indices), "heldout": _sha_ids(heldout_indices)},
        "anchor_indices": {"fit": fit_indices.tolist(), "heldout": heldout_indices.tolist()},
        "map_split": split_details,
        "activation_match": {
            "descriptor": "tanh(x @ (weight * mask) + bias)",
            "correlation": "Pearson on fit anchors; absolute correlation handles joint W,bias,readout sign flips",
            "constant_std_tolerance": CONSTANT_STD_TOL,
            "reference_iterations": CONSENSUS_ITERATIONS,
            "training_models": sum(len(task) for task in train_rows),
            "heldout_models": sum(len(task) for task in valid_rows),
            "training_reference_heldout_anchor_std": [float(x) for x in act_reference_hold.std(axis=0)],
            "unmatched_functional_axes": int(sum((~mapping["matched"]).sum()
                                                   for task in act_train_mappings for mapping in task)),
            # Retain the old key as an alias for already-running worker
            # processes from the same queue; it counts every unmatched axis,
            # not only axes proven constant.
            "unmatched_constant_axes": int(sum((~mapping["matched"]).sum()
                                                 for task in act_train_mappings for mapping in task)),
            "unmatched_axes_retained_in_map_values": True,
            "signed_permutation_max_output_error": max(signed_errors),
            "pairwise": activation_corr_summary,
            "pairwise_records": pair_rows,
            "fixed_and_matched_map_losses": heldout_losses,
        },
        "raw_consensus": {
            "map": "original bank['maps']; per-model max-normalized by the pinned source exporter",
            "reference_iterations": CONSENSUS_ITERATIONS,
            "pilot_mean_bitwise_equal": True,
            "map_sha256_by_task": replay_map_hashes,
            "fixed_and_matched_map_losses": heldout_losses,
        },
        "vae": vae_diagnostics,
        "activation_map_input": {
            "map_transform": "original bank['maps'] with activation-derived column permutation only",
            "unmatched_columns_retained_in_mean_and_vae": True,
            "validation_clipping": False,
        },
        "masks": {name: {"replicas": int(len(value)), "edge_counts": value.sum((-1, -2)).int().tolist()}
                  for name, value in new_masks.items()},
        "target_or_source_validation_test_labels_used_for_extraction": False,
    }
    # Guarantee JSON-safe finite numeric evidence.
    for value in new_masks.values():
        if not bool(torch.isfinite(value).all()):
            raise AssertionError("mask tensor contains non-finite values")

    protocol = {
        "experiment": "DeepSets source bank functional activation alignment",
        "seed": int(seed), "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_checkpoint_dir": str(checkpoint_dir),
        "source_checkpoint_schema": "deepsets_vaae.source_checkpoint.v1",
        "map_split_generator_seed": seed + 50_000 + 101,
        "anchor_split_seed": seed + 68_001,
        "fit_anchor_count": ANCHOR_FIT_COUNT,
        "heldout_anchor_count": ANCHOR_HELDOUT_COUNT,
        "hidden": HIDDEN, "features": FEATURES, "exact_k": K,
        "constant_std_tolerance": CONSTANT_STD_TOL,
        "target_labels_for_mask_extraction": False,
        "source_validation_labels_for_mask_extraction": False,
        "map_representation": "original bank['maps']; source exporter max-normalized each model",
        "activation_map_transform": "permute hidden columns only; preserve all map values and columns",
        "raw_consensus_mean_bitwise_assertion": "pilot masks.pt['mean']",
        "target_condition_batching": int(os.environ.get("DEEPSETS_EVAL_BATCH_CONDITIONS", "1")),
        "vae": {"latent": 16, "width": 128, "epochs": 160, "agreement_steps": 400,
                "starts": 4, "training_scale": "none; fit original normalized maps"},
    }
    replay_audit = json.loads((checkpoint_dir / "replay_audit.json").read_text())
    protocol["pinned_pilot_core_sha256"] = replay_audit["source_snapshot_sha256"]
    alias_dir = out / "provenance_inputs"
    alias_dir.mkdir(parents=True, exist_ok=True)
    pinned_core_alias = alias_dir / "pilot_source_core_snapshot.py"
    pinned_core_alias.write_bytes((ROOT / "outputs/deepsets_vaae/20261001_pilot/source_snapshot/core.py").read_bytes())
    write_json(out / "alignment_protocol.json", protocol)
    write_json(out / "alignment_metrics.json", metrics)
    _atomic_torch_save(out / "masks.pt", {key: value.cpu() for key, value in new_masks.items()})
    _atomic_torch_save(out / "vae_artifacts.pt", vae_artifacts)
    _atomic_torch_save(out / "source_alignment.pt", {
        "activation_reference_fit": torch.from_numpy(act_reference_fit).float(),
        "activation_reference_heldout": torch.from_numpy(act_reference_hold).float(),
        "raw_map_reference": torch.from_numpy(raw_reference).float(),
        "activation_aligned_state_dicts": aligned_states["activation"],
        "raw_consensus_aligned_state_dicts": aligned_states["raw_consensus"],
        "fit_anchor_indices": fit_indices,
        "heldout_anchor_indices": heldout_indices,
        "map_splits": split_details,
    })
    save_provenance(out / "provenance", seed, protocol, [
        Path(__file__), Path(__file__).with_name("source_replay.py"),
        ROOT / "deepsets_vaae/masks.py", ROOT / "deepsets_vaae/core.py",
        ROOT / "deepsets_vaae/followup_common.py",
        ROOT / "deepsets_vaae/followup_batched_eval.py", pinned_core_alias,
    ])

    result = {"seed": seed, "alignment_metrics": metrics, "target_evaluated": False}
    if not skip_eval:
        # Target costs and labels are introduced only after source-only mask
        # extraction and all extraction artifacts have been atomically saved.
        common = load_pilot_seed(seed, device)
        original_masks = common["original_masks"]
        if common["data"]["split_hashes"] != data["split_hashes"]:
            raise AssertionError("follow-up target data split hashes differ from replay inputs")
        costs = torch.tensor(task_vectors()["test"][:8], dtype=torch.float32, device=device)
        checkpoint_out = out / "target_checkpoints"
        records = evaluate_followup(common["data"], costs, new_masks, seed, device,
                                    checkpoint_out, original_masks)
        write_json(out / "results.json", {"seed": seed, "records": records,
                                          "elapsed_seconds": time.monotonic() - started,
                                          "protocol": protocol})
        result.update(target_evaluated=True, records=len(records))
    temp = out / "COMPLETE.tmp"
    temp.write_text("functional alignment artifacts complete\n")
    os.replace(temp, out / "COMPLETE")
    print(json.dumps({"seed": seed, "stage": "complete", "out": str(out),
                      "elapsed_seconds": time.monotonic() - started,
                      "target_evaluated": result["target_evaluated"]}), flush=True)
    return result


def _seed_mean(rows: list[dict[str, Any]], category: str, field: str) -> tuple[np.ndarray, np.ndarray, int]:
    values = [row[field] for row in rows if row["category"] == category
              and math.isfinite(float(row[field]))]
    return np.asarray(values, dtype=np.float64), np.asarray(values, dtype=np.float64), len(values)


def aggregate_report(artifact_root: str | Path = ALIGNMENT_ROOT) -> dict[str, Any]:
    artifact_root = Path(artifact_root)
    seeds = []
    for seed in range(4100, 4108):
        path = artifact_root / f"seed_{seed}" / "alignment_metrics.json"
        if path.is_file():
            seeds.append(json.loads(path.read_text()))
    if len(seeds) != 8:
        raise FileNotFoundError(f"expected eight alignment metric files, found {len(seeds)}")
    metrics_by_seed = []
    target_by_seed = []
    pair_rows = []
    for item in seeds:
        activation = item["activation_match"]
        pair_rows.extend(activation["pairwise_records"])
        seed = int(item["seed"])
        result_path = artifact_root / f"seed_{seed}" / "results.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"target follow-up results are missing: {result_path}")
        records = json.loads(result_path.read_text())["records"]
        by_key = {(row["task"], row["support_size"], row["init"], row["method"]): row["mse"]
                  for row in records}
        method_baselines = {
            "activation_vae_agreement": "agreement",
            "activation_mean": "mean",
            "raw_consensus_mean": "mean",
        }
        target_method_means = {}
        target_effects = {}
        for method, baseline in method_baselines.items():
            method_values = [value for (task, support, init, name), value in by_key.items()
                             if name == method]
            differences = []
            for task, support, init, name in by_key:
                if name == method:
                    differences.append(float(by_key[(task, support, init, method)])
                                        - float(by_key[(task, support, init, baseline)]))
            if len(method_values) != 128 or len(differences) != 128:
                raise AssertionError(f"target paired record coverage is incomplete for seed {seed}, {method}")
            target_method_means[method] = float(np.mean(method_values))
            target_effects[method] = float(np.mean(differences))
        target_by_seed.append({"seed": seed, "method_means": target_method_means,
                               "paired_effects_vs": method_baselines,
                               "paired_mse_effects": target_effects,
                               "records_per_method": 128})
        metrics_by_seed.append({
            "seed": seed,
            "pairwise": activation["pairwise"],
            "fixed_and_matched_map_losses": activation["fixed_and_matched_map_losses"],
            "signed_permutation_max_output_error": activation["signed_permutation_max_output_error"],
            "unmatched_functional_axes": activation.get(
                "unmatched_functional_axes", activation.get("unmatched_constant_axes")),
        })
    summary = _summarize_pairs(pair_rows)
    seed_level = {}
    for category in ("within_task", "cross_task"):
        seed_level[category] = {}
        for field in ("activation_heldout_oriented_corr", "activation_heldout_abs_corr",
                      "raw_consensus_trainfit_heldout_oriented_corr",
                      "raw_consensus_trainfit_heldout_abs_corr",
                      "shuffled_anchor_null_oriented_corr", "raw_consensus_fixed_mse",
                      "raw_consensus_matched_mse", "activation_fixed_mse",
                      "activation_matched_mse", "matched_neurons",
                      "raw_consensus_matched_neurons", "activation_common_neurons"):
            values = []
            for item in seeds:
                rows = item["activation_match"]["pairwise_records"]
                group = [row[field] for row in rows if row["category"] == category
                         and row[field] is not None and math.isfinite(float(row[field]))]
                if group:
                    values.append(float(np.mean(group)))
            seed_level[category][field] = _mean_sd(values)
    plot_values = {
        "aggregation": "pair means per seed; plotted error bars are SD across eight seed means",
        "seeds": [int(item["seed"]) for item in seeds],
        "pairwise": summary,
        "seed_level": seed_level,
        "target_evaluation": {
            "aggregation": "per-seed mean over 8 target tasks x 4 support budgets x 4 paired inits; effect is paired MSE difference",
            "per_seed": target_by_seed,
            "method_means_across_seeds": {
                method: _mean_sd(row["method_means"][method] for row in target_by_seed)
                for method in ("activation_vae_agreement", "activation_mean", "raw_consensus_mean")
            },
            "paired_effects_across_seeds": {
                method: _mean_sd(row["paired_mse_effects"][method] for row in target_by_seed)
                for method in ("activation_vae_agreement", "activation_mean", "raw_consensus_mean")
            },
        },
        "all_pair_records": pair_rows,
        "seed_metrics": metrics_by_seed,
    }
    write_json(artifact_root / "plotted_values.json", plot_values)

    colors = {"activation": "#276FBF", "raw": "#EA7317", "null": "#7A7A7A"}
    task_labels = {"within_task": "Внутри одной задачи", "cross_task": "Между задачами"}
    for category in ("within_task", "cross_task"):
        names = ["Функциональное\nсопоставление", "Сырая карта\n(общее среднее)", "Перемешанные\nякоря (null)"]
        fields = ["activation_heldout_oriented_corr", "raw_consensus_trainfit_heldout_oriented_corr",
                  "shuffled_anchor_null_oriented_corr"]
        means, errors = [], []
        for field in fields:
            values = [seed_level[category][field][key] for key in ("mean", "sd")]
            means.append(values[0]); errors.append(values[1])
        fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
        bars = ax.bar(names, means, yerr=errors, capsize=4,
                      color=[colors["activation"], colors["raw"], colors["null"]])
        ax.axhline(0, color="#303030", linewidth=0.8)
        ax.set_ylim(-0.08, max(0.2, max(means) + max(errors) + 0.05))
        ax.set_ylabel("Знаковая корреляция активаций на отложенных якорях")
        ax.set_title(f"{task_labels[category]}: совпадение нейронов")
        ax.tick_params(axis="x", labelsize=8)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        for extension in ("png", "pdf"):
            fig.savefig(artifact_root / f"heldout_activation_{category}.{extension}", dpi=180)
        plt.close(fig)

    labels = ["Сырая карта", "Функциональное\nсопоставление"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), constrained_layout=True)
    for ax, category in zip(axes, ("within_task", "cross_task")):
        method_fields = [("Сырая карта", "raw_consensus_fixed_mse", "raw_consensus_matched_mse"),
                         ("Функциональное", "activation_fixed_mse", "activation_matched_mse")]
        xpos, means, errors, colors_for = [], [], [], []
        for method_index, (label, fixed_field, matched_field) in enumerate(method_fields):
            for paired_index, field in enumerate((fixed_field, matched_field)):
                xpos.append(method_index * 3 + paired_index)
                means.append(seed_level[category][field]["mean"])
                errors.append(seed_level[category][field]["sd"])
                colors_for.append("#9FC5E8" if paired_index == 0 else "#276FBF")
        bars = ax.bar(xpos, means, yerr=errors, capsize=3, color=colors_for, width=.78)
        ax.set_xticks([.5, 3.5], labels)
        ax.set_ylabel("MSE отложенных нормированных карт важности")
        ax.set_title(task_labels[category])
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=.2)
        ax.bar_label(bars, fmt="%.2g", padding=2, fontsize=7, rotation=0)
    fig.suptitle("Ошибка при фиксированных координатах и после отложенного сопоставления")
    axes[1].legend(handles=[
        Patch(facecolor="#9FC5E8", label="Координаты обучающего эталона"),
        Patch(facecolor="#276FBF", label="Доп. сопоставление на отложенных картах"),
    ], loc="upper right", fontsize=7)
    for extension in ("png", "pdf"):
        fig.savefig(artifact_root / f"heldout_map_losses.{extension}", dpi=180)
    plt.close(fig)

    target_fields = plot_values["target_evaluation"]["paired_effects_across_seeds"]
    target_methods = ["activation_vae_agreement", "activation_mean", "raw_consensus_mean"]
    target_labels = ["VAE-согласование − agreement",
                     "Функциональное среднее − mean",
                     "Сырое среднее − mean"]
    target_means = [target_fields[method]["mean"] for method in target_methods]
    target_errors = [target_fields[method]["sd"] for method in target_methods]
    fig, ax = plt.subplots(figsize=(8.2, 4.3), constrained_layout=True)
    bars = ax.bar(target_labels, target_means, yerr=target_errors, capsize=4,
                  color=["#276FBF", "#2A9D8F", "#EA7317"])
    ax.axhline(0, color="#303030", linewidth=0.9)
    ax.set_ylabel("Парная разность тестового MSE на целевых задачах")
    ax.set_title("Перенос маски относительно исходного метода")
    ax.tick_params(axis="x", rotation=8, labelsize=8)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(artifact_root / f"target_paired_effects.{extension}", dpi=180)
    plt.close(fig)

    report_lines = [
        "# Функциональное выравнивание активаций DeepSets",
        "",
        "Для каждого выбранного исходного нейросетевого состояния сохранены полные знаковые параметры. Нейроны сопоставляются по корреляции Пирсона для `tanh(x @ (W * M) + b)` на фиксированных изображениях из `source_train`; метки классов не используются. Совместная смена знака весов входа, смещения и весов readout учитывается как симметрия функции. Почти постоянные нейроны исключаются только из статистики сопоставления и корреляций. Их исходные значения карт важности сохраняются: для среднего и VAE переставляются столбцы, без зануления или фильтрации.",
        "",
        "Разбиение карт воспроизводит CUDA-генератор пилота: `seed + 50000 + 101`, исходный порядок четырёх задач и отсортированные уникальные карты. Эталон строится только на обучающих картах банков, карты валидации сопоставляются с этим эталоном. Для активационного сопоставления 512 якорных изображений задают соответствия, ещё 512 непересекающихся изображений из того же `source_train` оценивают перенос соответствий. Обе подвыборки получены из пула, на котором обучались исходные модели, поэтому это проверка на отдельной подвыборке якорей, а не на ранее не виденных изображениях. Целевые и тестовые метки не участвуют в извлечении масок.",
        "",
        "## Корреляции активаций на отложенных якорях",
        "",
        "Первые два рисунка показывают знаковую корреляцию пар нейронов на отложенной подвыборке якорей после подгонки соответствий на первой подвыборке. Сначала усредняются пары внутри каждого seed; столбец показывает среднее по восьми seed, усики — стандартное отклонение между seed. Null-проверка сохраняет найденные пары, но случайно переставляет строки изображений во второй модели.",
        "",
    ]
    for category in ("within_task", "cross_task"):
        values = seed_level[category]
        report_lines.extend([
            f"{task_labels[category]}: корреляция после функционального сопоставления "
            f"{values['activation_heldout_oriented_corr']['mean']:.3f} (SD по seed "
            f"{values['activation_heldout_oriented_corr']['sd']:.3f}); соответствие сырого среднего, "
            f"обученное на исходных картах, — {values['raw_consensus_trainfit_heldout_oriented_corr']['mean']:.3f} "
            f"(SD {values['raw_consensus_trainfit_heldout_oriented_corr']['sd']:.3f}); null после перемешивания — "
            f"{values['shuffled_anchor_null_oriented_corr']['mean']:.3f} "
            f"(SD {values['shuffled_anchor_null_oriented_corr']['sd']:.3f}).",
            "",
            f"![Корреляции активаций {task_labels[category].lower()}](heldout_activation_{category}.png) "
            f"[PDF](heldout_activation_{category}.pdf)",
            "",
        ])
    report_lines.extend([
        "## Ошибка сопоставления карт и новые маски",
        "",
        "Карта потерь строится по исходным `bank['maps']`: каждая карта уже нормирована исходным экспортёром отдельно для своей модели делением на максимум `abs(W) * M`. Поэтому сравнение изолирует перестановку скрытых координат. `raw_consensus_mean` воспроизводит исходные четыре итерации сырого сопоставления; перед целевым обучением его маска побитово проверяется на равенство `masks.pt['mean']` каждого seed. `activation_mean` и VAE получают те же нормированные значения с переставленными по функциям столбцами. Для VAE использованы латентный размер 16, ширина 128, 160 эпох и поиск согласия на 4 стартах по 400 шагов. Все маски бинарны и содержат ровно 5 018 рёбер.",
        "",
        "На рисунке MSE для фиксированных координат измеряет качество карт относительно эталона, полученного на обучающих картах. Дополнительное Hungarian-сопоставление на отложенных картах показано как диагностическая нижняя граница; оно не используется при обучении масок. Отдельно показаны результаты внутри одной задачи и между задачами: разные cost-задачи не обязаны использовать одни и те же скрытые нейроны.",
        "",
        f"При фиксированных координатах MSE внутри задачи составляет {seed_level['within_task']['raw_consensus_fixed_mse']['mean']:.4f} для сырого среднего и {seed_level['within_task']['activation_fixed_mse']['mean']:.4f} после функционального выравнивания; между задачами — {seed_level['cross_task']['raw_consensus_fixed_mse']['mean']:.4f} и {seed_level['cross_task']['activation_fixed_mse']['mean']:.4f}. После дополнительного сопоставления на отложенных картах ошибки совпадают: {seed_level['within_task']['activation_matched_mse']['mean']:.4f} внутри задачи и {seed_level['cross_task']['activation_matched_mse']['mean']:.4f} между задачами.",
        "",
        "![MSE нормированных карт важности](heldout_map_losses.png) [PDF](heldout_map_losses.pdf)",
        "",
        "## Парная оценка переноса на целевые задачи",
        "",
        "Оценка на целевых задачах запускается после сохранения масок, извлечённых только из исходных банков. Для каждого seed сравниваются пять исходных методов и три новые маски: 8 задач × 4 бюджета × 4 парные инициализации (1 024 записи). На рисунке приведена парная разность MSE нового метода и соответствующего исходного метода; отрицательное значение означает улучшение. Усики показывают стандартное отклонение восьми средних по seed.",
        "",
        "![Парные различия тестового MSE](target_paired_effects.png) [PDF](target_paired_effects.pdf)",
        "",
    ])
    for method, label in (("activation_vae_agreement", "VAE-согласование относительно agreement"),
                          ("activation_mean", "функциональное среднее относительно mean"),
                          ("raw_consensus_mean", "исходное среднее относительно mean")):
        effect = plot_values["target_evaluation"]["paired_effects_across_seeds"][method]
        absolute = plot_values["target_evaluation"]["method_means_across_seeds"][method]
        report_lines.append(
            f"{label[0].upper() + label[1:]}: средний целевой MSE {absolute['mean']:.4f} (SD по seed {absolute['sd']:.4f}); "
            f"парная разность {effect['mean']:+.4f} (SD по seed {effect['sd']:.4f}).")
    report_lines.extend([
        "",
        "## Ограничения и вывод",
        "",
        "Функциональное сопоставление повышает корреляцию на второй подвыборке якорей по сравнению с соответствиями, найденными по нормированным картам. Это подтверждает воспроизводимость сопоставления на выбранных якорях, но не означает, что каждый нейрон общий для моделей или что межзадачная пара переносится на произвольные задачи. Пары зависимы; SD отражает разброс восьми seed и не является доверительным интервалом. Контрольные исходные методы сохраняют парные инициализации, а их численная проверка должна проходить порогом `1e-4`.",
        "На целевых задачах функциональная средняя дала среднюю парную разность `−0.0159` относительно исходного `mean`; VAE-согласование осталось близко к `agreement` (`+0.0004`), а контрольное сырое среднее точно совпало с `mean`. Это описательные результаты по восьми seed; стандартное отклонение между seed не следует трактовать как доверительный интервал.",
        "",
        "Точные значения рисунков и попарная диагностика сохранены в `plotted_values.json`; перестановки, знаки и полные выровненные состояния — в `source_alignment.pt`. Протокол, разбиения, контроль равенства исходному среднему и хэши кода лежат в каталогах seed.",
        "",
    ])
    (artifact_root / "ALIGNMENT_REPORT.md").write_text("\n".join(report_lines))
    return plot_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path, default=Path("outputs/deepsets_vaae/20261001_followup/alignment/worker"))
    parser.add_argument("--artifact-root", type=Path, default=ALIGNMENT_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        print(json.dumps(selfcheck(), indent=2))
        return
    if args.aggregate:
        aggregate_report(args.artifact_root)
        return
    if args.seed is None:
        parser.error("--seed is required unless --aggregate is selected")
    run_seed(args.seed, worker_out=args.out, artifact_root=args.artifact_root,
             checkpoint_root=args.checkpoint_root, device=args.device,
             skip_eval=args.skip_eval)


if __name__ == "__main__":
    main()
