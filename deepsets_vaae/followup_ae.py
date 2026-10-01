"""Deterministic autoencoder follow-up for permutation-ambiguous DeepSets maps.

Each worker reads a frozen pilot seed, reproduces its 26/6 map split, compares
three deterministic reconstruction objectives, searches source-supported codes
for exact-K agreement masks, evaluates reconstruction against held-out maps,
then calls the shared paired target-transfer evaluator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from . import masks as mask_ops
from .followup_common import (FOLLOWUP, configure, evaluate_followup,
                              load_pilot_seed, save_provenance,
                              run_cuda_queue)
from .run import task_vectors, write_json


FEATURES = 784
HIDDEN = 32
FLAT_DIM = FEATURES * HIDDEN
LATENT = 16
WIDTH = 128
TRAIN_MAPS = 26
VALIDATION_MAPS = 6
EPOCHS = 160
LEARNING_RATE = 1e-3
K = 5018
AGREEMENT_STEPS = 400
AGREEMENT_STARTS = 4
AGREEMENT_LR = .03
AGREEMENT_TEMPERATURE = .5
ANCHOR_WEIGHT = .001
SOFTNESS_THRESHOLD = .02
SOFTNESS_WEIGHT = .1
METHODS = ("ae_flat_consensus", "ae_flat_hungarian", "ae_set_hungarian")


class FlatDeterministicAE(nn.Module):
    """Deterministic counterpart of the pilot's flat VAE architecture."""

    def __init__(self, flat_dim: int = FLAT_DIM, latent: int = LATENT,
                 width: int = WIDTH) -> None:
        super().__init__()
        self.flat_dim = flat_dim
        self.latent_dim = latent
        self.encoder = nn.Sequential(nn.Linear(flat_dim, width), nn.ReLU())
        self.code = nn.Linear(width, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, width), nn.ReLU(),
                                     nn.Linear(width, flat_dim))

    def encode(self, maps: torch.Tensor) -> torch.Tensor:
        return self.code(self.encoder(maps.reshape(maps.shape[0], -1)))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(codes).reshape(-1, FEATURES, HIDDEN)

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(maps))


class SetInvariantAE(nn.Module):
    """Shared column encoder with invariant pooling and learned output slots."""

    def __init__(self, features: int = FEATURES, hidden: int = HIDDEN,
                 latent: int = LATENT, width: int = WIDTH) -> None:
        super().__init__()
        self.features = features
        self.hidden = hidden
        self.latent_dim = latent
        self.column_encoder = nn.Sequential(nn.Linear(features, width), nn.ReLU(),
                                            nn.Linear(width, width), nn.ReLU())
        self.pool_projection = nn.Sequential(nn.Linear(width, width), nn.ReLU(),
                                             nn.Linear(width, latent))
        self.code_projection = nn.Linear(latent, width)
        self.slot_embeddings = nn.Parameter(torch.empty(hidden, width))
        nn.init.normal_(self.slot_embeddings, mean=0.0, std=.02)
        self.slot_decoder = nn.Sequential(nn.Linear(width, width), nn.ReLU(),
                                          nn.Linear(width, features))

    def encode(self, maps: torch.Tensor) -> torch.Tensor:
        columns = maps.transpose(1, 2)
        return self.pool_projection(self.column_encoder(columns).mean(dim=1))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        shared = self.code_projection(codes).unsqueeze(1)
        slots = shared + self.slot_embeddings.unsqueeze(0)
        logits = self.slot_decoder(F.relu(slots))
        return logits.transpose(1, 2)

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(maps))


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _pairwise_bce_cost(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Actual summed BCE for every target/prediction column pair.

    Inputs may be [B,F,H] or [F,H].  The identity
    BCEWithLogits(y,x)=softplus(x)-y*x avoids materializing [B,F,H,H].
    """
    if target.shape != logits.shape or target.ndim not in (2, 3):
        raise ValueError("pairwise BCE expects matching [F,H] or [B,F,H] maps")
    one = target.ndim == 2
    if one:
        target, logits = target.unsqueeze(0), logits.unsqueeze(0)
    positive = torch.einsum("bft,bfp->btp", target, logits)
    prediction_offset = F.softplus(logits).sum(dim=1)
    costs = prediction_offset[:, None, :] - positive
    return costs[0] if one else costs


def _assign(cost: torch.Tensor) -> torch.Tensor:
    """Return prediction-column index for each target column, detached."""
    if cost.ndim != 3 or cost.shape[1] != cost.shape[2]:
        raise ValueError("assignment expects [batch,target_columns,prediction_columns]")
    orders = []
    for matrix in cost.detach().cpu().numpy():
        rows, columns = linear_sum_assignment(matrix)
        order = np.empty(len(rows), dtype=np.int64)
        order[rows] = columns
        orders.append(order)
    return torch.as_tensor(np.stack(orders), dtype=torch.long, device=cost.device)


def _matched_bce_per_map(target: torch.Tensor, logits: torch.Tensor,
                         return_orders: bool = False):
    if target.ndim == 2:
        target, logits = target.unsqueeze(0), logits.unsqueeze(0)
    costs = _pairwise_bce_cost(target, logits)
    order = _assign(costs)
    selected = costs.gather(2, order.unsqueeze(-1)).squeeze(-1)
    values = selected.sum(dim=1)
    return (values, order) if return_orders else values


def _fixed_bce_per_map(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(dim=(1, 2))


def _training_loss(model: nn.Module, maps: torch.Tensor, matching: bool) -> torch.Tensor:
    logits = model(maps)
    if matching:
        return _matched_bce_per_map(maps, logits).mean()
    return _fixed_bce_per_map(maps, logits).mean()


def _seed_model(factory: Callable[[], nn.Module], seed: int,
                device: torch.device) -> nn.Module:
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = factory().to(device)
    return model


def _fit_model(model: nn.Module, train: torch.Tensor, validation: torch.Tensor,
               matching: bool, epochs: int = EPOCHS) -> tuple[nn.Module, dict[str, Any]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    train_history: list[float] = []
    validation_history: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss = _training_loss(model, train, matching)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            train_value = float(_training_loss(model, train, matching).cpu())
            validation_value = float(_training_loss(model, validation, matching).cpu())
        train_history.append(train_value)
        validation_history.append(validation_value)
        if validation_value < best_loss:
            best_loss = validation_value
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "epochs": epochs, "learning_rate": LEARNING_RATE,
        "train_maps": int(train.shape[0]), "validation_maps": int(validation.shape[0]),
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "train_loss_by_epoch": train_history,
        "validation_loss_by_epoch": validation_history,
        "loss": "sum of actual BCE over 784x32 per map, then mean over maps",
        "matching": matching,
    }


def _split_unique_maps(maps: torch.Tensor, generator: torch.Generator
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unique = mask_ops._unique_rows(maps)
    if unique.shape[0] == 1:
        order = torch.zeros(1, dtype=torch.long, device=maps.device)
        return unique, unique, order
    order = torch.randperm(unique.shape[0], generator=generator, device=maps.device)
    n_validation = max(1, int(round(.2 * unique.shape[0])))
    n_validation = min(n_validation, unique.shape[0] - 1)
    return unique[order[n_validation:]], unique[order[:n_validation]], order


def _probabilities(logits: torch.Tensor) -> torch.Tensor:
    return logits.sigmoid()


def _probability_logits(probabilities: torch.Tensor) -> torch.Tensor:
    probabilities = probabilities.clamp(1e-6, 1. - 1e-6)
    return probabilities.log() - torch.log1p(-probabilities)


def _topk_binary(probabilities: torch.Tensor, k: int = K) -> torch.Tensor:
    flat = probabilities.reshape(-1)
    output = torch.zeros_like(flat)
    output[flat.topk(k).indices] = 1.
    return output.reshape_as(probabilities)


def _metric_one(target: torch.Tensor, prediction_logits: torch.Tensor,
                *, k: int = K) -> dict[str, Any]:
    cost = _pairwise_bce_cost(target, prediction_logits)
    rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
    bce_sum = float(cost[torch.as_tensor(rows, device=cost.device),
                         torch.as_tensor(columns, device=cost.device)].sum().cpu())
    pred = prediction_logits.sigmoid()
    target_binary = _topk_binary(target, k).transpose(0, 1)
    pred_binary = _topk_binary(pred, k).transpose(0, 1)
    intersection = target_binary @ pred_binary.transpose(0, 1)
    target_count = target_binary.sum(dim=1, keepdim=True)
    prediction_count = pred_binary.sum(dim=1).unsqueeze(0)
    union = (target_count + prediction_count - intersection).clamp_min(1.)
    iou = intersection / union
    iou_rows, iou_columns = linear_sum_assignment(-iou.detach().cpu().numpy())
    matched_iou = float(iou[torch.as_tensor(iou_rows, device=iou.device),
                             torch.as_tensor(iou_columns, device=iou.device)].mean().cpu())
    return {
        "matched_bce_sum": bce_sum,
        "matched_bce_per_entry": bce_sum / FLAT_DIM,
        "topk_matched_iou": matched_iou,
        "prediction_mean": float(pred.mean().cpu()),
        "prediction_softness": float((pred * (1. - pred)).mean().cpu()),
        "target_topk_edges": int(target_binary.sum().cpu()),
        "prediction_topk_edges": int(pred_binary.sum().cpu()),
        "bce_assignment_target_to_prediction": columns.astype(int).tolist(),
        "iou_assignment_target_to_prediction": iou_columns.astype(int).tolist(),
    }


def _matched_l1(left: torch.Tensor, right: torch.Tensor) -> float:
    cost = (left.transpose(0, 1)[:, None, :] - right.transpose(0, 1)[None, :, :]).abs().mean(-1)
    rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
    return float(cost[torch.as_tensor(rows, device=cost.device),
                      torch.as_tensor(columns, device=cost.device)].mean().cpu())


def _reconstruction_report(target_maps: torch.Tensor,
                           predictions: dict[str, torch.Tensor],
                           task_ids: list[int]) -> dict[str, Any]:
    per_example: dict[str, list[dict[str, Any]]] = {name: [] for name in predictions}
    summary: dict[str, Any] = {}
    for name, logits in predictions.items():
        for index, target in enumerate(target_maps):
            metric = _metric_one(target, logits[index])
            metric.update(task=int(task_ids[index]), heldout_index=index)
            per_example[name].append(metric)
        within_target: dict[int, list[float]] = {task: [] for task in sorted(set(task_ids))}
        within_prediction: dict[int, list[float]] = {task: [] for task in sorted(set(task_ids))}
        cross_target: list[float] = []
        cross_prediction: list[float] = []
        for left in range(target_maps.shape[0]):
            for right in range(left + 1, target_maps.shape[0]):
                target_distance = _matched_l1(target_maps[left], target_maps[right])
                prediction_distance = _matched_l1(logits[left].sigmoid(), logits[right].sigmoid())
                if task_ids[left] == task_ids[right]:
                    within_target[task_ids[left]].append(target_distance)
                    within_prediction[task_ids[left]].append(prediction_distance)
                else:
                    cross_target.append(target_distance)
                    cross_prediction.append(prediction_distance)
        total = len(per_example[name])
        mean_within_target = float(np.mean([v for values in within_target.values() for v in values]))
        mean_within_prediction = float(np.mean([v for values in within_prediction.values() for v in values]))
        mean_cross_target = float(np.mean(cross_target)) if cross_target else 0.
        mean_cross_prediction = float(np.mean(cross_prediction)) if cross_prediction else 0.
        summary[name] = {
            "heldout_maps": total,
            "matched_bce_sum": float(np.mean([row["matched_bce_sum"] for row in per_example[name]])),
            "matched_bce_per_entry": float(np.mean([row["matched_bce_per_entry"] for row in per_example[name]])),
            "topk_matched_iou": float(np.mean([row["topk_matched_iou"] for row in per_example[name]])),
            "mean_prediction_softness": float(np.mean([row["prediction_softness"] for row in per_example[name]])),
            "within_task_target_pairwise_matched_l1_per_entry": mean_within_target,
            "within_task_prediction_pairwise_matched_l1_per_entry": mean_within_prediction,
            "within_task_diversity_ratio_prediction_over_target": mean_within_prediction / max(mean_within_target, 1e-12),
            "cross_task_target_pairwise_matched_l1_per_entry": mean_cross_target,
            "cross_task_prediction_pairwise_matched_l1_per_entry": mean_cross_prediction,
            "cross_task_diversity_ratio_prediction_over_target": mean_cross_prediction / max(mean_cross_target, 1e-12),
            "per_task": {},
        }
        for task in sorted(set(task_ids)):
            rows = [row for row in per_example[name] if row["task"] == task]
            summary[name]["per_task"][str(task)] = {
                "heldout_maps": len(rows),
                "matched_bce_per_entry": float(np.mean([row["matched_bce_per_entry"] for row in rows])),
                "topk_matched_iou": float(np.mean([row["topk_matched_iou"] for row in rows])),
                "within_task_target_pairwise_matched_l1_per_entry": float(np.mean(within_target[task])),
                "within_task_prediction_pairwise_matched_l1_per_entry": float(np.mean(within_prediction[task])),
                "within_task_diversity_ratio_prediction_over_target": float(
                    np.mean(within_prediction[task]) / max(np.mean(within_target[task]), 1e-12)),
            }
    return {"summary": summary, "per_example": per_example,
            "metric_protocol": {
                "heldout_targets": "raw held-out maps; held-out columns never enter training alignment",
                "matched_bce": "minimum actual BCEWithLogits over one-to-one column assignments; assignment detached",
                "topk_matched_iou": f"global exact top-{K} binary supports; Hungarian assignment maximizes mean column IoU",
                "diversity": "within-task and cross-task pairwise minimum column-matched L1, divided by 784*32; within-task values expose per-task instance collapse",
            }}


def _medoid_map(train_raw: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
    values = torch.cat([maps.detach().cpu() for maps in train_raw], dim=0).numpy().astype(np.float32)
    columns = values.transpose(0, 2, 1)
    count = len(columns)
    distances = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        left_columns = columns[left]
        for right in range(left):
            cost = np.abs(left_columns[:, None, :] - columns[right][None, :, :]).mean(axis=-1)
            rows, cols = linear_sum_assignment(cost)
            distance = float(cost[rows, cols].mean())
            distances[left, right] = distances[right, left] = distance
    index = int(distances.mean(axis=1).argmin())
    return torch.from_numpy(values[index].copy()), {
        "candidate_maps": count, "selection": "minimum average permutation-matched column L1 to all source-training maps",
        "selected_flat_train_index": index,
        "mean_distance_to_bank": float(distances[index].mean()),
        "all_candidate_mean_distances": distances.mean(axis=1).tolist(),
    }


def _softness_penalty(probabilities: torch.Tensor) -> torch.Tensor:
    softness = probabilities.mul(1. - probabilities).mean(dim=(1, 2))
    return (softness - SOFTNESS_THRESHOLD).clamp_min(0.).square()


def _direct_code_distance_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distances using explicit residuals in float64.

    ``torch.cdist`` may use the norm expansion ``||x||² + ||y||² - 2 x·y``.
    That loses precision for near-identical encoded maps and can falsely report
    zero nearest-neighbor radii.  The latent banks are small, so direct
    residuals are cheap and preserve the distances represented by the saved
    float32 codes.
    """
    left64, right64 = left.to(torch.float64), right.to(torch.float64)
    return (left64[:, None, :] - right64[None, :, :]).square().sum(-1).sqrt()


def _agreement_and_masks(models_by_method: dict[str, list[nn.Module]],
                         train_inputs_by_method: dict[str, list[torch.Tensor]],
                         train_raw: list[torch.Tensor], seed: int,
                         device: torch.device, steps: int = AGREEMENT_STEPS
                         ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    output_masks: dict[str, torch.Tensor] = {}
    reports: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for method_index, method in enumerate(METHODS):
        models = models_by_method[method]
        for model in models:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
        training_codes: list[torch.Tensor] = []
        radii: list[float] = []
        legacy_cdist_radii: list[float] = []
        direct_nn_medians: list[float] = []
        start_indices: list[list[int]] = []
        code_gen = torch.Generator(device=device).manual_seed(seed + 71001 + 1009 * method_index)
        for task, (model, train_input) in enumerate(zip(models, train_inputs_by_method[method])):
            with torch.no_grad():
                codes = model.encode(train_input)
            distances = _direct_code_distance_matrix(codes, codes)
            distances.fill_diagonal_(float("inf"))
            nearest_neighbors = distances.min(dim=1).values
            direct_median = float(nearest_neighbors.median().cpu())
            radius = direct_median * .5
            # Preserve the preregistered rule exactly.  A zero median nearest
            # neighbor distance remains zero even if other code pairs differ.
            if not math.isfinite(radius) or radius < 0.:
                raise FloatingPointError(f"invalid direct train-code radius: {radius}")
            legacy_distances = torch.cdist(codes, codes)
            legacy_distances.fill_diagonal_(float("inf"))
            legacy_nn_median = float(legacy_distances.min(dim=1).values.median().cpu())
            legacy_radius = legacy_nn_median * .5
            legacy_scale = max(float(codes.norm(dim=-1).median().cpu()), 1.)
            if legacy_radius <= 8. * torch.finfo(codes.dtype).eps * legacy_scale:
                legacy_radius = 0.0
            choices = torch.randperm(codes.shape[0], generator=code_gen, device=device)[:AGREEMENT_STARTS]
            training_codes.append(codes.detach())
            radii.append(radius)
            direct_nn_medians.append(direct_median)
            legacy_cdist_radii.append(legacy_radius)
            start_indices.append(choices.detach().cpu().tolist())
        codes = [nn.Parameter(training_codes[task][torch.as_tensor(start_indices[task], device=device)].detach().clone())
                 for task in range(len(models))]
        initial_codes = torch.stack([code.detach().clone() for code in codes])
        optimizer = torch.optim.Adam(codes, lr=AGREEMENT_LR)

        def decode(current_codes: list[torch.Tensor]):
            logits = [model.decode(code) for model, code in zip(models, current_codes)]
            raw_probabilities = [value.sigmoid() for value in logits]
            soft_topk_maps = [mask_ops.soft_topk(value.reshape(value.shape[0], -1), K,
                                                 AGREEMENT_TEMPERATURE).reshape(-1, FEATURES, HIDDEN)
                              for value in logits]
            aligned = torch.stack([soft_topk_maps[0], *[
                mask_ops._align_columns(soft_topk_maps[0], value) for value in soft_topk_maps[1:]]])
            return logits, raw_probabilities, soft_topk_maps, aligned

        def anchor_cost(logits: list[torch.Tensor]) -> torch.Tensor:
            task_costs = []
            for predicted_logit, targets in zip(logits, train_raw):
                # [starts, training examples, target columns, predicted columns]
                positive = torch.einsum("nft,sfp->sntp", targets, predicted_logit)
                offset = F.softplus(predicted_logit).sum(dim=1)
                costs = offset[:, None, None, :] - positive
                order = _assign(costs.reshape(-1, HIDDEN, HIDDEN)).reshape(
                    costs.shape[0], costs.shape[1], HIDDEN)
                selected = costs.gather(-1, order.unsqueeze(-1)).squeeze(-1).sum(-1)
                # The nearest source-training map is selected discretely; its
                # selected BCE remains connected to the decoder output.
                task_costs.append(selected.min(dim=1).values / FLAT_DIM)
            return torch.stack(task_costs)

        def raw_soft_disagreement(probabilities: list[torch.Tensor]) -> torch.Tensor:
            raw_stack = torch.stack(probabilities)
            return (raw_stack - raw_stack.mean(dim=0, keepdim=True)).square().mean((0, 2, 3))

        def global_hard_iou(binary: torch.Tensor) -> torch.Tensor:
            # Binary is [task,start,F,H].  Match columns by IoU so the hard
            # metric respects the same hidden-column permutation symmetry.
            scores = []
            reference = binary[0].transpose(1, 2)
            for task in range(1, binary.shape[0]):
                other = binary[task].transpose(1, 2)
                intersection = reference @ other.transpose(1, 2)
                reference_count = reference.sum(dim=2, keepdim=True)
                other_count = other.sum(dim=2).unsqueeze(1)
                union = (reference_count + other_count - intersection).clamp_min(1.)
                iou = intersection / union
                matched = []
                for start in range(binary.shape[1]):
                    rows, columns = linear_sum_assignment(-iou[start].detach().cpu().numpy())
                    matched.append(iou[start, torch.as_tensor(rows, device=iou.device),
                                        torch.as_tensor(columns, device=iou.device)].mean())
                scores.append(torch.stack(matched))
            return torch.stack(scores).mean(0)

        def project_to_code_support_(current: list[torch.Tensor]) -> None:
            # Project each code into the union of train-centered Euclidean
            # balls.  The centers, neighborhood radii, and initial starts use
            # training maps only.
            with torch.no_grad():
                for task, (code, support, radius) in enumerate(zip(current, training_codes, radii)):
                    nearest = _direct_code_distance_matrix(code.detach(), support).argmin(dim=1)
                    centers = support[nearest]
                    delta = code - centers
                    norm = delta.to(torch.float64).norm(dim=-1, keepdim=True)
                    scale = ((.95 * radius) / norm.clamp_min(1e-300)).clamp(max=1.)
                    code.copy_(centers + delta * scale.to(delta.dtype))

        def nearest_code_distances(code_stack: torch.Tensor) -> torch.Tensor:
            values = []
            for task, support in enumerate(training_codes):
                distance = _direct_code_distance_matrix(code_stack[task], support)
                values.append(distance.min(dim=1).values)
            return torch.stack(values)

        with torch.no_grad():
            initial_logits, _, initial_soft, initial_aligned = decode(codes)
            initial_agreement = (initial_aligned - initial_aligned.mean(dim=0, keepdim=True)).square().mean((0, 2, 3))
            initial_raw_agreement = raw_soft_disagreement(initial_soft)
            initial_anchor = anchor_cost(initial_logits)
            initial_softness = torch.stack([value.mul(1. - value).mean((1, 2)) for value in initial_soft])
            initial_hard = torch.stack([mask_ops._hard_topk(value.reshape(AGREEMENT_STARTS, -1), K)
                                        .reshape(AGREEMENT_STARTS, FEATURES, HIDDEN)
                                        for value in initial_logits])
            initial_hard_iou = global_hard_iou(initial_hard)
            best_objective = initial_agreement + ANCHOR_WEIGHT * initial_anchor.mean(0) + \
                SOFTNESS_WEIGHT * torch.stack([_softness_penalty(value) for value in initial_soft]).mean(0)
            best_codes = initial_codes.clone()
            best_step = torch.zeros(AGREEMENT_STARTS, dtype=torch.long, device=device)
        history = []
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            logits, _, soft, aligned = decode(codes)
            agreement = (aligned - aligned.mean(dim=0, keepdim=True)).square().mean((0, 2, 3))
            anchor = anchor_cost(logits)
            softness_penalty = torch.stack([_softness_penalty(value) for value in soft]).mean(0)
            objective = agreement + ANCHOR_WEIGHT * anchor.mean(0) + SOFTNESS_WEIGHT * softness_penalty
            objective.sum().backward()
            optimizer.step()
            project_to_code_support_(codes)
            with torch.no_grad():
                current_logits, _, current_soft, current_aligned = decode(codes)
                current_agreement = (current_aligned - current_aligned.mean(dim=0, keepdim=True)).square().mean((0, 2, 3))
                current_raw_agreement = raw_soft_disagreement(current_soft)
                current_anchor = anchor_cost(current_logits)
                current_softness = torch.stack([value.mul(1. - value).mean((1, 2)) for value in current_soft])
                current_objective = current_agreement + ANCHOR_WEIGHT * current_anchor.mean(0) + \
                    SOFTNESS_WEIGHT * torch.stack([_softness_penalty(value) for value in current_soft]).mean(0)
                improved = current_objective < best_objective
                best_objective = torch.where(improved, current_objective, best_objective)
                best_codes = torch.where(improved[None, :, None], torch.stack([c.detach() for c in codes]), best_codes)
                best_step = torch.where(improved, torch.full_like(best_step, step), best_step)
                if step == 1 or step % 50 == 0 or step == steps:
                    history.append({
                        "step": step,
                        "mean_agreement_mse": float(current_agreement.mean().cpu()),
                        "mean_raw_unmatched_soft_mse": float(current_raw_agreement.mean().cpu()),
                        "mean_bank_anchor_bce_per_entry": float(current_anchor.mean().cpu()),
                        "mean_softness": float(current_softness.mean().cpu()),
                        "mean_objective": float(current_objective.mean().cpu()),
                    })
        final_codes = [best_codes[index] for index in range(len(models))]
        with torch.no_grad():
            final_logits, _, final_soft, final_aligned = decode(final_codes)
            final_agreement = (final_aligned - final_aligned.mean(dim=0, keepdim=True)).square().mean((0, 2, 3))
            final_raw_agreement = raw_soft_disagreement(final_soft)
            final_anchor = anchor_cost(final_logits)
            final_softness = torch.stack([value.mul(1. - value).mean((1, 2)) for value in final_soft])
            hard = torch.stack([mask_ops._hard_topk(model.decode(final_codes[index]).reshape(AGREEMENT_STARTS, -1), K)
                                .reshape(AGREEMENT_STARTS, FEATURES, HIDDEN)
                                for index, model in enumerate(models)])
            final_hard_iou = global_hard_iou(hard)
            null_generator = torch.Generator(device=device).manual_seed(seed + 89001 + 1009 * method_index)
            random_null = torch.rand(len(models), AGREEMENT_STARTS, FLAT_DIM,
                                     generator=null_generator, device=device)
            random_null = mask_ops._hard_topk(random_null, K).reshape(
                len(models), AGREEMENT_STARTS, FEATURES, HIDDEN)
            random_null_iou = global_hard_iou(random_null)
            uniform_probability = K / FLAT_DIM
            output_masks[method] = hard[0].detach().cpu()
        final_code_distances = nearest_code_distances(best_codes)
        reports[method] = {
            "steps": steps, "starts": AGREEMENT_STARTS,
            "learning_rate": AGREEMENT_LR, "temperature": AGREEMENT_TEMPERATURE,
            "k": K, "initial_codes_from_real_training_maps": True,
            "code_support_rule": "union of Euclidean balls centered on encoded training maps; radius is exactly 0.5 times the median nearest-neighbor distance within the training codes, computed by explicit float64 residual norms; nearest-center selection, projection, and final distances use the same direct residual calculation; projection uses 0.95 of the radius",
            "code_support_radius_by_task": radii,
            "direct_nearest_neighbor_median_by_task": direct_nn_medians,
            "direct_half_median_train_nn_distance_by_task": radii,
            "legacy_float32_cdist_radius_by_task": legacy_cdist_radii,
            "legacy_float32_cdist_rule": "reproduced the original torch.cdist computation and 8*float32_epsilon*code_scale zero-radius threshold for comparison only",
            "initial_nearest_training_code_distance_by_task_start": nearest_code_distances(initial_codes).detach().cpu().tolist(),
            "final_nearest_training_code_distance_by_task_start": final_code_distances.detach().cpu().tolist(),
            "maximum_final_distance_over_radius_by_task": [
                float(final_code_distances[task].max().cpu()) / max(radii[task], 1e-12)
                for task in range(len(models))],
            "start_training_map_indices_by_task": start_indices,
            "bank_anchor": "before SoftTopK, for each decoded sigmoid map choose the nearest raw source-training map by Hungarian-matched actual BCE per entry; selected target detached",
            "bank_anchor_weight": ANCHOR_WEIGHT,
            "softness_penalty": {"measure": "mean p*(1-p)", "threshold": SOFTNESS_THRESHOLD,
                                 "weight": SOFTNESS_WEIGHT},
            "initial_agreement_mse_by_start": initial_agreement.detach().cpu().tolist(),
            "final_agreement_mse_by_start": final_agreement.detach().cpu().tolist(),
            "initial_raw_unmatched_soft_mse_by_start": initial_raw_agreement.detach().cpu().tolist(),
            "final_raw_unmatched_soft_mse_by_start": final_raw_agreement.detach().cpu().tolist(),
            "initial_anchor_bce_per_entry_by_task_start": initial_anchor.detach().cpu().tolist(),
            "final_anchor_bce_per_entry_by_task_start": final_anchor.detach().cpu().tolist(),
            "initial_softness_by_task_start": initial_softness.detach().cpu().tolist(),
            "final_softness_by_task_start": final_softness.detach().cpu().tolist(),
            "initial_hard_hungarian_column_iou_across_tasks_by_start": initial_hard_iou.detach().cpu().tolist(),
            "final_hard_hungarian_column_iou_across_tasks_by_start": final_hard_iou.detach().cpu().tolist(),
            "random_exact_k_hard_hungarian_column_iou_null_by_start": random_null_iou.detach().cpu().tolist(),
            "uniform_soft_consensus_null": {
                "constant_probability_per_entry": uniform_probability,
                "soft_agreement_mse_by_definition": 0.0,
                "softness_p_times_one_minus_p": uniform_probability * (1. - uniform_probability),
                "hard_topk_ties_are_not_evidence_of_shared_support": True,
            },
            "best_step_by_start": best_step.detach().cpu().tolist(), "history": history,
            "warning": "agreement is an optimization objective, not evidence that these masks transfer or recover a unique neuron correspondence",
        }
        artifacts[method] = {
            "initial_codes": initial_codes.detach().cpu(),
            "final_codes": best_codes.detach().cpu(),
            "training_codes_by_task": [value.detach().cpu() for value in training_codes],
        }
    return output_masks, reports, artifacts


def _self_check(device: torch.device) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(9321)
    target = torch.rand(3, 19, 7, generator=generator, device=device) * .8 + .1
    order = torch.stack([torch.randperm(7, generator=generator, device=device) for _ in range(3)])
    permuted = torch.stack([target[i, :, order[i]] for i in range(3)])
    logits = _probability_logits(permuted)
    costs = _pairwise_bce_cost(target, logits)
    recovered = _assign(costs)
    expected = torch.stack([torch.argsort(order[i]) for i in range(3)])
    if not torch.equal(recovered, expected):
        raise AssertionError(f"Hungarian did not recover the known column permutation: {recovered} != {expected}")
    reconstructed = torch.stack([permuted[i, :, recovered[i]] for i in range(3)])
    exact_error = float((reconstructed - target).abs().max().cpu())
    if exact_error > 1e-6:
        raise AssertionError(f"known-permutation reconstruction error is {exact_error}")
    train_target = torch.rand(2, 19, 7, generator=generator, device=device)
    test_logits = torch.randn(2, 19, 7, generator=generator, device=device, requires_grad=True)
    match_loss = _matched_bce_per_map(train_target, test_logits).mean()
    match_loss.backward()
    if test_logits.grad is None or not bool(torch.isfinite(test_logits.grad).all()):
        raise AssertionError("Hungarian-selected BCE has invalid gradients")
    ae = SetInvariantAE(features=19, hidden=7, latent=5, width=12).to(device)
    sample = torch.rand(4, 19, 7, generator=generator, device=device)
    column_order = torch.stack([torch.randperm(7, generator=generator, device=device) for _ in range(4)])
    sample_permuted = torch.stack([sample[i, :, column_order[i]] for i in range(4)])
    difference = float((ae.encode(sample) - ae.encode(sample_permuted)).abs().max().cpu())
    if difference > 2e-6:
        raise AssertionError(f"set encoder changed under column permutations: {difference}")
    return {"device": str(device), "known_permutation_max_reconstruction_error": exact_error,
            "known_permutation_recovered": True, "matching_gradient_finite": True,
            "set_encoder_permutation_max_code_delta": difference,
            "set_encoder_invariant_within_tolerance": True}


def _pipeline_smoke(device: torch.device) -> dict[str, Any]:
    """Exercise model fitting, support-constrained search, and mask export."""
    generator = torch.Generator(device=device).manual_seed(19411)
    train = [torch.rand(TRAIN_MAPS, FEATURES, HIDDEN, generator=generator, device=device) * .2
             for _ in range(4)]
    validation = [torch.rand(VALIDATION_MAPS, FEATURES, HIDDEN, generator=generator, device=device) * .2
                  for _ in range(4)]
    factories = {"ae_flat_consensus": FlatDeterministicAE,
                 "ae_flat_hungarian": FlatDeterministicAE,
                 "ae_set_hungarian": SetInvariantAE}
    matching = {"ae_flat_consensus": False,
                "ae_flat_hungarian": True,
                "ae_set_hungarian": True}
    models: dict[str, list[nn.Module]] = {name: [] for name in METHODS}
    train_by_method = {
        "ae_flat_consensus": train,
        "ae_flat_hungarian": train,
        "ae_set_hungarian": train,
    }
    for method_index, method in enumerate(METHODS):
        for task in range(4):
            init_family = 0 if method in ("ae_flat_consensus", "ae_flat_hungarian") else method_index
            model = _seed_model(factories[method], 19411 + init_family * 101 + task, device)
            model, _ = _fit_model(model, train[task], validation[task], matching[method], epochs=1)
            models[method].append(model)
    masks, diagnostics, _ = _agreement_and_masks(models, train_by_method, train,
                                                  seed=19411, device=device, steps=2)
    for name, value in masks.items():
        if tuple(value.shape) != (AGREEMENT_STARTS, FEATURES, HIDDEN):
            raise AssertionError(f"smoke {name}: wrong mask shape {tuple(value.shape)}")
        if not bool(torch.all(value.sum((1, 2)) == K)):
            raise AssertionError(f"smoke {name}: exact-K check failed")
    smoke_targets = torch.cat([validation[0][:2], validation[1][:2]])
    smoke_logits = torch.cat([models["ae_flat_consensus"][0](validation[0][:2]),
                              models["ae_flat_consensus"][1](validation[1][:2])])
    metric_check = _reconstruction_report(smoke_targets, {"smoke_flat": smoke_logits}, [0, 0, 1, 1])
    if "within_task_diversity_ratio_prediction_over_target" not in metric_check["summary"]["smoke_flat"]:
        raise AssertionError("within-task diversity metric was not generated")
    return {"device": str(device), "training_models": sum(map(len, models.values())),
            "training_epochs_per_model": 1, "agreement_steps": 2,
            "mask_shapes": {name: list(value.shape) for name, value in masks.items()},
            "exact_k": {name: value.sum((1, 2)).tolist() for name, value in masks.items()},
            "support_locality_check": {
                name: diagnostics[name]["maximum_final_distance_over_radius_by_task"]
                for name in METHODS},
            "within_task_metric_smoke": metric_check["summary"]["smoke_flat"],
            "passed": True}


def _write_status(out: Path, stage: str, started: float, **details: Any) -> None:
    write_json(out / "status.json", {"stage": stage, "elapsed_seconds": time.monotonic() - started,
                                    **details})
    print(json.dumps({"stage": stage, **details}, ensure_ascii=False), flush=True)


def _archive_old_attempt(out: Path, archive: Path) -> None:
    """Preserve the old masks and paired target checkpoints before repair."""
    archive.mkdir(parents=True, exist_ok=True)
    if (archive / "masks.pt").exists():
        raise FileExistsError(f"pre-repair artifacts already archived at {archive}")
    for name in ("masks.pt", "agreement_codes.pt", "mask_diagnostics.json",
                 "results.json", "status.json", "provenance.json"):
        source = out / name
        if source.exists():
            shutil.copy2(source, archive / name)
    source_weights = out / "weights"
    if source_weights.exists():
        target_weights = archive / "weights"
        try:
            shutil.copytree(source_weights, target_weights, copy_function=os.link)
            write_json(archive / "weights_archive.json", {
                "storage": "hard links to the pre-repair target checkpoints",
                "source": str(source_weights),
            })
        except OSError:
            if target_weights.exists():
                shutil.rmtree(target_weights)
            shutil.copytree(source_weights, target_weights)
            write_json(archive / "weights_archive.json", {
                "storage": "independent copies of the pre-repair target checkpoints",
                "source": str(source_weights),
            })
    for snapshot_name in ("source_snapshot", "source_snapshots"):
        source_snapshots = out / snapshot_name
        if source_snapshots.exists():
            snapshot_archive = archive / snapshot_name
            if snapshot_archive.exists():
                shutil.rmtree(snapshot_archive)
            shutil.copytree(source_snapshots, snapshot_archive)
            shutil.rmtree(source_snapshots)


def repair_masks_seed(seed: int, out: Path, agreement_steps: int = AGREEMENT_STEPS) -> dict[str, Any]:
    """Re-extract masks from saved AEs, correcting cdist support radii only.

    No autoencoder is trained here.  The source splits, trained state dicts,
    start-choice seeds, and all search hyperparameters are loaded/reused.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("AE mask repair workers must use CUDA")
    if agreement_steps != AGREEMENT_STEPS:
        raise ValueError(f"repair must retain the original {AGREEMENT_STEPS}-step search")
    device = torch.device("cuda:0")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
    configure(seed)
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    artifacts_path = out / "ae_artifacts.pt"
    splits_path = out / "map_splits.pt"
    if not artifacts_path.exists() or not splits_path.exists():
        raise FileNotFoundError(f"saved AE artifacts/splits required in {out}")
    old_masks = torch.load(out / "masks.pt", map_location="cpu", weights_only=True)
    if set(METHODS + ("ae_medoid",)) - set(old_masks):
        raise ValueError("old mask artifact is missing one or more AE methods/medoid")
    artifacts = torch.load(artifacts_path, map_location=device, weights_only=False)
    splits = torch.load(splits_path, map_location=device, weights_only=False)
    train_raw = [value.to(device=device, dtype=torch.float32)
                 for value in splits["raw_train_by_task"]]
    aligned_train = [value.to(device=device, dtype=torch.float32)
                     for value in splits["aligned_train_by_task"]]
    train_inputs = {
        "ae_flat_consensus": aligned_train,
        "ae_flat_hungarian": aligned_train,
        "ae_set_hungarian": train_raw,
    }
    factories: dict[str, Callable[[], nn.Module]] = {
        "ae_flat_consensus": FlatDeterministicAE,
        "ae_flat_hungarian": FlatDeterministicAE,
        "ae_set_hungarian": SetInvariantAE,
    }
    models: dict[str, list[nn.Module]] = {name: [] for name in METHODS}
    states = artifacts["model_state_dicts"]
    for method in METHODS:
        for state in states[method]:
            model = factories[method]().to(device)
            model.load_state_dict(state)
            model.eval()
            models[method].append(model)
    _write_status(out, "mask_repair_search", started, seed=seed,
                  device=str(device), reused_trained_models=True,
                  agreement_steps=agreement_steps)
    masks, diagnostics, codes = _agreement_and_masks(
        models, train_inputs, train_raw, seed, device, steps=agreement_steps)
    # The medoid mask is independent of latent support geometry and is retained
    # bit-for-bit as its own source-bank control.
    masks["ae_medoid"] = old_masks["ae_medoid"].detach().cpu().clone()
    for method, value in masks.items():
        if tuple(value.shape) != (AGREEMENT_STARTS, FEATURES, HIDDEN) or \
                not bool(torch.all((value == 0) | (value == 1))) or \
                not bool(torch.all(value.sum(dim=(1, 2)) == K)):
            raise AssertionError(f"invalid repaired exact-K output for {method}")
    comparison: dict[str, Any] = {}
    for method, value in masks.items():
        before = old_masks[method].to(dtype=value.dtype, device="cpu")
        after = value.to(device="cpu")
        xor = before != after
        per_replica = xor.reshape(xor.shape[0], -1).sum(dim=1).tolist()
        comparison[method] = {
            "bitwise_equal": bool(torch.equal(before, after)),
            "changed_replicas": int(sum(count > 0 for count in per_replica)),
            "replicas": len(per_replica),
            "flipped_edges_total": int(xor.sum()),
            "flipped_edges_by_replica": [int(count) for count in per_replica],
            "old_exact_k_by_replica": [int(x) for x in before.sum((1, 2)).tolist()],
            "new_exact_k_by_replica": [int(x) for x in after.sum((1, 2)).tolist()],
        }
    all_equal = all(value["bitwise_equal"] for value in comparison.values())
    archive = out.parent / "cdist_attempt" / out.name
    _archive_old_attempt(out, archive)

    # Replace the search-specific products while keeping trained states,
    # reconstruction outputs, and train/validation splits unchanged.
    artifacts["training_codes"] = codes
    artifacts["agreement_codes"] = codes
    artifacts["masks"] = masks
    artifacts["protocol"]["mask_extraction"]["support_distance_correction"] = (
        "repaired after detecting cancellation in torch.cdist; all NN radii, "
        "nearest centers, projections, and final distances use direct float64 residual norms")
    torch.save(artifacts, artifacts_path)
    torch.save(codes, out / "agreement_codes.pt")
    torch.save(masks, out / "masks.pt")
    write_json(out / "protocol.json", artifacts["protocol"])
    repair_report = {
        "seed": seed, "status": "complete",
        "reused_trained_models": True, "autoencoders_retrained": False,
        "agreement_steps": agreement_steps, "same_start_rng_and_hyperparameters": True,
        "all_masks_bitwise_unchanged": all_equal,
        "target_refit_required": not all_equal,
        "masks": comparison,
        "old_attempt_archive": str(archive),
        "legacy_radius_summary_by_method": {
            method: {
                "legacy_cdist_radius_by_task": diagnostics[method]["legacy_float32_cdist_radius_by_task"],
                "correct_direct_radius_by_task": diagnostics[method]["code_support_radius_by_task"],
                "direct_nn_median_by_task": diagnostics[method]["direct_nearest_neighbor_median_by_task"],
            } for method in METHODS
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(out / "mask_repair_comparison.json", repair_report)
    old_diagnostics = json.loads((archive / "mask_diagnostics.json").read_text()) \
        if (archive / "mask_diagnostics.json").exists() else {}
    write_json(out / "mask_diagnostics.json", {
        "seed": seed, "agreement": diagnostics,
        "medoid": old_diagnostics.get("medoid", {}),
        "mask_names": list(masks), "repair_comparison": comparison,
        "old_cdist_attempt_archived": True,
    })
    save_provenance(out, seed, artifacts["protocol"], [Path(__file__), Path(mask_ops.__file__),
                                                          Path(__file__).with_name("core.py"),
                                                          Path(__file__).with_name("followup_common.py")])
    if not all_equal:
        # Keep the old checkpoints in cdist_attempt and force a fresh paired
        # target fit for this seed; a partial-mask match is not reusable.
        shutil.rmtree(out / "weights", ignore_errors=True)
        (out / "results.json").unlink(missing_ok=True)
        _write_status(out, "awaiting_target_refit", started, seed=seed,
                      changed_methods=[name for name, row in comparison.items()
                                       if not row["bitwise_equal"]],
                      refit_required=True)
    else:
        _write_status(out, "repair_complete_targets_reusable", started, seed=seed,
                      refit_required=False)
    return repair_report


def _queue_archive_path(root: Path, name: str) -> Path:
    archive = root / "cdist_attempt"
    archive.mkdir(parents=True, exist_ok=True)
    destination = archive / name
    if destination.exists():
        raise FileExistsError(f"queue archive already exists: {destination}")
    return destination


def _unique_queue_archive_path(root: Path, stem: str) -> Path:
    archive = root / "cdist_attempt"
    archive.mkdir(parents=True, exist_ok=True)
    destination = archive / stem
    suffix = 1
    while destination.exists():
        destination = archive / f"{Path(stem).stem}_{suffix}{Path(stem).suffix}"
        suffix += 1
    return destination


def finalize_repaired_seed(root: Path, seed: int, queue_error: str | None = None) -> dict[str, Any]:
    seed_out = root / f"seed_{seed}"
    seed_archive = root / "cdist_attempt" / seed_out.name
    old_snapshot = seed_out / "source_snapshot"
    if old_snapshot.exists():
        saved_snapshot = seed_archive / "source_snapshot"
        if saved_snapshot.exists():
            shutil.rmtree(saved_snapshot)
        shutil.copytree(old_snapshot, saved_snapshot)
        shutil.rmtree(old_snapshot)
    artifacts = torch.load(seed_out / "ae_artifacts.pt", map_location="cpu", weights_only=False)
    write_json(seed_out / "protocol.json", artifacts["protocol"])
    save_provenance(seed_out, seed, artifacts["protocol"], [
        Path(__file__), Path(mask_ops.__file__), Path(__file__).with_name("core.py"),
        Path(__file__).with_name("followup_common.py")])
    comparison = json.loads((seed_out / "mask_repair_comparison.json").read_text())
    if comparison["target_refit_required"]:
        shutil.rmtree(seed_out / "weights", ignore_errors=True)
        (seed_out / "results.json").unlink(missing_ok=True)
    _write_status(seed_out, "awaiting_target_refit" if comparison["target_refit_required"]
                  else "repair_complete_targets_reusable", time.monotonic(), seed=seed,
                  refit_required=comparison["target_refit_required"],
                  recovered_after_queue_error=queue_error is not None)
    return comparison


def finalize_mask_repair_queue(root: Path, queue_error: str | None = None) -> dict[str, Any]:
    reports = {seed: finalize_repaired_seed(root, seed, queue_error)
               for seed in range(4100, 4108)}
    changed = [seed for seed, row in reports.items() if row["target_refit_required"]]
    aggregate = {
        "seeds": list(range(4100, 4108)), "changed_seeds": changed,
        "unchanged_seeds": [seed for seed, row in reports.items() if not row["target_refit_required"]],
        "per_seed": [json.loads((root / f"seed_{seed}" / "mask_repair_comparison.json").read_text())
                     for seed in range(4100, 4108)],
        "mask_repair_allocation": str(root / "cdist_attempt" / "mask_repair_allocation.json"),
        "queue_error_recovered": queue_error,
    }
    write_json(root / "mask_repair_aggregate.json", aggregate)
    return aggregate


def launch_mask_repair(root: Path) -> None:
    """Run six concurrent source-only mask repairs on the reserved GPUs."""
    root = root.resolve()
    allocation = root / "allocation.json"
    if allocation.exists():
        name = ("original_allocation.json" if not
                (root / "cdist_attempt" / "original_allocation.json").exists()
                else "mask_repair_failed_allocation.json")
        shutil.move(str(allocation), str(_unique_queue_archive_path(root, name)))
    archive = root / "cdist_attempt"
    for seed in range(4100, 4108):
        seed_out = root / f"seed_{seed}"
        seed_archive = archive / seed_out.name
        seed_archive.mkdir(parents=True, exist_ok=True)
        original_log = seed_out / "run.log"
        archived_log = seed_archive / "original_ae_run.log"
        if original_log.exists() and not archived_log.exists():
            shutil.copy2(original_log, archived_log)
    gpu_uuids = ["GPU-2e5ce2f9-206f-380c-9687-3743ff9f665c",
                 "GPU-6784dc4e-6ec9-2266-5d23-85bf1b1c2af3",
                 "GPU-f8501f2d-53bc-9087-6041-64ee69876325"]
    queue_error = None
    try:
        run_cuda_queue("deepsets_vaae.followup_ae", root, gpu_uuids,
                       extra_args=["--repair-masks"], seeds=range(4100, 4108),
                       workers_per_gpu=2)
    except RuntimeError as error:
        queue_error = str(error)
        incomplete = [seed for seed in range(4100, 4108)
                      if not (root / f"seed_{seed}" / "mask_repair_comparison.json").exists()]
        if incomplete:
            raise
        print(json.dumps({"stage": "mask_repair_finalization_recovery",
                          "queue_error": queue_error,
                          "note": "all deterministic search/comparison artifacts exist; repairing only provenance/status"}),
              flush=True)
    destination = _queue_archive_path(root, "mask_repair_allocation.json")
    shutil.move(str(allocation), str(destination))
    aggregate = finalize_mask_repair_queue(root, queue_error)
    print(json.dumps({"stage": "mask_repair_queue_complete", "changed_seeds": changed,
                      "unchanged_seeds": aggregate["unchanged_seeds"]}), flush=True)


def launch_target_refits(root: Path) -> None:
    """Refit only seeds whose repaired masks differ, using validated batching."""
    root = root.resolve()
    aggregate_path = root / "mask_repair_aggregate.json"
    if not aggregate_path.exists():
        raise FileNotFoundError("run --launch-mask-repair before target refits")
    aggregate = json.loads(aggregate_path.read_text())
    seeds = aggregate["changed_seeds"]
    if not seeds:
        print(json.dumps({"stage": "target_refit_skipped", "reason": "all masks are bitwise unchanged"}),
              flush=True)
        return
    allocation = root / "allocation.json"
    if allocation.exists():
        shutil.move(str(allocation), str(_queue_archive_path(root, "mask_repair_allocation.json")))
    archive = root / "cdist_attempt"
    for seed in seeds:
        seed_out = root / f"seed_{seed}"
        seed_archive = archive / seed_out.name
        repair_log = seed_out / "run.log"
        archived_repair_log = seed_archive / "mask_repair_run.log"
        if repair_log.exists() and not archived_repair_log.exists():
            shutil.copy2(repair_log, archived_repair_log)
    gpu_uuids = ["GPU-2e5ce2f9-206f-380c-9687-3743ff9f665c",
                 "GPU-6784dc4e-6ec9-2266-5d23-85bf1b1c2af3",
                 "GPU-f8501f2d-53bc-9087-6041-64ee69876325"]
    run_cuda_queue("deepsets_vaae.followup_ae", root, gpu_uuids,
                   extra_args=["--target-only"], seeds=seeds, workers_per_gpu=2,
                   env_overrides={"DEEPSETS_EVAL_BATCH_CONDITIONS": "8"})
    shutil.move(str(allocation), str(_queue_archive_path(root, "target_refit_allocation.json")))
    write_json(root / "target_refit_status.json", {
        "stage": "complete", "seeds": seeds,
        "execution_batch_conditions": 8,
        "allocation": str(archive / "target_refit_allocation.json"),
    })


def run_seed(seed: int, out: Path, target_only: bool = False,
             epochs: int = EPOCHS, agreement_steps: int = AGREEMENT_STEPS) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("AE follow-up workers must use CUDA")
    device = torch.device("cuda:0")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
    configure(seed)
    started = time.monotonic()
    if target_only:
        prepared = torch.load(out / "ae_artifacts.pt", map_location=device, weights_only=False)
        masks = torch.load(out / "masks.pt", map_location=device, weights_only=True)
        pilot = load_pilot_seed(seed, device)
        costs = torch.tensor(task_vectors()["test"], device=device, dtype=torch.float32)
        _write_status(out, "target_evaluation", started, seed=seed)
        records = evaluate_followup(pilot["data"], costs, masks, seed, device,
                                   out / "weights", pilot["original_masks"])
        write_json(out / "results.json", {"seed": seed, "records": records,
                                          "elapsed_seconds": time.monotonic() - started,
                                          "protocol": prepared["protocol"]})
        _write_status(out, "complete", started, seed=seed, target_records=len(records))
        return
    if (out / "results.json").exists():
        print(f"seed {seed}: completed artifacts already exist; skipping", flush=True)
        return
    if epochs != EPOCHS or agreement_steps != AGREEMENT_STEPS:
        raise ValueError("Full workers use the preregistered 160 epochs and 400 agreement steps")
    pilot = load_pilot_seed(seed, device)
    banks = pilot["banks"]
    raw = [torch.as_tensor(bank["maps"], dtype=torch.float32, device=device) for bank in banks]
    if any(tuple(value.shape) != (32, FEATURES, HIDDEN) for value in raw):
        raise ValueError(f"unexpected source bank dimensions: {[tuple(value.shape) for value in raw]}")
    split_generator = torch.Generator(device=device).manual_seed(seed + 50000 + 101)
    train_raw: list[torch.Tensor] = []
    validation_raw: list[torch.Tensor] = []
    split_orders: list[torch.Tensor] = []
    for maps in raw:
        train, valid, order = _split_unique_maps(maps, split_generator)
        if train.shape[0] != TRAIN_MAPS or valid.shape[0] != VALIDATION_MAPS:
            raise ValueError(f"seed {seed} split count is {train.shape[0]}/{valid.shape[0]}, expected 26/6")
        train_raw.append(train)
        validation_raw.append(valid)
        split_orders.append(order)
    aligned_train, consensus = mask_ops._consensus_align(train_raw)
    aligned_validation = [mask_ops._align_columns(consensus.expand_as(value), value)
                          for value in validation_raw]
    train_input_by_method = {
        "ae_flat_consensus": aligned_train,
        "ae_flat_hungarian": aligned_train,
        "ae_set_hungarian": train_raw,
    }
    validation_input_by_method = {
        "ae_flat_consensus": aligned_validation,
        "ae_flat_hungarian": aligned_validation,
        "ae_set_hungarian": validation_raw,
    }
    model_factories = {
        "ae_flat_consensus": FlatDeterministicAE,
        "ae_flat_hungarian": FlatDeterministicAE,
        "ae_set_hungarian": SetInvariantAE,
    }
    matching_by_method = {"ae_flat_consensus": False,
                          "ae_flat_hungarian": True,
                          "ae_set_hungarian": True}
    models_by_method: dict[str, list[nn.Module]] = {name: [] for name in METHODS}
    fit_reports: dict[str, list[dict[str, Any]]] = {name: [] for name in METHODS}
    model_states: dict[str, list[dict[str, torch.Tensor]]] = {name: [] for name in METHODS}
    param_counts: dict[str, list[int]] = {name: [] for name in METHODS}
    for method_index, method in enumerate(METHODS):
        for task in range(4):
            init_family = 0 if method in ("ae_flat_consensus", "ae_flat_hungarian") else method_index
            model = _seed_model(model_factories[method], seed + 60000 + init_family * 101 + task, device)
            initial_state_hash = _state_sha256(model)
            param_counts[method].append(_parameter_count(model))
            model, report = _fit_model(model, train_input_by_method[method][task],
                                       validation_input_by_method[method][task],
                                       matching_by_method[method], epochs=epochs)
            report.update(method=method, task=task, parameter_count=_parameter_count(model),
                          initial_state_sha256=initial_state_hash)
            models_by_method[method].append(model)
            fit_reports[method].append(report)
            model_states[method].append({key: value.detach().cpu().clone()
                                         for key, value in model.state_dict().items()})
            print(f"[ae] seed={seed} method={method} task={task} "
                  f"val={report['best_validation_loss']:.4f} epoch={report['best_epoch']}/{epochs} "
                  f"params={report['parameter_count']}", flush=True)
    if any(fit_reports["ae_flat_consensus"][task]["initial_state_sha256"] !=
           fit_reports["ae_flat_hungarian"][task]["initial_state_sha256"] for task in range(4)):
        raise AssertionError("fixed-loss and Hungarian-loss flat AEs did not share their initial weights")
    vae_models = []
    vae_states = torch.load(pilot["folder"] / "vae_artifacts.pt", map_location=device,
                            weights_only=False)["vae_state_dicts"]
    for task, state in enumerate(vae_states):
        model = mask_ops._MaskVAE(FLAT_DIM, LATENT, WIDTH).to(device)
        model.load_state_dict(state)
        model.eval()
        vae_models.append(model)
    medoid, medoid_info = _medoid_map(train_raw)
    predictions: dict[str, torch.Tensor] = {
        "ae_flat_consensus": torch.cat([model(value).detach() for model, value in
                                        zip(models_by_method["ae_flat_consensus"], aligned_validation)]),
        "ae_flat_hungarian": torch.cat([model(value).detach() for model, value in
                                        zip(models_by_method["ae_flat_hungarian"], aligned_validation)]),
        "ae_set_hungarian": torch.cat([model(value).detach() for model, value in
                                       zip(models_by_method["ae_set_hungarian"], validation_raw)]),
    }
    predictions["saved_vae"] = torch.cat([
        model.decode(model.encode(value.reshape(value.shape[0], -1))[0]).reshape(-1, FEATURES, HIDDEN).detach()
        for model, value in zip(vae_models, aligned_validation)])
    task_means = [value.mean(dim=0) for value in aligned_train]
    global_mean = torch.cat(aligned_train, dim=0).mean(dim=0)
    predictions["constant_task_mean"] = torch.cat([
        _probability_logits(task_means[task].clamp(1e-6, 1. - 1e-6)).unsqueeze(0).expand(VALIDATION_MAPS, -1, -1)
        for task in range(4)])
    predictions["constant_global_consensus"] = _probability_logits(
        global_mean.clamp(1e-6, 1. - 1e-6)).unsqueeze(0).expand(4 * VALIDATION_MAPS, -1, -1)
    predictions["constant_medoid"] = _probability_logits(
        medoid.to(device).clamp(1e-6, 1. - 1e-6)).unsqueeze(0).expand(4 * VALIDATION_MAPS, -1, -1)
    heldout_targets = torch.cat(validation_raw)
    task_ids = [task for task in range(4) for _ in range(VALIDATION_MAPS)]
    reconstruction_report = _reconstruction_report(heldout_targets, predictions, task_ids)
    medoid_mask = _topk_binary(medoid.to(device), K)
    train_input_for_search = train_input_by_method
    new_masks, agreement_report, search_artifacts = _agreement_and_masks(
        models_by_method, train_input_for_search, train_raw, seed, device, steps=agreement_steps)
    new_masks["ae_medoid"] = medoid_mask.unsqueeze(0).expand(AGREEMENT_STARTS, -1, -1).detach().cpu()
    for method, value in new_masks.items():
        if tuple(value.shape) != (AGREEMENT_STARTS, FEATURES, HIDDEN) or \
                not bool(torch.all((value == 0) | (value == 1))) or \
                not bool(torch.all(value.sum(dim=(1, 2)) == K)):
            raise AssertionError(f"invalid exact-K output for {method}")
    protocol = {
        "experiment": "DeepSets deterministic AE permutation/mask transfer follow-up",
        "seed": seed, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device), "source_tasks": 4, "source_maps_per_task": 32,
        "map_split_seed": seed + 50000 + 101, "training_maps_per_task": TRAIN_MAPS,
        "validation_maps_per_task": VALIDATION_MAPS,
        "validation_alignment": "held-out maps aligned only to consensus computed from training maps; both flat variants receive the same aligned maps so only their reconstruction loss changes",
        "raw_importance_maps_unchanged": True,
        "variants": {
            "ae_flat_consensus": "flat deterministic encoder/decoder; original train consensus alignment; fixed-coordinate BCE",
            "ae_flat_hungarian": "same flat deterministic architecture and initialization as fixed-loss AE; same train-only consensus order; per-map Hungarian assignment under actual BCE",
            "ae_set_hungarian": "shared column encoder with mean pooling; 32 learned decoder slots; per-map Hungarian BCE",
        },
        "flat_objective_comparison": {
            "same_initial_weights_by_task": True,
            "initial_state_sha256_by_task": [fit_reports["ae_flat_consensus"][task]["initial_state_sha256"]
                                              for task in range(4)],
            "matching_variant_hashes_by_task": [fit_reports["ae_flat_hungarian"][task]["initial_state_sha256"]
                                                 for task in range(4)],
            "same_aligned_training_and_validation_maps": True,
        },
        "architecture": {"latent": LATENT, "width": WIDTH,
                         "flat_autoencoder_parameter_count": param_counts["ae_flat_consensus"][0],
                         "set_autoencoder_parameter_count": param_counts["ae_set_hungarian"][0],
                         "parameter_count_note": "flat architecture matches the pilot VAE minus its log-variance head; set architecture has a shared column encoder and 32 distinguishable learned decoder slots"},
        "training": {"epochs": epochs, "optimizer": "Adam", "learning_rate": LEARNING_RATE,
                     "full_batch": True, "kl_weight": 0.0, "random_prior_search": False},
        "matching_cost": "sum over 784 pixels of BCEWithLogits per target/prediction column pair; Hungarian assignment detached, chosen differentiable costs backpropagate",
        "reconstruction_metrics": reconstruction_report["metric_protocol"],
        "saved_vae_comparison_caveat": "the saved VAE checkpoint is historical and was selected by its original BCE-plus-KL validation objective; all methods here are scored with the same held-out Hungarian BCE and top-K metrics, which were not used to select that VAE checkpoint",
        "mask_extraction": {"k": K, "density": K / FLAT_DIM, "starts": AGREEMENT_STARTS,
                            "steps": agreement_steps, "initialized_from_training_codes": True,
                            "target_labels_for_selection": False,
                            "agreement_limit": "Neither matching nor decoder agreement guarantees transfer or functional hidden-unit identity."},
        "target_evaluation": {"tasks": 8, "budgets": [32, 64, 128, 256],
                              "source_map_split": "26 training and 6 held-out maps per task; target split supplied by shared evaluator",
                              "paired_weights": "shared evaluator preserves pilot original five controls and reference model initialization"},
    }
    split_payload = {
        "raw_train_by_task": [value.detach().cpu() for value in train_raw],
        "raw_validation_by_task": [value.detach().cpu() for value in validation_raw],
        "aligned_train_by_task": [value.detach().cpu() for value in aligned_train],
        "aligned_validation_by_task": [value.detach().cpu() for value in aligned_validation],
        "training_consensus": consensus.detach().cpu(),
        "unique_map_split_orders": [value.detach().cpu() for value in split_orders],
    }
    torch.save(split_payload, out / "map_splits.pt")
    write_json(out / "training_histories.json", {"seed": seed, "models": fit_reports,
                                                  "parameter_counts_by_method": param_counts})
    reconstruction_payload = {
        "seed": seed, "target_maps": heldout_targets.detach().cpu(),
        "task_ids": task_ids,
        "predictions_logits": {name: value.detach().cpu() for name, value in predictions.items()},
        "report": reconstruction_report,
        "medoid": medoid_info,
        "medoid_map": medoid,
    }
    torch.save(reconstruction_payload, out / "reconstruction.pt")
    write_json(out / "reconstruction_metrics.json", {
        "seed": seed, "summary": reconstruction_report["summary"],
        "per_example": reconstruction_report["per_example"],
        "metric_protocol": reconstruction_report["metric_protocol"],
        "medoid": medoid_info,
    })
    torch.save({"seed": seed, "model_state_dicts": model_states,
                "saved_vae_state_dicts": [{key: value.detach().cpu() for key, value in model.state_dict().items()}
                                           for model in vae_models],
                "training_codes": search_artifacts,
                "agreement_codes": search_artifacts,
                "masks": new_masks,
                "protocol": protocol}, out / "ae_artifacts.pt")
    torch.save(search_artifacts, out / "agreement_codes.pt")
    torch.save(new_masks, out / "masks.pt")
    write_json(out / "mask_diagnostics.json", {"seed": seed, "agreement": agreement_report,
                                                 "medoid": medoid_info,
                                                 "mask_names": list(new_masks)})
    write_json(out / "protocol.json", protocol)
    save_provenance(out, seed, protocol, [Path(__file__), Path(mask_ops.__file__),
                                          Path(__file__).with_name("core.py"),
                                          Path(__file__).with_name("followup_common.py")])
    _write_status(out, "source_prepared", started, seed=seed,
                  models=sum(len(value) for value in models_by_method.values()),
                  reconstruction_methods=len(predictions), new_masks=list(new_masks))
    costs = torch.tensor(task_vectors()["test"], device=device, dtype=torch.float32)
    _write_status(out, "target_evaluation", started, seed=seed)
    records = evaluate_followup(pilot["data"], costs, new_masks, seed, device,
                               out / "weights", pilot["original_masks"])
    write_json(out / "results.json", {"seed": seed, "records": records,
                                      "elapsed_seconds": time.monotonic() - started,
                                      "protocol": protocol})
    _write_status(out, "complete", started, seed=seed, target_records=len(records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--target-only", action="store_true")
    parser.add_argument("--repair-masks", action="store_true",
                        help="re-extract agreement masks from saved AEs with direct latent distances")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--self-check-device", default="cpu")
    parser.add_argument("--pipeline-smoke", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--launch-mask-repair", action="store_true")
    parser.add_argument("--finalize-repaired-masks", action="store_true",
                        help="finalize completed mask repairs whose old provenance snapshot blocked status writing")
    parser.add_argument("--launch-target-refits", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--agreement-steps", type=int, default=AGREEMENT_STEPS)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mnist8m"))
    args = parser.parse_args()
    if args.self_check:
        report = _self_check(torch.device(args.self_check_device))
        print(json.dumps(report, indent=2), flush=True)
        return
    if args.pipeline_smoke:
        device = torch.device(args.self_check_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for pipeline smoke but is unavailable")
        report = _pipeline_smoke(device)
        print(json.dumps(report, indent=2), flush=True)
        return
    if args.launch:
        uuids = ["GPU-2e5ce2f9-206f-380c-9687-3743ff9f665c",
                 "GPU-6784dc4e-6ec9-2266-5d23-85bf1b1c2af3",
                 "GPU-f8501f2d-53bc-9087-6041-64ee69876325"]
        run_cuda_queue("deepsets_vaae.followup_ae", args.out or FOLLOWUP / "ae",
                       uuids, extra_args=[])
        return
    if args.launch_mask_repair:
        launch_mask_repair(args.out or FOLLOWUP / "ae")
        return
    if args.finalize_repaired_masks:
        root = (args.out or FOLLOWUP / "ae").resolve()
        queue_error = None
        allocation = root / "allocation.json"
        if allocation.exists():
            name = "mask_repair_failed_allocation.json"
            shutil.move(str(allocation), str(_unique_queue_archive_path(root, name)))
        aggregate = finalize_mask_repair_queue(root, queue_error)
        write_json(root / "mask_repair_aggregate.json", aggregate)
        print(json.dumps({"stage": "mask_repair_finalization_complete",
                          "changed_seeds": aggregate["changed_seeds"],
                          "unchanged_seeds": aggregate["unchanged_seeds"]}), flush=True)
        return
    if args.launch_target_refits:
        launch_target_refits(args.out or FOLLOWUP / "ae")
        return
    if args.seed is None or args.out is None:
        parser.error("--seed and --out are required unless --self-check or --launch is used")
    if args.repair_masks:
        repair_masks_seed(args.seed, args.out, agreement_steps=args.agreement_steps)
        return
    run_seed(args.seed, args.out, target_only=args.target_only,
             epochs=args.epochs, agreement_steps=args.agreement_steps)


if __name__ == "__main__":
    main()
