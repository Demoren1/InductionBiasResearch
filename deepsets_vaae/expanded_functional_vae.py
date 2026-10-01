"""Functional-map VAE extraction for expanded sparse source banks.

This module consumes source-only model checkpoints and data.  It never reads
digit labels, task costs, or target-task examples.  Hidden-unit assignments are
fit from the raw training maps once, then reused for functional maps and for
the held-out maps.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from . import followup_importance as importance
from . import masks as mask_ops


FEATURE_CALIBRATION_IMAGES = 4096
FUNCTION_IMAGE_SEED_OFFSET = 50_000
ALIGNMENT_SEED_OFFSET = 50_101
OUTPUT_FILES = (
    "masks.pt",
    "functional_vae_diagnostics.json",
    "functional_vae_artifacts.pt",
    "functional_vae_arrays.npz",
)


def _bank_items(banks: Any) -> tuple[list[dict[str, Any]], list[str]]:
    if isinstance(banks, Mapping):
        if "banks" in banks and isinstance(banks["banks"], Sequence):
            values = list(banks["banks"])
            names = [str(bank.get("name", bank.get("task", index)))
                     for index, bank in enumerate(values)]
        else:
            values = list(banks.values())
            names = [str(key) for key in banks]
    else:
        values = list(banks)
        names = [str(bank.get("name", bank.get("task", index)))
                 for index, bank in enumerate(values)]
    if len(values) != 4 or any(not isinstance(bank, Mapping) for bank in values):
        raise ValueError("expanded functional extraction requires four source-task banks")
    return [dict(bank) for bank in values], names


def _source_train_features(data: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    """Read the source-training image tensor without accessing labels."""
    split = data["source_train"]
    if hasattr(split, "features"):
        features = split.features
    elif isinstance(split, Mapping):
        features = split["features"]
    else:
        # ``core.Split`` is tuple-compatible; feature values are its first field.
        features = split[0]
    result = torch.as_tensor(features, dtype=torch.float32, device=device)
    if result.ndim != 2 or result.size(0) == 0:
        raise ValueError("data['source_train'] must provide non-empty [images, features]")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("source-training images must be finite")
    return result


def _state_for_scoring(bank: Mapping[str, Any], index: int,
                       device: torch.device) -> dict[str, torch.Tensor]:
    state = bank.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError(f"source task {index} lacks a batched state_dict")
    aliases = {"weight": ("weight", "weights"),
               "masks": ("masks", "mask"),
               "bias": ("bias",),
               "readout": ("readout",),
               "per_image_offset": ("per_image_offset",)}
    result: dict[str, torch.Tensor] = {}
    for canonical, choices in aliases.items():
        value = next((state[name] for name in choices if name in state), None)
        if value is None and canonical == "masks":
            value = bank.get("masks")
        if value is None:
            raise ValueError(f"source task {index} state_dict lacks {canonical}")
        result[canonical] = torch.as_tensor(value, dtype=torch.float32,
                                            device=device).detach()
    if result["weight"].ndim != 3:
        raise ValueError(f"source task {index} weights must have shape [models,F,H]")
    models, features, hidden = result["weight"].shape
    expected = {"masks": (models, features, hidden),
                "bias": (models, hidden),
                "readout": (models, hidden),
                "per_image_offset": (models,)}
    for key, shape in expected.items():
        if tuple(result[key].shape) != shape:
            raise ValueError(f"source task {index} {key} has shape {tuple(result[key].shape)}, expected {shape}")
    if not all(bool(torch.isfinite(value).all()) for value in result.values()):
        raise ValueError(f"source task {index} state_dict contains non-finite values")
    return result


def _column_orders(reference: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Return per-map Hungarian orders for equal [N,F,H] map batches."""
    if reference.shape != other.shape or reference.ndim != 3:
        raise ValueError("column alignment expects equal [N,F,H] tensors")
    left = reference.detach().transpose(1, 2)
    right = other.detach().transpose(1, 2)
    costs = (left[:, :, None] - right[:, None, :]).square().sum(-1).cpu().numpy()
    orders = []
    for cost in costs:
        rows, cols = linear_sum_assignment(cost)
        order = torch.empty(reference.size(-1), dtype=torch.long)
        order[torch.as_tensor(rows)] = torch.as_tensor(cols)
        orders.append(order)
    return torch.stack(orders).to(other.device)


def _apply_orders(maps: torch.Tensor, orders: torch.Tensor) -> torch.Tensor:
    if maps.shape[0] != orders.shape[0] or maps.shape[-1] != orders.shape[-1]:
        raise ValueError("column orders do not match map batch")
    return maps.gather(2, orders[:, None, :].expand_as(maps))


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(_jsonable(value), indent=2,
                                        ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _strict_fp32():
    """Disable TF32 while calculating source function-gradient maps."""
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _metric_bundle(prediction: torch.Tensor, target: torch.Tensor,
                   train_mean: torch.Tensor, *, source_k: int) -> dict[str, float | int]:
    """Per-entry reconstruction metrics and support diversity for one split."""
    pred = prediction.reshape(len(prediction), -1).clamp(1e-7, 1. - 1e-7)
    truth = target.reshape(len(target), -1).clamp(0., 1.)
    mean = train_mean.reshape(1, -1).expand_as(truth).clamp(1e-7, 1. - 1e-7)

    def bce(left: torch.Tensor, right: torch.Tensor) -> float:
        return float(torch.nn.functional.binary_cross_entropy(left, right).detach().cpu())

    def mse(left: torch.Tensor, right: torch.Tensor) -> float:
        return float((left - right).square().mean().detach().cpu())

    pred_hard = mask_ops._hard_topk(pred, source_k)
    target_hard = mask_ops._hard_topk(truth, source_k)
    mean_hard = mask_ops._hard_topk(mean, source_k)

    def row_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        intersection = (left * right).sum(-1)
        union = (left + right - left * right).sum(-1).clamp_min(1.)
        return intersection / union

    def diversity(hard: torch.Tensor) -> tuple[float, float, float]:
        if len(hard) < 2:
            return 0., 0., float(torch.unique(hard, dim=0).size(0) / max(len(hard), 1))
        # Pairwise-expanded indexing would allocate O(N^2 * F*H) values.  A
        # Gram matrix keeps the working set at O(N^2 + N*F*H).  Binary FP32
        # products and sums are exact here: every intersection is an integer
        # no larger than the 5,018-edge source support (well below 2^24).
        binary = hard.to(torch.float32)
        with _strict_fp32():
            intersections = binary @ binary.transpose(0, 1)
        counts = binary.sum(dim=1)
        rows, cols = torch.triu_indices(len(hard), len(hard), offset=1,
                                        device=hard.device)
        pair_intersections = intersections[rows, cols]
        unions = (counts[rows] + counts[cols] - pair_intersections).clamp_min(1.)
        iou = pair_intersections / unions
        hamming = (counts[rows] + counts[cols] - 2. * pair_intersections) / hard.size(1)
        unique_fraction = float(torch.unique(hard, dim=0).size(0) / len(hard))
        return float(iou.mean().cpu()), float(hamming.mean().cpu()), unique_fraction

    pred_iou = row_iou(pred_hard, target_hard).mean()
    mean_iou = row_iou(mean_hard, target_hard).mean()
    pred_pair_iou, pred_pair_hamming, pred_unique = diversity(pred_hard)
    target_pair_iou, target_pair_hamming, target_unique = diversity(target_hard)
    pred_variance = float(pred.var(dim=0, unbiased=False).mean().cpu())
    target_variance = float(truth.var(dim=0, unbiased=False).mean().cpu())
    baseline_bce = bce(mean, truth)
    baseline_mse = mse(mean, truth)
    return {
        "reconstruction_bce": bce(pred, truth),
        "train_mean_bce": baseline_bce,
        "reconstruction_mse": mse(pred, truth),
        "train_mean_mse": baseline_mse,
        "reconstruction_topk_iou": float(pred_iou.cpu()),
        "train_mean_topk_iou": float(mean_iou.cpu()),
        "reconstruction_mean_map_variance": pred_variance,
        "target_mean_map_variance": target_variance,
        "reconstruction_to_target_variance_ratio": pred_variance / max(target_variance, 1e-12),
        "diversity_reconstruction_pairwise_iou_mean": pred_pair_iou,
        "diversity_reconstruction_pairwise_hamming_mean": pred_pair_hamming,
        "diversity_reconstruction_unique_fraction": pred_unique,
        "diversity_target_pairwise_iou_mean": target_pair_iou,
        "diversity_target_pairwise_hamming_mean": target_pair_hamming,
        "diversity_target_unique_fraction": target_unique,
        "examples": len(target),
        "topk_for_reconstruction_fidelity": source_k,
    }


def _reconstruct(model: mask_ops._MaskVAE, maps: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(maps.reshape(len(maps), -1))
        return torch.sigmoid(model.decode(mu)).reshape_as(maps)


def extract_expanded_functional_masks(
    banks: Any,
    data: dict[str, Any],
    seed: int,
    device: str | torch.device,
    out: Path,
    *,
    vae_epochs: int = 160,
    agreement_steps: int = 400,
    starts: int = 4,
    large_train_maps: int = 205,
    small_train_maps: int = 26,
    validation_maps: int = 51,
    smoke: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Extract 30%-density masks from four expanded source banks.

    The three VAEs fit functional-small, functional-large, and raw-large maps.
    Each task's small training set is a prefix of its large set; one held-out
    validation split and one raw-derived alignment are shared by every fit.
    """
    if vae_epochs <= 0 or agreement_steps < 0 or starts <= 0:
        raise ValueError("invalid VAE, agreement, or restart count")
    if min(large_train_maps, small_train_maps, validation_maps) <= 0:
        raise ValueError("map counts must be positive")
    target_device = torch.device(device)
    bank_list, bank_names = _bank_items(banks)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    existing = [str(out / name) for name in OUTPUT_FILES if (out / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite expanded functional outputs: {existing}")

    raw_by_task = [torch.as_tensor(bank["maps"], dtype=torch.float32,
                                   device=target_device) for bank in bank_list]
    if any(maps.ndim != 3 for maps in raw_by_task):
        raise ValueError("each bank['maps'] must have shape [keep,F,H]")
    features, hidden = raw_by_task[0].shape[1:]
    n_maps = raw_by_task[0].size(0)
    if features != 784 or hidden != 32:
        raise ValueError(f"expanded source maps must have shape [keep,784,32], got {(features, hidden)}")
    if any(tuple(maps.shape[1:]) != (features, hidden) or maps.size(0) != n_maps
           for maps in raw_by_task):
        raise ValueError("all four banks must have the same [keep,784,32] shape")
    if n_maps < 2:
        raise ValueError("at least two maps per source task are required")
    if any(not bool(torch.isfinite(maps).all()) or float(maps.min()) < 0.
           or float(maps.max()) > 1. for maps in raw_by_task):
        raise ValueError("raw importance maps must be finite and lie in [0,1]")

    if smoke:
        n_validation = max(1, int(round(.2 * n_maps)))
        n_validation = min(n_validation, n_maps - 1)
        n_large = min(large_train_maps, n_maps - n_validation)
        n_small = min(small_train_maps, n_large, 2)
        actual_epochs, actual_steps = 2, 2
    else:
        if validation_maps >= n_maps:
            raise ValueError("validation_maps must leave at least one training map per task")
        n_validation = validation_maps
        n_large = min(large_train_maps, n_maps - n_validation)
        n_small = min(small_train_maps, n_large)
        actual_epochs, actual_steps = vae_epochs, agreement_steps
    if n_large < 1 or n_small < 1:
        raise ValueError("split leaves no training maps")

    # Raw source maps already carry the bank's per-map normalization.  Reapply
    # it explicitly so the raw and function VAE targets share [0,1] semantics.
    raw_normalized = torch.stack([
        importance._normalize_nonnegative(task_maps)
        for task_maps in raw_by_task
    ])
    if any(not bool(torch.isfinite(value).all()) for value in raw_normalized):
        raise ValueError("normalized raw maps contain non-finite values")

    image_pool = _source_train_features(data, target_device)
    image_seed = int(seed) + FUNCTION_IMAGE_SEED_OFFSET
    image_generator = torch.Generator(device=target_device).manual_seed(image_seed)
    image_count = min(FEATURE_CALIBRATION_IMAGES, image_pool.size(0))
    image_indices = torch.randperm(image_pool.size(0), generator=image_generator,
                                   device=target_device)[:image_count]
    calibration_images = image_pool[image_indices]
    if calibration_images.size(1) != features:
        raise ValueError(f"source-training images have {calibration_images.size(1)} features; expected {features}")

    function_by_task = []
    score_metadata = []
    for task, bank in enumerate(bank_list):
        state = _state_for_scoring(bank, task, target_device)
        if tuple(state["weight"].shape) != tuple(raw_by_task[task].shape):
            raise ValueError(f"source task {task} state/map dimensions differ")
        with _strict_fp32():
            scores = importance._function_gradient_scores(state, calibration_images)
        normalized = importance._normalize_nonnegative(scores)
        function_by_task.append(normalized)
        score_metadata.append({
            "task": bank_names[task],
            "definition": "mean abs(W_ij * x_i * readout_j * (1-tanh(preactivation_j)^2)) over source_train images",
            "implementation": "followup_importance._function_gradient_scores",
            "normalization": "per-model nonnegative max normalization via followup_importance._normalize_nonnegative",
            "arithmetic": "strict FP32; CUDA matmul and cuDNN TF32 disabled for score calculation",
            "source_train_image_count": int(image_count),
            "source_train_image_seed": image_seed,
            "source_train_image_indices": image_indices.detach().cpu().tolist(),
            "target_labels_used": False,
            "source_labels_used": False,
            "score_nonzero_fraction": float((normalized > 0).float().mean().cpu()),
            "score_mean": float(normalized.mean().cpu()),
            "score_max": float(normalized.max().cpu()),
        })
    function_normalized = torch.stack(function_by_task)

    # Reproduce followup_importance's one-per-task row permutation seed.  The
    # leading validation rows are never used to fit the consensus or any VAE.
    split_generator = torch.Generator(device=target_device).manual_seed(int(seed) + ALIGNMENT_SEED_OFFSET)
    split_orders: list[torch.Tensor] = []
    validation_indices: list[torch.Tensor] = []
    large_indices: list[torch.Tensor] = []
    for _ in range(4):
        order = torch.randperm(n_maps, generator=split_generator, device=target_device)
        split_orders.append(order)
        validation_indices.append(order[:n_validation])
        large_indices.append(order[n_validation:n_validation + n_large])
    small_indices = [rows[:n_small] for rows in large_indices]

    train_raw = [raw_normalized[index, rows] for index, rows in enumerate(large_indices)]
    alignment_orders = importance._raw_alignment_orders(train_raw)
    train_raw_aligned = [_apply_orders(task, order)
                         for task, order in zip(train_raw, alignment_orders)]
    raw_consensus = torch.cat(train_raw_aligned, dim=0).mean(0)
    train_function = [function_normalized[index, rows]
                      for index, rows in enumerate(large_indices)]
    train_function_aligned = [_apply_orders(task, order)
                              for task, order in zip(train_function, alignment_orders)]

    validation_raw_aligned: list[torch.Tensor] = []
    validation_function_aligned: list[torch.Tensor] = []
    validation_orders: list[torch.Tensor] = []
    for task in range(4):
        raw_valid = raw_normalized[task, validation_indices[task]]
        order = _column_orders(raw_consensus.expand_as(raw_valid), raw_valid)
        validation_orders.append(order)
        validation_raw_aligned.append(_apply_orders(raw_valid, order))
        function_valid = function_normalized[task, validation_indices[task]]
        validation_function_aligned.append(_apply_orders(function_valid, order))

    # Source maps are sparse at 20%; this support size is used only to score
    # reconstruction fidelity.  Extracted downstream masks use 30% density.
    flat_dim = features * hidden
    source_k = min(max(1, int(round(.2 * flat_dim))), flat_dim - 1)
    final_k = min(max(1, int(round(.3 * flat_dim))), flat_dim - 1)

    fit_specs = {
        "functional_vae_small": ("function", small_indices),
        "functional_vae_large": ("function", large_indices),
        "raw_vae_large": ("raw", large_indices),
    }
    models_by_method: dict[str, list[mask_ops._MaskVAE]] = {}
    vae_reports: dict[str, list[dict[str, Any]]] = {}
    reconstruction_metrics: dict[str, list[dict[str, Any]]] = {}
    example_predictions: dict[str, torch.Tensor] = {}
    state_dicts: dict[str, list[dict[str, torch.Tensor]]] = {}
    example_count = min(8, n_validation)

    for method, (representation, chosen_indices) in fit_specs.items():
        models: list[mask_ops._MaskVAE] = []
        reports: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        for task in range(4):
            if representation == "function":
                full_train = train_function_aligned[task]
                validation = validation_function_aligned[task]
                # ``small_indices`` is deliberately the nested prefix of the
                # large split, so reuse the already-aligned large-map prefix.
                task_small = full_train[:n_small]
            else:
                full_train = train_raw_aligned[task]
                validation = validation_raw_aligned[task]
                task_small = full_train
            task_train = task_small if method == "functional_vae_small" else full_train
            model_seed = int(seed) + task * 1009
            model, report = mask_ops._fit_vae(
                task_train, validation, flat_dim=flat_dim, latent=16, width=128,
                epochs=actual_epochs, seed=model_seed, device=target_device,
            )
            report = {**report, "task": bank_names[task], "seed": model_seed,
                      "representation": representation,
                      "training_subset": "nested_prefix_of_large_split"
                      if method == "functional_vae_small" else "large_training_split"}
            reconstruction = _reconstruct(model, validation)
            training_reconstruction = _reconstruct(model, task_train)
            train_mean = task_train.mean(dim=0)
            heldout_metric = _metric_bundle(reconstruction, validation, train_mean,
                                             source_k=source_k)
            training_metric = _metric_bundle(training_reconstruction, task_train,
                                             train_mean, source_k=source_k)
            metric = {
                "task": bank_names[task],
                "training": training_metric,
                "heldout": heldout_metric,
                **{f"heldout_{key}": value for key, value in heldout_metric.items()},
                # Keep these direct aliases for concise comparison tables.
                "train_mean_bce": heldout_metric["train_mean_bce"],
                "train_mean_mse": heldout_metric["train_mean_mse"],
                "train_mean_topk_iou": heldout_metric["train_mean_topk_iou"],
            }
            reports.append(report)
            metrics.append(metric)
            models.append(model)
            if task == 0:
                example_predictions[method] = reconstruction[:example_count].detach().cpu()
        models_by_method[method] = models
        vae_reports[method] = reports
        reconstruction_metrics[method] = metrics
        state_dicts[method] = [
            {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            for model in models
        ]

    alignment_orders = [value.detach() for value in alignment_orders]
    agreements: dict[str, Any] = {}
    masks: dict[str, torch.Tensor] = {}
    agreement_initial_codes: dict[str, torch.Tensor] = {}
    for method in fit_specs:
        agreement, report, initial_z = mask_ops._search_agreement(
            models_by_method[method], starts=starts, steps=actual_steps,
            seed=int(seed), k=final_k, shape=(features, hidden), device=target_device,
        )
        masks[method] = agreement.detach().cpu()
        agreements[method] = report
        agreement_initial_codes[method] = initial_z.detach().cpu()

    small_function_train = [task[:n_small] for task in train_function_aligned]
    large_function_mean = torch.cat(train_function_aligned, dim=0).mean(0)
    small_function_mean = torch.cat(small_function_train, dim=0).mean(0)
    masks["functional_mean_small"] = mask_ops._hard_topk(
        small_function_mean.reshape(1, -1).expand(starts, -1), final_k
    ).reshape(starts, features, hidden).detach().cpu()
    masks["functional_mean_large"] = mask_ops._hard_topk(
        large_function_mean.reshape(1, -1).expand(starts, -1), final_k
    ).reshape(starts, features, hidden).detach().cpu()

    random_seed = int(seed) + 81_337
    random_generator = torch.Generator(device=target_device).manual_seed(random_seed)
    masks["random"] = mask_ops._hard_topk(
        torch.rand(starts, flat_dim, generator=random_generator, device=target_device), final_k
    ).reshape(starts, features, hidden).detach().cpu()
    masks["dense"] = torch.ones(starts, features, hidden, device="cpu")
    method_order = ("functional_mean_small", "functional_mean_large",
                    "functional_vae_small", "functional_vae_large",
                    "raw_vae_large", "random", "dense")
    masks = {name: masks[name] for name in method_order}

    for name, value in masks.items():
        expected = features * hidden if name == "dense" else final_k
        if value.shape != (starts, features, hidden) or not bool(torch.all(value.sum((1, 2)) == expected)):
            raise AssertionError(f"{name} mask failed shape/exact-cardinality validation")
        if not bool(torch.all((value == 0) | (value == 1))):
            raise AssertionError(f"{name} mask is not binary")

    example_arrays: dict[str, np.ndarray] = {
        "heldout_example_source_bank_indices": validation_indices[0][:example_count].detach().cpu().numpy(),
        "heldout_example_raw_maps": validation_raw_aligned[0][:example_count].detach().cpu().numpy(),
        "heldout_example_function_maps": validation_function_aligned[0][:example_count].detach().cpu().numpy(),
    }
    for method, predictions in example_predictions.items():
        example_arrays[f"heldout_example_{method}_reconstruction"] = predictions.numpy()
    example_arrays.update({
        "heldout_source_raw_maps": example_arrays["heldout_example_raw_maps"],
        "heldout_source_function_maps": example_arrays["heldout_example_function_maps"],
        "heldout_source_functional_vae_large_reconstructions":
            example_arrays["heldout_example_functional_vae_large_reconstruction"],
        "heldout_source_raw_vae_large_reconstructions":
            example_arrays["heldout_example_raw_vae_large_reconstruction"],
    })

    split_diagnostics = []
    for task in range(4):
        split_diagnostics.append({
            "task": bank_names[task],
            "permutation": split_orders[task].detach().cpu().tolist(),
            "validation_indices": validation_indices[task].detach().cpu().tolist(),
            "large_train_indices": large_indices[task].detach().cpu().tolist(),
            "small_train_indices": small_indices[task].detach().cpu().tolist(),
            "n_validation": int(n_validation),
            "n_large_train": int(n_large),
            "n_small_train": int(n_small),
            "raw_alignment_orders_train": alignment_orders[task].detach().cpu().tolist(),
            "raw_alignment_orders_validation": validation_orders[task].detach().cpu().tolist(),
            "source_bank_candidate_ids": bank_list[task].get("selected_candidates"),
            "source_split_hashes": bank_list[task].get("source_split_hashes", {}),
        })

    diagnostics: dict[str, Any] = {
        "experiment": "expanded source banks: functional maps and VAE sample-count controls",
        "seed": int(seed),
        "device": str(target_device),
        "shape": [features, hidden],
        "source_tasks": bank_names,
        "source_bank_maps_per_task": int(n_maps),
        "source_density": .2,
        "final_density": .3,
        "source_topk_for_reconstruction_fidelity": int(source_k),
        "k": int(final_k),
        "density": final_k / flat_dim,
        "smoke": bool(smoke),
        "effective_hyperparameters": {
            "vae_epochs": actual_epochs, "agreement_steps": actual_steps,
            "starts": starts, "large_train_maps": int(n_large),
            "small_train_maps": int(n_small), "validation_maps": int(n_validation),
            "latent_dim": 16, "width": 128,
            "vae_objective": "sum BCE-with-logits per map + 0.1 * KL, averaged over maps",
            "selection": "lowest held-out source-map VAE objective; posterior mean decoder",
        },
        "score_metadata": {
            "raw": "expanded bank maps, per-model nonnegative max-normalized before VAE fitting",
            "function_gradient": score_metadata,
            "image_calibration_seed": image_seed,
            "image_calibration_count": int(image_count),
            "image_calibration_indices": image_indices.detach().cpu().tolist(),
            "labels_accessed": False,
            "alignment": "four-round raw-only Hungarian consensus on large training maps; identical orders applied to functional maps",
            "validation_alignment": "raw held-out maps aligned to final large-training raw consensus; same per-map orders applied to functional held-out maps",
        },
        "splits": split_diagnostics,
        "vae": vae_reports,
        "reconstruction_metrics": reconstruction_metrics,
        "agreement": agreements,
        "methods": list(method_order),
        "random_seed": random_seed,
        "_artifacts": {
            "state_dicts": "functional_vae_artifacts.pt: vae_state_dicts keyed by VAE method",
            "score_maps": "functional_vae_arrays.npz: raw_maps and function_maps [task,map,F,H]",
            "heldout_examples": "functional_vae_arrays.npz: first min(8,n_validation) validation maps for source task 0",
        },
    }
    artifacts = {
        "vae_state_dicts": state_dicts,
        "agreement_initial_z_first": agreement_initial_codes,
        "raw_consensus": raw_consensus.detach().cpu(),
        "alignment_orders_train": [value.detach().cpu() for value in alignment_orders],
        "alignment_orders_validation": [value.detach().cpu() for value in validation_orders],
        "split_permutations": [value.detach().cpu() for value in split_orders],
        "source_train_image_indices": image_indices.detach().cpu(),
    }
    arrays: dict[str, np.ndarray] = {
        "raw_maps": raw_normalized.detach().cpu().numpy(),
        "function_maps": function_normalized.detach().cpu().numpy(),
        "train_alignment_orders": np.stack([value.detach().cpu().numpy() for value in alignment_orders]),
        "validation_alignment_orders": np.stack([value.detach().cpu().numpy() for value in validation_orders]),
        "split_permutations": np.stack([value.detach().cpu().numpy() for value in split_orders]),
        "validation_indices": np.stack([value.detach().cpu().numpy() for value in validation_indices]),
        "large_train_indices": np.stack([value.detach().cpu().numpy() for value in large_indices]),
        "small_train_indices": np.stack([value.detach().cpu().numpy() for value in small_indices]),
        "source_train_image_indices": image_indices.detach().cpu().numpy(),
        "raw_train_aligned": torch.stack(train_raw_aligned).detach().cpu().numpy(),
        "function_train_aligned": torch.stack(train_function_aligned).detach().cpu().numpy(),
        "raw_validation_aligned": torch.stack(validation_raw_aligned).detach().cpu().numpy(),
        "function_validation_aligned": torch.stack(validation_function_aligned).detach().cpu().numpy(),
        **example_arrays,
    }
    _atomic_torch_save(out / "masks.pt", masks)
    _atomic_json(out / "functional_vae_diagnostics.json", diagnostics)
    _atomic_torch_save(out / "functional_vae_artifacts.pt", artifacts)
    _atomic_npz(out / "functional_vae_arrays.npz", **arrays)
    return masks, diagnostics
