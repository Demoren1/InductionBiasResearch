"""Converged VAE fitting and label-free extraction for expanded source maps."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import expanded_functional_vae as functional
from . import masks as mask_ops


EVAL_EVERY = 10
LATENT_DIM = 16
WIDTH = 128
LEARNING_RATE = 1e-3
KL_WEIGHT = .1
DEFAULT_MIN_STEPS = 1000
DEFAULT_MAX_STEPS = 20000
DEFAULT_WINDOW = 200
DEFAULT_PATIENCE = 400
DEFAULT_RELATIVE_TOLERANCE = .001
DEFAULT_VALIDATION_IMPROVEMENT_TOLERANCE = .0001
DEFAULT_FIT_SEED_OFFSET = 50_000
DEFAULT_AGREEMENT_STEPS = 400
MASK_DENSITY = .3


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                        allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _float_maps(value: Any, device: torch.device, name: str) -> torch.Tensor:
    maps = torch.as_tensor(value, dtype=torch.float32, device=device)
    if maps.ndim != 3 or tuple(maps.shape[1:]) != (784, 32) or maps.size(0) < 1:
        raise ValueError(f"{name} must have shape [N,784,32]")
    if not bool(torch.isfinite(maps).all()) or float(maps.min()) < 0. or float(maps.max()) > 1.:
        raise ValueError(f"{name} must be finite and lie in [0,1]")
    return maps


def _tensor_hash(value: torch.Tensor) -> str:
    content = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(content).hexdigest()


def _relative_change(previous: float, current: float) -> float:
    return abs(current - previous) / max(abs(previous), 1e-12)


def _window_pair(values: list[float], count: int) -> tuple[float, float] | None:
    if count < 1 or len(values) < 2 * count:
        return None
    previous = sum(values[-2 * count:-count]) / count
    current = sum(values[-count:]) / count
    return previous, current


def _relative_trend(values: list[float], updates_per_window: int) -> float | None:
    """Absolute linear drift over one window, scaled by the previous mean."""
    count = len(values)
    if count < 2:
        return None
    center_x = (count - 1) / 2.
    center_y = sum(values) / count
    denominator = sum((i - center_x) ** 2 for i in range(count))
    if denominator <= 0.:
        return 0.
    slope = sum((i - center_x) * (value - center_y)
                for i, value in enumerate(values)) / denominator
    return abs(slope * updates_per_window) / max(abs(center_y), 1e-12)


def _plateau_metrics(evaluations: list[dict[str, Any]], stochastic: list[float],
                     *, step: int, window: int, patience: int,
                     last_significant_improvement_step: int,
                     relative_tolerance: float,
                     consecutive_checks: int) -> tuple[dict[str, Any], int]:
    eval_count = window // EVAL_EVERY
    eval_rows = [row for row in evaluations if row["step"] > 0]
    deterministic = [float(row["deterministic_train_objective"]) for row in eval_rows]
    validation = [float(row["validation_objective"]) for row in eval_rows]
    deterministic_pair = _window_pair(deterministic, eval_count)
    validation_pair = _window_pair(validation, eval_count)
    stochastic_pair = _window_pair(stochastic, window)
    values: dict[str, Any] = {
        "window_updates": window,
        "evaluation_interval_updates": EVAL_EVERY,
        "relative_tolerance": relative_tolerance,
        "validation_improvement_patience_updates": patience,
        "updates_since_significant_validation_improvement":
            step - last_significant_improvement_step,
        "last_significant_validation_improvement_step": last_significant_improvement_step,
        "stochastic_window_mean_previous": None,
        "stochastic_window_mean_current": None,
        "deterministic_train_window_mean_previous": None,
        "deterministic_train_window_mean_current": None,
        "validation_window_mean_previous": None,
        "validation_window_mean_current": None,
        "stochastic_train_relative_window_change": None,
        "deterministic_train_relative_window_change": None,
        "validation_relative_window_change": None,
        "stochastic_train_linear_trend_relative_drift": None,
        "deterministic_train_linear_trend_relative_drift": None,
        "validation_linear_trend_relative_drift": None,
        "plateau_checks_consecutive": consecutive_checks,
        "required_consecutive_checks": 3,
        "eligible": False,
    }
    if deterministic_pair is None or validation_pair is None or stochastic_pair is None:
        return values, 0

    prev, curr = stochastic_pair
    values["stochastic_window_mean_previous"] = prev
    values["stochastic_window_mean_current"] = curr
    values["stochastic_train_relative_window_change"] = _relative_change(prev, curr)
    values["stochastic_train_linear_trend_relative_drift"] = _relative_trend(
        stochastic[-2 * window:], window)

    prev, curr = deterministic_pair
    values["deterministic_train_window_mean_previous"] = prev
    values["deterministic_train_window_mean_current"] = curr
    values["deterministic_train_relative_window_change"] = _relative_change(prev, curr)
    values["deterministic_train_linear_trend_relative_drift"] = _relative_trend(
        deterministic[-2 * eval_count:], eval_count)

    prev, curr = validation_pair
    values["validation_window_mean_previous"] = prev
    values["validation_window_mean_current"] = curr
    values["validation_relative_window_change"] = _relative_change(prev, curr)
    values["validation_linear_trend_relative_drift"] = _relative_trend(
        validation[-2 * eval_count:], eval_count)

    plateau = all(
        values[key] is not None and values[key] <= relative_tolerance
        for key in (
            "stochastic_train_relative_window_change",
            "deterministic_train_relative_window_change",
            "validation_relative_window_change",
            "stochastic_train_linear_trend_relative_drift",
            "deterministic_train_linear_trend_relative_drift",
            "validation_linear_trend_relative_drift",
        )
    )
    patience_ok = step - last_significant_improvement_step >= patience
    eligible = plateau and patience_ok
    streak = consecutive_checks + 1 if eligible else 0
    values.update(plateau_checks_consecutive=streak, eligible=eligible,
                  plateau_objectives_passed=plateau,
                  validation_improvement_patience_passed=patience_ok)
    return values, streak


def _deterministic_objective(model: mask_ops._MaskVAE, flat_maps: torch.Tensor) -> torch.Tensor:
    mu, logvar = model.encode(flat_maps)
    return mask_ops._vae_loss(model.decode(mu), flat_maps, mu, logvar)


def _cpu_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _write_fit_outputs(out: Path, *, report: dict[str, Any], evaluations: list[dict[str, Any]],
                       stochastic_losses: list[float], best_state: dict[str, torch.Tensor],
                       config: dict[str, Any]) -> None:
    _atomic_torch_save(out / "best_model.pt", {
        "state_dict": _cpu_state(best_state), "config": config, "report": report,
    })
    curve_payload = {
        "objective": "full-batch stochastic ELBO: summed BCE-with-logits + 0.1 * KL, mean over maps",
        "deterministic_objective": "same ELBO evaluated at posterior means",
        "evaluation_interval_updates": EVAL_EVERY,
        "stochastic_train_loss_by_update": stochastic_losses,
        "evaluations": evaluations,
    }
    _atomic_json(out / "loss_curves.json", curve_payload)
    eval_steps = np.asarray([row["step"] for row in evaluations], dtype=np.int64)
    _atomic_npz(
        out / "loss_curves.npz",
        stochastic_train_loss_by_update=np.asarray(stochastic_losses, dtype=np.float32),
        evaluation_step=eval_steps,
        stochastic_train_window_mean=np.asarray(
            [np.nan if row.get("stochastic_train_window_mean") is None
             else row["stochastic_train_window_mean"] for row in evaluations], dtype=np.float32),
        deterministic_train_objective=np.asarray(
            [row["deterministic_train_objective"] for row in evaluations], dtype=np.float32),
        validation_objective=np.asarray(
            [row["validation_objective"] for row in evaluations], dtype=np.float32),
    )
    _atomic_json(out / "fit_report.json", report)


def fit_vae_to_plateau(
    train: Any,
    valid: Any,
    *,
    seed: int,
    device: str | torch.device,
    out: Path,
    min_steps: int = DEFAULT_MIN_STEPS,
    max_steps: int = DEFAULT_MAX_STEPS,
    window: int = DEFAULT_WINDOW,
    patience: int = DEFAULT_PATIENCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    validation_improvement_tolerance: float = DEFAULT_VALIDATION_IMPROVEMENT_TOLERANCE,
) -> tuple[mask_ops._MaskVAE, dict[str, Any]]:
    """Fit the existing VAE until stochastic, train, and validation losses plateau.

    Training uses one full stochastic ELBO update per step.  Validation and
    posterior-mean training objectives are evaluated every ten updates.  At
    the cap, the last optimizer/RNG state is saved for a later larger-cap
    continuation; the returned model always contains the best validation
    checkpoint.
    """
    if min_steps < 0 or max_steps < 1:
        raise ValueError("require nonnegative min_steps and positive max_steps")
    if window < EVAL_EVERY or window % EVAL_EVERY:
        raise ValueError("window must be a positive multiple of the 10-update evaluation interval")
    if patience < 0 or relative_tolerance <= 0. or validation_improvement_tolerance < 0.:
        raise ValueError("invalid plateau thresholds")
    target_device = torch.device(device)
    train_maps = _float_maps(train, target_device, "train")
    valid_maps = _float_maps(valid, target_device, "valid")
    if train_maps.shape[1:] != valid_maps.shape[1:]:
        raise ValueError("training and validation maps must have the same feature shape")
    flat_dim = int(train_maps[0].numel())
    train_flat = train_maps.reshape(len(train_maps), flat_dim)
    valid_flat = valid_maps.reshape(len(valid_maps), flat_dim)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    resume_path = out / "resume_checkpoint.pt"
    best_path = out / "best_model.pt"

    config = {
        "seed": int(seed), "device": str(target_device),
        "map_shape": list(train_maps.shape[1:]), "flat_dim": flat_dim,
        "train_maps": int(len(train_maps)), "validation_maps": int(len(valid_maps)),
        "train_sha256": _tensor_hash(train_maps), "validation_sha256": _tensor_hash(valid_maps),
        "latent_dim": LATENT_DIM, "width": WIDTH,
        "learning_rate": LEARNING_RATE, "kl_weight": KL_WEIGHT,
        "min_steps": int(min_steps), "window": int(window), "patience": int(patience),
        "relative_tolerance": float(relative_tolerance),
        "validation_improvement_tolerance": float(validation_improvement_tolerance),
        "evaluation_interval_updates": EVAL_EVERY,
    }
    fixed_config = dict(config)

    if best_path.exists() and not resume_path.exists():
        saved = torch.load(best_path, map_location=target_device, weights_only=False)
        saved_config = saved.get("config", {})
        if any(saved_config.get(key) != value for key, value in fixed_config.items()):
            raise ValueError("existing converged fit does not match training data/configuration")
        old_report = saved.get("report")
        if not isinstance(old_report, dict) or not old_report.get("converged"):
            raise FileExistsError(f"fit artifacts exist without resumable state: {out}")
        with mask_ops._local_torch_seed(int(seed), target_device):
            model = mask_ops._MaskVAE(flat_dim, LATENT_DIM, WIDTH).to(target_device)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        return model, old_report

    with mask_ops._local_torch_seed(int(seed), target_device):
        model = mask_ops._MaskVAE(flat_dim, LATENT_DIM, WIDTH).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    latent_rng = torch.Generator(device=target_device).manual_seed(int(seed) + 991)
    evaluations: list[dict[str, Any]] = []
    stochastic_losses: list[float] = []
    step = 0
    best_step = 0
    best_validation = float("inf")
    best_state = _cpu_state(model.state_dict())
    last_significant_improvement_step = 0
    plateau_streak = 0
    latest_convergence: dict[str, Any] = {"eligible": False, "plateau_checks_consecutive": 0}
    previous_cap = 0
    resume_from_step = 0

    if resume_path.exists():
        saved = torch.load(resume_path, map_location=target_device, weights_only=False)
        saved_config = saved.get("config", {})
        if any(saved_config.get(key) != value for key, value in fixed_config.items()):
            raise ValueError("resume checkpoint does not match training data/configuration")
        if saved.get("converged"):
            state = saved["best_state"]
            with mask_ops._local_torch_seed(int(seed), target_device):
                model = mask_ops._MaskVAE(flat_dim, LATENT_DIM, WIDTH).to(target_device)
            model.load_state_dict(state)
            model.eval()
            return model, saved["report"]
        model.load_state_dict(saved["last_model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        latent_rng.set_state(saved["latent_rng_state"])
        step = int(saved["step"])
        resume_from_step = step
        previous_cap = int(saved["max_steps_cap"])
        evaluations = saved["evaluations"]
        stochastic_losses = saved["stochastic_train_loss_by_update"]
        best_step = int(saved["best_step"])
        best_validation = float(saved["best_validation_objective"])
        best_state = saved["best_state"]
        last_significant_improvement_step = int(saved["last_significant_validation_improvement_step"])
        plateau_streak = int(saved["plateau_streak"])
        latest_convergence = saved["convergence_metrics"]
        if max_steps <= previous_cap:
            model.load_state_dict(best_state)
            model.eval()
            return model, saved["report"]

    elif any((out / filename).exists() for filename in
             ("fit_report.json", "loss_curves.json", "loss_curves.npz")):
        raise FileExistsError(f"fit outputs exist without resume checkpoint: {out}")

    if not evaluations:
        model.eval()
        with torch.no_grad():
            initial_train = _deterministic_objective(model, train_flat)
            initial_valid = _deterministic_objective(model, valid_flat)
            initial_values = torch.stack((initial_train, initial_valid)).detach().cpu().tolist()
        best_validation = float(initial_values[1])
        best_state = _cpu_state(model.state_dict())
        evaluations.append({"step": 0,
                            "deterministic_train_objective": float(initial_values[0]),
                            "validation_objective": float(initial_values[1]),
                            "stochastic_train_window_mean": None})

    converged = False
    stochastic_chunk: list[torch.Tensor] = []
    gradient_finite_chunk: list[torch.Tensor] = []
    while step < max_steps:
        step += 1
        model.train()
        logits, mu, logvar = model(train_flat, latent_rng)
        loss = mask_ops._vae_loss(logits, train_flat, mu, logvar)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_flags = [torch.isfinite(parameter.grad).all()
                      for parameter in model.parameters() if parameter.grad is not None]
        finite_grad = torch.stack(grad_flags).all() if grad_flags else torch.tensor(False, device=target_device)
        gradient_finite_chunk.append(finite_grad.detach())
        stochastic_chunk.append(loss.detach())
        optimizer.step()

        if step % EVAL_EVERY == 0 or step == max_steps:
            stochastic_tensor = torch.stack(stochastic_chunk)
            check_tensor = torch.stack(gradient_finite_chunk).to(torch.float32)
            model.eval()
            with torch.no_grad():
                deterministic_train = _deterministic_objective(model, train_flat)
                validation = _deterministic_objective(model, valid_flat)
                packed = torch.cat((stochastic_tensor,
                                    deterministic_train.reshape(1),
                                    validation.reshape(1),
                                    check_tensor)).detach().cpu()
            chunk_size = len(stochastic_chunk)
            chunk_losses = packed[:chunk_size].tolist()
            deterministic_value = float(packed[chunk_size])
            validation_value = float(packed[chunk_size + 1])
            finite_flags = packed[chunk_size + 2:]
            if not all(math.isfinite(value) for value in chunk_losses +
                       [deterministic_value, validation_value]) or not bool(torch.all(finite_flags == 1)):
                raise FloatingPointError(f"non-finite loss or gradient at update {step}")
            stochastic_losses.extend(float(value) for value in chunk_losses)
            stochastic_window_mean = float(sum(chunk_losses) / len(chunk_losses))
            stochastic_chunk.clear()
            gradient_finite_chunk.clear()

            if validation_value < best_validation:
                relative_improvement = _relative_change(best_validation, validation_value)
                if relative_improvement > validation_improvement_tolerance:
                    last_significant_improvement_step = step
                best_validation = validation_value
                best_step = step
                best_state = _cpu_state(model.state_dict())

            evaluations.append({
                "step": step,
                "stochastic_train_window_mean": stochastic_window_mean,
                "deterministic_train_objective": deterministic_value,
                "validation_objective": validation_value,
            })
            if step % EVAL_EVERY == 0:
                latest_convergence, plateau_streak = _plateau_metrics(
                    evaluations, stochastic_losses, step=step, window=window,
                    patience=patience,
                    last_significant_improvement_step=last_significant_improvement_step,
                    relative_tolerance=relative_tolerance,
                    consecutive_checks=plateau_streak,
                )
                evaluations[-1]["convergence_metrics"] = latest_convergence
                if step >= min_steps and plateau_streak >= 3:
                    converged = True
                    break

    stop_step = step
    status = "converged" if converged else "max_steps_reached"
    report = {
        "seed": int(seed), "device": str(target_device),
        "converged": converged, "convergence_status": status,
        "stop_step": stop_step, "max_steps_cap": max_steps,
        "best_step": best_step, "best_validation_objective": best_validation,
        "last_validation_objective": evaluations[-1]["validation_objective"],
        "last_deterministic_train_objective": evaluations[-1]["deterministic_train_objective"],
        "last_stochastic_train_objective": stochastic_losses[-1] if stochastic_losses else None,
        "train_maps": int(len(train_maps)), "validation_maps": int(len(valid_maps)),
        "epochs": stop_step, "latent_dim": LATENT_DIM, "width": WIDTH,
        "learning_rate": LEARNING_RATE, "kl_weight": KL_WEIGHT,
        "objective": "full-batch stochastic ELBO: summed BCE-with-logits + 0.1 * KL, mean over maps",
        "deterministic_objective": "same ELBO evaluated at posterior means",
        "convergence_scope": "numerical objective plateau on fixed source maps; this is not a global-optimality guarantee",
        "convergence_criteria": {
            "min_steps": min_steps, "max_steps": max_steps, "window_updates": window,
            "evaluation_interval_updates": EVAL_EVERY,
            "relative_window_tolerance": relative_tolerance,
            "validation_improvement_tolerance": validation_improvement_tolerance,
            "validation_improvement_patience_updates": patience,
            "required_consecutive_eligible_checks": 3,
            "objectives_required": ["stochastic_train", "deterministic_train", "validation"],
            "linear_trend_drift_required": True,
        },
        "convergence_metrics": latest_convergence,
        "initialization": "local torch RNG seed; stochastic latent generator seed+991",
        "resume_supported": True,
        "resume_from_step": None if resume_from_step == 0 else resume_from_step,
    }
    _write_fit_outputs(out, report=report, evaluations=evaluations,
                       stochastic_losses=stochastic_losses,
                       best_state=best_state, config=config)
    if converged:
        if resume_path.exists():
            resume_path.unlink()
    else:
        _atomic_torch_save(resume_path, {
            "config": config, "max_steps_cap": max_steps, "step": stop_step,
            "last_model_state": _cpu_state(model.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "latent_rng_state": latent_rng.get_state(),
            "evaluations": evaluations,
            "stochastic_train_loss_by_update": stochastic_losses,
            "best_state": best_state, "best_step": best_step,
            "best_validation_objective": best_validation,
            "last_significant_validation_improvement_step": last_significant_improvement_step,
            "plateau_streak": plateau_streak,
            "convergence_metrics": latest_convergence,
            "converged": False, "report": report,
        })
    model.load_state_dict(best_state)
    model.eval()
    return model, report


def _extract_metric(model: mask_ops._MaskVAE, train_maps: torch.Tensor,
                    valid_maps: torch.Tensor, *, source_k: int) -> dict[str, Any]:
    train_reconstruction = functional._reconstruct(model, train_maps)
    valid_reconstruction = functional._reconstruct(model, valid_maps)
    train_mean = train_maps.mean(0)
    train_report = functional._metric_bundle(train_reconstruction, train_maps,
                                             train_mean, source_k=source_k)
    valid_report = functional._metric_bundle(valid_reconstruction, valid_maps,
                                             train_mean, source_k=source_k)
    return {
        "training": train_report, "heldout": valid_report,
        **{f"heldout_{key}": value for key, value in valid_report.items()},
        "train_mean_bce": valid_report["train_mean_bce"],
        "train_mean_mse": valid_report["train_mean_mse"],
        "train_mean_topk_iou": valid_report["train_mean_topk_iou"],
        "heldout_reconstruction_mse": valid_report["reconstruction_mse"],
    }


def extract_converged_functional_masks(
    input_seed_folder: Path,
    out: Path,
    seed: int,
    device: str | torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Fit converged VAE families using cached, source-only aligned maps."""
    source = Path(input_seed_folder)
    out = Path(out)
    target_device = torch.device(device)
    arrays_path = source / "functional_vae_arrays.npz"
    masks_path = source / "masks.pt"
    if not arrays_path.is_file() or not masks_path.is_file():
        raise FileNotFoundError(f"source cache requires {arrays_path} and {masks_path}")
    output_files = ("masks.pt", "functional_vae_diagnostics.json",
                    "functional_vae_arrays.npz", "functional_vae_artifacts.pt")
    out.mkdir(parents=True, exist_ok=True)
    diagnostics_path = out / "functional_vae_diagnostics.json"
    previous_diagnostics = None
    if diagnostics_path.exists():
        previous_diagnostics = json.loads(diagnostics_path.read_text())
        if previous_diagnostics.get("converged"):
            raise FileExistsError(f"converged extraction already exists: {out}")
    elif any((out / name).exists() for name in output_files if name != "functional_vae_diagnostics.json"):
        raise FileExistsError(f"partial extraction outputs exist without diagnostics: {out}")
    effective_max_steps = DEFAULT_MAX_STEPS
    if previous_diagnostics is not None:
        effective_max_steps = int(previous_diagnostics.get("training", {}).get(
            "max_steps_per_fit", DEFAULT_MAX_STEPS)) + DEFAULT_MAX_STEPS

    with np.load(arrays_path) as archive:
        required = ("raw_train_aligned", "function_train_aligned",
                    "raw_validation_aligned", "function_validation_aligned")
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"source cache lacks aligned map arrays: {missing}")
        arrays = {key: archive[key] for key in archive.files}
    raw_train = torch.as_tensor(arrays["raw_train_aligned"], dtype=torch.float32,
                                device=target_device)
    function_train = torch.as_tensor(arrays["function_train_aligned"], dtype=torch.float32,
                                     device=target_device)
    raw_valid = torch.as_tensor(arrays["raw_validation_aligned"], dtype=torch.float32,
                                device=target_device)
    function_valid = torch.as_tensor(arrays["function_validation_aligned"], dtype=torch.float32,
                                     device=target_device)
    if (raw_train.ndim != 4 or tuple(raw_train.shape[::3]) != (4, 32)
            or tuple(raw_train.shape[2:]) != (784, 32)):
        raise ValueError(f"raw_train_aligned must have shape [4,N,784,32], got {tuple(raw_train.shape)}")
    if function_train.shape != raw_train.shape:
        raise ValueError("raw and functional training map arrays must have matching shapes")
    if (raw_valid.ndim != 4 or raw_valid.shape[0] != 4
            or tuple(raw_valid.shape[2:]) != (784, 32)):
        raise ValueError("raw_validation_aligned must have shape [4,N,784,32]")
    if function_valid.shape != raw_valid.shape:
        raise ValueError("raw and functional validation map arrays must have matching shapes")
    for name, value in (("raw train", raw_train), ("functional train", function_train),
                        ("raw validation", raw_valid), ("functional validation", function_valid)):
        if not bool(torch.isfinite(value).all()) or float(value.min()) < 0. or float(value.max()) > 1.:
            raise ValueError(f"{name} maps must be finite and lie in [0,1]")

    old_masks = torch.load(masks_path, map_location="cpu", weights_only=True)
    base_names = ("functional_mean_small", "functional_mean_large", "random", "dense")
    if any(name not in old_masks for name in base_names):
        raise ValueError(f"source masks lack direct controls: {base_names}")
    large_count = raw_train.size(1)
    small_count = min(26, large_count)
    validation_count = raw_valid.size(1)
    if min(large_count, small_count, validation_count) < 1:
        raise ValueError("source cache has empty train or validation maps")

    families = {
        "functional_vae_small": (function_train, function_valid, small_count),
        "functional_vae_large": (function_train, function_valid, large_count),
        "raw_vae_large": (raw_train, raw_valid, large_count),
    }
    starts = int(torch.as_tensor(old_masks["random"]).shape[0])
    models_by_family: dict[str, list[mask_ops._MaskVAE]] = {}
    vae_reports: dict[str, list[dict[str, Any]]] = {}
    reconstruction_metrics: dict[str, list[dict[str, Any]]] = {}
    state_dicts: dict[str, list[dict[str, torch.Tensor]]] = {}
    output_arrays = {key: np.asarray(value) for key, value in arrays.items()}
    example_count = min(8, validation_count)
    for family, (training_maps, validation_maps, train_count) in families.items():
        models: list[mask_ops._MaskVAE] = []
        reports: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        states: list[dict[str, torch.Tensor]] = []
        for task in range(4):
            task_train = training_maps[task, :train_count]
            task_valid = validation_maps[task]
            model_seed = int(seed) + DEFAULT_FIT_SEED_OFFSET + task * 1009
            fit_out = out / "fits" / family / f"task_{task}"
            model, fit_report = fit_vae_to_plateau(
                task_train, task_valid, seed=model_seed, device=target_device,
                out=fit_out, max_steps=effective_max_steps,
            )
            fit_report = dict(fit_report)
            fit_report.update({
                "task": task, "family": family,
                "loss_curve_path": str(fit_out / "loss_curves.json"),
                "checkpoint_path": str(fit_out / "best_model.pt"),
                "train_maps": int(len(task_train)),
                "validation_maps": int(len(task_valid)),
            })
            metric = _extract_metric(model, task_train, task_valid, source_k=5018)
            metric["task"] = task
            models.append(model)
            reports.append(fit_report)
            metrics.append(metric)
            states.append(_cpu_state(model.state_dict()))
            if task == 0:
                validation_prediction = functional._reconstruct(model, task_valid)
                output_arrays[f"heldout_example_{family}_reconstruction"] = (
                    validation_prediction[:example_count].detach().cpu().numpy())
        models_by_family[family] = models
        vae_reports[family] = reports
        reconstruction_metrics[family] = metrics
        state_dicts[family] = states

    agreement_reports: dict[str, Any] = {}
    new_masks: dict[str, torch.Tensor] = {}
    agreement_initial: dict[str, torch.Tensor] = {}
    final_k = 7526
    agreement_seed = int(seed) + DEFAULT_FIT_SEED_OFFSET
    for family, models in models_by_family.items():
        values, report, initial_z = mask_ops._search_agreement(
            models, starts=starts, steps=DEFAULT_AGREEMENT_STEPS,
            seed=agreement_seed, k=final_k, shape=(784, 32), device=target_device,
        )
        new_masks[family] = values.detach().cpu()
        agreement_reports[family] = report
        agreement_initial[family] = initial_z.detach().cpu()

    method_order = ("functional_mean_small", "functional_mean_large",
                    "functional_vae_small", "functional_vae_large",
                    "raw_vae_large", "random", "dense")
    masks: dict[str, torch.Tensor] = {
        "functional_mean_small": torch.as_tensor(old_masks["functional_mean_small"]).clone(),
        "functional_mean_large": torch.as_tensor(old_masks["functional_mean_large"]).clone(),
        **new_masks,
        "random": torch.as_tensor(old_masks["random"]).clone(),
        "dense": torch.as_tensor(old_masks["dense"]).clone(),
    }
    masks = {name: masks[name] for name in method_order}
    for name, mask in masks.items():
        expected = 25088 if name == "dense" else final_k
        if tuple(mask.shape) != (starts, 784, 32) or not bool(torch.all(mask.sum((1, 2)) == expected)):
            raise AssertionError(f"{name} returned an invalid shape/cardinality")
        if not bool(torch.all((mask == 0) | (mask == 1))):
            raise AssertionError(f"{name} mask is not binary")

    output_arrays["heldout_source_raw_maps"] = output_arrays["heldout_example_raw_maps"]
    output_arrays["heldout_source_function_maps"] = output_arrays["heldout_example_function_maps"]
    output_arrays["heldout_source_functional_vae_large_reconstructions"] = output_arrays[
        "heldout_example_functional_vae_large_reconstruction"]
    output_arrays["heldout_source_raw_vae_large_reconstructions"] = output_arrays[
        "heldout_example_raw_vae_large_reconstruction"]
    for family in families:
        recon_key = f"heldout_example_{family}_reconstruction"
        source_key = f"heldout_source_{family}_reconstructions"
        output_arrays[source_key] = output_arrays[recon_key]

    all_converged = all(report["converged"]
                        for family_reports in vae_reports.values()
                        for report in family_reports)
    diagnostics: dict[str, Any] = {
        "experiment": "converged functional VAE sample-count and representation comparison",
        "seed": agreement_seed,
        "experiment_seed": int(seed),
        "device": str(target_device),
        "shape": [784, 32], "k": final_k, "density": final_k / (784 * 32),
        "source_tasks": [0, 1, 2, 3],
        "source_train_maps_per_task": int(large_count),
        "small_train_maps_per_task": int(small_count),
        "validation_maps_per_task": int(validation_count),
        "converged": all_converged,
        "convergence_status": "all_12_fits_converged" if all_converged else "one_or_more_fits_reached_cap",
        "target_labels_used": False,
        "source_data": "cached source-only raw/function maps and the source-map validation split",
        "training": {
            "method": "full-batch stochastic ELBO, summed BCE-with-logits + 0.1*KL averaged over maps",
            "monitoring": "stochastic train loss each update; deterministic posterior-mean train and validation objectives every 10 updates",
            "plateau": "stochastic train, deterministic train, and validation adjacent-window means and linear trends; three consecutive eligible checks; validation improvement patience",
            "max_steps_per_fit": effective_max_steps,
        },
        "methods": list(method_order),
        "vae": vae_reports,
        "families": vae_reports,
        "reconstruction_metrics": reconstruction_metrics,
        "agreement": agreement_reports,
        "source_cache": str(arrays_path),
        "source_mask_cache": str(masks_path),
    }
    artifact_payload = {
        "vae_state_dicts": state_dicts,
        "agreement_initial_z_first": agreement_initial,
        "input_seed_folder": str(source),
        "fit_reports": vae_reports,
    }
    _atomic_torch_save(out / "masks.pt", masks)
    _atomic_json(out / "functional_vae_diagnostics.json", diagnostics)
    _atomic_torch_save(out / "functional_vae_artifacts.pt", artifact_payload)
    _atomic_npz(out / "functional_vae_arrays.npz", **output_arrays)
    return masks, diagnostics
