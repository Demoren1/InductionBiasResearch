"""Exact source-only replay diagnostics for the converged functional VAEs.

The production fits and target evaluations are frozen.  This module replays
only the cached source-map VAE fits on the same assigned GPUs, decomposes the
optimized ELBO into reconstruction and KL terms, and measures latent usage at
the best-validation and final-stop checkpoints.  It does not load target data
or labels, and writes only beneath ``latent_diagnostics``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from . import expanded_converged_vae as converged
from . import masks as mask_ops
from .followup_common import configure


ROOT = Path(__file__).resolve().parents[1]
CONVERGED_ROOT = ROOT / "outputs/deepsets_vaae/20261001_converged_functional_vae"
DEFAULT_OUT = CONVERGED_ROOT / "latent_diagnostics"
SEEDS = tuple(range(4100, 4108))
FAMILIES = ("functional_vae_small", "functional_vae_large", "raw_vae_large")
TASKS = tuple(range(4))
EVAL_EVERY = 10
LATENT_DIM = 16
WIDTH = 128
KL_WEIGHT = .1
ACTIVE_UNIT_VARIANCE_THRESHOLD = .01
AGREEMENT_RADIUS = 3. * float(np.sqrt(LATENT_DIM))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        temp.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False,
                                   allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        temp.write_text(value, encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _cpu_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _state_comparison(actual: Mapping[str, torch.Tensor],
                      expected: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    if set(actual) != set(expected):
        return {"exact_match": False, "max_abs_delta": None,
                "missing_keys": sorted(set(expected) - set(actual)),
                "extra_keys": sorted(set(actual) - set(expected))}
    max_delta = 0.0
    exact = True
    for key in actual:
        left = actual[key].detach().cpu()
        right = torch.as_tensor(expected[key]).detach().cpu()
        if left.shape != right.shape:
            return {"exact_match": False, "max_abs_delta": None,
                    "shape_mismatch": key}
        if left.numel():
            delta = float((left.to(torch.float64) - right.to(torch.float64)).abs().max())
            max_delta = max(max_delta, delta)
        exact = exact and torch.equal(left, right)
    return {"exact_match": exact, "max_abs_delta": max_delta}


def _component_terms(logits: torch.Tensor, target: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean summed BCE and per-coordinate mean KL, matching _vae_loss."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(-1).mean()
    per_dim_kl = -.5 * (1. + logvar - mu.square() - logvar.exp()).mean(0)
    return bce, per_dim_kl


def _target_entropy(maps: torch.Tensor) -> dict[str, Any]:
    """Bernoulli entropy floor for each soft source target map."""
    flat = maps.reshape(len(maps), -1).clamp(0., 1.)
    # xlogy defines 0*log(0)=0, including exact binary source-map entries.
    per_map = -(torch.special.xlogy(flat, flat) +
                torch.special.xlogy(1. - flat, 1. - flat)).sum(dim=1)
    return {
        "per_map_sum": per_map.detach().cpu().tolist(),
        "mean_per_map_sum": float(per_map.mean().detach().cpu()),
        "mean_per_entry": float(per_map.mean().detach().cpu()) / flat.size(1),
        "std_per_map_sum": float(per_map.std(unbiased=False).detach().cpu()),
        "min_per_map_sum": float(per_map.min().detach().cpu()),
        "max_per_map_sum": float(per_map.max().detach().cpu()),
        "maps": int(len(maps)),
        "features_per_map": int(flat.size(1)),
    }


def _deterministic_components(model: mask_ops._MaskVAE,
                              flat_maps: torch.Tensor) -> dict[str, Any]:
    mu, logvar = model.encode(flat_maps)
    logits = model.decode(mu)
    # Keep the exact original loss implementation for checkpoint selection.
    objective = mask_ops._vae_loss(logits, flat_maps, mu, logvar)
    bce, kl_by_dimension = _component_terms(logits, flat_maps, mu, logvar)
    return {
        "objective": float(objective.detach().cpu()),
        "bce": float(bce.detach().cpu()),
        "kl_total": float(kl_by_dimension.sum().detach().cpu()),
        "kl_per_dimension": kl_by_dimension.detach().cpu().numpy().astype(np.float64),
    }


def _latent_statistics(model: mask_ops._MaskVAE, maps: torch.Tensor) -> dict[str, Any]:
    flat = maps.reshape(len(maps), -1)
    with torch.no_grad():
        mu, logvar = model.encode(flat)
        mean_variance = mu.var(dim=0, unbiased=False)
        covariance = (mu - mu.mean(dim=0)).transpose(0, 1) @ (mu - mu.mean(dim=0))
        covariance = covariance / max(len(mu), 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.)
        noise_by_dimension = logvar.exp().mean(dim=0)
        mu_norm = mu.norm(dim=1)
        signal_sum = mean_variance.sum()
        noise_sum = noise_by_dimension.sum()
        eigen_sum = eigenvalues.sum()
        if float(eigen_sum) > 0.:
            probabilities = eigenvalues / eigen_sum
            positive = probabilities > 0.
            effective_rank = float(torch.exp(-(probabilities[positive] *
                                               probabilities[positive].log()).sum()))
        else:
            effective_rank = 0.0
        eigen_square_sum = eigenvalues.square().sum()
        participation_ratio = (float(eigen_sum.square() / eigen_square_sum)
                               if float(eigen_square_sum) > 0. else 0.0)
        signal_value = float(signal_sum)
        noise_value = float(noise_sum)
        return {
            "posterior_mean_variance_by_dimension": mean_variance.detach().cpu().tolist(),
            "active_units_threshold": ACTIVE_UNIT_VARIANCE_THRESHOLD,
            "active_units": int((mean_variance > ACTIVE_UNIT_VARIANCE_THRESHOLD).sum()),
            "posterior_mean_covariance_eigenvalues": eigenvalues.detach().cpu().tolist(),
            "posterior_mean_covariance_effective_rank": effective_rank,
            "posterior_mean_covariance_participation_ratio": participation_ratio,
            "posterior_noise_variance_by_dimension": noise_by_dimension.detach().cpu().tolist(),
            "posterior_mean_signal_variance_sum": signal_value,
            "posterior_noise_variance_sum": noise_value,
            "posterior_noise_to_signal_ratio": noise_value / max(signal_value, 1e-12),
            "posterior_mean_signal_variance_mean_per_dimension": signal_value / mu.size(1),
            "posterior_noise_variance_mean_per_dimension": noise_value / mu.size(1),
            "posterior_mean_vector_norm_mean": float(mu_norm.mean()),
            "posterior_mean_vector_norm_p50": float(torch.quantile(mu_norm, .50)),
            "posterior_mean_vector_norm_p95": float(torch.quantile(mu_norm, .95)),
            "posterior_mean_vector_norm_max": float(mu_norm.max()),
            "agreement_search_radius": AGREEMENT_RADIUS,
            "fraction_posterior_mean_norm_above_agreement_radius":
                float((mu_norm > AGREEMENT_RADIUS).to(torch.float32).mean()),
            "maps": int(len(maps)),
        }


def replay_fit(
    train: torch.Tensor,
    valid: torch.Tensor,
    *,
    fit_seed: int,
    stop_step: int,
    expected_stochastic_trace: list[float] | np.ndarray,
    expected_best_state: Mapping[str, torch.Tensor],
    device: str | torch.device,
    eval_every: int = EVAL_EVERY,
    latent_dim: int = LATENT_DIM,
    width: int = WIDTH,
) -> dict[str, Any]:
    """Replay one full-batch source-map fit and return losses/parity/latent stats."""
    if stop_step < 1 or stop_step % eval_every:
        raise ValueError("stop_step must be a positive multiple of eval_every")
    target_device = torch.device(device)
    train = torch.as_tensor(train, dtype=torch.float32, device=target_device)
    valid = torch.as_tensor(valid, dtype=torch.float32, device=target_device)
    if train.ndim < 2 or valid.ndim < 2 or train.shape[1:] != valid.shape[1:]:
        raise ValueError("train and valid must have matching [N,...] shapes")
    if train.size(0) < 1 or valid.size(0) < 1:
        raise ValueError("train and validation splits must be nonempty")
    if (not bool(torch.isfinite(train).all()) or not bool(torch.isfinite(valid).all())
            or float(train.min()) < 0. or float(train.max()) > 1.
            or float(valid.min()) < 0. or float(valid.max()) > 1.):
        raise ValueError("source map inputs must be finite and in [0,1]")
    flat_dim = int(train[0].numel())
    train_flat = train.reshape(len(train), flat_dim)
    valid_flat = valid.reshape(len(valid), flat_dim)
    if len(expected_stochastic_trace) != stop_step:
        raise ValueError("stored stochastic trace length differs from stop_step")

    with mask_ops._local_torch_seed(int(fit_seed), target_device):
        model = mask_ops._MaskVAE(flat_dim, latent_dim, width).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=converged.LEARNING_RATE)
    latent_rng = torch.Generator(device=target_device).manual_seed(int(fit_seed) + 991)
    entropy_train = _target_entropy(train_flat)
    entropy_valid = _target_entropy(valid_flat)

    update_losses: list[float] = []
    evaluation_rows: list[dict[str, Any]] = []
    stochastic_chunk: list[torch.Tensor] = []
    bce_chunk: list[torch.Tensor] = []
    kl_chunk: list[torch.Tensor] = []
    kl_dimension_chunk: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        initial_train = _deterministic_components(model, train_flat)
        initial_valid = _deterministic_components(model, valid_flat)
    initial_train["bce_minus_target_bernoulli_entropy"] = (
        initial_train["bce"] - entropy_train["mean_per_map_sum"])
    initial_valid["bce_minus_target_bernoulli_entropy"] = (
        initial_valid["bce"] - entropy_valid["mean_per_map_sum"])
    evaluation_rows.append({
        "step": 0,
        "stochastic_elbo_window_mean": None,
        "stochastic_bce_window_mean": None,
        "stochastic_kl_total_window_mean": None,
        "stochastic_kl_per_dimension_window_mean": None,
        "deterministic_train": initial_train,
        "validation": initial_valid,
    })
    best_step = 0
    best_validation = initial_valid["objective"]
    best_state = _cpu_state(model.state_dict())

    for step in range(1, stop_step + 1):
        model.train()
        logits, mu, logvar = model(train_flat, latent_rng)
        loss = mask_ops._vae_loss(logits, train_flat, mu, logvar)
        # Decomposition is detached and does not affect gradients, optimizer, or RNG.
        with torch.no_grad():
            bce, kl_by_dimension = _component_terms(
                logits.detach(), train_flat, mu.detach(), logvar.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        stochastic_chunk.append(loss.detach())
        bce_chunk.append(bce.detach())
        kl_chunk.append(kl_by_dimension.sum().detach())
        kl_dimension_chunk.append(kl_by_dimension.detach())

        if step % eval_every == 0 or step == stop_step:
            model.eval()
            with torch.no_grad():
                deterministic_train = _deterministic_components(model, train_flat)
                validation = _deterministic_components(model, valid_flat)
            deterministic_train["bce_minus_target_bernoulli_entropy"] = (
                deterministic_train["bce"] - entropy_train["mean_per_map_sum"])
            validation["bce_minus_target_bernoulli_entropy"] = (
                validation["bce"] - entropy_valid["mean_per_map_sum"])
            chunk_losses = torch.stack(stochastic_chunk).detach().cpu().tolist()
            chunk_bces = torch.stack(bce_chunk).detach().cpu().tolist()
            chunk_kls = torch.stack(kl_chunk).detach().cpu().tolist()
            chunk_kl_dimensions = torch.stack(kl_dimension_chunk).detach().cpu().numpy()
            update_losses.extend(float(value) for value in chunk_losses)
            evaluation_rows.append({
                "step": step,
                "stochastic_elbo_window_mean": float(np.mean(chunk_losses)),
                "stochastic_bce_window_mean": float(np.mean(chunk_bces)),
                "stochastic_kl_total_window_mean": float(np.mean(chunk_kls)),
                "stochastic_kl_per_dimension_window_mean":
                    chunk_kl_dimensions.mean(axis=0).astype(np.float64),
                "deterministic_train": deterministic_train,
                "validation": validation,
            })
            stochastic_chunk.clear()
            bce_chunk.clear()
            kl_chunk.clear()
            kl_dimension_chunk.clear()
            if validation["objective"] < best_validation:
                best_validation = validation["objective"]
                best_step = step
                best_state = _cpu_state(model.state_dict())

    expected = [float(value) for value in expected_stochastic_trace]
    deltas = np.abs(np.asarray(update_losses, dtype=np.float64) -
                    np.asarray(expected, dtype=np.float64))
    trace_exact = bool(np.array_equal(np.asarray(update_losses), np.asarray(expected)))
    trace_max_delta = float(deltas.max(initial=0.))
    best_state_comparison = _state_comparison(best_state, expected_best_state)
    stop_state = _cpu_state(model.state_dict())

    # Statistics are measured at both source-validation-selected best and final stop states.
    latent_stats: dict[str, dict[str, dict[str, Any]]] = {}
    for checkpoint_name, state in (("best", best_state), ("stop", stop_state)):
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            latent_stats[checkpoint_name] = {
                "training": _latent_statistics(model, train),
                "validation": _latent_statistics(model, valid),
            }

    return {
        "fit_seed": int(fit_seed),
        "stop_step": int(stop_step),
        "best_step": int(best_step),
        "best_validation_objective": float(best_validation),
        "stochastic_trace_exact_match": trace_exact,
        "stochastic_trace_max_abs_delta": trace_max_delta,
        "best_state_exact_match": best_state_comparison["exact_match"],
        "best_state_max_abs_delta": best_state_comparison.get("max_abs_delta"),
        "best_state_comparison": best_state_comparison,
        "latent_stats": latent_stats,
        "source_target_bernoulli_entropy": {
            "training": entropy_train, "validation": entropy_valid,
        },
        "stochastic_elbo_by_update": np.asarray(update_losses, dtype=np.float32),
        "evaluations": evaluation_rows,
        "stop_state": stop_state,
    }


def _fit_data(arrays: Mapping[str, np.ndarray], family: str,
              task: int) -> tuple[np.ndarray, np.ndarray]:
    if family == "functional_vae_small":
        return arrays["function_train_aligned"][task, :26], arrays["function_validation_aligned"][task]
    if family == "functional_vae_large":
        return arrays["function_train_aligned"][task], arrays["function_validation_aligned"][task]
    if family == "raw_vae_large":
        return arrays["raw_train_aligned"][task], arrays["raw_validation_aligned"][task]
    raise ValueError(f"unknown family {family}")


def _reconstruction_metrics(source_diagnostics: Mapping[str, Any], family: str,
                            task: int) -> Any:
    by_family = source_diagnostics.get("reconstruction_metrics", {})
    values = by_family.get(family, []) if isinstance(by_family, Mapping) else []
    if task < len(values):
        return values[task]
    return None


def _latent_arrays(fit_rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    checkpoints = ("best", "stop")
    splits = ("training", "validation")
    vector_fields = {
        "latent_mean_variance_by_dimension": "posterior_mean_variance_by_dimension",
        "latent_covariance_eigenvalues": "posterior_mean_covariance_eigenvalues",
        "latent_posterior_noise_variance_by_dimension": "posterior_noise_variance_by_dimension",
    }
    scalar_fields = {
        "latent_active_units": "active_units",
        "latent_effective_rank": "posterior_mean_covariance_effective_rank",
        "latent_participation_ratio": "posterior_mean_covariance_participation_ratio",
        "latent_signal_variance_sum": "posterior_mean_signal_variance_sum",
        "latent_posterior_noise_variance_sum": "posterior_noise_variance_sum",
        "latent_noise_to_signal_ratio": "posterior_noise_to_signal_ratio",
        "latent_mean_vector_norm_mean": "posterior_mean_vector_norm_mean",
        "latent_mean_vector_norm_p50": "posterior_mean_vector_norm_p50",
        "latent_mean_vector_norm_p95": "posterior_mean_vector_norm_p95",
        "latent_mean_vector_norm_max": "posterior_mean_vector_norm_max",
        "latent_fraction_mean_vector_norm_above_agreement_radius":
            "fraction_posterior_mean_norm_above_agreement_radius",
    }
    out: dict[str, np.ndarray] = {}
    for key, metric in vector_fields.items():
        out[key] = np.asarray([
            [[row["latent_stats"][checkpoint][split][metric] for split in splits]
             for checkpoint in checkpoints]
            for row in fit_rows
        ], dtype=np.float32)
    for key, metric in scalar_fields.items():
        out[key] = np.asarray([
            [[row["latent_stats"][checkpoint][split][metric] for split in splits]
             for checkpoint in checkpoints]
            for row in fit_rows
        ], dtype=np.float64 if key not in ("latent_active_units",) else np.int16)
    return out


def replay_seed(seed: int, out: Path, *, device: str | torch.device) -> dict[str, Any]:
    """Replay all twelve frozen fits for one source seed and commit artifacts."""
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        if (out / "COMPLETE").is_file():
            saved = json.loads((out / "latent_diagnostics.json").read_text(encoding="utf-8"))
            if saved.get("experiment_seed") == seed and saved.get("complete") is True:
                return saved
        raise FileExistsError(f"latent output exists without a reusable COMPLETE marker: {out}")
    out.mkdir(parents=True, exist_ok=True)
    configure(seed)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but CUDA is unavailable")

    seed_root = CONVERGED_ROOT / f"seed_{seed}"
    source_folder = seed_root / "functional"
    arrays_path = source_folder / "functional_vae_arrays.npz"
    diagnostics_path = source_folder / "functional_vae_diagnostics.json"
    if not (seed_root / "COMPLETE").is_file() or not (CONVERGED_ROOT / "COMPLETE").is_file():
        raise FileNotFoundError("original converged seed/root must be COMPLETE before latent replay")
    if not arrays_path.is_file() or not diagnostics_path.is_file():
        raise FileNotFoundError(f"missing frozen source cache for seed {seed}: {source_folder}")
    source_diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if source_diagnostics.get("converged") is not True:
        raise ValueError(f"source fit diagnostics are not converged for seed {seed}")
    if int(source_diagnostics.get("experiment_seed", -1)) != seed:
        raise ValueError("source diagnostics experiment seed does not match requested seed")

    with np.load(arrays_path, allow_pickle=False) as archive:
        missing = [key for key in ("raw_train_aligned", "function_train_aligned",
                                   "raw_validation_aligned", "function_validation_aligned")
                   if key not in archive.files]
        if missing:
            raise ValueError(f"source cache lacks required source-only arrays: {missing}")
        source_arrays = {key: archive[key] for key in archive.files}
    fit_rows: list[dict[str, Any]] = []
    update_offsets = [0]
    evaluation_offsets = [0]
    update_steps: list[np.ndarray] = []
    update_losses: list[np.ndarray] = []
    evaluation_steps: list[np.ndarray] = []
    curve_lists: dict[str, list[np.ndarray]] = {
        "stochastic_elbo_window_mean": [],
        "stochastic_bce_window_mean": [],
        "stochastic_kl_total_window_mean": [],
        "stochastic_kl_per_dimension_window_mean": [],
        "deterministic_train_bce": [],
        "deterministic_train_kl_total": [],
        "deterministic_train_kl_per_dimension": [],
        "deterministic_train_objective": [],
        "validation_bce": [],
        "validation_kl_total": [],
        "validation_kl_per_dimension": [],
        "validation_objective": [],
        "deterministic_train_bce_minus_target_bernoulli_entropy": [],
        "validation_bce_minus_target_bernoulli_entropy": [],
    }

    for family in FAMILIES:
        for task in TASKS:
            fit_dir = source_folder / "fits" / family / f"task_{task}"
            report_path = fit_dir / "fit_report.json"
            curve_path = fit_dir / "loss_curves.json"
            checkpoint_path = fit_dir / "best_model.pt"
            if not all(path.is_file() for path in (report_path, curve_path, checkpoint_path)):
                raise FileNotFoundError(f"incomplete frozen fit artifacts: {fit_dir}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            curves = json.loads(curve_path.read_text(encoding="utf-8"))
            if report.get("converged") is not True or report.get("convergence_status") != "converged":
                raise ValueError(f"frozen fit did not converge: {fit_dir}")
            stop_step = int(report["stop_step"])
            best_step = int(report["best_step"])
            fit_seed = int(report["seed"])
            experiment_fit_seed = seed + converged.DEFAULT_FIT_SEED_OFFSET + task * 1009
            diagnostic_fit = source_diagnostics.get("vae", {}).get(family, [])[task]
            if (fit_seed != experiment_fit_seed
                    or report.get("family", family) != family
                    or int(report.get("task", task)) != task
                    or int(diagnostic_fit.get("seed", -1)) != fit_seed
                    or int(diagnostic_fit.get("stop_step", -1)) != int(report.get("stop_step", -2))
                    or int(diagnostic_fit.get("best_step", -1)) != int(report.get("best_step", -2))):
                raise ValueError(f"fit identity mismatch: {fit_dir}")
            expected_trace = curves.get("stochastic_train_loss_by_update", [])
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "state_dict" not in checkpoint:
                raise ValueError(f"frozen best checkpoint lacks state_dict: {checkpoint_path}")
            train_np, valid_np = _fit_data(source_arrays, family, task)
            replay = replay_fit(
                torch.from_numpy(np.ascontiguousarray(train_np)),
                torch.from_numpy(np.ascontiguousarray(valid_np)),
                fit_seed=fit_seed, stop_step=stop_step,
                expected_stochastic_trace=expected_trace,
                expected_best_state=checkpoint["state_dict"], device=device,
            )
            if replay["best_step"] != best_step:
                raise RuntimeError(f"best checkpoint step mismatch for {fit_dir}: "
                                   f"replay {replay['best_step']} vs frozen {best_step}")
            if not replay["stochastic_trace_exact_match"] or not replay["best_state_exact_match"]:
                raise RuntimeError(f"replay parity failed for {fit_dir}: "
                                   f"trace={replay['stochastic_trace_max_abs_delta']}, "
                                   f"state={replay['best_state_max_abs_delta']}")

            stored_evaluations = curves.get("evaluations", [])
            replay_evaluations = replay["evaluations"]
            if len(stored_evaluations) != len(replay_evaluations):
                raise RuntimeError(f"deterministic evaluation count mismatch for {fit_dir}")
            expected_steps = np.asarray([int(row.get("step", -1)) for row in stored_evaluations], dtype=np.int64)
            actual_steps = np.asarray([int(row["step"]) for row in replay_evaluations], dtype=np.int64)
            if not np.array_equal(expected_steps, actual_steps):
                raise RuntimeError(f"deterministic evaluation steps differ for {fit_dir}")
            expected_train = np.asarray(
                [float(row["deterministic_train_objective"]) for row in stored_evaluations], dtype=np.float64)
            expected_valid = np.asarray(
                [float(row["validation_objective"]) for row in stored_evaluations], dtype=np.float64)
            replay_train = np.asarray(
                [float(row["deterministic_train"]["objective"]) for row in replay_evaluations], dtype=np.float64)
            replay_valid = np.asarray(
                [float(row["validation"]["objective"]) for row in replay_evaluations], dtype=np.float64)
            train_delta = np.abs(replay_train - expected_train)
            valid_delta = np.abs(replay_valid - expected_valid)
            deterministic_train_exact = bool(np.array_equal(replay_train, expected_train))
            deterministic_valid_exact = bool(np.array_equal(replay_valid, expected_valid))
            if not deterministic_train_exact or not deterministic_valid_exact:
                raise RuntimeError(f"deterministic objective parity failed for {fit_dir}: "
                                   f"train={train_delta.max(initial=0.)}, "
                                   f"validation={valid_delta.max(initial=0.)}")

            evaluations = replay["evaluations"]
            steps = np.asarray([int(row["step"]) for row in evaluations], dtype=np.int32)
            evaluation_steps.append(steps)
            update_steps.append(np.arange(1, stop_step + 1, dtype=np.int32))
            update_losses.append(replay["stochastic_elbo_by_update"])
            update_offsets.append(update_offsets[-1] + stop_step)
            evaluation_offsets.append(evaluation_offsets[-1] + len(steps))

            def scalar_curve(field: str, split: str | None = None) -> np.ndarray:
                values = []
                for row in evaluations:
                    if split is None:
                        value = row[field]
                    else:
                        value = row[split][field]
                    values.append(np.nan if value is None else float(value))
                return np.asarray(values, dtype=np.float32)

            def vector_curve(field: str, split: str) -> np.ndarray:
                values = []
                for row in evaluations:
                    value = row[split][field]
                    values.append(np.zeros(LATENT_DIM, dtype=np.float32)
                                  if value is None else np.asarray(value, dtype=np.float32))
                return np.stack(values)

            for key in ("stochastic_elbo_window_mean", "stochastic_bce_window_mean",
                        "stochastic_kl_total_window_mean"):
                curve_lists[key].append(scalar_curve(key))
            curve_lists["stochastic_kl_per_dimension_window_mean"].append(
                np.stack([np.zeros(LATENT_DIM, dtype=np.float32)
                          if row["stochastic_kl_per_dimension_window_mean"] is None
                          else np.asarray(row["stochastic_kl_per_dimension_window_mean"], dtype=np.float32)
                          for row in evaluations]))
            for split, prefix in (("deterministic_train", "deterministic_train"),
                                  ("validation", "validation")):
                curve_lists[f"{prefix}_bce"].append(scalar_curve("bce", split))
                curve_lists[f"{prefix}_kl_total"].append(scalar_curve("kl_total", split))
                curve_lists[f"{prefix}_kl_per_dimension"].append(vector_curve("kl_per_dimension", split))
                curve_lists[f"{prefix}_objective"].append(scalar_curve("objective", split))
                excess_key = ("deterministic_train_bce_minus_target_bernoulli_entropy"
                              if split == "deterministic_train"
                              else "validation_bce_minus_target_bernoulli_entropy")
                curve_lists[excess_key].append(scalar_curve(
                    "bce_minus_target_bernoulli_entropy", split))

            fit_id = f"{family}/task_{task}"
            rows_evaluations = []
            for row in evaluations:
                stoch_fields = {
                    key: (None if row[key] is None else float(row[key]))
                    for key in ("stochastic_elbo_window_mean", "stochastic_bce_window_mean",
                                "stochastic_kl_total_window_mean")
                }
                stoch_fields["stochastic_kl_per_dimension_window_mean"] = (
                    None if row["stochastic_kl_per_dimension_window_mean"] is None
                    else np.asarray(row["stochastic_kl_per_dimension_window_mean"]).tolist())
                rows_evaluations.append({
                    "step": int(row["step"]), **stoch_fields,
                    "deterministic_train": {
                        key: row["deterministic_train"][key]
                        for key in ("bce", "kl_total", "objective", "kl_per_dimension",
                                    "bce_minus_target_bernoulli_entropy")
                    },
                    "validation": {
                        key: row["validation"][key]
                        for key in ("bce", "kl_total", "objective", "kl_per_dimension",
                                    "bce_minus_target_bernoulli_entropy")
                    },
                })

            comparison = {
                "stochastic_trace_exact_match": replay["stochastic_trace_exact_match"],
                "stochastic_trace_max_abs_delta": replay["stochastic_trace_max_abs_delta"],
                "deterministic_train_curve_exact_match": deterministic_train_exact,
                "deterministic_train_curve_max_abs_delta": float(train_delta.max(initial=0.)),
                "validation_curve_exact_match": deterministic_valid_exact,
                "validation_curve_max_abs_delta": float(valid_delta.max(initial=0.)),
                "best_state_exact_match": replay["best_state_exact_match"],
                "best_state_max_abs_delta": replay["best_state_max_abs_delta"],
                "best_state_tensor_comparison": replay["best_state_comparison"],
            }
            fit_rows.append({
                "fit_id": fit_id, "family": family, "task": task,
                "fit_seed": fit_seed, "train_maps": len(train_np),
                "validation_maps": len(valid_np), "stop_step": stop_step,
                "best_step": best_step,
                "objective": report.get("objective"),
                "deterministic_objective": report.get("deterministic_objective"),
                "checkpoint_parity": comparison,
                "reconstruction_metrics": _reconstruction_metrics(source_diagnostics, family, task),
                "source_target_bernoulli_entropy": replay["source_target_bernoulli_entropy"],
                "latent_stats": replay["latent_stats"],
                "evaluation_curve": rows_evaluations,
                "source_fit_report_path": str(report_path),
                "source_loss_curve_path": str(curve_path),
                "source_best_checkpoint_path": str(checkpoint_path),
            })
            del replay, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()

    identities = np.asarray([row["fit_id"] for row in fit_rows], dtype="U64")
    npz_payload: dict[str, np.ndarray] = {
        "fit_id": identities,
        "family": np.asarray([row["family"] for row in fit_rows], dtype="U32"),
        "task": np.asarray([row["task"] for row in fit_rows], dtype=np.int16),
        "fit_seed": np.asarray([row["fit_seed"] for row in fit_rows], dtype=np.int64),
        "stop_step": np.asarray([row["stop_step"] for row in fit_rows], dtype=np.int32),
        "best_step": np.asarray([row["best_step"] for row in fit_rows], dtype=np.int32),
        "update_offsets": np.asarray(update_offsets, dtype=np.int64),
        "update_step": np.concatenate(update_steps),
        "stochastic_elbo_by_update": np.concatenate(update_losses),
        "evaluation_offsets": np.asarray(evaluation_offsets, dtype=np.int64),
        "evaluation_step": np.concatenate(evaluation_steps),
    }
    for key, values in curve_lists.items():
        npz_payload[key] = np.concatenate(values, axis=0)
    npz_payload.update(_latent_arrays(fit_rows))
    json_payload = {
        "experiment": "converged_functional_vae_latent_diagnostics",
        "seed": seed + converged.DEFAULT_FIT_SEED_OFFSET,
        "experiment_seed": seed,
        "complete": True,
        "source_only": True,
        "target_data_or_labels_loaded": False,
        "device": str(device),
        "torch_version": torch.__version__,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "threads": torch.get_num_threads(),
        "source_artifacts": {
            "arrays_path": str(arrays_path), "arrays_sha256": _sha(arrays_path),
            "diagnostics_path": str(diagnostics_path), "diagnostics_sha256": _sha(diagnostics_path),
        },
        "replay_protocol": {
            "families": list(FAMILIES), "tasks": list(TASKS),
            "latent_dim": LATENT_DIM, "width": WIDTH,
            "learning_rate": converged.LEARNING_RATE, "kl_weight": KL_WEIGHT,
            "training": "full-batch stochastic ELBO replay with same local model seed and latent RNG seed+991",
            "evaluation_interval_updates": EVAL_EVERY,
            "deterministic_evaluation": "posterior-mean BCE plus analytic KL on source train/validation maps",
            "active_unit_variance_threshold": ACTIVE_UNIT_VARIANCE_THRESHOLD,
            "checkpoint_selection": "replayed strict-minimum source validation posterior-mean ELBO",
        },
        "fit_count": len(fit_rows),
        "all_stochastic_traces_exact": all(row["checkpoint_parity"]["stochastic_trace_exact_match"]
                                            for row in fit_rows),
        "all_deterministic_train_curves_exact": all(
            row["checkpoint_parity"]["deterministic_train_curve_exact_match"] for row in fit_rows),
        "all_validation_curves_exact": all(
            row["checkpoint_parity"]["validation_curve_exact_match"] for row in fit_rows),
        "all_best_states_exact": all(row["checkpoint_parity"]["best_state_exact_match"]
                                      for row in fit_rows),
        "fits": fit_rows,
        "arrays_path": str(out / "latent_diagnostics.npz"),
    }
    if (len(fit_rows) != 12 or not json_payload["all_stochastic_traces_exact"]
            or not json_payload["all_deterministic_train_curves_exact"]
            or not json_payload["all_validation_curves_exact"]
            or not json_payload["all_best_states_exact"]):
        raise RuntimeError("latent replay did not produce twelve exact-parity fits")
    _atomic_npz(out / "latent_diagnostics.npz", **npz_payload)
    _atomic_json(out / "latent_diagnostics.json", json_payload)
    _atomic_text(out / "COMPLETE", "complete\n")
    return json_payload


def _reference_toy_fit(train: torch.Tensor, valid: torch.Tensor, *, seed: int,
                       steps: int) -> tuple[list[float], dict[str, torch.Tensor], list[float], list[float]]:
    """Small independent fixture matching the original fit's update sequence."""
    device = train.device
    with mask_ops._local_torch_seed(seed, device):
        model = mask_ops._MaskVAE(train.size(1), LATENT_DIM, WIDTH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=converged.LEARNING_RATE)
    rng = torch.Generator(device=device).manual_seed(seed + 991)
    with torch.no_grad():
        mu, logvar = model.encode(valid)
        best_validation = float(mask_ops._vae_loss(model.decode(mu), valid, mu, logvar))
    best_state = _cpu_state(model.state_dict())
    losses = []
    train_objectives = []
    validation_objectives = []
    model.eval()
    with torch.no_grad():
        mu_train, lv_train = model.encode(train)
        train_objectives.append(float(mask_ops._vae_loss(model.decode(mu_train), train,
                                                         mu_train, lv_train)))
        validation_objectives.append(best_validation)
    for step in range(1, steps + 1):
        model.train()
        logits, mu, logvar = model(train, rng)
        loss = mask_ops._vae_loss(logits, train, mu, logvar)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        losses.append(float(loss.detach().cpu()))
        optimizer.step()
        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                mu_train, lv_train = model.encode(train)
                train_objective = float(mask_ops._vae_loss(model.decode(mu_train), train,
                                                            mu_train, lv_train))
                mu, logvar = model.encode(valid)
                validation = float(mask_ops._vae_loss(model.decode(mu), valid, mu, logvar))
            train_objectives.append(train_objective)
            validation_objectives.append(validation)
            if validation < best_validation:
                best_validation = validation
                best_state = _cpu_state(model.state_dict())
    return losses, best_state, train_objectives, validation_objectives


def cpu_smoke() -> None:
    """Bounded deterministic replay/parity smoke test on tiny CPU maps."""
    configure(17)
    generator = torch.Generator().manual_seed(909)
    train = torch.rand((5, 32), generator=generator)
    valid = torch.rand((3, 32), generator=generator)
    steps = 20
    trace, best_state, reference_train, reference_valid = _reference_toy_fit(
        train, valid, seed=2317, steps=steps)
    result = replay_fit(
        train, valid, fit_seed=2317, stop_step=steps,
        expected_stochastic_trace=trace, expected_best_state=best_state, device="cpu",
    )
    assert result["stochastic_trace_exact_match"]
    assert result["best_state_exact_match"]
    assert result["stochastic_trace_max_abs_delta"] == 0.
    assert result["best_state_max_abs_delta"] == 0.
    assert len(result["evaluations"]) == steps // EVAL_EVERY + 1
    replay_train = [row["deterministic_train"]["objective"] for row in result["evaluations"]]
    replay_valid = [row["validation"]["objective"] for row in result["evaluations"]]
    assert np.array_equal(np.asarray(replay_train), np.asarray(reference_train))
    assert np.array_equal(np.asarray(replay_valid), np.asarray(reference_valid))
    assert set(result["latent_stats"]) == {"best", "stop"}
    for checkpoint in result["latent_stats"].values():
        for split in checkpoint.values():
            assert split["posterior_mean_variance_by_dimension"]
            assert np.isfinite(split["posterior_mean_covariance_effective_rank"])
            assert np.isfinite(split["posterior_noise_to_signal_ratio"])
            assert len(split["posterior_mean_covariance_eigenvalues"]) == LATENT_DIM
    serializable = {
        "evaluations": result["evaluations"],
        "latent_stats": result["latent_stats"],
        "entropy": result["source_target_bernoulli_entropy"],
    }
    encoded = json.dumps(_jsonable(serializable), allow_nan=False)
    assert "kl_per_dimension" in encoded
    print(json.dumps({"cpu_smoke": "passed", "updates": steps,
                      "trace_exact": result["stochastic_trace_exact_match"],
                      "state_exact": result["best_state_exact_match"]}), flush=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_allocation(path: Path, payload: dict[str, Any]) -> None:
    _atomic_json(path, payload)


def _launch_all(out: Path) -> None:
    out = Path(out).resolve()
    if not (CONVERGED_ROOT / "COMPLETE").is_file():
        raise RuntimeError(f"original converged production root is not COMPLETE: {CONVERGED_ROOT}")
    for seed in SEEDS:
        seed_root = CONVERGED_ROOT / f"seed_{seed}"
        if not (seed_root / "COMPLETE").is_file():
            raise RuntimeError(f"original converged seed {seed} is not COMPLETE")

    root_allocation_path = CONVERGED_ROOT / "allocation.json"
    selection_path = CONVERGED_ROOT / "gpu_selection.json"
    if not root_allocation_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError("original GPU UUID allocation/selection files are required")
    root_allocation = _read_json(root_allocation_path)
    selection = _read_json(selection_path)
    if root_allocation.get("complete") is not True or root_allocation.get("failed_seeds"):
        raise RuntimeError("original GPU allocation does not record a clean COMPLETE run")
    workers = root_allocation.get("workers", [])
    uuid_by_seed = {int(row["seed"]): str(row["uuid"]) for row in workers
                    if int(row.get("exit_code", -1)) == 0}
    selected = [str(value) for value in selection.get("selected", [])]
    if set(uuid_by_seed) != set(SEEDS) or len(selected) != len(SEEDS) or \
            set(uuid_by_seed.values()) != set(selected) or len(set(selected)) != len(selected):
        raise RuntimeError("original per-seed UUID allocation is incomplete or inconsistent")

    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    )
    utilization = {}
    for line in query.stdout.splitlines():
        uuid, percent = [part.strip() for part in line.split(",", 1)]
        utilization[uuid] = int(percent)
    busy = {uuid: utilization.get(uuid, None) for uuid in selected
            if utilization.get(uuid, None) != 0}
    if busy:
        raise RuntimeError(f"originally assigned GPUs are not idle: {busy}")

    allocation_path = out / "latent_allocation.json"
    if allocation_path.exists():
        old = _read_json(allocation_path)
        if old.get("complete") is True and not old.get("failed_seeds"):
            print(f"latent replay already complete: {allocation_path}", flush=True)
            return
        attempts = out / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        attempt_index = 1
        while (attempts / f"attempt_{attempt_index:02d}").exists():
            attempt_index += 1
        archive = attempts / f"attempt_{attempt_index:02d}"
        archive.mkdir()
        shutil.copy2(allocation_path, archive / "latent_allocation.json")
        for source_name, archive_name in (
            (CONVERGED_ROOT / "protocol.json", "converged_protocol.json"),
            (CONVERGED_ROOT / "gpu_selection.json", "original_gpu_selection.json"),
            (CONVERGED_ROOT / "allocation.json", "original_allocation.json"),
        ):
            if source_name.is_file():
                shutil.copy2(source_name, archive / archive_name)
        snapshot = archive / "source_snapshot"
        snapshot.mkdir()
        current_module = Path(__file__).resolve()
        shutil.copy2(current_module, snapshot / "expanded_latent_diagnostics_retry_source.py")
        pycache_file = current_module.parent / "__pycache__" / (current_module.stem + ".cpython-310.pyc")
        if pycache_file.is_file():
            shutil.copy2(pycache_file, snapshot / "expanded_latent_diagnostics_compiled.pyc")
        _atomic_json(archive / "attempt_metadata.json", {
            "archived_at_unix_seconds": time.time(),
            "failed_source_sha256_before_retry": old.get("source_sha256"),
            "current_source_sha256": _sha(current_module),
            "original_converged_root": str(CONVERGED_ROOT),
            "archive_note": "First attempt failed during fit identity preflight before replay training; allocation and logs are preserved.",
        })
        old_logs = out / "logs"
        if old_logs.exists():
            shutil.move(str(old_logs), str(archive / "logs"))
        for seed in SEEDS:
            partial_seed = out / f"seed_{seed}"
            if partial_seed.exists():
                shutil.move(str(partial_seed), str(archive / f"seed_{seed}"))
        allocation_path.unlink()
    for seed in SEEDS:
        seed_out = out / f"seed_{seed}"
        if seed_out.exists() and any(seed_out.iterdir()):
            raise FileExistsError(f"latent seed output already exists: {seed_out}")

    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    initial = {
        "experiment": "converged_functional_vae_latent_diagnostics",
        "complete": False, "launched": False, "seeds": list(SEEDS),
        "workers_per_gpu": 1,
        "original_allocation_path": str(root_allocation_path),
        "original_allocation_sha256": _sha(root_allocation_path),
        "original_gpu_selection_path": str(selection_path),
        "original_gpu_selection_sha256": _sha(selection_path),
        "source_sha256": _sha(Path(__file__).resolve()),
        "workers": [{"seed": seed, "uuid": uuid_by_seed[seed], "pid": None,
                     "exit_code": None, "status": "pending",
                     "output": str(out / f"seed_{seed}"),
                     "log": str(logs / f"seed_{seed}.log")}
                    for seed in SEEDS],
        "failed_seeds": [],
    }
    _write_allocation(allocation_path, initial)
    processes: dict[int, subprocess.Popen] = {}
    streams = {}
    env_base = os.environ.copy()
    env_base["PYTHONUNBUFFERED"] = "1"
    for row in initial["workers"]:
        seed = int(row["seed"])
        log_stream = Path(row["log"]).open("w", encoding="utf-8")
        streams[seed] = log_stream
        env = dict(env_base)
        env["CUDA_VISIBLE_DEVICES"] = row["uuid"]
        command = [sys.executable, "-m", "deepsets_vaae.expanded_latent_diagnostics",
                   "--seed", str(seed), "--out", row["output"], "--device", "cuda:0"]
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log_stream,
                                stderr=subprocess.STDOUT, text=True)
        processes[seed] = proc
        row["pid"] = proc.pid
        row["status"] = "running"
    initial["launched"] = True
    initial["launcher_pid"] = os.getpid()
    initial["started_unix_seconds"] = time.time()
    _write_allocation(allocation_path, initial)
    print(json.dumps({"latent_replay": "launched", "allocation": str(allocation_path),
                      "workers": [{"seed": row["seed"], "uuid": row["uuid"], "pid": row["pid"]}
                                  for row in initial["workers"]]}), flush=True)

    while any(proc.poll() is None for proc in processes.values()):
        for row in initial["workers"]:
            proc = processes[int(row["seed"])]
            code = proc.poll()
            if code is not None and row["exit_code"] is None:
                row["exit_code"] = int(code)
                row["status"] = "complete" if code == 0 else "failed"
        _write_allocation(allocation_path, initial)
        time.sleep(5.)
    for seed, proc in processes.items():
        code = int(proc.wait())
        row = next(item for item in initial["workers"] if int(item["seed"]) == seed)
        row["exit_code"] = code
        row["status"] = "complete" if code == 0 else "failed"
        if code == 0 and not (out / f"seed_{seed}" / "COMPLETE").is_file():
            row["status"] = "failed_no_marker"
            row["exit_code"] = 97
    for stream in streams.values():
        stream.close()
    initial["failed_seeds"] = [row["seed"] for row in initial["workers"]
                                if row["exit_code"] != 0]
    initial["complete"] = not initial["failed_seeds"]
    initial["finished_unix_seconds"] = time.time()
    _write_allocation(allocation_path, initial)
    print(json.dumps({"latent_replay": "complete" if initial["complete"] else "failed",
                      "allocation": str(allocation_path),
                      "failed_seeds": initial["failed_seeds"]}), flush=True)
    if not initial["complete"]:
        raise RuntimeError(f"latent diagnostic replay failed for seeds {initial['failed_seeds']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    if args.cpu_smoke:
        cpu_smoke()
        return
    if args.launch:
        _launch_all(args.out)
        return
    if args.seed is None:
        parser.error("--seed is required unless --launch or --cpu-smoke is used")
    target = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    replay_seed(args.seed, args.out, device=target)
    print(json.dumps({"seed": args.seed, "status": "complete", "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
