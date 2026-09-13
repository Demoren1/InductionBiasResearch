"""Learn amortized latent mappings between two frozen pattern-VAE decoders.

The adapter is trained without gold masks or task labels.  A source latent is
drawn from the prior and held fixed; only the adapter is optimized so that the
two decoded soft exact-K masks agree after a detached Hungarian column match.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .decoder_agreement import align_columns, hard_topk, soft_topk
from .run_decoder_agreement import load_models


class ConstantAdapter(torch.nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.value = torch.nn.Parameter(torch.zeros(latent_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.value.expand_as(z)


class LinearAdapter(torch.nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(latent_dim, latent_dim)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(latent_dim))
            self.linear.bias.zero_()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


class ResidualMLPAdapter(torch.nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or 2 * latent_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        # Linear and nonlinear mappings both start from identity.
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(z)


ADAPTERS = {
    "constant": ConstantAdapter,
    "linear": LinearAdapter,
    "mlp": ResidualMLPAdapter,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def project_ball(z: torch.Tensor, radius: float) -> torch.Tensor:
    if radius <= 0:
        raise ValueError("radius must be positive")
    norm = z.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(z.dtype).tiny)
    return z * (radius / norm).clamp(max=1.0)


def decode(model: torch.nn.Module, z: torch.Tensor, k: int,
           temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model.decode(z, z.new_zeros(len(z), int(getattr(model, "cond_dim", 0))))
    side = math.isqrt(int(model.mask_dim))
    if side * side != int(model.mask_dim):
        raise ValueError("mask_dim must be square for column alignment")
    soft = soft_topk(logits, k, temperature).reshape(-1, side, side)
    hard = hard_topk(logits, k).reshape(-1, side, side)
    return soft, hard


def latent_bank(n: int, latent_dim: int, seed: int, radius: float,
                device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return project_ball(torch.randn(n, latent_dim, generator=generator), radius).to(device)


def fixed_permutation(reference: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Infer one column permutation from an entire validation bank."""
    if reference.shape != other.shape or reference.ndim != 3:
        raise ValueError("expected equal (N,L,H) tensors")
    ref = reference.detach().permute(2, 0, 1).reshape(reference.size(2), -1)
    alt = other.detach().permute(2, 0, 1).reshape(other.size(2), -1)
    cost = (ref[:, None, :] - alt[None, :, :]).square().sum(-1).cpu().numpy()
    rows, cols = linear_sum_assignment(cost)
    order = torch.empty(reference.size(2), dtype=torch.long, device=other.device)
    order[torch.as_tensor(rows, device=other.device)] = torch.as_tensor(cols, device=other.device)
    return order


def mask_metrics(source_soft: torch.Tensor, source_hard: torch.Tensor,
                 target_soft: torch.Tensor, target_hard: torch.Tensor,
                 fixed_order: torch.Tensor | None = None) -> dict[str, float]:
    aligned_soft = align_columns(source_soft, target_soft)
    aligned_hard = align_columns(source_hard, target_hard)
    intersection = (source_hard.bool() & aligned_hard.bool()).sum((1, 2)).float()
    union = (source_hard.bool() | aligned_hard.bool()).sum((1, 2)).float()
    result = {
        "soft_mse": float((source_soft - aligned_soft).square().mean()),
        "hard_iou": float((intersection / union.clamp_min(1)).mean()),
        "hard_exact": float((source_hard == aligned_hard).all(2).all(1).float().mean()),
        "hard_hamming": float((source_hard - aligned_hard).abs().sum((1, 2)).mean()),
        "source_unique": int(torch.unique(source_hard.flatten(1), dim=0).size(0)),
        "target_unique": int(torch.unique(target_hard.flatten(1), dim=0).size(0)),
    }
    if fixed_order is not None:
        fixed_soft = target_soft.index_select(-1, fixed_order)
        fixed_hard = target_hard.index_select(-1, fixed_order)
        fixed_intersection = (source_hard.bool() & fixed_hard.bool()).sum((1, 2)).float()
        fixed_union = (source_hard.bool() | fixed_hard.bool()).sum((1, 2)).float()
        result.update({
            "fixed_soft_mse": float((source_soft - fixed_soft).square().mean()),
            "fixed_hard_iou": float((fixed_intersection / fixed_union.clamp_min(1)).mean()),
            "fixed_hard_exact": float(
                (source_hard == fixed_hard).all(2).all(1).float().mean()),
        })
    return result


@torch.no_grad()
def evaluate_mapping(source_model: torch.nn.Module, target_model: torch.nn.Module,
                     z_source: torch.Tensor, z_target: torch.Tensor, k: int,
                     temperature: float, fixed_order: torch.Tensor | None = None,
                     radius: float | None = None) -> tuple[dict, dict]:
    source_soft, source_hard = decode(source_model, z_source, k, temperature)
    target_soft, target_hard = decode(target_model, z_target, k, temperature)
    metrics = mask_metrics(source_soft, source_hard, target_soft, target_hard, fixed_order)
    norms = z_target.norm(dim=1)
    metrics.update({
        "target_norm_mean": float(norms.mean()),
        "target_norm_max": float(norms.max()),
    })
    if radius is not None:
        metrics["target_at_radius_fraction"] = float((norms >= radius - 1e-5).float().mean())
    tensors = {
        "z_source": z_source.cpu(), "z_target": z_target.cpu(),
        "source_hard": source_hard.cpu(), "target_hard": target_hard.cpu(),
    }
    return metrics, tensors


def train_adapter(adapter: torch.nn.Module, source_model: torch.nn.Module,
                  target_model: torch.nn.Module, z_train: torch.Tensor,
                  z_val: torch.Tensor, *, steps: int, batch_size: int,
                  eval_every: int, lr: float, radius: float, k: int,
                  temperature: float, seed: int) -> tuple[torch.nn.Module, dict]:
    adapter.train()
    device = z_train.device
    adapter.to(device)
    with torch.no_grad():
        train_soft, _ = decode(source_model, z_train, k, temperature)
        val_source_soft, _ = decode(source_model, z_val, k, temperature)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_state = copy.deepcopy(adapter.state_dict())
    best_val = float("inf")
    history = []
    started = time.monotonic()
    for step in range(1, steps + 1):
        indices = torch.randint(len(z_train), (batch_size,), generator=generator).to(device)
        source = train_soft.index_select(0, indices)
        z = z_train.index_select(0, indices)
        target_soft, _ = decode(target_model, project_ball(adapter(z), radius), k, temperature)
        per_example = (source - align_columns(source, target_soft)).square().mean((1, 2))
        optimizer.zero_grad(set_to_none=True)
        per_example.sum().backward()
        optimizer.step()
        if step % eval_every == 0 or step == steps:
            adapter.eval()
            with torch.no_grad():
                val_target_soft, _ = decode(
                    target_model, project_ball(adapter(z_val), radius), k, temperature)
                val = float((val_source_soft - align_columns(
                    val_source_soft, val_target_soft)).square().mean())
            if val < best_val:
                best_val = val
                best_state = copy.deepcopy(adapter.state_dict())
            history.append({"step": step, "train_soft_mse": float(per_example.mean()),
                            "val_soft_mse": val})
            print(f"[{adapter.__class__.__name__}] step={step}/{steps} "
                  f"train={per_example.mean().item():.6f} val={val:.6f}", flush=True)
            adapter.train()
    adapter.load_state_dict(best_state)
    adapter.eval()
    return adapter, {"best_val_soft_mse": best_val, "history": history,
                     "elapsed_seconds": time.monotonic() - started}


def optimize_target_latents(source_model: torch.nn.Module, target_model: torch.nn.Module,
                            z_source: torch.Tensor, z_initial: torch.Tensor, *,
                            steps: int, lr: float, radius: float, k: int,
                            temperature: float) -> tuple[torch.Tensor, dict]:
    """Fair one-sided Adam reference: source latents never move."""
    with torch.no_grad():
        source_soft, _ = decode(source_model, z_source, k, temperature)
    z_target = torch.nn.Parameter(z_initial.detach().clone())
    optimizer = torch.optim.Adam([z_target], lr=lr)
    with torch.no_grad():
        initial_target, _ = decode(target_model, z_target, k, temperature)
        initial_loss = (source_soft - align_columns(source_soft, initial_target)).square().mean((1, 2))
        best_loss = initial_loss.clone()
        best_z = z_target.detach().clone()
    for step in range(1, steps + 1):
        target_soft, _ = decode(target_model, z_target, k, temperature)
        loss = (source_soft - align_columns(source_soft, target_soft)).square().mean((1, 2))
        optimizer.zero_grad(set_to_none=True)
        loss.sum().backward()
        optimizer.step()
        with torch.no_grad():
            z_target.copy_(project_ball(z_target, radius))
            after_soft, _ = decode(target_model, z_target, k, temperature)
            after = (source_soft - align_columns(source_soft, after_soft)).square().mean((1, 2))
            improved = after < best_loss
            best_loss = torch.where(improved, after, best_loss)
            best_z = torch.where(improved[:, None], z_target.detach(), best_z)
        if step % 200 == 0 or step == steps:
            print(f"[z2-only Adam] step={step}/{steps} best={best_loss.mean().item():.6f}", flush=True)
    return best_z, {"initial_soft_mse": float(initial_loss.mean()),
                    "best_soft_mse": float(best_loss.mean())}


def _adapter_output(adapter: torch.nn.Module, z: torch.Tensor, radius: float) -> torch.Tensor:
    with torch.no_grad():
        return project_ball(adapter(z), radius)


def run_direction(source_model: torch.nn.Module, target_model: torch.nn.Module,
                  direction: str, settings: dict, device: torch.device,
                  seed: int) -> dict:
    latent_dim = int(source_model.latent_dim)
    if latent_dim != int(target_model.latent_dim):
        raise ValueError("this experiment requires equal latent dimensions")
    z_train = latent_bank(settings["n_train"], latent_dim, seed + 1, settings["radius"], device)
    z_val = latent_bank(settings["n_val"], latent_dim, seed + 2, settings["radius"], device)
    z_test = latent_bank(settings["n_test"], latent_dim, seed + 3, settings["radius"], device)
    z_independent = latent_bank(settings["n_test"], latent_dim, seed + 4,
                                settings["radius"], device)
    methods: dict[str, dict] = {}

    # Two controls require no fitting.
    with torch.no_grad():
        val_source_soft, _ = decode(source_model, z_val, settings["k"], settings["temperature"])
    for name, z_target in (("independent", z_independent), ("identity", z_test)):
        with torch.no_grad():
            val_target = z_val if name == "identity" else latent_bank(
                settings["n_val"], latent_dim, seed + 5, settings["radius"], device)
            val_target_soft, _ = decode(target_model, val_target, settings["k"], settings["temperature"])
            order = fixed_permutation(val_source_soft, val_target_soft)
        metrics, tensors = evaluate_mapping(source_model, target_model, z_test, z_target,
                                            settings["k"], settings["temperature"], order,
                                            settings["radius"])
        methods[name] = {"metrics": metrics, "tensors": tensors,
                         "fixed_order": order.cpu(), "training": None}

    fitted = {}
    for offset, (name, adapter_type) in enumerate(ADAPTERS.items()):
        torch.manual_seed(seed + 100 + offset)
        adapter = adapter_type(latent_dim)
        adapter, training = train_adapter(
            adapter, source_model, target_model, z_train, z_val,
            steps=settings["steps"], batch_size=settings["batch_size"],
            eval_every=settings["eval_every"], lr=settings["lr"],
            radius=settings["radius"], k=settings["k"],
            temperature=settings["temperature"], seed=seed + 200)
        with torch.no_grad():
            val_target_soft, _ = decode(target_model, _adapter_output(adapter, z_val, settings["radius"]),
                                        settings["k"], settings["temperature"])
            order = fixed_permutation(val_source_soft, val_target_soft)
            z_target = _adapter_output(adapter, z_test, settings["radius"])
        metrics, tensors = evaluate_mapping(source_model, target_model, z_test, z_target,
                                            settings["k"], settings["temperature"], order,
                                            settings["radius"])
        methods[name] = {"metrics": metrics, "tensors": tensors,
                         "fixed_order": order.cpu(), "training": training,
                         "state_dict": {key: value.cpu() for key, value in adapter.state_dict().items()}}
        fitted[name] = adapter

    # Expensive references use a fixed prefix of the held-out bank only.
    n_ref = min(settings["n_adam_test"], settings["n_test"])
    ref_source = z_test[:n_ref]
    ref_initial = z_independent[:n_ref]
    reference = {}
    starts = {"independent_init": ref_initial,
              "mlp_init": _adapter_output(fitted["mlp"], ref_source, settings["radius"])}
    for name, initial in starts.items():
        best_z, optimization = optimize_target_latents(
            source_model, target_model, ref_source, initial,
            steps=settings["adam_steps"], lr=settings["adam_lr"],
            radius=settings["radius"], k=settings["k"], temperature=settings["temperature"])
        metrics, tensors = evaluate_mapping(source_model, target_model, ref_source, best_z,
                                            settings["k"], settings["temperature"],
                                            radius=settings["radius"])
        reference[name] = {"metrics": metrics, "tensors": tensors,
                           "optimization": optimization}
    # Same subset for an honest amortization-gap comparison.
    subset = {}
    for name in ("independent", "identity", "constant", "linear", "mlp"):
        tensors = methods[name]["tensors"]
        # Recompute soft masks because the compact artifact stores only hard masks.
        with torch.no_grad():
            ss, sh = decode(source_model, ref_source, settings["k"], settings["temperature"])
            ts, th = decode(target_model, tensors["z_target"][:n_ref].to(device),
                            settings["k"], settings["temperature"])
        subset[name] = mask_metrics(ss, sh, ts, th)
    return {"direction": direction, "methods": methods, "adam_reference": reference,
            "adam_subset_methods": subset,
            "split_seeds": {"train": seed + 1, "validation": seed + 2,
                            "test": seed + 3, "independent": seed + 4}}


def run_pair(pair_root: Path, out: Path, device: torch.device, settings: dict) -> None:
    if (out / "result.pt").exists():
        print(f"[skip] {out}/result.pt already exists", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    models, model_provenance = load_models(pair_root, device)
    source_protocol = json.loads((pair_root / "protocol.json").read_text())
    pair = source_protocol["model_seeds"]
    result = {"model_seeds": pair, "settings": settings, "directions": {}}
    for index, (source, target) in enumerate(((models[0], models[1]), (models[1], models[0]))):
        direction = f"{pair[index]}_to_{pair[1 - index]}"
        print(f"[direction] {direction}", flush=True)
        result["directions"][direction] = run_direction(
            source, target, direction, settings, device,
            settings["seed"] + 10_000 * pair[0] + 1_000 * pair[1] + 100 * index)
    torch.save(result, out / "result.pt")
    protocol = {
        "experiment": "amortized one-sided latent adapter between two frozen VAE decoders",
        "source_pair_root": str(pair_root), "model_seeds": pair,
        "selection": "minimum validation soft agreement MSE; test, gold, and task labels excluded",
        "settings": settings,
        "methods": ["independent", "identity", "constant", "linear", "mlp",
                    "z2-only Adam from independent prior", "z2-only Adam from MLP adapter"],
    }
    dump_json(out / "protocol.json", protocol)
    dump_json(out / "provenance.json", {
        "protocol_sha256": sha256_file(out / "protocol.json"),
        "result_sha256": sha256_file(out / "result.pt"),
        "source_protocol_sha256": sha256_file(pair_root / "protocol.json"),
        "models": model_provenance,
        "source_sha256": sha256_file(Path(__file__)),
        "device": str(device), "torch_version": str(torch.__version__),
    })
    print(f"[complete] {out}", flush=True)
