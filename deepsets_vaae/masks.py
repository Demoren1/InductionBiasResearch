"""Label-free extraction of fixed-cardinality masks from DeepSets banks.

The input banks contain only successful, per-task importance maps.  This
module deliberately never receives the digit-score vectors, set examples, or
downstream validation labels.  It aligns hidden units on the source banks,
fits one unconditional VAE per source task, and searches the frozen decoders
for masks on which they agree.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from typing import Any, Iterable

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F


class _SoftTopK(torch.autograd.Function):
    """Differentiable sigmoid projection with exactly ``k`` active mass."""

    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
        if logits.ndim < 1 or not 0 < k < logits.size(-1) or temperature <= 0:
            raise ValueError("soft_topk requires 0 < k < last dimension and T > 0")
        lo = logits.detach().amin(-1, keepdim=True) - 40.0 * temperature
        hi = logits.detach().amax(-1, keepdim=True) + 40.0 * temperature
        desired = torch.full_like(lo, float(k))
        for _ in range(64):
            threshold = (lo + hi) * .5
            count = torch.sigmoid((logits.detach() - threshold) / temperature).sum(-1, keepdim=True)
            lo = torch.where(count > desired, threshold, lo)
            hi = torch.where(count > desired, hi, threshold)
        output = torch.sigmoid((logits - (lo + hi) * .5) / temperature)
        ctx.save_for_backward(output)
        ctx.temperature = temperature
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        (output,) = ctx.saved_tensors
        weight = output * (1.0 - output)
        mean = (grad_output * weight).sum(-1, keepdim=True)
        mean = mean / weight.sum(-1, keepdim=True).clamp_min(torch.finfo(output.dtype).tiny)
        return weight * (grad_output - mean) / ctx.temperature, None, None


def soft_topk(logits: torch.Tensor, k: int, temperature: float = .5) -> torch.Tensor:
    return _SoftTopK.apply(logits, int(k), float(temperature))


def _hard_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    result = torch.zeros_like(logits)
    return result.scatter(-1, logits.topk(k, dim=-1).indices, 1.0)


def _align_columns(reference: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Hungarian-align ``other`` hidden columns, retaining its autograd path."""
    if reference.shape != other.shape or reference.ndim not in (2, 3):
        raise ValueError("alignment expects equal [F,H] or [N,F,H] tensors")
    one = reference.ndim == 2
    if one:
        reference, other = reference.unsqueeze(0), other.unsqueeze(0)
    # Assignment is discrete and detached; gather itself is differentiable.
    left = reference.detach().transpose(1, 2)
    right = other.detach().transpose(1, 2)
    costs = (left[:, :, None] - right[:, None, :]).square().sum(-1).cpu().numpy()
    orders = []
    for cost in costs:
        rows, cols = linear_sum_assignment(cost)
        order = torch.empty(reference.size(-1), dtype=torch.long)
        order[torch.as_tensor(rows)] = torch.as_tensor(cols)
        orders.append(order)
    order = torch.stack(orders).to(other.device)
    result = other.gather(2, order[:, None, :].expand_as(other))
    return result[0] if one else result


def _unique_rows(maps: torch.Tensor) -> torch.Tensor:
    """Remove exact duplicate importance maps before the train/validation split."""
    flat = maps.reshape(maps.size(0), -1)
    return torch.unique(flat, dim=0, sorted=True).reshape(-1, *maps.shape[1:])


def _validated_maps(value: Any, device: torch.device) -> torch.Tensor:
    maps = torch.as_tensor(value, dtype=torch.float32, device=device)
    if maps.ndim != 3:
        raise ValueError("each bank['maps'] must have shape [keep,F,H]")
    if maps.size(0) == 0:
        raise ValueError("a bank has no successful maps")
    if not bool(torch.isfinite(maps).all()) or float(maps.min()) < 0. or float(maps.max()) > 1.:
        raise ValueError("importance maps must be finite BCE targets in [0, 1]")
    # Preserve the learned importance magnitude.  In this pilot the source
    # banks already have an exact-K binary support; turning them into binary
    # maps here would discard the within-support evidence that distinguishes
    # successful solutions.  Fixed-K is imposed only on generated outputs.
    return maps


def _split_unique(maps: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    maps = _unique_rows(maps)
    if maps.size(0) == 1:
        return maps, maps
    order = torch.randperm(maps.size(0), generator=generator, device=maps.device)
    n_val = max(1, int(round(.2 * maps.size(0))))
    n_val = min(n_val, maps.size(0) - 1)
    return maps[order[n_val:]], maps[order[:n_val]]


def _consensus_align(train_by_task: list[torch.Tensor], iterations: int = 4) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Iteratively align training maps to a common pixel-coordinate consensus."""
    all_maps = torch.cat(train_by_task, dim=0)
    consensus = all_maps[0]
    aligned: list[torch.Tensor] = []
    for _ in range(iterations):
        aligned = [_align_columns(consensus.expand_as(task), task) for task in train_by_task]
        consensus = torch.cat(aligned, dim=0).mean(0)
    return aligned, consensus


@contextmanager
def _local_torch_seed(seed: int, device: torch.device):
    """Construct networks reproducibly without mutating the caller's RNG state."""
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        yield


class _MaskVAE(nn.Module):
    def __init__(self, flat_dim: int, latent_dim: int, width: int) -> None:
        super().__init__()
        self.flat_dim, self.latent_dim = flat_dim, latent_dim
        self.encoder = nn.Sequential(nn.Linear(flat_dim, width), nn.ReLU())
        self.mu = nn.Linear(width, latent_dim)
        self.logvar = nn.Linear(width, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, width), nn.ReLU(), nn.Linear(width, flat_dim))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden).clamp(-12., 12.)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        noise = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
        return self.decode(mu + noise * (.5 * logvar).exp()), mu, logvar


def _vae_loss(logits: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    reconstruction = F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(-1)
    kl = -.5 * (1. + logvar - mu.square() - logvar.exp()).sum(-1)
    return (reconstruction + .1 * kl).mean()


def _fit_vae(train: torch.Tensor, valid: torch.Tensor, *, flat_dim: int, latent: int,
             width: int, epochs: int, seed: int, device: torch.device) -> tuple[_MaskVAE, dict[str, Any]]:
    with _local_torch_seed(seed, device):
        model = _MaskVAE(flat_dim, latent, width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = torch.Generator(device=device).manual_seed(seed + 991)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    initial_validation = float("nan")
    last_validation = float("nan")
    best_step = -1
    train_history: list[float] = []
    for epoch in range(epochs):
        if epoch == 0:
            model.eval()
            with torch.no_grad():
                initial_x = valid.reshape(valid.size(0), -1)
                initial_mu, initial_logvar = model.encode(initial_x)
                initial_validation = float(_vae_loss(model.decode(initial_mu), initial_x,
                                                    initial_mu, initial_logvar))
        model.train()
        logits, mu, logvar = model(train.reshape(train.size(0), -1), rng)
        loss = _vae_loss(logits, train.reshape(train.size(0), -1), mu, logvar)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            # Validation uses posterior means, so checkpoint selection is not
            # changed by sampling noise or any downstream task information.
            x = valid.reshape(valid.size(0), -1)
            mu_v, logvar_v = model.encode(x)
            validation = _vae_loss(model.decode(mu_v), x, mu_v, logvar_v)
            last_validation = float(validation)
        train_history.append(float(loss.detach().cpu()))
        if float(validation) < best_validation:
            best_validation = float(validation)
            best_state = copy.deepcopy(model.state_dict())
            best_step = epoch + 1
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, {"train_loss_last": train_history[-1], "validation_loss_initial": initial_validation,
                   "validation_loss_last": last_validation, "validation_loss_best": best_validation,
                   "best_step": best_step,
                   "epochs": epochs, "train_maps": int(train.size(0)), "validation_maps": int(valid.size(0))}


def _freeze(models: Iterable[nn.Module]) -> None:
    for model in models:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None


def _project_ball_(z: torch.Tensor, radius: float) -> None:
    with torch.no_grad():
        norm = z.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(z.dtype).tiny)
        z.mul_((radius / norm).clamp(max=1.))


def _aligned_stack(masks: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([masks[0], *[_align_columns(masks[0], item) for item in masks[1:]]])


def _agreement_loss(aligned: torch.Tensor) -> torch.Tensor:
    return (aligned - aligned.mean(dim=0, keepdim=True)).square().mean(dim=(0, 2, 3))


def _iou_by_source(aligned_hard: torch.Tensor) -> torch.Tensor:
    """IoU against the first aligned source, including its all-one row."""
    reference = aligned_hard[0]
    scores = [torch.ones(reference.size(0), device=reference.device)]
    for current in aligned_hard[1:]:
        intersection = (reference * current).sum(dim=(1, 2))
        union = (reference + current - reference * current).sum(dim=(1, 2)).clamp_min(1.)
        scores.append(intersection / union)
    return torch.stack(scores)


def _pairwise_iou(aligned_hard: torch.Tensor) -> torch.Tensor:
    """Mean IoU of every non-reference source for each restart."""
    by_source = _iou_by_source(aligned_hard)
    return by_source[1:].mean(0) if by_source.size(0) > 1 else by_source[0]


def _search_agreement(models: list[_MaskVAE], *, starts: int, steps: int, seed: int,
                      k: int, shape: tuple[int, int], device: torch.device) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    _freeze(models)
    generator = torch.Generator(device=device).manual_seed(seed + 4099)
    radius = math.sqrt(models[0].latent_dim) * 3.
    codes = [nn.Parameter(torch.randn(starts, model.latent_dim, generator=generator, device=device)) for model in models]
    for code in codes:
        _project_ball_(code, radius)
    initial_codes = torch.stack([code.detach().clone() for code in codes])

    def evaluate(current: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        soft = [soft_topk(model.decode(code), k, .5).reshape(starts, *shape) for model, code in zip(models, current)]
        aligned = _aligned_stack(soft)
        return soft, aligned, _agreement_loss(aligned)

    with torch.no_grad():
        initial_soft, initial_aligned, initial_loss = evaluate(codes)
        best_loss = initial_loss.clone()
        best_codes = initial_codes.clone()
        best_steps = torch.zeros(starts, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(codes, lr=.03)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, _, loss = evaluate(codes)
        loss.sum().backward()
        optimizer.step()
        for code in codes:
            _project_ball_(code, radius)
        with torch.no_grad():
            _, _, current_loss = evaluate(codes)
            improved = current_loss < best_loss
            best_loss = torch.where(improved, current_loss, best_loss)
            current_codes = torch.stack([code.detach() for code in codes])
            best_codes = torch.where(improved[None, :, None], current_codes, best_codes)
            best_steps = torch.where(improved, torch.full_like(best_steps, step + 1), best_steps)
    with torch.no_grad():
        final_codes = [best_codes[index] for index in range(len(models))]
        final_soft, final_aligned, final_loss = evaluate(final_codes)
        def hard_from(codes_: torch.Tensor) -> torch.Tensor:
            return torch.stack([_hard_topk(model.decode(codes_[i]), k).reshape(starts, *shape)
                                for i, model in enumerate(models)])
        initial_hard = hard_from(initial_codes)
        final_hard = hard_from(best_codes)
        initial_hard_aligned = _aligned_stack(list(initial_hard))
        final_hard_aligned = _aligned_stack(list(final_hard))
    initial_iou_by_source = _iou_by_source(initial_hard_aligned)
    final_iou_by_source = _iou_by_source(final_hard_aligned)
    diagnostics = {
        "soft_loss_initial": initial_loss.detach().cpu().tolist(),
        "soft_loss_final": final_loss.detach().cpu().tolist(),
        "hard_iou_initial": _pairwise_iou(initial_hard_aligned).detach().cpu().tolist(),
        "hard_iou_final": _pairwise_iou(final_hard_aligned).detach().cpu().tolist(),
        "hard_iou_by_source_initial": initial_iou_by_source.detach().cpu().tolist(),
        "hard_iou_by_source_final": final_iou_by_source.detach().cpu().tolist(),
        "softness_initial": (initial_aligned * (1. - initial_aligned)).mean(dim=(0, 2, 3)).detach().cpu().tolist(),
        "softness_final": (final_aligned * (1. - final_aligned)).mean(dim=(0, 2, 3)).detach().cpu().tolist(),
        "best_step": best_steps.detach().cpu().tolist(),
        "radius": radius,
    }
    return final_hard[0].detach(), diagnostics, initial_codes[0].detach()


def extract_masks(banks: list[dict[str, Any]], seed: int, device: str | torch.device,
                  vae_epochs: int = 160, agreement_steps: int = 400, starts: int = 4,
                  density: float = .2, latent: int = 16, width: int = 128) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return label-free fixed-K baselines and agreement masks.

    ``banks`` must contain at least two dictionaries, each with ``maps`` of
    shape ``[keep, F, H]``.  Returned masks are CPU float tensors of shape
    ``[starts, F, H]`` and all methods except ``dense`` have exactly the same
    number of active connections.
    """
    if len(banks) < 2:
        raise ValueError("agreement requires maps from at least two source tasks")
    if starts <= 0 or vae_epochs <= 0 or agreement_steps < 0 or latent <= 0 or width <= 0:
        raise ValueError("invalid non-positive extraction hyperparameter")
    if not 0. < density < 1.:
        raise ValueError("density must be strictly between zero and one")
    target_device = torch.device(device)
    first = torch.as_tensor(banks[0]["maps"])
    if first.ndim != 3:
        raise ValueError("each bank['maps'] must have shape [keep,F,H]")
    _, features, hidden = first.shape
    if any(torch.as_tensor(bank["maps"]).ndim != 3 or tuple(torch.as_tensor(bank["maps"]).shape[1:]) != (features, hidden)
           for bank in banks):
        raise ValueError("all source maps must have the same [F,H] shape")
    flat_dim = features * hidden
    k = int(round(density * flat_dim))
    k = min(max(k, 1), flat_dim - 1)
    generator = torch.Generator(device=target_device).manual_seed(seed + 101)
    raw = [_validated_maps(bank["maps"], target_device) for bank in banks]
    train_raw, validation_raw = zip(*[_split_unique(task, generator) for task in raw])
    aligned_train, consensus = _consensus_align(list(train_raw))
    # Validation maps are aligned only to the training consensus: neither
    # their columns nor their values take part in the baseline/VAE fit.
    aligned_validation = [_align_columns(consensus.expand_as(task), task) for task in validation_raw]

    models: list[_MaskVAE] = []
    vae_diagnostics = []
    for index, (train, valid) in enumerate(zip(aligned_train, aligned_validation)):
        model, report = _fit_vae(train, valid, flat_dim=flat_dim, latent=latent, width=width,
                                 epochs=vae_epochs, seed=seed + index * 1009, device=target_device)
        models.append(model)
        report["task"] = str(banks[index].get("name", banks[index].get("task", index)))
        vae_diagnostics.append(report)
        print("[masks] vae task={task} val={before:.4f}->{after:.4f} best={best:.4f} step={step}/{epochs}".format(
            task=report["task"], before=report["validation_loss_initial"],
            after=report["validation_loss_last"], best=report["validation_loss_best"],
            step=report["best_step"], epochs=vae_epochs), flush=True)

    agreement, agreement_report, agreement_initial_z = _search_agreement(
        models, starts=starts, steps=agreement_steps, seed=seed, k=k,
        shape=(features, hidden), device=target_device,
    )
    print("[masks] agreement soft={:.6f}->{:.6f} hard-IoU={:.4f}->{:.4f}".format(
        sum(agreement_report["soft_loss_initial"]) / starts,
        sum(agreement_report["soft_loss_final"]) / starts,
        sum(agreement_report["hard_iou_initial"]) / starts,
        sum(agreement_report["hard_iou_final"]) / starts), flush=True)
    with torch.no_grad():
        # Same first-decoder initial z as agreement makes this baseline a fair
        # test of cross-task agreement rather than a more fortunate prior draw.
        single = _hard_topk(models[0].decode(agreement_initial_z), k).reshape(starts, features, hidden)
    mean_logits = torch.cat(aligned_train, dim=0).mean(0).reshape(-1)
    mean = _hard_topk(mean_logits.expand(starts, -1), k).reshape(starts, features, hidden)
    random_scores = torch.rand(starts, flat_dim, generator=generator, device=target_device)
    random = _hard_topk(random_scores, k).reshape(starts, features, hidden)
    dense = torch.ones(starts, features, hidden, device=target_device)
    masks = {"agreement": agreement.cpu(), "mean": mean.cpu(), "single_vae": single.cpu(),
             "random": random.cpu(), "dense": dense.cpu()}
    diagnostics: dict[str, Any] = {
        "shape": [features, hidden], "k": k, "density": k / flat_dim,
        "source_tasks": [str(bank.get("name", bank.get("task", index))) for index, bank in enumerate(banks)],
        "source_unique_maps": [int(_unique_rows(task).size(0)) for task in raw],
        "vae": vae_diagnostics, "agreement": agreement_report,
        "alignment": {"iterations": 4, "train_consensus_mean": float(consensus.mean().detach().cpu())},
        # Tensor payload is intentionally separate from JSON serializable
        # diagnostics.  The caller may torch.save it after removing this key.
        "_artifacts": {"vae_state_dicts": [{key: value.detach().cpu() for key, value in model.state_dict().items()}
                                             for model in models],
                       "agreement_initial_z_first": agreement_initial_z.detach().cpu(),
                       "consensus": consensus.detach().cpu()},
    }
    return masks, diagnostics
