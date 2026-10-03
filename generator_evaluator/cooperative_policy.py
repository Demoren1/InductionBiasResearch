"""Cooperative search policy for one generator per target pattern.

The generators are deliberately independent: a generator only receives a
score-function update from its own pattern.  They cooperate through a shared
quality ensemble and through *measured* common-good masks.  In particular,
predicted cross-task quality is useful for acquisition, but is never used as
an elite distillation target.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adapters import FunctionalBank
from .data import topology_id
from .models import QualityEnsemble, TransformerMaskGenerator
from .training import generator_update, propose_candidates
from .quality_objectives import quality_objective_cost, validate_quality_objective


class DensityConditionedGenerator(nn.Module):
    """A mask generator whose raw bank profiles are conditioned on requested density.

    ``token_dim`` remains the dimension of a stored functional map.  The
    wrapped Transformer sees one additional, constant channel on every bank
    token, so a sparse map is not implicitly treated as a request for the
    same output density.  No bank or hidden-neuron position is introduced.
    """

    def __init__(
        self,
        token_dim: int,
        features: int,
        hidden: int,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        noise_dim: int = 16,
        quality_dim: int = 1,
        *,
        target_k: int | None = None,
    ) -> None:
        super().__init__()
        total = features * hidden
        if target_k is None:
            target_k = total
        if not 1 <= target_k <= total:
            raise ValueError("target_k must be within the output mask size")
        self.token_dim = token_dim
        self.features = features
        self.hidden = hidden
        self.noise_dim = noise_dim
        self.quality_dim = quality_dim
        self.width = width
        self.target_k = int(target_k)
        self.inner = TransformerMaskGenerator(
            token_dim + 1, features, hidden, width, heads, layers, noise_dim, quality_dim
        )

    def set_budget(self, k: int) -> None:
        if not 1 <= int(k) <= self.features * self.hidden:
            raise ValueError("k must be within the output mask size")
        self.target_k = int(k)

    def _validate_bank(self, tokens: Tensor, quality: Tensor | None) -> tuple[int, int]:
        if not isinstance(tokens, Tensor) or tokens.ndim != 4 or not tokens.is_floating_point():
            raise ValueError("tokens must be a floating [B,R,H,D] tensor")
        if not torch.isfinite(tokens).all().item():
            raise ValueError("tokens must be finite")
        batch, solutions, source_hidden, dimension = tokens.shape
        if solutions < 1 or source_hidden < 1 or dimension != self.token_dim:
            raise ValueError("tokens have invalid bank dimensions")
        if quality is not None:
            if (not isinstance(quality, Tensor) or quality.ndim != 3 or
                    quality.shape != (batch, solutions, self.quality_dim) or
                    not quality.is_floating_point() or not torch.isfinite(quality).all().item()):
                raise ValueError("quality has invalid bank dimensions")
        return batch, solutions

    def _density(self, density: Tensor | float | None, batch: int, tokens: Tensor) -> Tensor:
        if density is None:
            value = tokens.new_full((batch,), self.target_k / (self.features * self.hidden))
        else:
            value = torch.as_tensor(density, device=tokens.device, dtype=tokens.dtype)
            if value.ndim == 0:
                value = value.expand(batch)
            if value.shape != (batch,):
                raise ValueError("density must be a scalar or one value per bank")
        if not torch.isfinite(value).all().item() or bool((value <= 0).any()) or bool((value > 1).any()):
            raise ValueError("density must be finite and in (0, 1]")
        return value

    def _augment(self, tokens: Tensor, density: Tensor | float | None) -> Tensor:
        batch, _ = self._validate_bank(tokens, None)
        return self._augment_validated(tokens, density, batch)

    def _augment_validated(self, tokens: Tensor, density: Tensor | float | None, batch: int) -> Tensor:
        """Append requested density after the public token boundary was checked."""
        values = self._density(density, batch, tokens)
        channel = values[:, None, None, None].expand(*tokens.shape[:-1], 1)
        return torch.cat((tokens, channel), dim=-1)

    def encode_bank(self, tokens: Tensor, quality: Tensor | None = None, *, density: Tensor | float | None = None) -> Tensor:
        batch, solutions = self._validate_bank(tokens, quality)
        return self.inner._encode_bank_validated(self._augment_validated(tokens, density, batch), quality,
                                                  batch, solutions)

    def forward(
        self,
        tokens: Tensor,
        noise: Tensor,
        quality: Tensor | None = None,
        *,
        density: Tensor | float | None = None,
    ) -> Tensor:
        batch, _ = self._validate_bank(tokens, quality)
        # ``inner`` owns validation of noise at the public boundary; density
        # conditioning has already checked the shared bank inputs.
        if (not isinstance(noise, Tensor) or noise.ndim != 2 or not noise.is_floating_point() or
                noise.shape != (batch, self.noise_dim) or not torch.isfinite(noise).all().item()):
            raise ValueError("noise has invalid generator dimensions")
        return self.inner._forward_validated(self._augment_validated(tokens, density, batch), noise, quality, batch)


def _hungarian(cost: Tensor) -> list[int]:
    """Return minimum-cost column assignment for a square CPU-sized matrix."""
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("Hungarian matching requires a square cost matrix")
    n = cost.shape[0]
    # Classic O(n^3) potential implementation.  The small assignment lives on
    # CPU because it chooses a discrete target permutation and has no gradient.
    values = cost.detach().double().cpu().tolist()
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [float("inf")] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = values[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [0] * n  # target/row -> prediction/column
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


def align_elite_to_logits(elite: Tensor, logits: Tensor) -> Tensor:
    """Place an elite's hidden columns in the coordinate order preferred by logits."""
    if elite.ndim != 2 or logits.ndim != 2 or elite.shape != logits.shape:
        raise ValueError("elite and logits must have equal [features, hidden] shapes")
    if not bool(((elite == 0) | (elite == 1)).all()):
        raise ValueError("elite must be a binary hard mask")
    probability = logits.detach().sigmoid().clamp(1e-6, 1 - 1e-6)
    target = elite.detach()
    # cost[target column, output coordinate] = BCE for that pairing.
    cost = -(target.T[:, None, :] * probability.T[None, :, :].log() +
             (1 - target.T[:, None, :]) * (1 - probability.T[None, :, :]).log()).sum(-1)
    target_to_output = _hungarian(cost)
    aligned = torch.empty_like(target)
    for target_column, output_column in enumerate(target_to_output):
        aligned[:, output_column] = target[:, target_column]
    return aligned


def cooperative_generator_update(
    model: nn.Module,
    ensemble: QualityEnsemble,
    tokens: Tensor,
    quality: Tensor | None,
    contexts: Tensor,
    dense_quality: Tensor,
    optimizer: torch.optim.Optimizer,
    k: int,
    rng: torch.Generator | None = None,
    *,
    task_index: int,
    elite_masks: Tensor | None = None,
    agreement_weight: float = 0.1,
    permutation_weight: float = 1.0,
    accumulate: bool = False,
) -> dict[str, float]:
    """Update one pattern generator and optionally distil real common-good elites.

    The policy update intentionally passes a one-task context into
    :func:`generator_update`; global quality is used when candidates are ranked,
    not as an accidental shared objective for every individual generator.
    """
    if not 0 <= task_index < len(contexts) or agreement_weight < 0:
        raise ValueError("invalid task_index or agreement_weight")
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    quality = None if quality is None else quality.to(device)
    if hasattr(model, "set_budget"):
        model.set_budget(k)  # type: ignore[attr-defined]
    own_context, own_dense = contexts[task_index:task_index + 1], dense_quality[task_index:task_index + 1]
    logs = generator_update(
        model, ensemble, tokens, quality, own_context, own_dense, optimizer, k, rng,
        permutation_weight=permutation_weight,
        **({"accumulate": True} if accumulate else {}),
    )
    if elite_masks is None or len(elite_masks) == 0 or agreement_weight == 0:
        logs.update(elite_distillation_loss=0.0, elite_count=0.0)
        return logs
    elite_masks = torch.as_tensor(elite_masks, device=tokens.device, dtype=tokens.dtype)
    if elite_masks.ndim != 3 or elite_masks.shape[1:] != (model.features, model.hidden):
        raise ValueError("elite_masks must be [N, features, hidden]")
    if not bool(((elite_masks == 0) | (elite_masks == 1)).all()):
        raise ValueError("elite masks must be binary")
    # A common elite may have been found at a different density.  It is not a
    # valid target for the current exact-K policy draw, so simply omit it.
    elite_masks = elite_masks[elite_masks.sum((1, 2)) == k]
    if len(elite_masks) == 0:
        logs.update(elite_distillation_loss=0.0, elite_count=0.0)
        return logs
    raw_tokens = tokens.unsqueeze(0) if tokens.ndim == 3 else tokens
    raw_quality = None if quality is None else (quality.unsqueeze(0) if quality.ndim == 2 else quality)
    if raw_tokens.shape[0] != 1:
        raise ValueError("cooperative updates require one functional bank")
    draws = 2
    noise = torch.randn((draws, model.noise_dim), device=tokens.device, dtype=tokens.dtype, generator=rng)
    logits = model(raw_tokens.expand(draws, *raw_tokens.shape[1:]), noise,
                   None if raw_quality is None else raw_quality.expand(draws, *raw_quality.shape[1:]))
    selected = torch.randint(len(elite_masks), (draws,), device=tokens.device, generator=rng)
    targets = torch.stack([align_elite_to_logits(elite_masks[row], logits[index])
                           for index, row in enumerate(selected.tolist())])
    distillation = F.binary_cross_entropy_with_logits(logits, targets)
    loss = agreement_weight * distillation
    if not accumulate:
        optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if not accumulate:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
    logs.update(elite_distillation_loss=float(distillation.detach().cpu()), elite_count=float(len(elite_masks)))
    return logs


def _random_mask(features: int, hidden: int, k: int, device: torch.device, dtype: torch.dtype,
                 rng: torch.Generator | None) -> Tensor:
    mask = torch.zeros(features * hidden, device=device, dtype=dtype)
    rows = _randperm(mask.numel(), rng, device)[:k]
    mask[rows] = 1
    return mask.reshape(features, hidden)


def _rng_device(rng: torch.Generator | None, fallback: torch.device) -> torch.device:
    """The tensor sampled by a Generator must live on its device type."""
    if rng is None:
        return fallback
    return torch.device(rng.device)


def _randperm(size: int, rng: torch.Generator | None, output_device: torch.device) -> Tensor:
    """Draw with ``rng`` then move only discrete indices to the pool device."""
    return torch.randperm(size, generator=rng, device=_rng_device(rng, output_device)).to(output_device)


def _rand_index(size: int, rng: torch.Generator | None, fallback: torch.device) -> int:
    return int(torch.randint(size, (), generator=rng, device=_rng_device(rng, fallback)).cpu())


def _mutate(mask: Tensor, k: int, rng: torch.Generator | None) -> Tensor:
    flat = mask.flatten().clone()
    on, off = (flat == 1).nonzero().flatten(), (flat == 0).nonzero().flatten()
    if len(on) == 0 or len(off) == 0:
        return flat.reshape_as(mask)
    remove = on[_rand_index(len(on), rng, mask.device)]
    add = off[_rand_index(len(off), rng, mask.device)]
    flat[remove], flat[add] = 0, 1
    if int(flat.sum()) != k:
        raise AssertionError("edge-swap mutation changed mask budget")
    return flat.reshape_as(mask)


def propose_at_budget(
    model: nn.Module,
    bank: FunctionalBank,
    k: int,
    count: int,
    rng: torch.Generator | None = None,
) -> Tensor:
    """Generate exact-K candidates while explicitly conditioning on that budget."""
    if hasattr(model, "set_budget"):
        model.set_budget(k)  # type: ignore[attr-defined]
    device = next(model.parameters()).device
    return propose_candidates(model, bank.tokens.to(device),
                              None if bank.quality is None else bank.quality.to(device),
                              k, count, rng)


def propose_shared_pool(
    models: Mapping[str, nn.Module],
    banks: Mapping[str, FunctionalBank],
    k: int,
    candidates_per_generator: int,
    rng: torch.Generator | None,
    random_count: int = 4,
    mutation_count: int = 4,
    elites: Tensor | None = None,
    excluded_topologies: Sequence[str] | None = None,
    proposal_trace: dict[str, Any] | None = None,
    paired_proposals: bool = False,
    shared_noise: Tensor | None = None,
) -> tuple[Tensor, list[str]]:
    """Pool exact-K proposals, random exploration, and edge-swap elite variants.

    When ``proposal_trace`` is supplied, it is populated before global
    canonical deduplication.  Thus it diagnoses overlap of proposal families,
    rather than falsely reporting that a family made no proposal because its
    candidate was already in the common pool.
    """
    if not models or set(models) != set(banks) or candidates_per_generator < 1:
        raise ValueError("models and banks must have matching nonempty keys")
    first = next(iter(models.values()))
    features, hidden = first.features, first.hidden
    if not 1 <= k <= features * hidden:
        raise ValueError("k must be within mask size")
    excluded = set(excluded_topologies or ())
    pool: list[Tensor] = []
    sources: list[str] = []
    if proposal_trace is not None:
        proposal_trace["generators"] = {}
    paired_count = ((candidates_per_generator + 1) // 2
                    if shared_noise is not None or paired_proposals else 0)
    if paired_count:
        if any(model.noise_dim != first.noise_dim for model in models.values()):
            raise ValueError("paired proposals require the same latent dimension")
        if shared_noise is None:
            shared_noise = torch.randn((paired_count, first.noise_dim), generator=rng,
                                       device=_rng_device(rng, torch.device("cpu")))
        else:
            if (not isinstance(shared_noise, Tensor) or shared_noise.ndim not in (1, 2) or
                    shared_noise.shape[-1] != first.noise_dim or not shared_noise.is_floating_point() or
                    not bool(torch.isfinite(shared_noise).all())):
                raise ValueError("shared_noise must be a finite floating [noise_dim] or [count, noise_dim] tensor")
            if shared_noise.ndim == 1:
                shared_noise = shared_noise.unsqueeze(0).expand(paired_count, -1)
            elif shared_noise.shape[0] == 1:
                shared_noise = shared_noise.expand(paired_count, -1)
            elif shared_noise.shape[0] < paired_count:
                raise ValueError("shared_noise must have one row or at least one row per paired candidate")
            else:
                shared_noise = shared_noise[:paired_count]

    def add(mask: Tensor, source: str) -> None:
        mask = mask.detach().float()
        if mask.shape != (features, hidden) or int(mask.sum()) != k:
            raise ValueError("all pooled masks must have common dimensions and exact K")
        identity = topology_id(mask)
        if identity not in excluded:
            excluded.add(identity)
            pool.append(mask)
            sources.append(source)

    for name, model in models.items():
        if (model.features, model.hidden) != (features, hidden):
            raise ValueError("all generators must have matching output coordinates")
        bank = banks[name]
        if paired_count:
            model.set_budget(k)
            device = next(model.parameters()).device
            tokens = bank.tokens.to(device)
            quality = None if bank.quality is None else bank.quality.to(device)
            with torch.no_grad():
                logits = model(tokens.expand(paired_count, *tokens.shape[1:]),
                               shared_noise.to(device=device, dtype=tokens.dtype),
                               None if quality is None else quality.expand(paired_count, *quality.shape[1:]))
                hard = torch.zeros_like(logits.flatten(1))
                hard.scatter_(1, logits.flatten(1).topk(k, dim=1).indices, 1.)
                proposed = hard.reshape_as(logits)
            stochastic_count = candidates_per_generator - paired_count
            if stochastic_count:
                proposed = torch.cat((proposed, propose_at_budget(model, bank, k, stochastic_count, rng)))
        else:
            proposed = propose_at_budget(model, bank, k, candidates_per_generator, rng)
        if proposal_trace is not None:
            # These IDs are canonical with respect to hidden-column order but
            # preserve output-row coordinates, exactly the intended agreement
            # relation.  Do not apply exclusions here.
            unique = sorted({topology_id(mask) for mask in proposed})
            proposal_trace["generators"][name] = {
                "topology_ids": unique,
                "sampled_count": int(len(proposed)),
                "unique_count": int(len(unique)),
                "paired_topology_ids": [topology_id(mask) for mask in proposed[:paired_count]],
            }
        for mask in proposed:
            add(mask.cpu(), f"generator:{name}")
    for _ in range(random_count):
        add(_random_mask(features, hidden, k, torch.device("cpu"), torch.float32, rng), "random")
    if elites is not None and len(elites):
        elite_rows = torch.as_tensor(elites).detach().cpu().float()
        elite_rows = elite_rows[elite_rows.sum((1, 2)) == k]
        for index in range(mutation_count if len(elite_rows) else 0):
            add(_mutate(elite_rows[index % len(elite_rows)], k, rng), "mutation")
    if not pool:
        return torch.empty(0, features, hidden), []
    return torch.stack(pool), sources


def rank_shared_pool(masks: Tensor, ensemble: QualityEnsemble, contexts: Tensor,
                     dense_quality: Tensor,
                     quality_objective: str = "worst") -> dict[str, Tensor]:
    """Predict every proposed mask on every pattern using the global ensemble."""
    validate_quality_objective(quality_objective)
    if masks.ndim != 3 or contexts.ndim != 2 or dense_quality.shape != (len(contexts),):
        raise ValueError("invalid pool, contexts, or dense quality")
    count, tasks = len(masks), len(contexts)
    device = next(ensemble.parameters(), masks).device
    scorer_masks = masks.to(device)
    context_batch = contexts.to(scorer_masks).repeat(count, 1)
    mask_batch = scorer_masks[:, None].expand(-1, tasks, -1, -1).reshape(-1, *masks.shape[1:])
    with torch.no_grad():
        mean, std = ensemble.predict(mask_batch, context_batch)
    mean, std = mean.reshape(count, tasks), std.reshape(count, tasks)
    delta = mean - dense_quality.to(mean).unsqueeze(0)
    return {"mean": mean, "std": std, "delta": delta,
            "worst_delta": delta.max(dim=1).values,
            "max_std": std.max(dim=1).values,
            "objective_cost": quality_objective_cost(delta, quality_objective, task_dim=1)}


def mixed_acquisition(masks: Tensor, sources: Sequence[str], ranks: Mapping[str, Tensor], budget: int,
                      rng: torch.Generator | None = None) -> tuple[Tensor, list[str], list[str]]:
    """Mix predicted-good, uncertain, and random candidates, retaining provenance."""
    if budget < 1 or len(masks) != len(sources) or "worst_delta" not in ranks or "max_std" not in ranks:
        raise ValueError("invalid mixed acquisition inputs")
    promising_cost = ranks.get("objective_cost", ranks["worst_delta"])
    promising = torch.argsort(promising_cost).tolist()
    uncertain = torch.argsort(ranks["max_std"], descending=True).tolist()
    random_order = _randperm(len(masks), rng, masks.device).tolist()
    pools = (("promising", promising), ("uncertain", uncertain), ("random", random_order))
    position, selected, labels, trace = [0, 0, 0], [], [], []
    while len(selected) < min(budget, len(masks)):
        progressed = False
        for slot, (label, order) in enumerate(pools):
            while position[slot] < len(order) and order[position[slot]] in selected:
                position[slot] += 1
            if position[slot] < len(order):
                row = order[position[slot]]
                position[slot] += 1
                selected.append(row), labels.append(label), trace.append(sources[row])
                progressed = True
                if len(selected) == min(budget, len(masks)):
                    break
        if not progressed:
            break
    return torch.tensor(selected, device=masks.device), labels, trace


def select_common_elites(masks: Tensor, real_quality: Tensor, dense_quality: Tensor, *,
                         margin: float = 0.0, limit: int = 8,
                         quality_objective: str = "worst") -> Tensor:
    """Keep only masks measured good on *every* train pattern; never backfill."""
    validate_quality_objective(quality_objective)
    if masks.ndim != 3 or real_quality.ndim != 2 or real_quality.shape[0] != len(masks):
        raise ValueError("masks and real_quality must have matching candidate rows")
    if dense_quality.shape != (real_quality.shape[1],) or limit < 1:
        raise ValueError("invalid dense quality or elite limit")
    delta = real_quality - dense_quality.to(real_quality).unsqueeze(0)
    qualified = (delta <= margin).all(dim=1)
    rows = qualified.nonzero().flatten()
    if len(rows) == 0:
        return masks.new_empty((0, *masks.shape[1:]))
    costs = quality_objective_cost(delta.index_select(0, rows), quality_objective, task_dim=1)
    ordered = rows[torch.argsort(costs)[:limit]]
    return masks.index_select(0, ordered)
