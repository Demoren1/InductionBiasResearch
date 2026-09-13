"""Evolutionary latent search scored only through binary-mask MLP training."""

from __future__ import annotations

from typing import Any, Sequence

import torch

import config
from data.generate import make_dataset
from evaluation.single_z_task import (
    _assert_unchanged,
    _freeze_and_snapshot,
    _fresh_mlp,
    _hard_from_logits,
    _mlp_forward,
    _per_network_bce,
)
from evaluation.z_star_noise import project, seed_for


def _paired_params(n_parents: int, n_candidates: int, seed: int,
                   device: torch.device) -> list[torch.nn.Parameter]:
    """Give every candidate of one parent exactly the same initial MLP."""
    base = _fresh_mlp(n_parents, seed, device)
    return [torch.nn.Parameter(value.detach().repeat_interleave(n_candidates, dim=0))
            for value in base]


def fit_and_score(
    masks: torch.Tensor,
    pattern: str,
    device: torch.device,
    *,
    steps: int,
    model_seed: int,
    train_seed_base: int,
    eval_seed: int,
    return_accuracy: bool = False,
) -> dict[str, torch.Tensor]:
    """Train paired MLPs on hard masks and return per-candidate held-out scores."""
    if masks.ndim != 4 or masks.shape[-2:] != (config.SEQ_LEN, config.H):
        raise ValueError("masks must have shape (parents, candidates, 8, 8)")
    if not torch.equal(masks, masks.round()) or not (masks.sum((-1, -2)) == config.K_ACTIVE).all():
        raise ValueError("sampling evaluator accepts only binary exact-K masks")
    n_parents, n_candidates = masks.shape[:2]
    flat_masks = masks.to(device).reshape(-1, config.SEQ_LEN, config.H)
    params = _paired_params(n_parents, n_candidates, model_seed, device)
    optimizer = torch.optim.Adam(params, lr=config.LR)
    pat_int = int(pattern, 2)
    for step in range(steps):
        data = make_dataset(pattern, config.TRAIN_BATCH_SIZE,
                            train_seed_base + pat_int * 1_000_000 + step,
                            config.POS_FRACTION)
        xb, yb = data["x"].to(device), data["y"].to(device)
        loss = _per_network_bce(_mlp_forward(xb, *params, flat_masks), yb)
        optimizer.zero_grad(set_to_none=True)
        loss.sum().backward()
        optimizer.step()
    evaluation = make_dataset(pattern, config.N_VAL_SAMPLES, eval_seed + pat_int,
                              config.POS_FRACTION)
    xb, yb = evaluation["x"].to(device), evaluation["y"].to(device)
    with torch.no_grad():
        logits = _mlp_forward(xb, *params, flat_masks)
        bce = _per_network_bce(logits, yb).reshape(n_parents, n_candidates)
        result = {"bce": bce.cpu()}
        if return_accuracy:
            accuracy = ((logits > 0) == yb[:, None]).float().mean(0)
            result["accuracy"] = accuracy.reshape(n_parents, n_candidates).cpu()
    return result


def decode_hard(model: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    shape = z.shape
    flat = z.reshape(-1, shape[-1])
    with torch.no_grad():
        logits = model.decode(flat, flat.new_empty(len(flat), 0))
        masks = _hard_from_logits(logits)
    return masks.reshape(*shape[:-1], config.SEQ_LEN, config.H)


def sample_candidates(parents: torch.Tensor, radii: Sequence[float], radius: float,
                      seed: int) -> torch.Tensor:
    """Return parent plus fixed-L2 mutations, all projected into the latent ball."""
    if not radii:
        raise ValueError("at least one mutation radius is required")
    device = parents.device
    generator = torch.Generator(device=device).manual_seed(seed)
    directions = torch.randn(len(parents), len(radii), parents.size(1),
                             generator=generator, device=device)
    directions /= directions.norm(dim=2, keepdim=True).clamp_min(
        torch.finfo(directions.dtype).tiny)
    scale = parents.new_tensor(list(radii))[None, :, None]
    mutations = project((parents[:, None, :] + scale * directions).reshape(-1, parents.size(1)),
                        radius).reshape(len(parents), len(radii), parents.size(1))
    return torch.cat([parents[:, None, :], mutations], dim=1)


def unique_mask_counts(masks: torch.Tensor) -> list[int]:
    flat = masks.reshape(masks.size(0), masks.size(1), -1).cpu()
    return [int(torch.unique(row, dim=0).size(0)) for row in flat]


def evolutionary_search(
    model: torch.nn.Module,
    initial_z: torch.Tensor,
    pattern: str,
    device: str | torch.device,
    *,
    radius: float,
    radii_schedule: Sequence[Sequence[float]],
    screen_steps: int,
    refine_steps: int,
    finalists: int,
    final_eval_steps: int,
    seed: int,
) -> dict[str, Any]:
    """Run a (1+lambda) hard-mask search with paired MLP accept/reject scoring."""
    if finalists < 2 or finalists > 1 + len(radii_schedule[0]):
        raise ValueError("finalists must contain the parent and at least one mutation")
    if any(len(radii) != len(radii_schedule[0]) for radii in radii_schedule):
        raise ValueError("all generations must use the same mutation count")
    device = torch.device(device)
    before = _freeze_and_snapshot(model, device)
    parents = project(initial_z.detach().to(device).float(), radius)
    initial = parents.detach().cpu().clone()
    initial_masks = decode_hard(model, parents).cpu()
    history = []
    pat_int = int(pattern, 2)

    for generation, radii in enumerate(radii_schedule):
        candidate_z = sample_candidates(
            parents, radii, radius,
            seed_for("hard-sampling", seed, pat_int, generation),
        )
        candidate_masks = decode_hard(model, candidate_z)
        screen = fit_and_score(
            candidate_masks, pattern, device, steps=screen_steps,
            model_seed=seed_for("screen-model", seed, pat_int, generation),
            train_seed_base=110_000_000 + generation * 100_000,
            eval_seed=81000,
        )["bce"].to(device)

        # Parent is always refined. The remaining slots are the best mutations.
        mutation_indices = screen[:, 1:].topk(finalists - 1, dim=1, largest=False).indices + 1
        refined_indices = torch.cat([
            torch.zeros(len(parents), 1, dtype=torch.long, device=device), mutation_indices
        ], dim=1)
        batch_index = torch.arange(len(parents), device=device)[:, None]
        refined_z = candidate_z[batch_index, refined_indices]
        refined_masks = candidate_masks[batch_index, refined_indices]
        refined = fit_and_score(
            refined_masks, pattern, device, steps=refine_steps,
            model_seed=seed_for("refine-model", seed, pat_int, generation),
            train_seed_base=210_000_000 + generation * 100_000,
            eval_seed=81000,
        )["bce"].to(device)
        best_mutation_value, best_mutation_offset = refined[:, 1:].min(dim=1)
        best_mutation_index = best_mutation_offset + 1
        best_mutation_differs = (
            refined_masks[torch.arange(len(parents), device=device), best_mutation_index]
            != refined_masks[:, 0]
        ).any(-1).any(-1)
        accepted = (best_mutation_value < refined[:, 0] - 1e-6) & best_mutation_differs
        winners = torch.where(accepted, best_mutation_index, torch.zeros_like(best_mutation_index))
        previous = parents
        parents = refined_z[torch.arange(len(parents), device=device), winners].detach()
        different = (candidate_masks[:, 1:] != candidate_masks[:, :1]).any(-1).any(-1)
        history.append({
            "generation": generation,
            "radii": list(map(float, radii)),
            "accepted": accepted.cpu(),
            "accepted_fraction": float(accepted.float().mean()),
            "candidate_different_support_fraction": float(different.float().mean()),
            "unique_mask_counts": unique_mask_counts(candidate_masks),
            "screen_parent_bce": screen[:, 0].cpu(),
            "screen_best_mutation_bce": screen[:, 1:].min(dim=1).values.cpu(),
            "refine_parent_bce": refined[:, 0].cpu(),
            "refine_winner_bce": refined.min(dim=1).values.cpu(),
            "selected_candidate_index": refined_indices[
                torch.arange(len(parents), device=device), winners].cpu(),
            "step_l2": (parents - previous).norm(dim=1).cpu(),
        })
        print(f"[hard-sampling] pattern={pattern} generation={generation+1}/{len(radii_schedule)} "
              f"accepted={float(accepted.float().mean()):.3f} "
              f"different={float(different.float().mean()):.3f} "
              f"unique={float(torch.tensor(history[-1]['unique_mask_counts']).float().mean()):.2f}",
              flush=True)

    final = parents.cpu()
    final_masks = decode_hard(model, parents).cpu()
    paired_masks = torch.stack([initial_masks, final_masks], dim=1)
    final_eval = fit_and_score(
        paired_masks, pattern, device, steps=final_eval_steps,
        model_seed=seed_for("final-eval-model", seed, pat_int),
        train_seed_base=310_000_000,
        eval_seed=1000,
        return_accuracy=True,
    )
    _assert_unchanged(model, before)
    return {
        "initial_z": initial, "final_z": final,
        "initial_masks": initial_masks, "final_masks": final_masks,
        "history": history,
        "final_eval_initial_bce": final_eval["bce"][:, 0],
        "final_eval_final_bce": final_eval["bce"][:, 1],
        "final_eval_initial_accuracy": final_eval["accuracy"][:, 0],
        "final_eval_final_accuracy": final_eval["accuracy"][:, 1],
        "decoder_unchanged": True,
        "settings": {
            "parents": len(initial), "radius": radius,
            "generations": len(radii_schedule),
            "mutations": len(radii_schedule[0]),
            "radii_schedule": [list(map(float, row)) for row in radii_schedule],
            "screen_steps": screen_steps, "refine_steps": refine_steps,
            "finalists": finalists, "final_eval_steps": final_eval_steps,
            "seed": seed,
            "selection": "parent-retaining paired hard-mask validation BCE",
            "gold_used": False,
        },
    }
