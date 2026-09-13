"""High-fidelity hard-mask sampling with replicated paired accept/reject."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from evaluation.hard_mask_sampling import (
    decode_hard,
    fit_and_score,
    sample_candidates,
    unique_mask_counts,
)
from evaluation.single_z_task import _assert_unchanged, _freeze_and_snapshot
from evaluation.z_star_noise import project, seed_for


def robust_winners(scores: torch.Tensor, masks: torch.Tensor,
                   min_improvement: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a mutation only when it beats parent in every score replicate."""
    if scores.ndim != 3 or masks.ndim != 4 or scores.shape[1:] != masks.shape[:2]:
        raise ValueError("scores/masks shape mismatch")
    mean_scores = scores.mean(0)
    mutation_offset = mean_scores[:, 1:].argmin(dim=1)
    mutation_index = mutation_offset + 1
    parents = torch.arange(scores.size(1), device=scores.device)
    candidate_scores = scores[:, parents, mutation_index]
    parent_scores = scores[:, :, 0]
    differs = (masks[parents, mutation_index] != masks[:, 0]).any(-1).any(-1)
    accepted = (candidate_scores < parent_scores - min_improvement).all(0) & differs
    winners = torch.where(accepted, mutation_index, torch.zeros_like(mutation_index))
    return winners, accepted


def evolutionary_search_robust(
    model: torch.nn.Module,
    initial_z: torch.Tensor,
    pattern: str,
    device: str | torch.device,
    *,
    radius: float,
    radii_schedule: Sequence[Sequence[float]],
    screen_steps: int,
    refine_steps: int,
    refine_replicates: int,
    finalists: int,
    final_eval_steps: int,
    min_improvement: float,
    seed: int,
) -> dict[str, Any]:
    """Hard sampling whose accepted mutation wins multiple full-fidelity fits."""
    if refine_replicates < 2:
        raise ValueError("robust search requires at least two refinement replicates")
    device = torch.device(device)
    before = _freeze_and_snapshot(model, device)
    parents = project(initial_z.detach().to(device).float(), radius)
    initial = parents.cpu().clone()
    initial_masks = decode_hard(model, parents).cpu()
    history = []
    pat_int = int(pattern, 2)

    for generation, radii in enumerate(radii_schedule):
        candidate_z = sample_candidates(
            parents, radii, radius,
            seed_for("robust-hard-sampling", seed, pat_int, generation),
        )
        candidate_masks = decode_hard(model, candidate_z)
        screen = fit_and_score(
            candidate_masks, pattern, device, steps=screen_steps,
            model_seed=seed_for("robust-screen-model", seed, pat_int, generation),
            train_seed_base=410_000_000 + generation * 100_000,
            eval_seed=82000,
        )["bce"].to(device)
        mutation_indices = screen[:, 1:].topk(finalists - 1, dim=1, largest=False).indices + 1
        refined_indices = torch.cat([
            torch.zeros(len(parents), 1, dtype=torch.long, device=device), mutation_indices
        ], dim=1)
        parent_index = torch.arange(len(parents), device=device)[:, None]
        refined_z = candidate_z[parent_index, refined_indices]
        refined_masks = candidate_masks[parent_index, refined_indices]
        replicate_scores = []
        for replicate in range(refine_replicates):
            replicate_scores.append(fit_and_score(
                refined_masks, pattern, device, steps=refine_steps,
                model_seed=seed_for("robust-refine-model", seed, pat_int, generation, replicate),
                train_seed_base=510_000_000 + generation * 1_000_000 + replicate * 100_000,
                eval_seed=82000,
            )["bce"])
        scores = torch.stack(replicate_scores).to(device)
        winners, accepted = robust_winners(scores, refined_masks, min_improvement)
        previous = parents
        parents = refined_z[torch.arange(len(parents), device=device), winners].detach()
        different = (candidate_masks[:, 1:] != candidate_masks[:, :1]).any(-1).any(-1)
        history.append({
            "generation": generation, "radii": list(map(float, radii)),
            "accepted": accepted.cpu(), "accepted_fraction": float(accepted.float().mean()),
            "candidate_different_support_fraction": float(different.float().mean()),
            "unique_mask_counts": unique_mask_counts(candidate_masks),
            "screen_parent_bce": screen[:, 0].cpu(),
            "screen_best_mutation_bce": screen[:, 1:].min(1).values.cpu(),
            "refine_scores": scores.cpu(),
            "selected_candidate_index": refined_indices[
                torch.arange(len(parents), device=device), winners].cpu(),
            "step_l2": (parents - previous).norm(dim=1).cpu(),
        })
        print(f"[robust-sampling] pattern={pattern} generation={generation+1}/{len(radii_schedule)} "
              f"accepted={float(accepted.float().mean()):.3f} "
              f"different={float(different.float().mean()):.3f}", flush=True)

    final = parents.cpu()
    final_masks = decode_hard(model, parents).cpu()
    paired_masks = torch.stack([initial_masks, final_masks], dim=1)
    final_eval = fit_and_score(
        paired_masks, pattern, device, steps=final_eval_steps,
        model_seed=seed_for("robust-final-eval", seed, pat_int),
        train_seed_base=610_000_000, eval_seed=1000, return_accuracy=True,
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
            "generations": len(radii_schedule), "mutations": len(radii_schedule[0]),
            "radii_schedule": [list(map(float, row)) for row in radii_schedule],
            "screen_steps": screen_steps, "refine_steps": refine_steps,
            "refine_replicates": refine_replicates, "finalists": finalists,
            "final_eval_steps": final_eval_steps, "min_improvement": min_improvement,
            "seed": seed,
            "selection": "mutation must beat parent in every paired full-fidelity replicate",
            "gold_used": False,
        },
    }
