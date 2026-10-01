"""Build larger sparse source banks for the MNIST set-score experiment.

The training and selection loop follows :mod:`deepsets_vaae.source_replay`'s
instrumented pilot builder.  The expanded bank retains more candidate models,
so its gradient is normalized by the original 128-model denominator to keep
each model's Adam update scale aligned with the pilot.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from . import core


def build_expanded_bank(
    data: dict[str, Any],
    costs: torch.Tensor | Sequence[float],
    seed: int,
    device: str | torch.device,
    *,
    candidates: int = 1024,
    keep: int = 256,
    steps: int = 800,
    batch_size: int = 32,
    set_size: int = 5,
    density: float = 0.2,
) -> dict[str, Any]:
    """Train candidate source models and return the ``keep`` best checkpoints.

    Each candidate receives an independent mask and model initialization.  A
    fixed 128-set source-validation sample selects its best checkpoint, and
    the returned bank is ordered by each candidate's best validation loss.
    """
    if min(candidates, keep, steps, batch_size, set_size) < 1 or keep > candidates:
        raise ValueError("invalid positive bank sizes")
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")

    target_device = torch.device(device)
    source_train = data["source_train"]
    source_validation = data["source_validation"]
    if not isinstance(source_train, core.Split):
        source_train = core.Split(*source_train)
    if not isinstance(source_validation, core.Split):
        source_validation = core.Split(*source_validation)
    if source_train.features.device != target_device:
        source_train = core.Split(*(value.to(target_device) for value in source_train))
        source_validation = core.Split(*(value.to(target_device) for value in source_validation))

    task_costs = core.centred_costs(costs, target_device)
    mask_generator = torch.Generator(device=target_device).manual_seed(seed + 31)
    data_generator = torch.Generator(device=target_device).manual_seed(seed + 32)
    masks = core._exact_random_masks(candidates, 32, density, device=target_device,
                                     generator=mask_generator)
    model = core.MaskedDeepSets(masks, seed=seed + 101).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    val_x, val_y = core._sets(source_validation, task_costs, 128, set_size, data_generator)

    best_loss = torch.full((candidates,), float("inf"), device=target_device)
    best_state = {name: parameter.detach().clone()
                  for name, parameter in model.named_parameters()}
    curves: list[dict[str, Any]] = []
    check_every = max(1, min(50, steps // 8))

    for step in range(1, steps + 1):
        model.train()
        x, y = core._sets(source_train, task_costs, batch_size, set_size, data_generator)
        prediction = model(x)
        per_model = prediction.sub(y[None]).square().mean(dim=1) / set_size
        # Keep the pilot's per-model gradient normalization as the bank grows.
        loss = per_model.sum() / 128
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % check_every == 0 or step == steps:
            model.eval()
            validation = core._losses(model, val_x, val_y, set_size)
            improved = validation < best_loss
            for name, parameter in model.named_parameters():
                best_state[name][improved] = parameter.detach()[improved]
            best_loss = torch.minimum(best_loss, validation)
            curves.append({
                "step": step,
                "train_normalized_mse": float(per_model.mean().item()),
                "validation_mean_normalized_mse": float(validation.mean().item()),
            })

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(best_state[name])
    chosen = best_loss.argsort()[:keep]
    state_dict = {"masks": masks[chosen].detach().cpu().float().contiguous()}
    for name in ("weight", "bias", "readout", "per_image_offset"):
        state_dict[name] = best_state[name][chosen].detach().cpu().contiguous()

    return {
        "maps": model.importance()[chosen].detach().cpu(),
        "masks": masks[chosen].detach().cpu().bool(),
        "weights": best_state["weight"][chosen].detach().cpu(),
        "validation_losses": best_loss[chosen].detach().cpu().tolist(),
        "selected_candidates": chosen.detach().cpu().tolist(),
        "training_curves": curves,
        "best_validation_normalized_mse": best_loss.detach().cpu().tolist(),
        "edges_per_mask": int(masks[0].sum().item()),
        "source_split_hashes": {
            name: data.get("split_hashes", {}).get(name)
            for name in ("source_train", "source_validation")
        },
        "state_dict": state_dict,
        "gradient_normalization": {
            "reduction": "per_model_sum_divided_by_128",
            "denominator": 128,
            "purpose": "preserve the pilot's per-model Adam gradient scale as candidates increase",
        },
    }
