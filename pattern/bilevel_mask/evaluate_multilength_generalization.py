"""Evaluate frozen multi-length generators on unseen pattern lengths."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .generated_sharing import SharingGenerator, _network_losses, _project, adapt_weights, assignments, sharing_forward
from .multilength_sharing import (
    MultiLengthConfig,
    _fit_dense,
    _fit_shared,
    _fixed_generated,
    _oracle_fixed,
    _sample_task_indices,
    _task_groups,
    all_tasks,
    make_data,
)


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _load_config(raw: dict[str, Any], unseen_lengths: tuple[int, ...],
                 outer_steps: int) -> MultiLengthConfig:
    raw = dict(raw)
    raw["pattern_lengths"] = unseen_lengths
    raw["outer_steps"] = outer_steps
    return MultiLengthConfig(**raw)


def _hard_score(
    generator: SharingGenerator,
    z: torch.Tensor,
    data: Any,
    config: MultiLengthConfig,
    device: torch.device,
) -> tuple[float, torch.Tensor, list[float]]:
    with torch.no_grad():
        candidates = assignments(generator, z, config, config.temperature_end, "hard")
    losses: list[torch.Tensor] = []
    for start in range(0, len(data.tasks), config.checkpoint_task_chunk):
        stop = min(start + config.checkpoint_task_chunk, len(data.tasks))
        indices = torch.arange(start, stop, device=device)
        batch = data.subset(indices)
        groups = _task_groups(batch.tasks, config, "length_latent", device)
        structure = candidates[groups]
        weights = adapt_weights(
            structure, batch.support_x, batch.support_y, config,
            steps=config.inner_steps,
            seed=_seed(config.seed, "unseen-checkpoint", start),
            create_graph=False,
        )
        with torch.no_grad():
            losses.append(_network_losses(
                sharing_forward(batch.validation_x, weights, structure), batch.validation_y,
            ))
    validation = torch.cat(losses)
    length_rows = []
    for length in config.pattern_lengths:
        mask = torch.tensor([task.length == length for task in data.tasks], device=device)
        length_rows.append(validation[mask].mean(0))
    by_length = torch.stack(length_rows)
    restarts = by_length.argmin(1)
    selected = by_length[torch.arange(len(config.pattern_lengths), device=device), restarts]
    return float(selected.mean()), restarts, [float(value) for value in selected]


def optimize_unseen_latents(
    generator: SharingGenerator,
    initial: torch.Tensor,
    config: MultiLengthConfig,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    tasks = all_tasks(config.pattern_lengths)
    data = make_data(tasks, config, device, "unseen-latent-search")
    rng = torch.Generator(device="cpu").manual_seed(_seed(config.seed, "unseen-z"))
    z_value = torch.randn(
        len(config.pattern_lengths), config.train_restarts, config.latent_dim, generator=rng,
    ).to(device)
    z_value[:, :1] = initial
    z = nn.Parameter(z_value)
    _project(z, config.z_radius)
    optimizer = torch.optim.Adam([z], lr=config.latent_lr)
    best_score, best_restarts, initial_scores = _hard_score(generator, z, data, config, device)
    best_z = z[torch.arange(len(config.pattern_lengths), device=device), best_restarts][:, None].detach().clone()
    history: list[dict[str, Any]] = [{
        "step": 0, "hard_validation_bce": best_score,
        "hard_validation_by_length": initial_scores, "improved": True,
    }]
    stale = 0
    for step in range(1, config.outer_steps + 1):
        ratio = min(1.0, (step - 1) / max(1, config.temperature_anneal_steps - 1))
        temperature = config.temperature_start * (
            config.temperature_end / config.temperature_start
        ) ** ratio
        task_rng = torch.Generator(device="cpu").manual_seed(
            _seed(config.seed, "unseen-task-batch", step)
        )
        indices = _sample_task_indices(tasks, config, task_rng, device)
        batch = data.subset(indices)
        groups = _task_groups(batch.tasks, config, "length_latent", device)
        structure = assignments(generator, z, config, temperature, "soft")[groups]
        weights = adapt_weights(
            structure, batch.support_x, batch.support_y, config,
            steps=config.inner_steps,
            seed=_seed(config.seed, "unseen-adapt", step),
            create_graph=True,
        )
        validation = _network_losses(
            sharing_forward(batch.validation_x, weights, structure), batch.validation_y,
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        validation.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_([z], 10.0))
        optimizer.step()
        _project(z, config.z_radius)
        row: dict[str, Any] = {
            "step": step, "soft_validation_bce": float(validation.detach()),
            "temperature": temperature, "gradient_norm": gradient_norm,
        }
        if step % config.checkpoint_every == 0 or step == config.outer_steps:
            score, restarts, length_scores = _hard_score(generator, z, data, config, device)
            improved = score < best_score - config.outer_min_delta
            row.update({
                "hard_validation_bce": score,
                "hard_validation_by_length": length_scores,
                "improved": improved,
            })
            if improved:
                best_score = score
                best_z = z[
                    torch.arange(len(config.pattern_lengths), device=device), restarts
                ][:, None].detach().clone()
                stale = 0
            elif step >= config.temperature_anneal_steps:
                stale += 1
        history.append(row)
        if step % 10 == 0:
            print(
                f"UNSEEN seed={config.seed} step={step}/{config.outer_steps} "
                f"soft={float(validation.detach()):.6f}"
                + (f" hard={row['hard_validation_bce']:.6f}" if "hard_validation_bce" in row else ""),
                flush=True,
            )
        if step >= config.temperature_anneal_steps and stale >= config.outer_patience:
            break
    history.append({
        "selected_validation_bce": best_score,
        "completed_steps": step,
        "converged": step >= config.temperature_anneal_steps and stale >= config.outer_patience,
    })
    return best_z, history


def _category_maps(generator: SharingGenerator, z: torch.Tensor,
                   config: MultiLengthConfig) -> list[list[list[int]]]:
    with torch.no_grad():
        hard = assignments(generator, z, config, config.temperature_end, "hard").squeeze(1)
        active = hard.sum(-1).bool()
        category = hard.argmax(-1).add(1)
        category = torch.where(active, category, torch.zeros_like(category))
    return category.cpu().tolist()


def _evaluate(
    generator: SharingGenerator,
    z: torch.Tensor,
    config: MultiLengthConfig,
    variant: str,
    device: torch.device,
    method: str,
    tag: str,
) -> dict[str, Any]:
    evaluation_config = replace(
        config,
        support_per_class=config.evaluation_support_per_class,
        validation_per_class=config.evaluation_validation_per_class,
        query_per_class=config.evaluation_query_per_class,
    )
    tasks = all_tasks(evaluation_config.pattern_lengths)
    data = make_data(tasks, evaluation_config, device, tag)
    structure = _fixed_generated(generator, z, tasks, evaluation_config, variant, device)
    return _fit_shared(structure, data, evaluation_config, device, method, tag)


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--unseen-lengths", default="4,6")
    parser.add_argument("--latent-steps", type=int, default=500)
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    unseen_lengths = tuple(int(value) for value in args.unseen_lengths.split(","))
    config = _load_config(checkpoint["config"], unseen_lengths, args.latent_steps)
    source_variant = checkpoint["variant"]
    generator = SharingGenerator(config).to(device)
    generator.load_state_dict(checkpoint["generator_state"])
    generator.requires_grad_(False)
    generator.eval()
    trained_z = checkpoint["z"].to(device)

    result: dict[str, Any] = {
        "config": config.to_dict(), "source_variant": source_variant,
        "source_checkpoint": str(args.checkpoint), "strategies": {}, "masks": {},
    }
    if source_variant == "global":
        result["strategies"]["global_zero_shot"] = _evaluate(
            generator, trained_z, config, "global", device,
            "global z, zero-shot", "unseen-test",
        )
        result["masks"]["global_zero_shot"] = {
            str(length): _category_maps(generator, trained_z, config)[0]
            for length in unseen_lengths
        }
    elif source_variant == "length_latent":
        if trained_z.shape[0] != 3 or len(unseen_lengths) != 2:
            raise ValueError("interpolation expects trained lengths 3,5,7 and unseen lengths 4,6")
        interpolated = torch.stack((
            0.5 * (trained_z[0] + trained_z[1]),
            0.5 * (trained_z[1] + trained_z[2]),
        ))
        _project(interpolated, config.z_radius)
        adapted, history = optimize_unseen_latents(generator, interpolated, config, device)
        result["latent_search_history"] = history
        result["strategies"]["latent_interpolation"] = _evaluate(
            generator, interpolated, config, "length_latent", device,
            "midpoint latent, zero-shot", "unseen-test",
        )
        result["strategies"]["latent_adaptation"] = _evaluate(
            generator, adapted, config, "length_latent", device,
            "optimized unseen latent", "unseen-test",
        )
        result["masks"]["latent_interpolation"] = dict(zip(
            map(str, unseen_lengths), _category_maps(generator, interpolated, config),
        ))
        result["masks"]["latent_adaptation"] = dict(zip(
            map(str, unseen_lengths), _category_maps(generator, adapted, config),
        ))
        result["adapted_z"] = adapted.cpu().tolist()
    else:
        raise ValueError(f"unknown source variant: {source_variant}")

    evaluation_config = replace(
        config,
        support_per_class=config.evaluation_support_per_class,
        validation_per_class=config.evaluation_validation_per_class,
        query_per_class=config.evaluation_query_per_class,
    )
    tasks = all_tasks(evaluation_config.pattern_lengths)
    data = make_data(tasks, evaluation_config, device, "unseen-test")
    oracle = _oracle_fixed(tasks, evaluation_config, device)
    dense = torch.ones(
        len(tasks), evaluation_config.eval_restarts,
        evaluation_config.seq_len, evaluation_config.hidden, device=device,
    )
    result["strategies"]["analytic_sharing"] = _fit_shared(
        oracle, data, evaluation_config, device, "analytic cyclic sharing", "unseen-test",
    )
    result["strategies"]["dense"] = _fit_dense(
        dense, data, evaluation_config, device, "dense MLP", "unseen-test",
    )
    result["masks"]["trained"] = {}
    trained_lengths = (3, 5, 7)
    train_config = _load_config(checkpoint["config"], trained_lengths, args.latent_steps)
    trained_maps = _category_maps(generator, trained_z, train_config)
    if source_variant == "global":
        result["masks"]["trained"] = {str(length): trained_maps[0] for length in trained_lengths}
    else:
        result["masks"]["trained"] = dict(zip(map(str, trained_lengths), trained_maps))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
