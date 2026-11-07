"""Optimize continuous assignments while task codebooks keep learning."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import BASE, load_archive, save_json, search_configurations
from .model import (
    BatchedAdam,
    assignment_variance,
    balanced_loss,
    clipped,
    initial_numeric,
    metrics,
    schedule,
    soft_predict,
)


def roots(plan, output, phase):
    configurations = search_configurations(plan)
    if phase == "pilot":
        selected = configurations
        fixed = plan["search"].get("fixed_configuration_ids")
        if fixed is not None:
            selected = [config for config in selected if config["id"] in fixed]
        seeds = plan["search"]["pilot_seeds"]
        orders = ["first"]
    else:
        selection = json.loads((output / "pilot_selection.json").read_text())
        ids = selection["top_configurations"]
        selected = [config for config in configurations if config["id"] in ids]
        seeds = plan["search"]["confirmation_seeds"]
        orders = plan["search"]["confirmation_orders"]
    return [
        {**config, "search_seed": seed, "order": order}
        for config in selected
        for seed in seeds
        for order in orders
    ]


def train(plan, output, phase, device, shard, shards):
    specifications = roots(plan, output, phase)
    selected = [
        row for number, row in enumerate(specifications) if number % shards == shard
    ]
    destination = output / phase / "search" / f"shard_{shard}"
    destination.mkdir(parents=True, exist_ok=False)
    save_json(destination / "roots.json", selected)
    data = load_archive(output / "source.npz")
    support_x = torch.tensor(data["support_x"], device=device)
    support_y = torch.tensor(data["support_y"], device=device)[None, :, None]
    query_x = torch.tensor(data["query_x"], device=device)
    query_y = torch.tensor(data["query_y"], device=device)[None, :, None]
    spec = plan["search"]
    classes = plan.get("classes", plan["codes"] + 1)
    if classes not in (plan["codes"] + 1, 2 * plan["codes"] + 1):
        raise ValueError("Invalid free or sign-paired assignment class count")
    pair_coefficients = torch.tensor(
        [row.get("pair_penalty", 0.0) for row in selected], device=device
    )
    variance_coefficients = torch.tensor(
        [row.get("variance_penalty", 0.0) for row in selected], device=device
    )
    use_pair_penalty = bool(pair_coefficients.any())
    use_variance_penalty = bool(variance_coefficients.any())
    if (pair_coefficients < 0).any() or (variance_coefficients < 0).any():
        raise ValueError("Assignment regularizer coefficients must be nonnegative")
    if use_pair_penalty and classes != 2 * plan["codes"] + 1:
        raise ValueError("Pair exclusivity requires explicit C/-C assignment classes")
    steps = spec[phase + "_steps"]
    snapshots = set(spec[phase + "_snapshots"])
    alpha = torch.tensor(
        np.stack(
            [
                np.random.default_rng(
                    plan["seed"] + 1000007 * row["search_seed"]
                ).normal(0, spec["initial_logit_std"], size=(11, 8, classes))
                for row in selected
            ]
        ),
        dtype=torch.float32,
        device=device,
    ).requires_grad_()
    zero_probability = spec["initial_zero_probability"]
    with torch.no_grad():
        alpha[..., 0] += np.log(
            (classes - 1) * zero_probability / (1 - zero_probability)
        )
    numeric = initial_numeric(
        [plan["seed"] + 1000007 * row["search_seed"] + 31 for row in selected],
        len(plan["source_patterns"]),
        spec["numeric_restarts"],
        spec["initial_codebook_norm"],
        device,
        codes=plan["codes"],
        ordered_signed=spec.get("ordered_signed_initialization", False),
    ).requires_grad_()
    numeric_optimizer = BatchedAdam(numeric, [row["numeric_lr"] for row in selected])
    alpha_optimizer = BatchedAdam(alpha, [row["architecture_lr"] for row in selected])
    unrolled = torch.tensor(
        [row["order"] == "unrolled" for row in selected], device=device
    )
    use_unrolled = bool(unrolled.any())
    rng = torch.Generator(device=device).manual_seed(plan["seed"] + 700001)
    paired = spec.get("paired_batch_streams", False)
    query_rng = torch.Generator(device=device).manual_seed(plan["seed"] + 700003)
    update_mode = spec.get("update_mode", "alternating")
    if update_mode not in ("alternating", "synchronous_split", "joint_support"):
        raise ValueError("Unknown architecture/codebook update mode")
    if update_mode != "alternating" and use_unrolled:
        raise ValueError("Synchronous ablations use direct gradients")
    history, saved_steps, saved_alpha, saved_numeric = [], [], [], []
    saved_soft, saved_hard = [], []
    started = time.perf_counter()

    def numeric_step(probabilities, ids=None):
        if ids is None:
            ids = torch.randint(
                len(support_x), (spec["batch"],), generator=rng, device=device
            )
        loss = balanced_loss(
            soft_predict(support_x[ids], probabilities.detach(), numeric),
            support_y[..., ids],
        )
        loss = loss + spec["numeric_l2"] * numeric.square().sum(-1) / 2
        gradient = torch.autograd.grad(loss.sum(), numeric)[0]
        numeric_optimizer.step(clipped(gradient, spec["numeric_gradient_clip"], (-1,)))

    with torch.no_grad():
        initial_probabilities = alpha.softmax(-1)
    for _ in range(spec["numeric_warmup"]):
        numeric_step(initial_probabilities)
    for step in range(steps + 1):
        temperature, entropy_penalty = schedule(selected, step, steps, spec, device)
        if step % spec["checkpoint_every"] == 0 or step == steps:
            with torch.no_grad():
                probabilities = (alpha / temperature).softmax(-1)
                training = metrics(
                    soft_predict(support_x, probabilities, numeric), support_y
                )
                query = metrics(soft_predict(query_x, probabilities, numeric), query_y)
                hard = torch.nn.functional.one_hot(
                    probabilities.argmax(-1), classes
                ).float()
                inherited = metrics(soft_predict(query_x, hard, numeric), query_y)
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(
                    -1
                )
                record = {
                    "step": step,
                    "numeric_updates": numeric_optimizer.steps,
                    "train": training.mean((1, 2)).cpu().tolist(),
                    "query": query.mean((1, 2)).cpu().tolist(),
                    "inherited_hard_query": inherited.mean((1, 2)).cpu().tolist(),
                    "entropy": entropy.mean((1, 2)).cpu().tolist(),
                    "maximum_probability": probabilities.amax(-1)
                    .mean((1, 2))
                    .cpu()
                    .tolist(),
                    "hard_active_edges": (hard.argmax(-1) > 0)
                    .sum((1, 2))
                    .cpu()
                    .tolist(),
                    "temperature": temperature.flatten().cpu().tolist(),
                    "seconds": time.perf_counter() - started,
                }
                if not torch.isfinite(query).all() or not torch.isfinite(numeric).all():
                    raise FloatingPointError(
                        "Non-finite search state; inspect saved diagnostics"
                    )
                history.append(record)
                save_json(destination / "history.json", history)
                if step in snapshots:
                    saved_steps.append(step)
                    saved_alpha.append(alpha.detach().cpu().numpy().copy())
                    saved_numeric.append(numeric.detach().cpu().numpy().copy())
                    saved_soft.append(query.cpu().numpy())
                    saved_hard.append(inherited.cpu().numpy())
                    np.savez_compressed(
                        destination / "snapshots.npz",
                        steps=np.array(saved_steps),
                        alpha=np.stack(saved_alpha),
                        numeric=np.stack(saved_numeric),
                        soft_query=np.stack(saved_soft),
                        inherited_hard_query=np.stack(saved_hard),
                    )
                if step % 500 == 0 or step == steps:
                    print(
                        f"{phase} shard={shard} step={step} "
                        f"soft_BCE={float(query[..., 0].mean()):.4f} "
                        f"hard_BCE={float(inherited[..., 0].mean()):.4f} "
                        f"entropy={float(entropy.mean()):.3f} "
                        f"seconds={record['seconds']:.1f}",
                        flush=True,
                    )
        if step == steps:
            break
        with torch.no_grad():
            frozen_probabilities = (alpha / temperature).softmax(-1)
        batches = None
        query_ids = None
        if paired:
            batches = [
                torch.randint(
                    len(support_x), (spec["batch"],), generator=rng, device=device
                )
                for _ in range(3)
            ]
            query_ids = torch.randint(
                len(query_x), (spec["batch"],), generator=query_rng, device=device
            )
        simultaneous_gradient = None
        if update_mode == "alternating":
            for number in range(spec["numeric_steps_per_outer"]):
                numeric_step(
                    frozen_probabilities, None if batches is None else batches[number]
                )
        probabilities = (alpha / temperature).softmax(-1)
        if update_mode != "alternating":
            ids = (
                batches[0]
                if batches is not None
                else torch.randint(
                    len(support_x), (spec["batch"],), generator=rng, device=device
                )
            )
            support_loss = balanced_loss(
                soft_predict(
                    support_x[ids],
                    probabilities
                    if update_mode == "joint_support"
                    else probabilities.detach(),
                    numeric,
                ),
                support_y[..., ids],
            )
            numerical_objective = (
                support_loss + spec["numeric_l2"] * numeric.square().sum(-1) / 2
            )
            simultaneous_gradient = torch.autograd.grad(
                numerical_objective.sum(),
                numeric,
                retain_graph=update_mode == "joint_support",
            )[0]
            architecture_numeric = numeric.detach()
        elif use_unrolled:
            ids = torch.randint(
                len(support_x), (spec["batch"],), generator=rng, device=device
            )
            current = numeric.detach().requires_grad_()
            training_loss = balanced_loss(
                soft_predict(support_x[ids], probabilities, current),
                support_y[..., ids],
            )
            training_loss = (
                training_loss + spec["numeric_l2"] * current.square().sum(-1) / 2
            )
            gradient = torch.autograd.grad(
                training_loss.sum(), current, create_graph=True
            )[0]
            gradient = clipped(gradient, spec["numeric_gradient_clip"], (-1,))
            virtual = numeric_optimizer.virtual(current, gradient)
            architecture_numeric = torch.where(
                unrolled[:, None, None, None], virtual, current.detach()
            )
        else:
            architecture_numeric = numeric.detach()
        if update_mode == "joint_support":
            objective = support_loss.mean((1, 2))
        else:
            ids = (
                query_ids
                if query_ids is not None
                else torch.randint(
                    len(query_x), (spec["batch"],), generator=rng, device=device
                )
            )
            objective = balanced_loss(
                soft_predict(query_x[ids], probabilities, architecture_numeric),
                query_y[..., ids],
            ).mean((1, 2))
        entropy = (
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean((1, 2))
        )
        objective = objective + entropy_penalty * entropy
        sparsity = torch.tensor(
            [row.get("sparsity_penalty", 0.0) for row in selected], device=device
        )
        objective = objective + sparsity * (1 - probabilities[..., 0]).mean((1, 2))
        if use_pair_penalty:
            codes = plan["codes"]
            conflict = (
                (probabilities[..., 1 : codes + 1] * probabilities[..., codes + 1 :])
                .sum(-1)
                .mean((1, 2))
            )
            objective = objective + pair_coefficients * conflict
        if use_variance_penalty:
            variance = assignment_variance(probabilities, numeric).mean((1, 2, 3, 4))
            objective = objective + variance_coefficients * variance
        gradient = torch.autograd.grad(objective.sum(), alpha)[0]
        if simultaneous_gradient is not None:
            numeric_optimizer.step(
                clipped(simultaneous_gradient, spec["numeric_gradient_clip"], (-1,))
            )
        alpha_optimizer.step(
            clipped(gradient, spec["architecture_gradient_clip"], (1, 2, 3))
        )
        with torch.no_grad():
            alpha.sub_(alpha.mean(-1, keepdim=True))
    torch.save(
        {
            "alpha": alpha.detach().cpu(),
            "numeric": numeric.detach().cpu(),
            "alpha_first": alpha_optimizer.first.cpu(),
            "alpha_second": alpha_optimizer.second.cpu(),
            "numeric_first": numeric_optimizer.first.cpu(),
            "numeric_second": numeric_optimizer.second.cpu(),
            "steps": steps,
            "numeric_steps": numeric_optimizer.steps,
        },
        destination / "last.pt",
    )
    save_json(
        destination / "complete.json",
        {
            "seconds": time.perf_counter() - started,
            "roots": len(selected),
            "steps": steps,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BASE / "results/pattern_20261104"
    )
    parser.add_argument("--phase", choices=("pilot", "confirmation"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan = json.loads((args.output / "protocol.json").read_text())
    train(
        plan,
        args.output,
        args.phase,
        torch.device(args.device),
        args.shard,
        args.shards,
    )


if __name__ == "__main__":
    main()
