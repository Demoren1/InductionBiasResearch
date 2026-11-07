"""Explicit soft architecture search and hard numerical refitting."""

import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import save_json
from .model import (
    BatchedOptimizer,
    clip_gradient,
    fixed_weights,
    hard_indices,
    initial_numeric,
    predict,
    score,
    soft_weights,
    source_tensors,
)


def configurations(specification):
    configurations = []
    for numeric, architecture, radius, sparsity in itertools.product(
        specification["numeric_rates"],
        specification["architecture_rates"],
        specification["radii"],
        specification["sparsity"],
    ):
        configurations.append(
            {
                "numeric_lr": numeric,
                "architecture_lr": architecture,
                "radius": radius,
                "sparsity": sparsity,
                "mode": "joint",
            }
        )
    for radius, sparsity in itertools.product(
        specification["radii"], specification["sparsity"]
    ):
        configurations.append(
            {
                "numeric_lr": 0.01,
                "architecture_lr": 0.01,
                "radius": radius,
                "sparsity": sparsity,
                "mode": "alternating",
            }
        )
    return [{**row, "configuration": i} for i, row in enumerate(configurations)]


def search(output, plan, specification, rows, phase, device):
    destination = output / phase
    destination.mkdir(parents=True, exist_ok=False)
    save_json(destination / "rows.json", rows)
    spec = plan["search"]
    steps = spec[phase + "_steps"]
    codes = specification["codes"]
    shape = (specification["inputs"], specification["hidden"], 2 * codes + 1)
    arrays = []
    for row in rows:
        rng = np.random.default_rng(plan["seed"] + 100003 * row["seed"])
        arrays.append(rng.normal(0, spec["initial_logit_std"], shape))
    alpha = torch.tensor(np.array(arrays), device=device, dtype=torch.float32)
    with torch.no_grad():
        probability = spec["zero_probability"]
        alpha[..., 0] += math.log(2 * codes * probability / (1 - probability))
    alpha.requires_grad_()
    numeric = initial_numeric(
        rows,
        spec["restarts"],
        specification,
        plan["seed"] + 901,
        device,
        probabilities=alpha.detach().softmax(-1),
    ).requires_grad_()
    numeric_optimizer = BatchedOptimizer(numeric, rows)
    alpha_optimizer = BatchedOptimizer(alpha, rows, "architecture_lr")
    data = source_tensors(output, device)
    variance = specification["target_variance"]
    alternating = torch.tensor(
        [row["mode"] == "alternating" for row in rows], device=device
    )
    sparsity = numeric.new_tensor([row["sparsity"] for row in rows])
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + 1001)
    snapshots = {round(fraction * steps) for fraction in spec["snapshots"]}
    histories, saved = [], []
    started = time.perf_counter()

    def numerical_loss(probabilities, ids):
        weights = soft_weights(probabilities, numeric, codes)
        prediction = predict(data["support_x"][ids], weights, numeric, specification)
        loss = (prediction - data["support_y"][ids]).square().mean(-1) / variance
        return loss + spec["numeric_l2"] * numeric.square().sum(-1) / 2

    for _ in range(spec["numeric_warmup"]):
        ids = torch.randint(
            len(data["support_x"]), (spec["batch"],), device=device, generator=generator
        )
        loss = numerical_loss(alpha.detach().softmax(-1), ids)
        gradient = torch.autograd.grad(loss.sum(), numeric)[0]
        numeric_optimizer.step(clip_gradient(gradient, spec["gradient_clip"], (-1,)))
    for step in range(steps + 1):
        progress = step / max(steps, 1)
        cooling = min(1.0, max(0.0, (progress - 0.3) / 0.7))
        temperature = spec["minimum_temperature"] ** cooling
        probabilities = (alpha / temperature).softmax(-1)
        if step % spec["checkpoint_every"] == 0 or step == steps:
            with torch.no_grad():
                weights = soft_weights(probabilities, numeric, codes)
                query = score(
                    data["query_x"], data["query_y"], weights, numeric, specification
                )
                ids = torch.arange(min(512, len(data["support_x"])), device=device)
                training = score(
                    data["support_x"][ids],
                    data["support_y"][ids],
                    weights,
                    numeric,
                    specification,
                )
                indices = hard_indices(probabilities)
                hard = score(
                    data["query_x"],
                    data["query_y"],
                    fixed_weights(indices.long(), numeric, codes),
                    numeric,
                    specification,
                )
                entropy = (
                    -(probabilities * probabilities.clamp_min(1e-12).log())
                    .sum(-1)
                    .mean((1, 2))
                )
                histories.append(
                    {
                        "step": step,
                        "train_mse": training.cpu().tolist(),
                        "query_mse": query.cpu().tolist(),
                        "hard_query_mse": hard.cpu().tolist(),
                        "entropy": entropy.cpu().tolist(),
                        "seconds": time.perf_counter() - started,
                    }
                )
                if not torch.isfinite(query).all():
                    raise FloatingPointError("Non-finite architecture fit")
                save_json(destination / "history.json", histories)
                if step in snapshots:
                    saved.append(
                        {
                            "step": step,
                            "probabilities": probabilities.detach().cpu().numpy(),
                            "numeric": numeric.detach().cpu().numpy(),
                        }
                    )
                    np.savez_compressed(
                        destination / "snapshots.npz",
                        steps=np.array([item["step"] for item in saved]),
                        probabilities=np.stack(
                            [item["probabilities"] for item in saved]
                        ),
                        numeric=np.stack([item["numeric"] for item in saved]),
                    )
                print(
                    f"{specification['name']} {phase} step={step} soft_NMSE={query.mean().item() / variance:.5f} hard_NMSE={hard.mean().item() / variance:.5f} seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )
        if step == steps:
            break
        support_ids = torch.randint(
            len(data["support_x"]), (spec["batch"],), generator=generator, device=device
        )
        if alternating.any():
            loss = numerical_loss(probabilities.detach(), support_ids)
            gradient = torch.autograd.grad(loss.sum(), numeric)[0]
            numeric_optimizer.step(
                clip_gradient(gradient, spec["gradient_clip"], (-1,)),
                active=alternating,
            )
        weights = soft_weights(probabilities, numeric, codes)
        support_prediction = predict(
            data["support_x"][support_ids], weights, numeric, specification
        )
        support_loss = (
            support_prediction - data["support_y"][support_ids]
        ).square().mean(-1) / variance
        objective = support_loss.mean(-1)
        if alternating.any():
            query_ids = torch.randint(
                len(data["query_x"]),
                (spec["batch"],),
                generator=generator,
                device=device,
            )
            query_prediction = predict(
                data["query_x"][query_ids],
                soft_weights(probabilities, numeric.detach(), codes),
                numeric.detach(),
                specification,
            )
            query_loss = (query_prediction - data["query_y"][query_ids]).square().mean(
                -1
            ) / variance
            objective = torch.where(alternating, query_loss.mean(-1), objective)
        regularized = support_loss + spec["numeric_l2"] * numeric.square().sum(-1) / 2
        numerical_gradient = torch.autograd.grad(
            regularized.sum(), numeric, retain_graph=True
        )[0]
        entropy = (
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean((1, 2))
        )
        objective = objective + sparsity * (1 - probabilities[..., 0]).mean((1, 2))
        objective = (
            objective + spec["late_entropy"] * max(0, 2 * progress - 1) * entropy
        )
        alpha_gradient = torch.autograd.grad(objective.sum(), alpha)[0]
        numeric_optimizer.step(
            clip_gradient(numerical_gradient, spec["gradient_clip"], (-1,)),
            active=~alternating,
        )
        alpha_optimizer.step(
            clip_gradient(alpha_gradient, spec["alpha_clip"], (1, 2, 3))
        )
        with torch.no_grad():
            alpha.sub_(alpha.mean(-1, keepdim=True))
    save_json(
        destination / "complete.json",
        {
            "seconds": time.perf_counter() - started,
            "steps": steps,
            "numeric_updates_per_root": numeric_optimizer.steps.flatten()
            .cpu()
            .tolist(),
            "roots": len(rows),
            "numeric_restarts": spec["restarts"],
        },
    )
    return saved


def fit(
    output,
    plan,
    specification,
    rows,
    device,
    indices=None,
    initial=None,
    steps=None,
    restarts=None,
    mask=None,
    initial_direction=None,
):
    """Query alone selects each numerical checkpoint; selection is evaluated once."""
    spec = plan["fit"]
    steps = spec["steps"] if steps is None else steps
    restarts = spec["restarts"] if restarts is None else restarts
    data = source_tensors(output, device)
    codes = specification["codes"]
    dense = indices is None
    if dense:
        weights = []
        for row_number, row in enumerate(rows):
            rng = np.random.default_rng(plan["seed"] + 100003 * row["seed"] + 7727)
            item = rng.normal(
                size=(restarts, specification["inputs"], specification["hidden"])
            )
            if initial_direction is not None:
                item = np.asarray(initial_direction)[
                    None, :, None
                ] + 0.02 * item / math.sqrt(specification["inputs"])
            if mask is not None:
                item *= mask[row_number]
            norms = np.linalg.norm(item.reshape(restarts, -1), axis=-1)
            radius = row["radius"] * (
                math.sqrt(specification["hidden"])
                if specification["kind"] == "deepsets"
                else 1
            )
            weights.append(item * (radius / norms)[:, None, None])
        weights_parameter = torch.tensor(
            np.array(weights), device=device, dtype=torch.float32
        ).requires_grad_()
        if specification.get("head") == "full":
            numeric = torch.zeros(
                len(rows),
                restarts,
                codes + 2 * specification["hidden"] + 1,
                device=device,
            )
            numeric[..., codes : codes + specification["hidden"]] = 0.1
            numeric[..., codes + specification["hidden"] : -1] = 1.0
            numeric[..., -1] = specification["target_mean"]
        else:
            numeric = torch.zeros(len(rows), restarts, codes + 3, device=device)
            numeric[..., codes] = 0.1 if specification["kind"] == "deepsets" else 0.0
            numeric[..., codes + 1] = 1.0
            numeric[..., codes + 2] = (
                specification["target_mean"]
                if specification["kind"] == "deepsets"
                else 0.0
            )
        indices_tensor = None
    else:
        indices_tensor = torch.as_tensor(indices, device=device, dtype=torch.long)
        numeric = (
            initial_numeric(
                rows,
                restarts,
                specification,
                plan["seed"] + 11017,
                device,
                indices=indices_tensor,
            )
            if initial is None
            else torch.as_tensor(initial, device=device, dtype=torch.float32).clone()
        )
        weights_parameter = None
    numeric.requires_grad_()
    parameters = (
        [numeric] if weights_parameter is None else [numeric, weights_parameter]
    )
    optimizers = [BatchedOptimizer(parameter, rows) for parameter in parameters]
    best_loss = torch.full(numeric.shape[:2], torch.inf, device=device)
    best_numeric = numeric.detach().clone()
    best_weights = torch.zeros(
        len(rows),
        restarts,
        specification["inputs"],
        specification["hidden"],
        device=device,
    )
    best_step = torch.zeros(numeric.shape[:2], device=device, dtype=torch.long)
    history_train, history_query, history_steps = [], [], []
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + 15019)
    started = time.perf_counter()
    for step in range(steps + 1):
        weights = (
            weights_parameter
            if dense
            else fixed_weights(indices_tensor, numeric, codes)
        )
        if mask is not None:
            weights = (
                weights
                * torch.as_tensor(mask, device=device, dtype=weights.dtype)[:, None]
            )
        if step % spec["checkpoint_every"] == 0 or step == steps:
            with torch.no_grad():
                query = score(
                    data["query_x"], data["query_y"], weights, numeric, specification
                )
                improved = query < best_loss
                best_loss[improved] = query[improved]
                best_numeric[improved] = numeric[improved]
                best_weights[improved] = weights[improved]
                best_step[improved] = step
                training = score(
                    data["support_x"][:512],
                    data["support_y"][:512],
                    weights,
                    numeric,
                    specification,
                )
                history_steps.append(step)
                history_train.append(training.cpu().numpy())
                history_query.append(query.cpu().numpy())
                if not torch.isfinite(query).all():
                    raise FloatingPointError("Non-finite numerical fit")
            if step % 1000 == 0 or step == steps:
                print(
                    f"{specification['name']} fit rows={len(rows)} restarts={restarts} step={step} best_NMSE={best_loss.mean().item() / specification['target_variance']:.5f} seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )
        if step == steps:
            break
        ids = torch.randint(
            len(data["support_x"]), (spec["batch"],), device=device, generator=generator
        )
        prediction = predict(data["support_x"][ids], weights, numeric, specification)
        loss = (prediction - data["support_y"][ids]).square().mean(-1) / specification[
            "target_variance"
        ]
        regularizer = sum(
            parameter.square().flatten(2).sum(-1) for parameter in parameters
        )
        objective = loss + spec["numeric_l2"] * regularizer / 2
        gradients = torch.autograd.grad(objective.sum(), parameters)
        for parameter, optimizer, gradient in zip(parameters, optimizers, gradients):
            optimizer.step(
                clip_gradient(
                    gradient, spec["gradient_clip"], tuple(range(2, parameter.ndim))
                )
            )
    selection = score(
        data["selection_x"],
        data["selection_y"],
        best_weights,
        best_numeric,
        specification,
    )
    train_score = score(
        data["support_x"], data["support_y"], best_weights, best_numeric, specification
    )
    return {
        "numeric": best_numeric.cpu().numpy(),
        "weights": best_weights.cpu().numpy(),
        "query_mse": best_loss.cpu().numpy(),
        "selection_mse": selection.cpu().numpy(),
        "train_mse": train_score.cpu().numpy(),
        "step": best_step.cpu().numpy(),
        "history_steps": np.array(history_steps),
        "history_train": np.array(history_train),
        "history_query": np.array(history_query),
        "seconds": np.array(time.perf_counter() - started),
    }


def save_fit(path, result, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **result)
    save_json(
        path.with_suffix(".json"), {"rows": rows, "seconds": float(result["seconds"])}
    )
