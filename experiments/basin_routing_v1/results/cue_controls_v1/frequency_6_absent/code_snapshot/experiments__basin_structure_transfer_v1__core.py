"""Fresh numeric fitting for a sine network with tied input weights."""

import time

import numpy as np
import torch
from torch.nn import functional as F


def canonicalize(indices):
    """Name groups by first occurrence; do not change any equality."""
    result = np.empty_like(indices)
    for source, target in zip(
        indices.reshape(-1, indices.shape[-1]),
        result.reshape(-1, indices.shape[-1]),
    ):
        labels = {}
        for position, value in enumerate(source):
            target[position] = labels.setdefault(int(value), len(labels))
    return result


def make_tasks(plan, phase, device):
    """Synthetic teacher construction; hidden values are not solver inputs."""
    rng = np.random.default_rng(plan["seed"])
    library = canonicalize(
        np.stack(
            [
                rng.permutation(plan["inputs"]) % plan["groups"]
                for _ in range(plan["families"] + 1)
            ]
        )
    )
    per_family = plan[f"{phase}_tasks_per_family"]
    families = np.repeat(np.arange(plan["families"]), per_family)
    if phase == "evaluation":
        families = np.r_[
            families, np.full(plan["foreign_tasks"], plan["families"])
        ]
    offset = {"source": 1000, "calibration": 2000, "evaluation": 3000}[phase]
    rng = np.random.default_rng(plan["seed"] + offset)
    indices = library[families]
    codes = rng.normal(size=(len(families), plan["groups"]))
    codes /= np.linalg.norm(codes, axis=1, keepdims=True)
    direction = np.take_along_axis(codes, indices, axis=1)
    direction /= np.sqrt(plan["inputs"] / plan["groups"])
    source = phase == "source"
    high = rng.uniform(
        *plan["source_frequency" if source else "high_frequency"],
        len(families),
    )
    low = rng.uniform(*plan["low_frequency"], len(families))
    alpha = (
        np.zeros(len(families))
        if source
        else rng.uniform(*plan["low_amplitude"], len(families))
    )
    return {
        "direction": torch.tensor(direction, device=device).float(),
        "high": torch.tensor(high, device=device).float(),
        "low": torch.tensor(low, device=device).float(),
        "alpha": torch.tensor(alpha, device=device).float(),
        "indices": indices,
        "families": families,
    }


def target(x, spec):
    projection = torch.bmm(x, spec["direction"].unsqueeze(2)).squeeze(2)
    return spec["alpha"][:, None] * torch.sin(
        projection * spec["low"][:, None]
    ) + torch.sin(projection * spec["high"][:, None])


def population_loss(weights, amplitude, spec):
    """Exact Gaussian MSE, used for reporting and geometric diagnostics."""
    low = spec["direction"] * spec["low"][:, None]
    high = spec["direction"] * spec["high"][:, None]

    def kernel(left, right):
        return 0.5 * (
            (-(left - right).square().sum(-1) / 2).exp()
            - (-(left + right).square().sum(-1) / 2).exp()
        )

    alpha = spec["alpha"]
    energy = (
        alpha.square() * kernel(low, low)
        + kernel(high, high)
        + 2 * alpha * kernel(low, high)
    )
    correlation = alpha[:, None] * kernel(weights, low[:, None]) + kernel(
        weights, high[:, None]
    )
    return (
        amplitude.square() * kernel(weights, weights)
        - 2 * amplitude * correlation
        + energy[:, None]
    )


def fit(
    plan,
    spec,
    configs,
    starts,
    seed,
    device,
    indices=None,
    release_initial=False,
    source_weights=None,
    steps=None,
    query_seed=None,
    initial_direction=None,
):
    """Train on fresh minibatches; only empirical query chooses checkpoints.

    Code coordinates use an orthonormal group basis. Every start has the
    same specified weight norm, independent of the number of parameters.
    Numerical initializations and minibatches have independent RNG streams.
    """
    tasks, dimensions = spec["direction"].shape
    steps = plan["steps"] if steps is None else steps
    config_count = len(configs)
    replicas = config_count * starts
    rng = torch.Generator(device=device).manual_seed(seed)
    base = torch.randn(tasks, starts, dimensions, generator=rng, device=device)
    amplitude_base = 1 + 0.1 * torch.randn(
        tasks, starts, generator=rng, device=device
    )
    assignment, divisor = None, None
    if indices is not None:
        assignment = torch.as_tensor(indices, device=device).long()
        counts = F.one_hot(assignment, plan["groups"]).sum(2)
        coefficient = torch.zeros(tasks, starts, plan["groups"], device=device)
        coefficient.scatter_add_(2, assignment, base)
        coefficient /= counts.clamp_min(1).sqrt()
        base = coefficient
        divisor = counts.gather(2, assignment).sqrt()
        if release_initial:
            base = base.gather(2, assignment) / divisor
            assignment, divisor = None, None
    if source_weights is not None:
        picked = torch.randint(
            len(source_weights), (tasks, starts), generator=rng, device=device
        )
        base = source_weights[picked].clone()
    if initial_direction is not None:
        if indices is not None or source_weights is not None:
            raise ValueError("Task directions require a separate dense fit")
        base = torch.as_tensor(initial_direction, device=device).float()
        base = base[:, None].expand(-1, starts, -1).clone()
    base /= base.norm(dim=2, keepdim=True).clamp_min(1e-12)
    radius = torch.tensor([item["radius"] for item in configs], device=device)
    initial = base[:, None] * radius[None, :, None, None]
    values = (
        initial.expand(-1, -1, -1, -1).reshape(tasks, replicas, -1).clone()
    )
    amplitude = amplitude_base[:, None].expand(-1, config_count, -1)
    amplitude = amplitude.reshape(tasks, replicas).clone()
    if assignment is not None:
        assignment = assignment[:, None].expand(-1, config_count, -1, -1)
        assignment = assignment.reshape(tasks, replicas, dimensions)
        divisor = divisor[:, None].expand(-1, config_count, -1, -1)
        divisor = divisor.reshape(tasks, replicas, dimensions)
    rate = torch.tensor([item["lr"] for item in configs], device=device)
    rate = rate.repeat_interleave(starts)[None, :]
    velocity = [torch.zeros_like(values), torch.zeros_like(amplitude)]
    second = [torch.zeros_like(values), torch.zeros_like(amplitude)]
    best_query = torch.full((tasks, replicas), torch.inf, device=device)
    best_weights = torch.zeros(tasks, replicas, dimensions, device=device)
    best_amplitude = torch.zeros_like(amplitude)
    best_step = torch.zeros_like(amplitude, dtype=torch.long)
    data_rng = torch.Generator(device=device).manual_seed(seed + 100000)
    query_rng = torch.Generator(device=device).manual_seed(
        seed + 200000 if query_seed is None else query_seed
    )
    query_x = torch.randn(
        tasks, plan["query"], dimensions, generator=query_rng, device=device
    )
    query_y = target(query_x, spec)
    history = []
    optimizer = configs[0]["optimizer"]
    if any(item["optimizer"] != optimizer for item in configs):
        raise ValueError("One optimizer is required per batched fit")
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(steps + 1):
        values = values.detach().requires_grad_()
        amplitude = amplitude.detach().requires_grad_()
        weights = (
            values
            if assignment is None
            else (values.gather(2, assignment) / divisor)
        )
        if step % plan["log_every"] == 0 or step == steps:
            with torch.no_grad():
                query_prediction = (
                    torch.sin(torch.bmm(query_x, weights.transpose(1, 2)))
                    * amplitude[:, None]
                )
                query = (
                    (query_prediction - query_y[:, :, None]).square().mean(1)
                )
                better = query < best_query
                best_query[better] = query[better]
                best_weights[better] = weights[better]
                best_amplitude[better] = amplitude[better]
                best_step[better] = step
                history.append(
                    {
                        "step": step,
                        "query": query.cpu().tolist(),
                        "population": population_loss(weights, amplitude, spec)
                        .cpu()
                        .tolist(),
                    }
                )
        if step == steps:
            break
        x = torch.randn(
            tasks, plan["batch"], dimensions, generator=data_rng, device=device
        )
        y = target(x, spec)
        prediction = torch.sin(torch.bmm(x, weights.transpose(1, 2)))
        prediction = prediction * amplitude[:, None]
        loss = (prediction - y[:, :, None]).square().mean(1).sum()
        gradients = torch.autograd.grad(loss, (values, amplitude))
        with torch.no_grad():
            for index, (parameter, gradient) in enumerate(
                zip((values, amplitude), gradients)
            ):
                if optimizer == "adam":
                    velocity[index].mul_(0.9).add_(gradient, alpha=0.1)
                    second[index].mul_(0.999).addcmul_(
                        gradient, gradient, value=0.001
                    )
                    update = velocity[index] / (1 - 0.9 ** (step + 1))
                    denominator = (
                        second[index] / (1 - 0.999 ** (step + 1))
                    ).sqrt()
                    update = update / (denominator + 1e-8)
                elif optimizer == "sgdm":
                    velocity[index].mul_(plan["momentum"]).add_(gradient)
                    update = velocity[index]
                elif optimizer == "sgd":
                    update = gradient
                else:
                    raise ValueError(f"Unknown optimizer: {optimizer}")
                parameter -= (
                    rate[:, :, None] if index == 0 else rate
                ) * update
    torch.cuda.synchronize(device)
    with torch.no_grad():
        shape = (tasks, config_count, starts)
        result = {
            "query": best_query.cpu().numpy().reshape(shape),
            "population": population_loss(best_weights, best_amplitude, spec)
            .cpu()
            .numpy()
            .reshape(shape),
            "weights": best_weights.cpu().numpy().reshape(*shape, dimensions),
            "amplitude": best_amplitude.cpu().numpy().reshape(shape),
            "step": best_step.cpu().numpy().reshape(shape),
            "final_weights": weights.detach()
            .cpu()
            .numpy()
            .reshape(*shape, dimensions),
            "final_amplitude": amplitude.detach().cpu().numpy().reshape(shape),
            "initial_values": initial.cpu().numpy(),
        }
    return (
        result,
        history,
        {
            "seconds": time.perf_counter() - started,
            "steps": steps,
            "starts": starts,
            "configs": configs,
            "numeric_parameters": values.shape[-1] + 1,
            "fixed_indexator": assignment is not None,
            "source_numeric_initialization": source_weights is not None,
            "task_direction_initialization": initial_direction is not None,
            "common_minibatches_across_attempts": True,
        },
    )
