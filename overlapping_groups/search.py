"""Paired CUDA latent searches using only adaptation support/query labels."""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .masks import canonical_mask, topk
from .models.task import expanded_targets, initialization
from .problem import EDGES, logits


def templates(family, functions, replicas, seed, device):
    seeds = [
        seed + 100003 * family + 1009 * function + replica
        for function in functions
        for replica in range(replicas)
    ]
    state = initialization(
        seeds, device, {"initialization": "positive_weights_zero_readout"}
    )
    return {
        key: value.reshape(len(functions), replicas, *value.shape[1:])
        for key, value in state.items()
    }, seeds


def repeated_template(template, candidates):
    return {
        key: value.repeat(1, candidates, *([1] * (value.ndim - 2)))
        for key, value in template.items()
    }


def penalty(state, mask):
    result = (state["w"] * mask).square().sum((-2, -1))
    for key in ["b", "a"]:
        result = result + state[key].square().sum(-1)
    return result + state["c"].square()


class Fitness:
    """Fit fresh MLPs; checkpoint long differentiable optimizer trajectories."""

    def __init__(self, data, template, options):
        self.data, self.template, self.options = data, template, options
        self.hard, self.differentiable = 0, 0
        self.optimizer = options.get("inner_optimizer", "sgd")
        if self.optimizer not in ["sgd", "adam"]:
            raise ValueError("Unknown inner optimizer")

    def trajectory(self, masks, checkpoints=None, differentiable=False):
        if masks.device.type != "cuda":
            raise RuntimeError("Adaptation fits require CUDA")
        options = self.options
        count = len(masks)
        if differentiable:
            self.differentiable += count
        else:
            self.hard += count
        replicas = options["replicas"]
        mask = canonical_mask(masks).repeat_interleave(replicas, 0)
        mask = mask[None].expand(len(self.template["w"]), -1, -1, -1)
        keys = list(self.template)
        x, y = self.data["support"]
        checkpoints = checkpoints or [options["inner_steps"]]
        if max(checkpoints) != options["inner_steps"]:
            raise ValueError("Last checkpoint must equal inner_steps")

        def update(values, first, second, step):
            state = dict(zip(keys, values))
            slope = (
                options["negative_slope"]
                if step < options["inner_warmup_steps"]
                else 0.0
            )
            prediction = logits(state, mask, x, slope, packed=True)
            loss = F.binary_cross_entropy_with_logits(
                prediction, expanded_targets(y, prediction), reduction="none"
            ).mean(-1)
            loss = loss + 0.5 * options["l2"] * penalty(state, mask)
            gradients = torch.autograd.grad(
                loss.sum(), values, create_graph=differentiable
            )
            if self.optimizer == "sgd":
                values = tuple(
                    value - options["inner_lr"] * gradient
                    for value, gradient in zip(values, gradients)
                )
            else:
                first = tuple(
                    0.9 * moment + 0.1 * gradient
                    for moment, gradient in zip(first, gradients)
                )
                second = tuple(
                    0.999 * moment + 0.001 * gradient.square()
                    for moment, gradient in zip(second, gradients)
                )
                eps = options.get("inner_adam_eps", 1e-6)
                values = tuple(
                    value
                    - options["inner_lr"]
                    * (mean / (1 - 0.9 ** (step + 1)))
                    / (variance / (1 - 0.999 ** (step + 1)) + eps**2).sqrt()
                    for value, mean, variance in zip(values, first, second)
                )
            if not differentiable:
                values = tuple(
                    value.detach().requires_grad_() for value in values
                )
                first = tuple(value.detach() for value in first)
                second = tuple(value.detach() for value in second)
            return values, first, second

        with torch.enable_grad():
            values = tuple(
                value.detach().requires_grad_()
                for value in repeated_template(self.template, count).values()
            )
            first = tuple(torch.zeros_like(value) for value in values)
            second = tuple(torch.zeros_like(value) for value in values)
            scores = {}
            current = 0
            checkpointed = differentiable and options["inner_steps"] > 128
            for stop in sorted(checkpoints):
                chunk = (
                    options.get("gradient_checkpoint_steps", 32)
                    if checkpointed
                    else stop
                )
                while current < stop:
                    end = min(current + chunk, stop)
                    if checkpointed:

                        def block(*flat, start=current, end=end):
                            size = len(keys)
                            state, mean, variance = (
                                flat[:size],
                                flat[size : 2 * size],
                                flat[2 * size :],
                            )
                            for step in range(start, end):
                                state, mean, variance = update(
                                    state, mean, variance, step
                                )
                            return (*state, *mean, *variance)

                        flat = checkpoint(
                            block,
                            *values,
                            *first,
                            *second,
                            use_reentrant=False,
                            preserve_rng_state=False,
                        )
                        size = len(keys)
                        values, first, second = (
                            flat[:size],
                            flat[size : 2 * size],
                            flat[2 * size :],
                        )
                    else:
                        for step in range(current, end):
                            values, first, second = update(
                                values, first, second, step
                            )
                    current = end
                prediction = logits(
                    dict(zip(keys, values)),
                    mask,
                    self.data["query"][0],
                    packed=True,
                )
                losses = F.binary_cross_entropy_with_logits(
                    prediction,
                    expanded_targets(self.data["query"][1], prediction),
                    reduction="none",
                ).mean(-1)
                losses = losses.reshape(len(values[0]), count, replicas).mean(
                    (0, 2)
                )
                if not torch.isfinite(losses).all():
                    raise RuntimeError("Nonfinite adaptation-query BCE")
                scores[stop] = losses if differentiable else losses.detach()
        return scores

    def __call__(self, masks, differentiable=False):
        return self.trajectory(masks, differentiable=differentiable)[
            self.options["inner_steps"]
        ]

    def budget(self):
        total = self.hard + self.differentiable
        fits = total * len(self.template["w"]) * self.options["replicas"]
        return {
            "candidate_fits": total,
            "hard_candidate_fits": self.hard,
            "differentiable_candidate_fits": self.differentiable,
            "individual_mlp_fits": fits,
            "inner_optimizer": self.optimizer,
            "individual_optimizer_steps": fits * self.options["inner_steps"],
        }


def hard_scores(decoder, z, fitness, batch):
    values = []
    for start in range(0, len(z), batch):
        with torch.no_grad():
            masks = topk(decoder(z[start : start + batch]))
        values.append(fitness(masks))
    return torch.cat(values)


def budget(options):
    return options["starts"] * (
        1
        + options["steps"]
        * (options["gumbel_samples"] + len(options["proposal_scales"]))
    )


def interpolate(pair, step, steps):
    return pair[0] * (pair[1] / pair[0]) ** (step / max(1, steps - 1))


def gumbel_masks(scores, temperature, noise, generator):
    uniform = torch.rand(scores.shape, generator=generator).to(scores.device)
    uniform = uniform.clamp(1e-7, 1 - 1e-7)
    perturbed = scores - noise * (-uniform.log()).log()
    remaining = perturbed.flatten(-2)
    relaxed = torch.zeros_like(remaining)
    for _ in range(EDGES):
        selection = (remaining / temperature).softmax(-1)
        relaxed = relaxed + selection
        remaining = remaining + (1 - selection).clamp_min(1e-7).log()
    relaxed = relaxed.reshape_as(scores)
    return topk(perturbed) + (relaxed - relaxed.detach())


def search(decoder, initial_z, fitness, options, method, seed, progress):
    z = initial_z.detach().clone().requires_grad_(method == "gumbel")
    scores = hard_scores(decoder, z, fitness, options["hard_batch"])
    initial_scores = scores.clone()
    optimizer = (
        torch.optim.Adam([z], lr=options["z_lr"])
        if method == "gumbel"
        else None
    )
    generator = torch.Generator().manual_seed(
        seed + (701 if method == "gumbel" else 709)
    )
    history = []

    def record(step, **extra):
        row = {
            "step": step,
            "mean_bce": float(scores.mean()),
            "best_bce": float(scores.min()),
            "restart_bce": scores.detach().cpu().tolist(),
            **fitness.budget(),
            **extra,
        }
        history.append(row)
        progress(method, history)

    record(0)
    for step in range(options["steps"]):
        previous = z.detach().clone()
        if method == "gumbel":
            temperature = interpolate(
                options["temperature"], step, options["steps"]
            )
            noise = interpolate(
                options["gumbel_noise"], step, options["steps"]
            )
            optimizer.zero_grad(set_to_none=True)
            sampled = gumbel_masks(
                decoder(z).repeat(options["gumbel_samples"], 1, 1),
                temperature,
                noise,
                generator,
            )
            loss = (
                fitness(sampled, differentiable=True)
                .reshape(options["gumbel_samples"], len(z))
                .mean(0)
            )
            loss.sum().backward()
            if z.grad is None or not torch.isfinite(z.grad).all():
                raise RuntimeError("Invalid latent gradient")
            norms = z.grad.detach().norm(dim=-1, keepdim=True)
            gradient_norm = float(norms.mean())
            with torch.no_grad():
                z.grad.mul_(
                    (
                        options["z_clip_norm"] / norms.clamp_min(1e-12)
                    ).clamp_max(1)
                )
            optimizer.step()
            direction = z.detach() - previous
            proposals = torch.stack(
                [
                    previous + scale * direction
                    for scale in options["proposal_scales"]
                ]
            )
            extra = {
                "temperature": temperature,
                "gumbel_noise": noise,
                "gradient_norm": gradient_norm,
            }
            del sampled, loss
        else:
            count = options["gumbel_samples"] + len(options["proposal_scales"])
            radius = interpolate(
                options["evolution_radius"], step, options["steps"]
            )
            directions = torch.randn(count, *z.shape, generator=generator).to(
                z.device
            )
            proposals = previous[None] + radius * directions
            extra = {"radius": radius}
        proposal_scores = hard_scores(
            decoder, proposals.flatten(0, 1), fitness, options["hard_batch"]
        ).reshape(proposals.shape[:2])
        best_scores, winners = proposal_scores.min(0)
        accepted = best_scores < scores
        chosen = proposals[winners, torch.arange(len(z), device=z.device)]
        with torch.no_grad():
            z.copy_(torch.where(accepted[:, None], chosen, previous))
        scores = torch.minimum(scores, best_scores)
        record(step + 1, accepted_restarts=int(accepted.sum()), **extra)
    if fitness.budget()["candidate_fits"] != budget(options):
        raise RuntimeError(
            "Search did not consume the declared candidate-fit budget"
        )
    selected = int(scores.argmin())
    with torch.no_grad():
        masks = canonical_mask(topk(decoder(z)))
    return {
        "z": z.detach().cpu(),
        "masks": masks.cpu(),
        "scores": scores.cpu(),
        "selected": selected,
        "initial_z": initial_z.cpu(),
        "initial_scores": initial_scores.cpu(),
        "history": history,
        "budget": fitness.budget(),
    }
