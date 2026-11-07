"""Strong known-objective source extractors for G23's common-I protocol."""

import math
import time

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.codebook_projection import (
    exact_signed_projection,
)

from .shared_structure_numeric import fit_codes


def prediction(x, weight, plan):
    return (x @ weight + plan["bias"]).relu().sum(-1) / math.sqrt(plan["hidden"])


def project(weight, plan):
    label, center, quantized = exact_signed_projection(
        weight[None], plan["codes"], weight.new_zeros(1), weight.new_ones(1)
    )
    label = torch.where(label > plan["codes"], plan["codes"] - label, label)[0]
    return label, center[0], quantized[0]


@torch.no_grad()
def dense_loss(x, y, weight, plan, point_weights=None):
    value = weight.new_zeros(())
    normalizer = len(y) if point_weights is None else point_weights.sum()
    for begin in range(0, len(y), plan["numeric_chunk"]):
        end = min(len(y), begin + plan["numeric_chunk"])
        residual = prediction(x[begin:end], weight, plan) - y[begin:end]
        value += (
            residual.square().sum()
            if point_weights is None
            else (residual.square() * point_weights[begin:end]).sum()
        )
    return value / normalizer


@torch.no_grad()
def information(x, y, weight, plan, point_weights=None):
    entries = weight.numel()
    gram, score, loss = (
        weight.new_zeros((entries, entries)),
        weight.new_zeros(entries),
        weight.new_zeros(()),
    )
    normalizer = len(y) if point_weights is None else point_weights.sum()
    for begin in range(0, len(y), plan["numeric_chunk"]):
        end = min(len(y), begin + plan["numeric_chunk"])
        local = x[begin:end]
        activation = local @ weight + plan["bias"]
        jacobian = (local[:, :, None] * (activation > 0)[:, None, :]).flatten(
            1
        ) / math.sqrt(plan["hidden"])
        residual = y[begin:end] - activation.relu().sum(-1) / math.sqrt(plan["hidden"])
        weights = (
            weight.new_ones(end - begin)
            if point_weights is None
            else point_weights[begin:end]
        )
        gram += jacobian.T @ (jacobian * weights[:, None])
        score += jacobian.T @ (residual * weights)
        loss += (residual.square() * weights).sum()
    return gram / normalizer, score / normalizer, loss / (2 * normalizer)


@torch.no_grad()
def gn_step(x, y, weight, plan, ridge, point_weights=None, *, mu=0.0, target=None):
    gram, score, loss = information(x, y, weight, plan, point_weights)
    scale = torch.linalg.eigvalsh(gram)[-1].clamp_min(1e-12)
    matrix = gram + (scale * ridge + mu) * torch.eye(
        weight.numel(), device=weight.device, dtype=weight.dtype
    )
    objective = loss
    if mu:
        difference = weight.flatten() - target
        score -= mu * difference
        objective = loss + mu * difference.square().sum() / 2
    direction = torch.linalg.solve(matrix, score).reshape_as(weight)
    if not torch.isfinite(direction).all():
        raise FloatingPointError("Nonfinite native GN direction")
    for exponent in range(plan["numeric_line_search"]):
        candidate = weight + 2.0**-exponent * direction
        proposed = dense_loss(x, y, candidate, plan, point_weights) / 2
        if mu:
            proposed += mu * (candidate.flatten() - target).square().sum() / 2
        if proposed < objective:
            return candidate, {
                "accepted": True,
                "fraction": 2.0**-exponent,
                "half_mse_before": float(loss),
                "objective_after": float(proposed),
            }
    return weight, {"accepted": False, "half_mse_before": float(loss)}


def train(method, setting, restart, family, task, data, plan):
    started = time.monotonic()
    x, y = data["train_x"], data["train_y"]
    vx, vy = data["fit_validation_x"], data["fit_validation_y"]
    generator = torch.Generator(device=x.device).manual_seed(
        plan["seed"] + family * 71 + task * 1103 + restart * 100003
    )
    weight = torch.randn(
        (plan["inputs"], plan["hidden"]),
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    ) / math.sqrt(plan["inputs"])
    point_weights = (
        data["train_repeats"] if setting["loss_weights"] == "physical" else None
    )
    indices, latent, fitted, history = [], [], [], []
    l_solves = 0

    @torch.no_grad()
    def checkpoint(step, index, centers, current, raw, numerical=None):
        code, polishing = fit_codes(
            index,
            x,
            y,
            vx,
            vy,
            centers,
            plan,
            weights=point_weights,
            max_steps=plan["source_polish_steps"],
        )
        raw_mse = dense_loss(x, y, raw, plan, point_weights)
        indices.append(index.cpu().numpy().astype(np.int16))
        latent.append(current.detach().cpu().numpy())
        fitted.append(code.cpu().numpy())
        history.append(
            {
                "step": step,
                "l_solves": l_solves,
                "training_mse_latent": float(
                    dense_loss(x, y, current.detach(), plan, point_weights)
                ),
                "training_mse_raw_hard": float(raw_mse),
                "validation_mse": polishing["validation_mse"],
                "polish": polishing,
                "numerical": numerical,
                "seconds": time.monotonic() - started,
            }
        )
        print(
            f"native {method} step={step} val={polishing['validation_mse']:.6g} seconds={time.monotonic() - started:.2f}",
            flush=True,
        )

    if method in ("lc", "dense_dp"):
        with torch.no_grad():
            gram, _, _ = information(x, y, weight, plan, point_weights)
            scale = float(torch.linalg.eigvalsh(gram)[-1])
            index, centers, quantized = project(weight, plan)
            checkpoint(0, index, centers, weight, quantized)
            dual = weight.new_zeros(weight.numel())
            positive = scale * np.geomspace(
                setting.get("mu_initial", 1e-6),
                10.0,
                plan["native_stages"] - plan["native_warmup_stages"],
            )
            mus = (
                np.r_[np.zeros(plan["native_warmup_stages"]), positive]
                if method == "lc"
                else np.zeros(plan["native_stages"])
            )
            previous_mu = 0.0
            for stage, mu in enumerate(mus, 1):
                if method == "lc":
                    dual *= previous_mu / mu if previous_mu > 0 and mu > 0 else 0.0
                target = quantized.flatten() - dual
                accepted = 0
                for _ in range(plan["native_l_steps"]):
                    weight, numerical = gn_step(
                        x,
                        y,
                        weight,
                        plan,
                        setting.get("ridge", 1e-10),
                        point_weights,
                        mu=float(mu),
                        target=target,
                    )
                    l_solves += 1
                    accepted += numerical["accepted"]
                    if not numerical["accepted"]:
                        break
                index, centers, quantized = project(
                    (weight.flatten() + dual).reshape_as(weight), plan
                )
                if method == "lc" and mu > 0:
                    dual += weight.flatten() - quantized.flatten()
                previous_mu = mu
                checkpoint(
                    stage * plan["native_l_steps"],
                    index,
                    centers,
                    weight,
                    quantized,
                    numerical,
                )
                if method == "dense_dp" and not accepted:
                    break
    else:
        with torch.no_grad():
            for _ in range(plan["gradient_warmup_solves"]):
                weight, numerical = gn_step(x, y, weight, plan, 1e-10, point_weights)
                l_solves += 1
                if not numerical["accepted"]:
                    break
            index, centers, quantized = project(weight, plan)
        if method == "lutq":
            state = weight.clone().requires_grad_()
            parameters = [state]
        else:
            dictionary = torch.cat((centers[:1] * 0, centers, -centers))
            spread = weight.std().clamp_min(0.05)
            state = (
                (-(weight[..., None] - dictionary).square() / (0.05 * spread.square()))
                .detach()
                .requires_grad_()
            )
            centers = centers.detach().requires_grad_()
            parameters = [state, centers]
        optimizer = torch.optim.Adam(parameters, lr=setting["lr"])
        for step in range(plan["gradient_steps"] + 1):
            if method == "lutq":
                with torch.no_grad():
                    dictionary = torch.cat((centers[:1] * 0, centers, -centers))
                    label = (state.detach()[..., None] - dictionary).abs().argmin(-1)
                    index = torch.where(
                        label > plan["codes"], plan["codes"] - label, label
                    )
                    for code_id in range(1, plan["codes"] + 1):
                        active = index.abs() == code_id
                        if active.any():
                            centers[code_id - 1] = (
                                state.detach()[active] * index.sign()[active]
                            ).mean()
                    dictionary = torch.cat((centers[:1] * 0, centers, -centers))
                    label = (state.detach()[..., None] - dictionary).abs().argmin(-1)
                    index = torch.where(
                        label > plan["codes"], plan["codes"] - label, label
                    )
                    raw = dictionary[label]
                effective = state + (raw - state).detach()
                current = state
                entropy = None
            else:
                cooling = max(0.0, (step / plan["gradient_steps"] - 0.3) / 0.7)
                temperature = plan["minimum_temperature"] ** cooling
                probability = (state / temperature).softmax(-1)
                dictionary = torch.cat((centers[:1] * 0, centers, -centers))
                effective = (probability * dictionary).sum(-1)
                label = probability.detach().argmax(-1)
                index = torch.where(label > plan["codes"], plan["codes"] - label, label)
                raw, current = dictionary[label].detach(), effective
                entropy = (
                    -(probability * probability.clamp_min(1e-30).log()).sum(-1).mean()
                )
            if (
                step % plan["gradient_checkpoint"] == 0
                or step == plan["gradient_steps"]
            ):
                checkpoint(step, index.detach(), centers.detach(), current, raw)
            if step == plan["gradient_steps"]:
                break
            rows = torch.randint(
                len(x), (plan["gradient_batch"],), device=x.device, generator=generator
            )
            loss = (prediction(x[rows], effective, plan) - y[rows]).square()
            if point_weights is not None:
                loss *= point_weights[rows] / point_weights.mean()
            objective = loss.mean()
            if entropy is not None:
                objective += plan["categorical_entropy"] * entropy
                if setting["nuclear"]:
                    objective += (
                        setting["nuclear"]
                        * torch.linalg.svdvals(probability.flatten(0, 1)).sum()
                        / math.sqrt(weight.numel())
                    )
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            if method != "lutq":
                with torch.no_grad():
                    state.sub_(state.mean(-1, keepdim=True))
                    centers.clamp_(-plan["axis_code_max"], plan["axis_code_max"])
    winner = int(np.argmin([record["validation_mse"] for record in history]))
    return {
        "index": indices[winner],
        "centers": fitted[winner],
        "selected_checkpoint": winner,
        "validation_mse": history[winner]["validation_mse"],
        "trace_indices": np.stack(indices),
        "trace_latent_weights": np.stack(latent),
        "trace_fitted_C": np.stack(fitted),
        "history": history,
        "l_solves": l_solves,
        "seconds": time.monotonic() - started,
    }
