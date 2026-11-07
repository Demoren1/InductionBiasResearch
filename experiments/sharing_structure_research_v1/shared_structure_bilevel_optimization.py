"""Shared categorical T/V hypergradients for the nonlinear Yeh control."""

import math

import torch
from torch.nn import functional as F


class ScenarioAdam:
    """Adam with independent moments and a learning rate per batch scenario."""

    def __init__(self, parameter, rates):
        self.first = torch.zeros_like(parameter)
        self.second = torch.zeros_like(parameter)
        self.rates = torch.as_tensor(
            rates, device=parameter.device, dtype=parameter.dtype
        ).reshape(-1, *([1] * (parameter.ndim - 1)))
        self.steps = 0

    @torch.no_grad()
    def step(self, parameter, gradient):
        self.steps += 1
        self.first.lerp_(gradient, 0.1)
        self.second.lerp_(gradient.square(), 0.001)
        numerator = self.first / (1 - 0.9**self.steps)
        denominator = (self.second / (1 - 0.999**self.steps)).sqrt() + 1e-8
        parameter.add_(-self.rates * numerator / denominator)


def signed_basis(probability, codes):
    return probability[..., 1 : codes + 1] - probability[..., codes + 1 :]


def logits_gradient(probability, basis_gradient):
    probability_gradient = torch.cat(
        (torch.zeros_like(basis_gradient[..., :1]), basis_gradient, -basis_gradient),
        dim=-1,
    )
    return probability * (
        probability_gradient
        - (probability * probability_gradient).sum(-1, keepdim=True)
    )


def index_features(x, index, codes):
    basis = F.one_hot(index.abs().long(), codes + 1)[..., 1:].to(x.dtype)
    basis *= index.sign()[..., None]
    return torch.einsum("nd,sdhk->snhk", x, basis)


def prediction_and_gate(feature, code, plan):
    activation = torch.einsum("snhk,stk->stnh", feature, code) + plan["bias"]
    return (
        activation.relu().sum(-1) / math.sqrt(plan["hidden"]),
        (activation > 0).to(code.dtype),
    )


@torch.no_grad()
def code_mse(feature, code, y, plan):
    total = code.new_zeros(code.shape[:2])
    for begin in range(0, y.shape[-1], plan["numeric_chunk"]):
        end = min(y.shape[-1], begin + plan["numeric_chunk"])
        prediction, _ = prediction_and_gate(feature[:, begin:end], code, plan)
        total += (prediction - y[None, :, begin:end]).square().sum(-1)
    return total / y.shape[-1]


@torch.no_grad()
def code_statistics(feature, code, y, plan):
    scenarios, tasks, codes = code.shape
    gram = code.new_zeros((scenarios, tasks, codes, codes))
    score, mse = torch.zeros_like(code), code.new_zeros((scenarios, tasks))
    for begin in range(0, y.shape[-1], plan["numeric_chunk"]):
        end = min(y.shape[-1], begin + plan["numeric_chunk"])
        local = feature[:, begin:end]
        prediction, gate = prediction_and_gate(local, code, plan)
        jacobian = torch.einsum("snhk,stnh->stnk", local, gate) / math.sqrt(
            plan["hidden"]
        )
        residual = prediction - y[None, :, begin:end]
        gram += torch.einsum("stnk,stnl->stkl", jacobian, jacobian)
        score += torch.einsum("stnk,stn->stk", jacobian, residual)
        mse += residual.square().sum(-1)
    points = y.shape[-1]
    return gram / points, score / points, mse / points


@torch.no_grad()
def train_codes(feature, code, y, scale, optimizer, generator, plan):
    tasks, points = y.shape
    for _ in range(plan["bilevel_inner_steps"]):
        rows = torch.randint(
            points, (plan["gradient_batch"],), device=y.device, generator=generator
        )
        local = feature[:, rows]
        prediction, gate = prediction_and_gate(local, code, plan)
        jacobian = torch.einsum("snhk,stnh->stnk", local, gate) / math.sqrt(
            plan["hidden"]
        )
        gradient = torch.einsum(
            "stnk,stn->stk", jacobian, prediction - y[None, :, rows]
        ) / (tasks * len(rows) * scale)
        gradient += plan["bilevel_ridge"] / tasks * code
        optimizer.step(code, gradient)


@torch.no_grad()
def validation_gradient(probability, code, x, y, scale, plan):
    tasks, points = y.shape
    basis = signed_basis(probability, plan["codes"])
    basis_gradient, code_gradient = torch.zeros_like(basis), torch.zeros_like(code)
    mse = code.new_zeros(code.shape[:2])
    for begin in range(0, points, plan["numeric_chunk"]):
        end = min(points, begin + plan["numeric_chunk"])
        local = torch.einsum("nd,sdhk->snhk", x[begin:end], basis)
        prediction, gate = prediction_and_gate(local, code, plan)
        residual = prediction - y[None, :, begin:end]
        jacobian = torch.einsum("snhk,stnh->stnk", local, gate) / math.sqrt(
            plan["hidden"]
        )
        code_gradient += torch.einsum("stnk,stn->stk", jacobian, residual)
        coefficient = torch.einsum("stnh,stk->snhk", gate * residual[..., None], code)
        basis_gradient += torch.einsum("nd,snhk->sdhk", x[begin:end], coefficient)
        mse += residual.square().sum(-1)
    code_gradient /= tasks * points * scale
    basis_gradient /= tasks * points * scale * math.sqrt(plan["hidden"])
    return logits_gradient(probability, basis_gradient), code_gradient, mse / points


@torch.no_grad()
def mixed_gradient(probability, code, feature, x, y, inverse, scale, plan):
    tasks, points = y.shape
    gradient = torch.zeros_like(signed_basis(probability, plan["codes"]))
    for begin in range(0, points, plan["numeric_chunk"]):
        end = min(points, begin + plan["numeric_chunk"])
        local = feature[:, begin:end]
        prediction, gate = prediction_and_gate(local, code, plan)
        residual = prediction - y[None, :, begin:end]
        jv = torch.einsum("snhk,stk,stnh->stn", local, inverse, gate) / math.sqrt(
            plan["hidden"]
        )
        coefficient = torch.einsum("stnh,stk->snhk", gate * jv[..., None], code)
        coefficient += torch.einsum(
            "stnh,stk->snhk", gate * residual[..., None], inverse
        )
        gradient += torch.einsum("nd,snhk->sdhk", x[begin:end], coefficient)
    gradient /= tasks * points * scale * math.sqrt(plan["hidden"])
    return logits_gradient(probability, gradient)


def regularizer_gradient(logits, plan):
    leaf = logits.detach().requires_grad_()
    probability = leaf.softmax(-1)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1).mean((1, 2))
    nuclear = torch.linalg.svdvals(probability.flatten(1, 2)).sum(-1) / math.sqrt(
        plan["inputs"] * plan["hidden"]
    )
    objective = plan["bilevel_entropy"] * entropy + plan["bilevel_nuclear"] * nuclear
    gradient = torch.autograd.grad(objective.sum(), leaf)[0]
    return gradient.detach(), entropy.detach(), nuclear.detach()


@torch.no_grad()
def inverse_actions(hessian, gradient, rates, modes, iterations):
    direct = torch.linalg.solve(hessian, gradient[..., None])[..., 0]
    eta = rates[:, None, None]
    vector, literal = gradient.clone(), gradient.clone()
    for _ in range(iterations):
        vector -= eta * torch.einsum("stkl,stl->stk", hessian, vector)
        literal += vector
    actual = torch.where(modes[:, None, None], direct, literal)

    def residual(value):
        difference = torch.einsum("stkl,stl->stk", hessian, value) - gradient
        return difference.norm(dim=-1) / gradient.norm(dim=-1).clamp_min(1e-30)

    return actual, {
        "actual_inverse_residual_max_task": residual(actual).amax(1),
        "direct_inverse_residual_max_task": residual(direct).amax(1),
        "literal_inverse_residual_max_task": residual(literal).amax(1),
        "scaled_neumann_inverse_residual_max_task": residual(eta * literal).amax(1),
    }


@torch.no_grad()
def polish_indices(index, code, x, y, vx, vy, plan):
    """Batched form of G23's fixed-I, validation-selected warm-C GN polish."""
    feature = index_features(x, index, plan["codes"])
    validation_feature = index_features(vx, index, plan["codes"])
    code = code.clone().clamp(-plan["axis_code_max"], plan["axis_code_max"])
    best_score = code.new_full(code.shape[:2], torch.inf)
    best_code = code.clone()
    best_step = torch.zeros(code.shape[:2], device=code.device, dtype=torch.long)
    alive = torch.ones(code.shape[:2], device=code.device, dtype=torch.bool)
    identity = torch.eye(plan["codes"], device=code.device, dtype=code.dtype)
    for step in range(plan["source_polish_steps"] + 1):
        validation = code_mse(validation_feature, code, vy, plan)
        better = validation < best_score
        best_score[better], best_code[better], best_step[better] = (
            validation[better],
            code[better],
            step,
        )
        if step == plan["source_polish_steps"] or not alive.any():
            break
        gram, score, train = code_statistics(feature, code, y, plan)
        scale = torch.linalg.eigvalsh(gram)[..., -1].clamp_min(1e-12)
        direction = torch.linalg.solve(
            gram + plan["numeric_ridge"] * scale[..., None, None] * identity,
            -score[..., None],
        )[..., 0]
        accepted, proposed = ~alive, code.clone()
        for exponent in range(plan["numeric_line_search"]):
            candidate = code + 2.0**-exponent * direction
            feasible = candidate.abs().amax(-1) <= plan["axis_code_max"]
            value = code_mse(feature, candidate, y, plan)
            take = (~accepted) & feasible & (value < train)
            proposed[take] = candidate[take]
            accepted |= take
            if accepted.all():
                break
        alive &= accepted
        code = proposed
    if not torch.isfinite(best_score).all() or not torch.isfinite(best_code).all():
        raise FloatingPointError("Nonfinite fixed-I checkpoint polish")
    return best_code, best_score, best_step


def derivative_diagnostic(logits, code, x, y, scale, plan):
    """Measure local derivatives on observed data, without teacher or meta Y."""
    logits = logits[:1].detach().requires_grad_()
    code = code[:1].detach().requires_grad_()
    x, y = x[:128], y[:, :128]
    probability = logits.softmax(-1)
    feature = torch.einsum("nd,sdhk->snhk", x, signed_basis(probability, plan["codes"]))
    prediction, _ = prediction_and_gate(feature, code, plan)
    tasks = y.shape[0]
    loss = (prediction - y[None]).square().sum() / (2 * tasks * y.shape[1] * scale)
    loss = loss + plan["bilevel_ridge"] / (2 * tasks) * code.square().sum()
    auto_score = torch.autograd.grad(loss, code, create_graph=True)[0]
    inverse = torch.linspace(
        -0.7, 0.9, code.numel(), device=code.device, dtype=code.dtype
    ).reshape_as(code)
    auto_mixed = torch.autograd.grad((auto_score * inverse).sum(), logits)[0]

    def code_objective(value):
        fitted, _ = prediction_and_gate(feature.detach(), value.reshape_as(code), plan)
        return (fitted - y[None]).square().mean() / (2 * scale) + plan[
            "bilevel_ridge"
        ] / (2 * tasks) * value.square().sum()

    auto_hessian = torch.autograd.functional.hessian(code_objective, code.flatten())
    gram, score, _ = code_statistics(feature.detach(), code.detach(), y, plan)
    identity = torch.eye(plan["codes"], device=code.device, dtype=code.dtype)
    blocks = gram[0] / (tasks * scale) + plan["bilevel_ridge"] / tasks * identity
    hessian = torch.block_diag(*blocks)
    manual_score = (
        score / (tasks * scale) + plan["bilevel_ridge"] / tasks * code.detach()
    )
    manual_mixed = mixed_gradient(
        probability.detach(),
        code.detach(),
        feature.detach(),
        x,
        y,
        inverse.detach(),
        scale,
        plan,
    )
    direct, direct_code, _ = validation_gradient(
        probability.detach(), code.detach(), x, y, scale, plan
    )
    leaf_prediction, _ = prediction_and_gate(
        torch.einsum(
            "nd,sdhk->snhk",
            x,
            signed_basis(logits.softmax(-1), plan["codes"]),
        ),
        code,
        plan,
    )
    auto_direct, auto_direct_code = torch.autograd.grad(
        (leaf_prediction - y[None]).square().mean() / (2 * scale), (logits, code)
    )
    values = {
        "H_relative_error": float(
            (hessian - auto_hessian).norm() / auto_hessian.norm().clamp_min(1e-30)
        ),
        "C_gradient_relative_error": float(
            (manual_score - auto_score.detach()).norm()
            / auto_score.detach().norm().clamp_min(1e-30)
        ),
        "mixed_gradient_relative_error": float(
            (manual_mixed - auto_mixed).norm() / auto_mixed.norm().clamp_min(1e-30)
        ),
        "direct_P_gradient_relative_error": float(
            (direct - auto_direct).norm() / auto_direct.norm().clamp_min(1e-30)
        ),
        "direct_C_gradient_relative_error": float(
            (direct_code - auto_direct_code).norm()
            / auto_direct_code.norm().clamp_min(1e-30)
        ),
        "observed_points": len(x),
        "tasks": tasks,
        "teacher_or_meta_Y_opened": False,
    }
    if any(
        not math.isfinite(value) or value > 1e-8
        for name, value in values.items()
        if name.endswith("relative_error")
    ):
        raise ArithmeticError(f"Local derivative diagnostic failed: {values}")
    return values
