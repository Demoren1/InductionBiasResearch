"""Conditional full-map generators for the aligned DeepSets functional bank.

All models here operate directly on ``[batch, 784, 32]`` normalized-logit
maps.  They deliberately have no VAE/PCA bottleneck: a generative objective is
asked to model the entire 25,088-dimensional map distribution.

The time-conditioned vector field follows the common DDPM/flow-matching
parameterization (Ho et al., 2020, https://arxiv.org/abs/2006.11239; Lipman et
al., 2022, https://arxiv.org/abs/2210.02747).  The GAN pair uses a scalar,
unbounded critic and is compatible with the WGAN-GP objective (Gulrajani et
al., 2017, https://arxiv.org/abs/1704.00028).  HyperGAN (Ratzlaff and Fuxin,
2019, https://arxiv.org/abs/1901.11058) motivates generating complete weight
objects, but is not implemented here.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_FEATURES = 784
DEFAULT_HIDDEN = 32
DEFAULT_TASKS = 4


def parameter_count(module: nn.Module) -> int:
    """Number of trainable scalar parameters in ``module``."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _validate_map(x: Tensor, *, features: int, hidden: int) -> None:
    if x.ndim != 3 or x.shape[1:] != (features, hidden):
        raise ValueError(f"expected maps [batch, {features}, {hidden}], got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError("maps must have a floating point dtype")


def _task_indices(task: Tensor | int, *, batch: int, tasks: int, device: torch.device) -> Tensor:
    """Normalize a scalar or per-example task id to ``[batch]`` longs.

    Shape and dtype handling is always checked.  Value checks run on CPU; on
    CUDA they are intentionally skipped because reductions followed by Python
    control flow synchronize the device on every field evaluation.  Production
    callers use fixed, prevalidated task ids in ``[0, tasks)``.
    """
    value = torch.as_tensor(task, device=device)
    if value.ndim == 0:
        value = value.expand(batch)
    elif value.ndim == 1 and value.shape[0] == batch:
        pass
    else:
        raise ValueError(f"task must be scalar or [{batch}], got {tuple(value.shape)}")
    if value.is_floating_point():
        if value.device.type == "cpu" and not torch.equal(value, value.round()):
            raise ValueError("task indices must be integral")
        value = value.long()
    else:
        value = value.long()
    if value.device.type == "cpu" and value.numel() and (value.min() < 0 or value.max() >= tasks):
        raise ValueError(f"task indices must lie in [0, {tasks})")
    return value


class FourierTimeEmbedding(nn.Module):
    """Fixed sinusoidal features followed by a learned projection."""

    def __init__(self, width: int, frequencies: int = 32) -> None:
        super().__init__()
        if width < 2 or frequencies < 1:
            raise ValueError("width must be >= 2 and frequencies must be positive")
        # Geometric frequencies cover both short and long interpolation times.
        freq = torch.exp(torch.linspace(0.0, math.log(1000.0), frequencies))
        self.register_buffer("frequencies", freq, persistent=False)
        self.project = nn.Sequential(
            nn.Linear(2 * frequencies, width), nn.SiLU(), nn.Linear(width, width)
        )

    def forward(self, t: Tensor | float, *, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        value = torch.as_tensor(t, device=device, dtype=dtype)
        if value.ndim == 0:
            value = value.expand(batch)
        elif value.ndim == 1 and value.shape[0] == batch:
            pass
        elif value.ndim == 2 and value.shape == (batch, 1):
            value = value[:, 0]
        else:
            raise ValueError(f"t must be scalar, [{batch}], or [{batch}, 1], got {tuple(value.shape)}")
        # Avoid a host synchronization in each CUDA forward.  Training uses
        # finite schedule values; CPU tests retain the defensive check.
        if value.device.type == "cpu" and not torch.isfinite(value).all():
            raise ValueError("t must be finite")
        phase = value[:, None] * self.frequencies.to(dtype=dtype)[None, :] * (2.0 * math.pi)
        return self.project(torch.cat((phase.sin(), phase.cos()), dim=-1))


class _ResidualBlock(nn.Module):
    """Per-sample residual MLP block; no dropout or batch-dependent state."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear_1 = nn.Linear(width, width)
        self.linear_2 = nn.Linear(width, width)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        h = self.norm(x) + condition
        h = self.linear_1(F.silu(h))
        h = self.linear_2(F.silu(h))
        return x + h


class FlatTimeField(nn.Module):
    """Conditional vector field ``v(x, t, task)`` over full DeepSets maps.

    The same field can predict DDPM noise/velocity or the flow-matching
    velocity; the caller selects the training target and sampler.  ``x`` and
    the output retain map shape, avoiding an information-losing latent code.
    """

    def __init__(self, features: int = DEFAULT_FEATURES, hidden: int = DEFAULT_HIDDEN,
                 tasks: int = DEFAULT_TASKS, width: int = 256) -> None:
        super().__init__()
        if min(features, hidden, tasks, width) < 1:
            raise ValueError("features, hidden, tasks, and width must be positive")
        self.features, self.hidden, self.tasks, self.width = features, hidden, tasks, width
        self.flat_dim = features * hidden
        self.input = nn.Linear(self.flat_dim, width)
        self.time = FourierTimeEmbedding(width)
        self.task = nn.Embedding(tasks, width)
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList((_ResidualBlock(width), _ResidualBlock(width), _ResidualBlock(width)))
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, self.flat_dim)

    def forward(self, x: Tensor, t: Tensor | float, task: Tensor | int) -> Tensor:
        _validate_map(x, features=self.features, hidden=self.hidden)
        batch = x.shape[0]
        ids = _task_indices(task, batch=batch, tasks=self.tasks, device=x.device)
        condition = self.condition(self.time(t, batch=batch, device=x.device, dtype=x.dtype) + self.task(ids))
        h = self.input(x.reshape(batch, self.flat_dim))
        for block in self.blocks:
            h = block(h, condition)
        return self.output(F.silu(self.output_norm(h))).reshape(batch, self.features, self.hidden)


class ConditionalGenerator(nn.Module):
    """Generate an unbounded full-map logit from Gaussian noise and task id."""

    def __init__(self, features: int = DEFAULT_FEATURES, hidden: int = DEFAULT_HIDDEN,
                 tasks: int = DEFAULT_TASKS, width: int = 256, latent: int = 64) -> None:
        super().__init__()
        if min(features, hidden, tasks, width, latent) < 1:
            raise ValueError("features, hidden, tasks, width, and latent must be positive")
        self.features, self.hidden, self.tasks = features, hidden, tasks
        self.width, self.latent, self.flat_dim = width, latent, features * hidden
        self.task = nn.Embedding(tasks, width)
        self.input = nn.Linear(latent, width)
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList((_ResidualBlock(width), _ResidualBlock(width), _ResidualBlock(width)))
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, self.flat_dim)

    def forward(self, z: Tensor, task: Tensor | int) -> Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent:
            raise ValueError(f"expected z [batch, {self.latent}], got {tuple(z.shape)}")
        if not z.is_floating_point():
            raise TypeError("z must have a floating point dtype")
        batch = z.shape[0]
        ids = _task_indices(task, batch=batch, tasks=self.tasks, device=z.device)
        condition = self.condition(self.task(ids))
        h = self.input(z)
        for block in self.blocks:
            h = block(h, condition)
        # No sigmoid: normalized logit maps are real-valued and WGAN-GP needs
        # a generator with unrestricted support.
        return self.output(F.silu(self.output_norm(h))).reshape(batch, self.features, self.hidden)


class ConditionalCritic(nn.Module):
    """Unbounded WGAN-GP critic for complete maps, conditioned on task id."""

    def __init__(self, features: int = DEFAULT_FEATURES, hidden: int = DEFAULT_HIDDEN,
                 tasks: int = DEFAULT_TASKS, width: int = 256) -> None:
        super().__init__()
        if min(features, hidden, tasks, width) < 1:
            raise ValueError("features, hidden, tasks, and width must be positive")
        self.features, self.hidden, self.tasks = features, hidden, tasks
        self.width, self.flat_dim = width, features * hidden
        self.task = nn.Embedding(tasks, width)
        self.input = nn.Linear(self.flat_dim, width)
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList((_ResidualBlock(width), _ResidualBlock(width), _ResidualBlock(width)))
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 1)

    def forward(self, x: Tensor, task: Tensor | int) -> Tensor:
        _validate_map(x, features=self.features, hidden=self.hidden)
        batch = x.shape[0]
        ids = _task_indices(task, batch=batch, tasks=self.tasks, device=x.device)
        condition = self.condition(self.task(ids))
        h = self.input(x.reshape(batch, self.flat_dim))
        for block in self.blocks:
            h = block(h, condition)
        # WGAN critic score: deliberately no sigmoid or probability semantics.
        return self.output(F.silu(self.output_norm(h))).squeeze(-1)


def wgan_gradient_penalty(critic: ConditionalCritic, real: Tensor, fake: Tensor,
                          task: Tensor | int, *, target_norm: float = 1.0) -> Tensor:
    """Per-example WGAN-GP penalty with a differentiable critic-gradient norm.

    The returned scalar retains the graph needed for the critic's second
    backward pass. ``real`` and ``fake`` must have the same complete-map shape.
    """
    if real.shape != fake.shape:
        raise ValueError("real and fake must have identical shapes")
    _validate_map(real, features=critic.features, hidden=critic.hidden)
    _validate_map(fake, features=critic.features, hidden=critic.hidden)
    if target_norm <= 0:
        raise ValueError("target_norm must be positive")
    alpha = torch.rand(real.shape[0], 1, 1, device=real.device, dtype=real.dtype)
    mixed = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    scores = critic(mixed, task)
    gradients, = torch.autograd.grad(scores.sum(), mixed, create_graph=True, retain_graph=True)
    norms = gradients.reshape(real.shape[0], -1).norm(2, dim=1)
    return ((norms - target_norm) ** 2).mean()


def selfcheck() -> dict[str, Any]:
    """CPU shape, finite-value, and WGAN-GP double-backward smoke check."""
    device = torch.device("cpu")
    torch.manual_seed(17)
    batch, features, hidden, tasks, width, latent = 2, 7, 3, 4, 16, 5
    maps = torch.randn(batch, features, hidden, device=device)
    ids = torch.tensor([0, 3], device=device)
    field = FlatTimeField(features, hidden, tasks, width).to(device)
    generated = ConditionalGenerator(features, hidden, tasks, width, latent).to(device)
    critic = ConditionalCritic(features, hidden, tasks, width).to(device)

    field_out = field(maps, torch.tensor([0.1, 0.9]), ids)
    z = torch.randn(batch, latent, device=device)
    fake = generated(z, ids)
    scores = critic(fake, ids)
    penalty = wgan_gradient_penalty(critic, maps, fake.detach(), ids)
    # This is the critical WGAN-GP path: differentiating the gradient penalty
    # through critic parameters must work without batch statistics/dropout.
    critic.zero_grad(set_to_none=True)
    (scores.mean() + 10.0 * penalty).backward()
    finite = all(torch.isfinite(value).all().item() for value in (field_out, fake, scores, penalty))
    critic_grad_finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                             for parameter in critic.parameters())
    if not finite or not critic_grad_finite:
        raise AssertionError("non-finite output or gradient in generative_flat selfcheck")
    return {
        "passed": True,
        "field_shape": list(field_out.shape),
        "generator_shape": list(fake.shape),
        "critic_shape": list(scores.shape),
        "gradient_penalty": float(penalty.detach()),
        "parameters": {
            "field": parameter_count(field),
            "generator": parameter_count(generated),
            "critic": parameter_count(critic),
        },
    }


if __name__ == "__main__":
    print(selfcheck())
