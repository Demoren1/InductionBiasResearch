"""Core parameterizations and exact lower solve for learned sharing schemes.

The assignment convention follows Yeh et al. (2022): ``A[i, j] == 1`` means
that coordinate ``i`` uses shared parameter ``j``.  Thus every row of an
integral assignment matrix contains one one.  The relaxed objects returned by
this module are row-stochastic categorical distributions.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


AssignmentMode = Literal["soft", "hard", "ste"]


def assignment_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    mode: AssignmentMode = "soft",
) -> torch.Tensor:
    """Convert categorical logits to a row-stochastic assignment matrix.

    ``ste`` has an exactly one-hot forward pass and the softmax gradient in
    the backward pass.  Leading dimensions, if any, are treated as batches.
    """
    if logits.ndim < 2:
        raise ValueError("logits must have item and category dimensions")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    soft = torch.softmax(logits / temperature, dim=-1)
    if mode == "soft":
        return soft
    hard = F.one_hot(soft.argmax(dim=-1), num_classes=soft.shape[-1]).to(soft.dtype)
    if mode == "hard":
        return hard
    if mode == "ste":
        return hard + soft - soft.detach()
    raise ValueError(f"unknown assignment mode: {mode}")


class CoordinateAssignmentGenerator(nn.Module):
    """Decode assignment logits from a latent and item coordinates.

    The generator has no task-specific matrix parameters.  A batch of latents
    yields a batch of matrices, which lets a single ``G_psi`` model a family
    of structures while each task only carries its latent ``z``.
    """

    def __init__(
        self,
        n_items: int,
        n_categories: int | None = None,
        *,
        latent_dim: int = 8,
        width: int = 64,
        seed: int = 0,
        item_coordinates: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if n_items <= 0:
            raise ValueError("n_items must be positive")
        n_categories = n_items if n_categories is None else n_categories
        if n_categories <= 0 or latent_dim <= 0 or width <= 0:
            raise ValueError("n_categories, latent_dim, and width must be positive")
        self.n_items = int(n_items)
        self.n_categories = int(n_categories)
        self.latent_dim = int(latent_dim)

        # A caller can provide, for example, a two-dimensional (output,
        # input) coordinate for each item.  The default remains the original
        # normalized one-dimensional item index.
        if item_coordinates is None:
            item_coordinates = torch.linspace(-1.0, 1.0, self.n_items)[:, None]
        if item_coordinates.ndim != 2 or item_coordinates.shape[0] != self.n_items:
            raise ValueError("item_coordinates must have shape (n_items, coordinate_dim)")
        if item_coordinates.shape[1] == 0:
            raise ValueError("item_coordinates must contain at least one coordinate")
        item_coordinates = item_coordinates.detach().to(dtype=torch.float32)
        self.register_buffer("coordinates", item_coordinates)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.network = nn.Sequential(
                nn.Linear(self.latent_dim + item_coordinates.shape[-1], width),
                nn.Tanh(),
                nn.Linear(width, width),
                nn.Tanh(),
                nn.Linear(width, self.n_categories),
            )
            # A small, non-zero final layer breaks the uniform-assignment
            # symmetry without making the initial categorical choices sharp.
            nn.init.normal_(self.network[-1].weight, std=0.03)
            nn.init.zeros_(self.network[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``z.shape[:-1] + (items, categories)``."""
        if z.shape[-1] != self.latent_dim:
            raise ValueError(f"expected latent_dim={self.latent_dim}, got {z.shape[-1]}")
        prefix = z.shape[:-1]
        coord = self.coordinates.view(*(1 for _ in prefix), *self.coordinates.shape)
        coord = coord.expand(*prefix, -1, -1)
        latent = z[..., None, :].expand(*prefix, self.n_items, -1)
        return self.network(torch.cat((latent, coord), dim=-1))

    def assignment(
        self,
        z: torch.Tensor,
        *,
        temperature: float = 1.0,
        mode: AssignmentMode = "soft",
    ) -> torch.Tensor:
        return assignment_from_logits(self(z), temperature=temperature, mode=mode)


class DirectAssignment(nn.Module):
    """The paper's direct, unconstrained-logit parameterization of ``A``."""

    def __init__(
        self,
        n_items: int,
        n_categories: int | None = None,
        *,
        batch_size: int = 1,
        seed: int = 0,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        n_categories = n_items if n_categories is None else n_categories
        if min(n_items, n_categories, batch_size) <= 0 or init_scale < 0:
            raise ValueError("matrix dimensions must be positive and init_scale non-negative")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        initial = torch.empty(batch_size, n_items, n_categories)
        initial.uniform_(-init_scale, init_scale, generator=generator)
        self.logits = nn.Parameter(initial)

    def forward(self) -> torch.Tensor:
        return self.logits

    def assignment(self, *, temperature: float = 1.0, mode: AssignmentMode = "soft") -> torch.Tensor:
        return assignment_from_logits(self.logits, temperature=temperature, mode=mode)


def _labels(assignment: torch.Tensor) -> torch.Tensor:
    """Extract hard labels from either labels or an assignment matrix."""
    if assignment.ndim == 1:
        return assignment.to(torch.long)
    if assignment.ndim != 2:
        raise ValueError("partition distance expects a vector or a two-dimensional assignment")
    return assignment.argmax(dim=-1).to(torch.long)


def _maximum_weight_matching(weights: torch.Tensor) -> int:
    """Maximum-weight perfect matching (Kuhn--Munkres), for a small CPU table."""
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("the contingency table must be square")
    # This is evaluation-only and intentionally avoids a SciPy dependency in
    # the core.  Convert maximum integer weights to a minimum-cost problem.
    matrix = weights.detach().to(device="cpu", dtype=torch.float64)
    n = int(matrix.shape[0])
    max_weight = float(matrix.max()) if n else 0.0
    cost = (max_weight - matrix).tolist()
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    matching = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        matching[0] = i
        j0 = 0
        min_value = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = matching[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < min_value[j]:
                        min_value[j] = current
                        way[j] = j0
                    if min_value[j] < delta:
                        delta, j1 = min_value[j], j
            for j in range(n + 1):
                if used[j]:
                    u[matching[j]] += delta
                    v[j] -= delta
                else:
                    min_value[j] -= delta
            j0 = j1
            if matching[j0] == 0:
                break
        while True:
            j1 = way[j0]
            matching[j0] = matching[j1]
            j0 = j1
            if j0 == 0:
                break
    return int(sum(matrix[matching[j] - 1, j - 1].item() for j in range(1, n + 1)))


def partition_distance(first: torch.Tensor, second: torch.Tensor) -> int:
    """Invariant partition distance between two hard/relaxed assignments.

    Category names are immaterial: the result is the minimum number of items
    that must move after optimally relabelling the categories of either input.
    """
    left, right = _labels(first), _labels(second)
    if left.shape != right.shape:
        raise ValueError("both partitions must cover the same number of items")
    n_categories = max(int(left.max()) if left.numel() else 0, int(right.max()) if right.numel() else 0) + 1
    contingency = torch.zeros(n_categories, n_categories, dtype=torch.long)
    contingency.index_put_((left.cpu(), right.cpu()), torch.ones_like(left.cpu()), accumulate=True)
    return int(left.numel() - _maximum_weight_matching(contingency))


def solve_shared_mean(
    samples: torch.Tensor,
    assignment: torch.Tensor,
    *,
    ridge: float | None = 1e-6,
) -> torch.Tensor:
    """Fit and return the shared Gaussian mean through differentiable OLS/ridge.

    ``samples`` is ``(N, K)`` or ``(B, N, K)`` and ``assignment`` is ``(K, C)``
    or ``(B, K, C)``.  A positive ``ridge`` uses a stable normal-equation
    solve; ``None`` or zero uses the Moore--Penrose OLS solution.
    """
    if samples.ndim == 2:
        samples = samples.unsqueeze(0)
        squeeze = True
    elif samples.ndim == 3:
        squeeze = False
    else:
        raise ValueError("samples must have shape (N, K) or (B, N, K)")
    if assignment.ndim == 2:
        assignment = assignment.unsqueeze(0)
    if assignment.ndim != 3:
        raise ValueError("assignment must have shape (K, C) or (B, K, C)")
    batch, _, dimensions = samples.shape
    if assignment.shape[-2] != dimensions:
        raise ValueError("assignment item dimension must equal sample dimension")
    if assignment.shape[0] not in {1, batch}:
        raise ValueError("assignment batch must be one or equal sample batch")
    if assignment.shape[0] == 1 and batch != 1:
        assignment = assignment.expand(batch, -1, -1)
    empirical_mean = samples.mean(dim=-2)
    if ridge is None or ridge <= 0:
        coefficients = torch.matmul(torch.linalg.pinv(assignment), empirical_mean.unsqueeze(-1))
    else:
        gram = assignment.transpose(-2, -1) @ assignment
        identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        rhs = assignment.transpose(-2, -1) @ empirical_mean.unsqueeze(-1)
        coefficients = torch.linalg.solve(gram + float(ridge) * identity, rhs)
    estimate = (assignment @ coefficients).squeeze(-1)
    return estimate.squeeze(0) if squeeze else estimate


def assignment_regularizers(assignment: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return categorical entropy and the nuclear norm from Eqs. 20--21."""
    shifted = assignment + 1e-6
    entropy = -(shifted * shifted.log()).sum(dim=-1).mean(dim=-1)
    nuclear = torch.linalg.svdvals(assignment).sum(dim=-1)
    return entropy, nuclear


def release_assignment_regularizers(assignment: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the regularizers as computed by the public PyTorch release.

    The released implementation spells the second term as
    ``trace(sqrt(A.T @ A))``.  Because ``sqrt`` is elementwise in PyTorch,
    this is the sum of column L2 norms, not the matrix nuclear norm.
    """
    shifted = assignment + 1e-6
    entropy = -(shifted * shifted.log()).sum(dim=-1).mean(dim=-1)
    column_group = assignment.square().sum(dim=-2).clamp_min(1e-24).sqrt().sum(dim=-1)
    return entropy, column_group
