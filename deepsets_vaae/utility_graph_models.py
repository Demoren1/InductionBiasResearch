"""Conditional, hidden-permutation-equivariant fields for whole-mask utility.

The field works on labelled feature positions and exchangeable hidden neurons.
Node and edge contexts carry the source-bank functional summaries; the task
condition is continuous and contains no task ID embedding. Edge updates use
factorized projections, so node features are not copied into a wide tensor at
every one of the ``features * hidden`` edges.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _mlp(in_features: int, out_features: int, hidden_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.SiLU(),
        nn.Linear(hidden_features, out_features),
    )


class _FactorizedBipartiteBlock(nn.Module):
    """Edge-to-node-to-edge update with no broadcast node concatenations."""

    def __init__(self, width: int, edge_width: int) -> None:
        super().__init__()
        self.feature_norm = nn.LayerNorm(width)
        self.hidden_norm = nn.LayerNorm(width)
        self.edge_norm = nn.LayerNorm(edge_width)

        self.edge_from_edge = nn.Linear(edge_width, edge_width, bias=False)
        self.edge_from_feature = nn.Linear(width, edge_width, bias=False)
        self.edge_from_hidden = nn.Linear(width, edge_width, bias=False)
        self.edge_from_state = nn.Linear(1, edge_width, bias=False)
        self.edge_from_global = nn.Linear(width, edge_width, bias=False)
        self.edge_delta = _mlp(edge_width, edge_width, 2 * edge_width)

        self.feature_message = _mlp(edge_width, width, width)
        self.hidden_message = _mlp(edge_width, width, width)
        self.feature_from_node = nn.Linear(width, width, bias=False)
        self.feature_from_message = nn.Linear(width, width, bias=False)
        self.feature_from_global = nn.Linear(width, width, bias=False)
        self.hidden_from_node = nn.Linear(width, width, bias=False)
        self.hidden_from_message = nn.Linear(width, width, bias=False)
        self.hidden_from_global = nn.Linear(width, width, bias=False)
        self.feature_delta = _mlp(width, width, 2 * width)
        self.hidden_delta = _mlp(width, width, 2 * width)

    def forward(
        self, feature: Tensor, hidden: Tensor, edge: Tensor, state: Tensor, condition: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        f = self.feature_norm(feature)
        h = self.hidden_norm(hidden)
        e = self.edge_norm(edge)
        edge_pre = (
            self.edge_from_edge(e)
            + self.edge_from_feature(f)[:, :, None, :]
            + self.edge_from_hidden(h)[:, None, :, :]
            + self.edge_from_state(state[..., None])
            + self.edge_from_global(condition)[:, None, None, :]
        )
        edge = edge + self.edge_delta(F.silu(edge_pre))

        # Mean aggregation is symmetric over the opposite side of the graph.
        feature_message = self.feature_message(edge).mean(dim=2)
        hidden_message = self.hidden_message(edge).mean(dim=1)
        feature_pre = (
            self.feature_from_node(f)
            + self.feature_from_message(feature_message)
            + self.feature_from_global(condition)[:, None, :]
        )
        hidden_pre = (
            self.hidden_from_node(h)
            + self.hidden_from_message(hidden_message)
            + self.hidden_from_global(condition)[:, None, :]
        )
        feature = feature + self.feature_delta(F.silu(feature_pre))
        hidden = hidden + self.hidden_delta(F.silu(hidden_pre))
        return feature, hidden, edge


class UtilityGraphField(nn.Module):
    """Predict per-edge mask logits or a velocity field.

    Args:
        node_dim: Width of ``node_context[B, hidden, node_dim]``.
        edge_dim: Width of ``edge_context[B, features, hidden, edge_dim]``.
        task_dim: Width of the continuous task summary ``task_context[B, task_dim]``.
        features: Number of labelled input positions (784 for the DeepSets bank).
        hidden: Number of exchangeable hidden units (32 for the DeepSets bank).
        width: Internal node/global width.
        edge_width: Compact edge width used at every bipartite edge.
        depth: Number of edge-updating message-passing blocks.

    There are learned identities for fixed feature positions and no identities
    for hidden units. The scalar state is added directly to the output, keeping
    a full-coordinate residual route even though edge messages use a compact
    width. Permuting the hidden axis jointly in state and contexts permutes the
    output by the same amount.
    """

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        task_dim: int,
        features: int = 784,
        hidden: int = 32,
        width: int = 32,
        edge_width: int = 8,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if min(node_dim, edge_dim, task_dim, features, hidden, width, edge_width, depth) < 1:
            raise ValueError("all dimensions and depth must be positive")
        self.features = features
        self.hidden = hidden
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.task_dim = task_dim
        self.width = width
        self.edge_width = edge_width

        self.task_encoder = _mlp(task_dim, width, width)
        self.time_encoder = _mlp(3, width, width)
        self.feature_embedding = nn.Parameter(torch.empty(features, width))
        nn.init.normal_(self.feature_embedding, std=width ** -0.5)
        self.node_encoder = nn.Sequential(nn.Linear(node_dim, width), nn.SiLU(), nn.LayerNorm(width))
        self.edge_encoder = nn.Linear(edge_dim, edge_width)
        self.state_encoder = nn.Linear(1, edge_width)
        self.global_edge_encoder = nn.Linear(width, edge_width, bias=False)
        self.blocks = nn.ModuleList(
            [_FactorizedBipartiteBlock(width, edge_width) for _ in range(depth)]
        )
        self.readout_feature = nn.Linear(width, edge_width, bias=False)
        self.readout_hidden = nn.Linear(width, edge_width, bias=False)
        self.readout_global = nn.Linear(width, edge_width, bias=False)
        self.readout_state = nn.Linear(1, edge_width, bias=False)
        self.readout = _mlp(edge_width, 1, 2 * edge_width)
        self.residual_scale = nn.Parameter(torch.ones(()))

    def forward(
        self,
        state: Tensor,
        time: Tensor,
        node_context: Tensor,
        edge_context: Tensor,
        task_context: Tensor,
    ) -> Tensor:
        if state.ndim != 3 or state.shape[1:] != (self.features, self.hidden):
            raise ValueError(f"state must have shape [B, {self.features}, {self.hidden}]")
        batch = state.shape[0]
        if time.shape != (batch,):
            raise ValueError("time must have shape [B]")
        if node_context.shape != (batch, self.hidden, self.node_dim):
            raise ValueError(f"node_context must have shape [B, {self.hidden}, {self.node_dim}]")
        if edge_context.shape != (batch, self.features, self.hidden, self.edge_dim):
            raise ValueError(
                f"edge_context must have shape [B, {self.features}, {self.hidden}, {self.edge_dim}]"
            )
        if task_context.shape != (batch, self.task_dim):
            raise ValueError(f"task_context must have shape [B, {self.task_dim}]")

        time_features = torch.stack(
            (time, torch.sin(2.0 * math.pi * time), torch.cos(2.0 * math.pi * time)), dim=-1
        )
        condition = self.task_encoder(task_context) + self.time_encoder(time_features)
        feature = self.feature_embedding[None, :, :] + condition[:, None, :]
        hidden = self.node_encoder(node_context) + condition[:, None, :]
        edge = (
            self.edge_encoder(edge_context)
            + self.state_encoder(state[..., None])
            + self.global_edge_encoder(condition)[:, None, None, :]
        )
        for block in self.blocks:
            feature, hidden, edge = block(feature, hidden, edge, state, condition)
        readout_features = (
            self.readout_feature(feature)[:, :, None, :]
            + self.readout_hidden(hidden)[:, None, :, :]
            + self.readout_global(condition)[:, None, None, :]
            + self.readout_state(state[..., None])
            + edge
        )
        return self.residual_scale * state + self.readout(readout_features).squeeze(-1)


def exact_topk(scores: Tensor, k: int) -> Tensor:
    """Return binary masks with exactly ``k`` active entries per example.

    For rank-3 inputs, K is global across the final ``[features, hidden]`` map.
    Tied cutoff scores are resolved by ``torch.topk``'s index choice; exact-K
    and additive-shift invariance still hold, but equivariance at a tied cutoff
    is not promised.
    """
    if scores.ndim < 1:
        raise ValueError("scores must have at least one dimension")
    if scores.ndim == 1:
        flat = scores.reshape(1, -1)
    else:
        flat = scores.reshape(scores.shape[0], -1)
    if not 0 < k <= flat.shape[-1]:
        raise ValueError(f"k must be in [1, {flat.shape[-1]}]")
    indices = flat.topk(k, dim=-1).indices
    mask = torch.zeros_like(flat).scatter_(-1, indices, 1.0)
    return mask.reshape_as(scores)


def mask_to_flow_endpoint(mask: Tensor) -> Tensor:
    """Map binary ``{0,1}`` masks to the FM endpoint convention ``{-1,+1}``."""
    if not mask.is_floating_point():
        mask = mask.to(dtype=torch.float32)
    return mask.mul(2.0).sub(1.0)


def flow_matching_losses(
    model: UtilityGraphField,
    endpoint: Tensor,
    node_context: Tensor,
    edge_context: Tensor,
    task_context: Tensor,
    *,
    noise: Tensor | None = None,
    time: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Independent-coupling FM MSE, returned once per example for weighting."""
    if endpoint.ndim != 3:
        raise ValueError("endpoint must have shape [B, features, hidden]")
    batch = endpoint.shape[0]
    if noise is None:
        noise = torch.randn(endpoint.shape, dtype=endpoint.dtype, device=endpoint.device,
                            generator=generator)
    elif noise.shape != endpoint.shape:
        raise ValueError("noise and endpoint shapes must match")
    if time is None:
        time = torch.rand((batch,), dtype=endpoint.dtype, device=endpoint.device,
                          generator=generator)
    elif time.shape != (batch,):
        raise ValueError("time must have shape [B]")
    path_time = time[:, None, None]
    state = noise * (1.0 - path_time) + endpoint * path_time
    target_velocity = endpoint - noise
    prediction = model(state, time, node_context, edge_context, task_context)
    return (prediction - target_velocity).square().flatten(1).mean(dim=1)


def flow_matching_loss(
    model: UtilityGraphField,
    endpoint: Tensor,
    node_context: Tensor,
    edge_context: Tensor,
    task_context: Tensor,
    **kwargs,
) -> Tensor:
    """Mean independent-coupling FM loss across examples."""
    return flow_matching_losses(
        model, endpoint, node_context, edge_context, task_context, **kwargs
    ).mean()


@torch.no_grad()
def sample_flow(
    model: UtilityGraphField,
    node_context: Tensor,
    edge_context: Tensor,
    task_context: Tensor,
    *,
    steps: int = 12,
    method: Literal["euler", "heun"] = "euler",
    initial_noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Integrate the learned field from independent Gaussian noise to time 1."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if method not in ("euler", "heun"):
        raise ValueError("method must be 'euler' or 'heun'")
    batch = node_context.shape[0]
    expected_shape = (batch, model.features, model.hidden)
    if initial_noise is None:
        state = torch.randn(expected_shape, dtype=node_context.dtype, device=node_context.device,
                            generator=generator)
    else:
        if initial_noise.shape != expected_shape:
            raise ValueError(f"initial_noise must have shape {expected_shape}")
        state = initial_noise.clone()
    dt = 1.0 / steps
    for step in range(steps):
        time = torch.full((batch,), step * dt, dtype=state.dtype, device=state.device)
        velocity = model(state, time, node_context, edge_context, task_context)
        proposal = state + dt * velocity
        if method == "heun":
            next_time = torch.full((batch,), (step + 1) * dt, dtype=state.dtype, device=state.device)
            next_velocity = model(proposal, next_time, node_context, edge_context, task_context)
            state = state + 0.5 * dt * (velocity + next_velocity)
        else:
            state = proposal
    return state
