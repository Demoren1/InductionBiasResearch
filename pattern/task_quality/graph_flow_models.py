"""Compact permutation-equivariant graph models for edge-mask proposals.

The model scores the 11 x 8 input-to-hidden edges directly. Input positions
have learned positional embeddings; hidden vertices are represented only by
their supplied functional context and are processed with shared weights. The
same deterministic network can score masks at zero state or predict a flow
velocity at arbitrary states and times.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from .core import ACTIVE_EDGES, HIDDEN, SEQ_LEN


class _MessageBlock(nn.Module):
    """One edge-updating message-passing block with mean vertex aggregation."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.input_update = nn.Sequential(
            nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, width)
        )
        self.hidden_update = nn.Sequential(
            nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, width)
        )
        self.global_update = nn.Sequential(
            nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width)
        )
        self.edge_update = nn.Sequential(
            nn.Linear(4 * width, width), nn.GELU(), nn.Linear(width, width)
        )
        self.input_norm = nn.LayerNorm(width)
        self.hidden_norm = nn.LayerNorm(width)
        self.global_norm = nn.LayerNorm(width)
        self.edge_norm = nn.LayerNorm(width)

    def forward(
        self,
        edge: torch.Tensor,
        input_nodes: torch.Tensor,
        hidden_nodes: torch.Tensor,
        global_node: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = edge.size(0)
        input_message = edge.mean(dim=2)
        hidden_message = edge.mean(dim=1)
        global_message = edge.mean(dim=(1, 2))

        input_nodes = self.input_norm(
            input_nodes
            + self.input_update(
                torch.cat(
                    (input_nodes, input_message,
                     global_node[:, None, :].expand(-1, SEQ_LEN, -1)),
                    dim=-1,
                )
            )
        )
        hidden_nodes = self.hidden_norm(
            hidden_nodes
            + self.hidden_update(
                torch.cat(
                    (hidden_nodes, hidden_message,
                     global_node[:, None, :].expand(-1, HIDDEN, -1)),
                    dim=-1,
                )
            )
        )
        global_node = self.global_norm(
            global_node + self.global_update(torch.cat((global_node, global_message), dim=-1))
        )

        edge_delta = self.edge_update(
            torch.cat(
                (
                    edge,
                    input_nodes[:, :, None, :].expand(-1, -1, HIDDEN, -1),
                    hidden_nodes[:, None, :, :].expand(-1, SEQ_LEN, -1, -1),
                    global_node[:, None, None, :].expand(-1, SEQ_LEN, HIDDEN, -1),
                ),
                dim=-1,
            )
        )
        edge = self.edge_norm(edge + edge_delta)
        return edge, input_nodes, hidden_nodes, global_node


class GraphVelocity(nn.Module):
    """Shared edge scorer and flow-velocity model on the 11-by-8 graph.

    Args:
        node_dim: Feature width of ``node_context[B, 8, F]``.
        edge_dim: Feature width of ``edge_context[B, 11, 8, E]``.
        task_dim: Feature width of ``task_context[B, T]``.
        width: Internal width, either 32 or 48.
        layers: Number of message-passing blocks (three by default).

    No hidden-node identity embeddings are used, so jointly permuting the
    hidden axis of node context, edge state, and edge context permutes scores
    by the same amount.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        task_dim: int,
        width: int = 32,
        layers: int = 3,
    ) -> None:
        super().__init__()
        if min(node_dim, edge_dim, task_dim) < 1:
            raise ValueError("node_dim, edge_dim, and task_dim must be positive")
        if width not in (32, 48):
            raise ValueError("width must be 32 or 48")
        if layers < 1:
            raise ValueError("layers must be positive")
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.task_dim = int(task_dim)
        self.width = int(width)
        self.layers = int(layers)

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, width), nn.GELU(), nn.Linear(width, width), nn.LayerNorm(width)
        )
        self.edge_context_encoder = nn.Sequential(
            nn.Linear(edge_dim, width), nn.GELU(), nn.Linear(width, width), nn.LayerNorm(width)
        )
        self.edge_state_encoder = nn.Sequential(
            nn.Linear(1, width), nn.GELU(), nn.Linear(width, width)
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, width), nn.GELU(), nn.Linear(width, width), nn.LayerNorm(width)
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(4, width), nn.GELU(), nn.Linear(width, width)
        )
        self.input_positions = nn.Parameter(torch.empty(SEQ_LEN, width))
        nn.init.normal_(self.input_positions, std=width ** -0.5)
        self.blocks = nn.ModuleList(_MessageBlock(width) for _ in range(layers))
        self.score_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def _check_inputs(
        self,
        edge_state: torch.Tensor,
        time: torch.Tensor,
        node_context: torch.Tensor,
        edge_context: torch.Tensor,
        task_context: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        if edge_state.ndim != 3 or edge_state.shape[1:] != (SEQ_LEN, HIDDEN):
            raise ValueError(f"edge_state must have shape [B, {SEQ_LEN}, {HIDDEN}]")
        batch = edge_state.shape[0]
        if node_context.shape != (batch, HIDDEN, self.node_dim):
            raise ValueError(f"node_context must have shape [B, {HIDDEN}, {self.node_dim}]")
        if edge_context.shape != (batch, SEQ_LEN, HIDDEN, self.edge_dim):
            raise ValueError(
                f"edge_context must have shape [B, {SEQ_LEN}, {HIDDEN}, {self.edge_dim}]"
            )
        if task_context.shape != (batch, self.task_dim):
            raise ValueError(f"task_context must have shape [B, {self.task_dim}]")
        if time.ndim == 2 and time.shape == (batch, 1):
            time = time[:, 0]
        if time.shape != (batch,):
            raise ValueError("time must have shape [B] or [B, 1]")
        inputs = (time, node_context, edge_context, task_context)
        if any(value.device != edge_state.device for value in inputs):
            raise ValueError("edge_state, time, and contexts must be on the same device")
        if any(value.dtype != edge_state.dtype for value in inputs):
            raise ValueError("edge_state, time, and contexts must have the same dtype")
        return batch, time

    def forward(
        self,
        edge_state: torch.Tensor,
        time: torch.Tensor,
        node_context: torch.Tensor,
        edge_context: torch.Tensor,
        task_context: torch.Tensor,
    ) -> torch.Tensor:
        """Return one scalar score or velocity per edge, shaped ``[B, 11, 8]``."""
        batch, time = self._check_inputs(
            edge_state, time, node_context, edge_context, task_context
        )
        time_features = torch.stack(
            (time, torch.sin(torch.pi * time), torch.cos(torch.pi * time), 2.0 * time - 1.0),
            dim=-1,
        )
        input_nodes = self.input_positions[None, :, :].expand(batch, -1, -1)
        hidden_nodes = self.node_encoder(node_context)
        global_node = self.task_encoder(task_context) + self.time_encoder(time_features)
        edge = (
            self.edge_state_encoder(edge_state.unsqueeze(-1))
            + self.edge_context_encoder(edge_context)
            + input_nodes[:, :, None, :]
            + hidden_nodes[:, None, :, :]
        )
        for block in self.blocks:
            edge, input_nodes, hidden_nodes, global_node = block(
                edge, input_nodes, hidden_nodes, global_node
            )
        return self.score_head(edge).squeeze(-1)


# Descriptive alias for code that uses the model for deterministic logits.
PatternEdgeScoreGNN = GraphVelocity


def score_to_mask(scores: torch.Tensor, k: int = ACTIVE_EDGES) -> torch.Tensor:
    """Convert edge scores to an exact top-k binary mask.

    Centering each flattened candidate makes the ranking explicitly invariant
    to adding a constant to all scores in that candidate.
    """
    if scores.ndim < 2:
        raise ValueError("scores must have at least two dimensions")
    flat = scores.flatten(start_dim=-2)
    if not 0 < k <= flat.shape[-1]:
        raise ValueError(f"k must be in [1, {flat.shape[-1]}]")
    centered = flat - flat.mean(dim=-1, keepdim=True)
    indices = centered.topk(k, dim=-1).indices
    return torch.zeros_like(flat).scatter_(-1, indices, 1.0).reshape_as(scores)


def mask_to_flow_endpoint(
    mask: torch.Tensor,
    encoding: Literal["signed", "standardized"] = "signed",
) -> torch.Tensor:
    """Encode binary masks as flow endpoints (±1 by default).

    ``standardized`` maps the two binary values to zero-mean, unit-population-
    variance values for each candidate, when both classes are present.
    """
    if mask.ndim < 2:
        raise ValueError("mask must have at least two dimensions")
    # The helper is used inside GPU training loops, so avoid a value check
    # that would synchronize the device on every batch. Callers supply hard
    # binary masks by contract.
    if not mask.is_floating_point():
        mask = mask.to(dtype=torch.get_default_dtype())
    if encoding == "signed":
        return mask.mul(2).sub(1)
    if encoding == "standardized":
        flat = mask.flatten(start_dim=-2)
        mean = flat.mean(dim=-1, keepdim=True)
        std = flat.var(dim=-1, keepdim=True, unbiased=False).sqrt()
        scale = std.clamp_min(torch.finfo(mask.dtype).eps)
        return ((flat - mean) / scale).reshape_as(mask)
    raise ValueError("encoding must be 'signed' or 'standardized'")


def flow_matching_coupling(
    mask: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    encoding: Literal["signed", "standardized"] = "signed",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw an independent-Gaussian linear path and its constant velocity.

    Returns ``(state_t, time, velocity_target)`` for straight interpolation
    from standard Gaussian noise at t=0 to the encoded mask at t=1.
    """
    endpoint = mask_to_flow_endpoint(mask, encoding=encoding)
    batch = endpoint.shape[0]
    if noise is None:
        noise = torch.randn(
            endpoint.shape, device=endpoint.device, dtype=endpoint.dtype, generator=generator
        )
    elif noise.shape != endpoint.shape:
        raise ValueError("noise must have the same shape as mask")
    elif noise.device != endpoint.device or noise.dtype != endpoint.dtype:
        raise ValueError("noise and mask must have the same device and dtype")
    if time is None:
        time = torch.rand(batch, device=endpoint.device, dtype=endpoint.dtype, generator=generator)
    else:
        time = torch.as_tensor(time, device=endpoint.device, dtype=endpoint.dtype)
        if time.ndim == 0:
            time = time.expand(batch)
        elif time.ndim == 2 and time.shape == (batch, 1):
            time = time[:, 0]
        if time.shape != (batch,):
            raise ValueError("time must be scalar or have shape [B]")
    t = time[:, None, None]
    state = noise.lerp(endpoint, t)
    return state, time, endpoint - noise


def flow_matching_loss(
    model: GraphVelocity,
    mask: torch.Tensor,
    node_context: torch.Tensor,
    edge_context: torch.Tensor,
    task_context: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    encoding: Literal["signed", "standardized"] = "signed",
) -> torch.Tensor:
    """Mean-squared conditional-flow velocity loss for mask endpoints."""
    state, time, target_velocity = flow_matching_coupling(
        mask, noise=noise, time=time, generator=generator, encoding=encoding
    )
    predicted = model(state, time, node_context, edge_context, task_context)
    return F.mse_loss(predicted, target_velocity)


def elite_mask_bce_loss(
    model: GraphVelocity,
    mask: torch.Tensor,
    node_context: torch.Tensor,
    edge_context: torch.Tensor,
    task_context: torch.Tensor,
    *,
    edge_state: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fit deterministic zero-state edge logits to elite binary masks."""
    # As with flow endpoints, masks are binary by contract. Cast to the model
    # input dtype without forcing a device synchronization for validation.
    mask = mask.to(dtype=node_context.dtype)
    if edge_state is None:
        edge_state = torch.zeros_like(mask)
    if time is None:
        time = torch.zeros(mask.shape[0], device=mask.device, dtype=mask.dtype)
    logits = model(edge_state, time, node_context, edge_context, task_context)
    return F.binary_cross_entropy_with_logits(logits, mask.to(dtype=logits.dtype))


@torch.no_grad()
def sample_flow_endpoint(
    model: GraphVelocity,
    node_context: torch.Tensor,
    edge_context: torch.Tensor,
    task_context: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    solver: Literal["euler", "heun"] = "euler",
    steps: int | None = None,
) -> torch.Tensor:
    """Integrate from Gaussian noise and return endpoint edge scores.

    Defaults are Euler with 12 steps or Heun with 8 steps. Pass ``steps`` to
    set a different positive step count explicitly.
    """
    if solver not in ("euler", "heun"):
        raise ValueError("solver must be 'euler' or 'heun'")
    if steps is None:
        steps = 12 if solver == "euler" else 8
    if steps < 1:
        raise ValueError("steps must be positive")
    batch = node_context.shape[0]
    shape = (batch, SEQ_LEN, HIDDEN)
    if noise is None:
        noise = torch.randn(
            shape,
            device=node_context.device,
            dtype=node_context.dtype,
            generator=generator,
        )
    elif noise.shape != shape:
        raise ValueError(f"noise must have shape {shape}")
    elif noise.device != node_context.device or noise.dtype != node_context.dtype:
        raise ValueError("noise and contexts must have the same device and dtype")

    state = noise
    dt = 1.0 / steps
    for index in range(steps):
        t_value = index * dt
        time = torch.full((batch,), t_value, device=state.device, dtype=state.dtype)
        velocity = model(state, time, node_context, edge_context, task_context)
        if solver == "euler":
            state = state + dt * velocity
        else:
            next_time = torch.full(
                (batch,), t_value + dt, device=state.device, dtype=state.dtype
            )
            predicted = state + dt * velocity
            next_velocity = model(predicted, next_time, node_context, edge_context, task_context)
            state = state + 0.5 * dt * (velocity + next_velocity)
    return state


@torch.no_grad()
def sample_flow_mask(
    model: GraphVelocity,
    node_context: torch.Tensor,
    edge_context: torch.Tensor,
    task_context: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    solver: Literal["euler", "heun"] = "euler",
    steps: int | None = None,
    k: int = ACTIVE_EDGES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the integrated endpoint scores and their exact top-k mask."""
    scores = sample_flow_endpoint(
        model,
        node_context,
        edge_context,
        task_context,
        noise=noise,
        generator=generator,
        solver=solver,
        steps=steps,
    )
    return scores, score_to_mask(scores, k=k)
