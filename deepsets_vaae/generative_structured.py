"""Permutation-equivariant conditional fields for functional weight maps.

The maps used by the DeepSets experiments have shape ``[batch, 784, 32]``:
rows are labelled MNIST pixels and columns are exchangeable hidden neurons.
Consequently a hidden-neuron permutation must commute with a velocity field.

``BipartiteGNNField`` follows the edge/node/global message-passing pattern of
the graph meta network in section 4.2 of Piven et al. (2026), adapted to the
single bipartite weight matrix here.  Pixel identities are learnt node labels;
hidden nodes deliberately have no identifier.  ``SetTransformerField`` treats
the 32 columns as a set of 784-dimensional tokens and uses SAB blocks from
Lee et al. (2019).  It is an equivariant encoder rather than the invariant
pooling/readout part of a Set Transformer.

Both modules are only velocity parameterizations.  The flow-matching loss,
normalization, sampling ODE and choice of sparsity are intentionally owned by
the experiment runner.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, Iterable, Tuple, Type

import torch
from torch import Tensor, nn


def _mlp(in_features: int, out_features: int, hidden_features: int) -> nn.Sequential:
    """Small shared MLP used for node, edge and readout updates."""
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.SiLU(),
        nn.Linear(hidden_features, out_features),
    )


def _factorized_edge_mlp(mlp: nn.Sequential, pixel: Tensor, hidden: Tensor, edge: Tensor) -> Tensor:
    """Apply an edge MLP without materializing repeated node features.

    ``mlp`` receives ``cat(pixel_i, hidden_j, edge_ij)`` at every edge.  Its
    first affine map is separable over these three concatenated slices, so the
    same result can be formed from two node projections and one edge projection.
    This preserves the existing ``nn.Sequential`` parameters/state-dict names
    while avoiding a ``[B, F, H, 2*width+edge_width]`` temporary.
    """
    first, activation, second = mlp
    if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
        raise TypeError("factorized edge MLP expects Linear/activation/Linear")
    pixel_width = pixel.shape[-1]
    hidden_width = hidden.shape[-1]
    weights = first.weight
    pixel_weight = weights[:, :pixel_width]
    hidden_weight = weights[:, pixel_width:pixel_width + hidden_width]
    edge_weight = weights[:, pixel_width + hidden_width:]
    pixel_projection = torch.nn.functional.linear(pixel, pixel_weight)
    hidden_projection = torch.nn.functional.linear(hidden, hidden_weight)
    edge_projection = torch.nn.functional.linear(edge, edge_weight, first.bias)
    combined = (
        edge_projection
        + pixel_projection[:, :, None, :]
        + hidden_projection[:, None, :, :]
    )
    return second(activation(combined))


class _Condition(nn.Module):
    """Joint continuous-time and discrete-task conditioning without token IDs."""

    def __init__(self, tasks: int, width: int) -> None:
        super().__init__()
        if tasks < 1:
            raise ValueError("tasks must be positive")
        self.tasks = tasks
        self.width = width
        self.task_embedding = nn.Embedding(tasks, width)
        self.register_buffer(
            "frequencies",
            torch.exp(torch.linspace(0.0, math.log(1000.0), width // 2)),
            persistent=False,
        )
        self.time_mlp = _mlp(width, width, width)
        self.combine = _mlp(2 * width, width, width)

    def forward(self, t: Tensor, task: Tensor) -> Tensor:
        if t.ndim != 1 or task.ndim != 1 or t.shape != task.shape:
            raise ValueError("t and task must both have shape [batch]")
        if task.dtype != torch.long:
            raise TypeError("task must be a torch.long tensor")
        # Reductions followed by Python truth conversion synchronize CUDA.
        # The experiment runner validates GPU task labels; retain this useful
        # defensive range check for inexpensive CPU calls and self-checks.
        if task.device.type == "cpu" and task.numel() and (
            task.min() < 0 or task.max() >= self.tasks
        ):
            raise ValueError(f"task values must lie in [0, {self.tasks})")
        # Fixed Fourier features make a scalar integration time expressive while
        # preserving all hidden-neuron symmetries.
        frequencies = self.frequencies.to(dtype=t.dtype)
        phase = t[:, None] * frequencies[None, :] * (2.0 * math.pi)
        time_features = torch.cat((phase.sin(), phase.cos()), dim=-1)
        if time_features.shape[-1] < self.width:
            time_features = torch.nn.functional.pad(
                time_features, (0, self.width - time_features.shape[-1])
            )
        time_features = self.time_mlp(time_features)
        return self.combine(torch.cat((time_features, self.task_embedding(task)), dim=-1))


class _BipartiteBlock(nn.Module):
    """One equivariant edge-to-node-to-edge message-passing block."""

    def __init__(self, width: int, edge_width: int) -> None:
        super().__init__()
        self.pixel_norm = nn.LayerNorm(width)
        self.hidden_norm = nn.LayerNorm(width)
        self.edge_norm = nn.LayerNorm(edge_width)
        self.edge_update = _mlp(2 * width + edge_width, edge_width, 2 * width)
        self.pixel_message = _mlp(edge_width, width, width)
        self.hidden_message = _mlp(edge_width, width, width)
        self.pixel_update = _mlp(2 * width, width, 2 * width)
        self.hidden_update = _mlp(2 * width, width, 2 * width)

    def forward(self, pixel: Tensor, hidden: Tensor, edge: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # pixel: [B, F, W], hidden: [B, H, W], edge: [B, F, H, E].
        p = self.pixel_norm(pixel)
        h = self.hidden_norm(hidden)
        e = self.edge_norm(edge)
        edge = edge + _factorized_edge_mlp(self.edge_update, p, h, e)
        # Means are symmetric aggregators over the exchangeable hidden nodes
        # and, respectively, the fixed but collectively informative pixels.
        pixel_messages = self.pixel_message(edge).mean(dim=2)
        hidden_messages = self.hidden_message(edge).mean(dim=1)
        pixel = pixel + self.pixel_update(torch.cat((p, pixel_messages), dim=-1))
        hidden = hidden + self.hidden_update(torch.cat((h, hidden_messages), dim=-1))
        return pixel, hidden, edge


class BipartiteGNNField(nn.Module):
    """Conditional GMN-style field on a labelled-pixel / hidden-neuron graph.

    The output transforms as ``v(x[..., perm]) == v(x)[..., perm]``.  There
    are no learned hidden-neuron embeddings; all hidden nodes receive the same
    condition before they are differentiated by incident edge values.
    """

    def __init__(
        self,
        features: int = 784,
        hidden: int = 32,
        tasks: int = 4,
        width: int = 64,
        edge_width: int = 16,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if features < 1 or hidden < 1 or width < 2 or edge_width < 1 or depth < 1:
            raise ValueError("features, hidden, width, edge_width and depth must be positive")
        self.features, self.hidden, self.tasks = features, hidden, tasks
        self.condition = _Condition(tasks, width)
        # These are labels for fixed input pixels, not labels for hidden units.
        self.pixel_embedding = nn.Embedding(features, width)
        self.pixel_condition = nn.Linear(width, width, bias=False)
        self.hidden_condition = nn.Linear(width, width, bias=False)
        self.edge_encoder = nn.Linear(1, edge_width)
        self.blocks = nn.ModuleList(
            [_BipartiteBlock(width, edge_width) for _ in range(depth)]
        )
        self.readout = _mlp(2 * width + edge_width, 1, 2 * width)
        # A scalar residual gives the field a cheap coordinate-wise path without
        # breaking equivariance.  Its initial zero value does not assume x is a
        # useful velocity before fitting.
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor, t: Tensor, task: Tensor) -> Tensor:
        _check_input(x, t, task, self.features, self.hidden)
        batch = x.shape[0]
        condition = self.condition(t, task)
        pixel_ids = torch.arange(self.features, device=x.device)
        pixel = self.pixel_embedding(pixel_ids)[None].expand(batch, -1, -1)
        pixel = pixel + self.pixel_condition(condition)[:, None, :]
        hidden = self.hidden_condition(condition)[:, None, :].expand(-1, self.hidden, -1)
        edge = self.edge_encoder(x.unsqueeze(-1))
        for block in self.blocks:
            pixel, hidden, edge = block(pixel, hidden, edge)
        velocity = _factorized_edge_mlp(self.readout, pixel, hidden, edge).squeeze(-1)
        return velocity + self.residual_scale * x


class _SAB(nn.Module):
    """Set Attention Block: maps a set of hidden-neuron tokens to a set."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.feed_forward = _mlp(width, width, 2 * width)

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        return tokens + self.feed_forward(self.norm2(tokens))


class SetTransformerField(nn.Module):
    """SAB Set-Transformer field over the unordered hidden-neuron columns.

    Each column is a token containing all 784 labelled pixel values.  The
    input/output linear maps keep their pixel-coordinate semantics, whereas
    SAB attention shares all operations across hidden tokens and receives no
    hidden positional encoding.
    """

    def __init__(
        self,
        features: int = 784,
        hidden: int = 32,
        tasks: int = 4,
        width: int = 128,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if features < 1 or hidden < 1 or width < 2 or depth < 1:
            raise ValueError("features, hidden, width and depth must be positive")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.features, self.hidden, self.tasks = features, hidden, tasks
        self.condition = _Condition(tasks, width)
        self.input_projection = nn.Linear(features, width)
        self.blocks = nn.ModuleList([_SAB(width, heads) for _ in range(depth)])
        self.output_projection = nn.Linear(width, features)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor, t: Tensor, task: Tensor) -> Tensor:
        _check_input(x, t, task, self.features, self.hidden)
        # [B, H, F] is a set; no token receives an ordinal/ID feature.
        tokens = self.input_projection(x.transpose(1, 2))
        tokens = tokens + self.condition(t, task)[:, None, :]
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_projection(tokens).transpose(1, 2) + self.residual_scale * x


def _check_input(x: Tensor, t: Tensor, task: Tensor, features: int, hidden: int) -> None:
    if x.ndim != 3 or x.shape[1:] != (features, hidden):
        raise ValueError(
            f"x must have shape [batch, {features}, {hidden}], got {tuple(x.shape)}"
        )
    if not x.is_floating_point():
        raise TypeError("x must be floating point")
    if t.shape != (x.shape[0],) or task.shape != (x.shape[0],):
        raise ValueError("t and task must have shape [batch]")


def parameter_count(module: nn.Module) -> int:
    """Number of trainable parameters, convenient for experiment manifests."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


@torch.no_grad()
def _permutation_error(module: nn.Module, x: Tensor, t: Tensor, task: Tensor) -> float:
    module.eval()
    permutation = torch.randperm(x.shape[-1], device=x.device)
    reference = module(x, t, task)
    permuted = module(x[:, :, permutation], t, task)
    denominator = reference[:, :, permutation].abs().max().clamp_min(1e-6)
    return float((permuted - reference[:, :, permutation]).abs().max() / denominator)


def self_check(device: str | torch.device = "cpu") -> Dict[str, Dict[str, float | int | Tuple[int, ...]]]:
    """Quick CPU/CUDA shape, gradient, finiteness and equivariance check.

    The relative hidden-permutation error is asserted below ``1e-5``.  Small
    dimensions make this suitable for the module CLI; production maps remain
    the default ``[B, 784, 32]`` at construction time.
    """
    chosen_device = torch.device(device)
    torch.manual_seed(17)
    batch, features, hidden, tasks, width = 3, 17, 5, 4, 32
    x = torch.randn(batch, features, hidden, device=chosen_device)
    t = torch.rand(batch, device=chosen_device)
    task = torch.tensor([0, 2, 3], dtype=torch.long, device=chosen_device)
    models: Iterable[Tuple[str, Type[nn.Module], Dict[str, int]]] = (
        ("bipartite_gnn", BipartiteGNNField,
         {"features": features, "hidden": hidden, "tasks": tasks, "width": width, "edge_width": 12}),
        ("set_transformer", SetTransformerField,
         {"features": features, "hidden": hidden, "tasks": tasks, "width": width, "heads": 4}),
    )
    report: Dict[str, Dict[str, float | int | Tuple[int, ...]]] = {}
    for name, constructor, kwargs in models:
        model = constructor(**kwargs).to(chosen_device)
        value = model(x, t, task)
        loss = value.square().mean()
        loss.backward()
        finite_grads = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        relative_error = _permutation_error(model, x, t, task)
        if value.shape != x.shape or not torch.isfinite(value).all() or not finite_grads:
            raise AssertionError(f"{name}: invalid output or gradient")
        if relative_error >= 1e-5:
            raise AssertionError(f"{name}: permutation relative error {relative_error:.3e}")
        report[name] = {
            "parameters": parameter_count(model),
            "shape": tuple(value.shape),
            "permutation_relative_error": relative_error,
        }
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Self-check structured flow-matching fields")
    parser.add_argument("--device", default="cpu", help="torch device, default: cpu")
    arguments = parser.parse_args()
    print(json.dumps(self_check(arguments.device), indent=2))


if __name__ == "__main__":
    _main()
