"""Parameter-subspace models for the length-conditioned pattern experiment.

Each task-specific MLP is represented by two short vectors ``v1, v2``.  A
shared pair of matrices ``U = (U1, U2)`` expands them into the two affine
layers of an MLP.  Biases are included in the expanded matrices, so they are
subject to the same structural prior as ordinary weights.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


UTuple = Tuple[torch.Tensor, torch.Tensor]
VTuple = Tuple[torch.Tensor, torch.Tensor]


def _normalize_columns(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return a matrix whose nonzero columns have unit Euclidean norm."""
    return matrix / torch.linalg.vector_norm(matrix, dim=0, keepdim=True).clamp_min(eps)


def _check_dimensions(seq_len: int, hidden: int, rank1: int, rank2: int) -> None:
    if seq_len < 1 or hidden < 1 or rank1 < 1 or rank2 < 1:
        raise ValueError("seq_len, hidden, rank1, and rank2 must be positive")


def _output_size(seq_len: int, hidden: int, rank1: int, rank2: int) -> int:
    return (seq_len + 1) * hidden * rank1 + (hidden + 1) * rank2


class FullUGenerator(nn.Module):
    """Generate an unfactorized, layerwise parameter basis from pattern length.

    ``U1`` has shape ``((seq_len + 1) * hidden, rank1)`` and expands the
    input-to-hidden affine map.  ``U2`` has shape ``(hidden + 1, rank2)`` and
    expands the hidden-to-logit affine map.  The extra rows are bias terms.
    """

    def __init__(
        self,
        seq_len: int = 32,
        hidden: int = 32,
        rank1: int = 16,
        rank2: int = 4,
        width: int = 64,
        length_min: int = 3,
        length_max: int = 8,
        eps: float = 1e-8,
        depth: int = 2,
        condition_length: bool = True,
    ) -> None:
        super().__init__()
        _check_dimensions(seq_len, hidden, rank1, rank2)
        if width < 1 or depth < 1 or length_min < 1 or length_max < length_min:
            raise ValueError("invalid generator width or pattern-length range")
        self.seq_len = seq_len
        self.hidden = hidden
        self.rank1 = rank1
        self.rank2 = rank2
        self.length_min = length_min
        self.length_max = length_max
        self.eps = eps
        self.depth = depth
        self.condition_length = condition_length
        output_size = _output_size(seq_len, hidden, rank1, rank2)
        # Keep the depth=2 module layout byte-for-byte compatible with the
        # original generator: Linear, SiLU, Linear, SiLU, Linear.  In
        # particular, existing checkpoints retain their network.0/2/4 keys.
        layers: list[nn.Module] = [nn.Linear(1, width), nn.SiLU()]
        for _ in range(1, depth):
            layers.extend((nn.Linear(width, width), nn.SiLU()))
        layers.append(nn.Linear(width, output_size))
        self.network = nn.Sequential(*layers)

    def _normalized_length(self, length: int | torch.Tensor) -> torch.Tensor:
        parameter = next(self.parameters())
        if isinstance(length, torch.Tensor):
            if length.numel() != 1:
                raise ValueError("FullUGenerator accepts one scalar length at a time")
            value = length.reshape(1).to(device=parameter.device, dtype=parameter.dtype)
        else:
            value = torch.tensor([length], device=parameter.device, dtype=parameter.dtype)
        if not self.length_min <= float(value.detach().cpu()) <= self.length_max:
            raise ValueError(
                f"length must be in [{self.length_min}, {self.length_max}], got {length}"
            )
        midpoint = (self.length_min + self.length_max) / 2.0
        half_range = max((self.length_max - self.length_min) / 2.0, 1.0)
        if not self.condition_length:
            # This is deliberately an input change only.  The unconditional
            # control has precisely the same hypernetwork and parameter count
            # as a conditional generator of the same width and depth.
            return torch.zeros((1, 1), device=value.device, dtype=value.dtype)
        return ((value - midpoint) / half_range).reshape(1, 1)

    def forward(self, length: int | torch.Tensor) -> UTuple:
        output = self.network(self._normalized_length(length)).reshape(-1)
        first_size = (self.seq_len + 1) * self.hidden * self.rank1
        u1 = output[:first_size].reshape((self.seq_len + 1) * self.hidden, self.rank1)
        u2 = output[first_size:].reshape(self.hidden + 1, self.rank2)
        return _normalize_columns(u1, self.eps), _normalize_columns(u2, self.eps)


class LearnedUTable(nn.Module):
    """One directly learned full ``U`` per known pattern length control."""

    def __init__(
        self,
        seq_len: int = 32,
        hidden: int = 32,
        rank1: int = 16,
        rank2: int = 4,
        length_min: int = 3,
        length_max: int = 8,
        eps: float = 1e-8,
        seed: int = 0,
    ) -> None:
        super().__init__()
        _check_dimensions(seq_len, hidden, rank1, rank2)
        if length_min < 1 or length_max < length_min:
            raise ValueError("invalid pattern-length range")
        self.seq_len, self.hidden = seq_len, hidden
        self.rank1, self.rank2 = rank1, rank2
        self.length_min, self.length_max, self.eps = length_min, length_max, eps
        count = length_max - length_min + 1
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.u1_table = nn.Parameter(torch.randn(
            count, (seq_len + 1) * hidden, rank1, generator=generator
        ) * 0.1)
        self.u2_table = nn.Parameter(torch.randn(
            count, hidden + 1, rank2, generator=generator
        ) * 0.1)

    def forward(self, length: int | torch.Tensor) -> UTuple:
        if isinstance(length, torch.Tensor):
            if length.numel() != 1:
                raise ValueError("LearnedUTable accepts one scalar length at a time")
            length = int(length.detach().cpu().item())
        if not self.length_min <= length <= self.length_max:
            raise ValueError(f"unknown length {length}")
        index = length - self.length_min
        return (
            _normalize_columns(self.u1_table[index], self.eps),
            _normalize_columns(self.u2_table[index], self.eps),
        )


class RandomU(nn.Module):
    """Seeded full-U control, independently sampled for each pattern length."""

    def __init__(
        self,
        seq_len: int = 32,
        hidden: int = 32,
        rank1: int = 16,
        rank2: int = 4,
        length_min: int = 3,
        length_max: int = 8,
        seed: int = 0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        _check_dimensions(seq_len, hidden, rank1, rank2)
        if length_min < 1 or length_max < length_min:
            raise ValueError("invalid pattern-length range")
        self.seq_len, self.hidden = seq_len, hidden
        self.rank1, self.rank2 = rank1, rank2
        self.length_min, self.length_max = length_min, length_max
        self.seed, self.eps = int(seed), eps
        # Lets callers move this stateless control with the other modules.
        self.register_buffer("_anchor", torch.empty(0))

    def forward(self, length: int | torch.Tensor) -> UTuple:
        if isinstance(length, torch.Tensor):
            if length.numel() != 1:
                raise ValueError("RandomU accepts one scalar length at a time")
            length = int(length.detach().cpu().item())
        if not self.length_min <= length <= self.length_max:
            raise ValueError(f"unknown length {length}")
        # The private CPU generator is deliberately recreated per call: neither
        # call order nor the process-wide PyTorch RNG changes this U.
        mixed_seed = (self.seed + 1_000_003 * int(length)) % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(mixed_seed)
        u1 = torch.randn(
            (self.seq_len + 1) * self.hidden, self.rank1, generator=generator
        ).to(device=self._anchor.device, dtype=self._anchor.dtype)
        u2 = torch.randn(self.hidden + 1, self.rank2, generator=generator).to(
            device=self._anchor.device, dtype=self._anchor.dtype
        )
        return _normalize_columns(u1, self.eps), _normalize_columns(u2, self.eps)


def init_v(u: UTuple, seed: int, scale: float = 0.1) -> VTuple:
    """Create a task vector with a private CPU RNG and gradients enabled."""
    if scale < 0:
        raise ValueError("scale must be non-negative")
    u1, u2 = u
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    v1 = (torch.randn(u1.shape[1], generator=generator) * scale).to(
        device=u1.device, dtype=u1.dtype
    ).requires_grad_(True)
    v2 = (torch.randn(u2.shape[1], generator=generator) * scale).to(
        device=u2.device, dtype=u2.dtype
    ).requires_grad_(True)
    return v1, v2


def forward_with_u(
    x: torch.Tensor,
    u: UTuple,
    v: VTuple,
    seq_len: int = 32,
    hidden: int = 32,
) -> torch.Tensor:
    """Evaluate an MLP whose affine layers are expanded from ``U`` and ``v``."""
    u1, u2 = u
    v1, v2 = v
    if x.ndim != 2 or x.shape[1] != seq_len:
        raise ValueError(f"x must have shape (batch, {seq_len})")
    if u1.shape != ((seq_len + 1) * hidden, v1.numel()):
        raise ValueError("U1 and v1 are incompatible with seq_len and hidden")
    if u2.shape != (hidden + 1, v2.numel()):
        raise ValueError("U2 and v2 are incompatible with hidden")
    if x.device != u1.device or x.device != u2.device:
        raise ValueError("x and U must be on the same device")
    if x.dtype != u1.dtype or x.dtype != u2.dtype:
        raise ValueError("x and U must have the same dtype")

    w1 = (u1 @ v1).reshape(seq_len + 1, hidden)
    w2 = u2 @ v2
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    hidden_values = F.relu(torch.cat((x, ones), dim=1) @ w1)
    return torch.cat((hidden_values, ones), dim=1) @ w2


def adapt_v(
    u: UTuple,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    lr: float,
    seed: int,
    create_graph: bool,
    batch_size: int | None = None,
    *,
    optimizer: str = "sgd",
    init_scale: float = 0.1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
) -> VTuple:
    """Adapt a fresh ``v`` by functional SGD or Adam while retaining U's graph.

    With ``create_graph=True`` the returned vectors remain differentiably
    connected to ``u`` through every optimizer update, which is the full MAML-style
    meta-gradient.  With ``False`` each update is detached to keep evaluation
    inexpensive.  Adam uses the usual bias-corrected update and keeps its
    moments in the graph only for full meta-gradients.
    """
    if steps < 0 or lr < 0:
        raise ValueError("steps and lr must be non-negative")
    if optimizer not in {"sgd", "adam"}:
        raise ValueError("optimizer must be 'sgd' or 'adam'")
    if init_scale < 0:
        raise ValueError("init_scale must be non-negative")
    if not 0 <= adam_beta1 < 1 or not 0 <= adam_beta2 < 1 or adam_eps < 0:
        raise ValueError("invalid Adam hyperparameters")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("y must have shape (batch,) matching x")
    if y.device != x.device:
        raise ValueError("x and y must be on the same device")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    seq_len = x.shape[1]
    hidden = u[1].shape[0] - 1
    v = init_v(u, seed, scale=init_scale)
    first_moment = tuple(torch.zeros_like(value) for value in v)
    second_moment = tuple(torch.zeros_like(value) for value in v)
    batch_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 17_171)
    n_examples = x.shape[0]
    for _ in range(steps):
        if batch_size is None or batch_size >= n_examples:
            xb, yb = x, y
        else:
            indices = torch.randperm(n_examples, generator=batch_generator)[:batch_size]
            indices = indices.to(x.device)
            xb, yb = x[indices], y[indices]
        logits = forward_with_u(xb, u, v, seq_len=seq_len, hidden=hidden)
        loss = F.binary_cross_entropy_with_logits(logits, yb.to(dtype=logits.dtype))
        gradients = torch.autograd.grad(loss, v, create_graph=create_graph)
        if optimizer == "sgd":
            updated = tuple(value - lr * gradient for value, gradient in zip(v, gradients))
        else:
            first_moment = tuple(
                adam_beta1 * moment + (1.0 - adam_beta1) * gradient
                for moment, gradient in zip(first_moment, gradients)
            )
            second_moment = tuple(
                adam_beta2 * moment + (1.0 - adam_beta2) * gradient.square()
                for moment, gradient in zip(second_moment, gradients)
            )
            bias_correction1 = 1.0 - adam_beta1 ** (_ + 1)
            bias_correction2 = 1.0 - adam_beta2 ** (_ + 1)

            def stable_sqrt(value: torch.Tensor) -> torch.Tensor:
                """Keep Adam's higher derivatives finite at zero variance.

                The tiny clamp changes only the exactly-zero (or sub-1e-16)
                variance case.  For ordinary nonzero gradients the forward
                update is the same as :class:`torch.optim.Adam`, whose epsilon
                remains outside the square root.
                """
                return torch.sqrt(value.clamp_min(1e-16))

            updated = tuple(
                value - lr * (moment / bias_correction1) /
                (stable_sqrt(variance / bias_correction2) + adam_eps)
                for value, moment, variance in zip(v, first_moment, second_moment)
            )
        if create_graph:
            v = updated  # type: ignore[assignment]
        else:
            v = tuple(value.detach().requires_grad_(True) for value in updated)  # type: ignore[assignment]
            if optimizer == "adam":
                first_moment = tuple(value.detach() for value in first_moment)
                second_moment = tuple(value.detach() for value in second_moment)
    return v


def ideal_u(
    length: int,
    seq_len: int = 32,
    hidden: int = 32,
    rank1: int = 16,
    rank2: int = 4,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> UTuple:
    """Build the analytic shared-window basis for a fixed pattern length.

    It is a diagnostic oracle, never a training target.  The first ``length``
    coordinates of ``v1`` are literal weights shared across all windows.  Its
    next two coordinates are active and inactive hidden biases respectively.
    """
    _check_dimensions(seq_len, hidden, rank1, rank2)
    n_windows = seq_len - length + 1
    if length < 1 or n_windows < 1:
        raise ValueError("length must be between 1 and seq_len")
    if hidden < n_windows:
        raise ValueError("analytic detector needs one hidden unit per window")
    if rank1 < length + 2 or rank2 < 2:
        raise ValueError("analytic detector needs rank1 >= length + 2 and rank2 >= 2")
    u1 = torch.zeros((seq_len + 1) * hidden, rank1, device=device, dtype=dtype)
    for offset in range(length):
        for start in range(n_windows):
            u1[(start + offset) * hidden + start, offset] = 1.0
    for start in range(n_windows):
        u1[seq_len * hidden + start, length] = 1.0
    for unit in range(n_windows, hidden):
        u1[seq_len * hidden + unit, length + 1] = 1.0

    u2 = torch.zeros(hidden + 1, rank2, device=device, dtype=dtype)
    u2[:n_windows, 0] = 1.0
    u2[hidden, 1] = 1.0
    return _normalize_columns(u1), _normalize_columns(u2)


def gold_v(
    pattern: str | torch.Tensor,
    u: UTuple,
    seq_len: int = 32,
    hidden: int = 32,
) -> VTuple:
    """Coordinates which make :func:`ideal_u` an exact pattern detector."""
    if isinstance(pattern, str):
        bits = torch.tensor([int(bit) for bit in pattern], device=u[0].device)
    else:
        bits = pattern.reshape(-1).to(device=u[0].device)
    if bits.numel() < 1 or not bool(((bits == 0) | (bits == 1)).all()):
        raise ValueError("pattern must contain binary 0/1 values")
    length = bits.numel()
    n_windows = seq_len - length + 1
    if hidden < n_windows or u[0].shape[1] < length + 2 or u[1].shape[1] < 2:
        raise ValueError("U does not have enough coordinates for the analytic solution")
    dtype = u[0].dtype
    v1 = torch.zeros(u[0].shape[1], device=u[0].device, dtype=dtype)
    scale = torch.sqrt(torch.tensor(float(n_windows), device=u[0].device, dtype=dtype))
    v1[:length] = bits.to(dtype=dtype).mul(2).sub(1).mul(scale)
    v1[length] = -(length - 1) * scale
    if hidden > n_windows:
        # u1's inactive-bias column is normalized over exactly hidden-n_windows
        # units, so this makes every unused hidden bias -1.
        v1[length + 1] = -torch.sqrt(
            torch.tensor(float(hidden - n_windows), device=u[0].device, dtype=dtype)
        )
    v2 = torch.zeros(u[1].shape[1], device=u[1].device, dtype=u[1].dtype)
    v2[0] = scale.to(dtype=u[1].dtype)
    v2[1] = -0.5
    return v1, v2


def ideal_solution(
    pattern: str | torch.Tensor,
    seq_len: int = 32,
    hidden: int = 32,
    rank1: int = 16,
    rank2: int = 4,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[UTuple, VTuple]:
    """Return the analytic diagnostic ``(U, v)`` for one pattern."""
    length = len(pattern) if isinstance(pattern, str) else int(pattern.numel())
    u = ideal_u(length, seq_len, hidden, rank1, rank2, device=device, dtype=dtype)
    return u, gold_v(pattern, u, seq_len, hidden)
