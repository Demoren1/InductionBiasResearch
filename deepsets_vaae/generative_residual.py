"""Full-rank residual vector field for conditional flat flow matching.

``FlatTimeField`` has the form ``O g(Ax, c(t, task))`` after flattening a map
of dimension ``D=784*32``.  Its input-output Jacobian is therefore
``O J_g A`` and has rank at most its width (256), regardless of ``D``.  That
is a material restriction for Gaussian flow matching: its velocity can need a
coordinate-wise term such as ``-x`` even when the data endpoint is zero.

``ResidualFlatTimeField`` leaves that architecture intact and adds
``a(t, task) x``.  Its Jacobian is ``O J_g A + a I_D`` and can have full rank.
This is an identity path over the original 25,088 coordinates, not a PCA or a
new latent representation.  The residual scalar is zero-initialized, so the
wrapper is bit-exactly equivalent to :class:`FlatTimeField` at construction.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .generative_flat import FlatTimeField, _task_indices, parameter_count


class ResidualFlatTimeField(nn.Module):
    """``FlatTimeField`` plus a zero-initialized conditional identity skip.

    The base is constructed before the new scalar head.  Consequently, for a
    fixed torch seed its parameters exactly match a separately constructed
    ``FlatTimeField(features, hidden, tasks, width)``.  No base parameter is
    changed by adding this wrapper.
    """

    def __init__(self, features: int = 784, hidden: int = 32, tasks: int = 4,
                 width: int = 256) -> None:
        super().__init__()
        # Keep this first: its random draws must agree with the frozen model.
        self.base = FlatTimeField(features=features, hidden=hidden, tasks=tasks, width=width)
        # Deliberately construct after ``base``; zeroing makes the wrapper an
        # exact no-op initially while retaining a learnable time/task scalar.
        self.coefficient = nn.Linear(width, 1)
        with torch.no_grad():
            self.coefficient.weight.zero_()
            self.coefficient.bias.zero_()

    @property
    def features(self) -> int:
        return self.base.features

    @property
    def hidden(self) -> int:
        return self.base.hidden

    @property
    def tasks(self) -> int:
        return self.base.tasks

    def residual_coefficient(self, t: Tensor | float, task: Tensor | int, *,
                             batch: int, device: torch.device,
                             dtype: torch.dtype) -> Tensor:
        """Return the per-map skip coefficient ``a(t, task)`` as ``[batch]``.

        This intentionally recomputes the inexpensive width-dimensional
        condition used by ``base``.  It never projects the map through another
        bottleneck and avoids modifying the frozen ``FlatTimeField`` module.
        """
        ids = _task_indices(task, batch=batch, tasks=self.base.tasks, device=device)
        condition = self.base.condition(
            self.base.time(t, batch=batch, device=device, dtype=dtype) + self.base.task(ids)
        )
        return self.coefficient(condition).squeeze(-1)

    def forward(self, x: Tensor, t: Tensor | float, task: Tensor | int) -> Tensor:
        base_output = self.base(x, t, task)
        coefficient = self.residual_coefficient(
            t, task, batch=x.shape[0], device=x.device, dtype=x.dtype
        )
        return base_output + coefficient[:, None, None] * x


def _jacobian_matrix(model: nn.Module, x: Tensor, t: Tensor, task: Tensor) -> Tensor:
    """Dense toy-map Jacobian, used only by :func:`selfcheck`."""
    jacobian = torch.autograd.functional.jacobian(
        lambda value: model(value, t, task).reshape(-1), x, vectorize=True
    )
    dimension = x.numel()
    return jacobian.reshape(dimension, dimension)


def selfcheck() -> dict[str, Any]:
    """CPU checks for exact initialization, rank, finite values, and gradients.

    The rank check uses a 21-coordinate toy map.  A plain width-8 field has
    Jacobian rank at most 8; setting the wrapper's skip coefficient to one
    produces a full-rank 21-by-21 Jacobian.
    """
    device = torch.device("cpu")
    torch.manual_seed(431)
    old = FlatTimeField(features=7, hidden=3, tasks=4, width=8).to(device)
    torch.manual_seed(431)
    wrapper = ResidualFlatTimeField(features=7, hidden=3, tasks=4, width=8).to(device)
    if old.state_dict().keys() != wrapper.base.state_dict().keys():
        raise AssertionError("wrapper base state does not match FlatTimeField state")
    if any(not torch.equal(old_value, wrapper.base.state_dict()[name])
           for name, old_value in old.state_dict().items()):
        raise AssertionError("wrapper construction changed FlatTimeField parameters")

    x = torch.randn(1, 7, 3, device=device)
    t = torch.tensor([0.37], device=device)
    task = torch.tensor([2], device=device)
    old_output = old(x, t, task)
    initial_output = wrapper(x, t, task)
    if not torch.equal(old_output, initial_output):
        raise AssertionError("zero-initialized residual is not bit-exactly a FlatTimeField")

    old_rank = int(torch.linalg.matrix_rank(_jacobian_matrix(old, x, t, task)).item())
    if old_rank > 8:
        raise AssertionError(f"plain field rank {old_rank} exceeds width 8")
    with torch.no_grad():
        wrapper.coefficient.bias.fill_(1.0)
    residual_rank = int(torch.linalg.matrix_rank(_jacobian_matrix(wrapper, x, t, task)).item())
    if residual_rank != x.numel():
        raise AssertionError(f"residual field rank {residual_rank} is not full rank {x.numel()}")

    # With loss <v(x), x>, the coefficient bias derivative includes ||x||^2
    # and therefore must be nonzero for this sampled nonzero map.
    wrapper.zero_grad(set_to_none=True)
    output = wrapper(x, t, task)
    (output * x).sum().backward()
    coefficient_gradient = wrapper.coefficient.bias.grad
    if coefficient_gradient is None or not torch.isfinite(coefficient_gradient).all() or \
            float(coefficient_gradient.abs().max()) == 0.0:
        raise AssertionError("residual time/task coefficient has no finite nonzero gradient")
    if not torch.isfinite(output).all():
        raise AssertionError("residual field produced non-finite output")
    return {
        "passed": True,
        "shape": list(output.shape),
        "old_jacobian_rank": old_rank,
        "residual_jacobian_rank": residual_rank,
        "toy_dimension": x.numel(),
        "residual_coefficient_gradient": float(coefficient_gradient.detach().abs().max()),
        "parameters": {
            "base": parameter_count(wrapper.base),
            "residual_wrapper": parameter_count(wrapper),
            "added": parameter_count(wrapper) - parameter_count(wrapper.base),
        },
    }


if __name__ == "__main__":
    print(selfcheck())
