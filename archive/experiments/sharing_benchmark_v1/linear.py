"""Shared linear algebra for observed-data selection and fresh adaptation."""

import numpy as np
import torch


def index_basis(index, codes):
    labels = torch.arange(1, codes + 1, device=index.device)
    return (index[..., None].abs() == labels).to(torch.float32) * index.sign()[
        ..., None
    ]


def design(basis, x):
    return torch.einsum("soik,tni->stnok", basis, x).flatten(2, 3)


def fit_codes(features, y, ridge=0.0):
    target = y.flatten(1, 2)
    gram = features.transpose(-1, -2) @ features / features.shape[-2]
    cross = (features.transpose(-1, -2) @ target[None, ..., None])[..., 0]
    cross /= features.shape[-2]
    if ridge:
        identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        return torch.linalg.solve(gram + ridge * identity, cross)
    return (torch.linalg.pinv(gram, hermitian=True, rtol=1e-10) @ cross[..., None])[
        ..., 0
    ]


def adapted(basis, x, y, ridge=0.0):
    return fit_codes(design(basis, x), y, ridge)


def prediction(basis, codes, x):
    weights = torch.einsum("soik,stk->stoi", basis, codes)
    return torch.einsum("stoi,tni->stno", weights, x)


@torch.no_grad()
def adaptation_errors(basis, support_x, support_y, query_x, query_y, chunk=32):
    errors = []
    for local in basis.split(chunk):
        local = local.double()
        codes = adapted(local, support_x.double(), support_y.double())
        residual = prediction(local, codes, query_x.double()) - query_y.double()[None]
        errors.append(residual.square().mean((-1, -2)))
    return torch.cat(errors)


def load_observations(folder, device):
    with np.load(folder / "observations.npz") as archive:
        return {key: torch.tensor(archive[key], device=device) for key in archive.files}


class Adam:
    """Independent first-axis optimizers with per-configuration learning rates."""

    def __init__(self, parameters, rates):
        self.parameters = parameters
        self.rates = rates
        self.first = [torch.zeros_like(parameter) for parameter in parameters]
        self.second = [torch.zeros_like(parameter) for parameter in parameters]
        self.steps = 0

    def step(self, gradients):
        self.steps += 1
        with torch.no_grad():
            for parameter, gradient, first, second in zip(
                self.parameters, gradients, self.first, self.second
            ):
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("Non-finite optimization gradient")
                norm = gradient.flatten(1).norm(dim=1).clamp_min(1e-12)
                scale = (100 / norm).clamp_max(1)
                gradient = gradient * scale.reshape((-1,) + (1,) * (gradient.ndim - 1))
                first.mul_(0.9).add_(gradient, alpha=0.1)
                second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
                direction = (first / (1 - 0.9**self.steps)) / (
                    (second / (1 - 0.999**self.steps)).sqrt() + 1e-8
                )
                rates = self.rates.reshape((-1,) + (1,) * (parameter.ndim - 1))
                parameter.add_(-rates * direction)
