"""The reference NF encoder, shared VAE decoder and partwise objective.

GroupedLinear/GroupedNF are the unchanged operations from
overlapping_groups/models/vae.py. Keeping these small layers here makes this
experiment independent of the temporary prototype and other experiment code.
Parameter names, initialization order and decoder shapes match the six saved
reference checkpoints.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .core import COLUMNS, DIMENSION, KEEP, ROWS


class GroupedLinear(nn.Module):
    """Independent task-specific linear maps evaluated with batched GEMM."""

    def __init__(self, tasks, incoming, outgoing, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(tasks, outgoing, incoming))
        self.bias = (
            nn.Parameter(torch.empty(tasks, outgoing)) if bias else None
        )
        bound = 1 / math.sqrt(incoming)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        shape = x.shape[:-1]
        result = torch.bmm(
            x.reshape(x.shape[0], -1, x.shape[-1]), self.weight.transpose(1, 2)
        )
        if self.bias is not None:
            result = result + self.bias[:, None]
        return result.reshape(*shape, self.weight.shape[1])


class GroupedNF(nn.Module):
    """Column-equivariant Neural Functional layer for independent tasks."""

    def __init__(self, tasks, incoming, outgoing):
        super().__init__()
        self.local = GroupedLinear(tasks, incoming, outgoing)
        self.context = GroupedLinear(tasks, incoming, outgoing, bias=False)

    def forward(self, columns):
        return self.local(columns) + self.context(
            columns.mean(2, keepdim=True)
        )


class Encoder(nn.Module):
    def __init__(self, tasks, latent_dim):
        super().__init__()
        channels, width = 8, 64
        self.first = GroupedNF(tasks, ROWS * 2, ROWS * channels)
        self.second = GroupedNF(tasks, ROWS * channels, ROWS * channels)
        self.edges = GroupedLinear(tasks, ROWS * channels, width)
        self.posterior = GroupedLinear(tasks, width, width)
        self.moments = GroupedLinear(tasks, width, latent_dim * 2)

    def forward(self, features):
        columns = features.permute(0, 1, 3, 2, 4).flatten(-2)
        columns = F.gelu(self.first(columns))
        columns = F.gelu(self.second(columns))
        hidden = F.gelu(self.edges(columns.mean(2)))
        hidden = F.gelu(self.posterior(hidden))
        mu, logvar = self.moments(hidden).chunk(2, -1)
        return mu, logvar.clamp(-12, 8)


class MaskVAE(nn.Module):
    def __init__(self, tasks, latent_dim=16):
        super().__init__()
        self.encoder = Encoder(tasks, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, DIMENSION),
        )

    def decode(self, latent):
        return self.decoder(latent).reshape(*latent.shape[:-1], ROWS, COLUMNS)

    def forward(self, features, sample=True):
        mu, logvar = self.encoder(features)
        latent = mu
        if sample:
            latent = mu + torch.randn_like(mu) * (logvar * 0.5).exp()
        return self.decode(latent), mu, logvar


def hard_mask(scores, straight_through=False):
    flat = scores.flatten(-2)
    indices = flat.detach().argsort(dim=-1, descending=True, stable=True)[
        ..., :KEEP
    ]
    hard = torch.zeros_like(flat).scatter_(-1, indices, 1).reshape_as(scores)
    if straight_through:
        soft = scores.sigmoid()
        return hard + soft - soft.detach()
    return hard


def partwise_forward(model, features):
    """Reconstruct each real source column with independent foreign code."""
    mu, logvar = model.encoder(features)
    latent = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
    nuisance = torch.randn_like(latent)
    first = model.decode(torch.cat((latent[..., :8], nuisance[..., 8:]), -1))
    second = model.decode(torch.cat((nuisance[..., :8], latent[..., 8:]), -1))
    return torch.stack((first[..., 0], second[..., 1]), -1), mu, logvar


def objective(output, target, beta):
    """Canonical labelled-output reconstruction and the reference KL penalty."""
    scores, mu, logvar = output
    bce = F.binary_cross_entropy_with_logits(
        scores, target, pos_weight=scores.new_tensor(7)
    )
    mse = (hard_mask(scores, True) - target).square().mean()
    reconstruction = 0.8 * bce + 0.2 * mse
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
    loss = reconstruction + beta * kl
    exact = (hard_mask(scores) == target).all(-1).all(-1).float().mean()
    recall = (hard_mask(scores) * target).sum((-1, -2)).div(KEEP).mean()
    return loss, {
        "loss": float(loss.detach()),
        "reconstruction": float(reconstruction.detach()),
        "kl": float(kl.detach()),
        "exact": float(exact.detach()),
        "recall_at_2": float(recall.detach()),
    }
