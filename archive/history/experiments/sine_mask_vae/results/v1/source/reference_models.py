"""The repository's grouped NF encoder and shared VAE, adapted to 8x2."""

import torch
from torch import nn
from torch.nn import functional as F

from overlapping_groups.models.vae import GroupedLinear, GroupedNF

from core import COLUMNS, DIMENSION, KEEP, ROWS


def hard_mask(scores, straight_through=False):
    flat = scores.flatten(-2)
    indices = flat.detach().argsort(dim=-1, descending=True, stable=True)[
        ..., :KEEP
    ]
    hard = torch.zeros_like(flat).scatter_(-1, indices, 1)
    hard = hard.reshape_as(scores)
    if straight_through:
        soft = scores.sigmoid()
        return hard + soft - soft.detach()
    return hard


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


def objective(output, target, beta):
    scores, mu, logvar = output
    hard = hard_mask(scores, True)
    costs = []
    for labels in (target, target.flip(-1)):
        bce = F.binary_cross_entropy_with_logits(
            scores,
            labels,
            reduction="none",
            pos_weight=scores.new_tensor((DIMENSION - KEEP) / KEEP),
        ).mean((-2, -1))
        mse = (hard - labels).square().mean((-2, -1))
        costs.append(0.8 * bce + 0.2 * mse)
    reconstruction = torch.minimum(*costs).mean()
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
    loss = reconstruction + beta * kl
    with torch.no_grad():
        predicted = hard_mask(scores)
        exact = torch.maximum(
            (predicted == target).all(-1).all(-1).float(),
            (predicted == target.flip(-1)).all(-1).all(-1).float(),
        ).mean()
    return loss, {
        "loss": float(loss.detach()),
        "reconstruction": float(reconstruction.detach()),
        "kl": float(kl.detach()),
        "exact": float(exact),
    }
