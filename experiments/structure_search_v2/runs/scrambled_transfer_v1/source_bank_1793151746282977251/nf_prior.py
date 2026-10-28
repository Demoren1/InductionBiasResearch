"""Hidden-unit invariant NF encoders with a shared structural VAE decoder."""

import torch
from torch import nn


class NFEncoder(nn.Module):
    def __init__(self, inputs, latent):
        super().__init__()
        self.edges = nn.Sequential(
            nn.Linear(3, 8), nn.GELU(), nn.Linear(8, 8), nn.GELU()
        )
        self.body = nn.Sequential(nn.Linear(inputs * 8, 64), nn.GELU())
        self.mu = nn.Linear(64, latent)
        self.logvar = nn.Linear(64, latent)

    def forward(self, features):
        hidden = self.edges(features).mean(2).flatten(1)
        hidden = self.body(hidden)
        return self.mu(hidden), self.logvar(hidden).clamp(-12, 8)


class NFVAE(nn.Module):
    def __init__(self, tasks, inputs, outputs, latent=16):
        super().__init__()
        self.encoders = nn.ModuleList([NFEncoder(inputs, latent) for _ in range(tasks)])
        self.decoder = nn.Sequential(
            nn.Linear(latent, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, outputs),
        )
        self.latent = latent

    def encode(self, features, sample=True):
        moments = [
            encoder(features[task]) for task, encoder in enumerate(self.encoders)
        ]
        mu = torch.stack([item[0] for item in moments])
        logvar = torch.stack([item[1] for item in moments])
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        return self.decoder(z), mu, logvar


def normalized_features(weights, gradient, response):
    features = torch.stack([weights, gradient, response], -1)
    return features / features.abs().amax((1, 2), keepdim=True).clamp_min(1e-8)
