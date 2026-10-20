"""Ablations for canonical labels and independent decoder heads."""

import torch
from torch import nn
from torch.nn import functional as F

from models import MaskVAE, hard_mask, objective as original_objective


class ImprovedVAE(MaskVAE):
    def __init__(self, tasks, factorized=False):
        super().__init__(tasks, latent_dim=16)
        self.factorized = factorized
        if factorized:
            self.decoder = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(8, 128),
                        nn.GELU(),
                        nn.Linear(128, 128),
                        nn.GELU(),
                        nn.Linear(128, 8),
                    )
                    for _ in range(2)
                ]
            )

    def decode(self, latent):
        if not self.factorized:
            return super().decode(latent)
        return torch.stack(
            [
                head(latent[..., index * 8 : (index + 1) * 8])
                for index, head in enumerate(self.decoder)
            ],
            -1,
        )


def partwise_forward(model, features):
    """Reconstruct each real source column with random nuisance coordinates.

    Decoder architecture and target masks remain unchanged. The other half
    of the latent code is independent noise, so it cannot predict this part.
    """
    mu, logvar = model.encoder(features)
    latent = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
    nuisance = torch.randn_like(latent)
    first = model.decode(torch.cat((latent[..., :8], nuisance[..., 8:]), -1))
    second = model.decode(torch.cat((nuisance[..., :8], latent[..., 8:]), -1))
    scores = torch.stack((first[..., 0], second[..., 1]), -1)
    return scores, mu, logvar


def objective(output, target, beta, canonical=False):
    if not canonical:
        return original_objective(output, target, beta)
    scores, mu, logvar = output
    bce = F.binary_cross_entropy_with_logits(
        scores,
        target,
        pos_weight=scores.new_tensor(7),
    )
    mse = (hard_mask(scores, True) - target).square().mean()
    reconstruction = 0.8 * bce + 0.2 * mse
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
    loss = reconstruction + beta * kl
    exact = (hard_mask(scores) == target).all(-1).all(-1).float().mean()
    return loss, {
        "loss": float(loss.detach()),
        "reconstruction": float(reconstruction.detach()),
        "kl": float(kl.detach()),
        "exact": float(exact.detach()),
    }
