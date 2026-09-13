from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FixedVAE(nn.Module):
    """Unconditional VAE for flattened 32x32 importance maps."""

    def __init__(self, mask_dim: int = 1024, latent_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.mask_dim = int(mask_dim)
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.cond_dim = 0
        self.enc_fc1 = nn.Linear(self.mask_dim, self.hidden)
        self.enc_fc2 = nn.Linear(self.hidden, self.hidden)
        self.enc_mu = nn.Linear(self.hidden, self.latent_dim)
        self.enc_logvar = nn.Linear(self.hidden, self.latent_dim)
        self.dec_fc1 = nn.Linear(self.latent_dim, self.hidden)
        self.dec_fc2 = nn.Linear(self.hidden, self.hidden)
        self.dec_out = nn.Linear(self.hidden, self.mask_dim)

    def encode(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.enc_fc1(value))
        hidden = F.relu(self.enc_fc2(hidden))
        return self.enc_mu(hidden), self.enc_logvar(hidden)

    def decode(self, latent: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"latent must have shape (N, {self.latent_dim})")
        if condition is not None and condition.shape != (latent.shape[0], 0):
            raise ValueError("FixedVAE is unconditional and accepts only an empty condition")
        hidden = F.relu(self.dec_fc1(latent))
        hidden = F.relu(self.dec_fc2(hidden))
        return self.dec_out(hidden)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(value)
        latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(latent), mu, logvar


def vae_loss(logits: torch.Tensor, target: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none").sum(dim=1).mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
    return reconstruction + beta * kl, reconstruction, kl


def load_model(path: str, device: str | torch.device) -> FixedVAE:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = FixedVAE(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval().requires_grad_(False)

