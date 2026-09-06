"""Length-conditioned VAE for 32x32 unsigned first-layer importance maps.

The only condition is a *continuous scalar* pattern length.  In particular,
pattern bits are deliberately absent: a decoder trained here may use length,
but cannot memorize a pattern-specific map.  The scalar is normalized from the
protocol range [3, 8] to [-1, 1], so intermediate values such as 5.5 are a
valid decoder input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


LENGTH_MIN = 3.0
LENGTH_MAX = 8.0


def normalize_length(lengths: torch.Tensor | float | int,
                     *, device: torch.device | None = None) -> torch.Tensor:
    """Convert length(s) to a float ``(N, 1)`` condition in [-1, 1].

    This intentionally does not round or convert to an integer.  The allowed
    *training* lengths are validated by the trainer; this lower-level function
    also accepts interpolation points (for example 5.5).
    """
    result = torch.as_tensor(lengths, dtype=torch.float32, device=device)
    if result.ndim == 0:
        result = result.reshape(1, 1)
    elif result.ndim == 1:
        result = result.unsqueeze(-1)
    elif result.ndim != 2 or result.shape[-1] != 1:
        raise ValueError("lengths must be a scalar, (N,), or (N, 1)")
    return 2.0 * (result - LENGTH_MIN) / (LENGTH_MAX - LENGTH_MIN) - 1.0


@dataclass(frozen=True)
class CVAEConfig:
    mask_dim: int = 1024
    latent_dim: int = 32
    hidden: int = 256

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class LengthCVAE(nn.Module):
    """VAE whose encoder and decoder both receive a scalar length condition."""

    def __init__(self, mask_dim: int = 1024, latent_dim: int = 32,
                 hidden: int = 256) -> None:
        super().__init__()
        if min(mask_dim, latent_dim, hidden) <= 0:
            raise ValueError("mask_dim, latent_dim, and hidden must be positive")
        self.mask_dim = int(mask_dim)
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        # One scalar condition, never an embedding or one-hot vector.
        self.enc_fc1 = nn.Linear(self.mask_dim + 1, self.hidden)
        self.enc_fc2 = nn.Linear(self.hidden, self.hidden)
        self.enc_mu = nn.Linear(self.hidden, self.latent_dim)
        self.enc_logvar = nn.Linear(self.hidden, self.latent_dim)
        self.dec_fc1 = nn.Linear(self.latent_dim + 1, self.hidden)
        self.dec_fc2 = nn.Linear(self.hidden, self.hidden)
        self.dec_out = nn.Linear(self.hidden, self.mask_dim)

    def model_config(self) -> dict[str, int]:
        return CVAEConfig(self.mask_dim, self.latent_dim, self.hidden).to_dict()

    def condition(self, lengths: torch.Tensor | float | int,
                  n: int | None = None) -> torch.Tensor:
        """Return normalized scalar conditions, optionally broadcasting one length."""
        device = next(self.parameters()).device
        c = normalize_length(lengths, device=device)
        if n is None:
            return c
        if c.shape[0] == 1:
            return c.expand(n, -1)
        if c.shape[0] != n:
            raise ValueError(f"got {c.shape[0]} lengths for batch of {n}")
        return c

    def encode(self, x: torch.Tensor, lengths: torch.Tensor | float | int) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[-1] != self.mask_dim:
            raise ValueError(f"x must have shape (N, {self.mask_dim})")
        c = self.condition(lengths, x.shape[0]).to(dtype=x.dtype)
        h = F.relu(self.enc_fc1(torch.cat((x, c), dim=-1)))
        h = F.relu(self.enc_fc2(h))
        return self.enc_mu(h), self.enc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode_lengths(self, z: torch.Tensor,
                       lengths: torch.Tensor | float | int) -> torch.Tensor:
        """Decode latents at continuous scalar length(s), returning raw logits."""
        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError(f"z must have shape (N, {self.latent_dim})")
        c = self.condition(lengths, z.shape[0]).to(dtype=z.dtype)
        return self.decode_condition(z, c)

    def decode_condition(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Decode a normalized scalar condition; internal helper for sampling."""
        if condition.shape != (z.shape[0], 1):
            raise ValueError("condition must have shape (N, 1)")
        h = F.relu(self.dec_fc1(torch.cat((z, condition), dim=-1)))
        h = F.relu(self.dec_fc2(h))
        return self.dec_out(h)

    def decode(self, z: torch.Tensor, lengths: torch.Tensor | float | int) -> torch.Tensor:
        """Compatibility alias for :meth:`decode_lengths`."""
        return self.decode_lengths(z, lengths)

    @torch.no_grad()
    def decode_probabilities(self, z: torch.Tensor,
                             lengths: torch.Tensor | float | int) -> torch.Tensor:
        """Decode unsigned importance probabilities with ``sigmoid(logits)``."""
        return torch.sigmoid(self.decode_lengths(z, lengths))

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor | float | int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, lengths)
        return self.decode_lengths(self.reparameterize(mu, logvar), lengths), mu, logvar

    @torch.no_grad()
    def posterior_mean_logits(self, x: torch.Tensor,
                              lengths: torch.Tensor | float | int) -> torch.Tensor:
        """Deterministic reconstruction used for validation/checkpoint selection."""
        mu, _ = self.encode(x, lengths)
        return self.decode_lengths(mu, lengths)

    @torch.no_grad()
    def prior_logits(self, lengths: torch.Tensor | float | int,
                     n: int | None = None,
                     generator: torch.Generator | None = None) -> torch.Tensor:
        c = self.condition(lengths, n)
        z = torch.randn(c.shape[0], self.latent_dim, device=c.device, generator=generator)
        return self.decode_condition(z, c)

    @torch.no_grad()
    def sample_topk(self, lengths: torch.Tensor | float | int, target_k: int,
                    n: int | None = None,
                    generator: torch.Generator | None = None) -> torch.Tensor:
        """Prior samples converted to exact-cardinality hard masks by global top-k."""
        if not 0 < target_k <= self.mask_dim:
            raise ValueError("target_k must lie in [1, mask_dim]")
        c = self.condition(lengths, n)
        z = torch.randn(c.shape[0], self.latent_dim, device=c.device, generator=generator)
        logits = self.decode_condition(z, c)
        output = torch.zeros_like(logits)
        output.scatter_(1, logits.topk(target_k, dim=1).indices, 1.0)
        return output


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean analytic KL(q(z|x) || N(0, I)) over a batch."""
    return -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=-1).mean()


def cvae_loss(logits: torch.Tensor, targets: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor,
              beta: float = 0.1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BCE summed per map then averaged over batch, plus ``beta * KL``."""
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    recon = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").sum(dim=-1).mean()
    kl = kl_divergence(mu, logvar)
    return recon + beta * kl, recon, kl


def checkpoint_model(checkpoint: dict[str, Any], device: str | torch.device = "cpu") -> LengthCVAE:
    """Restore a model from a trainer checkpoint, rejecting incomplete metadata."""
    try:
        model = LengthCVAE(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
    except KeyError as error:
        raise KeyError("checkpoint lacks model_config or model_state") from error
    return model
