"""Unconditional VAE over binary/importance mask space.

Masks are flattened vectors of size mask_dim = SEQ_LEN * H (0/1 or [0,1]
importance entries).

This is an *unconditional* VAE: config.CVAE_COND_DIM == 0, so the mask is
shared across all patterns (no pattern condition).  The condition() helper
returns an empty (N, 0) tensor, keeping the encoder/decoder signatures and
call sites unchanged so callers may still pass a dummy pattern tensor.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config


class CVAE(nn.Module):
    """Unconditional variational autoencoder (mask shared, no pattern cond).

    Args:
        mask_dim: flattened mask dimension (SEQ_LEN * H).
        latent_dim: size of the latent vector z.
        hidden: width of the MLP hidden layers.
        cond_dim: length of the condition vector (default CVAE_COND_DIM = 0).
    """

    def __init__(self, mask_dim: int, latent_dim: int, hidden: int,
                 cond_dim: int | None = None):
        super().__init__()
        self.mask_dim = mask_dim
        self.latent_dim = latent_dim
        self.cond_dim = config.CVAE_COND_DIM if cond_dim is None else cond_dim

        enc_in = mask_dim + self.cond_dim
        self.enc_fc1 = nn.Linear(enc_in, hidden)
        self.enc_fc2 = nn.Linear(hidden, hidden)
        self.enc_mu = nn.Linear(hidden, latent_dim)
        self.enc_logvar = nn.Linear(hidden, latent_dim)

        dec_in = latent_dim + self.cond_dim
        self.dec_fc1 = nn.Linear(dec_in, hidden)
        self.dec_fc2 = nn.Linear(hidden, hidden)
        self.dec_out = nn.Linear(hidden, mask_dim)

    def condition(self, patterns_pm1: torch.Tensor) -> torch.Tensor:
        n = patterns_pm1.size(0)
        if self.cond_dim == 0:
            return patterns_pm1.new_zeros(n, 0)
        return patterns_pm1.float().clone()

    def prior_inputs(self, patterns_pm1: torch.Tensor, n_per: int,
                     generator: torch.Generator | None = None) -> tuple:
        """Return independent prior latents and repeated conditions."""
        c = self.condition(patterns_pm1)
        n_total = c.size(0) * n_per
        z = torch.randn(n_total, self.latent_dim, device=c.device,
                        generator=generator)
        return z, c.repeat_interleave(n_per, dim=0)

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple:
        h = F.relu(self.enc_fc1(torch.cat([x, c], dim=-1)))
        h = F.relu(self.enc_fc2(h))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """(N, latent) x (N, cond_dim) cond -> raw logits (N, mask_dim)."""
        h = F.relu(self.dec_fc1(torch.cat([z, c], dim=-1)))
        h = F.relu(self.dec_fc2(h))
        return self.dec_out(h)

    def importance(self, logits: torch.Tensor) -> torch.Tensor:
        """Map decoder logits to non-negative importance values."""
        return torch.tanh(logits).abs() if config.SIGNED_IMPORTANCE \
            else torch.sigmoid(logits)

    def forward(self, x: torch.Tensor,
                patterns_pm1: torch.Tensor) -> tuple:
        """x: (N, mask_dim) 0/1, patterns_pm1: (N, 4) +-1
        -> (logits, mu, logvar)."""
        c = self.condition(patterns_pm1)
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, c)
        return logits, mu, logvar

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor,
                    patterns_pm1: torch.Tensor) -> torch.Tensor:
        """Modal (per-pixel argmax) reconstruction in {0, 1}.

        NOTE: for Bernoulli masks with active fraction p << 0.5 the modal
        reconstruction is the all-zero mask (each bit is individually most
        likely zero). Use prob() / reconstruct_sample() for meaningful
        reconstructions.
        """
        logits, _, _ = self.forward(x, patterns_pm1)
        return (self.importance(logits) > 0.5).float()

    @torch.no_grad()
    def prob(self, x: torch.Tensor,
             patterns_pm1: torch.Tensor) -> torch.Tensor:
        """Per-pixel activation probabilities p(mask=1) -> (N, mask_dim).

        Uses the deterministic posterior mode z = mu(z | x, pattern).
        """
        c = self.condition(patterns_pm1)
        mu, _ = self.encode(x, c)
        logits = self.decode(mu, c)
        return self.importance(logits)

    @torch.no_grad()
    def reconstruct_sample(self, x: torch.Tensor,
                           patterns_pm1: torch.Tensor,
                           generator: torch.Generator | None = None,
                           ) -> torch.Tensor:
        """Stochastic reconstruction: Bernoulli(p) from the posterior mode."""
        p = self.prob(x, patterns_pm1)
        return torch.bernoulli(p, generator=generator)

    @torch.no_grad()
    def sample_topk(self, patterns_pm1: torch.Tensor, n_per: int,
                    k_active: int,
                    generator: torch.Generator | None = None,
                    ) -> torch.Tensor:
        """Sample n_per diverse masks per pattern, each with k_active ones.

        z ~ N(0, I) gives different decoder outputs per sample, so the
        top-k positions vary across the n_per masks.  If SIGNED_IMPORTANCE,
        importance = |tanh(logits)|; otherwise sigmoid(logits).
        """
        z, c = self.prior_inputs(patterns_pm1, n_per, generator)
        logits = self.decode(z, c)
        p = self.importance(logits)
        _, top = p.topk(k_active, dim=-1)
        out = torch.zeros_like(p)
        out.scatter_(-1, top, 1.0)
        return out

    @torch.no_grad()
    def sample_det(self, patterns_pm1: torch.Tensor,
                   k_active: int) -> torch.Tensor:
        """Deterministic top-k masks using z = 0 (the prior mode).

        If config.SIGNED_IMPORTANCE: decoder uses tanh, so importance = |tanh|.
        Otherwise: sigmoid, importance = sigmoid(logits).
        """
        c = self.condition(patterns_pm1)                     # (N, cond_dim)
        z = torch.zeros(c.size(0), self.latent_dim, device=c.device)
        logits = self.decode(z, c)
        p = self.importance(logits)
        _, top = p.topk(k_active, dim=-1)
        out = torch.zeros_like(p)
        out.scatter_(-1, top, 1.0)
        return out

    @torch.no_grad()
    def sample(self, patterns_pm1: torch.Tensor, n_per: int,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample n_per masks for each pattern -> (N*M, mask_dim).

        Masks are drawn from Bernoulli(p) with p = sigmoid(decoder logits),
        not thresholded, so the sampled sparsity matches the data (~p_active).
        """
        z, c = self.prior_inputs(patterns_pm1, n_per, generator)
        logits = self.decode(z, c)
        p = self.importance(logits)
        return torch.bernoulli(p, generator=generator)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Return mean KL(q(z|x) || N(0, I))."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()


def cvae_loss(logits: torch.Tensor, x: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor,
              beta: float = 1.0) -> tuple:
    """Return total, reconstruction BCE, and KL."""
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="mean")
    kl = kl_divergence(mu, logvar)
    total = recon + beta * kl
    return total, recon, kl


def _pattern_to_y(pat: str, n: int) -> torch.Tensor:
    """(n,) copies of a pattern as a (n, PATTERN_LEN) +-1 tensor."""
    pm1 = config.pattern_to_pm1(pat).float()
    return pm1.unsqueeze(0).expand(n, -1)


def checkpoint_file(ckpt_root, pattern: str, name: str) -> Path:
    """Return a pattern-specific checkpoint path."""
    return Path(ckpt_root) / f"pattern_{pattern}" / name


def load_selected_masks(patterns: list[str], ckpt_root) -> tuple:
    """Load best10pct masks from each pattern dir.

    Returns (x, y, info): x (N, mask_dim) float, y (N, 4) +-1 pattern,
    info (n_pats,) dict with pattern, n_masks, sparsity.
    """
    x_parts, y_parts, info = [], [], []
    for pat in patterns:
        path = checkpoint_file(ckpt_root, pat, "best10pct.pt")
        if not path.exists():
            raise FileNotFoundError(
                f"Runs scripts/03_select.sh first: missing {path}")
        d = torch.load(path, weights_only=True)
        m = d["masks"]                          # (n, SEQ_LEN, H) 0/1
        n = m.size(0)
        x_parts.append(m.reshape(n, -1).float())
        y_parts.append(_pattern_to_y(pat, n))
        info.append({"pattern": pat, "n_masks": n,
                     "sparsity": (m == 1).float().mean().item()})
        print(f"[cvae] pattern={pat}: {n} masks, "
              f"{info[-1]['sparsity']:.3f} ones")
    return torch.cat(x_parts), torch.cat(y_parts), info


def make_loaders(x, y, val_fraction, batch_size, seed) -> tuple:
    """Shuffled train/val TensorDataLoaders."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.size(0), generator=g)
    n_val = int(x.size(0) * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds = TensorDataset(x[train_idx], y[train_idx])
    val_ds = TensorDataset(x[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader, train_idx, val_idx


def make_importance_loss(loss: str, reduction: str):
    """Build an importance-map loss callable.

    Returns f(logits, importance, mu, logvar, beta) -> (total, recon, kl).

    loss="mse": pred = sigmoid(logits) (tanh if config.SIGNED_IMPORTANCE),
        recon = MSE(pred, importance).
    loss="bce": recon = BCE_with_logits(logits, importance); importance
        targets are soft labels in [0, 1], which is valid for BCE.

    reduction="mean": recon averaged over all elements.
    reduction="sum": recon summed over the mask dim (dim=-1) and averaged
        over the batch: F.mse_loss(..., reduction="none").sum(dim=-1).mean()
        (same for BCE with logits).

    kl is identical to the existing definition. total = recon + beta * kl.
    """
    if loss not in ("mse", "bce"):
        raise ValueError(f"loss must be 'mse' or 'bce', got {loss!r}")
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")

    def f(logits: torch.Tensor, importance: torch.Tensor,
          mu: torch.Tensor, logvar: torch.Tensor,
          beta: float = 1.0) -> tuple:
        if loss == "bce":
            if reduction == "mean":
                recon = F.binary_cross_entropy_with_logits(
                    logits, importance, reduction="mean")
            else:
                recon = F.binary_cross_entropy_with_logits(
                    logits, importance, reduction="none").sum(dim=-1).mean()
        else:  # mse
            if config.SIGNED_IMPORTANCE:
                pred = torch.tanh(logits)
            else:
                pred = torch.sigmoid(logits)
            if reduction == "mean":
                recon = F.mse_loss(pred, importance, reduction="mean")
            else:
                recon = F.mse_loss(pred, importance,
                                   reduction="none").sum(dim=-1).mean()
        kl = kl_divergence(mu, logvar)
        total = recon + beta * kl
        return total, recon, kl

    return f


def load_importance_maps(patterns: list[str], ckpt_root,
                         importance_name: str = "importance.pt",
                         top_frac: float | None = None) -> tuple:
    """Load per-pattern importance maps (continuous [0,1] entries).

    importance_name: filename in each pattern dir, e.g. "importance.pt" for
    the raw |W1|*mask maps or "importance_aligned.pt" for column-aligned maps.

    top_frac: if set (e.g. 0.1), keep only the lowest-val_loss round(n *
    top_frac) maps per pattern (at least 1), sorted by the stored val_loss.
    Default None keeps all maps (unchanged behavior).
    """
    x_parts, y_parts, info = [], [], []
    for pat in patterns:
        path = checkpoint_file(ckpt_root, pat, importance_name)
        if not path.exists():
            raise FileNotFoundError(f"Run evaluation/importance.py first: {path}")
        d = torch.load(path, weights_only=True)
        imp = d["importance"]
        n = imp.size(0)
        if top_frac is not None:
            n_keep = max(1, round(n * top_frac))
            val_loss = d["val_loss"]
            idx = torch.argsort(val_loss)[:n_keep]
            imp = imp[idx]
            print(f"[cvae] pattern={pat}: {n_keep}/{n} maps "
                  f"(top {top_frac:.0%} by val_loss)")
        x_parts.append(imp.reshape(imp.size(0), -1).float())
        y_parts.append(_pattern_to_y(pat, imp.size(0)))
        info.append({"pattern": pat, "n_masks": imp.size(0),
                     "sparsity": 0.0})  # importance, not binary
        if top_frac is None:
            print(f"[cvae] pattern={pat}: {n} importance maps")
    return torch.cat(x_parts), torch.cat(y_parts), info


def generate_ideal_masks(patterns: list[str], n_per: int,
                         noise: float = 0.05, seed: int = 42) -> tuple:
    """Create synthetic Toeplitz masks, one skeleton per pattern.

    The support follows a Toeplitz structure tied to the hidden index:
    hidden column h -> window w = h % N_WINDOWS, ones on
    rows [w : w + PATTERN_LEN] of column h.  This encodes a fixed,
    pattern-independent skeleton so we can test whether conditioning on the
    pattern changes the map.

    Independent Bernoulli(noise) bit-flips are added so that every mask
    looks slightly different (while keeping the strong skeleton structure).

    Returns (x, y, info) with the same format as load_selected_masks.
    """
    g = torch.Generator().manual_seed(seed)
    x_parts, y_parts, info = [], [], []
    for pat in patterns:
        m = torch.zeros(n_per, config.SEQ_LEN, config.H)
        for h in range(config.H):
            w = h % config.N_WINDOWS
            m[:, w:w + config.PATTERN_LEN, h] = 1.0
        if noise > 0:
            flip = (torch.rand(m.size(0), m.size(1), m.size(2),
                               generator=g) < noise).float()
            m = (m + flip) % 2
        st = m.float().mean().item()
        x_parts.append(m.reshape(n_per, -1).float())
        y_parts.append(_pattern_to_y(pat, n_per))
        info.append({"pattern": pat, "n_masks": n_per,
                     "sparsity": st})
        print(f"[cvae] pattern={pat}: {n_per} ideal masks, "
              f"sparsity={st:.3f}")
    return torch.cat(x_parts), torch.cat(y_parts), info
