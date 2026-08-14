"""Conditional VAE over binary mask space, conditioned on (kernel, offset).

Masks are flattened binary vectors of size mask_dim = L * H (0/1 entries).

The condition is the *continuous* MA kernel size and the target offset:
each is normalized (k / config.CVAE_COND_SCALE_K, s / config.CVAE_COND_SCALE_S)
and concatenated directly to the encoder/decoder inputs, so the model can be
queried for any (k, s) value, including held-out ones that were never seen in
training (interpolation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config


class CVAE(nn.Module):
    """Conditional variational autoencoder ((k, s) condition).

    Args:
        mask_dim: flattened mask dimension (L * H).
        latent_dim: size of the latent vector z.
        hidden: width of the MLP hidden layers.
    """

    def __init__(self, mask_dim: int, latent_dim: int, hidden: int):
        super().__init__()
        self.mask_dim = mask_dim
        self.latent_dim = latent_dim

        enc_in = mask_dim + 2
        self.enc_fc1 = nn.Linear(enc_in, hidden)
        self.enc_fc2 = nn.Linear(hidden, hidden)
        self.enc_mu = nn.Linear(hidden, latent_dim)
        self.enc_logvar = nn.Linear(hidden, latent_dim)

        dec_in = latent_dim + 2
        self.dec_fc1 = nn.Linear(dec_in, hidden)
        self.dec_fc2 = nn.Linear(hidden, hidden)
        self.dec_out = nn.Linear(hidden, mask_dim)

    def condition(self, kernels: torch.Tensor,
                  offsets: torch.Tensor) -> torch.Tensor:
        """(N,) kernel & offset tensors -> (N, 2) normalized conditions."""
        return torch.stack([kernels.float() / config.CVAE_COND_SCALE_K,
                            offsets.float() / config.CVAE_COND_SCALE_S],
                           dim=-1)

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple:
        h = F.relu(self.enc_fc1(torch.cat([x, c], dim=-1)))
        h = F.relu(self.enc_fc2(h))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """(N, latent) x (N, 2) cond -> raw logits (N, mask_dim)."""
        h = F.relu(self.dec_fc1(torch.cat([z, c], dim=-1)))
        h = F.relu(self.dec_fc2(h))
        return self.dec_out(h)

    def forward(self, x: torch.Tensor, kernels: torch.Tensor,
                offsets: torch.Tensor) -> tuple:
        """x: (N, mask_dim) 0/1, kernels/offsets: (N,) float
        -> (logits, mu, logvar)."""
        c = self.condition(kernels, offsets)
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, c)
        return logits, mu, logvar

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, kernels: torch.Tensor,
                    offsets: torch.Tensor) -> torch.Tensor:
        """Modal (per-pixel argmax) reconstruction in {0, 1}.

        NOTE: for Bernoulli masks with active fraction p << 0.5 the modal
        reconstruction is the all-zero mask (each bit is individually most
        likely zero). Use prob() / reconstruct_sample() for meaningful
        reconstructions.
        """
        logits, _, _ = self.forward(x, kernels, offsets)
        return (torch.sigmoid(logits) > 0.5).float()

    @torch.no_grad()
    def prob(self, x: torch.Tensor, kernels: torch.Tensor,
             offsets: torch.Tensor) -> torch.Tensor:
        """Per-pixel activation probabilities p(mask=1) -> (N, mask_dim).

        Uses the deterministic posterior mode z = mu(z | x, k, s).
        """
        c = self.condition(kernels, offsets)
        mu, _ = self.encode(x, c)
        logits = self.decode(mu, c)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def reconstruct_sample(self, x: torch.Tensor, kernels: torch.Tensor,
                           offsets: torch.Tensor,
                           generator: torch.Generator | None = None,
                           ) -> torch.Tensor:
        """Stochastic reconstruction: Bernoulli(p) from the posterior mode."""
        p = self.prob(x, kernels, offsets)
        return torch.bernoulli(p, generator=generator)

    @torch.no_grad()
    def sample_topk(self, kernels: torch.Tensor, offsets: torch.Tensor,
                    n_per_kernel: int, k_active: int,
                    generator: torch.Generator | None = None,
                    ) -> torch.Tensor:
        """Sample masks with exactly ``k_active`` ones per mask.

        The positions are chosen as the top-*k* entries in the decoder
        probability map, so the structure (bottom-row concentration) is
        preserved while the output density matches the target sparsity.
        """
        c = self.condition(kernels, offsets)
        z = torch.randn(c.size(0), self.latent_dim,
                        device=c.device, generator=generator)
        z = z.unsqueeze(1).expand(-1, n_per_kernel, -1).reshape(
            -1, self.latent_dim)
        c = c.unsqueeze(1).expand(-1, n_per_kernel, -1).reshape(-1, 2)
        logits = self.decode(z, c)
        p = torch.sigmoid(logits)
        _, top = p.topk(k_active, dim=-1)
        out = torch.zeros_like(p)
        out.scatter_(-1, top, 1.0)
        return out

    @torch.no_grad()
    def sample_det(self, kernels: torch.Tensor, offsets: torch.Tensor,
                   k_active: int) -> torch.Tensor:
        """Deterministic top-k masks using z = 0 (the prior mode).

        Useful when training has collapsed the KL (common for structured
        targets with a strong deterministic component).  Returns masks with
        exactly ``k_active`` ones chosen from the decoder p-map.
        """
        c = self.condition(kernels, offsets)                   # (N, 2)
        z = torch.zeros(c.size(0), self.latent_dim, device=c.device)
        logits = self.decode(z, c)
        p = torch.sigmoid(logits)
        _, top = p.topk(k_active, dim=-1)
        out = torch.zeros_like(p)
        out.scatter_(-1, top, 1.0)
        return out

    @torch.no_grad()
    def sample(self, kernels: torch.Tensor, offsets: torch.Tensor,
               n_per_kernel: int,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample n_per_kernel masks for each (k, s) -> (N*M, mask_dim).

        Masks are drawn from Bernoulli(p) with p = sigmoid(decoder logits),
        not thresholded, so the sampled sparsity matches the data (~p_active).
        """
        c = self.condition(kernels, offsets)                     # (N, 2)
        z = torch.randn(c.size(0), self.latent_dim,
                        device=c.device, generator=generator)
        z = z.unsqueeze(1).expand(-1, n_per_kernel, -1).reshape(
            -1, self.latent_dim)
        c = c.unsqueeze(1).expand(-1, n_per_kernel, -1).reshape(-1, 2)
        logits = self.decode(z, c)
        p = torch.sigmoid(logits)
        return torch.bernoulli(p, generator=generator)


def cvae_loss(logits: torch.Tensor, x: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor,
              beta: float = 1.0) -> tuple:
    """Return (total, recon_bce, kl) as (dim-0 scalar) tensors."""
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="mean")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    total = recon + beta * kl
    return total, recon, kl


def load_selected_masks(kernels, offsets, ckpt_root) -> tuple:
    """Load best10pct masks from each (kernel, offset) dir.

    Returns (x, y, info): x (N, mask_dim) float, y (N, 2) float [k, s],
    info (n_pairs,) dict with kernel, offset, n_masks, sparsity.
    """
    x_parts, y_parts, info = [], [], []
    for k, s in zip(kernels, offsets):
        path = config.kernel_dir(k, s) / "best10pct.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Runs scripts/03_select.sh first: missing {path}")
        d = torch.load(path, weights_only=True)
        m = d["masks"]                          # (n, L, H) 0/1
        n = m.size(0)
        x_parts.append(m.reshape(n, -1).float())
        ks = torch.stack([torch.full((n,), float(k)),
                          torch.full((n,), float(s))], dim=1)
        y_parts.append(ks)
        info.append({"kernel": k, "offset": s, "n_masks": n,
                     "sparsity": (m == 1).float().mean().item()})
        print(f"[cvae] kernel={k} offset={s}: {n} masks, "
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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader, train_idx, val_idx


def cvae_loss_importance(logits: torch.Tensor, importance: torch.Tensor,
                         mu: torch.Tensor, logvar: torch.Tensor,
                         beta: float = 1.0) -> tuple:
    """MSE loss on continuous importance targets [0, 1]."""
    recon = F.mse_loss(torch.sigmoid(logits), importance, reduction="mean")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    total = recon + beta * kl
    return total, recon, kl


def load_importance_maps(kernels, offsets, ckpt_root) -> tuple:
    """Load per-(k,s) importance.pt (continuous [0,1] maps)."""
    x_parts, y_parts, info = [], [], []
    for k, s in zip(kernels, offsets):
        path = config.kernel_dir(k, s) / "importance.pt"
        if not path.exists():
            raise FileNotFoundError(f"Run evaluation/importance.py first: {path}")
        d = torch.load(path, weights_only=True)
        imp = d["importance"]
        n = imp.size(0)
        x_parts.append(imp.reshape(n, -1).float())
        ks = torch.stack([torch.full((n,), float(k)),
                          torch.full((n,), float(s))], dim=1)
        y_parts.append(ks)
        info.append({"kernel": k, "offset": s, "n_masks": n,
                     "sparsity": 0.0})  # importance, not binary
        print(f"[cvae] kernel={k} offset={s}: {n} importance maps")
    return torch.cat(x_parts), torch.cat(y_parts), info


def generate_ideal_masks(kernels, n_per_kernel: int, offsets,
                         in_dim: int, hidden: int,
                         noise: float = 0.05, seed: int = 42) -> tuple:
    """Create synthetic masks concentrated on rows [L-s-k : L-s).

    For each (k, s) the base mask has support == 1 on all ``hidden``
    columns of the last (k) rows located right above the offset region
    (the theoretically ideal support for the shifted MA(k, s) task).
    Independent Bernoulli(noise) bit-flips are added so that every mask
    looks slightly different (while keeping the strong bottom-row
    structure).

    Returns (x, y, info) with the same format as load_selected_masks.
    """
    g = torch.Generator().manual_seed(seed)
    x_parts, y_parts, info = [], [], []
    for k, s in zip(kernels, offsets):
        m = torch.zeros(n_per_kernel, in_dim, hidden)
        m[:, in_dim - s - k: in_dim - s] = 1.0
        if noise > 0:
            flip = (torch.rand(m.size(0), m.size(1), m.size(2),
                               generator=g) < noise).float()
            m = (m + flip) % 2
        st = m.float().mean().item()
        x_parts.append(m.reshape(n_per_kernel, -1).float())
        ks = torch.stack([torch.full((n_per_kernel,), float(k)),
                          torch.full((n_per_kernel,), float(s))], dim=1)
        y_parts.append(ks)
        info.append({"kernel": k, "offset": s, "n_masks": n_per_kernel,
                     "sparsity": st})
        print(f"[cvae] kernel={k} offset={s}: {n_per_kernel} ideal masks, "
              f"sparsity={st:.3f}")
    return torch.cat(x_parts), torch.cat(y_parts), info