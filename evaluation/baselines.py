"""
Baselines for mask generation on the shifted MA task.

- MeanImportance: average importance map per (k,s) from the training data,
  then top-K -> binary mask. For held-out (k,s) uses the nearest trained offset.

- DeterministicRegressor: a small MLP mapping (k/10, s/10) -> importance_map,
  trained on all training (k,s) importance maps with MSE loss, then top-K.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from models.cvae import load_importance_maps  # noqa: E402


class MeanImportance:
    def __init__(self):
        self._means = {}

    def _load(self, k, s):
        # Load mean importance from kernel_dir(k,s)/importance.pt
        d = torch.load(config.kernel_dir(k, s) / "importance.pt",
                       weights_only=True)
        return d["importance"].mean(dim=0)  # (L,H)

    def __call__(self, k, s, k_active):
        key = (k, s)
        if key not in self._means:
            # Try exact load; if fails, use nearest offset for same kernel
            try:
                self._means[key] = self._load(k, s)
            except FileNotFoundError:
                tr_offsets = config.OFFSETS
                ns = min(tr_offsets, key=lambda t: abs(t - s))
                try:
                    self._means[key] = self._load(k, ns)
                except FileNotFoundError:
                    tr_kernels = config.KERNELS
                    nk = min(tr_kernels, key=lambda t: abs(t - k))
                    self._means[key] = self._load(nk, ns)
        p = self._means[key].flatten()
        _, top = p.topk(k_active)
        m = torch.zeros_like(p)
        m[top] = 1.0
        return m.reshape(config.L, config.H)


class DetRegressor(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, config.MASK_DIM),
        )

    def forward(self, cond):
        return self.net(cond)  # logits (N,512)

    @torch.no_grad()
    def mask(self, k, s, k_active, device=None):
        if device is None:
            device = next(self.parameters()).device
        c = torch.tensor([[float(k) / config.CVAE_COND_SCALE_K,
                           float(s) / config.CVAE_COND_SCALE_S]],
                         device=device)
        with torch.no_grad():
            p = torch.sigmoid(self.forward(c)).flatten()
        _, top = p.topk(k_active)
        m = torch.zeros_like(p)
        m[top] = 1.0
        return m.reshape(config.L, config.H)


def train_det_reg(epochs=100, lr=1e-3, batch_size=256):
    """Train the deterministic regressor on all training importance maps."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[det_reg] using {device}")
    train_kernels = [p[0] for p in config.CVAE_TRAIN_PAIRS]
    train_offsets = [p[1] for p in config.CVAE_TRAIN_PAIRS]
    x, y, _ = load_importance_maps(train_kernels, train_offsets,
                                   config.CKPT_DIR)
    # x: (N,512), y: (N,2) = [kernels, offsets]
    cond = torch.stack([
        y[:, 0].float() / config.CVAE_COND_SCALE_K,
        y[:, 1].float() / config.CVAE_COND_SCALE_S,
    ], dim=1)  # (N,2)
    # Split train/val
    n_val = int(x.size(0) * 0.15)
    perm = torch.randperm(x.size(0))
    tr_idx, va_idx = perm[n_val:], perm[:n_val]
    x = x.to(device)
    cond = cond.to(device)
    model = DetRegressor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf")
    for ep in range(epochs):
        model.train()
        for i in range(0, len(tr_idx), batch_size):
            idx = tr_idx[i:i + batch_size]
            loss = nn.functional.mse_loss(
                torch.sigmoid(model(cond[idx])), x[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(
                torch.sigmoid(model(cond[va_idx])), x[va_idx])
        if vl < best:
            best = vl
            torch.save(model.state_dict(),
                       config.OUTPUTS / "eval" / "det_reg.pt")
        if ep % 20 == 0:
            print(f"[det_reg] ep {ep}: val_mse={vl.item():.5f}")
    model.load_state_dict(torch.load(config.OUTPUTS / "eval" / "det_reg.pt"))
    return model


if __name__ == "__main__":
    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    (config.OUTPUTS / "eval").mkdir(exist_ok=True)
    train_det_reg()
    print("det_reg saved -> outputs/eval/det_reg.pt")