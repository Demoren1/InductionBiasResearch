import sys
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from models.cvae import load_importance_maps


class MeanImportance:
    def __init__(self):
        self._means = {}

    def _load(self, pat):
        # Load mean importance from pattern_dir(pat)/importance.pt
        d = torch.load(config.pattern_dir(pat) / "importance.pt",
                       weights_only=True)
        return d["importance"].mean(dim=0)  # (L,H)

    def __call__(self, pat, k_active):
        if config.CVAE_COND_DIM == 0:
            # Unconditional: average ALL available pattern maps (shared
            # support). Ignore pat except as a cache key "all".
            if "all" not in self._means:
                if pat in self._means:
                    # stale entry from a previous conditional call
                    self._means.pop(pat, None)
                all_pats = config.PATTERNS
                parts = [self._load(p) for p in all_pats]
                self._means["all"] = torch.stack(parts).mean(dim=0)
            mean = self._means["all"]
        else:
            if pat not in self._means:
                try:
                    self._means[pat] = self._load(pat)
                except FileNotFoundError:
                    parts = [self._load(p)
                             for p in config.CVAE_TRAIN_PATTERNS]
                    self._means[pat] = torch.stack(parts).mean(dim=0)
            mean = self._means[pat]
        p = mean.flatten().abs() if config.SIGNED_IMPORTANCE \
            else mean.flatten()
        _, top = p.topk(k_active)
        m = torch.zeros_like(p)
        m[top] = 1.0
        return m.reshape(config.SEQ_LEN, config.H)


class DetRegressor(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, config.MASK_DIM),
        )

    def forward(self, cond):
        return self.net(cond)  # logits (N, MASK_DIM)

    @torch.no_grad()
    def mask(self, pat, k_active, device=None):
        if device is None:
            device = next(self.parameters()).device
        if config.CVAE_COND_DIM == 0:
            # Unconditional: net still takes a (1,4) input but we feed a
            # dummy zeros cond so it learns a constant (shared) map.
            c = torch.zeros(1, 4, device=device)
        elif isinstance(pat, str):
            c = config.pattern_to_pm1(pat).view(1, 4).to(device)
        else:
            c = pat.view(1, 4).to(device)
        with torch.no_grad():
            logits = self.forward(c).flatten()
            if config.SIGNED_IMPORTANCE:
                p = torch.tanh(logits).abs()
            else:
                p = torch.sigmoid(logits)
        _, top = p.topk(k_active)
        m = torch.zeros_like(p)
        m[top] = 1.0
        return m.reshape(config.SEQ_LEN, config.H)


def train_det_reg(epochs=100, lr=1e-3, batch_size=256):
    """Train the deterministic regressor on all training importance maps."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[det_reg] using {device}")
    x, y, _ = load_importance_maps(config.CVAE_TRAIN_PATTERNS,
                                   config.CKPT_DIR)
    # x: (N, MASK_DIM), y: (N, 4) +-1 pattern condition
    x = x.float()
    y = y.float()
    if config.CVAE_COND_DIM == 0:
        # Unconditional: fit a constant function (same as the mean map), so
        # the pattern condition is ignored by zeroing the targets.
        y = torch.zeros_like(y)
    # Split train/val
    n_val = int(x.size(0) * 0.15)
    perm = torch.randperm(x.size(0))
    tr_idx, va_idx = perm[n_val:], perm[:n_val]
    x = x.to(device)
    y = y.to(device)
    model = DetRegressor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf")
    save_path = config.EVAL_DIR / "det_reg.pt"
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        model.train()
        for i in range(0, len(tr_idx), batch_size):
            idx = tr_idx[i:i + batch_size]
            logits = model(y[idx])
            pred = torch.tanh(logits) if config.SIGNED_IMPORTANCE \
                else torch.sigmoid(logits)
            loss = nn.functional.mse_loss(pred, x[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(y[va_idx])
            pred = torch.tanh(logits) if config.SIGNED_IMPORTANCE \
                else torch.sigmoid(logits)
            vl = nn.functional.mse_loss(pred, x[va_idx])
        if vl < best:
            best = vl
            torch.save(model.state_dict(), save_path)
        if ep % 20 == 0:
            print(f"[det_reg] ep {ep}: val_mse={vl.item():.5f}")
    model.load_state_dict(torch.load(save_path))
    return model


if __name__ == "__main__":
    train_det_reg()
