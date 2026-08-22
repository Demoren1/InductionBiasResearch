import sys
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from models.cvae import load_importance_maps


class MeanImportance:
    """Build top-k masks from mean importance maps."""

    def __init__(self):
        self._means = {}

    def _load(self, pat):
        d = torch.load(config.pattern_dir(pat) / "importance.pt",
                       weights_only=True)
        return d["importance"].mean(dim=0)

    def _mean_for(self, pat: str) -> torch.Tensor:
        key = "all" if config.CVAE_COND_DIM == 0 else pat
        if key not in self._means:
            patterns = config.PATTERNS if key == "all" else [pat]
            self._means[key] = torch.stack([self._load(p) for p in patterns]).mean(dim=0)
        return self._means[key]

    def __call__(self, pat, k_active):
        mean = self._mean_for(pat)
        p = mean.flatten().abs() if config.SIGNED_IMPORTANCE \
            else mean.flatten()
        _, top = p.topk(k_active)
        m = torch.zeros_like(p)
        m[top] = 1.0
        return m.reshape(config.SEQ_LEN, config.H)


class DetRegressor(nn.Module):
    """Predict an importance map from a pattern condition."""

    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, config.MASK_DIM),
        )

    def forward(self, cond):
        return self.net(cond)

    @torch.no_grad()
    def mask(self, pat, k_active, device=None):
        if device is None:
            device = next(self.parameters()).device
        if config.CVAE_COND_DIM == 0:
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
    x, y, _ = load_importance_maps(config.PATTERNS,
                                   config.CKPT_DIR)
    x = x.float()
    y = y.float()
    if config.CVAE_COND_DIM == 0:
        y = torch.zeros_like(y)
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
            torch.save({name: value.detach().cpu()
                        for name, value in model.state_dict().items()}, save_path)
        if ep % 20 == 0:
            print(f"[det_reg] ep {ep}: val_mse={vl.item():.5f}")
    model.load_state_dict(torch.load(save_path, weights_only=True,
                                     map_location=device))
    return model


if __name__ == "__main__":
    train_det_reg()
