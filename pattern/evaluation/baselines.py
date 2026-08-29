import sys
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from models.cvae import load_importance_maps


class MeanImportance:
    """Build top-k masks from a meta-training importance-map prior.

    ``train_patterns`` is deliberately an explicit constructor argument: a
    held-out task must never contribute its own maps to this baseline's
    prior.  Omitting it retains the historic all-pattern experiment.
    """

    def __init__(self, train_patterns=None, importance_name="importance.pt",
                 top_frac=None):
        self.train_patterns = list(config.PATTERNS if train_patterns is None
                                   else train_patterns)
        if not self.train_patterns:
            raise ValueError("train_patterns must contain at least one pattern")
        if top_frac is not None and not 0.0 < top_frac <= 1.0:
            raise ValueError("top_frac must be in (0, 1]")
        self.importance_name = importance_name
        self.top_frac = top_frac
        self._means = {}

    def _load(self, pat):
        d = torch.load(config.pattern_dir(pat) / self.importance_name,
                       weights_only=True)
        importance = d["importance"]
        if self.top_frac is not None:
            n_keep = max(1, round(importance.size(0) * self.top_frac))
            if "val_loss" not in d:
                raise KeyError(f"{self.importance_name} for {pat} has no val_loss "
                               "required for top_frac selection")
            idx = torch.argsort(d["val_loss"])[:n_keep]
            importance = importance[idx]
        return importance.mean(dim=0)

    def _mean_for(self, pat: str) -> torch.Tensor:
        # ``pat`` is intentionally not part of the cache key.  The prior is
        # pooled exclusively over the supplied meta-training tasks, including
        # when evaluating a conditional model on an unseen task.
        key = "meta_train"
        if key not in self._means:
            self._means[key] = torch.stack(
                [self._load(p) for p in self.train_patterns]).mean(dim=0)
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


def train_det_reg(epochs=100, lr=1e-3, batch_size=256, *, patterns=None,
                  checkpoint_path=None, importance_name="importance.pt",
                  top_frac=None, seed=config.CVAE_SEED,
                  ckpt_root=config.CKPT_DIR):
    """Train the deterministic regressor on explicitly supplied meta-train maps.

    The checkpoint stores its data provenance so an OOD evaluation can audit
    which tasks the learned baseline was allowed to observe.
    """
    train_patterns = list(config.PATTERNS if patterns is None else patterns)
    if not train_patterns:
        raise ValueError("patterns must contain at least one pattern")
    if top_frac is not None and not 0.0 < top_frac <= 1.0:
        raise ValueError("top_frac must be in (0, 1]")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[det_reg] using {device}; patterns={train_patterns}; seed={seed}")
    x, y, _ = load_importance_maps(train_patterns, ckpt_root,
                                   importance_name=importance_name,
                                   top_frac=top_frac)
    x = x.float()
    y = y.float()
    if config.CVAE_COND_DIM == 0:
        y = torch.zeros_like(y)
    n_val = max(1, int(x.size(0) * 0.15))
    if n_val >= x.size(0):
        raise ValueError("need at least two importance maps to train det_reg")
    perm = torch.randperm(x.size(0), generator=torch.Generator().manual_seed(seed))
    tr_idx, va_idx = perm[n_val:], perm[:n_val]
    x = x.to(device)
    y = y.to(device)
    model = DetRegressor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf")
    save_path = (config.EVAL_DIR / "det_reg.pt" if checkpoint_path is None
                 else Path(checkpoint_path))
    save_path.parent.mkdir(parents=True, exist_ok=True)
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
            torch.save({
                "state_dict": {name: value.detach().cpu()
                               for name, value in model.state_dict().items()},
                "patterns": train_patterns,
                "importance_name": importance_name,
                "top_frac": top_frac,
                "seed": seed,
            }, save_path)
        if ep % 20 == 0:
            print(f"[det_reg] ep {ep}: val_mse={vl.item():.5f}")
    payload = torch.load(save_path, weights_only=True, map_location=device)
    model.load_state_dict(payload.get("state_dict", payload))
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a deterministic importance-map baseline.")
    parser.add_argument("--patterns", nargs="+", default=None,
                        help="meta-train patterns (default: all patterns)")
    parser.add_argument("--checkpoint_path", type=Path,
                        default=config.EVAL_DIR / "det_reg.pt")
    parser.add_argument("--importance_name", default="importance.pt")
    parser.add_argument("--top_frac", type=float, default=0.0,
                        help="lowest-val-loss fraction per task; 0 keeps all")
    parser.add_argument("--seed", type=int, default=config.CVAE_SEED)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--ckpt_root", type=Path, default=config.CKPT_DIR)
    args = parser.parse_args()
    train_det_reg(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                  patterns=args.patterns, checkpoint_path=args.checkpoint_path,
                  importance_name=args.importance_name,
                  top_frac=args.top_frac or None, seed=args.seed,
                  ckpt_root=args.ckpt_root)
