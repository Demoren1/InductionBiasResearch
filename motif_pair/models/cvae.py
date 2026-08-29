"""Conditional and unconditional VAEs for sparse first-layer masks.

The conditional model receives only the task gap as an eight-way one-hot
vector.  In particular, it never receives the two motifs, so OOD evaluation
tests transfer of the gap-dependent structural rule rather than memorisation
of task identities.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config


def _config(name: str, default):
    return getattr(config, name, default)


def task_ids(tasks: Iterable) -> list[str]:
    """Canonical task ids, retaining order (order is part of provenance)."""
    return [config.task_id(task) if hasattr(config, "task_id") else str(task) for task in tasks]


def read_split_provenance(split_path: Path) -> dict:
    """Read the current split manifest and return its immutable provenance.

    The digest deliberately covers the raw file bytes, rather than a parsed
    JSON representation: changing the manifest in place must invalidate every
    downstream artifact, even if the task ids happen to be unchanged.
    """
    split_path = Path(split_path).resolve()
    raw = split_path.read_bytes()
    payload = json.loads(raw)
    tasks = payload.get("train_tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(t, str) for t in tasks):
        raise ValueError(f"{split_path} must contain a nonempty string list 'train_tasks'")
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"{split_path} contains duplicate train_tasks")
    return {
        "split_path": str(split_path),
        "split_sha256": hashlib.sha256(raw).hexdigest(),
        "split_train_tasks": tasks,
    }


def verify_split_provenance(payload: dict, expected: dict, *, label: str,
                            train_key: str = "split_train_tasks") -> None:
    """Reject an artifact not made from exactly the currently requested split.

    ``train_key`` permits CVAE checkpoints to retain their historical
    ``train_tasks`` field while candidate/importance artifacts use
    ``split_train_tasks``.  SHA and *ordered* task-list checks are both
    mandatory when this function is requested by a production CLI.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"{label} lacks provenance metadata")
    seen_hash = payload.get("split_sha256")
    seen_tasks = payload.get(train_key)
    if seen_hash is None or seen_tasks is None:
        raise ValueError(f"{label} lacks split SHA256 or {train_key} provenance")
    if seen_hash != expected["split_sha256"]:
        raise ValueError(
            f"{label} split SHA256 mismatch: artifact={seen_hash}, "
            f"current={expected['split_sha256']}"
        )
    if list(seen_tasks) != list(expected["split_train_tasks"]):
        raise ValueError(
            f"{label} train_tasks mismatch: artifact={seen_tasks}, "
            f"current={expected['split_train_tasks']}"
        )


def verify_generator_config(payload: dict, *, importance_name: str,
                            top_frac: float, label: str) -> None:
    """Make a generator checkpoint's target preprocessing explicit at eval."""
    if payload.get("importance_name") != importance_name:
        raise ValueError(
            f"{label} importance_name mismatch: checkpoint={payload.get('importance_name')!r}, "
            f"requested={importance_name!r}"
        )
    value = payload.get("top_frac")
    if value is None or not math.isclose(float(value), float(top_frac), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{label} top_frac mismatch: checkpoint={value!r}, requested={top_frac!r}"
        )


def task_condition(tasks, *, device=None) -> torch.Tensor:
    """Return gap-only one-hot conditions for a task or batch of tasks."""
    if isinstance(tasks, torch.Tensor):
        x = tasks.to(device) if device is not None else tasks
        if x.ndim == 2 and x.shape[-1] == _config("COND_DIM", 8):
            return x.float()
        if x.ndim == 1 and x.numel() == _config("COND_DIM", 8):
            return x.float().unsqueeze(0)
        tasks = x.detach().cpu().tolist()
    # Experiment tasks are normally canonical strings or ``config.Task``
    # objects; only a *list* denotes a batch.  Do not inspect the concrete
    # Task type here, since that would make this module coupled to config's
    # dataclass implementation.
    if not isinstance(tasks, list):
        tasks = [tasks]
    if hasattr(config, "task_to_condition"):
        rows = [torch.as_tensor(config.task_to_condition(t), dtype=torch.float32)
                for t in tasks]
        result = torch.stack(rows)
    else:  # Kept for small standalone smoke tests.
        gaps = list(_config("GAPS", range(3, 11)))
        values = []
        for task in tasks:
            gap = task[-1] if isinstance(task, tuple) else int(task)
            values.append(gaps.index(gap))
        result = F.one_hot(torch.tensor(values), num_classes=len(gaps)).float()
    return result.to(device) if device is not None else result


class CVAE(nn.Module):
    """Bernoulli-mask VAE, conditional on a gap one-hot when ``cond_dim > 0``."""

    def __init__(self, mask_dim: int | None = None, latent_dim: int | None = None,
                 hidden: int | None = None, cond_dim: int | None = None):
        super().__init__()
        self.mask_dim = int(mask_dim if mask_dim is not None else _config("MASK_DIM", 256))
        self.latent_dim = int(latent_dim if latent_dim is not None else _config("LATENT_DIM", 32))
        self.hidden = int(hidden if hidden is not None else _config("CVAE_HIDDEN", 256))
        self.cond_dim = int(cond_dim if cond_dim is not None else _config("COND_DIM", 8))
        self.encoder = nn.Sequential(
            nn.Linear(self.mask_dim + self.cond_dim, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(self.hidden, self.latent_dim)
        self.logvar = nn.Linear(self.hidden, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.cond_dim, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.mask_dim),
        )

    def condition(self, tasks, *, device=None) -> torch.Tensor:
        if self.cond_dim == 0:
            if isinstance(tasks, torch.Tensor):
                n = tasks.shape[0] if tasks.ndim > 1 else 1
                dev = device or tasks.device
            else:
                n = len(tasks) if isinstance(tasks, (list, tuple)) else 1
                dev = device
            return torch.zeros(n, 0, device=dev)
        c = task_condition(tasks, device=device)
        if c.shape[-1] != self.cond_dim:
            raise ValueError(f"condition dim {c.shape[-1]} != model cond_dim {self.cond_dim}")
        return c

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        h = self.encoder(torch.cat((x, c), dim=-1))
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat((z, c), dim=-1))

    def forward(self, x: torch.Tensor, tasks):
        c = self.condition(tasks, device=x.device)
        mu, logvar = self.encode(x, c)
        return self.decode(self.reparameterize(mu, logvar), c), mu, logvar

    @torch.no_grad()
    def sample_topk(self, tasks, n_per: int, k_active: int, *,
                    generator: torch.Generator | None = None) -> torch.Tensor:
        c = self.condition(tasks, device=next(self.parameters()).device)
        c = c.repeat_interleave(n_per, dim=0)
        z = torch.randn(c.shape[0], self.latent_dim, device=c.device, generator=generator)
        scores = torch.sigmoid(self.decode(z, c))
        top = scores.topk(k_active, dim=-1).indices
        masks = torch.zeros_like(scores)
        masks.scatter_(1, top, 1.0)
        return masks


def kl_per_latent_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Return the per-example KL contribution of every latent dimension."""
    if mu.shape != logvar.shape or mu.ndim != 2:
        raise ValueError("mu and logvar must have equal [batch, latent_dim] shapes")
    return -0.5 * (1 + logvar - mu.square() - logvar.exp())


def vae_loss(logits: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, beta: float = 0.1):
    """BCE summed over mask dimensions, then averaged over samples, plus beta-KL."""
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(-1).mean()
    kl = kl_per_latent_dim(mu, logvar).sum(dim=-1).mean()
    return recon + beta * kl, recon, kl


def posterior_diagnostics(mu: torch.Tensor, logvar: torch.Tensor, *,
                          active_kl_threshold: float = 0.01,
                          active_mu_variance_threshold: float = 1e-2) -> dict[str, float | int | list[float]]:
    """Summarize posterior use without inspecting any OOD task.

    A dimension is *active* when its mean per-example KL exceeds
    ``active_kl_threshold`` nats.  Reporting the complete per-dimension KL
    vector makes this threshold auditable rather than a hidden model-selection
    heuristic.
    """
    if active_kl_threshold < 0 or active_mu_variance_threshold < 0:
        raise ValueError("posterior activity thresholds must be nonnegative")
    per_dim = kl_per_latent_dim(mu, logvar).mean(dim=0)
    mu_variance_per_dim = mu.var(dim=0, unbiased=False)
    posterior_std = (0.5 * logvar).exp()
    active = per_dim >= active_kl_threshold
    active_by_mu_variance = mu_variance_per_dim > active_mu_variance_threshold
    return {
        "kl_per_dim": per_dim.detach().cpu().tolist(),
        "active_latent_dims": int(active.sum().item()),
        "active_latent_fraction": float(active.float().mean().item()),
        "posterior_mu_variance_per_dim": mu_variance_per_dim.detach().cpu().tolist(),
        "active_mu_variance_dims": int(active_by_mu_variance.sum().item()),
        "active_mu_variance_fraction": float(active_by_mu_variance.float().mean().item()),
        "mean_kl_per_dim": float(per_dim.mean().item()),
        "posterior_mu_mean": float(mu.mean().item()),
        "posterior_mu_abs_mean": float(mu.abs().mean().item()),
        "posterior_mu_std": float(mu.std(unbiased=False).item()),
        "posterior_logvar_mean": float(logvar.mean().item()),
        "posterior_std_mean": float(posterior_std.mean().item()),
        "posterior_std_std": float(posterior_std.std(unbiased=False).item()),
    }


def _task_dir(task, ckpt_root: Path) -> Path:
    task_id = config.task_id(task) if hasattr(config, "task_id") else str(task)
    return ckpt_root / f"task_{task_id}"


def _extract_maps(payload):
    if isinstance(payload, torch.Tensor):
        return payload, None
    for key in ("importance", "importance_maps", "maps", "masks"):
        if key in payload:
            maps = payload[key]
            break
    else:
        raise KeyError("importance artifact needs importance/maps/masks")
    losses = next((payload[k] for k in ("val_loss", "val_losses", "losses") if k in payload), None)
    return maps, losses


def canonicalize_hidden_columns(maps: torch.Tensor) -> torch.Tensor:
    """Assign exchangeable hidden columns to deterministic circular anchors.

    The assignment is a batched GPU greedy matching.  Its score uses only a
    generic local three-position window and never the task gap or labels, so
    canonicalization removes column-permutation noise without injecting the
    desired pair geometry.  The returned shape is unchanged: ``[N, 16, 16]``.
    """
    if maps.ndim != 3 or maps.shape[1:] != (_config("SEQ_LEN", 16), _config("H", 16)):
        raise ValueError("canonicalization expects [N, SEQ_LEN, H] maps")
    n, seq_len, hidden = maps.shape
    if seq_len != hidden:
        raise ValueError("anchor canonicalization requires H == SEQ_LEN")
    # Greedy matching must not break score ties by the arbitrary incoming
    # hidden-column index.  Sort sources by their own support signature first;
    # exact signature ties are identical for binary targets and therefore
    # interchangeable.  Powers of two encode every 16-bit support uniquely.
    signature_weights = maps.new_tensor([2.0 ** position for position in range(seq_len)])
    signatures = (maps * signature_weights.view(1, -1, 1)).sum(dim=1)
    source_order = torch.argsort(signatures, dim=1, stable=True)
    maps = torch.gather(maps, 2, source_order[:, None, :].expand(-1, seq_len, -1))
    weights = maps.new_tensor((1.0, 0.5, 0.25))
    scores = torch.stack([
        sum(weights[offset] * maps[:, (anchor + offset) % seq_len, :]
            for offset in range(len(weights)))
        for anchor in range(seq_len)
    ], dim=2)  # [N, source-column, target-anchor]
    available_columns = torch.ones(n, hidden, dtype=torch.bool, device=maps.device)
    available_anchors = torch.ones(n, hidden, dtype=torch.bool, device=maps.device)
    source_for_anchor = torch.empty(n, hidden, dtype=torch.long, device=maps.device)
    batch = torch.arange(n, device=maps.device)
    for _ in range(hidden):
        available = available_columns.unsqueeze(2) & available_anchors.unsqueeze(1)
        best = scores.masked_fill(~available, float("-inf")).flatten(1).argmax(dim=1)
        source_column = torch.div(best, hidden, rounding_mode="floor")
        target_anchor = best % hidden
        source_for_anchor[batch, target_anchor] = source_column
        available_columns[batch, source_column] = False
        available_anchors[batch, target_anchor] = False
    return torch.gather(maps, 2, source_for_anchor[:, None, :].expand(-1, seq_len, -1))


def load_top_importance(tasks: Iterable, ckpt_root: Path | None = None,
                        importance_name: str = "importance.pt", top_frac: float = .1,
                        device: torch.device | str = "cpu", *,
                        split_path: Path | None = None,
                        expected_provenance: dict | None = None):
    """Load continuous raw importance maps for explicitly supplied tasks.

    The default artifact is ``importance.pt``.  ``top_frac`` selects models by
    their source validation BCE but leaves the selected maps continuous; no
    binary-mask conversion happens on this training path.
    """
    if split_path is not None and expected_provenance is not None:
        raise ValueError("pass only one of split_path or expected_provenance")
    if split_path is not None:
        expected_provenance = read_split_provenance(split_path)
    task_list = list(tasks)
    if expected_provenance is not None and task_ids(task_list) != list(expected_provenance["split_train_tasks"]):
        raise ValueError("requested generator train tasks do not exactly match split.json train_tasks")
    ckpt_root = Path(ckpt_root or _config("CKPT_DIR", "outputs/checkpoints"))
    xs, cs, provenance = [], [], []
    for task in task_list:
        path = _task_dir(task, ckpt_root) / importance_name
        if not path.exists():
            raise FileNotFoundError(f"missing raw importance artifact: {path}")
        payload = torch.load(path, weights_only=True, map_location=device)
        if expected_provenance is not None:
            verify_split_provenance(payload, expected_provenance,
                                    label=f"importance artifact {path}")
        expected_task = config.task_id(task) if hasattr(config, "task_id") else str(task)
        if isinstance(payload, dict) and payload.get("task") not in (None, expected_task):
            raise ValueError(f"importance artifact {path} task mismatch: "
                             f"artifact={payload.get('task')!r}, expected={expected_task!r}")
        maps, losses = _extract_maps(payload)
        maps = maps.reshape(maps.shape[0], -1).float()
        if maps.shape[1] != _config("MASK_DIM", 256):
            raise ValueError(f"{path}: expected mask_dim={_config('MASK_DIM', 256)}, got {maps.shape[1]}")
        # A preselected artifact has already been filtered by validation BCE;
        # selecting it a second time would silently turn the requested 10%
        # into 1%.
        already_selected = isinstance(payload, dict) and "top_fraction" in payload
        n = len(maps) if already_selected else max(1, int(len(maps) * top_frac))
        if losses is not None and not already_selected:
            idx = torch.as_tensor(losses).flatten().topk(n, largest=False).indices
            maps = maps[idx]
        else:
            maps = maps[:n]
        maps = canonicalize_hidden_columns(
            maps.reshape(len(maps), _config("SEQ_LEN", 16), _config("H", 16))
        ).flatten(1)
        xs.append(maps)
        cs.append(task_condition([task], device=maps.device).expand(len(maps), -1))
        provenance.append({"task": expected_task,
                           "source": str(path), "n_selected": len(maps),
                           "hidden_columns_canonicalized": True,
                           **({key: payload[key] for key in ("split_path", "split_sha256", "split_train_tasks")}
                              if isinstance(payload, dict) and expected_provenance is not None else {})})
    return torch.cat(xs), torch.cat(cs), provenance


def make_loaders(x, c, batch_size: int, seed: int, val_fraction: float = .15):
    if len(x) < 2:
        raise ValueError("at least two selected masks are required")
    p = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    n_val = min(max(1, round(len(x) * val_fraction)), len(x) - 1)
    val, train = p[:n_val], p[n_val:]
    return (DataLoader(TensorDataset(x[train], c[train]), batch_size=batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(seed)),
            DataLoader(TensorDataset(x[val], c[val]), batch_size=batch_size))
