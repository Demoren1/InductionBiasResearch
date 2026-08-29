"""Train a VAE/CVAE on continuous ``importance.pt`` maps with collapse diagnostics.

All validation is an internally seeded split of the explicitly supplied
meta-training tasks.  This module never loads held-out/OOD tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from models.cvae import (  # noqa: E402
    CVAE,
    load_top_importance,
    make_loaders,
    posterior_diagnostics,
    read_split_provenance,
    task_ids,
    vae_loss,
)


def _value(name, default):
    return getattr(config, name, default)


def require_cuda(device_name: str = "cuda") -> torch.device:
    """Return CUDA or fail explicitly instead of silently falling back to CPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VAE/CVAE training; refusing CPU fallback")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("only CUDA devices are supported for VAE/CVAE training")
    return device


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, required=True,
                   help="current split.json; training refuses artifacts from any other manifest")
    p.add_argument("--tasks", nargs="+", required=True, help="meta-train task ids only")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--variant", choices=("cvae", "vae"), default="cvae")
    p.add_argument("--epochs", type=int, default=_value("CVAE_EPOCHS", 80))
    p.add_argument("--batch_size", type=int, default=_value("CVAE_BATCH_SIZE", 256))
    p.add_argument("--lr", type=float, default=_value("CVAE_LR", 1e-3))
    p.add_argument("--beta", type=float, default=_value("CVAE_BETA", .1))
    p.add_argument("--latent_dim", type=int, default=_value("LATENT_DIM", 32))
    p.add_argument("--hidden", type=int, default=_value("CVAE_HIDDEN", 256))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_frac", type=float, default=.1,
                   help="lowest-BCE fraction selected from continuous importance.pt maps")
    p.add_argument("--importance_name", default="importance.pt",
                   help="continuous raw importance artifact; binary best-mask files are not the default")
    p.add_argument("--ckpt_root", type=Path, default=_value("CKPT_DIR", "outputs/checkpoints"))
    p.add_argument("--val_fraction", type=float, default=.15,
                   help="internal meta-train validation fraction")
    p.add_argument("--active_kl_threshold", type=float, default=.01,
                   help="reporting threshold in nats for KL-active dimensions")
    p.add_argument("--active_mu_variance_threshold", type=float, default=1e-2,
                   help="collapse guard: active dim iff Var_x[posterior mean] exceeds this")
    p.add_argument("--min_active_dims", type=int, default=2,
                   help="collapse guard: minimum dimensions active by posterior-mean variance")
    p.add_argument("--min_val_kl", type=float, default=1.0,
                   help="collapse guard: minimum internal-validation mean KL in nats")
    p.add_argument("--min_posterior_z0_recon_gap", type=float, default=0.0,
                   help="collapse guard: require posterior reconstruction to beat z=0 decoding by this amount")
    p.add_argument("--tail_fraction", type=float, default=.10,
                   help="strict stability window: final fraction of epochs that must all pass the guard")
    p.add_argument("--tail_min_epochs", type=int, default=5,
                   help="strict stability window has at least this many final epochs")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--device", choices=("cuda",), default="cuda")
    return p


def collapse_guard(metrics: dict[str, Any], *, min_active_dims: int,
                   min_val_kl: float, active_mu_variance_threshold: float,
                   min_posterior_z0_recon_gap: float) -> dict[str, Any]:
    """Evaluate anti-collapse criteria from internal validation metrics only."""
    if min_active_dims < 0 or min_val_kl < 0 or active_mu_variance_threshold < 0:
        raise ValueError("collapse thresholds must be nonnegative")
    finite = ("total", "recon", "kl", "posterior_mu_std", "posterior_std_mean",
              "posterior_vs_z0_recon_gap")
    failures: list[str] = []
    nonfinite = [name for name in finite if not math.isfinite(float(metrics[name]))]
    if nonfinite:
        failures.append("nonfinite:" + ",".join(nonfinite))
    if int(metrics["active_mu_variance_dims"]) < min_active_dims:
        failures.append(f"active_mu_variance_dims<{min_active_dims}")
    if float(metrics["kl"]) < min_val_kl:
        failures.append(f"kl<{min_val_kl:g}")
    if float(metrics["posterior_vs_z0_recon_gap"]) <= min_posterior_z0_recon_gap:
        failures.append(f"posterior_vs_z0_recon_gap<={min_posterior_z0_recon_gap:g}")
    return {
        "passed": not failures,
        "failures": failures,
        "criteria": {
            "min_val_kl": min_val_kl,
            "min_active_mu_variance_dims": min_active_dims,
            "active_mu_variance_threshold": active_mu_variance_threshold,
            "min_posterior_z0_recon_gap": min_posterior_z0_recon_gap,
            "finite_metrics": list(finite),
        },
        "observed": {
            "kl": float(metrics["kl"]),
            "active_mu_variance_dims": int(metrics["active_mu_variance_dims"]),
            "posterior_vs_z0_recon_gap": float(metrics["posterior_vs_z0_recon_gap"]),
            "posterior_mu_std": float(metrics["posterior_mu_std"]),
            "posterior_std_mean": float(metrics["posterior_std_mean"]),
        },
    }


def tail_stability(history: list[dict[str, Any]], *, tail_fraction: float = .10,
                   tail_min_epochs: int = 5) -> dict[str, Any]:
    """Require every final-window epoch to pass: early transients are not evidence.

    This is intentionally stricter than a majority vote.  A late collapsed
    posterior is unsuitable for sampling even if an initialization transient
    briefly had high KL.
    """
    if not history:
        raise ValueError("cannot assess tail stability of empty history")
    if not 0 < tail_fraction <= 1 or tail_min_epochs < 1:
        raise ValueError("tail_fraction must be in (0, 1] and tail_min_epochs positive")
    tail_length = min(len(history), max(tail_min_epochs, math.ceil(tail_fraction * len(history))))
    tail = history[-tail_length:]
    evidence = [{
        "epoch": row["epoch"],
        "passed": row["collapse_guard"]["passed"],
        "failures": row["collapse_guard"]["failures"],
        "val_kl": row["val"]["kl"],
        "active_mu_variance_dims": row["val"]["active_mu_variance_dims"],
        "posterior_vs_z0_recon_gap": row["val"]["posterior_vs_z0_recon_gap"],
        "val_recon": row["val"]["recon"],
    } for row in tail]
    return {
        "stable": all(row["passed"] for row in evidence),
        "policy": "all epochs in the final tail must pass the internal-validation collapse guard",
        "rationale": "prevents a high-KL initialization transient from being promoted after late collapse",
        "tail_fraction": tail_fraction,
        "tail_min_epochs": tail_min_epochs,
        "tail_length": tail_length,
        "tail_start_epoch": tail[0]["epoch"],
        "tail_end_epoch": tail[-1]["epoch"],
        "epochs": evidence,
    }


def _epoch(model: CVAE, loader, *, beta: float, device: torch.device,
           optimizer: torch.optim.Optimizer | None,
           active_kl_threshold: float,
           active_mu_variance_threshold: float) -> dict[str, Any]:
    """Run one epoch; every tensor diagnostic remains on CUDA until JSON output."""
    training = optimizer is not None
    model.train(training)
    sums = {"total": 0.0, "recon": 0.0, "kl": 0.0, "z0_recon": 0.0}
    mus, logvars, n_examples = [], [], 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for xb, cb in loader:
            xb = xb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)
            logits, mu, logvar = model(xb, cb)
            total, recon, kl = vae_loss(logits, xb, mu, logvar, beta)
            # z=0 keeps the same condition but removes sample-specific latent
            # information.  A positive gap is direct evidence against collapse.
            condition = model.condition(cb, device=device)
            z0_logits = model.decode(torch.zeros_like(mu), condition)
            z0_recon = F.binary_cross_entropy_with_logits(
                z0_logits, xb, reduction="none").sum(dim=-1).mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()
            batch_n = xb.size(0)
            n_examples += batch_n
            for name, value in (("total", total), ("recon", recon), ("kl", kl),
                                ("z0_recon", z0_recon)):
                sums[name] += float(value.detach().item()) * batch_n
            mus.append(mu.detach())
            logvars.append(logvar.detach())
    if not n_examples:
        raise RuntimeError("empty VAE/CVAE loader")
    metrics: dict[str, Any] = {name: value / n_examples for name, value in sums.items()}
    metrics["posterior_vs_z0_recon_gap"] = metrics["z0_recon"] - metrics["recon"]
    metrics.update(posterior_diagnostics(
        torch.cat(mus), torch.cat(logvars),
        active_kl_threshold=active_kl_threshold,
        active_mu_variance_threshold=active_mu_variance_threshold,
    ))
    metrics["active_kl_threshold"] = active_kl_threshold
    metrics["active_mu_variance_threshold"] = active_mu_variance_threshold
    metrics["n_examples"] = n_examples
    return metrics


def _checkpoint_payload(model: CVAE, args, tasks: list[str], sources: list[dict],
                        epoch: int, train_metrics: dict, val_metrics: dict,
                        guard: dict, split_provenance: dict | None = None) -> dict[str, Any]:
    return {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "variant": args.variant,
        "mask_dim": model.mask_dim,
        "latent_dim": model.latent_dim,
        "hidden": model.hidden,
        "cond_dim": model.cond_dim,
        "beta": args.beta,
        "top_frac": args.top_frac,
        "importance_name": args.importance_name,
        "train_tasks": [config.task_id(task) if hasattr(config, "task_id") else str(task)
                        for task in tasks],
        "sources": sources,
        "seed": args.seed,
        "selected_epoch": epoch,
        "internal_train_metrics": train_metrics,
        "internal_val_metrics": val_metrics,
        "collapse_guard": guard,
        "selection_rule": "lowest internal validation reconstruction among guard-passing epochs "
                          "in the final stable tail only",
        **(split_provenance or {}),
    }


def run_training(args) -> dict[str, Any]:
    """Fit one beta value and return detailed train-only validation diagnostics."""
    if not 0 < args.top_frac <= 1:
        raise ValueError("--top_frac must be in (0, 1]")
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val_fraction must be in (0, 1)")
    if args.epochs < 1 or args.log_every < 1:
        raise ValueError("--epochs and --log_every must be positive")
    if args.beta < 0:
        raise ValueError("--beta must be nonnegative")
    if not 0 < args.tail_fraction <= 1 or args.tail_min_epochs < 1:
        raise ValueError("--tail_fraction must be in (0, 1] and --tail_min_epochs positive")
    device = require_cuda(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tasks = list(args.tasks)
    # ``run_training`` remains usable as a small library helper without a
    # manifest, but every command-line production run has a required --split.
    split_path = getattr(args, "split", None)
    split_provenance = read_split_provenance(split_path) if split_path is not None else None
    if split_provenance is not None and task_ids(tasks) != split_provenance["split_train_tasks"]:
        raise ValueError("--tasks must exactly equal --split train_tasks (including order)")
    x, c, sources = load_top_importance(tasks, args.ckpt_root, args.importance_name,
                                        args.top_frac, device=device,
                                        expected_provenance=split_provenance)
    train_loader, val_loader = make_loaders(x, c, args.batch_size, args.seed,
                                            val_fraction=args.val_fraction)
    cond_dim = _value("COND_DIM", 8) if args.variant == "cvae" else 0
    model = CVAE(_value("MASK_DIM", 256), args.latent_dim, args.hidden, cond_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # These are generated artifacts owned by this exact run directory.  Remove
    # stale promotions before training so an unstable rerun cannot leave an old
    # checkpoint looking valid beside a new ineligible selection.json.
    (args.out_dir / "best.pt").unlink(missing_ok=True)
    (args.out_dir / "best_tail.pt").unlink(missing_ok=True)

    history: list[dict[str, Any]] = []
    tail_length = min(args.epochs, max(args.tail_min_epochs,
                                       math.ceil(args.tail_fraction * args.epochs)))
    tail_start_epoch = args.epochs - tail_length + 1
    best_tail_rank: tuple[float, float] | None = None
    best_tail_row: dict[str, Any] | None = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = _epoch(model, train_loader, beta=args.beta, device=device,
                               optimizer=optimizer,
                               active_kl_threshold=args.active_kl_threshold,
                               active_mu_variance_threshold=args.active_mu_variance_threshold)
        val_metrics = _epoch(model, val_loader, beta=args.beta, device=device, optimizer=None,
                             active_kl_threshold=args.active_kl_threshold,
                             active_mu_variance_threshold=args.active_mu_variance_threshold)
        guard = collapse_guard(
            val_metrics, min_active_dims=args.min_active_dims, min_val_kl=args.min_val_kl,
            active_mu_variance_threshold=args.active_mu_variance_threshold,
            min_posterior_z0_recon_gap=args.min_posterior_z0_recon_gap,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics,
               "collapse_guard": guard}
        history.append(row)
        # Do not save an early candidate: a checkpoint can only be promoted
        # after the complete final window has established stable non-collapse.
        rank = (float(val_metrics["recon"]), float(val_metrics["total"]))
        if epoch >= tail_start_epoch and guard["passed"] and (
                best_tail_rank is None or rank < best_tail_rank):
            best_tail_rank, best_tail_row = rank, row
            torch.save(_checkpoint_payload(model, args, tasks, sources, epoch,
                                           train_metrics, val_metrics, guard, split_provenance),
                       args.out_dir / "best_tail.pt")
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"[{args.variant} beta={args.beta:g}] epoch={epoch:03d} "
                  f"train(total={train_metrics['total']:.4f},recon={train_metrics['recon']:.4f},"
                  f"kl={train_metrics['kl']:.4f},active={train_metrics['active_mu_variance_dims']}) "
                  f"val(total={val_metrics['total']:.4f},recon={val_metrics['recon']:.4f},"
                  f"kl={val_metrics['kl']:.4f},active={val_metrics['active_mu_variance_dims']},"
                  f"z0gap={val_metrics['posterior_vs_z0_recon_gap']:.4f},"
                  f"guard={'pass' if guard['passed'] else 'FAIL'})", flush=True)
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    tail = tail_stability(history, tail_fraction=args.tail_fraction,
                          tail_min_epochs=args.tail_min_epochs)
    eligible = tail["stable"] and best_tail_row is not None
    selected = best_tail_row if eligible else None
    if eligible:
        # Atomic replacement only occurs after the whole tail is known stable.
        (args.out_dir / "best_tail.pt").replace(args.out_dir / "best.pt")
    else:
        # A passing early epoch or partial tail is diagnostic evidence only;
        # never leave it under a checkpoint-like name after a failed rerun.
        (args.out_dir / "best_tail.pt").unlink(missing_ok=True)
    summary = {
        "variant": args.variant,
        "beta": args.beta,
        "importance_name": args.importance_name,
        "top_frac": args.top_frac,
        "train_tasks": tasks,
        **(split_provenance or {}),
        "selection_rule": "within beta/seed: lowest internal validation reconstruction among "
                          "guard-passing final-tail epochs; no OOD task was loaded or consulted",
        "tail_stability": tail,
        "selected_epoch": selected["epoch"] if selected else None,
        "selected_train": selected["train"] if selected else None,
        "selected_val": selected["val"] if selected else None,
        "collapse_guard": selected["collapse_guard"] if selected else None,
        "selection_status": "eligible" if eligible else "tail_unstable_not_promoted",
    }
    (args.out_dir / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    if eligible:
        print(f"saved {args.out_dir / 'best.pt'} ({summary['selection_status']})", flush=True)
    else:
        print(f"no checkpoint promoted ({summary['selection_status']}); see {args.out_dir / 'selection.json'}",
              flush=True)
    return summary


def main() -> None:
    run_training(parser().parse_args())


if __name__ == "__main__":
    main()
