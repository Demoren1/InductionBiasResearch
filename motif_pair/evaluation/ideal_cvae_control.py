"""Controlled scalar CVAE trained only on canonical ideal masks for seen gaps.

This diagnostic deliberately separates two questions.  Training uses identical,
gap-indexed ideal supports from the meta-train gaps only.  Evaluation then asks
whether scalar conditioning and frozen-decoder latent search can reconstruct or
express that structure at each declared gap.  It is not a task-performance or
zero-shot selection experiment: held-out ideals appear only after checkpoint
selection, as diagnostic targets for posterior/oracle metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from data.generate import ideal_mask  # noqa: E402
from evaluation.oracle_ideal import hard_topk, optimize_ideal  # noqa: E402
from evaluation.structural import best_permutation_iou  # noqa: E402
from models.cvae import CVAE, posterior_diagnostics, vae_loss  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent / "outputs" / "ideal_cvae_control" / "20260906"
N_PER_SEEN_GAP = 408
SEED = 42
EPOCHS = 80
BATCH_SIZE = 256
LR = 1e-3
N_STARTS = 64
ORACLE_STEPS = 1000
ORACLE_LR = .03
ORACLE_TEMPERATURE = .5
RADII = (8., 16.)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu(item) for item in value]
    return value


@dataclass(frozen=True)
class Profile:
    name: str
    split: Path
    beta: float


def profiles() -> tuple[Profile, ...]:
    base = Path(__file__).resolve().parent.parent / "outputs" / "ood"
    return (
        Profile("interpolation", base / "gap_interp_g05_g08_seed_42" / "split.json", .1),
        Profile("extrapolation", base / "gap_extrap_g03_g04_seed_42" / "split.json", .3),
    )


def gap_lists(split: dict[str, Any]) -> tuple[list[int], list[int]]:
    train = [config.parse_task(task).gap for task in split["train_tasks"]]
    test = [config.parse_task(task).gap for task in split["test_tasks"]]
    seen, heldout = sorted(set(train)), sorted(set(test))
    if len(seen) != 6 or len(heldout) != 2 or set(seen) | set(heldout) != set(config.GAPS):
        raise ValueError("expected six seen and two held-out gaps covering config.GAPS")
    return seen, heldout


def task_for_gap(gap: int) -> config.Task:
    return config.Task("000", "001", gap)


def condition(gaps: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Explicit scalar convention, rather than inheriting a process default."""
    return ((gaps.to(device=device, dtype=torch.float32) - min(config.GAPS)) /
            (max(config.GAPS) - min(config.GAPS))).reshape(-1, 1)


def build_seen_data(seen: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    masks, gaps = [], []
    for gap in seen:
        target = ideal_mask(task_for_gap(gap)).reshape(-1).float()
        masks.append(target.expand(N_PER_SEEN_GAP, -1).clone())
        gaps.append(torch.full((N_PER_SEEN_GAP,), gap, dtype=torch.long))
    x, g = torch.cat(masks), torch.cat(gaps)
    # The templates have canonical aligned columns by design; no random column
    # permutations are introduced in this isolating control.
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(x), generator=generator)
    val_n = int(round(.15 * len(x)))
    val, train = order[:val_n], order[val_n:]
    return x, g, train, val


def _metrics(model: CVAE, loader: DataLoader, device: torch.device, beta: float,
             optimizer: torch.optim.Optimizer | None) -> dict[str, Any]:
    train = optimizer is not None
    model.train(train)
    sums = {key: 0. for key in ("total", "recon", "kl", "z0_recon")}
    mus, logvars, count = [], [], 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, gaps in loader:
            x = x.to(device, non_blocking=True)
            c = condition(gaps, device)
            logits, mu, logvar = model(x, c)
            total, recon, kl = vae_loss(logits, x, mu, logvar, beta)
            z0 = model.decode(torch.zeros_like(mu), c)
            z0_recon = F.binary_cross_entropy_with_logits(z0, x, reduction="none").sum(-1).mean()
            if train:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()
            n = len(x); count += n
            for key, value in (("total", total), ("recon", recon), ("kl", kl), ("z0_recon", z0_recon)):
                sums[key] += float(value.detach()) * n
            mus.append(mu.detach()); logvars.append(logvar.detach())
    result = {key: value / count for key, value in sums.items()}
    result["posterior_vs_z0_recon_gap"] = result["z0_recon"] - result["recon"]
    result.update(posterior_diagnostics(torch.cat(mus), torch.cat(logvars)))
    result["n_examples"] = count
    return result


def hard(logits: torch.Tensor) -> torch.Tensor:
    return hard_topk(logits, config.K_ACTIVE).reshape(-1, config.SEQ_LEN, config.H).cpu()


def iou_summary(masks: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    values = torch.tensor([best_permutation_iou(mask, target)["iou"] for mask in masks])
    return {"mean_iou": float(values.mean()), "max_iou": float(values.max()),
            "exact_count": int((values == 1).sum()), "n": len(values), "ious": values}


def evaluate_basic(model: CVAE, gap: int, initial_z: torch.Tensor, device: torch.device) -> dict[str, Any]:
    target = ideal_mask(task_for_gap(gap)).float()
    x = target.reshape(1, -1).to(device)
    c1 = condition(torch.tensor([gap]), device)
    with torch.no_grad():
        mu, logvar = model.encode(x, c1)
        reconstruction = hard(model.decode(mu, c1))
        z0 = hard(model.decode(torch.zeros_like(mu), c1))
        prior_c = c1.expand(N_STARTS, -1).contiguous()
        prior = hard(model.decode(initial_z.to(device), prior_c))
    return {
        "target": target, "condition": c1.cpu(), "posterior_mu": mu.cpu(),
        "posterior_logvar": logvar.cpu(), "reconstruction": reconstruction,
        "z0": z0, "prior": prior,
        "reconstruction_stats": iou_summary(reconstruction, target),
        "z0_stats": iou_summary(z0, target), "prior_stats": iou_summary(prior, target),
    }


def strip_iou_tensor(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stats.items() if key != "ious"}


def _slice_oracle_batch(record: dict[str, Any], group: int, group_size: int) -> dict[str, Any]:
    """Recover one independent gap's 64 trajectories from a joint batch.

    ``optimize_ideal`` sums per-row losses and the decoder has no batch-coupled
    layers, so this batched execution is mathematically the same set of
    independent searches while avoiding eight Python/CUDA launch sequences.
    """
    start, stop = group * group_size, (group + 1) * group_size
    result: dict[str, Any] = {"decoder_unchanged": bool(record["decoder_unchanged"])}
    for stage in ("initial", "best_soft", "best_hard"):
        result[stage] = {key: value[start:stop].clone() for key, value in record[stage].items()}
    result["best_soft_steps"] = record["best_soft_steps"][start:stop].clone()
    result["best_hard_steps"] = record["best_hard_steps"][start:stop].clone()
    result["condition"] = record["condition"][start:stop].clone()
    result["target"] = record["target"][start:stop].clone()
    result["group_ids"] = torch.zeros(group_size, dtype=torch.long)
    result["history"] = [
        {"step": row["step"], **row["groups"][group]}
        for row in record["history"]
    ]
    return result


def train_one(profile: Profile, split: dict[str, Any], output: Path, device: torch.device) -> dict[str, Any]:
    seen, heldout = gap_lists(split)
    x, gaps, train_indices, val_indices = build_seen_data(seen)
    if len(x) != 2448 or len(train_indices) != 2081 or len(val_indices) != 367:
        raise AssertionError("control dataset cardinality changed")
    train_ds = TensorDataset(x[train_indices], gaps[train_indices])
    val_ds = TensorDataset(x[val_indices], gaps[val_indices])
    loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=loader_generator,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    if len(train_loader) != 9:
        raise AssertionError("expected nine training batches per epoch")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = CVAE(config.MASK_DIM, 32, 256, 1, condition_encoding="scalar").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = []
    best = None
    for epoch in range(1, EPOCHS + 1):
        train = _metrics(model, train_loader, device, profile.beta, optimizer)
        val = _metrics(model, val_loader, device, profile.beta, None)
        row = {"epoch": epoch, "train": train, "val": val}
        history.append(row)
        # Deterministic targets can validly need no sample-specific latent.  We
        # retain KL diagnostics but intentionally do not gate promotion on them.
        if epoch >= 73 and (best is None or val["recon"] < best[0]):
            best = (val["recon"], epoch, cpu({key: value.clone() for key, value in model.state_dict().items()}), row)
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"[{profile.name}] epoch={epoch:03d} train_recon={train['recon']:.4f} "
                  f"val_recon={val['recon']:.4f} val_kl={val['kl']:.4f} "
                  f"active={val['active_mu_variance_dims']}", flush=True)
    if best is None:
        raise AssertionError("no final-tail checkpoint")
    _, epoch, state, selected = best
    checkpoint = {
        "state_dict": state, "variant": "cvae", "mask_dim": config.MASK_DIM,
        "latent_dim": 32, "hidden": 256, "cond_dim": 1,
        **config.condition_metadata("scalar"), "beta": profile.beta, "seed": SEED,
        "selected_epoch": epoch, "selection_rule": "lowest posterior-mean BCE on duplicate internal validation in epochs 73--80; no KL guard",
        "train_gaps": seen, "heldout_gaps": heldout,
    }
    torch.save(checkpoint, output / "model.pt")
    (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "training_split.pt").write_bytes(b"") if False else torch.save(
        {"x": x, "gaps": gaps, "train_indices": train_indices, "val_indices": val_indices}, output / "training_split.pt")
    # All post-selection diagnostics must use exactly the checkpoint that was
    # selected from the final tail, not the in-memory epoch-80 weights.
    model.load_state_dict(state)
    return {"model": model, "seen": seen, "heldout": heldout, "history": history,
            "selected": selected, "selected_epoch": epoch, "checkpoint": checkpoint}


def load_existing(profile: Profile, split: dict[str, Any], output: Path,
                  device: torch.device) -> dict[str, Any]:
    """Load the immutable selected checkpoint for a validation-only rerun."""
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    model = CVAE(checkpoint["mask_dim"], checkpoint["latent_dim"], checkpoint["hidden"],
                 checkpoint["cond_dim"], condition_encoding=checkpoint["condition_encoding"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    history = json.loads((output / "training_history.json").read_text())
    selected_epoch = int(checkpoint["selected_epoch"])
    if not 73 <= selected_epoch <= 80:
        raise AssertionError("selected checkpoint is outside the preregistered tail")
    seen, heldout = gap_lists(split)
    return {"model": model, "seen": seen, "heldout": heldout, "history": history,
            "selected": history[selected_epoch - 1], "selected_epoch": selected_epoch,
            "checkpoint": checkpoint}


def run_profile(profile: Profile, protocol_sha: str, initial_z: torch.Tensor, device: torch.device,
                *, existing: bool = False) -> dict[str, Any]:
    split = json.loads(profile.split.read_text())
    out = ROOT / profile.name
    if existing:
        trained = load_existing(profile, split, out, device)
    else:
        out.mkdir(parents=True, exist_ok=False)
        trained = train_one(profile, split, out, device)
    model: CVAE = trained["model"]
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    results: dict[str, Any] = {}
    summary_gaps: dict[str, Any] = {}
    # Basic posterior/z0/prior diagnostics are evaluated separately because a
    # posterior mean is defined per target.  The expensive oracle is then one
    # batched call per radius, with 64 independently optimized rows per gap.
    basics = {gap: evaluate_basic(model, gap, initial_z, device) for gap in config.GAPS}
    groups = len(config.GAPS)
    z_batch = initial_z.to(device).unsqueeze(0).expand(groups, -1, -1).contiguous().reshape(-1, 32)
    target_batch = torch.cat([basics[gap]["target"].to(device).expand(N_STARTS, -1, -1)
                              for gap in config.GAPS], dim=0).contiguous()
    condition_batch = torch.cat([basics[gap]["condition"].to(device).expand(N_STARTS, -1)
                                 for gap in config.GAPS], dim=0).contiguous()
    group_ids = torch.arange(groups, device=device).repeat_interleave(N_STARTS)
    oracle_batches = {
        str(int(radius)): optimize_ideal(model, z_batch, target_batch, condition_batch,
                                         steps=ORACLE_STEPS, lr=ORACLE_LR,
                                         temperature=ORACLE_TEMPERATURE, radius=radius,
                                         group_ids=group_ids)
        for radius in RADII
    }
    for gap in config.GAPS:
        basic = basics[gap]
        group = list(config.GAPS).index(gap)
        oracle_by_radius = {radius: cpu(_slice_oracle_batch(record, group, N_STARTS))
                            for radius, record in oracle_batches.items()}
        results[str(gap)] = {"basic": cpu(basic), "oracle": oracle_by_radius}
        row = {"label": "seen" if gap in trained["seen"] else "heldout",
               "reconstruction": strip_iou_tensor(basic["reconstruction_stats"]),
               "z0": strip_iou_tensor(basic["z0_stats"]),
               "prior": strip_iou_tensor(basic["prior_stats"])}
        for radius in RADII:
            record = oracle_by_radius[str(int(radius))]
            row[f"oracle_r{int(radius)}_best_soft"] = strip_iou_tensor(iou_summary(record["best_soft"]["hard"], basic["target"]))
            row[f"oracle_r{int(radius)}_best_hard"] = strip_iou_tensor(iou_summary(record["best_hard"]["hard"], basic["target"]))
        summary_gaps[str(gap)] = row
        print(f"[{profile.name}] gap={gap} recon={row['reconstruction']['mean_iou']:.3f} "
              f"z0={row['z0']['mean_iou']:.3f} r8={row['oracle_r8_best_hard']['mean_iou']:.3f}", flush=True)
    payload = {"metadata": {"profile": profile.name, "beta": profile.beta,
                               "split": str(profile.split), "split_sha256": sha256(profile.split),
                               "protocol_sha256": protocol_sha, "source_sha256": sha256(Path(__file__)),
                               "model_sha256": sha256(out / "model.pt"), "initial_z_sha256": hashlib.sha256(initial_z.numpy().tobytes()).hexdigest()},
               "initial_z": initial_z.cpu(), "results": results}
    torch.save(payload, out / "evaluation.pt")
    compact = {"metadata": payload["metadata"], "selected_epoch": trained["selected_epoch"],
               "selected_internal_val": trained["selected"]["val"], "seen_gaps": trained["seen"],
               "heldout_gaps": trained["heldout"], "gaps": summary_gaps,
               "final_epoch": trained["history"][-1],
               "selection_note": "internal validation consists of duplicates of seen canonical targets and measures reconstruction only"}
    (out / "summary.json").write_text(json.dumps(compact, indent=2) + "\n")
    return compact


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--evaluate-existing", action="store_true",
                      help="redecode only the already selected control checkpoints")
    args = args.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if ROOT.exists() and not args.evaluate_existing:
        raise FileExistsError(f"refusing to overwrite existing output {ROOT}")
    if not ROOT.exists():
        ROOT.mkdir(parents=True)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    source_profiles = {}
    for profile in profiles():
        split = json.loads(profile.split.read_text())
        seen, heldout = gap_lists(split)
        source_profiles[profile.name] = {"split": str(profile.split), "split_sha256": sha256(profile.split),
                                         "seen_gaps": seen, "heldout_gaps": heldout, "beta": profile.beta}
    protocol = {"purpose": "ideal-target scalar-CVAE control; held-out ideals not used until after checkpoint selection",
                "profiles": source_profiles, "target": "canonical ideal_mask(gap), no random column permutation",
                "condition": {**config.condition_metadata("scalar"), "explicit_formula": "(gap - 3) / 7"},
                "architecture": {"latent_dim": 32, "hidden": 256, "mask_dim": 256},
                "training": {"seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
                             "examples_per_seen_gap": N_PER_SEEN_GAP, "total_examples": 2448,
                             "internal_train": 2081, "internal_val": 367, "train_batches_per_epoch": 9,
                             "selection": "min posterior-mean BCE in epochs 73--80; KL is reported, not a gate"},
                "evaluation": {"all_gaps": list(config.GAPS), "prior_starts": N_STARTS,
                               "oracle_starts": N_STARTS, "oracle_steps": ORACLE_STEPS,
                               "oracle_lr": ORACLE_LR, "oracle_temperature": ORACLE_TEMPERATURE,
                               "radii": list(RADII), "metric": "existing best_permutation_iou"}}
    if args.evaluate_existing:
        existing_protocol = json.loads((ROOT / "protocol.json").read_text())
        if existing_protocol != protocol:
            raise AssertionError("existing protocol differs; refusing to re-evaluate")
    else:
        (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    protocol_sha = sha256(ROOT / "protocol.json")
    generator = torch.Generator().manual_seed(20260906)
    initial_z = torch.randn(N_STARTS, 32, generator=generator)
    summaries = {profile.name: run_profile(profile, protocol_sha, initial_z, device,
                                           existing=args.evaluate_existing)
                 for profile in profiles()}
    manifest = {"protocol": protocol, "summaries": summaries,
                "artifacts": {name: {"summary_sha256": sha256(ROOT / name / "summary.json"),
                                      "evaluation_sha256": sha256(ROOT / name / "evaluation.pt")}
                              for name in summaries}}
    (ROOT / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({name: value["selected_epoch"] for name, value in summaries.items()}))


if __name__ == "__main__":
    main()
