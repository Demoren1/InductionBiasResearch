"""Evaluate generated masks on entirely held-out pattern lengths."""
import argparse
from functools import lru_cache
from pathlib import Path
import os
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from meta_pattern.data import PatternTask, sample_dataset
from .common import (atomic_save, bank_path, exact_k, ideal_mask, load_protocol,
                     patterns_by_split, seed_for)
from .cvae import LengthCVAE
from .mlp import BatchedMaskedMLP, fit_mlp

METHODS = ("cvae", "wrong_lower", "wrong_upper", "cvae_z0", "mean_interp", "random", "ideal")


def hard_topk(values, k):
    flat = values.reshape(values.shape[0], -1)
    result = torch.zeros_like(flat)
    # Stable sorting makes ties reproducible; no test labels enter this choice.
    indices = torch.argsort(flat, dim=1, descending=True, stable=True)[:, :k]
    return result.scatter_(1, indices, 1.0)


def matched_iou(masks, target):
    result = []
    for mask in masks:
        intersections = (mask.T @ target).numpy()
        rows, cols = linear_sum_assignment(intersections, maximize=True)
        intersection = float(intersections[rows, cols].sum())
        result.append(intersection / (float(mask.sum() + target.sum()) - intersection))
    return torch.tensor(result)


def mean_maps(out, c):
    result = {}
    for k in c.train_lengths:
        maps = []
        for pat in patterns_by_split(c)["train"]:
            if len(pat) == k:
                value = torch.load(bank_path(out, pat), map_location="cpu", weights_only=False)
                maps.append(value["selected"]["importance"].mean(0))
        result[k] = torch.stack(maps).mean(0)
    return result


def make_mask_pool(out, c, seed, length, device):
    state = torch.load(Path(out) / "cvae" / f"seed{seed}" / "best.pt", map_location="cpu", weights_only=False)
    model = LengthCVAE(**state["model_config"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    lower = max(k for k in c.train_lengths if k < length)
    upper = min(k for k in c.train_lengths if k > length)
    n, k_active = c.eval_masks, exact_k(length, c)
    generator = torch.Generator().manual_seed(seed_for("mask_z", c.seed, seed, length))
    z = torch.randn(n, c.latent_dim, generator=generator).to(device)
    masks = []
    with torch.no_grad():
        for condition in (length, lower, upper):
            logits = model.decode_lengths(z, torch.full((n,), float(condition), device=device))
            masks.append(hard_topk(logits, k_active).cpu().reshape(n, c.seq_len, c.hidden))
        logits = model.decode_lengths(torch.zeros_like(z), torch.full((n,), float(length), device=device))
        masks.append(hard_topk(logits, k_active).cpu().reshape(n, c.seq_len, c.hidden))
    means = mean_maps(out, c)
    fraction = (length - lower) / (upper - lower)
    interpolation = means[lower] * (1 - fraction) + means[upper] * fraction
    masks.append(hard_topk(interpolation[None].expand(n, -1, -1), k_active).reshape(n, c.seq_len, c.hidden))
    generator = torch.Generator().manual_seed(seed_for("random_masks", c.seed, length))
    random_values = torch.rand(n, c.seq_len * c.hidden, generator=generator)
    masks.append(hard_topk(random_values, k_active).reshape(n, c.seq_len, c.hidden))
    target = ideal_mask(length, c)
    masks.append(target[None].expand(n, -1, -1).clone())
    masks = torch.stack(masks)
    if not torch.all(masks.sum(dim=(-1, -2)) == k_active):
        raise AssertionError("Mask cardinalities differ between methods")
    return {"methods": METHODS, "masks": masks, "z": z.cpu(), "length": length,
            "lower": lower, "upper": upper, "k_active": k_active,
            "iou": torch.stack([matched_iou(m, target) for m in masks]),
            "checkpoint_epoch": state["epoch"]}


def run(out, seed, length, shard, shards, device):
    if not 0 <= shard < shards:
        raise ValueError("Invalid shard")
    c = load_protocol(out)
    if length not in c.heldout_lengths:
        raise ValueError("Only held-out interpolation lengths may be evaluated")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = Path(out) / "evaluation" / f"seed{seed}_k{length}_shard{shard}.pt"
    if path.exists():
        raise FileExistsError(path)
    pool = make_mask_pool(out, c, seed, length, device)
    patterns = [p for p in patterns_by_split(c)["test"] if len(p) == length][shard::shards]
    rows = []
    started = time.monotonic()
    for pattern in patterns:
        task = PatternTask(pattern)
        for repeat in range(c.eval_repeats):
            base = seed_for("evaluation", c.seed, pattern, repeat)
            support = sample_dataset(task, c.support_pool_size, seed=seed_for(base, "support"), split="support",
                                     split_seed=c.input_split_seed)
            tests = {kind: sample_dataset(task, c.eval_test_size, seed=seed_for(base, kind), split="test",
                                         split_seed=c.input_split_seed, balanced=kind == "balanced")
                     for kind in ("balanced", "natural")}
            flat_masks = pool["masks"].reshape(-1, c.seq_len, c.hidden)
            model = BatchedMaskedMLP(flat_masks, seed=seed_for(base, "weights"))
            # Match initial MLP weights for each mask ordinal across every method.
            with torch.no_grad():
                for parameter in model.parameters():
                    base_weights = parameter[:c.eval_masks].clone()
                    parameter.copy_(base_weights.repeat((len(METHODS),) + (1,) * (parameter.ndim - 1)))
            model.to(device)
            fit = fit_mlp(model, support["x"].to(device), support["y"].to(device), steps=c.eval_steps,
                          batch_size=c.eval_batch, lr=c.eval_lr, seed=seed_for(base, "minibatches"))
            for kind, data in tests.items():
                metrics = model.validation(data["x"].to(device), data["y"].to(device), batch_size=256)
                rows.append({"pattern": pattern, "length": length, "repeat": repeat, "distribution": kind,
                             "accuracy": metrics["accuracy"].cpu().reshape(len(METHODS), c.eval_masks),
                             "bce": metrics["bce"].cpu().reshape(len(METHODS), c.eval_masks),
                             "positive_fraction": float(data["y"].mean()), "fit": fit})
        print(f"EVAL seed={seed} length={length} pattern={pattern} elapsed={time.monotonic()-started:.1f}s", flush=True)
    atomic_save(path, {"config": c.to_dict(), "seed": seed, "length": length, "shard": shard,
                       "patterns": patterns, "pool": pool, "rows": rows, "seconds": time.monotonic() - started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    run(**vars(parser.parse_args()))
