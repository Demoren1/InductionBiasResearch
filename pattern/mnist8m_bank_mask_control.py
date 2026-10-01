"""Paired retraining control for top-bank and random MNIST8m masks.

This is a standalone diagnostic. It leaves the bank and VAE artifacts unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from pattern.mnist8m_raw_mlp_bce import (
    FEATURES,
    HIDDEN,
    K,
    balanced_batch,
    exact_masks,
    predict,
)
from pattern.mnist8m_importance_bce import save


def run(task: int, root: Path, out: Path, device: torch.device, *,
        pairs: int, repeats: int, max_steps: int, density: float,
        importance_topk: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    total = FEATURES * HIDDEN
    connections = round(density * total)
    if not 0 < connections <= K:
        raise ValueError(f"density must select between 1 and {K} links")
    suffix = (f"_importance_{density:.4f}".replace(".", "p")
              if importance_topk else "")
    path = out / f"task{task}{suffix}.pt"
    if path.exists():
        raise FileExistsError(path)
    data = torch.load(root / "features.pt", map_location="cpu", weights_only=True)
    bank = torch.load(root / f"bank_task{task}.pt", map_location="cpu",
                      weights_only=True)
    digit = int(bank["digit"])
    if pairs > len(bank["masks"]):
        raise ValueError(f"requested {pairs} pairs, only {len(bank['masks'])} selected")
    if importance_topk:
        importance = bank["importance"][:pairs].reshape(pairs, total).to(device)
        selected_flat = torch.zeros_like(importance)
        selected_flat.scatter_(1, importance.topk(connections, dim=1).indices, 1.)
        selected = selected_flat.reshape(pairs, FEATURES, HIDDEN)
        mask_generator = torch.Generator(device=device).manual_seed(950000 + digit)
        random_values = torch.rand(pairs, total, generator=mask_generator,
                                   device=device)
        random_flat = torch.zeros_like(random_values)
        random_flat.scatter_(1, random_values.topk(connections, dim=1).indices, 1.)
        random = random_flat.reshape(pairs, FEATURES, HIDDEN)
    else:
        if connections != K:
            raise ValueError("bank support comparison uses its original 20% density")
        selected = bank["masks"][:pairs].to(device)
        random = exact_masks(pairs, 950000 + digit, device)
    masks = torch.stack((selected, random)).repeat_interleave(repeats, dim=1)
    count = pairs * repeats

    generator = torch.Generator(device=device).manual_seed(960000 + digit)
    initial_weight = torch.randn(count, FEATURES, HIDDEN, generator=generator,
                                 device=device) * .08
    initial_readout = torch.randn(count, HIDDEN, generator=generator,
                                  device=device) * .08
    weight = torch.nn.Parameter(initial_weight.unsqueeze(0).repeat(2, 1, 1, 1))
    bias = torch.nn.Parameter(torch.zeros(2, count, HIDDEN, device=device))
    readout = torch.nn.Parameter(initial_readout.unsqueeze(0).repeat(2, 1, 1))
    offset = torch.nn.Parameter(torch.zeros(2, count, device=device))
    optimizer = torch.optim.Adam((weight, bias, readout, offset), lr=.003)
    train = tuple(x.to(device) for x in data["train"])
    validation = tuple(x.to(device) for x in data["validation"])
    test = tuple(x.to(device) for x in data["test"])
    train_gen = torch.Generator(device=device).manual_seed(970000 + digit)
    val_gen = torch.Generator(device=device).manual_seed(980000 + digit)
    val_x, val_y = balanced_batch(validation, digit, 512, val_gen)
    best = torch.full((2, count), float("inf"), device=device)
    best_step = torch.zeros(2, count, dtype=torch.int32, device=device)
    best_values = None
    stale = 0
    for step in tqdm(range(1, max_steps + 1), desc=f"digit {digit} paired masks",
                     unit="step", mininterval=2):
        x, y = balanced_batch(train, digit, 128, train_gen)
        logits = predict(x, masks, weight, bias, readout, offset)
        losses = F.binary_cross_entropy_with_logits(
            logits, y.expand_as(logits), reduction="none").mean(-1)
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        optimizer.step()
        if step % 100 != 0:
            continue
        with torch.no_grad():
            logits = predict(val_x, masks, weight, bias, readout, offset)
            positive = val_y.bool()
            val = .5 * (
                F.softplus(-logits[..., positive]).mean(-1)
                + F.softplus(logits[..., ~positive]).mean(-1))
            improved = val < best - 1e-4
            best = torch.where(improved, val, best)
            best_step = torch.where(improved, step, best_step)
            values = (weight.detach(), bias.detach(), readout.detach(),
                      offset.detach())
            if best_values is None:
                best_values = tuple(value.clone() for value in values)
            else:
                best_values = tuple(torch.where(
                    improved.reshape(*improved.shape,
                                     *((1,) * (value.ndim - improved.ndim))),
                    value, previous)
                    for value, previous in zip(values, best_values))
            stale = 0 if improved.any() else stale + 1
            if stale >= 15:
                break
    assert best_values is not None

    bce_positive = torch.zeros(2, count, device=device)
    bce_negative = torch.zeros(2, count, device=device)
    correct_positive = torch.zeros(2, count, device=device)
    correct_negative = torch.zeros(2, count, device=device)
    n_positive = n_negative = 0
    with torch.no_grad():
        test_x, test_digits = test
        for x, labels in zip(test_x.split(256), test_digits.split(256)):
            logits = predict(x, masks, *best_values)
            positive = labels == digit
            negative = ~positive
            n_positive += int(positive.sum())
            n_negative += int(negative.sum())
            bce_positive += F.softplus(-logits[..., positive]).sum(-1)
            bce_negative += F.softplus(logits[..., negative]).sum(-1)
            correct_positive += (logits[..., positive] > 0).sum(-1)
            correct_negative += (logits[..., negative] <= 0).sum(-1)
    bce = .5 * (bce_positive / n_positive + bce_negative / n_negative)
    accuracy = .5 * (correct_positive / n_positive
                     + correct_negative / n_negative)
    payload = {
        "digit": digit,
        "method_order": ["selected_bank", "fresh_random"],
        "balanced_accuracy": accuracy.reshape(2, pairs, repeats).cpu(),
        "balanced_bce": bce.reshape(2, pairs, repeats).cpu(),
        "validation_bce": best.reshape(2, pairs, repeats).cpu(),
        "best_step": best_step.reshape(2, pairs, repeats).cpu(),
        "selected_indices": bank["selected_indices"][:pairs],
        "settings": {"pairs": pairs, "repeats": repeats,
                     "max_steps": max_steps, "actual_steps": step,
                     "importance_topk": importance_topk,
                     "connections": connections,
                     "random_seed": 950000 + digit,
                     "same_initial_weights_within_pair": True,
                     "same_train_batches_within_pair": True,
                     "validation_batch": 512,
                     "test_positive": n_positive,
                     "test_negative": n_negative,
                     "mask_density": connections / total},
    }
    save(payload, path)
    for key in ("balanced_accuracy", "balanced_bce"):
        means = payload[key].mean((1, 2)).tolist()
        print(f"digit {digit}: {key} selected={means[0]:.5f}, "
              f"random={means[1]:.5f}, delta={means[0]-means[1]:+.5f}",
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True, choices=(0, 1))
    parser.add_argument("--root", type=Path,
                        default=Path("pattern/outputs/mnist8m_raw_mlp_bce/pair38"))
    parser.add_argument("--out", type=Path,
                        default=Path("pattern/outputs/mnist8m_raw_mlp_bce/mask_control"))
    parser.add_argument("--pairs", type=int, default=410)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--density", type=float, default=K / (FEATURES * HIDDEN))
    parser.add_argument("--importance-topk", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.task, args.root, args.out, torch.device(args.device),
        pairs=args.pairs, repeats=args.repeats, max_steps=args.max_steps,
        density=args.density, importance_topk=args.importance_topk)


if __name__ == "__main__":
    main()
