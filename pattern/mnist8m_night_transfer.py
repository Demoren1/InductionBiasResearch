"""Retrain one 10-class MLP per frozen mask on the MNIST8m digit split."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pattern.mnist8m_importance_bce import save


FEATURES, HIDDEN, CLASSES = 784, 64, 10


def capped_topk(scores: torch.Tensor, connections: int, cap: int) -> torch.Tensor:
    if scores.shape != (FEATURES, HIDDEN):
        raise ValueError(f"unexpected map shape: {scores.shape}")
    if cap * FEATURES < connections:
        raise ValueError("pixel cap cannot fit the requested connections")
    allowed = torch.zeros_like(scores, dtype=torch.bool)
    allowed.scatter_(1, scores.topk(cap, dim=1).indices, True)
    eligible = scores.masked_fill(~allowed, -torch.inf).flatten()
    mask = torch.zeros_like(eligible)
    mask.scatter_(0, eligible.topk(connections).indices, 1.)
    return mask.reshape(FEATURES, HIDDEN)


def build_masks(source: Path, density: float) -> tuple[list[str], torch.Tensor, tuple[int, int]]:
    banks = [torch.load(source / f"bank_task{task}.pt", map_location="cpu",
                        weights_only=True, mmap=True) for task in (0, 1)]
    pair = tuple(int(bank["digit"]) for bank in banks)
    features = torch.load(source / "features.pt", map_location="cpu", weights_only=True)
    if tuple(features["pair"]) != pair:
        raise ValueError("features and bank use different digit pairs")
    connections = round(density * FEATURES * HIDDEN)
    cap = min(HIDDEN, math.ceil(connections / 650))
    for bank in banks:
        if int(bank["masks"][0].sum()) != connections:
            raise ValueError("bank density does not match requested density")
    result = torch.load(source / "search_shared_lambda1.pt", map_location="cpu",
                        weights_only=True)
    logits = result["logits"][:, result["chosen_start"]]
    means = [bank["importance"].mean(0) for bank in banks]
    names = ["agreement", "mean", "vae0", "vae1", "direct0", "direct1"]
    masks = [capped_topk(logits.mean(0), connections, cap),
             capped_topk(torch.stack(means).mean(0), connections, cap),
             capped_topk(logits[0], connections, cap),
             capped_topk(logits[1], connections, cap),
             banks[0]["masks"][0].float(),
             banks[1]["masks"][0].float()]
    for index in range(4):
        generator = torch.Generator().manual_seed(20260927 + index)
        names.append(f"random{index}")
        masks.append(capped_topk(torch.rand(FEATURES, HIDDEN, generator=generator),
                                 connections, cap))
    names.append("dense")
    masks.append(torch.ones(FEATURES, HIDDEN))
    return names, torch.stack(masks), pair


def predict(x: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor,
            bias: torch.Tensor, readout: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    hidden = torch.tanh(torch.einsum("bf,nfh->nbh", x, weight * mask) + bias[:, None])
    return torch.einsum("nbh,nhc->nbc", hidden, readout) + offset[:, None]


def plot_masks(payload: dict, path: Path) -> None:
    names = payload["names"]
    display = (("agreement", "Agreement"), ("mean", "Средняя"),
               ("vae0", "VAE 0"), ("vae1", "VAE 1"),
               ("direct0", "Карта 0"), ("direct1", "Карта 1"),
               ("random0", "Случайная"))
    fig, axes = plt.subplots(2, 4, figsize=(12, 9))
    for ax, (name, label) in zip(axes.flat, display):
        ax.imshow(payload["masks"][names.index(name)].numpy(), aspect="auto",
                  interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
        ax.set_title(label)
        ax.set_xlabel("Скрытый нейрон")
        ax.set_ylabel("Пиксель")
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--density", type=float, choices=(.02, .05), required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps < 100:
        parser.error("--steps must be at least 100")
    if args.out.exists():
        plot_path = args.out.with_name("transfer_masks.png")
        if not plot_path.exists():
            payload = torch.load(args.out, map_location="cpu", weights_only=True)
            plot_masks(payload, plot_path)
        print(f"reusing {args.out}", flush=True)
        return
    torch.set_num_threads(2)
    device = torch.device(args.device)
    names, masks_cpu, pair = build_masks(args.source, args.density)
    features = torch.load(args.source / "features.pt", map_location="cpu", weights_only=True)
    visible_digits = features.get("visible_digits", list(range(CLASSES)))
    heldout_digits = [digit for digit in range(CLASSES) if digit not in visible_digits]
    train = tuple(t.to(device) for t in features["train"])
    validation = tuple(t.to(device) for t in features["validation"])
    test = tuple(t.to(device) for t in features["test"])
    repeats = 4
    methods = len(names)
    count = methods * repeats
    masks = masks_cpu.to(device).repeat_interleave(repeats, dim=0)
    generator = torch.Generator(device=device).manual_seed(272700)
    initial_weight = torch.randn(repeats, FEATURES, HIDDEN, generator=generator,
                                 device=device) * .08
    initial_readout = torch.randn(repeats, HIDDEN, CLASSES, generator=generator,
                                  device=device) * .08
    weight = torch.nn.Parameter(initial_weight.repeat(methods, 1, 1))
    bias = torch.nn.Parameter(torch.zeros(count, HIDDEN, device=device))
    readout = torch.nn.Parameter(initial_readout.repeat(methods, 1, 1))
    offset = torch.nn.Parameter(torch.zeros(count, CLASSES, device=device))
    optimizer = torch.optim.Adam((weight, bias, readout, offset), lr=.003)
    best = torch.full((count,), float("inf"), device=device)
    best_step = torch.zeros(count, dtype=torch.int32, device=device)
    best_values = None
    stale = 0
    train_x, train_y = train
    val_x, val_y = validation
    for step in tqdm(range(1, args.steps + 1), desc=f"10-class {pair}", unit="step",
                     mininterval=2):
        indices = torch.randint(len(train_y), (128,), generator=generator, device=device)
        logits = predict(train_x[indices], masks, weight, bias, readout, offset)
        losses = F.cross_entropy(logits.transpose(1, 2),
                                 train_y[indices].expand(count, -1), reduction="none")
        optimizer.zero_grad(set_to_none=True)
        losses.mean(-1).sum().backward()
        optimizer.step()
        if step % 100 == 0:
            with torch.no_grad():
                total = torch.zeros(count, device=device)
                for x, labels in zip(val_x.split(256), val_y.split(256)):
                    prediction = predict(x, masks, weight, bias, readout, offset)
                    total += F.cross_entropy(prediction.transpose(1, 2),
                                             labels.expand(count, -1),
                                             reduction="none").sum(-1)
                score = total / len(val_y)
                improved = score < best - 1e-4
                best = torch.where(improved, score, best)
                best_step = torch.where(improved, step, best_step)
                values = (weight.detach(), bias.detach(), readout.detach(),
                          offset.detach())
                if best_values is None:
                    best_values = tuple(value.clone() for value in values)
                else:
                    best_values = tuple(torch.where(
                        improved.reshape(-1, *((1,) * (value.ndim - 1))),
                        value, prior) for value, prior in zip(values, best_values))
                stale = 0 if improved.any() else stale + 1
                if stale >= 15:
                    break
    assert best_values is not None
    test_x, test_y = test
    correct = torch.zeros(count, device=device)
    by_digit = torch.zeros(count, CLASSES, device=device)
    for x, labels in zip(test_x.split(256), test_y.split(256)):
        with torch.no_grad():
            prediction = predict(x, masks, *best_values).argmax(-1)
            matches = prediction == labels
            correct += matches.sum(-1)
            for digit in range(CLASSES):
                by_digit[:, digit] += matches[:, labels == digit].sum(-1)
    counts = torch.bincount(test_y, minlength=CLASSES).to(device)
    payload = {"names": names, "pair": pair, "density": args.density,
               "visible_digits": visible_digits, "heldout_digits": heldout_digits,
               "masks": masks_cpu, "accuracy": (correct / len(test_y)).reshape(methods, repeats).cpu(),
               "recall": (by_digit / counts).reshape(methods, repeats, CLASSES).permute(0, 2, 1).cpu(),
               "validation_ce": best.reshape(methods, repeats).cpu(),
               "best_step": best_step.reshape(methods, repeats).cpu(),
               "settings": {"max_steps": args.steps, "actual_steps": step,
                            "plateau": stale >= 15, "repeats": repeats,
                            "train_images": len(train_y), "validation_images": len(val_y),
                            "test_images": len(test_y),
                            "selection": "one fixed first anchor per pair and density"}}
    save(payload, args.out)
    plot_masks(payload, args.out.with_name("transfer_masks.png"))
    print(f"saved {args.out}; steps={step}; plateau={stale >= 15}", flush=True)


if __name__ == "__main__":
    main()
