"""End-to-end learned vector masks on the native length-8 pattern task.

The learned mask scores are an outer product of two 8-vectors or a Kronecker
product of 2x4 and 4x2 factors. An unconstrained 8x8 score matrix, a fixed
random exact-32 mask, the ideal mask, and a dense network are paired controls.
All sparse methods use hard top-32 masks in the forward pass and a straight-
through gradient for learned scores. This is a within-task feasibility pilot:
mask parameters see pattern labels during training, unlike label-free VAEs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from data.generate import ideal_mask, make_dataset  # noqa: E402
from evaluation.ood_split import make_split  # noqa: E402
from models.mlp import BatchedMaskedMLP, generate_fixed_sparsity_masks  # noqa: E402

METHODS = ("outer", "kronecker", "matrix", "random", "ideal", "dense")


def hard_topk_st(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Exact-k binary forward mask with an identity straight-through gradient."""
    flat = scores.flatten(1)
    indices = flat.topk(k, dim=1).indices
    hard = torch.zeros_like(flat).scatter_(1, indices, 1.0).view_as(scores)
    return hard + scores - scores.detach()


class LearnedMaskMLPs(nn.Module):
    def __init__(self, repeats: int, seed: int, pattern: str):
        super().__init__()
        self.repeats = repeats
        n = len(METHODS) * repeats
        generator = torch.Generator().manual_seed(seed * 10000 + int(pattern, 2))
        initial_w1 = torch.randn(repeats, 8, 8, generator=generator) * 0.1
        initial_w2 = torch.randn(repeats, 8, generator=generator) * 0.1
        self.w1 = nn.Parameter(initial_w1.repeat(len(METHODS), 1, 1))
        self.b1 = nn.Parameter(torch.zeros(n, 8))
        self.w2 = nn.Parameter(initial_w2.repeat(len(METHODS), 1))
        self.b2 = nn.Parameter(torch.zeros(n))
        self.outer_input = nn.Parameter(torch.randn(repeats, 8, generator=generator) * 0.1)
        self.outer_hidden = nn.Parameter(torch.randn(repeats, 8, generator=generator) * 0.1)
        self.kron_left = nn.Parameter(torch.randn(repeats, 2, 4, generator=generator) * 0.1)
        self.kron_right = nn.Parameter(torch.randn(repeats, 4, 2, generator=generator) * 0.1)
        self.matrix_scores = nn.Parameter(torch.randn(repeats, 8, 8, generator=generator) * 0.1)
        random_masks = generate_fixed_sparsity_masks(
            repeats, 8, 8, 32, seed=seed * 1000 + int(pattern, 2))
        self.register_buffer("random_masks", random_masks)
        self.register_buffer("ideal_masks", ideal_mask().float().expand(repeats, -1, -1))
        self.register_buffer("dense_masks", torch.ones(repeats, 8, 8))

    def masks(self) -> torch.Tensor:
        a = torch.sigmoid(self.outer_input)
        b = torch.sigmoid(self.outer_hidden)
        outer = a[:, :, None] * b[:, None, :]
        left = torch.sigmoid(self.kron_left)
        right = torch.sigmoid(self.kron_right)
        # Batched Kronecker: (2,4) kron (4,2) -> (8,8).
        kron = (left[:, :, :, None, None] * right[:, None, None, :, :])
        kron = kron.permute(0, 1, 3, 2, 4).reshape(self.repeats, 8, 8)
        matrix = torch.sigmoid(self.matrix_scores)
        return torch.cat((
            hard_topk_st(outer, 32), hard_topk_st(kron, 32),
            hard_topk_st(matrix, 32), self.random_masks,
            self.ideal_masks, self.dense_masks,
        ), dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_w1 = self.w1 * self.masks()
        hidden = F.relu(torch.einsum("bi,mih->bmh", x, masked_w1) + self.b1)
        return torch.einsum("bmh,mh->bm", hidden, self.w2) + self.b2


@torch.no_grad()
def evaluate(model: LearnedMaskMLPs, data: dict, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = data["x"].to(model.w1.device)
    y = data["y"].to(model.w1.device)
    bce = torch.zeros(len(model.w1), device=x.device)
    correct = torch.zeros_like(bce)
    for start in range(0, len(x), batch_size):
        xb, yb = x[start:start + batch_size], y[start:start + batch_size]
        logits = model(xb)
        target = yb[:, None].expand_as(logits)
        bce += F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(0)
        correct += ((logits > 0) == (target > 0.5)).sum(0)
    return bce / len(x), correct / len(x)


def run_pattern(pattern: str, args: argparse.Namespace, device: torch.device) -> dict:
    torch.manual_seed(args.seed + int(pattern, 2))
    model = LearnedMaskMLPs(args.repeats, args.seed, pattern).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.train_pool_size:
        pool = make_dataset(pattern, args.train_pool_size,
                            seed=100000 + int(pattern, 2) * 10000,
                            pos_fraction=config.POS_FRACTION)
        pool_x, pool_y = pool["x"].to(device), pool["y"].to(device)
        batch_generator = torch.Generator(device=device).manual_seed(
            args.seed * 100000 + int(pattern, 2))
    validation = make_dataset(pattern, args.eval_samples,
                              seed=200000 + int(pattern, 2),
                              pos_fraction=config.POS_FRACTION)
    test = make_dataset(pattern, args.eval_samples,
                        seed=300000 + int(pattern, 2),
                        pos_fraction=config.POS_FRACTION)
    n = len(METHODS) * args.repeats
    best_val_bce = torch.full((n,), float("inf"), device=device)
    best_val_acc = torch.zeros(n, device=device)
    best_mask = torch.zeros(n, 8, 8, device=device)
    best_w1 = torch.empty_like(model.w1)
    best_b1 = torch.empty_like(model.b1)
    best_w2 = torch.empty_like(model.w2)
    best_b2 = torch.empty_like(model.b2)
    best_outer_input = torch.empty_like(model.outer_input)
    best_outer_hidden = torch.empty_like(model.outer_hidden)
    best_kron_left = torch.empty_like(model.kron_left)
    best_kron_right = torch.empty_like(model.kron_right)
    best_matrix_scores = torch.empty_like(model.matrix_scores)
    best_step = torch.zeros(n, dtype=torch.int64, device=device)
    for step in range(1, args.steps + 1):
        if args.train_pool_size:
            indices = torch.randint(len(pool_x), (args.batch_size,),
                                    generator=batch_generator, device=device)
            x, y = pool_x[indices], pool_y[indices]
        else:
            train = make_dataset(pattern, args.batch_size,
                                 seed=100000 + int(pattern, 2) * 10000 + step,
                                 pos_fraction=config.POS_FRACTION)
            x, y = train["x"].to(device), train["y"].to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y[:, None].expand_as(logits))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            val_bce, val_acc = evaluate(model, validation, args.eval_batch_size)
            improved = val_bce < best_val_bce
            best_val_bce = torch.where(improved, val_bce, best_val_bce)
            best_val_acc = torch.where(improved, val_acc, best_val_acc)
            best_mask = torch.where(improved[:, None, None], model.masks().detach(), best_mask)
            best_w1[improved] = model.w1.detach()[improved]
            best_b1[improved] = model.b1.detach()[improved]
            best_w2[improved] = model.w2.detach()[improved]
            best_b2[improved] = model.b2.detach()[improved]
            outer_improved = improved[:args.repeats]
            kron_improved = improved[args.repeats:2 * args.repeats]
            matrix_improved = improved[2 * args.repeats:3 * args.repeats]
            best_outer_input[outer_improved] = model.outer_input.detach()[outer_improved]
            best_outer_hidden[outer_improved] = model.outer_hidden.detach()[outer_improved]
            best_kron_left[kron_improved] = model.kron_left.detach()[kron_improved]
            best_kron_right[kron_improved] = model.kron_right.detach()[kron_improved]
            best_matrix_scores[matrix_improved] = model.matrix_scores.detach()[matrix_improved]
            best_step = torch.where(improved, torch.full_like(best_step, step), best_step)
            model.train()
    fixed = BatchedMaskedMLP(n, 8, 8).to(device)
    fixed.load_masks(best_mask)
    with torch.no_grad():
        fixed.w1.copy_(best_w1)
        fixed.b1.copy_(best_b1)
        fixed.w2.copy_(best_w2.unsqueeze(-1))
        fixed.b2.copy_(best_b2.unsqueeze(-1))
    best_test_bce, best_test_acc = evaluate(fixed, test, args.eval_batch_size)
    torch.save({
        "pattern": pattern,
        "methods": METHODS,
        "repeats": args.repeats,
        "mask": best_mask.cpu(),
        "w1": best_w1.cpu(),
        "b1": best_b1.cpu(),
        "w2": best_w2.cpu(),
        "b2": best_b2.cpu(),
        "outer_input_logits": best_outer_input.cpu(),
        "outer_hidden_logits": best_outer_hidden.cpu(),
        "kron_left_logits": best_kron_left.cpu(),
        "kron_right_logits": best_kron_right.cpu(),
        "matrix_logits": best_matrix_scores.cpu(),
        "best_val_bce": best_val_bce.cpu(),
        "best_step": best_step.cpu(),
    }, args.out.parent / f"pattern_{pattern}_states.pt")
    gold = ideal_mask().float().to(device)
    methods = {}
    for index, method in enumerate(METHODS):
        sl = slice(index * args.repeats, (index + 1) * args.repeats)
        masks = best_mask[sl]
        intersection = (masks * gold).sum((1, 2))
        union = masks.sum((1, 2)) + gold.sum() - intersection
        methods[method] = {
            "test_acc": float(best_test_acc[sl].mean()),
            "test_bce": float(best_test_bce[sl].mean()),
            "val_acc": float(best_val_acc[sl].mean()),
            "val_bce": float(best_val_bce[sl].mean()),
            "test_acc_repeats": best_test_acc[sl].cpu().tolist(),
            "test_bce_repeats": best_test_bce[sl].cpu().tolist(),
            "best_steps": best_step[sl].cpu().tolist(),
            "active_connections": masks.sum((1, 2)).cpu().tolist(),
            "gold_iou": float((intersection / union).mean()),
            "masks": masks.cpu().int().tolist(),
        }
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-pool-size", type=int, default=0,
                        help="optional cached training pool; 0 uses fresh batches")
    parser.add_argument("--all-patterns", action="store_true",
                        help="evaluate all 16 pattern tasks instead of four held-out tasks")
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--out", type=Path,
                        default=config.OUTPUTS / "vector_outer_pilot" / "summary.json")
    args = parser.parse_args()
    if args.repeats < 1 or args.steps < 1 or args.eval_every < 1:
        parser.error("repeats, steps and eval-every must be positive")
    if (config.SEQ_LEN, config.H, config.K_ACTIVE) != (8, 8, 32):
        parser.error("This pilot assumes the native 8x8, exact-32 task")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    split = make_split(args.split_seed)
    patterns = list(config.PATTERNS if args.all_patterns else split["test_patterns"])
    summary = {
        "protocol": {
            "split": split,
            "evaluated_patterns": patterns,
            "seed": args.seed,
            "repeats": args.repeats,
            "steps": args.steps,
            "eval_every": args.eval_every,
            "train_batch_size": args.batch_size,
            "train_pool_size": args.train_pool_size,
            "validation_and_test_samples_each": args.eval_samples,
            "mask_shape": [8, 8],
            "sparse_connections": 32,
            "selection": "minimum validation BCE for each method and repeat",
            "final_metric": "fresh-test accuracy/BCE at validation-selected checkpoint",
            "mask_training": "supervised end-to-end with straight-through hard top-32",
            "device": str(device),
        },
        "patterns": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        summary["patterns"][pattern] = run_pattern(pattern, args, device)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        row = summary["patterns"][pattern]
        print(f"[vector-outer] {pattern} " + " ".join(
            f"{name}={item['test_acc']:.4f}" for name, item in row.items()), flush=True)
    summary["macro_test_acc"] = {
        name: sum(row[name]["test_acc"] for row in summary["patterns"].values())
        / len(summary["patterns"]) for name in METHODS
    }
    summary["macro_test_bce"] = {
        name: sum(row[name]["test_bce"] for row in summary["patterns"].values())
        / len(summary["patterns"]) for name in METHODS
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print("[vector-outer] macro", summary["macro_test_acc"], flush=True)


if __name__ == "__main__":
    main()
