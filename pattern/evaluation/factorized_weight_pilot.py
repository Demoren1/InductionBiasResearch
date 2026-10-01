"""Train many vector-factorized first layers on length-8 pattern tasks.

Here the *weights themselves* are outer or Kronecker products of learned
factors. Factor masks give exactly 32 active matrix entries (50%), matching
the native pattern experiment. Dense factorized and free-matrix controls
separate factorization limits from sparsity limits. No VAE is trained here.
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
from models.mlp import generate_fixed_sparsity_masks  # noqa: E402

METHODS = (
    "outer_input4", "outer_hidden4", "outer_dense",
    "kron_left4", "kron_right4", "kron_dense",
    "matrix_random32", "matrix_ideal32", "matrix_dense",
)


def half_masks(n: int, generator: torch.Generator) -> torch.Tensor:
    ranks = torch.rand(n, 8, generator=generator).argsort(1)
    return torch.zeros(n, 8).scatter_(1, ranks[:, :4], 1.0)


class FactorizedWeightMLPs(nn.Module):
    def __init__(self, repeats: int, seed: int, pattern: str):
        super().__init__()
        self.repeats = repeats
        generator = torch.Generator().manual_seed(seed * 10000 + int(pattern, 2))
        outer_n = kron_n = matrix_n = 3 * repeats
        scale = 0.1 ** 0.5
        self.outer_a = nn.Parameter(torch.randn(outer_n, 8, generator=generator) * scale)
        self.outer_b = nn.Parameter(torch.randn(outer_n, 8, generator=generator) * scale)
        self.kron_a = nn.Parameter(torch.randn(kron_n, 2, 4, generator=generator) * scale)
        self.kron_b = nn.Parameter(torch.randn(kron_n, 4, 2, generator=generator) * scale)
        self.matrix_w = nn.Parameter(torch.randn(matrix_n, 8, 8, generator=generator) * 0.1)
        output_init = torch.randn(repeats, 8, generator=generator) * 0.1
        self.w2 = nn.Parameter(output_init.repeat(len(METHODS), 1))
        self.b1 = nn.Parameter(torch.zeros(len(METHODS) * repeats, 8))
        self.b2 = nn.Parameter(torch.zeros(len(METHODS) * repeats))

        outer_a_mask = torch.ones(outer_n, 8)
        outer_b_mask = torch.ones(outer_n, 8)
        outer_a_mask[:repeats] = half_masks(repeats, generator)
        outer_b_mask[repeats:2 * repeats] = half_masks(repeats, generator)
        self.register_buffer("outer_a_mask", outer_a_mask)
        self.register_buffer("outer_b_mask", outer_b_mask)

        kron_a_mask = torch.ones(kron_n, 8)
        kron_b_mask = torch.ones(kron_n, 8)
        kron_a_mask[:repeats] = half_masks(repeats, generator)
        kron_b_mask[repeats:2 * repeats] = half_masks(repeats, generator)
        self.register_buffer("kron_a_mask", kron_a_mask.reshape(kron_n, 2, 4))
        self.register_buffer("kron_b_mask", kron_b_mask.reshape(kron_n, 4, 2))

        matrix_mask = torch.ones(matrix_n, 8, 8)
        matrix_mask[:repeats] = generate_fixed_sparsity_masks(
            repeats, 8, 8, 32, seed=seed * 1000 + int(pattern, 2))
        matrix_mask[repeats:2 * repeats] = ideal_mask().float()
        self.register_buffer("matrix_mask", matrix_mask)

    def first_layer(self) -> torch.Tensor:
        a = self.outer_a * self.outer_a_mask
        b = self.outer_b * self.outer_b_mask
        outer = a[:, :, None] * b[:, None, :]
        a = self.kron_a * self.kron_a_mask
        b = self.kron_b * self.kron_b_mask
        kron = (a[:, :, :, None, None] * b[:, None, None, :, :])
        kron = kron.permute(0, 1, 3, 2, 4).reshape(-1, 8, 8)
        matrix = self.matrix_w * self.matrix_mask
        return torch.cat((outer, kron, matrix), dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(torch.einsum("bi,mih->bmh", x, self.first_layer()) + self.b1)
        return torch.einsum("bmh,mh->bm", hidden, self.w2) + self.b2


@torch.no_grad()
def evaluate(model: FactorizedWeightMLPs, data: dict, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = data["x"].to(model.b1.device)
    y = data["y"].to(model.b1.device)
    bce = torch.zeros(len(model.b1), device=x.device)
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
    model = FactorizedWeightMLPs(args.repeats, args.seed, pattern).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
    best_val = torch.full((n,), float("inf"), device=device)
    best_step = torch.zeros(n, dtype=torch.int64, device=device)
    best = {name: torch.empty_like(getattr(model, name)) for name in (
        "outer_a", "outer_b", "kron_a", "kron_b", "matrix_w", "w2", "b1", "b2")}
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(pool_x), (args.batch_size,),
                                generator=batch_generator, device=device)
        x, y = pool_x[indices], pool_y[indices]
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y[:, None].expand_as(logits))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            val_bce, _ = evaluate(model, validation, args.eval_batch_size)
            improved = val_bce < best_val
            best_val = torch.where(improved, val_bce, best_val)
            best_step[improved] = step
            r = args.repeats
            for name in ("outer_a", "outer_b", "kron_a", "kron_b", "matrix_w"):
                start = {"outer_a": 0, "outer_b": 0, "kron_a": 3*r,
                         "kron_b": 3*r, "matrix_w": 6*r}[name]
                part = improved[start:start + 3*r]
                best[name][part] = getattr(model, name).detach()[part]
            for name in ("w2", "b1", "b2"):
                best[name][improved] = getattr(model, name).detach()[improved]
            model.train()
    with torch.no_grad():
        for name, tensor in best.items():
            getattr(model, name).copy_(tensor)
    model.eval()
    test_bce, test_acc = evaluate(model, test, args.eval_batch_size)
    first_layer = model.first_layer().detach().cpu()
    active = (first_layer != 0).sum((1, 2))
    expected = torch.tensor([32, 32, 64, 32, 32, 64, 32, 32, 64]).repeat_interleave(args.repeats)
    if not torch.equal(active, expected):
        raise AssertionError(f"active counts differ: {active.tolist()}")
    state = {
        "pattern": pattern,
        "methods": METHODS,
        "repeats": args.repeats,
        "best_val_bce": best_val.cpu(),
        "best_steps": best_step.cpu(),
        "test_acc": test_acc.cpu(),
        "test_bce": test_bce.cpu(),
        "first_layer": first_layer,
        "outer_a_mask": model.outer_a_mask.cpu(),
        "outer_b_mask": model.outer_b_mask.cpu(),
        "kron_a_mask": model.kron_a_mask.cpu(),
        "kron_b_mask": model.kron_b_mask.cpu(),
        "matrix_mask": model.matrix_mask.cpu(),
        **{name: tensor.cpu() for name, tensor in best.items()},
    }
    torch.save(state, args.out.parent / f"pattern_{pattern}_states.pt")
    result = {}
    for index, method in enumerate(METHODS):
        sl = slice(index * args.repeats, (index + 1) * args.repeats)
        local_val = best_val[sl]
        order = local_val.argsort()
        top_n = max(1, round(args.repeats * 0.1))
        result[method] = {
            "mean_test_acc": float(test_acc[sl].mean()),
            "mean_test_bce": float(test_bce[sl].mean()),
            "top10pct_test_acc": float(test_acc[sl][order[:top_n]].mean()),
            "top10pct_test_bce": float(test_bce[sl][order[:top_n]].mean()),
            "best_val_selected_test_acc": float(test_acc[sl][order[0]]),
            "active_connections": int(expected[sl][0]),
            "fraction_best_at_limit": float((best_step[sl] == args.steps).float().mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-pool-size", type=int, default=32768)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patterns", nargs="+", default=config.PATTERNS)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--out", type=Path,
                        default=config.OUTPUTS / "factorized_weight_pilot" / "summary.json")
    args = parser.parse_args()
    if args.repeats < 1 or args.steps < 1 or args.train_pool_size < args.batch_size:
        parser.error("invalid repeats, steps, or training pool size")
    if any(pattern not in config.PATTERNS for pattern in args.patterns):
        parser.error("unknown pattern")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"protocol": {
        "seed": args.seed,
        "patterns": args.patterns,
        "repeats_per_method_per_pattern": args.repeats,
        "steps": args.steps,
        "eval_every": args.eval_every,
        "train_pool_size": args.train_pool_size,
        "eval_samples": args.eval_samples,
        "sparse_connections": 32,
        "factorization": "first-layer weights = masked a outer masked b or masked A kron masked B",
        "selection": "per-model checkpoint by validation BCE; top 10% by validation BCE",
        "device": str(device),
    }, "patterns": {}}
    for pattern in args.patterns:
        row = run_pattern(pattern, args, device)
        summary["patterns"][pattern] = row
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[factorized-weight] {pattern} " + " ".join(
            f"{name}={row[name]['top10pct_test_acc']:.4f}" for name in METHODS), flush=True)
    summary["macro_top10pct_test_acc"] = {
        name: sum(row[name]["top10pct_test_acc"] for row in summary["patterns"].values())
        / len(summary["patterns"]) for name in METHODS
    }
    summary["macro_mean_test_acc"] = {
        name: sum(row[name]["mean_test_acc"] for row in summary["patterns"].values())
        / len(summary["patterns"]) for name in METHODS
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print("[factorized-weight] macro top10", summary["macro_top10pct_test_acc"], flush=True)


if __name__ == "__main__":
    main()
