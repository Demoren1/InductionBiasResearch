"""Test whether directly meta-learning U helps on MNIST8m digit-sum tasks.

Every task assigns a different real-valued score to each digit.  A set's label
is the sum of its digits' task scores.  The 784->300->100 image encoder and the
30->1 readout are shared; only v in the final 100->30 layer adapts on labeled
support sets.  U is learned across tasks or fixed at the same random start.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


INPUTS = 100
OUTPUTS = 30
WEIGHTS = (INPUTS + 1) * OUTPUTS  # Includes the last layer's bias row.


class ImageSum(nn.Module):
    def __init__(self, *, learn_u: bool, rank: int = 16, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.first = nn.Linear(784, 300)
        self.second = nn.Linear(300, INPUTS)
        self.readout = nn.Linear(OUTPUTS, 1, bias=False)
        for layer in (self.first, self.second, self.readout):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        initial_u = torch.randn(WEIGHTS, rank)
        initial_u = F.normalize(initial_u, dim=0) * math.sqrt(WEIGHTS) * 0.05
        if learn_u:
            self.u_raw = nn.Parameter(initial_u)
        else:
            self.register_buffer("u_raw", initial_u)
        self.initial_v = nn.Parameter(torch.randn(rank) * 0.1)
        self.rank = rank
        self.learn_u = learn_u

    def u(self) -> torch.Tensor:
        return F.normalize(self.u_raw, dim=0) * math.sqrt(WEIGHTS) * 0.05

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.second(torch.tanh(self.first(images))))

    def predict(self, hidden: torch.Tensor, mask: torch.Tensor,
                v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        last_weight = (u @ v).reshape(INPUTS + 1, OUTPUTS)
        with_bias = F.pad(hidden, (0, 1), value=1.0)
        last = torch.tanh(with_bias @ last_weight)
        per_image = self.readout(last).squeeze(-1)
        return (per_image * mask).sum(dim=1)


def load_pool(data_dir: Path, *, seed: int, per_digit: int,
              split: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(split)
    images = np.load(data_dir / "images.npy", mmap_mode="r")
    labels = np.load(data_dir / "labels.npy", mmap_mode="r")
    if images.shape != (8_100_000, 784) or labels.shape != (8_100_000,):
        raise ValueError("Expected converted MNIST8m arrays")
    # Training uses the first 1/8 of MNIST8m, as in the released benchmark;
    # validation and test use separate later blocks, with no shared images.
    block = len(labels) // 8
    block_id = {"train": 0, "validation": 1, "test": 2}[split]
    lo, hi = block_id * block, (block_id + 1) * block
    local_labels = np.asarray(labels[lo:hi])
    rng = np.random.default_rng(seed + 1009 * block_id)
    ids = []
    for digit in range(10):
        candidates = np.flatnonzero(local_labels == digit)
        if len(candidates) < per_digit:
            raise ValueError(f"Insufficient images of digit {digit}")
        ids.append(rng.choice(candidates, per_digit, replace=False) + lo)
    ids = np.concatenate(ids)
    rng.shuffle(ids)
    # Keep source pixels as uint8 on device; cast only the sampled episode.
    pixels = torch.from_numpy(np.asarray(images[ids], dtype=np.uint8)).to(device)
    digits = torch.from_numpy(np.asarray(labels[ids], dtype=np.int64)).to(device)
    return pixels, digits


def sample_episode(pool: tuple[torch.Tensor, torch.Tensor], *, support: int,
                   query: int, max_train_length: int, rng: torch.Generator,
                   query_length: int | None = None,
                   digit_scores: torch.Tensor | None = None) -> tuple[tuple, tuple]:
    pixels, digits = pool
    device = pixels.device
    q_capacity = query_length or max_train_length
    n_slots = support * max_train_length + query * q_capacity
    if n_slots > len(pixels):
        raise ValueError(f"Need {n_slots} distinct images, pool has {len(pixels)}")
    chosen = torch.randperm(len(pixels), generator=rng, device=device)[:n_slots]
    support_ids = chosen[:support * max_train_length].reshape(support, max_train_length)
    query_ids = chosen[support * max_train_length:].reshape(query, q_capacity)
    support_lengths = torch.randint(1, max_train_length + 1, (support,),
                                    generator=rng, device=device)
    query_lengths = (torch.full((query,), query_length, dtype=torch.long,
                                device=device) if query_length is not None else
                     torch.randint(1, max_train_length + 1, (query,),
                                   generator=rng, device=device))
    if digit_scores is None:
        digit_scores = torch.randn(10, generator=rng, device=device)

    def make(ids: torch.Tensor, lengths: torch.Tensor) -> tuple:
        mask = torch.arange(ids.shape[1], device=device)[None] < lengths[:, None]
        x = pixels[ids].float().div_(255.0)
        target = (digit_scores[digits[ids]] * mask).sum(dim=1)
        return x, mask, target, lengths

    return make(support_ids, support_lengths), make(query_ids, query_lengths)


def normalized_mse(prediction: torch.Tensor, target: torch.Tensor,
                   lengths: torch.Tensor) -> torch.Tensor:
    return ((prediction - target).square() / lengths).mean()


def adapt_query(model: ImageSum, support: tuple, query: tuple, *,
                steps: int, lr: float, meta_gradient: bool) -> tuple:
    sx, smask, sy, slength = support
    qx, qmask, qy, qlength = query
    support_h = model.encode(sx)
    query_h = model.encode(qx)
    u = model.u()
    v = model.initial_v
    for _ in range(steps):
        support_prediction = model.predict(support_h, smask, v, u)
        support_loss = normalized_mse(support_prediction, sy, slength)
        gradient, = torch.autograd.grad(support_loss, v,
                                        create_graph=meta_gradient)
        v = v - lr * gradient
    prediction = model.predict(query_h, qmask, v, u)
    return normalized_mse(prediction, qy, qlength), (prediction - qy).abs().mean()


def evaluate(model: ImageSum, pool: tuple, *, seed: int, tasks: int,
             support: int, query: int, lr: float,
             horizons: tuple[int, ...], lengths: tuple[int, ...]) -> dict:
    model.eval()
    pixels, _ = pool
    rng = torch.Generator(device=pixels.device).manual_seed(seed)
    scores = [torch.randn(10, generator=rng, device=pixels.device)
              for _ in range(tasks)]
    results: dict[str, dict] = {}
    for length in lengths:
        by_horizon = {str(h): {"normalized_mse": [], "mae": []} for h in horizons}
        for score in scores:
            support_episode, query_episode = sample_episode(
                pool, support=support, query=query, max_train_length=5,
                rng=rng, query_length=length, digit_scores=score)
            sx, smask, sy, slength = support_episode
            qx, qmask, qy, qlength = query_episode
            # Gradients are needed only for the freshly adapted v.
            with torch.no_grad():
                sh, qh, u = model.encode(sx), model.encode(qx), model.u()
            v = model.initial_v.detach().clone().requires_grad_(True)
            for step in range(max(horizons) + 1):
                if step in horizons:
                    prediction = model.predict(qh, qmask, v, u)
                    by_horizon[str(step)]["normalized_mse"].append(
                        normalized_mse(prediction, qy, qlength).detach().item())
                    by_horizon[str(step)]["mae"].append(
                        (prediction - qy).abs().mean().detach().item())
                if step == max(horizons):
                    break
                support_prediction = model.predict(sh, smask, v, u)
                support_loss = normalized_mse(support_prediction, sy, slength)
                gradient, = torch.autograd.grad(support_loss, v)
                v = (v - lr * gradient).detach().requires_grad_(True)
        results[str(length)] = {
            step: {metric: float(np.mean(values)) for metric, values in metrics.items()}
            | {"mae_per_task": metrics["mae"]}
            for step, metrics in by_horizon.items()}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("learned", "random"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mnist8m"))
    parser.add_argument("--out", type=Path,
                        default=Path("deepsets_z/mnist8m/outputs/meta_u"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--inner-lr", type=float, default=0.5)
    parser.add_argument("--outer-lr", type=float, default=0.0003)
    parser.add_argument("--tasks-per-step", type=int, default=2)
    parser.add_argument("--support", type=int, default=32)
    parser.add_argument("--query", type=int, default=32)
    parser.add_argument("--train-images-per-digit", type=int, default=2000)
    parser.add_argument("--eval-images-per-digit", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--val-tasks", type=int, default=8)
    parser.add_argument("--test-tasks", type=int, default=32)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    counts = (args.steps, args.inner_steps, args.tasks_per_step, args.support,
              args.query, args.train_images_per_digit,
              args.eval_images_per_digit, args.eval_every, args.val_tasks,
              args.test_tasks, args.rank)
    if min(counts) < 1 or min(args.inner_lr, args.outer_lr) <= 0:
        parser.error("counts and learning rates must be positive")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    train = load_pool(args.data_dir, seed=args.seed, split="train",
                      per_digit=args.train_images_per_digit, device=device)
    validation = load_pool(args.data_dir, seed=args.seed, split="validation",
                           per_digit=args.eval_images_per_digit, device=device)
    test = load_pool(args.data_dir, seed=args.seed, split="test",
                     per_digit=args.eval_images_per_digit, device=device)
    model = ImageSum(learn_u=args.arm == "learned", rank=args.rank,
                     seed=args.seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_lr)
    rng = torch.Generator(device=device).manual_seed(args.seed + 1200)
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"{args.arm}_seed{args.seed}"
    result_path = args.out / f"{name}.json"
    latest_path = args.out / f"{name}_latest.pt"
    best_path = args.out / f"{name}_best.pt"
    if result_path.exists() and not args.resume:
        raise FileExistsError(f"Result already exists: {result_path}; use --resume")
    history: list[dict] = []
    best_val = math.inf
    best_step = 0
    start_step = 0
    elapsed_before = 0.0
    if args.resume:
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        # CUDA generators serialize their state as a CPU ByteTensor.  A
        # map_location to CUDA moves it, but set_state still requires CPU.
        rng.set_state(saved["rng"].cpu())
        history, best_val, best_step = (saved["history"], saved["best_val"],
                                        saved["best_step"])
        start_step, elapsed_before = saved["step"], saved["elapsed_seconds"]
    start_time = time.monotonic() - elapsed_before
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(args.tasks_per_step):
            episode = sample_episode(train, support=args.support,
                                     query=args.query, max_train_length=5,
                                     rng=rng)
            loss, _ = adapt_query(model, *episode, steps=args.inner_steps,
                                  lr=args.inner_lr, meta_gradient=True)
            losses.append(loss)
        outer_loss = torch.stack(losses).mean()
        outer_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, validation, seed=args.seed + 9000,
                           tasks=args.val_tasks, support=args.support,
                           query=args.query, lr=args.inner_lr,
                           horizons=(args.inner_steps,), lengths=(1, 3, 5))
            score = float(np.mean([val[str(length)][str(args.inner_steps)][
                "normalized_mse"] for length in (1, 3, 5)]))
            row = {"step": step, "train_normalized_mse": outer_loss.item(),
                   "validation_score": score, "validation": val,
                   "elapsed_seconds": time.monotonic() - start_time}
            history.append(row)
            if score < best_val:
                best_val, best_step = score, step
                torch.save(model.state_dict(), best_path)
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "rng": rng.get_state(),
                        "history": history, "best_val": best_val,
                        "best_step": best_step, "step": step,
                        "elapsed_seconds": row["elapsed_seconds"]}, latest_path)
            result = {
                "config": {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()},
                "model": {"last_layer": "100 -> 30", "rank": args.rank,
                          "u_shape": [WEIGHTS, args.rank],
                          "u_trainable": model.learn_u,
                          "adapted_parameters": args.rank,
                          "parameters": sum(p.numel() for p in model.parameters())},
                "best_step": best_step, "best_validation_score": best_val,
                "history": history}
            temporary = result_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(result_path)
            print(f"{name} step={step} train={outer_loss.item():.4f} "
                  f"val={score:.4f} best={best_step} "
                  f"elapsed={row['elapsed_seconds']:.0f}s", flush=True)
    model.load_state_dict(torch.load(best_path, map_location=device,
                                     weights_only=True))
    test_metrics = evaluate(model, test, seed=args.seed + 19000,
                            tasks=args.test_tasks, support=args.support,
                            query=args.query, lr=args.inner_lr,
                            horizons=(0, args.inner_steps, 20),
                            lengths=(1, 3, 5, 10, 20))
    result["test"] = test_metrics
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"DONE {name} best_step={best_step} "
          f"test_length5_mae={test_metrics['5'][str(args.inner_steps)]['mae']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
