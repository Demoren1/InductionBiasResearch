"""Image-conditioned 32-value gates for a shared first-layer U on MNIST8m.

For every image x, an encoder emits g(x) = 1 + tanh(E(x)) in R^32. Its
first-layer matrix is reshape(U @ (v_task * g(x))). U and the encoder are shared
across tasks; only v_task and a 31-value readout adapt on support labels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .meta_u import load_pool, normalized_mse
from .meta_u_first import (HIDDEN, INPUT_PIXELS, RANK, FirstLayerU,
                           shifted_episode)


ARMS = ("generated", "generated_shuffled", "generated_ortho",
        "generated_shuffled_ortho", "random", "learned")
ENCODERS = ("mlp32", "mlp64", "mlp128", "mlp256", "mlp512",
            "mlp256x2", "conv32", "conv64")


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


class ConvImageEncoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels // 2, 3, padding=1), nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(channels // 2, channels, 3, padding=1), nn.Tanh(),
            nn.AvgPool2d(2), nn.Flatten())
        self.head = nn.Linear(channels * 7 * 7, RANK)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images.reshape(-1, 1, 28, 28)))


def make_image_encoder(name: str) -> tuple[nn.Module, nn.Linear]:
    if name.startswith("mlp") and name != "mlp256x2":
        width = int(name[3:])
        encoder = nn.Sequential(nn.Linear(INPUT_PIXELS, width), nn.Tanh(),
                                nn.Linear(width, RANK))
        return encoder, encoder[-1]
    if name == "mlp256x2":
        encoder = nn.Sequential(
            nn.Linear(INPUT_PIXELS, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, RANK))
        return encoder, encoder[-1]
    if name.startswith("conv"):
        encoder = ConvImageEncoder(int(name[4:]))
        return encoder, encoder.head
    raise ValueError(name)


class ImageCodeU(nn.Module):
    def __init__(self, arm: str, seed: int, encoder: str = "mlp64") -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        if encoder not in ENCODERS:
            raise ValueError(encoder)
        self.base = FirstLayerU(arm, seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 3091)
            self.image_encoder, final_layer = make_image_encoder(encoder)
            nn.init.normal_(final_layer.weight, std=0.01)
            nn.init.zeros_(final_layer.bias)

    def prepare(self, images: torch.Tensor, u: torch.Tensor,
                *, image_code: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        shape = images.shape[:-1]
        expanded = F.pad(images, (0, 1), value=1.0) @ u.reshape(
            INPUT_PIXELS + 1, HIDDEN * RANK)
        projections = expanded.reshape(*shape, HIDDEN, RANK)
        if image_code:
            code = self.image_encoder(images.reshape(-1, INPUT_PIXELS))
            gate = 1.0 + torch.tanh(code).reshape(*shape, RANK)
        else:
            gate = images.new_ones((*shape, RANK))
        return projections, gate

    def predict(self, prepared: tuple, mask: torch.Tensor,
                v1: torch.Tensor, readout: torch.Tensor) -> torch.Tensor:
        projections, gate = prepared
        first = torch.tanh((projections * (gate * v1)[..., None, :]).sum(-1))
        second = torch.tanh(self.base.second(first))
        third = torch.tanh(self.base.third(second))
        per_image = F.pad(third, (0, 1), value=1.0) @ readout
        return (per_image * mask).sum(dim=1)


def adapt_query(model: ImageCodeU, support: tuple, query: tuple,
                *, u: torch.Tensor, steps: int, lr_v1: float,
                lr_readout: float, meta_gradient: bool) -> tuple:
    sp = model.prepare(support[0], u)
    qp = model.prepare(query[0], u)
    v1, readout = model.base.initial_v1, model.base.initial_readout
    for _ in range(steps):
        prediction = model.predict(sp, support[1], v1, readout)
        loss = normalized_mse(prediction, support[2], support[3])
        gv, gr = torch.autograd.grad(loss, (v1, readout),
                                    create_graph=meta_gradient)
        v1, readout = v1 - lr_v1 * gv, readout - lr_readout * gr
    prediction = model.predict(qp, query[1], v1, readout)
    return (normalized_mse(prediction, query[2], query[3]),
            (prediction - query[2]).abs().mean())


def evaluate(model: ImageCodeU, pool: tuple, *, seed: int, tasks: int,
             support: int, query: int, steps: int, lr_v1: float,
             lr_readout: float, lengths: tuple[int, ...],
             image_code: bool = True) -> dict:
    model.eval()
    rng = torch.Generator(device=pool[0].device).manual_seed(seed)
    scores = [torch.randn(10, generator=rng, device=pool[0].device)
              for _ in range(tasks)]
    with torch.no_grad():
        u = model.base.u()
    result = {}
    for length in lengths:
        rows = []
        for score in scores:
            s, q, shift = shifted_episode(pool, support=support, query=query,
                                          rng=rng, query_length=length,
                                          digit_scores=score)
            with torch.no_grad():
                sp = model.prepare(s[0], u, image_code=image_code)
                qp = model.prepare(q[0], u, image_code=image_code)
            v1 = model.base.initial_v1.detach().clone().requires_grad_(True)
            readout = model.base.initial_readout.detach().clone().requires_grad_(True)
            for _ in range(steps):
                prediction = model.predict(sp, s[1], v1, readout)
                loss = normalized_mse(prediction, s[2], s[3])
                gv, gr = torch.autograd.grad(loss, (v1, readout))
                v1 = (v1 - lr_v1 * gv).detach().requires_grad_(True)
                readout = (readout - lr_readout * gr).detach().requires_grad_(True)
            prediction = model.predict(qp, q[1], v1, readout)
            rows.append({"normalized_mse": normalized_mse(
                prediction, q[2], q[3]).detach().item(),
                "mae": (prediction - q[2]).abs().mean().detach().item(),
                "shift": shift})
        result[str(length)] = {
            "normalized_mse": float(np.mean([r["normalized_mse"] for r in rows])),
            "mae": float(np.mean([r["mae"] for r in rows])),
            "mae_per_task": [r["mae"] for r in rows]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--encoder", choices=ENCODERS, default="mlp64")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mnist8m"))
    parser.add_argument("--out", type=Path, default=Path(
        "deepsets_z/mnist8m/outputs/meta_u_first_image_code"))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--lr-v1", type=float, default=0.1)
    parser.add_argument("--lr-readout", type=float, default=0.03)
    parser.add_argument("--outer-lr", type=float, default=0.0002)
    parser.add_argument("--tasks-per-step", type=int, default=2)
    parser.add_argument("--support", type=int, default=32)
    parser.add_argument("--query", type=int, default=32)
    parser.add_argument("--train-images-per-digit", type=int, default=3000)
    parser.add_argument("--eval-images-per-digit", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--val-tasks", type=int, default=16)
    parser.add_argument("--test-tasks", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.inner_steps, args.tasks_per_step, args.support,
           args.query, args.train_images_per_digit,
           args.eval_images_per_digit, args.eval_every,
           args.val_tasks, args.test_tasks) < 1:
        parser.error("All counts must be positive")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    train = load_pool(args.data_dir, seed=args.seed, split="train",
                      per_digit=args.train_images_per_digit, device=device)
    validation = load_pool(args.data_dir, seed=args.seed, split="validation",
                           per_digit=args.eval_images_per_digit, device=device)
    test = load_pool(args.data_dir, seed=args.seed, split="test",
                     per_digit=args.eval_images_per_digit, device=device)
    model = ImageCodeU(args.arm, args.seed, args.encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_lr)
    rng = torch.Generator(device=device).manual_seed(args.seed + 1200)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = (f"{args.arm}_seed{args.seed}" if args.encoder == "mlp64"
            else f"{args.arm}_{args.encoder}_seed{args.seed}")
    result_path = args.out / f"{stem}.json"
    latest_path = args.out / f"{stem}_latest.pt"
    best_path = args.out / f"{stem}_best.pt"
    if result_path.exists() and not args.resume:
        raise FileExistsError(result_path)
    best_score, best_step, history, start_step, elapsed_before = (
        math.inf, 0, [], 0, 0.0)
    if args.resume:
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        signature = {key: value for key, value in vars(args).items()
                     if key not in ("device", "out", "resume")}
        if saved.get("config") is not None and saved["config"] != {
                key: str(value) if isinstance(value, Path) else value
                for key, value in signature.items()}:
            raise ValueError(f"Checkpoint settings differ: {latest_path}")
        if saved.get("config") is None and args.encoder != "mlp64":
            raise ValueError(f"Checkpoint lacks encoder settings: {latest_path}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["rng"].cpu())
        best_score, best_step = saved["best_score"], saved["best_step"]
        history, start_step = saved["history"], saved["step"]
        elapsed_before = saved["elapsed_seconds"]
    output = {"config": {k: str(v) if isinstance(v, Path) else v
                         for k, v in vars(args).items()},
              "model": {"u_shape": [235500, RANK],
                        "encoder": args.encoder,
                        "trainable_parameters": sum(
                            p.numel() for p in model.parameters()),
                        "image_code_parameters": sum(
                            p.numel() for p in model.image_encoder.parameters()),
                        "adapted_parameters": [RANK, 31]},
              "best_step": best_step,
              "best_validation_score": best_score,
              "history": history}
    start = time.monotonic() - elapsed_before
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        shared_u = model.base.u()
        losses = []
        for _ in range(args.tasks_per_step):
            s, q, _ = shifted_episode(train, support=args.support,
                                      query=args.query, rng=rng)
            loss, _ = adapt_query(model, s, q, u=shared_u,
                                  steps=args.inner_steps, lr_v1=args.lr_v1,
                                  lr_readout=args.lr_readout,
                                  meta_gradient=True)
            losses.append(loss)
        outer_loss = torch.stack(losses).mean()
        outer_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, validation, seed=args.seed + 9000,
                           tasks=args.val_tasks, support=args.support,
                           query=args.query, steps=args.inner_steps,
                           lr_v1=args.lr_v1, lr_readout=args.lr_readout,
                           lengths=(1, 3, 5))
            score = float(np.mean([val[str(length)]["normalized_mse"]
                                   for length in (1, 3, 5)]))
            row = {"step": step, "train_normalized_mse": outer_loss.item(),
                   "validation_score": score,
                   "elapsed_seconds": time.monotonic() - start}
            history.append(row)
            if score < best_score:
                best_score, best_step = score, step
                atomic_torch_save(model.state_dict(), best_path)
            atomic_torch_save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng": rng.get_state(), "best_score": best_score,
                        "best_step": best_step, "history": history,
                        "step": step,
                        "config": {
                            key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()
                            if key not in ("device", "out", "resume")},
                        "elapsed_seconds": row["elapsed_seconds"]}, latest_path)
            output["best_step"] = best_step
            output["best_validation_score"] = best_score
            temporary = result_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(output, indent=2) + "\n")
            temporary.replace(result_path)
            print(f"{stem} step={step} train={outer_loss.item():.4f} "
                  f"val={score:.4f} best={best_step} "
                  f"elapsed={row['elapsed_seconds']:.0f}s", flush=True)
    model.load_state_dict(torch.load(best_path, map_location=device,
                                     weights_only=True))
    output["test"] = evaluate(model, test, seed=args.seed + 19000,
                              tasks=args.test_tasks, support=args.support,
                              query=args.query, steps=args.inner_steps,
                              lr_v1=args.lr_v1,
                              lr_readout=args.lr_readout,
                              lengths=(1, 3, 5, 10, 20))
    output["test_no_image_code"] = evaluate(
        model, test, seed=args.seed + 19000,
        tasks=args.test_tasks, support=args.support, query=args.query,
        steps=args.inner_steps, lr_v1=args.lr_v1,
        lr_readout=args.lr_readout, lengths=(1, 3, 5, 10, 20),
        image_code=False)
    result_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"DONE {stem} best={best_step} "
          f"test5_mae={output['test']['5']['mae']:.4f} "
          f"no_code={output['test_no_image_code']['5']['mae']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
