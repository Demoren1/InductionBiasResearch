"""First-layer U on MNIST8m tasks with shifted query images.

The 300 first-layer outputs are indexed as three 10x10 feature maps.  A learned
or fixed random U is compared with an analytic strided 3x3 convolution U.
Coordinate-generated U and a shuffled-coordinate control are also available.
For every task, v1 for the first layer and a 31-number readout are adapted on
unshifted support sets; query images are translated by three pixels.
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

from .meta_u import load_pool, normalized_mse, sample_episode


INPUT_PIXELS = 784
HIDDEN = 300
RANK = 32
N_WEIGHTS = (INPUT_PIXELS + 1) * HIDDEN
SHIFT_CHOICES = ((0, -3), (0, 3), (-3, 0), (3, 0))
GENERATED_ARMS = ("generated", "generated_shuffled", "generated_ortho",
                  "generated_shuffled_ortho")


def generator_coordinates() -> torch.Tensor:
    """Input pixel and output-map positions for all first-layer weights."""
    input_index = torch.arange(INPUT_PIXELS + 1)
    output_index = torch.arange(HIDDEN)
    input_row = (input_index // 28).clamp(max=27).float() / 27
    input_col = (input_index % 28).float() / 27
    input_row[-1] = input_col[-1] = 0
    output_row = ((output_index % 100) // 10).float() / 9
    output_col = (output_index % 10).float() / 9
    axes = (input_row[:, None].expand(-1, HIDDEN),
            input_col[:, None].expand(-1, HIDDEN),
            output_row[None, :].expand(INPUT_PIXELS + 1, -1),
            output_col[None, :].expand(INPUT_PIXELS + 1, -1))
    features = []
    for axis in axes:
        features.append(axis)
        for frequency in (1, 3):
            features.extend((torch.sin(2 * math.pi * frequency * axis),
                             torch.cos(2 * math.pi * frequency * axis)))
    channels = F.one_hot(output_index // 100, num_classes=3).float()
    features.extend(channels[:, channel][None, :].expand(INPUT_PIXELS + 1, -1)
                    for channel in range(3))
    features.append((input_index == INPUT_PIXELS).float()[:, None].expand(
        -1, HIDDEN))
    return torch.stack(features, dim=-1).reshape(N_WEIGHTS, -1)


def convolution_u() -> torch.Tensor:
    """Exactly shared 3x3 filters at stride 3, with one bias per channel."""
    result = torch.zeros(N_WEIGHTS, RANK)
    for channel in range(3):
        for out_row in range(10):
            for out_col in range(10):
                output_index = channel * 100 + out_row * 10 + out_col
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        row, col = 3 * out_row + dr, 3 * out_col + dc
                        if 0 <= row < 28 and 0 <= col < 28:
                            input_index = row * 28 + col
                            parameter_index = channel * 9 + (dr + 1) * 3 + dc + 1
                            result[input_index * HIDDEN + output_index,
                                   parameter_index] = 1.0
                result[INPUT_PIXELS * HIDDEN + output_index, 27 + channel] = 1.0
    return result


def translate(images: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Translate a batch of 28x28 images with zero-filled boundaries."""
    if dy == 0 and dx == 0:
        return images
    shape = images.shape
    grids = images.reshape(-1, 28, 28)
    padded = F.pad(grids, (3, 3, 3, 3))
    shifted = padded[:, 3 - dy:31 - dy, 3 - dx:31 - dx]
    return shifted.reshape(shape)


class FirstLayerU(nn.Module):
    def __init__(self, arm: str, seed: int) -> None:
        super().__init__()
        if arm not in ("learned", "random", "convolution", "dense",
                       *GENERATED_ARMS):
            raise ValueError(arm)
        torch.manual_seed(seed)
        self.second = nn.Linear(300, 100)
        self.third = nn.Linear(100, 30)
        for layer in (self.second, self.third):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        if arm == "convolution":
            initial_u = convolution_u()
        elif arm in GENERATED_ARMS:
            # Preserve the same downstream layers and initial v as the direct-U
            # controls without retaining a full random basis.
            initial_u = None
            torch.randn(N_WEIGHTS, RANK)
        else:
            initial_u = torch.randn(N_WEIGHTS, RANK)
        if initial_u is not None:
            initial_u = F.normalize(initial_u, dim=0) * math.sqrt(N_WEIGHTS) * 0.08
        if arm == "learned":
            self.u_raw = nn.Parameter(initial_u)
        elif arm in ("random", "convolution"):
            self.register_buffer("u_raw", initial_u)
        elif arm in GENERATED_ARMS:
            coordinates = generator_coordinates()
            if "shuffled" in arm:
                permutation = torch.randperm(N_WEIGHTS,
                                             generator=torch.Generator().manual_seed(seed + 3011))
                coordinates = coordinates[permutation]
            self.register_buffer("coordinates", coordinates, persistent=False)
            width = 64
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 2041)
                self.generator = nn.Sequential(
                    nn.Linear(coordinates.shape[1], width), nn.Tanh(),
                    nn.Linear(width, width), nn.Tanh(),
                    nn.Linear(width, RANK))
                nn.init.normal_(self.generator[-1].weight, std=0.03)
                nn.init.zeros_(self.generator[-1].bias)
            # Remove only the initial per-output DC component.  The fixed
            # offset prevents smooth coordinate features from saturating tanh
            # at initialization without constraining subsequent learned U.
            with torch.no_grad():
                initial_dc = self.generator(coordinates).reshape(
                    INPUT_PIXELS + 1, HIDDEN, RANK)[:INPUT_PIXELS].mean(0)
            self.register_buffer("initial_dc", initial_dc)
        initial_v1 = torch.randn(RANK) * 0.1
        if arm == "dense":
            # Match the random-U arm's initial first-layer matrix exactly.
            self.dense_weight = nn.Parameter(
                (initial_u @ initial_v1).reshape(INPUT_PIXELS + 1, HIDDEN))
            self.register_buffer("initial_v1", initial_v1)
        else:
            self.initial_v1 = nn.Parameter(initial_v1)
        self.initial_readout = nn.Parameter(torch.randn(31) * 0.05)
        if arm == "convolution":
            with torch.no_grad():
                self.initial_v1[27:30].zero_()
        self.arm = arm

    def u(self) -> torch.Tensor | None:
        if self.arm == "dense":
            return None
        if self.arm in GENERATED_ARMS:
            raw = self.generator(self.coordinates).reshape(
                INPUT_PIXELS + 1, HIDDEN, RANK)
            raw = torch.cat((raw[:INPUT_PIXELS] - self.initial_dc[None],
                             raw[INPUT_PIXELS:]), dim=0).reshape(N_WEIGHTS, RANK)
            if self.arm.endswith("ortho"):
                gram = raw.T @ raw / N_WEIGHTS
                chol = torch.linalg.cholesky(gram +
                    1e-6 * torch.eye(RANK, device=raw.device, dtype=raw.dtype))
                return torch.linalg.solve_triangular(
                    chol, raw.T, upper=False).T * 0.08
            return F.normalize(raw, dim=0) * math.sqrt(N_WEIGHTS) * 0.08
        return F.normalize(self.u_raw, dim=0) * math.sqrt(N_WEIGHTS) * 0.08

    def predict(self, images: torch.Tensor, mask: torch.Tensor,
                v1: torch.Tensor, readout: torch.Tensor,
                u: torch.Tensor | None) -> torch.Tensor:
        first_weight = (self.dense_weight if self.arm == "dense" else
                        (u @ v1).reshape(INPUT_PIXELS + 1, HIDDEN))
        pixels_with_bias = F.pad(images, (0, 1), value=1.0)
        first = torch.tanh(pixels_with_bias @ first_weight)
        second = torch.tanh(self.second(first))
        third = torch.tanh(self.third(second))
        per_image = F.pad(third, (0, 1), value=1.0) @ readout
        return (per_image * mask).sum(dim=1)


def shifted_episode(pool: tuple, *, support: int, query: int,
                    rng: torch.Generator, query_length: int | None = None,
                    digit_scores: torch.Tensor | None = None) -> tuple:
    s, q = sample_episode(pool, support=support, query=query,
                          max_train_length=5, rng=rng,
                          query_length=query_length, digit_scores=digit_scores)
    shift_index = int(torch.randint(0, len(SHIFT_CHOICES), (1,),
                                    generator=rng, device=pool[0].device).item())
    dy, dx = SHIFT_CHOICES[shift_index]
    return s, (translate(q[0], dy, dx), *q[1:]), (dy, dx)


def adapt_query(model: FirstLayerU, support: tuple, query: tuple,
                *, steps: int, lr_v1: float, lr_readout: float,
                meta_gradient: bool, u: torch.Tensor | None = None) -> tuple:
    sx, smask, sy, slength = support
    qx, qmask, qy, qlength = query
    if u is None:
        u = model.u()
    v1, readout = model.initial_v1, model.initial_readout
    for _ in range(steps):
        prediction = model.predict(sx, smask, v1, readout, u)
        loss = normalized_mse(prediction, sy, slength)
        if model.arm == "dense":
            gr, = torch.autograd.grad(loss, (readout,),
                                      create_graph=meta_gradient)
            readout = readout - lr_readout * gr
        else:
            gv, gr = torch.autograd.grad(loss, (v1, readout),
                                        create_graph=meta_gradient)
            v1, readout = v1 - lr_v1 * gv, readout - lr_readout * gr
    prediction = model.predict(qx, qmask, v1, readout, u)
    return normalized_mse(prediction, qy, qlength), (prediction - qy).abs().mean()


def evaluate(model: FirstLayerU, pool: tuple, *, seed: int, tasks: int,
             support: int, query: int, steps: int, lr_v1: float,
             lr_readout: float, lengths: tuple[int, ...]) -> dict:
    model.eval()
    rng = torch.Generator(device=pool[0].device).manual_seed(seed)
    scores = [torch.randn(10, generator=rng, device=pool[0].device)
              for _ in range(tasks)]
    with torch.no_grad():
        fixed_u = model.u()
    result = {}
    for length in lengths:
        rows = []
        for score in scores:
            s, q, shift = shifted_episode(pool, support=support, query=query,
                                          rng=rng, query_length=length,
                                          digit_scores=score)
            # Only adapted vectors require gradients during evaluation.
            v1 = model.initial_v1.detach().clone().requires_grad_(
                model.arm != "dense")
            readout = model.initial_readout.detach().clone().requires_grad_(True)
            for _ in range(steps):
                prediction = model.predict(s[0], s[1], v1, readout, fixed_u)
                loss = normalized_mse(prediction, s[2], s[3])
                if model.arm == "dense":
                    gr, = torch.autograd.grad(loss, (readout,))
                else:
                    gv, gr = torch.autograd.grad(loss, (v1, readout))
                    v1 = (v1 - lr_v1 * gv).detach().requires_grad_(True)
                readout = (readout - lr_readout * gr).detach().requires_grad_(True)
            prediction = model.predict(q[0], q[1], v1, readout, fixed_u)
            rows.append({"normalized_mse": normalized_mse(
                prediction, q[2], q[3]).detach().item(),
                "mae": (prediction - q[2]).abs().mean().detach().item(),
                "shift": shift})
        result[str(length)] = {
            "normalized_mse": float(np.mean([r["normalized_mse"] for r in rows])),
            "mae": float(np.mean([r["mae"] for r in rows])),
            "mae_per_task": [r["mae"] for r in rows]}
    return result


@torch.no_grad()
def convolution_overlap(model: FirstLayerU) -> float | None:
    if model.arm == "dense":
        return None
    u = model.u()
    reference = convolution_u().to(u.device)
    reference = F.normalize(reference[:, :30], dim=0)
    projected = reference.T @ u
    return float((projected.square().sum() / u.square().sum()).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("learned", "random", "convolution",
                                          "dense", *GENERATED_ARMS),
                        required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mnist8m"))
    parser.add_argument("--out", type=Path,
                        default=Path("deepsets_z/mnist8m/outputs/meta_u_first"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--lr-v1", type=float, default=0.1)
    parser.add_argument("--lr-readout", type=float, default=0.03)
    parser.add_argument("--outer-lr", type=float, default=0.0002)
    parser.add_argument("--tasks-per-step", type=int, default=2)
    parser.add_argument("--support", type=int, default=32)
    parser.add_argument("--query", type=int, default=32)
    parser.add_argument("--train-images-per-digit", type=int, default=3000)
    parser.add_argument("--eval-images-per-digit", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--val-tasks", type=int, default=8)
    parser.add_argument("--test-tasks", type=int, default=32)
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
    model = FirstLayerU(args.arm, args.seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_lr)
    rng = torch.Generator(device=device).manual_seed(args.seed + 1200)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.arm}_seed{args.seed}"
    result_path = args.out / f"{stem}.json"
    latest_path = args.out / f"{stem}_latest.pt"
    best_path = args.out / f"{stem}_best.pt"
    if result_path.exists() and not args.resume:
        raise FileExistsError(result_path)
    best_score, best_step, history, start_step, elapsed_before = (
        math.inf, 0, [], 0, 0.0)
    if args.resume:
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["rng"].cpu())
        best_score, best_step = saved["best_score"], saved["best_step"]
        history, start_step = saved["history"], saved["step"]
        elapsed_before = saved["elapsed_seconds"]
    start = time.monotonic() - elapsed_before
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        shared_u = model.u()
        for _ in range(args.tasks_per_step):
            support, query, _ = shifted_episode(
                train, support=args.support, query=args.query, rng=rng)
            loss, _ = adapt_query(model, support, query,
                                  steps=args.inner_steps,
                                  lr_v1=args.lr_v1,
                                  lr_readout=args.lr_readout,
                                  meta_gradient=True, u=shared_u)
            losses.append(loss)
        outer_loss = torch.stack(losses).mean()
        outer_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, validation, seed=args.seed + 9000,
                           tasks=args.val_tasks, support=args.support,
                           query=args.query, steps=args.inner_steps,
                           lr_v1=args.lr_v1,
                           lr_readout=args.lr_readout,
                           lengths=(1, 3, 5))
            score = float(np.mean([val[str(length)]["normalized_mse"]
                                   for length in (1, 3, 5)]))
            row = {"step": step, "train_normalized_mse": outer_loss.item(),
                   "validation_score": score,
                   "elapsed_seconds": time.monotonic() - start}
            history.append(row)
            if score < best_score:
                best_score, best_step = score, step
                torch.save(model.state_dict(), best_path)
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng": rng.get_state(), "best_score": best_score,
                        "best_step": best_step, "history": history,
                        "step": step,
                        "elapsed_seconds": row["elapsed_seconds"]}, latest_path)
            output = {"config": {k: str(v) if isinstance(v, Path) else v
                                 for k, v in vars(args).items()},
                      "model": {"u_shape": ([N_WEIGHTS, RANK]
                                            if args.arm != "dense" else None),
                                "u_trainable": args.arm in (
                                    "learned", *GENERATED_ARMS),
                                "generator_parameters": (
                                    sum(p.numel() for p in model.generator.parameters())
                                    if args.arm in GENERATED_ARMS
                                    else None),
                                "adapted_parameters": ([RANK, 31]
                                                       if args.arm != "dense"
                                                       else [31])},
                      "best_step": best_step,
                      "best_validation_score": best_score,
                      "history": history}
            temporary = result_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(output, indent=2) + "\n")
            temporary.replace(result_path)
            print(f"{stem} step={step} train={outer_loss.item():.4f} "
                  f"val={score:.4f} best={best_step} "
                  f"elapsed={row['elapsed_seconds']:.0f}s", flush=True)
    model.load_state_dict(torch.load(best_path, map_location=device,
                                     weights_only=True))
    output["convolution_overlap"] = convolution_overlap(model)
    output["test"] = evaluate(model, test, seed=args.seed + 19000,
                              tasks=args.test_tasks, support=args.support,
                              query=args.query, steps=args.inner_steps,
                              lr_v1=args.lr_v1,
                              lr_readout=args.lr_readout,
                              lengths=(1, 3, 5, 10, 20))
    result_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"DONE {stem} best={best_step} "
          f"test5_mae={output['test']['5']['mae']:.4f} "
          f"conv_overlap={output['convolution_overlap']}", flush=True)


if __name__ == "__main__":
    main()
