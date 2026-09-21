"""Compare generated, directly learned, and fixed U on Aug-Omniglot.

The shared U acts on the output, input, and spatial axes of each convolution.
For every episode, V and the classifier start from a learned initialization and
are adapted on that episode's support images. Only the outer loop changes U.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.datasets import Omniglot


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "omniglot"
OUTPUT_ROOT = ROOT / "aug_omniglot" / "outputs"
CONV_SHAPES = [(32, 1), (32, 32), (32, 32), (32, 32)]


def load_data(device: torch.device) -> tuple[dict[str, Tensor], dict[str, list[int]], dict]:
    """Load official background/evaluation split; hold out five train alphabets."""
    cache = DATA_ROOT / "prepared_28.pt"
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
    else:
        xs, names, sections = [], [], []
        for section, background in (("background", True), ("evaluation", False)):
            ds = Omniglot(str(DATA_ROOT), background=background, download=True)
            for image_idx in range(len(ds)):
                image, cls = ds[image_idx]
                image = image.resize((28, 28), Image.Resampling.BILINEAR)
                xs.append(np.asarray(image, dtype=np.uint8).copy())
                names.append(ds._characters[cls])
                sections.append(section)
        saved = {"images": torch.from_numpy(np.stack(xs)), "names": names, "sections": sections}
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, cache)
    background_alphabets = sorted({name.split("/")[0] for name, sec in zip(saved["names"], saved["sections"]) if sec == "background"})
    rng = np.random.default_rng(2020)
    val_alphabets = sorted(rng.choice(background_alphabets, size=5, replace=False).tolist())
    groups = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    for idx, (name, section) in enumerate(zip(saved["names"], saved["sections"])):
        split = "test" if section == "evaluation" else "val" if name.split("/")[0] in val_alphabets else "train"
        groups[split][name].append(idx)
    class_indices = {split: [indices for _, indices in sorted(chars.items())] for split, chars in groups.items()}
    images = 1.0 - saved["images"].to(device=device, dtype=torch.float32).unsqueeze(1) / 255.0
    manifest = {"train_classes": len(class_indices["train"]), "val_classes": len(class_indices["val"]), "test_classes": len(class_indices["test"]), "val_alphabets": val_alphabets, "images_per_class": 20}
    return {"images": images}, class_indices, manifest


def episode(images: Tensor, classes: list[list[int]], n_way: int, shots: int, queries: int, seed: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(classes), size=n_way, replace=False)
    support, query = [], []
    for c in chosen:
        picked = rng.choice(classes[c], size=shots + queries, replace=False)
        support.extend(picked[:shots].tolist())
        query.extend(picked[shots:].tolist())
    s_idx = torch.tensor(support, device=images.device)
    q_idx = torch.tensor(query, device=images.device)
    s_y = torch.arange(n_way, device=images.device).repeat_interleave(shots)
    q_y = torch.arange(n_way, device=images.device).repeat_interleave(queries)
    return images[s_idx], s_y, images[q_idx], q_y


def augment_queries(images: Tensor, seed: int) -> Tensor:
    """Batch GPU version of random resized crop, two flips, and ±30° rotation."""
    generator = torch.Generator(device=images.device).manual_seed(seed)
    n = images.shape[0]
    rand = lambda *size: torch.rand(size, generator=generator, device=images.device)
    area = .8 + .2 * rand(n)
    aspect = torch.exp(math.log(.75) + (math.log(4 / 3) - math.log(.75)) * rand(n))
    width = (area * aspect).sqrt().clamp(max=1.0)
    height = (area / aspect).sqrt().clamp(max=1.0)
    left = (1 - width) * (2 * rand(n) - 1)
    top = (1 - height) * (2 * rand(n) - 1)
    fx = torch.where(rand(n) < .5, -1.0, 1.0)
    fy = torch.where(rand(n) < .5, -1.0, 1.0)
    angle = (rand(n) - .5) * (math.pi / 3)
    cos, sin = angle.cos(), angle.sin()
    theta = torch.zeros((n, 2, 3), device=images.device)
    theta[:, 0, 0] = width * fx * cos
    theta[:, 0, 1] = width * fx * sin
    theta[:, 1, 0] = -height * fy * sin
    theta[:, 1, 1] = height * fy * cos
    theta[:, 0, 2] = left
    theta[:, 1, 2] = top
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class SharedU(nn.Module):
    def __init__(self, kind: str, generator_width: int = 64, generator_scale: float = .1):
        super().__init__()
        self.kind = kind
        self.generator_scale = generator_scale
        self.keys = []
        self.direct = nn.ParameterDict()
        self.fixed = {}
        if kind == "generated":
            self.generator = nn.Sequential(nn.Linear(11, generator_width), nn.Tanh(), nn.Linear(generator_width, generator_width), nn.Tanh(), nn.Linear(generator_width, 1))
            nn.init.zeros_(self.generator[-1].weight)
            nn.init.zeros_(self.generator[-1].bias)
        for layer, (out_ch, in_ch) in enumerate(CONV_SHAPES):
            for axis, size in enumerate((out_ch, in_ch, 9)):
                key = f"l{layer}_a{axis}"
                self.keys.append(key)
                identity = torch.eye(size)
                if kind == "direct":
                    self.direct[key] = nn.Parameter(identity)
                elif kind == "random":
                    # Deterministic orthogonal coordinates; held fixed by design.
                    rng = torch.Generator().manual_seed(7823 + layer * 3 + axis)
                    matrix, _ = torch.linalg.qr(torch.randn((size, size), generator=rng))
                    self.register_buffer(key, matrix)
                elif kind == "identity":
                    self.register_buffer(key, identity)
                elif kind == "generated":
                    row, col = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
                    row, col = row.float(), col.float()
                    denom = max(size - 1, 1)
                    features = torch.zeros((size, size, 11))
                    features[:, :, axis] = 1.0
                    features[:, :, 3 + layer] = 1.0
                    features[:, :, 7] = 2 * row / denom - 1
                    features[:, :, 8] = 2 * col / denom - 1
                    features[:, :, 9] = (row - col) / denom
                    features[:, :, 10] = (row == col).float()
                    self.register_buffer(key + "_coords", features)
                else:
                    raise ValueError(kind)

    def matrices(self) -> dict[str, Tensor]:
        result = {}
        for key in self.keys:
            if self.kind == "direct":
                result[key] = self.direct[key]
            elif self.kind in ("random", "identity"):
                result[key] = getattr(self, key)
            else:
                coords = getattr(self, key + "_coords")
                result[key] = torch.eye(coords.shape[0], device=coords.device) + self.generator_scale * self.generator(coords).squeeze(-1)
        return result


class MetaConv(nn.Module):
    def __init__(self, kind: str, n_way: int, generator_width: int = 64,
                 generator_scale: float = .1, inner_lr_mode: str = "clamp",
                 inner_init: float = .4, separate_u_seed: int | None = None):
        super().__init__()
        if inner_lr_mode not in ("clamp", "softplus"):
            raise ValueError(inner_lr_mode)
        if inner_init <= 0:
            raise ValueError("inner_init must be positive")
        self.inner_lr_mode = inner_lr_mode
        if separate_u_seed is None:
            self.u = SharedU(kind, generator_width, generator_scale)
        else:
            # U initialization must not change the RNG state used for V. This
            # gives matched V initialization across architecture variants.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(separate_u_seed)
                self.u = SharedU(kind, generator_width, generator_scale)
        self.v = nn.ParameterDict()
        for layer, (out_ch, in_ch) in enumerate(CONV_SHAPES):
            self.v[f"conv{layer}"] = nn.Parameter(torch.empty(out_ch, in_ch, 3, 3))
            nn.init.kaiming_normal_(self.v[f"conv{layer}"], nonlinearity="relu")
            self.v[f"bn_w{layer}"] = nn.Parameter(torch.ones(out_ch))
            self.v[f"bn_b{layer}"] = nn.Parameter(torch.zeros(out_ch))
        self.v["head_w"] = nn.Parameter(torch.empty(n_way, 32))
        nn.init.kaiming_normal_(self.v["head_w"], nonlinearity="linear")
        self.v["head_b"] = nn.Parameter(torch.zeros(n_way))
        raw_init = inner_init if inner_lr_mode == "clamp" else math.log(math.expm1(inner_init))
        self.inner_lrs = nn.ParameterDict({
            **{f"layer{layer}": nn.Parameter(torch.tensor(raw_init)) for layer in range(4)},
            "head": nn.Parameter(torch.tensor(raw_init)),
        })

    def effective_inner_lrs(self) -> dict[str, Tensor]:
        transform = (lambda value: value.clamp(.001, 1.0)) if self.inner_lr_mode == "clamp" else F.softplus
        return {key: transform(value) for key, value in self.inner_lrs.items()}

    def forward(self, x: Tensor, v: dict[str, Tensor], u: dict[str, Tensor]) -> Tensor:
        for layer in range(4):
            out_ch, in_ch = CONV_SHAPES[layer]
            weight = v[f"conv{layer}"].reshape(out_ch, in_ch, 9)
            weight = torch.einsum("oa,abk->obk", u[f"l{layer}_a0"], weight)
            weight = torch.einsum("ib,obk->oik", u[f"l{layer}_a1"], weight)
            weight = torch.einsum("sk,oik->ois", u[f"l{layer}_a2"], weight).reshape(out_ch, in_ch, 3, 3)
            x = F.conv2d(x, weight, padding=1)
            x = F.batch_norm(x, None, None, v[f"bn_w{layer}"], v[f"bn_b{layer}"], training=True)
            x = F.max_pool2d(F.relu(x), 2)
        return F.linear(x.flatten(1), v["head_w"], v["head_b"])

    def adapt(self, sx: Tensor, sy: Tensor, u: dict[str, Tensor], steps: int, second_order: bool) -> dict[str, Tensor]:
        params = dict(self.v.items())
        for _ in range(steps):
            loss = F.cross_entropy(self.forward(sx, params, u), sy)
            grads = torch.autograd.grad(loss, tuple(params.values()), create_graph=second_order)
            rates = self.effective_inner_lrs()
            params = {
                key: value - rates["head" if key.startswith("head") else f"layer{key[-1]}"] * grad
                for (key, value), grad in zip(params.items(), grads)
            }
        return params


def evaluate(model: MetaConv, data: Tensor, classes: list[list[int]], split: str, args: argparse.Namespace, tasks: int) -> dict:
    model.eval()
    u = {k: v.detach() for k, v in model.u.matrices().items()}
    losses, accuracies = [], []
    for idx in range(tasks):
        seed = args.seed * 100_000_000 + (2 if split == "test" else 1) * 1_000_000 + idx
        sx, sy, qx, qy = episode(data, classes, args.ways, args.shots, args.queries, seed)
        qx = augment_queries(qx, seed + 9187)
        adapted = model.adapt(sx, sy, u, args.eval_inner_steps, second_order=False)
        with torch.no_grad():
            logits = model(qx, adapted, u)
            losses.append(F.cross_entropy(logits, qy).item())
            accuracies.append((logits.argmax(-1) == qy).float().mean().item())
    model.train()
    return {"loss": float(np.mean(losses)), "accuracy": float(np.mean(accuracies)), "accuracy_se": float(np.std(accuracies, ddof=1) / math.sqrt(tasks)), "tasks": tasks}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["generated", "direct", "random", "identity"], required=True)
    p.add_argument("--generator-width", type=int, default=64)
    p.add_argument("--generator-scale", type=float, default=.1)
    p.add_argument("--inner-lr-mode", choices=["clamp", "softplus"], default="clamp")
    p.add_argument("--inner-init", type=float, default=.4)
    p.add_argument("--match-v-init", action="store_true")
    p.add_argument("--select-metric", choices=["loss", "accuracy"], default="loss")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--reset-patience-on-resume", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ways", type=int, default=5)
    p.add_argument("--shots", type=int, default=1)
    p.add_argument("--queries", type=int, default=5)
    p.add_argument("--meta-batch", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--min-steps", type=int, default=1600)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--val-tasks", type=int, default=40)
    p.add_argument("--test-tasks", type=int, default=400)
    p.add_argument("--eval-inner-steps", type=int, default=3)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    data, splits, manifest = load_data(device)
    model = MetaConv(args.kind, args.ways, args.generator_width,
                     args.generator_scale, args.inner_lr_mode, args.inner_init,
                     args.seed + 2041 if args.match_v_init else None).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    suffix = f"_w{args.generator_width}" if args.kind == "generated" and args.generator_width != 64 else ""
    output = args.output or OUTPUT_ROOT / f"{args.kind}{suffix}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output"] = str(output)
    (output / "config.json").write_text(json.dumps({"config": config, "data": manifest, "parameters": sum(p.numel() for p in model.parameters())}, indent=2, ensure_ascii=False))
    metrics_file = output / "history.jsonl"
    start_step = 0
    best_score = float("inf") if args.select_metric == "loss" else -float("inf")
    bad_rounds = 0
    checkpoint = output / "latest.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = saved["step"]
        best_score = saved.get("best_score", saved.get("best_loss"))
        bad_rounds = 0 if args.reset_patience_on_resume else saved["bad_rounds"]
        print(f"RESUME {args.kind} step={start_step}", flush=True)
    start_time = time.time()
    for step in range(start_step + 1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        batch_loss, batch_accuracy = 0.0, 0.0
        for task_idx in range(args.meta_batch):
            # Each task has its own autograd graph; generator outputs cannot be
            # reused after a previous task's backward pass frees that graph.
            shared_u = model.u.matrices()
            task_seed = args.seed * 100_000_000 + step * 100 + task_idx
            sx, sy, qx, qy = episode(data["images"], splits["train"], args.ways, args.shots, args.queries, task_seed)
            qx = augment_queries(qx, task_seed + 9187)
            adapted = model.adapt(sx, sy, shared_u, 1, second_order=True)
            logits = model(qx, adapted, shared_u)
            task_loss = F.cross_entropy(logits, qy)
            (task_loss / args.meta_batch).backward()
            batch_loss += task_loss.item() / args.meta_batch
            batch_accuracy += (logits.argmax(-1) == qy).float().mean().item() / args.meta_batch
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % 20 == 0 or step == 1:
            print(f"STEP {step} {args.kind} train_loss={batch_loss:.4f} train_acc={batch_accuracy:.3f} elapsed={time.time()-start_time:.0f}s", flush=True)
        if step % args.eval_every == 0:
            val = evaluate(model, data["images"], splits["val"], "val", args, args.val_tasks)
            record = {"step": step, "train_loss": batch_loss, "train_accuracy": batch_accuracy, "val": val, "elapsed_s": time.time() - start_time}
            with metrics_file.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            score = val[args.select_metric]
            improved = (score < best_score - .002) if args.select_metric == "loss" else (score > best_score + .002)
            if improved:
                best_score, bad_rounds = score, 0
                torch.save({"model": model.state_dict(), "step": step, "val": val}, output / "best.pt")
            else:
                bad_rounds += 1
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "best_score": best_score, "bad_rounds": bad_rounds}, checkpoint)
            print(f"VAL {step} {args.kind} loss={val['loss']:.4f} acc={val['accuracy']:.4f} best_{args.select_metric}={best_score:.4f} bad={bad_rounds}", flush=True)
            if step >= args.min_steps and bad_rounds >= args.patience:
                print(f"EARLY_STOP {step} {args.kind}", flush=True)
                break
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test = None if args.skip_test else evaluate(model, data["images"], splits["test"], "test", args, args.test_tasks)
    result = {"kind": args.kind, "seed": args.seed, "best_step": best["step"], "validation": best["val"], "test": test, "elapsed_s": time.time() - start_time, "parameters_total": sum(p.numel() for p in model.parameters()), "parameters_u": sum(p.numel() for p in model.u.parameters()), "inner_lrs": {key: value.item() for key, value in model.effective_inner_lrs().items()}}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(f"RESULT {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
