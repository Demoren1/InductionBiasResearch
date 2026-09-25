"""Few-shot Omniglot with one support-conditioned U per episode.

The generator sees only the five unmodified support images. The same U is used
for support adaptation and query prediction; query images are never generator
inputs. ``static`` is the directly learned U control from the same Conv4 model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .run import CONV_SHAPES, MetaConv, augment_queries, episode, load_data


DEFAULT_ROOT = Path(__file__).resolve().parent / "outputs" / "conditional_u"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


class SupportConditionedU(nn.Module):
    """A learned common U plus a small, support-dependent low-rank correction."""

    def __init__(self, aggregation: str, bases: int = 4, rank: int = 4,
                 scale: float = 0.1):
        super().__init__()
        if aggregation not in ("mean", "transformer"):
            raise ValueError(aggregation)
        if bases < 1 or rank < 1 or scale <= 0:
            raise ValueError("bases, rank and scale must be positive")
        self.aggregation = aggregation
        self.bases = bases
        self.scale = scale
        self.keys: list[str] = []
        self.base = nn.ParameterDict()
        self.left = nn.ParameterDict()
        self.right = nn.ParameterDict()
        for layer, (out_ch, in_ch) in enumerate(CONV_SHAPES):
            for axis, size in enumerate((out_ch, in_ch, 9)):
                key = f"l{layer}_a{axis}"
                self.keys.append(key)
                effective_rank = min(rank, size)
                self.base[key] = nn.Parameter(torch.eye(size))
                self.left[key] = nn.Parameter(
                    torch.randn(bases, size, effective_rank) / math.sqrt(effective_rank))
                self.right[key] = nn.Parameter(
                    torch.randn(bases, size, effective_rank) / math.sqrt(effective_rank))

        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        if aggregation == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=128, dropout=0.0,
                batch_first=True, norm_first=True)
            self.set_encoder = nn.TransformerEncoder(layer, num_layers=1,
                                                      enable_nested_tensor=False)
        else:
            self.set_encoder = nn.Identity()
        self.coefficients = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, len(self.keys) * bases))
        nn.init.zeros_(self.coefficients[-1].weight)
        nn.init.zeros_(self.coefficients[-1].bias)

    def matrices(self, support_images: Tensor) -> dict[str, Tensor]:
        codes = self.image_encoder(support_images).unsqueeze(0)
        codes = self.set_encoder(codes).mean(dim=1).squeeze(0)
        coefficients = self.coefficients(codes).view(len(self.keys), self.bases).tanh()
        result = {}
        for idx, key in enumerate(self.keys):
            left, right = self.left[key], self.right[key]
            basis = left @ right.transpose(-1, -2)
            result[key] = self.base[key] + self.scale * torch.einsum(
                "k,kij->ij", coefficients[idx], basis)
        return result


class ConditionalMetaConv(MetaConv):
    def __init__(self, arm: str, n_way: int, bases: int, rank: int, scale: float,
                 inner_init: float):
        if arm not in ("static", "mean", "transformer"):
            raise ValueError(arm)
        # Construct V before the image encoder, so all arms have exactly the
        # same initial V for a given seed.
        super().__init__("direct" if arm == "static" else "identity", n_way,
                         inner_lr_mode="softplus", inner_init=inner_init)
        self.arm = arm
        if arm != "static":
            self.u = SupportConditionedU(arm, bases, rank, scale)

    def matrices_for(self, support_images: Tensor) -> dict[str, Tensor]:
        if self.arm == "static":
            return self.u.matrices()
        return self.u.matrices(support_images)


def evaluate(model: ConditionalMetaConv, images: Tensor, classes: list[list[int]],
             args: argparse.Namespace, split: str, count: int,
             wrong_context: bool = False,
             report_progress: Callable[[int], None] | None = None) -> dict:
    model.eval()
    losses, accuracies = [], []
    for idx in range(count):
        seed = args.seed * 100_000_000 + (2 if split == "test" else 1) * 1_000_000 + idx
        sx, sy, qx, qy = episode(images, classes, args.ways, args.shots,
                                 args.queries, seed)
        qx = augment_queries(qx, seed + 9187)
        context = sx
        if wrong_context:
            chosen = set(np.random.default_rng(seed).choice(
                len(classes), size=args.ways, replace=False).tolist())
            other_classes = [item for index, item in enumerate(classes)
                             if index not in chosen]
            context = episode(images, other_classes, args.ways, args.shots,
                              args.queries, seed + 700_000)[0]
        with torch.no_grad():
            u = {key: value.detach() for key, value in
                 model.matrices_for(context).items()}
        adapted = model.adapt(sx, sy, u, args.eval_inner_steps,
                              second_order=False)
        with torch.no_grad():
            logits = model(qx, adapted, u)
            losses.append(F.cross_entropy(logits, qy).item())
            accuracies.append((logits.argmax(-1) == qy).float().mean().item())
        if report_progress is not None and ((idx + 1) % 10 == 0 or idx + 1 == count):
            report_progress(idx + 1)
    model.train()
    return {"loss": float(np.mean(losses)),
            "accuracy": float(np.mean(accuracies)),
            "accuracy_se": float(np.std(accuracies, ddof=1) / math.sqrt(count)),
            "tasks": count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["static", "mean", "transformer"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ways", type=int, default=5)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--bases", type=int, default=4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--meta-batch", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--min-steps", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-tasks", type=int, default=100)
    parser.add_argument("--test-tasks", type=int, default=1000)
    parser.add_argument("--eval-inner-steps", type=int, default=3)
    parser.add_argument("--inner-init", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_steps < args.eval_every or args.min_steps > args.max_steps:
        parser.error("Require max-steps >= eval-every and min-steps <= max-steps")
    output = args.output or DEFAULT_ROOT / f"{args.arm}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output"] = str(output)
    previous_config = output / "config.json"
    if previous_config.exists():
        previous = json.loads(previous_config.read_text())["config"]
        comparable = lambda item: {key: value for key, value in item.items()
                                   if key not in ("device", "max_steps")}
        if comparable(previous) != comparable(config) or args.max_steps < previous["max_steps"]:
            parser.error(f"Existing run uses different configuration: {output}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    data, splits, manifest = load_data(device)
    images = data["images"]
    model = ConditionalMetaConv(args.arm, args.ways, args.bases, args.rank,
                                args.scale, args.inner_init).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    atomic_json(previous_config, {"config": config, "data": manifest,
                                  "parameters": sum(p.numel() for p in model.parameters()),
                                  "parameters_u": sum(p.numel() for p in model.u.parameters())})
    latest = output / "latest.pt"
    best_path = output / "best.pt"
    progress_path = output / "progress.json"
    start_step, best_score, bad_rounds = 0, float("inf"), 0
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = saved["step"]
        best_score = saved["best_score"]
        bad_rounds = saved["bad_rounds"]
        torch.set_rng_state(saved["cpu_rng"].cpu())
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device=device)
        print(f"RESUME {args.arm} seed={args.seed} step={start_step}", flush=True)
    # A capped run can be extended with a larger --max-steps. Its old result
    # must not make the launcher mistake an interrupted extension for success.
    (output / "result.json").unlink(missing_ok=True)
    start_time = time.time()
    atomic_json(progress_path, {"stage": "train", "step": start_step,
                                "max_steps": args.max_steps})
    stop_reason = "max_steps"
    step = start_step
    for step in range(start_step + 1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss, train_accuracy = 0.0, 0.0
        for task_idx in range(args.meta_batch):
            task_seed = args.seed * 100_000_000 + step * 100 + task_idx
            sx, sy, qx, qy = episode(images, splits["train"], args.ways,
                                     args.shots, args.queries, task_seed)
            qx = augment_queries(qx, task_seed + 9187)
            u = model.matrices_for(sx)
            adapted = model.adapt(sx, sy, u, 1, second_order=True)
            logits = model(qx, adapted, u)
            loss = F.cross_entropy(logits, qy)
            (loss / args.meta_batch).backward()
            train_loss += loss.item() / args.meta_batch
            train_accuracy += (logits.argmax(-1) == qy).float().mean().item() / args.meta_batch
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % 20 == 0 or step == 1:
            atomic_json(progress_path, {"stage": "train", "step": step,
                                        "max_steps": args.max_steps,
                                        "train_loss": train_loss,
                                        "train_accuracy": train_accuracy})
            print(f"STEP {step} {args.arm} seed={args.seed} "
                  f"loss={train_loss:.4f} acc={train_accuracy:.3f} "
                  f"elapsed={time.time()-start_time:.0f}s", flush=True)
        if step % args.eval_every != 0:
            continue
        atomic_json(progress_path, {"stage": "validation", "step": step,
                                    "max_steps": args.max_steps})
        val = evaluate(model, images, splits["val"], args, "val", args.val_tasks)
        record = {"step": step, "train_loss": train_loss,
                  "train_accuracy": train_accuracy, "val": val,
                  "elapsed_s": time.time() - start_time}
        with (output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if val["loss"] < best_score - 0.002:
            best_score, bad_rounds = val["loss"], 0
            torch.save({"model": model.state_dict(), "step": step, "val": val}, best_path)
        else:
            bad_rounds += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "best_score": best_score,
                    "bad_rounds": bad_rounds, "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None},
                   latest)
        print(f"VAL {step} {args.arm} seed={args.seed} "
              f"acc={val['accuracy']:.4f} loss={val['loss']:.4f} "
              f"best_loss={best_score:.4f} bad={bad_rounds}/{args.patience}", flush=True)
        if step >= args.min_steps and bad_rounds >= args.patience:
            print(f"EARLY_STOP {step} {args.arm} seed={args.seed}", flush=True)
            stop_reason = "validation_plateau"
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_total = args.test_tasks * (2 if args.arm != "static" else 1)
    atomic_json(progress_path, {"stage": "test", "step": step,
                                "max_steps": args.max_steps,
                                "test_done": 0, "test_total": test_total})
    def report_test(done: int) -> None:
        atomic_json(progress_path, {"stage": "test", "step": step,
                                    "max_steps": args.max_steps,
                                    "test_done": done, "test_total": test_total})

    test = evaluate(model, images, splits["test"], args, "test", args.test_tasks,
                    report_progress=report_test)
    wrong_context = (evaluate(model, images, splits["test"], args, "test",
                              args.test_tasks, wrong_context=True,
                              report_progress=lambda done: report_test(args.test_tasks + done))
                     if args.arm != "static" else None)
    result = {"arm": args.arm, "seed": args.seed,
              "best_step": best["step"], "last_step": step,
              "stop_reason": stop_reason,
              "validation": best["val"], "test": test,
              "test_wrong_context": wrong_context,
              "elapsed_s": time.time() - start_time,
              "parameters_total": sum(p.numel() for p in model.parameters()),
              "parameters_u": sum(p.numel() for p in model.u.parameters())}
    atomic_json(output / "result.json", result)
    atomic_json(progress_path, {"stage": "done", "step": step,
                                "max_steps": args.max_steps,
                                "test_done": test_total, "test_total": test_total,
                                "test_accuracy": test["accuracy"]})
    print(f"RESULT {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
