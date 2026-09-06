"""Train a full-U generator or table through differentiable task adaptation."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from .common import (dataset, make_model, save_checkpoint, seed_for, setup, source_hashes,
                     task_splits, validation, write_json)
from .config import Config
from .models import adapt_v, forward_with_u


def stable_clip_grad_norm_(parameters, max_norm, eps=1e-6):
    """Clip finite gradients using a float64 reduction for the global norm.

    The default PyTorch float32 reduction can overflow while every individual
    gradient remains finite (for example, values around 1e23).  Such a batch
    is still usable: compute its norm in float64 and rescale it.  Actual
    non-finite gradient elements remain a hard failure.
    """
    if max_norm <= 0 or eps < 0:
        raise ValueError("max_norm must be positive and eps must be non-negative")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    total_squared = torch.zeros((), device=gradients[0].device, dtype=torch.float64)
    for gradient in gradients:
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Non-finite gradient before clipping")
        total_squared = total_squared + gradient.detach().to(torch.float64).square().sum()
    total_norm = torch.sqrt(total_squared)
    if not bool(torch.isfinite(total_norm)):
        raise FloatingPointError("Non-finite float64 gradient norm before clipping")
    scale = (float(max_norm) / (total_norm + float(eps))).clamp(max=1.0)
    if bool(scale < 1.0):
        for gradient in gradients:
            gradient.mul_(scale.to(dtype=gradient.dtype))
    return float(total_norm)


def _check_resume_config(current, saved):
    """Reject a resume whose model/data semantics changed."""
    previous_config = Config(**saved["config"])
    permitted_changes = {"outer_steps", "validate_every"}
    for key, value in current.to_dict().items():
        if key not in permitted_changes and value != previous_config.to_dict()[key]:
            raise ValueError(f"Cannot resume with changed {key}")


def _resume_best(resume, previous, current, device):
    """Load the historical best artifact accompanying a latest checkpoint.

    A continuation writes into a new directory, so retaining just ``latest``
    would otherwise reset best-model selection.  Old checkpoints did not save
    ``best_val_bce`` in ``latest.pt``; their sibling ``best.pt`` is therefore
    the authoritative backward-compatible source.
    """
    sibling = Path(resume).with_name("best.pt")
    if sibling.exists():
        best_state = torch.load(sibling, map_location=device, weights_only=False)
        if not isinstance(best_state, dict) or "config" not in best_state or "model" not in best_state:
            raise ValueError(f"Invalid sibling best checkpoint: {sibling}")
        _check_resume_config(current, best_state)
        score = best_state.get("val_bce")
    else:
        # A legacy standalone latest checkpoint only ever recorded its current
        # validation loss, in which case that model is the only recoverable
        # historical best.  New-format latest files with a lower recorded best
        # require their sibling artifact rather than silently selecting a
        # different model.
        best_state = previous
        score = previous.get("best_val_bce", previous.get("val_bce"))
        current_score = previous.get("val_bce")
        if ("best_val_bce" in previous and current_score is not None
                and float(score) < float(current_score) and not sibling.exists()):
            raise FileNotFoundError(
                f"Missing historical best checkpoint next to {resume}: expected {sibling}"
            )
    try:
        score = float(score)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Resume checkpoint has no validation score: {sibling}") from error
    if not math.isfinite(score):
        raise ValueError(f"Resume checkpoint has non-finite historical best score: {sibling}")
    return best_state, score, sibling


def train(c, out, device="cuda", resume=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {out}")
    setup(c.seed, device)
    model = make_model(c, device)
    splits = task_splits(c)
    protocol = {"config": c.to_dict(), "task_splits": {s: [t.pattern for t in ts] for s, ts in splits.items()},
                "source_sha256": source_hashes(), "torch": torch.__version__, "device": str(device),
                "inner_optimizer": f"functional {c.inner_optimizer}; fresh seeded initialization per episode",
                "outer_gradient": "full differentiation through all inner steps",
                "selection": "equal-length mean query BCE on validation patterns at train lengths"}
    write_json(out / "protocol.json", protocol)
    learnable = c.method in {"generator", "table"}
    optimizer = torch.optim.Adam(model.parameters(), lr=c.outer_lr) if learnable else None
    best, started = math.inf, time.monotonic()
    start_step = 0
    if resume is not None:
        previous = torch.load(resume, map_location=device, weights_only=False)
        _check_resume_config(c, previous)
        model.load_state_dict(previous["model"])
        if optimizer is not None:
            optimizer.load_state_dict(previous["optimizer"])
        start_step = previous["step"]
        if c.outer_steps <= start_step:
            raise ValueError("Resume requires a larger total outer-step budget")
        best_state, best, best_source = _resume_best(resume, previous, c, device)
        # Copy rather than link: a completed continuation remains self-contained
        # if the source experiment directory is later archived or moved.
        save_checkpoint(out / "best.pt", best_state)
        protocol["resumed_from"] = str(Path(resume).resolve())
        protocol["resumed_best_from"] = str(best_source.resolve())
        protocol["resumed_best_step"] = best_state.get("step")
        write_json(out / "protocol.json", protocol)
    train_by_length = {k: [t for t in splits["train"] if t.length == k] for k in c.train_lengths}

    def checkpoint(step):
        nonlocal best
        model.eval()
        score, rows = validation(c, model, device)
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite validation loss")
        improved = score < best
        if improved:
            best = score
        state = {"config": c.to_dict(), "model": model.state_dict(), "step": step, "val_bce": score,
                 "best_val_bce": best, "source_sha256": protocol["source_sha256"],
                 "optimizer": optimizer.state_dict() if optimizer else None}
        save_checkpoint(out / "latest.pt", state)
        if improved:
            save_checkpoint(out / "best.pt", state)
        with (out / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": step, "bce": score, "best": improved, "tasks": rows}) + "\n")
        print(f"VALIDATE method={c.method} seed={c.seed} step={step} bce={score:.6f} best={best:.6f}", flush=True)
        model.train()

    checkpoint(start_step)
    for step in range(start_step + 1, c.outer_steps + 1 if learnable else 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        # Cycle lengths for equal exposure; randomly choose a pattern within that length.
        for slot in range(c.tasks_per_step):
            length = c.train_lengths[((step - 1) * c.tasks_per_step + slot) % len(c.train_lengths)]
            episode_seed = seed_for("train", c.seed if c.data_seed is None else c.data_seed, step, slot)
            task = random.Random(episode_seed).choice(train_by_length[length])
            support = dataset(c, task, c.support_size, seed_for(episode_seed, "support"), "support", device)
            query = dataset(c, task, c.query_size, seed_for(episode_seed, "query"), "query", device)
            u = model(length)
            v = adapt_v(u, support["x"], support["y"], steps=c.inner_steps, lr=c.inner_lr,
                        seed=seed_for(episode_seed, "v"), create_graph=True, batch_size=c.batch_size,
                        optimizer=c.inner_optimizer, init_scale=c.init_scale)
            loss = F.binary_cross_entropy_with_logits(
                forward_with_u(query["x"], u, v, seq_len=c.seq_len, hidden=c.hidden), query["y"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite meta loss at step {step}")
            (loss / c.tasks_per_step).backward()
            total += loss.detach().item() / c.tasks_per_step
        grad_norm = stable_clip_grad_norm_(model.parameters(), c.grad_clip)
        optimizer.step()
        row = {"step": step, "query_bce": total, "gradient_norm_before_clip": float(grad_norm),
               "seconds": time.monotonic() - started}
        with (out / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if step == 1 or step % 10 == 0:
            print(f"TRAIN method={c.method} seed={c.seed} step={step} query_bce={total:.6f} grad={float(grad_norm):.6g}", flush=True)
        if step % c.validate_every == 0 or step == c.outer_steps:
            checkpoint(step)
    write_json(out / "done.json", {"best_val_bce": best, "seconds": time.monotonic() - started,
                                    "trained_steps": c.outer_steps if learnable else 0})
    return out / "best.pt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", type=Path)
    p.add_argument("--method", choices=("generator", "table", "random", "ideal"), default="generator")
    for name in ("seed", "outer_steps", "inner_steps", "tasks_per_step", "support_size", "query_size",
                 "batch_size", "validate_every", "val_tasks_per_length", "rank1", "rank2", "width", "generator_depth"):
        p.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(Config(), name))
    for name in ("inner_lr", "outer_lr", "init_scale"):
        p.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(Config(), name))
    p.add_argument("--train-lengths", nargs="+", type=int, default=list(Config().train_lengths))
    p.add_argument("--inner-optimizer", choices=["sgd", "adam"], default="sgd")
    p.add_argument("--unconditional", action="store_true")
    p.add_argument("--all-unseen-patterns", action="store_true")
    p.add_argument("--data-seed", type=int)
    args = vars(p.parse_args())
    out, device = args.pop("out"), args.pop("device")
    resume = args.pop("resume")
    args["condition_length"] = not args.pop("unconditional")
    args["train_lengths"] = tuple(args["train_lengths"])
    train(Config(**args), out, device, resume)


if __name__ == "__main__":
    main()
