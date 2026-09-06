"""Gold-oracle latent reachability diagnostic for a frozen motif-pair CVAE.

This is deliberately *not* a zero-shot experiment: the ideal mask is used in
both the objective and iterate selection.  It answers the narrower question
whether the decoder can express the known gap-specific structure when only
``z`` is allowed to move.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from data.generate import ideal_mask  # noqa: E402
from evaluation.eval_generated_masks import _check_provenance, _load_model  # noqa: E402
from models.cvae import checkpoint_condition_metadata, read_split_provenance  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _SoftTopK(torch.autograd.Function):
    """Sigmoid top-k projection whose last dimension sums exactly to ``k``."""

    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
        if logits.ndim < 1 or not 0 < k < logits.shape[-1] or temperature <= 0:
            raise ValueError("invalid soft top-k inputs")
        lo = logits.detach().amin(-1, keepdim=True) - 40 * temperature
        hi = logits.detach().amax(-1, keepdim=True) + 40 * temperature
        desired = torch.full_like(lo, float(k))
        for _ in range(64):
            threshold = (lo + hi) * .5
            count = torch.sigmoid((logits.detach() - threshold) / temperature).sum(-1, keepdim=True)
            lo = torch.where(count > desired, threshold, lo)
            hi = torch.where(count > desired, hi, threshold)
        result = torch.sigmoid((logits - (lo + hi) * .5) / temperature)
        ctx.save_for_backward(result)
        ctx.temperature = temperature
        return result

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        (result,) = ctx.saved_tensors
        weight = result * (1 - result)
        mean = (grad_output * weight).sum(-1, keepdim=True)
        mean = mean / weight.sum(-1, keepdim=True).clamp_min(torch.finfo(result.dtype).tiny)
        return weight * (grad_output - mean) / ctx.temperature, None, None


def soft_topk(logits: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
    return _SoftTopK.apply(logits, int(k), float(temperature))


def hard_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    if not 0 < k <= logits.shape[-1]:
        raise ValueError("invalid hard top-k k")
    result = torch.zeros_like(logits)
    return result.scatter(-1, logits.topk(k, dim=-1).indices, 1.)


def align_columns(reference: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Align ``other`` hidden columns to ``reference`` with one CPU transfer.

    Assignment costs are detached, but the returned batched gather preserves
    the ordinary gradient path through ``other``.
    """
    if reference.shape != other.shape or reference.ndim not in (2, 3):
        raise ValueError("expected equal [N,L,H] or [L,H] masks")
    one = reference.ndim == 2
    if one:
        reference, other = reference.unsqueeze(0), other.unsqueeze(0)
    # [N,H,H]: target-column versus source-column squared distances.
    left = reference.detach().transpose(1, 2)
    right = other.detach().transpose(1, 2)
    costs = (left[:, :, None, :] - right[:, None, :, :]).square().sum(-1).cpu().numpy()
    orders = []
    for cost in costs:
        rows, columns = linear_sum_assignment(cost)
        order = torch.empty(reference.shape[-1], dtype=torch.long)
        order[torch.as_tensor(rows)] = torch.as_tensor(columns)
        orders.append(order)
    order = torch.stack(orders).to(other.device)
    aligned = other.gather(2, order[:, None, :].expand_as(other))
    return aligned[0] if one else aligned


def _project_(z: torch.Tensor, radius: float) -> None:
    with torch.no_grad():
        z.mul_((radius / z.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1))


def _validate_target(target: torch.Tensor) -> int:
    if target.ndim not in (2, 3) or not bool(((target == 0) | (target == 1)).all()):
        raise ValueError("target must be a binary [input, hidden] matrix or batch")
    supports = target.sum(dim=(-2, -1)) if target.ndim == 3 else target.sum().reshape(1)
    if not bool((supports == supports[0]).all()):
        raise ValueError("every target must have the same cardinality")
    k = int(supports[0])
    if not 0 < k < target.shape[-2] * target.shape[-1]:
        raise ValueError("target must have nontrivial cardinality")
    return k


def optimize_ideal(model: torch.nn.Module, initial_z: torch.Tensor, target: torch.Tensor,
                   condition: torch.Tensor, *, steps: int = 1000, lr: float = .03,
                   temperature: float = .5, radius: float = 8.,
                   group_ids: torch.Tensor | None = None) -> dict[str, Any]:
    """Find diagnostic witnesses for one fixed condition without changing model weights."""
    if steps < 0 or lr <= 0 or temperature <= 0 or radius <= 0:
        raise ValueError("invalid optimization settings")
    k = _validate_target(target)
    if initial_z.ndim != 2:
        raise ValueError("initial_z must have shape [starts, latent]")
    n = initial_z.shape[0]
    if condition.ndim != 2 or condition.shape[1] != getattr(model, "cond_dim", condition.shape[1]):
        raise ValueError("condition has wrong shape for decoder")
    if condition.shape[0] == 1:
        condition = condition.expand(n, -1)
    if condition.shape[0] != n:
        raise ValueError("condition must have one row or one row per start")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    z = torch.nn.Parameter(initial_z.detach().clone())
    _project_(z, radius)
    start_z = z.detach().clone()
    target_batch = target.to(z.device, dtype=z.dtype)
    if target_batch.ndim == 2:
        target_batch = target_batch.expand(n, -1, -1)
    elif target_batch.shape[0] != n:
        raise ValueError("batched target must have one item per start")
    condition = condition.to(z.device, dtype=z.dtype)
    if group_ids is not None:
        group_ids = group_ids.to(z.device, dtype=torch.long)
        if group_ids.shape != (n,) or int(group_ids.min()) < 0:
            raise ValueError("group_ids must have one nonnegative item per start")
        group_count = int(group_ids.max()) + 1
    else:
        group_count = 0
    optimizer = torch.optim.Adam([z], lr=lr)
    best_loss = torch.full((n,), float("inf"), device=z.device)
    best_iou = torch.full((n,), -1., device=z.device)
    tie_loss = best_loss.clone()
    best_soft_z = z.detach().clone()
    best_hard_z = z.detach().clone()
    best_soft_steps = torch.zeros(n, dtype=torch.long, device=z.device)
    best_hard_steps = torch.zeros(n, dtype=torch.long, device=z.device)
    history: list[dict[str, float | int]] = []

    def evaluate(codes: torch.Tensor):
        logits = model.decode(codes, condition)
        if logits.shape != (n, target_batch.shape[-2] * target_batch.shape[-1]):
            raise ValueError("decoder output does not match target dimensions")
        soft = soft_topk(logits, k, temperature).reshape_as(target_batch)
        aligned_soft = align_columns(target_batch, soft)
        loss = (aligned_soft - target_batch).square().mean((1, 2))
        with torch.no_grad():
            hard = hard_topk(logits.detach(), k).reshape_as(target_batch)
            aligned_hard = align_columns(target_batch, hard)
            intersection = (aligned_hard * target_batch).sum((1, 2))
            iou = intersection / (2 * k - intersection)
        return loss, iou, soft, hard, aligned_hard

    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, iou, _, _, _ = evaluate(z)
        if not bool(torch.isfinite(loss).all()):
            raise RuntimeError("non-finite oracle objective")
        with torch.no_grad():
            improved = loss < best_loss
            best_loss = torch.where(improved, loss, best_loss)
            best_soft_z[improved] = z[improved]
            best_soft_steps[improved] = step
            improved_hard = (iou > best_iou) | ((iou == best_iou) & (loss < tie_loss))
            best_iou = torch.where(improved_hard, iou, best_iou)
            tie_loss = torch.where(improved_hard, loss, tie_loss)
            best_hard_z[improved_hard] = z[improved_hard]
            best_hard_steps[improved_hard] = step
            row: dict[str, Any] = {"step": step, "soft_loss_mean": float(loss.mean()),
                                   "hard_iou_mean": float(iou.mean()),
                                   "exact_now": int((iou == 1).sum()),
                                   "exact_ever": int((best_iou == 1).sum())}
            if group_ids is not None:
                row["groups"] = [
                    {"soft_loss_mean": float(loss[group_ids == group].mean()),
                     "hard_iou_mean": float(iou[group_ids == group].mean()),
                     "exact_now": int((iou[group_ids == group] == 1).sum()),
                     "exact_ever": int((best_iou[group_ids == group] == 1).sum())}
                    for group in range(group_count)
                ]
            history.append(row)
        if step % 100 == 0 or step == steps:
            print(f"[oracle] step={step}/{steps} loss={float(loss.mean()):.6f} "
                  f"IoU={float(iou.mean()):.4f} exact-ever={int((best_iou == 1).sum())}/{n}", flush=True)
        if step < steps:
            loss.sum().backward()
            optimizer.step()
            _project_(z, radius)

    def pack(codes: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            loss, iou, soft, hard, aligned_hard = evaluate(codes)
        return {"z": codes.detach().cpu(), "loss": loss.cpu(), "iou": iou.cpu(),
                "soft": soft.cpu(), "hard": hard.cpu(), "aligned_hard": aligned_hard.cpu()}

    for name, value in model.state_dict().items():
        if not torch.equal(before[name], value):
            raise AssertionError(f"frozen decoder state changed: {name}")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("frozen decoder received gradients")
    return {"initial": pack(start_z), "best_soft": pack(best_soft_z),
            "best_hard": pack(best_hard_z), "best_soft_steps": best_soft_steps.cpu(),
            "best_hard_steps": best_hard_steps.cpu(), "history": history,
            "condition": condition.detach().cpu(), "target": target_batch.detach().cpu(),
            "group_ids": None if group_ids is None else group_ids.cpu(), "decoder_unchanged": True}


def _stats(record: dict[str, torch.Tensor], radius: float) -> dict[str, Any]:
    iou, loss, z = record["iou"], record["loss"], record["z"]
    return {"mean_iou": float(iou.mean()), "max_iou": float(iou.max()),
            "min_iou": float(iou.min()), "exact_count": int((iou == 1).sum()),
            "mean_loss": float(loss.mean()), "mean_z_norm": float(z.norm(dim=1).mean()),
            "at_radius": int((z.norm(dim=1) >= radius - 1e-5).sum())}


def _task_for_gap(gap: int) -> config.Task:
    # Conditions and ideal support depend only on the gap.  Fixed valid motifs
    # keep this provenance explicit without exposing task identities to z.
    return config.Task("000", "001", gap)


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run outside the sandbox")
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    split = json.loads(args.split.read_text())
    expected_split = read_split_provenance(args.split)
    model, checkpoint = _load_model(args.checkpoint, device)
    _check_provenance(checkpoint, expected_split, "CVAE", importance_name=args.importance_name,
                      top_frac=args.top_frac)
    expected_condition = config.condition_metadata(config.CONDITION_ENCODING_SCALAR)
    if (model.cond_dim != 1 or checkpoint.get("condition_encoding") != config.CONDITION_ENCODING_SCALAR
            or checkpoint_condition_metadata(checkpoint) != expected_condition
            or any(checkpoint.get(key) != value for key, value in expected_condition.items())):
        raise ValueError("oracle requires an explicitly scalar, one-dimensional gap-conditioned CVAE")
    # CPU generator makes the common starts byte-identical across GPU profiles.
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    initial_z = torch.randn(args.n_starts, model.latent_dim, generator=generator).to(device)
    seen_gaps = sorted({config.parse_task(task).gap for task in split["train_tasks"]})
    heldout_gaps = sorted(set(config.GAPS) - set(seen_gaps))
    metadata = {
        "oracle": True, "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint), "split": str(args.split.resolve()),
        "split_sha256": expected_split["split_sha256"], "split_train_tasks": expected_split["split_train_tasks"],
        "source_sha256": _sha256(Path(__file__)), "torch_version": str(torch.__version__),
        "device": str(device), "settings": {"radius": args.radius, "steps": args.steps,
                                                 "n_starts": args.n_starts, "seed": args.seed,
                                                 "lr": args.lr, "temperature": args.temperature},
        "seen_gaps": seen_gaps, "heldout_gaps": heldout_gaps,
        "condition_encoding": checkpoint.get("condition_encoding"),
    }
    if args.protocol is not None:
        metadata["protocol"] = str(args.protocol.resolve())
        metadata["protocol_sha256"] = _sha256(args.protocol)
        protocol = json.loads(args.protocol.read_text())
        matching_profiles = [name for name, profile in protocol.get("profiles", {}).items()
                             if profile.get("checkpoint") == str(args.checkpoint.resolve())
                             and profile.get("split") == str(args.split.resolve())]
        if len(matching_profiles) != 1:
            raise ValueError("checkpoint/split do not identify exactly one pre-registered profile")
        profile = protocol["profiles"][matching_profiles[0]]
        checks = {
            "checkpoint_sha256": _sha256(args.checkpoint),
            "split_sha256": expected_split["split_sha256"],
            "n_starts": args.n_starts, "seed": args.seed, "steps": args.steps,
            "lr": args.lr, "temperature": args.temperature,
        }
        expected = {"checkpoint_sha256": profile["checkpoint_sha256"],
                    "split_sha256": profile["split_sha256"],
                    **{name: protocol[name] for name in ("n_starts", "seed", "steps", "lr", "temperature")}}
        if checks != expected or args.radius not in protocol.get("radii", []):
            raise ValueError("arguments or input hashes differ from the pre-registered protocol")
        metadata["protocol_profile"] = matching_profiles[0]
    gaps = tuple(config.GAPS)
    grouped_z = initial_z.repeat(len(gaps), 1)
    targets = torch.stack([ideal_mask(_task_for_gap(gap)) for gap in gaps]).to(device)
    targets = targets.repeat_interleave(args.n_starts, dim=0)
    conditions = torch.cat([model.condition([_task_for_gap(gap)], device=device).expand(args.n_starts, -1)
                            for gap in gaps])
    group_ids = torch.arange(len(gaps), device=device).repeat_interleave(args.n_starts)
    batch_result = optimize_ideal(model, grouped_z, targets, conditions,
                                  steps=args.steps, lr=args.lr, temperature=args.temperature,
                                  radius=args.radius, group_ids=group_ids)
    results: dict[str, Any] = {}
    summary: dict[str, Any] = {"metadata": metadata, "gaps": {}}
    for ordinal, gap in enumerate(gaps):
        selector = slice(ordinal * args.n_starts, (ordinal + 1) * args.n_starts)
        result = {key: ({field: item[selector].clone() for field, item in value.items()}
                        if key in ("initial", "best_soft", "best_hard") else value[selector].clone()
                        if key in ("best_soft_steps", "best_hard_steps") else value)
                  for key, value in batch_result.items() if key not in ("target", "condition", "group_ids")}
        result["target"] = batch_result["target"][selector].clone()
        result["condition"] = batch_result["condition"][selector].clone()
        result["history"] = [
            {key: value for key, value in row.items() if key != "groups"} | row["groups"][ordinal]
            for row in batch_result["history"]
        ]
        results[str(gap)] = result
        summary["gaps"][str(gap)] = {"label": "seen" if gap in seen_gaps else "heldout",
                                      "prior": _stats(result["initial"], args.radius),
                                      "best_soft": _stats(result["best_soft"], args.radius),
                                      "best_hard": _stats(result["best_hard"], args.radius)}
    torch.save({"metadata": metadata, "initial_z": initial_z.detach().cpu(), "results": results},
               args.out_dir / "result.pt")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved {args.out_dir / 'result.pt'}", flush=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--radius", type=float, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--n-starts", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--lr", type=float, default=.03)
    p.add_argument("--temperature", type=float, default=.5)
    p.add_argument("--top-frac", type=float, default=.1)
    p.add_argument("--importance-name", default="importance.pt")
    p.add_argument("--protocol", type=Path, help="pre-registered manifest whose hash is recorded")
    p.add_argument("--device", choices=("cuda",), default="cuda")
    return p


if __name__ == "__main__":
    parsed = parser().parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    run(parsed)
