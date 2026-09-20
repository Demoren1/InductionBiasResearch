"""Image-routed mixture of v experts for the generated first-layer U."""

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

from .meta_u import load_pool, normalized_mse
from .meta_u_first import HIDDEN, INPUT_PIXELS, RANK, FirstLayerU, shifted_episode
from .meta_u_first_image_code import ConvImageEncoder, atomic_torch_save


LENGTHS = (1, 3, 5, 10, 20)


class VExpertMoE(nn.Module):
    def __init__(self, experts: int, seed: int) -> None:
        super().__init__()
        if experts < 1 or experts > RANK:
            raise ValueError("Number of experts must be between 1 and 32")
        self.n_experts = experts
        self.base = FirstLayerU("generated_ortho", seed)
        initial_v = self.base.initial_v1.detach().clone()
        self.base.initial_v1.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 5107)
            if experts == 1:
                initial_experts = initial_v[None]
                self.router = None
            else:
                noise = 0.08 * torch.randn(experts, RANK)
                noise -= noise.mean(dim=0, keepdim=True)
                initial_experts = initial_v[None] + noise
                self.router = ConvImageEncoder(32, out_features=experts)
                nn.init.normal_(self.router.head.weight, std=0.01)
                nn.init.zeros_(self.router.head.bias)
        self.v_experts = nn.Parameter(initial_experts)
        # At uniform routing, this gives the old model's initial v and
        # roughly matches the scale of its inner-loop update across K.
        self.initial_coefficients = nn.Parameter(
            torch.full((experts,), 1.0 / math.sqrt(experts)))

    def route(self, images: torch.Tensor, *, uniform: bool = False) -> torch.Tensor:
        shape = images.shape[:-1]
        if uniform or self.router is None:
            return images.new_full((*shape, self.n_experts), 1 / self.n_experts)
        logits = self.router(images.reshape(-1, INPUT_PIXELS))
        return logits.softmax(-1).reshape(*shape, self.n_experts)

    def prepare(self, images: torch.Tensor, u: torch.Tensor,
                *, uniform_route: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        expanded = F.pad(images, (0, 1), value=1.0) @ u.reshape(
            INPUT_PIXELS + 1, HIDDEN * RANK)
        projections = expanded.reshape(*images.shape[:-1], HIDDEN, RANK)
        return projections, self.route(images, uniform=uniform_route)

    def predict(self, prepared: tuple[torch.Tensor, torch.Tensor],
                mask: torch.Tensor, coefficients: torch.Tensor,
                readout: torch.Tensor) -> torch.Tensor:
        projections, route = prepared
        effective_v = (route * (coefficients * math.sqrt(self.n_experts))) @ (
            self.v_experts)
        first = torch.tanh((projections * effective_v[..., None, :]).sum(-1))
        second = torch.tanh(self.base.second(first))
        third = torch.tanh(self.base.third(second))
        per_image = F.pad(third, (0, 1), value=1.0) @ readout
        return (per_image * mask).sum(dim=1)

    def orthogonality_loss(self) -> torch.Tensor:
        if self.n_experts == 1:
            return self.v_experts.new_zeros(())
        unit = F.normalize(self.v_experts, dim=1)
        gram = unit @ unit.T
        off_diagonal = gram - torch.diag_embed(gram.diagonal())
        return off_diagonal.square().sum() / (
            self.n_experts * (self.n_experts - 1))


def adapt_query(model: VExpertMoE, support: tuple, query: tuple, *,
                u: torch.Tensor, steps: int, lr_coeff: float,
                lr_readout: float, meta_gradient: bool) -> torch.Tensor:
    sp = model.prepare(support[0], u)
    qp = model.prepare(query[0], u)
    coefficients = model.initial_coefficients
    readout = model.base.initial_readout
    for _ in range(steps):
        prediction = model.predict(sp, support[1], coefficients, readout)
        loss = normalized_mse(prediction, support[2], support[3])
        gc, gr = torch.autograd.grad(
            loss, (coefficients, readout), create_graph=meta_gradient)
        coefficients = coefficients - lr_coeff * gc
        readout = readout - lr_readout * gr
    prediction = model.predict(qp, query[1], coefficients, readout)
    return normalized_mse(prediction, query[2], query[3])


def evaluate(model: VExpertMoE, pool: tuple, *, seed: int, tasks: int,
             support: int, query: int, steps: int, lr_coeff: float,
             lr_readout: float, lengths: tuple[int, ...],
             uniform_route: bool = False, shuffle_route: bool = False) -> dict:
    if uniform_route and shuffle_route:
        raise ValueError("Choose one routing ablation")
    model.eval()
    rng = torch.Generator(device=pool[0].device).manual_seed(seed)
    shuffle_rng = torch.Generator(device=pool[0].device).manual_seed(seed + 41000)
    scores = [torch.randn(10, generator=rng, device=pool[0].device)
              for _ in range(tasks)]
    with torch.no_grad():
        u = model.base.u()
    result = {}
    for length in lengths:
        rows = []
        for score in scores:
            s, q, _ = shifted_episode(
                pool, support=support, query=query, rng=rng,
                query_length=length, digit_scores=score)
            with torch.no_grad():
                sp = model.prepare(s[0], u, uniform_route=uniform_route)
                qp = model.prepare(q[0], u, uniform_route=uniform_route)
                if shuffle_route:
                    def shuffled(prepared: tuple, mask: torch.Tensor) -> tuple:
                        projections, route = prepared
                        route = route.clone()
                        active = route[mask]
                        permutation = torch.randperm(
                            len(active), generator=shuffle_rng,
                            device=active.device)
                        route[mask] = active[permutation]
                        return projections, route

                    sp = shuffled(sp, s[1])
                    qp = shuffled(qp, q[1])
            coefficients = model.initial_coefficients.detach().clone().requires_grad_(True)
            readout = model.base.initial_readout.detach().clone().requires_grad_(True)
            for _ in range(steps):
                prediction = model.predict(sp, s[1], coefficients, readout)
                loss = normalized_mse(prediction, s[2], s[3])
                gc, gr = torch.autograd.grad(loss, (coefficients, readout))
                coefficients = (coefficients - lr_coeff * gc).detach().requires_grad_(True)
                readout = (readout - lr_readout * gr).detach().requires_grad_(True)
            prediction = model.predict(qp, q[1], coefficients, readout)
            rows.append((normalized_mse(
                prediction, q[2], q[3]).detach().item(),
                (prediction - q[2]).abs().mean().detach().item()))
        result[str(length)] = {
            "normalized_mse": float(np.mean([row[0] for row in rows])),
            "mae": float(np.mean([row[1] for row in rows])),
            "mae_per_task": [row[1] for row in rows]}
    return result


@torch.no_grad()
def routing_diagnostics(model: VExpertMoE, pool: tuple) -> dict:
    model.eval()
    pixels, digits = pool
    total = torch.zeros(10, model.n_experts, device=pixels.device)
    counts = torch.zeros(10, device=pixels.device)
    entropy_sum = 0.0
    for start in range(0, len(pixels), 256):
        images = pixels[start:start + 256].float().div_(255.0)
        labels = digits[start:start + 256]
        route = model.route(images)
        total.index_add_(0, labels, route)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
        entropy_sum += float((-(route * route.clamp_min(1e-12).log()).sum(-1)).sum())
    per_digit = total / counts[:, None]
    overall = total.sum(0) / counts.sum()
    return {
        "mean_route_by_true_digit": per_digit.cpu().tolist(),
        "mean_expert_usage": overall.cpu().tolist(),
        "mean_routing_entropy": entropy_sum / len(pixels),
        "normalized_routing_entropy": (entropy_sum / len(pixels) /
                                       math.log(model.n_experts)
                                       if model.n_experts > 1 else 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--ortho-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mnist8m"))
    parser.add_argument("--out", type=Path, default=Path(
        "deepsets_z/mnist8m/outputs/meta_u_first_v_moe"))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--lr-coeff", type=float, default=0.1)
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
    if args.experts < 1 or args.experts > RANK or args.ortho_weight < 0:
        parser.error("Expected 1..32 experts and non-negative ortho weight")
    if min(args.steps, args.inner_steps, args.tasks_per_step, args.support,
           args.query, args.train_images_per_digit, args.eval_images_per_digit,
           args.eval_every, args.val_tasks, args.test_tasks) < 1:
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
    model = VExpertMoE(args.experts, args.seed).to(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.outer_lr)
    rng = torch.Generator(device=device).manual_seed(args.seed + 1200)
    args.out.mkdir(parents=True, exist_ok=True)
    weight_tag = f"{args.ortho_weight:g}".replace(".", "p")
    stem = f"moe_k{args.experts}_ortho{weight_tag}_seed{args.seed}"
    result_path = args.out / f"{stem}.json"
    latest_path = args.out / f"{stem}_latest.pt"
    best_path = args.out / f"{stem}_best.pt"
    if result_path.exists() and not args.resume:
        raise FileExistsError(result_path)
    signature = {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()
                 if key not in ("device", "out", "resume")}
    best_score, best_step, history, start_step, elapsed_before = (
        math.inf, 0, [], 0, 0.0)
    if args.resume:
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        if saved["config"] != signature:
            raise ValueError(f"Checkpoint settings differ: {latest_path}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["rng"].cpu())
        best_score, best_step = saved["best_score"], saved["best_step"]
        history, start_step = saved["history"], saved["step"]
        elapsed_before = saved["elapsed_seconds"]
    output = {
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "model": {
            "u_shape": [235500, RANK], "v_experts_shape": [args.experts, RANK],
            "router_parameters": (sum(p.numel() for p in model.router.parameters())
                                  if model.router is not None else 0),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad),
            "adapted_parameters": [args.experts, 31]},
        "best_step": best_step, "best_validation_score": best_score,
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
            losses.append(adapt_query(
                model, s, q, u=shared_u, steps=args.inner_steps,
                lr_coeff=args.lr_coeff, lr_readout=args.lr_readout,
                meta_gradient=True))
        data_loss = torch.stack(losses).mean()
        orth_loss = model.orthogonality_loss()
        outer_loss = data_loss + args.ortho_weight * orth_loss
        outer_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 5.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val = evaluate(
                model, validation, seed=args.seed + 9000,
                tasks=args.val_tasks, support=args.support, query=args.query,
                steps=args.inner_steps, lr_coeff=args.lr_coeff,
                lr_readout=args.lr_readout, lengths=(1, 3, 5))
            score = float(np.mean(
                [val[str(length)]["normalized_mse"] for length in (1, 3, 5)]))
            row = {"step": step, "train_normalized_mse": data_loss.item(),
                   "train_orthogonality": orth_loss.item(),
                   "validation_score": score,
                   "elapsed_seconds": time.monotonic() - start}
            history.append(row)
            if score < best_score:
                best_score, best_step = score, step
                atomic_torch_save(model.state_dict(), best_path)
            atomic_torch_save({
                "config": signature, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "rng": rng.get_state(),
                "best_score": best_score, "best_step": best_step,
                "history": history, "step": step,
                "elapsed_seconds": row["elapsed_seconds"]}, latest_path)
            output["best_step"] = best_step
            output["best_validation_score"] = best_score
            temporary = result_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(output, indent=2) + "\n")
            temporary.replace(result_path)
            print(f"{stem} step={step} train={data_loss.item():.4f} "
                  f"orth={orth_loss.item():.4f} val={score:.4f} "
                  f"best={best_step} elapsed={row['elapsed_seconds']:.0f}s",
                  flush=True)
    model.load_state_dict(torch.load(
        best_path, map_location=device, weights_only=True))
    output["test"] = evaluate(
        model, test, seed=args.seed + 19000, tasks=args.test_tasks,
        support=args.support, query=args.query, steps=args.inner_steps,
        lr_coeff=args.lr_coeff, lr_readout=args.lr_readout, lengths=LENGTHS)
    output["test_uniform_route"] = evaluate(
        model, test, seed=args.seed + 19000, tasks=args.test_tasks,
        support=args.support, query=args.query, steps=args.inner_steps,
        lr_coeff=args.lr_coeff, lr_readout=args.lr_readout, lengths=LENGTHS,
        uniform_route=True)
    output["expert_orthogonality"] = model.orthogonality_loss().item()
    output["routing"] = routing_diagnostics(model, test)
    result_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"DONE {stem} best={best_step} "
          f"test5_mae={output['test']['5']['mae']:.4f} "
          f"uniform={output['test_uniform_route']['5']['mae']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
