"""Task-aware search for 1–10 frozen pattern-specific VAEs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from pattern.length11.agreement import align, column_orders, decode, load_models, topk
from pattern.length11.bank import all_inputs, labels
from pattern.length11.settings import EDGES, HIDDEN, ROOT, SEQ_LEN


def run(patterns: tuple[str, ...], replicate: int, out: Path, device: torch.device,
        *, starts: int = 16, max_steps: int = 30000, coefficient: float = .5,
        repeats: int = 2, warm_start: Path | None = None) -> Path:
    n = len(patterns)
    if not 1 <= n <= 10 or len(set(patterns)) != n:
        raise ValueError("expected 1–10 distinct patterns")
    destination = out / "joint_multi" / f"n{n}" / f"rep{replicate}.pt"
    settings = {"starts": starts, "max_steps": max_steps,
                "coefficient": coefficient, "repeats": repeats}
    if destination.exists():
        prior = torch.load(destination, map_location="cpu", weights_only=True)
        if (prior["patterns"] != list(patterns) or prior["replicate"] != replicate or
                any(prior["settings"][key] != value for key, value in settings.items())):
            raise ValueError(f"existing search has a different protocol: {destination}")
        return destination
    torch.set_num_threads(2)
    models = load_models(patterns, replicate, out, device)
    seed = 20260930 + 100 * replicate + n
    torch.manual_seed(seed)
    z = torch.nn.Parameter(torch.randn(n, starts, 32, device=device))
    warm_count = 0
    if warm_start is not None and warm_start.exists():
        previous = torch.load(warm_start, map_location="cpu", weights_only=True)
        previous_z = previous["z"]
        if previous_z.shape != (n - 1, starts, 32):
            raise ValueError(f"warm start shape mismatch: {warm_start}")
        warm_count = starts // 2
        with torch.no_grad():
            z[:n - 1, :warm_count] = previous_z[:, :warm_count].to(device)
    with torch.no_grad():
        orders = column_orders(decode(models, z))
    shape = (n, repeats, n, starts)
    w1 = torch.nn.Parameter(torch.randn(*shape, SEQ_LEN, HIDDEN, device=device) * .1)
    b1 = torch.nn.Parameter(torch.zeros(*shape, HIDDEN, device=device))
    w2 = torch.nn.Parameter(torch.randn(*shape, HIDDEN, device=device) * .1)
    b2 = torch.nn.Parameter(torch.zeros(*shape, device=device))
    optimizer = torch.optim.Adam([
        {"params": [z], "lr": .02},
        {"params": [w1, b1, w2, b2], "lr": .001},
    ])
    x_cpu, bits = all_inputs()
    order = torch.randperm(len(x_cpu), generator=torch.Generator().manual_seed(1729))
    support = order[:round(.8 * len(x_cpu))]
    train, val = support[:round(.8 * len(support))], support[round(.8 * len(support)):]
    x = x_cpu.to(device)
    y_cpu = torch.stack([labels(bits, pattern) for pattern in patterns])
    y = y_cpu.to(device)
    positives = [train[y_cpu[task, train] == 1] for task in range(n)]
    negatives = [train[y_cpu[task, train] == 0] for task in range(n)]
    generator = torch.Generator().manual_seed(seed + 1)

    def logits_and_masks() -> tuple[torch.Tensor, torch.Tensor]:
        raw = align(decode(models, z), orders)
        flat = raw.flatten(2)
        hard = topk(flat)
        kth = flat.topk(EDGES, dim=-1).values[..., -1:].detach()
        soft = torch.sigmoid((flat - kth) / .25)
        masks = (hard + soft - soft.detach()).reshape(n, starts, SEQ_LEN, HIDDEN)
        return raw, masks

    def mlp_scores(mask: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        weighted = w1 * mask[None, None]
        hidden = torch.einsum("tbi,trvsih->tbrvsh", xb, weighted)
        hidden = F.relu(hidden + b1[:, None])
        return torch.einsum("tbrvsh,trvsh->tbrvs", hidden, w2) + b2[:, None]

    best_score = torch.full((starts,), float("inf"), device=device)
    best_z = z.detach().clone()
    history = []
    stale = 0
    progress = tqdm(range(1, max_steps + 1), desc=f"joint n{n} rep{replicate}",
                    unit="step", mininterval=2)
    for step in progress:
        raw, masks = logits_and_masks()
        chosen = torch.stack([
            torch.cat((
                positives[task][torch.randint(len(positives[task]), (64,), generator=generator)],
                negatives[task][torch.randint(len(negatives[task]), (64,), generator=generator)],
            )) for task in range(n)
        ]).to(device)
        scores = mlp_scores(masks, x[chosen])
        target = y.gather(1, chosen)[:, :, None, None, None]
        task_loss = F.binary_cross_entropy_with_logits(
            scores, target.expand_as(scores), reduction="none").mean(1).mean((0, 1, 2))
        mse = (raw - raw.mean(0, keepdim=True)).square().mean((0, 2, 3))
        objective = (task_loss + coefficient * mse).sum()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            z *= (12. / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1)
        if step % 500 == 0:
            with torch.no_grad():
                raw, masks = logits_and_masks()
                ids = val.to(device)
                xb = x[ids][None].expand(n, -1, -1)
                val_scores = mlp_scores(masks, xb)
                targets = y[:, ids]
                errors = F.binary_cross_entropy_with_logits(
                    val_scores, targets[:, :, None, None, None].expand_as(val_scores),
                    reduction="none")
                by_task = []
                for task in range(n):
                    pos = targets[task] == 1
                    balanced = .5 * (errors[task, pos].mean(0) +
                                     errors[task, ~pos].mean(0))
                    by_task.append(balanced.mean((0, 1)))
                val_task = torch.stack(by_task).mean(0)
                val_mse = (raw - raw.mean(0, keepdim=True)).square().mean((0, 2, 3))
                val_score = val_task + coefficient * val_mse
                previous_best = float(best_score.min())
                improved = val_score < best_score - 1e-4
                best_score = torch.where(improved, val_score, best_score)
                best_z = torch.where(improved[None, :, None], z.detach(), best_z)
                stale = 0 if float(best_score.min()) < previous_best - 1e-4 else stale + 1
                history.append((step, float(best_score.min()), float(val_task.mean()),
                                float(val_mse.mean())))
                progress.set_postfix(score=f"{float(best_score.min()):.3f}")
                if stale >= 8:
                    break
    with torch.no_grad():
        aligned = align(decode(models, best_z), orders)
        chosen_start = int(best_score.argmin())
        payload = {
            "patterns": list(patterns), "replicate": replicate,
            "z": best_z.cpu(), "chosen_start": chosen_start,
            "logits": aligned.cpu(), "masks": topk(aligned).cpu(),
            "score": best_score.cpu(), "history": torch.tensor(history),
            "settings": {**settings, "warm_starts": warm_count,
                         "actual_steps": step, "stopped_on_plateau": stale >= 8,
                         "objective": "mean task BCE on every VAE top-32 mask for every pattern + coefficient * raw-logit MSE"},
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(destination)
    print(f"joint n{n} rep{replicate}: score={float(best_score.min()):.4f} "
          f"steps={step} plateau={stale >= 8}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patterns", nargs="+", required=True)
    parser.add_argument("--replicate", type=int, choices=range(4), required=True)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--coefficient", type=float, default=.5)
    parser.add_argument("--warm-start", type=Path)
    args = parser.parse_args()
    run(tuple(args.patterns), args.replicate, args.out, torch.device(args.device),
        starts=args.starts, max_steps=args.max_steps,
        coefficient=args.coefficient, warm_start=args.warm_start)


if __name__ == "__main__":
    main()
