"""Jointly train latent codes and task MLPs on each VAE's hard mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from pattern.length11.agreement import PAIRS, align, column_orders, decode, load_models, topk
from pattern.length11.bank import all_inputs, labels
from pattern.length11.settings import EDGES, HIDDEN, ROOT, SEQ_LEN


STARTS = 16
REPEATS = 2
STEPS = 10000
LAMBDA = .5


def run(pair_index: int, replicate: int, device: torch.device,
        out: Path = ROOT, *, variant: str = "joint", max_steps: int = STEPS) -> Path:
    pair = PAIRS[pair_index]
    destination = out / variant / f"pair{pair_index:02d}_rep{replicate}.pt"
    if destination.exists():
        return destination
    models = load_models(pair, replicate, out, device)
    n = len(pair)
    seed = 20260929 + pair_index * 100 + replicate
    torch.manual_seed(seed)
    z = torch.nn.Parameter(torch.randn(n, STARTS, 32, device=device))
    with torch.no_grad():
        orders = column_orders(decode(models, z))
    shape = (len(pair), REPEATS, n, STARTS)
    w1 = torch.nn.Parameter(torch.randn(*shape, SEQ_LEN, HIDDEN, device=device) * .1)
    b1 = torch.nn.Parameter(torch.zeros(*shape, HIDDEN, device=device))
    w2 = torch.nn.Parameter(torch.randn(*shape, HIDDEN, device=device) * .1)
    b2 = torch.nn.Parameter(torch.zeros(*shape, device=device))
    mlp_params = [w1, b1, w2, b2]
    optimizer = torch.optim.Adam([{"params": [z], "lr": .02},
                                  {"params": mlp_params, "lr": .001}])
    x, bits = all_inputs()
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(1729))
    support = order[:round(.8 * len(x))]
    train, val = support[:round(.8 * len(support))], support[round(.8 * len(support)):]
    x = x.to(device)
    task_data = []
    for pattern in pair:
        y_cpu = labels(bits, pattern)
        pos = train[y_cpu[train] == 1]
        neg = train[y_cpu[train] == 0]
        task_data.append((y_cpu.to(device), pos, neg))
    generator = torch.Generator().manual_seed(seed + 1)

    def logits_and_mask() -> tuple[torch.Tensor, torch.Tensor]:
        raw = align(decode(models, z), orders)
        flat = raw.flatten(2)
        hard = topk(flat)
        kth = flat.topk(EDGES, dim=-1).values[..., -1:].detach()
        soft = torch.sigmoid((flat - kth) / .25)
        gate = (hard + soft - soft.detach()).reshape(n, STARTS, SEQ_LEN, HIDDEN)
        return raw, gate

    def task_scores(gates: torch.Tensor, ids: torch.Tensor, task: int) -> torch.Tensor:
        gates = gates[None].expand(REPEATS, -1, -1, -1, -1)
        weight = w1[task] * gates
        hidden = torch.einsum("bi,rsnih->brsnh", x[ids], weight)
        hidden = F.relu(hidden + b1[task][None])
        return torch.einsum("brsnh,rsnh->brsn", hidden, w2[task]) + b2[task][None]

    best_score = torch.full((STARTS,), float("inf"), device=device)
    best_z = z.detach().clone()
    history = []
    stale = 0
    progress = tqdm(range(1, max_steps + 1), desc=f"joint pair{pair_index} rep{replicate}",
                    unit="step", mininterval=2)
    for step in progress:
        raw, gates = logits_and_mask()
        losses = []
        for task, (y, pos, neg) in enumerate(task_data):
            chosen = torch.cat((
                pos[torch.randint(len(pos), (64,), generator=generator)],
                neg[torch.randint(len(neg), (64,), generator=generator)]
            )).to(device)
            scores = task_scores(gates, chosen, task)
            bce = F.binary_cross_entropy_with_logits(
                scores, y[chosen, None, None, None].expand_as(scores),
                reduction="none").mean(0)
            losses.append(bce.mean((0, 1)))  # mean repeats and decoders, one value per start
        task_loss = torch.stack(losses).mean(0)
        mse = (raw - raw.mean(0, keepdim=True)).square().mean((0, 2, 3))
        objective = (task_loss + LAMBDA * mse).sum()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            z *= (12. / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1)
        if step % 500 == 0:
            with torch.no_grad():
                raw, gates = logits_and_mask()
                values = []
                for task, (y, _, _) in enumerate(task_data):
                    ids = val.to(device)
                    scores = task_scores(gates, ids, task)
                    errors = F.binary_cross_entropy_with_logits(
                        scores, y[ids, None, None, None].expand_as(scores),
                        reduction="none")
                    pos = y[ids] == 1
                    balanced = .5 * (errors[pos].mean(0) + errors[~pos].mean(0))
                    values.append(balanced.mean((0, 1)))
                val_task = torch.stack(values).mean(0)
                val_mse = (raw - raw.mean(0, keepdim=True)).square().mean((0, 2, 3))
                score = val_task + LAMBDA * val_mse
                previous_best = float(best_score.min())
                improved = score < best_score - 1e-4
                best_score = torch.where(improved, score, best_score)
                best_z = torch.where(improved[None, :, None], z.detach(), best_z)
                stale = 0 if float(best_score.min()) < previous_best - 1e-4 else stale + 1
                history.append((step, float(best_score.min()),
                                float(val_task.mean()), float(val_mse.mean())))
                progress.set_postfix(score=f"{float(best_score.min()):.3f}")
                if stale >= 8:
                    break
    with torch.no_grad():
        aligned = align(decode(models, best_z), orders)
        chosen = int(best_score.argmin())
        payload = {"pair": list(pair), "replicate": replicate,
                   "z": best_z.cpu(), "chosen_start": chosen,
                   "logits": aligned.cpu(), "masks": topk(aligned).cpu(),
                   "score": best_score.cpu(), "history": torch.tensor(history),
                   "settings": {"starts": STARTS, "mlp_repeats": REPEATS,
                                "max_steps": max_steps, "lambda": LAMBDA,
                                "early_stop_patience_checks": 8,
                                "objective": "per-VAE hard-mask task BCE on both tasks + raw-logit MSE"}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(destination)
    print(f"joint pair{pair_index} rep{replicate}: score={float(best_score.min()):.4f} "
          f"steps={step}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=int, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--variant", default="joint")
    parser.add_argument("--max-steps", type=int, default=STEPS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.pair, args.replicate, torch.device(args.device), args.out,
        variant=args.variant, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
