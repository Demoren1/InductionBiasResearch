"""Evaluate length-11 agreement masks against tasks, gold, and random masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from pattern.length11.agreement import PAIRS, topk
from pattern.length11.bank import all_inputs, labels
from pattern.length11.settings import EDGES, HIDDEN, ROOT, SEQ_LEN
from pattern.length_interp.bank import random_exact_k_masks
from pattern.length_interp.mlp import BatchedMaskedMLP
from pattern.length_interp.mlp import ideal_mask


CHECK_EVERY = 500
MAX_STEPS = 20000
PATIENCE = 8
BATCH = 128
MLP_REPEATS = 4


def gold_mask() -> torch.Tensor:
    return ideal_mask("0000", seq_len=SEQ_LEN, hidden=HIDDEN)


def gold_iou(mask: torch.Tensor) -> float:
    gold = gold_mask()
    cost = -torch.matmul(gold.T, mask).numpy()
    rows, cols = linear_sum_assignment(cost)
    aligned = mask[:, cols[rows.argsort()]]
    overlap = (gold * aligned).sum()
    return float(overlap / (2 * EDGES - overlap))


def candidates(data: dict, seed: int) -> tuple[list[str], torch.Tensor]:
    j = data["chosen_start"]
    final = data["final_masks"][:, j]
    initial = data["initial_masks"][:, j]
    consensus_final = topk(data["final_logits"][:, j].mean(0))
    consensus_initial = topk(data["initial_logits"][:, j].mean(0))
    masks = [*final, consensus_final, *initial, consensus_initial]
    names = ([f"final_vae{i}" for i in range(len(final))] +
             ["final_consensus"] +
             [f"initial_vae{i}" for i in range(len(initial))] +
             ["initial_consensus"])
    random = random_exact_k_masks(16, seq_len=SEQ_LEN, hidden=HIDDEN,
                                  k_active=EDGES, seed=seed)
    masks.extend(random)
    names.extend(f"random_{i:02d}" for i in range(len(random)))
    masks.append(gold_mask())
    names.append("analytic")
    return names, torch.stack(masks)


def balanced(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pos = targets == 1
    return .5 * (values[pos].mean(0) + values[~pos].mean(0))


def fit_and_evaluate(masks: torch.Tensor, pattern: str, seed: int,
                     device: torch.device) -> dict:
    torch.set_num_threads(2)
    x, bits = all_inputs()
    y = labels(bits, pattern)
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(1729))
    support, test = order[:round(.8 * len(x))], order[round(.8 * len(x)):]
    train, val = support[:round(.8 * len(support))], support[round(.8 * len(support)):]
    pos = train[y[train] == 1]
    neg = train[y[train] == 0]
    x, y = x.to(device), y.to(device)
    n_cases = len(masks)
    model = BatchedMaskedMLP(masks.repeat(MLP_REPEATS, 1, 1), seed=seed).to(device)
    with torch.no_grad():
        for param in (model.w1, model.b1, model.w2, model.b2):
            for repeat in range(MLP_REPEATS):
                block = param[repeat * n_cases:(repeat + 1) * n_cases]
                block[:] = block[:1]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val = torch.full((model.n_models,), float("inf"), device=device)
    best_state = {key: value.detach().clone() for key, value in
                  model.named_parameters()}
    stale = torch.zeros(model.n_models, dtype=torch.int, device=device)
    generator = torch.Generator().manual_seed(seed + 1)
    history = []
    progress = tqdm(range(1, MAX_STEPS + 1), desc=f"task {pattern}",
                    unit="step", mininterval=2)
    for step in progress:
        selected = torch.cat((
            pos[torch.randint(len(pos), (BATCH // 2,), generator=generator)],
            neg[torch.randint(len(neg), (BATCH - BATCH // 2,), generator=generator)]
        )).to(device)
        xb, yb = x[selected], y[selected]
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(
            logits, yb[:, None].expand_as(logits), reduction="none").mean(0)
        optimizer.zero_grad(set_to_none=True)
        loss.sum().backward()
        optimizer.step()
        if step % CHECK_EVERY == 0:
            with torch.no_grad():
                vx, vy = x[val.to(device)], y[val.to(device)]
                logits = model(vx)
                losses = F.binary_cross_entropy_with_logits(
                    logits, vy[:, None].expand_as(logits), reduction="none")
                score = balanced(losses, vy)
                improved = score < best_val - 1e-4
                best_val = torch.where(improved, score, best_val)
                stale = torch.where(improved, 0, stale + 1)
                for key, value in model.named_parameters():
                    old = best_state[key]
                    old[improved] = value.detach()[improved]
                history.append((step, float(score.mean()), float(best_val.mean())))
                progress.set_postfix(val=f"{float(best_val.mean()):.3f}")
                if (stale >= PATIENCE).all():
                    break
    with torch.no_grad():
        for key, value in model.named_parameters():
            value.copy_(best_state[key])
        tx, ty = x[test.to(device)], y[test.to(device)]
        logits = model(tx)
        bce = balanced(F.binary_cross_entropy_with_logits(
            logits, ty[:, None].expand_as(logits), reduction="none"), ty)
        accuracy = balanced(((logits > 0) == ty[:, None]).float(), ty)
    return {"bce": bce.reshape(MLP_REPEATS, n_cases).mean(0).cpu(),
            "accuracy": accuracy.reshape(MLP_REPEATS, n_cases).mean(0).cpu(),
            "per_mlp_bce": bce.reshape(MLP_REPEATS, n_cases).cpu(),
            "per_mlp_accuracy": accuracy.reshape(MLP_REPEATS, n_cases).cpu(),
            "best_val_bce": best_val.reshape(MLP_REPEATS, n_cases).cpu(),
            "steps": step, "mlp_repeats": MLP_REPEATS,
            "history": torch.tensor(history)}


def run(pair_index: int, replicate: int, out: Path, device: torch.device) -> Path:
    pair = PAIRS[pair_index]
    folder = out / "evaluation_v2" / f"pair{pair_index:02d}"
    destination = folder / f"rep{replicate}.pt"
    if destination.exists():
        return destination
    data = torch.load(out / "agreement" / "_".join(pair) /
                      f"rep{replicate}.pt", map_location="cpu", weights_only=True)
    names, masks = candidates(data, seed=20260926 + pair_index * 100 + replicate)
    by_task = {}
    for pattern in pair:
        by_task[pattern] = fit_and_evaluate(
            masks, pattern, seed=20260927 + pair_index * 1000 +
            replicate * 100 + int(pattern, 2), device=device)
    payload = {"pair": list(pair), "replicate": replicate, "names": names,
               "masks": masks, "iou": torch.tensor([gold_iou(m) for m in masks]),
               "tasks": by_task}
    folder.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(destination)
    print(f"eval pair={pair_index} rep={replicate}: "
          f"final consensus IoU={float(payload['iou'][names.index('final_consensus')]):.3f}",
          flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    for index in range(args.shard, len(PAIRS), args.shards):
        for replicate in range(4):
            run(index, replicate, args.out, torch.device(args.device))


if __name__ == "__main__":
    main()
