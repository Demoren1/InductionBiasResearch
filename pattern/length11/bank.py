"""Build exact-32 importance-map banks on length-11 pattern tasks."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from pattern.length_interp.bank import random_exact_k_masks, select_top_indices
from pattern.length_interp.mlp import BatchedMaskedMLP
from pattern.length11.settings import BankConfig, EDGES, HIDDEN, PATTERNS, ROOT, SEQ_LEN


def all_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    bits = ((torch.arange(1 << SEQ_LEN)[:, None] >>
             torch.arange(SEQ_LEN - 1, -1, -1)) & 1).float()
    return 2 * bits - 1, bits


def labels(bits: torch.Tensor, pattern: str) -> torch.Tensor:
    target = torch.tensor([int(value) for value in pattern], dtype=bits.dtype)
    return (bits.unfold(1, len(pattern), 1) == target).all(dim=-1).any(dim=-1).float()


def hash_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def run_pattern(pattern: str, out: Path, device: torch.device,
                config: BankConfig = BankConfig()) -> Path:
    if pattern not in PATTERNS:
        raise ValueError(f"invalid pattern: {pattern}")
    destination = out / "bank" / f"pattern_{pattern}.pt"
    if destination.exists():
        payload = torch.load(destination, map_location="cpu", weights_only=True)
        if (payload["protocol"] != config.__dict__ or payload["pattern"] != pattern or
                payload["importance"].shape !=
                (math.ceil(config.bank_mlps * config.top_fraction), SEQ_LEN, HIDDEN)):
            raise ValueError(f"bank protocol mismatch: {destination}")
        return destination
    x, bits = all_inputs()
    y = labels(bits, pattern)
    generator = torch.Generator().manual_seed(config.input_split_seed)
    indices = torch.randperm(len(x), generator=generator)
    train_ids = indices[:round(.8 * len(x))]
    query_ids = indices[round(.8 * len(x)):]
    positives = train_ids[y[train_ids] == 1]
    negatives = train_ids[y[train_ids] == 0]
    if min(len(positives), len(negatives)) == 0:
        raise ValueError("train partition has only one class")
    seed = config.seed * 100 + int(pattern, 2)
    masks = random_exact_k_masks(config.bank_mlps, seq_len=SEQ_LEN, hidden=HIDDEN,
                                 k_active=EDGES, seed=seed + 1)
    model = BatchedMaskedMLP(masks, seed=seed + 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.bank_lr)
    sampler = torch.Generator().manual_seed(seed + 3)
    train_x, train_y = x.to(device), y.to(device)
    steps = tqdm(range(config.bank_steps), desc=f"bank {pattern}", unit="step",
                 mininterval=2)
    for step in steps:
        pos = positives[torch.randint(len(positives), (config.bank_batch // 2,),
                                      generator=sampler)]
        neg = negatives[torch.randint(len(negatives),
                                      (config.bank_batch - len(pos),),
                                      generator=sampler)]
        chosen = torch.cat((pos, neg)).to(device)
        xb, yb = train_x.index_select(0, chosen), train_y.index_select(0, chosen)
        logits = model(xb)
        per_model = F.binary_cross_entropy_with_logits(
            logits, yb[:, None].expand_as(logits), reduction="none").mean(dim=0)
        optimizer.zero_grad(set_to_none=True)
        per_model.sum().backward()
        optimizer.step()
        if step % 500 == 0:
            steps.set_postfix(train=f"{float(per_model.mean()):.3f}")
    with torch.no_grad():
        qx, qy = train_x[query_ids.to(device)], train_y[query_ids.to(device)]
        logits = model(qx)
        losses = F.binary_cross_entropy_with_logits(
            logits, qy[:, None].expand_as(logits), reduction="none")
        positive = qy == 1
        bce = .5 * (losses[positive].mean(dim=0) + losses[~positive].mean(dim=0))
        correct = (logits > 0) == qy[:, None]
        accuracy = .5 * (correct[positive].float().mean(dim=0) +
                         correct[~positive].float().mean(dim=0))
        selected = select_top_indices(bce, config.top_fraction)
        weights = model.w1.detach().cpu().index_select(0, selected)
        selected_masks = masks.index_select(0, selected)
        importance = (weights * selected_masks).abs()
        importance /= importance.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    payload = {
        "protocol": dict(config.__dict__), "pattern": pattern,
        "shape": [SEQ_LEN, HIDDEN], "edges": EDGES,
        "train_ids_sha256": hash_tensor(train_ids),
        "query_ids_sha256": hash_tensor(query_ids),
        "n_train_inputs": len(train_ids), "n_query_inputs": len(query_ids),
        "n_bank": config.bank_mlps, "n_selected": len(selected),
        "all_query_bce": bce.cpu(), "all_query_accuracy": accuracy.cpu(),
        "selected_indices": selected, "selected_masks": selected_masks,
        "importance": importance.cpu(),
        "selected_query_bce": bce.cpu().index_select(0, selected),
        "selected_query_accuracy": accuracy.cpu().index_select(0, selected),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    print(f"bank {pattern}: selected BCE={float(payload['selected_query_bce'].mean()):.4f} "
          f"accuracy={float(payload['selected_query_accuracy'].mean()):.4f}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--bank-mlps", type=int, default=BankConfig.bank_mlps)
    parser.add_argument("--pattern", choices=PATTERNS)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if (args.pattern is None) == (args.shard is None):
        raise ValueError("specify exactly one of --pattern or --shard")
    if args.shard is not None and not 0 <= args.shard < args.shards:
        raise ValueError("invalid shard")
    torch.set_num_threads(2)
    patterns = (args.pattern,) if args.pattern else PATTERNS[args.shard::args.shards]
    for pattern in patterns:
        run_pattern(pattern, args.out, torch.device(args.device),
                    BankConfig(bank_mlps=args.bank_mlps))


if __name__ == "__main__":
    main()
