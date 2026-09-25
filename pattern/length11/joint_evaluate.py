"""Evaluate task-regularized agreement against the same random controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pattern.length11.agreement import PAIRS, topk
from pattern.length11.evaluate import fit_and_evaluate, gold_iou, gold_mask
from pattern.length11.settings import EDGES, HIDDEN, ROOT, SEQ_LEN
from pattern.length_interp.bank import random_exact_k_masks


def run(pair_index: int, replicate: int, device: torch.device,
        out: Path = ROOT, *, variant: str = "joint") -> Path:
    folder = out / f"evaluation_{variant}" / f"pair{pair_index:02d}"
    destination = folder / f"rep{replicate}.pt"
    if destination.exists():
        return destination
    joint = torch.load(out / variant / f"pair{pair_index:02d}_rep{replicate}.pt",
                       map_location="cpu", weights_only=True)
    pair = PAIRS[pair_index]
    plain = torch.load(out / "agreement" / "_".join(pair) / f"rep{replicate}.pt",
                       map_location="cpu", weights_only=True)
    j = joint["chosen_start"]
    masks = [*joint["masks"][:, j], topk(joint["logits"][:, j].mean(0)),
             topk(plain["final_logits"][:, plain["chosen_start"]].mean(0))]
    names = ["joint_vae0", "joint_vae1", "joint_consensus", "plain_consensus"]
    random = random_exact_k_masks(16, seq_len=SEQ_LEN, hidden=HIDDEN,
                                  k_active=EDGES,
                                  seed=20260926 + pair_index * 100 + replicate)
    masks.extend(random)
    names.extend(f"random_{i:02d}" for i in range(16))
    masks.append(gold_mask())
    names.append("analytic")
    masks = torch.stack(masks)
    by_task = {}
    for pattern in pair:
        by_task[pattern] = fit_and_evaluate(
            masks, pattern, seed=20260927 + pair_index * 1000 +
            replicate * 100 + int(pattern, 2), device=device)
    payload = {"pair": list(pair), "replicate": replicate,
               "names": names, "masks": masks,
               "iou": torch.tensor([gold_iou(mask) for mask in masks]),
               "tasks": by_task}
    folder.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(destination)
    print(f"joint eval pair={pair_index} rep={replicate}: "
          f"IoU={float(payload['iou'][2]):.3f}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=int, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--variant", default="joint")
    args = parser.parse_args()
    run(args.pair, args.replicate, torch.device(args.device), args.out,
        variant=args.variant)


if __name__ == "__main__":
    main()
