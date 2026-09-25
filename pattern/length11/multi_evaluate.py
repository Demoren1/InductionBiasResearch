"""Evaluate 1–10 VAE searches with random, initial, and analytic controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pattern.length11.agreement import topk
from pattern.length11.evaluate import fit_and_evaluate, gold_iou, gold_mask
from pattern.length11.settings import EDGES, HIDDEN, ROOT, SEQ_LEN
from pattern.length_interp.bank import random_exact_k_masks


def run(patterns: tuple[str, ...], replicate: int, out: Path, device: torch.device) -> Path:
    n = len(patterns)
    destination = out / "evaluation_multi" / f"n{n}" / f"rep{replicate}.pt"
    if destination.exists():
        previous = torch.load(destination, map_location="cpu", weights_only=True)
        if previous["patterns"] != list(patterns) or previous["replicate"] != replicate:
            raise ValueError(f"evaluation protocol mismatch: {destination}")
        return destination
    joint = torch.load(out / "joint_multi" / f"n{n}" / f"rep{replicate}.pt",
                       map_location="cpu", weights_only=True)
    plain = torch.load(out / "agreement" / "_".join(patterns) /
                       f"rep{replicate}.pt", map_location="cpu", weights_only=True)
    if joint["patterns"] != list(patterns) or plain["patterns"] != list(patterns):
        raise ValueError("search outputs have different pattern order")
    j = joint["chosen_start"]
    k = plain["chosen_start"]
    names = [f"joint_vae{i}" for i in range(n)]
    masks = [*joint["masks"][:, j]]
    masks.append(topk(joint["logits"][:, j].mean(0)))
    names.append("joint_consensus")
    masks.append(topk(plain["final_logits"][:, k].mean(0)))
    names.append("plain_consensus")
    masks.append(topk(plain["initial_logits"][:, k].mean(0)))
    names.append("initial_consensus")
    random = random_exact_k_masks(16, seq_len=SEQ_LEN, hidden=HIDDEN,
                                  k_active=EDGES, seed=20261001 + replicate)
    masks.extend(random)
    names.extend(f"random_{i:02d}" for i in range(16))
    masks.append(gold_mask())
    names.append("analytic")
    masks = torch.stack(masks)
    by_task = {}
    for pattern in patterns:
        by_task[pattern] = fit_and_evaluate(
            masks, pattern, seed=20261002 + 100 * replicate + int(pattern, 2),
            device=device)
    payload = {
        "patterns": list(patterns), "replicate": replicate,
        "names": names, "masks": masks,
        "iou": torch.tensor([gold_iou(mask) for mask in masks]),
        "tasks": by_task,
        "settings": {"random_masks": 16, "mlp_repeats": 4,
                     "max_mlp_steps": 20000,
                     "task_score": "balanced accuracy and BCE on the fixed input query split"},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(destination)
    print(f"evaluate n{n} rep{replicate}: joint IoU="
          f"{float(payload['iou'][names.index('joint_consensus')]):.3f}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patterns", nargs="+", required=True)
    parser.add_argument("--replicate", type=int, choices=range(4), required=True)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(tuple(args.patterns), args.replicate, args.out, torch.device(args.device))


if __name__ == "__main__":
    main()
