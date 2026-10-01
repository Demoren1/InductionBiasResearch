"""Align hidden-unit columns of selected MLP importance maps before VAE training."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from pattern.mnist8m_importance_bce import save


def topk_iou(maps: torch.Tensor, template: torch.Tensor, k: int) -> float:
    target = torch.zeros_like(maps, dtype=torch.bool)
    target.scatter_(1, maps.topk(k, dim=1).indices, True)
    candidate = torch.zeros_like(template, dtype=torch.bool)
    candidate.scatter_(0, template.topk(k).indices, True)
    intersection = (target & candidate).sum(1)
    return float((intersection / (2 * k - intersection)).float().mean())


def align_one(mapping: torch.Tensor, template: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = mapping.shape[1]
    target = torch.nn.functional.normalize(template, p=2, dim=0, eps=1e-8)
    source = torch.nn.functional.normalize(mapping, p=2, dim=0, eps=1e-8)
    similarity = (target.T @ source).numpy()
    rows, columns = linear_sum_assignment(-similarity)
    order = np.empty(h, dtype=np.int64)
    order[rows] = columns
    order = torch.from_numpy(order)
    return mapping[:, order], order


def align_bank(bank: dict, task: int, reference: torch.Tensor,
               reference_index: int, reference_task: int) -> tuple[dict, dict]:
    maps = bank["importance"]
    masks = bank["masks"]
    count, pixels, hidden = maps.shape
    generator = torch.Generator().manual_seed(3130 + task)
    split = torch.randperm(count, generator=generator)
    n_val = max(32, round(.15 * count))
    train_indices, held_indices = split[n_val:], split[:n_val]
    template = reference
    aligned_maps = torch.empty_like(maps)
    aligned_masks = torch.empty_like(masks)
    orders = torch.empty(count, hidden, dtype=torch.int64)
    for index in tqdm(range(count), desc=f"task {task} finalize", mininterval=2):
        aligned, order = align_one(maps[index], template)
        aligned_maps[index] = aligned
        aligned_masks[index] = masks[index][:, order]
        orders[index] = order
    k = int(masks[0].sum())
    raw_train_mean = maps[train_indices].mean(0).flatten()
    aligned_train_mean = aligned_maps[train_indices].mean(0).flatten()
    raw_iou = topk_iou(maps[held_indices].flatten(1), raw_train_mean, k)
    aligned_iou = topk_iou(aligned_maps[held_indices].flatten(1),
                           aligned_train_mean, k)
    prepared = dict(bank)
    prepared["importance"] = aligned_maps
    prepared["masks"] = aligned_masks
    prepared["column_orders"] = orders
    prepared["settings"] = {**bank["settings"],
                            "alignment": "cosine Hungarian to one shared VAE train map",
                            "alignment_reference_index": reference_index,
                            "alignment_reference_task": reference_task,
                            "alignment_template_split": "same as VAE train split"}
    diagnostics = {"task": task, "digit": bank["digit"], "density": k / (pixels * hidden),
                   "reference_index": reference_index,
                   "raw_heldout_mean_iou": raw_iou,
                   "aligned_heldout_mean_iou": aligned_iou,
                   "changed_columns_fraction": float((orders != torch.arange(hidden)).float().mean())}
    return prepared, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-task-reference", action="store_true",
                        help="Use a separate anchor for each task (diagnostic only)")
    parser.add_argument("--reference-task", type=int, choices=(0, 1), default=0)
    parser.add_argument("--reference-index", type=int,
                        help="Index of a reference map in the VAE training split")
    args = parser.parse_args()
    if args.per_task_reference and args.reference_index is not None:
        parser.error("--reference-index cannot be combined with --per-task-reference")
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("features.pt",):
        target = args.out / name
        if not target.exists():
            temporary = target.with_suffix(".tmp")
            shutil.copy2(args.source / name, temporary)
            temporary.replace(target)
    banks = [torch.load(args.source / f"bank_task{task}.pt",
                        map_location="cpu", weights_only=True)
             for task in (0, 1)]
    train_indices = []
    for task, bank in enumerate(banks):
        count = len(bank["importance"])
        split = torch.randperm(count, generator=torch.Generator().manual_seed(3130 + task))
        n_val = max(32, round(.15 * count))
        train_indices.append(split[n_val:])
    if args.reference_index is not None:
        selected = args.reference_index
        if selected not in train_indices[args.reference_task].tolist():
            parser.error("reference map must be in the VAE training split")
    diagnostics_rows = []
    for task, bank in enumerate(banks):
        reference_task = task if args.per_task_reference else args.reference_task
        reference_index = (int(train_indices[reference_task][0])
                           if args.reference_index is None else args.reference_index)
        reference = banks[reference_task]["importance"][reference_index]
        target_path = args.out / f"bank_task{task}.pt"
        if target_path.exists():
            existing = torch.load(target_path, map_location="cpu", weights_only=True)
            settings = existing["settings"]
            if (settings.get("alignment_reference_task") != reference_task or
                    settings.get("alignment_reference_index") != reference_index):
                raise ValueError(f"existing bank uses another reference: {target_path}")
            train = train_indices[task]
            train_set = set(train.tolist())
            held = train.new_tensor([index for index in range(len(banks[task]["importance"]))
                                     if index not in train_set])
            k = int(existing["masks"][0].sum())
            raw_maps = banks[task]["importance"].flatten(1)
            aligned_maps = existing["importance"].flatten(1)
            diagnostics_rows.append({
                "task": task, "digit": banks[task]["digit"],
                "density": k / raw_maps.shape[1],
                "reference_index": reference_index,
                "raw_heldout_mean_iou": topk_iou(raw_maps[held],
                                                   raw_maps[train].mean(0), k),
                "aligned_heldout_mean_iou": topk_iou(aligned_maps[held],
                                                       aligned_maps[train].mean(0), k),
                "changed_columns_fraction": float((existing["column_orders"] !=
                                                    torch.arange(existing["column_orders"].shape[1])).float().mean())
            })
            print(f"reusing {target_path}", flush=True)
            continue
        prepared, diagnostics = align_bank(bank, task, reference,
                                           reference_index, reference_task)
        save(prepared, target_path)
        diagnostics_rows.append(diagnostics)
        print(diagnostics, flush=True)
    temporary = args.out / "alignment_diagnostics.json.tmp"
    temporary.write_text(json.dumps(diagnostics_rows, indent=2) + "\n")
    temporary.replace(args.out / "alignment_diagnostics.json")


if __name__ == "__main__":
    main()
