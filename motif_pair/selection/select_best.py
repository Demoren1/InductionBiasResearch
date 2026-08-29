"""Select the lowest-validation-BCE candidate masks for meta-train tasks only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from models.cvae import read_split_provenance, verify_split_provenance  # noqa: E402


def train_tasks_from_split(split_json: Path) -> list[str]:
    payload = json.loads(split_json.read_text())
    tasks = payload.get("train_tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(t, str) for t in tasks):
        raise ValueError(f"{split_json} must contain a nonempty string list 'train_tasks'")
    return tasks


def load_all_checkpoints(task: str, device: torch.device | str = "cpu",
                         ckpt_root: Path | None = None,
                         expected_provenance: dict | None = None) -> dict:
    """Load, validate, and concatenate every candidate-bank shard for ``task``."""
    directory = Path(ckpt_root or config.CKPT_DIR) / f"task_{task}"
    files = sorted(directory.glob("gpu*_round*.pt"))
    if not files:
        raise FileNotFoundError(f"no candidate checkpoints found in {directory}")
    heads = [torch.load(path, weights_only=True, map_location=device) for path in files]
    if any(head.get("task") != task for head in heads):
        raise ValueError(f"checkpoint task provenance mismatch in {directory}")
    provenance_keys = ("split_path", "split_sha256", "split_train_tasks")
    for key in provenance_keys:
        values = [head.get(key) for head in heads]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"inconsistent {key} across candidate shards for {task}")
    if expected_provenance is not None:
        for path, head in zip(files, heads):
            verify_split_provenance(head, expected_provenance,
                                    label=f"candidate shard {path}")
    indices = torch.cat([head["global_idx"] for head in heads])
    order = torch.argsort(indices)
    if indices.numel() != torch.unique(indices).numel():
        raise ValueError(f"duplicate global candidate indexes in {directory}")
    params = {key: torch.cat([head["params"][key] for head in heads])[order]
              for key in heads[0]["params"]}
    return {
        "task": task,
        "global_idx": indices[order],
        "val_loss": torch.cat([head["val_loss"] for head in heads])[order],
        "val_acc": torch.cat([head["val_acc"] for head in heads])[order],
        "masks": torch.cat([head["masks"] for head in heads])[order],
        "params": params,
        "n_mlps_per_task": heads[0].get("n_mlps_per_task"),
        "mask_probability": heads[0].get("mask_probability"),
        **{key: heads[0].get(key) for key in provenance_keys},
    }


def select_best(all_candidates: dict, top_fraction: float) -> dict:
    """Keep the lowest-BCE fraction of a task's independently trained MLPs."""
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    n_candidates = all_candidates["val_loss"].numel()
    n_selected = max(1, int(round(n_candidates * top_fraction)))
    chosen = torch.argsort(all_candidates["val_loss"])[:n_selected]
    selected = {
        key: all_candidates[key][chosen]
        for key in ("global_idx", "val_loss", "val_acc", "masks")
    }
    selected["params"] = {key: value[chosen] for key, value in all_candidates["params"].items()}
    selected.update({
        "task": all_candidates["task"],
        "top_fraction": top_fraction,
        "n_candidates": n_candidates,
        "n_selected": n_selected,
        "n_mlps_per_task": all_candidates["n_mlps_per_task"],
        "mask_probability": all_candidates["mask_probability"],
        "split_path": all_candidates["split_path"],
        "split_sha256": all_candidates["split_sha256"],
        "split_train_tasks": all_candidates["split_train_tasks"],
    })
    return selected


def _cpu_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tensors(item) for key, item in value.items()}
    return value


def select(task: str, top_fraction: float, device: torch.device | str = "cpu",
           ckpt_root: Path | None = None,
           expected_provenance: dict | None = None) -> Path:
    all_candidates = load_all_checkpoints(task, device=device, ckpt_root=ckpt_root,
                                          expected_provenance=expected_provenance)
    selected = select_best(all_candidates, top_fraction)
    directory = Path(ckpt_root or config.CKPT_DIR) / f"task_{task}"
    out = directory / "best10pct.pt"
    torch.save(_cpu_tensors(selected), out)
    losses, acc = selected["val_loss"], selected["val_acc"]
    (directory / "best10pct_summary.txt").write_text(
        f"task={task}\n"
        f"total_mlps={selected['n_candidates']}\n"
        f"n_selected={selected['n_selected']} (top {top_fraction:.0%})\n"
        f"val_loss (BCE) min={losses.min():.6f} p50={losses.median():.6f} max={losses.max():.6f}\n"
        f"val_acc min={acc.min():.6f} p50={acc.median():.6f} max={acc.max():.6f}\n"
    )
    print(f"[select] task={task}: kept {selected['n_selected']}/{selected['n_candidates']} "
          f"(BCE {losses.min():.6f}..{losses.max():.6f}) -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_json", type=Path, required=True)
    parser.add_argument("--task", help="one train task; default: every task in split.json")
    parser.add_argument("--top_fraction", type=float, default=getattr(config, "TOP_FRACTION", 0.1))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--ckpt_root", type=Path, default=config.CKPT_DIR)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for candidate selection")
    device = torch.device(args.device)
    expected = read_split_provenance(args.split_json)
    allowed = expected["split_train_tasks"]
    tasks = [args.task] if args.task else allowed
    if any(task not in allowed for task in tasks):
        raise ValueError("selection is restricted to split.json train_tasks")
    for task in tasks:
        select(task, args.top_fraction, device=device, ckpt_root=args.ckpt_root,
               expected_provenance=expected)


if __name__ == "__main__":
    main()
