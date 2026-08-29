"""Extract normalized raw ``|W1 * mask|`` importance maps from candidate banks."""

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
    tasks = json.loads(split_json.read_text()).get("train_tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(t, str) for t in tasks):
        raise ValueError(f"{split_json} must contain a nonempty string list 'train_tasks'")
    return tasks


def importance_from(w1: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Return per-candidate [0,1] maps from raw masked first-layer weights."""
    if w1.shape != masks.shape:
        raise ValueError(f"w1 and masks must have equal shapes; got {w1.shape} and {masks.shape}")
    raw_abs = (w1 * masks.to(dtype=w1.dtype)).abs()
    maximum = raw_abs.flatten(start_dim=1).amax(dim=1, keepdim=True).clamp_min(1e-9)
    return raw_abs / maximum.view(-1, 1, 1)


def load_rounds(task: str, device: torch.device | str = "cpu",
                ckpt_root: Path | None = None,
                expected_provenance: dict | None = None) -> dict:
    directory = Path(ckpt_root or config.CKPT_DIR) / f"task_{task}"
    files = sorted(directory.glob("gpu*_round*.pt"))
    if not files:
        raise FileNotFoundError(f"no round checkpoints found in {directory}")
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
    if indices.numel() != torch.unique(indices).numel():
        raise ValueError(f"duplicate global candidate indexes in {directory}")
    order = torch.argsort(indices)
    return {
        "global_idx": indices[order],
        "w1": torch.cat([head["params"]["w1"] for head in heads])[order],
        "masks": torch.cat([head["masks"] for head in heads])[order],
        "val_loss": torch.cat([head["val_loss"] for head in heads])[order],
        "val_acc": torch.cat([head["val_acc"] for head in heads])[order],
        "split_path": heads[0].get("split_path"),
        "split_sha256": heads[0].get("split_sha256"),
        "split_train_tasks": heads[0].get("split_train_tasks"),
    }


def _cpu_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tensors(item) for key, item in value.items()}
    return value


def save_importance(task: str, device: torch.device | str = "cpu",
                    ckpt_root: Path | None = None,
                    expected_provenance: dict | None = None) -> Path:
    source = load_rounds(task, device=device, ckpt_root=ckpt_root,
                         expected_provenance=expected_provenance)
    importance = importance_from(source.pop("w1"), source.pop("masks"))
    out = Path(ckpt_root or config.CKPT_DIR) / f"task_{task}" / "importance.pt"
    payload = {"task": task, "n_mlps": importance.size(0),
               "importance": importance, **source}
    torch.save(_cpu_tensors(payload), out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_json", type=Path, required=True)
    parser.add_argument("--task", help="one train task; default: every task in split.json")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--ckpt_root", type=Path, default=config.CKPT_DIR)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for importance extraction")
    device = torch.device(args.device)
    expected = read_split_provenance(args.split_json)
    allowed = expected["split_train_tasks"]
    tasks = [args.task] if args.task else allowed
    if any(task not in allowed for task in tasks):
        raise ValueError("importance extraction is restricted to split.json train_tasks")
    for task in tasks:
        out = save_importance(task, device=device, ckpt_root=args.ckpt_root,
                              expected_provenance=expected)
        print(f"[importance] task={task}: saved {out}")


if __name__ == "__main__":
    main()
