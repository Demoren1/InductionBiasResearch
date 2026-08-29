"""Train a masked-MLP candidate bank for one meta-training motif-pair task."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data.generate import configure_compute_device  # noqa: E402
from models.mlp import BatchedMaskedMLP, generate_masks, get_train_batch  # noqa: E402


def _config(name: str, default):
    return getattr(config, name, default)


def own_mlp_slice(n_total: int, worker_id: int, n_workers: int) -> tuple[int, int]:
    """Return the non-overlapping global candidate slice owned by a worker."""
    if n_workers < 1 or not 0 <= worker_id < n_workers:
        raise ValueError("worker_id must be in [0, n_workers)")
    base, remainder = divmod(n_total, n_workers)
    start = worker_id * base + min(worker_id, remainder)
    return start, base + int(worker_id < remainder)


def load_train_tasks(split_path: Path) -> list[str]:
    """Read and validate the explicit meta-train task list from ``split.json``."""
    payload = json.loads(split_path.read_text())
    tasks = payload.get("train_tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(t, str) for t in tasks):
        raise ValueError(f"{split_path} must contain a nonempty string list 'train_tasks'")
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"{split_path} contains duplicate train tasks")
    return tasks


def split_provenance(split_path: Path, train_tasks: list[str]) -> dict:
    raw = split_path.read_bytes()
    return {
        "split_path": str(split_path.resolve()),
        "split_sha256": hashlib.sha256(raw).hexdigest(),
        "split_train_tasks": train_tasks,
    }


def train_round(model: BatchedMaskedMLP, task: str, *, steps: int,
                batch_size: int, lr: float, seed: int,
                x_val: torch.Tensor, y_val: torch.Tensor) -> tuple[float, float]:
    """Optimize one in-memory bank and return mean validation metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = model.w1.device
    for step in range(steps):
        x, y = get_train_batch(task, batch_size, seed + step, device=device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.unsqueeze(1).expand_as(logits))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        every = _config("EVAL_EVERY", 200)
        if step and step % every == 0:
            val_loss = model.val_loss(x_val, y_val, _config("VAL_BATCH_SIZE", 256)).mean().item()
            val_acc = model.val_acc(x_val, y_val, _config("VAL_BATCH_SIZE", 256)).mean().item()
            print(f"  [task={task}] step {step}/{steps} train_bce={loss.item():.5f} "
                  f"val_bce={val_loss:.5f} val_acc={val_acc:.4f}", flush=True)
    val_loss = model.val_loss(x_val, y_val, _config("VAL_BATCH_SIZE", 256)).mean().item()
    val_acc = model.val_acc(x_val, y_val, _config("VAL_BATCH_SIZE", 256)).mean().item()
    return val_loss, val_acc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="task id listed in split.json train_tasks")
    parser.add_argument("--split_json", type=Path, required=True)
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="logical worker id, not physical CUDA id")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--n_mlps", type=int,
                        default=_config("N_MLPS_PER_TASK", 512),
                        help="total candidate MLPs for this task")
    parser.add_argument("--mlps_per_gpu", type=int, default=_config("MLPS_PER_GPU", 512))
    parser.add_argument("--train_steps", type=int, default=_config("TRAIN_STEPS", 1000))
    parser.add_argument("--batch_size", type=int, default=_config("TRAIN_BATCH_SIZE", 128))
    parser.add_argument("--lr", type=float, default=_config("LR", 1e-3))
    parser.add_argument("--data_dir", type=Path, default=_config("DATA_DIR", "outputs/data"),
                        help="directory containing this split's val_<task>.pt artifacts")
    parser.add_argument("--ckpt_root", type=Path, default=_config("CKPT_DIR", "outputs/checkpoints"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    split_path = args.split_json.resolve()
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    train_tasks = load_train_tasks(split_path)
    if args.task not in train_tasks:
        raise ValueError(f"refusing to train task {args.task!r}: it is not in {split_path}'s train_tasks")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for candidate-bank training")
    configure_compute_device("cuda")

    data_dir = args.data_dir.resolve()
    val_path = data_dir / f"val_{args.task}.pt"
    if not val_path.is_file():
        raise FileNotFoundError(f"missing {val_path}; run scripts/01_generate_data.sh first")
    val = torch.load(val_path, weights_only=True)
    expected_data_provenance = split_provenance(split_path, train_tasks)
    for key in ("split_sha256", "split_train_tasks"):
        if val.get(key) != expected_data_provenance[key]:
            raise ValueError(
                f"validation artifact provenance mismatch for {key}: "
                f"artifact={val.get(key)!r}, split={expected_data_provenance[key]!r}"
            )
    device = torch.device("cuda:0")
    x_val, y_val = val["x"].to(device), val["y"].to(device)

    n_total = args.n_mlps
    if n_total < 1:
        raise ValueError("--n_mlps must be positive")
    start, count = own_mlp_slice(n_total, args.gpu_id, args.num_gpus)
    task_dir = args.ckpt_root / f"task_{args.task}"
    task_dir.mkdir(parents=True, exist_ok=True)
    print(f"[worker {args.gpu_id}] task={args.task} owns MLPs [{start}, {start + count})", flush=True)

    # Every global candidate has a deterministic mask irrespective of the GPU layout.
    task_seed = int.from_bytes(hashlib.sha256(args.task.encode()).digest()[:8], "little")
    all_masks = generate_masks(n_total, _config("SEQ_LEN", 16), _config("H", 16),
                               _config("P", 0.375), task_seed, device=device)
    provenance = expected_data_provenance
    provenance["data_dir"] = str(data_dir)
    started = time.time()
    for round_index, offset in enumerate(range(0, count, args.mlps_per_gpu)):
        n_here = min(args.mlps_per_gpu, count - offset)
        global_start = start + offset
        seed = task_seed + global_start * 7
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = BatchedMaskedMLP(n_here, _config("SEQ_LEN", 16), _config("H", 16)).to(device)
        masks = all_masks[global_start:global_start + n_here]
        model.load_masks(masks)
        round_started = time.time()
        mean_loss, mean_acc = train_round(model, args.task, steps=args.train_steps,
                                          batch_size=args.batch_size, lr=args.lr, seed=seed,
                                          x_val=x_val, y_val=y_val)
        checkpoint = {
            "task": args.task,
            "global_idx": torch.arange(global_start, global_start + n_here),
            "params": BatchedMaskedMLP.state_as_dict(model.w1, model.b1, model.w2, model.b2),
            "masks": masks.cpu(),
            "val_loss": model.val_loss(x_val, y_val, _config("VAL_BATCH_SIZE", 256)).cpu(),
            "val_acc": model.val_acc(x_val, y_val, _config("VAL_BATCH_SIZE", 256)).cpu(),
            "n_mlps_per_task": n_total,
            "mask_probability": _config("P", 0.375),
            "train_steps": args.train_steps,
            **provenance,
        }
        out = task_dir / f"gpu{args.gpu_id}_round{round_index:03d}.pt"
        torch.save(checkpoint, out)
        print(f"[worker {args.gpu_id}] saved {out} in {time.time() - round_started:.1f}s "
              f"(mean val_bce={mean_loss:.5f}, val_acc={mean_acc:.4f})", flush=True)
    print(f"[worker {args.gpu_id}] task={args.task} completed in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
