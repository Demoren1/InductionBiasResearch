"""Reproducible episodes, model construction and artifact helpers."""
import hashlib
import json
import os
import random
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import Config
from .data import PatternTask, build_task_splits, sample_dataset
from .models import FullUGenerator, LearnedUTable, RandomU, adapt_v, forward_with_u, ideal_u


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def source_hashes():
    root = Path(__file__).resolve().parent
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py")) if "outputs" not in p.relative_to(root).parts}


def setup(seed, device):
    torch.set_num_threads(2)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run outside the sandbox or use --device cpu")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class IdealModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer("device_anchor", torch.empty(0))

    def forward(self, length):
        c = self.config
        return tuple(u.to(self.device_anchor.device) for u in ideal_u(
            length, seq_len=c.seq_len, hidden=c.hidden, rank1=c.rank1, rank2=c.rank2))


def make_model(c, device):
    kwargs = dict(seq_len=c.seq_len, hidden=c.hidden, rank1=c.rank1, rank2=c.rank2)
    if c.method == "generator":
        model = FullUGenerator(**kwargs, width=c.width, depth=c.generator_depth,
                               condition_length=c.condition_length,
                               length_min=min(c.lengths), length_max=max(c.lengths))
    elif c.method == "table":
        model = LearnedUTable(**kwargs, length_min=min(c.lengths), length_max=max(c.lengths), seed=c.seed)
    elif c.method == "random":
        model = RandomU(**kwargs, length_min=min(c.lengths), length_max=max(c.lengths), seed=c.seed)
    else:
        model = IdealModel(c)
    return model.to(device)


@lru_cache(maxsize=32)
def task_splits(c):
    splits = build_task_splits(lengths=c.lengths, seed=c.task_split_seed)
    if c.all_unseen_patterns:
        splits["train"] = [t for t in splits["train"] if t.length in c.train_lengths]
        splits["val"] = [t for t in splits["val"] if t.length in c.train_lengths]
        splits["test"] = [t for t in splits["test"] if t.length in c.train_lengths]
        for k in c.lengths:
            if k not in c.train_lengths:
                splits["test"].extend(PatternTask(format(i, f"0{k}b")) for i in range(2**k))
    return splits


def dataset(c, task, n, seed, split, device, balanced=True):
    data = sample_dataset(task, n, seed=seed, split=split, split_seed=c.input_split_seed,
                          balanced=balanced, seq_len=c.seq_len)
    return {k: v.to(device) if k in ("x", "y") else v for k, v in data.items()}


def metrics(logits, labels):
    prediction = logits > 0
    positive, negative = labels.bool(), ~labels.bool()
    result = {"bce": F.binary_cross_entropy_with_logits(logits, labels).item(),
              "accuracy": (prediction == positive).float().mean().item(),
              "positive_fraction": labels.mean().item()}
    for name, group in (("tpr", positive), ("tnr", negative)):
        result[name] = ((prediction[group] == positive[group]).float().mean().item()
                        if group.any() else None)
    return result


def validation(c, model, device):
    """Only held-out validation PATTERNS at meta-train LENGTHS; no final test data."""
    rows = []
    for k in c.train_lengths:
        tasks = [t for t in task_splits(c)["val"] if t.length == k]
        if c.val_tasks_per_length and c.val_tasks_per_length < len(tasks):
            tasks = random.Random(seed_for("validation_subset", c.task_split_seed, k)).sample(tasks, c.val_tasks_per_length)
        for task in tasks:
            base = seed_for("validation", c.seed if c.data_seed is None else c.data_seed, task.pattern)
            support = dataset(c, task, c.support_size, seed_for(base, "support"), "support", device)
            query = dataset(c, task, c.query_size, seed_for(base, "query"), "query", device)
            u = tuple(t.detach() for t in model(k))
            v = adapt_v(u, support["x"], support["y"], steps=c.inner_steps, lr=c.inner_lr,
                        seed=seed_for(base, "v"), create_graph=False, batch_size=c.batch_size,
                        optimizer=c.inner_optimizer, init_scale=c.init_scale)
            with torch.no_grad():
                score = metrics(forward_with_u(query["x"], u, v, seq_len=c.seq_len, hidden=c.hidden), query["y"])
            rows.append({"pattern": task.pattern, "length": k, **score})
    # Equal length weights, regardless of how many validation patterns each length has.
    score = sum(sum(r["bce"] for r in rows if r["length"] == k) /
                sum(r["length"] == k for r in rows) for k in c.train_lengths) / len(c.train_lengths)
    return score, rows
