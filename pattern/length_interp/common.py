import hashlib
import json
from functools import lru_cache
from pathlib import Path
import random

import torch

from meta_pattern.data import build_task_splits
from .config import Config


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def source_hashes():
    root = Path(__file__).resolve().parent
    result = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob("*.py"))}
    data = root.parents[1] / "meta_pattern/data.py"
    result["../../meta_pattern/data.py"] = hashlib.sha256(data.read_bytes()).hexdigest()
    return result


def load_protocol(out):
    return Config(**json.loads((Path(out) / "protocol.json").read_text())["config"])


@lru_cache(maxsize=16)
def patterns_by_split(c):
    original = build_task_splits(lengths=c.train_lengths, seed=c.task_split_seed)
    result = {}
    for split in ("train", "val"):
        selected = []
        for k in c.train_lengths:
            patterns = [t.pattern for t in original[split] if t.length == k]
            if c.task_limit and len(patterns) > c.task_limit:
                patterns = random.Random(seed_for("task_cap", c.task_split_seed, split, k)).sample(patterns, c.task_limit)
            selected.extend(sorted(patterns))
        result[split] = selected
    result["test"] = []
    for k in c.heldout_lengths:
        patterns = [format(i, f"0{k}b") for i in range(2**k)]
        if c.eval_task_limit and len(patterns) > c.eval_task_limit:
            patterns = random.Random(seed_for("eval_cap", c.task_split_seed, k)).sample(patterns, c.eval_task_limit)
        result["test"].extend(sorted(patterns))
    return result


def bank_path(out, pattern):
    return Path(out) / "bank" / f"k{len(pattern)}" / f"pattern_{pattern}.pt"


def exact_k(length, c):
    return c.hidden * length


def ideal_mask(length, c):
    mask = torch.zeros(c.seq_len, c.hidden)
    for h in range(c.hidden):
        start = h % (c.seq_len - length + 1)
        mask[start:start + length, h] = 1
    return mask
