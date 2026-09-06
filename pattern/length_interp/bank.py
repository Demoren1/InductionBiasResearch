"""Train and select the per-pattern MLP bank for the interpolation CVAE."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import torch

from meta_pattern.data import PatternTask, sample_dataset

from .mlp import BatchedMaskedMLP, fit_mlp


def _sha256_tensor(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def random_exact_k_masks(
    count: int, *, seq_len: int, hidden: int, k_active: int, seed: int
) -> torch.Tensor:
    """Independent random masks with exactly ``k_active`` entries per model.

    The construction uses a local CPU generator so a shard's masks do not
    depend on GPU number, run ordering, or the process-wide RNG.  Equal
    cardinality to the analytic local mask is the sole architectural control;
    it must not itself supply a locality prior to the CVAE training bank.
    """
    total = seq_len * hidden
    if count < 1 or not 0 <= k_active <= total:
        raise ValueError("invalid count or exact mask cardinality")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    masks = torch.zeros(count, total, dtype=torch.float32)
    for row in range(count):
        masks[row, torch.randperm(total, generator=generator)[:k_active]] = 1.0
    return masks.reshape(count, seq_len, hidden)


def select_top_indices(validation_bce: torch.Tensor, top_fraction: float) -> torch.Tensor:
    """Indices of the lowest-loss models, with a deterministic index tie-break."""
    if validation_bce.ndim != 1 or validation_bce.numel() < 1:
        raise ValueError("validation_bce must be a nonempty vector")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    n_keep = max(1, math.ceil(validation_bce.numel() * top_fraction))
    # stable=True makes equal losses resolve to the original model index.
    return torch.argsort(validation_bce.detach().cpu(), stable=True)[:n_keep]


def _sample_pool(
    task: PatternTask,
    count: int,
    *,
    seed: int,
    split_seed: int,
    seq_len: int,
) -> dict[str, torch.Tensor]:
    """Build a support pool in bounded sampler calls.

    For k=3 a 65k balanced all-at-once rejection sample can spend most of its
    time looking for rare negatives.  Chunks retain the global split guarantee
    and make this failure mode observable and recoverable per chunk.  Duplicate
    support examples, if any, are harmless for the stated SGD objective.
    """
    chunk = min(8192, count)
    pieces: list[dict[str, torch.Tensor]] = []
    total = 0
    ordinal = 0
    while total < count:
        n = min(chunk, count - total)
        try:
            part = sample_dataset(task, n, seed=int(seed + ordinal * 1_000_003), split="support",
                                  split_seed=split_seed, balanced=True, seq_len=seq_len)
        except RuntimeError:
            # Smaller chunks have the same distribution but a less punitive
            # rejection cap for the remaining class.
            if n <= 256:
                raise
            chunk = max(256, n // 2)
            continue
        pieces.append(part)
        total += n
        ordinal += 1
    return {key: torch.cat([part[key] for part in pieces], dim=0) for key in ("x", "y", "ids")}


def train_masked(
    masks: torch.Tensor,
    pattern: str,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str | torch.device,
    support_pool_size: int = 65_536,
    split_seed: int = 1729,
    validation_data: dict[str, torch.Tensor] | None = None,
) -> tuple[BatchedMaskedMLP, dict[str, Any]]:
    """Train a batched bank only on support and score it only on query."""
    if masks.ndim != 3:
        raise ValueError("masks must have shape [models, seq_len, hidden]")
    task = PatternTask(pattern)
    seq_len, hidden = masks.shape[1:]
    if task.length > seq_len:
        raise ValueError("pattern is longer than the mask input")
    support = _sample_pool(task, support_pool_size, seed=seed + 11, split_seed=split_seed, seq_len=seq_len)
    if validation_data is None:
        validation_data = sample_dataset(task, 2048, seed=seed + 29, split="query",
                                         split_seed=split_seed, balanced=True, seq_len=seq_len)
    for key in ("x", "y"):
        if key not in validation_data:
            raise ValueError("validation_data must contain x and y")

    model = BatchedMaskedMLP(masks, seed=seed).to(device)
    support_x, support_y = support["x"].to(device), support["y"].to(device)
    validation_x, validation_y = validation_data["x"].to(device), validation_data["y"].to(device)
    fit = fit_mlp(model, support_x, support_y, steps=steps, batch_size=batch_size, lr=lr, seed=seed + 47)
    score = model.validation(validation_x, validation_y)
    diagnostics: dict[str, Any] = {
        "fit": fit,
        "validation_bce": score["bce"].detach().cpu(),
        "validation_accuracy": score["accuracy"].detach().cpu(),
        "support_ids_hash": _sha256_tensor(support["ids"]),
        "query_ids_hash": _sha256_tensor(validation_data.get("ids", torch.empty(0, dtype=torch.int64))),
        "support_size": int(support_x.size(0)),
        "query_size": int(validation_x.size(0)),
        "support_unique_ids": int(torch.unique(support["ids"]).numel()),
    }
    return model, diagnostics


def train_one_pattern(config: Any, out: str | Path, pattern: str, device: str | torch.device) -> Path:
    """Train one full bank and atomically persist its selected top fraction."""
    from . import common  # parent-owned protocol helpers; import lazily for tests

    destination = Path(common.bank_path(Path(out), pattern))
    protocol_hash = hashlib.sha256(repr(config).encode()).hexdigest()
    if destination.exists():
        prior = torch.load(destination, map_location="cpu", weights_only=True)
        expected = {
            "protocol_hash": protocol_hash,
            "pattern": pattern,
            "selection_split": "query",
            "n_bank": int(config.bank_mlps),
        }
        mismatches = [key for key, value in expected.items() if prior.get(key) != value]
        if "importance" not in prior or "importance" not in prior.get("selected", {}):
            mismatches.append("importance contract")
        if mismatches:
            raise RuntimeError(f"refusing to reuse incompatible bank {destination}: {mismatches}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    n_models = int(config.bank_mlps)
    task_seed = common.seed_for("bank", config.seed, pattern)
    mask_cardinality = int(config.hidden) * len(pattern)
    masks = random_exact_k_masks(
        n_models, seq_len=int(config.seq_len), hidden=int(config.hidden),
        k_active=mask_cardinality, seed=common.seed_for("bank-masks", config.seed, pattern),
    )
    query = sample_dataset(
        PatternTask(pattern), int(config.bank_val_size),
        seed=common.seed_for("bank-query", config.seed, pattern), split="query",
        split_seed=int(config.input_split_seed), balanced=True, seq_len=int(config.seq_len),
    )
    model, diagnostics = train_masked(
        masks, pattern, steps=int(config.bank_steps), batch_size=int(config.bank_batch),
        lr=float(config.bank_lr), seed=task_seed, device=device,
        support_pool_size=int(config.support_pool_size), split_seed=int(config.input_split_seed),
        validation_data=query,
    )
    chosen = select_top_indices(diagnostics["validation_bce"], float(config.top_fraction))
    state = model.cpu_state()
    importance = (state["w1"] * masks).abs().index_select(0, chosen)
    # Each selected MLP supplies a map in [0, 1]; this prevents an arbitrary
    # global scale of one inner optimization trajectory becoming a CVAE label.
    importance = importance / importance.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    payload = {
        "schema_version": 1,
        "pattern": pattern,
        "length": len(pattern),
        "selection_split": "query",
        "top_fraction": float(config.top_fraction),
        "n_bank": n_models,
        "n_selected": int(chosen.numel()),
        # Canonical CVAE contract.  The nested copy below retains the selected
        # MLP bundle in one place for downstream audits.
        "importance": importance,
        "selected_indices": chosen,
        "selected": {
            "masks": masks.index_select(0, chosen),
            "params": {name: value.index_select(0, chosen) for name, value in state.items()},
            "importance": importance,
            "validation_bce": diagnostics["validation_bce"].index_select(0, chosen),
            "validation_accuracy": diagnostics["validation_accuracy"].index_select(0, chosen),
        },
        # Retain all query scores so selection remains independently auditable.
        "all_validation_bce": diagnostics["validation_bce"],
        "all_validation_accuracy": diagnostics["validation_accuracy"],
        "mask_cardinality": int(masks[0].sum().item()),
        "seeds": {"model": task_seed, "query": common.seed_for("bank-query", config.seed, pattern)},
        "diagnostics": {key: value for key, value in diagnostics.items() if key not in {"validation_bce", "validation_accuracy"}},
        "protocol_hash": protocol_hash,
    }
    common.atomic_save(destination, payload)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise SystemExit("--shard must be in [0, --shards)")
    # Must be in the environment before the first CUDA operation; deterministic
    # GEMMs otherwise fail on CUDA 10.2+.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from . import common
    config = common.load_protocol(args.out)
    splits = common.patterns_by_split(config)
    # Train and validation patterns supply CVAE examples.  Test patterns (the
    # held-out lengths) are intentionally absent from this stage.
    patterns = [str(item) for name in ("train", "val") for item in splits[name]]
    selected = patterns[args.shard::args.shards]
    for pattern in selected:
        result = train_one_pattern(config, args.out, pattern, args.device)
        print(f"[bank] {pattern} -> {result}", flush=True)


if __name__ == "__main__":
    main()
