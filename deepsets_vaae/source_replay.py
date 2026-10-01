"""Verified replay that restores the full selected source-bank models.

The pilot banks intentionally contain their historical fields (maps, masks,
weights, selection indices, and validation losses).  The original training
code kept checkpointed bias and readout in memory only.  This module replays
that exact training loop from the pinned pilot source snapshot, audits its
selection against the saved banks, then adds a batched ``state_dict`` for all
selected models.

The batched state dict can be loaded into the original ``MaskedDeepSets``
class constructed with the saved masks.  Its leading dimension is the bank
candidate dimension (normally 32).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "outputs/deepsets_vaae/20261001_pilot"
SNAPSHOT_CORE = PILOT_ROOT / "source_snapshot/core.py"
DEFAULT_CHECKPOINT_ROOT = ROOT / "outputs/deepsets_vaae/20261001_followup/source_checkpoints"
DEFAULT_DATA_DIR = ROOT / "datasets/mnist8m"

WEIGHT_RTOL = 2e-5
WEIGHT_ATOL = 2e-6
LOSS_RTOL = 2e-5
LOSS_ATOL = 2e-6
MAP_RTOL = 2e-5
MAP_ATOL = 2e-6


class ReplayMismatch(RuntimeError):
    """Raised when deterministic replay does not reproduce a saved bank."""


_ORIGINAL_CORE: ModuleType | None = None


def _original_core() -> ModuleType:
    """Load the immutable pilot snapshot instead of a mutable working module."""
    global _ORIGINAL_CORE
    if _ORIGINAL_CORE is None:
        if not SNAPSHOT_CORE.is_file():
            raise FileNotFoundError(f"pinned pilot source is missing: {SNAPSHOT_CORE}")
        spec = importlib.util.spec_from_file_location("deepsets_vaae_pilot_core", SNAPSHOT_CORE)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load pinned source snapshot {SNAPSHOT_CORE}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ORIGINAL_CORE = module
    return _ORIGINAL_CORE


def task_vectors(seed: int = 20261001) -> dict[str, Any]:
    """Return the pilot's fixed source/validation/test task vectors."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(14, 10))
    vectors -= vectors.mean(axis=1, keepdims=True)
    vectors /= vectors.std(axis=1, keepdims=True)
    return {"seed": seed, "source": vectors[:4].tolist(),
            "validation": vectors[4:6].tolist(), "test": vectors[6:].tolist()}


def _instrumented_build_bank(
    data: dict[str, Any],
    costs: torch.Tensor | Sequence[float],
    seed: int,
    device: str | torch.device,
    *,
    hidden: int = 32,
    density: float = 0.2,
    candidates: int = 128,
    keep: int = 32,
    steps: int = 800,
    batch_size: int = 32,
    set_size: int = 5,
) -> dict[str, Any]:
    """Pilot ``build_bank`` with one added output: the selected full state.

    The body below follows ``source_snapshot/core.py::build_bank`` line for
    line.  Keeping the optimizer, random generators, checkpoint cadence and
    candidate selection unchanged is necessary for a meaningful replay audit.
    """
    core = _original_core()
    if min(hidden, candidates, keep, steps, batch_size, set_size) < 1 or keep > candidates:
        raise ValueError("invalid positive bank sizes")
    target_device = torch.device(device)
    source_train, source_validation = data["source_train"], data["source_validation"]
    if not isinstance(source_train, core.Split):
        source_train = core.Split(*source_train)
    if not isinstance(source_validation, core.Split):
        source_validation = core.Split(*source_validation)
    if source_train.features.device != target_device:
        source_train = core.Split(*(value.to(target_device) for value in source_train))
        source_validation = core.Split(*(value.to(target_device) for value in source_validation))
    task_costs = core.centred_costs(costs, target_device)
    mask_generator = torch.Generator(device=target_device).manual_seed(seed + 31)
    data_generator = torch.Generator(device=target_device).manual_seed(seed + 32)
    masks = core._exact_random_masks(candidates, hidden, density,
                                     device=target_device, generator=mask_generator)
    model = core.MaskedDeepSets(masks, seed=seed + 101).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    val_x, val_y = core._sets(source_validation, task_costs, max(128, batch_size * 4),
                              set_size, data_generator)
    best_loss = torch.full((candidates,), float("inf"), device=target_device)
    best_state = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    curves: list[dict[str, Any]] = []
    check_every = max(1, min(50, steps // 8))
    for step in range(1, steps + 1):
        model.train()
        x, y = core._sets(source_train, task_costs, batch_size, set_size, data_generator)
        prediction = model(x)
        per_model = prediction.sub(y[None]).square().mean(dim=1) / set_size
        loss = per_model.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % check_every == 0 or step == steps:
            model.eval()
            validation = core._losses(model, val_x, val_y, set_size)
            improved = validation < best_loss
            for name, parameter in model.named_parameters():
                best_state[name][improved] = parameter.detach()[improved]
            best_loss = torch.minimum(best_loss, validation)
            curves.append({"step": step, "train_normalized_mse": float(loss.item()),
                           "validation_mean_normalized_mse": float(validation.mean().item())})
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(best_state[name])
    chosen = best_loss.argsort()[:keep]
    state_dict = {"masks": masks[chosen].detach().cpu().float().contiguous()}
    for name in ("weight", "bias", "readout", "per_image_offset"):
        state_dict[name] = best_state[name][chosen].detach().cpu().contiguous()
    maps = model.importance()[chosen].detach().cpu()
    result = {
        "maps": maps,
        "masks": masks[chosen].detach().cpu().bool(),
        "weights": best_state["weight"][chosen].detach().cpu(),
        "validation_losses": best_loss[chosen].detach().cpu().tolist(),
        "selected_candidates": chosen.detach().cpu().tolist(),
        "training_curves": curves,
        "best_validation_normalized_mse": best_loss.detach().cpu().tolist(),
        "edges_per_mask": int(masks[0].sum().item()),
        "source_split_hashes": {name: data.get("split_hashes", {}).get(name)
                                for name in ("source_train", "source_validation")},
        "state_dict": state_dict,
    }
    return result


def _to_cpu_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    result = torch.as_tensor(value).detach().cpu()
    return result.to(dtype=dtype) if dtype is not None else result


def _assert_close(name: str, actual: Any, expected: Any, *, rtol: float, atol: float) -> float:
    left = _to_cpu_tensor(actual, dtype=torch.float64)
    right = _to_cpu_tensor(expected, dtype=torch.float64)
    if left.shape != right.shape:
        raise ReplayMismatch(f"{name} shape mismatch: replay {tuple(left.shape)}, saved {tuple(right.shape)}")
    difference = (left - right).abs()
    maximum = float(difference.max().item()) if difference.numel() else 0.0
    if not torch.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False):
        raise ReplayMismatch(f"{name} mismatch (max_abs={maximum:g}, rtol={rtol:g}, atol={atol:g})")
    return maximum


def replay_bank(
    data: dict[str, Any],
    costs: torch.Tensor | Sequence[float],
    seed: int,
    device: str | torch.device,
    original_bank: Mapping[str, Any],
    *,
    hidden: int = 32,
    density: float = 0.2,
    candidates: int = 128,
    keep: int = 32,
    steps: int = 800,
    batch_size: int = 32,
    set_size: int = 5,
) -> dict[str, Any]:
    """Replay and audit one old bank; return legacy fields plus full weights.

    ``state_dict`` tensors are batched across selected candidates and include
    masks, W, bias, readout and per-image offset.  A mismatch in selected
    indices or masks is fatal; floating-point tensors use the tolerances above.
    """
    replayed = _instrumented_build_bank(
        data, costs, seed, device, hidden=hidden, density=density,
        candidates=candidates, keep=keep, steps=steps, batch_size=batch_size,
        set_size=set_size,
    )
    failures: list[str] = []
    if list(replayed["selected_candidates"]) != list(original_bank["selected_candidates"]):
        failures.append("selected_candidates differ")
    old_masks = torch.as_tensor(original_bank["masks"]).detach().cpu().bool()
    if not torch.equal(replayed["masks"], old_masks):
        failures.append("selected masks differ")
    if replayed["source_split_hashes"] != original_bank.get("source_split_hashes", {}):
        failures.append("source split hashes differ")
    errors: dict[str, float] = {}
    try:
        errors["weights_max_abs"] = _assert_close(
            "weights", replayed["weights"], original_bank["weights"],
            rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)
        errors["validation_losses_max_abs"] = _assert_close(
            "validation_losses", replayed["validation_losses"], original_bank["validation_losses"],
            rtol=LOSS_RTOL, atol=LOSS_ATOL)
        errors["maps_max_abs"] = _assert_close(
            "maps", replayed["maps"], original_bank["maps"],
            rtol=MAP_RTOL, atol=MAP_ATOL)
    except ReplayMismatch as exc:
        failures.append(str(exc))
    if failures:
        raise ReplayMismatch("; ".join(failures))

    # Copy the original dictionary first so old consumers retain their fields
    # and semantics. Replace only floating outputs with verified replay values.
    expanded = dict(original_bank)
    for key in ("maps", "masks", "weights", "validation_losses", "selected_candidates",
                "training_curves", "best_validation_normalized_mse", "edges_per_mask",
                "source_split_hashes"):
        expanded[key] = replayed[key]
    expanded["state_dict"] = replayed["state_dict"]
    expanded["replay_metadata"] = {
        "schema": "deepsets_vaae.source_checkpoint.v1",
        "source_snapshot": str(SNAPSHOT_CORE.relative_to(ROOT)),
        "replay_seed": int(seed),
        "device": str(device),
        "audit": "passed",
        "tolerances": {"weights": {"rtol": WEIGHT_RTOL, "atol": WEIGHT_ATOL},
                       "validation_losses": {"rtol": LOSS_RTOL, "atol": LOSS_ATOL},
                       "maps": {"rtol": MAP_RTOL, "atol": MAP_ATOL}},
        "max_abs_errors": errors,
    }
    return expanded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def replay_seed(
    seed: int,
    device: str | torch.device,
    *,
    pilot_root: str | Path = PILOT_ROOT,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict[str, Any]:
    """Replay all four source tasks and atomically publish a complete seed."""
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(device)
    pilot_root, checkpoint_root, data_dir = (Path(pilot_root).resolve(),
                                             Path(checkpoint_root).resolve(),
                                             Path(data_dir).resolve())
    seed_out = checkpoint_root / f"seed_{seed}"
    complete = seed_out / "COMPLETE"
    if complete.exists():
        audit_path = seed_out / "replay_audit.json"
        audit = json.loads(audit_path.read_text())
        if audit.get("status") == "passed" and audit.get("seed") == seed:
            return audit
        raise ReplayMismatch(f"invalid existing completion marker at {complete}")

    core = _original_core()
    started = time.monotonic()
    print(f"[source_replay] seed={seed} device={device} loading data", flush=True)
    data = core.load_data(data_dir, seed, device, per_digit_train=1000,
                          per_digit_validation=300, per_digit_test=300)
    tasks = task_vectors()
    task_audits = []
    for task_index, vector in enumerate(tasks["source"]):
        print(f"[source_replay] seed={seed} task={task_index} replaying 128 candidates", flush=True)
        original_path = pilot_root / f"seed_{seed}" / f"bank_{task_index}.pt"
        if not original_path.is_file():
            raise FileNotFoundError(f"legacy source bank is missing: {original_path}")
        original_bank = torch.load(original_path, map_location="cpu", weights_only=False)
        expanded = replay_bank(
            data, torch.tensor(vector, dtype=torch.float32, device=device),
            seed + 1000 * task_index, device, original_bank,
            hidden=32, density=0.2, candidates=128, keep=32,
            steps=800, batch_size=32, set_size=5,
        )
        destination = seed_out / f"bank_{task_index}.pt"
        _atomic_torch_save(destination, expanded)
        print(f"[source_replay] seed={seed} task={task_index} audit=passed "
              f"selected={len(expanded['selected_candidates'])} elapsed={time.monotonic() - started:.1f}s",
              flush=True)
        task_audits.append({
            "task": task_index,
            "original_bank": str(original_path.relative_to(ROOT)),
            "original_sha256": _sha256(original_path),
            "checkpoint": str(destination.relative_to(ROOT)),
            "checkpoint_sha256": _sha256(destination),
            "selected_candidates": expanded["selected_candidates"],
            "selected_masks_exact": True,
            "source_split_hashes": expanded["source_split_hashes"],
            **expanded["replay_metadata"],
        })

    audit = {
        "schema": "deepsets_vaae.source_replay_audit.v1",
        "status": "passed",
        "seed": int(seed),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "elapsed_seconds": time.monotonic() - started,
        "source_snapshot": str(SNAPSHOT_CORE.relative_to(ROOT)),
        "source_snapshot_sha256": _sha256(SNAPSHOT_CORE),
        "data_dir": str(data_dir),
        "data_split_hashes": data["split_hashes"],
        "tasks": task_audits,
    }
    _atomic_json(seed_out / "replay_audit.json", audit)
    temp_complete = seed_out / "COMPLETE.tmp"
    temp_complete.write_text("all four banks replayed and audited\n")
    os.replace(temp_complete, complete)
    return audit


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    # Accepted for the shared CUDA queue's standard worker command. The
    # immutable checkpoints always go to --checkpoint-root.
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pilot-root", type=Path, default=PILOT_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    result = replay_seed(args.seed, args.device, pilot_root=args.pilot_root,
                         checkpoint_root=args.checkpoint_root, data_dir=args.data_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
