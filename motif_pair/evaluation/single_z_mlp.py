"""Task-loss search of one frozen scalar-CVAE latent code.

For each outer step a fresh bank of MLPs is trained on a *detached* decoder
mask.  The resulting MLP weights are frozen, then a search-validation BCE is
differentiated only through a newly decoded soft mask into ``z``.  Thus this
is a direct partial derivative through a trained, frozen downstream model;
it deliberately does not differentiate through the MLP optimiser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from data.generate import make_dataset  # noqa: E402
from evaluation.eval_generated_masks import _check_provenance, _load_model, _seed  # noqa: E402
from evaluation.oracle_ideal import hard_topk, soft_topk  # noqa: E402
from models.cvae import checkpoint_condition_metadata, read_split_provenance  # noqa: E402
from models.mlp import BatchedMaskedMLP, get_train_batch  # noqa: E402


OUTER_STEPS = 30
INNER_STEPS = 400
N_STARTS = 64
TEMPERATURE = .5
RADIUS = 8.
Z_LR = .05
INNER_LR = 1e-3
INNER_BATCH_SIZE = 128
SEARCH_VAL_SAMPLES = 1024


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_(z: torch.Tensor, radius: float) -> None:
    with torch.no_grad():
        z.mul_((radius / z.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1.))


def _freeze_decoder(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False); p.grad = None
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_decoder_unchanged(model: torch.nn.Module, before: dict[str, torch.Tensor]) -> None:
    for name, value in model.state_dict().items():
        if not torch.equal(before[name], value):
            raise AssertionError(f"frozen CVAE changed: {name}")
    if any(p.grad is not None for p in model.parameters()):
        raise AssertionError("frozen CVAE received gradients")


def _snapshot_mlp(model: BatchedMaskedMLP) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_mlp_unchanged(model: BatchedMaskedMLP, before: dict[str, torch.Tensor]) -> None:
    for name, value in model.state_dict().items():
        if not torch.equal(before[name], value):
            raise AssertionError(f"trained frozen MLP changed during z update: {name}")
    if any(p.grad is not None for p in model.parameters()):
        raise AssertionError("trained frozen MLP received gradients during z update")


def _per_network_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y.unsqueeze(1).expand_as(logits), reduction="none").mean(0)


def _forward_live_mask(model: BatchedMaskedMLP, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Use frozen MLP weights but retain the gradient from a live mask to z."""
    hidden = F.relu(torch.einsum("bi,mih->bmh", x, model.w1 * mask) + model.b1)
    return torch.einsum("bmh,mh->bm", hidden, model.w2.squeeze(-1)) + model.b2.squeeze(-1)


def _decode(model: torch.nn.Module, z: torch.Tensor, condition: torch.Tensor,
            temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model.decode(z, condition)
    if logits.shape != (len(z), config.MASK_DIM):
        raise ValueError("decoder output does not have motif mask shape")
    return logits, soft_topk(logits, config.K_ACTIVE, temperature).reshape(len(z), config.SEQ_LEN, config.H)


def _fresh_mlp(n_starts: int, seed: int, device: torch.device) -> BatchedMaskedMLP:
    # Preserve process-global RNG state; each outer MLP is reproducible and
    # independent of which other task shards happen to run in parallel.
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda": torch.cuda.manual_seed_all(seed)
        return BatchedMaskedMLP(n_starts, config.SEQ_LEN, config.H).to(device)


def optimize_task_z(model: torch.nn.Module, initial_z: torch.Tensor, task: str,
                    *, device: str | torch.device, outer_steps: int = OUTER_STEPS,
                    inner_steps: int = INNER_STEPS, z_lr: float = Z_LR,
                    radius: float = RADIUS, temperature: float = TEMPERATURE) -> dict[str, Any]:
    """Adapt ``z`` to a task loss, keeping decoder and each trained MLP fixed for the z update."""
    if initial_z.shape != (N_STARTS, int(getattr(model, "latent_dim", -1))):
        raise ValueError("initial_z must be the exact 64xlatent prior block")
    if outer_steps <= 0 or inner_steps < 0 or z_lr <= 0 or radius <= 0 or temperature <= 0:
        raise ValueError("invalid optimisation settings")
    device = torch.device(device)
    before_decoder = _freeze_decoder(model)
    z = torch.nn.Parameter(initial_z.detach().to(device).clone())
    if bool((z.norm(dim=1) > radius + 1e-5).any()):
        raise ValueError("saved prior z lies outside the declared radius")
    condition = model.condition([task], device=device).expand(N_STARTS, -1)
    validation = make_dataset(task, SEARCH_VAL_SAMPLES, _seed(task, 80_000), config.POS_FRACTION)
    x_val, y_val = validation["x"].to(device), validation["y"].to(device)
    z_optimizer = torch.optim.Adam([z], lr=z_lr)
    history: list[dict[str, Any]] = []
    initial_z_cpu = z.detach().cpu().clone()

    for outer in range(outer_steps):
        # Training uses the current mask as a constant: inner SGD cannot
        # leave a graph or an accumulated gradient on z.
        with torch.no_grad():
            _, detached_soft = _decode(model, z, condition, temperature)
        mlp = _fresh_mlp(N_STARTS, _seed(task, 90_000_003) + outer, device)
        mlp.load_masks(detached_soft)
        inner_optimizer = torch.optim.Adam(mlp.parameters(), lr=INNER_LR)
        inner_losses: list[float] = []
        for step in range(inner_steps):
            xb, yb = get_train_batch(task, INNER_BATCH_SIZE,
                                     _seed(task, 100_000_003) + outer * 100_000 + step, device=device)
            train_loss = _per_network_bce(mlp(xb), yb)
            inner_optimizer.zero_grad(set_to_none=True); train_loss.sum().backward(); inner_optimizer.step()
            inner_losses.append(float(train_loss.detach().mean()))
        for p in mlp.parameters():
            p.requires_grad_(False); p.grad = None
        mlp_before_z = _snapshot_mlp(mlp)

        z_optimizer.zero_grad(set_to_none=True)
        _, live_soft = _decode(model, z, condition, temperature)
        search_loss = _per_network_bce(_forward_live_mask(mlp, x_val, live_soft), y_val)
        search_loss.sum().backward()
        if z.grad is None or not bool(torch.isfinite(z.grad).all()):
            raise AssertionError("frozen-MLP search validation BCE did not yield a finite z gradient")
        grad_norm = z.grad.detach().norm(dim=1)
        z_optimizer.step(); _project_(z, radius)
        _assert_mlp_unchanged(mlp, mlp_before_z)
        with torch.no_grad():
            _, post_soft = _decode(model, z, condition, temperature)
            post_loss = _per_network_bce(_forward_live_mask(mlp, x_val, post_soft), y_val)
        history.append({"outer": outer, "inner_train_loss_mean": inner_losses,
                        "pre_update_search_val_bce": search_loss.detach().cpu().tolist(),
                        "post_update_search_val_bce": post_loss.detach().cpu().tolist(),
                        "z_grad_norm": grad_norm.cpu().tolist(),
                        "z_norm_after_update": z.detach().norm(dim=1).cpu().tolist()})
        if (outer + 1) % 5 == 0 or outer + 1 == outer_steps:
            print(f"[single-z] {task} outer={outer + 1}/{outer_steps} "
                  f"search-bce={float(search_loss.mean()):.5f} z-norm={float(z.detach().norm(dim=1).mean()):.3f}", flush=True)

    with torch.no_grad():
        final_logits, final_soft = _decode(model, z, condition, temperature)
        final_hard = hard_topk(final_logits, config.K_ACTIVE)
    _assert_decoder_unchanged(model, before_decoder)
    return {"initial_z": initial_z_cpu, "final_z": z.detach().cpu(), "final_soft": final_soft.flatten(1).cpu(),
            "final_hard": final_hard.cpu(), "history": history, "decoder_unchanged": True,
            "gradient_semantics": "partial derivative through live soft-top96 mask and trained frozen MLP; inner optimization not unrolled"}


def _single_protocol(root: Path) -> dict[str, Any]:
    parent = root / "protocol.json"
    if not parent.exists(): raise FileNotFoundError(f"missing parent protocol {parent}")
    return {"experiment": "single_frozen_cvae_task_loss_z", "parent_protocol": str(parent.resolve()),
            "parent_protocol_sha256": _sha256(parent), "source": str(Path(__file__).resolve()),
            "source_sha256": _sha256(Path(__file__)), "profiles": json.loads(parent.read_text())["profiles"],
            "settings": {"n_starts": N_STARTS, "outer_steps": OUTER_STEPS, "inner_steps": INNER_STEPS,
                         "inner_batch_size": INNER_BATCH_SIZE, "inner_lr": INNER_LR, "z_lr": Z_LR,
                         "radius": RADIUS, "temperature": TEMPERATURE, "k": config.K_ACTIVE,
                         "search_validation_samples": SEARCH_VAL_SAMPLES, "search_validation_seed": "_seed(task, 80000)",
                         "inner_mlp_seed": "_seed(task, 90000003) + outer", "inner_training_seed": "_seed(task, 100000003) + outer*100000 + step",
                         "final_evaluation_train_seed": "_seed(task, 1_000_003) + step", "final_evaluation_test_seed": "_seed(task, 20_000)",
                         "selection": "last z after 30 updates; no validation-best, test, or gold selection",
                         "gradient_semantics": "MLP trains on detached soft mask; then weights freeze and validation BCE differentiates only z through live soft mask; no unrolling"}}


def ensure_protocol(root: Path) -> tuple[Path, dict[str, Any]]:
    path = root / "single_z_protocol.json"; current = _single_protocol(root)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != current: raise ValueError("existing single-z protocol differs from current source/parent protocol")
    else:
        path.write_text(json.dumps(current, indent=2) + "\n")
    return path, current


def _load_prior_block(search_path: Path, gap: int, latent_dim: int) -> tuple[torch.Tensor, str]:
    payload = torch.load(search_path, weights_only=True, map_location="cpu")
    gaps = payload.get("gaps"); z = payload.get("latents", {}).get("prior", {}).get("z1")
    if not isinstance(gaps, torch.Tensor) or not isinstance(z, torch.Tensor):
        raise ValueError("pair-search masks.pt lacks gaps or first-decoder prior latent codes")
    if gaps.tolist() != list(config.GAPS) or z.shape != (len(config.GAPS) * N_STARTS, latent_dim):
        raise ValueError("pair-search prior z schema does not match 8x64xlatent protocol")
    ordinal = gaps.tolist().index(int(gap))
    return z[ordinal * N_STARTS:(ordinal + 1) * N_STARTS].clone(), _sha256(search_path)


def _validate_profile(root: Path, profile: str, device: torch.device):
    protocol_path, protocol = ensure_protocol(root)
    if profile not in protocol["profiles"]: raise ValueError(f"unknown profile {profile!r}")
    spec = protocol["profiles"][profile]
    checkpoint, split = Path(spec["checkpoint"]), Path(spec["split"])
    expected = read_split_provenance(split)
    model, payload = _load_model(checkpoint, device)
    _check_provenance(payload, expected, "single-z CVAE", importance_name="importance.pt", top_frac=.1)
    expected_condition = config.condition_metadata(config.CONDITION_ENCODING_SCALAR)
    if (payload.get("seed") != 42 or model.cond_dim != 1 or payload.get("condition_encoding") != config.CONDITION_ENCODING_SCALAR
            or checkpoint_condition_metadata(payload) != expected_condition):
        raise ValueError("single-z requires the pre-registered scalar CVAE seed 42")
    if _sha256(checkpoint) != spec["checkpoint_sha256"] or expected["split_sha256"] != spec["split_sha256"]:
        raise ValueError("profile checkpoint/split differs from parent protocol")
    return model, checkpoint, split, expected, protocol_path


def _save_task(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        previous = torch.load(path, weights_only=True, map_location="cpu")
        for key in ("task", "metadata"):
            if previous.get(key) != payload.get(key): raise ValueError(f"resume artifact mismatch at {path}")
        return
    torch.save(payload, path)


def run(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    if args.prepare_protocol:
        path, _ = ensure_protocol(root); print(f"saved/validated {path}"); return
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required; run outside sandbox")
    device = torch.device(args.device)
    model, checkpoint, split_path, split_provenance, protocol_path = _validate_profile(root, args.profile, device)
    split = json.loads(split_path.read_text()); tasks = list(split["test_tasks"])
    if len(tasks) != 16: raise ValueError("single-z task comparator expects exactly 16 held-out tasks")
    indexes = list(range(len(tasks))) if args.task_indices is None else list(args.task_indices)
    if not indexes or len(set(indexes)) != len(indexes) or any(i < 0 or i >= len(tasks) for i in indexes):
        raise ValueError("task-indices must be distinct held-out task indexes")
    search_path = root / args.profile / "search" / "masks.pt"
    out = root / args.profile / "single_z"; out.mkdir(parents=True, exist_ok=True)
    metadata = {"protocol": str(protocol_path.resolve()), "protocol_sha256": _sha256(protocol_path),
                "source_sha256": _sha256(Path(__file__)), "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint), "split": str(split_path.resolve()), **split_provenance,
                "parent_masks": str(search_path.resolve()), "parent_masks_sha256": _sha256(search_path),
                "settings": _single_protocol(root)["settings"]}
    for index in indexes:
        task = tasks[index]; prior, search_sha = _load_prior_block(search_path, config.parse_task(task).gap, model.latent_dim)
        if search_sha != metadata["parent_masks_sha256"]: raise RuntimeError("pair-search artifact changed during single-z run")
        result = optimize_task_z(model, prior, task, device=device)
        _save_task(out / f"task_{task}.pt", {"task": task, "task_index": index, "metadata": metadata, **result})
    files = [out / f"task_{task}.pt" for task in tasks]
    if all(path.exists() for path in files):
        saved = [torch.load(path, weights_only=True, map_location="cpu") for path in files]
        if any(row["metadata"] != metadata for row in saved): raise ValueError("cannot aggregate inconsistent task shards")
        torch.save({"tasks": tasks, "masks": torch.stack([row["final_hard"] for row in saved]),
                    "latents": torch.stack([row["final_z"] for row in saved]), "metadata": metadata}, out / "masks.pt")
        print(f"saved {out / 'masks.pt'}", flush=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--profile", choices=("interp", "extrap"), default="interp")
    p.add_argument("--task-indices", type=int, nargs="+")
    p.add_argument("--device", choices=("cuda",), default="cuda"); p.add_argument("--prepare-protocol", action="store_true")
    return p


if __name__ == "__main__":
    parsed = parser().parse_args(); torch.set_num_threads(2); torch.use_deterministic_algorithms(True); run(parsed)
