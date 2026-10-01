"""Run a paired DeepSets density sweep from the frozen 20261001 pilot banks.

The three density-dependent masks are a nested random ranking, a nested
function-gradient ranking, and a separately optimized VAE-agreement mask.
Each density keeps all target checkpoints for the eight exploratory test
tasks and the two independent density-selection cost tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import masks as mask_ops
from .core import evaluate_masks, load_data
from .followup_batched_eval import evaluate_masks_batched
from .followup_common import PILOT, configure, load_pilot_seed, run_cuda_queue
from .followup_importance import _raw_alignment_orders
from .run import task_vectors, write_json


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs/deepsets_vaae/20261001_density_sweep"
IMPORTANCE_ROOT = ROOT / "outputs/deepsets_vaae/20261001_followup/importance/fp32_final"
SEEDS = tuple(range(4100, 4108))
GPU_UUIDS = [
    "GPU-2e5ce2f9-206f-380c-9687-3743ff9f665c",
    "GPU-6784dc4e-6ec9-2266-5d23-85bf1b1c2af3",
    "GPU-f8501f2d-53bc-9087-6041-64ee69876325",
    "GPU-7bb2c2a2-451a-7632-d931-fc64f8901744",
    "GPU-ebb2cc3f-3769-6b77-f2de-166234901bb6",
    "GPU-fa392093-03b9-0a49-7959-ed393750e978",
    "GPU-2ac4168d-7c26-2108-6c70-778e5b97f743",
    "GPU-3c5e3dfd-1a22-964d-5405-e4dd5ac73795",
]
DENSITIES = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0)
FEATURES = 784
HIDDEN = 32
FLAT_DIM = FEATURES * HIDDEN
REFERENCE_MODELS = 20
SUPPORT_SIZES = (32, 64, 128, 256)
ORIGINAL_METHODS = ("agreement", "mean", "single_vae", "random", "dense")
NEW_METHODS = ("random_density", "function_gradient_density", "agreement_density")
TEST_TARGET_SEED_OFFSET = 100_000
VALIDATION_TARGET_SEED_OFFSET = 200_000
CONTROL_TOLERANCE = 1e-4

SOURCE_FILES = (
    Path(__file__),
    ROOT / "deepsets_vaae/core.py",
    ROOT / "deepsets_vaae/masks.py",
    ROOT / "deepsets_vaae/followup_batched_eval.py",
    ROOT / "deepsets_vaae/followup_common.py",
    ROOT / "deepsets_vaae/followup_importance.py",
    ROOT / "deepsets_vaae/run.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def _slug(density: float) -> str:
    return f"rho_{density:.3f}".replace(".", "p")


def _edge_count(density: float) -> int:
    return int(round(density * FLAT_DIM))


def _prepare_root(out: Path = OUT_ROOT) -> dict[str, str]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    hashes: dict[str, str] = {}
    for source in SOURCE_FILES:
        source = source.resolve()
        content = source.read_bytes()
        target = snapshot / source.name
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"Frozen density-sweep source changed: {target}")
        if not target.exists():
            target.write_bytes(content)
        hashes[source.name] = hashlib.sha256(content).hexdigest()
    return hashes


def _root_protocol(hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "experiment": "DeepSets target-quality versus connection-density sweep",
        "date_moscow": "2026-10-01",
        "seeds": list(SEEDS),
        "densities": list(DENSITIES),
        "edge_counts": {str(rho): _edge_count(rho) for rho in DENSITIES},
        "density_directory_slugs": {_slug(rho): rho for rho in DENSITIES},
        "original_methods_first": list(ORIGINAL_METHODS),
        "new_methods": list(NEW_METHODS),
        "model": {"family": "DeepSets", "features": FEATURES, "hidden": HIDDEN,
                  "activation": "tanh", "pooling": "sum over five images",
                  "set_size": 5, "shared_item_encoder": True},
        "target_evaluation": {
            "test_cost_tasks": 8, "validation_cost_tasks": 2,
            "support_sizes": list(SUPPORT_SIZES), "steps": 800, "batch_size": 32,
            "validation_sets": 128, "test_sets": 512, "paired_initializations": 4,
            "initialization_reference_models": REFERENCE_MODELS,
            "gradient_denominator": REFERENCE_MODELS,
            "batch_conditions": 8, "kernel_mode": "reference",
            "test_target_seed_offset": TEST_TARGET_SEED_OFFSET,
            "validation_target_seed_offset": VALIDATION_TARGET_SEED_OFFSET,
            "test_task_vectors": "pilot protocol task_vectors.test[0:8]",
            "validation_task_vectors": "pilot protocol task_vectors.validation[0:2]",
            "density_selection_metric": "validation_mse only on the two validation cost tasks",
            "test_task_use": "descriptive curves; the eight test costs were inspected during pilot exploration",
        },
        "mask_extraction": {
            "random_density": "one original-seed+50000+101 split generator; four _split_unique passes; then torch.rand(4,25088); stable descending rank prefixes",
            "function_gradient_density": "final importance/fp32_final function_gradient arrays selected by saved training row ids; same four-round raw-bank Hungarian column orders; mean consensus; stable descending rank prefixes",
            "agreement_density": "restore four frozen pilot MaskVAE states; rerun masks._search_agreement for each interior K with seed+50000, 400 steps, four starts; all-off/all-on endpoints bypass search",
            "tie_policy": "stable descending argsort; equal scores retain ascending flattened feature-hidden index",
            "target_labels_for_mask_extraction": False,
            "vae_retrained": False,
            "source_banks_retrained": False,
        },
        "control_audit": {"test_rows_per_density": 640,
                          "maximum_absolute_test_mse_delta": CONTROL_TOLERANCE,
                          "reference": "outputs/deepsets_vaae/20261001_pilot/seed_s/results.json"},
        "outputs": {
            "seed_payload": "seed_s/results.json with all density-tagged test and validation records",
            "density_payload": "seed_s/rho_slug/results.json",
            "test_states": "seed_s/rho_slug/weights/target_task{task}_budget{budget}.pt (32 files)",
            "validation_states": "seed_s/rho_slug/validation_weights/target_task{task}_budget{budget}.pt (8 files)",
            "masks": "seed_s/rho_slug/masks.pt containing the three new [4,784,32] mask tensors",
            "agreement_diagnostics": "seed_s/rho_slug/agreement_diagnostics.json",
            "raw_rankings": "seed_s/ranking_arrays.npz",
            "complete_marker": "seed_s/COMPLETE only after all eleven densities finish",
        },
        "gpu_uuids": GPU_UUIDS,
        "gpu_memory_considered": False,
        "source_sha256": hashes,
    }


def _load_function_rank(seed: int, banks: list[dict[str, Any]], device: torch.device):
    arrays_path = IMPORTANCE_ROOT / f"seed_{seed}" / "importance_arrays.npz"
    transfer_path = IMPORTANCE_ROOT / f"seed_{seed}" / "transfer_masks.pt"
    if not arrays_path.is_file() or not transfer_path.is_file():
        raise FileNotFoundError(f"missing final function-gradient artifacts for seed {seed}")
    with np.load(arrays_path) as archive:
        if "training_row_indices_by_task" not in archive or "function_gradient" not in archive:
            raise ValueError(f"incomplete function-gradient arrays: {arrays_path}")
        rows_np = np.asarray(archive["training_row_indices_by_task"], dtype=np.int64)
        values_np = np.asarray(archive["function_gradient"], dtype=np.float32)
    if rows_np.ndim != 2 or rows_np.shape[0] != 4 or values_np.shape != (4, 32, FEATURES, HIDDEN):
        raise ValueError(f"unexpected final function-gradient array shapes for seed {seed}")
    rows = torch.as_tensor(rows_np, dtype=torch.long, device=device)
    raw_bank_maps = [torch.as_tensor(bank["maps"], dtype=torch.float32, device=device)
                     for bank in banks]
    raw_train = [bank.index_select(0, rows[task]) for task, bank in enumerate(raw_bank_maps)]
    orders = _raw_alignment_orders(raw_train)
    maps = torch.as_tensor(values_np, dtype=torch.float32, device=device)
    aligned: list[torch.Tensor] = []
    for task, order in enumerate(orders):
        task_values = maps[task].index_select(0, rows[task])
        aligned.append(task_values.gather(2, order[:, None, :].expand_as(task_values)))
    consensus = torch.cat(aligned, dim=0).mean(dim=0).reshape(-1)
    transfer = torch.load(transfer_path, map_location="cpu", weights_only=True)
    transfer_mask = transfer["importance_function_gradient"].detach().cpu().bool()
    if tuple(transfer_mask.shape) != (4, FEATURES, HIDDEN):
        raise ValueError(f"unexpected transfer mask shape: {tuple(transfer_mask.shape)}")
    return consensus, rows_np, [order.detach().cpu().numpy() for order in orders], transfer_mask


def _random_rank(seed: int, banks: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    """Reproduce extract_masks' split-generator draws, omitting VAE fitting."""
    generator = torch.Generator(device=device).manual_seed(seed + 50_000 + 101)
    for bank in banks:
        maps = torch.as_tensor(bank["maps"], dtype=torch.float32, device=device)
        unique = mask_ops._unique_rows(maps)
        if unique.size(0) <= 1:
            continue
        # The split draw is intentionally retained because it precedes the
        # original extractor's random-mask draw on the shared generator.
        torch.randperm(unique.size(0), generator=generator, device=device)
    return torch.rand(4, FLAT_DIM, generator=generator, device=device)


def _stable_orders(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim == 1:
        scores = scores[None]
    return torch.argsort(scores, dim=-1, descending=True, stable=True)


def _masks_from_order(order: torch.Tensor, k: int) -> torch.Tensor:
    if order.ndim == 1:
        order = order[None].expand(4, -1)
    mask = torch.zeros((order.shape[0], FLAT_DIM), dtype=torch.float32, device=order.device)
    if k:
        mask.scatter_(1, order[:, :k], 1.0)
    return mask.reshape(-1, FEATURES, HIDDEN)


def _load_vae_models(seed: int, device: torch.device):
    artifact_path = PILOT / f"seed_{seed}" / "vae_artifacts.pt"
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if len(artifact.get("vae_state_dicts", [])) != 4:
        raise ValueError(f"pilot VAE artifact must contain four models: {artifact_path}")
    models = []
    for state in artifact["vae_state_dicts"]:
        model = mask_ops._MaskVAE(FLAT_DIM, latent_dim=16, width=128).to(device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models, artifact_path


def _extract_density_masks(seed: int, density: float, k: int,
                           random_order: torch.Tensor, function_order: torch.Tensor,
                           vae_models: list[torch.nn.Module], device: torch.device,
                           original_masks: dict[str, torch.Tensor],
                           transfer_function_mask: torch.Tensor):
    random_mask = _masks_from_order(random_order, k)
    function_mask = _masks_from_order(function_order, k)
    if k == 0:
        agreement_mask = torch.zeros((4, FEATURES, HIDDEN), device=device)
        agreement_diagnostics = {"density": density, "edges": k, "bypassed_search": True,
                                 "endpoint": "all_off", "starts": 4, "steps": 0}
    elif k == FLAT_DIM:
        agreement_mask = torch.ones((4, FEATURES, HIDDEN), device=device)
        agreement_diagnostics = {"density": density, "edges": k, "bypassed_search": True,
                                 "endpoint": "all_on", "starts": 4, "steps": 0}
    else:
        agreement_mask, diagnostics, _ = mask_ops._search_agreement(
            vae_models, starts=4, steps=400, seed=seed + 50_000, k=k,
            shape=(FEATURES, HIDDEN), device=device)
        agreement_mask = agreement_mask.float()
        agreement_diagnostics = {"density": density, "edges": k, "bypassed_search": False,
                                 "steps": 400, "starts": 4, "seed": seed + 50_000,
                                 "diagnostics": diagnostics}
    masks = {"random_density": random_mask,
             "function_gradient_density": function_mask,
             "agreement_density": agreement_mask}
    for name, value in masks.items():
        counts = value.sum(dim=(-1, -2))
        if not bool(torch.all(counts == k)):
            raise AssertionError(f"{name} has wrong density cardinality: {counts.tolist()} != {k}")
        if not bool(torch.all((value == 0) | (value == 1))):
            raise AssertionError(f"{name} contains non-binary mask values")
    exact: dict[str, bool] = {}
    if k == _edge_count(0.20):
        expected = {
            "random_density": original_masks["random"].detach().cpu().bool(),
            "function_gradient_density": transfer_function_mask,
            "agreement_density": original_masks["agreement"].detach().cpu().bool(),
        }
        for name, reference in expected.items():
            match = torch.equal(masks[name].detach().cpu().bool(), reference)
            exact[name] = match
            if not match:
                raise AssertionError(f"20% extraction does not reproduce its required control: {name}")
    return masks, agreement_diagnostics, exact


def _augment(rows: list[dict[str, Any]], *, seed: int, density: float,
             edges: int, phase: str) -> list[dict[str, Any]]:
    return [dict(row, seed=seed, density=density, edges=edges, phase=phase) for row in rows]


def _control_audit(seed: int, records: list[dict[str, Any]]) -> dict[str, Any]:
    old_rows = json.loads((PILOT / f"seed_{seed}" / "results.json").read_text())["records"]
    key = lambda row: (row["task"], row["support_size"], row["method"], row["init"])
    old = {key(row): row for row in old_rows if row["method"] in ORIGINAL_METHODS}
    controls = [row for row in records if row["method"] in ORIGINAL_METHODS]
    expected_keys = set(old)
    if len(expected_keys) != 640 or set(map(key, controls)) != expected_keys:
        raise AssertionError(f"density control coverage differs for seed {seed}: {len(controls)} rows")
    deltas = []
    for row in controls:
        previous = old[key(row)]
        delta = abs(float(row["mse"]) - float(previous["mse"]))
        deltas.append({"task": row["task"], "support_size": row["support_size"],
                       "method": row["method"], "init": row["init"],
                       "pilot_mse": float(previous["mse"]), "density_mse": float(row["mse"]),
                       "absolute_delta": delta})
    maximum = max(row["absolute_delta"] for row in deltas)
    audit = {"seed": seed, "phase": "test", "control_records": len(deltas),
             "methods": list(ORIGINAL_METHODS), "maximum_absolute_mse_delta": maximum,
             "mean_absolute_mse_delta": sum(row["absolute_delta"] for row in deltas) / len(deltas),
             "threshold": CONTROL_TOLERANCE, "passed": maximum <= CONTROL_TOLERANCE,
             "rows": deltas}
    if not audit["passed"]:
        raise AssertionError(f"original controls do not reproduce for seed {seed}: {maximum}")
    return audit


def _write_rankings(out: Path, random_scores: torch.Tensor, random_order: torch.Tensor,
                    function_scores: torch.Tensor, function_order: torch.Tensor,
                    training_rows: np.ndarray, alignment_orders: list[np.ndarray],
                    function_transfer_mask: torch.Tensor) -> None:
    path = out / "ranking_arrays.npz"
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            random_scores=random_scores.detach().cpu().numpy(),
            random_order=random_order.detach().cpu().numpy(),
            function_gradient_consensus=function_scores.detach().cpu().numpy(),
            function_gradient_order=function_order.detach().cpu().numpy(),
            training_row_indices_by_task=training_rows,
            raw_alignment_orders=np.stack(alignment_orders),
            function_gradient_20pct_mask=function_transfer_mask.numpy().astype(np.uint8),
            tie_policy=np.asarray("stable descending; flattened index ascending"),
        )
    temporary.replace(path)


def _nested_audit(previous: dict[str, torch.Tensor], current: dict[str, torch.Tensor],
                  previous_edges: int, edges: int) -> dict[str, Any]:
    rows = {}
    for name in ("random_density", "function_gradient_density"):
        small = previous[name].bool()
        large = current[name].bool()
        nested = bool(torch.all(small <= large))
        rows[name] = {"nested": nested,
                      "previous_edges": previous_edges, "current_edges": edges,
                      "added_edges": int((large & ~small).sum().item())}
        if not nested:
            raise AssertionError(f"fixed-score ranking is not nested: {name}, {previous_edges}->{edges}")
    return rows


def _evaluate(data: dict, costs: list[list[float]], masks: dict[str, torch.Tensor],
              seed: int, device: torch.device, artifact_dir: Path,
              *, steps: int = 800, batch_size: int = 32, support_sizes=SUPPORT_SIZES,
              validation_sets: int = 128, test_sets: int = 512) -> list[dict[str, Any]]:
    return evaluate_masks_batched(
        data, costs, masks, seed, device, support_sizes=support_sizes,
        steps=steps, batch_size=batch_size, set_size=5,
        validation_sets=validation_sets, test_sets=test_sets,
        artifact_dir=artifact_dir,
        initialization_reference_models=REFERENCE_MODELS,
        batch_conditions=8, kernel_mode="reference")


def _write_density_protocol(folder: Path, seed: int, density: float, edges: int,
                            hashes: dict[str, str], uuid: str | None) -> None:
    _atomic_json(folder / "protocol.json", {
        "seed": seed, "density": density, "edges": edges,
        "directory_slug": _slug(density), "source_sha256": hashes,
        "cuda_visible_devices": uuid,
        "methods": {"new": list(NEW_METHODS), "original_controls_first": list(ORIGINAL_METHODS)},
        "target_seeds": {"test": seed + TEST_TARGET_SEED_OFFSET,
                         "validation": seed + VALIDATION_TARGET_SEED_OFFSET},
        "validation_selection_metric": "validation_mse only on task_vectors.validation cost tasks",
        "test_costs_are_descriptive": True,
    })


def run_seed(seed: int, out: Path, hashes: dict[str, str]) -> None:
    started = time.monotonic()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE").exists():
        raise FileExistsError(f"Seed is already complete: {out}")
    configure(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("density sweep workers require one assigned CUDA device")
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES")
    original = load_pilot_seed(seed, device)
    data = original["data"]
    original_masks = original["original_masks"]
    original_names = tuple(original_masks)
    if original_names != ORIGINAL_METHODS:
        raise ValueError(f"original control order changed: {original_names}")
    test_costs = original["protocol"]["task_vectors"]["test"]
    validation_costs = original["protocol"]["task_vectors"]["validation"]
    if len(test_costs) != 8 or len(validation_costs) != 2:
        raise ValueError("pilot protocol must contain eight test and two validation cost tasks")

    function_scores, training_rows, alignment_orders, transfer_function_mask = _load_function_rank(
        seed, original["banks"], device)
    function_order = _stable_orders(function_scores)[0]
    random_scores = _random_rank(seed, original["banks"], device)
    random_order = _stable_orders(random_scores)
    function_at_20 = _masks_from_order(function_order, _edge_count(0.20)).detach().cpu().bool()
    function_selected = function_scores[function_order[_edge_count(0.20) - 1]]
    function_next = function_scores[function_order[_edge_count(0.20)]]
    if not bool(function_selected > function_next):
        raise AssertionError("function-gradient score cutoff at 20% is tied")
    if not torch.equal(function_at_20, transfer_function_mask):
        raise AssertionError("function-gradient ranking differs from final 20% transfer mask")
    if not torch.equal(_masks_from_order(random_order, _edge_count(0.20)).cpu().bool(),
                       original_masks["random"].detach().cpu().bool()):
        raise AssertionError("random ranking differs from original 20% random mask")
    _write_rankings(out, random_scores, random_order, function_scores, function_order,
                    training_rows, alignment_orders, transfer_function_mask)
    vae_models, vae_path = _load_vae_models(seed, device)
    vae_hash = _sha256(vae_path)

    aggregate_records: list[dict[str, Any]] = []
    aggregate_validation: list[dict[str, Any]] = []
    nested_from: dict[str, torch.Tensor] | None = None
    previous_edges: int | None = None
    per_density_metrics: list[dict[str, Any]] = []
    status_path = out / "status.json"
    _atomic_json(status_path, {"seed": seed, "stage": "loaded_inputs_and_rankings",
                               "device": str(device), "cuda_visible_devices": uuid,
                               "elapsed_seconds": time.monotonic() - started})
    for density in DENSITIES:
        density_folder = out / _slug(density)
        density_folder.mkdir(parents=True, exist_ok=True)
        result_path = density_folder / "results.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            if float(existing["density"]) != density:
                raise ValueError(f"density payload mismatch at {density_folder}")
            if not (density_folder / "masks.pt").is_file():
                raise ValueError(f"completed density payload lacks masks: {density_folder}")
            test_rows = existing["records"]
            validation_rows = existing["validation_records"]
            audit_path = density_folder / "control_audit.json"
            if not audit_path.is_file() or not json.loads(audit_path.read_text())["passed"]:
                raise ValueError(f"completed density payload lacks control audit: {density_folder}")
            loaded_masks = torch.load(density_folder / "masks.pt", map_location=device,
                                      weights_only=True)
            if nested_from is not None and previous_edges is not None:
                _nested_audit(nested_from, loaded_masks, previous_edges, int(existing["edges"]))
            nested_from, previous_edges = loaded_masks, int(existing["edges"])
            aggregate_records.extend(test_rows)
            aggregate_validation.extend(validation_rows)
            per_density_metrics.append({"density": density, "edges": int(existing["edges"]),
                                        "slug": _slug(density), "resumed": True,
                                        "control_audit_passed": True})
            continue

        k = _edge_count(density)
        masks, agreement_diagnostics, exact = _extract_density_masks(
            seed, density, k, random_order, function_order, vae_models, device,
            original_masks, transfer_function_mask)
        if nested_from is not None and previous_edges is not None:
            nested = _nested_audit(nested_from, masks, previous_edges, k)
        else:
            nested = None
        if density == 0.20:
            for method, matched in exact.items():
                if not matched:
                    raise AssertionError(f"20% mask equality failed: {method}")
        torch.save({name: value.detach().cpu() for name, value in masks.items()},
                   density_folder / "masks.pt")
        _atomic_json(density_folder / "agreement_diagnostics.json", {
            "seed": seed, "density": density, "edges": k,
            "vae_artifact": str(vae_path), "vae_artifact_sha256": vae_hash,
            "vae_retrained": False, **agreement_diagnostics})
        mask_audit = {"seed": seed, "density": density, "edges": k,
                      "method_edge_counts": {name: value.sum((-1, -2)).detach().cpu().tolist()
                                             for name, value in masks.items()},
                      "binary_masks": True, "exact_20_percent_controls": exact,
                      "fixed_ranking_nested_from_previous_density": nested,
                      "random_tie_policy": "stable descending; flattened index ascending",
                      "function_gradient_tie_policy": "stable descending; flattened index ascending"}
        _atomic_json(density_folder / "mask_audit.json", mask_audit)
        _write_density_protocol(density_folder, seed, density, k, hashes, uuid)

        all_masks = {name: value.to(device) for name, value in original_masks.items()}
        all_masks.update(masks)
        weights_folder = density_folder / "weights"
        validation_weights_folder = density_folder / "validation_weights"
        test_records = _evaluate(
            data, test_costs, all_masks, seed + TEST_TARGET_SEED_OFFSET,
            device, weights_folder)
        validation_records = _evaluate(
            data, validation_costs, all_masks, seed + VALIDATION_TARGET_SEED_OFFSET,
            device, validation_weights_folder)
        test_rows = _augment(test_records, seed=seed, density=density, edges=k, phase="test")
        validation_rows = _augment(validation_records, seed=seed, density=density,
                                   edges=k, phase="validation")
        audit = _control_audit(seed, test_rows)
        _atomic_json(density_folder / "control_audit.json", audit)
        payload = {"seed": seed, "density": density, "edges": k,
                   "records": test_rows, "validation_records": validation_rows}
        _atomic_json(result_path, payload)
        aggregate_records.extend(test_rows)
        aggregate_validation.extend(validation_rows)
        per_density_metrics.append({"density": density, "edges": k, "slug": _slug(density),
                                    "test_records": len(test_rows),
                                    "validation_records": len(validation_rows),
                                    "control_audit_passed": audit["passed"],
                                    "control_max_delta": audit["maximum_absolute_mse_delta"],
                                    "twenty_percent_masks_exact": exact,
                                    "nested_ranking_audit": nested})
        _atomic_json(status_path, {"seed": seed, "stage": "density_complete",
                                   "density": density, "edges": k,
                                   "completed_densities": len(per_density_metrics),
                                   "elapsed_seconds": time.monotonic() - started,
                                   "records": len(aggregate_records),
                                   "validation_records": len(aggregate_validation)})
        print(json.dumps({"seed": seed, "density": density, "edges": k,
                          "test_records": len(test_rows),
                          "validation_records": len(validation_rows),
                          "control_max_delta": audit["maximum_absolute_mse_delta"],
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
        nested_from, previous_edges = masks, k

    # Rebuild complete aggregate payload from every per-density result, which
    # also makes a resumed run produce the same deterministic row ordering.
    aggregate_records = []
    aggregate_validation = []
    for density in DENSITIES:
        payload = json.loads((out / _slug(density) / "results.json").read_text())
        aggregate_records.extend(payload["records"])
        aggregate_validation.extend(payload["validation_records"])
    _atomic_json(out / "results.json", {"seed": seed, "densities": list(DENSITIES),
                                        "records": aggregate_records,
                                        "validation_records": aggregate_validation})
    _atomic_json(out / "density_audit.json", {
        "seed": seed, "densities": per_density_metrics,
        "all_control_audits_passed": all(
            json.loads((out / _slug(rho) / "control_audit.json").read_text())["passed"]
            for rho in DENSITIES),
        "all_mask_audits_present": all((out / _slug(rho) / "mask_audit.json").is_file()
                                        for rho in DENSITIES),
        "fixed_random_and_function_rankings_nested": True,
        "test_records": len(aggregate_records),
        "validation_records": len(aggregate_validation),
        "elapsed_seconds": time.monotonic() - started,
    })
    _atomic_json(status_path, {"seed": seed, "stage": "complete",
                               "densities": len(DENSITIES),
                               "records": len(aggregate_records),
                               "validation_records": len(aggregate_validation),
                               "elapsed_seconds": time.monotonic() - started})
    (out / "COMPLETE").write_text("all density conditions and audits completed\n")


def run_preflight(seed: int, out: Path, hashes: dict[str, str]) -> None:
    """Check all three 20% masks and run a reduced paired evaluator smoke."""
    configure(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("density-sweep preflight requires a GPU")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    original = load_pilot_seed(seed, device)
    banks, original_masks = original["banks"], original["original_masks"]
    function_scores, training_rows, alignment_orders, transfer_function = _load_function_rank(
        seed, banks, device)
    function_order = _stable_orders(function_scores)[0]
    random_scores = _random_rank(seed, banks, device)
    random_order = _stable_orders(random_scores)
    vae_models, vae_path = _load_vae_models(seed, device)
    masks, agreement_diag, exact = _extract_density_masks(
        seed, 0.20, _edge_count(0.20), random_order, function_order,
        vae_models, device, original_masks, transfer_function)
    if exact != {name: True for name in NEW_METHODS}:
        raise AssertionError(f"incomplete exact 20% preflight: {exact}")
    _write_rankings(out, random_scores, random_order, function_scores, function_order,
                    training_rows, alignment_orders, transfer_function)
    torch.save({name: value.detach().cpu() for name, value in masks.items()}, out / "masks_20pct.pt")
    _atomic_json(out / "agreement_diagnostics_20pct.json", {
        "seed": seed, "vae_artifact": str(vae_path), "vae_artifact_sha256": _sha256(vae_path),
        **agreement_diag})

    all_masks = {name: value.to(device) for name, value in original_masks.items()}
    all_masks.update(masks)
    costs = original["protocol"]["task_vectors"]["test"][:1]
    kwargs = {"support_sizes": (32,), "steps": 2, "batch_size": 32,
              "set_size": 5, "validation_sets": 8, "test_sets": 8}
    new_rows = evaluate_masks_batched(
        original["data"], costs, all_masks, seed + TEST_TARGET_SEED_OFFSET,
        device, artifact_dir=out / "smoke_batched_weights",
        initialization_reference_models=REFERENCE_MODELS,
        batch_conditions=8, kernel_mode="reference", **kwargs)
    old_rows = evaluate_masks(
        original["data"], costs, original_masks,
        seed + TEST_TARGET_SEED_OFFSET, device,
        artifact_dir=None, initialization_reference_models=REFERENCE_MODELS,
        **kwargs)
    old_by_key = {(row["task"], row["support_size"], row["method"], row["init"]): row
                  for row in old_rows}
    compare_rows = [row for row in new_rows if row["method"] in ORIGINAL_METHODS]
    deltas = []
    for row in compare_rows:
        key = (row["task"], row["support_size"], row["method"], row["init"])
        previous = old_by_key[key]
        deltas.append({"method": row["method"], "init": row["init"],
                       "batched_mse": float(row["mse"]), "reference_mse": float(previous["mse"]),
                       "absolute_delta": abs(float(row["mse"]) - float(previous["mse"]))})
    max_delta = max(row["absolute_delta"] for row in deltas)
    smoke_audit = {"seed": seed, "task": 0, "support_size": 32,
                   "optimizer_steps": 2, "test_sets": 8,
                   "compared_original_control_rows": len(deltas),
                   "maximum_absolute_mse_delta": max_delta,
                   "threshold": CONTROL_TOLERANCE, "passed": max_delta <= CONTROL_TOLERANCE,
                   "rows": deltas}
    if not smoke_audit["passed"]:
        raise AssertionError(f"batched evaluator preflight failed: {max_delta}")
    _atomic_json(out / "preflight.json", {
        "seed": seed, "density": 0.20, "edges": _edge_count(0.20),
        "source_sha256": hashes, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "exact_masks": exact, "function_gradient_cutoff_unique": True,
        "smoke_control_audit": smoke_audit,
        "agreement_diagnostics": agreement_diag,
        "status": "passed",
    })
    print(json.dumps({"preflight": "passed", "seed": seed,
                      "exact_masks": exact, "smoke_max_delta": max_delta}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.launch:
        hashes = _prepare_root(OUT_ROOT)
        _atomic_json(OUT_ROOT / "protocol.json", _root_protocol(hashes))
        run_cuda_queue(
            "deepsets_vaae.density_sweep", OUT_ROOT, GPU_UUIDS,
            seeds=SEEDS, workers_per_gpu=1,
            env_overrides={"DEEPSETS_EVAL_BATCH_CONDITIONS": "8"})
        missing = [seed for seed in SEEDS if not (OUT_ROOT / f"seed_{seed}" / "COMPLETE").is_file()]
        if missing:
            raise RuntimeError(f"density-sweep workers lack COMPLETE markers: {missing}")
        (OUT_ROOT / "COMPLETE").write_text("all eight seeds and all eleven densities completed\n")
    elif args.preflight:
        if args.seed is None:
            parser.error("--preflight requires --seed")
        hashes = _prepare_root(OUT_ROOT)
        _atomic_json(OUT_ROOT / "protocol.json", _root_protocol(hashes))
        destination = args.out or OUT_ROOT / "preflight" / f"seed_{args.seed}"
        run_preflight(args.seed, destination, hashes)
    else:
        if args.seed is None or args.out is None:
            parser.error("worker mode requires --seed and --out")
        hashes = _prepare_root(OUT_ROOT)
        protocol_path = OUT_ROOT / "protocol.json"
        if not protocol_path.is_file():
            raise FileNotFoundError(f"launch protocol is missing: {protocol_path}")
        frozen_hashes = json.loads(protocol_path.read_text()).get("source_sha256")
        if frozen_hashes != hashes:
            raise ValueError("worker source hashes differ from the frozen launch protocol")
        run_seed(args.seed, args.out, hashes)


if __name__ == "__main__":
    main()
