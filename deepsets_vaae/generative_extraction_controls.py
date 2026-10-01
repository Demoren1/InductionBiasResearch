"""Source-only extraction controls for the full functional-map generator study.

The controls answer a narrow question: can alignment, logit averaging, or a
pixel marginal prior alone account for a generated-mask result?  They consume
only the canonical 4 x 205 source-training maps and already-completed
source-only GNN samples to make five scores, then
run the same fixed-budget paired target evaluator as the generator masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .core import load_data
from .followup_batched_eval import evaluate_masks_batched
from .followup_common import configure, run_cuda_queue
from .generative_run import canonicalize
from .masks import _hard_topk
from .run import write_json


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/deepsets_vaae/20261001_converged_functional_vae"
GENERATOR_ROOT = ROOT / "outputs/deepsets_vaae/20261001_other_generators"
DEFAULT_OUT = ROOT / "outputs/deepsets_vaae/20261001_generative_extraction_controls"
SEEDS = list(range(4100, 4108))
CONTROL_METHODS = ["functional_mean_small", "functional_mean_large", "functional_vae_small",
                   "functional_vae_large", "raw_vae_large", "random", "dense"]
NEW_METHODS = ["functional_realign_mean", "functional_realign_logit_mean",
               "functional_pixel_marginal", "gnn_sample_agreement",
               "empirical_sample_agreement"]
METHODS = CONTROL_METHODS + NEW_METHODS
TARGET_EDGES = 7526
SOURCE_FILES = ["generative_extraction_controls.py", "generative_run.py", "masks.py", "core.py",
                "followup_batched_eval.py", "followup_common.py", "run.py"]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _save_pt(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _input_manifest() -> dict[str, str]:
    required = [INPUT / "protocol.json"]
    for seed in SEEDS:
        folder = INPUT / f"seed_{seed}"
        required += [folder / "functional/functional_vae_arrays.npz", folder / "masks.pt",
                     folder / "results.json", folder / "data_provenance.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing immutable source inputs: {missing[:4]}")
    return {path.relative_to(INPUT).as_posix(): _sha(path) for path in required}


def _gnn_sample_manifest() -> dict[str, str]:
    paths = [GENERATOR_ROOT / f"seed_{seed}/gnn_flow/{name}"
             for seed in SEEDS for name in ['samples.pt', 'fit.json', 'COMPLETE']]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing completed primary GNN samples: {missing[:4]}")
    return {path.relative_to(GENERATOR_ROOT).as_posix(): _sha(path) for path in paths}


def protocol() -> dict:
    original = _json(INPUT / "protocol.json")
    if original.get("seeds") != SEEDS:
        raise ValueError("unexpected immutable source seed set")
    return {
        "experiment": "Source-only functional-map extraction controls",
        "date_moscow": "2026-10-01",
        "input_root": str(INPUT),
        "input_sha256": _input_manifest(),
        "gnn_sample_root": str(GENERATOR_ROOT),
        "gnn_sample_sha256": _gnn_sample_manifest(),
        "input_protocol_sha256": _sha(INPUT / "protocol.json"),
        "seeds": SEEDS,
        "methods": METHODS,
        "control_methods": CONTROL_METHODS,
        "new_methods": NEW_METHODS,
        "source_scope": {
            "maps": "canonical source FUNCTIONAL training maps only",
            "shape": [4, 205, 784, 32],
            "pooled_train_maps": 820,
            "alignment": "Hungarian each of 820 maps to original pooled functional TRAIN mean",
            "target_data_loaded_after_source_masks": True,
            "target_labels_used_for_extraction": False,
        },
        "scores": {
            "functional_realign_mean": "mean of all 820 canonicalized source maps",
            "functional_realign_logit_mean": "sigmoid(mean(logit(clamp(canonicalized maps, 1e-4))))",
            "functional_pixel_marginal": "original pooled source-map mean over tasks, maps, and 32 hidden columns; repeat the 784-pixel vector across columns",
            "gnn_sample_agreement": "four coherent multi-task agreements over the 4 x 32 saved GNN source samples",
            "empirical_sample_agreement": "same agreement operator over 4 x 32 fixed-seed empirical source-map samples",
        },
        "agreement": {"candidates_per_task": 32, "starts": "task-0 candidates 0..3",
                      "rounds": 20, "selection": "per-task nearest squared-L2 map to current center",
                      "update": "mean of the four selected maps", "reference": "original pooled functional TRAIN mean"},
        "masking": {"hard_top_k": TARGET_EDGES, "replicas": 4,
                    "new_mask_shape": [4, 784, 32], "all_non_dense_masks_have_exact_k": True},
        "target_eval": {"support_sizes": [256], "steps": 800, "batch_size": 32, "set_size": 5,
                        "validation_sets": 128, "test_sets": 512, "batch_conditions": 8,
                        "kernel_mode": "reference", "initialization_reference_models": 20,
                        "gradient_denominator": 20, "old_seed_offset": 100000,
                        "fresh_seed_offset": 300000},
        "paired_replay": {"original_controls": CONTROL_METHODS, "mse_tolerance": 1e-4,
                          "reference_rows": "immutable converged VAE study, budget 256 only"},
        "comparison_status": "exploratory; old and fresh task vectors were inspected previously",
        "adaptive_sparsity": "deferred; K remains fixed at 7,526 (30%). Future K selection must be source-only or nested validation.",
        "task_vectors": original["task_vectors"],
    }


def freeze_sources(out: Path, spec: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    hashes = {}
    for name in SOURCE_FILES:
        source = ROOT / "deepsets_vaae" / name
        payload = source.read_bytes()
        destination = snapshot / name
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError(f"frozen source differs: {name}")
        destination.write_bytes(payload)
        hashes[name] = _sha(source)
    spec["source_sha256"] = hashes
    destination = out / "protocol.json"
    if destination.exists() and _json(destination) != spec:
        raise ValueError("existing protocol differs; choose a fresh output root")
    write_json(destination, spec)


def _verify_frozen(out: Path, spec: dict, seed: int) -> None:
    if _sha(INPUT / "protocol.json") != spec["input_protocol_sha256"]:
        raise ValueError("immutable source protocol changed")
    for name, expected in spec["source_sha256"].items():
        current = ROOT / "deepsets_vaae" / name
        snapshot = out.parent / "source_snapshot" / name
        if _sha(current) != expected or _sha(snapshot) != expected:
            raise ValueError(f"production or snapshot helper changed: {name}")
    for name, expected in spec["input_sha256"].items():
        if name.startswith(f"seed_{seed}/") and _sha(INPUT / name) != expected:
            raise ValueError(f"immutable source input changed: {name}")
    for gnn_name, expected in spec['gnn_sample_sha256'].items():
        if gnn_name.startswith(f'seed_{seed}/') and _sha(GENERATOR_ROOT / gnn_name) != expected:
            raise ValueError("completed primary GNN sample input changed")


def _load_source_train(seed: int, device: torch.device) -> torch.Tensor:
    path = INPUT / f"seed_{seed}/functional/functional_vae_arrays.npz"
    with np.load(path) as arrays:
        train = torch.as_tensor(arrays["function_train_aligned"], device=device)
    if train.shape != (4, 205, 784, 32):
        raise ValueError(f"invalid source training map shape: {tuple(train.shape)}")
    if not torch.isfinite(train).all() or not ((train >= 0).all() and (train <= 1).all()):
        raise ValueError("source functional maps must be finite probabilities")
    return train.float()


def _four_replicas(score: torch.Tensor) -> torch.Tensor:
    if score.shape != (784, 32) or not torch.isfinite(score).all():
        raise ValueError(f"invalid source score shape/value: {tuple(score.shape)}")
    hard = _hard_topk(score.reshape(1, -1), TARGET_EDGES).reshape(1, 784, 32)
    return hard.expand(4, -1, -1).clone()


def _agreement_centers(candidates: torch.Tensor, *, rounds: int = 20) -> tuple[torch.Tensor, dict[str, Tensor]]:
    """Find four coherent cross-task centers without using target information.

    For each task-0 start, all four source tasks choose their nearest complete
    map to the current center, then the center becomes the mean of those four
    selections.  This retains joint edge configurations that a marginal mean
    can discard.
    """
    if candidates.shape != (4, 32, 784, 32) or not torch.isfinite(candidates).all():
        raise ValueError(f"agreement candidates must be finite [4,32,784,32], got {tuple(candidates.shape)}")
    centers = candidates[0, :4].clone()
    history = [centers.clone()]
    indices, objectives = [], []
    for _ in range(rounds):
        # [starts, tasks, candidates], with the full 25,088-coordinate L2
        # score computed in one batched operation.
        distance = (candidates[None] - centers[:, None, None]).square().mean((-1, -2))
        choice = distance.argmin(dim=-1)  # [starts, tasks]
        chosen = torch.stack([candidates[task, choice[:, task]] for task in range(4)], dim=1)
        indices.append(choice)
        objectives.append(distance.gather(2, choice[:, :, None]).squeeze(-1).sum(1))
        centers = chosen.mean(1)
        history.append(centers.clone())
    details = {"initial_indices": indices[0], "final_indices": indices[-1],
               "objective_history": torch.stack(objectives, dim=1),
               "center_history": torch.stack(history, dim=1)}
    return centers, details


def _hard_centers(centers: torch.Tensor) -> torch.Tensor:
    if centers.shape != (4, 784, 32):
        raise ValueError(f"agreement centers must be [4,784,32], got {tuple(centers.shape)}")
    return _hard_topk(centers.reshape(4, -1), TARGET_EDGES).reshape_as(centers)


def _gnn_candidates(seed: int, reference: torch.Tensor, device: torch.device) -> torch.Tensor:
    path = GENERATOR_ROOT / f"seed_{seed}/gnn_flow/samples.pt"
    fit = _json(path.parent / 'fit.json')
    if not (path.parent / 'COMPLETE').is_file() or fit.get('experiment_seed') != seed or fit.get('converged') is not True:
        raise ValueError('GNN candidates must come from the completed converged source fit')
    saved = torch.load(path, map_location=device, weights_only=True)
    candidates = torch.as_tensor(saved.get("samples"), device=device, dtype=torch.float32)
    if candidates.shape != (4, 32, 784, 32):
        raise ValueError(f"invalid primary GNN sample shape: {tuple(candidates.shape)}")
    if not torch.isfinite(candidates).all() or not ((candidates >= 0).all() and (candidates <= 1).all()):
        raise ValueError("primary GNN samples must be finite probabilities")
    # The frozen primary source-only program canonicalized against this same
    # pooled train mean; its input manifest identifies the unchanged arrays.
    return candidates


@torch.no_grad()
def build_source_controls(seed: int, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Construct all source-only scores before any target data is requested."""
    train = _load_source_train(seed, device)
    original_pooled_mean = train.mean((0, 1))
    aligned, orders = canonicalize(train, original_pooled_mean)
    scores = {
        "functional_realign_mean": aligned.mean((0, 1)),
        "functional_realign_logit_mean": torch.sigmoid(
            torch.logit(aligned.clamp(1e-4, 1 - 1e-4)).mean((0, 1))
        ),
        "functional_pixel_marginal": train.mean((0, 1, 3))[:, None].expand(784, 32).clone(),
    }
    masks = {name: _four_replicas(score) for name, score in scores.items()}
    gnn_candidates = _gnn_candidates(seed, original_pooled_mean, device)
    empirical_generator = torch.Generator(device=device).manual_seed(seed + 900000)
    empirical_indices = torch.randperm(205, device=device, generator=empirical_generator)[:32]
    empirical_candidates, empirical_orders = canonicalize(train[:, empirical_indices], original_pooled_mean)
    agreement_details = {}
    for name, candidates in (("gnn_sample_agreement", gnn_candidates),
                             ("empirical_sample_agreement", empirical_candidates)):
        centers, details = _agreement_centers(candidates)
        masks[name] = _hard_centers(centers)
        agreement_details[name] = {
            **details, "candidate_sha256": hashlib.sha256(
                candidates.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            "center_sha256": hashlib.sha256(centers.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        }
    diagnostics = {
        "seed": seed,
        "source_train_shape": list(train.shape),
        "source_train_sha256": hashlib.sha256(train.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "original_pooled_mean_sha256": hashlib.sha256(
            original_pooled_mean.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "canonicalized_train_sha256": hashlib.sha256(
            aligned.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "canonicalization_orders_sha256": hashlib.sha256(
            orders.contiguous().numpy().tobytes()).hexdigest(),
        "score_sha256": {name: hashlib.sha256(score.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                         for name, score in scores.items()},
        "scores": {name: score.detach().cpu() for name, score in scores.items()},
        "canonicalization_orders": orders,
        "original_pooled_mean": original_pooled_mean.detach().cpu(),
        "empirical_indices": empirical_indices.detach().cpu(),
        "empirical_canonicalization_orders": empirical_orders,
        "agreement": {name: {key: value.detach().cpu() if isinstance(value, Tensor) else value
                              for key, value in details.items()}
                      for name, details in agreement_details.items()},
    }
    return masks, diagnostics


def _validate_masks(masks: dict[str, torch.Tensor]) -> None:
    if list(masks) != METHODS:
        raise ValueError(f"wrong method ordering: {list(masks)}")
    for name, value in masks.items():
        value = torch.as_tensor(value)
        expected = 784 * 32 if name == "dense" else TARGET_EDGES
        if value.shape != (4, 784, 32) or not torch.all((value == 0) | (value == 1)):
            raise ValueError(f"{name}: mask must be binary [4,784,32]")
        if not torch.all(value.sum((1, 2)) == expected):
            raise ValueError(f"{name}: wrong hard-K cardinality")


def _validate_records(rows: list[dict]) -> None:
    expected = {(task, 256, method, init) for task in range(8) for method in METHODS for init in range(4)}
    seen = {(int(row["task"]), int(row["support_size"]), str(row["method"]), int(row["init"])) for row in rows}
    if len(rows) != len(expected) or seen != expected:
        raise ValueError(f"target record coverage mismatch: {len(rows)} rows, expected {len(expected)}")
    for row in rows:
        for name, value in row.items():
            if isinstance(value, (float, int)) and not math.isfinite(float(value)):
                raise ValueError(f"non-finite target record value {name}")


def _checkpoint_keys(task: int) -> set[tuple[int, int, str, int]]:
    return {(task, 256, method, init) for method in METHODS for init in range(4)}


def _validate_checkpoints(folder: Path, seed_masks: dict[str, torch.Tensor]) -> list[dict]:
    expected = {f"target_task{task}_budget256.pt" for task in range(8)}
    present = {path.name for path in folder.glob("*.pt")}
    if present != expected:
        raise ValueError(f"checkpoint coverage mismatch in {folder}: {sorted(present ^ expected)}")
    expected_names = [method for method in METHODS for _ in range(4)]
    expected_replicas = list(range(4)) * len(METHODS)
    expected_masks = torch.cat([torch.as_tensor(seed_masks[name]) for name in METHODS], dim=0)
    records: list[dict] = []
    for name in expected:
        saved = torch.load(folder / name, map_location="cpu", weights_only=False)
        task = int(name.removeprefix("target_task").removesuffix("_budget256.pt"))
        if saved.get("task") != task or saved.get("support_size") != 256:
            raise ValueError(f"checkpoint task/budget mismatch: {folder / name}")
        if saved.get("method_names") != expected_names or saved.get("replica_indices") != expected_replicas:
            raise ValueError(f"invalid method checkpoint: {folder / name}")
        method_rows = saved.get("methods", [])
        expected_method_rows = [{"model_index": index, "method": method, "init": init}
                                for index, (method, init) in enumerate(zip(expected_names, expected_replicas))]
        if method_rows != expected_method_rows:
            raise ValueError(f"invalid method-index mapping: {folder / name}")
        checkpoint_records = saved.get("records", [])
        keys = {(int(row["task"]), int(row["support_size"]), str(row["method"]), int(row["init"]))
                for row in checkpoint_records}
        if len(checkpoint_records) != len(expected_names) or keys != _checkpoint_keys(task):
            raise ValueError(f"invalid record/replica coverage: {folder / name}")
        masks = saved.get("masks")
        weight, effective = saved.get("weight"), saved.get("effective_weight")
        if masks is None or weight is None or effective is None or tuple(masks.shape) != (len(expected_names), 784, 32):
            raise ValueError(f"invalid checkpoint masks: {folder / name}")
        if not torch.equal(masks, expected_masks):
            raise ValueError(f"checkpoint masks differ from saved source masks: {folder / name}")
        if tuple(weight.shape) != tuple(masks.shape) or tuple(effective.shape) != tuple(masks.shape):
            raise ValueError(f"invalid checkpoint weight shape: {folder / name}")
        if not torch.equal(effective, weight * masks) or not torch.equal(effective[masks == 0], torch.zeros_like(effective[masks == 0])):
            raise ValueError(f"W*M is not exact in checkpoint: {folder / name}")
        for tensor_name in ("weight", "effective_weight", "bias", "readout", "per_image_offset"):
            value = saved.get(tensor_name)
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise ValueError(f"non-finite/missing checkpoint tensor {tensor_name}: {folder / name}")
        for row in checkpoint_records:
            for field, value in row.items():
                if isinstance(value, (float, int)) and not math.isfinite(float(value)):
                    raise ValueError(f"non-finite checkpoint metric {field}: {folder / name}")
        records.extend(checkpoint_records)
    _validate_records(records)
    return sorted(records, key=lambda row: (row["task"], row["support_size"], row["method"], row["init"]))


def _audit_controls(rows: list[dict], reference_rows: list[dict]) -> dict:
    key = lambda row: (int(row["task"]), int(row["support_size"]), str(row["method"]), int(row["init"]))
    actual = {key(row): row for row in rows if row["method"] in CONTROL_METHODS}
    reference = {key(row): row for row in reference_rows
                 if row["method"] in CONTROL_METHODS and int(row["support_size"]) == 256}
    if set(actual) != set(reference):
        raise ValueError("original-control replay key coverage differs")
    delta = [abs(float(actual[name]["mse"]) - float(reference[name]["mse"])) for name in actual]
    maximum = max(delta) if delta else float("inf")
    return {"records": len(delta), "max_mse_delta": maximum, "tolerance": 1e-4,
            "passed": maximum <= 1e-4}


def worker(out: Path, seed: int) -> None:
    configure(seed)
    spec = _json(out.parent / "protocol.json")
    _verify_frozen(out, spec, seed)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE").is_file():
        completed = _json(out / "results.json")
        if len(completed.get("records", [])) == 384 and len(completed.get("fresh_records", [])) == 384:
            return
        raise ValueError("existing COMPLETE seed lacks required records")
    started = time.monotonic()
    write_json(out / "protocol.json", spec | {"seed": seed,
                                                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    write_json(out / "status.json", {"seed": seed, "stage": "source_controls_before_target_data"})
    # This entire block is source-only; do not move load_data above it.
    new_masks, diagnostics = build_source_controls(seed, device)
    original_masks = torch.load(INPUT / f"seed_{seed}/masks.pt", map_location="cpu", weights_only=True)
    if list(original_masks) != CONTROL_METHODS:
        raise ValueError("immutable original control mask order mismatch")
    masks = {name: torch.as_tensor(original_masks[name]).cpu() for name in CONTROL_METHODS}
    masks.update({name: value.cpu() for name, value in new_masks.items()})
    _validate_masks(masks)
    _save_pt(out / "masks.pt", masks)
    _save_pt(out / "scores.pt", diagnostics)
    write_json(out / "source_controls.json", {
        key: value for key, value in diagnostics.items() if key not in
        {"scores", "canonicalization_orders", "original_pooled_mean", "empirical_indices",
         "empirical_canonicalization_orders", "agreement"}
    })

    # Target data enters only after masks/scores have been frozen above.
    write_json(out / "status.json", {"seed": seed, "stage": "loading_target_data"})
    data = load_data(ROOT / "datasets/mnist8m", seed, device, per_digit_train=1000,
                     per_digit_validation=300, per_digit_test=300)
    provenance = _json(INPUT / f"seed_{seed}/data_provenance.json")
    if data.get("split_hashes") != provenance.get("split_hashes") or data.get("row_ids_pairwise_disjoint") is not True:
        raise ValueError("target data split provenance differs from immutable study")
    shutil.copyfile(INPUT / f"seed_{seed}/data_provenance.json", out / "data_provenance.json")
    evaluation = spec["target_eval"]
    kw = dict(support_sizes=(256,), steps=800, batch_size=32, set_size=5, validation_sets=128,
              test_sets=512, batch_conditions=8, kernel_mode="reference",
              initialization_reference_models=20)
    write_json(out / "status.json", {"seed": seed, "stage": "target_old"})
    old = evaluate_masks_batched(data, spec["task_vectors"]["test"], masks, seed + 100000, device,
                                 artifact_dir=out / "weights", **kw)
    write_json(out / "status.json", {"seed": seed, "stage": "target_fresh"})
    fresh = evaluate_masks_batched(data, spec["task_vectors"]["fresh_test"], masks, seed + 300000, device,
                                   artifact_dir=out / "fresh_weights", **kw)
    _validate_records(old)
    _validate_records(fresh)
    _validate_checkpoints(out / "weights", masks)
    _validate_checkpoints(out / "fresh_weights", masks)
    reference = _json(INPUT / f"seed_{seed}/results.json")
    audit = {"records": _audit_controls(old, reference["records"]),
             "fresh_records": _audit_controls(fresh, reference["fresh_records"])}
    if not all(value["passed"] for value in audit.values()):
        raise AssertionError(f"original control replay failed: {audit}")
    write_json(out / "control_audit.json", audit)
    write_json(out / "results.json", {"seed": seed, "records": old, "fresh_records": fresh,
                                        "control_audit": audit, "elapsed_seconds": time.monotonic() - started,
                                        "source_controls_constructed_before_target_data": True,
                                        "target_evaluation": evaluation})
    write_json(out / "status.json", {"seed": seed, "stage": "complete", "records": len(old),
                                       "fresh_records": len(fresh), "elapsed_seconds": time.monotonic() - started})
    (out / "COMPLETE").write_text("complete\n")


def launch(out: Path) -> None:
    spec = protocol()
    freeze_sources(out, spec)
    prior_path = GENERATOR_ROOT / "gpu_selection.json"
    prior = _json(prior_path)
    selected = list(prior.get("selected", []))
    if len(selected) != 8 or len(set(selected)) != 8:
        raise ValueError("other-generators run does not contain eight unique GPU UUIDs")
    observation = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
                                  "--format=csv,noheader,nounits"], check=True, text=True,
                                 capture_output=True).stdout
    write_json(out / "gpu_selection.json", {
        "selected": selected, "selection_source": str(prior_path), "selection_source_sha256": _sha(prior_path),
        "current_observation": observation, "reuse_existing_generator_allocation": True,
        "utilization_idle_required": False, "memory_considered": False,
        "reason": "user explicitly authorized overlap with the active generator run; memory ignored",
    })
    run_cuda_queue("deepsets_vaae.generative_extraction_controls", out, selected, seeds=SEEDS,
                   workers_per_gpu=1, env_overrides={"DEEPSETS_EVAL_BATCH_CONDITIONS": "8"})
    (out / "COMPLETE").write_text("all8 complete\n")


def _verify_original_snapshot(root: Path, spec: dict) -> dict[str, str]:
    """Validate the failed-run snapshot without comparing it to repair code."""
    observed = {}
    for name, expected in spec["source_sha256"].items():
        snapshot = root / "source_snapshot" / name
        if not snapshot.is_file() or (actual := _sha(snapshot)) != expected:
            raise ValueError(f"original frozen source snapshot hash mismatch: {name}")
        observed[name] = actual
    if _sha(INPUT / "protocol.json") != spec["input_protocol_sha256"]:
        raise ValueError("immutable source protocol changed since failed run")
    for name, expected in spec["input_sha256"].items():
        if _sha(INPUT / name) != expected:
            raise ValueError(f"immutable source input changed since failed run: {name}")
    for name, expected in spec["gnn_sample_sha256"].items():
        if _sha(GENERATOR_ROOT / name) != expected:
            raise ValueError(f"immutable GNN sample input changed since failed run: {name}")
    return observed


def _snapshot_recovery_source(root: Path) -> dict[str, str]:
    """Record the corrected validator separately; never overwrite source_snapshot."""
    snapshot = root / "recovery_source_snapshot"
    snapshot.mkdir(exist_ok=True)
    source = Path(__file__)
    destination = snapshot / source.name
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    # Keep any earlier recovery attempt intact.  A content-addressed child is
    # necessary because the first verifier pass predated the final COMPLETE
    # marker fix; neither it nor the original failed-run snapshot is replaced.
    versioned = snapshot / digest / source.name
    versioned.parent.mkdir(exist_ok=True)
    if versioned.exists() and versioned.read_bytes() != payload:
        raise ValueError("content-addressed recovery source snapshot differs")
    versioned.write_bytes(payload)
    return {versioned.relative_to(root).as_posix(): _sha(versioned)}


def _validate_seed_recovery_artifacts(root: Path, seed: int, spec: dict) -> tuple[list[dict], list[dict], dict]:
    folder = root / f"seed_{seed}"
    expected_seed_protocol = spec | {"seed": seed}
    seed_protocol = _json(folder / "protocol.json")
    comparable = {key: value for key, value in seed_protocol.items() if key != "cuda_visible_devices"}
    if comparable != expected_seed_protocol:
        raise ValueError(f"seed protocol differs from original root protocol: {folder}")
    source_masks = torch.load(folder / "masks.pt", map_location="cpu", weights_only=True)
    _validate_masks(source_masks)
    for required in (folder / "scores.pt", folder / "source_controls.json", folder / "data_provenance.json",
                     folder / "weights", folder / "fresh_weights"):
        if not required.exists():
            raise FileNotFoundError(f"missing completed artifact: {required}")
    provenance = folder / "data_provenance.json"
    reference_provenance = INPUT / f"seed_{seed}/data_provenance.json"
    if _sha(provenance) != _sha(reference_provenance) or _sha(provenance) != spec["input_sha256"][
            f"seed_{seed}/data_provenance.json"]:
        raise ValueError(f"data provenance mismatch for seed {seed}")
    old = _validate_checkpoints(folder / "weights", source_masks)
    fresh = _validate_checkpoints(folder / "fresh_weights", source_masks)
    reference = _json(INPUT / f"seed_{seed}/results.json")
    audit = {"records": _audit_controls(old, reference["records"]),
             "fresh_records": _audit_controls(fresh, reference["fresh_records"])}
    if not all(part["passed"] for part in audit.values()):
        raise AssertionError(f"original-control replay failed for seed {seed}: {audit}")
    return old, fresh, audit


def recover(root: Path) -> None:
    """Finalize a verifier-only recovery from saved checkpoints.

    The original workers completed both target evaluations and then failed in a
    checkpoint validator that expected one row per method instead of four
    replicas.  This command reads existing artifacts only; it neither loads
    target data nor calls the evaluator.  It writes reconstructed result JSON,
    recovery diagnostics, and COMPLETE markers only after every check passes.
    """
    root = root.resolve()
    spec = _json(root / "protocol.json")
    original_hashes = _verify_original_snapshot(root, spec)
    recovery_hashes = _snapshot_recovery_source(root)
    recovery_dir = root / "recovery"
    recovery_dir.mkdir(exist_ok=True)
    before = {str(path.relative_to(root)): _sha(path)
              for seed in SEEDS for split in ("weights", "fresh_weights")
              for path in sorted((root / f"seed_{seed}" / split).glob("*.pt"))}
    if len(before) != len(SEEDS) * 2 * 8:
        raise ValueError(f"expected 128 existing checkpoints before recovery, found {len(before)}")
    seed_diagnostics = {}
    for seed in SEEDS:
        old, fresh, audit = _validate_seed_recovery_artifacts(root, seed, spec)
        folder = root / f"seed_{seed}"
        result = {"seed": seed, "records": old, "fresh_records": fresh, "control_audit": audit,
                  "source_controls_constructed_before_target_data": True,
                  "target_evaluation": spec["target_eval"],
                  "recovered_from_saved_checkpoints": True,
                  "recovery_note": "No target data, target evaluator, training, or checkpoint was rerun."}
        existing = folder / "results.json"
        if existing.exists() and _json(existing) != result:
            raise ValueError(f"existing result differs from checkpoint-derived result: {existing}")
        if not existing.exists():
            write_json(existing, result)
        audit_path = folder / "control_audit.json"
        if audit_path.exists() and _json(audit_path) != audit:
            raise ValueError(f"existing control audit differs: {audit_path}")
        if not audit_path.exists():
            write_json(audit_path, audit)
        diagnostic = {"seed": seed, "status": "passed", "records": len(old), "fresh_records": len(fresh),
                      "control_audit": audit, "validator_failure":
                      "checkpoint validator incorrectly expected 12 method rows; saved checkpoints correctly contain 12 methods × 4 paired replicas = 48 rows",
                      "work_repeated": False}
        write_json(recovery_dir / f"seed_{seed}.json", diagnostic)
        # Preserve failed-worker status.json and run.log; recovery has its own status record.
        write_json(folder / "recovery_status.json", diagnostic)
        seed_diagnostics[str(seed)] = diagnostic
    after = {str(path.relative_to(root)): _sha(path)
             for seed in SEEDS for split in ("weights", "fresh_weights")
             for path in sorted((root / f"seed_{seed}" / split).glob("*.pt"))}
    if before != after:
        raise AssertionError("a saved checkpoint changed during verifier-only recovery")
    # All eight seeds have now passed every artifact, numeric, and replay
    # check.  Only at this point may their previous failed-validator state be
    # upgraded to COMPLETE; original status.json and run.log remain intact.
    for seed in SEEDS:
        (root / f"seed_{seed}" / "COMPLETE").write_text("complete\n")
    summary = {"status": "passed", "mode": "verifier_only_recovery", "seeds": SEEDS,
               "original_snapshot_sha256": original_hashes, "recovery_source_sha256": recovery_hashes,
               "checkpoint_count": len(before), "checkpoint_sha256_unchanged": True,
               "failure_cause": "validator assumed one checkpoint row per method instead of four paired replicas",
               "no_training_or_evaluation_repeated": True, "seed_diagnostics": seed_diagnostics}
    write_json(root / "recovery.json", summary)
    (recovery_dir / "COMPLETE").write_text("all8 verifier-only recovery checks passed\n")
    (root / "COMPLETE").write_text("all8 recovered from saved validated checkpoints\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    if args.launch and args.recover:
        parser.error("--launch and --recover are mutually exclusive")
    if args.recover:
        recover(args.out)
    elif args.launch:
        launch(args.out)
    else:
        if args.seed is None:
            parser.error("--seed is required for a worker")
        worker(args.out, args.seed)


if __name__ == "__main__":
    main()
