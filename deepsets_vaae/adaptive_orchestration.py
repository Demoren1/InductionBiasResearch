"""Launch, validate, and globally freeze the adaptive-density experiment.

This module is deliberately separate from the training code.  It treats a
completed worker directory as an untrusted artifact: an aggregate can only be
made after every record, mask and saved effective weight has been checked.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import torch

from . import adaptive_data as data


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/deepsets_vaae/20261001_adaptive_density"
SEEDS = tuple(range(4100, 4108))
SELECTION_METHODS = 111
FINAL_LABELS = (
    "functional_selected", "gnn_selected", "pixelprior_selected", "joint_primary",
    "random_matched_joint", "dense_tuned", "dense_default", "functional_fixed30",
    "gnn_fixed30", "pixelprior_fixed30",
)
SOURCE_STAGE_FILES = ("selection_records.json", "selection_provenance.json", "selection_masks_states.pt")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _seed_dir(root: Path, seed: int) -> Path:
    return root / f"seed_{seed}"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error


def _expected_cardinality(name: str) -> int:
    if name.startswith("dense_lr"):
        return data.FLAT
    marker = "_rho"
    if marker not in name:
        raise ValueError(f"cannot infer density from mask name {name}")
    rho_text = name.split(marker, 1)[1].split("_lr", 1)[0]
    return data.k_for_rho(float(rho_text))


def _expected_selection_names(seed: int) -> set[str]:
    # The complete name grid is deterministic and this also catches a missing
    # logical method even if its records were simply omitted.
    return set(data.selection_masks(seed, "cpu")[0])


def _validate_masks_and_states(path: Path, expected_names: set[str], cardinalities: dict[str, int], *, phase: str) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"cannot read checkpoint artifact {path}: {error}") from error
    masks = payload.get("masks")
    states = payload.get("checkpoints")
    if not isinstance(masks, dict) or set(masks) != expected_names:
        raise ValueError(f"top-level masks do not match expected names in {path}")
    for name, mask in masks.items():
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (4, data.FEATURES, data.HIDDEN):
            raise ValueError(f"bad mask shape for {name}")
        if not torch.isfinite(mask).all() or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError(f"mask {name} is not finite binary")
        counts = mask.reshape(4, -1).sum(1)
        if not bool((counts == cardinalities[name]).all()):
            raise ValueError(f"wrong cardinality for {name}: {counts.tolist()}")
    if not isinstance(states, dict) or not states:
        raise ValueError(f"missing checkpoint states in {path}")
    checked = 0
    for key, state in states.items():
        if not isinstance(state, dict) or not {"weight", "masks", "effective_weight"} <= set(state):
            raise ValueError(f"incomplete state {key}")
        weight, mask, effective = state["weight"], state["masks"], state["effective_weight"]
        if not all(isinstance(x, torch.Tensor) for x in (weight, mask, effective)):
            raise ValueError(f"non-tensor state {key}")
        if weight.shape != mask.shape or effective.shape != weight.shape or weight.ndim != 4:
            raise ValueError(f"incompatible W/M tensors in state {key}")
        if not torch.isfinite(weight).all() or not torch.isfinite(effective).all():
            raise ValueError(f"non-finite W/M state {key}")
        if not torch.allclose(effective, weight * mask, rtol=0, atol=0):
            raise ValueError(f"effective W*M mismatch in state {key}")
        if not bool((effective[mask == 0] == 0).all()):
            raise ValueError(f"masked effective weights not zero in state {key}")
        try:
            group = ast.literal_eval(key)
            budgets = {int(item[1]) for item in group}
        except (ValueError, SyntaxError, TypeError, IndexError):
            raise ValueError(f"unparseable condition group key {key}")
        if phase == "confirmation" and len(budgets) != 1:
            raise ValueError(f"confirmation state group must contain one support budget: {key}")
        budget = next(iter(budgets))
        logical_names = list(masks) if phase == "selection" else [name for name in masks if name.endswith(f"_b{budget}")]
        expected_mask = torch.cat([masks[name] for name in logical_names])
        if mask.shape[1:] != expected_mask.shape or mask.shape[0] != len(group):
            raise ValueError(f"state mask shape does not match condition group {key}")
        if not torch.equal(mask, expected_mask[None].expand_as(mask)):
            raise ValueError(f"state mask differs from its saved logical mask for {key}")
        checked += 1
    return {"mask_count": len(masks), "state_groups": checked}


def _validate_record_grid(records: list[dict[str, Any]], expected_names: set[str], *, phase: str) -> None:
    required = {"phase", "task", "support_size", "method", "init", "rho", "lr", "score_mse", "converged", "status"}
    expected_tasks = range(16 if phase == "selection" else 32)
    budgets = data.BUDGETS
    if phase == "selection":
        names_per_budget = {budget: expected_names for budget in budgets}
        expected_total = 16 * len(budgets) * len(expected_names) * 4
    else:
        names_per_budget = {budget: {f"{label}_b{budget}" for label in FINAL_LABELS} for budget in budgets}
        expected_total = 32 * len(budgets) * len(FINAL_LABELS) * 4
    if len(records) != expected_total:
        raise ValueError(f"{phase} record count {len(records)} != {expected_total}")
    seen: set[tuple[int, int, str, int]] = set()
    for row in records:
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError(f"incomplete {phase} record")
        task, budget, method, init = int(row["task"]), int(row["support_size"]), str(row["method"]), int(row["init"])
        if task not in expected_tasks or budget not in budgets or method not in names_per_budget[budget] or init not in range(4):
            raise ValueError(f"out-of-grid {phase} record {(task, budget, method, init)}")
        if str(row["phase"]) != phase or str(row["status"]) not in {"converged", "max_cap"}:
            raise ValueError(f"invalid {phase} status")
        if bool(row["converged"]) != (row["status"] == "converged"):
            raise ValueError(f"inconsistent convergence status")
        if not np.isfinite(float(row["score_mse"])):
            raise ValueError("non-finite stage score")
        key = (task, budget, method, init)
        if key in seen:
            raise ValueError(f"duplicate record {key}")
        seen.add(key)
    if len(seen) != expected_total:
        raise ValueError("record grid is incomplete")


def _validate_provenance(root: Path, seed: int, phase: str, provenance: dict[str, Any]) -> None:
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        raise ValueError("protocol.json is required before validation")
    protocol = _read_json(protocol_path)
    for rel, expected in protocol.get("input_sha256", {}).items():
        path = ROOT / rel
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"source input hash guard failed: {rel}")
    expected_stage = phase
    if provenance.get("stage") != expected_stage or not provenance.get("row_ids_pairwise_disjoint") or not provenance.get("exact_pixel_pairs_disjoint"):
        raise ValueError(f"invalid {phase} provenance for seed {seed}")
    if set(provenance.get("split_hashes", {})) != ({"selection_train", "selection_checkpoint", "selection_score"} if phase == "selection" else {"confirmation_train", "confirmation_checkpoint", "confirmation_test"}):
        raise ValueError(f"unexpected split hashes in {phase} provenance")
    if phase == "confirmation" and not provenance.get("excluded_selection_split_hashes"):
        raise ValueError("confirmation provenance does not prove selection exclusion")


def validate_stage(root: Path, seed: int, phase: str, *, require_marker: bool = True) -> dict[str, Any]:
    """Validate a single finished worker artifact without changing it."""
    folder = _seed_dir(root, seed)
    marker = folder / f"{phase.upper()}_COMPLETE"
    if require_marker and not marker.is_file():
        raise ValueError(f"missing completion marker {marker}")
    records_path = folder / f"{phase}_records.json"
    provenance_path = folder / f"{phase}_provenance.json"
    state_path = folder / f"{phase}_masks_states.pt"
    if not all(path.is_file() for path in (records_path, provenance_path, state_path)):
        raise ValueError(f"incomplete {phase} artifact for seed {seed}")
    records = _read_json(records_path)
    if not isinstance(records, list):
        raise ValueError("records must be a JSON list")
    if phase == "selection":
        names = _expected_selection_names(seed)
    else:
        names = {f"{label}_b{budget}" for budget in data.BUDGETS for label in FINAL_LABELS}
    _validate_record_grid(records, names, phase=phase)
    _validate_provenance(root, seed, phase, _read_json(provenance_path))
    cardinalities: dict[str, int] = {}
    for name in names:
        rhos = {float(row["rho"]) for row in records if row["method"] == name}
        if len(rhos) != 1:
            raise ValueError(f"method {name} has inconsistent density in records")
        cardinalities[name] = data.k_for_rho(rhos.pop())
    mask_summary = _validate_masks_and_states(state_path, names, cardinalities, phase=phase)
    return {"seed": seed, "phase": phase, "records": len(records), "artifacts": {path.name: _sha(path) for path in (records_path, provenance_path, state_path)}, **mask_summary}


def _source_artifact_manifest(root: Path) -> list[dict[str, Any]]:
    return [validate_stage(root, seed, "selection") for seed in SEEDS]


def _mean_scores(records: Iterable[dict[str, Any]], *, budget: int, method: str) -> tuple[float, dict[str, float], int]:
    rows = [row for row in records if int(row["support_size"]) == budget and row["method"] == method]
    if not rows:
        raise ValueError(f"no source scores for {method} budget {budget}")
    by_seed: dict[str, list[float]] = {}
    for row in rows:
        by_seed.setdefault(str(row["seed"]), []).append(float(row["score_mse"]))
    if set(map(int, by_seed)) != set(SEEDS):
        raise ValueError(f"method {method} is not present for all eight seeds")
    seed_means = {seed: float(np.mean(values)) for seed, values in by_seed.items()}
    return float(np.mean(list(seed_means.values()))), seed_means, len(rows)


def _config(name: str, family: str, rho: float, lr: float, mean: float, seeds: dict[str, float], records: int) -> dict[str, Any]:
    return {"method": name, "family": family, "rho": float(rho), "lr": float(lr), "stage_selection_mean_nmse": mean,
            "seed_means_nmse": seeds, "record_count": records}


def _rank_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    # Error is primary.  Only an exact numerical tie lets smaller K decide.
    return min(candidates, key=lambda row: (row["stage_selection_mean_nmse"], data.k_for_rho(row["rho"]), row["method"]))


def _source_gate(records: list[dict[str, Any]], configs: dict[str, dict[str, Any]], budget: int) -> dict[str, Any]:
    # Phase B reports every listed comparator, therefore a capped source fit
    # for any of them is a real optimization issue, not something to hide
    # behind the primary comparison.
    target_methods = {row["method"] for row in configs.values()}
    rows = [row for row in records if int(row["support_size"]) == budget and row["method"] in target_methods]
    failures = [row for row in rows if not bool(row["converged"])]
    return {"eligible_for_confirmation": not failures, "checked_methods": sorted(target_methods),
            "checked_records": len(rows), "max_cap_records": len(failures),
            "status": "PASS" if not failures else "BLOCKED_UNCONVERGED_SOURCE_FIT"}


def _selection_difference(records: list[dict[str, Any]], budget: int, candidate: str, dense: str) -> dict[str, Any]:
    _, candidate_seed, _ = _mean_scores(records, budget=budget, method=candidate)
    _, dense_seed, _ = _mean_scores(records, budget=budget, method=dense)
    values = np.array([candidate_seed[str(seed)] - dense_seed[str(seed)] for seed in SEEDS], dtype=float)
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    return {"candidate_minus_dense_seedmean_nmse": float(values.mean()), "seed_t95_ci": [float(values.mean() - 2.365 * standard_error), float(values.mean() + 2.365 * standard_error)],
            "seed_count": len(values), "description": "descriptive source-selection comparison; not confirmation evidence"}


def aggregate_selection(root: Path) -> dict[str, Any]:
    """Validate all eight selections, then write one immutable global choice."""
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        raise ValueError("workers must freeze protocol.json before aggregation")
    artifact_manifest = _source_artifact_manifest(root)
    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        for row in _read_json(_seed_dir(root, seed) / "selection_records.json"):
            row = dict(row); row["seed"] = seed; records.append(row)
    by_budget: dict[str, Any] = {}
    source_fit_gates: dict[str, Any] = {}
    for budget in data.BUDGETS:
        dense_candidates = []
        for lr in data.LRS:
            name = f"dense_lr{lr:g}"; mean, seeds, count = _mean_scores(records, budget=budget, method=name)
            dense_candidates.append(_config(name, "dense", 1.0, lr, mean, seeds, count))
        dense = _rank_candidate(dense_candidates)
        family_winners: dict[str, dict[str, Any]] = {}
        for family in ("functional", "gnn", "pixelprior"):
            sparse_candidates = []
            for rho in data.SPARSE_RHOS:
                for lr in data.LRS:
                    name = f"{family}_rho{rho:g}_lr{lr:g}"
                    mean, seeds, count = _mean_scores(records, budget=budget, method=name)
                    sparse_candidates.append(_config(name, family, rho, lr, mean, seeds, count))
            sparse = _rank_candidate(sparse_candidates)
            # A sparse source is allowed into confirmation only if it strictly
            # improves the equally tuned dense baseline.  K resolves ties
            # among sparse candidates only, never a tie with dense.
            family_winners[family] = sparse if sparse["stage_selection_mean_nmse"] < dense["stage_selection_mean_nmse"] else dense
        strict_joint = [row for row in family_winners.values() if row["stage_selection_mean_nmse"] < dense["stage_selection_mean_nmse"]]
        joint = _rank_candidate(strict_joint) if strict_joint else dense
        if joint["rho"] >= 1.0:
            random_matched = dict(dense)
            random_matched["family"] = "dense_alias_for_random"
        else:
            random_name = f"random_rho{joint['rho']:g}_lr{joint['lr']:g}"
            mean, seeds, count = _mean_scores(records, budget=budget, method=random_name)
            random_matched = _config(random_name, "random", joint["rho"], joint["lr"], mean, seeds, count)
        fixed = {}
        for family in ("functional", "gnn", "pixelprior"):
            name = f"{family}_rho0.3_lr0.002"; mean, seeds, count = _mean_scores(records, budget=budget, method=name)
            fixed[family] = _config(name, family, .30, .002, mean, seeds, count)
        config = {
            "functional_selected": family_winners["functional"], "gnn_selected": family_winners["gnn"],
            "pixelprior_selected": family_winners["pixelprior"], "joint_primary": joint,
            "random_matched_joint": random_matched, "dense_tuned": dense,
            "dense_default": next(row for row in dense_candidates if row["lr"] == .002),
            "functional_fixed30": fixed["functional"], "gnn_fixed30": fixed["gnn"], "pixelprior_fixed30": fixed["pixelprior"],
        }
        gate = _source_gate(records, config, budget)
        source_fit_gates[str(budget)] = {
            **gate,
            "joint_minus_dense_descriptive": _selection_difference(records, budget, joint["method"], dense["method"]),
        }
        by_budget[str(budget)] = config
    result = {
        "selection_protocol_sha256": _sha(protocol_path),
        "selection_artifact_manifest": artifact_manifest,
        "selection_artifact_manifest_sha256": hashlib.sha256(json.dumps(artifact_manifest, sort_keys=True).encode()).hexdigest(),
        "selection_rule": "global per budget: a sparse family must strictly beat equally tuned dense mean selection_score across 8 seeds, 16 tasks, 4 initializations; density breaks only exact ties among sparse candidates, otherwise dense is selected",
        "dense_rule": "dense shares the identical three-LR grid and is selected by minimum mean stage score",
        "random_rule": "random is a matched control and never eligible for joint selection",
        "primary_confirmation_comparison": "joint_primary versus dense_tuned at budget 256; predeclared single primary comparison",
        "by_budget": by_budget,
        "source_fit_gates": source_fit_gates,
    }
    existing = root / "selected_density.json"
    if existing.exists():
        current = _read_json(existing)
        if current != result:
            raise ValueError("selected_density.json is immutable and differs from current validated selection")
        return current
    _json(existing, result)
    return result


def confirmation_ready(root: Path, seed: int | None = None) -> bool:
    """Strict phase-B guard: all source workers and their bound aggregate pass."""
    selected_path = root / "selected_density.json"
    protocol_path = root / "protocol.json"
    if not selected_path.is_file() or not protocol_path.is_file():
        return False
    try:
        selected = _read_json(selected_path)
        if selected.get("selection_protocol_sha256") != _sha(protocol_path):
            return False
        manifest = _source_artifact_manifest(root)
        if manifest != selected.get("selection_artifact_manifest"):
            return False
        manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if manifest_hash != selected.get("selection_artifact_manifest_sha256"):
            return False
        if any(not selected["source_fit_gates"][str(budget)]["eligible_for_confirmation"] for budget in data.BUDGETS):
            return False
        return seed is None or seed in SEEDS
    except (ValueError, KeyError, TypeError):
        return False


def _read_gpu_selection(root: Path) -> list[str]:
    payload = _read_json(root / "gpu_selection.json")
    selected = payload.get("selected")
    if not isinstance(selected, list) or len(selected) != len(SEEDS) or len(set(selected)) != len(SEEDS):
        raise ValueError("gpu_selection.json must preserve exactly eight unique initially idle UUIDs")
    if payload.get("memory_considered") is not False:
        raise ValueError("GPU allocation must explicitly ignore existing memory")
    return [str(value) for value in selected]


def launch(root: Path, phase: str, *, python: str, dry_run: bool = False) -> dict[str, Any]:
    """Launch one isolated worker per pre-reserved UUID; never reselect cards."""
    # Freeze once in the parent before any worker is started.  Worker-side
    # freeze is then a verification, so eight processes never race to create
    # a protocol or source snapshot.
    from . import adaptive_density
    adaptive_density.freeze(root)
    if phase == "confirmation" and not confirmation_ready(root):
        raise RuntimeError("confirmation is blocked: global selection artifacts or source-fit gate failed validation")
    devices = _read_gpu_selection(root)
    allocation_path = root / f"{phase}_allocation.json"
    allocation = {"phase": phase, "memory_considered": False, "assignments": []}
    outcomes: list[dict[str, Any]] = []
    pending: list[tuple[int, str, Path, Path, list[str]]] = []
    # Build and validate the complete allocation first.  A mismatching old
    # allocation or a corrupt "complete" worker must fail before a single
    # new process can be started.
    for seed, uuid in zip(SEEDS, devices):
        folder = _seed_dir(root, seed); marker = folder / f"{phase.upper()}_COMPLETE"
        log_path = folder / f"{phase}_worker.log"
        command = [python, "-m", "deepsets_vaae.adaptive_density", "--launch", "--phase", phase, "--seed", str(seed), "--out", str(folder), "--device", "cuda"]
        # Allocation is immutable.  Runtime status is intentionally kept out
        # of it, otherwise a valid completed worker would make a resume look
        # like an allocation mutation.
        allocation["assignments"].append({"seed": seed, "gpu_uuid": uuid, "command": command,
                                          "log": str(log_path.relative_to(root))})
        if marker.is_file():
            validate_stage(root, seed, phase)
            outcomes.append({"seed": seed, "action": "validated_reuse"})
            continue
        outcomes.append({"seed": seed, "action": "launch"})
        pending.append((seed, uuid, folder, log_path, command))
    if allocation_path.exists():
        old = _read_json(allocation_path)
        if old != allocation:
            raise ValueError(f"{allocation_path.name} exists with different assignments; refusing overwrite")
    else:
        _json(allocation_path, allocation)
    processes: list[tuple[int, subprocess.Popen[str], Any]] = []
    for seed, uuid, folder, log_path, command in pending:
        folder.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = uuid
            stream = log_path.open("w")
            processes.append((seed, subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True), stream))
    exits = {}
    for seed, process, stream in processes:
        exits[str(seed)] = process.wait(); stream.close()
    result = {"allocation": allocation, "outcomes": outcomes, "exit_codes": exits, "dry_run": dry_run}
    if exits and any(code != 0 for code in exits.values()):
        raise RuntimeError(f"some {phase} workers failed: {exits}")
    return result


def smoke(root: Path) -> dict[str, Any]:
    """Pure CPU selection checks: quality beats smaller K and dense gets same LR grid."""
    synthetic: list[dict[str, Any]] = []
    for seed in SEEDS:
        for task in range(16):
            for init in range(4):
                base = {"seed": seed, "task": task, "support_size": 256, "init": init, "converged": True, "status": "converged"}
                for method, rho, lr, score in (("functional_rho0.1_lr0.002", .1, .002, .500), ("functional_rho0.3_lr0.002", .3, .002, .400),
                                                ("dense_lr0.0005", 1., .0005, .440), ("dense_lr0.002", 1., .002, .430), ("dense_lr0.005", 1., .005, .450)):
                    synthetic.append({**base, "method": method, "rho": rho, "lr": lr, "score_mse": score})
    quality = _rank_candidate([_config("functional_rho0.1_lr0.002", "functional", .1, .002, *_mean_scores(synthetic, budget=256, method="functional_rho0.1_lr0.002")),
                               _config("functional_rho0.3_lr0.002", "functional", .3, .002, *_mean_scores(synthetic, budget=256, method="functional_rho0.3_lr0.002"))])
    dense = _rank_candidate([_config(name, "dense", 1., lr, *_mean_scores(synthetic, budget=256, method=name)) for name, lr in (("dense_lr0.0005", .0005), ("dense_lr0.002", .002), ("dense_lr0.005", .005))])
    immutable = {"same_lr_grid_dense_endpoint": dense["method"] == "dense_lr0.002", "quality_not_minimum_k": quality["method"] == "functional_rho0.3_lr0.002"}
    result = {**immutable, "status": "PASS" if all(immutable.values()) else "FAIL"}
    _json(root / "smoke_adaptive_orchestration.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--phase", choices=("selection", "confirmation"), default="selection")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default="/home/udeneev-av/miniconda3/envs/ras/bin/python")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(); root = args.out
    actions = sum((args.launch, args.aggregate, args.validate, args.smoke))
    if actions != 1:
        raise SystemExit("choose exactly one of --launch, --aggregate, --validate, --smoke")
    if args.smoke:
        print(json.dumps(smoke(root), sort_keys=True)); return
    if args.aggregate:
        print(json.dumps(aggregate_selection(root), sort_keys=True)); return
    if args.validate:
        seeds = (args.seed,) if args.seed is not None else SEEDS
        print(json.dumps([validate_stage(root, seed, args.phase) for seed in seeds], sort_keys=True)); return
    print(json.dumps(launch(root, args.phase, python=args.python, dry_run=args.dry_run), sort_keys=True))


if __name__ == "__main__":
    main()
