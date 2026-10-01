"""Sharded worker for immutable adaptive-density source recovery v2.

This worker never freezes or mutates the original experiment root and never
opens the phase-B gate.  An external orchestrator owns those decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import adaptive_data as data
from .adaptive_continued_eval import (ContinuedConfig, evaluate_continued, prefix_equivalence_test,
                                      tiny_prefix_tensor_test)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/deepsets_vaae/20261001_adaptive_density/source_recovery_v2"
OLD_ROOT = ROOT / "outputs/deepsets_vaae/20261001_adaptive_density"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _selection_conditions(shard: int) -> list[tuple[int, int]]:
    if shard not in range(8): raise ValueError("selection shard must be 0..7")
    return [(task, budget) for task in range(2 * shard, 2 * shard + 2) for budget in data.BUDGETS]


def _confirmation_conditions(shard: int) -> tuple[list[tuple[int, int]], int]:
    if shard not in range(16): raise ValueError("confirmation shard must be 0..15")
    budget = data.BUDGETS[shard // 4]; first = 8 * (shard % 4)
    return [(task, budget) for task in range(first, first + 8)], budget


def _selected_masks(selection_root: Path, seed: int, device: str) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    selected_path = selection_root / "selected_density.json"
    selected = json.loads(selected_path.read_text())
    all_masks, _ = data.selection_masks(seed, device)
    labels = ("functional_selected", "gnn_selected", "pixelprior_selected", "joint_primary",
              "random_matched_joint", "dense_tuned", "dense_default", "functional_fixed30",
              "gnn_fixed30", "pixelprior_fixed30")
    masks: dict[str, torch.Tensor] = {}; manifest: list[dict[str, Any]] = []
    for budget in data.BUDGETS:
        configs = selected["by_budget"][str(budget)]
        if set(configs) != set(labels): raise ValueError("selected density lacks final methods")
        for label in labels:
            source = configs[label]; name = f"{label}_b{budget}"
            masks[name] = all_masks[source["method"]]
            manifest.append({"method": name, "family": source["family"], "rho": source["rho"], "lr": source["lr"], "budget": budget})
    return masks, manifest


def _assert_wm(artifacts: dict[str, Any]) -> None:
    for location in ("best_states", "last_states"):
        for state in artifacts[location].values():
            if not torch.equal(state["effective_weight"], state["weight"] * state["masks"]):
                raise AssertionError(f"W*M mismatch in {location}")
    for payload in artifacts["step6000"].values():
        state = payload["last_state"]
        if not torch.equal(state["effective_weight"], state["weight"] * state["masks"]):
            raise AssertionError("W*M mismatch at replay boundary")


def _validate(records: list[dict], artifacts: dict[str, Any], phase: str, shard: int) -> None:
    expected = 3552 if phase == "selection" else 320
    if len(records) != expected: raise AssertionError(f"{phase} shard {shard} records {len(records)} != {expected}")
    keys = {(row["task"], row["support_size"], row["method"], row["init"]) for row in records}
    if len(keys) != len(records) or not all(np.isfinite(row["score_mse"]) for row in records):
        raise AssertionError("non-finite or duplicate records")
    if any(row["status"] not in {"converged", "max_cap"} for row in records): raise AssertionError("bad status")
    _assert_wm(artifacts)


def run(phase: str, seed: int, shard: int, out: Path, frozen_root: Path, selection_root: Path, device: str) -> Path:
    """Run exactly one predetermined recovery shard; no external gate is consulted."""
    protocol = frozen_root / "protocol.json"
    if not protocol.is_file(): raise FileNotFoundError("validated frozen protocol is required")
    old_protocol = json.loads(protocol.read_text())
    if phase == "selection":
        conditions = _selection_conditions(shard); splits, provenance = data.selection_data(seed, device)
        masks, manifest = data.selection_masks(seed, device); costs = data.centred_tasks(20261005, 16); score_key = "selection_score"
    else:
        conditions, budget = _confirmation_conditions(shard); splits, provenance = data.confirmation_data(seed, device)
        all_masks, all_manifest = _selected_masks(selection_root, seed, device)
        names = [row["method"] for row in all_manifest if row["budget"] == budget]
        masks = {name: all_masks[name] for name in names}; manifest = [row for row in all_manifest if row["budget"] == budget]
        costs = data.centred_tasks(20261006, 32); score_key = "confirmation_test"
    folder = out / "shards" / phase / f"seed_{seed}"; folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"shard_{shard}"
    if (stem.with_suffix(".COMPLETE")).exists(): raise FileExistsError("validated shard already exists")
    records, artifacts = evaluate_continued(splits, costs, masks, manifest, seed=seed, device=device, phase=phase,
                                            # Always prepare all four budget prefixes with the original RNG draw shapes;
                                            # conditions_override merely selects this shard's fitted rows.
                                            cfg=ContinuedConfig(), score_key=score_key, support_sizes=data.BUDGETS,
                                            conditions_override=conditions)
    _validate(records, artifacts, phase, shard)
    meta = {"schema": "adaptive_density.source_recovery_v2.shard.v1", "phase": phase, "seed": seed, "shard": shard,
            "conditions": conditions, "records": len(records), "source_frozen_protocol": str(protocol),
            "source_frozen_protocol_sha256": _sha(protocol), "source_input_sha256": old_protocol.get("input_sha256", {}),
            "selection_root": str(selection_root) if phase == "confirmation" else None,
            "schedule": artifacts["schedule"], "provenance": provenance,
            "all_selected_controls_must_converge_before_confirmation": phase == "selection"}
    _write_json(stem.with_suffix(".json"), meta)
    temporary = stem.with_suffix(".pt.tmp")
    torch.save({"records": records, "artifacts": artifacts, "meta": meta}, temporary)
    os.replace(temporary, stem.with_suffix(".pt"))
    stem.with_suffix(".COMPLETE").write_text("validated\n")
    return stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("selection", "confirmation"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frozen-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.out.mkdir(parents=True, exist_ok=True)
        payload = {"prefix_schedule_and_floor": prefix_equivalence_test(),
                   "prefix_100_steps_matches_frozen_tensors_and_masks": tiny_prefix_tensor_test(), "gpu_launched": False}
        _write_json(args.out / "smoke_adaptive_continued_worker.json", payload); print(json.dumps(payload)); return
    print(run(args.phase, args.seed, args.shard, args.out, args.frozen_root, args.selection_root, args.device))


if __name__ == "__main__": main()
