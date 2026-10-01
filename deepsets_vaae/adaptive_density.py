"""Two-stage adaptive density experiment; production is intentionally opt-in."""
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
from .adaptive_eval import (EvalConfig, evaluate_adaptive, logical_order_mapping_test,
                            pairing_and_freeze_test, toy_adam_lr_test)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/deepsets_vaae/20261001_adaptive_density"
SOURCE_FILES = ("adaptive_density.py", "adaptive_data.py", "adaptive_eval.py", "adaptive_orchestration.py", "core.py")


def _json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _root(out: Path) -> Path:
    return out if out.name == "20261001_adaptive_density" else out.parent


def freeze(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "source_snapshot"; snapshot.mkdir(exist_ok=True)
    sources = {}
    for name in SOURCE_FILES:
        source = ROOT / "deepsets_vaae" / name; target = snapshot / name
        if target.exists() and target.read_bytes() != source.read_bytes(): raise ValueError("frozen adaptive source differs")
        if not target.exists():
            temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
            temporary.write_bytes(source.read_bytes())
            os.replace(temporary, target)
        # A concurrent first writer is acceptable only when it wrote identical bytes.
        if target.read_bytes() != source.read_bytes(): raise ValueError("concurrent frozen source differs")
        sources[name] = _sha(source)
    inputs = {}
    for seed in range(4100, 4108):
        for path in (data.FUNCTIONAL_ROOT / f"seed_{seed}/functional/functional_vae_arrays.npz",
                     data.FUNCTIONAL_ROOT / f"seed_{seed}/data_provenance.json",
                     data.GNN_ROOT / f"seed_{seed}/gnn_flow/samples.pt"):
            inputs[str(path.relative_to(ROOT))] = _sha(path)
    selection = data.centred_tasks(20261005, 16); confirmation = data.centred_tasks(20261006, 32)
    for path, values in ((root / "selection_tasks.npy", selection), (root / "confirmation_tasks.npy", confirmation)):
        if path.exists() and not np.array_equal(np.load(path), values): raise ValueError("frozen task vectors differ")
        if not path.exists():
            temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npy")
            np.save(temporary, values)
            os.replace(temporary, path)
        if not np.array_equal(np.load(path), values): raise ValueError("concurrent frozen task vectors differ")
    protocol = {"experiment": "adaptive density selection then independent confirmation", "seeds": list(range(4100, 4108)),
                "source_files_sha256": sources, "input_sha256": inputs, "selection_task_rng": 20261005,
                "confirmation_task_rng": 20261006, "selection_tasks": 16, "confirmation_tasks": 32,
                "rhos": list(data.RHOS), "sparse_rhos": list(data.SPARSE_RHOS), "lrs": list(data.LRS),
                "budgets": list(data.BUDGETS), "selection_methods": "3 families x 9 sparse densities x 3 LR, random x 9 x 3 LR, dense x 3 LR",
                "dense_endpoint": "one all-on mask per LR, aliases rho=1 across source families", "kernel": "TF32 FP32 bmm",
                "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "python": os.sys.version},
                "limitation": "Exact pixel and row exclusions are enforced; augmentation-group handwriting identity unavailable."}
    existing = root / "protocol.json"
    if existing.exists():
        if json.loads(existing.read_text()) != protocol: raise ValueError("existing protocol differs; choose a fresh output root")
    else:
        _json(existing, protocol)
    return protocol


def _seed_dir(out: Path, seed: int) -> Path:
    return out if out.name == f"seed_{seed}" else out / f"seed_{seed}"


def _validate_stage(records: list[dict], artifacts: dict, masks: dict[str, torch.Tensor], phase: str) -> None:
    """Reject incomplete/non-finite output before a COMPLETE sentinel is written."""
    if not records: raise AssertionError("stage has no records")
    required = {"phase", "task", "support_size", "method", "init", "family", "rho", "lr",
                "score_mse", "checkpoint_mse", "converged", "status"}
    if any(not required.issubset(row) for row in records): raise AssertionError("record schema incomplete")
    for row in records:
        if not np.isfinite([row["score_mse"], row["checkpoint_mse"]]).all(): raise AssertionError("non-finite stage metric")
        if row["status"] not in {"converged", "max_cap"}: raise AssertionError("unknown convergence status")
    tasks = {row["task"] for row in records}; budgets = {row["support_size"] for row in records}
    expected_tasks = set(range(16 if phase == "selection" else 32))
    if tasks != expected_tasks or budgets != set(data.BUDGETS): raise AssertionError("unexpected task or budget coverage")
    method_inits_by_budget = {budget: {(row["method"], row["init"]) for row in records if row["support_size"] == budget}
                              for budget in budgets}
    expected = {(task, budget, method, init) for task in tasks for budget in budgets
                for method, init in method_inits_by_budget[budget]}
    observed = {(row["task"], row["support_size"], row["method"], row["init"]) for row in records}
    if observed != expected or len(observed) != len(records): raise AssertionError("record coverage is not rectangular/exact")
    for state in artifacts["states"].values():
        for key in ("masks", "weight", "effective_weight"):
            if key not in state or not torch.isfinite(state[key]).all(): raise AssertionError(f"invalid checkpoint {key}")
        if not bool(((state["masks"] == 0) | (state["masks"] == 1)).all()): raise AssertionError("non-binary checkpoint mask")
        if not torch.equal(state["effective_weight"], state["weight"] * state["masks"]): raise AssertionError("W*M checkpoint differs from exact product")
    for mask in masks.values():
        if mask.shape != (4, data.FEATURES, data.HIDDEN) or not bool(torch.isfinite(mask).all()):
            raise AssertionError("invalid saved mask")


def _save_stage(folder: Path, phase: str, records: list[dict], provenance: dict, artifacts: dict, masks: dict[str, torch.Tensor]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / f"{phase.upper()}_COMPLETE").exists(): raise FileExistsError(f"{phase} already complete; refusing overwrite")
    _validate_stage(records, artifacts, masks, phase)
    _json(folder / f"{phase}_records.json", records); _json(folder / f"{phase}_provenance.json", provenance)
    np.savez_compressed(folder / f"{phase}_curves.npz", curves=np.array([artifacts["curves"]], dtype=object))
    checkpoints = {}
    for key, state in artifacts["states"].items():
        checkpoints[key] = dict(state)
    torch.save({"masks": {k: v.cpu() for k, v in masks.items()}, "checkpoints": checkpoints,
                "optimization_adequacy_step800": artifacts.get("step800", {})},
               folder / f"{phase}_masks_states.pt")
    (folder / f"{phase.upper()}_COMPLETE").write_text("validated\n")


def run_selection(root: Path, seed: int, out: Path, device: str) -> None:
    freeze(root); folder = _seed_dir(out, seed)
    if (folder / "SELECTION_COMPLETE").exists(): return
    splits, provenance = data.selection_data(seed, device)
    masks, manifest = data.selection_masks(seed, device); data.assert_mask_protocol(masks)
    records, artifacts = evaluate_adaptive(splits, np.load(root / "selection_tasks.npy"), masks, manifest, seed=seed,
                                           device=device, phase="selection", cfg=EvalConfig(), score_key="selection_score")
    provenance["mask_manifest"] = manifest
    _save_stage(folder, "selection", records, provenance, artifacts, masks)


def _final_masks(root: Path, seed: int, device: str) -> tuple[dict[str, torch.Tensor], list[dict]]:
    selected = json.loads((root / "selected_density.json").read_text())
    if selected.get("selection_protocol_sha256") != _sha(root / "protocol.json"): raise ValueError("selected density is not bound to this protocol")
    all_masks, _ = data.selection_masks(seed, device); result = {}; manifest = []
    # Parent aggregation supplies one global configuration for every seed/budget.
    for budget in data.BUDGETS:
        configs = selected["by_budget"][str(budget)]
        required = {"functional_selected", "gnn_selected", "pixelprior_selected", "joint_primary",
                    "random_matched_joint", "dense_tuned", "dense_default",
                    "functional_fixed30", "gnn_fixed30", "pixelprior_fixed30"}
        if set(configs) != required:
            raise ValueError(f"budget {budget} must declare exactly the predeclared ten final methods")
        for label, source in configs.items():
            if source["method"] not in all_masks: raise ValueError(f"unknown selected method {source['method']}")
            name = f"{label}_b{budget}"; result[name] = all_masks[source["method"]]
            manifest.append({"method": name, "family": source.get("family", label), "rho": source.get("rho", 1.0), "lr": source["lr"], "budget": budget})
    return result, manifest


def run_confirmation(root: Path, seed: int, out: Path, device: str) -> None:
    freeze(root); folder = _seed_dir(out, seed)
    if not confirmation_ready(root, folder): raise RuntimeError("confirmation is guarded until validated selection and frozen density complete")
    if (folder / "CONFIRMATION_COMPLETE").exists(): return
    splits, provenance = data.confirmation_data(seed, device)
    masks, manifest = _final_masks(root, seed, device)
    # Evaluate per budget so each selected density has its own final mask while all methods remain paired.
    records=[]; all_artifacts={"curves": {}, "states": {}, "step800": {}}
    for budget in data.BUDGETS:
        names = [row["method"] for row in manifest if row["budget"] == budget]
        rows = [row for row in manifest if row["budget"] == budget]
        rec, artifact = evaluate_adaptive(splits, np.load(root / "confirmation_tasks.npy"), {n: masks[n] for n in names}, rows,
                                          seed=seed, device=device, phase="confirmation", cfg=EvalConfig(), score_key="confirmation_test",
                                          support_sizes=(budget,))
        records.extend(rec); all_artifacts["curves"].update(artifact["curves"]); all_artifacts["states"].update(artifact["states"]); all_artifacts["step800"].update(artifact["step800"])
    provenance["mask_manifest"] = manifest
    _save_stage(folder, "confirmation", records, provenance, all_artifacts, masks)


def confirmation_ready(root: Path, folder: Path) -> bool:
    # The orchestration validator binds all eight phase-A artifacts and their
    # current immutable-input hashes into selected_density before phase B.
    from .adaptive_orchestration import confirmation_ready as global_ready
    try:
        return bool(global_ready(root, int(folder.name.removeprefix("seed_"))))
    except ValueError:
        return False


def smoke(out: Path) -> dict[str, Any]:
    """CPU checks of cardinality/endpoints, paired starts, LR chunks and stage guard."""
    device = "cpu"; masks, _ = data.selection_masks(4100, device); data.assert_mask_protocol(masks)
    dense = [value for name, value in masks.items() if name.startswith("dense_")]
    exact = all(int(value[0].sum()) == data.k_for_rho(.30) for name, value in masks.items() if "rho0.3" in name)
    endpoint = all(torch.equal(value, torch.ones_like(value)) for value in dense)
    lr_ok = toy_adam_lr_test(device); pairing_freeze = pairing_and_freeze_test(device)
    logical_mapping = logical_order_mapping_test(device)
    guard = not confirmation_ready(out, out / "seed_4100")
    result = {"exact_density_prefix": exact, "dense_endpoint": endpoint, "standard_adam_lr_chunks": lr_ok,
              "initialization_pairing_and_freeze_no_adam_drift": pairing_freeze,
              "three_lr_logical_order_snapshot_loss_freeze": logical_mapping, "confirmation_guard": guard,
              "status": "PASS" if all((exact, endpoint, lr_ok, pairing_freeze, logical_mapping, guard)) else "FAIL"}
    _json(out / "smoke_adaptive_density.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--phase", choices=("selection", "confirmation"), default="selection")
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); root = _root(args.out)
    if args.smoke:
        result = smoke(root); print(json.dumps(result)); return
    if not args.launch: raise SystemExit("Refusing production execution without --launch")
    if args.phase == "selection": run_selection(root, args.seed, args.out, args.device)
    else: run_confirmation(root, args.seed, args.out, args.device)


if __name__ == "__main__": main()
