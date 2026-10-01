"""Leakage-safe LR tuning, held-out evaluation, and report orchestration.

The runner is deliberately split into explicit stages so that validation
scores are globally frozen before any held-out test labels are constructed:

``tune --seed`` -> ``freeze-lr`` -> ``test --seed`` -> ``merge``.

Each ``tune`` and ``test`` call batches all candidate child fits for one
support budget through :func:`meta.fit_child_batch`.  The child fitter writes
its complete Adam/RNG/history checkpoint every evaluation interval; completed
test records are stored per seed to avoid concurrent writers sharing a file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from meta_pattern.data import build_task_splits

from .core import (
    build_task_support_query,
    build_test_pool,
    common_probe,
    sample_balanced,
)
from .evaluate import (
    EXPECTED_OUTER_SEEDS,
    EXPECTED_TEST_TASK_IDS,
    METHODS,
    analytic_labels,
    build_comparison_masks,
    score_children_batched,
    random_exact32_mask,
    validate_eval_record,
    validate_method_mask,
)
from .generator import FEATURE_DIM, Generator, generate
from .meta import fit_child_batch
from .structure import gold_toeplitz_mask


DEFAULT_ROOT = Path("/home/udeneev-av/ResearchProject/pattern/outputs/task_quality_toeplitz_20261001")
LR_GRID = (0.001, 0.003, 0.01)
EXPECTED_BUDGETS = (32, 128)
EXPECTED_INIT_IDS = (0, 1, 2, 3)
MAX_STEPS = 12000
MAX_EXTENDED_STEPS = 48000


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**31 - 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_bank(root: Path, seed: int) -> tuple[dict[str, Any], Path]:
    path = root / f"seed_{seed}" / "bank" / "bank.pt"
    if not path.is_file():
        raise FileNotFoundError(f"source bank is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "feature" not in payload or payload["feature"].ndim != 3:
        raise ValueError(f"bank feature tensor has an invalid shape in {path}")
    return payload, path


def _load_meta_models(root: Path, seed: int, device: torch.device) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for method in ("transformer_mask", "free_mask"):
        path = root / f"seed_{seed}" / method / "meta" / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"best meta checkpoint is missing: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("method") != method or int(checkpoint.get("seed", -1)) != seed:
            raise ValueError(f"meta checkpoint provenance mismatch: {path}")
        model = Generator(feature_dim=FEATURE_DIM, mode=method).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        models[method] = {
            "model": model,
            "path": path,
            "sha256": _sha256(path),
            "checkpoint": checkpoint,
        }
    return models


def _edge_q(bank: Mapping[str, Any], probe: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return saved source-only edge responses, reconstructing older banks."""
    if "edge_q" in bank:
        values = torch.as_tensor(bank["edge_q"]).detach().cpu().float()
        if values.ndim != 4 or tuple(values.shape[-2:]) != (11, 8):
            raise ValueError("bank edge_q must have shape [maps, probe, 11, 8]")
        return values
    required = {"weff", "b", "a"}
    if not required.issubset(bank):
        raise ValueError("bank is missing edge_q and the source weights needed to reconstruct it")
    x = probe["x"].detach().cpu().float()
    weight = torch.as_tensor(bank["weff"]).detach().cpu().float()
    bias = torch.as_tensor(bank["b"]).detach().cpu().float()
    readout = torch.as_tensor(bank["a"]).detach().cpu().float()
    preactivation = torch.einsum("pi,mih->pmh", x, weight) + bias[None, :, :]
    return (x[:, None, :, None] * weight[None]
            * readout[None, :, None, :] * (preactivation > 0).to(weight.dtype))


def _task_lookup(split_name: str) -> list[Any]:
    splits = build_task_splits([4], seed=42)
    tasks = list(splits[split_name])
    if split_name == "test" and tuple(task.task_id for task in tasks) != EXPECTED_TEST_TASK_IDS:
        raise RuntimeError("held-out task split no longer matches the registered protocol")
    return tasks


def _draw_support(task_pool: Mapping[str, Any], seed: int, budget: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample = sample_balanced(task_pool["support"], budget, generator)
    positives = int((sample["y"] > 0.5).sum())
    if positives != budget // 2:
        raise RuntimeError("balanced support sampler did not produce the required class balance")
    return sample


def _generator_masks(
    models: Mapping[str, Mapping[str, Any]],
    bank_feature: torch.Tensor,
    support: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, np.ndarray]:
    x = support["x"].to(device=device, dtype=torch.float32).unsqueeze(0)
    y = support["y"].to(device=device, dtype=torch.float32).unsqueeze(0)
    output: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for method in ("transformer_mask", "free_mask"):
            model = models[method]["model"]
            selected_bank = bank_feature.to(device) if method == "transformer_mask" else None
            masks, _ = generate(model, selected_bank, x, y)
            output[method] = masks[0].detach().cpu().numpy().astype(np.uint8)
    return output


def _condition_masks(
    *,
    models: Mapping[str, Mapping[str, Any]],
    bank: Mapping[str, Any],
    support: Mapping[str, torch.Tensor],
    probe: Mapping[str, torch.Tensor],
    device: torch.device,
    outer_seed: int,
    task_id: str,
) -> dict[str, Any]:
    learned = _generator_masks(models, torch.as_tensor(bank["feature"]), support, device)
    # Each child initialization gets its own random support order. The order is
    # paired across LR values and reused across support budgets.
    random_order_seed = _stable_seed("random-mask", outer_seed, 0)
    result = build_comparison_masks(
        learned["transformer_mask"], learned["free_mask"], _edge_q(bank, probe),
        random_seed=random_order_seed,
    )
    random_layouts = {
        init_id: random_exact32_mask(_stable_seed("random-mask", outer_seed, init_id))
        for init_id in EXPECTED_INIT_IDS
    }
    orders = [tuple(layout["nested_order"].tolist()) for layout in random_layouts.values()]
    if len(set(orders)) != len(EXPECTED_INIT_IDS):
        raise RuntimeError("random_exact32 must use four independent nested edge orders")
    result["random_exact32"] = {
        init_id: layout["mask"] for init_id, layout in random_layouts.items()
    }
    result["random_exact32_nested_order"] = {
        init_id: layout["nested_order"] for init_id, layout in random_layouts.items()
    }
    return result


def _mask_for_spec(masks_by_condition: Mapping[tuple[str, int], Mapping[str, Any]],
                   spec: Mapping[str, Any]) -> np.ndarray:
    condition = (str(spec["task_id"]), int(spec["budget"]))
    method = str(spec["method"])
    value = masks_by_condition[condition][method]
    if method == "random_exact32":
        value = value[int(spec["init_id"])]
    return np.asarray(value, dtype=np.uint8)


def _candidate_init_seed(task_id: str, budget: int, init_id: int) -> int:
    # The same fresh initialization and minibatch stream is paired across mask
    # methods, LR values, and outer seeds for an identical task/budget/init ID.
    return _stable_seed("child-init", task_id, budget, init_id)


def _stack_candidates(
    specs: Sequence[Mapping[str, Any]],
    masks_by_condition: Mapping[tuple[str, int], Mapping[str, np.ndarray]],
    pools_by_condition: Mapping[tuple[str, int], Mapping[str, Any]],
    query_full: bool = True,
) -> tuple[torch.Tensor, ...]:
    masks: list[np.ndarray] = []
    support_x: list[torch.Tensor] = []
    support_y: list[torch.Tensor] = []
    query_x: list[torch.Tensor] = []
    query_y: list[torch.Tensor] = []
    child_seeds: list[int] = []
    learning_rates: list[float] = []
    for spec in specs:
        task_id, budget = str(spec["task_id"]), int(spec["budget"])
        condition = (task_id, budget)
        method = str(spec["method"])
        mask = validate_method_mask(method, _mask_for_spec(masks_by_condition, spec))["mask"]
        pool = pools_by_condition[condition]
        masks.append(mask)
        support_x.append(pool["sampled_support"]["x"])
        support_y.append(pool["sampled_support"]["y"])
        query_x.append(pool["query"]["x"])
        query_y.append(pool["query"]["y"])
        child_seeds.append(_candidate_init_seed(task_id, budget, int(spec["init_id"])))
        learning_rates.append(float(spec.get("lr", 0.001)))
    return (
        torch.as_tensor(np.stack(masks), dtype=torch.float32),
        torch.stack(support_x).float(), torch.stack(support_y).float(),
        torch.stack(query_x).float(), torch.stack(query_y).float(),
        torch.as_tensor(child_seeds, dtype=torch.int64),
        torch.as_tensor(learning_rates, dtype=torch.float32),
    )


def _fit_candidates(
    specs: Sequence[Mapping[str, Any]],
    masks_by_condition: Mapping[tuple[str, int], Mapping[str, np.ndarray]],
    pools_by_condition: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    device: torch.device,
    checkpoint_path: Path,
    min_steps: int,
    max_steps: int,
    max_updates: int | None,
) -> dict[str, Any]:
    manifest_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".manifest.json")
    conditions = sorted({(str(spec["task_id"]), int(spec["budget"])) for spec in specs})
    manifest = {
        "protocol": "task_quality_child_fit_manifest_v1",
        "checkpoint": str(checkpoint_path),
        "candidates": [
            {**dict(spec), "child_init_seed": _candidate_init_seed(
                str(spec["task_id"]), int(spec["budget"]), int(spec["init_id"])),
             "mask": _mask_for_spec(masks_by_condition, spec).astype(int).tolist()}
            for spec in specs
        ],
        "pools": {
            f"{task_id}/budget_{budget}": {
                "support_ids": pools_by_condition[(task_id, budget)]["sampled_support"]["ids"].tolist(),
                "query_ids": pools_by_condition[(task_id, budget)]["query"]["ids"].tolist(),
            }
            for task_id, budget in conditions
        },
        "test_labels_materialized": False,
    }
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise ValueError(f"resumed child fit manifest does not match the current inputs: {manifest_path}")
    else:
        _atomic_json(manifest, manifest_path)
    mask, x_support, y_support, x_query, y_query, seeds, rates = _stack_candidates(
        specs, masks_by_condition, pools_by_condition)
    if max_steps > MAX_EXTENDED_STEPS:
        raise ValueError(f"initial child step cap cannot exceed {MAX_EXTENDED_STEPS}")
    stored_cap: int | None = None
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        stored_cap = int(checkpoint.get("max_steps", 0))
        if not 0 < stored_cap <= MAX_EXTENDED_STEPS:
            raise ValueError(f"child checkpoint has an invalid saved cap: {stored_cap}")
    first_cap = max(int(max_steps), stored_cap or int(max_steps))
    caps = [first_cap]
    while caps[-1] < MAX_EXTENDED_STEPS:
        caps.append(min(MAX_EXTENDED_STEPS, caps[-1] * 2))
    result: dict[str, Any] | None = None
    previous_cap: int | None = None
    for cap in caps:
        is_extension = ((previous_cap is not None)
                        or (stored_cap is not None and cap > stored_cap))
        result = fit_child_batch(
            mask, x_support, y_support,
            x_query=x_query, y_query=y_query, seeds=seeds,
            learning_rates=rates, device=device, max_steps=cap,
            min_steps=min_steps, batch_size=128, momentum=0.9, beta2=0.999,
            decay_every=2000, minimum_lr_factor=1.0 / 16.0,
            eval_every=100, patience_steps=400, plateau_relative=0.01,
            checkpoint_path=checkpoint_path, resume=checkpoint_path.exists(),
            extend=is_extension,
            max_updates=max_updates,
        )
        if result["termination_reason"] == "checkpoint_chunk":
            return result
        converged = torch.as_tensor(result["converged"], dtype=torch.bool)
        if bool(converged.all()):
            break
        # An explicitly chunked invocation never silently advances to a new
        # cap. Rerun without --max-updates to enter the registered extensions.
        if max_updates is not None:
            result["termination_reason"] = "checkpoint_chunk"
            return result
        if cap >= MAX_EXTENDED_STEPS:
            break
        previous_cap = cap
    assert result is not None
    if not bool(torch.as_tensor(result["selection_complete"]).all()):
        raise RuntimeError("child checkpoint selection was not completed on query data")
    if not torch.isfinite(torch.as_tensor(result["best_query_balanced_bce"])).all():
        raise RuntimeError("child query checkpoint scores contain non-finite values")
    return result


def _fit_curves(result: Mapping[str, Any], specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    for history_row in result["history"]:
        step = int(torch.as_tensor(history_row["step"]).item())
        support_losses = torch.as_tensor(history_row["support_balanced_bce"]).reshape(-1)
        query_losses = torch.as_tensor(history_row["query_balanced_bce"]).reshape(-1)
        for index, spec in enumerate(specs):
            curves.append({
                "method": str(spec["method"]), "budget": int(spec["budget"]),
                "task_id": str(spec["task_id"]), "seed": int(spec["seed"]),
                "init_id": int(spec["init_id"]), "step": step,
                "support_balanced_bce": float(support_losses[index]),
                "query_balanced_bce": float(query_losses[index]),
            })
    return curves


def _save_curve_archive(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        method=np.asarray([row["method"] for row in rows], dtype="U32"),
        budget=np.asarray([row["budget"] for row in rows], dtype=np.int64),
        task_id=np.asarray([row["task_id"] for row in rows], dtype="U16"),
        seed=np.asarray([row["seed"] for row in rows], dtype=np.int64),
        init_id=np.asarray([row["init_id"] for row in rows], dtype=np.int64),
        step=np.asarray([row["step"] for row in rows], dtype=np.int64),
        support_balanced_bce=np.asarray([row["support_balanced_bce"] for row in rows], dtype=np.float64),
        query_balanced_bce=np.asarray([row["query_balanced_bce"] for row in rows], dtype=np.float64),
    )
    temporary.replace(path)


def tune_seed(
    root: str | Path,
    seed: int,
    *,
    device: str | torch.device,
    min_steps: int = 1000,
    max_steps: int = MAX_STEPS,
    max_updates: int | None = None,
) -> dict[str, Any]:
    """Tune each method/budget LR using only the two meta-validation tasks."""
    if seed not in EXPECTED_OUTER_SEEDS:
        raise ValueError(f"seed must be one of {EXPECTED_OUTER_SEEDS}")
    if not 1 <= min_steps <= max_steps <= MAX_EXTENDED_STEPS:
        raise ValueError(f"fit limits must satisfy 1 <= min <= max <= {MAX_EXTENDED_STEPS}")
    root = Path(root)
    device = torch.device(device)
    bank, bank_path = _load_bank(root, seed)
    models = _load_meta_models(root, seed, device)
    probe = common_probe(n_probe=128, seed=seed, split_seed=1729)
    val_tasks = _task_lookup("val")
    val_ids = tuple(task.task_id for task in val_tasks)
    if len(val_tasks) != 2:
        raise RuntimeError("protocol requires exactly two meta-validation tasks")

    output_dir = root / f"seed_{seed}" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    for budget in EXPECTED_BUDGETS:
        pools: dict[tuple[str, int], Mapping[str, Any]] = {}
        masks_by_condition: dict[tuple[str, int], Mapping[str, np.ndarray]] = {}
        specs: list[dict[str, Any]] = []
        for task_index, task in enumerate(val_tasks):
            task_pools = build_task_support_query(task, probe["ids"], split_seed=1729)
            sampled = _draw_support(
                task_pools, _stable_seed("val-support", seed, task.task_id, budget), budget)
            condition_pool = {"sampled_support": sampled, "query": task_pools["query"]}
            condition = (task.task_id, budget)
            pools[condition] = condition_pool
            masks_by_condition[condition] = _condition_masks(
                models=models, bank=bank, support=sampled, probe=probe,
                device=device, outer_seed=seed, task_id=task.task_id)
            for method in METHODS:
                for lr in LR_GRID:
                    for init_id in EXPECTED_INIT_IDS:
                        specs.append({
                            "seed": seed, "task_index": task_index, "task_id": task.task_id,
                            "budget": budget, "method": method, "lr": lr,
                            "init_id": init_id,
                        })
        if len(specs) != 144:
            raise RuntimeError(f"expected 144 validation candidates per budget, got {len(specs)}")
        checkpoint_path = output_dir / f"tune_budget_{budget}.checkpoint.pt"
        fitted = _fit_candidates(
            specs, masks_by_condition, pools, device=device,
            checkpoint_path=checkpoint_path, min_steps=min_steps,
            max_steps=max_steps, max_updates=max_updates)
        if fitted["termination_reason"] == "checkpoint_chunk":
            return {"stage": "tune", "seed": seed, "budget": budget,
                    "complete": False, "checkpoint": str(checkpoint_path),
                    "step": int(fitted["total_steps"]),
                    "message": "checkpoint chunk saved; rerun the same command to resume"}

        scores = torch.as_tensor(fitted["best_query_balanced_bce"]).detach().cpu().numpy()
        status = fitted["fit_status"]
        for index, spec in enumerate(specs):
            run_rows.append({
                **spec,
                "best_query_balanced_bce": float(scores[index]),
                "best_step": int(torch.as_tensor(fitted["best_steps"])[index]),
                "fit_status": status[index],
                "mask": _mask_for_spec(masks_by_condition, spec).astype(int).tolist(),
                "random_nested_order": (
                    masks_by_condition[(spec["task_id"], budget)]
                    ["random_exact32_nested_order"][int(spec["init_id"])].tolist()
                    if spec["method"] == "random_exact32" else None),
                "support_ids": pools[(spec["task_id"], budget)]["sampled_support"]["ids"].tolist(),
                "query_ids": pools[(spec["task_id"], budget)]["query"]["ids"].tolist(),
            })
        _atomic_torch({
            "protocol": "task_quality_validation_child_fit_v1",
            "seed": seed, "budget": budget, "specs": specs,
            "masks": _stack_candidates(specs, masks_by_condition, pools)[0],
            "best_params": fitted["best_params"], "last_params": fitted["last_params"],
            "best_steps": fitted["best_steps"],
            "best_query_balanced_bce": fitted["best_query_balanced_bce"],
            "fit_status": fitted["fit_status"], "history": fitted["history"],
            "checkpoint_path": str(checkpoint_path),
            "bank_sha256": _sha256(bank_path),
            "meta_checkpoint_sha256": {name: data["sha256"] for name, data in models.items()},
        }, output_dir / f"tune_budget_{budget}.pt")

    scores_by_method_budget: dict[str, dict[str, dict[str, float]]] = {}
    for method in METHODS:
        scores_by_method_budget[method] = {}
        for budget in EXPECTED_BUDGETS:
            scores_by_method_budget[method][str(budget)] = {}
            for rate in LR_GRID:
                values = [row["best_query_balanced_bce"] for row in run_rows
                          if row["method"] == method and int(row["budget"]) == budget
                          and math.isclose(float(row["lr"]), rate)]
                if len(values) != len(val_tasks) * len(EXPECTED_INIT_IDS):
                    raise RuntimeError("validation LR scores do not cover both tasks and four child inits")
                scores_by_method_budget[method][str(budget)][f"{rate:g}"] = float(np.mean(values))
    payload = {
        "protocol": "task_quality_lr_tuning_v1", "seed": seed,
        "selection_split": "meta_val_query", "task_ids": list(val_ids),
        "budgets": list(EXPECTED_BUDGETS), "methods": list(METHODS),
        "lr_grid": list(LR_GRID), "init_ids": list(EXPECTED_INIT_IDS),
        "score": "mean best query balanced BCE across both validation tasks and four fresh-child initializations",
        "used_test": False,
        "bank_sha256": _sha256(bank_path),
        "meta_checkpoint_sha256": {name: data["sha256"] for name, data in models.items()},
        "scores_by_method_budget_lr": scores_by_method_budget,
        "candidate_results": run_rows,
    }
    _atomic_json(payload, output_dir / "val_tuning.json")
    return {"stage": "tune", "seed": seed, "complete": True,
            "output": str(output_dir / "val_tuning.json"),
            "candidate_count": len(run_rows)}


def freeze_learning_rates(root: str | Path) -> dict[str, Any]:
    """Freeze global per-method/per-budget rates from all four seed val runs."""
    root = Path(root)
    per_seed: dict[int, dict[str, Any]] = {}
    tuning_provenance: dict[str, Any] = {}
    for seed in EXPECTED_OUTER_SEEDS:
        path = root / f"seed_{seed}" / "eval" / "val_tuning.json"
        if not path.is_file():
            raise FileNotFoundError(f"validation tuning for seed {seed} is missing: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("protocol") != "task_quality_lr_tuning_v1" or int(data.get("seed", -1)) != seed:
            raise ValueError(f"validation tuning provenance mismatch: {path}")
        if data.get("used_test") is not False or data.get("selection_split") != "meta_val_query":
            raise ValueError(f"validation tuning cannot use held-out test data: {path}")
        expected_val_ids = [task.task_id for task in _task_lookup("val")]
        if (data.get("methods") != list(METHODS)
                or data.get("budgets") != list(EXPECTED_BUDGETS)
                or data.get("init_ids") != list(EXPECTED_INIT_IDS)
                or data.get("lr_grid") != list(LR_GRID)
                or data.get("task_ids") != expected_val_ids):
            raise ValueError(f"validation tuning design is incomplete or changed: {path}")
        candidate_rows = data.get("candidate_results", [])
        if len(candidate_rows) != 288:
            raise ValueError(f"validation tuning must contain 288 candidates: {path}")
        for method in METHODS:
            for budget in EXPECTED_BUDGETS:
                for rate in LR_GRID:
                    matching = [row for row in candidate_rows
                                if row.get("method") == method
                                and int(row.get("budget", -1)) == budget
                                and math.isclose(float(row.get("lr", np.nan)), rate)]
                    if len(matching) != len(expected_val_ids) * len(EXPECTED_INIT_IDS):
                        raise ValueError(f"validation tuning has incomplete LR coverage: {path}")
                    observed_pairs = {(str(row.get("task_id")), int(row.get("init_id", -1)))
                                      for row in matching}
                    expected_pairs = {(task_id, init_id) for task_id in expected_val_ids
                                      for init_id in EXPECTED_INIT_IDS}
                    if observed_pairs != expected_pairs:
                        raise ValueError(f"validation tuning has duplicate or missing task/init cells: {path}")
                    if not np.isfinite([row["best_query_balanced_bce"] for row in matching]).all():
                        raise ValueError(f"validation tuning scores must be finite: {path}")
                    recomputed = float(np.mean([
                        float(row["best_query_balanced_bce"]) for row in matching]))
                    recorded = float(data["scores_by_method_budget_lr"][method]
                                     [str(budget)][f"{rate:g}"])
                    if not math.isclose(recorded, recomputed, rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError(f"validation tuning summary disagrees with its candidates: {path}")
        per_seed[seed] = data
        tuning_provenance[str(seed)] = {
            "val_tuning_sha256": _sha256(path),
            "bank_sha256": data.get("bank_sha256"),
            "meta_checkpoint_sha256": data.get("meta_checkpoint_sha256"),
        }

    selections: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        selections[method] = {}
        for budget in EXPECTED_BUDGETS:
            scores: dict[str, float] = {}
            for rate in LR_GRID:
                all_seed_scores = [
                    float(per_seed[seed]["scores_by_method_budget_lr"][method][str(budget)][f"{rate:g}"])
                    for seed in EXPECTED_OUTER_SEEDS
                ]
                if not np.isfinite(all_seed_scores).all():
                    raise ValueError("non-finite LR validation score")
                scores[f"{rate:g}"] = float(np.mean(all_seed_scores))
            chosen = min(LR_GRID, key=lambda rate: scores[f"{rate:g}"])
            selections[method][str(budget)] = {
                "selected_lr": float(chosen), "grid": list(LR_GRID),
                "scores_by_lr": scores, "selection_split": "meta_val_query",
                "used_test": False,
                "task_ids": list(_task_lookup("val")[index].task_id for index in range(2)),
                "budgets": list(EXPECTED_BUDGETS),
                "outer_seeds": list(EXPECTED_OUTER_SEEDS),
                "init_ids": list(EXPECTED_INIT_IDS),
                "n_seeds": len(EXPECTED_OUTER_SEEDS),
                "n_inits": len(EXPECTED_INIT_IDS),
                "score": "mean best query balanced BCE over two validation tasks, four outer seeds and four fresh-child initializations",
            }
    payload = {
        "protocol": "task_quality_global_lr_selection_v1",
        "selection_split": "meta_val_query", "used_test": False,
        "tuning_provenance": tuning_provenance,
        "methods": selections,
    }
    _atomic_json(payload, root / "lr_selection.json")
    return payload


def _validate_frozen_tuning_provenance(root: Path, lr_payload: Mapping[str, Any]) -> None:
    """Verify every artifact used by global LR selection is still unchanged."""
    provenance = lr_payload.get("tuning_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("global LR freeze is missing validation artifact hashes")
    for seed in EXPECTED_OUTER_SEEDS:
        expected = provenance.get(str(seed))
        if not isinstance(expected, Mapping):
            raise ValueError(f"global LR freeze has no provenance for seed {seed}")
        eval_dir = root / f"seed_{seed}" / "eval"
        tuning_path = eval_dir / "val_tuning.json"
        bank_path = root / f"seed_{seed}" / "bank" / "bank.pt"
        if not tuning_path.is_file() or not bank_path.is_file():
            raise FileNotFoundError(f"validation provenance artifacts are missing for seed {seed}")
        if _sha256(tuning_path) != expected.get("val_tuning_sha256"):
            raise ValueError(f"validation tuning artifact changed after global LR freeze: {tuning_path}")
        if _sha256(bank_path) != expected.get("bank_sha256"):
            raise ValueError(f"source bank changed after global LR freeze: {bank_path}")
        meta_hashes = expected.get("meta_checkpoint_sha256")
        if not isinstance(meta_hashes, Mapping):
            raise ValueError(f"global LR freeze lacks meta checkpoint hashes for seed {seed}")
        for method in ("transformer_mask", "free_mask"):
            meta_path = root / f"seed_{seed}" / method / "meta" / "best.pt"
            if not meta_path.is_file() or _sha256(meta_path) != meta_hashes.get(method):
                raise ValueError(f"{method} checkpoint changed after global LR freeze: {meta_path}")


def _existing_seed_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                validate_eval_record(record)
                records.append(record)
    return records


def _write_seed_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def _serialize_params(params: Mapping[str, torch.Tensor], index: int) -> dict[str, list[float]]:
    return {name: torch.as_tensor(value[index]).detach().cpu().float().numpy().tolist()
            for name, value in params.items()}


def test_seed(
    root: str | Path,
    seed: int,
    *,
    device: str | torch.device,
    min_steps: int = 1000,
    max_steps: int = MAX_STEPS,
    max_updates: int | None = None,
) -> dict[str, Any]:
    """Freeze all query-selected children before creating any test-label pool."""
    if seed not in EXPECTED_OUTER_SEEDS:
        raise ValueError(f"seed must be one of {EXPECTED_OUTER_SEEDS}")
    if not 1 <= min_steps <= max_steps <= MAX_EXTENDED_STEPS:
        raise ValueError(f"fit limits must satisfy 1 <= min <= max <= {MAX_EXTENDED_STEPS}")
    root = Path(root)
    device = torch.device(device)
    lr_path = root / "lr_selection.json"
    if not lr_path.is_file():
        raise FileNotFoundError("run freeze-lr after all four tune stages before test evaluation")
    lr_payload = json.loads(lr_path.read_text(encoding="utf-8"))
    if lr_payload.get("protocol") != "task_quality_global_lr_selection_v1" or lr_payload.get("used_test") is not False:
        raise ValueError("global LR freeze is missing or has invalid provenance")
    _validate_frozen_tuning_provenance(root, lr_payload)
    bank, bank_path = _load_bank(root, seed)
    models = _load_meta_models(root, seed, device)
    frozen_provenance = lr_payload.get("tuning_provenance", {}).get(str(seed))
    if not isinstance(frozen_provenance, Mapping):
        raise ValueError(f"global LR freeze is missing seed {seed} artifact hashes")
    if _sha256(bank_path) != frozen_provenance.get("bank_sha256"):
        raise ValueError("source bank changed after meta-validation LR selection")
    expected_meta_hashes = frozen_provenance.get("meta_checkpoint_sha256", {})
    for method, model_data in models.items():
        if expected_meta_hashes.get(method) != model_data["sha256"]:
            raise ValueError(f"{method} checkpoint changed after LR selection")
    probe = common_probe(n_probe=128, seed=seed, split_seed=1729)
    tasks = _task_lookup("test")
    output_dir = root / f"seed_{seed}" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "test_records.jsonl"
    records = _existing_seed_records(records_path)
    curve_rows: list[dict[str, Any]] = []
    curve_path = output_dir / "child_curves.npz"
    if curve_path.is_file():
        with np.load(curve_path, allow_pickle=False) as archive:
            curve_rows = [dict(method=str(m), budget=int(b), task_id=str(t), seed=int(s),
                               init_id=int(i), step=int(st),
                               support_balanced_bce=float(tr), query_balanced_bce=float(qr))
                          for m, b, t, s, i, st, tr, qr in zip(
                              archive["method"], archive["budget"], archive["task_id"],
                              archive["seed"], archive["init_id"], archive["step"],
                              archive["support_balanced_bce"], archive["query_balanced_bce"])]

    records_by_budget = {budget: [row for row in records if int(row["budget"]) == budget]
                         for budget in EXPECTED_BUDGETS}
    expected_records_per_budget = len(tasks) * len(METHODS) * len(EXPECTED_INIT_IDS)
    for budget, rows in records_by_budget.items():
        if rows and len(rows) != expected_records_per_budget:
            raise ValueError(f"budget {budget} has {len(rows)} records, expected {expected_records_per_budget}")
    lr_sha256 = _sha256(lr_path)
    frozen_paths = {budget: output_dir / f"test_budget_{budget}.frozen.pt"
                    for budget in EXPECTED_BUDGETS}

    # Phase one fits all four test tasks at each budget. No test pool is built
    # during this phase, so the test labels cannot influence another condition.
    for budget in EXPECTED_BUDGETS:
        frozen_path = frozen_paths[budget]
        if frozen_path.is_file():
            frozen = torch.load(frozen_path, map_location="cpu", weights_only=False)
            if (frozen.get("protocol") != "task_quality_frozen_test_batch_v1"
                    or int(frozen.get("seed", -1)) != seed
                    or int(frozen.get("budget", -1)) != budget
                    or frozen.get("lr_selection_sha256") != lr_sha256
                    or frozen.get("bank_sha256") != _sha256(bank_path)
                    or frozen.get("meta_checkpoint_sha256")
                    != {name: data["sha256"] for name, data in models.items()}):
                raise ValueError(f"frozen query batch provenance changed: {frozen_path}")
            if _sha256(Path(frozen["batch_checkpoint"])) != frozen.get("batch_checkpoint_sha256"):
                raise ValueError(f"query-selected batch checkpoint changed: {frozen_path}")
            continue
        if records_by_budget[budget]:
            raise ValueError(f"test records exist without their frozen query batch: {frozen_path}")
        pools_by_condition: dict[tuple[str, int], Mapping[str, Any]] = {}
        masks_by_condition: dict[tuple[str, int], Mapping[str, Any]] = {}
        specs: list[dict[str, Any]] = []
        for task_index, task in enumerate(tasks):
            task_id = task.task_id
            task_pools = build_task_support_query(task, probe["ids"], split_seed=1729)
            sampled = _draw_support(
                task_pools, _stable_seed("test-support", seed, task_id, budget), budget)
            condition = (task_id, budget)
            pools_by_condition[condition] = {
                "sampled_support": sampled, "query": task_pools["query"]}
            masks_by_condition[condition] = _condition_masks(
                models=models, bank=bank, support=sampled, probe=probe,
                device=device, outer_seed=seed, task_id=task_id)
            for method in METHODS:
                for init_id in EXPECTED_INIT_IDS:
                    specs.append({
                        "seed": seed, "task_index": task_index, "task_id": task_id,
                        "budget": budget, "method": method, "init_id": init_id,
                        "lr": float(lr_payload["methods"][method][str(budget)]["selected_lr"]),
                    })
        if len(specs) != 96:
            raise RuntimeError(f"expected 96 final candidates per support budget, got {len(specs)}")
        checkpoint_path = output_dir / f"test_budget_{budget}.checkpoint.pt"
        fitted = _fit_candidates(
            specs, masks_by_condition, pools_by_condition,
            device=device, checkpoint_path=checkpoint_path,
            min_steps=min_steps, max_steps=max_steps, max_updates=max_updates)
        if fitted["termination_reason"] == "checkpoint_chunk":
            return {"stage": "test", "seed": seed, "budget": budget,
                    "complete": False, "checkpoint": str(checkpoint_path),
                    "step": int(fitted["total_steps"]),
                    "message": "checkpoint chunk saved; rerun the same command to resume"}
        candidate_masks = np.stack([_mask_for_spec(masks_by_condition, spec) for spec in specs])

        frozen_payload = {
            "protocol": "task_quality_frozen_test_batch_v1", "seed": seed,
            "budget": budget, "specs": specs,
            "masks": torch.as_tensor(candidate_masks, dtype=torch.uint8),
            "best_params": {name: value.detach().cpu() for name, value in fitted["best_params"].items()},
            "last_params": {name: value.detach().cpu() for name, value in fitted["last_params"].items()},
            "best_steps": torch.as_tensor(fitted["best_steps"]).detach().cpu(),
            "best_query_balanced_bce": torch.as_tensor(
                fitted["best_query_balanced_bce"]).detach().cpu(),
            "fit_status": fitted["fit_status"], "history": fitted["history"],
            "support_ids_by_task": {
                task.task_id: pools_by_condition[(task.task_id, budget)]["sampled_support"]["ids"]
                for task in tasks},
            "query_ids_by_task": {
                task.task_id: pools_by_condition[(task.task_id, budget)]["query"]["ids"]
                for task in tasks},
            "random_nested_order_by_init": {
                task.task_id: {
                    init_id: masks_by_condition[(task.task_id, budget)]
                    ["random_exact32_nested_order"][init_id]
                    for init_id in EXPECTED_INIT_IDS
                } for task in tasks},
            "batch_checkpoint": str(checkpoint_path),
            "batch_checkpoint_sha256": _sha256(checkpoint_path),
            "lr_selection_sha256": lr_sha256,
            "bank_sha256": _sha256(bank_path),
            "meta_checkpoint_sha256": {name: data["sha256"] for name, data in models.items()},
            "checkpoint_selected_on": "query", "frozen_before_test": True,
            "test_labels_materialized": False,
        }
        _atomic_torch(frozen_payload, frozen_path)

    for budget, frozen_path in frozen_paths.items():
        frozen = torch.load(frozen_path, map_location="cpu", weights_only=False)
        if _sha256(Path(frozen["batch_checkpoint"])) != frozen.get("batch_checkpoint_sha256"):
            raise ValueError(f"query-selected checkpoint changed before test access: {frozen_path}")
    if all(len(records_by_budget[budget]) == expected_records_per_budget
           for budget in EXPECTED_BUDGETS):
        return {"stage": "test", "seed": seed, "complete": True,
                "record_count": len(records), "records": str(records_path),
                "curves": str(curve_path)}
    if (_sha256(lr_path) != lr_sha256
            or _sha256(bank_path) != frozen_provenance.get("bank_sha256")
            or any(_sha256(data["path"]) != data["sha256"]
                   for data in models.values())):
        raise ValueError("LR, source bank, or meta checkpoint changed before test-label access")

    # Both support budgets now have frozen masks and query-selected child
    # checkpoints for all four patterns. Only now are any held-out test labels
    # constructed, and they are used for scoring only.
    test_pools = {task.task_id: build_test_pool(task, split_seed=1729) for task in tasks}
    for budget in EXPECTED_BUDGETS:
        if records_by_budget[budget]:
            continue
        frozen_path = frozen_paths[budget]
        frozen = torch.load(frozen_path, map_location="cpu", weights_only=False)
        specs = frozen["specs"]
        candidate_masks = torch.as_tensor(frozen["masks"], dtype=torch.float32)
        params = {name: value.to(device) for name, value in frozen["best_params"].items()}
        per_run_test_x = torch.stack([test_pools[str(spec["task_id"])]
                                      ["x"] for spec in specs])
        per_run_test_y = torch.stack([test_pools[str(spec["task_id"])]
                                      ["y"] for spec in specs])
        scores = score_children_batched(
            params, candidate_masks,
            {"x": per_run_test_x, "y": per_run_test_y}, device=device)
        curves_as_result = {"history": frozen["history"]}
        curve_rows.extend(_fit_curves(curves_as_result, specs))
        _save_curve_archive(curve_rows, curve_path)
        checkpoint_path = Path(frozen["batch_checkpoint"])
        if _sha256(checkpoint_path) != frozen["batch_checkpoint_sha256"]:
            raise ValueError("frozen query checkpoint changed before test scoring")
        child_checkpoint_sha = frozen["batch_checkpoint_sha256"]
        for index, (spec, test_score) in enumerate(zip(specs, scores)):
            task_id = str(spec["task_id"])
            task_index = int(spec["task_index"])
            method = str(spec["method"])
            condition_support = frozen["support_ids_by_task"][task_id]
            condition_query = frozen["query_ids_by_task"][task_id]
            mask = candidate_masks[index].to(torch.uint8).cpu()
            child_path = output_dir / f"child_{task_id}_b{budget}_{method}_init{spec['init_id']}.pt"
            per_child = {
                "protocol": "task_quality_frozen_child_v1", "spec": spec,
                "mask": mask,
                "best_params": {name: value[index] for name, value in frozen["best_params"].items()},
                "last_params": {name: value[index] for name, value in frozen["last_params"].items()},
                "best_step": int(frozen["best_steps"][index]),
                "best_query_balanced_bce": float(frozen["best_query_balanced_bce"][index]),
                "fit_status": frozen["fit_status"][index],
                "checkpoint_selected_on": "query", "frozen_before_test": True,
                "support_ids": condition_support, "query_ids": condition_query,
                "batch_checkpoint": str(checkpoint_path),
                "batch_checkpoint_sha256": child_checkpoint_sha,
            }
            _atomic_torch(per_child, child_path)
            method_lr = lr_payload["methods"][method][str(budget)]
            best_params = frozen["best_params"]
            record = {
                "method": method, "seed": seed, "task_id": task_id,
                "task_index": task_index, "budget": budget,
                "init_id": int(spec["init_id"]), "mask": mask.numpy().astype(int).tolist(),
                "weight": _serialize_params(best_params, index)["w"],
                "bias": _serialize_params(best_params, index)["b"],
                "readout": _serialize_params(best_params, index)["a"],
                "output_bias": float(best_params["c"][index]),
                "checkpoint_selected_on": "query", "frozen_before_test": True,
                "test_labels_used_for_selection": False,
                "support_ids": condition_support.tolist(),
                "query_ids": condition_query.tolist(),
                "test_ids": test_pools[task_id]["ids"].tolist(),
                "lr_selection": method_lr,
                "fit_status": frozen["fit_status"][index],
                "best_step": int(frozen["best_steps"][index]),
                "best_query_balanced_bce": float(frozen["best_query_balanced_bce"][index]),
                "child_checkpoint": str(child_path),
                "child_checkpoint_sha256": _sha256(child_path),
                "batch_checkpoint": str(checkpoint_path),
                "batch_checkpoint_sha256": child_checkpoint_sha,
                "mask_provenance": {
                    "meta_checkpoint": str(models[method]["path"])
                    if method in models else None,
                    "meta_checkpoint_sha256": models[method]["sha256"]
                    if method in models else None,
                    "bank_checkpoint": str(bank_path), "bank_sha256": _sha256(bank_path),
                    "random_nested_order": (
                        frozen["random_nested_order_by_init"][task_id]
                        [int(spec["init_id"])].tolist()
                        if method == "random_exact32" else None),
                    "test_labels_used": False,
                },
                "test": test_score,
            }
            validate_eval_record(record)
            records.append(record)
        _write_seed_records(records_path, records)
        records_by_budget[budget] = [row for row in records if int(row["budget"]) == budget]

    return {"stage": "test", "seed": seed, "complete": len(records) == 4 * 2 * len(METHODS) * 4,
            "record_count": len(records), "records": str(records_path),
            "curves": str(curve_path)}


def _collect_meta_curves(root: Path) -> None:
    rows: list[tuple[str, int, int, float, float, float]] = []
    for seed in EXPECTED_OUTER_SEEDS:
        for method in ("transformer_mask", "free_mask"):
            path = root / f"seed_{seed}" / method / "meta" / "best.pt"
            if not path.is_file():
                continue
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            history = checkpoint.get("history", {})
            steps = torch.as_tensor(history.get("step", torch.empty(0))).reshape(-1)
            train = torch.as_tensor(history.get("train_monitor_query_bce", torch.empty(0))).reshape(-1)
            val = torch.as_tensor(history.get("val_query_bce", torch.empty(0))).reshape(-1)
            outer_lr = float(checkpoint.get("config", {}).get("outer_learning_rate", np.nan))
            if not (steps.numel() == train.numel() == val.numel()):
                continue
            if not np.isfinite(outer_lr):
                continue
            for step, train_loss, val_loss in zip(steps, train, val):
                rows.append((method, seed, int(step), float(train_loss), float(val_loss), outer_lr))
    if not rows:
        return
    np.savez_compressed(
        root / "meta_curves.npz",
        method=np.asarray([row[0] for row in rows], dtype="U32"),
        seed=np.asarray([row[1] for row in rows], dtype=np.int64),
        step=np.asarray([row[2] for row in rows], dtype=np.int64),
        train_loss=np.asarray([row[3] for row in rows], dtype=np.float64),
        val_loss=np.asarray([row[4] for row in rows], dtype=np.float64),
        lr=np.asarray([row[5] for row in rows], dtype=np.float64),
    )


def merge_and_report(root: str | Path) -> dict[str, Any]:
    """Merge per-seed records after all test stages and build final artifacts."""
    root = Path(root)
    merged: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for seed in EXPECTED_OUTER_SEEDS:
        seed_dir = root / f"seed_{seed}" / "eval"
        records_path = seed_dir / "test_records.jsonl"
        if not records_path.is_file():
            raise FileNotFoundError(f"test records for seed {seed} are missing")
        merged.extend(_existing_seed_records(records_path))
        curve_path = seed_dir / "child_curves.npz"
        if curve_path.is_file():
            with np.load(curve_path, allow_pickle=False) as archive:
                curves.extend(dict(method=str(m), budget=int(b), task_id=str(t), seed=int(s),
                                   init_id=int(i), step=int(st),
                                   support_balanced_bce=float(tr), query_balanced_bce=float(qr))
                              for m, b, t, s, i, st, tr, qr in zip(
                                  archive["method"], archive["budget"], archive["task_id"],
                                  archive["seed"], archive["init_id"], archive["step"],
                                  archive["support_balanced_bce"], archive["query_balanced_bce"]))
    merged.sort(key=lambda row: (row["method"], int(row["seed"]), int(row["task_index"]),
                                 int(row["budget"]), int(row["init_id"])))
    root.mkdir(parents=True, exist_ok=True)
    _write_seed_records(root / "records.jsonl", merged)
    _save_curve_archive(curves, root / "child_curves.npz")
    _collect_meta_curves(root)
    from .report import generate_report
    return generate_report(root, require_complete=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in ("tune", "test"):
        command = subparsers.add_parser(name)
        command.add_argument("--seed", type=int, required=True, choices=EXPECTED_OUTER_SEEDS)
        command.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        command.add_argument("--device", default="cpu")
        command.add_argument("--min-steps", type=int, default=1000)
        command.add_argument("--max-steps", type=int, default=MAX_STEPS)
        command.add_argument("--max-updates", type=int,
                             help="run one resumable optimizer chunk; omit for full fitting")
    freeze = subparsers.add_parser("freeze-lr")
    freeze.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.stage == "tune":
        result = tune_seed(args.root, args.seed, device=args.device,
                           min_steps=args.min_steps, max_steps=args.max_steps,
                           max_updates=args.max_updates)
    elif args.stage == "freeze-lr":
        result = freeze_learning_rates(args.root)
    elif args.stage == "test":
        result = test_seed(args.root, args.seed, device=args.device,
                           min_steps=args.min_steps, max_steps=args.max_steps,
                           max_updates=args.max_updates)
    else:
        result = merge_and_report(args.root)
    print(json.dumps(result, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
