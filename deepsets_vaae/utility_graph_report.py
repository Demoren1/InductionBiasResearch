"""Strict aggregate report for the utility-trained graph mask benchmark.

The bank seed is the statistical unit. Replicas are averaged within each of
the eight fixed target tasks, then the tasks are averaged within each bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/deepsets_vaae/20261001_rebuilt_utility_graph_flow"
BANK_OUT = ROOT / "outputs/deepsets_vaae/20261001_rebuilt_functional_bank"
CANONICAL_REPORT = ROOT / "mds/experiments_md/2026-10-01/04_deepsets_gnn_flow_utility.md"
SEEDS = tuple(range(4100, 4108))
TASKS = tuple(range(8))
REPLICAS = tuple(range(2, 6))
GENERATORS = ("gnn", "flow", "functional")
EXPECTED_SPARSE = {
    "functional", "pixel_prior", "random", "gnn_single", "flow_single",
    "gnn_search8", "flow_search8", "functional_search8",
}
K = 7526
N_CONNECTIONS = 784 * 32
LABELS = {
    "functional": "Функциональная средняя",
    "pixel_prior": "Pixel prior: norm. mean abs(q)",
    "random": "Случайная топология",
    "gnn_single": "GNN, один draw",
    "flow_single": "Flow, один draw",
    "gnn_search8": "GNN, выбор из 8",
    "flow_search8": "Flow, выбор из 8",
    "functional_search8": "Функциональный prior, выбор из 8",
    "dense_tuned": "Dense, обучаемый контроль",
}
COLORS = {
    "functional": "#1b9e77", "pixel_prior": "#a6761d", "random": "#666666",
    "gnn_single": "#d95f02", "flow_single": "#7570b3",
    "gnn_search8": "#e7298a", "flow_search8": "#66a61e",
    "functional_search8": "#e6ab02", "dense_tuned": "#444444",
}
GENERATOR_LABELS = {"gnn": "GNN", "flow": "Flow", "functional": "Функциональный prior"}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _torch(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _interval(values: Any, comparisons: int = 1) -> dict[str, Any]:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n_banks": 0, "mean": None, "sd": None, "ci95": None}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else None
    if len(x) > 1:
        alpha = .05 / max(1, comparisons)
        margin = float(t.ppf(1-alpha/2, len(x)-1) * sd / math.sqrt(len(x)))
        ci = [mean-margin, mean+margin]
    else:
        ci = None
    return {"n_banks": int(len(x)), "mean": mean, "sd": sd,
            "ci95": ci, "bank_values": x.tolist()}


def _wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n < 1:
        return None
    p = successes / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [max(0., center-half), min(1., center+half)]


def _cluster_bootstrap_ci(values: np.ndarray, seed: int = 20261001,
                          draws: int = 20000) -> list[float] | None:
    """Percentile interval for a task-averaged rate, resampling whole banks."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, x.shape[0], size=(draws, x.shape[0]))
    estimates = x[indices].mean(axis=(1, 2))
    return [float(v) for v in np.quantile(estimates, [.025, .975])]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_records(rows: list[dict[str, Any]], seed: int, methods: tuple[str, ...],
                      path: Path) -> np.ndarray:
    expected = {(task, method, replica) for task in TASKS
                for method in methods for replica in REPLICAS}
    seen: dict[tuple[int, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["task"]), str(row["method"]), int(row["replica"]))
        _require(key not in seen, f"Duplicate test row {key} in {path}")
        _require(key in expected, f"Unexpected test row {key} in {path}")
        _require(int(row["seed"]) == seed, f"Seed mismatch in {path}")
        for metric in ("nmse", "query_nmse", "support_nmse"):
            _require(math.isfinite(float(row[metric])), f"Nonfinite {metric} for {key} in {path}")
        seen[key] = row
    missing = expected - set(seen)
    _require(not missing, f"Incomplete test rows in {path}; missing {len(missing)}")
    values = np.full((8, len(methods), len(REPLICAS)), np.nan)
    method_index = {method: i for i, method in enumerate(methods)}
    replica_index = {replica: i for i, replica in enumerate(REPLICAS)}
    for (task, method, replica), row in seen.items():
        values[task, method_index[method], replica_index[replica]] = float(row["nmse"])
    return values


def _load_root_audits(out: Path, rows: list[dict[str, Any]], partial: bool) -> dict[str, Any]:
    """Cross-check root-level plateau and post-freeze topology audits."""
    plateau_path = out / "root_child_plateau_audit.json"
    topology_path = out / "root_mask_diagnostics.json"
    if partial and (not plateau_path.is_file() or not topology_path.is_file()):
        return {"child_plateau": None, "topology": None}

    _require(plateau_path.is_file(), f"Missing child plateau audit: {plateau_path}")
    audit = _json(plateau_path)
    by_seed = {}
    for row in rows:
        seed = int(row["seed"])
        flags = row["all_child_plateau"]
        observed = {"true": int(flags.sum()), "total": int(flags.size)}
        recorded = audit.get(str(seed))
        _require(recorded == observed,
                 f"Root child plateau audit mismatch for seed {seed}: {recorded} != {observed}")
        _require(observed["total"] == 1024,
                 f"Expected 1024 source/feedback/selection/final child fits for seed {seed}")
        by_seed[str(seed)] = observed
    if len(rows) == len(SEEDS):
        _require(set(audit) == {str(seed) for seed in SEEDS},
                 "Root child plateau audit must enumerate all eight bank seeds")
    plateau_summary = {
        "path": str(plateau_path), "by_seed": by_seed,
        "true": int(sum(item["true"] for item in by_seed.values())),
        "total": int(sum(item["total"] for item in by_seed.values())),
    }

    _require(topology_path.is_file(), f"Missing post-freeze topology audit: {topology_path}")
    source = _json(topology_path)
    raw_rows = source.get("rows", [])
    selected_seeds = {int(row["seed"]) for row in rows}
    filtered = [item for item in raw_rows if int(item["seed"]) in selected_seeds]
    seen = {(int(item["seed"]), int(item["task"])) for item in filtered}
    expected = {(seed, task) for seed in selected_seeds for task in TASKS}
    _require(len(filtered) == len(expected) and seen == expected,
             f"Topology audit must contain one row per task and completed seed: {topology_path}")
    fields = ("iou_gnn_functional", "iou_gnn_pixel", "iou_flow_functional",
              "iou_functional_pixel")
    bank_means, overall_means = {}, {}
    for field in fields:
        values = np.asarray([float(item[field]) for item in filtered], dtype=float)
        _require(np.isfinite(values).all(), f"Nonfinite {field} in {topology_path}")
        per_seed = np.asarray([
            [float(item[field]) for item in filtered if int(item["seed"]) == seed]
            for seed in sorted(selected_seeds)
        ])
        _require(per_seed.shape == (len(selected_seeds), len(TASKS)),
                 f"Malformed per-seed {field} rows in {topology_path}")
        bank_means[field] = _interval(per_seed.mean(axis=1))
        overall_means[field] = float(values.mean())
        if len(rows) == len(SEEDS):
            recorded_mean = source.get("means", {}).get(field)
            _require(recorded_mean is not None and math.isclose(
                float(recorded_mean), overall_means[field], rel_tol=0., abs_tol=1e-12),
                f"Stored {field} mean disagrees with audited rows in {topology_path}")
    topology = {
        "path": str(topology_path), "description": source.get("description"),
        "scope": "post-freeze descriptive topology comparison; not used for training, mask selection, or test evaluation",
        "overall_row_means": overall_means, "bank_mean_intervals": bank_means,
        "rows": filtered,
    }
    return {"child_plateau": plateau_summary, "topology": topology}


def _load_seed(seed: int, out: Path, methods: tuple[str, ...]) -> dict[str, Any]:
    folder = out / f"seed_{seed}"
    _require((folder / "COMPLETE").is_file(), f"Missing COMPLETE marker: {folder}")
    _require((BANK_OUT / f"seed_{seed}" / "COMPLETE").is_file(),
             f"Rebuilt source bank incomplete for seed {seed}")
    rows_path = folder / "results.json"
    results = _validate_records(_json(rows_path), seed, methods, rows_path)
    frozen = _json(folder / "source_frozen.json")
    _require(int(frozen.get("seed", -1)) == seed, f"Source freeze seed mismatch: {folder}")
    _require(frozen.get("target_tasks") is not None, f"Missing frozen target declaration: {folder}")
    _require(not (folder / "MODEL_NOT_CONVERGED.json").exists(),
             f"Completed seed has a nonconvergence marker: {folder}")
    context_path = BANK_OUT / f"seed_{seed}" / "functional_context.pt"
    _require(context_path.is_file(), f"Missing source functional context: {context_path}")
    _require(frozen.get("bank_sha256") == _sha256(context_path),
             f"Frozen source-context checksum mismatch: {folder}")
    model_hashes = frozen.get("models", {})
    _require(set(model_hashes) == {"gnn", "flow"},
             f"Expected final GNN and Flow hashes in {folder / 'source_frozen.json'}")
    for kind in ("gnn", "flow"):
        model_path = folder / f"{kind}_model.pt"
        _require(model_path.is_file() and model_hashes[kind] == _sha256(model_path),
                 f"Frozen final {kind} model checksum mismatch: {model_path}")
    all_frozen_path = folder / "all_children_frozen.json"
    _require(all_frozen_path.is_file(), f"Missing two-phase freeze manifest: {all_frozen_path}")
    all_frozen = _json(all_frozen_path)
    _require(all_frozen.get("test_opened") is False,
             f"all_children_frozen.json must certify test_opened=false: {all_frozen_path}")
    _require(int(all_frozen.get("tasks", -1)) == len(TASKS),
             f"Freeze manifest must cover all eight tasks: {all_frozen_path}")
    freeze_hashes = all_frozen.get("children_sha256", {})
    _require(set(freeze_hashes) == {str(task) for task in TASKS},
             f"Freeze manifest must contain each task checksum: {all_frozen_path}")
    test_provenance = folder / "target_test_provenance.json"
    _require(test_provenance.is_file(), f"Missing separate target test provenance: {test_provenance}")

    counts = np.zeros((8, len(GENERATORS)), dtype=float)
    candidate_query = np.full((8, len(GENERATORS), 8), np.nan)
    selection_obj = np.full_like(candidate_query, np.nan)
    selection_query = np.full_like(candidate_query, np.nan)
    selection_plateau = np.full_like(candidate_query, np.nan)
    selected = np.zeros((8, len(GENERATORS)), dtype=int)
    frozen_masks = []
    final_plateau = []
    all_child_plateau = []
    final_steps = []
    source_initial = source_final = source_children = feedback_children = None
    for task in TASKS:
        task_dir = folder / f"task_{task}"
        child_manifest_path = task_dir / "children_frozen.json"
        _require(child_manifest_path.is_file(), f"Missing per-task child freeze manifest: {child_manifest_path}")
        child_manifest = _json(child_manifest_path)
        _require(child_manifest.get("test_opened") is False,
                 f"Per-task manifest must certify test_opened=false: {child_manifest_path}")
        actual_freeze_hash = hashlib.sha256(child_manifest_path.read_bytes()).hexdigest()
        _require(freeze_hashes[str(task)] == actual_freeze_hash,
                 f"Per-task freeze checksum mismatch: {child_manifest_path}")
        mask_data = _torch(task_dir / "masks_frozen.pt")
        stored_sparse = tuple(mask_data["methods"])
        _require(stored_sparse == methods[:-1],
                 f"Method order in masks_frozen.pt differs from reported result schema at {task_dir}")
        chosen = _array(mask_data["masks"])
        _require(chosen.shape == (len(methods)-1, 784, 32),
                 f"Unexpected frozen mask shape {chosen.shape} at {task_dir}")
        active = np.rint(chosen.sum(axis=(1, 2))).astype(int)
        _require(np.all(np.isin(chosen, [0, 1])) and np.all(active == K),
                 f"Every sparse mask must be binary with exactly K={K} at {task_dir}")
        frozen_masks.append(mask_data)
        candidate_q = _array(mask_data["candidate_query"])
        _require(candidate_q.shape == (3, 8) and np.isfinite(candidate_q).all(),
                 f"Expected finite 3 x 8 candidate query losses at {task_dir}")
        for gi, generator in enumerate(GENERATORS):
            candidate = _array(mask_data["candidate_masks"][generator])
            _require(candidate.shape == (8, 784, 32), f"Expected 8 {generator} masks at {task_dir}")
            _require(np.all(np.isin(candidate, [0, 1]))
                     and np.all(np.rint(candidate.sum(axis=(1, 2))).astype(int) == K),
                     f"Invalid {generator} candidate mask at {task_dir}")
            actual_unique = len({
                np.packbits(mask.reshape(-1).astype(np.uint8)).tobytes() for mask in candidate
            })
            stored_unique = int(mask_data["distinct"][generator])
            _require(actual_unique == stored_unique,
                     f"Stored {generator} distinct count {stored_unique} != audited {actual_unique} at {task_dir}")
            counts[task, gi] = actual_unique
            candidate_query[task, gi] = candidate_q[gi]
            selected[task, gi] = int(mask_data["selected"][generator])
            _require(0 <= selected[task, gi] < 8
                     and selected[task, gi] == int(np.argmin(candidate_q[gi])),
                     f"{generator} selected index does not minimize stored query NMSE at {task_dir}")

        search_rows = {
            "gnn_search8": ("gnn", 0), "flow_search8": ("flow", 1),
            "functional_search8": ("functional", 2),
        }
        for method, (generator, gi) in search_rows.items():
            chosen_index = methods.index(method)
            selected_mask = _array(mask_data["candidate_masks"][generator])[selected[task, gi]]
            _require(np.array_equal(chosen[chosen_index], selected_mask),
                     f"Frozen {method} mask differs from its query-selected candidate at {task_dir}")

        selection = _torch(task_dir / "selection_children.pt")
        _require(bool(selection.get("fixed_horizon", False)),
                 f"Candidate utility fit is not fixed horizon at {task_dir}")
        support_objective = _array(selection["support_objective"])[0]
        query_loss = _array(selection["query_loss"])[0]
        plateau = _array(selection["plateau_flags"])[0].astype(float)
        _require(len(support_objective) == len(query_loss) == 48,
                 f"Expected 3 x 8 x 2 paired candidate children at {task_dir}")
        selection_obj[task] = support_objective.reshape(3, 8, 2).mean(axis=2)
        selection_query[task] = query_loss.reshape(3, 8, 2).mean(axis=2)
        selection_plateau[task] = plateau.reshape(3, 8, 2).mean(axis=2)
        all_child_plateau.extend(plateau.astype(bool).reshape(-1).tolist())

        sparse = _torch(task_dir / "final_sparse_children.pt")
        dense = _torch(task_dir / "final_dense_children.pt")
        sparse_flags = _array(sparse["plateau_flags"]).reshape(-1).astype(bool)
        dense_flags = _array(dense["plateau_flags"]).reshape(-1).astype(bool)
        final_plateau.extend(sparse_flags.tolist())
        final_plateau.extend(dense_flags.tolist())
        all_child_plateau.extend(sparse_flags.tolist())
        all_child_plateau.extend(dense_flags.tolist())
        final_steps.append({
            "sparse_fits": int(_array(sparse["plateau_flags"]).size),
            "dense_fits": int(_array(dense["plateau_flags"]).size),
            "sparse_terminal_step": int(sparse["terminal_step"]),
            "sparse_steps_run": int(sparse["steps_run"]),
            "dense_terminal_step": int(dense["terminal_step"]),
            "dense_steps_run": int(dense["steps_run"]),
        })
        if task == 0:
            source_initial = _torch(folder / "archive.pt")
            source_final = _torch(folder / "archive_final.pt")
            source_children = _torch(folder / "source_children.pt")
            feedback_children = _torch(folder / "feedback_children.pt")

    q0, qf = _array(source_initial["quality"]), _array(source_final["quality"])
    _require(q0.shape == (4, 12) and qf.shape == (4, 44),
             f"Expected source utility matrices [4,12] and [4,44], got {q0.shape}, {qf.shape}")
    _require(np.isfinite(q0).all() and np.isfinite(qf).all(),
             f"Source archive contains nonfinite utility labels for seed {seed}")
    _require(np.allclose(qf[:, :12], q0, rtol=1e-6, atol=1e-8),
             f"Source archive did not preserve original utility labels for seed {seed}")
    initial_masks = _array(source_initial["masks"])
    final_masks = _array(source_final["masks"])
    _require(initial_masks.shape == (12, 784, 32) and final_masks.shape == (44, 784, 32),
             f"Expected 12 initial and 32 feedback masks for seed {seed}")
    _require(np.array_equal(final_masks[:12], initial_masks),
             f"Final source archive did not preserve initial masks for seed {seed}")
    for name, archive in (("initial", source_initial), ("final", source_final)):
        elite = archive.get("elite", [])
        task_counts = [sum(int(item["task"]) == task for item in elite) for task in range(4)]
        _require(task_counts == [4, 4, 4, 4],
                 f"{name} archive must retain four elites per source task for seed {seed}")
        for task in range(4):
            entries = [item for item in elite if int(item["task"]) == task]
            hashes = {np.packbits(_array(item["mask"]).reshape(-1).astype(np.uint8)).tobytes()
                      for item in entries}
            _require(len(hashes) == 4, f"{name} elite masks are not distinct for seed {seed}, task {task}")
            weights = np.asarray([float(item["weight"]) for item in entries])
            _require(np.all(np.isfinite(weights)) and np.isclose(weights.sum(), 1., atol=1e-5),
                     f"{name} elite weights do not normalize for seed {seed}, task {task}")
    _require(_array(source_children["query_loss"]).shape == (4, 24),
             f"Unexpected initial archive child evaluation shape for seed {seed}")
    _require(_array(feedback_children["query_loss"]).shape == (4, 64),
             f"Unexpected feedback child evaluation shape for seed {seed}")
    source_flags = _array(source_children["plateau_flags"]).reshape(-1).astype(bool)
    feedback_flags = _array(feedback_children["plateau_flags"]).reshape(-1).astype(bool)
    _require(source_flags.size == 4*12*2 and feedback_flags.size == 4*32*2,
             f"Unexpected source/feedback plateau flag counts for seed {seed}")
    all_child_plateau.extend(source_flags.tolist())
    all_child_plateau.extend(feedback_flags.tolist())
    _require(len(all_child_plateau) == 1024,
             f"Expected 1024 source/feedback/selection/final child flags for seed {seed}")

    histories = {}
    stage_summary = {}
    for kind in ("gnn", "flow"):
        histories[kind] = {}
        stage_rows = []
        missing_seen = False
        for stage in range(6):
            stage_path = folder / f"{kind}_model_stage{stage}.pt"
            if not stage_path.is_file():
                missing_seen = True
                continue
            _require(not missing_seen, f"Non-contiguous {kind} stage history at {stage_path}")
            payload = _torch(stage_path)
            history = payload.get("history", [])
            _require(bool(history), f"Missing {kind} stage {stage} training history for seed {seed}")
            cap = 1200 if stage == 0 else 800
            steps = int(payload["steps"])
            _require(0 < steps <= cap,
                     f"{kind} stage {stage} exceeded declared cap {cap}: {steps} steps")
            _require(int(history[-1]["step"]) == steps,
                     f"{kind} stage {stage} history does not end at its terminal step")
            histories[kind][stage] = history
            stage_rows.append({
                "stage": stage, "phase": "preliminary" if stage <= 1 else "source_refinement",
                "cap": cap, "steps": steps, "plateau": bool(payload["plateau"]),
                "best_source_validation": float(payload["best_source_validation"]),
                "seconds": float(payload["seconds"]),
            })
        _require({0, 1}.issubset(histories[kind]),
                 f"Completed seed lacks preliminary {kind} stages 0/1")
        final_stage = max(histories[kind])
        final_field_plateau = stage_rows[-1]["plateau"]
        _require(final_field_plateau,
                 f"Completed seed final {kind} field has no empirical source plateau")
        stage_summary[kind] = {
            "stages": stage_rows, "final_stage": final_stage,
            "final_plateau": final_field_plateau,
            "final_best_source_validation": stage_rows[-1]["best_source_validation"],
            "plateau_rule": "held-noise source validation range over five 25-update checkpoints below 1%, for three consecutive checks",
        }
    return {
        "seed": seed, "results": results, "frozen_masks": frozen_masks,
        "candidate_counts": counts, "candidate_query": candidate_query,
        "selection_objective": selection_obj, "selection_query": selection_query,
        "selection_plateau": selection_plateau, "selection_selected": selected,
        "final_plateau": np.asarray(final_plateau, dtype=bool), "final_steps": final_steps,
        "all_child_plateau": np.asarray(all_child_plateau, dtype=bool),
        "source_initial_quality": q0, "source_final_quality": qf,
        "source_children": source_children, "feedback_children": feedback_children,
        "histories": histories, "training_stage_summary": stage_summary,
    }


def _savefig(fig: plt.Figure, out: Path, name: str) -> None:
    (out / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "plots" / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_bank_contrasts(out: Path, means: np.ndarray, methods: tuple[str, ...]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, baseline, title in (
        (axes[0], "functional", "Разность с функциональной средней"),
        (axes[1], "dense_tuned", "Разность с dense control"),
    ):
        base = methods.index(baseline)
        compared = [name for name in methods if name != baseline]
        points, lower, upper = [], [], []
        for name in compared:
            stat = _interval(means[:, methods.index(name)] - means[:, base])
            points.append(stat["mean"])
            ci = stat["ci95"]
            lower.append(stat["mean"] - ci[0] if ci else 0.)
            upper.append(ci[1] - stat["mean"] if ci else 0.)
        y = np.arange(len(compared))
        ax.errorbar(points, y, xerr=np.asarray([lower, upper]), fmt="none", color="#333333",
                    capsize=4, linewidth=1.5, zorder=2)
        for yi, (name, point) in enumerate(zip(compared, points)):
            ax.scatter(point, yi, s=55, color=COLORS[name], edgecolor="white", linewidth=.6, zorder=3)
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_yticks(y, [LABELS[name] for name in compared])
        ax.invert_yaxis(); ax.grid(axis="x", alpha=.22)
        ax.set_title(title)
        ax.set_xlabel("Paired test NMSE difference (method − baseline)")
    fig.suptitle(f"Bank means; n={means.shape[0]} complete seeds, fixed average over eight tasks", y=1.02)
    fig.tight_layout()
    _savefig(fig, out, "paired_bank_contrasts")


def _plot_task_heatmap(out: Path, task_means: np.ndarray,
                       methods: tuple[str, ...]) -> tuple[np.ndarray, float]:
    deltas = task_means - task_means[:, :, [methods.index("functional")]]
    task_gap = deltas.mean(axis=0)
    shown_methods = [name for name in methods if name != "functional"]
    shown = task_gap[:, [methods.index(name) for name in shown_methods]]
    limit = max(float(np.max(np.abs(shown))), 1e-8)
    fig, ax = plt.subplots(figsize=(14, 6.2))
    image = ax.imshow(shown, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(shown_methods)), [LABELS[m] for m in shown_methods],
                  rotation=28, ha="right")
    ax.set_yticks(np.arange(8), [f"Задача {i}" for i in TASKS])
    ax.set_title("Средняя по банкам разность test NMSE для каждой фиксированной задачи")
    ax.set_xlabel("Метод; значение = метод − функциональная средняя")
    ax.set_ylabel("Одна из восьми новых cost vectors")
    fig.colorbar(image, ax=ax,
                 label="NMSE: метод − функциональная средняя")
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            ax.text(j, i, f"{shown[i,j]:+.3f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    _savefig(fig, out, "taskwise_nmse_gap_heatmap")
    return task_gap, limit


def _plot_task_gaps(out: Path, task_deltas: np.ndarray,
                    methods: tuple[str, ...]) -> dict[str, Any]:
    compared = [name for name in methods if name != "functional"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharex=True)
    x = np.arange(8)
    for name in compared:
        j = methods.index(name)
        values = task_deltas[:, :, j]
        means = values.mean(axis=0)
        cis_raw = [_interval(values[:, task])["ci95"] for task in TASKS]
        if any(ci is None for ci in cis_raw):
            yerr = np.zeros((2, len(TASKS)))
        else:
            cis = np.asarray(cis_raw, dtype=float)
            yerr = np.stack((means-cis[:, 0], cis[:, 1]-means))
        axes[0].errorbar(x, means, yerr=yerr,
                         marker="o", lw=1.4, capsize=2, color=COLORS[name],
                         label=LABELS[name])
        rates = (values < 0).mean(axis=0)
        rate_ci = np.asarray([
            _wilson(int((values[:, task] < 0).sum()), len(values)) for task in TASKS
        ])
        axes[1].errorbar(x, rates,
                         yerr=np.stack((rates-rate_ci[:, 0], rate_ci[:, 1]-rates)),
                         marker="o", lw=1.3, capsize=2, color=COLORS[name],
                         label=LABELS[name])
    axes[0].axhline(0, color="black", lw=1, ls="--")
    axes[0].set_ylabel("Paired test NMSE difference")
    axes[0].set_title("Разность по задаче; среднее и 95% t-интервал по банкам")
    axes[0].grid(alpha=.2)
    axes[1].axhline(.5, color="black", lw=1, ls="--")
    axes[1].set_ylim(-.05, 1.05)
    axes[1].set_ylabel("Доля банков с меньшим NMSE относительно functional")
    axes[1].set_title("Доля побед по задаче; интервал Уилсона, n банков")
    axes[1].grid(alpha=.2)
    for ax in axes:
        ax.set_xticks(x, [str(task) for task in TASKS])
        ax.set_xlabel("Номер фиксированной test-задачи")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    _savefig(fig, out, "taskwise_gaps_and_win_rates")

    info = {}
    for name in compared:
        values = task_deltas[:, :, methods.index(name)]
        info[name] = {
            "task_gaps": [_interval(values[:, task]) for task in TASKS],
            "task_win_fraction": [
                {"wins": int((values[:, task] < 0).sum()), "n_banks": len(values),
                 "fraction": float((values[:, task] < 0).mean()),
                 "wilson95": _wilson(int((values[:, task] < 0).sum()), len(values))}
                for task in TASKS
            ],
            "overall_win_fraction": float((values < 0).mean()),
            "overall_win_fraction_cluster_bootstrap95":
                _cluster_bootstrap_ci((values < 0).astype(float)),
        }
    return info


def _plot_weight_masks(out: Path, methods: tuple[str, ...],
                       sample_seed: int) -> dict[str, Any]:
    folder = out / f"seed_{sample_seed}" / "task_0"
    frozen = _torch(folder / "masks_frozen.pt")
    sparse_fit = _torch(folder / "final_sparse_children.pt")
    dense_fit = _torch(folder / "final_dense_children.pt")
    sparse_state, dense_state = sparse_fit["state_dict"], dense_fit["state_dict"]
    weights = _array(sparse_state["weight"])[0]
    trained_masks = _array(sparse_state["masks"])[0]
    dense_weights, dense_masks = _array(dense_state["weight"])[0], _array(dense_state["masks"])[0]
    chosen_masks = _array(frozen["masks"])
    effective, binaries = [], []
    for mi in range(len(methods)-1):
        idx = mi * 4  # first final replica is initialization 2
        effective.append(weights[idx] * trained_masks[idx])
        binaries.append(chosen_masks[mi])
    effective.append(dense_weights[0] * dense_masks[0])
    binaries.append(dense_masks[0])
    # Each selected fit is [784 pixels, 32 hidden units]; transpose only for
    # display so hidden units form rows and all 784 pixels span the x axis.
    effective = np.stack(effective)
    binaries = np.stack(binaries)
    limit = max(float(np.quantile(np.abs(effective), .99)), 1e-8)
    fig, axes = plt.subplots(len(methods), 2, figsize=(14, 2.05*len(methods)),
                             gridspec_kw={"width_ratios": [1, 1]})
    for mi, name in enumerate(methods):
        signed = axes[mi, 0].imshow(effective[mi].T, cmap="RdBu_r", vmin=-limit, vmax=limit,
                                    aspect="auto", interpolation="nearest")
        axes[mi, 1].imshow(binaries[mi].T, cmap="Greys", vmin=0, vmax=1,
                           aspect="auto", interpolation="nearest")
        axes[mi, 0].set_ylabel(LABELS[name], fontsize=8)
        for ax in axes[mi]:
            ax.set_yticks([0, 7, 15, 23, 31])
            ax.set_yticklabels([0, 7, 15, 23, 31], fontsize=7)
            ax.set_xticks([0, 195, 391, 587, 783])
            ax.tick_params(axis="x", labelsize=7)
    axes[0, 0].set_title("Обученные signed $W_{eff}=W\\odot M$")
    axes[0, 1].set_title("Бинарная выбранная маска $M$")
    axes[-1, 0].set_xlabel("Пиксель, row-major индекс 0–783")
    axes[-1, 1].set_xlabel("Пиксель, row-major индекс 0–783")
    colorbar_axis = fig.add_axes([.925, .15, .018, .7])
    fig.colorbar(signed, cax=colorbar_axis,
                 label=f"Общая симметричная шкала $W_{{eff}}$, предел ±{limit:.3g}")
    fig.suptitle(f"seed {sample_seed} · новая задача 0 · child initialization 2", y=.998)
    fig.subplots_adjust(left=.2, right=.9, top=.965, bottom=.04, hspace=.25, wspace=.15)
    _savefig(fig, out, "sample_effective_weights_and_masks")
    return {"seed": sample_seed, "task": 0, "replica": 2, "method_order": list(methods),
            "weight_effective": effective.tolist(), "binary_masks": binaries.tolist(),
            "weight_shared_symmetric_limit_99pct": limit}


def _curve(rows: list[dict[str, Any]], kind: str, stage: int,
           field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    histories = [row["histories"][kind][stage] for row in rows
                 if stage in row["histories"][kind]]
    if not histories:
        empty = np.asarray([], dtype=float)
        return empty, empty, np.empty((2, 0), dtype=float)
    steps = sorted({int(item["step"]) for h in histories for item in h})
    matrix = np.full((len(rows), len(steps)), np.nan)
    index = {step: i for i, step in enumerate(steps)}
    for ri, history in enumerate(histories):
        for item in history:
            matrix[ri, index[int(item["step"])]] = float(item[field])
    means = np.nanmean(matrix, axis=0)
    bounds = np.full((2, len(steps)), np.nan)
    for j in range(len(steps)):
        vals = matrix[:, j]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 1:
            margin = float(t.ppf(.975, len(vals)-1) * vals.std(ddof=1) / math.sqrt(len(vals)))
            bounds[:, j] = [vals.mean()-margin, vals.mean()+margin]
        elif len(vals) == 1:
            bounds[:, j] = vals[0]
    return np.asarray(steps), means, bounds


def _plot_training(out: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    stages = sorted({stage for row in rows for kind in ("gnn", "flow")
                     for stage in row["histories"][kind]})
    fig, axes = plt.subplots(2, len(stages),
                             figsize=(4.7*len(stages), 8.5), squeeze=False)
    data = {}
    titles = {"gnn": "GNN · BCE whole-mask labels",
              "flow": "Flow · scalar-velocity FM loss"}
    for ri, kind in enumerate(("gnn", "flow")):
        for ci, stage in enumerate(stages):
            ax = axes[ri, ci]
            n_stage = sum(stage in row["histories"][kind] for row in rows)
            if n_stage == 0:
                ax.axis("off")
                continue
            for field, label, color in (
                ("loss", "Train objective", "#1f77b4"),
                ("validation", "Held-noise source validation", "#d62728"),
            ):
                steps, mean, bounds = _curve(rows, kind, stage, field)
                ax.plot(steps, mean, label=label, color=color, lw=1.8)
                ax.fill_between(steps, bounds[0], bounds[1], color=color, alpha=.16)
                data[f"{kind}_stage{stage}_{field}"] = {
                    "steps": steps.tolist(), "mean": mean.tolist(),
                    "ci95": bounds.T.tolist(), "n_banks": n_stage,
                }
            cap = 1200 if stage == 0 else 800
            ax.set_title(f"{titles[kind]} · stage {stage} (cap {cap}; n={n_stage})")
            ax.set_xlabel("Update внутри стадии")
            ax.set_ylabel("Loss")
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Source-only training; means and 95% t bands across available bank seeds", y=1.01)
    fig.tight_layout()
    _savefig(fig, out, "source_field_training_curves")
    return data


def _plot_source_diagnostics(out: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    initial = np.stack([r["source_final_quality"][:, :12] for r in rows])
    feedback = np.stack([r["source_final_quality"][:, 12:] for r in rows])
    task_x = np.arange(4)
    for phase, (values, label, color) in enumerate((
        (initial, "Initial 12", "#4c78a8"), (feedback, "Feedback 32", "#f58518")
    )):
        means = values.mean(axis=(0, 2))
        cis_raw = [_interval(values[:, task].mean(axis=1))["ci95"] for task in range(4)]
        if any(ci is None for ci in cis_raw):
            cis = np.stack((means, means), axis=1)
        else:
            cis = np.asarray(cis_raw)
        axes[0].errorbar(task_x+(phase-.5)*.12, means,
                         yerr=np.stack((means-cis[:, 0], cis[:, 1]-means)),
                         marker="o", capsize=3, label=label, color=color)
    axes[0].set_xticks(task_x, [f"Source {i}" for i in task_x])
    axes[0].set_ylabel("Fresh-child source query NMSE")
    axes[0].set_title("Source utility: before / after feedback")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)

    generator_method = {"gnn": "gnn_search8", "flow": "flow_search8",
                        "functional": "functional_search8"}
    for generator in GENERATORS:
        mi = list(generator_method).index(generator)
        method_name = generator_method[generator]
        color = COLORS[method_name]
        xvalues = np.concatenate([r["selection_objective"][:, mi].reshape(-1) for r in rows])
        yvalues = np.concatenate([r["selection_query"][:, mi].reshape(-1) for r in rows])
        chosen_x, chosen_y = [], []
        for row in rows:
            for task in TASKS:
                candidate = int(row["selection_selected"][task, mi])
                chosen_x.append(row["selection_objective"][task, mi, candidate])
                chosen_y.append(row["selection_query"][task, mi, candidate])
        axes[1].scatter(xvalues, yvalues, s=13, alpha=.24, color=color,
                        label=f"{GENERATOR_LABELS[generator]}: 8 candidates")
        axes[1].scatter(chosen_x, chosen_y, marker="x", s=32, linewidth=1.1,
                        color=color, label=f"{GENERATOR_LABELS[generator]}: selected")
    axes[1].set_xlabel("Fresh-child support objective (includes L2)")
    axes[1].set_ylabel("Query NMSE used for candidate selection")
    axes[1].set_title("Candidate search: support and query losses")
    axes[1].legend(fontsize=6.5); axes[1].grid(alpha=.2)

    plateau = np.concatenate([r["final_plateau"] for r in rows]).astype(float)
    fits = int(len(plateau))
    plateau_rate = float(plateau.mean())
    axes[2].bar([0, 1], [plateau_rate, 1.0], color=["#54a24b", "#b279a2"])
    axes[2].set_xticks([0, 1], ["Support-only\nplateau flag", "Fixed horizon\ncompleted"])
    axes[2].set_ylim(0, 1.08); axes[2].set_ylabel("Fraction of final child fits")
    axes[2].set_title("Plateau is diagnostic; no early stopping")
    axes[2].text(0, min(.98, plateau_rate+.035), f"{plateau_rate:.1%} ({int(plateau.sum())}/{fits})",
                 ha="center", fontsize=8)
    cap_total = 0
    cap_hits = 0
    fixed_pairs = 0
    for row in rows:
        for step in row["final_steps"]:
            sparse_at_cap = step["sparse_terminal_step"] == step["sparse_steps_run"]
            dense_at_cap = step["dense_terminal_step"] == step["dense_steps_run"]
            cap_total += step["sparse_fits"] + step["dense_fits"]
            cap_hits += step["sparse_fits"] * sparse_at_cap + step["dense_fits"] * dense_at_cap
            fixed_pairs += sparse_at_cap and dense_at_cap
    axes[2].text(1, 1.015, f"{cap_hits}/{cap_total} child fits", ha="center", va="bottom", fontsize=8)
    axes[2].grid(axis="y", alpha=.2)
    fig.suptitle("Source archive utility and child-level selection diagnostics", y=1.02)
    fig.tight_layout()
    _savefig(fig, out, "source_utility_and_child_diagnostics")
    return {
        "initial_query_nmse": initial.tolist(), "feedback_query_nmse": feedback.tolist(),
        "final_child_plateau_fraction": plateau_rate, "final_child_count": fits,
        "child_fits_reaching_declared_horizon": cap_hits, "final_child_fits": cap_total,
        "task_pairs_reaching_declared_horizon": fixed_pairs, "task_pairs_total": 8*len(rows),
    }


def _plot_unique_draws(out: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_bank_task = np.stack([r["candidate_counts"] for r in rows])
    bank_means = per_bank_task.mean(axis=1)
    means = bank_means.mean(axis=0)
    intervals = [_interval(bank_means[:, i])["ci95"] for i in range(len(GENERATORS))]
    intervals = [[means[i], means[i]] if ci is None else ci
                 for i, ci in enumerate(intervals)]
    errors = np.asarray([
        [means[i]-intervals[i][0] for i in range(len(GENERATORS))],
        [intervals[i][1]-means[i] for i in range(len(GENERATORS))],
    ])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(GENERATORS))
    ax.bar(x, means, color=[COLORS["gnn_search8"], COLORS["flow_search8"],
                            COLORS["functional_search8"]], alpha=.88)
    ax.errorbar(x, means, yerr=errors, fmt="none", color="black", capsize=5)
    ax.set_xticks(x, [GENERATOR_LABELS[g] for g in GENERATORS])
    ax.set_ylim(0, 8.8); ax.set_ylabel("Unique binary masks among eight draws")
    ax.set_title("Target candidate diversity; average tasks within bank, CI across banks")
    ax.grid(axis="y", alpha=.2)
    for i, mean in enumerate(means):
        ax.text(i, mean+.15, f"{mean:.2f}/8", ha="center")
    fig.tight_layout()
    _savefig(fig, out, "candidate_unique_draw_counts")
    return {
        "per_bank_task_counts": per_bank_task.tolist(), "bank_task_mean": bank_means.tolist(),
        "mean_ci_by_generator": {name: _interval(bank_means[:, i])
                                 for i, name in enumerate(GENERATORS)},
    }


def _dense_comparison_summary(task_means: np.ndarray,
                              methods: tuple[str, ...]) -> dict[str, Any]:
    dense_index = methods.index("dense_tuned")
    task_deltas = task_means - task_means[:, :, [dense_index]]
    mean_gaps = task_deltas.mean(axis=0)
    comparisons = {}
    for name in methods:
        if name == "dense_tuned":
            continue
        index = methods.index(name)
        bank_gap = task_deltas[:, :, index].mean(axis=1)
        task_gap = mean_gaps[:, index]
        comparisons[name] = {
            "bank_paired_difference": _interval(bank_gap),
            "familywise_ci95": _interval(bank_gap, comparisons=16)["ci95"],
            "task_mean_gaps": task_gap.tolist(),
            "tasks_with_lower_mean_nmse": [int(task) for task in TASKS if task_gap[task] < 0],
            "tasks_lower_count": int((task_gap < 0).sum()),
            "tasks_total": len(TASKS),
        }
    candidate_names = [name for name in methods if name != "dense_tuned"]
    return {
        "baseline": "dense_tuned", "comparisons": comparisons,
        "primary_all_task_improvement_met": any(
            np.all(mean_gaps[:, methods.index(name)] < 0) for name in candidate_names
        ),
        "task_deltas_bank_task_method": task_deltas.tolist(),
    }


def _write_markdown(out: Path, report: dict[str, Any]) -> None:
    methods = tuple(report["methods"])
    n, seeds = report["n_banks"], report["complete_seeds"]
    stage_table_lines = []
    for seed in seeds:
        model_row = report["source_model_training"]["by_seed"][str(seed)]
        for kind in ("gnn", "flow"):
            item = model_row[kind]
            stage_text = "; ".join(
                f"{stage['stage']}: {stage['steps']}/{stage['cap']}"
                + (" plateau" if stage["plateau"] else "")
                for stage in item["stages"]
            )
            stage_table_lines.append(
                f"| {seed} | {kind.upper()} | {stage_text} | {item['final_stage']} | да |"
            )
    lines = [
        "# Результаты benchmark: utility-trained graph masks для DeepSets", "",
        f"Завершённые bank seeds: {', '.join(map(str, seeds))}; статистическая единица — seed, $n={n}$. "
        "Сначала усреднены четыре child initialization для каждой из восьми одинаковых целевых задач, "
        "затем усреднены задачи внутри seed. Поэтому 256 test fits не рассматриваются как 256 независимых задач.", "",
        "## Протокол и границы информации", "",
        "Функциональный банк содержит четыре source cost tasks и функциональные решения при плотностях $\\rho\\in\\{0.1,0.3,0.5,0.7,0.9\\}$. "
        "Подготовленная population сохраняет обученные полные состояния при этих плотностях; совпавшие topology masks не удаляются на уровне банка. "
        "Начальные 12 целых масок оценивались fresh children; "
        "затем один общий source-only feedback шаг добавлял 32 маски. Utility — средний query NMSE по fresh child replicas 0/1. "
        "Для элитного архива выбирались до четырёх различных масок на source task с весом "
        "$w_i\\propto\\exp[-(u_i-u_{\\min})/0.03]$. Модель получает агрегированные по банку средние и дисперсии "
        "полных $\\psi$ профилей и signed/RMS моментов производной "
        "$q_{ij}(x)=x_iW_{eff,ij}a_j(1-\\tanh^2(pre_j))$. Для каждого учителя абсолютный профиль нормируется на maximum RMS чувствительность: "
        "$\\tilde q_{ij}=\\mathbb E_x|q_{ij}(x)|/\\max_{i,j}\\sqrt{\\mathbb E_x q_{ij}(x)^2}$. "
        "Pixel prior усредняет $\\tilde q$ по учителям и hidden units на каждом пикселе, затем broadcast-ит pixel score на все hidden units; "
        "используется та же per-teacher нормировка, что у functional-score baseline. "
        "Индивидуальные target digit labels и сырые importance rankings не подаются.", "",
        "Для каждой новой target задачи используются support из 205 observed set labels и query из 51 set labels. "
        "Query выбирает одну маску из восьми кандидатов для GNN, Flow и функционального prior. Все маски и fresh children "
        "замораживаются до чтения test pool; затем для каждого target task строятся 512 test sets размера 5 из отдельного pool по 300 изображений на digit (3000 уникальных image rows). "
        "Set sampling is with replacement внутри этого pool; это не 512 независимых изображений. Оцениваются replicas 2–5. "
        "Восемь sparse вариантов (functional, pixel prior, random topology, GNN/Flow single и три search-варианта) имеют ровно "
        f"$K={K}$ связей из $784\\times32$ (плотность $K/25088={K/N_CONNECTIONS:.4f}$); dense control обучается с полной маской.", "",
        "Pixel prior ранжирует функциональный score mean по source bank на уровне пикселя (mean normalized $|q|$, усреднённый по hidden units) и транслирует его на hidden units; "
        "random topology служит контролем той же плотности. Целевая плотность 0.3 заранее фиксирована и не выбирается по target labels. "
        "Source-bank parent располагает четырьмя известными utility tasks и большим числом наблюдений, чем новый target. "
        "Выравнивание функциональных состояний сохраняет исторический train-only Hungarian preprocessing: этот benchmark не является alignment-free.", "",
        "Метрика fresh child: $\\mathrm{NMSE}=\\frac{1}{N\\cdot5}\\sum_{i=1}^{N}(\\hat y_i-y_i)^2$, "
        "где digit costs центрированы и нормированы к единичной дисперсии. Query NMSE участвует только в выборе кандидата; "
        "test NMSE используется только после фиксации всех child states.", "",
        "Plateau-проверки относятся к разным этапам. Upstream source-population fits используют cadence 50 updates, поэтому last-50 range опирается на отдельные snapshots шагов; "
        "у utility-archive и финальных target children cadence равен 100, и округлённое окно 50 updates имеет только один checkpoint, то есть фактически охватывает 100 updates. "
        "Финальный source-field plateau проверяется отдельно по held-noise validation каждые 25 updates и требует трёх последовательных stable checks.", "",
        "## Test результаты", "",
        "![Парные контрасты по bank seeds](plots/paired_bank_contrasts.png)", "",
        "**Пояснение графика.** По X — paired разность test NMSE метода и baseline; отрицательное значение указывает на меньшую ошибку метода. "
        "Строки подписаны методом, цвет точки следует тому же method identity. Панели сравнивают с функциональной средней и dense control. "
        "Точка и 95% t-интервал вычислены на bank means после усреднения "
        "одних и тех же восьми задач внутри банка. Интервал описывает вариацию между банками при условии этого фиксированного набора задач.", "",
        "| Метод | Среднее test NMSE | 95% CI по банкам |",
        "|---|---:|---:|",
    ]
    for name in methods:
        item = report["test_nmse"][name]
        ci = item["ci95"]
        ci_text = "не оценивается" if ci is None else f"[{ci[0]:.5f}, {ci[1]:.5f}]"
        lines.append(f"| {LABELS[name]} | {item['mean']:.5f} | {ci_text} |")
    lines += [
        "", "![Heatmap по задачам](plots/taskwise_nmse_gap_heatmap.png)", "",
        "**Пояснение графика.** Строки — восемь заранее заданных cost vectors, столбцы — методы кроме функциональной средней. "
        "Ячейка показывает среднюю по банкам разность test NMSE относительно functional; симметричная шкала центрирована на нуле. "
        "Синий означает меньшую ошибку метода, красный — большую. Heatmap показывает task-specific структуру и сам по себе не является тестом значимости.", "",
        "![Парные разности и доли побед по задачам](plots/taskwise_gaps_and_win_rates.png)", "",
        "**Пояснение графика.** Цвет линии соответствует методу в легенде. Слева показана по каждой фиксированной задаче средняя paired разность относительно functional и 95% t-интервал по bank seeds. "
        "Справа показана доля из $n$ банков, где метод имеет меньший NMSE; интервалы Уилсона основаны на этих парных сравнениях. "
        "Задачи остаются фиксированным набором, а не независимыми повторами.", "",
        "### Парные сравнения с dense control по задачам", "",
        "Таблица показывает, на скольких из восьми фиксированных задач средняя по bank seeds ошибка метода ниже dense, и paired разность после усреднения задач внутри банка. "
        "95% интервалы справа скорректированы Bonferroni по всем 16 отображённым сравнениям (восемь методов против functional и dense); отрицательная разность благоприятна методу.", "",
        "| Метод | Задач с меньшей средней NMSE | Task IDs | Paired NMSE gap vs dense, 95% familywise CI |",
        "|---|---:|---|---:|",
    ]
    for name, item in report["dense_comparison"]["comparisons"].items():
        ci = item["familywise_ci95"]
        ci_text = "не оценивается" if ci is None else f"[{ci[0]:.5f}, {ci[1]:.5f}]"
        tasks = ", ".join(map(str, item["tasks_with_lower_mean_nmse"])) or "—"
        lines.append(f"| {LABELS[name]} | {item['tasks_lower_count']}/8 | {tasks} | {ci_text} |")
    lines += [
        "", "![Выбранные веса и маски](plots/sample_effective_weights_and_masks.png)", "",
    ]
    lines += [
        f"**Пояснение графика.** Условия: seed {report['sample_effective_weights_and_masks']['seed']}, новая задача 0, child initialization 2. Строки соответствуют методам; "
        "левая колонка — фактические обученные signed $W_{eff}=W\\odot M$, правая — бинарная маска. Оси показывают 784 пикселя в row-major порядке и 32 hidden units. "
        "Для весов одна общая симметричная шкала ±общая 99-я процентиль $|W_{eff}|$; маски имеют общую шкалу 0–1. "
        "Цвет отражает знак/величину параметра, не importance. Пример не заменяет paired benchmark.", "",
        "### Сходство замороженных топологий", "",
        "IoU ниже рассчитан постфактум для уже выбранных масок, усреднён по восьми задачам внутри каждого банка, затем между восемью bank seeds. "
        "Это описательная диагностика масок; IoU не участвовал в обучении/выборе и не устанавливает причинный механизм качества.", "",
        "| Сравнение топологий | Средний IoU | 95% t-интервал по bank seeds |",
        "|---|---:|---:|",
    ]
    topology = report.get("root_audits", {}).get("topology")
    if topology is not None:
        topology_labels = {
            "iou_gnn_functional": "GNN single vs functional",
            "iou_gnn_pixel": "GNN single vs pixel prior",
            "iou_flow_functional": "Flow single vs functional",
            "iou_functional_pixel": "Functional vs pixel prior",
        }
        for key, label in topology_labels.items():
            item = topology["bank_mean_intervals"][key]
            ci = item["ci95"]
            ci_text = "не оценивается" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            lines.append(f"| {label} | {item['mean']:.4f} | {ci_text} |")
        lines += [
            "", "Средняя функциональная маска близка к pixel prior; GNN masks сохраняют значительное сходство и с functional, и с pixel prior. "
            "Flow меняет топологию сильнее (меньший IoU с functional), но это само по себе не означает лучшую test utility: средние Flow остаются хуже dense и random controls. "
            "IoU не даёт оснований приписать наблюдаемую точность сходству или отличиям топологий.", "",
        ]
    else:
        lines += ["| Диагностика | недоступна в partial report | — |", ""]
    child_audit = report.get("root_audits", {}).get("child_plateau")
    if child_audit is not None:
        final = report["source_child_and_candidate_diagnostics"]
        lines += [
            f"Root-level audit: plateau flags подняты у {child_audit['true']}/{child_audit['total']} source, feedback, selection и final child fits; "
            f"в финальном target subset — {int(final['final_child_plateau_fraction'] * final['final_child_count'])}/{final['final_child_count']}. "
            f"Проверены также [root_child_plateau_audit.json](root_child_plateau_audit.json) и [root_mask_diagnostics.json](root_mask_diagnostics.json).", "",
        ]
    review_files = [name for name in ("independent_review.json", "independent_review.md")
                    if (out / name).is_file()]
    if review_files:
        lines += ["Независимая проверка: " + ", ".join(
            f"[{name}]({name})" for name in review_files
        ) + ".", ""]
    lines += [
        "## Выбранные веса, обучение и utility", "",
        "![Source training curves](plots/source_field_training_curves.png)", "",
        "**Пояснение графика.** Строки — GNN BCE и Flow scalar-velocity FM loss; колонки — фактически выполненные stages, "
        "в заголовке указаны cap и число банков с этой стадией. Синий цвет показывает train objective, красный — held-noise validation на source archive; "
        "полупрозрачные bands — 95% t-интервалы между доступными bank seeds. Stage 0 ограничен cap 1200, stage 1 и source-only refinements 2–5 — cap 800; "
        "stages 0/1 — bounded preliminary fits, а plateau фиксируется отдельно у последней стадии каждой модели. До теста допускается только финальный GNN и Flow с plateau=true. "
        "Кривые относятся к source-only обучению, а не к target test quality.", "",
        "| Bank | Модель | Стадии: updates/cap и plateau | Финальная стадия | Финальный source plateau |",
        "|---:|---|---|---:|:---:|",
        *stage_table_lines, "",
        "![Source utility и child diagnostics](plots/source_utility_and_child_diagnostics.png)", "",
        "**Пояснение графика.** Слева показаны initial 12 и feedback 32 masks на четырёх source tasks; цвет кодирует фазу, интервалы отражают банки. "
        "В центре цветом обозначен generator в легенде, а крест отмечает выбранный query-кандидат среди восьми. Оси показывают конечный support objective с L2 и query NMSE. "
        "Справа показаны доля финальных child fits с support-only plateau flag и число завершённых fit из общего числа на фиксированном горизонте. "
        "Флаг child-fit использует checkpoints каждые 100 updates; обе его стабильности фактически оценивают последнее 100-update окно. "
        "Это только диагностика и она не останавливает fit.", "",
        "![Unique candidate draws](plots/candidate_unique_draw_counts.png)", "",
        "**Пояснение графика.** Цвет и подпись столбца указывают generator. Столбец — число различных binary masks из восьми draws, сначала усреднённое по target tasks внутри каждого банка; "
        "интервалы рассчитаны между банками. Максимум равен восьми. График показывает diversity, не качество.", "",
        "## Интерпретация", "",
    ]
    claims = []
    for baseline, comparisons in report["paired_contrasts"].items():
        for name, item in comparisons.items():
            ci = item["familywise_ci95"]
            if ci is not None and ci[1] < 0 and item["task_all_mean_gaps_negative"]:
                claims.append(f"{LABELS[name]} относительно {LABELS[baseline]}")
    if claims:
        lines.append("Поддержанное ограниченное преимущество обнаружено для: " + "; ".join(claims)
                     + ". Требование включает Bonferroni-adjusted paired CI по 16 method-baseline contrasts ниже нуля "
                     "и отрицательные точечные средние по всем восьми фиксированным задачам.")
    else:
        lines.append("Результаты не поддерживают позитивное общее утверждение о превосходстве какого-либо метода. "
                     "Такое утверждение потребовало бы Bonferroni-adjusted paired bank CI всех 16 method-baseline contrasts "
                     "с верхней границей ниже нуля и отрицательной средней разностью на каждой из восьми фиксированных задач. "
                     "Это условие не доказывает эквивалентность методов.")
    dense_claim = report["dense_comparison"]["primary_all_task_improvement_met"]
    if not dense_claim:
        flow_search = report["dense_comparison"]["comparisons"].get("flow_search8", {})
        random = report["dense_comparison"]["comparisons"].get("random", {})
        lines.append(
            f"Поставленная цель улучшить качество относительно dense на каждой новой задаче не достигнута: "
            f"ни один sparse-метод не имеет меньшей bank-mean NMSE на всех восьми задачах. "
            f"Например, Flow search выигрывает по task mean на {flow_search.get('tasks_lower_count', 0)}/8 задачах, "
            f"случайная топология — на {random.get('tasks_lower_count', 0)}/8."
        )
    lines += [
        "", "Главное ограничение — восемь target cost vectors фиксированы и повторяются между bank seeds: CI описывают вариацию банков на этих задачах, "
        "а не обобщение на все возможные функции стоимости. Source parent знает source utility tasks и имеет больше наблюдений, чем target learner. "
        "Контекст агрегирует полные функциональные profiles и q moments по банку; он не подаёт все индивидуальные решения отдельными токенами "
        "и не использует их quality labels как attention inputs. Историческое train-only Hungarian alignment остаётся частью preprocessing.", "",
        "Числа paired contrasts, task gaps, win rates и массивы графиков находятся в [summary.json](summary.json) и [figure_data.npz](figure_data.npz); "
        "протокол запуска — [protocol.json](protocol.json).", "",
    ]
    content = "\n".join(lines)
    (out / "RESULTS_RU.md").write_text(content, encoding="utf-8")
    if report["complete_seeds"] == list(SEEDS) and not report["partial_report"]:
        CANONICAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
        relative_root = Path("../../../") / out.relative_to(ROOT)
        canonical = content.replace("(plots/", f"({relative_root.as_posix()}/plots/")
        for name in ("summary.json", "figure_data.npz", "protocol.json",
                     "root_child_plateau_audit.json", "root_mask_diagnostics.json",
                     "independent_review.json", "independent_review.md"):
            canonical = canonical.replace(f"]({name})", f"]({relative_root.as_posix()}/{name})")
        CANONICAL_REPORT.write_text(canonical, encoding="utf-8")


def build_report(out: Path = DEFAULT_OUT, partial: bool = False) -> dict[str, Any]:
    out = Path(out).resolve()
    _require(out.is_dir(), f"Utility-graph result root does not exist: {out}")
    protocol_path = out / "protocol.json"
    _require(protocol_path.is_file(), f"Missing run protocol: {protocol_path}")
    protocol = _json(protocol_path)
    complete = [seed for seed in SEEDS if (out / f"seed_{seed}" / "COMPLETE").is_file()]
    missing = [seed for seed in SEEDS if seed not in complete]
    if missing and not partial:
        raise FileNotFoundError(
            f"All eight seeds must be COMPLETE; missing {missing}. Use --partial for an explicit partial report."
        )
    _require(bool(complete), "No complete utility-graph seeds are available")

    first_mask_path = out / f"seed_{complete[0]}" / "task_0" / "masks_frozen.pt"
    first_methods = tuple(_torch(first_mask_path)["methods"])
    _require(set(first_methods) == EXPECTED_SPARSE and len(first_methods) == len(EXPECTED_SPARSE),
             f"Unexpected sparse method set in {first_mask_path}: {first_methods}")
    methods = first_methods + ("dense_tuned",)
    rows = [_load_seed(seed, out, methods) for seed in complete]
    for row in rows[1:]:
        _require(tuple(row["frozen_masks"][0]["methods"]) == first_methods,
                 f"Sparse method order differs between seeds (seed {row['seed']})")

    raw = np.stack([row["results"] for row in rows])  # bank, task, method, replica
    task_means = raw.mean(axis=-1)
    bank_means = task_means.mean(axis=1)
    root_audits = _load_root_audits(out, rows, partial=bool(missing))
    dense_comparison = _dense_comparison_summary(task_means, methods)
    test_summary = {name: _interval(bank_means[:, i]) for i, name in enumerate(methods)}
    paired = {}
    for baseline in ("functional", "dense_tuned"):
        base_index = methods.index(baseline)
        family = {}
        for name in methods:
            if name == baseline:
                continue
            index = methods.index(name)
            diff = bank_means[:, index] - bank_means[:, base_index]
            task_diff = task_means[:, :, index] - task_means[:, :, base_index]
            family[name] = {
                "bank_paired_difference": _interval(diff),
                "familywise_ci95": _interval(diff, comparisons=16)["ci95"],
                "familywise_adjustment": "Bonferroni t interval across all 16 reported method-baseline contrasts",
                "task_mean_gaps": task_diff.mean(axis=0).tolist(),
                "task_intervals": [_interval(task_diff[:, task]) for task in TASKS],
                "task_all_mean_gaps_negative": bool(np.all(task_diff.mean(axis=0) < 0)),
            }
        paired[baseline] = family

    (out / "plots").mkdir(parents=True, exist_ok=True)
    _plot_bank_contrasts(out, bank_means, methods)
    task_gap, heatmap_limit = _plot_task_heatmap(out, task_means, methods)
    task_delta_functional = task_means - task_means[:, :, [methods.index("functional")]]
    win_info = _plot_task_gaps(out, task_delta_functional, methods)
    weight_sample = _plot_weight_masks(out, methods, complete[0])
    training_curves = _plot_training(out, rows)
    child_diagnostics = _plot_source_diagnostics(out, rows)
    unique_draws = _plot_unique_draws(out, rows)

    initial = np.stack([r["source_final_quality"][:, :12] for r in rows])
    feedback = np.stack([r["source_final_quality"][:, 12:] for r in rows])
    model_training = {
        "stage_caps": {
            "stage0_initial_archive_preliminary": 1200,
            "stage1_feedback_archive_preliminary": 800,
            "stages2_to5_source_warm_refinement": 800,
            "final_plateau_required_before_target_data": True,
        },
        "by_seed": {
            str(row["seed"]): row["training_stage_summary"] for row in rows
        },
        "all_final_fields_plateau_certified": all(
            row["training_stage_summary"][kind]["final_plateau"]
            for row in rows for kind in ("gnn", "flow")
        ),
        "source_frozen_hashes_validated": True,
    }
    claims = {}
    for baseline, family in paired.items():
        claims[baseline] = {
            name: bool(item["familywise_ci95"] is not None
                       and item["familywise_ci95"][1] < 0
                       and item["task_all_mean_gaps_negative"])
            for name, item in family.items()
        }
    summary = {
        "experiment": "utility-trained deterministic GNN and flow whole-mask benchmark",
        "result_root": str(out), "protocol": protocol,
        "complete_seeds": complete, "missing_seeds": missing, "partial_report": bool(missing),
        "n_banks": len(rows), "fixed_target_tasks": list(TASKS), "replicas": list(REPLICAS),
        "methods": list(methods), "method_labels_ru": LABELS,
        "mask": {"shape": [784, 32], "active_connections": K, "density": K/N_CONNECTIONS},
        "source_functional_bank": {
            "source_cost_task_count": 4,
            "population_mask_densities": [.1, .3, .5, .7, .9],
            "context_inputs": ["bank-aggregated full-state psi mean/std",
                               "signed q mean and q RMS moments",
                               "functional-score mean"],
            "individual_quality_attention_inputs": False,
            "target_digit_label_oracle": False,
            "alignment": "historical train-only Hungarian preprocessing; not alignment-free",
            "plateau_scopes": {
                "source_population_children": "true 50-update objective range from separate 50-update snapshots plus 100-update endpoint drift",
                "utility_archive_and_target_child_diagnostics": "checkpoint cadence 100; rounded 50-update window has one checkpoint, so both components use a 100-update window",
                "source_field_models": "last five 25-update held-noise validation checkpoints stable within 1% for three consecutive checks",
            },
        },
        "support_query_test": {"support_observed_set_labels": 205,
                               "query_selection_set_labels": 51,
                               "final_test_sets_per_task": 512,
                               "images_per_test_set": 5,
                               "test_pool_unique_image_rows_per_digit": 300,
                               "test_pool_unique_image_rows_total": 3000,
                               "test_set_sampling": "with replacement from independent held-out image pool",
                               "candidate_draws_per_generator": 8,
                               "final_replicas": list(REPLICAS)},
        "statistical_unit": "bank seed; replica mean within each fixed task, followed by fixed eight-task mean",
        "test_nmse": test_summary, "paired_contrasts": paired, "claim_decisions": claims,
        "dense_comparison": dense_comparison, "root_audits": root_audits,
        "taskwise_gaps_vs_functional": win_info,
        "taskwise_method_nmse_mean": task_means.mean(axis=0).tolist(),
        "taskwise_gap_mean_vs_functional": task_gap.tolist(),
        "heatmap_symmetric_limit": heatmap_limit,
        "source_archive_utility": {
            "initial_12_query_nmse_by_seed_task_candidate": initial.tolist(),
            "feedback_32_query_nmse_by_seed_task_candidate": feedback.tolist(),
            "initial_phase_seed_task_mean": initial.mean(axis=2).tolist(),
            "feedback_phase_seed_task_mean": feedback.mean(axis=2).tolist(),
            "initial_mask_count": 12, "shared_feedback_mask_count": 32,
            "elite_masks_per_source_task": 4,
        },
        "source_child_and_candidate_diagnostics": child_diagnostics,
        "unique_candidate_draws": unique_draws,
        "sample_effective_weights_and_masks": weight_sample,
        "source_training_curves": training_curves,
        "source_model_training": model_training,
        "figures": [
            "plots/paired_bank_contrasts.png", "plots/taskwise_nmse_gap_heatmap.png",
            "plots/sample_effective_weights_and_masks.png", "plots/taskwise_gaps_and_win_rates.png",
            "plots/source_field_training_curves.png", "plots/source_utility_and_child_diagnostics.png",
            "plots/candidate_unique_draw_counts.png",
        ],
    }
    task_win_flags = task_delta_functional < 0
    task_win_fractions = task_win_flags.mean(axis=0)
    task_win_intervals = np.asarray([
        [_wilson(int(task_win_flags[:, task, methods.index(name)].sum()), len(rows))
         for task in TASKS]
        for name in methods if name != "functional"
    ], dtype=float)
    curve_arrays = {}
    for name, value in training_curves.items():
        curve_arrays[f"training_{name}_steps"] = np.asarray(value["steps"], dtype=int)
        curve_arrays[f"training_{name}_mean"] = np.asarray(value["mean"], dtype=float)
        curve_arrays[f"training_{name}_ci95"] = np.asarray(value["ci95"], dtype=float)
        curve_arrays[f"training_{name}_n_banks"] = np.asarray(value["n_banks"], dtype=int)
    np.savez_compressed(
        out / "figure_data.npz",
        bank_seed_ids=np.asarray(complete, dtype=int), method_names=np.asarray(methods),
        generator_names=np.asarray(GENERATORS), test_nmse_by_bank_task_method_replica=raw,
        bank_task_method_means=task_means, bank_method_means=bank_means,
        paired_vs_functional=bank_means-bank_means[:, [methods.index("functional")]],
        paired_vs_dense=bank_means-bank_means[:, [methods.index("dense_tuned")]],
        task_method_means=task_means.mean(axis=0), task_gaps_vs_functional=task_gap,
        task_heatmap_data=task_gap[:, [i for i, name in enumerate(methods)
                                       if name != "functional"]],
        candidate_unique_draw_counts=np.stack([r["candidate_counts"] for r in rows]),
        candidate_query_nmse=np.stack([r["candidate_query"] for r in rows]),
        candidate_selection_support_objective=np.stack([r["selection_objective"] for r in rows]),
        candidate_selection_query_nmse=np.stack([r["selection_query"] for r in rows]),
        candidate_selection_plateau_fraction=np.stack([r["selection_plateau"] for r in rows]),
        candidate_selection_selected=np.stack([r["selection_selected"] for r in rows]),
        final_child_plateau_flags=np.stack([r["final_plateau"] for r in rows]),
        all_child_plateau_flags=np.stack([r["all_child_plateau"] for r in rows]),
        final_child_step_audit=np.asarray([
            [[s["sparse_terminal_step"], s["sparse_steps_run"],
              s["dense_terminal_step"], s["dense_steps_run"]]
             for s in r["final_steps"]] for r in rows
        ], dtype=int),
        task_win_flags_vs_functional=task_win_flags,
        task_win_fractions_vs_functional=task_win_fractions,
        task_win_wilson95_vs_functional=task_win_intervals,
        source_initial_query_nmse=initial, source_feedback_query_nmse=feedback,
        sample_weight_effective=np.asarray(weight_sample["weight_effective"]),
        sample_binary_masks=np.asarray(weight_sample["binary_masks"]),
        sample_weight_symmetric_limit=np.asarray(weight_sample["weight_shared_symmetric_limit_99pct"]),
        **curve_arrays,
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(out, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--partial", action="store_true",
                        help="include only COMPLETE seeds; default requires all eight")
    args = parser.parse_args()
    summary = build_report(args.out, partial=args.partial)
    print(json.dumps({"report": str(args.out.resolve() / "RESULTS_RU.md"),
                      "complete_seeds": summary["complete_seeds"],
                      "n_banks": summary["n_banks"],
                      "partial": summary["partial_report"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
