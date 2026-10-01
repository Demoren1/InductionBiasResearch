"""Build an auditable source-only report for the rebuilt DeepSets bank.

The default mode requires all 8 seed folders, their 4 sparse+dense task shards,
per-seed completion markers, and the root ``COMPLETE`` marker. ``--partial`` is
for an explicitly labelled interim report only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/deepsets_vaae/20261001_rebuilt_functional_bank"
DEFAULT_SEEDS = tuple(range(4100, 4108))
RHOS = (0.1, 0.3, 0.5, 0.7, 0.9)
RECIPES = ("random", "dense_functional")
TASKS = range(4)


def _json_read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_seeds(source: Path, protocol: dict[str, Any] | None) -> list[int]:
    if protocol:
        raw = protocol.get("seeds") or protocol.get("seed_values")
        if isinstance(raw, list) and raw:
            values = [int(value) for value in raw]
            if len(set(values)) != len(values):
                raise ValueError("protocol seed list contains duplicates")
            return values
    found = sorted(
        int(match.group(1)) for child in source.glob("seed_*")
        if (match := re.fullmatch(r"seed_(\d+)", child.name))
    )
    return found or list(DEFAULT_SEEDS)


def _load_pt(path: Path) -> dict[str, Any]:
    item = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(item, dict):
        raise ValueError(f"{path} does not contain a dictionary")
    return item


def _flat_metric(raw: dict[str, Any], key: str, expected: int | None = None) -> np.ndarray:
    if key not in raw:
        raise ValueError(f"missing {key!r} in shard")
    value = torch.as_tensor(raw[key]).detach().float().cpu().numpy().reshape(-1)
    if expected is not None and len(value) != expected:
        raise ValueError(f"{key} has {len(value)} values, expected {expected}")
    if not np.isfinite(value).all():
        raise ValueError(f"{key} contains non-finite values")
    return value.astype(np.float64)


def _mask_fingerprint(mask: Tensor) -> str:
    array = mask.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy()
    packed = np.packbits(array.reshape(-1), bitorder="little")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _effective_state_fingerprint(state: dict[str, Tensor], index: int) -> str:
    effective_weight = state["weight"][index].detach().cpu() * state["masks"][index].detach().cpu()
    parts = (effective_weight, state["bias"][index], state["readout"][index],
             state["per_image_offset"][index:index + 1])
    digest = hashlib.sha256()
    for tensor in parts:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _history_record(raw: dict[str, Any], method: str, seed: int, task: int) -> dict[str, Any]:
    history = raw.get("history")
    if not isinstance(history, dict):
        raise ValueError(f"{method} seed {seed} task {task}: missing fit history")
    steps = torch.as_tensor(history.get("steps", []), dtype=torch.long).cpu().numpy().reshape(-1)
    train = torch.as_tensor(history.get("trainNMSE", history.get("population_loss", []))).float().cpu().numpy()
    query = torch.as_tensor(history.get("queryNMSE", [])).float().cpu().numpy()
    if train.ndim != 2 or query.shape != train.shape or train.shape[0] != len(steps):
        raise ValueError(f"{method} seed {seed} task {task}: malformed train/query history")
    if not np.isfinite(train).all() or not np.isfinite(query).all():
        raise ValueError(f"{method} seed {seed} task {task}: non-finite history")
    if len(steps) < 1 or np.any(np.diff(steps) <= 0):
        raise ValueError(f"{method} seed {seed} task {task}: invalid checkpoint steps")
    flags = _flat_metric(raw, "plateau_flags", train.shape[1]).astype(bool)
    stopped = _flat_metric(raw, "plateau_stopped", train.shape[1]).astype(bool) if "plateau_stopped" in raw else flags
    capped = _flat_metric(raw, "capped", train.shape[1]).astype(bool) if "capped" in raw else ~stopped
    stopping = _flat_metric(raw, "stopping_steps", train.shape[1]).astype(np.int64)
    return {
        "seed": seed,
        "task": task,
        "method": method,
        "steps": steps.astype(np.int64),
        "trainNMSE": train.mean(axis=1).astype(np.float64),
        "queryNMSE": query.mean(axis=1).astype(np.float64),
        "candidate_count": int(train.shape[1]),
        "plateau_candidates": int(flags.sum()),
        "plateau_stopped_candidates": int(stopped.sum()),
        "capped_candidates": int(capped.sum()),
        "terminal_step": int(raw.get("terminal_step", steps[-1])),
        "candidate_stopping_steps": stopping,
        "all_plateau": bool(flags.all()),
    }


def _cell_values(rows: list[dict[str, Any]]) -> dict[tuple[int, str, int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["task"], row["rho_tenths"], row["recipe"])].append(row)
    return grouped


def _seed_aggregates(rows: list[dict[str, Any]]) -> dict[tuple[int, str, int], dict[str, Any]]:
    bank_cells = _cell_values(rows)
    by_seed: dict[tuple[int, str, int], list[dict[str, float]]] = defaultdict(list)
    for (seed, _task, rho, recipe), candidates in bank_cells.items():
        by_seed[(seed, recipe, rho)].append({
            "sparse": float(np.mean([r["sparse_nmse"] for r in candidates])),
            "dense": float(np.mean([r["paired_dense_nmse"] for r in candidates])),
            "gap": float(np.mean([r["paired_gap"] for r in candidates])),
            "win_rate": float(np.mean([r["paired_gap"] < 0 for r in candidates])),
            "abs_gap": float(np.mean([abs(r["paired_gap"]) for r in candidates])),
        })
    result: dict[tuple[int, str, int], dict[str, Any]] = {}
    for key, task_values in by_seed.items():
        # Four source tasks are repeated within a seed, not four extra seeds.
        result[key] = {
            metric: float(np.mean([value[metric] for value in task_values]))
            for metric in ("sparse", "dense", "gap", "win_rate", "abs_gap")
        }
    return result


def _mean_ci(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    count = len(array)
    if count == 0:
        return {"n_seeds": 0, "mean": None, "sd": None,
                "ci95_low": None, "ci95_high": None, "seed_values": []}
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if count > 1 else None
    radius = float(student_t.ppf(.975, count - 1) * sd / math.sqrt(count)) if count > 1 else None
    return {"n_seeds": count, "mean": mean, "sd": sd,
            "ci95_low": mean - radius if radius is not None else None,
            "ci95_high": mean + radius if radius is not None else None,
            "seed_values": array.tolist()}


def _summarize(rows: list[dict[str, Any]], dense_rows: list[dict[str, Any]],
               fits: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    seed_cells = _seed_aggregates(rows)
    table: list[dict[str, Any]] = []
    for rho in RHOS:
        for recipe in RECIPES:
            selected = [r for r in rows if r["rho_tenths"] == round(10 * rho) and r["recipe"] == recipe]
            per_seed = [seed_cells[(seed, recipe, round(10 * rho))]
                        for seed in seeds if (seed, recipe, round(10 * rho)) in seed_cells]
            gap_ci = _mean_ci([r["gap"] for r in per_seed])
            sparse_ci = _mean_ci([r["sparse"] for r in per_seed])
            dense_ci = _mean_ci([r["dense"] for r in per_seed])
            bank_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            for item in selected:
                bank_groups[(item["seed"], item["task"])].append(item)
            table.append({
                "rho": rho,
                "recipe": recipe,
                "candidate_count": len(selected),
                "bank_count": len(bank_groups),
                "sparse_audit_nmse_by_seed": sparse_ci,
                "paired_dense_audit_nmse_by_seed": dense_ci,
                "paired_gap_by_seed": gap_ci,
                "paired_sparse_wins": int(sum(r["paired_gap"] < 0 for r in selected)),
                "paired_ties": int(sum(r["paired_gap"] == 0 for r in selected)),
                "paired_n": len(selected),
                "mean_absolute_paired_gap_by_seed": _mean_ci([r["abs_gap"] for r in per_seed]),
                "mean_candidate_nmse": float(np.mean([r["sparse_nmse"] for r in selected])) if selected else None,
                "mean_candidate_paired_dense_nmse": float(np.mean([r["paired_dense_nmse"] for r in selected])) if selected else None,
                "mean_candidate_paired_gap": float(np.mean([r["paired_gap"] for r in selected])) if selected else None,
                "mean_absolute_candidate_gap": float(np.mean([abs(r["paired_gap"]) for r in selected])) if selected else None,
            })

    dense_seed_task: dict[tuple[int, int], float] = {}
    for item in dense_rows:
        dense_seed_task.setdefault((item["seed"], item["task"]), []).append(item["audit_nmse"])
    dense_by_seed: dict[int, float] = {}
    for seed in seeds:
        task_values = [np.mean(values) for (s, _task), values in dense_seed_task.items() if s == seed]
        if task_values:
            dense_by_seed[seed] = float(np.mean(task_values))

    convergence = {}
    for method in ("dense", "sparse"):
        method_fits = [fit for fit in fits if fit["method"] == method]
        stopping = np.concatenate([fit["candidate_stopping_steps"] for fit in method_fits]) if method_fits else np.asarray([], dtype=np.int64)
        convergence[method] = {
            "fit_count": len(method_fits),
            "fit_all_plateau_count": sum(fit["all_plateau"] for fit in method_fits),
            "candidate_count": int(sum(fit["candidate_count"] for fit in method_fits)),
            "plateau_candidate_count": int(sum(fit["plateau_candidates"] for fit in method_fits)),
            "plateau_stopped_candidate_count": int(sum(fit["plateau_stopped_candidates"] for fit in method_fits)),
            "capped_candidate_count": int(sum(fit["capped_candidates"] for fit in method_fits)),
            "stopping_steps_min": int(stopping.min()) if stopping.size else None,
            "stopping_steps_max": int(stopping.max()) if stopping.size else None,
            "fit_terminal_steps_min": min((fit["terminal_step"] for fit in method_fits), default=None),
            "fit_terminal_steps_max": max((fit["terminal_step"] for fit in method_fits), default=None),
        }

    mc_sparse = np.asarray([row["mc_sparse_nmse"] for row in rows], dtype=np.float64)
    exact_sparse = np.asarray([row["sparse_nmse"] for row in rows], dtype=np.float64)
    mc_paired_dense = np.asarray([row["mc_paired_dense_nmse"] for row in rows], dtype=np.float64)
    exact_paired_dense = np.asarray([row["paired_dense_nmse"] for row in rows], dtype=np.float64)
    mc_sparse_gap = mc_sparse - mc_paired_dense
    exact_sparse_gap = exact_sparse - exact_paired_dense
    mc_dense = np.asarray([row["mc_audit_nmse"] for row in dense_rows], dtype=np.float64)
    exact_dense = np.asarray([row["audit_nmse"] for row in dense_rows], dtype=np.float64)

    return {
        "source_task_count": 4,
        "seed_count": len(seeds),
        "seeds": seeds,
        "candidate_count_sparse": len(rows),
        "candidate_count_dense": len(dense_rows),
        "exact_audit": {
            "sparse_nmse_mean": float(exact_sparse.mean()),
            "paired_dense_nmse_mean": float(exact_paired_dense.mean()),
            "paired_gap_mean": float(exact_sparse_gap.mean()),
            "sparse_paired_wins": int((exact_sparse_gap < 0).sum()),
            "paired_ties": int((exact_sparse_gap == 0).sum()),
            "n_sparse_fits": int(len(exact_sparse)),
            "n_paired_comparisons": int(len(exact_sparse_gap)),
        },
        "monte_carlo_secondary": {
            "sparse_nmse_mean": float(mc_sparse.mean()),
            "paired_dense_nmse_mean": float(mc_paired_dense.mean()),
            "paired_gap_mean": float(mc_sparse_gap.mean()),
            "sparse_paired_wins": int((mc_sparse_gap < 0).sum()),
            "n_sparse_fits": int(len(mc_sparse)),
            "sparse_abs_error_vs_exact_mean": float(np.abs(mc_sparse - exact_sparse).mean()),
            "sparse_abs_error_vs_exact_max": float(np.abs(mc_sparse - exact_sparse).max()),
            "dense_abs_error_vs_exact_mean": float(np.abs(mc_dense - exact_dense).mean()),
            "dense_abs_error_vs_exact_max": float(np.abs(mc_dense - exact_dense).max()),
            "stored_vs_replay_max_abs_error": None,
            "set_samples_per_fit": 2048,
            "set_size": 5,
        },
        "density_recipe_table": table,
        "dense_audit_nmse_by_seed": _mean_ci(list(dense_by_seed.values())),
        "dense_audit_nmse_by_seed_values": {str(seed): value for seed, value in dense_by_seed.items()},
        "convergence": convergence,
        "statistical_unit": "eight seed aggregates; first average candidates within seed/task/cell, then four fixed task banks within seed",
        "interpretation_limit": "descriptive intervals over seed aggregates; source pools may overlap and four task banks are repeated fixed tasks, not independent seeds",
    }


def _plot_quality(summary: dict[str, Any], output: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    colors = {"random": "#3973ac", "dense_functional": "#d17c18"}
    for ax, recipe in zip(axes, RECIPES):
        cells = [row for row in summary["density_recipe_table"] if row["recipe"] == recipe]
        x = np.asarray([row["rho"] for row in cells])
        values = [row["sparse_audit_nmse_by_seed"] for row in cells]
        means = np.asarray([np.nan if v["mean"] is None else v["mean"] for v in values])
        low = np.asarray([means[i] - values[i]["ci95_low"] if np.isfinite(means[i]) and values[i]["ci95_low"] is not None else 0.0 for i in range(len(values))])
        high = np.asarray([values[i]["ci95_high"] - means[i] if np.isfinite(means[i]) and values[i]["ci95_high"] is not None else 0.0 for i in range(len(values))])
        ax.errorbar(x, means, yerr=np.vstack((low, high)), marker="o", capsize=3,
                    color=colors[recipe], label="sparse masks; exact audit")
        dense = summary["dense_audit_nmse_by_seed"]["mean"]
        if dense is not None:
            ax.axhline(dense, color="#333333", linestyle="--", label="dense controls; exact audit")
        ax.set_title("Случайные маски" if recipe == "random" else "Dense-functional маски")
        ax.set_xlabel("Плотность ρ (доля активных рёбер)")
        ax.set_xticks(RHOS)
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Точный empirical-population NMSE по 3000 audit-изображениям")
    fig.suptitle("Точный audit всех сохранённых масок по плотности")
    target = output / "audit_nmse_by_density.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def _plot_mc_against_exact(rows: list[dict[str, Any]], output: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(.08, .92, len(RHOS)))
    exact_values = [r["sparse_nmse"] for r in rows] + [r["paired_dense_nmse"] for r in rows]
    mc_values = [r["mc_sparse_nmse"] for r in rows] + [r["mc_paired_dense_nmse"] for r in rows]
    lower = min(min(exact_values), min(mc_values))
    upper = max(max(exact_values), max(mc_values))
    pad = .025 * max(upper - lower, 1e-6)
    limits = (lower - pad, upper + pad)
    for rho, color in zip(RHOS, colors):
        selected = [row for row in rows if row["rho"] == rho]
        axes[0].scatter([r["sparse_nmse"] for r in selected],
                        [r["mc_sparse_nmse"] for r in selected],
                        s=9, alpha=.27, color=color, label=f"ρ={rho:.1f}")
        axes[1].scatter([r["paired_dense_nmse"] for r in selected],
                        [r["mc_paired_dense_nmse"] for r in selected],
                        s=9, alpha=.27, color=color, label=f"ρ={rho:.1f}")
    axes[0].set_title("Sparse candidates")
    axes[1].set_title("Paired dense controls")
    for ax in axes:
        ax.plot(limits, limits, color="#222222", linestyle="--", linewidth=1, label="y = x")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_xlabel("Точный NMSE empirical audit-популяции")
        ax.set_ylabel("Monte Carlo NMSE; 2048 наборов × 5 изображений")
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Monte Carlo — вторичная оценка; exact audit — основной источник")
    target = output / "monte_carlo_vs_exact_audit.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def _plot_paired_gaps(seed_cells: dict[tuple[int, str, int], dict[str, Any]], output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 5.2), constrained_layout=True)
    positions: list[float] = []
    data: list[list[float]] = []
    labels: list[str] = []
    colors: list[str] = []
    for rho_idx, rho in enumerate(RHOS):
        for recipe_idx, recipe in enumerate(RECIPES):
            values = [seed_cells[(seed, recipe, round(10*rho))]["gap"]
                      for seed in sorted({key[0] for key in seed_cells})
                      if (seed, recipe, round(10*rho)) in seed_cells]
            if not values:
                continue
            positions.append(rho_idx + (-.17 if recipe_idx == 0 else .17))
            data.append(values)
            labels.append("Случайная" if recipe == "random" else "Dense-functional")
            colors.append("#3973ac" if recipe == "random" else "#d17c18")
    boxes = ax.boxplot(data, positions=positions, widths=.28, patch_artist=True,
                       showfliers=False, medianprops={"color": "black", "linewidth": 1.3})
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(.55)
    ax.axhline(0, color="#333333", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(RHOS)), [f"ρ={rho:.1f}" for rho in RHOS])
    ax.set_xlabel("Плотность ρ; рядом показаны две генерации масок")
    ax.set_ylabel("Парная разность NMSE (sparse − dense); ниже нуля лучше sparse")
    ax.set_title("Разброс средних парных разностей по 8 seed")
    ax.grid(axis="y", alpha=.25)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#3973ac", alpha=.55, label="Случайная"),
                       Patch(facecolor="#d17c18", alpha=.55, label="Dense-functional")],
              frameon=False)
    target = output / "paired_dense_gaps.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def _plot_histories(fits: list[dict[str, Any]], output: Path) -> tuple[Path | None, dict[str, Any]]:
    curves: dict[str, list[dict[str, Any]]] = {
        method: [fit for fit in fits if fit["method"] == method] for method in ("dense", "sparse")
    }
    common_steps: dict[str, list[int]] = {}
    for method, records in curves.items():
        if not records:
            common_steps[method] = []
            continue
        intersection = set(records[0]["steps"].tolist())
        for record in records[1:]:
            intersection.intersection_update(record["steps"].tolist())
        common_steps[method] = sorted(int(step) for step in intersection)
    if not any(common_steps.values()):
        return None, common_steps

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for method, label, color in (("dense", "Dense controls", "#333333"),
                                 ("sparse", "Sparse candidates", "#3973ac")):
        records = curves[method]
        steps = common_steps[method]
        if not records or not steps:
            continue
        seed_series: dict[int, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
        for record in records:
            lookup = {int(step): idx for idx, step in enumerate(record["steps"])}
            for step in steps:
                index = lookup[step]
                seed_series[record["seed"]][step].append((record["trainNMSE"][index], record["queryNMSE"][index]))
        train_mean, train_sd, query_mean, query_sd = [], [], [], []
        for metric_index in (0, 1):
            values_by_seed = []
            for seed in sorted(seed_series):
                values_by_seed.append([
                    float(np.mean([task_value[metric_index] for task_value in seed_series[seed][step]]))
                    for step in steps
                ])
            values = np.asarray(values_by_seed, dtype=np.float64)
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
            if metric_index == 0:
                train_mean, train_sd = mean, sd
            else:
                query_mean, query_sd = mean, sd
        axes[0].plot(steps, train_mean, label=label, color=color)
        axes[0].fill_between(steps, train_mean-train_sd, train_mean+train_sd, color=color, alpha=.14)
        axes[1].plot(steps, query_mean, label=label, color=color)
        axes[1].fill_between(steps, query_mean-query_sd, query_mean+query_sd, color=color, alpha=.14)
    axes[0].set_title("Полная source-популяция")
    axes[0].set_ylabel("Ожидаемый NMSE на iid-наборах из 5 изображений")
    axes[1].set_title("Диагностика на source-validation наборах")
    axes[1].set_ylabel("NMSE: среднее SSE набора / 5")
    for ax in axes:
        ax.set_xlabel("Шаг Adam")
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    fig.suptitle("Обучение dense и sparse банков; средние по моделям внутри shard")
    target = output / "source_population_training_curves.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target, common_steps


def _find_recipe_row(recipes: list[dict[str, Any]], rho: float,
                     recipe: str, variant: int, replica: int) -> int:
    for index, item in enumerate(recipes):
        if (round(float(item["rho"]), 6) == round(rho, 6)
                and item["recipe"] == recipe
                and int(item["variant"]) == variant
                and int(item["replica"]) == replica):
            return index
    raise ValueError(f"missing recipe rho={rho}, recipe={recipe}, variant={variant}, replica={replica}")


def _plot_masks_weights(source: Path, output: Path) -> tuple[Path | None, dict[str, Any] | None]:
    seed, task, replica = 4100, 0, 0
    dense_path = source / f"seed_{seed}" / f"dense_{task}.pt"
    sparse_path = source / f"seed_{seed}" / f"bank_{task}.pt"
    if not dense_path.is_file() or not sparse_path.is_file():
        return None, None
    dense = _load_pt(dense_path)
    sparse = _load_pt(sparse_path)
    recipes = sparse.get("candidate_recipe")
    if not isinstance(recipes, list):
        return None, None
    dense_state = dense.get("source_state_dict", dense.get("state_dict"))
    sparse_state = sparse.get("source_state_dict", sparse.get("state_dict"))
    if not isinstance(dense_state, dict) or not isinstance(sparse_state, dict):
        return None, None
    entries: list[tuple[str, Tensor, Tensor]] = []
    dense_mask = torch.as_tensor(dense_state["masks"][replica]).float()
    dense_eff = torch.as_tensor(dense_state["weight"][replica]).float() * dense_mask
    entries.append(("Dense control", dense_mask, dense_eff))
    for rho in RHOS:
        index = _find_recipe_row(recipes, rho, "dense_functional", 0, replica)
        mask = torch.as_tensor(sparse_state["masks"][index]).float()
        eff = torch.as_tensor(sparse_state["weight"][index]).float() * mask
        entries.append((f"ρ={rho:.1f}", mask, eff))
    weight_limit = max(float(eff.abs().max()) for _, _, eff in entries)
    weight_limit = max(weight_limit, 1e-12)
    fig, axes = plt.subplots(len(entries), 2, figsize=(13.4, 14.5), constrained_layout=True)
    for row, (label, mask, eff) in enumerate(entries):
        axes[row, 0].imshow(mask.T.detach().cpu().numpy(), aspect="auto", interpolation="nearest",
                            cmap="Greys", vmin=0, vmax=1)
        image = axes[row, 1].imshow(eff.T.detach().cpu().numpy(), aspect="auto", interpolation="nearest",
                                    cmap="coolwarm", vmin=-weight_limit, vmax=weight_limit)
        axes[row, 0].set_ylabel(f"{label}\nHidden neuron")
        axes[row, 0].set_xlabel("Feature position")
        axes[row, 1].set_ylabel("Hidden neuron")
        axes[row, 1].set_xlabel("Feature position")
    axes[0, 0].set_title("Binary support mask (0 white, 1 black)")
    axes[0, 1].set_title("Signed effective weight W × M (shared symmetric scale)")
    fig.colorbar(image, ax=axes[:, 1].tolist(), shrink=.7, label="Signed effective weight")
    fig.suptitle("Actual fitted structures: seed 4100, task 0, replica 0; dense-functional variant 0")
    target = output / "representative_masks_and_signed_weights.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    meta = {"seed": seed, "task": task, "replica": replica,
            "sparse_recipe": "dense_functional", "variant": 0,
            "variant_noise_scale": 0.0,
            "weight_shared_symmetric_limit": weight_limit,
            "rho_values": list(RHOS)}
    return target, meta


def _context_figures(source: Path, output: Path) -> list[dict[str, str]]:
    context_path = source / "seed_4100" / "functional_context.pt"
    if not context_path.is_file():
        return []
    context = _load_pt(context_path)
    paths = context.get("figure_paths", [])
    wanted = {
        "psi_two_teachers": ("ψ: профили двух aligned source-teachers",
                             "Два примера для seed 4100, task 0: x — индекс probe-изображения (128 probes), y — aligned hidden neuron; цвет — вклад ψ на общей симметричной шкале. Held-out teacher показан только как teacher-профиль и исключен из pooled context. Это не оценка реконструкции генератора."),
        "q_signed_mean": ("Подписанная карта вклада q",
                          "Ось x — input feature (784), y — hidden neuron (32); цвет — средний по probes signed q-вклад в единицах функции teacher, симметричная шкала центрирована в нуле. Профиль seed 4100, task 0."),
        "q_abs_mean": ("Средний абсолютный вклад q",
                       "Ось x — input feature (784), y — hidden neuron (32); цвет — среднее по probes абсолютное значение q-вклада в единицах функции teacher. Профиль seed 4100, task 0; знак вклада опущен."),
    }
    found = []
    for raw_path in paths:
        path = Path(raw_path)
        item = wanted.get(path.stem)
        if item is None:
            continue
        absolute = path if path.is_absolute() else source / path
        if not absolute.is_file():
            continue
        relative = os.path.relpath(absolute, output)
        found.append({"name": path.stem, "path": relative, "title": item[0], "caption": item[1]})
    return found


def _context_split_summary(source: Path, seeds: list[int]) -> dict[str, Any]:
    teacher_counts: list[int] = []
    train_counts: list[int] = []
    heldout_counts: list[int] = []
    train_only_flags: list[bool] = []
    cross_mask_topologies = 0
    train_rows_matching_heldout_mask = 0
    cross_effective_states = 0
    checked_cells = 0
    for seed in seeds:
        manifest = _json_read(source / f"seed_{seed}" / "functional_context_manifest.json")
        if manifest is None:
            raise FileNotFoundError(f"seed_{seed}/functional_context_manifest.json is missing")
        teacher_counts.extend(int(x) for x in manifest["teacher_counts"])
        train_counts.extend(int(x) for x in manifest["train_counts"])
        heldout_counts.extend(int(x) for x in manifest["heldout_counts"])
        train_only_flags.extend((bool(manifest.get("alignment_training_rows_only")),
                                 bool(manifest.get("pooled_context_and_score_baselines_training_rows_only")),
                                 not bool(manifest.get("heldout_teachers_used_as_context_inputs"))))
        context = _load_pt(source / f"seed_{seed}" / "functional_context.pt")
        for task in TASKS:
            bank = _load_pt(source / f"seed_{seed}" / f"bank_{task}.pt")
            state = bank["state_dict"]
            train_rows = torch.as_tensor(context["train_rows"][task]).long().tolist()
            heldout_rows = torch.as_tensor(context["heldout_rows"][task]).long().tolist()
            train_masks = {_mask_fingerprint(state["masks"][i]) for i in train_rows}
            heldout_masks = {_mask_fingerprint(state["masks"][i]) for i in heldout_rows}
            cross_mask_topologies += len(train_masks & heldout_masks)
            train_rows_matching_heldout_mask += sum(
                _mask_fingerprint(state["masks"][i]) in heldout_masks for i in train_rows
            )
            train_states = {_effective_state_fingerprint(state, i) for i in train_rows}
            heldout_states = {_effective_state_fingerprint(state, i) for i in heldout_rows}
            cross_effective_states += len(train_states & heldout_states)
            checked_cells += 1
    return {
        "teacher_count_per_task_seed_minmax": [min(teacher_counts), max(teacher_counts)],
        "train_count_per_task_seed_minmax": [min(train_counts), max(train_counts)],
        "heldout_count_per_task_seed_minmax": [min(heldout_counts), max(heldout_counts)],
        "train_only_context_manifest_checks_passed": bool(train_only_flags) and all(train_only_flags),
        "split_cells_checked": checked_cells,
        "train_heldout_shared_mask_topology_intersections_across_cells": cross_mask_topologies,
        "train_rows_whose_mask_topology_also_occurs_in_heldout_across_cells": train_rows_matching_heldout_mask,
        "train_heldout_exact_effective_state_intersections_across_cells": cross_effective_states,
        "interpretation": "Train/held-out is a local 80/20 teacher split per task and seed. Some binary masks recur across the split; no exact effective-state fingerprint (W*M,bias,readout,offset) recurs. Held-out teachers are excluded from pooled context; this is not a generator held-out reconstruction evaluation.",
    }


def _topology_summary(
    global_masks: set[str],
    global_effective_states: set[str],
    global_dense_masks: set[str],
    global_dense_effective_states: set[str],
    recipe_masks: dict[tuple[int, str], set[str]],
    shard_counts: list[dict[str, Any]],
) -> dict[str, Any]:
    per_cell = []
    for rho in RHOS:
        for recipe in RECIPES:
            key = (round(rho * 10), recipe)
            items = [r for r in shard_counts if r["rho_recipe_local_unique"].get(f"{rho:.1f}|{recipe}") is not None]
            local_unique_sum = sum(r["rho_recipe_local_unique"][f"{rho:.1f}|{recipe}"] for r in items)
            cell_solutions = len(items) * 32
            per_cell.append({
                "rho": rho,
                "recipe": recipe,
                "solution_count": cell_solutions,
                "global_unique_mask_topologies": len(recipe_masks[key]),
                "solution_rows_reusing_global_topology": cell_solutions - len(recipe_masks[key]),
                "sum_unique_per_seed_task_shard": int(local_unique_sum),
                "within_shard_duplicate_rows": int(cell_solutions - local_unique_sum),
            })
    local_counts = [int(item["unique_masks"]) for item in shard_counts]
    local_state_counts = [int(item["unique_effective_states"]) for item in shard_counts]
    sparse_fit_count = len(shard_counts) * 320
    return {
        "sparse_solution_fit_count": sparse_fit_count,
        "sparse_unique_mask_topologies_global": len(global_masks),
        "sparse_solution_rows_reusing_an_existing_global_mask": sparse_fit_count - len(global_masks),
        "unique_effective_states_global": len(global_effective_states),
        "unique_masks_per_seed_task_shard_minmax": [min(local_counts), max(local_counts)],
        "unique_effective_states_per_seed_task_shard_minmax": [min(local_state_counts), max(local_state_counts)],
        "dense_solution_fit_count": len(shard_counts) * 8,
        "dense_unique_mask_topologies_global": len(global_dense_masks),
        "dense_unique_effective_states_global": len(global_dense_effective_states),
        "unique_effective_states_all_fits_global": len(global_effective_states | global_dense_effective_states),
        "density_recipe_counts": per_cell,
        "seed_task_counts": [{k: v for k, v in row.items() if k != "rho_recipe_local_unique"}
                              for row in shard_counts],
        "fingerprint_definition": "Exact SHA-256 of packed binary mask; effective state fingerprint hashes contiguous exact tensors (W*M, bias, readout, per-image offset), including dtype and shape.",
        "interpretation": "A solution fit is a separately initialized/trained row; repeated mask topologies remain separate fits. Exact effective-state hashes are checked separately.",
    }


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _fmt_ci(value: dict[str, Any]) -> str:
    low, high = value["ci95_low"], value["ci95_high"]
    if value["mean"] is None:
        return "нет данных"
    if low is None or high is None:
        return f"{value['mean']:.4f} (n={value['n_seeds']})"
    return f"{value['mean']:.4f} [{low:.4f}, {high:.4f}]"


def _write_report(
    source: Path,
    output: Path,
    partial: bool,
    expected_seeds: list[int],
    missing: list[str],
    summary: dict[str, Any],
    fit_records: list[dict[str, Any]],
    figure_paths: dict[str, Path | None],
    paired_plot_meta: dict[str, Any],
    context_figures: list[dict[str, str]],
    structure_meta: dict[str, Any] | None,
) -> None:
    status_line = (
        "**Статус: частичный предварительный отчет.** Он включает только имеющиеся shard-файлы и не предназначен для итоговых выводов."
        if partial else "**Статус: полный отчет.** Все ожидаемые seed/task shards и маркеры завершения проверены."
    )
    cap_text = summary.get("population_cap") if summary.get("population_cap") is not None else "не сохранён"
    minimum_text = summary.get("population_minimum") if summary.get("population_minimum") is not None else "не сохранён"
    decay_text = summary.get("population_decay_every") if summary.get("population_decay_every") is not None else "не сохранён"
    missing_text = "\n".join(f"- `{item}`" for item in missing) if missing else "- Нет."
    table_rows = []
    for row in summary["density_recipe_table"]:
        sparse = _fmt_ci(row["sparse_audit_nmse_by_seed"])
        dense = _fmt_ci(row["paired_dense_audit_nmse_by_seed"])
        gap = _fmt_ci(row["paired_gap_by_seed"])
        win = f"{row['paired_sparse_wins']}/{row['paired_n']}"
        table_rows.append([
            f"{row['rho']:.1f}",
            "случайная" if row["recipe"] == "random" else "dense-functional",
            str(row["candidate_count"]), sparse, dense, gap, win,
        ])

    conv_rows = []
    for method, label in (("dense", "Dense controls"), ("sparse", "Sparse candidates")):
        item = summary["convergence"][method]
        conv_rows.append([
            label, f"{item['fit_count']}/{item['fit_all_plateau_count']}",
            f"{item['plateau_candidate_count']}/{item['candidate_count']}",
            f"{item['capped_candidate_count']}/{item['candidate_count']}",
            f"{item['stopping_steps_min']}–{item['stopping_steps_max']}",
            f"{item['fit_terminal_steps_min']}–{item['fit_terminal_steps_max']}",
        ])

    figures_text = []
    descriptions = {
        "audit_nmse_by_density.png": f"**Рисунок 1.** Ось x — плотность маски $\\rho$; ось y — точное значение эмпирической iid-популяционной ошибки, рассчитанное по 3000 audit-изображениям. Точки усреднены по seed после усреднения по четырем task-bank ячейкам; интервалы — описательные 95% t-интервалы по {summary['seed_count']} seed. Пунктир — exact dense-контроль; все варианты и replicas включены.",
        "paired_dense_gaps.png": f"**Рисунок 2.** Ось x — плотность и тип генерации, ось y — точная разность sparse minus парный dense NMSE; отрицательная разность означает меньшую ошибку sparse. Каждый box содержит до {summary['seed_count']} средних по seed, где четыре фиксированные задачи уже усреднены; это не распределение независимых кандидатов.",
        "source_population_training_curves.png": "**Рисунок 3.** Ось x — шаг Adam; слева y — exact source-population NMSE без L2, справа y — диагностический NMSE на одних и тех же 512 source-validation наборах. Для каждой кривой значения сначала усредняют кандидатов внутри task×seed shard, затем четыре cost-вектора внутри seed; линии — среднее по восьми seed, полоса — межseed стандартное отклонение. Для каждой группы отображены только checkpoint-шаги, присутствующие во всех её shards. Plateau проверяется по отдельной fit-цели с L2.",
        "representative_masks_and_signed_weights.png": "**Рисунок 4.** По строкам — dense control и sparse-маски пяти плотностей; слева бинарная поддержка (0 белый, 1 черный), справа фактические знаковые эффективные веса $W\\odot M$ на общей симметричной шкале с нулем в центре. Ось x — входной признак, y — скрытый нейрон. Это один конкретный пример: seed 4100, task 0, replica 0, dense-functional variant 0; рисунок не является усредненной картой.",
        "monte_carlo_vs_exact_audit.png": "**Рисунок 5.** Ось x — exact audit для конечной эмпирической source-популяции из 3000 изображений; ось y — Monte Carlo оценка по 2048 iid-наборам длины 5. Левая панель: каждая точка — sparse fit; правая: парный dense score показан в строках сравнения и поэтому повторяется для кандидатов одного dense replica. Цвет обозначает density sparse-кандидата. Пунктир $y=x$ — совпадение оценивателей. Оценка Monte Carlo вторична и не заменяет exact audit.",
    }
    for name in ("audit_nmse_by_density.png", "paired_dense_gaps.png",
                 "source_population_training_curves.png", "representative_masks_and_signed_weights.png",
                 "monte_carlo_vs_exact_audit.png"):
        path = figure_paths.get(name)
        if path is not None and path.is_file():
            rel = os.path.relpath(path, output)
            figures_text.extend([f"![{name}]({rel})", "", descriptions[name], ""])
    for item in context_figures:
        figures_text.extend([
            f"![{item['title']}]({item['path']})", "",
            f"**Функциональный профиль.** {item['caption']}", "",
        ])
    figures_text_joined = "\n".join(figures_text)

    structure_text = ""
    if structure_meta and figure_paths.get("representative_masks_and_signed_weights.png"):
        structure_text = (
            f"Выбранная иллюстрация: seed {structure_meta['seed']}, task {structure_meta['task']}, "
            f"replica {structure_meta['replica']}, recipe {structure_meta['sparse_recipe']}, "
            f"variant {structure_meta['variant']} (noise scale {structure_meta['variant_noise_scale']:.1f}). "
            "Парный dense-контроль использует тот же replica ID и исходную инициализацию.\n\n"
        )
    dense_mean = summary["dense_audit_nmse_by_seed"]["mean"]
    old_vs_new = (
        f"В прежнем bank-протоколе сообщалось среднее около 0.5832, тогда как новый dense-контроль здесь дает "
        f"{dense_mean:.4f}. Это только историческое ориентировочное сопоставление: старый банк использовал "
        "другой набор данных и обучение примерно на 20% источника, новый банк смешивает плотности и использует "
        "population-цель. Значения не являются парным matched-сравнением и не идентифицируют причинный эффект."
    )
    protocol_path = source / "protocol.json"
    protocol_link = f"[protocol.json]({os.path.relpath(protocol_path, output)})" if protocol_path.is_file() else "protocol.json пока отсутствует"
    context_manifest = source / "seed_4100" / "functional_context_manifest.json"
    context_link = f"[functional_context_manifest.json]({os.path.relpath(context_manifest, output)})" if context_manifest.is_file() else "functional_context_manifest.json пока отсутствует"
    partial_missing_block = f"\n\nОжидаемые, но отсутствующие или некорректные файлы/маркеры:\n\n{missing_text}\n" if partial else ""
    provenance = summary.get("solver_selection_provenance")
    if provenance:
        solver_note = (
            f"Сначала bank v3 предложил lr={provenance.get('lr')}, λ_sparse={provenance.get('sparse_l2')} и "
            f"λ_dense={provenance.get('dense_l2')} при стохастическом fresh-set обучении. Затем отдельные "
            "population pilot и L2 pilot с точной эмпирической population-целью на двух held-out meta-validation "
            "cost-задачах при предложенном lr=0.002 варьировали и независимо проверяли L2-регуляризацию. "
            "Фактический production снимок использовал этот "
            f"optimizer при аналитической population-цели, decay={provenance.get('population_decay_every')}, "
            f"cap={provenance.get('cap')}, minimum={provenance.get('minimum')}. Предварительное предложение, "
            "подтверждение и production-run — отдельные этапы; audit NMSE не выбирал кандидаты или настройки."
        )
    else:
        solver_note = "Артефакт выбора solver settings не найден; итоговые настройки приведены в per-shard metadata, если они сохранены."
    report = fr"""# Перестроенный функциональный банк DeepSets: отчет по source-аудиту

{status_line}
{partial_missing_block}

Источник: `{source}`. Ожидались seed {expected_seeds[0]}–{expected_seeds[-1]} и четыре фиксированных source cost-вектора из одной synthetic task family. Набор содержит **{summary['seed_count']} seed**, **{summary['source_task_count']} повторяемых cost-векторов** и **{summary['candidate_count_sparse']} sparse fit-решений** плюс **{summary['candidate_count_dense']} dense-control fit-решений**. Четыре вектора — не четыре дополнительных независимых seed или четыре family; часть source-пулов между seed может пересекаться.

## Цель и протокол

Для известной source-функции с индивидуальной целью $c[\mathrm{{digit}}]$ определим ошибку одного изображения $e(x)=g(x)-c[\mathrm{{digit}}]$. Для iid-наборов размера пять точная ожидаемая нормированная ошибка равна

$$
\mathbb{{E}}\left[\frac{{(\sum_{{i=1}}^5 e_i)^2}}{{5}}\right]=\mathbb{{E}}[e^2]+4\,\mathbb{{E}}[e]^2.
$$

Эта формула использует привилегию source audit: на этапе построения банка известны индивидуальная целевая метка $c[\\mathrm{{digit}}]$ каждого изображения и полный source image pool. Для будущей target-задачи такие метки отдельных изображений недоступны. DeepSets подгоняются к source-population цели на полном source image pool; градиент получает один packed матричный проход по изображениям и кандидатам. Регуляризатор fit-цели равен

$$
0.5\lambda(\|W\odot M\|_2^2+\|b\|_2^2+\|a\|_2^2+o^2).
$$

Решение о plateau/продлении использует только source population objective. Source-validation query и отдельный audit-набор нужны для диагностики и таблиц; они не управляют градиентом или остановкой.

Sparse shards сохраняют все 320 решений на task: пять плотностей $\rho\in\{{0.1,0.3,0.5,0.7,0.9\}}$, две recipe (`random`, `dense_functional`), четыре варианта шума и восемь начальных replicas. Dense control сохраняет восемь replicas отдельно. Ни один кандидат не отбирался по audit NMSE. Для каждого sparse-кандидата показано сравнение с dense-моделью той же task и replica; парный разрыв — sparse NMSE минус dense NMSE. Отрицательное значение благоприятствует sparse.

{solver_note} Population fit использовал cap {cap_text}, minimum {minimum_text} и decay каждые {decay_text} шагов; дополнительные шаги задавались plateau-флагом source objective.

Это source-аудит. Он не переносит индивидуальные source-цели $c[\mathrm{{digit}}]$ на target task: target child по-прежнему получает только наблюдаемые метки 5-image наборов (205 train и 51 диагностический query набор), а не метки отдельных изображений. Истинная карта важности $U$ не строилась.

## Аудитная ошибка и парное сравнение

Основная метрика берется из `exact_audit.pt`: это точное значение iid-ожидаемой нормированной ошибки для эмпирического audit-пула из 3000 изображений, рассчитанное как $\\mathbb{{E}}[e^2]+4\\mathbb{{E}}[e]^2$. «Точное» относится к конечной эмпирической популяции, а не к неизвестному непрерывному распределению изображений. В каждой ячейке таблицы candidate-level means включают все варианты и replicas. Интервалы в средних построены на seed-уровне: сначала кандидаты внутри task×seed ячейки, затем четыре фиксированные задачи усредняются внутри seed, затем считается среднее по восьми seed.

{_md_table(['ρ','Recipe','Кандидатов','Sparse exact NMSE: среднее [95% описат. интервал]','Парный dense exact NMSE','Exact парная разность','Sparse wins / n'], table_rows)}

Exact sparse mean: **{summary['exact_audit']['sparse_nmse_mean']:.6f}**; paired dense mean: **{summary['exact_audit']['paired_dense_nmse_mean']:.6f}**; средний парный разрыв sparse − dense: **{summary['exact_audit']['paired_gap_mean']:.6f}**. Sparse выигрывает у matched dense в **{summary['exact_audit']['sparse_paired_wins']}/{summary['exact_audit']['n_paired_comparisons']}** строках (сравнение той же task/replica). Отдельный dense mean по seed: **{_fmt_ci(summary['dense_audit_nmse_by_seed'])}**.

Сохраненный Monte Carlo audit из 2048 случайных iid-наборов по 5 изображений приведен только как вторичная оценка: среднее sparse **{summary['monte_carlo_secondary']['sparse_nmse_mean']:.6f}**, paired dense **{summary['monte_carlo_secondary']['paired_dense_nmse_mean']:.6f}**; средняя абсолютная разница Monte Carlo с exact равна {summary['monte_carlo_secondary']['sparse_abs_error_vs_exact_mean']:.6g} для sparse и {summary['monte_carlo_secondary']['dense_abs_error_vs_exact_mean']:.6g} для dense. Артефакт `audit_replay.pt` повторно вычисляет сохраненные MC-выборки: на {summary['monte_carlo_secondary']['replayed_metric_count']} значениях максимум |stored − replay| составляет **{summary['monte_carlo_secondary']['stored_vs_replay_max_abs_error']:.1f}**. Это проверка воспроизводимости сэмплирования, а не доказательство равенства Monte Carlo и exact audit.

{old_vs_new}

## Сходимость

Plateau-флаг относится к population objective, не к audit/query score. Ниже приведены фактические counts по сохраненным child fits; `fit_count / all_plateau` показывает число shard fits и сколько из них имели plateau-флаг у каждого кандидата к terminal checkpoint. `plateau candidates` и `capped candidates` — counts по отдельным решениям.

{_md_table(['Группа','Fits / все plateau','Plateau candidates','Ограничены cap','Индивидуальный stopping step, min–max','Terminal fit step, min–max'], conv_rows)}

## Структура кандидатов и повторения

Число fit-строк не равно числу уникальных бинарных масок. В {summary['topology']['sparse_solution_fit_count']} отдельных sparse fit-решениях найдено {summary['topology']['sparse_unique_mask_topologies_global']} точных бинарных топологий; {summary['topology']['sparse_solution_rows_reusing_an_existing_global_mask']} строк повторяют уже встреченную маску. Внутри отдельного seed×task bank число уникальных масок варьируется от {summary['topology']['unique_masks_per_seed_task_shard_minmax'][0]} до {summary['topology']['unique_masks_per_seed_task_shard_minmax'][1]} при 320 строках. При этом точный hash эффективного состояния $(W\\odot M,b,a,o)$ дает {summary['topology']['unique_effective_states_global']} различных обученных состояний: одинаковая support-маска не объединяет разные fit-решения. Для density/recipe разреза по количеству решений и глобальных топологий см. `summary.json`. Dense controls содержат {summary['topology']['dense_solution_fit_count']} fit-решений, но всего {summary['topology']['dense_unique_mask_topologies_global']} топологию all-on.

Dense controls содержат {summary['topology']['dense_solution_fit_count']} отдельных fit-решений с одной mask-топологией, но {summary['topology']['dense_unique_effective_states_global']} различными эффективными состояниями. Sparse и dense вместе дают {summary['topology']['unique_effective_states_all_fits_global']} уникальных exact effective-state hashes. Для нового source-банка внешняя статистическая единица — до восьми seed, а не 32 task×seed shard и не тысячи масок. Указанные t-интервалы описательны: source-пулы могут пересекаться, задач всего четыре, поэтому интервал не следует читать как формальное обобщение на популяцию новых задач.

## Графики

{figures_text_joined}

## Контекст и ограничения

Полные банки содержат 320 sparse-кандидатов на seed/task плюс отдельные dense controls; контекстный extractor использует label-free alignment и pooled train-only функциональные признаки $\\psi$ и $q$. По каждому из {summary['context_split']['split_cells_checked']} seed×task teacher-шардов 256 из 320 строк вошли в train context, 64 held-out строки исключены из alignment и pooled context. В {summary['context_split']['train_heldout_shared_mask_topology_intersections_across_cells']} из этих разбиений встречаются общие точные mask-топологии: {summary['context_split']['train_rows_whose_mask_topology_also_occurs_in_heldout_across_cells']} train-строк имеют маску, встречающуюся и в held-out части; точных совпадений эффективного состояния $(W\\odot M,b,a,o)$ нет. Это повторение масок, не held-out функциональная реконструкция, и отчет не заявляет такую оценку. Audit NMSE, candidate quality и recipe не входят в context-векторы. Знаки $q$ различают положительный и отрицательный вклад; $|q|$ показывает величину без знака.

Данные охватывают четыре фиксированных source cost-вектора из одной synthetic task family и восемь seed на доступных MNIST8m пулах. Плотность/recipe сравнения полезны как диагностика этого source bank и его генератора, но не устанавливают перенос на произвольные задачи. Результаты для графика реальных масок/весов — только один прозрачно выбранный пример. {structure_text}

## Артефакты

- [summary.json](summary.json) — агрегаты, интервалы, paired wins и convergence counts.
- [plot_arrays.npz](plot_arrays.npz) — все candidate-level audit значения и seed-level plot arrays.
- [exact_audit.pt, seed 4100](seed_4100/exact_audit.pt) и [audit_replay.pt, seed 4100](seed_4100/audit_replay.pt) — примеры основных per-seed артефактов; аналогичные файлы сохранены во всех восьми seed. [solver_selection_provenance.json]({os.path.relpath(source / 'solver_selection_provenance.json', output)}) фиксирует цепочку выбора solver.
- {protocol_link} и per-seed {context_link} — протокол, split и определение функциональных признаков.
- Статус готовности: `{ 'partial' if partial else 'complete' }`.

"""
    report = report.replace("\\\\", "\\")
    (output / "RESULTS_RU.md").write_text(report, encoding="utf-8")


def _validate_report_markdown(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    display_pattern = re.compile(r"(?ms)^\$\$\s*\n.+?^\$\$\s*$")
    display_blocks = list(display_pattern.finditer(text))
    if not display_blocks:
        raise ValueError("RESULTS_RU.md is missing display math blocks")
    remainder = display_pattern.sub("", text)
    if "$$" in remainder or remainder.count("$") % 2:
        raise ValueError("RESULTS_RU.md has unmatched inline/display math delimiters")
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    for target in links:
        if re.match(r"^[a-z]+://", target, flags=re.IGNORECASE):
            continue
        local_path = (path.parent / target).resolve()
        if not local_path.is_file():
            raise FileNotFoundError(f"broken local Markdown link: {target} -> {local_path}")


def _write_master_report(path: Path, source: Path, summary: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    full_report = source / "RESULTS_RU.md"
    report_rel = os.path.relpath(full_report, path.parent)
    fig_paths = [source / "figures" / name for name in (
        "audit_nmse_by_density.png", "paired_dense_gaps.png",
        "representative_masks_and_signed_weights.png", "monte_carlo_vs_exact_audit.png",
        "source_population_training_curves.png")]
    fig_paths.extend(source / "seed_4100" / "figures" / name
                     for name in ("psi_two_teachers.png", "q_signed_mean.png", "q_abs_mean.png"))
    figure_blocks = []
    captions = {
        "audit_nmse_by_density.png": "По X — доля активных связей; по Y — точный ожидаемый NMSE для iid-наборов из пяти элементов на 3000 audit-изображениях. Панели разделяют random и dense-functional маски; пунктир показывает dense. Точки — средние по восьми seed, интервалы описательные. Средние sparse-оценки уступают dense при всех проверенных плотностях.",
        "paired_dense_gaps.png": "По X — плотность и способ построения маски; по Y — exact sparse NMSE минус paired dense NMSE. В каждом box — восемь seed-средних по четырём исходным задачам. Отрицательная разность благоприятна sparse; это не распределение тысяч независимых задач.",
        "representative_masks_and_signed_weights.png": "Пример seed 4100, task 0, replica 0, dense-functional variant 0. По X — пиксель 0–783, по Y — hidden-нейрон 0–31. Слева binary mask: белый — 0, чёрный — 1. Справа реальные signed $W\\odot M$ на общей симметричной шкале: красный — положительный вес, синий — отрицательный. Это отдельные обученные состояния, не усреднение.",
        "monte_carlo_vs_exact_audit.png": "По X — точный ожидаемый audit NMSE; по Y — вторичная Monte Carlo оценка по 2048 iid-наборам из пяти элементов. Панели показывают sparse и paired dense; цвет обозначает плотность. Пунктир $y=x$ показывает совпадение оценок. Разброс отражает конечное число сэмплированных наборов.",
        "source_population_training_curves.png": "По X — шаг Adam. Слева — точный source NMSE без L2, справа — NMSE на 512 source-validation наборах. Линии — средние по восьми seed для dense и sparse, полосы — межseed стандартное отклонение. Plateau проверяется отдельно по fit-цели с L2; query остаётся диагностикой.",
        "psi_two_teachers.png": "Seed 4100, task 0: по X — 128 общих probe-изображений, по Y — aligned hidden-нейрон. Цвет показывает signed вклад $\\psi_j(x)$ на общей симметричной шкале. Показаны train и held-out учителя; held-out исключён из context. Это полные профили вкладов, без дополнительного отсечения значений.",
        "q_signed_mean.png": "Seed 4100: по X — пиксель, по Y — aligned hidden-нейрон. Цвет — средний signed $q_{ij}(x)$ на probe-выборке; шкала симметрична относительно нуля. Знак различает положительный и отрицательный вклад связи.",
        "q_abs_mean.png": "Seed 4100: по X — пиксель, по Y — aligned hidden-нейрон. Цвет — средний абсолютный $q_{ij}(x)$ на probe-выборке. Это производная статистика полного функционального представления; бинарная маска хранится отдельно.",
    }
    for fig in fig_paths:
        if fig.is_file():
            rel = os.path.relpath(fig, path.parent)
            figure_blocks.append(f"![{fig.name}]({rel})\n\n{captions[fig.name]}")
    figure_blocks_text = "\n\n".join(figure_blocks)
    exact = summary["exact_audit"]
    topology = summary["topology"]
    context = summary["context_split"]
    report = fr"""# Перестроенный функциональный банк DeepSets

## Результат

Основной source-audit — точная эмпирическая iid-population метрика по 3000 audit-изображениям. Средний NMSE sparse решений равен {exact['sparse_nmse_mean']:.6f}, paired dense controls — {exact['paired_dense_nmse_mean']:.6f}; sparse лучше в {exact['sparse_paired_wins']} из {exact['n_paired_comparisons']} matched сравнений task/replica. В банке восемь seed, четыре фиксированных source cost-вектора из одной synthetic task family, {summary['candidate_count_sparse']} sparse fit-решений и {summary['candidate_count_dense']} dense-control решений.

$$
\\mathbb{{E}}\\left[\\frac{{(\\sum_{{i=1}}^5 e_i)^2}}{{5}}\\right]=\\mathbb{{E}}[e^2]+4\\mathbb{{E}}[e]^2
$$

Это source-bank сравнение. Известные индивидуальные метки $c[\\mathrm{{digit}}]$ доступны при построении source teachers; они не передаются target-задаче, где остаются только наблюдаемые метки наборов. Четыре cost-вектора из одной synthetic family не являются четырьмя независимыми seed или четырьмя families.

## Топология против fit-решений

В {topology['sparse_solution_fit_count']} sparse fit-строк найдено {topology['sparse_unique_mask_topologies_global']} точных бинарных mask-топологий и {topology['unique_effective_states_global']} уникальных точных эффективных состояний $(W\\odot M,b,a,o)$. Идентичная support-маска поэтому не означает идентичный fit. У dense controls {topology['dense_solution_fit_count']} отдельных fit-решений используют одну topology all-on и {topology['dense_unique_effective_states_global']} эффективных состояний. Train/held-out teacher split — 256/64 в каждой task×seed ячейке; {context['train_heldout_shared_mask_topology_intersections_across_cells']} ячейки имеют повтор маски через split, но exact effective-state дубликатов нет. Held-out teachers исключены из pooled context; held-out reconstruction для обученного генератора здесь не измерялся.

## Exact и Monte Carlo audit

Вторичный Monte Carlo audit использует 2048 сэмплированных iid-наборов по 5 изображений. Replay сохраненных индексов дает максимальное отличие от сохраненных MC значений {summary['monte_carlo_secondary']['stored_vs_replay_max_abs_error']:.1f}; это проверяет replay, но exact audit остается главным. Solver provenance разделяет первоначальное stochastic v3 предложение, exact-population pilots с варьированием L2 при предложенном lr=0.002 на двух meta-validation cost vectors и actual production snapshot. Все {summary['convergence']['sparse']['candidate_count']}/{summary['convergence']['sparse']['candidate_count']} sparse и {summary['convergence']['dense']['candidate_count']}/{summary['convergence']['dense']['candidate_count']} dense fits получили plateau-флаг.

Полный отчет, методика, caveats, density/recipe таблицы и контекстные фигуры: [{full_report.name}]({report_rel}).

{figure_blocks_text}
"""
    report = report.replace("\\\\", "\\")
    path.write_text(report, encoding="utf-8")
    _validate_report_markdown(path)


def build_report(source: Path, output: Path, *, partial: bool = False,
                 master_report: Path | None = None) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    protocol = _json_read(source / "protocol.json")
    selection = _json_read(source / "bank_v3_pilot" / "selection.json")
    solver_provenance = _json_read(source / "solver_selection_provenance.json")
    expected_seeds = _expected_seeds(source, protocol)
    expected = [f"seed_{seed}/{'bank' if kind == 'sparse' else 'dense'}_{task}.pt"
                for seed in expected_seeds for task in TASKS for kind in ("sparse", "dense")]
    expected.extend(f"seed_{seed}/{name}" for seed in expected_seeds
                    for name in ("exact_audit.pt", "audit_replay.pt", "functional_context.pt",
                                 "functional_context_manifest.json"))
    expected.append("solver_selection_provenance.json")
    missing: list[str] = []
    if protocol is None:
        missing.append("protocol.json")
    if not (source / "COMPLETE").is_file():
        missing.append("COMPLETE")
    for seed in expected_seeds:
        if not (source / f"seed_{seed}" / "COMPLETE").is_file():
            missing.append(f"seed_{seed}/COMPLETE")
    missing.extend(path for path in expected if not (source / path).is_file())
    if not partial and missing:
        preview = "\n".join(f"- {item}" for item in missing[:30])
        raise FileNotFoundError(
            "Refusing to build the final report before all expected shards and completion markers exist. "
            "Use --partial only for an explicitly interim report. Missing:\n" + preview
        )
    if not partial and (len(expected_seeds) != 8 or len(list(TASKS)) != 4):
        raise ValueError("the complete report requires eight seeds and four fixed source tasks")

    rows: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    invalid: list[str] = []
    all_mask_hashes: set[str] = set()
    all_effective_hashes: set[str] = set()
    all_dense_mask_hashes: set[str] = set()
    all_dense_effective_hashes: set[str] = set()
    masks_by_recipe: dict[tuple[int, str], set[str]] = defaultdict(set)
    topology_shards: list[dict[str, Any]] = []
    replay_max_error = 0.0
    replay_checks = 0
    for seed in expected_seeds:
        seed_dir = source / f"seed_{seed}"
        # Partial reports consume only seed folders that have committed their
        # completion marker; an actively written shard is never read mid-save.
        if partial and not (seed_dir / "COMPLETE").is_file():
            continue
        try:
            exact = _load_pt(seed_dir / "exact_audit.pt")
            replay = _load_pt(seed_dir / "audit_replay.pt")
            if int(exact.get("seed", -1)) != seed or int(replay.get("seed", -1)) != seed:
                raise ValueError("exact audit / replay seed metadata mismatch")
            if exact.get("split_hash") != replay.get("split_hash"):
                raise ValueError("exact audit and Monte Carlo replay use different audit splits")
            source_ids = torch.as_tensor(exact["source_ids"]).detach().cpu()
            replay_source_ids = torch.as_tensor(replay["split_source_ids"]).detach().cpu()
            if not torch.equal(source_ids, replay_source_ids):
                raise ValueError("exact audit and replay source image IDs differ")
            exact_sparse_all = torch.as_tensor(exact["sparse_nmse"]).detach().float().cpu().numpy()
            exact_dense_all = torch.as_tensor(exact["dense_nmse"]).detach().float().cpu().numpy()
            exact_paired_all = torch.as_tensor(exact["paired_dense_nmse"]).detach().float().cpu().numpy()
            exact_gap_all = torch.as_tensor(exact["paired_difference"]).detach().float().cpu().numpy()
            if exact_sparse_all.shape != (4, 320) or exact_dense_all.shape != (4, 8):
                raise ValueError(f"unexpected exact audit shapes: {exact_sparse_all.shape}, {exact_dense_all.shape}")
            for key in ("stored_sparse_nmse", "replay_sparse_nmse", "sparse_absdiff"):
                if torch.as_tensor(replay[key]).shape != (4, 320):
                    raise ValueError(f"unexpected replay {key} shape")
            for key in ("stored_dense_nmse", "replay_dense_nmse", "dense_absdiff"):
                if torch.as_tensor(replay[key]).shape != (4, 8):
                    raise ValueError(f"unexpected replay {key} shape")
            replay_error = max(
                float(torch.as_tensor(replay["sparse_absdiff"]).abs().max()),
                float(torch.as_tensor(replay["dense_absdiff"]).abs().max()),
            )
            replay_max_error = max(replay_max_error, replay_error)
            if not np.isfinite(exact_sparse_all).all() or not np.isfinite(exact_dense_all).all():
                raise ValueError("exact audit contains non-finite values")
        except Exception as exc:
            invalid.append(f"seed_{seed}/exact_audit: {type(exc).__name__}: {exc}")
            if not partial:
                raise ValueError(invalid[-1]) from exc
            continue
        for task in TASKS:
            sparse_path, dense_path = seed_dir / f"bank_{task}.pt", seed_dir / f"dense_{task}.pt"
            if not sparse_path.is_file() or not dense_path.is_file():
                continue
            try:
                sparse = _load_pt(sparse_path)
                dense = _load_pt(dense_path)
                mc_sparse_nmse = _flat_metric(sparse, "audit_nmse")
                recipes = sparse.get("candidate_recipe")
                exact_recipes = exact.get("candidate_recipe", [None] * 4)[task]
                if not isinstance(recipes, list) or len(recipes) != len(mc_sparse_nmse):
                    raise ValueError("candidate_recipe rows do not match sparse audit vector")
                if len(recipes) != 320 or len(exact_recipes) != 320:
                    raise ValueError("sparse bank and exact audit must each contain all 320 candidate rows")
                mc_dense_nmse = _flat_metric(dense, "audit_nmse")
                mc_paired_dense = _flat_metric(sparse, "paired_dense_audit_nmse", len(mc_sparse_nmse))
                mc_paired_gap = _flat_metric(sparse, "paired_difference", len(mc_sparse_nmse))
                if len(mc_dense_nmse) != 8:
                    raise ValueError("dense control must contain eight replicas")
                exact_sparse = np.asarray(exact_sparse_all[task], dtype=np.float64)
                exact_dense = np.asarray(exact_dense_all[task], dtype=np.float64)
                paired_dense = np.asarray(exact_paired_all[task], dtype=np.float64)
                paired_gap = np.asarray(exact_gap_all[task], dtype=np.float64)
                rebuilt_dense = np.asarray([exact_dense[int(item["replica"])] for item in recipes])
                if not np.allclose(paired_dense, rebuilt_dense, rtol=1e-6, atol=1e-7):
                    raise ValueError("exact paired dense values do not match the same-replica exact dense controls")
                if not np.allclose(paired_gap, exact_sparse - paired_dense, rtol=1e-6, atol=1e-7):
                    raise ValueError("exact paired difference is inconsistent with sparse minus paired dense")
                replay_sparse = torch.as_tensor(replay["replay_sparse_nmse"]).detach().float().cpu().numpy()[task]
                replay_dense = torch.as_tensor(replay["replay_dense_nmse"]).detach().float().cpu().numpy()[task]
                replay_stored_sparse = torch.as_tensor(replay["stored_sparse_nmse"]).detach().float().cpu().numpy()[task]
                replay_stored_dense = torch.as_tensor(replay["stored_dense_nmse"]).detach().float().cpu().numpy()[task]
                replay_max_error = max(replay_max_error,
                    float(np.max(np.abs(replay_sparse - mc_sparse_nmse))),
                    float(np.max(np.abs(replay_dense - mc_dense_nmse))),
                    float(np.max(np.abs(replay_stored_sparse - mc_sparse_nmse))),
                    float(np.max(np.abs(replay_stored_dense - mc_dense_nmse))))
                replay_checks += len(mc_sparse_nmse) + len(mc_dense_nmse)
                if not np.allclose(mc_paired_dense,
                                   np.asarray([mc_dense_nmse[int(item["replica"])] for item in recipes]),
                                   rtol=1e-5, atol=1e-6):
                    raise ValueError("Monte Carlo paired dense values do not match same-replica controls")
                if not np.allclose(mc_paired_gap, mc_sparse_nmse - mc_paired_dense,
                                   rtol=1e-5, atol=1e-6):
                    raise ValueError("Monte Carlo paired difference is inconsistent")
                recipe_counts: dict[tuple[float, str], int] = defaultdict(int)
                for recipe_row in recipes:
                    recipe_counts[(round(float(recipe_row["rho"]), 1), str(recipe_row["recipe"]))] += 1
                if any(recipe_counts[(rho, name)] != 32 for rho in RHOS for name in RECIPES):
                    raise ValueError("expected all 32 variant×replica candidates in each density/recipe cell")
                candidate_rows = []
                task_dense_rows = []
                sparse_state = sparse["state_dict"]
                dense_state = dense.get("source_state_dict", dense["state_dict"])
                if len(sparse_state["masks"]) != 320 or len(dense_state["masks"]) != 8:
                    raise ValueError("mask tensor counts do not match saved candidate rows")
                shard_mask_hashes: set[str] = set()
                shard_state_hashes: set[str] = set()
                shard_recipe_hashes: dict[str, set[str]] = defaultdict(set)
                for index, recipe in enumerate(recipes):
                    rho = round(float(recipe["rho"]), 1)
                    name = str(recipe["recipe"])
                    if rho not in RHOS or name not in RECIPES:
                        raise ValueError(f"unexpected recipe cell: {recipe}")
                    if recipe != exact_recipes[index]:
                        raise ValueError(f"exact audit candidate_recipe differs at row {index}")
                    mask_hash = _mask_fingerprint(sparse_state["masks"][index])
                    state_hash = _effective_state_fingerprint(sparse_state, index)
                    shard_mask_hashes.add(mask_hash)
                    shard_state_hashes.add(state_hash)
                    all_mask_hashes.add(mask_hash)
                    all_effective_hashes.add(state_hash)
                    masks_by_recipe[(round(10 * rho), name)].add(mask_hash)
                    shard_recipe_hashes[f"{rho:.1f}|{name}"].add(mask_hash)
                    candidate_rows.append({
                        "seed": seed, "task": task, "rho": rho, "rho_tenths": round(10*rho),
                        "recipe": name, "variant": int(recipe["variant"]),
                        "replica": int(recipe["replica"]),
                        "sparse_nmse": float(exact_sparse[index]),
                        "paired_dense_nmse": float(paired_dense[index]),
                        "paired_gap": float(paired_gap[index]),
                        "mc_sparse_nmse": float(mc_sparse_nmse[index]),
                        "mc_paired_dense_nmse": float(mc_paired_dense[index]),
                        "mc_paired_gap": float(mc_paired_gap[index]),
                        "mask_hash": mask_hash,
                        "effective_state_hash": state_hash,
                    })
                for replica, score in enumerate(exact_dense):
                    dense_mask_hash = _mask_fingerprint(dense_state["masks"][replica])
                    dense_state_hash = _effective_state_fingerprint(dense_state, replica)
                    all_dense_mask_hashes.add(dense_mask_hash)
                    all_dense_effective_hashes.add(dense_state_hash)
                    task_dense_rows.append({"seed": seed, "task": task, "replica": replica,
                                            "audit_nmse": float(score),
                                            "mc_audit_nmse": float(mc_dense_nmse[replica]),
                                            "mask_hash": dense_mask_hash,
                                            "effective_state_hash": dense_state_hash})
                topology_shards.append({
                    "seed": seed, "task": task, "candidate_count": len(recipes),
                    "unique_masks": len(shard_mask_hashes),
                    "unique_effective_states": len(shard_state_hashes),
                    "rho_recipe_local_unique": {key: len(value) for key, value in shard_recipe_hashes.items()},
                })
                sparse_fit = _history_record(sparse, "sparse", seed, task)
                dense_fit = _history_record(dense, "dense", seed, task)
                rows.extend(candidate_rows)
                dense_rows.extend(task_dense_rows)
                fit_records.extend((sparse_fit, dense_fit))
            except Exception as exc:
                invalid.append(f"seed_{seed}/task_{task}: {type(exc).__name__}: {exc}")
                if not partial:
                    raise ValueError(invalid[-1]) from exc
            finally:
                if "sparse" in locals():
                    del sparse
                if "dense" in locals():
                    del dense
    if invalid:
        missing.extend(f"invalid {item}" for item in invalid)
    if not rows or not dense_rows:
        raise ValueError("no valid sparse+dense task shards found; nothing to report")

    output.mkdir(parents=True, exist_ok=True)
    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = _summarize(rows, dense_rows, fit_records, sorted({r["seed"] for r in rows}))
    summary["topology"] = _topology_summary(all_mask_hashes, all_effective_hashes,
                                            all_dense_mask_hashes, all_dense_effective_hashes,
                                            masks_by_recipe,
                                            topology_shards)
    summary["context_split"] = _context_split_summary(source, sorted({r["seed"] for r in rows}))
    summary["monte_carlo_secondary"]["stored_vs_replay_max_abs_error"] = float(replay_max_error)
    summary["monte_carlo_secondary"]["replayed_metric_count"] = int(replay_checks)
    summary["monte_carlo_secondary"]["replay_split_hash_verified_per_seed"] = len({r["seed"] for r in rows})
    summary["exact_audit"]["audit_images_per_seed"] = 3000
    summary["exact_audit"]["objective"] = "E[e^2]+4(E[e])^2 exactly on the finite empirical audit-image pool"
    summary["monte_carlo_secondary"]["objective"] = "2048 sampled iid image sets of size five from the same audit pool"
    seed_cells = _seed_aggregates(rows)
    fig_quality = _plot_quality(summary, figures_dir)
    fig_gaps = _plot_paired_gaps(seed_cells, figures_dir)
    fig_mc = _plot_mc_against_exact(rows, figures_dir)
    fig_history, common_history_steps = _plot_histories(fit_records, figures_dir)
    fig_structure, structure_meta = _plot_masks_weights(source, figures_dir)
    context_figs = _context_figures(source, output)
    figure_paths = {
        fig_quality.name: fig_quality,
        fig_gaps.name: fig_gaps,
        "source_population_training_curves.png": fig_history,
        "representative_masks_and_signed_weights.png": fig_structure,
        fig_mc.name: fig_mc,
    }

    row_dtype = [
        ("seed", "i8"), ("task", "i4"), ("rho", "f4"), ("recipe", "U32"),
        ("variant", "i4"), ("replica", "i4"), ("sparse_nmse", "f8"),
        ("paired_dense_nmse", "f8"), ("paired_gap", "f8"),
        ("mc_sparse_nmse", "f8"), ("mc_paired_dense_nmse", "f8"), ("mc_paired_gap", "f8"),
        ("mask_hash", "U64"), ("effective_state_hash", "U64"),
    ]
    raw_rows = np.asarray([
        (r["seed"], r["task"], r["rho"], r["recipe"], r["variant"], r["replica"],
         r["sparse_nmse"], r["paired_dense_nmse"], r["paired_gap"],
         r["mc_sparse_nmse"], r["mc_paired_dense_nmse"], r["mc_paired_gap"],
         r["mask_hash"], r["effective_state_hash"])
        for r in rows
    ], dtype=row_dtype)
    dense_dtype = [("seed", "i8"), ("task", "i4"), ("replica", "i4"),
                   ("audit_nmse", "f8"), ("mc_audit_nmse", "f8"),
                   ("mask_hash", "U64"), ("effective_state_hash", "U64")]
    raw_dense = np.asarray([(r["seed"], r["task"], r["replica"], r["audit_nmse"],
                             r["mc_audit_nmse"], r["mask_hash"], r["effective_state_hash"])
                            for r in dense_rows], dtype=dense_dtype)
    curve_arrays: dict[str, np.ndarray] = {}
    for method in ("dense", "sparse"):
        items = [fit for fit in fit_records if fit["method"] == method]
        common = common_history_steps.get(method, []) if common_history_steps else []
        if items and common:
            by_seed: dict[int, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
            for fit in items:
                lookup = {int(step): i for i, step in enumerate(fit["steps"])}
                for step in common:
                    index = lookup[int(step)]
                    by_seed[fit["seed"]][int(step)].append((fit["trainNMSE"][index], fit["queryNMSE"][index]))
            seed_ids = sorted(by_seed)
            curve_arrays[f"{method}_history_steps"] = np.asarray(common, dtype=np.int64)
            curve_arrays[f"{method}_history_seed_ids"] = np.asarray(seed_ids, dtype=np.int64)
            curve_arrays[f"{method}_history_train_seedmeans"] = np.asarray([
                [np.mean([x[0] for x in by_seed[seed][step]]) for step in common] for seed in seed_ids
            ], dtype=np.float64)
            curve_arrays[f"{method}_history_query_seedmeans"] = np.asarray([
                [np.mean([x[1] for x in by_seed[seed][step]]) for step in common] for seed in seed_ids
            ], dtype=np.float64)
    np.savez_compressed(output / "plot_arrays.npz", candidates=raw_rows,
                        dense_controls=raw_dense, **curve_arrays)

    summary.update({
        "status": "partial" if partial or missing else "complete",
        "source_root": str(source),
        "expected_seeds": expected_seeds,
        "expected_task_count": 4,
        "missing_or_invalid": missing,
        "root_complete_marker": (source / "COMPLETE").is_file(),
        "context_figures": context_figs,
        "representative_structure": structure_meta,
        "fit_history_common_steps": common_history_steps,
        "fit_history_records": [{key: fit[key] for key in (
            "seed", "task", "method", "candidate_count", "plateau_candidates",
            "plateau_stopped_candidates", "capped_candidates", "terminal_step", "all_plateau"
        )} for fit in fit_records],
        "figure_files": [str(path.relative_to(output)) for path in figure_paths.values() if path is not None],
        "paired_gap_sign": "sparse audit NMSE minus dense audit NMSE for the same task and replica; negative favors sparse",
        "context_quality_nonuse": "quality, audit NMSE, and candidate_recipe are not context inputs; pooled contexts use train teachers only",
        "source_population_objective": "E[e^2]+4(E[e])^2, e=g(x)-c[digit]; exactly E[(sum_5 e_i)^2/5] for iid sets of size 5",
        "audit_nmse_definition": "Primary: exact E[e^2]+4(E[e])^2 on the finite 3000-image empirical audit population, equivalent to E[(sum_5 e_i)^2/5] for iid sampling from that pool. Secondary: 2048 sampled sets of size five.",
        "audit_artifact_roles": {"primary": "per-seed exact_audit.pt", "secondary_mc": "per-seed audit_replay.pt and shard audit_nmse vectors"},
        "stored_mc_replay_max_abs_error": float(replay_max_error),
        "stored_mc_replay_checked_values": int(replay_checks),
        "selection_rule": "all candidates retained; no audit-score filtering or candidate winner selection",
        "historical_old_bank_audit_nmse_reference": 0.5832,
        "historical_old_bank_comparison_note": "not matched/paired: old source data and training protocol (~20% of source pool) differed from this mixed-density population-fit bank",
        "solver_selection": selection,
        "solver_selection_provenance": solver_provenance,
        "population_cap": protocol.get("population_cap") if protocol else None,
        "population_minimum": protocol.get("population_minimum") if protocol else None,
        "population_decay_every": protocol.get("population_decay_every") if protocol else None,
    })
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                            encoding="utf-8")
    _write_report(source, output, bool(partial or missing), expected_seeds, missing,
                  summary, fit_records, figure_paths, {}, context_figs, structure_meta)
    _validate_report_markdown(output / "RESULTS_RU.md")
    if master_report is not None:
        if partial or missing:
            raise ValueError("refusing to write the final master report from a partial/incomplete bank")
        _write_master_report(master_report, output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_SOURCE / "report",
                        help="report output folder (default: source bank/report)")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE,
                        help="source-bank output containing protocol.json and seed folders")
    parser.add_argument("--partial", action="store_true",
                        help="write an explicitly labelled interim report from available complete shards")
    parser.add_argument("--master-report", type=Path, default=None,
                        help="also write a concise cross-report summary to this Markdown file")
    args = parser.parse_args()
    result = build_report(args.source_root, args.out, partial=args.partial,
                          master_report=args.master_report)
    print(json.dumps({"status": result["status"], "output": str(args.out.resolve()),
                      "sparse_candidates": result["candidate_count_sparse"],
                      "dense_controls": result["candidate_count_dense"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
