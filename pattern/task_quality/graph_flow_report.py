"""Build the Russian report and CPU figures for graph-flow mask quality.

Run from the repository root with::

    python -m pattern.task_quality.graph_flow_report

The builder consumes only frozen experiment artifacts. Gold structure is used
only for post-hoc column alignment in the reporting audit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .structure import align_mask_to_gold, signed_weight_audit


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "graph_flow_quality_20261001"
SEEDS = (8100, 8102)
METHODS = (
    "gnn_direct",
    "gnn_flow_single",
    "gnn_search8",
    "gnn_flow_search8",
    "functional_search8",
    "uniform_functional",
    "dense",
)
LABELS = {
    "gnn_direct": "GNN direct",
    "gnn_flow_single": "Flow single",
    "gnn_search8": "GNN search-8",
    "gnn_flow_search8": "Flow search-8",
    "functional_search8": "Functional search-8",
    "uniform_functional": "Uniform functional",
    "dense": "Dense",
}
SUPPORT_ONLY = {"gnn_direct", "gnn_flow_single", "uniform_functional", "dense"}
QUERY_SEARCH = {"gnn_search8", "gnn_flow_search8", "functional_search8"}
FIGURE_NAMES = (
    "figure_01_paired_losses.png",
    "figure_02_test_task_heatmap.png",
    "figure_03_masks_signed_weights.png",
    "figure_04_outer_training_curves.png",
    "figure_05_shared_child_losses.png",
    "figure_06_source_archive_utility.png",
    "figure_07_shared_child_history.png",
)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"),
                "minimum": float("nan"), "maximum": float("nan")}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _format(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _method_group(method: str) -> str:
    if method in QUERY_SEARCH:
        return "query-search"
    if method in SUPPORT_ONLY:
        return "support-only / fixed baseline"
    return "other"


def _history_value(history: list[dict[str, Any]], step: int, name: str,
                   index: int) -> float:
    match = next((item for item in reversed(history) if int(item["step"]) == step), None)
    if match is None:
        raise ValueError(f"missing child-history step {step}")
    values = match[name]
    if isinstance(values, torch.Tensor):
        return float(values[index].item())
    return float(values[index])


def _load_primary(root: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    seed_data: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        folder = root / f"seed_{seed}"
        records = _read_json(folder / "records.json")
        fit = torch.load(folder / "final_children_frozen.pt", map_location="cpu", weights_only=False)
        selected = torch.load(folder / "selected_masks_frozen.pt", map_location="cpu", weights_only=False)
        archive = torch.load(folder / "archive.pt", map_location="cpu", weights_only=False)
        training = _read_json(folder / "training_summary.json")
        evaluation = _read_json(folder / "evaluation_summary.json")
        if tuple(selected["methods"]) != METHODS:
            raise ValueError(f"unexpected method order for seed {seed}: {selected['methods']}")
        if not fit["fixed_horizon"] or int(fit["steps"]) != 1400 or int(fit["init_offset"]) != 2:
            raise ValueError(f"seed {seed} final-child protocol differs from fixed 1400 / init 2..5")
        if fit["replicas"] != 4 or len(records) != 168:
            raise ValueError(f"unexpected frozen evaluation row count for seed {seed}")
        if archive["masks"].shape != (10, 28, 11, 8):
            raise ValueError(f"unexpected source archive shape for seed {seed}")
        terminal = next((row for row in reversed(fit["history"])
                         if int(row["step"]) == 1400), None)
        if terminal is None:
            raise ValueError(f"seed {seed} is missing terminal child objective history")

        seed_rows = []
        for record in records:
            task_index = fit["task_ids"].index(record["task_id"])
            method_index = METHODS.index(record["method"])
            replica_index = int(record["init_id"]) - int(fit["init_offset"])
            if not 0 <= replica_index < fit["replicas"]:
                raise ValueError(f"unexpected final init id in seed {seed}: {record['init_id']}")
            flat_index = (task_index * len(METHODS) + method_index) * fit["replicas"] + replica_index
            saved_mask = fit["masks"][flat_index].numpy().astype(np.uint8)
            record_mask = np.asarray(record["mask"], dtype=np.uint8)
            if not np.array_equal(saved_mask, record_mask):
                raise ValueError(f"record mask differs from frozen child fit: {seed} {record['task_id']} {record['method']}")
            selected_mask = selected["masks"][task_index, method_index].numpy().astype(np.uint8)
            if not np.array_equal(selected_mask, saved_mask):
                raise ValueError(f"selected mask differs from final child mask: {seed} {record['task_id']} {record['method']}")
            if record["split"] == "test":
                if set(record["score_ids"]) & set(record["support_ids"]):
                    raise ValueError(f"test IDs overlap support IDs: seed={seed}, task={record['task_id']}")
                if set(record["score_ids"]) & set(record["query_ids"]):
                    raise ValueError(f"test IDs overlap query IDs: seed={seed}, task={record['task_id']}")
            enriched = dict(record)
            enriched.update({
                "flat_index": flat_index,
                "support_bce_terminal": _history_value(fit["history"], 1400, "support_bce", flat_index),
                "objective_terminal": _history_value(fit["history"], 1400, "objective", flat_index),
                "query_bce_terminal": _history_value(fit["history"], 1400, "query_bce", flat_index),
                "mask_array": saved_mask,
                "weight_array": fit["best_params"]["w"][flat_index].numpy().astype(np.float64),
            })
            seed_rows.append(enriched)
        all_rows.extend(seed_rows)
        seed_data[seed] = {
            "folder": folder,
            "rows": seed_rows,
            "fit": fit,
            "selected": selected,
            "archive": archive,
            "training": training,
            "evaluation": evaluation,
            "terminal_child_plateau": int(fit["converged"].sum()),
            "terminal_child_count": int(fit["converged"].numel()),
        }
    return all_rows, seed_data


def _test_task_means(rows: list[dict[str, Any]], metric: str,
                     method: str | None = None) -> dict[tuple[int, str], float]:
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        if row["split"] != "test" or (method is not None and row["method"] != method):
            continue
        grouped.setdefault((int(row["seed"]), row["task_id"]), []).append(float(row[metric]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _mean_by_method(rows: list[dict[str, Any]], method: str, metric: str) -> list[float]:
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        if row["split"] == "test" and row["method"] == method:
            grouped.setdefault((int(row["seed"]), row["task_id"]), []).append(float(row[metric]))
    return [float(np.mean(values)) for values in grouped.values()]


def _dense_data(root: Path) -> dict[str, Any]:
    selection = _read_json(root / "dense_control_selection.json")
    rows: list[dict[str, Any]] = []
    tuning_by_seed = {}
    for seed in SEEDS:
        folder = root / f"seed_{seed}" / "dense_control"
        rows.extend(_read_json(folder / "records.json"))
        tuning_by_seed[seed] = _read_json(folder / "tuning.json")
    return {"rows": rows, "selection": selection, "tuning_by_seed": tuning_by_seed}


def _test_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["split"] == "test"]


def _structure_audit(all_rows: list[dict[str, Any]],
                     seed_data: dict[int, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    entries = []
    for row in all_rows:
        mask = np.asarray(row["mask_array"], dtype=np.uint8)
        weight = np.asarray(row["weight_array"], dtype=np.float64)
        effective = weight * mask
        alignment = align_mask_to_gold(mask)
        order = np.asarray(alignment["gold_to_candidate"], dtype=np.int64)
        aligned_mask = mask[:, order]
        aligned_effective = effective[:, order]
        weight_audit = signed_weight_audit(weight, mask)
        # The support-only Hungarian order is reused for the actual signed
        # effective weights; weights never participate in the assignment.
        if not np.allclose(aligned_effective, weight_audit["masked_effective_weight"], atol=0, rtol=0):
            raise AssertionError("W_eff alignment must reuse the mask-only Hungarian permutation")
        entries.append({
            "seed": int(row["seed"]), "split": row["split"], "task_id": row["task_id"],
            "method": row["method"], "init_id": int(row["init_id"]),
            "active_edges": int(mask.sum()),
            "gold_to_candidate_order_posthoc": order.tolist(),
            "candidate_to_gold_order_posthoc": alignment["candidate_to_gold"],
            "mask_iou_posthoc": float(alignment["iou"]),
            "mask_precision_posthoc": float(alignment["precision"]),
            "mask_recall_posthoc": float(alignment["recall"]),
            "gold_column_overlap": alignment["gold_column_overlap"],
            "gold_column_exact_match": alignment["gold_column_exact_match"],
            "exact_recover_all_gold_windows_once": bool(alignment["exact_recover_all_windows_once"]),
            "relative_offset_histogram_posthoc": alignment["relative_offset_histogram"],
            "aligned_mask": aligned_mask.tolist(),
            "aligned_signed_effective_weight": aligned_effective.tolist(),
            "signed_effective_weight_l2": float(np.linalg.norm(aligned_effective)),
            "signed_effective_weight_toeplitz_energy_all_offsets": float(
                weight_audit["masked_effective_weight_structure"]["all_offset_toeplitz_explained_energy"]
            ),
            "signed_effective_weight_energy_gold_band_0_to_3": float(
                weight_audit["masked_effective_weight_structure"]["oracle_band_0_to_k_minus_1_explained_energy"]
            ),
            "alignment_policy": "Hungarian assignment uses support overlap only; reporting only",
        })

    test_entries = [entry for entry in entries if entry["split"] == "test"]
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for entry in test_entries:
        grouped.setdefault((entry["seed"], entry["method"], entry["task_id"]), []).append(entry)
    selected_masks = []
    for (seed, method, task_id), group in grouped.items():
        masks = [np.asarray(item["aligned_mask"], dtype=np.uint8) for item in group]
        if any(not np.array_equal(masks[0], mask) for mask in masks[1:]):
            raise ValueError(f"selected mask changed across child initializations: {seed}/{task_id}/{method}")
        selected_masks.append(group[0])
    selected_grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    weight_grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for entry in selected_masks:
        selected_grouped.setdefault((entry["seed"], entry["method"]), []).append(entry)
    for entry in test_entries:
        weight_grouped.setdefault((entry["seed"], entry["method"]), []).append(entry)
    summaries = []
    for (seed, method), group in sorted(selected_grouped.items()):
        unique_masks = {tuple(np.asarray(item["aligned_mask"], dtype=np.uint8).ravel()) for item in group}
        weight_group = weight_grouped[(seed, method)]
        summaries.append({
            "seed": seed, "method": method, "test_contexts": len(group),
            "child_weight_fits": len(weight_group),
            "distinct_test_masks": len(unique_masks),
            "mean_posthoc_iou": float(np.mean([item["mask_iou_posthoc"] for item in group])),
            "exact_gold_window_masks": int(sum(item["exact_recover_all_gold_windows_once"] for item in group)),
            "mean_weff_energy_gold_band_0_to_3": float(np.mean([
                item["signed_effective_weight_energy_gold_band_0_to_3"] for item in weight_group
            ])),
        })
    source_uniform = {}
    for seed, data in seed_data.items():
        masks = data["archive"]["masks"]
        source_uniform[str(seed)] = {
            "source_tasks": int(masks.shape[0]),
            "distinct_uniform_functional_masks": int(len({
                masks[task, 0].numpy().astype(np.uint8).tobytes() for task in range(masks.shape[0])
            })),
            "per_source_task_candidate_counts": [12, 20, 28],
        }
    structure = {
        "scope": "Frozen primary final child masks and first-layer signed W_eff = W * M; validation and test records.",
        "gold_use": "Post-hoc reporting reference only. No gold mask, U, or gold alignment entered candidate generation, child fitting, or checkpoint selection.",
        "alignment": "For each mask, Hungarian assignment maximizes support overlap with the canonical four-edge-window mask. The same gold_to_candidate permutation is applied to M and W_eff.",
        "weff_definition": "W_eff[i,h] = trained signed first-layer weight W[i,h] multiplied by the saved binary mask M[i,h].",
        "records_count": len(entries),
        "test_summary": summaries,
        "source_uniform_mask_note": source_uniform,
        "entries": entries,
    }
    example_rows = [row for row in entries if row["seed"] == 8100 and row["split"] == "test"
                    and row["task_id"] == "k4:0010" and row["init_id"] == 2]
    example = {row["method"]: row for row in example_rows}
    if set(example) != set(METHODS):
        raise ValueError("expected the seven primary methods for the fixed weight example")
    return structure, example


def _curve_data(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"gnn": {}, "fm": {}}
    for seed in SEEDS:
        folder = root / f"seed_{seed}"
        gnn_path = folder / "gnn_stage_2_curves.json"
        checkpoint = torch.load(folder / "fm_frozen.pt", map_location="cpu", weights_only=False)
        fm_stage = int(checkpoint["stage"])
        fm_path = folder / f"fm_stage_{fm_stage}_curves.json"
        result["gnn"][seed] = {"stage": 2, "rows": _read_json(gnn_path)}
        result["fm"][seed] = {"stage": fm_stage, "rows": _read_json(fm_path)}
    return result


def _plot_paired_losses(fig_dir: Path, test_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    method_colors = {
        "support-only / fixed baseline": "#2878B5",
        "query-search": "#E07A24",
        "other": "#737373",
    }
    metric_specs = (
        ("support_bce_terminal", "Support BCE при шаге 1400"),
        ("balanced_bce", "Сбалансированный BCE на test ID"),
    )
    x = np.arange(len(METHODS))
    offsets = (-0.12, 0.12)
    for ax, (metric, title) in zip(axes, metric_specs):
        # Every point in the summary is a task/seed mean over four new child
        # initializations; error bars show descriptive SD across eight units.
        for index, method in enumerate(METHODS):
            by_task: dict[tuple[int, str], list[float]] = {}
            for row in test_rows:
                if row["method"] == method:
                    by_task.setdefault((int(row["seed"]), row["task_id"]), []).append(float(row[metric]))
            means = np.asarray([np.mean(v) for v in by_task.values()], dtype=float)
            mean = float(means.mean())
            sd = float(means.std(ddof=1)) if means.size > 1 else 0.0
            category = _method_group(method)
            marker = "o" if category == "support-only / fixed baseline" else "s"
            ax.errorbar(index + offsets[0 if metric == "support_bce_terminal" else 1], mean,
                        yerr=sd, fmt=marker, color=method_colors[category], capsize=3,
                        markersize=6, lw=1.3, zorder=3)
        ax.set_xticks(x, [LABELS[method] for method in METHODS], rotation=32, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Balanced BCE, меньше — лучше")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=method_colors["support-only / fixed baseline"],
               label="Без task-wise поиска маски", markersize=7),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=method_colors["query-search"],
               label="Query-search-8", markersize=7),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Поддержка и независимый тест: парные task/seed средние", y=1.08, fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / FIGURE_NAMES[0], dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_test_heatmap(fig_dir: Path, test_rows: list[dict[str, Any]], task_keys: list[tuple[int, str]]) -> None:
    matrix = np.zeros((len(task_keys), len(METHODS)), dtype=float)
    for i, task in enumerate(task_keys):
        for j, method in enumerate(METHODS):
            values = [float(row["balanced_bce"]) for row in test_rows
                      if (int(row["seed"]), row["task_id"]) == task and row["method"] == method]
            if len(values) != 4:
                raise ValueError(f"expected four child initializations for {task}/{method}")
            matrix[i, j] = float(np.mean(values))
    fig, ax = plt.subplots(figsize=(12.2, 6.3))
    im = ax.imshow(matrix, aspect="auto", cmap="magma_r", vmin=0.45, vmax=max(.78, float(matrix.max())))
    ax.set_xticks(np.arange(len(METHODS)), [LABELS[name] for name in METHODS], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(task_keys)), [f"{seed} · {task}" for seed, task in task_keys])
    ax.set_xlabel("Метод")
    ax.set_ylabel("Тестовая задача и seed")
    ax.set_title("Test balanced BCE по задачам (среднее четырёх инициализаций)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            rgba = im.cmap(im.norm(matrix[i, j]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "black" if luminance > 0.53 else "white"
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("Balanced BCE (меньше лучше)")
    fig.tight_layout()
    fig.savefig(fig_dir / FIGURE_NAMES[1], dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_masks_weights(fig_dir: Path, example: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(20.5, 6.6))
    grid = fig.add_gridspec(2, len(METHODS) + 1,
                            width_ratios=[1] * len(METHODS) + [.065],
                            height_ratios=[1, 1.12], hspace=.58, wspace=.20,
                            left=.055, right=.98, bottom=.10, top=.80)
    axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(len(METHODS))]
                       for row in range(2)])
    mask_cbar_axis = fig.add_subplot(grid[0, -1])
    weight_cbar_axis = fig.add_subplot(grid[1, -1])
    vmax = max(float(np.abs(np.asarray(row["aligned_signed_effective_weight"])).max())
               for row in example.values())
    mask_image = None
    weight_image = None
    for j, method in enumerate(METHODS):
        row = example[method]
        mask = np.asarray(row["aligned_mask"], dtype=np.uint8)
        weights = np.asarray(row["aligned_signed_effective_weight"], dtype=float)
        ax = axes[0, j]
        mask_image = ax.imshow(mask, cmap="Greys", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        ax.set_title(f"{LABELS[method]}\nK={int(mask.sum())}", fontsize=9)
        ax.set_xticks(range(8), range(8), fontsize=7)
        ax.set_yticks(range(11), range(11), fontsize=7)
        if j == 0:
            ax.set_ylabel("M: вход i")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("скрытый столбец h*", fontsize=8)

        ax = axes[1, j]
        weight_image = ax.imshow(weights, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                                 interpolation="nearest", aspect="auto")
        ax.set_xticks(range(8), range(8), fontsize=7)
        ax.set_yticks(range(11), range(11), fontsize=7)
        if j == 0:
            ax.set_ylabel("W_eff = W × M: вход i")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("скрытый столбец h*", fontsize=8)
    fig.colorbar(mask_image, cax=mask_cbar_axis,
                 label="0 — выключено; 1 — включено")
    fig.colorbar(weight_image, cax=weight_cbar_axis,
                 label="Значение W_eff")
    fig.suptitle("Замороженные маски и знаковые эффективные веса · seed 8100, task 0010, init 2\n"
                 "Столбцы переставлены только после обучения по Hungarian-сопоставлению маски",
                 fontsize=13, y=1.02)
    fig.savefig(fig_dir / FIGURE_NAMES[2], dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_training_curves(fig_dir: Path, curves: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    colors = {8100: "#2878B5", 8102: "#D55E00"}
    for ax, kind, title, objective in (
        (axes[0], "gnn", "GNN: финальная source-стадия 2", "BCE по elite masks"),
        (axes[1], "fm", "FM: последняя continuation/polish стадия", "MSE velocity matching"),
    ):
        for seed in SEEDS:
            item = curves[kind][seed]
            rows = item["rows"]
            steps = [int(row["step"]) for row in rows]
            color = colors[seed]
            ax.plot(steps, [float(row["fixed_monitor_loss"]) for row in rows],
                    color=color, lw=1.7, label=f"{seed}, stage {item['stage']}: fixed monitor")
            ax.plot(steps, [float(row["stochastic_loss"]) for row in rows],
                    color=color, lw=1.0, ls="--", alpha=.7,
                    label=f"{seed}, stage {item['stage']}: stochastic")
        ax.set_title(title)
        ax.set_xlabel("Шаг оптимизатора")
        ax.set_ylabel(f"{objective} (собственная шкала панели)")
        ax.grid(alpha=.22)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Внешние модели: фактические loss-кривые, не качество масок", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / FIGURE_NAMES[3], dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_shared_child(fig_dir: Path, example_rows: list[dict[str, Any]]) -> None:
    metrics = ("support_bce_terminal", "objective_terminal", "query_bce_terminal")
    labels = ("Support BCE", "Support + L2 objective", "Query BCE")
    x = np.arange(len(METHODS))
    width = .24
    fig, ax = plt.subplots(figsize=(13.8, 5.4))
    colors = ("#2878B5", "#6B6B6B", "#E07A24")
    for j, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = [next(float(row[metric]) for row in example_rows if row["method"] == method)
                  for method in METHODS]
        ax.bar(x + (j - 1) * width, values, width, label=label, color=color)
    ax.set_xticks(x, [LABELS[name] for name in METHODS], rotation=28, ha="right")
    ax.set_ylabel("Balanced BCE / регуляризованная цель")
    ax.set_title("Один общий пример: seed 8100, k4:0010, init 2, терминальный шаг 1400")
    ax.grid(axis="y", alpha=.22)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(fig_dir / FIGURE_NAMES[4], dpi=180, bbox_inches="tight")
    plt.close(fig)


def _shared_child_history(seed_data: dict[int, dict[str, Any]]) -> dict[str, Any]:
    fit = seed_data[8100]["fit"]
    task_index = fit["task_ids"].index("k4:0010")
    replica_index = 2 - int(fit["init_offset"])
    if replica_index != 0:
        raise ValueError("shared child-history example must use init 2")
    flat_indices = {
        method: (task_index * len(METHODS) + method_index) * fit["replicas"] + replica_index
        for method_index, method in enumerate(METHODS)
    }
    history = fit["history"]
    return {
        "steps": np.asarray([int(item["step"]) for item in history], dtype=np.int32),
        "support_bce": np.stack([
            np.asarray([float(item["support_bce"][flat_indices[method]]) for item in history])
            for method in METHODS
        ]),
        "objective": np.stack([
            np.asarray([float(item["objective"][flat_indices[method]]) for item in history])
            for method in METHODS
        ]),
        "query_bce": np.stack([
            np.asarray([float(item["query_bce"][flat_indices[method]]) for item in history])
            for method in METHODS
        ]),
        "flat_indices": flat_indices,
        "sampling_interval": "history is recorded every 25 updates through terminal update 1400",
    }


def _plot_shared_child_history(fig_dir: Path, history: dict[str, Any]) -> None:
    colors = plt.get_cmap("tab10").colors
    specs = (
        ("support_bce", "Support balanced BCE"),
        ("objective", "Support BCE + L2 objective"),
        ("query_bce", "Query balanced BCE"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.1), sharex=True)
    for ax, (key, title) in zip(axes, specs):
        for index, method in enumerate(METHODS):
            ax.plot(history["steps"], history[key][index], color=colors[index],
                    lw=1.65, label=LABELS[method])
        ax.set_title(title)
        ax.set_xlabel("Шаги обучения ребёнка")
        ax.grid(alpha=.23)
    axes[0].set_ylabel("Balanced BCE / регуляризованная цель")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Фактические траектории ребёнка · seed 8100, k4:0010, init 2\n"
                 "Точки сохранены каждые 25 обновлений, до фиксированного конца 1400",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / FIGURE_NAMES[6], dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_source_archive(fig_dir: Path, seed_data: dict[int, dict[str, Any]]) -> dict[str, Any]:
    counts = (12, 20, 28)
    x = np.asarray(counts)
    summary: dict[str, Any] = {}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        values = seed_data[seed]["archive"]["query_bce"].numpy().astype(float)
        if values.shape != (10, 28):
            raise ValueError(f"unexpected source utility array for seed {seed}: {values.shape}")
        best_by_stage = np.stack([values[:, :count].min(axis=1) for count in counts], axis=1)
        uniform = values[:, 0]
        for task_i in range(values.shape[0]):
            ax.plot(x, best_by_stage[task_i], color="#B9C5D2", lw=.85, alpha=.8)
        mean_best = best_by_stage.mean(axis=0)
        mean_uniform = float(uniform.mean())
        ax.plot(x, mean_best, color="#1F5A85", marker="o", lw=2.6,
                label="Средний лучший query BCE архива")
        ax.axhline(mean_uniform, color="#C04B26", ls="--", lw=1.8,
                   label="Uniform functional, среднее")
        ax.set_xticks(x)
        ax.set_xlabel("Кандидатов в накопленном архиве")
        ax.set_title(f"Seed {seed}: 10 source-задач")
        ax.grid(alpha=.23)
        ax.legend(frameon=False, fontsize=8)
        summary[str(seed)] = {
            "source_tasks": int(values.shape[0]),
            "candidate_counts": list(counts),
            "mean_best_query_bce_by_stage": mean_best.tolist(),
            "mean_uniform_query_bce": mean_uniform,
            "per_task_best_query_bce_by_stage": best_by_stage.tolist(),
            "per_task_uniform_query_bce": uniform.tolist(),
        }
    axes[0].set_ylabel("Source query balanced BCE (меньше лучше)")
    fig.suptitle("Накопление whole-mask utility для обучения обеих моделей", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / FIGURE_NAMES[5], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary


def _dense_summary(dense: dict[str, Any]) -> dict[str, Any]:
    selection = dense["selection"]
    tuning = selection["scores"]
    test_rows = dense["rows"]
    methods = sorted({row["method"] for row in test_rows})
    test_summary = {}
    for method in methods:
        by_task: dict[tuple[int, str], list[float]] = {}
        by_seed = {}
        for row in test_rows:
            if row["method"] == method:
                by_task.setdefault((int(row["seed"]), row["task_id"]), []).append(float(row["balanced_bce"]))
        values = [float(np.mean(group)) for group in by_task.values()]
        for seed in SEEDS:
            seed_values = [float(np.mean(vals)) for (row_seed, _), vals in by_task.items() if row_seed == seed]
            by_seed[str(seed)] = _describe(seed_values)
        test_summary[method] = {"task_seed_mean": _describe(values), "by_seed": by_seed}
    tuning_seeds = {}
    for seed, data in dense["tuning_by_seed"].items():
        tuning_seeds[str(seed)] = {
            "uses_test": data.get("uses_test"),
            "config_count": len(data["scores"]),
            "meta_validation_scores": data["scores"],
        }
    return {
        "disclosure": selection.get("disclosure"),
        "uses_test_for_selection": selection.get("uses_test_for_selection"),
        "selection": selection["selection"],
        "aggregate_meta_validation_grid": tuning,
        "per_seed_meta_validation_grid": tuning_seeds,
        "test_results_exploratory": test_summary,
        "test_record_count": len(test_rows),
        "selected_test_plateau_rows": {
            method: int(sum(bool(row["converged"]) for row in test_rows if row["method"] == method))
            for method in methods
        },
    }


def _build_arrays(fig_dir: Path, rows: list[dict[str, Any]], test_rows: list[dict[str, Any]],
                  task_keys: list[tuple[int, str]], seed_data: dict[int, dict[str, Any]],
                  dense: dict[str, Any], example: dict[str, Any], curves: dict[str, Any],
                  child_history: dict[str, Any]) -> dict[str, Any]:
    shape = (len(task_keys), len(METHODS), 4)
    support = np.full(shape, np.nan)
    objective = np.full(shape, np.nan)
    train_query = np.full(shape, np.nan)
    test = np.full(shape, np.nan)
    init_ids = np.arange(2, 6, dtype=int)
    for i, task_key in enumerate(task_keys):
        seed, task_id = task_key
        for j, method in enumerate(METHODS):
            for k, init_id in enumerate(init_ids):
                found = [row for row in test_rows if int(row["seed"]) == seed and row["task_id"] == task_id
                         and row["method"] == method and int(row["init_id"]) == init_id]
                if len(found) != 1:
                    raise ValueError(f"missing/duplicate test row for {task_key}/{method}/init{init_id}")
                item = found[0]
                support[i, j, k] = item["support_bce_terminal"]
                objective[i, j, k] = item["objective_terminal"]
                train_query[i, j, k] = item["query_bce_terminal"]
                test[i, j, k] = item["balanced_bce"]

    example_masks = np.stack([np.asarray(example[name]["aligned_mask"], dtype=np.uint8) for name in METHODS])
    example_weff = np.stack([np.asarray(example[name]["aligned_signed_effective_weight"], dtype=np.float64)
                             for name in METHODS])
    arrays: dict[str, Any] = {
        "methods": np.asarray(METHODS, dtype="U32"),
        "task_keys": np.asarray([f"{seed}/{task}" for seed, task in task_keys], dtype="U32"),
        "init_ids": init_ids,
        "support_bce_terminal": support,
        "objective_terminal": objective,
        "training_query_bce_terminal": train_query,
        "heldout_test_balanced_bce": test,
        "heldout_test_task_method_mean": test.mean(axis=2),
        "example_8100_0010_init2_aligned_masks": example_masks,
        "example_8100_0010_init2_aligned_signed_weff": example_weff,
        "example_child_history_steps": child_history["steps"],
        "example_child_history_support_bce": child_history["support_bce"],
        "example_child_history_objective": child_history["objective"],
        "example_child_history_query_bce": child_history["query_bce"],
        "source_query_bce_8100": seed_data[8100]["archive"]["query_bce"].numpy(),
        "source_query_bce_8102": seed_data[8102]["archive"]["query_bce"].numpy(),
        "dense_control_methods": np.asarray(sorted({row["method"] for row in dense["rows"]}), dtype="U32"),
    }
    dense_by_task = {}
    for method in sorted({row["method"] for row in dense["rows"]}):
        values = []
        for seed, task in task_keys:
            values.append([float(np.mean([row["balanced_bce"] for row in dense["rows"]
                                          if int(row["seed"]) == seed and row["task_id"] == task
                                          and row["method"] == method]))])
        dense_by_task[f"dense_test_{method}"] = np.asarray(values, dtype=np.float64)
    arrays.update(dense_by_task)
    for kind in ("gnn", "fm"):
        for seed in SEEDS:
            curve_rows = curves[kind][seed]["rows"]
            arrays[f"{kind}_seed{seed}_steps"] = np.asarray([row["step"] for row in curve_rows], dtype=np.int32)
            arrays[f"{kind}_seed{seed}_fixed_monitor_loss"] = np.asarray(
                [row["fixed_monitor_loss"] for row in curve_rows], dtype=np.float64
            )
            arrays[f"{kind}_seed{seed}_stochastic_loss"] = np.asarray(
                [row["stochastic_loss"] for row in curve_rows], dtype=np.float64
            )
    np.savez_compressed(fig_dir.parent / "plot_arrays.npz", **arrays)
    return arrays


def _write_summary(root: Path, all_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]],
                   seed_data: dict[int, dict[str, Any]], dense: dict[str, Any],
                   dense_summary: dict[str, Any], source_utility: dict[str, Any],
                   structure: dict[str, Any], task_keys: list[tuple[int, str]]) -> dict[str, Any]:
    methods = {}
    for method in METHODS:
        means = _mean_by_method(test_rows, method, "balanced_bce")
        support_means = _mean_by_method(test_rows, method, "support_bce_terminal")
        per_seed = {}
        for seed in SEEDS:
            per_seed[str(seed)] = _describe([
                float(np.mean([row["balanced_bce"] for row in test_rows
                               if row["seed"] == seed and row["method"] == method and row["task_id"] == task]))
                for task in sorted({row["task_id"] for row in test_rows if row["seed"] == seed})
            ])
        methods[method] = {
            "category": _method_group(method),
            "test_balanced_bce_over_task_seed_means": _describe(means),
            "terminal_support_balanced_bce_over_task_seed_means": _describe(support_means),
            "test_balanced_bce_by_seed": per_seed,
        }
    test_method_means = {method: methods[method]["test_balanced_bce_over_task_seed_means"]["mean"]
                         for method in METHODS}
    best_method = min(test_method_means, key=test_method_means.get)
    flow_single = test_method_means["gnn_flow_single"]
    flow_search = test_method_means["gnn_flow_search8"]
    uniform = test_method_means["uniform_functional"]
    task_means = {method: _test_task_means(test_rows, "balanced_bce", method) for method in METHODS}
    flow_vs_uniform_by_task = {}
    for key in sorted(task_means["uniform_functional"]):
        flow_vs_uniform_by_task[f"{key[0]}/{key[1]}"] = {
            "flow_search8": task_means["gnn_flow_search8"][key],
            "uniform_functional": task_means["uniform_functional"][key],
            "difference_flow_minus_uniform": (
                task_means["gnn_flow_search8"][key] - task_means["uniform_functional"][key]
            ),
        }
    training = {}
    for seed in SEEDS:
        item = seed_data[seed]
        training[str(seed)] = {
            "source_tasks": item["training"]["source_tasks"],
            "unique_source_whole_masks_per_task": item["training"]["candidates_per_task"],
            "child_fits": item["training"]["child_fits"],
            "elapsed_seconds": item["training"]["elapsed_seconds"],
            "outer_model_stages": item["training"]["models"],
            "final_children_plateau_flags": {
                "plateau": item["terminal_child_plateau"], "count": item["terminal_child_count"]
            },
            "evaluation": item["evaluation"],
        }
    convergence = {
        "main_source_candidate_fits": {"plateau": 1120, "count": 1120},
        "main_search_selection_fits": {"plateau": 576, "count": 576},
        "main_final_children_including_val_and_test": {"plateau": 336, "count": 336},
        "main_primary_total": {"plateau": 2032, "count": 2032},
        "dense_tuning_grid": {"plateau": 175, "count": 192,
                               "note": "17 capped unused hyperparameter fits; not selected for final test rows"},
        "selected_dense_final_support_and_query_protocols": {"plateau": 32, "count": 32,
                               "note": "The two checkpoint protocols reuse the same 32 selected terminal fits."},
        "pilot": {"plateau": 32, "count": 32},
        "meaning": "Empirical plateau flag under saved finite solver checks; not proof of a global optimum.",
    }
    replay = _read_json(root / "independent_review.json")
    unique_by_seed = {}
    for seed in SEEDS:
        audit = replay["primary_final"][str(seed)]
        task_order = seed_data[seed]["fit"]["task_ids"]
        unique_by_seed[str(seed)] = {}
        for method, counts in audit["unique_masks_per_search_pool"].items():
            unique_by_seed[str(seed)][method] = {
                task: int(count) for task, count in zip(task_order, counts)
                if task in {item for s, item in task_keys if s == seed}
            }
    summary = {
        "title": "Графовые модели и flow matching для целочисленных масок ребер",
        "source_root": str(root),
        "seeds": list(SEEDS),
        "primary_method_order": list(METHODS),
        "test_task_seed_units": len(task_keys),
        "test_patterns": sorted({task for _, task in task_keys}),
        "final_test_rows": len(test_rows),
        "methods": methods,
        "best_primary_mean_method": best_method,
        "flow_search_vs_single_mean_difference": flow_search - flow_single,
        "flow_search_vs_uniform_mean_difference": flow_search - uniform,
        "flow_search_and_uniform_by_test_task_seed": flow_vs_uniform_by_task,
        "search_pool_unique_mask_counts_by_test_task": unique_by_seed,
        "independent_cpu_replay_verdict": replay.get("verdict"),
        "exploratory_dense_controls": dense_summary,
        "primary_protocol": {
            "child_optimizer": "Adam; fixed 1400 terminal updates; lr=0.1; L2 coefficient=0.01",
            "child_selection": "No child checkpoint selection on query; terminal trained weights used.",
            "candidate_selection": "Eight whole-mask candidates; two paired query-fit initializations for selection; four fresh initializations 2..5 for final test scoring.",
            "test_metric": "Balanced BCE on frozen global test IDs, disjoint from support and query IDs.",
            "support_metrics": "Terminal balanced support BCE and regularized objective from the saved step-1400 child history.",
            "source_bank": "960 existing functional solutions per seed; source banks were previously query-curated.",
            "context": "Full 187-feature functional tokens pooled by centroid-aligned hidden columns; teacher-quality channel zeroed before pooling; edge context uses signed/absolute/variance summaries plus occupancy.",
            "training_labels": "Four distinct whole-mask elites per source task, labeled by actual fresh-weight child retraining. GNN BCE and FM velocity MSE use the same weighted elite archive.",
            "fm_endpoint": "Signed binary endpoints +/-1, independent Gaussian coupling; reported sampler uses Euler with 12 steps.",
            "no_ste": True,
            "no_U_or_gold_training": True,
        },
        "convergence_flags": convergence,
        "training_by_seed": training,
        "source_archive_utility": source_utility,
        "posthoc_structure_summary": structure["test_summary"],
        "scope_limits": [
            "Two source banks/seeds and four test patterns per seed; the four test patterns form one orbit under reversal/complement, not four independent orbit families.",
            "No formal significance tests: task-level spread is descriptive and there are only eight task-seed test units.",
            "The functional bank was source-query curated before this experiment; the aligned full-feature prior and uniform functional baseline are strong prior-informed comparisons.",
            "Centroid-based hidden alignment is explicit preprocessing. Network permutation equivariance does not establish invariance of the upstream centroid/alignment procedure.",
            "Dense tuning was added after primary scores had been inspected; treat it as a separate exploratory comparison.",
        ],
    }
    (root / "summary.json").write_text(json.dumps(_plain(summary), ensure_ascii=False, indent=2,
                                                     allow_nan=False) + "\n", encoding="utf-8")
    return summary


def _write_report(root: Path, summary: dict[str, Any], dense: dict[str, Any],
                  dense_summary: dict[str, Any], source_utility: dict[str, Any],
                  structure: dict[str, Any]) -> None:
    means = {method: summary["methods"][method]["test_balanced_bce_over_task_seed_means"]["mean"]
             for method in METHODS}
    best = min(means, key=means.get)
    dense_means = {method: info["task_seed_mean"]["mean"]
                   for method, info in dense_summary["test_results_exploratory"].items()}
    selected = dense_summary["selection"]
    val_scores = dense_summary["aggregate_meta_validation_grid"]
    runtime_text = "; ".join(
        f"seed {seed}: source {summary['training_by_seed'][str(seed)]['elapsed_seconds']:.1f} с, "
        f"candidate evaluation {summary['training_by_seed'][str(seed)]['evaluation']['elapsed_seconds']:.1f} с"
        for seed in SEEDS
    )
    tuning_rows = []
    chosen_pair = (selected["dense_tuned_support"]["lr"], selected["dense_tuned_support"]["l2"])
    for row in val_scores:
        chosen = (row["lr"], row["l2"]) == chosen_pair
        tuning_rows.append(
            f"| {row['lr']:.3g} | {row['l2']:.4g} | {row['terminal_query']:.3f} | "
            f"{row['trajectory_best_query']:.3f} | {'**выбран**' if chosen else ''} |"
        )
    method_table = "\n".join(
        f"| {LABELS[method]} | {_format(means[method])} | "
        f"{_format(summary['methods'][method]['test_balanced_bce_by_seed']['8100']['mean'])} | "
        f"{_format(summary['methods'][method]['test_balanced_bce_by_seed']['8102']['mean'])} | "
        f"{summary['methods'][method]['category']} |"
        for method in METHODS
    )
    dense_table = "\n".join(
        f"| {method} | {_format(dense_means[method])} | "
        f"{_format(dense_summary['test_results_exploratory'][method]['by_seed']['8100']['mean'])} | "
        f"{_format(dense_summary['test_results_exploratory'][method]['by_seed']['8102']['mean'])} |"
        for method in sorted(dense_means)
    )
    source_lines = []
    for seed in SEEDS:
        item = source_utility[str(seed)]
        means_for_seed = item["mean_best_query_bce_by_stage"]
        source_lines.append(
            f"- seed {seed}: средний лучший source-query BCE после 12/20/28 масок "
            f"{', '.join(_format(v) for v in means_for_seed)}; uniform — {_format(item['mean_uniform_query_bce'])}."
        )
    direct_structure = {entry["seed"]: entry for entry in structure["test_summary"]
                        if entry["method"] == "gnn_direct"}
    flow_structure = {entry["seed"]: entry for entry in structure["test_summary"]
                      if entry["method"] == "gnn_flow_search8"}
    uniform_structure = {entry["seed"]: entry for entry in structure["test_summary"]
                         if entry["method"] == "uniform_functional"}
    direct_values = ", ".join(
        f"{seed}: IoU {_format(direct_structure[seed]['mean_posthoc_iou'], 4)}, "
        f"{direct_structure[seed]['exact_gold_window_masks']}/4 точных, "
        f"{direct_structure[seed]['distinct_test_masks']} разных масок"
        for seed in SEEDS
    )
    flow_values = ", ".join(
        f"{seed}: IoU {_format(flow_structure[seed]['mean_posthoc_iou'], 4)}, "
        f"{flow_structure[seed]['exact_gold_window_masks']}/4 точных"
        for seed in SEEDS
    )
    uniform_values = ", ".join(
        f"{seed}: {uniform_structure[seed]['exact_gold_window_masks']}/4 точных контекста, "
        f"{uniform_structure[seed]['distinct_test_masks']} уникальная маска"
        for seed in SEEDS
    )
    flow_deltas = summary["flow_search_and_uniform_by_test_task_seed"]
    per_task_flow = {}
    for key, value in flow_deltas.items():
        task = key.split("/", 1)[1]
        per_task_flow.setdefault(task, []).append(value["difference_flow_minus_uniform"])
    flow_task_text = "; ".join(
        f"{task}: Δ={np.mean(values):+.3f}"
        for task, values in sorted(per_task_flow.items())
    )
    distinct_ranges = {}
    for method in ("gnn_search8", "gnn_flow_search8", "functional_search8"):
        values = [count for seed in SEEDS
                  for count in summary["search_pool_unique_mask_counts_by_test_task"][str(seed)][method].values()]
        distinct_ranges[method] = (min(values), max(values))
    report = f"""# Графовые маски и flow matching: короткое сравнение

## Результат

На восьми парных единицах «seed × тестовая задача» средний test balanced BCE ниже у **{LABELS[best]}** ({means[best]:.3f}); ниже — лучше. При этом преимущество невелико и описательное: двух seed-банков и одной орбиты из четырёх связанных паттернов недостаточно для вывода о переносе на новые орбиты. Ни один GNN/flow вариант не превзошёл среднюю маску `uniform_functional` по среднему test BCE. Flow search-8 улучшил flow single в среднем на {-summary['flow_search_vs_single_mean_difference']:.3f} BCE, но остался выше uniform на {summary['flow_search_vs_uniform_mean_difference']:.3f}. Сравнение масок показывает почти оконную структуру у графовых предложений, однако это не дало преимущества над уже хорошей функциональной средней.

Первичные числа — среднее сначала по четырём независимым финальным инициализациям ребёнка внутри задачи, затем по четырём test-паттернам и двум seed-банкам. Значения по seed приведены отдельно; статистические тесты не заявляются.

| Метод | Test balanced BCE, среднее | Seed 8100 | Seed 8102 | Категория |
|---|---:|---:|---:|---|
{method_table}

![Парные support/test потери](figures/{FIGURE_NAMES[0]})

**Рис. 1.** По оси X — семь первичных методов; по оси Y — balanced BCE. Слева показан support BCE на терминальном шаге обучения ребёнка 1400, справа — BCE на независимых глобальных test ID. Синие круги обозначают предложения без task-wise query-поиска маски, оранжевые квадраты — search-8. Точка — среднее по задаче и четырём инициализациям; усики — описательное стандартное отклонение восьми task-seed средних, не доверительный интервал. Dense — обучаемый ребёнок со всеми 88 рёбрами.

![Test BCE по задачам](figures/{FIGURE_NAMES[1]})

**Рис. 2.** Строки — четыре test-паттерна для каждого из двух seed; столбцы — первичные методы; цвет и число показывают средний balanced BCE по четырём новым начальным весам ребёнка. Обратная шкала magma делает меньшие потери светлее. Эта разбивка показывает разницу между задачами, но четыре паттерна связаны одним симметрийным орбитальным классом.

## Что обучалось и как оценивалось

Для каждого seed использован существующий банк из 960 функциональных решений. Перед графовой моделью скрытые столбцы выровнены по центроидной функциональной сигнатуре. Узловой контекст содержит среднее и стандартное отклонение полного 187-мерного функционального профиля; учительский канал качества обнулён до pooling. Рёберный контекст собирает среднее/разброс `q_signed`, `q_abs`, `q_variance` и частоту включения ребра. Это явный prior от bank-функциональности. Сами банки ранее формировались с использованием source-query качества, поэтому `uniform_functional` — сильный prior-informed baseline, а не нейтральный нулевой ориентир.

Кандидаты — целые бинарные маски с ровно 32 рёбрами. Для utility каждое предложение оценивалось реальным переобучением свежего ребёнка; GNN и FM получили один и тот же архив из четырёх различных лучших целых масок на source-задачу, взвешенный source-query BCE. GNN учился предсказывать маски через BCE; FM учился velocity matching от независимого гауссовского шума к endpoint со значениями $\\{{-1,+1\\}}$. Золотые четырёхрёберные окна используются только в пост-хок структурном аудите; $U$ не участвовала в обучении. Straight-through градиенты не использовались.

FM обучался на прямом независимом гауссовском coupling:

$$
x_t=(1-t)\\epsilon+t z,\\qquad v^*=z-\\epsilon,\\qquad
\\mathcal{{L}}_{{FM}}=\\mathbb{{E}}\\left[\\|v_\\theta(x_t,t,c)-v^*\\|_2^2\\right],
\\quad z\\in\\{{-1,+1\\}}^{{11\\times8}}.
$$

Для методов с search-8 query BCE выбирал кандидата среди восьми слотов по двум парным seed-и детям с init 0 и 1; слоты не гарантируют восемь различных масок из-за точных дубликатов. На test-задачах число уникальных масок составляло {distinct_ranges['gnn_search8'][0]}–{distinct_ranges['gnn_search8'][1]} для GNN, {distinct_ranges['gnn_flow_search8'][0]}–{distinct_ranges['gnn_flow_search8'][1]} для FM и {distinct_ranges['functional_search8'][0]}–{distinct_ranges['functional_search8'][1]} для functional search-8. Выбранную маску заново оценивали четырьмя независимыми весовыми инициализациями init 2–5. В одиночных методах кандидат не выбирался по task query. Финальные дети всех методов оптимизировали одинаково: Adam, $1400$ фиксированных шагов, $\\mathrm{{lr}}=0.1$, $\\lambda_{{L2}}=0.01$, с использованием support. Test BCE вычислялся по отдельным ID; test-метки не участвовали в выборе масок или весовых checkpoint-ов.

На конкретном общем примере различаются support BCE, регуляризованная цель и query BCE при шаге 1400:

![Потери общего дочернего примера](figures/{FIGURE_NAMES[4]})

**Рис. 5.** По оси X — семь методов; по оси Y — значения loss для seed 8100, задачи `k4:0010`, init 2. Синие, серые и оранжевые столбцы — соответственно balanced BCE на support, objective `support BCE + L2`, и balanced BCE на задаче query. Query здесь относится к разрешённому candidate-selection набору, а не к независимым финальным test ID. Значения сняты с сохранённой истории на шаге 1400; по этому query не выбирался финальный весовой checkpoint.

Архив начинался с 12 масок на source-задачу и пополнялся двумя раундами по 8 цельных предложений, всего 28. На графике ниже сравниваются лучшая накопленная query utility и uniform mask на том же source-query наборе:

![Source utility по раундам](figures/{FIGURE_NAMES[5]})

**Рис. 6.** По оси X — накопленные префиксы из 12, 20 и 28 whole-mask кандидатов; по оси Y — source-query balanced BCE, усреднённый по двум child-инициализациям. Тонкие серые линии показывают десять source-задач, синяя — средний минимум среди всех кандидатов в префиксе, пунктир — средний uniform-functional кандидат. Архивы вложены, поэтому лучшая source-query utility не может ухудшаться с добавлением кандидатов. Это training/selection objective для elite labels, не оценка обобщения.

{chr(10).join(source_lines)}

Для того же ребёнка сохранённая история показывает ход оптимизации до фиксированного конца:

![История обучения общего дочернего примера](figures/{FIGURE_NAMES[6]})

**Рис. 7.** Ось X — шаг ребёнка от 25 до 1400; точки истории записывались раз в 25 шагов. Три панели показывают support balanced BCE, сумму support BCE и L2-штрафа и query balanced BCE для seed 8100, `k4:0010`, init 2; цвет обозначает метод. Кривые относятся к одному выбранному ребёнку на метод и описывают его конечный оптимизационный запуск, не распределение по четырём инициализациям и не внешнее обучение GNN/FM. Финальный checkpoint у всех методов взят после 1400 шагов.

## Структура сохранённых масок и веса ребёнка

Визуализация использует ровно сохранённые маски и обученные signed first-layer веса для seed 8100, `k4:0010`, init 2. Показан $W_{{eff}}=W\\odot M$. Чтобы сопоставить столбцы только для анализа, Hungarian-назначение вычислялось по пересечению бинарной маски с каноническими четырёхрёберными окнами; та же перестановка применена к $M$ и $W_{{eff}}$. Gold-маска не строила ни одной кандидатной маски и не входила в обучение. Подробные записи по всем сохранённым финальным детям находятся в `structure.json`.

![Маски и signed эффективные веса](figures/{FIGURE_NAMES[2]})

**Рис. 3.** Две строки по семь панелей: сверху — бинарная маска, снизу — фактический $W_{{eff}}$. Ось Y — входная позиция $i=0\\dots10$, ось X — постфактум сопоставленный скрытый столбец $h^*$. Верхняя шкала: белый — выключено, чёрный — включено; нижняя RdBu_r: красный — положительный вес, синий — отрицательный, белый около нуля; шкала общая для семи методов этого примера. У шестёрки разреженных методов $K=32$, у dense $K=88$. При полной dense-маске Hungarian-перестановка неоднозначна: все назначения имеют одинаковое пересечение, поэтому порядок её столбцов для сравнения весов условен. Пример иллюстрирует сохранённые параметры, но не является выборкой для оценки среднего качества.

В структурном аудите на четырёх test-контекстах восстановлена почти оконная поддержка: GNN direct имеет средний post-hoc IoU {direct_values}; flow search-8 — {flow_values}. Uniform functional даёт {uniform_values}. Для uniform одна bank-маска повторяется по четырём задачам каждого seed: это две независимые bank-маски, а не восемь независимых восстановлений. Оконная структура GNN/flow не даёт выигрыша по child utility относительно функциональной средней.

## Оптимизация внешней модели и результаты дополнительного dense контроля

![Кривые обучения GNN и FM](figures/{FIGURE_NAMES[3]})

**Рис. 4.** По оси X — шаг внешнего оптимизатора; по оси Y — сохранённая stochastic loss (пунктир) и loss на одном фиксированном noise/time мониторинге (сплошная). Цвета различают seed 8100 и 8102. Левый panel — GNN stage 2 с BCE; справа последние FM продолжения: stage 3 у seed 8100 и stage 4 у seed 8102 с velocity-MSE. Масштабы panel-ов различны, значения между GNN и FM напрямую не сравниваются. Плато — эмпирический мониторный критерий, а не гарантия глобальной сходимости.

В основном запуске plateau-флаг отмечен у 1120/1120 source candidate fits, 576/576 search-selection fits и 336/336 финальных детей (включая validation и test). Все четыре замороженные финальные GNN/FM-модели также остановились по plateau-критерию. Промежуточные FM-стадии, достигшие cap, сохранены отдельно: начальные FM-стадии были ограничены 600 шагами; финальное продолжение seed 8102 достигло cap 1200 и было дополнено ограниченным polish на 350 шагах. Флаг сообщает только о выполнении заданной эмпирической проверки на фиксированных конечных данных.

Зафиксированное время: {runtime_text}. Эти значения — длительность конкретных CPU/GPU запусков из run summaries, не оценка алгоритмической сложности.

Дополнительный dense strength control был запущен **после просмотра первичных test результатов** и потому помечен exploratory. Для каждого seed на двух meta-validation паттернах сравнивались 12 сочетаний `lr × L2` (4 learning rate × 3 коэффициента). Общий выбранный конфиг — `lr=0.1`, `L2=0.01`; test не использовался при выборе. В сетке из 12 сочетаний plateau достигнут в 175/192 запусков; 17 конфигурационных запусков достигли ограничения по шагам и не оказались выбранными. Финальные dense fits достигли plateau в 32/32 случаях; оба правила выбора checkpoint используют эти же fits. В следующей таблице приведены усреднённые по двум seed meta-validation terminal-query и trajectory-best-query значения, не test scores:

| lr | L2 | Meta-val terminal query BCE | Meta-val trajectory-best query BCE | |
|---:|---:|---:|---:|---|
{chr(10).join(tuning_rows)}

Отдельная test-оценка двух dense протоколов:

| Dense variant | Test balanced BCE, среднее | Seed 8100 | Seed 8102 |
|---|---:|---:|---:|
{dense_table}

`dense_tuned_support` использует terminal support-trained weights; `dense_tuned_query` выбирает trajectory checkpoint по task query и поэтому не сопоставим с фиксированным primary протоколом выбора весов. Оба значения — дополнительная диагностика dense baseline, не доказательство сильного dense решения: всего два однородных meta-validation паттерна, малое число банков, и короткий fixed-1400 протокол отличается от прежнего long/query-best baseline.

В primary test среднем порядок таков: uniform functional — {means['uniform_functional']:.3f}, GNN direct — {means['gnn_direct']:.3f}, GNN search-8 — {means['gnn_search8']:.3f}, flow search-8 — {means['gnn_flow_search8']:.3f}, functional search-8 — {means['functional_search8']:.3f}, flow single — {means['gnn_flow_single']:.3f}, dense — {means['dense']:.3f}. Flow search-8 ниже flow single, но выше uniform. По task-level среднему flow-search против uniform: {flow_task_text}. Это не подтверждает общее преимущество learned GNN/flow proposal над prior-informed functional baseline.

## Ограничения

- Всего два source seed-банка и четыре test паттерна на seed; тестовые паттерны образуют одну орбиту при обращении и дополнении последовательности. Не измеряется перенос на независимые task-орбиты.
- Не приводятся формальные p-values или доверительные интервалы для обобщения: восемь task-seed средних здесь служат только описанием.
- Functional bank уже query-curated; centroid alignment и функциональные q-признаки — заметный структурный prior. Сравнение не является тестом инвариантности всей предобработки к перестановкам банковских столбцов.
- Dense hyperparameter tuning — отдельная post-hoc exploratory проверка после просмотра первичного результата; её test-показатели следует читать отдельно.
- Успешное воспроизведение окна после post-hoc alignment не означает, что модель обнаружила неизвестную причинную структуру или улучшила результат переобучения ребёнка.

Машиночитаемые данные для построения графиков сохранены в `plot_arrays.npz`; сводка протокола и результатов — в `summary.json`; подробная post-hoc масочная/весовая структура — в `structure.json`.
"""
    (root / "RESULTS_RU.md").write_text(report, encoding="utf-8")


def build(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.titlesize": 13,
        "savefig.facecolor": "white",
    })
    all_rows, seed_data = _load_primary(root)
    test_rows = _test_rows(all_rows)
    task_keys = sorted({(int(row["seed"]), row["task_id"]) for row in test_rows})
    if len(task_keys) != 8:
        raise ValueError(f"expected eight task-seed test units, got {len(task_keys)}")
    dense = _dense_data(root)
    dense_summary = _dense_summary(dense)
    curves = _curve_data(root)
    structure, example = _structure_audit(all_rows, seed_data)
    child_history = _shared_child_history(seed_data)
    example_rows = [row for row in all_rows if row["seed"] == 8100 and row["split"] == "test"
                    and row["task_id"] == "k4:0010" and row["init_id"] == 2]
    source_utility = _plot_source_archive(fig_dir, seed_data)

    _plot_paired_losses(fig_dir, test_rows)
    _plot_test_heatmap(fig_dir, test_rows, task_keys)
    _plot_masks_weights(fig_dir, example)
    _plot_training_curves(fig_dir, curves)
    _plot_shared_child(fig_dir, example_rows)
    _plot_shared_child_history(fig_dir, child_history)
    # Figure 6 was already rendered above while collecting its numeric data.
    structure_path = root / "structure.json"
    structure_path.write_text(json.dumps(_plain(structure), ensure_ascii=False, indent=2,
                                         allow_nan=False) + "\n", encoding="utf-8")
    summary = _write_summary(root, all_rows, test_rows, seed_data, dense, dense_summary,
                             source_utility, structure, task_keys)
    _build_arrays(fig_dir, all_rows, test_rows, task_keys, seed_data, dense,
                  example, curves, child_history)
    _write_report(root, summary, dense, dense_summary, source_utility, structure)

    expected = [fig_dir / name for name in FIGURE_NAMES]
    if any(not path.exists() or path.stat().st_size == 0 for path in expected):
        raise RuntimeError("one or more report figures were not produced")
    report = (root / "RESULTS_RU.md").read_text(encoding="utf-8")
    if report.count("$$") % 2:
        raise RuntimeError("unbalanced display-math delimiters in report")
    for line in report.splitlines():
        if line.startswith("![") and "](figures/" in line:
            name = line.split("](figures/", 1)[1].split(")", 1)[0]
            if not (fig_dir / name).exists():
                raise RuntimeError(f"broken figure link: {name}")
    return {
        "report": str(root / "RESULTS_RU.md"),
        "figures": [str(path) for path in expected],
        "plot_arrays": str(root / "plot_arrays.npz"),
        "summary": str(root / "summary.json"),
        "structure": str(structure_path),
        "primary_means": {
            method: summary["methods"][method]["test_balanced_bce_over_task_seed_means"]["mean"]
            for method in METHODS
        },
        "dense_means": {
            method: item["task_seed_mean"]["mean"]
            for method, item in dense_summary["test_results_exploratory"].items()
        },
    }


if __name__ == "__main__":
    torch.set_num_threads(1)
    result = build()
    print(json.dumps(result, ensure_ascii=False, indent=2))
