"""Evidence-preserving report generator for the adaptive-density experiment.

It never trains or selects models.  It reads frozen worker artifacts and emits
figures together with the arrays used to draw them.  Before phase B exists the
output is explicitly a source-selection preliminary report.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import adaptive_data as data
from .adaptive_orchestration import confirmation_ready, validate_stage


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/deepsets_vaae/20261001_adaptive_density"
SEEDS = tuple(range(4100, 4108))
FAMILIES = ("functional", "gnn", "pixelprior")
FINAL_LABELS = ("functional_selected", "gnn_selected", "pixelprior_selected", "joint_primary",
                "random_matched_joint", "dense_tuned", "dense_default", "functional_fixed30",
                "gnn_fixed30", "pixelprior_fixed30")
COLORS = {"functional": "#1b9e77", "gnn": "#7570b3", "pixelprior": "#d95f02", "random": "#666666", "dense": "#111111"}


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _save(fig: plt.Figure, directory: Path, name: str) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        target = directory / f"{name}.{suffix}"
        fig.savefig(target, dpi=180, bbox_inches="tight")
        paths.append(str(target.name))
    plt.close(fig)
    return paths


def _complete(root: Path, phase: str) -> bool:
    return all((root / f"seed_{seed}" / f"{phase.upper()}_COMPLETE").is_file() for seed in SEEDS)


def _records(root: Path, phase: str) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        path = root / f"seed_{seed}" / f"{phase}_records.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in _json(path):
            item = dict(row); item["seed"] = seed; rows.append(item)
    return rows


def _seed_means(rows: list[dict[str, Any]], budget: int, method: str) -> np.ndarray:
    values = []
    for seed in SEEDS:
        take = [float(row["score_mse"]) for row in rows if row["seed"] == seed and int(row["support_size"]) == budget and row["method"] == method]
        if not take:
            raise ValueError(f"missing {method}, b={budget}, seed={seed}")
        values.append(np.mean(take))
    return np.asarray(values, dtype=float)


def _mean(rows: list[dict[str, Any]], budget: int, method: str) -> float:
    return float(_seed_means(rows, budget, method).mean())


def _paired_primary(rows: list[dict[str, Any]], budget: int = 256, *, draws: int = 2000) -> dict[str, Any]:
    a, b = f"joint_primary_b{budget}", f"dense_tuned_b{budget}"
    raw = np.empty((len(SEEDS), 32, 4), dtype=float)
    dense = np.empty_like(raw)
    for si, seed in enumerate(SEEDS):
        lookup = {(int(r["task"]), int(r["init"]), r["method"]): float(r["score_mse"])
                  for r in rows if r["seed"] == seed and int(r["support_size"]) == budget}
        for task in range(32):
            for init in range(4):
                raw[si, task, init] = lookup[(task, init, a)] - lookup[(task, init, b)]
                dense[si, task, init] = lookup[(task, init, b)]
    seed_difference = raw.mean(axis=(1, 2))
    center = float(seed_difference.mean())
    se = float(seed_difference.std(ddof=1) / np.sqrt(8))
    rng = np.random.default_rng(20261007)
    boots = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled_seeds = rng.integers(0, 8, 8)
        sampled_tasks = rng.integers(0, 32, 32)
        boots[draw] = raw[sampled_seeds][:, sampled_tasks, :].mean()
    percent = -100 * center / float(dense.mean())
    return {"comparison": "joint_primary_b256 minus dense_tuned_b256", "difference_nmse": center,
            "difference_t95_ci_seed8": [center - 2.365 * se, center + 2.365 * se],
            "relative_improvement_percent": percent, "seed_differences_nmse": seed_difference.tolist(),
            "two_way_seed_task_bootstrap_rng": 20261007, "two_way_bootstrap_draws": draws,
            "two_way_bootstrap_95_ci": np.quantile(boots, [.025, .975]).tolist(),
            "raw_difference_shape": list(raw.shape)}


def _selection_plot(root: Path, rows: list[dict[str, Any]], selected: dict[str, Any], plots: Path, numeric: Path) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    output: dict[str, Any] = {}
    for ax, budget in zip(axes.flat, data.BUDGETS):
        entry: dict[str, Any] = {"rho": list(data.SPARSE_RHOS)}
        for family in FAMILIES:
            vals, lrs = [], []
            for rho in data.SPARSE_RHOS:
                scores = [(lr, _mean(rows, budget, f"{family}_rho{rho:g}_lr{lr:g}")) for lr in data.LRS]
                lr, score = min(scores, key=lambda x: x[1]); vals.append(score); lrs.append(lr)
            ax.plot(data.SPARSE_RHOS, vals, marker="o", color=COLORS[family], label=family)
            entry[family] = {"mean_nmse_min_over_lr": vals, "selected_lr_per_rho": lrs}
        random_vals = []
        for rho in data.SPARSE_RHOS:
            random_vals.append(min(_mean(rows, budget, f"random_rho{rho:g}_lr{lr:g}") for lr in data.LRS))
        dense = [(lr, _mean(rows, budget, f"dense_lr{lr:g}")) for lr in data.LRS]
        dense_lr, dense_score = min(dense, key=lambda x: x[1])
        ax.plot(data.SPARSE_RHOS, random_vals, marker="o", ls="--", color=COLORS["random"], label="random")
        ax.axhline(dense_score, color=COLORS["dense"], ls=":", label=f"dense, lr={dense_lr:g}")
        ax.set_xscale("log"); ax.set_title(f"support budget {budget}"); ax.set_xlabel("retained density $\\rho$"); ax.set_ylabel("selection score NMSE")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
        entry["random"] = {"mean_nmse_min_over_lr": random_vals}; entry["dense"] = {"mean_nmse_by_lr": dict(dense), "best_lr": dense_lr, "best_nmse": dense_score}
        entry["frozen_config"] = selected.get("by_budget", {}).get(str(budget), {})
        output[str(budget)] = entry
    fig.suptitle("Stage A: source selection score; LR minimized within each curve (descriptive)")
    paths = _save(fig, plots, "selection_density_curves")
    np.savez_compressed(numeric / "selection_density_curves.npz", payload=np.array([output], dtype=object))
    return {"paths": paths, "values": output}


def _confirmation_plot(rows: list[dict[str, Any]], plots: Path, numeric: Path) -> dict[str, Any]:
    mean = np.empty((len(FINAL_LABELS), len(data.BUDGETS)), dtype=float)
    sd = np.empty_like(mean)
    for mi, label in enumerate(FINAL_LABELS):
        for bi, budget in enumerate(data.BUDGETS):
            v = _seed_means(rows, budget, f"{label}_b{budget}"); mean[mi, bi] = v.mean(); sd[mi, bi] = v.std(ddof=1)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for mi, label in enumerate(FINAL_LABELS):
        ax.errorbar(data.BUDGETS, mean[mi], yerr=sd[mi], marker="o", capsize=2.5, label=label)
    ax.set_xscale("log", base=2); ax.set_xticks(data.BUDGETS, labels=[str(x) for x in data.BUDGETS]); ax.set_xlabel("support budget"); ax.set_ylabel("confirmation test NMSE")
    ax.set_title("Stage B: mean across 8 seed means; bars are seed SD")
    ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2)
    paths = _save(fig, plots, "confirmation_methods_by_budget")
    np.savez_compressed(numeric / "confirmation_methods_by_budget.npz", methods=np.asarray(FINAL_LABELS), budgets=np.asarray(data.BUDGETS), mean_nmse=mean, seed_sd=sd)
    return {"paths": paths, "methods": list(FINAL_LABELS), "budgets": list(data.BUDGETS), "mean_nmse": mean.tolist(), "seed_sd": sd.tolist()}


def _primary_plot(primary: dict[str, Any], plots: Path, numeric: Path) -> dict[str, Any]:
    values = np.asarray(primary["seed_differences_nmse"], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.axhline(0, color="black", lw=.8); ax.scatter(np.arange(1, 9), values, color="#1b9e77")
    ax.errorbar(9, primary["difference_nmse"], yerr=[[primary["difference_nmse"] - primary["difference_t95_ci_seed8"][0]], [primary["difference_t95_ci_seed8"][1] - primary["difference_nmse"]]], fmt="o", color="#d95f02", capsize=4, label="mean ± t95")
    ax.set_xticks(list(range(1, 9)) + [9], labels=[str(x) for x in SEEDS] + ["mean"]); ax.set_ylabel("joint − dense NMSE"); ax.set_title("Predeclared primary comparison, budget 256")
    ax.legend(); ax.grid(alpha=.25)
    paths = _save(fig, plots, "primary_joint_vs_dense")
    np.savez_compressed(numeric / "primary_joint_vs_dense.npz", seed_differences=values, t95_ci=np.asarray(primary["difference_t95_ci_seed8"]), bootstrap_ci=np.asarray(primary["two_way_bootstrap_95_ci"]))
    return {"paths": paths, **primary}


def _load_curves(root: Path, seed: int, phase: str) -> dict[str, Any]:
    path = root / f"seed_{seed}" / f"{phase}_curves.npz"
    with np.load(path, allow_pickle=True) as arrays:
        return arrays["curves"].item()


def _source_curve_plot(root: Path, selected: dict[str, Any], plots: Path, numeric: Path) -> dict[str, Any] | None:
    path = root / "seed_4100" / "selection_curves.npz"
    if not path.is_file() or not selected.get("by_budget"):
        return None
    curves = _load_curves(root, 4100, "selection")
    config = selected["by_budget"]["256"]
    names = []
    manifest = _json(root / "seed_4100" / "selection_provenance.json")["mask_manifest"]
    for row in manifest: names.extend([row["method"]] * 4)
    chosen, dense = config["joint_primary"]["method"], config["dense_tuned"]["method"]
    for key, payload in curves.items():
        group = ast.literal_eval(key)
        if (0, 256) not in group: continue
        condition = group.index((0, 256)); steps = np.asarray(payload["steps"]); train = np.asarray(payload["train"])[:, condition]; checkpoint = np.asarray(payload["checkpoint"])[:, condition]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
        stored: dict[str, Any] = {"steps": steps}
        for method, color in ((chosen, "#1b9e77"), (dense, "#111111")):
            idx = [i for i, name in enumerate(names) if name == method]
            for ax, values, title in zip(axes, (train[:, idx], checkpoint[:, idx]), ("stochastic train", "checkpoint validation")):
                mu = values.mean(1); lo, hi = np.quantile(values, [.1, .9], axis=1)
                ax.plot(steps, mu, color=color, label=method); ax.fill_between(steps, lo, hi, color=color, alpha=.18); ax.set_title(title); ax.set_xlabel("update"); ax.set_ylabel("NMSE"); ax.grid(alpha=.25)
            stored[method] = {"train": train[:, idx], "checkpoint": checkpoint[:, idx]}
        for ax in axes: ax.legend(fontsize=8)
        paths = _save(fig, plots, "source_curve_seed4100_task0_budget256")
        np.savez_compressed(numeric / "source_curve_seed4100_task0_budget256.npz", **stored)
        return {"paths": paths, "seed": 4100, "task": 0, "budget": 256, "methods": [chosen, dense], "band": "10th–90th percentile over four paired initializations"}
    raise ValueError("source curve has no fixed seed/task/budget condition")


def _heatmaps(root: Path, plots: Path, numeric: Path) -> dict[str, Any] | None:
    path = root / "seed_4100" / "confirmation_masks_states.pt"
    provenance = root / "seed_4100" / "confirmation_provenance.json"
    if not path.is_file() or not provenance.is_file(): return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    manifest = [row for row in _json(provenance)["mask_manifest"] if int(row["budget"]) == 256]
    names = [row["method"] for row in manifest]
    state = None; condition = None
    for key, candidate in payload["checkpoints"].items():
        group = ast.literal_eval(key)
        if (0, 256) in group: state, condition = candidate, group.index((0, 256)); break
    if state is None: raise ValueError("no confirmation state for seed4100 task0 budget256")
    effective = state["effective_weight"][condition].numpy(); masks = state["masks"][condition].numpy()
    matrices = np.stack([effective[i * 4] for i in range(10)]); binary = np.stack([masks[i * 4] for i in range(10)])
    scale = float(np.abs(matrices).max())
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharex=True, sharey=True)
    for ax, name, matrix in zip(axes.flat, names, matrices):
        image = ax.imshow(matrix.T, aspect="auto", cmap="coolwarm", vmin=-scale, vmax=scale, interpolation="nearest")
        ax.set_title(name.replace("_b256", ""), fontsize=8); ax.set_xlabel("input feature"); ax.set_ylabel("hidden unit")
    fig.colorbar(image, ax=axes, shrink=.7, label="effective weight $W\\odot M$")
    paths1 = _save(fig, plots, "confirmation_effective_weight_heatmaps")
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharex=True, sharey=True)
    for ax, name, matrix in zip(axes.flat, names, binary):
        image = ax.imshow(matrix.T, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(name.replace("_b256", ""), fontsize=8); ax.set_xlabel("input feature"); ax.set_ylabel("hidden unit")
    fig.colorbar(image, ax=axes, shrink=.7, label="mask: 0 forbidden, 1 retained")
    paths2 = _save(fig, plots, "confirmation_mask_density_heatmaps")
    np.savez_compressed(numeric / "confirmation_seed4100_task0_budget256_heatmaps.npz", methods=np.asarray(names), effective_weight=matrices, masks=binary, signed_scale=scale)
    return {"effective_weight_paths": paths1, "mask_paths": paths2, "seed": 4100, "task": 0, "budget": 256, "init": 0, "common_signed_scale": scale}


def _coverage(root: Path, phase: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"phase": phase, "records": len(rows), "expected_records": (28416 if phase == "selection" else 5120) * 8,
              "converged": sum(bool(x["converged"]) for x in rows), "max_cap": sum(not bool(x["converged"]) for x in rows), "states": {}}
    for seed in SEEDS:
        state = root / f"seed_{seed}" / f"{phase}_masks_states.pt"; curves = root / f"seed_{seed}" / f"{phase}_curves.npz"
        result["states"][str(seed)] = {"state_exists": state.is_file(), "curves_exists": curves.is_file(),
                                       "state_sha256": _sha(state) if state.is_file() else None, "curves_sha256": _sha(curves) if curves.is_file() else None}
    return result


def _input_guard(root: Path) -> dict[str, Any]:
    protocol = _json(root / "protocol.json")
    checked = {rel: (ROOT / rel).is_file() and _sha(ROOT / rel) == expected for rel, expected in protocol.get("input_sha256", {}).items()}
    return {"all_unchanged": all(checked.values()), "files": checked, "protocol_sha256": _sha(root / "protocol.json")}


def _confirmation_config_binding(root: Path, selected: dict[str, Any]) -> dict[str, Any]:
    """Check the saved phase-B manifest still agrees with frozen selection."""
    if not selected:
        return {"passed": False, "reason": "selected_density.json missing"}
    for seed in SEEDS:
        provenance = _json(root / f"seed_{seed}" / "confirmation_provenance.json")
        rows = {row["method"]: row for row in provenance.get("mask_manifest", [])}
        for budget in data.BUDGETS:
            for label in FINAL_LABELS:
                method = f"{label}_b{budget}"
                saved, frozen = rows.get(method), selected["by_budget"][str(budget)][label]
                if saved is None or float(saved["rho"]) != float(frozen["rho"]) or float(saved["lr"]) != float(frozen["lr"]):
                    return {"passed": False, "reason": f"confirmation mask manifest differs: seed={seed}, method={method}"}
    return {"passed": True, "selected_density_sha256": _sha(root / "selected_density.json")}


def _final_validation(root: Path, selection_done: bool, confirmation_done: bool, selected: dict[str, Any]) -> dict[str, Any]:
    """Run independent complete-artifact checks before any final claim."""
    if not selection_done or not confirmation_done:
        return {"passed": False, "reason": "completion sentinels absent"}
    try:
        selection = [validate_stage(root, seed, "selection") for seed in SEEDS]
        confirmation = [validate_stage(root, seed, "confirmation") for seed in SEEDS]
        source_gate = confirmation_ready(root)
        binding = _confirmation_config_binding(root, selected)
    except Exception as error:
        return {"passed": False, "reason": f"artifact validation failed: {type(error).__name__}: {error}"}
    passed = bool(source_gate and binding["passed"])
    return {"passed": passed, "source_gate_and_selected_binding": bool(source_gate), "confirmation_config_binding": binding,
            "selection_validated_artifacts": selection, "confirmation_validated_artifacts": confirmation,
            "reason": "PASS" if passed else "source gate or frozen selected-density binding failed"}


def _audit_file_stats(root: Path) -> dict[str, dict[str, int]]:
    result = {}
    for seed in SEEDS:
        for phase in ("selection", "confirmation"):
            for suffix in ("records.json", "provenance.json", "masks_states.pt", "curves.npz"):
                path = root / f"seed_{seed}" / f"{phase}_{suffix}"
                stat = path.stat()
                result[str(path.relative_to(root))] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    for name in ("protocol.json", "selected_density.json"):
        stat = (root / name).stat(); result[name] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def build(root: Path) -> dict[str, Any]:
    plots, numeric = root / "report_plots", root / "report_numeric"; plots.mkdir(exist_ok=True); numeric.mkdir(exist_ok=True)
    selection_done, confirmation_done = _complete(root, "selection"), _complete(root, "confirmation")
    selected = _json(root / "selected_density.json") if (root / "selected_density.json").is_file() else {}
    validation = _final_validation(root, selection_done, confirmation_done, selected)
    status = ("FINAL_CONFIRMATION" if validation["passed"] else
              "BLOCKED_FINAL_VALIDATION" if confirmation_done else "PRELIMINARY_SOURCE_SELECTION_ONLY")
    summary: dict[str, Any] = {"status": status, "input_guard": _input_guard(root), "selection_complete": selection_done, "confirmation_complete": confirmation_done,
                               "final_validation": validation, "single_primary": "joint_primary_b256 versus dense_tuned_b256; t95 across 8 seed means", "secondary": "all other curves/tables are exploratory"}
    if selection_done:
        selection = _records(root, "selection"); summary["selection_coverage"] = _coverage(root, "selection", selection)
        summary["selection_density_plot"] = _selection_plot(root, selection, selected, plots, numeric)
        summary["source_curve"] = _source_curve_plot(root, selected, plots, numeric)
    if validation["passed"]:
        confirmation = _records(root, "confirmation"); summary["confirmation_coverage"] = _coverage(root, "confirmation", confirmation)
        summary["confirmation_method_plot"] = _confirmation_plot(confirmation, plots, numeric)
        summary["primary"] = _primary_plot(_paired_primary(confirmation), plots, numeric)
        summary["heatmaps"] = _heatmaps(root, plots, numeric)
    _dump(root / "summary.json", summary)
    if validation["passed"]:
        # This compact manifest is independent of worker COMPLETE sentinels.
        # Reuse is valid only while all hash-listed artifacts still match.
        _dump(root / "validated_final_manifest.json", {"validation": validation, "protocol_sha256": _sha(root / "protocol.json"),
                                                         "selected_density_sha256": _sha(root / "selected_density.json"),
                                                         "stat_binding": _audit_file_stats(root)})
    appendix = _appendix(summary)
    (root / "REPORT27_ADAPTIVE_APPENDIX.md").write_text(appendix, encoding="utf-8")
    (root / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _appendix(summary: dict[str, Any]) -> str:
    if summary["status"] != "FINAL_CONFIRMATION":
        detail = ("Фаза B с независимым confirmation test ещё не завершена." if summary["status"] == "PRELIMINARY_SOURCE_SELECTION_ONLY"
                  else f"Фаза B завершила worker-sentinels, но независимая итоговая валидация заблокирована: {summary['final_validation']['reason']}.")
        return f"## Adaptive density: предварительный статус\n\n{detail} Источниковый подбор плотности и его графики сохранены, но вывод о превосходстве над dense не делается.\n"
    p = summary["primary"]
    return ("## Adaptive density: независимая проверка\n\n"
            f"Единственное заранее заданное primary-сравнение — `joint_primary` против `dense_tuned` при бюджете 256. "
            f"Разность NMSE (joint − dense) равна {p['difference_nmse']:.6f}, точечный t95 интервал по восьми seed means "
            f"[{p['difference_t95_ci_seed8'][0]:.6f}, {p['difference_t95_ci_seed8'][1]:.6f}]. "
            "Остальные методы и плотности имеют статус exploratory.\n")


def _report(summary: dict[str, Any]) -> str:
    lines = ["# Adaptive selection of connection density", "", f"Status: **{summary['status']}**.", "",
             "All figure arrays are in `report_numeric/`; every plot is emitted as PNG and PDF. Mathematical notation uses Markdown KaTeX delimiters.", "",
             "## Protocol", "", "Stage A chooses a global configuration per budget from source-only selection scores. A sparse method can enter Stage B only when it strictly improves the equally tuned dense mean; density resolves only exact sparse ties. Stage B is independent confirmation.", ""]
    if summary["selection_complete"]:
        lines += ["## Stage A density curves", "", "![Selection density curves](report_plots/selection_density_curves.png)", "",
                  "Each point minimizes over the three searched learning rates, so these curves are descriptive of the hyperparameter search. The horizontal dense line uses the same LR grid.", ""]
        frozen = summary["selection_density_plot"]["values"]
        lines += ["### Globally frozen choices", "", "| Budget | Functional | GNN | Pixel prior | Joint | Dense |", "|---:|---|---|---|---|---|"]
        for budget in data.BUDGETS:
            config = frozen[str(budget)].get("frozen_config", {})
            def cell(label: str) -> str:
                row = config.get(label, {})
                return "—" if not row else f"$\\rho={row['rho']:g}$, lr={row['lr']:g}"
            lines.append(f"| {budget} | {cell('functional_selected')} | {cell('gnn_selected')} | {cell('pixelprior_selected')} | {cell('joint_primary')} | {cell('dense_tuned')} |")
        lines += [""]
    if summary["status"] == "FINAL_CONFIRMATION":
        p = summary["primary"]
        lines += ["## Independent Stage B", "", "![Confirmation methods](report_plots/confirmation_methods_by_budget.png)", "",
                  "Bars are SD across eight seed means. This is a secondary exploratory overview.", "",
                  "![Primary paired effect](report_plots/primary_joint_vs_dense.png)", "",
                  f"The predeclared primary effect at budget 256 is $\\Delta={p['difference_nmse']:.6f}$ NMSE (joint minus dense), with seed-level t95 CI $[{p['difference_t95_ci_seed8'][0]:.6f}, {p['difference_t95_ci_seed8'][1]:.6f}]$. A negative value favours joint_primary.", "",
                  "![Actual effective weights](report_plots/confirmation_effective_weight_heatmaps.png)", "",
                  "Each panel is the saved $W\\odot M$ of seed 4100, task 0, budget 256, initialization 0. All panels share one signed colour scale.", "",
                  "![Actual masks](report_plots/confirmation_mask_density_heatmaps.png)", "",
                  "White cells are forbidden connections and black cells are retained connections for the exact same saved states.", ""]
    else:
        lines += ["## Interpretation", "", f"Final confirmation is unavailable: {summary['final_validation']['reason']}. This report deliberately makes no dense-improvement claim.", ""]
    return "\n".join(lines)


def smoke() -> dict[str, Any]:
    # A decisive synthetic paired effect exercises task/seed bootstrap and the
    # t interval without using a scientific artifact.
    rows = []
    for seed in SEEDS:
        for task in range(32):
            for init in range(4):
                base = {"seed": seed, "task": task, "init": init, "support_size": 256}
                rows += [{**base, "method": "joint_primary_b256", "score_mse": .8}, {**base, "method": "dense_tuned_b256", "score_mse": 1.0}]
    value = _paired_primary(rows, draws=100)
    checks = {"negative_effect": value["difference_nmse"] < 0, "ci_negative": value["difference_t95_ci_seed8"][1] < 0,
              "bootstrap_negative": value["two_way_bootstrap_95_ci"][1] < 0}
    return {**checks, "status": "PASS" if all(checks.values()) else "FAIL"}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, default=DEFAULT_OUT); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke: print(json.dumps(smoke(), sort_keys=True)); return
    print(json.dumps(build(args.out), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
