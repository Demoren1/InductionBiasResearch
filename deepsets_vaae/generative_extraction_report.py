"""Report the source-only extraction controls for generated functional maps."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/deepsets_vaae/20261001_generative_extraction_controls"
DEFAULT_PRIMARY = ROOT / "outputs/deepsets_vaae/20261001_other_generators_corrected"
SEEDS = tuple(range(4100, 4108))
TASKS = tuple(range(8))
REPLICAS = tuple(range(4))
K, F, H = 7526, 784, 32
CONTROLS = ("functional_mean_small", "functional_mean_large", "functional_vae_small",
            "functional_vae_large", "raw_vae_large", "random", "dense")
EXTRA = ("functional_realign_mean", "functional_realign_logit_mean", "functional_pixel_marginal",
         "gnn_sample_agreement", "empirical_sample_agreement")
METHODS = CONTROLS + EXTRA
T975_DF7 = 2.364624251


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=175, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _ci(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (8,) or not np.isfinite(array).all():
        raise ValueError("need eight finite paired seed values")
    mean = float(array.mean())
    sem = float(array.std(ddof=1) / math.sqrt(8))
    return {"n": 8, "df": 7, "mean": mean, "sem": sem,
            "ci95_low": mean - T975_DF7 * sem, "ci95_high": mean + T975_DF7 * sem,
            "seed_values": array.tolist()}


def _validate(out: Path) -> dict[str, Any]:
    if not (out / "COMPLETE").is_file():
        raise ValueError(f"Control root is incomplete: {out}")
    protocol = _json(out / "protocol.json")
    expected = {"seeds": list(SEEDS), "methods": list(METHODS), "new_methods": list(EXTRA)}
    bad = {k: (protocol.get(k), v) for k, v in expected.items() if protocol.get(k) != v}
    if bad:
        raise ValueError(f"protocol mismatch: {bad}")
    return protocol


def _records(out: Path) -> dict[str, dict[int, list[dict[str, Any]]]]:
    data = {"old": {}, "fresh": {}}
    expected = {(task, method, replica) for task in TASKS for method in METHODS for replica in REPLICAS}
    for seed in SEEDS:
        result = _json(out / f"seed_{seed}" / "results.json")
        if int(result.get("seed", -1)) != seed:
            raise ValueError(f"seed {seed} result id mismatch")
        for population, field in (("old", "records"), ("fresh", "fresh_records")):
            rows = result.get(field)
            if not isinstance(rows, list):
                raise ValueError(f"seed {seed}/{population}: no rows")
            found = set()
            for row in rows:
                key = (int(row.get("task", -1)), str(row.get("method", "")), int(row.get("init", -1)))
                if int(row.get("support_size", -1)) != 256 or key in found or key not in expected:
                    raise ValueError(f"seed {seed}/{population}: invalid row {key}")
                if not math.isfinite(float(row.get("mse", float("nan")))):
                    raise ValueError(f"seed {seed}/{population}: nonfinite MSE")
                found.add(key)
            if found != expected:
                raise ValueError(f"seed {seed}/{population}: {len(found)} cells, expected {len(expected)}")
            data[population][seed] = rows
    return data


def _summary(rows: dict[str, dict[int, list[dict[str, Any]]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for population in ("old", "fresh"):
        per_method = {}
        for method in METHODS:
            seed_values = []
            for seed in SEEDS:
                values = [float(row["mse"]) for row in rows[population][seed] if row["method"] == method]
                if len(values) != len(TASKS) * len(REPLICAS):
                    raise ValueError(f"missing {population}/{seed}/{method} values")
                seed_values.append(float(np.mean(values)))
            per_method[method] = _ci(seed_values)
        contrasts = {}
        for method in EXTRA:
            for baseline in ("functional_mean_large", "functional_vae_large", "random", "dense"):
                contrasts[f"{method}_minus_{baseline}"] = _ci((np.asarray(per_method[method]["seed_values"])
                                                                 - np.asarray(per_method[baseline]["seed_values"])).tolist())
        result[population] = {"methods": per_method, "contrasts": contrasts}
    return result


def _add_primary_gnn(summary: dict[str, Any], primary: Path) -> None:
    source = _json(primary / "summary.json")
    for population in ("old", "fresh"):
        values = source["target"][population]["256"]["gnn_flow"]["seed_values"]
        control = summary[population]["methods"]["gnn_sample_agreement"]["seed_values"]
        summary[population]["contrasts"]["gnn_sample_agreement_minus_primary_gnn_flow"] = _ci(
            (np.asarray(control) - np.asarray(values)).tolist())
        summary[population]["primary_gnn_flow"] = _ci(list(values))


def _plot_targets(out: Path, summary: dict[str, Any]) -> None:
    display = ("functional_mean_large", "functional_vae_large", "random", "dense") + EXTRA
    colors = {"functional_mean_large": "#2f7f5f", "functional_vae_large": "#d55e00",
              "random": "#777777", "dense": "#222222", "functional_realign_mean": "#4c78a8",
              "functional_realign_logit_mean": "#59a14f", "functional_pixel_marginal": "#e15759",
              "gnn_sample_agreement": "#9467bd", "empirical_sample_agreement": "#f28e2b"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    x = np.arange(len(display))
    for ax, population in zip(axes, ("old", "fresh")):
        cells = [summary[population]["methods"][method] for method in display]
        mean = [cell["mean"] for cell in cells]
        error = [[cell["mean"] - cell["ci95_low"] for cell in cells],
                 [cell["ci95_high"] - cell["mean"] for cell in cells]]
        ax.bar(x, mean, color=[colors[m] for m in display], yerr=error, capsize=2)
        ax.set_xticks(x, display, rotation=35, ha="right", fontsize=7)
        ax.set_title(f"{population}: budget 256, paired seed means")
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("NMSE = MSE / 5 (меньше лучше)")
    _save(fig, out, "extraction_target_methods")

    comparisons = [(method, "functional_mean_large") for method in EXTRA]
    comparisons += [("gnn_sample_agreement", "primary_gnn_flow")]
    fig, ax = plt.subplots(figsize=(12, 5))
    cells = []
    for method, baseline in comparisons:
        key = f"{method}_minus_{baseline}" if baseline != "primary_gnn_flow" else "gnn_sample_agreement_minus_primary_gnn_flow"
        cells.append((f"{method}\n− {baseline}", summary["fresh"]["contrasts"][key]))
    x = np.arange(len(cells))
    means = [cell[1]["mean"] for cell in cells]
    errors = [[cell[1]["mean"] - cell[1]["ci95_low"] for cell in cells],
              [cell[1]["ci95_high"] - cell[1]["mean"] for cell in cells]]
    ax.bar(x, means, yerr=errors, capsize=3, color="#4c78a8")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(x, [cell[0] for cell in cells], rotation=22, ha="right")
    ax.set_ylabel("paired Δ NMSE, fresh (ниже нуля лучше first method)")
    ax.set_title("Extraction controls: paired contrasts, 95% t-CI, df=7")
    ax.grid(axis="y", alpha=.2)
    _save(fig, out, "extraction_target_contrasts_fresh")


def _method_index(checkpoint: dict[str, Any], method: str) -> int:
    items = checkpoint.get("methods", [])
    matched = [int(row["model_index"]) for row in items if row.get("method") == method and int(row.get("init", -1)) == 0]
    if len(matched) != 1:
        raise ValueError(f"checkpoint lacks unique {method}/init0")
    return matched[0]


def _heatmap(out: Path) -> None:
    path = out / "seed_4100" / "fresh_weights" / "target_task0_budget256.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing fixed heatmap checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    weight, masks = torch.as_tensor(payload.get("weight")), torch.as_tensor(payload.get("masks"))
    if weight.ndim != 3 or tuple(weight.shape[1:]) != (F, H) or masks.shape != weight.shape:
        raise ValueError("unexpected target checkpoint tensors")
    methods = EXTRA + ("dense",)
    values = []
    for method in methods:
        index = _method_index(payload, method)
        values.append((weight[index] * masks[index]).numpy())
    scale = float(np.abs(np.stack(values)).max())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid W*M common scale")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), constrained_layout=True)
    for ax, method, value in zip(axes.flat, methods, values):
        image = ax.imshow(value.T, aspect="auto", cmap="coolwarm", vmin=-scale, vmax=scale, interpolation="nearest")
        ax.set_title(method, fontsize=9); ax.set_xlabel("pixel (784)"); ax.set_ylabel("hidden (32)")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.72, label="trained W × binary mask; shared signed scale")
    _save(fig, out, "extraction_fresh_task0_budget256_effective_weights")
    np.savez_compressed(out / "extraction_heatmap_arrays.npz", methods=np.asarray(methods),
                        effective_weights=np.stack(values), common_abs_scale=np.asarray(scale))


def _report(out: Path, summary: dict[str, Any]) -> None:
    table = []
    for method in EXTRA:
        value = summary["fresh"]["methods"][method]
        delta = summary["fresh"]["contrasts"][f"{method}_minus_functional_mean_large"]
        table.append(f"| {method} | {value['mean']:.4f} [{value['ci95_low']:.4f}; {value['ci95_high']:.4f}] | {delta['mean']:+.4f} [{delta['ci95_low']:+.4f}; {delta['ci95_high']:+.4f}] |")
    text = [
        "# Source-only extraction controls for generated functional maps", "",
        "This follow-up isolates extraction from source functional maps. It uses the immutable 4×205 aligned source-train maps, exact top-7526 output masks, and does not access target labels while constructing masks. Target evaluation is the existing paired 800-update protocol at budget 256. Both target populations remain exploratory.", "",
        "`functional_realign_mean` and `functional_realign_logit_mean` aggregate 820 maps after a fresh source-only Hungarian match to the original pooled functional-train mean. `functional_pixel_marginal` keeps only a pixel prior. `gnn_sample_agreement` applies the same discrete multi-task agreement operator to the saved 4×32 GNN samples; `empirical_sample_agreement` applies it to matched empirical source samples. These are extraction controls, not retrained generators.", "",
        "## Fresh target results", "", "| Method | NMSE [95% t-CI] | Δ vs functional_mean_large [95% t-CI] |", "|---|---:|---:|", *table, "",
        "A negative Δ is better for the first method. Intervals are paired across the eight seeds (df=7), conditional on fixed, previously viewed task vectors and without multiplicity correction.", "",
        "## Figures", "", "- [All extraction methods](extraction_target_methods.png): paired seed means and 95% t-CI for old/fresh populations.", "- [Fresh paired contrasts](extraction_target_contrasts_fresh.png): extractions versus direct functional mean, plus saved GNN-sample agreement versus the primary GNN sample-mean mask.", "- [Fixed W×M heatmap](extraction_fresh_task0_budget256_effective_weights.png): seed 4100, fresh task 0, budget 256, init 0; the signed scale is shared across the five extraction controls and dense.", "",
        "[summary.json](summary.json) and [extraction_figure_data.npz](extraction_figure_data.npz) preserve numerical values. The source-only construction, hashes and replay audit are in [protocol.json](protocol.json), [source_controls.json](seed_4100/source_controls.json) and per-seed artifacts.", "",
        "The bank is itself fixed at 20% source sparsity and every extraction uses 30% target top-K; this control cannot select an optimal density. Variable density requires a source bank/evaluation spanning multiple K, with source-only or nested held-out selection.",
    ]
    (out / "EXTRACTION_REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    (out / "EXTRACTION_CAPTIONS.md").write_text("""# Captions

`extraction_target_methods` compares target performance after changing only a source-only extraction score. `extraction_target_contrasts_fresh` uses seed-paired differences; zero is no difference. The W×M figure displays fitted target weights after validation checkpoint selection and shares its signed scale; it is not a source score heatmap. Neither figure establishes a causal architectural mechanism or generalization outside the eight fixed target tasks.
""", encoding="utf-8")


def build(out: Path, primary: Path) -> dict[str, Any]:
    out, primary = Path(out), Path(primary)
    protocol = _validate(out)
    rows = _records(out)
    summary = _summary(rows)
    _add_primary_gnn(summary, primary)
    _plot_targets(out, summary)
    _heatmap(out)
    arrays = {}
    for population in ("old", "fresh"):
        for method in METHODS:
            arrays[f"{population}_{method}_seedvalues"] = np.asarray(summary[population]["methods"][method]["seed_values"])
    for population in ("old", "fresh"):
        for key, cell in summary[population]["contrasts"].items():
            arrays[f"contrast_{population}_{key}"] = np.asarray(cell["seed_values"])
    np.savez_compressed(out / "extraction_figure_data.npz", **arrays)
    result = {"status": "PASS", "protocol": protocol, "target": summary}
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _report(out, summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    args = parser.parse_args()
    result = build(args.out, args.primary)
    print(json.dumps({"status": result["status"], "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
