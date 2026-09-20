"""Summarize MoE-v runs, orthogonality ablations and routing by digit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .v_moe_sweep import SPECS, stem


LENGTHS = (1, 3, 5, 10, 20)
COLORS = {0.0: "#225eaa", 0.05: "#12886f", 0.2: "#c2792d"}
CONFIG_KEYS = ("seed", "steps", "inner_steps", "lr_coeff", "lr_readout",
               "outer_lr", "tasks_per_step", "support", "query", "test_tasks",
               "train_images_per_digit", "eval_images_per_digit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(
        "deepsets_z/mnist8m/outputs/meta_u_first_v_moe"))
    parser.add_argument("--conv32", type=Path, default=Path(
        "deepsets_z/mnist8m/meta_u_first_encoder_results/"
        "generated_ortho_conv32_seed42.json"))
    parser.add_argument("--static-root", type=Path, default=Path(
        "deepsets_z/mnist8m/meta_u_first_results"))
    args = parser.parse_args()
    runs = {}
    for spec in SPECS:
        path = args.results / f"{stem(*spec, 42)}.json"
        run = json.loads(path.read_text())
        if ("test" not in run or "test_uniform_route" not in run
                or "test_shuffled_route" not in run
                or "routing" not in run or run["history"][-1]["step"] != 5000):
            raise ValueError(f"Incomplete run: {path}")
        if (run["config"]["experts"] != spec[0]
                or run["config"]["ortho_weight"] != spec[1]):
            raise ValueError(f"Wrong specification: {path}")
        runs[spec] = run
    reference = runs[SPECS[0]]["config"]
    for spec, run in runs.items():
        for key in CONFIG_KEYS:
            if run["config"][key] != reference[key]:
                raise ValueError(f"Mismatched {key}: {spec}")
    conv32 = json.loads(args.conv32.read_text())
    for key in ("seed", "steps", "inner_steps", "lr_readout", "outer_lr",
                "tasks_per_step", "support", "query", "test_tasks",
                "train_images_per_digit", "eval_images_per_digit"):
        if conv32["config"][key] != reference[key]:
            raise ValueError(f"Mismatched conv32 {key}")
    if conv32["config"]["lr_v1"] != reference["lr_coeff"]:
        raise ValueError("Mismatched inner learning rate")
    static = {name: json.loads((args.static_root /
                                f"{name}_seed42.json").read_text())
              for name in ("dense", "learned")}
    summary = {
        "seed": 42, "test_tasks_per_length": reference["test_tasks"],
        "metric": "mean absolute error on shifted held-out tasks; lower is better",
        "references": {
            "conv32_gate": {str(n): conv32["test"][str(n)]["mae"]
                            for n in LENGTHS},
            **{name: {str(n): run["test"][str(n)]["mae"]
                      for n in LENGTHS} for name, run in static.items()}},
        "runs": {},
    }
    for (k, weight), run in runs.items():
        summary["runs"][f"k{k}_lambda{weight:g}"] = {
            "experts": k, "orthogonality_weight": weight,
            "trainable_parameters": run["model"]["trainable_parameters"],
            "adapted_parameters": run["model"]["adapted_parameters"],
            "best_step": run["best_step"],
            "test_mae": {str(n): run["test"][str(n)]["mae"]
                         for n in LENGTHS},
            "uniform_route_mae": {
                str(n): run["test_uniform_route"][str(n)]["mae"]
                for n in LENGTHS},
            "shuffled_route_mae": {
                str(n): run["test_shuffled_route"][str(n)]["mae"]
                for n in LENGTHS},
            "expert_orthogonality": run["expert_orthogonality"],
            "routing_entropy": run["routing"]["normalized_routing_entropy"],
            "expert_usage": run["routing"]["mean_expert_usage"],
        }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "v_moe_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), layout="constrained")
    ax_quality, ax_lengths, ax_routes0, ax_routes_ortho = axes.flat
    for weight, color in COLORS.items():
        specs = [spec for spec in SPECS if spec[1] == weight]
        xs = [spec[0] for spec in specs]
        ys = [runs[spec]["test"]["5"]["mae"] for spec in specs]
        ax_quality.plot(xs, ys, color=color, marker="o",
                        linewidth=2, label=f"orthogonality λ={weight:g}")
    for name, label, color in (
            ("conv32_gate", "Continuous gate, conv32", "#585f6b"),
            ("dense", "Dense first layer", "#8d53a4"),
            ("learned", "Direct U", "#cc5145")):
        ax_quality.axhline(summary["references"][name]["5"],
                           color=color, linestyle="--", label=label)
    ax_quality.set_xticks([1, 10, 32])
    ax_quality.set_xlabel("Number of v experts")
    ax_quality.set_ylabel("Test MAE, set length 5 ↓")
    ax_quality.set_title("A. Quality versus experts and orthogonality")
    ax_quality.grid(alpha=0.25)
    ax_quality.legend(frameon=False, fontsize=8)

    for spec in SPECS:
        k, weight = spec
        row = runs[spec]["test"]
        label = f"K={k}, λ={weight:g}"
        ax_lengths.plot(LENGTHS, [row[str(n)]["mae"] for n in LENGTHS],
                        color=COLORS[weight], linestyle=("-" if k == 10 else
                        "--" if k == 32 else ":"), marker="o",
                        linewidth=1.6, markersize=3.5, label=label)
    ax_lengths.plot(LENGTHS,
                    [conv32["test"][str(n)]["mae"] for n in LENGTHS],
                    color="#585f6b", linewidth=2, label="Continuous gate")
    ax_lengths.set_xticks(LENGTHS)
    ax_lengths.set_xlabel("Query-set length")
    ax_lengths.set_ylabel("Test MAE ↓")
    ax_lengths.set_title("B. Generalization to longer sets")
    ax_lengths.grid(alpha=0.25)
    ax_lengths.legend(frameon=False, fontsize=7, ncol=2)

    for ax, spec, title in (
            (ax_routes0, (10, 0.0), "C. Routing by digit · K=10, λ=0"),
            (ax_routes_ortho, (10, 0.05),
             "D. Routing by digit · K=10, λ=0.05")):
        matrix = np.array(runs[spec]["routing"]["mean_route_by_true_digit"])
        image = ax.imshow(matrix, vmin=0, vmax=max(0.25, matrix.max()),
                          cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Expert index")
        ax.set_ylabel("True digit (diagnostic only)")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        fig.colorbar(image, ax=ax, shrink=0.8, label="Mean routing probability")
    fig.suptitle("MNIST8m · image-routed v experts · seed 42", fontsize=14)
    figure_path = args.results / "v_moe_summary.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.7), layout="constrained")
    for spec, run in runs.items():
        k, weight = spec
        history = run["history"]
        ax.plot([point["step"] for point in history],
                [point["validation_score"] for point in history],
                color=COLORS[weight],
                linestyle=("-" if k == 10 else "--" if k == 32 else ":"),
                linewidth=1.6, label=f"K={k}, λ={weight:g}")
    ax.plot([point["step"] for point in conv32["history"]],
            [point["validation_score"] for point in conv32["history"]],
            color="#585f6b", linewidth=2, label="Continuous gate")
    ax.set_xlabel("Outer optimization step")
    ax.set_ylabel("Normalized validation MSE ↓")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(args.results / "v_moe_convergence.png", dpi=180)
    plt.close(fig)
    print(f"Figure: {figure_path}")
    print("Run                 MAE5   shuffled5   MAE20   orth_loss  route_entropy")
    for spec in SPECS:
        k, weight = spec
        run = runs[spec]
        print(f"K={k:<2} λ={weight:<4g} "
              f"{run['test']['5']['mae']:.4f} "
              f"{run['test_shuffled_route']['5']['mae']:.4f} "
              f"{run['test']['20']['mae']:.4f} "
              f"{run['expert_orthogonality']:.4f} "
              f"{run['routing']['normalized_routing_entropy']:.4f}")
    print(f"References MAE5: conv32={conv32['test']['5']['mae']:.4f}, "
          f"dense={static['dense']['test']['5']['mae']:.4f}, "
          f"direct U={static['learned']['test']['5']['mae']:.4f}")


if __name__ == "__main__":
    main()
