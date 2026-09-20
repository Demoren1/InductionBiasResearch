"""Summarize a fixed-lambda sweep of v-expert counts, including old controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .meta_u_first import RANK


LENGTHS = (1, 3, 5, 10, 20)
REFERENCE_COUNTS = (10, 32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(
        "deepsets_z/mnist8m/outputs/meta_u_first_v_moe_counts"))
    parser.add_argument("--previous", type=Path, default=Path(
        "deepsets_z/mnist8m/meta_u_first_v_moe_results"))
    parser.add_argument("--conv32", type=Path, default=Path(
        "deepsets_z/mnist8m/meta_u_first_encoder_results/"
        "generated_ortho_conv32_seed42.json"))
    parser.add_argument("--static-root", type=Path, default=Path(
        "deepsets_z/mnist8m/meta_u_first_results"))
    args = parser.parse_args()
    runs = {}
    for count in REFERENCE_COUNTS:
        path = args.previous / f"moe_k{count}_ortho0p2_seed42.json"
        runs[count] = json.loads(path.read_text())
    for path in args.results.glob("moe_k*_ortho0p2_seed42.json"):
        run = json.loads(path.read_text())
        count = run["config"]["experts"]
        if count in runs:
            raise ValueError(f"Repeated expert count: {path}")
        runs[count] = run
    if len(runs) <= len(REFERENCE_COUNTS):
        raise ValueError("No completed new counts found")
    reference = runs[32]["config"]
    for count, run in runs.items():
        if (run["config"]["ortho_weight"] != 0.2
                or run["config"]["seed"] != 42
                or run["history"][-1]["step"] != 5000
                or "test" not in run or "test_shuffled_route" not in run):
            raise ValueError(f"Incomplete or mismatched K={count}")
        for key in ("steps", "inner_steps", "lr_coeff", "lr_readout",
                    "outer_lr", "tasks_per_step", "support", "query",
                    "test_tasks", "train_images_per_digit",
                    "eval_images_per_digit"):
            if run["config"][key] != reference[key]:
                raise ValueError(f"Mismatched {key}: K={count}")
    conv32 = json.loads(args.conv32.read_text())
    static = {arm: json.loads((args.static_root /
                               f"{arm}_seed42.json").read_text())
              for arm in ("dense", "learned")}
    counts = sorted(runs)
    summary = {
        "seed": 42, "orthogonality_weight": 0.2,
        "expert_vector_dimension": RANK,
        "test_tasks_per_length": reference["test_tasks"],
        "by_count": {},
        "references": {
            "continuous_conv32": {str(n): conv32["test"][str(n)]["mae"]
                                  for n in LENGTHS},
            **{arm: {str(n): data["test"][str(n)]["mae"]
                     for n in LENGTHS} for arm, data in static.items()}},
    }
    for count, run in runs.items():
        summary["by_count"][str(count)] = {
            "source": ("previous_sweep" if count in REFERENCE_COUNTS
                       else "count_sweep"),
            "trainable_parameters": run["model"]["trainable_parameters"],
            "adapted_parameters": run["model"]["adapted_parameters"],
            "best_step": run["best_step"],
            "best_validation_score": run["best_validation_score"],
            "mae": {str(n): run["test"][str(n)]["mae"] for n in LENGTHS},
            "shuffled_route_mae": {
                str(n): run["test_shuffled_route"][str(n)]["mae"]
                for n in LENGTHS},
            "expert_orthogonality": run["expert_orthogonality"],
            "welch_lower_bound": max(
                0.0, (count - RANK) / (RANK * (count - 1))),
            "routing_entropy": run["routing"]["normalized_routing_entropy"],
            "active_experts_gt_1pct": sum(
                usage > 0.01 for usage in run["routing"]["mean_expert_usage"]),
        }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "v_moe_count_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    fig = plt.figure(figsize=(13.2, 8.2), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    ax5 = fig.add_subplot(grid[0, 0])
    ax20 = fig.add_subplot(grid[0, 1])
    ax_lengths = fig.add_subplot(grid[1, 0])
    ax_ortho = fig.add_subplot(grid[1, 1])
    for ax, length, title in ((ax5, 5, "A. Test MAE, length 5"),
                              (ax20, 20, "B. Test MAE, length 20")):
        values = [runs[k]["test"][str(length)]["mae"] for k in counts]
        ax.plot(counts, values, color="#147f6f", marker="o", linewidth=2.2,
                label="MoE v, λ=0.2")
        for name, label, color in (
                ("continuous_conv32", "Continuous conv32 gate", "#717783"),
                ("dense", "Dense first layer", "#8d53a4"),
                ("learned", "Direct U", "#c95447")):
            ax.axhline(summary["references"][name][str(length)],
                       color=color, linestyle="--", linewidth=1.4,
                       label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(counts)
        ax.set_xticklabels([str(k) for k in counts])
        ax.set_xlabel("Number of v experts")
        ax.set_ylabel("Mean absolute error ↓")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    selected_count = min(counts, key=lambda k: runs[k]["best_validation_score"])
    for count, color, label in ((10, "#86b6a8", "K=10"),
                                (32, "#48a795", "K=32"),
                                (selected_count, "#0c6659",
                                 f"Validation choice K={selected_count}")):
        ax_lengths.plot(LENGTHS,
                        [runs[count]["test"][str(n)]["mae"] for n in LENGTHS],
                        color=color, marker="o", linewidth=2, label=label)
    for name, label, color in (
            ("continuous_conv32", "Continuous conv32 gate", "#717783"),
            ("dense", "Dense first layer", "#8d53a4"),
            ("learned", "Direct U", "#c95447")):
        ax_lengths.plot(LENGTHS,
                        [summary["references"][name][str(n)] for n in LENGTHS],
                        color=color, linestyle="--", linewidth=1.5,
                        label=label)
    ax_lengths.set_xticks(LENGTHS)
    ax_lengths.set_xlabel("Query-set length")
    ax_lengths.set_ylabel("Test MAE ↓")
    ax_lengths.set_title("C. Quality across set lengths")
    ax_lengths.grid(alpha=0.25)
    ax_lengths.legend(frameon=False, fontsize=8)

    ax_ortho.plot(counts,
                  [runs[k]["expert_orthogonality"] for k in counts],
                  color="#d08634", marker="o", label="Observed penalty")
    ax_ortho.plot(counts,
                  [summary["by_count"][str(k)]["welch_lower_bound"]
                   for k in counts], color="#364c91", linestyle="--",
                  label="Minimum possible in 32 dimensions")
    ax_ortho.set_xscale("log", base=2)
    ax_ortho.set_xticks(counts)
    ax_ortho.set_xticklabels([str(k) for k in counts])
    ax_ortho.set_xlabel("Number of v experts")
    ax_ortho.set_ylabel("Mean squared cosine")
    ax_ortho.set_title("D. Orthogonality above 32 experts")
    ax_ortho.grid(alpha=0.25)
    ax_ortho.legend(frameon=False, fontsize=8)
    fig.suptitle("MNIST8m · v-expert count sweep · λ=0.2 · seed 42",
                 fontsize=14)
    figure_path = args.results / "v_moe_count_summary.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(f"Figure: {figure_path}")
    print("K   params   adapted   val   MAE5   MAE20   shuffled5   orth   active")
    for count in counts:
        row = summary["by_count"][str(count)]
        print(f"{count:<3} {row['trainable_parameters']:<8} "
              f"{sum(row['adapted_parameters']):<9} "
              f"{row['best_validation_score']:.4f} "
              f"{row['mae']['5']:.4f} {row['mae']['20']:.4f} "
              f"{row['shuffled_route_mae']['5']:.4f} "
              f"{row['expert_orthogonality']:.4f} "
              f"{row['active_experts_gt_1pct']}")


if __name__ == "__main__":
    main()
