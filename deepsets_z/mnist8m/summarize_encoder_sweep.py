"""Summarize the seed-42 image-encoder sweep against the existing controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .encoder_sweep import ARM, DEFAULT_ENCODERS


LENGTHS = (1, 3, 5, 10, 20)
CONFIG_KEYS = ("arm", "seed", "steps", "inner_steps", "lr_v1", "lr_readout",
               "outer_lr", "tasks_per_step", "support", "query", "val_tasks",
               "test_tasks", "train_images_per_digit", "eval_images_per_digit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(
        "deepsets_z/mnist8m/outputs/meta_u_first_encoder_sweep"))
    parser.add_argument("--encoders", default=",".join(DEFAULT_ENCODERS),
                        help="Comma-separated names to summarize alongside mlp64")
    parser.add_argument("--baseline", type=Path, default=Path(
        "deepsets_z/mnist8m/meta_u_first_image_code_results/"
        "generated_ortho_seed42.json"))
    parser.add_argument("--static-root", type=Path, default=Path(
        "deepsets_z/mnist8m/meta_u_first_results"))
    args = parser.parse_args()
    names = tuple(name.strip() for name in args.encoders.split(",") if name.strip())
    if len(set(names)) != len(names) or "mlp64" in names:
        parser.error("Specify distinct new encoders; mlp64 is loaded as baseline")
    runs = {"mlp64": json.loads(args.baseline.read_text())}
    reference = runs["mlp64"]["config"]
    for name in names:
        path = args.results / f"{ARM}_{name}_seed42.json"
        run = json.loads(path.read_text())
        if ("test" not in run or "test_no_image_code" not in run
                or run["history"][-1]["step"] != reference["steps"]):
            raise ValueError(f"Incomplete run: {path}")
        for key in CONFIG_KEYS:
            if run["config"][key] != reference[key]:
                raise ValueError(f"Mismatched {key}: {path}")
        if run["config"]["encoder"] != name:
            raise ValueError(f"Wrong encoder: {path}")
        runs[name] = run
    static = {arm: json.loads((args.static_root /
                               f"{arm}_seed42.json").read_text())
              for arm in ("dense", "learned")}
    summary = {
        "seed": 42,
        "tasks_per_test_length": reference["test_tasks"],
        "metric": "mean absolute error on held-out shifted-query tasks; lower is better",
        "mlp64_source": str(args.baseline),
        "by_encoder": {},
        "controls": {arm: {str(n): static[arm]["test"][str(n)]["mae"]
                           for n in LENGTHS} for arm in static},
    }
    for name, run in runs.items():
        summary["by_encoder"][name] = {
            "encoder_parameters": run["model"]["image_code_parameters"],
            "total_parameters": run["model"]["trainable_parameters"],
            "best_step": run["best_step"],
            "mae": {str(n): run["test"][str(n)]["mae"] for n in LENGTHS},
            "code_disabled_mae": {
                str(n): run["test_no_image_code"][str(n)]["mae"]
                for n in LENGTHS},
        }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "encoder_sweep_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    fig = plt.figure(figsize=(13.5, 9), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.85))
    ax_capacity = fig.add_subplot(grid[0, 0])
    ax_lengths = fig.add_subplot(grid[0, 1])
    ax_steps = fig.add_subplot(grid[1, :])
    mlp_names = sorted((name for name in runs if name.startswith("mlp")),
                       key=lambda name: summary["by_encoder"][name][
                           "encoder_parameters"])
    conv_names = sorted((name for name in runs if name.startswith("conv")),
                        key=lambda name: summary["by_encoder"][name][
                            "encoder_parameters"])
    colors = {name: plt.cm.Blues(0.35 + 0.6 * index /
              max(1, len(mlp_names) - 1)) for index, name in enumerate(mlp_names)}
    colors.update({name: plt.cm.Greens(0.55 + 0.35 * index /
                   max(1, len(conv_names) - 1))
                   for index, name in enumerate(conv_names)})
    for name in runs:
        row = summary["by_encoder"][name]
        color = colors[name]
        marker = "s" if name.startswith("conv") else "o"
        ax_capacity.scatter(row["encoder_parameters"], row["mae"]["5"],
                            color=color, s=65, marker=marker, zorder=3)
        ax_capacity.annotate(name, (row["encoder_parameters"], row["mae"]["5"]),
                             xytext=(4, 5), textcoords="offset points", fontsize=8)
        ax_lengths.plot(LENGTHS, [row["mae"][str(n)] for n in LENGTHS],
                        marker=marker, markersize=4, linewidth=1.7,
                        color=color, label=name)
        history = runs[name]["history"]
        ax_steps.plot([point["step"] for point in history],
                      [point["validation_score"] for point in history],
                      color=color, linewidth=1.6, label=name)
    for arm, label, color in (("dense", "Dense", "#9c5caa"),
                              ("learned", "Direct U", "#cf594a")):
        ax_capacity.axhline(summary["controls"][arm]["5"], color=color,
                            linestyle="--", linewidth=1.5, label=label)
        ax_lengths.plot(LENGTHS, [summary["controls"][arm][str(n)]
                                  for n in LENGTHS], color=color,
                        linestyle="--", linewidth=1.5, label=label)
    ax_capacity.set_xscale("log")
    ax_capacity.set_xlabel("Trainable image-encoder parameters")
    ax_capacity.set_ylabel("Test MAE, set length 5 ↓")
    ax_capacity.set_title("A. Encoder size versus test quality")
    ax_capacity.grid(alpha=0.25)
    ax_capacity.legend(frameon=False)
    ax_lengths.set_xticks(LENGTHS)
    ax_lengths.set_xlabel("Query-set length")
    ax_lengths.set_ylabel("Test MAE ↓")
    ax_lengths.set_title("B. Generalization to longer sets")
    ax_lengths.grid(alpha=0.25)
    ax_lengths.legend(frameon=False, fontsize=8, ncol=2)
    ax_steps.set_xlabel("Outer optimization step")
    ax_steps.set_ylabel("Normalized validation MSE ↓")
    ax_steps.set_title("C. Meta-training convergence")
    ax_steps.set_yscale("log")
    ax_steps.grid(alpha=0.25)
    ax_steps.legend(frameon=False, fontsize=8, ncol=4)
    fig.suptitle("MNIST8m · spatial orthogonal U generator · image-encoder sweep · seed 42",
                 fontsize=14)
    figure_path = args.results / "encoder_sweep_summary.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(f"Figure: {figure_path}")
    print("Encoder             parameters   MAE length 5   MAE length 20")
    for name, row in sorted(summary["by_encoder"].items(),
                            key=lambda item: item[1]["mae"]["5"]):
        print(f"{name:18s} {row['encoder_parameters']:>10,d} "
              f"{row['mae']['5']:>13.4f} {row['mae']['20']:>15.4f}")
    print(f"Dense control: {summary['controls']['dense']['5']:.4f}; "
          f"direct U: {summary['controls']['learned']['5']:.4f}")


if __name__ == "__main__":
    main()
