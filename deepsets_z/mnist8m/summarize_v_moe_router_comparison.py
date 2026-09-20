"""Compare parameter-matched convolutional and MLP image routers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COUNTS = (48, 96, 128)
LENGTHS = (1, 3, 5, 10, 20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(
        "deepsets_z/mnist8m/outputs/meta_u_first_v_moe_router_mlp"))
    parser.add_argument("--conv", type=Path, default=Path(
        "deepsets_z/mnist8m/meta_u_first_v_moe_count_results"))
    parser.add_argument("--conv-long", type=Path,
                        help="K=96 conv JSON trained for 10000 steps")
    parser.add_argument("--mlp-long", type=Path,
                        help="K=96 MLP JSON trained for 10000 steps")
    args = parser.parse_args()
    if (args.conv_long is None) != (args.mlp_long is None):
        parser.error("Provide both --conv-long and --mlp-long")
    runs = {}
    summary = {"seed": 42, "orthogonality_weight": 0.2,
               "test_tasks_per_length": 64, "by_count": {}}
    for count in COUNTS:
        stem = f"moe_k{count}_ortho0p2_seed42"
        pair = {
            "conv": json.loads((args.conv / f"{stem}.json").read_text()),
            "mlp": json.loads((args.results / f"{stem}_mlp.json").read_text()),
        }
        reference = pair["conv"]["config"]
        for name, run in pair.items():
            config = run["config"]
            if (config.get("router", "conv") != name
                    or config["experts"] != count
                    or config["ortho_weight"] != 0.2
                    or config["seed"] != 42
                    or run["history"][-1]["step"] != 5000
                    or "test" not in run or "test_shuffled_route" not in run):
                raise ValueError(f"Incomplete or mismatched {name} K={count}")
            for key in ("steps", "inner_steps", "lr_coeff", "lr_readout",
                        "outer_lr", "tasks_per_step", "support", "query",
                        "test_tasks", "train_images_per_digit",
                        "eval_images_per_digit"):
                if config[key] != reference[key]:
                    raise ValueError(f"Mismatched {key}: {name} K={count}")
        conv_params = pair["conv"]["model"]["router_parameters"]
        mlp_params = pair["mlp"]["model"]["router_parameters"]
        if abs(mlp_params - conv_params) / conv_params > 0.005:
            raise ValueError(f"Router parameter mismatch: K={count}")
        runs[count] = pair
        row = {}
        for name, run in pair.items():
            row[name] = {
                "router_parameters": run["model"]["router_parameters"],
                "trainable_parameters": run["model"]["trainable_parameters"],
                "adapted_parameters": run["model"]["adapted_parameters"],
                "best_validation_score": run["best_validation_score"],
                "best_step": run["best_step"],
                "mae": {str(n): run["test"][str(n)]["mae"] for n in LENGTHS},
                "uniform_route_mae5": run["test_uniform_route"]["5"]["mae"],
                "shuffled_route_mae5": run["test_shuffled_route"]["5"]["mae"],
            }
        differences = {}
        for length in (5, 20):
            conv = np.asarray(pair["conv"]["test"][str(length)]["mae_per_task"])
            mlp = np.asarray(pair["mlp"]["test"][str(length)]["mae_per_task"])
            if len(conv) != 64 or len(mlp) != 64:
                raise ValueError(f"Expected 64 test tasks: K={count}")
            difference = mlp - conv
            rng = np.random.default_rng(20260920 + 1000 * count + length)
            sampled = difference[rng.integers(
                0, len(difference), size=(20000, len(difference)))].mean(axis=1)
            differences[str(length)] = {
                "mlp_minus_conv": float(difference.mean()),
                "paired_task_bootstrap_95pct": [float(x) for x in
                                                 np.quantile(sampled, (0.025, 0.975))],
            }
        row["differences"] = differences
        summary["by_count"][str(count)] = row
    args.results.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.7, 7.8), layout="constrained")
    colors = {"conv": "#147f6f", "mlp": "#c46b39"}
    labels = {"conv": "Conv router", "mlp": "MLP router"}
    for ax, length, title in ((axes[0, 0], 5, "A. Test MAE, length 5"),
                              (axes[0, 1], 20, "B. Test MAE, length 20")):
        for name in ("conv", "mlp"):
            ax.plot(COUNTS, [runs[k][name]["test"][str(length)]["mae"]
                                  for k in COUNTS], marker="o", linewidth=2.2,
                    color=colors[name], label=labels[name])
        ax.set_xticks(COUNTS)
        ax.set_xlabel("Number of v experts")
        ax.set_ylabel("Mean absolute error ↓")
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
    ax = axes[1, 0]
    for name in ("conv", "mlp"):
        ax.plot(COUNTS, [runs[k][name]["best_validation_score"]
                         for k in COUNTS], marker="o", color=colors[name],
                linewidth=2, label=labels[name])
    ax.set_xticks(COUNTS)
    ax.set_xlabel("Number of v experts")
    ax.set_ylabel("Best validation score ↓")
    ax.set_title("C. Checkpoint-selection criterion")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    ax = axes[1, 1]
    for name in ("conv", "mlp"):
        ax.plot(LENGTHS,
                [runs[96][name]["test"][str(n)]["mae"] for n in LENGTHS],
                marker="o", color=colors[name], linewidth=2, label=labels[name])
    ax.set_xticks(LENGTHS)
    ax.set_xlabel("Query-set length")
    ax.set_ylabel("Test MAE ↓")
    ax.set_title("D. Same K=96 across set lengths")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.suptitle("MNIST8m · parameter-matched v routers · λ=0.2 · seed 42")
    figure = args.results / "v_moe_router_summary.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(f"Figure: {figure}")
    if args.conv_long is not None:
        longer = {"conv": json.loads(args.conv_long.read_text()),
                  "mlp": json.loads(args.mlp_long.read_text())}
        for name, run in longer.items():
            config = run["config"]
            if (config.get("router", "conv") != name
                    or config["experts"] != 96
                    or config["ortho_weight"] != 0.2
                    or config["seed"] != 42
                    or config["steps"] != 10000
                    or run["history"][-1]["step"] != 10000
                    or "test" not in run):
                raise ValueError(f"Invalid 10000-step result: {name}")
        summary["longer_training_k96"] = {
            name: {
                "best_step": run["best_step"],
                "best_validation_score": run["best_validation_score"],
                "mae": {str(n): run["test"][str(n)]["mae"] for n in LENGTHS},
            } for name, run in longer.items()}
        summary["longer_training_k96"]["differences"] = {}
        for length in (5, 20):
            conv = np.asarray(longer["conv"]["test"][str(length)]["mae_per_task"])
            mlp = np.asarray(longer["mlp"]["test"][str(length)]["mae_per_task"])
            difference = mlp - conv
            rng = np.random.default_rng(20260920 + length)
            sampled = difference[rng.integers(
                0, len(difference), size=(20000, len(difference)))].mean(axis=1)
            summary["longer_training_k96"]["differences"][str(length)] = {
                "mlp_minus_conv": float(difference.mean()),
                "paired_task_bootstrap_95pct": [float(x) for x in
                                                 np.quantile(sampled, (0.025, 0.975))],
            }
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.1),
                                 layout="constrained")
        ax = axes[0]
        for name, run in longer.items():
            history = run["history"]
            ax.plot([row["step"] for row in history],
                    [row["validation_score"] for row in history],
                    color=colors[name], linewidth=2, label=labels[name])
        ax.axvline(5000, color="#7b828a", linestyle="--", linewidth=1)
        ax.set_xlabel("Outer training step")
        ax.set_ylabel("Validation score ↓")
        ax.set_title("A. K=96 convergence to 10000 steps")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        ax = axes[1]
        for name in ("conv", "mlp"):
            for length, linestyle in ((5, "-"), (20, "--")):
                values = (runs[96][name]["test"][str(length)]["mae"],
                          longer[name]["test"][str(length)]["mae"])
                ax.plot((5000, 10000), values, marker="o", linewidth=2,
                        linestyle=linestyle, color=colors[name],
                        label=f"{labels[name]}, length {length}")
        ax.set_xticks((5000, 10000))
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Test MAE ↓")
        ax.set_title("B. K=96 test error (5k: separate runs)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        figure = args.results / "v_moe_router_convergence.png"
        fig.savefig(figure, dpi=180)
        plt.close(fig)
        print(f"Figure: {figure}")
        for name, run in longer.items():
            print(f"10000 steps {name}: best={run['best_step']} "
                  f"MAE5={run['test']['5']['mae']:.4f} "
                  f"MAE20={run['test']['20']['mae']:.4f}")
    (args.results / "v_moe_router_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print("K  conv_params mlp_params conv5 mlp5 conv20 mlp20 MLP-minus-conv5")
    for count in COUNTS:
        row = summary["by_count"][str(count)]
        print(f"{count:<3} {row['conv']['router_parameters']:<11} "
              f"{row['mlp']['router_parameters']:<10} "
              f"{row['conv']['mae']['5']:.4f} "
              f"{row['mlp']['mae']['5']:.4f} "
              f"{row['conv']['mae']['20']:.4f} "
              f"{row['mlp']['mae']['20']:.4f} "
              f"{row['differences']['5']['mlp_minus_conv']:+.4f}")


if __name__ == "__main__":
    main()
