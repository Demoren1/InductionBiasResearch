"""Evaluate trained conv/MLP v routers on set lengths through 50."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .meta_u import load_pool
from .meta_u_first_v_moe import VExpertMoE, evaluate


# Preserve the original order of lengths 1, 3, 5, 10, 20 so those task
# episodes are the same as in the earlier tests.
EVALUATION_ORDER = (1, 3, 5, 10, 20, 15, 25, 30, 40, 50)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conv-dir", type=Path, default=Path(
        "deepsets_z/mnist8m/outputs/meta_u_first_v_moe_router_conv_10k"))
    parser.add_argument("--mlp-dir", type=Path, default=Path(
        "deepsets_z/mnist8m/outputs/meta_u_first_v_moe_router_mlp_10k"))
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mnist8m"))
    parser.add_argument("--out", type=Path, default=Path(
        "deepsets_z/mnist8m/outputs/meta_u_first_v_moe_router_converged"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", type=int, default=64)
    args = parser.parse_args()
    if args.tasks < 1:
        parser.error("--tasks must be positive")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    runs = {}
    models = {}
    for name, directory in (("conv", args.conv_dir), ("mlp", args.mlp_dir)):
        stem = "moe_k96_ortho0p2_seed42" + ("_mlp" if name == "mlp" else "")
        result_path = directory / f"{stem}.json"
        run = json.loads(result_path.read_text())
        config = run["config"]
        if (config["experts"] != 96 or config["seed"] != 42
                or config["ortho_weight"] != 0.2
                or config.get("router", "conv") != name
                or "stopping" not in run or "test" not in run):
            raise ValueError(f"Invalid completed run: {result_path}")
        model = VExpertMoE(96, 42, name).to(device)
        model.load_state_dict(torch.load(
            directory / f"{stem}_best.pt", map_location=device,
            weights_only=True))
        runs[name] = run
        models[name] = model
    reference = runs["conv"]["config"]
    for key in ("support", "query", "inner_steps", "lr_coeff",
                "lr_readout", "eval_images_per_digit"):
        if runs["mlp"]["config"][key] != reference[key]:
            raise ValueError(f"Mismatched {key} between routers")
    pool = load_pool(args.data_dir, seed=42, split="test",
                     per_digit=reference["eval_images_per_digit"],
                     device=device)
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}: best step {runs[name]['best_step']}",
              flush=True)
        results[name] = evaluate(
            model, pool, seed=42 + 19000, tasks=args.tasks,
            support=reference["support"], query=reference["query"],
            steps=reference["inner_steps"], lr_coeff=reference["lr_coeff"],
            lr_readout=reference["lr_readout"], lengths=EVALUATION_ORDER)
        for length in EVALUATION_ORDER:
            print(f"{name} length={length} "
                  f"MAE={results[name][str(length)]['mae']:.4f}", flush=True)
    rng = np.random.default_rng(20260920)
    summary = {
        "protocol": {
            "seed": 42, "test_tasks_per_length": args.tasks,
            "evaluation_order": EVALUATION_ORDER,
            "train_query_lengths": [1, 2, 3, 4, 5],
            "test_image_shift_pixels": 3,
            "support_sets": reference["support"],
            "adaptation_steps": reference["inner_steps"]},
        "training": {
            name: {"best_step": run["best_step"],
                   "best_validation_score": run["best_validation_score"],
                   "stopping": run["stopping"]}
            for name, run in runs.items()},
        "by_length": {}}
    for length in sorted(EVALUATION_ORDER):
        key = str(length)
        conv = np.asarray(results["conv"][key]["mae_per_task"])
        mlp = np.asarray(results["mlp"][key]["mae_per_task"])
        difference = mlp - conv
        sampled = difference[rng.integers(
            0, args.tasks, size=(20000, args.tasks))].mean(axis=1)
        summary["by_length"][key] = {
            name: results[name][key] for name in ("conv", "mlp")}
        summary["by_length"][key]["mlp_minus_conv"] = {
            "mae": float(difference.mean()),
            "paired_task_bootstrap_95pct": [float(x) for x in
                                             np.quantile(sampled, (0.025, 0.975))]}
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "v_moe_router_lengths_to50.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    lengths = sorted(EVALUATION_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5),
                             layout="constrained")
    for name, color, label in (("conv", "#147f6f", "Conv router"),
                               ("mlp", "#c46b39", "MLP router")):
        mae = [summary["by_length"][str(n)][name]["mae"] for n in lengths]
        axes[0].plot(lengths, mae, color=color, marker="o", linewidth=2.2,
                     label=label)
        axes[1].plot(lengths,
                     [summary["by_length"][str(n)][name]["normalized_mse"]
                      for n in lengths],
                     color=color, marker="o", linewidth=2.2, label=label)
    for ax in axes:
        ax.axvline(5, color="#747e87", linestyle="--", linewidth=1,
                   label="Longest training set")
        ax.set_xticks(lengths)
        ax.set_xlabel("Number of images in query set")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Test MAE ↓")
    axes[0].set_title("A. Absolute error")
    axes[1].set_ylabel("Test squared error / length ↓")
    axes[1].set_title("B. Training loss by test length")
    fig.suptitle("MNIST8m · image-routed v experts · K=96 · λ=0.2")
    figure_path = args.out / "v_moe_router_lengths_to50.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.2, 4.7), layout="constrained")
    for name, color, label in (("conv", "#147f6f", "Conv router"),
                               ("mlp", "#c46b39", "MLP router")):
        history = runs[name]["history"]
        steps = [row["step"] for row in history]
        scores = [row["validation_score"] for row in history]
        ax.plot(steps, scores, color=color, linewidth=1.9, label=label)
        ax.scatter([runs[name]["best_step"]],
                   [runs[name]["best_validation_score"]],
                   color=color, s=55, zorder=3)
        ax.axvline(runs[name]["stopping"]["stop_step"], color=color,
                   linestyle=":", linewidth=1.1)
    ax.set_xlabel("Outer training step")
    ax.set_ylabel("Validation score ↓")
    ax.set_title("Training to the validation-plateau criterion")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    convergence_path = args.out / "v_moe_router_convergence_to_stop.png"
    fig.savefig(convergence_path, dpi=180)
    plt.close(fig)
    print(f"Summary: {summary_path}\nFigures: {figure_path}, "
          f"{convergence_path}", flush=True)


if __name__ == "__main__":
    main()
