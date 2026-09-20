"""Compare coordinate-generated first-layer U with matched MNIST8m controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 43, 44)
LENGTHS = (1, 3, 5, 10, 20)
ARMS = ("generated", "generated_shuffled", "generated_ortho",
        "generated_shuffled_ortho", "learned", "dense", "random")
LABELS = {"generated": "spatial generator",
          "generated_shuffled": "shuffled coordinates",
          "generated_ortho": "spatial + orthogonal",
          "generated_shuffled_ortho": "shuffled + orthogonal",
          "learned": "direct U", "dense": "dense", "random": "fixed random U"}
COLORS = {"generated": "#15896b", "generated_shuffled": "#c77923",
          "generated_ortho": "#18aa86",
          "generated_shuffled_ortho": "#e19c3f",
          "learned": "#1769aa", "dense": "#8a55a0", "random": "#8e9299"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "deepsets_z/mnist8m/meta_u_first_results"))
    parser.add_argument("--out", type=Path, default=Path(
        "deepsets_z/mnist8m"))
    args = parser.parse_args()
    runs = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = args.root / f"{arm}_seed{seed}.json"
            data = json.loads(path.read_text())
            if "test" not in data or data["history"][-1]["step"] != 5000:
                raise ValueError(f"Incomplete result: {path}")
            runs[(arm, seed)] = data
    for seed in SEEDS:
        reference = runs[("learned", seed)]["config"]
        for arm in ARMS:
            config = runs[(arm, seed)]["config"]
            for key in ("seed", "steps", "inner_steps", "lr_v1", "lr_readout",
                        "outer_lr", "tasks_per_step", "support", "query",
                        "test_tasks", "train_images_per_digit",
                        "eval_images_per_digit"):
                if config[key] != reference[key]:
                    raise ValueError(f"Mismatched {key}: {arm}, seed {seed}")
    diagnostic = json.loads((args.out / "meta_u_first_diagnostics.json").read_text())
    summary = {"seeds": list(SEEDS), "by_length": {},
               "generator_parameters": runs[("generated", 42)]["model"][
                   "generator_parameters"],
               "overlap": {}, "initial_shift_error": {},
               "adapted_shift_error": {}, "basis_effective_rank": {},
               "initial_image_sensitivity": {},
               "adapted_image_sensitivity": {},
               "no_v_adaptation_mean_mae": diagnostic[
                   "no_v_adaptation_mean_mae"]}
    for length in LENGTHS:
        row = {}
        for arm in ARMS:
            values = np.array([runs[(arm, seed)]["test"][str(length)]["mae"]
                               for seed in SEEDS])
            row[arm] = {"mean_mae": float(values.mean()),
                        "sd_across_seeds": float(values.std(ddof=1)),
                        "by_seed": values.tolist()}
        summary["by_length"][str(length)] = row
    for arm in ARMS:
        summary["overlap"][arm] = [runs[(arm, seed)]["convolution_overlap"]
                                   for seed in SEEDS]
        summary["initial_shift_error"][arm] = [
            diagnostic["first_layer_shift_error"][f"{arm}_seed{seed}"]
            for seed in SEEDS]
        summary["adapted_shift_error"][arm] = [
            diagnostic["adapted_first_layer"][f"{arm}_seed{seed}"][
                "shift_error_mean"] for seed in SEEDS]
        summary["basis_effective_rank"][arm] = [
            diagnostic["basis_effective_rank"][f"{arm}_seed{seed}"]
            for seed in SEEDS]
        summary["initial_image_sensitivity"][arm] = [
            diagnostic["first_layer_image_sensitivity"][
                f"{arm}_seed{seed}"]["variance_to_energy"]
            for seed in SEEDS]
        summary["adapted_image_sensitivity"][arm] = [
            diagnostic["adapted_first_layer"][f"{arm}_seed{seed}"][
                "image_sensitivity_mean"] for seed in SEEDS]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "meta_u_first_generator_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        for arm in ARMS:
            history = runs[(arm, seed)]["history"]
            ax.plot([r["step"] for r in history],
                    [r["validation_score"] for r in history],
                    label=LABELS[arm], color=COLORS[arm], linewidth=1.6)
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("Meta-training step")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Validation MSE / set length ↓")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "meta_u_first_generator_convergence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for arm in ARMS:
        values = np.array([[runs[(arm, seed)]["test"][str(length)]["mae"]
                            for length in LENGTHS] for seed in SEEDS])
        for per_seed in values:
            ax.plot(LENGTHS, per_seed, color=COLORS[arm], alpha=0.18,
                    linewidth=0.8)
        ax.plot(LENGTHS, values.mean(axis=0), color=COLORS[arm],
                marker="o", label=LABELS[arm], linewidth=2)
    ax.set_xlabel("Set length")
    ax.set_ylabel("Shifted-query test MAE ↓")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "meta_u_first_generator_lengths.png", dpi=180)
    plt.close(fig)

    structure_arms = ("generated", "generated_shuffled", "generated_ortho",
                      "generated_shuffled_ortho", "learned", "random",
                      "convolution")
    rank = [float(np.mean(summary["basis_effective_rank"][arm]))
            if arm != "convolution" else
            diagnostic["basis_effective_rank"]["convolution_seed42"]
            for arm in structure_arms]
    shift = [float(np.mean(summary["adapted_shift_error"][arm]))
             if arm != "convolution" else
             diagnostic["adapted_first_layer"]["convolution_seed42"][
                 "shift_error_mean"]
             for arm in structure_arms]
    sensitivity = [float(np.mean(summary["adapted_image_sensitivity"][arm]))
                   if arm != "convolution" else
                   diagnostic["adapted_first_layer"][
                       "convolution_seed42"]["image_sensitivity_mean"]
                   for arm in structure_arms]
    positions = np.arange(len(structure_arms))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=True)
    for ax, values, xlabel in zip(axes, (rank, shift, sensitivity),
                                  ("Effective rank of U ↑",
                                   "Shift error after task adaptation ↓",
                                   "Image variance / energy after adaptation ↑")):
        ax.barh(positions, values,
                color=[COLORS.get(arm, "#348659") for arm in structure_arms])
        ax.set_yticks(positions, [LABELS.get(arm, "analytic conv, seed 42")
                                  for arm in structure_arms])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.out / "meta_u_first_generator_structure.png", dpi=180)
    plt.close(fig)
    print(json.dumps({length: summary["by_length"][length]
                      for length in ("5", "10", "20")}, indent=2))


if __name__ == "__main__":
    main()
