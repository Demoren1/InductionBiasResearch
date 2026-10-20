"""Plot saved VAE learning and a separate source-task SGD trajectory sample."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .settings import OPTIONS, RUN
from .core import correct_mask, fit, save_json, task
from .train import load_cards, restore, sampled_batch


def main():
    torch.set_num_threads(1)
    trace_path = RUN / "training_trajectories.npz"
    if not trace_path.exists():
        steps = list(range(16, 1025, 16))
        traces = {"steps": np.asarray(steps)}
        spec = task(0, 0)
        selection = json.loads((RUN / "selection.json").read_text())
        for name, masks, options in (
            (
                "dense",
                np.ones((128, 8, 2), dtype=np.float32),
                selection["dense_options"],
            ),
            ("correct_mask", correct_mask(spec, 128), OPTIONS),
        ):
            result = fit(
                masks, spec, 23911, **options, device="cpu", checkpoints=steps
            )
            traces[name] = np.stack(
                [result["history"][str(step)] for step in steps], 1
            )
        np.savez_compressed(trace_path, **traces)

    records = json.loads((RUN / "models.json").read_text())
    _, roles = load_cards()
    final_metrics = []
    complete_training_history = True
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for record in records:
        color = (
            "#277dad"
            if record["variant"] == "partwise_training"
            else "#ac638d"
        )
        label = f"{record['variant']} / {record['seed']}"
        history_path = (RUN / record["checkpoint"]).parent / "history.json"
        history = json.loads(history_path.read_text())
        complete_training_history &= all("train" in row for row in history)
        epochs = [row["epoch"] for row in history]
        axes[0, 0].plot(
            epochs,
            [row["loss"] for row in history],
            color=color,
            alpha=0.75,
            label=label,
        )
        if all("train" in row for row in history):
            axes[0, 0].plot(
                epochs,
                [row["train"]["loss"] for row in history],
                color=color,
                linestyle="--",
                alpha=0.6,
            )
        axes[0, 1].plot(
            epochs, [row["exact"] for row in history], color=color, alpha=0.75
        )
        model = restore(record)
        metrics = {"variant": record["variant"], "seed": record["seed"]}
        for role, name in enumerate(("train", "validation")):
            x, target = sampled_batch(
                roles[role], max(len(x) for x, _ in roles[role]), fixed=True
            )
            with torch.no_grad():
                scores, _, _ = model(x, sample=False)
                flat = scores.flatten(-2)
                indices = flat.argsort(dim=-1, descending=True, stable=True)[
                    ..., :2
                ]
                masks = (
                    torch.zeros_like(flat)
                    .scatter_(-1, indices, 1)
                    .reshape_as(scores)
                )
                values = [
                    (masks[i, : len(features)] * target[i, : len(features)])
                    .sum((-1, -2))
                    .mean()
                    / 2
                    for i, (features, _) in enumerate(roles[role])
                ]
                metrics[f"{name}_mask_recall_at_2"] = float(
                    torch.stack(values).mean()
                )
        final_metrics.append(metrics)
    axes[0, 0].set(
        title="NF/VAE: saved source-validation objective",
        xlabel="Epoch",
        ylabel="Reconstruction + KL",
    )
    axes[0, 0].legend(fontsize=6)
    axes[0, 1].set(
        title="NF/VAE: exact source-mask recovery",
        xlabel="Epoch",
        ylabel="Fraction correct",
        ylim=(-0.03, 1.03),
    )
    traces = np.load(trace_path)
    steps = traces["steps"]
    dense, sparse = traces["dense"], traces["correct_mask"]
    good = sparse[:, -1] < 0.1
    ax = axes[1, 0]
    ax.plot(
        steps,
        np.median(dense, axis=0),
        color="#808b96",
        label="Dense SGD, median",
    )
    for selected, label, color in (
        (good, "Correct mask, successful fits", "#277dad"),
        (~good, "Correct mask, failed fits", "#db9735"),
    ):
        if selected.any():
            ax.plot(steps, sparse[selected].mean(0), color=color, label=label)
    ax.set(
        title="Source-task SGD: expected prediction MSE",
        xlabel="SGD step",
        ylabel="Population MSE",
    )
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    for data, label, color in (
        (dense, "Dense SGD", "#808b96"),
        (sparse, "Known correct mask", "#277dad"),
    ):
        ax.plot(steps, (data < 0.1).mean(0), color=color, label=label)
    ax.set(
        title="Recovery of a good minimum during training",
        xlabel="SGD step",
        ylabel="Fraction with MSE < 0.1",
    )
    ax.legend(fontsize=8)
    fig.suptitle(
        "Learning curves; source trajectories are diagnostic, outside the search comparison"
    )
    fig.savefig(RUN / "training.png", dpi=140)
    plt.close(fig)
    save_json(
        RUN / "training_metrics.json",
        {
            "best_checkpoint_metrics": final_metrics,
            "trajectory_scope": "128 independent source-task fits per policy on CPU; known mask is diagnostic, not a search result",
            "history_limit": (
                "Train/validation loss, KL and Recall@2 are recorded; training uses the current KL warmup."
                if complete_training_history
                else "Original VAE logs contain validation loss/exact recovery, not per-epoch train loss or edge Recall@2. Final train/validation Recall@2 evaluated at frozen best checkpoints."
            ),
        },
    )


if __name__ == "__main__":
    main()
