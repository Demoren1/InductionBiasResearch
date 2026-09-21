"""Plot learning curves and paired held-out task comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from .run import OUTPUT_ROOT, MetaConv, augment_queries, episode, load_data


KINDS = ("generated", "generated_w80", "direct", "random", "identity")
COLORS = {"generated": "#176692", "generated_w80": "#2da5bd", "direct": "#ba5a3b", "random": "#8558a5", "identity": "#3d454c"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", type=int, default=400)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    data, splits, manifest = load_data(device)
    models, histories, best_steps = {}, {}, {}
    for kind in KINDS:
        folder = OUTPUT_ROOT / f"{kind}_seed{args.seed}"
        config = json.loads((folder / "config.json").read_text())["config"]
        model = MetaConv("generated" if kind.startswith("generated") else kind, config["ways"], config.get("generator_width", 64)).to(device)
        checkpoint = torch.load(folder / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models[kind] = model
        best_steps[kind] = checkpoint["step"]
        histories[kind] = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
    accuracies = {kind: [] for kind in KINDS}
    losses = {kind: [] for kind in KINDS}
    for task in range(args.tasks):
        task_seed = args.seed * 100_000_000 + 2_000_000 + task
        sx, sy, qx, qy = episode(data["images"], splits["test"], 5, 1, 5, task_seed)
        qx = augment_queries(qx, task_seed + 9187)
        for kind, model in models.items():
            u = {key: val.detach() for key, val in model.u.matrices().items()}
            v = model.adapt(sx, sy, u, 3, second_order=False)
            with torch.no_grad():
                logits = model(qx, v, u)
                accuracies[kind].append((logits.argmax(-1) == qy).float().mean().item())
                losses[kind].append(F.cross_entropy(logits, qy).item())
        if (task + 1) % 50 == 0:
            print(f"paired test {task + 1}/{args.tasks}", flush=True)
    arrays = {kind: np.array(vals) for kind, vals in accuracies.items()}
    rng = np.random.default_rng(12345)
    pairwise = {}
    for kind in KINDS[1:]:
        delta = arrays["generated"] - arrays[kind]
        bootstrap = rng.choice(delta, size=(3000, len(delta)), replace=True).mean(1)
        pairwise[kind] = {"mean_percentage_points": float(100 * delta.mean()), "ci95_percentage_points": [float(v) for v in 100 * np.quantile(bootstrap, [.025, .975])]}
    delta = arrays["generated_w80"] - arrays["direct"]
    bootstrap = rng.choice(delta, size=(3000, len(delta)), replace=True).mean(1)
    pairwise["generated_w80_vs_direct"] = {"mean_percentage_points": float(100 * delta.mean()), "ci95_percentage_points": [float(v) for v in 100 * np.quantile(bootstrap, [.025, .975])]}
    summary = {
        "seed": args.seed,
        "test_tasks": args.tasks,
        "data": manifest,
        "arms": {kind: {"accuracy": float(arrays[kind].mean()), "accuracy_se": float(arrays[kind].std(ddof=1) / np.sqrt(args.tasks)), "loss": float(np.mean(losses[kind])), "best_step": best_steps[kind]} for kind in KINDS},
        "generated_minus": pairwise,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "paired_comparison.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for kind in KINDS:
        h = histories[kind]
        steps = [item["step"] for item in h]
        axes[0].plot(steps, [100 * item["val"]["accuracy"] for item in h], label=kind, color=COLORS[kind])
        axes[1].plot(steps, [item["val"]["loss"] for item in h], label=kind, color=COLORS[kind])
    axes[0].set(title="Validation accuracy", xlabel="Outer step", ylabel="Accuracy, %")
    axes[1].set(title="Validation loss", xlabel="Outer step", ylabel="Cross-entropy")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=.2)
    axes[1].grid(alpha=.2)
    positions = np.arange(len(KINDS))
    means = [100 * arrays[kind].mean() for kind in KINDS]
    errors = [1.96 * 100 * arrays[kind].std(ddof=1) / np.sqrt(args.tasks) for kind in KINDS]
    axes[2].bar(positions, means, yerr=errors, color=[COLORS[kind] for kind in KINDS], capsize=4)
    axes[2].set_xticks(positions, KINDS, rotation=20)
    axes[2].set(title=f"Held-out tasks ({args.tasks} episodes)", ylabel="Accuracy, %")
    axes[2].grid(axis="y", alpha=.2)
    fig.savefig(OUTPUT_ROOT / "comparison.png", dpi=180)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
