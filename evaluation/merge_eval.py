"""Merge partial eval results from parallel GPU runs and plot the final figure.

Usage:
    python evaluation/merge_eval.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

OUT_DIR = config.OUTPUTS / "eval"
EVAL_KERNELS = [3, 4, 5, 6, 7, 8, 9, 10]
ORDER = ["random", "cvae", "top10%", "ideal"]


def main() -> None:
    merged = {}
    for f in sorted(OUT_DIR.glob("eval_results_gpu*.pt")):
        d = torch.load(f, weights_only=True)
        merged.update(d)
        print(f"[merge] {f.name} -> {len(d)} entries")

    torch.save(merged, OUT_DIR / "eval_results.pt")
    with open(OUT_DIR / "eval_results.json", "w") as f:
        json.dump({f"{k}:{s}:{m}": st for (k, s, m), st in merged.items()},
                  f, indent=2)

    config.PLOT_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Separate train and test results and plot them.
    train_pairs = [(k, s) for k, s in config.CVAE_TRAIN_PAIRS]
    test_pairs = [(k, s) for k, s in config.CVAE_TEST_PAIRS]
    
    # Build categorical x-axis: all 16 pairs sorted by (k, s)
    all_pairs = sorted(train_pairs + test_pairs, key=lambda p: (p[0], p[1]))
    x_labels = [f"k={k},s={s}" for k, s in all_pairs]
    x_positions = list(range(len(all_pairs)))
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    for i, m in enumerate(ORDER):
        color = f"C{i}"
        # Train pairs: filled circles
        train_x = [x_positions[all_pairs.index(p)] for p in train_pairs]
        train_y = [merged.get((p[0], p[1], m), {"mean": np.nan})["mean"] for p in train_pairs]
        ax.scatter(train_x, train_y, marker='o', color=color, s=70,
                   label=f"{m} train", zorder=3)
        # Test pairs: hollow diamonds
        test_x = [x_positions[all_pairs.index(p)] for p in test_pairs]
        test_y = [merged.get((p[0], p[1], m), {"mean": np.nan})["mean"] for p in test_pairs]
        ax.scatter(test_x, test_y, marker='D', facecolors='none', edgecolors=color,
                   linewidths=1.5, s=70, label=f"{m} test", zorder=2)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("mean val MSE (log)")
    ax.legend(ncols=2, fontsize=8, loc='upper left')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(config.PLOT_EVAL_DIR / "eval_mse.png", dpi=130)
    plt.close(fig)
    print(f"[merge] plot -> {config.PLOT_EVAL_DIR / 'eval_mse.png'}")
    print(f"[merge] json -> {OUT_DIR / 'eval_results.json'}")


if __name__ == "__main__":
    main()