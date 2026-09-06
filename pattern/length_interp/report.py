"""Report interpolation quality with pattern-level paired comparisons."""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .common import ideal_mask, load_protocol, patterns_by_split, write_json
from .evaluate import METHODS


def report(out):
    out = Path(out)
    c = load_protocol(out)
    values = defaultdict(list)
    pools = {}
    seen = set()
    for seed in c.cvae_seeds:
        for k in c.heldout_lengths:
            for shard in range(2):
                d = torch.load(out / "evaluation" / f"seed{seed}_k{k}_shard{shard}.pt", map_location="cpu", weights_only=False)
                if seed == c.cvae_seeds[0]:
                    pools[k] = d["pool"]
                assert tuple(d["pool"]["methods"]) == METHODS
                assert torch.all(d["pool"]["masks"].sum((-1, -2)) == c.hidden * k)
                for r in d["rows"]:
                    key = seed, r["pattern"], r["repeat"], r["distribution"]
                    if key in seen:
                        raise AssertionError(f"Duplicate evaluation row {key}")
                    seen.add(key)
                    values[(k, r["distribution"], r["pattern"])].append(r["accuracy"].mean(1).numpy())
    expected = len(patterns_by_split(c)["test"]) * len(c.cvae_seeds) * c.eval_repeats * 2
    if len(seen) != expected:
        raise AssertionError(f"Incomplete evaluation: {len(seen)} != {expected}")
    result = []
    rng = np.random.default_rng(42)
    for k in c.heldout_lengths:
        for distribution in ("balanced", "natural"):
            per_task = np.stack([np.mean(v, axis=0) for (length, kind, _), v in sorted(values.items())
                                 if length == k and kind == distribution])
            sample = rng.integers(len(per_task), size=(5000, len(per_task)))
            for j, method in enumerate(METHODS):
                differences = per_task[:, j] - per_task[:, METHODS.index("random")]
                boot = differences[sample].mean(1)
                result.append({"length": k, "distribution": distribution, "method": method,
                               "tasks": len(per_task), "accuracy": float(per_task[:, j].mean()),
                               "delta_vs_random": float(differences.mean()),
                               "task_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist()})
    write_json(out / "summary.json", {"config": c.to_dict(), "rows": result,
                                      "bootstrap_unit": "pattern; averages masks, MLP repeats and fixed CVAE seeds first"})
    lines = ["# Pattern-32 CVAE: length interpolation", "",
             f"CVAE train/validation lengths: {c.train_lengths}; unseen lengths: {c.heldout_lengths}.", "",
             "Balanced accuracy. Intervals resample tasks, conditional on the two fitted CVAE seeds.", "",
             "| Length | Method | Accuracy | Difference vs random, pp | Task bootstrap 95% interval, pp |",
             "|---:|---|---:|---:|---:|"]
    for r in result:
        if r["distribution"] != "balanced":
            continue
        lo, hi = r["task_bootstrap_ci95"]
        lines.append(f"| {r['length']} | {r['method']} | {100*r['accuracy']:.2f}% | {100*r['delta_vs_random']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ["", "Each method has exactly 32 × target_length active mask entries.",
              "Wrong-condition controls use the same latent draws and target-length cardinality as CVAE.",
              "The mean baseline interpolates train-only length means; ideal is a diagnostic support, followed by ordinary MLP training.",
              "No unseen-length labels enter bank selection, CVAE training, checkpoint selection or mask sampling.",
              "Natural-distribution results and per-task evidence are stored in JSON/PT files.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(len(c.heldout_lengths), len(METHODS), figsize=(19, 6), squeeze=False)
    for row, k in enumerate(c.heldout_lengths):
        target = ideal_mask(k, c)
        for j, method in enumerate(METHODS):
            mask = pools[k]["masks"][j, 0]
            rows, cols = linear_sum_assignment((mask.T @ target).numpy(), maximize=True)
            order = np.empty(c.hidden, dtype=int)
            order[cols] = rows
            axes[row, j].imshow(mask[:, order], vmin=0, vmax=1, cmap="Greys", interpolation="nearest")
            axes[row, j].set_title(f"k={k} {method}\nIoU={float(pools[k]['iou'][j,0]):.3f}", fontsize=9)
            axes[row, j].set_xticks([])
            axes[row, j].set_yticks([])
    fig.suptitle("One sampled mask per method; hidden columns matched to ideal for visualization only")
    fig.tight_layout()
    fig.savefig(out / "support_masks.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    report(parser.parse_args().out)
