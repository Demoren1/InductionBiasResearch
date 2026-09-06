"""Plot saved stage diagnostics without loading or modifying source models."""
import argparse
import json
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.oracle_ideal import align_columns


def run(root):
    summaries = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    display = {"canonical": "Input top-96", "posterior_mu": "Posterior mean top-96",
               "z0": "z=0 top-96", "internal_train_gap_mean": "Internal-train gap mean"}
    for axis, profile in zip(axes, ("interp", "extrap")):
        folder = root / profile
        summary = json.loads((folder / "summary.json").read_text())
        data = torch.load(folder / "diagnostic.pt", weights_only=True)
        gaps = sorted(int(g) for g in summary["groups"] if g != "all")
        for stage, label in display.items():
            axis.plot(gaps, [summary["groups"][str(g)]["val"][stage]["iou_gold"] for g in gaps],
                      "o-", label=label)
        axis.set(title=f"{profile}: internal validation, seen gaps only", xlabel="Gap",
                 ylabel="Mean top-96 IoU with ideal (column permutation)", ylim=(0, 1.02))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
        # A deterministic, non-selected example from each gap.
        examples, mask_axes = plt.subplots(len(gaps), 4, figsize=(10, 2.4 * len(gaps)), layout="constrained")
        for row, gap in enumerate(gaps):
            index = int(((data["gaps"] == gap) & data["is_internal_val"]).nonzero()[0])
            target = data["target"][index]
            masks = [align_columns(target, data["hard"][stage][index]) for stage in ("canonical", "posterior_mu", "z0")]
            masks.append(target)
            labels = [f"{display[stage]}\nIoU={data['metrics'][stage]['iou_gold'][index]:.3f}"
                      for stage in ("canonical", "posterior_mu", "z0")] + ["Ideal"]
            for a, mask, label in zip(mask_axes[row], masks, labels):
                a.imshow(mask, vmin=0, vmax=1, cmap="Greys")
                a.set(title=label, xticks=[], yticks=[])
            mask_axes[row, 0].set_ylabel(f"gap {gap}; row {index}")
        examples.suptitle(f"{profile}: first internal-val example per gap; gold alignment for display")
        examples.savefig(folder / "examples.png", dpi=150)
        plt.close(examples)
        # Derive threshold diagnostics from saved values, not assumptions.
        thresholds = {}
        for stage in ("raw", "canonical", "posterior_mu"):
            values = data["stages"][stage].flatten(1)
            threshold = values.topk(96, dim=1).values[:, -1]
            tie_count = (values == threshold[:, None]).sum(1)
            thresholds[stage] = {"zero_cutoff_count": int((threshold == 0).sum()),
                                 "mean_threshold": float(threshold.mean()),
                                 "mean_tie_count": float(tie_count.float().mean()),
                                 "max_tie_count": int(tie_count.max())}
        summaries[profile] = {"summary": summary, "thresholds": thresholds,
                              "max_abs_raw_canonical_iou_delta": float((data["metrics"]["raw"]["iou_gold"] -
                                                                           data["metrics"]["canonical"]["iou_gold"]).abs().max())}
    fig.savefig(root / "seen_gap_reconstruction.png", dpi=170)
    fig.savefig(root / "seen_gap_reconstruction.pdf")
    plt.close(fig)
    (root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps({p: {"thresholds": r["thresholds"], "max_raw_canonical_delta": r["max_abs_raw_canonical_iou_delta"]}
                      for p, r in summaries.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.root)
