"""Summarize completed paired and task-loss latent searches, with saved evaluations."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.generate import ideal_mask
from evaluation.oracle_ideal import align_columns
from evaluation.structural import best_permutation_iou


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root):
    protocol = json.loads((root / "protocol.json").read_text())
    audit = json.loads((root / "structural_audit.json").read_text())
    summary = {"protocol_sha256": sha(root / "protocol.json"),
               "single_z_protocol_sha256": sha(root / "single_z_protocol.json"), "profiles": {}}
    methods = ["prior1", "agreement1", "random_search1", "single_z", "conditional_mean", "random_exact96", "ideal"]
    labels = ["Prior42", "Pair42", "Random pair42", "Task z42", "Gap mean", "Random96", "Ideal"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    examples, example_axes = plt.subplots(4, 4, figsize=(10, 10), layout="constrained")
    for row_index, (profile, source) in enumerate(protocol["profiles"].items()):
        folder = root / profile
        evaluation = json.loads((folder / "evaluation_with_single/summary.json").read_text())
        single_audit = json.loads((folder / "single_z/audit.json").read_text())
        assert evaluation["provenance"]["mask_sha256"] == audit[profile]["masks_sha256"]
        assert evaluation["provenance"]["settings"]["single_z_masks_sha256"] == single_audit["masks_sha256"]
        assert evaluation["provenance"]["source_sha256"] == sha(ROOT / "evaluation/eval_saved_pair.py")
        rows = {}
        for method in evaluation["methods"]:
            if method == "single_z":
                iou = sum(v["mean_iou"] for v in single_audit["tasks"].values()) / len(single_audit["tasks"])
            else:
                iou = sum(audit[profile]["structural"][method][str(g)]["mean_iou"]
                          for g in source["heldout_gaps"]) / len(source["heldout_gaps"])
            rows[method] = {**evaluation["methods"][method], "mean_ideal_iou": iou}
        summary["profiles"][profile] = {"methods": rows, "comparisons": evaluation["comparisons"],
                                        "heldout_gaps": source["heldout_gaps"],
                                        "evaluation_sha256": sha(folder / "evaluation_with_single/summary.json")}
        for column, (key, title) in enumerate((("mean_acc", "Fresh MLP accuracy"), ("mean_ideal_iou", "Mask IoU with ideal"))):
            axis = axes[row_index, column]
            values = [rows[m][key] for m in methods]
            bars = axis.bar(range(len(methods)), values, color=["#777777", "#1f77b4", "#73a8cc", "#e68a32", "#2ca02c", "#aaaaaa", "#222222"])
            axis.bar_label(bars, fmt="%.3f", fontsize=8)
            axis.set(xticks=range(len(methods)), xticklabels=labels, ylim=(0, 1.08), title=f"{profile}: {title}")
            axis.tick_params(axis="x", labelrotation=25)
            axis.grid(axis="y", alpha=.2)
        search_masks = torch.load(folder / "search/masks.pt", weights_only=True, map_location="cpu")
        single_masks = torch.load(folder / "single_z/masks.pt", weights_only=True, map_location="cpu")
        for gap_ordinal, gap in enumerate(source["heldout_gaps"]):
            task_index = next(i for i, t in enumerate(single_masks["tasks"]) if t.endswith(f"G{gap:02d}"))
            task = single_masks["tasks"][task_index]
            gold = ideal_mask(task).float()
            gap_index = search_masks["gaps"].tolist().index(gap)
            masks = [search_masks["masks"][m][gap_index, 0].reshape(16, 16) for m in ("prior1", "agreement1")]
            masks += [single_masks["masks"][task_index, 0].reshape(16, 16), gold]
            for axis, mask, label in zip(example_axes[row_index * 2 + gap_ordinal], masks, ("Prior42", "Pair42", "Task z42", "Ideal")):
                iou = best_permutation_iou(mask, gold)["iou"]
                axis.imshow(align_columns(gold, mask), cmap="Greys", vmin=0, vmax=1)
                axis.set(title=f"{label}, IoU={iou:.3f}", xticks=[], yticks=[])
            example_axes[row_index * 2 + gap_ordinal, 0].set_ylabel(f"{profile}, gap {gap}")
    figure.savefig(root / "comparison.png", dpi=170)
    figure.savefig(root / "comparison.pdf")
    examples.suptitle("First task and latent ordinal 0 per gap; gold alignment for display only")
    examples.savefig(root / "comparison_masks.png", dpi=170)
    plt.close(figure)
    plt.close(examples)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/decoder_agreement/20260906")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.root)
