"""Validate saved oracle witnesses and plot the two gap-OOD profiles."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data.generate import ideal_mask
from evaluation.eval_generated_masks import _load_model
from evaluation.structural import best_permutation_iou


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    protocol = json.loads((root / "protocol.json").read_text())
    all_records, summaries = {}, {}
    common_start = None
    checked = 0
    for profile, source in protocol["profiles"].items():
        assert sha(source["checkpoint"]) == source["checkpoint_sha256"]
        assert sha(source["split"]) == source["split_sha256"]
        model, _ = _load_model(Path(source["checkpoint"]), device)
        model.requires_grad_(False)
        for radius in protocol["radii"]:
            key = f"{profile}_radius_{radius:g}"
            folder = root / key
            saved = torch.load(folder / "result.pt", weights_only=True, map_location="cpu")
            summary = json.loads((folder / "summary.json").read_text())
            assert saved["metadata"] == summary["metadata"]
            assert saved["metadata"]["checkpoint_sha256"] == source["checkpoint_sha256"]
            assert saved["metadata"]["split_sha256"] == source["split_sha256"]
            assert saved["metadata"]["protocol_sha256"] == sha(root / "protocol.json")
            assert saved["metadata"]["source_sha256"] == sha(Path(__file__).with_name("oracle_ideal.py"))
            if common_start is None:
                common_start = saved["initial_z"]
            assert torch.equal(common_start, saved["initial_z"])
            for gap in protocol["gaps"]:
                record = saved["results"][str(gap)]
                assert record["decoder_unchanged"]
                task = config.Task("000", "001", gap)
                target = ideal_mask(task).cpu().float()
                torch.testing.assert_close(record["target"], target.expand_as(record["target"]), rtol=0, atol=0)
                torch.testing.assert_close(record["condition"],
                                           torch.full_like(record["condition"], (gap - 3) / 7),
                                           rtol=0, atol=1e-7)
                for stage in ("initial", "best_soft", "best_hard"):
                    value = record[stage]
                    assert len(value["z"]) == protocol["n_starts"]
                    assert (value["z"].norm(dim=1) <= radius + 2e-5).all()
                    with torch.no_grad():
                        z = value["z"].to(device)
                        c = model.condition([task], device=device).expand(len(z), -1)
                        logits = model.decode(z, c)
                        hard = torch.zeros_like(logits).scatter(1, logits.topk(96, dim=1).indices, 1)
                    assert torch.equal(hard.cpu().reshape_as(value["hard"]), value["hard"])
                    assert (value["hard"].sum((1, 2)) == 96).all()
                    # Independent existing evaluator: different assignment implementation.
                    expected_iou = torch.tensor([best_permutation_iou(mask, target)["iou"]
                                                 for mask in value["hard"]])
                    torch.testing.assert_close(expected_iou, value["iou"], rtol=0, atol=1e-7)
                    stats = summary["gaps"][str(gap)]["prior" if stage == "initial" else stage]
                    assert abs(stats["mean_iou"] - float(expected_iou.mean())) < 1e-7
                    assert stats["exact_count"] == int((expected_iou == 1).sum())
                    checked += len(z)
                assert (record["best_soft"]["loss"] <= record["initial"]["loss"] + 1e-7).all()
                assert (record["best_hard"]["iou"] >= record["initial"]["iou"]).all()
            all_records[key], summaries[key] = saved, summary
    manifest = {"protocol": protocol, "runs": summaries,
                "validation": {"redecoded_masks": checked, "independent_iou": True,
                               "common_initial_z": True, "checkpoint_hashes_unchanged": True},
                "result_sha256": {key: sha(root / key / "result.pt") for key in all_records}}
    (root / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for axis, (profile, source) in zip(axes, protocol["profiles"].items()):
        for radius in protocol["radii"]:
            result = summaries[f"{profile}_radius_{radius:g}"]["gaps"]
            for stage, style in (("best_soft", "o-"), ("best_hard", "x--")):
                axis.plot(protocol["gaps"], [result[str(g)][stage]["mean_iou"] for g in protocol["gaps"]],
                          style, label=f"r={radius}, {stage}")
        result = summaries[f"{profile}_radius_{protocol['radii'][0]:g}"]["gaps"]
        axis.plot(protocol["gaps"], [result[str(g)]["prior"]["mean_iou"] for g in protocol["gaps"]],
                  "k:", label="Initial prior")
        for gap in source["heldout_gaps"]:
            axis.axvspan(gap - .18, gap + .18, color="orange", alpha=.18)
        axis.set(title=f"{profile}; shaded = held-out gaps", xlabel="Fixed gap",
                 ylabel="Mean IoU with ideal after column assignment", ylim=(0, 1.02))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(root / "reachability.png", dpi=170)
    fig.savefig(root / "reachability.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for row, profile in enumerate(protocol["profiles"]):
        for column, metric in enumerate(("soft_loss_mean", "hard_iou_mean")):
            axis = axes[row, column]
            for radius in protocol["radii"]:
                records = all_records[f"{profile}_radius_{radius:g}"]["results"]
                for gap in protocol["profiles"][profile]["heldout_gaps"]:
                    history = records[str(gap)]["history"]
                    axis.plot([v["step"] for v in history], [v[metric] for v in history],
                              label=f"gap={gap}, r={radius}")
            axis.set(title=profile, xlabel="Adam updates", ylabel=metric)
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    fig.savefig(root / "convergence.png", dpi=170)
    plt.close(fig)

    rows = [(profile, gap) for profile, source in protocol["profiles"].items()
            for gap in source["heldout_gaps"]]
    fig, axes = plt.subplots(len(rows), 4, figsize=(10, 10), layout="constrained")
    for row, (profile, gap) in enumerate(rows):
        records = all_records[f"{profile}_radius_8"]["results"][str(gap)]
        best = int(records["best_hard"]["iou"].argmax())
        images = [records["initial"]["aligned_hard"][0],
                  records["best_soft"]["aligned_hard"][0],
                  records["best_hard"]["aligned_hard"][best],
                  ideal_mask(config.Task("000", "001", gap))]
        titles = [f"Prior start 0: {records['initial']['iou'][0]:.3f}",
                  f"Oracle start 0: {records['best_soft']['iou'][0]:.3f}",
                  f"Best witness: {records['best_hard']['iou'][best]:.3f}", "Ideal"]
        for axis, matrix, title in zip(axes[row], images, titles):
            axis.imshow(matrix, cmap="Greys", vmin=0, vmax=1)
            axis.set(title=title, xticks=[], yticks=[])
        axes[row, 0].set_ylabel(f"{profile}, gap={gap}")
    fig.suptitle("Gold used for optimization and display alignment; radius 8")
    fig.savefig(root / "heldout_masks.png", dpi=170)
    plt.close(fig)
    print(json.dumps(manifest["validation"], indent=2))
    for key, summary in summaries.items():
        print(key)
        for gap, result in summary["gaps"].items():
            print(gap, result["label"], *(round(result[s]["mean_iou"], 4)
                  for s in ("prior", "best_soft", "best_hard")),
                  "max", round(result["best_hard"]["max_iou"], 4),
                  "exact", result["best_hard"]["exact_count"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(2)
    run(args.root, torch.device(args.device))
