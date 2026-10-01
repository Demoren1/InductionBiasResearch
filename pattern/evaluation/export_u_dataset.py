"""Export canonical binary pattern U records without training a generator.

Hidden columns are ordered by the center of their active input positions.
Nonzero sharing codes are ordered by their mean input-minus-hidden offset.
Neither step uses the analytic Toeplitz target. The saved task parameters are
permuted consistently, and reconstructed W is checked for every record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from pattern.evaluation.u_reparameterization_pilot import SharedUModel
from pattern.length11.evaluate import gold_iou


def one_record(directory: Path) -> dict:
    summary = json.loads((directory / "summary.json").read_text())
    if summary["method"] not in ("learned_binary", "offset_assignment") or summary["seq_len"] != 11:
        raise ValueError(f"expected binary length-11 U: {directory}")
    if summary.get("column_quota") or summary.get("free_cardinality"):
        raise ValueError(f"expected global exact-32 support: {directory}")
    model = SharedUModel(summary["method"], summary["seed"], 11)
    saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    with torch.no_grad():
        codes = model.assignment_matrix().argmax(1).reshape(11, 8)
        support = codes != 0
        positions = torch.arange(11)[:, None]
        counts = support.sum(0)
        centers = (support * positions).sum(0) / counts.clamp_min(1)
        centers = torch.where(counts > 0, centers, torch.full_like(centers, 100))
        hidden_permutation = torch.argsort(centers, stable=True)
        aligned = codes[:, hidden_permutation]
        category_offsets = []
        for category in range(1, 5):
            location = torch.where(aligned == category)
            mean_offset = (float((location[0] - location[1]).float().mean())
                           if len(location[0]) else float("inf"))
            category_offsets.append(mean_offset)
        category_order = sorted(range(1, 5), key=lambda code: (category_offsets[code - 1], code))
        canonical = torch.zeros_like(aligned)
        coefficients = torch.zeros_like(model.v)
        for new, old in enumerate(category_order, start=1):
            canonical[aligned == old] = new
            coefficients[:, new] = model.v[:, old]
        reconstructed = coefficients[:, canonical.long()]
        expected = model.first_layer()[:, :, hidden_permutation]
        if not torch.allclose(reconstructed, expected, atol=1e-6):
            raise AssertionError(f"canonical U changed W: {directory}")
        if int((canonical != 0).sum()) != 32:
            raise AssertionError(f"wrong support size: {directory}")
    return {
        "source": str(directory),
        "source_method": summary["method"],
        "seed": summary["seed"],
        "codes": canonical.to(torch.uint8),
        "U_full": F.one_hot(canonical.flatten().long(), 5).to(torch.uint8),
        "U_raw": F.one_hot(codes.flatten().long(), 5).to(torch.uint8),
        "hidden_permutation_new_to_old": hidden_permutation.to(torch.uint8),
        "code_order_new_to_old": torch.tensor([0, *category_order], dtype=torch.uint8),
        "v": coefficients,
        "b1": model.b1.detach()[:, hidden_permutation],
        "w2": model.w2.detach()[:, hidden_permutation],
        "b2": model.b2.detach().clone(),
        "first_layer": reconstructed,
        "toeplitz_iou_evaluation_only": json.loads(
            (directory / "toeplitz_audit.json").read_text())["toeplitz_support_iou"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-offset", action="store_true",
                        help="include learned offset-gated models as full binary U records")
    args = parser.parse_args()
    methods = ("learned_binary", "offset_assignment") if args.include_offset else ("learned_binary",)
    directories = [path for method in methods
                   for path in sorted(args.root.glob(f"{method}_seed[0-9]*"))
                   if path.name.removeprefix(f"{method}_seed").isdigit()]
    if not directories:
        raise ValueError("no direct binary U runs found")
    records = [one_record(path) for path in directories]
    supports = torch.stack([record["codes"] != 0 for record in records])
    frequency = supports.float().mean(0)
    consensus = torch.zeros(88)
    consensus[frequency.flatten().topk(32).indices] = 1
    consensus = consensus.reshape(11, 8)
    pairwise_iou = []
    for first in range(len(supports)):
        for second in range(first + 1, len(supports)):
            intersection = int((supports[first] & supports[second]).sum())
            pairwise_iou.append(intersection / (64 - intersection))
    result = {
        "kind": "canonical_full_binary_U",
        "n_independent_U": len(records),
        "shape": [11, 8],
        "U_full_shape": [88, 5],
        "active_edges": 32,
        "n_nonzero_codes": 4,
        "canonicalization": "hidden columns by support center; code IDs by mean relative offset",
        "gold_used_for_alignment": False,
        "VAE_trained": False,
        "seeds": [record["seed"] for record in records],
        "source_method_counts": {method: sum(record["source_method"] == method
                                      for record in records) for method in methods},
        "unique_supports": len({mask.numpy().tobytes() for mask in supports}),
        "mean_pairwise_canonical_support_iou": (
            sum(pairwise_iou) / len(pairwise_iou) if pairwise_iou else None),
        "mean_individual_toeplitz_iou_evaluation_only": sum(
            record["toeplitz_iou_evaluation_only"] for record in records) / len(records),
        "consensus_toeplitz_iou_evaluation_only": gold_iou(consensus),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"records": records, "support_frequency": frequency,
                "support_consensus_top32": consensus,
                "summary": result}, args.out / "records.pt")
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    main()
