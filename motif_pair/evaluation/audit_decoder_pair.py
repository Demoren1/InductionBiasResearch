"""Independently re-decode pair-search witnesses and audit gap fidelity."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from data.generate import ideal_mask
from evaluation.eval_generated_masks import _load_model
from evaluation.oracle_ideal import align_columns, soft_topk, hard_topk
from evaluation.structural import best_permutation_iou


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def run(root):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for re-decoding with original precision")
    protocol = json.loads((root / "protocol.json").read_text())
    outputs = {}
    targets = {g: ideal_mask(config.Task("000", "001", g)).float() for g in config.GAPS}
    for profile, source in protocol["profiles"].items():
        path = root / profile / "search/masks.pt"
        saved = torch.load(path, weights_only=True, map_location="cpu")
        summary = json.loads(path.with_name("summary.json").read_text())
        assert saved["metadata"] == summary["metadata"]
        metadata = saved["metadata"]
        assert metadata["protocol_sha256"] == sha(root / "protocol.json")
        assert metadata["protocol_profile"] == profile
        assert metadata["source_sha256"] == sha(ROOT / "evaluation/decoder_pair_search.py")
        gaps = saved["gaps"].tolist()
        assert gaps == protocol["search"]["gaps"]
        n = protocol["search"]["n_starts"]
        models = []
        for number in (1, 2):
            assert sha(metadata[f"checkpoint{number}"]) == metadata[f"checkpoint{number}_sha256"]
            model, _ = _load_model(Path(metadata[f"checkpoint{number}"]), torch.device("cuda"))
            model.requires_grad_(False)
            models.append(model)
        c = models[0].condition([config.Task("000", "001", g) for g in gaps], device="cuda").repeat_interleave(n, 0)
        decoded_count = 0
        pair = {}
        for stage, prefix in (("prior", "prior"), ("agreement", "agreement"), ("random_search", "random_search")):
            soft, hard = [], []
            for number, model in enumerate(models, 1):
                z = saved["latents"][stage][f"z{number}"].to("cuda")
                assert (z.norm(dim=1) <= 8.00002).all()
                logits = model.decode(z, c)
                hard.append(hard_topk(logits, 96).reshape(-1, 16, 16))
                soft.append(soft_topk(logits, 96, .5).reshape(-1, 16, 16))
                assert torch.equal(hard[-1].cpu().reshape(len(gaps), n, 256), saved["masks"][f"{prefix}{number}"])
                decoded_count += len(z)
            mse = (soft[0] - align_columns(soft[0], soft[1])).square().mean((1, 2)).cpu().reshape(len(gaps), n)
            torch.testing.assert_close(mse, saved["pair_stats"][stage]["soft_pair_loss"], rtol=1e-5, atol=1e-8)
            intersections = (hard[0] * align_columns(hard[0], hard[1])).sum((1, 2)).cpu().reshape(len(gaps), n)
            iou = intersections / (192 - intersections)
            torch.testing.assert_close(iou, saved["pair_stats"][stage]["hard_pair_iou"], rtol=0, atol=1e-7)
            pair[stage] = {str(g): {"mean_mse": float(mse[i].mean()), "mean_pair_iou": float(iou[i].mean()),
                                    "exact_pair_count": int((iou[i] == 1).sum()),
                                    "mean_norm1": float(saved["pair_stats"][stage]["norm1"][i].mean()),
                                    "mean_norm2": float(saved["pair_stats"][stage]["norm2"][i].mean())}
                           for i, g in enumerate(gaps)}
        assert (saved["pair_stats"]["agreement"]["soft_pair_loss"] <=
                saved["pair_stats"]["prior"]["soft_pair_loss"] + 1e-7).all()
        structural = {}
        for method, masks in saved["masks"].items():
            structural[method] = {}
            for index, gap in enumerate(gaps):
                values = torch.tensor([best_permutation_iou(m.reshape(16, 16), targets[gap])["iou"]
                                       for m in masks[index]])
                assert abs(float(values.double().mean()) - summary["metrics"][method][str(gap)]["mean_best_permutation_iou"]) < 1e-7
                entry = {"mean_iou": float(values.mean()), "max_iou": float(values.max()),
                         "exact_ideal_count": int((values == 1).sum())}
                if method in ("prior1", "agreement1", "random_search1"):
                    wrong_targets = [g for g in gaps if min(g, 16-g) != min(gap, 16-gap)]
                    other = {g: sum(best_permutation_iou(m.reshape(16, 16), targets[g])["iou"]
                                    for m in masks[index]) / n for g in wrong_targets}
                    strongest = max(other, key=other.get)
                    wrong_cond = summary["metrics"][method][str(gap)]["wrong_condition"]
                    entry.update(strongest_wrong_ideal_gap=strongest,
                                 strongest_wrong_ideal_iou=other[strongest],
                                 correct_minus_best_wrong_ideal=float(values.mean()) - other[strongest],
                                 same_z_wrong_conditions={g: r["mean_best_permutation_iou"] for g, r in wrong_cond.items()},
                                 correct_minus_best_wrong_condition=float(values.mean()) - max(r["mean_best_permutation_iou"] for r in wrong_cond.values()))
                structural[method][str(gap)] = entry
        outputs[profile] = {"masks_sha256": sha(path), "redecoded_count": decoded_count,
                            "pair": pair, "structural": structural}
        print(profile, "redecoded", decoded_count, flush=True)
    (root / "structural_audit.json").write_text(json.dumps(outputs, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/decoder_agreement/20260906")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    run(args.root)
