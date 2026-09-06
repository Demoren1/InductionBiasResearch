"""Read-only stage diagnostics: selected importance -> posterior-mean reconstruction.

No model training, latent optimization or source-artifact mutation. Gold is
used solely for diagnostic evaluation, on meta-train tasks of each profile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from data.generate import ideal_mask
from evaluation.eval_generated_masks import _load_model, _check_provenance
from evaluation.oracle_ideal import align_columns, hard_topk
from evaluation.structural import best_permutation_iou
from models.cvae import (canonicalize_hidden_columns, load_top_importance,
                         read_split_provenance, checkpoint_condition_metadata,
                         make_loaders, kl_per_latent_dim)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def piou(masks, targets):
    aligned = align_columns(targets, masks)
    intersection = (aligned * targets).sum((1, 2))
    union = masks.sum((1, 2)) + targets.sum((1, 2)) - intersection
    return intersection / union.clamp_min(1)


def score(scores, target):
    hard = hard_topk(scores.flatten(1), 96).reshape_as(target)
    aligned = align_columns(target, scores)
    values = scores.flatten(1)
    cutoff = values.topk(97, dim=1).values
    return hard, {
        "iou_gold": piou(hard, target),
        "gold_mass_fraction": (aligned * target).sum((1, 2)) / scores.sum((1, 2)).clamp_min(1e-12),
        "positive_count": (values > 0).sum(1).float(),
        "top96_zero_count": ((scores == 0) & (hard == 1)).sum((1, 2)).float(),
        "cutoff_tied": (cutoff[:, 95] == cutoff[:, 96]).float(),
    }


def reduce_metrics(metrics, subset):
    result = {name: float(value[subset].mean()) for name, value in metrics.items()}
    iou = metrics["iou_gold"][subset]
    result.update(n=int(subset.sum()), iou_min=float(iou.min()),
                  iou_max=float(iou.max()), exact_count=int((iou == 1).sum()))
    return result


@torch.no_grad()
def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Run this diagnostic on CUDA outside the sandbox")
    device = torch.device("cuda")
    protocol = json.loads(args.protocol.read_text())
    source = protocol["profiles"][args.profile]
    assert sha(source["checkpoint"]) == source["checkpoint_sha256"]
    assert sha(source["split"]) == source["split_sha256"]
    if args.out.exists():
        raise FileExistsError(args.out)
    split = read_split_provenance(Path(source["split"]))
    tasks = split["split_train_tasks"]
    model, checkpoint = _load_model(Path(source["checkpoint"]), device)
    model.requires_grad_(False)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    _check_provenance(checkpoint, split, "CVAE", importance_name="importance.pt", top_frac=.1)
    assert checkpoint_condition_metadata(checkpoint) == config.condition_metadata("scalar")
    bank_root = Path(source["split"]).parent / "checkpoints"
    x, c, sources = load_top_importance(tasks, bank_root, device=device,
                                       expected_provenance=split, condition_encoding="scalar")
    assert len(sources) == len(checkpoint["sources"])
    raw, indices, gaps, task_index, source_hashes = [], [], [], [], []
    offset = 0
    for ordinal, (actual, original) in enumerate(zip(sources, checkpoint["sources"])):
        path = Path(actual["source"])
        original_path = Path(original["source"])
        if not original_path.is_absolute():
            original_path = ROOT / original_path
        assert path.resolve() == original_path.resolve()
        assert actual["task"] == original["task"] and actual["n_selected"] == original["n_selected"]
        payload = torch.load(path, weights_only=True, map_location=device)
        assert "top_fraction" not in payload
        n = max(1, int(len(payload["importance"]) * .1))
        selection = payload["val_loss"].flatten().topk(n, largest=False).indices
        maps = payload["importance"][selection].float()
        assert torch.equal(canonicalize_hidden_columns(maps).flatten(1), x[offset:offset + n])
        raw.append(maps)
        indices.append(payload["global_idx"][selection])
        gap = config.parse_task(actual["task"]).gap
        gaps.extend([gap] * n)
        task_index.extend([ordinal] * n)
        source_hashes.append({"path": str(path.resolve()), "sha256": sha(path)})
        offset += n
    raw = torch.cat(raw)
    canonical = x.reshape_as(raw)
    # A permutation must preserve all continuous values exactly.
    assert torch.equal(raw.flatten(1).sort(1).values, canonical.flatten(1).sort(1).values)
    gaps = torch.tensor(gaps, device=device)
    train_loader, val_loader = make_loaders(x, c, 256, checkpoint["seed"], val_fraction=.15)
    n_val = len(val_loader.dataset)
    assert n_val == checkpoint["internal_val_metrics"]["n_examples"]
    assert len(train_loader.dataset) == checkpoint["internal_train_metrics"]["n_examples"]
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(checkpoint["seed"]))
    is_val = torch.zeros(len(x), dtype=torch.bool, device=device)
    is_val[perm[:n_val].to(device)] = True
    assert torch.equal(x[perm[:n_val]], val_loader.dataset.tensors[0])
    assert torch.equal(x[perm[n_val:]], train_loader.dataset.tensors[0])
    mu, logvar = model.encode(x, c)
    logits = model.decode(mu, c)
    assert torch.equal(logits, model.decode(mu, c))
    z0_logits = model.decode(torch.zeros_like(mu), c)
    recon = logits.sigmoid().reshape_as(raw)
    z0 = z0_logits.sigmoid().reshape_as(raw)
    target = torch.stack([ideal_mask(config.Task("000", "001", int(g))) for g in gaps.cpu()]).to(device).float()
    # Gap-mean control uses only internal-training examples, including when
    # evaluating internal validation. It is not the old all-meta-train baseline.
    gap_mean = torch.empty_like(canonical)
    for gap in gaps.unique():
        subset = gaps == gap
        # PyTorch 2.3.1 deterministic CUDA index_put cannot broadcast [16,16]
        # here; explicit contiguous expansion preserves the same calculation.
        mean = canonical[subset & ~is_val].mean(0)
        gap_mean[subset] = mean.expand(int(subset.sum()), -1, -1).contiguous()
    stages = {"raw": raw, "canonical": canonical, "posterior_mu": recon,
              "z0": z0, "internal_train_gap_mean": gap_mean}
    hard, metrics = {}, {}
    for name, value in stages.items():
        hard[name], metrics[name] = score(value, target)
    # Raw/canonical top-k may disagree because zero-score ties are resolved in
    # different column orders. Positive support and continuous mass avoid this.
    raw_positive_iou = piou((raw > 0).float(), target)
    canonical_positive_iou = piou((canonical > 0).float(), target)
    torch.testing.assert_close(raw_positive_iou, canonical_positive_iou, rtol=0, atol=1e-7)
    torch.testing.assert_close(metrics["raw"]["gold_mass_fraction"],
                               metrics["canonical"]["gold_mass_fraction"], rtol=0, atol=1e-6)
    input_hard = hard["canonical"]
    for name, value in stages.items():
        intersection = (hard[name] * input_hard).sum((1, 2))
        metrics[name]["iou_input_fixed_columns"] = intersection / (192 - intersection)
        metrics[name]["iou_input_permuted"] = piou(hard[name], input_hard)
        metrics[name]["mse_input"] = (value - canonical).square().mean((1, 2))
    for name, values in (("posterior_mu", logits), ("z0", z0_logits)):
        metrics[name]["bce_input_sum"] = F.binary_cross_entropy_with_logits(values, x, reduction="none").sum(1)
    kl = kl_per_latent_dim(mu, logvar).sum(1)
    val_kl = float(kl[is_val].mean())
    stored_kl = checkpoint["internal_val_metrics"]["kl"]
    assert abs(val_kl - stored_kl) < 1e-4, (val_kl, stored_kl)
    groups = {}
    for gap in sorted(set(gaps.cpu().tolist())) + ["all"]:
        gap_subset = torch.ones_like(is_val) if gap == "all" else gaps == gap
        groups[str(gap)] = {}
        for name, subset in (("all", gap_subset), ("train", gap_subset & ~is_val), ("val", gap_subset & is_val)):
            groups[str(gap)][name] = {stage: reduce_metrics(value, subset) for stage, value in metrics.items()}
            delta = metrics["posterior_mu"]["iou_gold"] - metrics["canonical"]["iou_gold"]
            groups[str(gap)][name]["paired_change"] = {
                "mean_iou_delta": float(delta[subset].mean()),
                "fraction_improved": float((delta[subset] > 1e-7).float().mean()),
                "fraction_worse": float((delta[subset] < -1e-7).float().mean()),
            }
    # Independent existing metric on every saved hard mask.
    for name, masks in hard.items():
        checked = torch.tensor([best_permutation_iou(m, t)["iou"]
                                for m, t in zip(masks.cpu(), target.cpu())])
        torch.testing.assert_close(checked, metrics[name]["iou_gold"].cpu(), rtol=0, atol=1e-7)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert sha(source["checkpoint"]) == source["checkpoint_sha256"]
    metadata = {"profile": args.profile, "source": source,
                "source_files": source_hashes,
                "diagnostic_source_sha256": sha(__file__),
                "torch_version": str(torch.__version__), "device": torch.cuda.get_device_name(),
                "reconstruction": "sigmoid(decode(encode(x,c).mu,c)); no posterior sampling",
                "n_examples": len(x), "internal_train": len(x) - n_val, "internal_val": n_val,
                "internal_split_seed": checkpoint["seed"], "internal_val_fraction": .15,
                "internal_val_kl_recomputed": val_kl, "internal_val_kl_checkpoint": stored_kl,
                "checks": ["source-selection matches production loader", "canonicalization preserves all values",
                           "positive-support PIoU invariant", "continuous gold-mass invariant",
                           "internal split matches loaders and checkpoint counts/KL",
                           "deterministic repeated posterior decode", "independent hard PIoU for all five stages",
                           "decoder state and source checkpoint unchanged"]}
    args.out.mkdir(parents=True)
    def cpu_dict(data):
        return {k: cpu_dict(v) if isinstance(v, dict) else v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}
    torch.save(cpu_dict({"metadata": metadata, "stages": stages, "hard": hard,
                        "metrics": metrics, "target": target, "condition": c,
                        "posterior_mu": mu, "posterior_logvar": logvar,
                        "gaps": gaps, "task_index": torch.tensor(task_index),
                        "tasks": tasks, "global_candidate_index": torch.cat(indices),
                        "is_internal_val": is_val, "raw_positive_iou": raw_positive_iou}), args.out / "diagnostic.pt")
    (args.out / "summary.json").write_text(json.dumps({"metadata": metadata, "groups": groups}, indent=2) + "\n")
    print(args.profile, "n", len(x), "validation KL", val_kl, "stored", stored_kl)
    for gap, subsets in groups.items():
        group = subsets["all"]
        print(gap, {s: round(group[s]["iou_gold"], 4) for s in stages},
              "delta", round(group["paired_change"]["mean_iou_delta"], 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("interp", "extrap"), required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "outputs/oracle_ideal/seed_20260906/protocol.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    run(args)
