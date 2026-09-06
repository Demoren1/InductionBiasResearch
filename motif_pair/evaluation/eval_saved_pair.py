"""Evaluate fixed saved decoder-agreement masks with fresh paired task MLPs.

Mask search is already complete. No task label, gold score or test metric is
used to choose masks. Reuses the established motif task/data/MLP protocol.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from data.generate import configure_compute_device, make_dataset
from evaluation.eval_generated_masks import _seed, _train_and_eval


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; use authorized GPU outside sandbox")
    device = torch.device("cuda")
    configure_compute_device("cuda")
    protocol = json.loads((args.root / "protocol.json").read_text())
    profile = protocol["profiles"][args.profile]
    split_path = Path(profile["split"])
    assert sha(split_path) == profile["split_sha256"]
    split = json.loads(split_path.read_text())
    mask_path = args.root / args.profile / "search/masks.pt"
    saved = torch.load(mask_path, weights_only=True, map_location="cpu")
    meta = saved["metadata"]
    assert meta["protocol_sha256"] == sha(args.root / "protocol.json")
    assert meta["protocol_profile"] == args.profile
    assert meta["split_sha256"] == profile["split_sha256"]
    assert Path(meta["split"]).resolve() == split_path.resolve()
    assert meta["checkpoint1_sha256"] == profile["checkpoint_sha256"]
    assert sha(meta["checkpoint1"]) == meta["checkpoint1_sha256"]
    assert sha(meta["checkpoint2"]) == meta["checkpoint2_sha256"]
    assert Path(meta["checkpoint2"]).resolve() == (args.root / args.profile / "vae_43/best.pt").resolve()
    assert meta["checkpoint_seeds"] == [42, 43] and meta["loader_seed"] == 42
    for key in ("seed", "n_starts", "steps", "lr", "temperature", "radius"):
        assert meta["settings"][key] == protocol["search"][key]
    assert meta["settings"]["random_proposals"] == protocol["search"]["random_pair_proposals"]
    assert meta["source_sha256"] == sha(ROOT / "evaluation/decoder_pair_search.py")
    gaps = saved["gaps"].tolist() if isinstance(saved["gaps"], torch.Tensor) else saved["gaps"]
    cfg = dict(protocol["evaluation"])
    methods = list(cfg["methods"])
    n = cfg["n_masks"]
    for method in methods:
        values = saved["masks"][method]
        assert values.shape == (len(gaps), n, 256)
        assert ((values == 0) | (values == 1)).all()
        assert (values.sum(-1) == 96).all()
    single = None
    if args.include_single:
        single_path = args.root / args.profile / "single_z/masks.pt"
        single = torch.load(single_path, weights_only=True, map_location="cpu")
        smeta = single["metadata"]
        assert smeta["protocol_sha256"] == sha(args.root / "single_z_protocol.json")
        assert smeta["checkpoint_sha256"] == meta["checkpoint1_sha256"]
        assert smeta["parent_masks_sha256"] == sha(mask_path)
        assert smeta["split_sha256"] == profile["split_sha256"]
        assert smeta["source_sha256"] == sha(ROOT / "evaluation/single_z_mlp.py")
        single_protocol = json.loads((args.root / "single_z_protocol.json").read_text())
        assert smeta["settings"] == single_protocol["settings"]
        assert single_protocol["parent_protocol_sha256"] == sha(args.root / "protocol.json")
        assert single["tasks"] == split["test_tasks"]
        assert single["masks"].shape == (len(single["tasks"]), n, 256)
        assert ((single["masks"] == 0) | (single["masks"] == 1)).all()
        assert (single["masks"].sum(-1) == 96).all()
        methods.append("single_z")
        cfg["methods"] = methods
        cfg["single_z_protocol_sha256"] = sha(args.root / "single_z_protocol.json")
        cfg["single_z_masks_sha256"] = sha(single_path)
    out = args.root / args.profile / ("evaluation_with_single" if args.include_single else "evaluation")
    out.mkdir(parents=True, exist_ok=True)
    provenance = {"mask_sha256": sha(mask_path), "protocol_sha256": sha(args.root / "protocol.json"),
                  "source_sha256": sha(__file__),
                  "evaluator_sha256": sha(ROOT / "evaluation/eval_generated_masks.py"),
                  "split_sha256": sha(split_path), "settings": cfg,
                  "device": torch.cuda.get_device_name(), "torch_version": str(torch.__version__)}
    records = {}
    for ordinal, task in enumerate(split["test_tasks"]):
        destination = out / f"{task}.json"
        if destination.exists():
            result = json.loads(destination.read_text())
            assert result["provenance"] == provenance, "Cannot resume changed masks/protocol/runtime"
            records[task] = result
            continue
        gap_index = gaps.index(config.parse_task(task).gap)
        masks = torch.cat([single["masks"][ordinal] if method == "single_z"
                           else saved["masks"][method][gap_index] for method in methods]).to(device)
        dataset = make_dataset(task, cfg["val_samples"], _seed(task, 20_000), .5)
        bce, acc = _train_and_eval(masks, task, cfg["mlp_steps"], cfg["batch_size"], cfg["lr"],
                                   dataset["x"].to(device), dataset["y"].to(device), paired_group_size=n)
        result = {"task": task, "provenance": provenance, "methods": {}}
        for index, method in enumerate(methods):
            lo, hi = index * n, (index + 1) * n
            result["methods"][method] = {"mean_bce": float(bce[lo:hi].mean()),
                                          "mean_acc": float(acc[lo:hi].mean()),
                                          "bce": bce[lo:hi].cpu().tolist(),
                                          "acc": acc[lo:hi].cpu().tolist()}
        destination.write_text(json.dumps(result, indent=2) + "\n")
        records[task] = result
        print(f"[{args.profile} {ordinal+1}/{len(split['test_tasks'])}] {task} "
              f"prior1={result['methods']['prior1']['mean_acc']:.4f} "
              f"agreement1={result['methods']['agreement1']['mean_acc']:.4f}", flush=True)
    assert sha(mask_path) == provenance["mask_sha256"]
    summary = {"profile": args.profile, "provenance": provenance, "tasks": records,
               "methods": {method: {metric: sum(record["methods"][method][metric]
                                               for record in records.values()) / len(records)
                                     for metric in ("mean_acc", "mean_bce")}
                           for method in methods}}
    # Crossed paired bootstrap: resample the 8 shared motif pairs and the 64
    # common mask/initialization ordinals, keeping both held-out gaps fixed.
    # This is conditional on the selected checkpoints/gaps, not a seed-level CI.
    pairs = sorted({(config.parse_task(t).a, config.parse_task(t).b) for t in records})
    heldout = sorted(profile["heldout_gaps"])
    generator = torch.Generator().manual_seed(20260906)
    pair_draws = torch.randint(len(pairs), (4000, len(pairs)), generator=generator)
    mask_draws = torch.randint(n, (4000, n), generator=generator)
    comparisons = {}
    compare = [("agreement1", "prior1"), ("agreement1", "random_search1"),
               ("agreement1", "conditional_mean"), ("agreement2", "prior2")]
    if args.include_single:
        compare += [("single_z", "prior1"), ("single_z", "agreement1")]
    for candidate, reference in compare:
        deltas = torch.tensor([[[records[config.Task(a, b, gap).id]["methods"][candidate]["acc"][i] -
                                 records[config.Task(a, b, gap).id]["methods"][reference]["acc"][i]
                                 for i in range(n)] for gap in heldout] for a, b in pairs])
        by_pair_mask = deltas.mean(1)
        resampled = by_pair_mask[pair_draws[:, :, None], mask_draws[:, None, :]].mean((1, 2))
        comparisons[f"{candidate}_minus_{reference}"] = {
            "accuracy_delta": float(deltas.mean()),
            "paired_bootstrap_95": torch.quantile(resampled, torch.tensor([.025, .975])).tolist(),
            "scope": "fixed checkpoints and heldout gaps; crossed resampling shared motif pairs and mask ordinals"}
    summary["comparisons"] = comparisons
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"methods": summary["methods"], "comparisons": comparisons}, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "outputs/decoder_agreement/20260906")
    p.add_argument("--profile", choices=("interp", "extrap"), required=True)
    p.add_argument("--include-single", action="store_true", help="evaluate task-loss single-z masks in the same paired MLP batch")
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    run(args)
