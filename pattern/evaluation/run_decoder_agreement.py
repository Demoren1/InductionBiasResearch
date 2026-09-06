"""Run and evaluate label-free agreement of two frozen pattern VAE decoders.

Search writes immutable mask artifacts before evaluation accesses the ideal
support or held-out task data. All 64 starts are evaluated; none are selected
using task performance or gold structure. See decoder_agreement.py for the
objective and its differentiable fixed-cardinality relaxation.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from models.cvae import CVAE
from evaluation.decoder_agreement import (
    align_columns, hard_topk, optimize_agreement, soft_topk,
)


DEFAULT_ROOT = config.OUTPUTS / "decoder_agreement" / "seed_20260906"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def load_models(root, device):
    protocol = json.loads((root / "protocol.json").read_text())
    split_path = config.ROOT.parent / protocol["split_source"]
    split = json.loads(split_path.read_text())
    if set(split["train_patterns"]) & set(protocol["evaluation"]["patterns"]):
        raise ValueError("Meta-train and evaluation task identities overlap")
    if split["test_patterns"] != protocol["evaluation"]["patterns"]:
        raise ValueError("Protocol evaluation tasks do not match the declared split")
    models, provenance = [], []
    for seed in protocol["model_seeds"]:
        folder = root / f"vae_{seed}"
        checkpoint = folder / "cvae_best.pt"
        meta = torch.load(folder / "cvae_meta.pt", map_location="cpu", weights_only=True)
        expected = {**protocol["vae"], "seed": seed,
                    "map_split_seed": protocol["map_split_seed"],
                    "patterns": split["train_patterns"]}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"{folder}: metadata {key}={meta.get(key)!r}, expected {value!r}")
        model = CVAE(config.MASK_DIM, config.LATENT_DIM, config.CVAE_HIDDEN)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.to(device).eval().requires_grad_(False)
        models.append(model)
        provenance.append({"seed": seed, "checkpoint": str(checkpoint),
                           "sha256": file_hash(checkpoint), "metadata": meta,
                           "task_split_sha256": file_hash(split_path)})
    if len(models) != 2 or len(set(protocol["model_seeds"])) != 2:
        raise ValueError("Exactly two distinct model seeds are required")
    return models, provenance


def decode(model, z, temperature):
    logits = model.decode(z, z.new_empty(len(z), 0))
    return soft_topk(logits, config.K_ACTIVE, temperature).reshape(-1, 8, 8), logits


def project(z, radius):
    return z * (radius / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)


@torch.no_grad()
def random_search(models, search, device):
    """Same number of latent-pair proposals per start, excluding gradients."""
    n, proposals = search["n_starts"], search["steps"] + 1
    generator = torch.Generator(device=device).manual_seed(search["seed"] + 1)
    best = torch.full((n,), float("inf"), device=device)
    best_z = [torch.empty(n, config.LATENT_DIM, device=device) for _ in models]
    # Evaluate 16 proposals/start per batch; retain the best by agreement only.
    for offset in range(0, proposals, 16):
        count = min(16, proposals - offset)
        zs = [project(torch.randn(count * n, config.LATENT_DIM,
                                  generator=generator, device=device), search["latent_radius"])
              for _ in models]
        masks = [decode(m, z, search["temperature"])[0] for m, z in zip(models, zs)]
        loss = (masks[0] - align_columns(masks[0], masks[1])).square().mean((1, 2))
        values, indices = loss.reshape(count, n).min(dim=0)
        improved = values < best
        best[improved] = values[improved]
        for target, z in zip(best_z, zs):
            candidate = z.reshape(count, n, -1)[indices, torch.arange(n, device=device)]
            target[improved] = candidate[improved]
        if offset % 160 == 0:
            print(f"[random-search] {offset + count}/{proposals} proposals/start", flush=True)
    result = {"loss": best.cpu(), "proposals_per_start": proposals}
    for i, (model, z) in enumerate(zip(models, best_z), 1):
        soft, logits = decode(model, z, search["temperature"])
        result[f"z{i}"] = z.cpu()
        result[f"masks{i}"] = hard_topk(logits, config.K_ACTIVE).reshape(n, 8, 8).cpu()
        result[f"soft{i}"] = soft.cpu()
    return result


def search_stage(root, device):
    protocol = json.loads((root / "protocol.json").read_text())
    search = protocol["search"]
    if (root / "masks.pt").exists():
        raise FileExistsError("masks.pt already exists; use --stage evaluate or a new output root")
    models, provenance = load_models(root, device)
    result = optimize_agreement(*models, n_starts=search["n_starts"],
                               steps=search["steps"], lr=search["lr"], seed=search["seed"],
                               temperature=search["temperature"], radius=search["latent_radius"],
                               device=device)
    torch.save(result, root / "optimization.pt")
    random_result = random_search(models, search, device)
    torch.save(random_result, root / "random_search.pt")
    masks = {
        "initial_vae1": result["initial_masks1"], "initial_vae2": result["initial_masks2"],
        "optimized_vae1": result["final_masks1"], "optimized_vae2": result["final_masks2"],
        "random_search_vae1": random_result["masks1"],
        "random_search_vae2": random_result["masks2"],
    }
    # The generated masks and provenance are fixed before any gold/task evaluation.
    torch.save(masks, root / "masks.pt")
    dump_json(root / "search_provenance.json", {"protocol_sha256": file_hash(root / "protocol.json"),
              "models": provenance, "mask_sha256": file_hash(root / "masks.pt"),
              "source_sha256": {name: file_hash(Path(__file__).parent / name)
                                for name in ("decoder_agreement.py", "run_decoder_agreement.py",
                                             "train_agreement_vaes.py")},
              "device": str(device), "torch_version": str(torch.__version__)})
    print("[search] masks frozen and saved; gold and held-out data were not accessed", flush=True)


def describe(values):
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std()),
            "min": float(array.min()), "max": float(array.max()), "values": array.tolist()}


def structure(masks):
    from data.generate import ideal_mask
    gold = ideal_mask().float().unsqueeze(0).expand(len(masks), -1, -1)
    aligned = align_columns(gold, masks.float())
    intersection = (gold * aligned).sum((1, 2))
    iou = intersection / (64 - intersection)
    windows = torch.zeros(config.N_WINDOWS, config.SEQ_LEN)
    for w in range(config.N_WINDOWS):
        windows[w, w:w + config.PATTERN_LEN] = 1
    # Exact-window coverage avoids arbitrary nearest-window tie breaks.
    exact = (masks.transpose(1, 2).unsqueeze(2) == windows[None, None]).all(-1)
    return {"iou": describe(iou.numpy()),
            "unique_masks": len(torch.unique(masks.flatten(1), dim=0)),
            "exact_window_columns_fraction": float(exact.any(-1).float().mean()),
            "mean_distinct_exact_windows": float(exact.any(1).sum(1).float().mean()),
            "fraction_masks_covering_each_exact_window": exact.any(1).float().mean(0).tolist(),
            "mean_row_coverage": masks.sum(2).mean(0).tolist()}


def pair_diagnostics(a, b):
    aligned = align_columns(a, b)
    intersection = (a * aligned).sum((1, 2))
    return {"iou": describe((intersection / (64 - intersection)).numpy()),
            "hamming": describe((a - aligned).abs().sum((1, 2)).numpy()),
            "exact_matches": int((a == aligned).all(2).all(1).sum())}


def eval_one_task(masks, pattern, settings, device):
    """Fresh MLPs, matched initial weights and training stream across methods."""
    import torch.nn.functional as F
    from data.generate import make_dataset
    from models.mlp import BatchedMaskedMLP, get_train_batch
    names = list(masks)
    n = settings["n_masks_per_method"]
    torch.manual_seed(settings["seed"] + int(pattern, 2))
    base = BatchedMaskedMLP(n, config.SEQ_LEN, config.H).to(device)
    model = BatchedMaskedMLP(n * len(names), config.SEQ_LEN, config.H).to(device)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            template = dict(base.named_parameters())[name]
            parameter.copy_(template.repeat((len(names),) + (1,) * (parameter.ndim - 1)))
    model.load_masks(torch.cat([masks[name] for name in names]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    for step in range(settings["steps"]):
        x, y = get_train_batch(pattern, settings["batch_size"], int(pattern, 2) * 10_003 + n + step)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        # Each method has the same gradient scale as its standalone n-mask run.
        loss = F.binary_cross_entropy_with_logits(logits, y[:, None].expand_as(logits)) * len(names)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 500 == 0:
            print(f"[eval] pattern={pattern} {step+1}/{settings['steps']}", flush=True)
    data = make_dataset(pattern, config.N_VAL_SAMPLES, 1000 + int(pattern, 2), config.POS_FRACTION)
    x, y = data["x"].to(device), data["y"].to(device)
    bce = model.val_loss(x, y, config.VAL_BATCH_SIZE).cpu().reshape(len(names), n)
    acc = model.val_acc(x, y, config.VAL_BATCH_SIZE).cpu().reshape(len(names), n)
    return {name: {"accuracy": describe(acc[i].numpy()), "bce": describe(bce[i].numpy())}
            for i, name in enumerate(names)}


def evaluate_stage(root, device):
    from data.generate import ideal_mask
    from models.mlp import generate_fixed_sparsity_masks
    protocol = json.loads((root / "protocol.json").read_text())
    provenance = json.loads((root / "search_provenance.json").read_text())
    assert file_hash(root / "protocol.json") == provenance["protocol_sha256"]
    assert file_hash(root / "masks.pt") == provenance["mask_sha256"]
    masks = torch.load(root / "masks.pt", map_location="cpu", weights_only=True)
    n = protocol["search"]["n_starts"]
    masks["random_exact32"] = generate_fixed_sparsity_masks(n, 8, 8, 32, protocol["search"]["seed"] + 2)
    masks["ideal"] = ideal_mask().float().unsqueeze(0).repeat(n, 1, 1)
    for value in masks.values():
        assert value.shape == (n, 8, 8) and (value.sum((1, 2)) == 32).all()
        assert ((value == 0) | (value == 1)).all()
    summary = {"protocol": protocol, "structure": {k: structure(v) for k, v in masks.items()},
               "agreement": {name: pair_diagnostics(masks[f"{name}_vae1"], masks[f"{name}_vae2"])
                             for name in ("initial", "optimized", "random_search")}, "tasks": {}}
    for pattern in protocol["evaluation"]["patterns"]:
        task_path = root / f"eval_{pattern}.json"
        if task_path.exists():
            saved = json.loads(task_path.read_text())
            assert saved["mask_sha256"] == provenance["mask_sha256"]
            assert saved["protocol_sha256"] == provenance["protocol_sha256"]
            result = saved["methods"]
        else:
            result = eval_one_task(masks, pattern, protocol["evaluation"], device)
            dump_json(task_path, {"mask_sha256": provenance["mask_sha256"],
                                 "protocol_sha256": provenance["protocol_sha256"], "methods": result})
        summary["tasks"][pattern] = result
    summary["downstream"] = {
        name: {"accuracy": describe([t[name]["accuracy"]["mean"] for t in summary["tasks"].values()]),
               "bce": describe([t[name]["bce"]["mean"] for t in summary["tasks"].values()])}
        for name in masks}
    dump_json(root / "summary.json", summary)
    print(json.dumps({name: {"iou": summary["structure"][name]["iou"]["mean"],
                            "accuracy": summary["downstream"][name]["accuracy"]["mean"]}
                      for name in masks}, indent=2), flush=True)
    render(root, masks, summary)


def render(root, masks, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"initial_vae1": "VAE 1: initial", "initial_vae2": "VAE 2: initial",
              "optimized_vae1": "VAE 1: optimized", "optimized_vae2": "VAE 2: optimized",
              "random_search_vae1": "VAE 1: random search", "random_search_vae2": "VAE 2: random search",
              "random_exact32": "Random exact-32", "ideal": "Ideal support"}
    names = list(masks)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for axis, source, key, title in ((axes[0], summary["structure"], "iou", "Ideal-support IoU (post-hoc)"),
                                    (axes[1], summary["downstream"], "accuracy", "Held-out MLP accuracy")):
        axis.barh(range(len(names)), [source[n][key]["mean"] for n in names], color=["#7295b6"]*2+["#328369"]*2+["#b39162"]*2+["#999999", "#444444"])
        axis.set_yticks(range(len(names)), [labels[n] for n in names])
        axis.invert_yaxis()
        axis.set_xlim(0, 1)
        axis.set_title(title)
        axis.grid(axis="x", alpha=.2)
    for extension in ("png", "pdf"):
        fig.savefig(root / f"summary.{extension}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9), layout="constrained")
    aligned_second = align_columns(masks["optimized_vae1"], masks["optimized_vae2"])
    for row in range(4):
        entries = [masks["initial_vae1"][row], masks["optimized_vae1"][row], aligned_second[row], masks["ideal"][row]]
        for col, matrix in enumerate(entries):
            axes[row, col].imshow(matrix, cmap="Greys", vmin=0, vmax=1)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(f"Start {row}")
    for axis, title in zip(axes[0], ("Initial VAE 1", "Optimized VAE 1", "Optimized VAE 2\nmatched to VAE 1", "Ideal reference\nnot used in search")):
        axis.set_title(title, fontsize=10)
    fig.suptitle("First four starts in fixed order (no quality-based selection)")
    fig.savefig(root / "mask_examples.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage", choices=["search", "evaluate", "all"], default="all")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; run outside the sandbox")
    if args.stage in ("search", "all"):
        search_stage(args.out_dir, device)
    if args.stage in ("evaluate", "all"):
        evaluate_stage(args.out_dir, device)


if __name__ == "__main__":
    main()
