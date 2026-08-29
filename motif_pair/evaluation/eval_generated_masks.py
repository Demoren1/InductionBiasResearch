"""Evaluate structural-prior masks on held-out motif-pair tasks only.

No selected-mask oracle or latent optimisation is implemented here: every
candidate MLP is newly initialised and trained after its mask is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from data.generate import configure_compute_device, make_dataset  # noqa: E402
from evaluation.baselines import ConditionalMean, gold_mask, random_bernoulli, random_exact  # noqa: E402
from evaluation.structural import best_permutation_iou  # noqa: E402
from models.cvae import (CVAE, read_split_provenance, task_condition,
                         verify_generator_config, verify_split_provenance)  # noqa: E402
from models.mlp import BatchedMaskedMLP, get_train_batch  # noqa: E402


METHODS = ("random", "random_exact96", "conditional_mean", "vae", "cvae", "cvae_wrong_gap", "ideal")


def _value(name, default): return getattr(config, name, default)


def _seed(task: str, seed: int) -> int:
    return seed + int.from_bytes(hashlib.sha256(task.encode()).digest()[:4], "little")


def _load_split(path: Path):
    split = json.loads(path.read_text())
    train, test = split.get("train_tasks"), split.get("test_tasks")
    if not isinstance(train, list) or not isinstance(test, list) or not train or not test:
        raise ValueError("split needs nonempty train_tasks and test_tasks")
    if set(train) & set(test): raise ValueError("split train/test tasks overlap")
    return split, train, test


def _load_model(path: Path, device):
    payload = torch.load(path, weights_only=True, map_location=device)
    model = CVAE(payload.get("mask_dim", config.MASK_DIM), payload.get("latent_dim", 32),
                 payload.get("hidden", 256), payload.get("cond_dim", 8))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


def _check_provenance(payload, expected_split, label, *, importance_name: str,
                      top_frac: float):
    verify_split_provenance(payload, expected_split, label=f"{label} checkpoint",
                            train_key="train_tasks")
    # Check the candidate-stage key as well when present.  It makes copied
    # checkpoints auditable without accepting a stale source chain.
    if "split_train_tasks" in payload:
        verify_split_provenance(payload, expected_split, label=f"{label} checkpoint")
    verify_generator_config(payload, importance_name=importance_name,
                            top_frac=top_frac, label=f"{label} checkpoint")


def _masks(task, method, n, *, cvae, vae, mean, generator, device):
    if method == "random": return random_bernoulli(n, generator=generator, device=device)
    if method == "random_exact96": return random_exact(n, generator=generator, device=device)
    if method == "conditional_mean": return mean(task).to(device).expand(n, -1).clone()
    if method == "ideal": return gold_mask(task).to(device).expand(n, -1).clone()
    if method == "vae": return vae.sample_topk([task], n, config.K_ACTIVE, generator=generator)
    if method in ("cvae", "cvae_wrong_gap"):
        condition = task_condition([task]).to(device)
        if method == "cvae_wrong_gap": condition = condition.roll(1, dims=-1)
        return cvae.sample_topk(condition, n, config.K_ACTIVE, generator=generator)
    raise ValueError(method)


def _train_and_eval(masks, task, steps, batch_size, lr, x_val, y_val,
                    paired_group_size: int | None = None):
    n = len(masks); device = x_val.device
    init_seed = _seed(task, 7_000_021)
    torch.manual_seed(init_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(init_seed)
    model = BatchedMaskedMLP(n, config.SEQ_LEN, config.H).to(device)
    if paired_group_size is not None:
        if paired_group_size < 1 or n % paired_group_size:
            raise ValueError("paired_group_size must divide the total number of masks")
        repeats = n // paired_group_size
        # Candidate ordinal j receives identical initial weights under every
        # method; only its connectivity mask differs.  This makes downstream
        # comparisons genuinely paired and removes avoidable initialization
        # variance from the OOD delta.
        with torch.no_grad():
            for parameter in (model.w1, model.b1, model.w2, model.b2):
                base = parameter[:paired_group_size].clone()
                parameter.copy_(base.repeat((repeats,) + (1,) * (parameter.ndim - 1)))
    model.load_masks(masks.reshape(n, config.SEQ_LEN, config.H))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    base = _seed(task, 1_000_003)
    for step in range(steps):
        xb, yb = get_train_batch(task, batch_size, base + step, device=device)
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb.unsqueeze(1).expand_as(logits))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return (model.val_loss(x_val, y_val, _value("VAL_BATCH_SIZE", 256)),
            model.val_acc(x_val, y_val, _value("VAL_BATCH_SIZE", 256)))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--cvae_ckpt", type=Path, required=True)
    p.add_argument("--vae_ckpt", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--steps", type=int, default=_value("EVAL_STEPS", 1000))
    p.add_argument("--n_masks", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=_value("TRAIN_BATCH_SIZE", 128))
    p.add_argument("--lr", type=float, default=_value("LR", 1e-3))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_frac", type=float, default=.1)
    p.add_argument(
        "--importance_name",
        default="importance.pt",
        help="Continuous importance artifact; --top_frac selects candidates by validation BCE.",
    )
    p.add_argument("--ckpt_root", type=Path, default=config.CKPT_DIR)
    p.add_argument("--device", choices=("cuda",), default="cuda")
    return p


def main():
    args = parser().parse_args(); split, train_tasks, test_tasks = _load_split(args.split)
    split_provenance = read_split_provenance(args.split)
    if train_tasks != split_provenance["split_train_tasks"]:
        raise RuntimeError("split parser/provenance disagreement")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for held-out downstream evaluation")
    device = torch.device(args.device)
    configure_compute_device(str(device))
    cvae = vae = None
    if any(m in args.methods for m in ("cvae", "cvae_wrong_gap")):
        cvae, cvae_meta = _load_model(args.cvae_ckpt, device)
        _check_provenance(cvae_meta, split_provenance, "CVAE",
                          importance_name=args.importance_name, top_frac=args.top_frac)
        if cvae.cond_dim != config.COND_DIM: raise ValueError("CVAE checkpoint is not gap-conditioned")
    if "vae" in args.methods:
        vae, vae_meta = _load_model(args.vae_ckpt, device)
        _check_provenance(vae_meta, split_provenance, "VAE",
                          importance_name=args.importance_name, top_frac=args.top_frac)
        if vae.cond_dim != 0: raise ValueError("VAE ablation checkpoint unexpectedly has a condition")
    mean = (ConditionalMean(train_tasks, ckpt_root=args.ckpt_root, importance_name=args.importance_name,
                            top_frac=args.top_frac, device=device,
                            split_path=args.split)
            if "conditional_mean" in args.methods else None)
    results = {"provenance": {"split": split, **split_provenance,
                              "methods": args.methods, "seed": args.seed,
                              "n_masks": args.n_masks, "steps": args.steps,
                              "top_frac": args.top_frac, "importance_name": args.importance_name,
                              "cvae_checkpoint": str(args.cvae_ckpt.resolve()),
                              "vae_checkpoint": str(args.vae_ckpt.resolve()),
                              "no_target_selection": True, "no_latent_optimization": True}, "tasks": {}}
    for task in test_tasks:
        val = make_dataset(task, _value("N_VAL_SAMPLES", 2048), _seed(task, 20_000), .5)
        x_val, y_val = val["x"].to(device), val["y"].to(device)
        task_result = {}
        runs = []
        for ordinal, method in enumerate(args.methods):
            # correct/wrong-gap is a controlled ablation: reuse exactly the
            # same latent z and change only the condition.
            seed_ordinal = (args.methods.index("cvae")
                            if method == "cvae_wrong_gap" and "cvae" in args.methods
                            else ordinal)
            gen = torch.Generator(device=device).manual_seed(
                _seed(task, args.seed + seed_ordinal * 10_000)
            )
            masks = _masks(task, method, args.n_masks, cvae=cvae, vae=vae, mean=mean, generator=gen, device=device)
            runs.append((method, masks))
        all_masks = torch.cat([m for _, m in runs])
        losses, acc = _train_and_eval(all_masks, task, args.steps, args.batch_size,
                                     args.lr, x_val, y_val,
                                     paired_group_size=args.n_masks)
        start = 0; gold = gold_mask(task).reshape(config.SEQ_LEN, config.H)
        for method, masks in runs:
            end = start + len(masks); l, a = losses[start:end], acc[start:end]; start = end
            structures = [best_permutation_iou(m.reshape(config.SEQ_LEN, config.H).cpu(), gold) for m in masks]
            task_result[method] = {"mean_bce": l.mean().item(), "mean_acc": a.mean().item(),
                                   "bce": l.detach().cpu().tolist(), "acc": a.detach().cpu().tolist(),
                                   "sparsity": masks.float().mean().item(), "structural": structures,
                                   "mean_best_permutation_iou": sum(s["iou"] for s in structures) / len(structures)}
            print(f"[eval] {task} {method}: acc={task_result[method]['mean_acc']:.4f} "
                  f"iou={task_result[method]['mean_best_permutation_iou']:.3f}", flush=True)
        results["tasks"][task] = task_result
    out = args.out_dir / "eval_results.json"; out.write_text(json.dumps(results, indent=2) + "\n")
    torch.save(results, args.out_dir / "eval_results.pt")
    print(f"saved -> {out}")


if __name__ == "__main__": main()
