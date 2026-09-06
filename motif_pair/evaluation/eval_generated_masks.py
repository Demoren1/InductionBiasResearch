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
from evaluation.baselines import (ConditionalMean, TrainOnlyGapMean, gold_mask,
                                  random_bernoulli, random_exact)  # noqa: E402
from evaluation.structural import best_permutation_iou  # noqa: E402
from models.cvae import (CVAE, checkpoint_condition_encoding, read_split_provenance,
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
    condition_encoding = checkpoint_condition_encoding(payload)
    model = CVAE(payload.get("mask_dim", config.MASK_DIM), payload.get("latent_dim", 32),
                 payload.get("hidden", 256), payload.get("cond_dim", config.COND_DIM),
                 condition_encoding=condition_encoding)
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


def _wrong_gaps(task: str, meta_train_gaps=None) -> tuple[int, ...]:
    """Choose every nearest incorrect condition from meta-training.

    The old one-hot-vector roll did not define a meaningful scalar-condition
    counterpart.  For gap-OOD, use all nearest structurally non-equivalent
    *meta-train* gaps, keeping motifs fixed and changing only the relation.
    Retaining both sides of an
    interpolation tie is essential: selecting the lower neighbour alone can
    make the ablation depend on an arbitrary tie-break rather than condition
    fidelity.  Extrapolation held-outs never compare CVAE to another unseen
    condition.  The no-candidate fallback retains the previous adjacent-gap
    behavior for direct legacy helper callers.
    """
    gap = config.parse_task(task).gap
    if meta_train_gaps is not None:
        # M_g and M_{L-g} are identical up to a hidden-column permutation, so
        # the complement is not a structurally wrong condition under PIoU.
        equivalent = {gap, (int(config.SEQ_LEN) - gap) % int(config.SEQ_LEN)}
        candidates = sorted({int(value) for value in meta_train_gaps if int(value) in config.GAPS}
                             - equivalent)
        if candidates:
            distance = min(abs(candidate - gap) for candidate in candidates)
            return tuple(candidate for candidate in candidates
                         if abs(candidate - gap) == distance)
        raise ValueError(f"no structurally non-equivalent meta-train gap for target gap {gap}")
    return (gap + 1 if gap == min(config.GAPS) else gap - 1,)


def _wrong_gap(task: str, meta_train_gaps=None) -> int:
    """Backward-compatible single-neighbour helper.

    Production evaluation uses :func:`_wrong_gaps`; this wrapper keeps direct
    callers deterministic while making the old lower-tie behavior explicit.
    """
    return _wrong_gaps(task, meta_train_gaps)[0]


def _task_at_gap(task: str, gap: int) -> str:
    parsed = config.parse_task(task)
    return config.Task(parsed.a, parsed.b, gap).id


def _masks(task, method, n, *, cvae, vae, mean, generator, device,
           meta_train_gaps=None):
    if method == "random": return random_bernoulli(n, generator=generator, device=device)
    if method == "random_exact96": return random_exact(n, generator=generator, device=device)
    if method == "conditional_mean": return mean(task).to(device).expand(n, -1).clone()
    if method == "ideal": return gold_mask(task).to(device).expand(n, -1).clone()
    if method == "vae": return vae.sample_topk([task], n, config.K_ACTIVE, generator=generator)
    if method == "cvae":
        # Ask the loaded model to encode the condition.  This is checkpoint
        # aware (scalar vs. legacy one-hot).
        condition = cvae.condition([task], device=device)
        return cvae.sample_topk(condition, n, config.K_ACTIVE, generator=generator)
    if method == "cvae_wrong_gap":
        # Every counterfactual condition receives exactly the same z samples
        # as correct-gap CVAE and as every other equally-near wrong gap.  The
        # concatenated result contains one n-sized block per queried gap.
        initial_state = generator.get_state()
        blocks = []
        for wrong_gap in _wrong_gaps(task, meta_train_gaps):
            generator.set_state(initial_state)
            condition = cvae.condition([_task_at_gap(task, wrong_gap)], device=device)
            blocks.append(cvae.sample_topk(condition, n, config.K_ACTIVE,
                                           generator=generator))
        return torch.cat(blocks)
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


def _bootstrap_strict_margin_ci(correct: torch.Tensor, wrong: list[torch.Tensor], *,
                                seed: int, n_resamples: int = 2_000) -> list[float]:
    """Percentile CI that recomputes the strongest wrong condition per resample."""
    correct = torch.as_tensor(correct, dtype=torch.float64, device="cpu").flatten()
    wrong = [torch.as_tensor(row, dtype=torch.float64, device="cpu").flatten()
             for row in wrong]
    if not wrong or any(row.shape != correct.shape for row in wrong):
        raise ValueError("strict-margin bootstrap needs equally shaped paired conditions")
    if correct.numel() < 2:
        value = float(correct.mean() - max(row.mean() for row in wrong))
        return [value, value]
    generator = torch.Generator().manual_seed(seed)
    indexes = torch.randint(correct.numel(), (n_resamples, correct.numel()),
                            generator=generator)
    correct_means = correct[indexes].mean(dim=1)
    wrong_means = torch.stack([row[indexes].mean(dim=1) for row in wrong], dim=1)
    margins = correct_means - wrong_means.max(dim=1).values
    bounds = torch.quantile(margins, torch.tensor([.025, .975], dtype=margins.dtype))
    return [float(bounds[0]), float(bounds[1])]


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
    meta_train_gaps = sorted({config.parse_task(task).gap for task in train_tasks})
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
        if cvae.cond_dim == 0: raise ValueError("CVAE checkpoint is not gap-conditioned")
    if "vae" in args.methods:
        vae, vae_meta = _load_model(args.vae_ckpt, device)
        _check_provenance(vae_meta, split_provenance, "VAE",
                          importance_name=args.importance_name, top_frac=args.top_frac)
        if vae.cond_dim != 0: raise ValueError("VAE ablation checkpoint unexpectedly has a condition")
    gap_heldout = split.get("split_kind") == "gap_heldout"
    mean = None
    mean_metadata = None
    if "conditional_mean" in args.methods:
        mean_class = TrainOnlyGapMean if gap_heldout else ConditionalMean
        mean_kwargs = dict(ckpt_root=args.ckpt_root, importance_name=args.importance_name,
                           top_frac=args.top_frac, device=device, split_path=args.split)
        if gap_heldout:
            # Its averages do not depend on the encoding, but recording it
            # proves the selected artifacts came from the same scalar/one-hot
            # run configuration.
            mean_kwargs["condition_encoding"] = (cvae.condition_encoding if cvae is not None else None)
        mean = mean_class(train_tasks, **mean_kwargs)
        if gap_heldout:
            mean_metadata = {
                "name": "train_only_gap_mean",
                "policy": mean.interpolation_policy,
                "available_train_gaps": list(mean.available_gaps),
                "per_test_gap": {str(gap): mean.describe_gap(gap)
                                 for gap in sorted({config.parse_task(t).gap for t in test_tasks})},
            }
    results = {"provenance": {"split": split, **split_provenance,
                              "methods": args.methods, "seed": args.seed,
                              "n_masks": args.n_masks, "steps": args.steps,
                              "top_frac": args.top_frac, "importance_name": args.importance_name,
                              "cvae_checkpoint": str(args.cvae_ckpt.resolve()),
                              "vae_checkpoint": str(args.vae_ckpt.resolve()),
                              "condition_encoding": (cvae.condition_encoding if cvae is not None else None),
                              "wrong_gap_policy": (
                                  "all nearest structurally non-equivalent meta-train gaps; "
                                  "interpolation ties retain both sides; "
                                  "same motifs and paired latent samples; "
                                  "adjacent-gap fallback only without train-gap metadata"
                              ),
                              "wrong_gap_meta_train_gaps": meta_train_gaps,
                              **({"conditional_mean": mean_metadata} if mean_metadata else {}),
                              "no_target_selection": True, "no_latent_optimization": True}, "tasks": {}}
    for task in test_tasks:
        val = make_dataset(task, _value("N_VAL_SAMPLES", 2048), _seed(task, 20_000), .5)
        x_val, y_val = val["x"].to(device), val["y"].to(device)
        task_result = {}
        runs = []
        for method in args.methods:
            # correct/wrong-gap is a controlled ablation: reuse exactly the
            # same latent z and change only the condition.  Use the canonical
            # global method order so a targeted --methods rerun reproduces the
            # same masks as the full evaluation.
            seed_ordinal = (METHODS.index("cvae")
                            if method == "cvae_wrong_gap"
                            else METHODS.index(method))
            gen = torch.Generator(device=device).manual_seed(
                _seed(task, args.seed + seed_ordinal * 10_000)
            )
            masks = _masks(task, method, args.n_masks, cvae=cvae, vae=vae, mean=mean,
                           generator=gen, device=device, meta_train_gaps=meta_train_gaps)
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
            if method == "cvae_wrong_gap":
                queried_gaps = _wrong_gaps(task, meta_train_gaps)
                expected = args.n_masks * len(queried_gaps)
                if len(masks) != expected:
                    raise RuntimeError(
                        f"wrong-gap block mismatch: got {len(masks)}, expected {expected}"
                    )
                by_condition_gap = {}
                for ordinal, wrong_gap in enumerate(queried_gaps):
                    block = slice(ordinal * args.n_masks, (ordinal + 1) * args.n_masks)
                    block_structures = structures[block]
                    by_condition_gap[str(wrong_gap)] = {
                        "mean_bce": l[block].mean().item(),
                        "mean_acc": a[block].mean().item(),
                        "mean_best_permutation_iou": (
                            sum(row["iou"] for row in block_structures) / len(block_structures)
                        ),
                        "bce": l[block].detach().cpu().tolist(),
                        "acc": a[block].detach().cpu().tolist(),
                        "structural": block_structures,
                    }
                task_result[method]["queried_condition_gaps"] = list(queried_gaps)
                task_result[method]["by_condition_gap"] = by_condition_gap
            print(f"[eval] {task} {method}: acc={task_result[method]['mean_acc']:.4f} "
                  f"iou={task_result[method]['mean_best_permutation_iou']:.3f}", flush=True)
        if "cvae" in task_result and "cvae_wrong_gap" in task_result:
            correct_ious = torch.tensor(
                [row["iou"] for row in task_result["cvae"]["structural"]],
                dtype=torch.float64,
            )
            wrong_rows = task_result["cvae_wrong_gap"]["by_condition_gap"]
            adversarial_gap, adversarial = max(
                wrong_rows.items(),
                key=lambda item: item[1]["mean_best_permutation_iou"],
            )
            adversarial_ious = torch.tensor(
                [row["iou"] for row in adversarial["structural"]],
                dtype=torch.float64,
            )
            paired_delta = correct_ious - adversarial_ious
            wrong_ious = [
                torch.tensor([item["iou"] for item in row["structural"]],
                             dtype=torch.float64)
                for row in wrong_rows.values()
            ]
            target_gap = config.parse_task(task).gap
            task_result["cvae"]["condition_fidelity"] = {
                "definition": "correct PIoU minus the largest mean PIoU among equally-near seen-gap queries",
                "target_gap": target_gap,
                "correct_mean_best_permutation_iou": float(correct_ious.mean()),
                "wrong_mean_best_permutation_iou_by_gap": {
                    gap: row["mean_best_permutation_iou"] for gap, row in wrong_rows.items()
                },
                "adversarial_wrong_gap": int(adversarial_gap),
                "strict_margin": float(paired_delta.mean()),
                "paired_latent_bootstrap_ci95": _bootstrap_strict_margin_ci(
                    correct_ious, wrong_ious,
                    seed=_seed(task, args.seed + 30_000_019)
                ),
                "n_paired_latents": len(paired_delta),
            }
        results["tasks"][task] = task_result
    out = args.out_dir / "eval_results.json"; out.write_text(json.dumps(results, indent=2) + "\n")
    torch.save(results, args.out_dir / "eval_results.pt")
    print(f"saved -> {out}")


if __name__ == "__main__": main()
