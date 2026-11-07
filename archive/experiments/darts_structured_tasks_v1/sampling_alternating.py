"""Separate support codebook updates from query architecture updates."""

import argparse
import itertools
import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import BASE, digest, save_json
from .model import (
    BatchedOptimizer,
    clip_gradient,
    fixed_weights,
    initial_numeric,
    predict,
    score,
    signed_values,
    source_tensors,
)
from .run import refit_search
from .sharing_ablation import fresh

FAMILIES = ("joint_support", "alternating_support", "alternating_query")
SAMPLES = 4


def configurations(regime):
    rows = []
    for family, numeric_rate, architecture_rate in itertools.product(
        FAMILIES, (0.003, 0.01), (0.003, 0.01, 0.03)
    ):
        rows.append(
            {
                "configuration": len(rows),
                "family": family,
                "mode": family,
                "numeric_lr": numeric_rate,
                "architecture_lr": architecture_rate,
                "radius": 0.5,
                "sparsity": 0.03 if regime == "full" else 0.0,
                "commitment": 0.0,
                "samples": SAMPLES,
                "minimum_temperature": 0.3,
            }
        )
    return rows


def prepare(parent, root):
    root.mkdir(parents=True, exist_ok=False)
    for regime in ("full", "masked"):
        output = root / regime
        output.mkdir()
        plan = json.loads((parent / "protocol.json").read_text())
        spec = json.loads((parent / "specification.json").read_text())
        spec.update(
            name=f"deepsets_sampling_alternating_{regime}",
            full_connectivity_search=regime == "full",
        )
        plan["search"].update(
            pilot_seeds=[221, 222],
            confirmation_seeds=[223, 224, 225],
            pilot_steps=12000,
            confirmation_steps=12000,
            numeric_warmup=0,
            restarts=1,
            top_configurations=len(FAMILIES),
            zero_probability=0.5 if regime == "full" else 1 / (2 * spec["codes"] + 1),
        )
        plan["sharing_heldout_offset"] = 7700000
        if regime == "full":
            mask = np.ones((spec["inputs"], spec["hidden"]), dtype=bool)
        else:
            with np.load(parent / "selected_states.npz") as archive:
                mask = archive["indices"] != 0
        np.savez_compressed(output / "search_support.npz", mask=mask)
        for name in ("source.npz", "teacher_audit_only.npz"):
            shutil.copy2(parent / name, output / name)
        save_json(output / "protocol.json", plan)
        save_json(output / "specification.json", spec)
        snapshot = output / "code_snapshot"
        snapshot.mkdir()
        for name in (
            "sampling_alternating.py",
            "alternating_report.py",
            "model.py",
            "train.py",
            "run.py",
            "data.py",
            "sharing_ablation.py",
            "sharing_report.py",
            "evaluate.py",
        ):
            shutil.copy2(BASE / name, snapshot / name)
        save_json(
            output / "registration.json",
            {
                "registered_utc": datetime.now(UTC).isoformat(),
                "regime": regime,
                "candidates": configurations(regime),
                "rules": [
                    "One previously inspected synthetic teacher; no transfer claim.",
                    "Full search allows every matrix position. Masked search reuses the same previously learned DARTS mask, with zero still available inside it.",
                    "Three matched families: joint support, alternating support, alternating query. Only the architecture data split and use of the updated C differ.",
                    "Exactly12000 C updates and12000 architecture updates per root in every family. All families execute two forward/backward stages per iteration.",
                    "C always receives only support MSE and numerical L2. Joint support computes the architecture gradient at C before its update; alternating families use updated C.",
                    "For alternating query, architecture loss uses query labels. Its C is detached: this is first-order alternating optimization, not an unrolled bilevel gradient.",
                    "The C stage detaches P, the P stage detaches C and the head. Numerical gradients cannot update P, and architecture gradients cannot update C.",
                    "Four hard Gumbel samples per stage, with the same sampled structures in both stages. An indexator is shared across all examples in the minibatch.",
                    "Pair initial logits/C, support/query minibatch indices and Gumbel noise across configurations at the same seed. One trainable numerical replica during search, then one inherited plus three fresh numerical starts in refit.",
                    "P=softmax(alpha) remains the categorical sampling law. Surrogate temperature1 to0.3 and late entropy0.02 are identical. No mass projection changes the sampling law in this ablation.",
                    "Use only source folds for tuning: support for C, query for P when specified and for fixed-I numerical checkpoints, selection for snapshot and hyperparameter selection. Query is not independent validation for alternating query.",
                    "Snapshots0/4000/8000/12000 freeze raw policy MAP as one-hot, followed by3000 C-fit steps. Mean selection error over four refits ranks each snapshot.",
                    "Six learning-rate configurations per family. Pilot221/222 selects one setting per family before confirmation223/224/225. Selection never uses teacher recovery or heldout error.",
                    "After source freeze, calibrate fresh-C fits, evaluate32 new C starts and draw new heldout inputs. Teacher alignment is audit only.",
                    "No software tests are added or run.",
                ],
                "source_sha256": digest(output / "source.npz"),
                "mask_sha256": digest(output / "search_support.npz"),
                "snapshot_sha256": {p.name: digest(p) for p in snapshot.glob("*.py")},
            },
        )


def policy_probabilities(alpha, mask):
    zero = torch.zeros_like(alpha)
    zero[..., 0] = 1
    return torch.where(mask[None, ..., None], alpha.softmax(-1), zero)


def sampled_loss(assignment, numeric, x, y, spec):
    weights = torch.einsum(
        "smdhk,srk->srmdh", assignment, signed_values(numeric, spec["codes"])
    ).flatten(1, 2)
    expanded = numeric[:, :, None].expand(-1, -1, assignment.shape[1], -1)
    predicted = predict(x, weights, expanded.flatten(1, 2), spec)
    return (predicted - y).square().mean(-1).reshape(
        len(numeric), numeric.shape[1], assignment.shape[1]
    ).mean(-1) / spec["target_variance"]


def search(output, plan, spec, rows, phase, device):
    folder = output / phase
    folder.mkdir(exist_ok=False)
    save_json(folder / "rows.json", rows)
    settings = plan["search"]
    codes, classes = spec["codes"], 2 * spec["codes"] + 1
    with np.load(output / "search_support.npz") as archive:
        mask_array = archive["mask"]
    mask = torch.tensor(mask_array, device=device)
    arrays = []
    for row in rows:
        rng = np.random.default_rng(plan["seed"] + 100003 * row["seed"])
        item = rng.normal(0, 1, (*mask_array.shape, classes))
        probability = settings["zero_probability"]
        item[..., 0] += math.log(2 * codes * probability / (1 - probability))
        arrays.append(item)
    alpha = torch.tensor(np.array(arrays), device=device, dtype=torch.float32)
    initial_policy = policy_probabilities(alpha, mask)
    initial_assignment = F.one_hot(initial_policy.argmax(-1), classes).float()
    numeric = initial_numeric(
        rows,
        1,
        spec,
        plan["seed"] + 901,
        device,
        probabilities=initial_assignment,
    ).requires_grad_()
    alpha.requires_grad_()
    np.savez_compressed(
        folder / "initial_state.npz",
        logits=alpha.detach().cpu().numpy(),
        numeric=numeric.detach().cpu().numpy(),
    )
    numeric_optimizer = BatchedOptimizer(numeric, rows)
    alpha_optimizer = BatchedOptimizer(alpha, rows, "architecture_lr")
    joint = torch.tensor([r["family"] == "joint_support" for r in rows], device=device)
    outer_query = torch.tensor(
        [r["family"] == "alternating_query" for r in rows], device=device
    )
    sparsity = numeric.new_tensor([r["sparsity"] for r in rows])
    seeds = sorted({row["seed"] for row in rows})
    seed_index = torch.tensor([seeds.index(r["seed"]) for r in rows], device=device)
    data = source_tensors(output, device)
    batch_generator = torch.Generator(device=device).manual_seed(plan["seed"] + 1001)
    query_generator = torch.Generator(device=device).manual_seed(plan["seed"] + 1002)
    sample_generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + 850001 + min(seeds)
    )
    diagnostic_generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + 860001 + min(seeds)
    )
    steps = settings[phase + "_steps"]
    checkpoints = {round(value * steps) for value in settings["snapshots"]}
    history, saved = [], []
    started = time.perf_counter()
    maximum_forward_error = 0.0
    gradient_norms = None
    for step in range(steps + 1):
        progress = step / max(steps, 1)
        cooling = min(1.0, max(0.0, (progress - 0.3) / 0.7))
        temperature = 0.3**cooling
        policy = policy_probabilities(alpha, mask)
        log_policy = policy.clamp_min(1e-30).log()
        entropy = -(policy * log_policy).sum(-1)[:, mask].mean(-1)
        if step % settings["checkpoint_every"] == 0 or step == steps:
            with torch.no_grad():
                map_assignment = F.one_hot(policy.argmax(-1), classes).float()
                selected = policy.argmax(-1)
                indices = torch.where(selected > codes, codes - selected, selected)
                hard_weights = fixed_weights(indices, numeric, codes)
                hard_query = score(
                    data["query_x"], data["query_y"], hard_weights, numeric, spec
                )
                uniform = torch.rand(
                    (len(seeds), 8, *mask_array.shape, classes),
                    generator=diagnostic_generator,
                    device=device,
                ).clamp_(1e-7, 1 - 1e-7)
                noise = -(-uniform.log()).log()
                sampled = F.one_hot(
                    (log_policy[:, None] + noise[seed_index]).argmax(-1), classes
                ).float()
                query = sampled_loss(
                    sampled, numeric, data["query_x"], data["query_y"], spec
                )
                training = sampled_loss(
                    sampled, numeric, data["support_x"], data["support_y"], spec
                )
                if not torch.isfinite(query).all():
                    raise FloatingPointError("Non-finite alternating-sampling search")
                history.append(
                    {
                        "step": step,
                        "query_mse": (query * spec["target_variance"]).tolist(),
                        "train_mse": (training * spec["target_variance"]).tolist(),
                        "hard_query_mse": hard_query.tolist(),
                        "entropy": entropy.tolist(),
                        "MAP_edges": (selected != 0).sum((1, 2)).tolist(),
                        "gradient_norms": gradient_norms,
                        "maximum_hard_forward_error": maximum_forward_error,
                        "seconds": time.perf_counter() - started,
                    }
                )
                save_json(folder / "history.json", history)
                if step in checkpoints:
                    fresh_numeric = initial_numeric(
                        [{**r, "aligned_initial_codes": False} for r in rows],
                        3,
                        spec,
                        plan["seed"] + 51017,
                        device,
                        indices=indices,
                    )
                    saved.append(
                        {
                            "step": step,
                            "probabilities": map_assignment.cpu().numpy(),
                            "policy_probabilities": policy.cpu().numpy(),
                            "logits": alpha.cpu().numpy(),
                            "numeric": torch.cat((numeric, fresh_numeric), 1)
                            .cpu()
                            .numpy(),
                            "search_numeric": numeric.cpu().numpy(),
                        }
                    )
                    np.savez_compressed(
                        folder / "snapshots.npz",
                        steps=np.array([item["step"] for item in saved]),
                        **{
                            key: np.stack([item[key] for item in saved])
                            for key in saved[0]
                            if key != "step"
                        },
                    )
                if step % 1000 == 0 or step == steps:
                    print(
                        f"{spec['name']} {phase} step={step} roots={len(rows)} "
                        f"query_NMSE={query.mean().item():.5g} "
                        f"seconds={time.perf_counter() - started:.1f}",
                        flush=True,
                    )
        if step == steps:
            break
        uniform = torch.rand(
            (len(seeds), SAMPLES, *mask_array.shape, classes),
            generator=sample_generator,
            device=device,
        ).clamp_(1e-7, 1 - 1e-7)
        noise = -(-uniform.log()).log()[seed_index]
        perturbed = log_policy[:, None] + noise
        relaxed = (perturbed / temperature).softmax(-1)
        hard = F.one_hot(perturbed.argmax(-1), classes).to(relaxed)
        assignment = hard + (relaxed - relaxed.detach())
        support_ids = torch.randint(
            len(data["support_x"]),
            (settings["batch"],),
            generator=batch_generator,
            device=device,
        )
        query_ids = torch.randint(
            len(data["query_x"]),
            (settings["batch"],),
            generator=query_generator,
            device=device,
        )
        # Numerical adaptation sees only support; P is explicitly detached.
        before = numeric.detach().clone()
        losses = sampled_loss(
            hard.detach(),
            numeric,
            data["support_x"][support_ids],
            data["support_y"][support_ids],
            spec,
        )
        objective = losses + settings["numeric_l2"] * numeric.square().sum(-1) / 2
        numeric_gradient = torch.autograd.grad(objective.sum(), numeric)[0]
        numeric_optimizer.step(
            clip_gradient(numeric_gradient, settings["gradient_clip"], (-1,))
        )
        # Joint uses the pre-update C. Alternating uses updated C, detached.
        outer_numeric = torch.where(joint[:, None, None], before, numeric.detach())
        support_x, support_y = (
            data["support_x"][support_ids],
            data["support_y"][support_ids],
        )
        query_x, query_y = data["query_x"][query_ids], data["query_y"][query_ids]
        x = torch.where(outer_query[:, None, None], query_x[None], support_x[None])
        y = torch.where(outer_query[:, None], query_y[None], support_y[None])
        weights = torch.einsum(
            "smdhk,srk->smdh", assignment, signed_values(outer_numeric, codes)
        )
        expanded = outer_numeric.expand(-1, SAMPLES, -1)
        hidden = torch.einsum("snd,smdh->smnh", x, weights)
        prediction = (hidden + expanded[..., codes, None, None]).relu().sum(
            -1
        ) * expanded[..., codes + 1, None] / math.sqrt(spec["hidden"]) + expanded[
            ..., codes + 2, None
        ]
        outer_loss = (prediction - y[:, None]).square().mean((1, 2)) / spec[
            "target_variance"
        ]
        objective = outer_loss + sparsity * (1 - policy[..., 0]).mean((1, 2))
        objective = (
            objective + settings["late_entropy"] * max(0, 2 * progress - 1) * entropy
        )
        alpha_gradient = torch.autograd.grad(objective.sum(), alpha)[0]
        alpha_optimizer.step(
            clip_gradient(alpha_gradient, settings["alpha_clip"], (1, 2, 3))
        )
        with torch.no_grad():
            alpha.sub_(alpha.mean(-1, keepdim=True))
            if step % settings["checkpoint_every"] == 0:
                error = float((assignment.detach() - hard).abs().max())
                maximum_forward_error = max(maximum_forward_error, error)
                if error != 0:
                    raise RuntimeError("Hard forward unexpectedly became a mixture")
                gradient_norms = {
                    "numeric": numeric_gradient.flatten(1).norm(dim=1).tolist(),
                    "architecture": alpha_gradient.flatten(1).norm(dim=1).tolist(),
                }
    save_json(
        folder / "complete.json",
        {
            "seconds": time.perf_counter() - started,
            "roots": len(rows),
            "numeric_updates_per_root": steps,
            "architecture_updates_per_root": steps,
            "numerical_search_replicas": 1,
            "hard_samples_per_stage": SAMPLES,
            "effective_structure_forwards": len(rows) * steps * SAMPLES * 2,
            "maximum_hard_forward_error": maximum_forward_error,
            "unrolled_gradient": False,
        },
    )


def run_phase(output, phase, device):
    plan = json.loads((output / "protocol.json").read_text())
    spec = json.loads((output / "specification.json").read_text())
    candidates = json.loads((output / "registration.json").read_text())["candidates"]
    if phase == "confirmation":
        selected = json.loads((output / "pilot_selection.json").read_text())[
            "top_configurations"
        ]
        candidates = [r for r in candidates if r["configuration"] in selected]
    rows = [
        {**r, "seed": seed}
        for r in candidates
        for seed in plan["search"][phase + "_seeds"]
    ]
    search(output, plan, spec, rows, phase, device)
    refit_search(output, plan, spec, phase, device)
    ranking = json.loads((output / phase / "ranking.json").read_text())
    families = {}
    for entry in ranking:
        families.setdefault(entry["per_seed"][0]["family"], entry)
    if phase == "pilot":
        save_json(
            output / "pilot_selection.json",
            {
                "families": families,
                "top_configurations": [r["configuration"] for r in families.values()],
                "ranking": ranking,
            },
        )
    else:
        save_json(output / "family_selection.json", {"families": families})
        save_json(
            output / "source_frozen.json",
            {
                "frozen_utc": datetime.now(UTC).isoformat(),
                "sha256": {
                    name: digest(output / name)
                    for name in (
                        "source.npz",
                        "family_selection.json",
                        "architecture_selection.json",
                        "confirmation/hard_fits.npz",
                        "confirmation/snapshots.npz",
                        "confirmation/rows.json",
                    )
                },
            },
        )
    print(
        json.dumps(
            {
                name: entry["selection_mse"] / spec["target_variance"]
                for name, entry in families.items()
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--regime", choices=("full", "masked", "both"), default="both")
    parser.add_argument(
        "--stage", choices=("prepare", "pilot", "confirmation", "fresh"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        if args.parent is None:
            parser.error("--parent is required for preparation")
        prepare(args.parent, args.root)
        return
    regimes = ("full", "masked") if args.regime == "both" else (args.regime,)
    device = torch.device(args.device)
    for regime in regimes:
        if args.stage == "fresh":
            fresh(args.root / regime, device)
        else:
            run_phase(args.root / regime, args.stage, device)


if __name__ == "__main__":
    main()
