"""Compare soft assignments, hard STE and categorical hard sampling."""

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

from experiments.darts_indexator_v1.model import canonical_signed_probabilities

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


def configurations(regime):
    methods = (
        ("soft", "soft", 1, False, 0.0),
        ("soft_commit", "soft", 1, False, 0.1),
        ("hard_argmax", "argmax", 1, False, 0.0),
        ("gumbel_1", "gumbel", 1, False, 0.0),
        ("gumbel_4", "gumbel", 4, False, 0.0),
        ("gumbel_1_mass", "gumbel", 1, True, 0.0),
    )
    architecture_rates = (0.003, 0.03) if regime == "masked" else (0.03,)
    sparsities = (0.0,) if regime == "masked" else (0.0, 0.03)
    rows = []
    for method, numeric_rate, architecture_rate, sparsity in itertools.product(
        methods, (0.003, 0.01), architecture_rates, sparsities
    ):
        family, estimator, samples, mass, commitment = method
        rows.append(
            {
                "configuration": len(rows),
                "family": family,
                "estimator": estimator,
                "samples": samples,
                "mass_transfer": mass,
                "commitment": commitment,
                "numeric_lr": numeric_rate,
                "architecture_lr": architecture_rate,
                "sparsity": sparsity,
                "radius": 0.5,
                "minimum_temperature": 0.3,
                "mode": "joint",
            }
        )
    return rows


def prepare(parent, root):
    root.mkdir(parents=True, exist_ok=False)
    for regime, heldout_offset in (("masked", 7400000), ("full", 7500000)):
        output = root / regime
        output.mkdir()
        plan = json.loads((parent / "protocol.json").read_text())
        spec = json.loads((parent / "specification.json").read_text())
        spec.update(
            name=f"deepsets_hard_sampling_{regime}",
            full_connectivity_search=regime == "full",
        )
        plan["search"].update(
            pilot_seeds=[201, 202],
            confirmation_seeds=[203, 204, 205],
            pilot_steps=12000,
            confirmation_steps=12000,
            numeric_warmup=0,
            restarts=1,
            top_configurations=6,
            zero_probability=1 / (2 * spec["codes"] + 1) if regime == "masked" else 0.5,
        )
        plan["sharing_heldout_offset"] = heldout_offset
        if regime == "masked":
            with np.load(parent / "selected_states.npz") as archive:
                mask = archive["indices"] != 0
        else:
            mask = np.ones((spec["inputs"], spec["hidden"]), dtype=bool)
        np.savez_compressed(output / "search_support.npz", mask=mask)
        for name in ("source.npz", "teacher_audit_only.npz"):
            shutil.copy2(parent / name, output / name)
        save_json(output / "protocol.json", plan)
        save_json(output / "specification.json", spec)
        snapshot = output / "code_snapshot"
        snapshot.mkdir()
        for name in (
            "hard_sampling.py",
            "sampling_report.py",
            "model.py",
            "train.py",
            "run.py",
            "data.py",
            "sharing_ablation.py",
            "sharing_report.py",
            "evaluate.py",
        ):
            shutil.copy2(BASE / name, snapshot / name)
        shutil.copy2(
            BASE.parent / "darts_indexator_v1/model.py",
            snapshot / "signed_probability_model.py",
        )
        save_json(
            output / "registration.json",
            {
                "registered_utc": datetime.now(UTC).isoformat(),
                "regime": regime,
                "candidates": configurations(regime),
                "rules": [
                    "Same previously inspected synthetic teacher; no claim of transfer to new functions.",
                    "The masked regime reuses one source-selected DARTS mask. The full regime permits all positions.",
                    "One common trainable signed codebook per search root; sampled structures do not have independent numerical parameters.",
                    "Soft, hard deterministic STE, hard Gumbel with one/four samples, and a separate Gumbel mass-transfer ablation.",
                    "Matched random logits and initial codebook across methods at each seed; radius normalized using the same raw hard assignment.",
                    "Each sampled complete indexator is shared by all examples in its minibatch.",
                    "Gumbel noise is paired across configurations at the same seed; one sample is the prefix of the four-sample estimator.",
                    "Both codebook and logits update jointly on support. This is not a bilevel validation gradient.",
                    "Hard forward with the soft Gumbel backward is a biased gradient estimator.",
                    "Sampling law is softmax(alpha), or its explicit mass projection. Temperature1 to0.3 affects only the surrogate for hard methods.",
                    "Late entropy coefficient0.02 on the policy; full search also ablates sparsity0/0.03. No teacher edge count is used.",
                    "Mass transfer before sampling changes the distribution; it is an experimental factor, not an equivalent rewrite of the stochastic loss.",
                    "Snapshots0/4000/8000/12000 store exactly the MAP policy assignment as one-hot. No additional canonicalization changes this assignment.",
                    "Each frozen indexator gets3000 support refit steps with one inherited and three fresh Gaussian C; query selects numerical checkpoints.",
                    "Pilot201/202 chooses one setting per family by mean hard-refit selection loss; confirmation203/204/205 uses frozen pilot settings.",
                    "Diagnostic expected losses average eight query structures and never select candidates. Final ranking uses refitted deterministic indexators.",
                    "After all source choices, fresh-C calibration precedes32 new initializations and a new heldout input draw. Teacher is only used in final audit.",
                    "No software tests are created or run.",
                ],
                "literature": ["https://arxiv.org/abs/1611.01144"],
                "source_sha256": digest(output / "source.npz"),
                "mask_sha256": digest(output / "search_support.npz"),
                "snapshot_sha256": {
                    path.name: digest(path) for path in snapshot.glob("*.py")
                },
            },
        )


def assignments(alpha, mass_transfer, mask):
    probabilities = alpha.softmax(-1)
    canonical = canonical_signed_probabilities(probabilities)
    policy = torch.where(mass_transfer[:, None, None, None], canonical, probabilities)
    zero = torch.zeros_like(policy)
    zero[..., 0] = 1
    return torch.where(mask[None, ..., None], policy, zero)


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
    initial_onehot = F.one_hot(alpha.argmax(-1), classes).float()
    zero = torch.zeros_like(initial_onehot)
    zero[..., 0] = 1
    initial_onehot = torch.where(mask[None, ..., None], initial_onehot, zero)
    numeric = initial_numeric(
        rows,
        1,
        spec,
        plan["seed"] + 901,
        device,
        probabilities=initial_onehot,
    ).requires_grad_()
    alpha.requires_grad_()
    np.savez_compressed(
        folder / "initial_state.npz",
        logits=alpha.detach().cpu().numpy(),
        numeric=numeric.detach().cpu().numpy(),
    )
    numeric_optimizer = BatchedOptimizer(numeric, rows)
    alpha_optimizer = BatchedOptimizer(alpha, rows, "architecture_lr")
    is_soft = torch.tensor([row["estimator"] == "soft" for row in rows], device=device)
    stochastic = torch.tensor(
        [row["estimator"] == "gumbel" for row in rows], device=device
    )
    mass_transfer = torch.tensor([row["mass_transfer"] for row in rows], device=device)
    sample_count = torch.tensor([row["samples"] for row in rows], device=device)
    maximum_samples = int(sample_count.max())
    sample_active = (
        torch.arange(maximum_samples, device=device)[None] < sample_count[:, None]
    )
    minima = numeric.new_tensor([row["minimum_temperature"] for row in rows])
    sparsity = numeric.new_tensor([row["sparsity"] for row in rows])
    commitment = numeric.new_tensor([row["commitment"] for row in rows])
    seeds = sorted({row["seed"] for row in rows})
    seed_index = torch.tensor([seeds.index(row["seed"]) for row in rows], device=device)
    data = source_tensors(output, device)
    batch_generator = torch.Generator(device=device).manual_seed(plan["seed"] + 1001)
    sample_generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + 810001 + min(seeds)
    )
    validation_generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + 820001 + min(seeds)
    )
    steps = settings[phase + "_steps"]
    checkpoints = {round(value * steps) for value in settings["snapshots"]}
    history, saved = [], []
    started = time.perf_counter()
    total_forward_error = 0.0
    for step in range(steps + 1):
        progress = step / max(steps, 1)
        cooling = min(1.0, max(0.0, (progress - 0.3) / 0.7))
        temperature = minima.pow(cooling)
        policy = assignments(alpha, mass_transfer, mask)
        log_policy = policy.clamp_min(1e-12).log()
        values = signed_values(numeric, codes)[:, 0]
        entropy = -(policy * log_policy).sum(-1)[:, mask].mean(-1)
        if step % settings["checkpoint_every"] == 0 or step == steps:
            with torch.no_grad():
                map_onehot = F.one_hot(policy.argmax(-1), classes).float()
                selected_classes = policy.argmax(-1)
                indices = torch.where(
                    selected_classes > codes, codes - selected_classes, selected_classes
                )
                hard_weights = fixed_weights(indices, numeric, codes)
                hard_query = score(
                    data["query_x"], data["query_y"], hard_weights, numeric, spec
                )
                relaxed_policy = (
                    log_policy / temperature[:, None, None, None]
                ).softmax(-1)
                relaxed_weights = torch.einsum("sdhk,sk->sdh", relaxed_policy, values)[
                    :, None
                ]
                soft_query = score(
                    data["query_x"], data["query_y"], relaxed_weights, numeric, spec
                )
                noise = torch.rand(
                    (len(seeds), 8, *mask_array.shape, classes),
                    generator=validation_generator,
                    device=device,
                ).clamp_(1e-7, 1 - 1e-7)
                gumbel = -(-noise.log()).log()
                sampled_classes = (log_policy[:, None] + gumbel[seed_index]).argmax(-1)
                sampled = F.one_hot(sampled_classes, classes).float()
                sampled_weights = torch.einsum("smdhk,sk->smdh", sampled, values)
                expanded = numeric.expand(-1, 8, -1)
                sampled_query = score(
                    data["query_x"], data["query_y"], sampled_weights, expanded, spec
                ).mean(-1, keepdim=True)
                sampled_train = score(
                    data["support_x"][:512],
                    data["support_y"][:512],
                    sampled_weights,
                    expanded,
                    spec,
                ).mean(-1, keepdim=True)
                soft_train = score(
                    data["support_x"][:512],
                    data["support_y"][:512],
                    relaxed_weights,
                    numeric,
                    spec,
                )
                hard_train = score(
                    data["support_x"][:512],
                    data["support_y"][:512],
                    hard_weights,
                    numeric,
                    spec,
                )
                query = torch.where(
                    is_soft[:, None],
                    soft_query,
                    torch.where(stochastic[:, None], sampled_query, hard_query),
                )
                training = torch.where(
                    is_soft[:, None],
                    soft_train,
                    torch.where(stochastic[:, None], sampled_train, hard_train),
                )
                if not torch.isfinite(query).all():
                    raise FloatingPointError("Non-finite hard-sampling search")
                history.append(
                    {
                        "step": step,
                        "query_mse": query.tolist(),
                        "train_mse": training.tolist(),
                        "hard_query_mse": hard_query.tolist(),
                        "soft_query_mse": soft_query.tolist(),
                        "sampled_query_mse": sampled_query.tolist(),
                        "entropy": entropy.tolist(),
                        "active_probability": (1 - policy[..., 0])[:, mask]
                        .mean(-1)
                        .tolist(),
                        "codebook": values[:, 1 : codes + 1].tolist(),
                        "maximum_hard_forward_error": total_forward_error,
                        "seconds": time.perf_counter() - started,
                    }
                )
                save_json(folder / "history.json", history)
                if step in checkpoints:
                    fresh_numeric = initial_numeric(
                        [{**row, "aligned_initial_codes": False} for row in rows],
                        3,
                        spec,
                        plan["seed"] + 51017,
                        device,
                        indices=indices,
                    )
                    saved.append(
                        {
                            "step": step,
                            "probabilities": map_onehot.cpu().numpy(),
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
                print(
                    f"{spec['name']} {phase} step={step} rows={len(rows)} "
                    f"search_NMSE={query.mean().item() / spec['target_variance']:.5g} "
                    f"MAP_NMSE={hard_query.mean().item() / spec['target_variance']:.5g} "
                    f"seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )
        if step == steps:
            break
        noise = torch.rand(
            (len(seeds), maximum_samples, *mask_array.shape, classes),
            generator=sample_generator,
            device=device,
        ).clamp_(1e-7, 1 - 1e-7)
        gumbel = -(-noise.log()).log()
        perturbed = (
            log_policy[:, None]
            + gumbel[seed_index] * stochastic[:, None, None, None, None]
        )
        relaxed = (perturbed / temperature[:, None, None, None, None]).softmax(-1)
        hard = F.one_hot(perturbed.argmax(-1), classes).to(relaxed)
        # The parenthesized difference is exactly zero in forward.
        straight = hard + (relaxed - relaxed.detach())
        assignment = torch.where(is_soft[:, None, None, None, None], relaxed, straight)
        if step % settings["checkpoint_every"] == 0:
            error = float((assignment[~is_soft].detach() - hard[~is_soft]).abs().max())
            total_forward_error = max(total_forward_error, error)
            if error != 0:
                raise RuntimeError("A hard training assignment became a soft mixture")
        weights = torch.einsum("smdhk,sk->smdh", assignment, values)
        support_ids = torch.randint(
            len(data["support_x"]),
            (settings["batch"],),
            generator=batch_generator,
            device=device,
        )
        prediction = predict(
            data["support_x"][support_ids],
            weights,
            numeric.expand(-1, maximum_samples, -1),
            spec,
        )
        losses = (prediction - data["support_y"][support_ids]).square().mean(-1)
        objective = (losses * sample_active).sum(-1) / sample_count
        objective = objective / spec["target_variance"]
        objective = objective + sparsity * (1 - policy[..., 0]).mean((1, 2))
        objective = (
            objective + settings["late_entropy"] * max(0, 2 * progress - 1) * entropy
        )
        detached_values = values.detach()
        average = torch.einsum("sdhk,sk->sdh", policy, detached_values)
        second = torch.einsum("sdhk,sk->sdh", policy, detached_values.square())
        energy = detached_values[:, 1 : codes + 1].square().mean(-1).clamp_min(1e-8)
        variance_penalty = (second - average.square()).clamp_min(0)[:, mask].mean(-1)
        objective = objective + commitment * variance_penalty / energy
        objective = (
            objective + settings["numeric_l2"] * numeric.square().sum((1, 2)) / 2
        )
        numeric_gradient, alpha_gradient = torch.autograd.grad(
            objective.sum(), (numeric, alpha)
        )
        numeric_optimizer.step(clip_gradient(numeric_gradient, 10, (1, 2)))
        alpha_optimizer.step(clip_gradient(alpha_gradient, 5, (1, 2, 3)))
        with torch.no_grad():
            alpha.sub_(alpha.mean(-1, keepdim=True))
    save_json(
        folder / "complete.json",
        {
            "seconds": time.perf_counter() - started,
            "steps": steps,
            "roots": len(rows),
            "numeric_restarts": 1,
            "maximum_hard_forward_error": total_forward_error,
            "effective_structure_forwards": int(
                sum(row["samples"] for row in rows) * steps
            ),
            "maximum_vectorized_samples": maximum_samples,
        },
    )


def run_phase(output, phase, device):
    plan = json.loads((output / "protocol.json").read_text())
    spec = json.loads((output / "specification.json").read_text())
    candidates = json.loads((output / "registration.json").read_text())["candidates"]
    if phase == "confirmation":
        chosen = json.loads((output / "pilot_selection.json").read_text())[
            "top_configurations"
        ]
        candidates = [row for row in candidates if row["configuration"] in chosen]
    rows = [
        {**row, "seed": seed}
        for row in candidates
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
                "top_configurations": [
                    entry["configuration"] for entry in families.values()
                ],
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
                family: {
                    "configuration": entry["configuration"],
                    "selection_nmse": entry["selection_mse"] / spec["target_variance"],
                }
                for family, entry in families.items()
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--regime", choices=("masked", "full", "both"), default="both")
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
    regimes = ("masked", "full") if args.regime == "both" else (args.regime,)
    device = torch.device(args.device)
    for regime in regimes:
        output = args.root / regime
        if args.stage == "fresh":
            fresh(output, device)
        else:
            run_phase(output, args.stage, device)


if __name__ == "__main__":
    main()
