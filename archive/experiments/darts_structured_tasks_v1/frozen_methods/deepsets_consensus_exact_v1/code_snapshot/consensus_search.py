"""Full-matrix sharing search with continuous/discrete consensus updates."""

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
import torch.nn.functional as F

from .codebook_projection import exact_signed_projection
from .data import BASE, digest, save_json
from .model import (
    BatchedOptimizer,
    clip_gradient,
    initial_numeric,
    predict,
    score,
    source_tensors,
)
from .run import refit_search
from .sharing_ablation import fresh


def configurations():
    rows = []
    for family, dual in (("quadratic", 0.0), ("damped_admm", 0.25), ("admm", 1.0)):
        for rho, charge in itertools.product((1.0, 100.0), (0.0, 0.003, 0.03, 0.3)):
            rows.append(
                {
                    "family": family,
                    "dual_step": dual,
                    "rho_final": rho,
                    "nonzero_penalty": charge,
                    "numeric_lr": 0.01,
                    "latent_lr": 0.01,
                    "radius": 0.5,
                    "mode": "joint",
                }
            )
    return [{**row, "configuration": i} for i, row in enumerate(rows)]


def prepare(parent, output, projection="lloyd"):
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((parent / "protocol.json").read_text())
    spec = json.loads((parent / "specification.json").read_text())
    spec.update(
        name="deepsets_consensus_exact"
        if projection == "dynamic"
        else "deepsets_consensus_full",
        full_connectivity_search=True,
    )
    plan["search"].update(
        pilot_seeds=[281, 282],
        confirmation_seeds=[283, 284, 285],
        pilot_steps=12000,
        confirmation_steps=12000,
        restarts=1,
        numeric_warmup=2000,
        top_configurations=3,
    )
    plan["sharing_heldout_offset"] = 9900000 if projection == "dynamic" else 8900000
    plan["consensus"] = {
        "block_steps": 50,
        "lloyd_iterations": 12,
        "rho_initial": 0.01,
        "projection": projection,
    }
    plan["rules"] = [
        "Full256-position search, no teacher mask/groups/edge count/numerical values. All positions can change until the source indexator is frozen.",
        "Optimize continuous Z and numerical heads on support with normalized task loss plus rho/2*mean((Z-Q+U)^2). Q=C[I] is a separate hard signed codebook projection.",
        "After2000 unstructured dense warmup steps initialize four positive code magnitudes from empirical Z quantiles. Continue10000 updates; all positions remain available. Warmup uses its best query state.",
        "Every50 updates jointly update I and C by12 signed Lloyd iterations minimizing mean((Z+U-C[I])^2)+2*lambda/rho*mean(I!=0); fixed zero code. Empty centers retain their old value.",
        "Then U += dual_step*(Z-Q). Compare no dual, dual_step.25 and1. Rho increases geometrically from.01 to1 or100; rescale U to preserve the unscaled multiplier when rho changes.",
        "This is inexact nonconvex consensus optimization. The Lloyd step is locally optimized, the inner continuous subproblem is not solved exactly, and no global convergence or structural recovery guarantee is claimed.",
        "Continuous search forward is Z, not a soft mixture. Save Q indexators explicitly and select them only after hard3000-step C refits with1inherited+3freshGaussian numerical initializations.",
        "Query selects numerical checkpoints, independent selection selects I/config by mean four-start loss. Pilot281/282 selects one setting per family; confirmation283/284/285 uses frozen settings.",
        "Fresh32 on the source function is an optimizer diagnostic, not structural transfer. Any new-coefficient transfer must keep learned I fixed and initialize all numerical parameters freshly.",
        "No new software tests; do not overwrite earlier result folders.",
    ]
    if projection == "dynamic":
        plan["rules"][3] = (
            "Every50 updates globally solve the scalar signed projection: sort abs(Z+U), allow a fixed-zero prefix, use dynamic programming for at most four contiguous positive clusters and charge2*lambda/rho per nonzero element. Signs restore signed labels. Double precision interval costs; no known support size, groups or code magnitudes."
        )
        plan["rules"][5] = (
            "The projection subproblem is globally solved by one-dimensional dynamic programming, but the overall continuous/nonconvex consensus optimization has no global recovery guarantee. Compare the same initial seeds, minibatches, settings and budgets as the Lloyd series."
        )
    for name in ("source.npz", "teacher_audit_only.npz"):
        shutil.copy2(parent / name, output / name)
    save_json(output / "protocol.json", plan)
    save_json(output / "specification.json", spec)
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for name in (
        "consensus_search.py",
        "consensus_report.py",
        "codebook_projection.py",
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
            "candidates": configurations(),
            "rules": plan["rules"],
            "source_sha256": digest(output / "source.npz"),
            "snapshot_sha256": {p.name: digest(p) for p in snapshot.glob("*.py")},
            "references": [
                "https://arxiv.org/abs/1811.01907",
                "https://journal.r-project.org/articles/RJ-2011-015/",
            ],
        },
    )


@torch.no_grad()
def project(values, centers, charge, rho, iterations):
    codes = centers.shape[1]
    nonzero = torch.arange(2 * codes + 1, device=values.device) != 0
    for iteration in range(iterations + 1):
        dictionary = torch.cat((centers[:, :1] * 0, centers, -centers), 1)
        cost = (values[..., None] - dictionary[:, None, None]).square()
        cost += (2 * charge / rho)[:, None, None, None] * nonzero
        labels = cost.argmin(-1)
        if iteration == iterations:
            break
        signed = torch.where(labels > codes, codes - labels, labels)
        for code in range(1, codes + 1):
            active = signed.abs() == code
            count = active.sum((1, 2))
            mean = (values.abs() * active).sum((1, 2)) / count.clamp_min(1)
            centers[:, code - 1] = torch.where(count > 0, mean, centers[:, code - 1])
        centers = centers.sort(-1).values
    quantized = dictionary.gather(1, labels.flatten(1)).reshape(values.shape)
    return labels, centers, quantized


def search(output, plan, spec, rows, phase, device):
    folder = output / phase
    folder.mkdir(exist_ok=False)
    save_json(folder / "rows.json", rows)
    settings = plan["search"]
    codes, classes = spec["codes"], 2 * spec["codes"] + 1
    arrays = []
    for row in rows:
        rng = np.random.default_rng(plan["seed"] + 100003 * row["seed"])
        matrix = rng.normal(size=(spec["inputs"], spec["hidden"]))
        matrix *= row["radius"] * math.sqrt(spec["hidden"]) / np.linalg.norm(matrix)
        arrays.append(matrix)
    latent = torch.tensor(np.array(arrays), device=device, dtype=torch.float32)
    latent.requires_grad_()
    heads = torch.zeros(len(rows), 1, 3, device=device)
    heads[..., 0], heads[..., 1], heads[..., 2] = 0.1, 1.0, spec["target_mean"]
    heads.requires_grad_()
    centers = torch.zeros(len(rows), codes, device=device)
    numeric_optimizer = BatchedOptimizer(heads, rows)
    latent_optimizer = BatchedOptimizer(latent, rows, "latent_lr")
    data = source_tensors(output, device)
    dual, quantized = torch.zeros_like(latent), torch.zeros_like(latent)
    labels = torch.zeros_like(latent, dtype=torch.long)
    charge = latent.new_tensor([row["nonzero_penalty"] for row in rows])
    dual_step = latent.new_tensor([row["dual_step"] for row in rows])
    rho_final = latent.new_tensor([row["rho_final"] for row in rows])
    rho_initial = plan["consensus"]["rho_initial"]
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + 1001)
    warmup = settings["numeric_warmup"]
    steps = settings[phase + "_steps"]
    levels = (torch.arange(codes, device=device) + 0.5) / codes
    checkpoints = {round(fraction * steps) for fraction in settings["snapshots"]}
    best_warm = latent.new_full((len(rows),), torch.inf)
    best_latent, best_heads = latent.detach().clone(), heads.detach().clone()
    previous_rho = latent.new_full((len(rows),), rho_initial)
    history, saved = [], []
    started = time.perf_counter()
    for step in range(steps + 1):
        if step == warmup:
            with torch.no_grad():
                latent.copy_(best_latent)
                heads.copy_(best_heads)
                centers = torch.quantile(latent.abs().flatten(1), levels, dim=1).T
                for optimizer in (numeric_optimizer, latent_optimizer):
                    optimizer.first.zero_()
                    optimizer.second.zero_()
                    optimizer.velocity.zero_()
                    optimizer.steps.zero_()
        block_step = plan["consensus"]["block_steps"]
        progress = max(0, (step - warmup) // block_step * block_step) / (steps - warmup)
        rho = rho_initial * (rho_final / rho_initial).pow(progress)
        if step >= warmup and (step - warmup) % plan["consensus"]["block_steps"] == 0:
            with torch.no_grad():
                dual *= (previous_rho / rho)[:, None, None]
                if plan["consensus"].get("projection", "lloyd") == "dynamic":
                    labels, centers, quantized = exact_signed_projection(
                        latent + dual, codes, charge, rho
                    )
                else:
                    labels, centers, quantized = project(
                        latent + dual,
                        centers,
                        charge,
                        rho,
                        plan["consensus"]["lloyd_iterations"],
                    )
                dual += dual_step[:, None, None] * (latent - quantized)
                previous_rho = rho
        numeric = torch.cat((centers[:, None], heads), -1)
        if step % settings["checkpoint_every"] == 0 or step == steps:
            with torch.no_grad():
                query = score(
                    data["query_x"], data["query_y"], latent[:, None], numeric, spec
                )
                training = score(
                    data["support_x"], data["support_y"], latent[:, None], numeric, spec
                )
                hard_query = score(
                    data["query_x"], data["query_y"], quantized[:, None], numeric, spec
                )
                if step < warmup:
                    better = query[:, 0] < best_warm
                    best_warm[better] = query[better, 0]
                    best_latent[better] = latent[better]
                    best_heads[better] = heads[better]
                if not torch.isfinite(query).all():
                    raise FloatingPointError("Non-finite consensus optimization")
                history.append(
                    {
                        "step": step,
                        "query_mse": query.tolist(),
                        "train_mse": training.tolist(),
                        "hard_query_mse": hard_query.tolist(),
                        "active_edges": (labels != 0).sum((1, 2)).tolist(),
                        "rho": rho.tolist(),
                        "consensus_mse": (latent - quantized)
                        .square()
                        .mean((1, 2))
                        .tolist(),
                        "dual_norm": dual.flatten(1).norm(dim=1).tolist(),
                        "codebook": centers.tolist(),
                        "seconds": time.perf_counter() - started,
                    }
                )
                save_json(folder / "history.json", history)
                if step in checkpoints:
                    index = torch.where(labels > codes, codes - labels, labels)
                    fresh_numeric = initial_numeric(
                        [{**r, "aligned_initial_codes": False} for r in rows],
                        3,
                        spec,
                        plan["seed"] + 51017,
                        device,
                        indices=index,
                    )
                    saved.append(
                        {
                            "step": step,
                            "probabilities": F.one_hot(labels, classes)
                            .float()
                            .cpu()
                            .numpy(),
                            "numeric": torch.cat((numeric, fresh_numeric), 1)
                            .cpu()
                            .numpy(),
                            "latent": latent.cpu().numpy(),
                            "dual": dual.cpu().numpy(),
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
                        f"{phase} step={step} roots={len(rows)} continuous_query_NMSE={query.mean().item() / spec['target_variance']:.5g} hard_query_NMSE={hard_query.mean().item() / spec['target_variance']:.5g} seconds={time.perf_counter() - started:.1f}",
                        flush=True,
                    )
        if step == steps:
            break
        ids = torch.randint(
            len(data["support_x"]),
            (settings["batch"],),
            generator=generator,
            device=device,
        )
        prediction = predict(data["support_x"][ids], latent[:, None], numeric, spec)
        task_loss = (prediction - data["support_y"][ids]).square().mean((1, 2)) / spec[
            "target_variance"
        ]
        penalty = (
            (latent - quantized + dual).square().mean((1, 2)) * rho / 2
            if step >= warmup
            else task_loss * 0
        )
        regularizer = (
            settings["numeric_l2"]
            * (heads.square().sum((1, 2)) + latent.square().sum((1, 2)))
            / 2
        )
        gradients = torch.autograd.grad(
            (task_loss + penalty + regularizer).sum(), (heads, latent)
        )
        numeric_optimizer.step(
            clip_gradient(gradients[0], settings["gradient_clip"], (1, 2))
        )
        latent_optimizer.step(
            clip_gradient(gradients[1], settings["gradient_clip"], (1, 2))
        )
    save_json(
        folder / "complete.json",
        {
            "seconds": time.perf_counter() - started,
            "steps": steps,
            "roots": len(rows),
            "numeric_restarts": 1,
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
                    row["configuration"] for row in families.values()
                ],
                "ranking": ranking,
            },
        )
    else:
        save_json(output / "family_selection.json", {"families": families})
        names = (
            "source.npz",
            "family_selection.json",
            "architecture_selection.json",
            "confirmation/hard_fits.npz",
            "confirmation/snapshots.npz",
            "confirmation/rows.json",
        )
        save_json(
            output / "source_frozen.json",
            {
                "frozen_utc": datetime.now(UTC).isoformat(),
                "sha256": {name: digest(output / name) for name in names},
            },
        )
    print(
        json.dumps(
            {
                name: row["selection_mse"] / spec["target_variance"]
                for name, row in families.items()
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--projection", choices=("lloyd", "dynamic"), default="lloyd")
    parser.add_argument(
        "--stage", choices=("prepare", "pilot", "confirmation", "fresh"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        if args.parent is None:
            parser.error("--parent is required for preparation")
        prepare(args.parent, args.root, args.projection)
    elif args.stage == "fresh":
        fresh(args.root, torch.device(args.device))
    else:
        run_phase(args.root, args.stage, torch.device(args.device))


if __name__ == "__main__":
    main()
