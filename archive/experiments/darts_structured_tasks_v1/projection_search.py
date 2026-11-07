"""Full nearest-code search with a cost for choosing nonzero codes."""

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
    initial_numeric,
    predict,
    score,
    signed_values,
    source_tensors,
)
from .run import refit_search
from .sharing_ablation import fresh


def configurations():
    candidates = []
    for family, center_loss in (("nearest", 0.01), ("task_centers", 0.0)):
        for rate in (0.003, 0.01):
            candidates.append(
                {
                    "family": family,
                    "numeric_lr": rate,
                    "latent_lr": rate,
                    "zero_cost": 0.0,
                    "center_loss": center_loss,
                    "warmup": 0,
                }
            )
    for family, warmup in (("proximal", 0), ("proximal_warm", 3000)):
        for cost, rate in itertools.product((0.001, 0.01, 0.1, 1.0), (0.003, 0.01)):
            candidates.append(
                {
                    "family": family,
                    "numeric_lr": rate,
                    "latent_lr": rate,
                    "zero_cost": cost,
                    "center_loss": 0.0,
                    "warmup": warmup,
                }
            )
    return [
        {**row, "configuration": i, "radius": 0.5, "commitment": 0.01, "mode": "joint"}
        for i, row in enumerate(candidates)
    ]


def prepare(parent, output):
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((parent / "protocol.json").read_text())
    spec = json.loads((parent / "specification.json").read_text())
    spec.update(name="deepsets_projection_full", full_connectivity_search=True)
    plan["search"].update(
        pilot_seeds=[261, 262],
        confirmation_seeds=[263, 264, 265],
        pilot_steps=12000,
        confirmation_steps=12000,
        restarts=1,
        numeric_warmup=0,
        top_configurations=4,
    )
    plan["sharing_heldout_offset"] = 7900000
    plan["rules"] = [
        "Full256-position search. No mask, point partition, teacher edge count, true code values or true indexator enter the learner.",
        "Signed dictionary[0,C1,...,C4,-C1,...,-C4]. Learn a latent real matrix Z; choose the code minimizing normalized squared distance plus a nonzero-code charge.",
        "The charge is a proximal assignment rule, not an unbiased task-gradient estimate for L0. It widens the zero decision region independently of tiny nonzero centers.",
        "The hard numerical forward is exactly the selected code. Z uses identity STE, C receives selected-code task gradients. This gradient estimator is biased.",
        "Compare nearest with centering loss, nearest with task-only centers, charged projection, and3000-step full dense numerical warmup followed by charged projection.",
        "Warmup trains every matrix position and freezes none. At the transition use the best query numerical warmup state and empirical absolute-value quantile centers; all256 positions remain trainable.",
        "All searches12000 updates including warmup. Nonzero charge ramps over the first60percent of discrete training; no late entropy or known edge budget.",
        "One numeric search replica per root. Every hard snapshot gets one inherited plus three fresh Gaussian C fits of3000 steps; query selects C checkpoints, mean selection chooses the structure/configuration.",
        "Pilot261/262 selects among registered settings per family, then confirmation263/264/265 runs frozen settings. Teacher is read only after source/fresh freeze.",
        "Fresh-C32 repeats use the same function and measure numerical convergence, not structural transfer. New-coefficient functions require a separate frozen-I evaluation.",
        "No new software tests. Do not overwrite previous experiments.",
    ]
    for name in ("source.npz", "teacher_audit_only.npz"):
        shutil.copy2(parent / name, output / name)
    save_json(output / "protocol.json", plan)
    save_json(output / "specification.json", spec)
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for name in (
        "projection_search.py",
        "projection_report.py",
        "model.py",
        "train.py",
        "data.py",
        "run.py",
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
                "https://arxiv.org/abs/1911.04951",
                "https://arxiv.org/abs/2305.08842",
                "https://arxiv.org/abs/1810.00861",
            ],
        },
    )


def search(output, plan, spec, rows, phase, device):
    folder = output / phase
    folder.mkdir(exist_ok=False)
    save_json(folder / "rows.json", rows)
    settings = plan["search"]
    codes, classes = spec["codes"], 2 * spec["codes"] + 1
    arrays = []
    for row in rows:
        rng = np.random.default_rng(plan["seed"] + 100003 * row["seed"])
        z = rng.normal(size=(spec["inputs"], spec["hidden"]))
        z *= row["radius"] * math.sqrt(spec["hidden"]) / np.linalg.norm(z)
        arrays.append(z)
    latent = torch.tensor(
        np.array(arrays), dtype=torch.float32, device=device, requires_grad=True
    )
    numeric = torch.zeros(len(rows), 1, codes + 3, device=device)
    numeric[..., codes] = 0.1
    numeric[..., codes + 1] = 1.0
    numeric[..., codes + 2] = spec["target_mean"]
    numeric.requires_grad_()
    latent_optimizer = BatchedOptimizer(latent, rows, "latent_lr")
    numeric_optimizer = BatchedOptimizer(numeric, rows)
    data = source_tensors(output, device)
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + 1001)
    warmups = np.array([r["warmup"] for r in rows])
    warm_steps = torch.tensor(warmups, device=device)
    charges = latent.new_tensor([r["zero_cost"] for r in rows])
    center_strength = latent.new_tensor([r["center_loss"] for r in rows])
    commitment = latent.new_tensor([r["commitment"] for r in rows])
    best_warm = latent.new_full((len(rows),), torch.inf)
    best_latent, best_numeric = latent.detach().clone(), numeric.detach().clone()
    levels = (torch.arange(codes, device=device) + 0.5) / codes
    steps = settings[phase + "_steps"]
    checkpoints = {round(f * steps) for f in settings["snapshots"]}
    history, saved = [], []
    started = time.perf_counter()
    maximum_forward_error = 0.0
    for step in range(steps + 1):
        transitioning = np.flatnonzero(warmups == step)
        if len(transitioning):
            ids = torch.tensor(transitioning, device=device)
            with torch.no_grad():
                if step:
                    latent[ids] = best_latent[ids]
                    numeric[ids] = best_numeric[ids]
                centers = torch.quantile(latent[ids].abs().flatten(1), levels, dim=1).T
                numeric[ids, 0, :codes] = centers.clamp_min(1e-4)
                for optimizer in (latent_optimizer, numeric_optimizer):
                    optimizer.first[ids] = 0
                    optimizer.second[ids] = 0
                    optimizer.velocity[ids] = 0
                    optimizer.steps[ids] = 0
        progress = ((step - warm_steps) / (steps - warm_steps)).clamp(0, 1)
        is_warm = step < warm_steps
        ramp = (progress / 0.6).clamp(0, 1)
        values = signed_values(numeric, codes)[:, 0]
        energy = numeric[..., :codes].detach().square().mean((1, 2)).clamp_min(1e-8)
        distance = (latent[..., None] - values[:, None, None]).square() / energy[
            :, None, None, None
        ]
        nonzero = torch.arange(classes, device=device) != 0
        cost = distance + (charges * ramp)[:, None, None, None] * nonzero
        selected = cost.argmin(-1)
        nearest = values.gather(1, selected.flatten(1)).reshape(latent.shape)
        # Exactly hard in forward; selected C and latent Z have distinct gradients.
        hard = nearest + (latent - latent.detach())
        weights = torch.where(is_warm[:, None, None], latent, hard)
        latent_penalty = (latent - nearest.detach()).square().mean((1, 2)) / energy
        center_penalty = (latent.detach() - nearest).square().mean((1, 2)) / energy
        penalty = (
            ramp
            * (commitment * latent_penalty + center_strength * center_penalty)
            * (~is_warm)
        )
        if step % settings["checkpoint_every"] == 0 or step == steps:
            with torch.no_grad():
                query = score(
                    data["query_x"], data["query_y"], weights[:, None], numeric, spec
                )
                training = score(
                    data["support_x"],
                    data["support_y"],
                    weights[:, None],
                    numeric,
                    spec,
                )
                better = is_warm & (query[:, 0] < best_warm)
                best_warm[better] = query[better, 0]
                best_latent[better] = latent[better]
                best_numeric[better] = numeric[better]
                if not torch.isfinite(query).all():
                    raise FloatingPointError("Non-finite charged projection search")
                if (~is_warm).any():
                    error = float((weights[~is_warm] - nearest[~is_warm]).abs().max())
                    maximum_forward_error = max(maximum_forward_error, error)
                    if error != 0:
                        raise RuntimeError(
                            "Discrete search forward contains a soft mixture"
                        )
                history.append(
                    {
                        "step": step,
                        "query_mse": query.tolist(),
                        "train_mse": training.tolist(),
                        "active_edges": (selected != 0).sum((1, 2)).tolist(),
                        "effective_nonzero_cost": (charges * ramp).tolist(),
                        "latent_commitment": latent_penalty.tolist(),
                        "codebook": values[:, 1 : codes + 1].tolist(),
                        "seconds": time.perf_counter() - started,
                    }
                )
                save_json(folder / "history.json", history)
                if step in checkpoints:
                    indices = torch.where(selected > codes, codes - selected, selected)
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
                            "probabilities": F.one_hot(selected, classes)
                            .float()
                            .cpu()
                            .numpy(),
                            "numeric": torch.cat((numeric, fresh_numeric), 1)
                            .cpu()
                            .numpy(),
                            "search_numeric": numeric.cpu().numpy(),
                            "latent": latent.cpu().numpy(),
                        }
                    )
                    np.savez_compressed(
                        folder / "snapshots.npz",
                        steps=np.array([r["step"] for r in saved]),
                        **{
                            key: np.stack([r[key] for r in saved])
                            for key in saved[0]
                            if key != "step"
                        },
                    )
                if step % 1000 == 0 or step == steps:
                    print(
                        f"{phase} step={step} roots={len(rows)} query_NMSE={query.mean().item() / spec['target_variance']:.5g} seconds={time.perf_counter() - started:.1f}",
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
        prediction = predict(data["support_x"][ids], weights[:, None], numeric, spec)
        loss = (prediction - data["support_y"][ids]).square().mean((1, 2)) / spec[
            "target_variance"
        ]
        objective = (
            loss
            + penalty
            + settings["numeric_l2"]
            * (numeric.square().sum((1, 2)) + latent.square().sum((1, 2)))
            / 2
        )
        numeric_gradient, latent_gradient = torch.autograd.grad(
            objective.sum(), (numeric, latent)
        )
        numeric_optimizer.step(
            clip_gradient(numeric_gradient, settings["gradient_clip"], (1, 2))
        )
        latent_optimizer.step(
            clip_gradient(latent_gradient, settings["gradient_clip"], (1, 2))
        )
    save_json(
        folder / "complete.json",
        {
            "seconds": time.perf_counter() - started,
            "steps": steps,
            "roots": len(rows),
            "numeric_restarts": 1,
            "maximum_hard_forward_error": maximum_forward_error,
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
        candidates = [r for r in candidates if r["configuration"] in chosen]
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
                name: r["selection_mse"] / spec["target_variance"]
                for name, r in families.items()
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
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
    elif args.stage == "fresh":
        fresh(args.root, torch.device(args.device))
    else:
        run_phase(args.root, args.stage, torch.device(args.device))


if __name__ == "__main__":
    main()
