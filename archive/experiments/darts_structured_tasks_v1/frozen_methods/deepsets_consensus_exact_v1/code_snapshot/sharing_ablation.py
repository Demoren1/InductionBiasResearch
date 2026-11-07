"""Paired DeepSets ablations of numerical assignment commitment."""

import argparse
import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .data import BASE, digest, save_json
from .model import (
    BatchedOptimizer,
    clip_gradient,
    fixed_weights,
    hard_indices,
    initial_numeric,
    predict,
    score,
    signed_values,
    soft_weights,
    source_tensors,
)
from .run import refit_search
from .train import fit, save_fit


def configurations():
    common = {
        "numeric_lr": 0.01,
        "architecture_lr": 0.1,
        "radius": 0.5,
        "sparsity": 0.03,
        "mode": "joint",
        "commitment": 0.0,
        "commitment_start": 0.0,
        "commitment_finish": 0.0,
        "minimum_temperature": 0.03,
        "cooling_start": 0.3,
        "late_entropy": 0.02,
    }
    candidates = [
        {**common, "family": "baseline"},
        {**common, "family": "no_sparsity", "sparsity": 0.0},
        {**common, "family": "no_entropy", "late_entropy": 0.0},
        {
            **common,
            "family": "gentle_cooling",
            "cooling_start": 0.5,
            "minimum_temperature": 0.1,
        },
    ]
    schedules = (
        ("constant", 0.0, 0.0, {}),
        ("early", 0.1, 0.7, {}),
        ("late", 0.5, 0.95, {}),
        ("early_no_sparsity", 0.1, 0.7, {"sparsity": 0.0}),
        ("early_no_entropy", 0.1, 0.7, {"late_entropy": 0.0}),
        ("early_slow_alpha", 0.1, 0.7, {"architecture_lr": 0.03}),
    )
    for family, start, finish, changes in schedules:
        for strength in (0.01, 0.1, 1.0):
            candidates.append(
                {
                    **common,
                    **changes,
                    "family": "commit_" + family,
                    "commitment": strength,
                    "commitment_start": start,
                    "commitment_finish": finish,
                }
            )
    return [{**row, "configuration": number} for number, row in enumerate(candidates)]


def search(output, plan, specification, rows, phase, device):
    """Optimize independent numeric replicas and a shared P per search seed."""
    destination = output / phase
    destination.mkdir(parents=True, exist_ok=False)
    save_json(destination / "rows.json", rows)
    settings = plan["search"]
    steps = settings[phase + "_steps"]
    codes = specification["codes"]
    shape = (specification["inputs"], specification["hidden"], 2 * codes + 1)
    alpha = torch.tensor(
        np.array(
            [
                np.random.default_rng(plan["seed"] + 100003 * row["seed"]).normal(
                    0, settings["initial_logit_std"], shape
                )
                for row in rows
            ]
        ),
        dtype=torch.float32,
        device=device,
    )
    probability = settings["zero_probability"]
    alpha[..., 0] += math.log(2 * codes * probability / (1 - probability))
    alpha.requires_grad_()
    numeric = initial_numeric(
        rows,
        settings["restarts"],
        specification,
        plan["seed"] + 901,
        device,
        probabilities=alpha.detach().softmax(-1),
    ).requires_grad_()
    numeric_optimizer = BatchedOptimizer(numeric, rows)
    alpha_optimizer = BatchedOptimizer(alpha, rows, "architecture_lr")
    data = source_tensors(output, device)
    variance = specification["target_variance"]
    constants = {
        key: numeric.new_tensor([row[key] for row in rows])
        for key in (
            "sparsity",
            "commitment",
            "commitment_start",
            "commitment_finish",
            "minimum_temperature",
            "cooling_start",
            "late_entropy",
        )
    }
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + 1001)
    checkpoints = {round(fraction * steps) for fraction in settings["snapshots"]}
    histories, saved = [], []
    started = time.perf_counter()

    def losses(probabilities, ids):
        weights = soft_weights(probabilities, numeric, codes)
        prediction = predict(data["support_x"][ids], weights, numeric, specification)
        return (prediction - data["support_y"][ids]).square().mean(-1) / variance

    for _ in range(settings["numeric_warmup"]):
        ids = torch.randint(
            len(data["support_x"]),
            (settings["batch"],),
            device=device,
            generator=generator,
        )
        loss = losses(alpha.detach().softmax(-1), ids)
        regularized = loss + settings["numeric_l2"] * numeric.square().sum(-1) / 2
        gradient = torch.autograd.grad(regularized.sum(), numeric)[0]
        numeric_optimizer.step(
            clip_gradient(gradient, settings["gradient_clip"], (-1,))
        )

    for step in range(steps + 1):
        progress = step / max(steps, 1)
        cooling = (
            (progress - constants["cooling_start"]) / (1 - constants["cooling_start"])
        ).clamp(0, 1)
        temperature = constants["minimum_temperature"].pow(cooling)
        probabilities = (alpha / temperature[:, None, None, None]).softmax(-1)
        code_values = signed_values(numeric.detach(), codes)
        mean = torch.einsum("sdhk,srk->srdh", probabilities, code_values)
        second = torch.einsum("sdhk,srk->srdh", probabilities, code_values.square())
        scale = numeric.detach()[..., :codes].square().mean(-1).clamp_min(1e-8)
        commitment = (
            (second - mean.square()).clamp_min(0).mean((-1, -2)) / scale
        ).mean(-1)
        entropy = (
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean((1, 2))
        )
        ramp = (
            (progress - constants["commitment_start"])
            / (
                constants["commitment_finish"] - constants["commitment_start"]
            ).clamp_min(1e-8)
        ).clamp(0, 1)
        ramp = torch.where(
            constants["commitment_finish"] == 0, torch.ones_like(ramp), ramp
        )
        strength = constants["commitment"] * ramp
        if step % settings["checkpoint_every"] == 0 or step == steps:
            with torch.no_grad():
                weights = soft_weights(probabilities, numeric, codes)
                query = score(
                    data["query_x"], data["query_y"], weights, numeric, specification
                )
                train = score(
                    data["support_x"][:512],
                    data["support_y"][:512],
                    weights,
                    numeric,
                    specification,
                )
                indices = hard_indices(probabilities)
                hard = score(
                    data["query_x"],
                    data["query_y"],
                    fixed_weights(indices.long(), numeric, codes),
                    numeric,
                    specification,
                )
                histories.append(
                    {
                        "step": step,
                        "train_mse": train.cpu().tolist(),
                        "query_mse": query.cpu().tolist(),
                        "hard_query_mse": hard.cpu().tolist(),
                        "entropy": entropy.cpu().tolist(),
                        "commitment": commitment.cpu().tolist(),
                        "commitment_strength": strength.cpu().tolist(),
                        "temperature": temperature.cpu().tolist(),
                        "seconds": time.perf_counter() - started,
                    }
                )
                if not torch.isfinite(query).all():
                    raise FloatingPointError("Non-finite architecture loss")
                save_json(destination / "history.json", histories)
                if step in checkpoints:
                    saved.append(
                        {
                            "step": step,
                            "probabilities": probabilities.detach().cpu().numpy(),
                            "numeric": numeric.detach().cpu().numpy(),
                        }
                    )
                    np.savez_compressed(
                        destination / "snapshots.npz",
                        steps=np.array([item["step"] for item in saved]),
                        probabilities=np.stack(
                            [item["probabilities"] for item in saved]
                        ),
                        numeric=np.stack([item["numeric"] for item in saved]),
                    )
                print(
                    f"{phase} step={step} rows={len(rows)} "
                    f"soft_NMSE={query.mean().item() / variance:.5f} "
                    f"hard_NMSE={hard.mean().item() / variance:.5f} "
                    f"seconds={time.perf_counter() - started:.1f}",
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
        support_loss = losses(probabilities, ids)
        regularized = (
            support_loss + settings["numeric_l2"] * numeric.square().sum(-1) / 2
        )
        numeric_gradient = torch.autograd.grad(
            regularized.sum(), numeric, retain_graph=True
        )[0]
        objective = support_loss.mean(-1)
        objective = objective + constants["sparsity"] * (
            1 - probabilities[..., 0]
        ).mean((1, 2))
        objective = (
            objective
            + (constants["late_entropy"] * max(0, 2 * progress - 1) * entropy)
            + strength * commitment
        )
        alpha_gradient = torch.autograd.grad(objective.sum(), alpha)[0]
        numeric_optimizer.step(
            clip_gradient(numeric_gradient, settings["gradient_clip"], (-1,))
        )
        alpha_optimizer.step(
            clip_gradient(alpha_gradient, settings["alpha_clip"], (1, 2, 3))
        )
        with torch.no_grad():
            alpha.sub_(alpha.mean(-1, keepdim=True))
    save_json(
        destination / "complete.json",
        {
            "seconds": time.perf_counter() - started,
            "steps": steps,
            "roots": len(rows),
            "numeric_restarts": settings["restarts"],
            "numeric_updates_per_root": numeric_optimizer.steps.flatten().tolist(),
        },
    )


def prepare(parent, output):
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((parent / "protocol.json").read_text())
    specification = json.loads((parent / "specification.json").read_text())
    specification["name"] = "deepsets_sharing_ablation"
    plan["search"].update(
        {
            "pilot_seeds": [43, 44],
            "confirmation_seeds": [111, 112, 113],
            "pilot_steps": 12000,
            "confirmation_steps": 12000,
            "top_configurations": 10,
        }
    )
    save_json(output / "protocol.json", plan)
    save_json(output / "specification.json", specification)
    for name in ("source.npz", "teacher_audit_only.npz"):
        shutil.copy2(parent / name, output / name)
    snapshots = output / "code_snapshot"
    snapshots.mkdir()
    for name in ("sharing_ablation.py", "model.py", "train.py", "run.py", "data.py"):
        shutil.copy2(BASE / name, snapshots / name)
    save_json(
        output / "registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "status": "exploratory ablation on previously studied DeepSets task",
            "parent": str(parent),
            "source_sha256": digest(output / "source.npz"),
            "candidates": configurations(),
            "rules": [
                "Exact signed dictionary [0,C1,...,C4,-C1,...,-C4].",
                "No teacher grouping, point roles, mask or active-edge count enters search.",
                "Paired initialization, warmup, minibatch stream and numerical budget per search seed.",
                "Commitment is normalized categorical numerical variance, including zero; C is detached in this penalty.",
                "Pilot seeds43/44 were studied previously; confirmation111/112/113 are new search initializations, not new functions.",
                "Select one setting per ablation family by mean pilot hard-refit selection MSE, then confirm on three new seeds.",
                "Each numerical checkpoint is selected by query; architecture snapshots by mean selection over four replicas.",
                "Freeze architecture choices before inspecting teacher recovery or generating the new held-out observations.",
                "No software tests. Keep earlier experiment artifacts intact.",
                "If commitment fails, investigate discrete-forward or weight-distance assignment in a separately registered continuation.",
            ],
            "snapshot_sha256": {
                path.name: digest(path) for path in snapshots.glob("*.py")
            },
        },
    )


def run_phase(output, phase, device):
    plan = json.loads((output / "protocol.json").read_text())
    spec = json.loads((output / "specification.json").read_text())
    candidates = json.loads((output / "registration.json").read_text())["candidates"]
    if phase == "confirmation":
        selected = json.loads((output / "pilot_selection.json").read_text())
        ids = selected["top_configurations"]
        candidates = [row for row in candidates if row["configuration"] in ids]
    rows = [
        {**row, "seed": seed}
        for row in candidates
        for seed in plan["search"][phase + "_seeds"]
    ]
    search(output, plan, spec, rows, phase, device)
    refit_search(output, plan, spec, phase, device)
    if phase == "pilot":
        ranking = json.loads((output / "pilot/ranking.json").read_text())
        winners = {}
        for entry in ranking:
            family = candidates[entry["configuration"]]["family"]
            winners.setdefault(family, entry)
        save_json(
            output / "pilot_selection.json",
            {
                "criterion": "Best mean pilot hard-refit selection per family",
                "top_configurations": [
                    entry["configuration"] for entry in winners.values()
                ],
                "families": winners,
                "ranking": ranking,
            },
        )
    else:
        ranking = json.loads((output / "confirmation/ranking.json").read_text())
        choices = {entry["per_seed"][0]["family"]: entry for entry in ranking}
        save_json(
            output / "family_selection.json",
            {"criterion": "Mean selection across new seeds", "families": choices},
        )
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
                    )
                },
            },
        )


def fresh(output, device):
    plan = json.loads((output / "protocol.json").read_text())
    spec = json.loads((output / "specification.json").read_text())
    selection = json.loads((output / "family_selection.json").read_text())["families"]
    with np.load(output / "confirmation/hard_fits.npz") as archive:
        fitted = dict(archive)
    proposals, rows = [], []
    for family, result in selection.items():
        chosen = min(result["per_seed"], key=lambda item: item["selection_mse"])
        index = fitted["indices"][chosen["row"]]
        for radius in (0.5, 1.5):
            for rate in (0.003, 0.01, 0.03):
                rows.append(
                    {
                        "family": family,
                        "source_row": chosen["row"],
                        "radius": radius,
                        "numeric_lr": rate,
                        "seed": 121,
                        "aligned_initial_codes": False,
                    }
                )
                proposals.append(index)
    calibration = fit(
        output, plan, spec, rows, device, indices=np.array(proposals), restarts=8
    )
    save_fit(output / "fresh_calibration", calibration, rows)
    choices = {}
    for number, row in enumerate(rows):
        value = float(calibration["selection_mse"][number].mean())
        if row["family"] not in choices or value < choices[row["family"]]["loss"]:
            choices[row["family"]] = {"row": number, "loss": value}
    eval_rows = [{**rows[choice["row"]], "seed": 122} for choice in choices.values()]
    eval_indices = np.array([proposals[c["row"]] for c in choices.values()])
    result = fit(
        output, plan, spec, eval_rows, device, indices=eval_indices, restarts=32
    )
    result["indices"] = eval_indices
    save_fit(output / "fresh_evaluation", result, eval_rows)
    save_json(
        output / "fresh_selection.json",
        {
            "calibration": choices,
            "rows": eval_rows,
            "query_selected_restarts": result["query_mse"].argmin(-1).tolist(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "pilot", "confirmation", "fresh"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare(args.parent, args.output)
    elif args.stage == "fresh":
        fresh(args.output, torch.device(args.device))
    else:
        run_phase(args.output, args.stage, torch.device(args.device))


if __name__ == "__main__":
    main()
