"""Compare categorical and weight-distance sharing on a learned DARTS mask."""

import argparse
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
    hard_indices,
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
    for method in ("logits", "distance_soft", "nearest_ste"):
        for warmup in (0, 1000, 3000):
            for rate in (0.003, 0.01):
                candidates.append(
                    {
                        "family": f"{method}_w{warmup}",
                        "method": method,
                        "warmup": warmup,
                        "numeric_lr": rate,
                        "architecture_lr": 0.03,
                        "latent_lr": rate,
                        "radius": 0.5,
                        "commitment": 0.0 if method == "logits" else 0.01,
                        "mode": "joint",
                    }
                )
    for rate in (0.003, 0.01):
        candidates.append(
            {
                "family": "logits_commit_w0",
                "method": "logits",
                "warmup": 0,
                "numeric_lr": rate,
                "architecture_lr": 0.03,
                "latent_lr": rate,
                "radius": 0.5,
                "commitment": 0.1,
                "mode": "joint",
            }
        )
    return [{**row, "configuration": number} for number, row in enumerate(candidates)]


def prepare(parent, output):
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((parent / "protocol.json").read_text())
    spec = json.loads((parent / "specification.json").read_text())
    spec["name"] = "deepsets_sharing_geometry"
    plan["search"].update(
        {
            "pilot_seeds": [151, 152],
            "confirmation_seeds": [153, 154, 155],
            "pilot_steps": 12000,
            "confirmation_steps": 12000,
            "numeric_warmup": 0,
            "restarts": 1,
            "top_configurations": 10,
        }
    )
    plan["sharing_heldout_offset"] = 7200000
    with np.load(parent / "selected_states.npz") as archive:
        mask = archive["indices"] != 0
    np.savez_compressed(output / "learned_mask.npz", mask=mask)
    for name in ("source.npz", "teacher_audit_only.npz"):
        shutil.copy2(parent / name, output / name)
    save_json(output / "protocol.json", plan)
    save_json(output / "specification.json", spec)
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for name in (
        "sharing_geometry.py",
        "sharing_ablation.py",
        "model.py",
        "train.py",
        "run.py",
        "data.py",
    ):
        shutil.copy2(BASE / name, snapshot / name)
    save_json(
        output / "registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "status": "Conditional sharing follow-up after commitment ablations",
            "mask_parent": str(parent / "selected_states.npz"),
            "mask_sha256": digest(output / "learned_mask.npz"),
            "source_sha256": digest(output / "source.npz"),
            "candidates": configurations(),
            "rules": [
                "The common mask was learned by the earlier source-selected DARTS run. No teacher indices or coordinate roles enter the learner.",
                "This is conditional sharing search on that learned support, not a new end-to-end mask search.",
                "All methods have 12000 search steps including optional masked numerical warmup0/1000/3000; one numeric search replica per seed.",
                "Centers initialize from generic empirical quantiles of absolute active latent weights; no k-means or teacher values are used.",
                "Zero and tied sign pairs remain available. Removed edges stay zero; the retained support may be further pruned.",
                "Warmup uses unconstrained masked numerical weights. Its best query checkpoint initializes quantizer and head.",
                "Warm logits initialize from distances at the same weights and centers as geometric variants; random logits are used without warmup.",
                "Distance-soft uses softmax of negative squared normalized distance, with temperature0.25 to0.001.",
                "Nearest-STE uses a genuinely nearest-code weight in forward; identity derivative for latent W plus direct grouped gradient for C. Its gradient is biased.",
                "Geometric variants add 0.01 ramped latent commitment plus a code-centering term. This is an alternative parameterization, not pure categorical DARTS.",
                "Every hard snapshot is evaluated with one inherited and three fresh Gaussian codebooks for3000 steps. No inherited optimizer state is reused.",
                "Choose one numerical rate per family using pilot hard-refit selection; confirm on three new search seeds.",
                "Freeze source choices before teacher recovery and a new held-out input draw. No software tests.",
            ],
            "literature": [
                "https://proceedings.neurips.cc/paper_files/paper/2017/file/86b122d4358357d834a87ce618a55de0-Paper.pdf",
                "https://arxiv.org/abs/1308.3432",
            ],
            "snapshot_sha256": {
                path.name: digest(path) for path in snapshot.glob("*.py")
            },
        },
    )


def search(output, plan, spec, rows, phase, device):
    folder = output / phase
    folder.mkdir(exist_ok=False)
    save_json(folder / "rows.json", rows)
    with np.load(output / plan.get("support_file", "learned_mask.npz")) as archive:
        mask_array = archive["mask"]
    mask = torch.tensor(mask_array, device=device)
    codes, width = spec["codes"], spec["hidden"]
    settings = plan["search"]
    steps = settings[phase + "_steps"]
    latent_arrays, alpha_arrays = [], []
    for row in rows:
        rng = np.random.default_rng(plan["seed"] + 100003 * row["seed"])
        latent = rng.normal(size=mask_array.shape) * mask_array
        latent *= row["radius"] * math.sqrt(width) / np.linalg.norm(latent)
        latent_arrays.append(latent)
        alpha_arrays.append(rng.normal(0, 1, (*mask_array.shape, 2 * codes + 1)))
    latent = torch.tensor(
        np.array(latent_arrays),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    alpha = torch.tensor(
        np.array(alpha_arrays),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    numeric = torch.zeros(len(rows), 1, codes + 3, device=device)
    numeric[..., codes] = 0.1
    numeric[..., codes + 1] = 1.0
    numeric[..., codes + 2] = spec["target_mean"]
    numeric.requires_grad_()
    latent_optimizer = BatchedOptimizer(latent, rows, "latent_lr")
    numeric_optimizer = BatchedOptimizer(numeric, rows)
    alpha_optimizer = BatchedOptimizer(alpha, rows, "architecture_lr")
    warmup = np.array([row["warmup"] for row in rows])
    categorical = torch.tensor(
        [row["method"] == "logits" for row in rows], device=device
    )
    discrete = torch.tensor(
        [row["method"] == "nearest_ste" for row in rows], device=device
    )
    strength = numeric.new_tensor([row["commitment"] for row in rows])
    latent_l1 = numeric.new_tensor([row.get("latent_l1", 0.0) for row in rows])
    has_l1 = any(row.get("latent_l1", 0.0) for row in rows)
    has_exact_assignment = any(
        row.get("exact_forward_assignment", False) for row in rows
    )
    exact_assignment = torch.tensor(
        [row.get("exact_forward_assignment", False) for row in rows], device=device
    )
    levels = (torch.arange(codes, device=device) + 0.5) / codes
    best_warm_loss = torch.full((len(rows),), torch.inf, device=device)
    best_warm_latent = latent.detach().clone()
    best_warm_numeric = numeric.detach().clone()
    data = source_tensors(output, device)
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + 1001)
    checkpoints = {round(value * steps) for value in settings["snapshots"]}
    history, saved = [], []
    started = time.perf_counter()
    for step in range(steps + 1):
        transitioning = np.flatnonzero(warmup == step)
        if len(transitioning):
            ids = torch.tensor(transitioning, device=device)
            with torch.no_grad():
                if step:
                    warm_query = score(
                        data["query_x"],
                        data["query_y"],
                        (latent[ids] * mask)[:, None],
                        numeric[ids],
                        spec,
                    )[:, 0]
                    improved = warm_query < best_warm_loss[ids]
                    improved_ids = ids[improved]
                    best_warm_loss[improved_ids] = warm_query[improved]
                    best_warm_latent[improved_ids] = latent[improved_ids]
                    best_warm_numeric[improved_ids] = numeric[improved_ids]
                    latent[ids] = best_warm_latent[ids]
                    numeric[ids] = best_warm_numeric[ids]
                centers = torch.quantile(latent[ids][:, mask].abs(), levels, dim=1).T
                numeric[ids, 0, :codes] = centers.clamp_min(1e-4)
                values = signed_values(numeric[ids], codes)[:, 0]
                energy = centers.square().mean(-1).clamp_min(1e-8)
                distances = (
                    latent[ids, ..., None] - values[:, None, None, :]
                ).square() / energy[:, None, None, None]
                if step:
                    alpha[ids] = -distances / 0.25
                for optimizer in (latent_optimizer, numeric_optimizer, alpha_optimizer):
                    optimizer.first[ids] = 0
                    optimizer.second[ids] = 0
                    optimizer.velocity[ids] = 0
                    optimizer.steps[ids] = 0
        progress_array = np.maximum(0, (step - warmup) / (steps - warmup))
        progress = numeric.new_tensor(progress_array)
        is_warm = torch.tensor(step < warmup, device=device)
        cooling = ((progress - 0.2) / 0.8).clamp(0, 1)
        temperature = 0.25 * (0.001 / 0.25) ** cooling
        values = signed_values(numeric, codes)[:, 0]
        energy = numeric[..., :codes].detach().square().mean((1, 2)).clamp_min(1e-8)
        distances = (latent[..., None] - values[:, None, None, :]).square() / energy[
            :, None, None, None
        ]
        geometry = (-distances / temperature[:, None, None, None]).softmax(-1)
        logits_temperature = 0.03**cooling
        free = (alpha / logits_temperature[:, None, None, None]).softmax(-1)
        probabilities = torch.where(categorical[:, None, None, None], free, geometry)
        zero = torch.zeros_like(probabilities)
        zero[..., 0] = 1
        probabilities = torch.where(mask[None, ..., None], probabilities, zero)
        soft = torch.einsum("sdhk,sk->sdh", probabilities, values)
        nearest = values.gather(1, distances.argmin(-1).flatten(1)).reshape(
            latent.shape
        )
        nearest = nearest * mask
        straight = latent + (nearest - latent).detach() + nearest - nearest.detach()
        weights = torch.where(discrete[:, None, None], straight, soft) * mask
        weights = torch.where(is_warm[:, None, None], latent * mask, weights)
        detached_values = signed_values(numeric.detach(), codes)[:, 0]
        average = torch.einsum("sdhk,sk->sdh", probabilities, detached_values)
        second = torch.einsum("sdhk,sk->sdh", probabilities, detached_values.square())
        categorical_commit = (second - average.square()).clamp_min(0)[:, mask].mean(
            -1
        ) / energy
        latent_commit = (
            (latent - nearest.detach()).square()[:, mask].mean(-1)
        ) / energy
        center_commit = (
            (latent.detach() - nearest).square()[:, mask].mean(-1)
        ) / energy
        ramp = ((progress - 0.1) / 0.6).clamp(0, 1)
        commitment = (
            torch.where(categorical, categorical_commit, latent_commit + center_commit)
            * strength
            * ramp
            * (~is_warm)
        )
        if step % settings["checkpoint_every"] == 0 or step == steps:
            with torch.no_grad():
                query = score(
                    data["query_x"], data["query_y"], weights[:, None], numeric, spec
                )
                improved = is_warm & (query[:, 0] < best_warm_loss)
                best_warm_loss[improved] = query[improved, 0]
                best_warm_latent[improved] = latent[improved]
                best_warm_numeric[improved] = numeric[improved]
                extraction = probabilities
                if has_exact_assignment:
                    one_hot = F.one_hot(distances.argmin(-1), 2 * codes + 1).to(
                        probabilities
                    )
                    one_hot = torch.where(mask[None, ..., None], one_hot, zero)
                    extraction = torch.where(
                        exact_assignment[:, None, None, None], one_hot, probabilities
                    )
                indices = hard_indices(extraction)
                hard = score(
                    data["query_x"],
                    data["query_y"],
                    fixed_weights(indices.long(), numeric, codes),
                    numeric,
                    spec,
                )
                history.append(
                    {
                        "step": step,
                        "query_mse": query.tolist(),
                        "hard_query_mse": hard.tolist(),
                        "commitment": commitment.tolist(),
                        "seconds": time.perf_counter() - started,
                    }
                )
                if not torch.isfinite(query).all():
                    raise FloatingPointError("Non-finite conditional sharing fit")
                save_json(folder / "history.json", history)
                if step in checkpoints:
                    fresh_numeric = initial_numeric(
                        [{**row, "aligned_initial_codes": False} for row in rows],
                        3,
                        spec,
                        plan["seed"] + 51017,
                        device,
                        indices=indices.long(),
                    )
                    saved.append(
                        {
                            "step": step,
                            "probabilities": extraction.cpu().numpy(),
                            "numeric": torch.cat((numeric, fresh_numeric), 1)
                            .cpu()
                            .numpy(),
                            "search_numeric": numeric.cpu().numpy(),
                            "latent": latent.cpu().numpy(),
                        }
                    )
                    np.savez_compressed(
                        folder / "snapshots.npz",
                        steps=np.array([item["step"] for item in saved]),
                        **{
                            key: np.stack([item[key] for item in saved])
                            for key in (
                                "probabilities",
                                "numeric",
                                "search_numeric",
                                "latent",
                            )
                        },
                    )
                print(
                    f"{phase} step={step} rows={len(rows)} "
                    f"query_NMSE={query.mean().item() / spec['target_variance']:.6g} "
                    f"hard_NMSE={hard.mean().item() / spec['target_variance']:.6g} "
                    f"seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )
        if step == steps:
            break
        ids = torch.randint(
            len(data["support_x"]),
            (settings["batch"],),
            device=device,
            generator=generator,
        )
        predicted = predict(data["support_x"][ids], weights[:, None], numeric, spec)
        loss = (predicted - data["support_y"][ids]).square().mean((1, 2))
        objective = loss / spec["target_variance"] + commitment
        objective = (
            objective
            + settings["numeric_l2"]
            * (numeric.square().sum((1, 2)) + latent.square().sum((1, 2)))
            / 2
        )
        if has_l1:
            objective = objective + latent_l1 * (
                latent.abs().mean((1, 2)) / energy.sqrt()
            )
        gradients = torch.autograd.grad(objective.sum(), (numeric, latent, alpha))
        numeric_optimizer.step(clip_gradient(gradients[0], 10, (1, 2)))
        latent_optimizer.step(
            clip_gradient(gradients[1], 10, (1, 2)), active=is_warm | ~categorical
        )
        alpha_optimizer.step(
            clip_gradient(gradients[2], 5, (1, 2, 3)), active=~is_warm & categorical
        )
        with torch.no_grad():
            latent.mul_(mask)
            alpha.sub_(alpha.mean(-1, keepdim=True))
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
        ids = json.loads((output / "pilot_selection.json").read_text())[
            "top_configurations"
        ]
        candidates = [row for row in candidates if row["configuration"] in ids]
    rows = [
        {**row, "seed": seed}
        for row in candidates
        for seed in plan["search"][phase + "_seeds"]
    ]
    search(output, plan, spec, rows, phase, device)
    refit_search(output, plan, spec, phase, device)
    ranking = json.loads((output / phase / "ranking.json").read_text())
    if phase == "pilot":
        families = {}
        for entry in ranking:
            family = entry["per_seed"][0]["family"]
            families.setdefault(family, entry)
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
        save_json(
            output / "family_selection.json",
            {
                "families": {
                    entry["per_seed"][0]["family"]: entry for entry in ranking
                },
            },
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
