"""Compare retained latent weights with cyclic codebook fitting and Z resets."""

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
    fixed_weights,
    initial_numeric,
    predict,
    score,
    source_tensors,
)
from .run import refit_search
from .sharing_ablation import fresh


def configurations(long_cycles=False):
    rows = []
    period = 1000 if long_cycles else 50
    splits = (800, 500, 200) if long_cycles else (40, 25, 10)

    def add(family, consensus, z_steps, copy_z, clear_moments, tau=None, cycle=None):
        cycle = period if cycle is None else cycle
        rows.append(
            {
                "family": family,
                "use_consensus": consensus,
                "z_steps": z_steps,
                "c_steps": cycle - z_steps,
                "cycle_updates": cycle,
                "copy_z": copy_z,
                "clear_z_moments": clear_moments,
                "projection_tau": tau,
                "dual_step": 0.25 if consensus else 0.0,
                "rho_final": 100.0,
                "nonzero_penalty": 0.3,
                "numeric_lr": 0.01,
                "latent_lr": 0.01,
                "codebook_lr": 0.01,
                "radius": 0.5,
                "mode": "joint",
            }
        )

    add("consensus_baseline", True, 50, False, False, cycle=50)
    if not long_cycles:
        for clear in (False, True):
            add("consensus_reset", True, 50, True, clear)
    for z_steps in splits:
        add("consensus_adapt_keep", True, z_steps, False, False)
        for clear in (False, True):
            add("consensus_adapt_reset", True, z_steps, True, clear)
    for z_steps, tau in itertools.product(splits, (0.0006, 0.006, 0.06)):
        add("plain_adapt_keep", False, z_steps, False, False, tau)
        for clear in (False, True):
            add("plain_adapt_reset", False, z_steps, True, clear, tau)
    return [{**row, "configuration": i} for i, row in enumerate(rows)]


def prepare(parent, output, frozen_method, long_cycles=False):
    method = json.loads((frozen_method / "method.json").read_text())
    for name, expected in method["origin_sha256"].items():
        if digest(parent / name) != expected:
            raise RuntimeError(f"Frozen baseline changed before preparation: {name}")
    for name, expected in method["copied_sha256"].items():
        if digest(frozen_method / name) != expected:
            raise RuntimeError(f"Frozen method copy changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((parent / "protocol.json").read_text())
    spec = json.loads((parent / "specification.json").read_text())
    spec.update(name=output.name, full_connectivity_search=True)
    plan["search"].update(
        pilot_seeds=[341, 342],
        confirmation_seeds=[343, 344, 345],
        pilot_steps=12000,
        confirmation_steps=12000,
        restarts=1,
        numeric_warmup=2000,
        top_configurations=6,
    )
    plan["sharing_heldout_offset"] = 14100000
    plan["reset"] = {
        "cycle_updates": 1000 if long_cycles else 50,
        "long_cycles": long_cycles,
        "rho_initial": 0.01,
        "rho_schedule_interval": 50,
    }
    plan["rules"] = [
        "The prior successful method is archived separately with actual executed code, protocol, selected states and origin hashes; this search does not initialize from its learned indexators or weights.",
        "Search every position of a16x16 matrix from random Z, with four free signed codes and a fixed zero. No teacher mask, groups, code values, point roles or active-edge count enter the learner.",
        "All methods share2000 dense warmup updates and200 cycles of50 task-gradient updates:12000 total. Heads update on every step. Z/C update counts differ by the registered split and are reported explicitly; FLOPs and wall time need not match.",
        "Frozen baseline: retain Z; exact projection every50 Z updates, rho.01 to100, lambda.3, dual.25, with no task-C fit inside the loop. Its hyperparameters are not retuned.",
        "A cycle first trains Z and heads on support, then projects to I/C, then trains only C and heads on support at fixed I, and optionally copies the fitted C[I] into Z. C fits have10/25/40 steps, with40/25/10 Z steps per cycle.",
        "Compare baseline, consensus plus projection-only reset, consensus plus task-C fit with/without reset, and plain task-Z/task-C alternation with/without reset. Plain alternation has no dual or consensus penalty; its projection pays constant tau.0006/.006/.06 per active entry.",
        "Projection globally solves the signed scalar clustering problem with a free zero-prefix. All positions can be reassigned on every cycle; no mask is permanently frozen during search.",
        "With consensus retain and rescale U as rho changes, then update U using Z minus the adapted C[I] BEFORE any Z copy. The task-C phase itself minimizes task loss, not the ADMM subproblem.",
        "A copied Z either keeps its Adam state or clears only Z moments/timestep, as registered. Heads keep their Adam state; C moments clear whenever the projection recomputes the codebook.",
        "At the warmup transition restore the best query warmup state and initialize a projection, but do not reset Z. Subsequent resets occur at completed cycle boundaries, after the C phase when present.",
        "Actual reset error, reset counts, distinct numerical columns, code usage, I changes, train/query/hard losses and Z/C/head update counts are recorded. Query does not supply training gradients or choose an inner-C checkpoint.",
        "Each snapshot0/4000/8000/12000 gets3000 hard-C-fit steps from1inherited+3freshGaussian starts. Query selects numerical checkpoints, mean selection chooses the snapshot/config. Pilot341/342 selects one setting per family; confirmation343/344/345 uses frozen settings.",
        "Teacher audit and a new source heldout follow source/fresh freeze. Fresh32 on the original function is an optimizer diagnostic. Structural transfer uses separately registered new-coefficient functions, fixed source-selected I and all fresh numerical parameters.",
        "No software tests, no mutation of old result folders or frozen baseline, no commit/push.",
    ]
    if long_cycles:
        plan["rules"][2] = (
            "All methods share2000 dense warmup and10000 further task-gradient updates. "
            "The unchanged baseline has200cycles of50Z updates; adapted methods have10cycles "
            "of1000updates with800/200,500/500 or200/800Z/C splits. Heads update every step. "
            "Total Z/C counts match the corresponding short split; block lengths and projection/reset cadence differ."
        )
        plan["rules"][4] = (
            "A long cycle trains Z and heads, projects to I/C, then trains C and heads for200/500/800support updates, "
            "and optionally copies fitted C[I] into Z. This checks inner-C training duration at fixed total budget."
        )
        plan["rules"][5] = (
            "Compare the unchanged baseline and four adapted families: consensus/plain, with/without reset. "
            "Projection-only reset was already checked in the short experiment. Plain uses tau.0006/.006/.06."
        )
        plan["rules"].append(
            "Long-block extension registered after observing the short-block source confirmation, "
            "before viewing either experiment's new transfer functions. Same341-345seeds give paired initializations, "
            "not an independent new task cohort. Rho advances/rescales U every50globalupdates, "
            "including during C blocks; dual increments only at completed per-row cycles."
        )
    for name in ("source.npz", "teacher_audit_only.npz"):
        shutil.copy2(parent / name, output / name)
    save_json(output / "protocol.json", plan)
    save_json(output / "specification.json", spec)
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for name in (
        "reset_search.py",
        "reset_report.py",
        "codebook_projection.py",
        "model.py",
        "train.py",
        "data.py",
        "run.py",
        "sharing_ablation.py",
        "sharing_report.py",
        "evaluate.py",
        "report.py",
        "structure_transfer.py",
        "transfer_report.py",
    ):
        shutil.copy2(BASE / name, snapshot / name)
    save_json(
        output / "registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "candidates": configurations(long_cycles),
            "rules": plan["rules"],
            "source_sha256": digest(output / "source.npz"),
            "frozen_method_path": str(frozen_method.resolve()),
            "frozen_method_sha256": digest(frozen_method / "method.json"),
            "frozen_bundle_sha256": {
                str(path.relative_to(frozen_method)): digest(path)
                for path in frozen_method.rglob("*")
                if path.is_file()
            },
            "snapshot_sha256": {
                path.name: digest(path) for path in snapshot.glob("*.py")
            },
        },
    )


def clear_moments(optimizer, indices):
    optimizer.first[indices] = 0
    optimizer.second[indices] = 0
    optimizer.velocity[indices] = 0
    optimizer.steps[indices] = 0


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
    centers = torch.zeros(len(rows), 1, codes, device=device, requires_grad=True)
    head_optimizer = BatchedOptimizer(heads, rows)
    latent_optimizer = BatchedOptimizer(latent, rows, "latent_lr")
    code_optimizer = BatchedOptimizer(centers, rows, "codebook_lr")
    data = source_tensors(output, device)
    dual = torch.zeros_like(latent)
    labels = torch.zeros_like(latent, dtype=torch.long)
    consensus = torch.tensor([row["use_consensus"] for row in rows], device=device)
    copy_z = torch.tensor([row["copy_z"] for row in rows], device=device)
    clear_z = torch.tensor([row["clear_z_moments"] for row in rows], device=device)
    z_steps = torch.tensor([row["z_steps"] for row in rows], device=device)
    charge = latent.new_tensor([row["nonzero_penalty"] for row in rows])
    tau = latent.new_tensor([row["projection_tau"] or 0.0 for row in rows])
    dual_step = latent.new_tensor([row["dual_step"] for row in rows])
    rho_final = latent.new_tensor([row["rho_final"] for row in rows])
    rho_initial = plan["reset"]["rho_initial"]
    cycle_steps = torch.tensor(
        [row.get("cycle_updates", plan["reset"]["cycle_updates"]) for row in rows],
        device=device,
    )
    rho_interval = plan["reset"].get("rho_schedule_interval", 50)
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + 1001)
    warmup = settings["numeric_warmup"]
    steps = settings[phase + "_steps"]
    checkpoints = {round(fraction * steps) for fraction in settings["snapshots"]}
    best_warm = latent.new_full((len(rows),), torch.inf)
    best_latent, best_heads = latent.detach().clone(), heads.detach().clone()
    previous_rho = latent.new_full((len(rows),), rho_initial)
    reset_counts = torch.zeros(len(rows), device=device, dtype=torch.long)
    projection_counts = torch.zeros_like(reset_counts)
    index_changes = torch.zeros_like(reset_counts)
    z_updates, c_updates = (
        torch.zeros_like(reset_counts),
        torch.zeros_like(reset_counts),
    )
    history, saved = [], []
    maximum_reset_error = 0.0
    started = time.perf_counter()
    all_rows = torch.arange(len(rows), device=device)
    for step in range(steps + 1):
        if step == warmup:
            with torch.no_grad():
                latent.copy_(best_latent)
                heads.copy_(best_heads)
                clear_moments(head_optimizer, all_rows)
                clear_moments(latent_optimizer, all_rows)
        offset = max(0, step - warmup)
        position = offset % cycle_steps
        cycle_end = (position == 0) & (step > warmup)
        progress = (offset // rho_interval * rho_interval) / (steps - warmup)
        rho = rho_initial * (rho_final / rho_initial).pow(progress)
        with torch.no_grad():
            if step >= warmup:
                if offset % rho_interval == 0:
                    dual *= (previous_rho / rho)[:, None, None]
                    previous_rho = rho
                if step == warmup:
                    projecting = torch.ones_like(consensus)
                else:
                    projecting = (z_steps == position) | (
                        cycle_end & (z_steps == cycle_steps)
                    )
                ids = projecting.nonzero().flatten()
                if len(ids):
                    projection_charge = torch.where(
                        consensus[ids], charge[ids], tau[ids] / 2
                    )
                    projection_rho = torch.where(
                        consensus[ids], rho[ids], torch.ones_like(rho[ids])
                    )
                    chosen, values, _ = exact_signed_projection(
                        latent[ids] + dual[ids],
                        codes,
                        projection_charge,
                        projection_rho,
                    )
                    index_changes[ids] += (labels[ids] != chosen).sum((1, 2))
                    projection_counts[ids] += 1
                    labels[ids] = chosen
                    centers[ids, 0] = values
                    clear_moments(code_optimizer, ids)
            numeric = torch.cat((centers, heads), -1)
            indices = torch.where(labels > codes, codes - labels, labels)
            hard = fixed_weights(indices, numeric, codes)
            updating = torch.ones_like(consensus) if step == warmup else cycle_end
            dual += (
                dual_step[:, None, None]
                * (latent - hard[:, 0])
                * updating[:, None, None]
            )
            log_now = step % settings["checkpoint_every"] == 0 or step == steps
            if log_now:
                before_query = score(
                    data["query_x"], data["query_y"], latent[:, None], numeric, spec
                )
                before_distance = (latent - hard[:, 0]).square().mean((1, 2))
            if cycle_end.any():
                ids = (copy_z & cycle_end).nonzero().flatten()
                if len(ids):
                    latent[ids] = hard[ids, 0]
                    reset_counts[ids] += 1
                    error = float((latent[ids] - hard[ids, 0]).abs().max())
                    maximum_reset_error = max(maximum_reset_error, error)
                    if error != 0:
                        raise RuntimeError("Z reset differs from the fitted C[I]")
                clear_moments(
                    latent_optimizer, (copy_z & clear_z & cycle_end).nonzero().flatten()
                )
        z_active = torch.ones_like(consensus) if step < warmup else position < z_steps
        numeric = torch.cat((centers, heads), -1)
        hard = fixed_weights(indices, numeric, codes)
        weights = torch.where(z_active[:, None, None, None], latent[:, None], hard)
        if log_now:
            with torch.no_grad():
                query = score(data["query_x"], data["query_y"], weights, numeric, spec)
                training = score(
                    data["support_x"], data["support_y"], weights, numeric, spec
                )
                hard_query = score(
                    data["query_x"], data["query_y"], hard, numeric, spec
                )
                hard_training = score(
                    data["support_x"], data["support_y"], hard, numeric, spec
                )
                if step < warmup:
                    better = query[:, 0] < best_warm
                    best_warm[better] = query[better, 0]
                    best_latent[better] = latent[better]
                    best_heads[better] = heads[better]
                if not torch.isfinite(query).all():
                    raise FloatingPointError("Non-finite cyclic reset fit")
                states = latent.detach().cpu().numpy()
                spread = (latent - latent.mean(-1, keepdim=True)).square().sum(
                    (1, 2)
                ) / latent.square().sum((1, 2)).clamp_min(1e-12)
                history.append(
                    {
                        "step": step,
                        "query_mse": query.tolist(),
                        "train_mse": training.tolist(),
                        "hard_query_mse": hard_query.tolist(),
                        "hard_train_mse": hard_training.tolist(),
                        "pre_reset_query_mse": before_query.tolist(),
                        "pre_reset_distance": before_distance.tolist(),
                        "active_edges": (labels != 0).sum((1, 2)).tolist(),
                        "groups_used": [
                            len(torch.unique(row[row != 0].abs())) for row in indices
                        ],
                        "distinct_columns": [
                            len(np.unique(row.T, axis=0)) for row in states
                        ],
                        "column_spread": spread.tolist(),
                        "rho": rho.tolist(),
                        "reset_counts": reset_counts.tolist(),
                        "projection_counts": projection_counts.tolist(),
                        "index_changes": index_changes.tolist(),
                        "codebook": centers[:, 0].tolist(),
                        "z_updates": z_updates.tolist(),
                        "c_updates": c_updates.tolist(),
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
                            "probabilities": F.one_hot(labels, classes)
                            .float()
                            .detach()
                            .cpu()
                            .numpy(),
                            "numeric": torch.cat((numeric, fresh_numeric), 1)
                            .detach()
                            .cpu()
                            .numpy(),
                            "latent": latent.detach().cpu().numpy(),
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
                        f"{phase} step={step} roots={len(rows)} current_query_NMSE={query.mean().item() / spec['target_variance']:.5g} hard_query_NMSE={hard_query.mean().item() / spec['target_variance']:.5g} seconds={time.perf_counter() - started:.1f}",
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
        prediction = predict(data["support_x"][ids], weights, numeric, spec)
        task_loss = (prediction - data["support_y"][ids]).square().mean((1, 2)) / spec[
            "target_variance"
        ]
        penalty = (
            (latent - hard[:, 0].detach() + dual).square().mean((1, 2))
            * rho
            / 2
            * consensus
            * z_active
            if step >= warmup
            else task_loss * 0
        )
        size = torch.where(
            z_active, latent.square().sum((1, 2)), centers.square().sum((1, 2))
        )
        regularizer = settings["numeric_l2"] * (size + heads.square().sum((1, 2))) / 2
        gradients = torch.autograd.grad(
            (task_loss + penalty + regularizer).sum(), (heads, latent, centers)
        )
        head_optimizer.step(
            clip_gradient(gradients[0], settings["gradient_clip"], (1, 2))
        )
        latent_optimizer.step(
            clip_gradient(gradients[1], settings["gradient_clip"], (1, 2)),
            active=z_active,
        )
        code_optimizer.step(
            clip_gradient(gradients[2], settings["gradient_clip"], (1, 2)),
            active=~z_active,
        )
        z_updates += z_active.long()
        c_updates += (~z_active).long()
    save_json(
        folder / "complete.json",
        {
            "seconds": time.perf_counter() - started,
            "steps": steps,
            "roots": len(rows),
            "numeric_restarts": 1,
            "maximum_reset_error": maximum_reset_error,
            "actual_z_updates": z_updates.tolist(),
            "actual_c_updates": c_updates.tolist(),
            "actual_head_updates": [steps] * len(rows),
            "actual_resets": reset_counts.tolist(),
            "actual_projections": projection_counts.tolist(),
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
    parser.add_argument("--frozen-method", type=Path)
    parser.add_argument(
        "--stage", choices=("prepare", "pilot", "confirmation", "fresh"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--long-cycles", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        if args.parent is None or args.frozen_method is None:
            parser.error("--parent and --frozen-method are required for preparation")
        prepare(args.parent, args.root, args.frozen_method, args.long_cycles)
    elif args.stage == "fresh":
        fresh(args.root, torch.device(args.device))
    else:
        run_phase(args.root, args.stage, torch.device(args.device))


if __name__ == "__main__":
    main()
