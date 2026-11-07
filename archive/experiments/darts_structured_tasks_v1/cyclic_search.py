"""Repeated frozen-C architecture phases and frozen-P numerical phases."""

import argparse
import itertools
import json
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
    soft_weights,
    source_tensors,
)
from .run import refit_search
from .sharing_ablation import fresh
from .sharing_report import evaluate

FAMILIES = ("soft_P_soft_C", "soft_P_hard_C", "hard_P_hard_C")
P_STEPS = 4
C_STEPS = 64
CYCLES = 200


def configurations():
    return [
        {
            "configuration": i,
            "family": family,
            "mode": "cyclic",
            "numeric_lr": numeric_lr,
            "architecture_lr": architecture_lr,
            "radius": 0.5,
            "sparsity": 0.03,
            "commitment": 0.0,
        }
        for i, (family, numeric_lr, architecture_lr) in enumerate(
            itertools.product(FAMILIES, (0.01, 0.03), (0.01, 0.03, 0.1))
        )
    ]


def prepare(parent, output):
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((parent / "protocol.json").read_text())
    spec = json.loads((parent / "specification.json").read_text())
    spec.update(name="deepsets_cyclic_full", full_connectivity_search=True)
    plan["search"].update(
        pilot_seeds=[241, 242],
        confirmation_seeds=[243, 244, 245],
        pilot_steps=CYCLES,
        confirmation_steps=CYCLES,
        restarts=4,
        numeric_warmup=0,
        top_configurations=3,
    )
    plan["sharing_heldout_offset"] = 7800000
    plan["cyclic"] = {"cycles": CYCLES, "P_steps": P_STEPS, "C_steps": C_STEPS}
    rules = [
        "Full256-position search only; no supplied mask, teacher connections, groups or edge count.",
        "A repeated cycle first freezes C/head and trains P for4 support minibatches, then freezes P and trains C/head for64 support minibatches. No parameter is frozen permanently.",
        "Keep soft logits P between all cycles; recompute argmax after each P phase. On the hard-C branch this MAP I stays fixed only for that numerical block.",
        "Three families: soft P/soft C control, user-proposed soft P/hard C, hard P/hard C. Hard P uses deterministic MAP forward and softmax STE backward.",
        "Four independent numerical starts share P. Average their losses for P; compute independent numerical gradients without averaging across starts.",
        "All families have200 cycles:800 P updates and12800 C updates per root, four numerical replicas; matched initial logits/C and phase minibatches.",
        "P=softmax(alpha), no temperature cooling or sampling. Late entropy0.02 and sparsity0.03 are common. Hard forward uses one code, no cancellation-mass projection alters MAP.",
        "Source selection only: pilot241/242 chooses one configuration per family among six learning-rate pairs; confirmation243/244/245 uses these fixed settings.",
        "Snapshots after0/67/133/200 complete cycles freeze raw MAP, then four inherited numerical starts get3000 C-fit updates with reset optimizer moments. Query selects numerical checkpoints and mean selection chooses snapshots/configurations.",
        "Both phases use support to isolate hard numerical adaptation; independent query/selection monitor generalization. This is block-coordinate training, not an unrolled bilevel gradient.",
        "Codebook optimizer moments persist across cycles and are reset only in final fixed-I fits. Numerical initialization is normalized against the same hard MAP matrix in all families.",
        "After source freeze, fresh-C calibration and32 new numerical initializations, then a new heldout draw. One previously inspected teacher; no transfer claim.",
        "No software tests are added or run.",
    ]
    plan["rules"] = rules
    for name in ("source.npz", "teacher_audit_only.npz"):
        shutil.copy2(parent / name, output / name)
    save_json(output / "protocol.json", plan)
    save_json(output / "specification.json", spec)
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for name in (
        "cyclic_search.py",
        "cyclic_report.py",
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
            "rules": rules,
            "source_sha256": digest(output / "source.npz"),
            "snapshot_sha256": {p.name: digest(p) for p in snapshot.glob("*.py")},
            "references": ["https://arxiv.org/abs/1812.00332"],
        },
    )


def search(output, plan, spec, rows, phase, device):
    folder = output / phase
    folder.mkdir(exist_ok=False)
    save_json(folder / "rows.json", rows)
    settings = plan["search"]
    classes = 2 * spec["codes"] + 1
    alpha = torch.tensor(
        np.array(
            [
                np.random.default_rng(plan["seed"] + 100003 * row["seed"]).normal(
                    0, 1, (spec["inputs"], spec["hidden"], classes)
                )
                for row in rows
            ]
        ),
        device=device,
        dtype=torch.float32,
    )
    alpha[..., 0] += np.log(2 * spec["codes"])
    initial_map = F.one_hot(alpha.argmax(-1), classes).float()
    numeric = initial_numeric(
        rows, 4, spec, plan["seed"] + 901, device, probabilities=initial_map
    ).requires_grad_()
    alpha.requires_grad_()
    np.savez_compressed(
        folder / "initial_state.npz",
        logits=alpha.detach().cpu().numpy(),
        numeric=numeric.detach().cpu().numpy(),
    )
    numeric_optimizer = BatchedOptimizer(numeric, rows)
    alpha_optimizer = BatchedOptimizer(alpha, rows, "architecture_lr")
    hard_P = torch.tensor([r["family"] == "hard_P_hard_C" for r in rows], device=device)
    hard_C = torch.tensor([r["family"] != "soft_P_soft_C" for r in rows], device=device)
    sparsity = numeric.new_tensor([r["sparsity"] for r in rows])
    data = source_tensors(output, device)
    p_generator = torch.Generator(device=device).manual_seed(plan["seed"] + 870001)
    c_generator = torch.Generator(device=device).manual_seed(plan["seed"] + 870002)
    cycles = plan["cyclic"]["cycles"]
    snapshots = {round(f * cycles) for f in settings["snapshots"]}
    history, saved = [], []
    started = time.perf_counter()
    changes = [0] * len(rows)
    maximum_forward_error = 0.0
    variance = spec["target_variance"]
    for cycle in range(cycles + 1):
        policy = alpha.softmax(-1)
        map_assignment = F.one_hot(policy.argmax(-1), classes).float()
        if cycle % 10 == 0 or cycle in snapshots:
            with torch.no_grad():
                relaxed_weights = soft_weights(policy, numeric, spec["codes"])
                hard_weights = soft_weights(map_assignment, numeric, spec["codes"])
                hard_query = score(
                    data["query_x"], data["query_y"], hard_weights, numeric, spec
                )
                soft_query = score(
                    data["query_x"], data["query_y"], relaxed_weights, numeric, spec
                )
                weights = torch.where(
                    hard_C[:, None, None, None], hard_weights, relaxed_weights
                )
                training = score(
                    data["support_x"], data["support_y"], weights, numeric, spec
                )
                query = torch.where(hard_C[:, None], hard_query, soft_query)
                if not torch.isfinite(query).all():
                    raise FloatingPointError("Non-finite cyclic search")
                history.append(
                    {
                        "cycle": cycle,
                        "step": cycle * C_STEPS,
                        "P_updates": cycle * P_STEPS,
                        "C_updates": cycle * C_STEPS,
                        "train_mse": training.tolist(),
                        "query_mse": query.tolist(),
                        "soft_query_mse": soft_query.tolist(),
                        "hard_query_mse": hard_query.tolist(),
                        "entropy": (
                            -(policy * policy.clamp_min(1e-12).log())
                            .sum(-1)
                            .mean((1, 2))
                        ).tolist(),
                        "MAP_edges": (map_assignment[..., 0] == 0).sum((1, 2)).tolist(),
                        "MAP_changes_last_cycle": changes,
                        "seconds": time.perf_counter() - started,
                    }
                )
                save_json(folder / "history.json", history)
                if cycle in snapshots:
                    saved.append(
                        {
                            "step": cycle * C_STEPS,
                            "probabilities": map_assignment.cpu().numpy(),
                            "policy_probabilities": policy.cpu().numpy(),
                            "logits": alpha.cpu().numpy(),
                            "numeric": numeric.cpu().numpy(),
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
                if cycle % 25 == 0 or cycle == cycles:
                    print(
                        f"{phase} cycle={cycle} roots={len(rows)} hard_NMSE={hard_query.mean().item() / variance:.5g} seconds={time.perf_counter() - started:.1f}",
                        flush=True,
                    )
        if cycle == cycles:
            break
        old_classes = alpha.detach().argmax(-1)
        fixed_numeric = numeric.detach()
        # P phase: all numerical parameters are frozen; logits remain soft.
        for _ in range(P_STEPS):
            ids = torch.randint(
                len(data["support_x"]),
                (settings["batch"],),
                generator=p_generator,
                device=device,
            )
            policy = alpha.softmax(-1)
            hard = F.one_hot(policy.argmax(-1), classes).float()
            straight = hard + (policy - policy.detach())
            assignment = torch.where(hard_P[:, None, None, None], straight, policy)
            prediction = predict(
                data["support_x"][ids],
                soft_weights(assignment, fixed_numeric, spec["codes"]),
                fixed_numeric,
                spec,
            )
            losses = (prediction - data["support_y"][ids]).square().mean(
                (1, 2)
            ) / variance
            entropy = -(policy * policy.clamp_min(1e-12).log()).sum(-1).mean((1, 2))
            objective = losses + sparsity * (1 - policy[..., 0]).mean((1, 2))
            objective = (
                objective
                + settings["late_entropy"] * max(0, 2 * cycle / cycles - 1) * entropy
            )
            gradient = torch.autograd.grad(objective.sum(), alpha)[0]
            alpha_optimizer.step(
                clip_gradient(gradient, settings["alpha_clip"], (1, 2, 3))
            )
            with torch.no_grad():
                alpha.sub_(alpha.mean(-1, keepdim=True))
                if hard_P.any():
                    maximum_forward_error = max(
                        maximum_forward_error,
                        float((assignment[hard_P].detach() - hard[hard_P]).abs().max()),
                    )
        # C phase: one current MAP, fixed only until the next P phase.
        with torch.no_grad():
            policy = alpha.softmax(-1)
            hard = F.one_hot(policy.argmax(-1), classes).float()
            fixed_assignment = torch.where(hard_C[:, None, None, None], hard, policy)
            changes = (alpha.argmax(-1) != old_classes).sum((1, 2)).tolist()
        for _ in range(C_STEPS):
            ids = torch.randint(
                len(data["support_x"]),
                (settings["batch"],),
                generator=c_generator,
                device=device,
            )
            prediction = predict(
                data["support_x"][ids],
                soft_weights(fixed_assignment, numeric, spec["codes"]),
                numeric,
                spec,
            )
            losses = (prediction - data["support_y"][ids]).square().mean(-1) / variance
            objective = losses + settings["numeric_l2"] * numeric.square().sum(-1) / 2
            gradient = torch.autograd.grad(objective.sum(), numeric)[0]
            numeric_optimizer.step(
                clip_gradient(gradient, settings["gradient_clip"], (-1,))
            )
    if maximum_forward_error != 0:
        raise RuntimeError("Hard P forward became a mixture")
    save_json(
        folder / "complete.json",
        {
            "seconds": time.perf_counter() - started,
            "roots": len(rows),
            "cycles": cycles,
            "P_updates_per_root": cycles * P_STEPS,
            "C_updates_per_root": cycles * C_STEPS,
            "numeric_restarts": 4,
            "maximum_hard_P_forward_error": maximum_forward_error,
            "gradient_through_C_adaptation": False,
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
    parser.add_argument(
        "--stage",
        choices=("prepare", "pilot", "confirmation", "fresh", "evaluate"),
        required=True,
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
    elif args.stage == "evaluate":
        evaluate(args.root)
    else:
        run_phase(args.root, args.stage, torch.device(args.device))


if __name__ == "__main__":
    main()
