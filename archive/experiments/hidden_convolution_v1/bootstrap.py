"""Refine source solutions on the support of the best inferred source model."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=ROOT / "runs/pilot_v2")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/pilot_v3")
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--alignment", choices=["support", "energy"], default="support")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    copied = [
        "source_data.npz",
        "source_teacher.npz",
        "calibration_data.npz",
        "calibration_teacher.npz",
        "target_data.npz",
        "target_teacher.npz",
        "solver_protocol.json",
        "teacher_protocol.json",
    ]
    for filename in copied:
        shutil.copy2(args.parent / filename, args.out / filename)
    bank = np.load(args.parent / "structure_bank.npz")
    train = bank["train"]
    reference = int(train[np.argmin(bank["quality"][train])])
    support_mask = bank["indices"][reference] != 0
    positions = np.flatnonzero(support_mask.flatten())
    source = np.load(args.parent / "source_models.npz")
    shutil.copy2(
        args.parent / "source_models.npz", args.out / "original_dense_source_models.npz"
    )
    orders = bank["column_orders"]
    if args.alignment == "energy":
        orders = []
        for matrix in source["weights"]:
            energy = matrix**2
            scores = energy.T @ support_mask.astype(float)
            scores /= np.maximum(energy.sum(0)[:, None], 1e-12)
            old, new = linear_sum_assignment(-scores)
            order = np.empty(36, dtype=int)
            order[new] = old
            orders.append(order)
        orders = np.array(orders)
    initial = np.stack(
        [matrix[:, order] for matrix, order in zip(source["weights"], orders)]
    )
    torch.set_num_threads(4)
    device = torch.device("cuda")
    numbers = torch.tensor(
        initial.reshape(-1, 2304)[:, positions], device=device
    ).requires_grad_()
    locations = torch.tensor(positions, device=device)[None].expand(len(numbers), -1)
    observations = np.load(args.out / "source_data.npz")
    spec = json.loads((args.out / "solver_protocol.json").read_text())
    support = spec["source_support"]
    x = torch.tensor(observations["x"], device=device)
    y = torch.tensor(observations["y"][source["task"]], device=device)
    scale = y[:, :support].var(1, unbiased=False).clamp_min(1e-8)
    optimizer = torch.optim.Adam([numbers], lr=0.003)
    torch.manual_seed(283050)
    best = numbers.detach().clone()
    quality = torch.full((len(y),), torch.inf, device=device)
    history = []

    def predict(batch):
        weights = numbers.new_zeros(len(numbers), 2304).scatter(1, locations, numbers)
        return (
            torch.einsum("bd,mdh->mbh", batch, weights.reshape(-1, 64, 36))
            .relu()
            .sum(-1)
            / 6
        )

    for step in range(args.steps + 1):
        if step % 128 == 0:
            with torch.no_grad():
                error = torch.zeros(len(y), device=device)
                for ids in torch.arange(support, len(x), device=device).split(256):
                    error += (predict(x[ids]) - y[:, ids]).square().sum(1)
                error /= (len(x) - support) * scale
                improve = error < quality
                quality[improve] = error[improve]
                best[improve] = numbers[improve]
                history.append({"step": step, "query_nmse": quality.cpu().tolist()})
            if step % 1024 == 0:
                print("BOOTSTRAP", step, float(quality.mean()), flush=True)
        if step == args.steps:
            break
        if step == args.steps // 2:
            optimizer.param_groups[0]["lr"] = 0.001
        ids = torch.randint(support, (256,), device=device)
        loss = ((predict(x[ids]) - y[:, ids]).square().mean(1) / scale).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    best_weights = torch.zeros(len(y), 2304, device=device).scatter(1, locations, best)
    np.savez_compressed(
        args.out / "source_models.npz",
        weights=best_weights.reshape(-1, 64, 36).cpu().numpy(),
        quality=quality.cpu().numpy(),
        task=source["task"],
        penalties=np.zeros(len(y)),
    )
    np.savez_compressed(
        args.out / "bootstrapped_support.npz",
        mask=support_mask,
        reference=np.array(reference),
        positions=positions,
    )
    audit = {
        "parent": str(args.parent),
        "reference_source_network": reference,
        "reference_source_task": int(bank["task"][reference]),
        "reference_query_nmse": float(bank["quality"][reference]),
        "numeric_parameters_per_source_model": len(positions),
        "source_refinement_steps": args.steps,
        "column_alignment": args.alignment,
        "teacher_layouts_read": False,
        "layout_files_copied_opaque_for_later_audit": True,
        "inputs": [
            "source inferred indexators",
            "source-query quality",
            "fitted weights",
            "source X/y",
        ],
        "copied_data_hashes": {
            name: hashlib.sha256((args.out / name).read_bytes()).hexdigest()
            for name in copied
            if name.endswith("_data.npz")
        },
    }
    (args.out / "bootstrap_complete.json").write_text(json.dumps(audit, indent=2))
    (args.out / "source_complete.json").write_text(
        json.dumps(
            {
                "steps": args.steps,
                "starts": 4,
                "tasks": 32,
                "read_files": [
                    "parent source bank",
                    "source_data.npz",
                    "solver_protocol.json",
                ],
                "best_source_query_mean": float(
                    quality.reshape(32, 4).min(1).values.mean()
                ),
                "support_is_learned_from_source": True,
            },
            indent=2,
        )
    )
    (args.out / "source_history.json").write_text(json.dumps(history))
    shutil.copy2(__file__, args.out / "source_snapshot.py")


if __name__ == "__main__":
    main()
