"""Extract I1 candidates from models trained on a fixed observed dataset."""

import argparse
import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .blind_contract import digest, load_sizes, save_json
from .blind_relu import scale_quantizations

BASE = Path(__file__).resolve().parent


def values(x, parameters, dimensions, hidden, outputs, *, jacobian=False):
    count = dimensions * hidden
    weight = parameters[:, :count].reshape(-1, dimensions, hidden)
    bias = parameters[:, count : count + hidden]
    head = parameters[:, count + hidden : -outputs].reshape(-1, hidden, outputs)
    preactivation = torch.einsum("bnd,bdh->bnh", x, weight) + bias[:, None]
    activation = preactivation.relu() / math.sqrt(hidden)
    fitted = (
        torch.einsum("bnh,bho->bno", activation, head) + parameters[:, None, -outputs:]
    )
    if not jacobian:
        return fitted
    gate = (preactivation > 0).to(x.dtype) / math.sqrt(hidden)
    weight_jac = torch.einsum("bnd,bnh,bho->bnodh", x, gate, head).flatten(-2)
    bias_jac = torch.einsum("bnh,bho->bnoh", gate, head)
    eye = torch.eye(outputs, device=x.device, dtype=x.dtype)
    head_jac = torch.einsum("bnh,oq->bnohq", activation, eye).flatten(-2)
    offset_jac = eye[None, None].expand(len(x), x.shape[1], -1, -1)
    return fitted, torch.cat(
        [weight_jac, bias_jac, head_jac, offset_jac], dim=-1
    ).flatten(1, 2)


@torch.no_grad()
def polish(initial, x, y, vx, vy, dimensions, hidden, outputs, device, steps):
    parameters = torch.tensor(initial, device=device, dtype=torch.float64)
    tx, target, tvx, validation = [
        torch.tensor(a, device=device, dtype=torch.float64) for a in (x, y, vx, vy)
    ]
    identity = torch.eye(parameters.shape[1], device=device, dtype=torch.float64)
    best, selected = (
        parameters.new_full((len(parameters),), torch.inf),
        parameters.clone(),
    )
    training, validation_history = [], []
    for step in range(steps + 1):
        fitted, jac = values(tx, parameters, dimensions, hidden, outputs, jacobian=True)
        residual = (target - fitted).flatten(1)
        train = residual.square().mean(1)
        score = (
            (values(tvx, parameters, dimensions, hidden, outputs) - validation)
            .square()
            .mean((1, 2))
        )
        better = score < best
        best[better], selected[better] = score[better], parameters[better]
        training.append(train.cpu().numpy())
        validation_history.append(score.cpu().numpy())
        if step == steps:
            break
        gram = jac.transpose(1, 2) @ jac / residual.shape[1]
        cross = jac.transpose(1, 2) @ residual[..., None] / residual.shape[1]
        scale = gram.diagonal(dim1=1, dim2=2).amax(1).clamp_min(1e-12)
        direction = torch.linalg.solve(
            gram + 1e-6 * scale[:, None, None] * identity, cross
        )[..., 0]
        accepted = train < 1e-24
        updated = parameters.clone()
        for exponent in range(12):
            proposal = parameters + 2.0 ** (-exponent) * direction
            loss = (
                (values(tx, proposal, dimensions, hidden, outputs) - target)
                .square()
                .mean((1, 2))
            )
            take = (~accepted) & torch.isfinite(proposal).all(1) & (loss < train)
            updated[take] = proposal[take]
            accepted |= take
            if accepted.all():
                break
        parameters = updated
    return (
        selected.cpu().numpy(),
        best.cpu().numpy(),
        np.stack(training),
        np.stack(validation_history),
    )


@torch.no_grad()
def uncertainty(x, y, parameters, dimensions, hidden, outputs, device):
    tx = torch.tensor(x[None], device=device, dtype=torch.float64)
    p = torch.tensor(parameters[None], device=device, dtype=torch.float64)
    prediction, jac = values(tx, p, dimensions, hidden, outputs, jacobian=True)
    residual = torch.tensor(y, device=device) - prediction[0]
    gram = jac[0].T @ jac[0]
    eigenvalues, vectors = torch.linalg.eigh(gram)
    keep = eigenvalues > eigenvalues[-1] * 1e-10
    rank = int(keep.sum())
    variance = float(residual.square().sum() / max(residual.numel() - rank, 1))
    inverse = (vectors[:, keep] / eigenvalues[keep]) @ vectors[:, keep].T
    covariance = (inverse * variance).cpu().numpy()
    weight = parameters[: dimensions * hidden].reshape(dimensions, hidden)
    norm = np.linalg.norm(weight, axis=0)
    normal = weight / np.maximum(norm, 1e-12)
    deviations = []
    for h in range(hidden):
        selected = np.arange(dimensions) * hidden + h
        jacobian = (np.eye(dimensions) - np.outer(normal[:, h], normal[:, h])) / max(
            norm[h], 1e-12
        )
        directional = jacobian @ covariance[np.ix_(selected, selected)] @ jacobian.T
        deviations.append(math.sqrt(max(float(np.max(directional.diagonal())), 0)))
    return np.array(deviations), variance, rank


def run(root, case, condition, device, steps, from_repair=False, polish_only=False):
    torch.set_num_threads(2)
    started = time.monotonic()
    sizes = load_sizes(root, case)
    dimensions, hidden, outputs, codes = [
        sizes[k] for k in ("inputs", "hidden", "outputs", "codes")
    ]
    dense_folder = root / "dense" / f"{case}_c{condition}"
    dense = dict(np.load(dense_folder / "fitted.npz"))
    fixed = dict(np.load(root / "public" / f"{case}_c{condition}" / "fixed_train.npz"))
    validation = dict(
        np.load(root / "public" / f"{case}_c{condition}" / "selection.npz")
    )
    tasks, keep = dense["all_parameters"].shape[0], 4
    if from_repair:
        keep = 1
        initial = np.load(dense_folder / "repaired.npz")["parameters"]
    else:
        chosen = dense["all_validation_mse"].argsort(axis=1)[:, :keep]
        initial = dense["all_parameters"][np.arange(tasks)[:, None], chosen].reshape(
            tasks * keep, -1
        )
    x, y = [
        np.repeat(
            np.stack([fixed[f"{name}_{task}"] for task in range(tasks)]), keep, axis=0
        )
        for name in ("x", "y")
    ]
    vx, vy = [
        np.repeat(
            np.stack([validation[f"{name}_{task}"] for task in range(tasks)]),
            keep,
            axis=0,
        )
        for name in ("x", "y")
    ]
    parameters, scores, train_history, validation_history = polish(
        initial, x, y, vx, vy, dimensions, hidden, outputs, device, steps
    )
    label = "extracted_after_repair" if from_repair else "polished"
    np.savez_compressed(
        dense_folder / f"{label}.npz",
        parameters=parameters.reshape(tasks, keep, -1),
        validation_mse=scores.reshape(tasks, keep),
        train_history=train_history,
        validation_history=validation_history,
    )
    if polish_only:
        save_json(
            dense_folder / f"{label}.json",
            {
                "frozen_utc": datetime.now(UTC).isoformat(),
                "seconds": time.monotonic() - started,
                "steps": steps,
                "teacher_access": False,
                "sha256": digest(dense_folder / f"{label}.npz"),
            },
        )
        return
    for task in range(tasks):
        model = task * keep + int(scores.reshape(tasks, keep)[task].argmin())
        parameter = parameters[model]
        weight = parameter[: dimensions * hidden].reshape(dimensions, hidden)
        norm = np.maximum(np.linalg.norm(weight, axis=0), 1e-12)
        normal = weight / norm
        bias = parameter[dimensions * hidden : dimensions * hidden + hidden] / norm
        deviations, variance, rank = uncertainty(
            x[model], y[model], parameter, dimensions, hidden, outputs, device
        )
        candidates = []
        for multiplier in (3, 5):
            proposed = scale_quantizations(
                normal,
                deviations,
                codes,
                33510007 + task * 1009,
                starts=24,
                zero_multiplier=multiplier,
            )
            for candidate in proposed:
                candidate["zero_multiplier"] = multiplier
            candidates.extend(proposed)
        candidates.sort(key=lambda row: row["projection_mse"])
        arrays = {
            "normal": normal,
            "bias": bias,
            "uncertainty": deviations,
            "dense_parameters": parameter,
        }
        for number, candidate in enumerate(candidates):
            for name in ("index", "codes", "scales"):
                arrays[f"{name}_{number}"] = candidate[name]
        folder = root / "source" / f"{case}_c{condition}_t{task}"
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "executed_code_snapshot").mkdir()
        for name in ("blind_dense_extract.py", "blind_relu.py"):
            shutil.copy2(BASE / name, folder / "executed_code_snapshot" / name)
        np.savez_compressed(folder / "learned.npz", **arrays)
        save_json(
            folder / "trace.json",
            {
                "complete": bool(candidates),
                "mode": "fixed_examples_dense_initialization",
                "noise": {
                    "variance_estimate": [variance] * outputs,
                    "estimator": "own_model_residual_with_empirical_jacobian_rank_correction",
                    "signal_variance_estimate": float(np.var(fixed[f"y_{task}"])),
                },
                "candidates": [
                    {
                        "number": number,
                        "projection_mse": c["projection_mse"],
                        "zero_multiplier": c["zero_multiplier"],
                    }
                    for number, c in enumerate(candidates)
                ],
                "jacobian_rank": rank,
                "dense_validation_mse": float(scores[model]),
                "public_sizes": sizes,
                "additional_output_queries": 0,
            },
        )
        save_json(
            folder / "frozen.json",
            {
                "frozen_utc": datetime.now(UTC).isoformat(),
                "sha256": {
                    name: digest(folder / name)
                    for name in ("learned.npz", "trace.json")
                },
            },
        )
    save_json(
        dense_folder / f"{label}.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "seconds": time.monotonic() - started,
            "steps": steps,
            "teacher_access": False,
            "sha256": digest(dense_folder / f"{label}.npz"),
        },
    )
    print(
        json.dumps(
            {
                "extracted": case,
                "condition": condition,
                "seconds": time.monotonic() - started,
                "best_dense_validation_mse": scores.reshape(tasks, keep)
                .min(axis=1)
                .tolist(),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--from-repair", action="store_true")
    parser.add_argument("--polish-only", action="store_true")
    args = parser.parse_args()
    run(
        args.root.resolve(),
        args.case,
        args.condition,
        torch.device(args.device),
        args.steps,
        args.from_repair,
        args.polish_only,
    )


if __name__ == "__main__":
    main()
