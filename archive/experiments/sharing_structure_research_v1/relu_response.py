"""G15: observed ReLU versus fixed-gate quadratic single-step response.

No new teacher queries, new optimizer trajectory, or confidence claim. G9
observations and starting parameters are regenerated/verified against hashes.
"""

import argparse
import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .curvature_sharing import forward, information, observations
from .nonlinear_moments import make_teacher

BASE = Path(__file__).resolve().parent


def register(root, parent):
    original = json.loads((parent / "protocol.json").read_text())
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "stage": "G15 observed single-step response on opened frozen G9; diagnosis only",
        "registered_utc": datetime.now(UTC).isoformat(),
        "parent": str(parent.resolve()),
        "parent_protocol_sha256": digest(parent / "protocol.json"),
        "jobs": [
            {"family": family, "initialization": initialization, "noise": 0.2}
            for family in original["families"]
            for initialization in original["initializations"]
        ],
        "relative_ridge": [0.0, 1e-6, 1e-4, 1e-2],
        "fractions": [2.0 ** (-k) for k in range(17)],
        "selection": "No new model/HPO selection; all directions/fractions retained. Best actual source decrease is a diagnostic, with its validation change reported separately. No teacher-I/C chooses a step.",
        "scope": "Fixed-gate quadratic identity is exact after its measured remainder. GN decrement is a bound to local linear optimum only; no automatic nonlinear parameter/CI guarantee. G9 original gate/results unchanged.",
        "data": "Exactly previous G9 source/readout/validation via observation hashes. New labels or teacher gradients never used. Make_teacher is used only to regenerate the fixed observations.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": {
            name: digest(BASE / name)
            for name in (
                "relu_response.py",
                "curvature_sharing.py",
                "nonlinear_moments.py",
            )
        },
        "checks": "No software tests. Necessary provenance/quadratic-identity scientific guards and lint only.",
    }
    save_json(root / "protocol.json", plan)
    shutil.copy2(BASE / "THEORY.md", root / "theory_at_registration.md")
    snapshot = root / "executed_code_snapshot"
    snapshot.mkdir()
    for name in plan["code_sha256"]:
        shutil.copy2(BASE / name, snapshot / name)
    (root / "models").mkdir()


@torch.no_grad()
def run_case(root, plan, job_id, device):
    started = time.monotonic()
    job = plan["jobs"][job_id]
    parent = Path(plan["parent"])
    original = json.loads((parent / "protocol.json").read_text())
    folder = parent / f"family_{job['family']}"
    frozen = json.loads((folder / "learner_frozen.json").read_text())
    for name, expected in frozen["sha256"].items():
        if digest(folder / name) != expected:
            raise RuntimeError("Frozen G9 source changed")
    metadata = json.loads((folder / "learner_records.json").read_text())
    teacher, _, _ = make_teacher(job["family"], original["function"], original, device)
    data, fingerprint = observations(original, job["family"], teacher, device)
    if fingerprint != metadata["observation_sha256"]:
        raise RuntimeError("G9 observations differ; response is not paired")
    key = f"{job['initialization']}__noise{job['noise']}"
    with np.load(folder / "learner_outputs.npz") as archive:
        theta = torch.tensor(archive[key + "__theta"], device=device)
    x, ys = data["source"]
    y = ys[:, original["noise"].index(job["noise"])]
    inputs, hidden = original["inputs"], original["hidden"]
    entries = inputs * hidden
    floor = (
        y.square().mean().clamp_min(1) * original["rounding_variance_floor_relative"]
    )
    hessian, score, mse, variance = information(x, y, theta, inputs, hidden, floor)
    eigenvalues = torch.linalg.eigvalsh(hessian)
    if float(eigenvalues.min()) <= 0:
        raise FloatingPointError("Full-rank response analysis requires SPD H")
    prediction = forward(x, theta, inputs, hidden)
    residual = y - prediction
    old_gates = x @ theta[:entries].reshape(inputs, hidden) + theta[-2] > 0
    identity = torch.eye(len(theta), device=device, dtype=torch.float64)
    vx, vy = data["validation"]
    vy = vy[:, original["noise"].index(job["noise"])]
    validation_mse = (forward(vx, theta, inputs, hidden) - vy).square().mean()
    scale = len(x) / variance
    output = []
    best_source = float(mse)
    selected_theta = theta.clone()
    maximum_identity_error = 0.0
    for relative_ridge in plan["relative_ridge"]:
        direction = torch.linalg.solve(
            hessian + relative_ridge * eigenvalues[-1] * identity, score
        )
        directional = x @ direction[:entries].reshape(inputs, hidden) + direction[-2]
        linear_response = (directional * old_gates).sum(-1) / math.sqrt(hidden)
        linear_response += direction[-1]
        gd = score @ direction
        curvature = direction @ hessian @ direction
        for fraction in plan["fractions"]:
            proposed = theta + fraction * direction
            prediction_new = forward(x, proposed, inputs, hidden)
            actual = mse - (y - prediction_new).square().mean()
            quadratic = 2 * fraction * gd - fraction**2 * curvature
            remainder = prediction_new - prediction - fraction * linear_response
            remainder_contribution = (
                2 * ((residual - fraction * linear_response) * remainder).mean()
                - remainder.square().mean()
            )
            identity_error = float((actual - quadratic - remainder_contribution).abs())
            maximum_identity_error = max(maximum_identity_error, identity_error)
            gates_new = (
                x @ proposed[:entries].reshape(inputs, hidden) + proposed[-2] > 0
            )
            validation_new = (
                (forward(vx, proposed, inputs, hidden) - vy).square().mean()
            )
            output.append(
                {
                    "relative_ridge": relative_ridge,
                    "fraction": fraction,
                    "actual_improvement": float(actual),
                    "quadratic_improvement": float(quadratic),
                    "scaled_actual_improvement": float(scale * actual),
                    "scaled_quadratic_improvement": float(scale * quadratic),
                    "scaled_remainder_contribution": float(
                        scale * remainder_contribution
                    ),
                    "gate_cross_fraction": float(
                        (gates_new != old_gates).double().mean()
                    ),
                    "remainder_rms": float(remainder.square().mean().sqrt()),
                    "validation_improvement": float(validation_mse - validation_new),
                    "positive_bias": bool(proposed[-2] > 0),
                }
            )
            if float(mse - actual) < best_source and proposed[-2] > 0:
                best_source = float(mse - actual)
                selected_theta = proposed.clone()
    if maximum_identity_error > 1e-10 * (1 + float(mse)):
        raise FloatingPointError("Observed quadratic/remainder identity disagrees")
    decrement = scale * score @ torch.linalg.solve(hessian, score)
    result = {
        "job": job,
        "source_mse": float(mse),
        "validation_mse": float(validation_mse),
        "variance": float(variance),
        "scaled_newton_decrement": float(decrement),
        "linear_contrast_shift_bound_se": float(decrement.sqrt()),
        "best_actual_source_improvement": float(mse) - best_source,
        "maximum_quadratic_identity_error": maximum_identity_error,
        "records": output,
        "observation_sha256": fingerprint,
        "parent_outputs_sha256": digest(folder / "learner_outputs.npz"),
        "protocol_sha256": digest(root / "protocol.json"),
        "seconds": time.monotonic() - started,
    }
    destination = root / "models" / f"job_{job_id:03d}"
    np.savez_compressed(
        destination.with_suffix(".npz"),
        theta=theta.cpu().numpy(),
        best_source_step_theta=selected_theta.cpu().numpy(),
    )
    save_json(destination.with_suffix(".json"), result)
    print(
        json.dumps(
            {
                "finished": job_id,
                "decrement": float(decrement),
                "seconds": result["seconds"],
            }
        ),
        flush=True,
    )


def freeze(root, plan):
    hashes = {}
    for job_id in range(len(plan["jobs"])):
        for suffix in ("npz", "json"):
            path = root / "models" / f"job_{job_id:03d}.{suffix}"
            hashes[str(path.relative_to(root))] = digest(path)
    save_json(
        root / "models_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(root / "protocol.json"),
            "sha256": hashes,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "freeze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root, args.parent)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    if args.action == "freeze":
        if (args.root / "models_frozen.json").exists():
            raise RuntimeError("Already frozen")
        freeze(args.root, plan)
        return
    for job_id in range(args.shard, len(plan["jobs"]), args.shards):
        if (args.root / "models" / f"job_{job_id:03d}.json").exists():
            raise RuntimeError("Refusing to repeat completed experiment")
        run_case(args.root, plan, job_id, args.device)


if __name__ == "__main__":
    main()
