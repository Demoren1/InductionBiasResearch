"""G16: independent one-step scores and an explicit pseudo-target bias audit.

Replicated labels studentize the design-conditional pseudo-target exactly in
the Gaussian model. This does not establish nonlinear true-parameter coverage.
"""

import argparse
import hashlib
import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import t

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .contrasts import resolve_graph
from .curvature_sharing import forward
from .interval_indexator import interval_indexator
from .nonlinear_moments import labels, make_teacher, project

BASE = Path(__file__).resolve().parent
METHODS = ("scalar_dp", "zero_student_dp", "signed_student_graph", "global_interval")


def register(root, parent):
    original = json.loads((parent / "protocol.json").read_text())
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        **{
            key: original[key]
            for key in (
                "seed",
                "families",
                "function",
                "points",
                "features",
                "point_hidden",
                "codes",
                "inputs",
                "hidden",
                "bias",
                "amplitude",
                "initializations",
            )
        },
        "stage": "G16 independent-score diagnosis on opened G9 geometries; known one-step control, not competitive confirmation",
        "registered_utc": datetime.now(UTC).isoformat(),
        "parent": str(parent.resolve()),
        "parent_protocol_sha256": digest(parent / "protocol.json"),
        "previous_goal_turn": "Progress: G14/G15 changed authoritative evidence and determined the next diagnosis",
        "noise": 0.2,
        "observations": [262144, 1048576, 4194304],
        "labels_per_input": 2,
        "chunk": 32768,
        "validation": 16384,
        "alpha_per_family": 0.05,
        "methods": list(METHODS),
        "jobs": [
            {"family": family, "initialization": initialization}
            for family in original["families"]
            for initialization in original["initializations"]
        ],
        "selection": "No teacher/validation selection of reference, n or method. All16 fixed G9 references and all3 nested independent-B budgets retained. Explicit point/readout algorithms; no native DARTS claim.",
        "confidence_scope": "Exact Student statements conditional on A,x refer to pseudo-target theta_B^circ, not true nonlinear W. Unknown true bias audited only after whole-cohort freeze. Alpha divided by4 references and3 looks, separately per algorithm/family. No cross-method/family guarantee.",
        "data": "New float64 Gaussian IID inputs and two independent noise0.2 labels at each x. Same nested B per family across references, B seed=seed+16000000+family. Independent validation seed=seed+17000000+family. Budget counts2n B labels plus previous reference-learning cost; expensive moment warm cost is not omitted or called matched baseline training.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": {
            name: digest(BASE / name)
            for name in (
                "independent_score.py",
                "curvature_sharing.py",
                "nonlinear_moments.py",
                "interval_indexator.py",
                "contrasts.py",
            )
        },
        "checks": "No software tests. Scientific covariance/noise-decomposition/provenance guards, minimum lint and figures only.",
    }
    save_json(root / "protocol.json", plan)
    shutil.copy2(BASE / "THEORY.md", root / "theory_at_registration.md")
    snapshot = root / "executed_code_snapshot"
    snapshot.mkdir()
    for name in plan["code_sha256"]:
        shutil.copy2(BASE / name, snapshot / name)
    (root / "models").mkdir()


def batches(plan, family, teacher, device):
    generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + 16000000 + family
    )
    for end in range(plan["chunk"], max(plan["observations"]) + 1, plan["chunk"]):
        x = torch.randn(
            (plan["chunk"], plan["inputs"]),
            generator=generator,
            device=device,
            dtype=torch.float64,
        )
        noise = torch.randn(
            (len(x), 2), generator=generator, device=device, dtype=torch.float64
        )
        clean = labels(x, teacher, plan)
        pair = clean[:, None] + plan["noise"] * noise
        yield end, x, pair.mean(-1), pair[:, 0] - pair[:, 1], clean


def design(x, theta, plan):
    entries = plan["inputs"] * plan["hidden"]
    weights = theta[:entries].reshape(plan["inputs"], plan["hidden"])
    gate = x @ weights + theta[-2] > 0
    jacobian = x.new_empty((len(x), entries + 2))
    jacobian[:, :entries] = (x[:, :, None] * gate[:, None]).flatten(1) / math.sqrt(
        plan["hidden"]
    )
    jacobian[:, -2] = gate.sum(-1) / math.sqrt(plan["hidden"])
    jacobian[:, -1] = 1
    return jacobian


def validation(plan, family, teacher, device):
    generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + 17000000 + family
    )
    x = torch.randn(
        (plan["validation"], plan["inputs"]),
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    y = labels(x, teacher, plan)
    y += plan["noise"] * torch.randn(
        len(x), generator=generator, device=device, dtype=torch.float64
    )
    return x, y


def readout(theta, covariance, n, plan):
    entries = plan["inputs"] * plan["hidden"]
    shape = (plan["inputs"], plan["hidden"])
    flat = theta[:entries]
    weights = flat.reshape(shape)
    local = covariance[:entries, :entries]
    se = local.diagonal().clamp_min(0).sqrt()
    alpha = plan["alpha_per_family"] / (
        len(plan["initializations"]) * len(plan["observations"])
    )
    critical = float(t.isf(alpha / (2 * entries), n))
    lower, upper = flat - critical * se, flat + critical * se
    result = {"scalar_dp": project(weights, plan["codes"])[0]}
    masked = weights.clone().flatten()
    masked[flat.abs() <= critical * se] = 0
    result["zero_student_dp"] = project(masked.reshape(shape), plan["codes"])[0]
    pair_critical = float(t.isf(alpha / (2 * entries**2), n))
    relations = {}
    for sign in (-1, 1):
        contrast_variance = (
            local.diagonal()[:, None] + local.diagonal()[None] - 2 * sign * local
        ).clamp_min(0)
        difference = flat[:, None] - sign * flat[None]
        relations[sign] = (
            (difference.abs() <= pair_critical * contrast_variance.sqrt()).cpu().numpy()
        )
    zeros = (flat.abs() <= pair_critical * se).cpu().numpy()
    inferred, graph, _ = resolve_graph(zeros, relations, plan["codes"], shape)
    if inferred is not None:
        result["signed_student_graph"] = inferred
    interval, interval_metadata = interval_indexator(
        lower.cpu().numpy().reshape(shape),
        upper.cpu().numpy().reshape(shape),
        plan["codes"],
    )
    if interval is not None:
        result["global_interval"] = interval
    return (
        result,
        {
            "graph": graph,
            "interval": interval_metadata,
            "critical": critical,
            "pair_critical": pair_critical,
        },
        lower,
        upper,
    )


def load_reference(plan, job, device):
    folder = Path(plan["parent"]) / f"family_{job['family']}"
    frozen = json.loads((folder / "learner_frozen.json").read_text())
    for name, expected in frozen["sha256"].items():
        if digest(folder / name) != expected:
            raise RuntimeError("Frozen G9 reference changed")
    key = f"{job['initialization']}__noise0.2__theta"
    with np.load(folder / "learner_outputs.npz") as archive:
        theta = torch.tensor(archive[key], device=device)
    return theta, digest(folder / "learner_outputs.npz")


@torch.no_grad()
def run_case(root, plan, job_id, device):
    started = time.monotonic()
    job = plan["jobs"][job_id]
    theta0, reference_hash = load_reference(plan, job, device)
    teacher, _, _ = make_teacher(job["family"], plan["function"], plan, device)
    vx, vy = validation(plan, job["family"], teacher, device)
    before_loss = float(
        (forward(vx, theta0, plan["inputs"], plan["hidden"]) - vy).square().mean()
    )
    gram = theta0.new_zeros((len(theta0), len(theta0)))
    cross = theta0.new_zeros(len(theta0))
    pair_square = theta0.new_zeros(())
    arrays = {"reference_theta": theta0.cpu().numpy()}
    records, fingerprints = [], {}
    fingerprint = hashlib.sha256()
    for n, x, mean_y, difference, _ in batches(plan, job["family"], teacher, device):
        jacobian = design(x, theta0, plan)
        response = mean_y - forward(x, theta0, plan["inputs"], plan["hidden"])
        gram += jacobian.T @ jacobian
        cross += jacobian.T @ response
        pair_square += difference.square().sum()
        fingerprint.update(x.cpu().numpy().tobytes())
        fingerprint.update(mean_y.cpu().numpy().tobytes())
        fingerprint.update(difference.cpu().numpy().tobytes())
        if n not in plan["observations"]:
            continue
        hessian = gram / n
        inverse = torch.linalg.inv(hessian)
        noise_variance = pair_square / (2 * n)
        covariance = noise_variance * inverse / (2 * n)
        theta = theta0 + torch.linalg.solve(gram, cross)
        inferred, metadata, lower, upper = readout(theta, covariance, n, plan)
        if not torch.isfinite(theta).all() or float(covariance.diagonal().min()) <= 0:
            raise FloatingPointError("Invalid independent-score estimate/covariance")
        key = f"n{n}"
        arrays[key + "__theta"] = theta.cpu().numpy()
        arrays[key + "__gram"] = hessian.cpu().numpy()
        arrays[key + "__covariance"] = covariance.cpu().numpy()
        arrays[key + "__lower"] = lower.cpu().numpy()
        arrays[key + "__upper"] = upper.cpu().numpy()
        for method, index in inferred.items():
            arrays[key + "__" + method] = index
        fingerprints[str(n)] = fingerprint.hexdigest()
        records.append(
            {
                "n": n,
                "B_label_budget": 2 * n,
                "noise_variance": float(noise_variance),
                "validation_mse": float(
                    (forward(vx, theta, plan["inputs"], plan["hidden"]) - vy)
                    .square()
                    .mean()
                ),
                "positive_bias": bool(theta[-2] > 0),
                "returned": {method: method in inferred for method in METHODS},
                "inference": metadata,
            }
        )
        print(
            json.dumps({"job": job_id, "n": n, "seconds": time.monotonic() - started}),
            flush=True,
        )
    folder = root / "models" / f"job_{job_id:03d}"
    np.savez_compressed(folder.with_suffix(".npz"), **arrays)
    save_json(
        folder.with_suffix(".json"),
        {
            "job": job,
            "records": records,
            "reference_validation_mse": before_loss,
            "observation_sha256": fingerprints,
            "reference_sha256": reference_hash,
            "protocol_sha256": digest(root / "protocol.json"),
            "seconds": time.monotonic() - started,
        },
    )


def freeze(root, plan):
    hashes = {}
    for i in range(len(plan["jobs"])):
        for suffix in ("npz", "json"):
            path = root / "models" / f"job_{i:03d}.{suffix}"
            hashes[str(path.relative_to(root))] = digest(path)
    save_json(
        root / "models_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(root / "protocol.json"),
            "sha256": hashes,
        },
    )


@torch.no_grad()
def audit_case(root, plan, job_id, device):
    job = plan["jobs"][job_id]
    folder = root / "models" / f"job_{job_id:03d}"
    frozen = json.loads((root / "models_frozen.json").read_text())
    for suffix in ("npz", "json"):
        path = folder.with_suffix("." + suffix)
        if digest(path) != frozen["sha256"][str(path.relative_to(root))]:
            raise RuntimeError("Frozen G16 model changed")
    with np.load(folder.with_suffix(".npz")) as archive:
        arrays = dict(archive)
    metadata = json.loads(folder.with_suffix(".json").read_text())
    theta0 = torch.tensor(arrays["reference_theta"], device=device)
    teacher, truth, _ = make_teacher(job["family"], plan["function"], plan, device)
    entries = plan["inputs"] * plan["hidden"]
    old_w = arrays["reference_theta"][:entries].reshape(truth.shape)
    cost = ((old_w[:, :, None] - teacher.cpu().numpy()[:, None, :]) ** 2).sum(0)
    source, target = linear_sum_assignment(cost)
    permutation = target[np.argsort(source)]
    truth_aligned = teacher[:, permutation]
    theta_true = torch.cat(
        (truth_aligned.flatten(), theta0.new_tensor([plan["bias"], 0.0]))
    )
    noise_score = theta0.new_zeros(len(theta0))
    records, audit_arrays = (
        [],
        {"truth_indices": truth, "truth_weights": teacher.cpu().numpy()},
    )
    fingerprint = hashlib.sha256()
    for n, x, mean_y, difference, clean in batches(
        plan, job["family"], teacher, device
    ):
        jacobian = design(x, theta0, plan)
        noise_score += jacobian.T @ (mean_y - clean)
        fingerprint.update(x.cpu().numpy().tobytes())
        fingerprint.update(mean_y.cpu().numpy().tobytes())
        fingerprint.update(difference.cpu().numpy().tobytes())
        if n not in plan["observations"]:
            continue
        if fingerprint.hexdigest() != metadata["observation_sha256"][str(n)]:
            raise RuntimeError("Audit regenerated different B observations")
        key = f"n{n}"
        hessian = torch.tensor(arrays[key + "__gram"], device=device)
        theta = torch.tensor(arrays[key + "__theta"], device=device)
        covariance = torch.tensor(arrays[key + "__covariance"], device=device)
        noise_shift = torch.linalg.solve(hessian, noise_score / n)
        pseudo_target = theta - noise_shift
        bias = pseudo_target - theta_true
        se = covariance.diagonal().sqrt()
        lower = torch.tensor(arrays[key + "__lower"], device=device)
        upper = torch.tensor(arrays[key + "__upper"], device=device)
        coverage_true = (lower <= theta_true[:entries]) & (
            theta_true[:entries] <= upper
        )
        coverage_pseudo = (lower <= pseudo_target[:entries]) & (
            pseudo_target[:entries] <= upper
        )
        recovery = {}
        for method in METHODS:
            if key + "__" + method in arrays:
                metric, aligned, _ = align_indices(
                    arrays[key + "__" + method], truth, plan["codes"]
                )
                recovery[method] = {
                    **metric,
                    "exact_I": metric["categorical_errors"] == 0,
                }
                audit_arrays[key + "__" + method + "__aligned"] = aligned
        records.append(
            {
                "n": n,
                "all_true_weight_intervals_cover": bool(coverage_true.all()),
                "all_pseudo_weight_intervals_cover": bool(coverage_pseudo.all()),
                "noncovered_true_weights": int((~coverage_true).sum()),
                "noncovered_pseudo_weights": int((~coverage_pseudo).sum()),
                "max_standardized_true_error": float(
                    ((theta - theta_true)[:entries] / se[:entries]).abs().max()
                ),
                "max_standardized_pseudo_error": float(
                    (noise_shift[:entries] / se[:entries]).abs().max()
                ),
                "max_standardized_linearization_bias": float(
                    (bias[:entries] / se[:entries]).abs().max()
                ),
                "rms_true_weight_error": float(
                    (theta - theta_true)[:entries].square().mean().sqrt()
                ),
                "rms_linearization_bias": float(bias[:entries].square().mean().sqrt()),
                "recovery": recovery,
            }
        )
        audit_arrays[key + "__bias"] = bias.cpu().numpy()
        audit_arrays[key + "__noise_shift"] = noise_shift.cpu().numpy()
        audit_arrays[key + "__truth_theta"] = theta_true.cpu().numpy()
    np.savez_compressed(
        folder.parent / (folder.name + "_audit_only.npz"), **audit_arrays
    )
    save_json(
        folder.parent / (folder.name + "_audit.json"),
        {
            "job": job,
            "records": records,
            "cohort_frozen_sha256": digest(root / "models_frozen.json"),
            "alignment": "Hidden permutation fixed by reference-to-teacher W distance, audit only; structural scores separately exhaust code/sign/hidden symmetries.",
        },
    )
    print(json.dumps({"audited": job_id}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "freeze", "audit"))
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
            raise RuntimeError("Cohort already frozen")
        freeze(args.root, plan)
        return
    for i in range(args.shard, len(plan["jobs"]), args.shards):
        suffix = "_audit.json" if args.action == "audit" else ".json"
        if (args.root / "models" / (f"job_{i:03d}" + suffix)).exists():
            raise RuntimeError("Refusing to overwrite completed scientific work")
        function = audit_case if args.action == "audit" else run_case
        function(args.root, plan, i, args.device)


if __name__ == "__main__":
    main()
