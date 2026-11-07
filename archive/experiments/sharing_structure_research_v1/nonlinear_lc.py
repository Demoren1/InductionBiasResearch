"""G17: known native LC versus dense GN and penalty-only on paired G16 data.

Objective-level signed/fixed-zero adaptation, not an unchanged author script.
No true structure enters fitting, refitting, checkpoint or hyperparameter choice.
"""

import argparse
import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.darts_structured_tasks_v1.codebook_projection import (
    exact_signed_projection,
)
from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .curvature_sharing import forward
from .independent_score import batches, design, validation
from .nonlinear_moments import labels, make_teacher

BASE = Path(__file__).resolve().parent
METHODS = ("lc", "penalty", "dense_gn")


def register(root, parent, warmup_stages=0):
    original = json.loads((parent / "protocol.json").read_text())
    if not (parent / "models_frozen.json").exists():
        raise RuntimeError("G16 source cohort must be frozen")
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
                "noise",
                "chunk",
                "validation",
            )
        },
        "stage": "G17 native nonlinear LC control on opened G16; pilot, not independent confirmation",
        "registered_utc": datetime.now(UTC).isoformat(),
        "parent": str(parent.resolve()),
        "parent_cohort_sha256": digest(parent / "models_frozen.json"),
        "source_n": 1048576,
        "source_labels": 2097152,
        "initializations": ["cold", "moment_pairs8388608"],
        "stages": 24,
        "warmup_stages": warmup_stages,
        "l_solves_per_stage": 6,
        "polish_solves": 12,
        "line_search_levels": 16,
        "settings": {
            "lc": [
                {
                    "mu_initial_relative": value,
                    "mu_final_relative": 10.0,
                    "ridge_relative": 1e-10,
                }
                for value in (1e-6, 1e-4)
            ],
            "penalty": [
                {
                    "mu_initial_relative": value,
                    "mu_final_relative": 10.0,
                    "ridge_relative": 1e-10,
                }
                for value in (1e-6, 1e-4)
            ],
            "dense_gn": [
                {
                    "mu_initial_relative": 0.0,
                    "mu_final_relative": 0.0,
                    "ridge_relative": value,
                }
                for value in (1e-10, 1e-4)
            ],
        },
        "jobs": [
            {
                "family": family,
                "initialization": initialization,
                "method": method,
                "setting": setting,
            }
            for family in original["families"]
            for initialization in ("cold", "moment_pairs8388608")
            for method in (("lc", "penalty") if warmup_stages else METHODS)
            for setting in range(2)
        ],
        "selection": "Source-only checkpoint by observed G16 validation MSE of hard-I+polished C,b,d. One global setting per method by mean selected validation over8 reference datasets. Freeze whole source cohort and choices before new true-I audit.",
        "adaptation": "Known LC augmented-Lagrangian objective with full GN L-step, exact scalar signed0+-K DP C-step, scaled dual rescaled when mu changes. Penalty-only fixes dual0. DenseGN has explicit additional DP readout. Same fixed-I numeric GN polish for all; it never changes I or feeds back into latent trajectory. Not literal author code, no general finite-run convergence guarantee.",
        "cost": "Exactly previous G16 middle-B inputs/replicated labels and independent validation, verified observation SHA. Common frozen G9 references, including expensive warm moment cost. Same max144 L-solves and12 refit-solves/checkpoint; actual accepted/attempted steps and times retained. Data-matched, not claimed FLOP/time-matched. No new source or target-C labels.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": {
            name: digest(BASE / name)
            for name in (
                "nonlinear_lc.py",
                "independent_score.py",
                "nonlinear_moments.py",
                "curvature_sharing.py",
            )
        },
        "projection_sha256": digest(
            BASE.parent / "darts_structured_tasks_v1/codebook_projection.py"
        ),
        "checks": "No software tests; necessary data/provenance/finiteness diagnostics, minimum Ruff and figures only.",
    }
    if warmup_stages:
        primary = root.parent
        plan["additional_control"] = {
            "primary_root": str(primary.resolve()),
            "primary_frozen_sha256": digest(primary / "models_frozen.json"),
            "policy": "Same-data warm-up control after primary diagnosis; preserve primary48. First4 stages mu=u=0, remaining20 positive; total max144 L-solves. Reuse frozen dense controls. Not independent confirmation.",
        }
    save_json(root / "protocol.json", plan)
    shutil.copy2(BASE / "THEORY.md", root / "theory_at_registration.md")
    snapshot = root / "executed_code_snapshot"
    snapshot.mkdir()
    for name in plan["code_sha256"]:
        shutil.copy2(BASE / name, snapshot / name)
    shutil.copy2(
        BASE.parent / "darts_structured_tasks_v1/codebook_projection.py", snapshot
    )
    (root / "models").mkdir()


def source_data(plan, job, device):
    parent = Path(plan["parent"])
    original = json.loads((parent / "protocol.json").read_text())
    source_job = next(
        i
        for i, item in enumerate(original["jobs"])
        if item == {"family": job["family"], "initialization": job["initialization"]}
    )
    path = parent / "models" / f"job_{source_job:03d}"
    frozen = json.loads((parent / "models_frozen.json").read_text())
    for suffix in (".npz", ".json"):
        candidate = path.with_suffix(suffix)
        if digest(candidate) != frozen["sha256"][str(candidate.relative_to(parent))]:
            raise RuntimeError("Frozen G16 source changed")
    metadata = json.loads(path.with_suffix(".json").read_text())
    with np.load(path.with_suffix(".npz")) as a:
        theta = torch.tensor(a["reference_theta"], device=device)
    teacher, _, _ = make_teacher(job["family"], plan["function"], plan, device)
    xs, ys, fingerprint = [], [], hashlib.sha256()
    for n, x, y, difference, _ in batches(original, job["family"], teacher, device):
        xs.append(x)
        ys.append(y)
        for value in (x, y, difference):
            fingerprint.update(value.cpu().numpy().tobytes())
        if n == plan["source_n"]:
            break
    if fingerprint.hexdigest() != metadata["observation_sha256"][str(plan["source_n"])]:
        raise RuntimeError("G17 training observations differ from frozen G16")
    vx, vy = validation(original, job["family"], teacher, device)
    return theta, torch.cat(xs), torch.cat(ys), vx, vy, fingerprint.hexdigest()


def information(x, y, theta, plan, mapping=None):
    dimensions = len(theta) if mapping is None else mapping.shape[1]
    gram = theta.new_zeros((dimensions, dimensions))
    cross = theta.new_zeros(dimensions)
    square = theta.new_zeros(())
    for begin in range(0, len(x), plan["chunk"]):
        local = x[begin : begin + plan["chunk"]]
        residual = y[begin : begin + len(local)] - forward(
            local, theta, plan["inputs"], plan["hidden"]
        )
        jacobian = design(local, theta, plan)
        if mapping is not None:
            jacobian = jacobian @ mapping
        gram += jacobian.T @ jacobian
        cross += jacobian.T @ residual
        square += residual.square().sum()
    return gram / len(x), cross / len(x), square / (2 * len(x))


def half_mse(x, y, theta, plan):
    return (forward(x, theta, plan["inputs"], plan["hidden"]) - y).square().mean() / 2


def gn_step(x, y, parameter, plan, ridge_relative, mu=0.0, target=None, mapping=None):
    theta = parameter if mapping is None else mapping @ parameter
    gram, score, loss = information(x, y, theta, plan, mapping)
    scale = torch.linalg.eigvalsh(gram)[-1].clamp_min(1e-12)
    matrix = gram + scale * ridge_relative * torch.eye(
        len(parameter), device=x.device, dtype=x.dtype
    )
    right = score.clone()
    if mu:
        matrix = matrix.clone()
        matrix.diagonal()[:-2] += mu
        difference = parameter[:-2] - target
        right[:-2] -= mu * difference
        objective = loss + mu * difference.square().sum() / 2
    else:
        objective = loss
    direction = torch.linalg.solve(matrix, right)
    if not torch.isfinite(direction).all():
        raise FloatingPointError("Nonfinite GN direction")
    metadata = {
        "accepted": False,
        "gradient_norm": float(right.norm()),
        "task_half_mse": float(loss),
        "penalized_objective": float(objective),
    }
    for exponent in range(plan["line_search_levels"]):
        fraction = 2.0 ** (-exponent)
        proposed = parameter + fraction * direction
        proposed_theta = proposed if mapping is None else mapping @ proposed
        if proposed_theta[-2] <= 1e-8:
            continue
        proposed_loss = half_mse(x, y, proposed_theta, plan)
        if mu:
            proposed_loss += mu * (proposed[:-2] - target).square().sum() / 2
        if proposed_loss < objective:
            return proposed, {
                **metadata,
                "accepted": True,
                "fraction": fraction,
                "new_penalized_objective": float(proposed_loss),
            }
    return parameter, metadata


def compress(weights, plan):
    index, center, quantized = exact_signed_projection(
        weights[None], plan["codes"], weights.new_zeros(1), weights.new_ones(1)
    )
    index = torch.where(index > plan["codes"], plan["codes"] - index, index)[0]
    return index, center[0], quantized[0]


def polish(x, y, theta, index, centers, plan):
    entries, codes = index.numel(), plan["codes"]
    basis = F.one_hot(index.abs().long(), codes + 1)[..., 1:].to(x.dtype)
    basis *= index.sign()[..., None]
    mapping = x.new_zeros((entries + 2, codes + 2))
    mapping[:entries, :codes] = basis.flatten(0, 1)
    mapping[-2:, -2:] = torch.eye(2, device=x.device, dtype=x.dtype)
    parameter = torch.cat((centers, theta[-2:])).clone()
    attempted, accepted = 0, 0
    for _ in range(plan["polish_solves"]):
        parameter, result = gn_step(x, y, parameter, plan, 1e-10, mapping=mapping)
        attempted += 1
        accepted += result["accepted"]
        if not result["accepted"]:
            break
    return parameter, mapping @ parameter, attempted, accepted


@torch.no_grad()
def run_case(root, plan, job_id, device):
    started = time.monotonic()
    job = plan["jobs"][job_id]
    setting = plan["settings"][job["method"]][job["setting"]]
    theta, x, y, vx, vy, fingerprint = source_data(plan, job, device)
    entries = plan["inputs"] * plan["hidden"]
    shape = (plan["inputs"], plan["hidden"])
    reference = theta.clone()
    hessian, _, _ = information(x, y, theta, plan)
    scale = float(torch.linalg.eigvalsh(hessian)[-1])
    if job["method"] == "dense_gn":
        mus = np.zeros(plan["stages"])
    else:
        warmup = plan.get("warmup_stages", 0)
        positive = scale * np.geomspace(
            setting["mu_initial_relative"],
            setting["mu_final_relative"],
            plan["stages"] - warmup,
        )
        mus = np.r_[np.zeros(warmup), positive]
    dual = theta.new_zeros(entries)
    index, centers, quantized = compress(theta[:entries].reshape(shape), plan)
    records, indices, latents, hard, raw_centers, fitted_centers = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    l_attempted = l_accepted = c_attempted = c_accepted = 0

    def checkpoint(stage, mu, numerical):
        nonlocal c_attempted, c_accepted
        coefficients, compressed, attempts, accepted = polish(
            x, y, theta, index, centers, plan
        )
        c_attempted += attempts
        c_accepted += accepted
        raw = torch.cat((quantized.flatten(), theta[-2:]))
        records.append(
            {
                "stage": stage,
                "mu": float(mu),
                "l_solves": l_attempted,
                "l_updates": l_accepted,
                "refit_solves": c_attempted,
                "refit_updates": c_accepted,
                "total_solves": l_attempted + c_attempted,
                "source_mse_latent": float(2 * half_mse(x, y, theta, plan)),
                "source_mse_raw_compressed": float(2 * half_mse(x, y, raw, plan)),
                "source_mse_polished": float(2 * half_mse(x, y, compressed, plan)),
                "validation_mse_raw": float(2 * half_mse(vx, vy, raw, plan)),
                "validation_mse": float(2 * half_mse(vx, vy, compressed, plan)),
                "relative_primal_residual": float(
                    (theta[:entries] - quantized.flatten()).norm()
                    / theta[:entries].norm().clamp_min(1)
                ),
                "active": int((index != 0).sum()),
                "groups": int(index.abs().max()),
                "last_l_numerical": numerical,
            }
        )
        indices.append(index.cpu().numpy().astype(np.int16))
        latents.append(theta.cpu().numpy())
        hard.append(compressed.cpu().numpy())
        raw_centers.append(centers.cpu().numpy())
        fitted_centers.append(coefficients.cpu().numpy())
        print(
            json.dumps(
                {
                    "job": job_id,
                    "stage": stage,
                    "val": records[-1]["validation_mse"],
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )

    checkpoint(0, 0.0, None)
    previous_mu = mus[0]
    dense_stopped = False
    for stage, mu in enumerate(mus, 1):
        if job["method"] == "lc":
            if mu > 0 and previous_mu > 0:
                dual *= previous_mu / mu
            else:
                dual.zero_()
        target = quantized.flatten() - dual
        stage_accepted = 0
        for _ in range(plan["l_solves_per_stage"]):
            theta, numerical = gn_step(
                x,
                y,
                theta,
                plan,
                setting["ridge_relative"],
                mu=float(mu),
                target=target,
            )
            l_attempted += 1
            l_accepted += numerical["accepted"]
            stage_accepted += numerical["accepted"]
            if not numerical["accepted"]:
                break
        if job["method"] == "dense_gn" and stage_accepted == 0:
            dense_stopped = True
            break
        index, centers, quantized = compress(
            (theta[:entries] + dual).reshape(shape), plan
        )
        if job["method"] == "lc" and mu > 0:
            dual += theta[:entries] - quantized.flatten()
        previous_mu = mu
        checkpoint(stage, mu, numerical)
    scores = np.array([record["validation_mse"] for record in records])
    winner = int(scores.argmin())
    arrays = {
        "trace_indices": np.stack(indices),
        "trace_latent_theta": np.stack(latents),
        "trace_hard_theta": np.stack(hard),
        "trace_raw_centers": np.stack(raw_centers),
        "trace_polished_coefficients": np.stack(fitted_centers),
        "selected_index": indices[winner],
        "selected_theta": hard[winner],
        "reference_theta": reference.cpu().numpy(),
    }
    if not np.isfinite(np.stack(hard)).all() or not np.isfinite(scores).all():
        raise FloatingPointError("Invalid nonlinear LC trajectory")
    path = root / "models" / f"job_{job_id:03d}"
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    save_json(
        path.with_suffix(".json"),
        {
            "job": job,
            "records": records,
            "selected_checkpoint": winner,
            "selected_validation_mse": float(scores[winner]),
            "dense_stopped": dense_stopped,
            "observation_sha256": fingerprint,
            "initial_information_scale": scale,
            "l_solves": l_attempted,
            "l_updates": l_accepted,
            "refit_solves": c_attempted,
            "refit_updates": c_accepted,
            "protocol_sha256": digest(root / "protocol.json"),
            "seconds": time.monotonic() - started,
        },
    )


def freeze(root, plan):
    methods = sorted({job["method"] for job in plan["jobs"]})
    hashes, scores = {}, {method: [[], []] for method in methods}
    for i, job in enumerate(plan["jobs"]):
        path = root / "models" / f"job_{i:03d}"
        for suffix in (".npz", ".json"):
            candidate = path.with_suffix(suffix)
            hashes[str(candidate.relative_to(root))] = digest(candidate)
        data = json.loads(path.with_suffix(".json").read_text())
        scores[job["method"]][job["setting"]].append(data["selected_validation_mse"])
    means = {
        method: np.mean(values, axis=1).tolist() for method, values in scores.items()
    }
    choices = {method: int(np.argmin(values)) for method, values in means.items()}
    save_json(
        root / "models_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(root / "protocol.json"),
            "sha256": hashes,
            "global_settings": choices,
            "mean_validation_mse": means,
        },
    )


def audit(root, plan, device):
    frozen = json.loads((root / "models_frozen.json").read_text())
    records = []
    for i, job in enumerate(plan["jobs"]):
        path = root / "models" / f"job_{i:03d}"
        for suffix in (".npz", ".json"):
            candidate = path.with_suffix(suffix)
            if digest(candidate) != frozen["sha256"][str(candidate.relative_to(root))]:
                raise RuntimeError("Frozen G17 model changed")
        meta = json.loads(path.with_suffix(".json").read_text())
        teacher, truth, _ = make_teacher(job["family"], plan["function"], plan, device)
        with np.load(path.with_suffix(".npz")) as a:
            recovery, aligned, _ = align_indices(
                a["selected_index"], truth, plan["codes"]
            )
            masks = a["trace_indices"] != 0
            theta = torch.tensor(a["selected_theta"], device=device)
        vx, _ = validation(plan, job["family"], teacher, device)
        clean_mse = float(
            (
                forward(vx, theta, plan["inputs"], plan["hidden"])
                - labels(vx, teacher, plan)
            )
            .square()
            .mean()
        )
        # Hidden permutations are matched by each full structural audit, not by
        # comparing raw column positions to the teacher's arbitrary neuron order.
        trace_recall = []
        with np.load(path.with_suffix(".npz")) as a:
            for index in a["trace_indices"]:
                current, _, _ = align_indices(index, truth, plan["codes"])
                trace_recall.append(current["mask_recall"])
        np.savez_compressed(
            path.parent / (path.name + "_audit_only.npz"),
            truth=truth,
            selected_aligned=aligned,
            truth_weights=teacher.cpu().numpy(),
        )
        records.append(
            {
                "job_id": i,
                "job": job,
                "selected_global_setting": job["setting"]
                == frozen["global_settings"][job["method"]],
                "recovery": {
                    **recovery,
                    "exact_I": recovery["categorical_errors"] == 0,
                },
                "selected_validation_mse": meta["selected_validation_mse"],
                "selected_clean_validation_mse": clean_mse,
                "trace_mask_recall": trace_recall,
                "trace_active": masks.sum((-1, -2)).tolist(),
            }
        )
    save_json(
        root / "audit.json",
        {
            "cohort_frozen_sha256": digest(root / "models_frozen.json"),
            "records": records,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "freeze", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--warmup-stages", type=int, choices=(0, 4), default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root, args.parent, args.warmup_stages)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    if args.action in ("freeze", "audit"):
        destination = args.root / (
            "models_frozen.json" if args.action == "freeze" else "audit.json"
        )
        if destination.exists():
            raise RuntimeError("Scientific phase already completed")
        if args.action == "freeze":
            freeze(args.root, plan)
        else:
            audit(args.root, plan, args.device)
    else:
        for i in range(args.shard, len(plan["jobs"]), args.shards):
            if (args.root / "models" / f"job_{i:03d}.json").exists():
                raise RuntimeError("Refusing to repeat completed fit")
            run_case(args.root, plan, i, args.device)


if __name__ == "__main__":
    main()
