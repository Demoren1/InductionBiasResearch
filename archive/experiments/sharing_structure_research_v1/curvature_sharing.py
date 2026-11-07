"""G9: full local information, numerical refinement and nonlinear Wald sharing.

The nonlinear Wald interpretation is asymptotic and local. It does not inherit
the exact finite-sample Gaussian F guarantee of the linear contrast graph.
"""

import argparse
import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .contrasts import resolve_graph
from .nonlinear_moments import labels, make_teacher, project, weight_error

BASE = Path(__file__).resolve().parent


def register(root, parent):
    root.mkdir(parents=True, exist_ok=False)
    source = json.loads((parent / "protocol.json").read_text())
    plan = {
        **{
            k: source[k]
            for k in (
                "seed",
                "data_seed_offset",
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
                "source",
                "readout_support",
                "validation",
                "initializations",
            )
        },
        "stage": "G9 local nonlinear-curvature diagnostic on opened G8 geometries, not independent confirmation",
        "registered_utc": datetime.now(UTC).isoformat(),
        "parent": str(parent.resolve()),
        "parent_protocol_sha256": digest(parent / "protocol.json"),
        "max_newton_steps": 40,
        "line_search_levels": 16,
        "ridge_relative": 1e-10,
        "rank_relative": 1e-10,
        "decrement_stop": 1e-6,
        "decrement_eligibility": 1e-4,
        "alpha_per_family": 0.05,
        "rounding_variance_floor_relative": 1e-14,
        "rounding_reason": "G8 observed labels were stored in float32. Use observed mean squared labels x1e-14 as a conservative numerical floor in noise0; not a finite-sample stochastic guarantee.",
        "methods": ["scalar_dp", "wald_full", "wald_diagonal", "zero_wald_then_dp"],
        "inference": "All E^2 zero/signed-pair Wald relations; alpha divided by8 converged source-fit views per family. Full inverse information includes beta/d nuisance. Exact knownK4 signed graph may abstain. Nonlinear normal approximation is local/asymptotic only, with rank/decrement checks; no exact finite-sample F claim.",
        "controls": "Same refined numeric estimate for all four decoders. Zero-Wald then DP is a known two-stage mask/quantization control, not the proposed direct graph. Diagonal covariance graph is an ablation without its own valid full-covariance guarantee.",
        "data": "Regenerate exactly G8 source/readout/validation tensors and verify observation hash. Use source only for numerical fit/covariance; independent validation reports functional fit. No teacher autograd or fixed mask. No software tests.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": digest(Path(__file__)),
    }
    save_json(root / "protocol.json", plan)


def forward(x, theta, inputs, hidden):
    weights = theta[: inputs * hidden].reshape(inputs, hidden)
    beta, offset = theta[-2:]
    return (x @ weights + beta).relu().sum(-1) / math.sqrt(hidden) + offset


def information(x, y, theta, inputs, hidden, floor):
    entries = inputs * hidden
    weights = theta[:entries].reshape(inputs, hidden)
    gates = (x @ weights + theta[-2]) > 0
    jacobian = x.new_empty((len(x), entries + 2))
    jacobian[:, :entries] = (x[:, :, None] * gates[:, None]).flatten(1) / math.sqrt(
        hidden
    )
    jacobian[:, -2] = gates.sum(-1) / math.sqrt(hidden)
    jacobian[:, -1] = 1
    gram = jacobian.T @ jacobian / len(x)
    residual = y - forward(x, theta, inputs, hidden)
    score = jacobian.T @ residual / len(x)
    mse = residual.square().mean()
    variance = torch.maximum(residual.square().sum() / (len(x) - entries - 2), floor)
    return gram, score, mse, variance


@torch.no_grad()
def refine(x, y, theta, plan):
    theta = theta.clone()
    inputs, hidden = plan["inputs"], plan["hidden"]
    floor = y.square().mean().clamp_min(1) * plan["rounding_variance_floor_relative"]
    history = []
    identity = torch.eye(len(theta), dtype=theta.dtype, device=theta.device)
    for step in range(plan["max_newton_steps"] + 1):
        gram, score, mse, variance = information(x, y, theta, inputs, hidden, floor)
        values, vectors = torch.linalg.eigh(gram)
        valid = values > values[-1] * plan["rank_relative"]
        inverse = (
            vectors * torch.where(valid, values.reciprocal(), 0)[None]
        ) @ vectors.T
        decrement = (len(x) * (score @ inverse @ score) / variance).clamp_min(0)
        history.append(
            {
                "step": step,
                "source_mse": float(mse),
                "variance": float(variance),
                "rank": int(valid.sum()),
                "scaled_newton_decrement": float(decrement),
            }
        )
        if step == plan["max_newton_steps"] or decrement <= plan["decrement_stop"]:
            break
        ridge = values[-1].clamp_min(1e-12) * plan["ridge_relative"]
        direction = torch.linalg.solve(gram + ridge * identity, score)
        accepted = False
        for exponent in range(plan["line_search_levels"]):
            fraction = 2.0 ** (-exponent)
            proposed = theta + fraction * direction
            if proposed[-2] <= 1e-10:
                continue
            proposed_loss = (forward(x, proposed, inputs, hidden) - y).square().mean()
            if proposed_loss < mse:
                theta, accepted = proposed, True
                history[-1]["line_search_fraction"] = fraction
                break
        if not accepted:
            history[-1]["line_search_stopped"] = True
            break
    rank = int(valid.sum())
    eligible = rank == len(theta) and float(decrement) <= plan["decrement_eligibility"]
    metadata = {
        "history": history,
        "source_mse": float(mse),
        "variance": float(variance),
        "variance_floor": float(floor),
        "rank": rank,
        "parameters": len(theta),
        "condition": float(values[-1] / values[0]) if values[0] > 0 else None,
        "scaled_newton_decrement": float(decrement),
        "inference_eligible": eligible,
    }
    return theta, variance / len(x) * inverse, metadata


@torch.no_grad()
def wald_relations(weights, covariance, codes, alpha, diagonal=False):
    flat = weights.flatten()
    entries = len(flat)
    if diagonal:
        covariance = torch.diag(covariance.diag())
    variance = covariance.diag()
    critical = math.sqrt(float(chi2.isf(alpha / entries**2, 1)))
    zero = flat.abs() <= critical * variance.clamp_min(1e-30).sqrt()
    relations = {}
    for sign in (-1, 1):
        contrast_variance = variance[:, None] + variance[None] - 2 * sign * covariance
        difference = flat[:, None] - sign * flat[None]
        relations[sign] = (
            (difference.abs() <= critical * contrast_variance.clamp_min(1e-30).sqrt())
            .cpu()
            .numpy()
        )
    index, metadata, components = resolve_graph(
        zero.cpu().numpy(), relations, codes, weights.shape
    )
    return (
        index,
        {
            **metadata,
            "critical_z": critical,
            "alpha_this_view": alpha,
            "interpretation": "Local/asymptotic nonlinear Wald, not exact F",
        },
        components,
        zero,
    )


def observations(plan, family, teacher, device):
    generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + plan["data_seed_offset"] + family
    )
    data, fingerprint = {}, hashlib.sha256()
    for split in ("source", "readout_support", "validation"):
        x = torch.randn(
            (plan[split], plan["inputs"]), generator=generator, device=device
        )
        clean = labels(x.double(), teacher, plan).float()
        noise = torch.randn(len(x), generator=generator, device=device)
        y = clean[:, None] + noise[:, None] * x.new_tensor(plan["noise"])
        fingerprint.update(x.cpu().numpy().tobytes())
        fingerprint.update(y.cpu().numpy().tobytes())
        data[split] = x.double(), y.double()
    return data, fingerprint.hexdigest()


def run_family(root, plan, family, device):
    folder = root / f"family_{family}"
    folder.mkdir(exist_ok=False)
    parent = Path(plan["parent"]) / f"family_{family}"
    teacher, truth, codes = make_teacher(family, plan["function"], plan, device)
    data, fingerprint = observations(plan, family, teacher, device)
    parent_records = json.loads((parent / "learner_records.json").read_text())
    if fingerprint != parent_records["observation_sha256"]:
        raise RuntimeError("Regenerated G8 observations differ")
    with np.load(parent / "learner_outputs.npz") as archive:
        source = dict(archive)
    entries = plan["inputs"] * plan["hidden"]
    alpha = plan["alpha_per_family"] / (
        len(plan["noise"]) * len(plan["initializations"])
    )
    arrays, records = {}, []
    started = time.perf_counter()
    for chosen in parent_records["selected"]:
        row, noise = chosen["chosen_row"], chosen["noise"]
        noise_id = plan["noise"].index(noise)
        weights = torch.tensor(
            source["all_weights"][row], device=device, dtype=torch.float64
        )
        amplitude, offset = source["all_readout"][row]
        theta = torch.cat(
            (
                (weights * float(amplitude)).flatten(),
                weights.new_tensor([amplitude, offset]),
            )
        )
        x, y = data["source"]
        theta, covariance, numerical = refine(x, y[:, noise_id], theta, plan)
        weights = theta[:entries].reshape(plan["inputs"], plan["hidden"])
        index, _, _ = project(weights, plan["codes"])
        key = f"{chosen['method']}__noise{noise}"
        arrays[key + "__theta"] = theta.cpu().numpy()
        arrays[key + "__covariance"] = covariance.cpu().numpy()
        arrays[key + "__scalar_dp"] = index
        graphs = {}
        for name, diagonal in (("wald_full", False), ("wald_diagonal", True)):
            if numerical["inference_eligible"]:
                inferred, metadata, components, zeros = wald_relations(
                    weights,
                    covariance[:entries, :entries],
                    plan["codes"],
                    alpha,
                    diagonal,
                )
                arrays[key + "__" + name + "_components"] = components
                if inferred is not None:
                    arrays[key + "__" + name] = inferred
                graphs[name] = metadata
            else:
                graphs[name] = {
                    "returned_index": False,
                    "reason": "Rank or stationarity check failed",
                }
        if numerical["inference_eligible"]:
            critical = math.sqrt(float(chi2.isf(alpha / entries**2, 1)))
            zeros = (
                weights.flatten().abs()
                <= critical * covariance.diag()[:entries].clamp_min(1e-30).sqrt()
            )
            masked = weights.clone().flatten()
            masked[zeros] = 0
            index, _, _ = project(masked.reshape(weights.shape), plan["codes"])
            arrays[key + "__zero_wald_then_dp"] = index
        vx, vy = data["validation"]
        validation_mse = float(
            (forward(vx, theta, plan["inputs"], plan["hidden"]) - vy[:, noise_id])
            .square()
            .mean()
        )
        records.append(
            {
                "method": chosen["method"],
                "noise": noise,
                "key": key,
                "validation_mse": validation_mse,
                "numerical": numerical,
                "graphs": graphs,
            }
        )
        print(
            json.dumps(
                {
                    "family": family,
                    "method": chosen["method"],
                    "noise": noise,
                    "seconds": time.perf_counter() - started,
                    "source_mse": numerical["source_mse"],
                    "eligible": numerical["inference_eligible"],
                }
            ),
            flush=True,
        )
    np.savez_compressed(folder / "learner_outputs.npz", **arrays)
    save_json(
        folder / "learner_records.json",
        {
            "records": records,
            "observation_sha256": fingerprint,
            "parent_outputs_sha256": digest(parent / "learner_outputs.npz"),
        },
    )
    save_json(
        folder / "learner_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(root / "protocol.json"),
            "sha256": {
                name: digest(folder / name)
                for name in ("learner_outputs.npz", "learner_records.json")
            },
        },
    )
    audit_arrays = {
        "truth_indices": truth,
        "truth_weights": teacher.cpu().numpy(),
        "truth_codes": codes,
    }
    for record in records:
        key = record["key"]
        theta = arrays[key + "__theta"]
        record["weight_relative_squared_error"] = weight_error(
            theta[:entries].reshape(truth.shape), teacher.cpu().numpy()
        )
        recovery = {}
        for method in plan["methods"]:
            name = key + "__" + method
            if name in arrays:
                recovery[method], aligned, _ = align_indices(
                    arrays[name], truth, plan["codes"]
                )
                audit_arrays[name + "__aligned"] = aligned
        record["recovery"] = recovery
    np.savez_compressed(folder / "audit_only.npz", **audit_arrays)
    save_json(
        folder / "audit.json",
        {
            "family": family,
            "records": records,
            "frozen_sha256": digest(folder / "learner_frozen.json"),
            "seconds": time.perf_counter() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--family", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root, args.parent)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    if args.family not in plan["families"]:
        raise ValueError("Family must be registered")
    run_family(args.root, plan, args.family, args.device)


if __name__ == "__main__":
    main()
