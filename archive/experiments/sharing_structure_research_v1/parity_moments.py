"""G7: matched-label parity and known joint-diagonalization ablations."""

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .nonlinear_moments import (
    labels,
    make_teacher,
    moment_sums,
    population_moments,
    project,
    weight_error,
)

BASE = Path(__file__).resolve().parent


def register(root):
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "stage": "G7 diagnostic factorial ablation on opened G6 geometries, not independent confirmation or novelty claim",
        "registered_utc": datetime.now(UTC).isoformat(),
        "seed": 2611056817,
        "data_seed_offset": 2000000,
        "families": [8401, 8402, 8403, 8404],
        "functions": ["original_codes", "fresh_gaussian_codes"],
        "points": 8,
        "features": 2,
        "point_hidden": 2,
        "codes": 4,
        "inputs": 16,
        "hidden": 16,
        "bias": 0.15,
        "amplitude": 1.0,
        "noise": [0.0, 0.2],
        "pairs": [16384, 131072, 1048576, 8388608, 67108864, 536870912],
        "chunk": 65536,
        "affine_pilot_pairs": 8192,
        "reconstruction_sketches": 8,
        "joint_sweeps": 50,
        "joint_angle_tolerance": 1e-8,
        "methods": [
            "iid_affine__sketch",
            "iid_affine__joint",
            "antithetic_affine__sketch",
            "antithetic_affine__joint",
        ],
        "labels": "Each IID view uses2n independent labels, each parity view n pairs at x/-x, also2n labels. The same16384 additional paired pilot labels estimate affine coefficients for both views. Different views do not exchange main-source labels. Noise at x/-x is independent. Counts concern actual noisy function evaluations, not free second branches.",
        "known": "Identical architectural and Gaussian-input assumptions as G6. Positive common bias/readout known qualitatively, their numerical values and W/I/roles not given. No teacher autograd.",
        "readout": "Additional exact signed scalar DP, fixed zero and at most4 centers, zero sparsity penalty. This is explicit quantization, not native neural assignments.",
        "stopping": "All prespecified budgets run. No teacher-driven early stopping or settings selection. Per-function models frozen before its structural audit. No software tests.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "executed_sha256": {
            name: digest(BASE / name)
            for name in ("parity_moments.py", "nonlinear_moments.py")
        },
    }
    save_json(root / "protocol.json", plan)


def joint_diagonalize(matrices, initial, sweeps, tolerance):
    """Known real symmetric Jacobi rotations, with an explicit objective guard."""
    orthogonal = initial.copy()
    rotated = np.einsum("ip,kij,jq->kpq", orthogonal, matrices, orthogonal)

    def objective(values):
        return max(
            float(
                np.square(values).sum()
                - np.square(np.diagonal(values, axis1=1, axis2=2)).sum()
            ),
            0.0,
        )

    before = objective(rotated)
    for sweep in range(sweeps):
        maximum_angle = 0.0
        for p in range(initial.shape[0] - 1):
            for q in range(p + 1, initial.shape[0]):
                a = rotated[:, p, p] - rotated[:, q, q]
                b = 2 * rotated[:, p, q]
                angle = 0.25 * math.atan2(2 * np.dot(a, b), np.dot(a, a) - np.dot(b, b))
                maximum_angle = max(maximum_angle, abs(angle))
                cosine, sine = math.cos(angle), math.sin(angle)
                first, second = rotated[:, p, :].copy(), rotated[:, q, :].copy()
                rotated[:, p, :] = cosine * first + sine * second
                rotated[:, q, :] = -sine * first + cosine * second
                first, second = rotated[:, :, p].copy(), rotated[:, :, q].copy()
                rotated[:, :, p] = cosine * first + sine * second
                rotated[:, :, q] = -sine * first + cosine * second
                first, second = orthogonal[:, p].copy(), orthogonal[:, q].copy()
                orthogonal[:, p] = cosine * first + sine * second
                orthogonal[:, q] = -sine * first + cosine * second
        if maximum_angle < tolerance:
            break
    after = objective(rotated)
    if after > before + 1e-9 * max(float(np.square(matrices).sum()), 1.0):
        raise ArithmeticError("Jacobi joint-diagonalization increased its objective")
    return orthogonal, {
        "joint_initial_offdiagonal": before,
        "joint_final_offdiagonal": after,
        "joint_sweeps": sweep + 1,
    }


@torch.no_grad()
def reconstruct_variants(second, third, sketches, plan):
    values, vectors = torch.linalg.eigh(second)
    common = {"min_eigenvalue": float(values[0]), "max_eigenvalue": float(values[-1])}
    if values[0] <= values[-1] * 1e-10:
        return {
            name: (None, {**common, "reason": "Nonpositive M2"})
            for name in ("sketch", "joint")
        }
    square = (vectors * values.sqrt()) @ vectors.T
    inverse = (vectors / values.sqrt()) @ vectors.T
    whitened = torch.einsum("ia,jb,kc,ijk->abc", inverse, inverse, inverse, third)
    best = None
    for sketch in sketches.T:
        _, orthogonal = torch.linalg.eigh(torch.einsum("ijk,k->ij", whitened, sketch))
        transformed = torch.einsum(
            "ia,jb,kc,ijk->abc", orthogonal, orthogonal, orthogonal, whitened
        )
        off = (
            transformed.square().sum()
            - torch.einsum("iii->i", transformed).square().sum()
        ).clamp_min(0)
        if best is None or off < best[0]:
            best = off, orthogonal
    off, orthogonal = best
    joint, joint_info = joint_diagonalize(
        whitened.permute(2, 0, 1).cpu().numpy(),
        orthogonal.cpu().numpy(),
        plan["joint_sweeps"],
        plan["joint_angle_tolerance"],
    )
    result = {}
    for name, basis, info in (
        ("sketch", orthogonal, {}),
        (
            "joint",
            torch.tensor(joint, dtype=second.dtype, device=second.device),
            joint_info,
        ),
    ):
        direction = square @ basis
        direction /= direction.norm(dim=0, keepdim=True)
        dual = inverse @ basis
        contraction = torch.einsum("ijk,ia,ja->ka", third, dual, dual)
        ratio = (contraction * direction).sum(0)
        transformed = torch.einsum("ia,jb,kc,ijk->abc", basis, basis, basis, whitened)
        residual = (
            transformed.square().sum()
            - torch.einsum("iii->i", transformed).square().sum()
        ).clamp_min(0) / whitened.square().sum()
        info = {**common, **info, "tensor_offdiagonal_fraction": float(residual)}
        if ratio.abs().min() < 1e-12:
            result[name] = None, {**info, "reason": "Vanishing third-order scale"}
        else:
            result[name] = -direction / ratio, info
    return result


def run_function(root, plan, family, function, device):
    folder = root / f"family_{family}" / function
    folder.mkdir(parents=True, exist_ok=False)
    weights, truth, codes = make_teacher(family, function, plan, device)
    seed = (
        plan["seed"]
        + plan["data_seed_offset"]
        + family
        + 100000 * (function != "original_codes")
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    dimensions = plan["inputs"]
    noise_scale = weights.new_tensor(plan["noise"])

    def observe(x):
        clean = labels(x, weights, plan)
        noise = torch.randn(len(x), generator=generator, device=device, dtype=x.dtype)
        return clean[:, None] + noise[:, None] * noise_scale

    pilot_x = torch.randn(
        (plan["affine_pilot_pairs"], dimensions),
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    pilot_x = torch.cat((pilot_x, -pilot_x))
    pilot_y = observe(pilot_x)
    pilot_design = torch.cat((torch.ones_like(pilot_x[:, :1]), pilot_x), -1)
    affine = torch.linalg.lstsq(pilot_design, pilot_y).solution
    sketches = torch.randn(
        (dimensions, plan["reconstruction_sketches"]),
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    cumulative = {
        view: [
            weights.new_zeros((2, dimensions, dimensions)),
            weights.new_zeros((2, dimensions, dimensions, dimensions)),
        ]
        for view in ("iid_affine", "antithetic_affine")
    }
    records, arrays = [], {}
    count, started = 0, time.perf_counter()
    for n in plan["pairs"]:
        while count < n:
            batch = min(plan["chunk"], n - count)
            x = torch.randn(
                (2 * batch, dimensions),
                generator=generator,
                device=device,
                dtype=torch.float64,
            )
            y = observe(x)
            design = torch.cat((torch.ones_like(x[:, :1]), x), -1)
            _, second, third = moment_sums(x, y - design @ affine)
            cumulative["iid_affine"][0].add_(second)
            cumulative["iid_affine"][1].add_(third)
            paired_x = torch.randn(
                (batch, dimensions),
                generator=generator,
                device=device,
                dtype=torch.float64,
            )
            positive, negative = observe(paired_x), observe(-paired_x)
            even = (positive + negative) / 2 - affine[:1]
            odd = (positive - negative) / 2 - paired_x @ affine[1:]
            _, second, third = moment_sums(paired_x, torch.cat((even, odd), -1))
            cumulative["antithetic_affine"][0].add_(second[:2])
            cumulative["antithetic_affine"][1].add_(third[2:])
            count += batch
        for view, totals in cumulative.items():
            divisor = 2 * n if view == "iid_affine" else n
            for noise_id, noise in enumerate(plan["noise"]):
                second, third = [
                    value[noise_id] / divisor for value in totals
                ]
                moment_key = f"{view}__noise{noise}__pairs{n}"
                arrays[f"{moment_key}__second"] = second.cpu().numpy()
                arrays[f"{moment_key}__third"] = third.cpu().numpy()
                results = reconstruct_variants(second, third, sketches, plan)
                for solver, (reconstructed, info) in results.items():
                    key = f"{view}__{solver}__noise{noise}__pairs{n}"
                    record = {
                        "method": f"{view}__{solver}",
                        "noise": noise,
                        "pairs": n,
                        "query_labels": 2 * n,
                        "pilot_labels": len(pilot_x),
                        "key": key,
                        "moment_key": moment_key,
                        **info,
                    }
                    if reconstructed is not None:
                        index, centers, quantized = project(
                            reconstructed, plan["codes"]
                        )
                        arrays[f"{key}__weights_over_bias"] = (
                            reconstructed.cpu().numpy()
                        )
                        arrays[f"{key}__indices"] = index
                        arrays[f"{key}__centers"] = centers.cpu().numpy()
                        arrays[f"{key}__quantized"] = quantized.cpu().numpy()
                    records.append(record)
        print(
            json.dumps(
                {
                    "family": family,
                    "function": function,
                    "pairs": n,
                    "labels_per_view": 2 * n,
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    np.savez_compressed(folder / "learner_outputs.npz", **arrays)
    save_json(folder / "learner_records.json", {"records": records})
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
    # Population values below are diagnostic audit only, after learner freeze.
    population_second, population_third = population_moments(weights, plan)
    audit_arrays = {
        "truth_indices": truth,
        "truth_weights": weights.cpu().numpy(),
        "truth_codes": codes,
    }
    for record in records:
        key, moment_key = record["key"], record["moment_key"]
        record["relative_second_error"] = float(
            (
                torch.tensor(arrays[f"{moment_key}__second"], device=device)
                - population_second
            ).norm()
            / population_second.norm()
        )
        record["relative_third_error"] = float(
            (
                torch.tensor(arrays[f"{moment_key}__third"], device=device)
                - population_third
            ).norm()
            / population_third.norm()
        )
        if f"{key}__indices" in arrays:
            recovery, aligned, _ = align_indices(
                arrays[f"{key}__indices"], truth, plan["codes"]
            )
            record["recovery"] = recovery
            record["weight_relative_squared_error"] = weight_error(
                arrays[f"{key}__weights_over_bias"],
                weights.cpu().numpy() / plan["bias"],
            )
            audit_arrays[f"{key}__aligned_indices"] = aligned
    np.savez_compressed(folder / "audit_only.npz", **audit_arrays)
    save_json(
        folder / "audit.json",
        {
            "family": family,
            "function": function,
            "records": records,
            "frozen_sha256": digest(folder / "learner_frozen.json"),
            "seconds": time.perf_counter() - started,
        },
    )
    print(
        json.dumps(
            {
                "family": family,
                "function": function,
                "complete": True,
                "seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    if args.family not in plan["families"]:
        raise ValueError("Family must be registered")
    for function in plan["functions"]:
        run_function(args.root, plan, args.family, function, args.device)


if __name__ == "__main__":
    main()
