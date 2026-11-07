"""Geometric diagnostics, independent of generator and optimizer selection."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import betainc

from experiments.basin_structure_transfer_v1.core import (
    make_tasks,
    population_loss,
)

ROOT = Path(__file__).resolve().parent


def toy_geometry(plan, device):
    """Matched-radius basin comparison for an explicitly artificial loss."""
    height, broad, narrow = 1.0, 0.05, 4.0
    target_norm, start_norm = np.sqrt(2.0), 2.5
    center_norm = narrow * target_norm / (narrow - broad)
    radius = np.sqrt(
        narrow * broad * target_norm**2 + height * (narrow - broad)
    ) / (narrow - broad)
    sine_squared = (radius / center_norm) ** 2
    cosine = np.sqrt(1 - sine_squared)
    samples, steps, rate = 100000, 2048, 0.1
    result = {
        "loss": "min(1 + 0.05 ||w||^2, 4 ||w-v||^2)",
        "target_norm": target_norm,
        "initial_radius": start_norm,
        "narrow_region_center_norm": center_norm,
        "narrow_region_radius": radius,
        "sin_squared_cap_angle": sine_squared,
        "initial_sphere_outside_narrow_ball": bool(
            start_norm > center_norm + radius
        ),
        "samples_per_dimension": samples,
        "gd_steps": steps,
        "gd_lr": rate,
        "scope": "known correct subspace; no learned-generator claim",
        "dimensions": {},
    }
    for dimension in (plan["groups"], plan["inputs"]):
        torch.manual_seed(plan["seed"] + dimension)
        weights = torch.randn(samples, dimension, device=device)
        weights *= start_norm / weights.norm(dim=1, keepdim=True)
        target = torch.zeros(dimension, device=device)
        target[0] = target_norm
        exact_hits = (weights[:, 0] / start_norm > cosine).sum()
        for _ in range(steps):
            narrow_loss = narrow * (weights - target).square().sum(1)
            broad_loss = height + broad * weights.square().sum(1)
            gradient = torch.where(
                (narrow_loss < broad_loss)[:, None],
                2 * narrow * (weights - target),
                2 * broad * weights,
            )
            weights -= rate * gradient
        error = narrow * (weights - target).square().sum(1)
        result["dimensions"][dimension] = {
            "continuous_flow_probability": float(
                0.5 * betainc((dimension - 1) / 2, 0.5, sine_squared)
            ),
            "sampled_continuous_flow_hits": int(exact_hits),
            "discrete_gd_hits": int((error < 1e-6).sum()),
            "broad_minimum_hits": int((weights.norm(dim=1) < 1e-6).sum()),
        }
    return result


def stationary_points(spec, device):
    """Bracket profiled stationary points, then inspect the full Hessian.

    Unconstrained refinement can jump from the intended low-frequency well
    to the good well. Bracketing the derivative avoids that mislabelling.
    """
    records = []
    for task in range(len(spec["families"])):
        one = {
            key: value[task : task + 1].double()
            for key, value in spec.items()
            if isinstance(value, torch.Tensor)
        }
        direction = one["direction"][0]
        direction_norm = float(direction.norm())
        direction = direction / direction.norm()
        record = {"task": task, "family": int(spec["families"][task])}
        low = float(one["low"][0]) * direction_norm
        high = float(one["high"][0]) * direction_norm
        alpha = float(one["alpha"][0])

        def profile(value):
            variance = -0.5 * np.expm1(-2 * value**2)
            correlation, derivative = 0.0, 0.0
            for coefficient, frequency in ((alpha, low), (1.0, high)):
                difference = np.exp(-0.5 * (value - frequency) ** 2)
                total = np.exp(-0.5 * (value + frequency) ** 2)
                correlation += coefficient * 0.5 * (difference - total)
                derivative += (
                    coefficient
                    * 0.5
                    * (
                        -(value - frequency) * difference
                        + (value + frequency) * total
                    )
                )
            amplitude = correlation / variance
            gradient = (
                amplitude**2 * 2 * value * np.exp(-2 * value**2)
                - 2 * amplitude * derivative
            )
            return gradient, amplitude

        grid = np.linspace(0.03, high + 2, 4001)
        gradients, _ = profile(grid)
        intervals = np.flatnonzero((gradients[:-1] < 0) & (gradients[1:] > 0))
        roots = [
            brentq(
                lambda value: profile(value)[0],
                grid[i],
                grid[i + 1],
                xtol=1e-14,
            )
            for i in intervals
        ]
        bad = [value for value in roots if value < (low + high) / 2]
        record["profile_minimum_frequencies"] = roots
        record["bad"] = None
        candidates = [
            ("good", min(roots, key=lambda value: abs(value - high)))
        ]
        if bad:
            candidates.append(
                ("bad", min(bad, key=lambda value: abs(value - low)))
            )
        for name, frequency in candidates:
            amplitude = profile(frequency)[1]
            full = torch.cat(
                (
                    frequency * direction,
                    torch.tensor(
                        [amplitude], device=device, dtype=torch.float64
                    ),
                )
            ).requires_grad_()

            def objective(parameters):
                return population_loss(
                    parameters[:-1][None, None],
                    parameters[-1][None, None],
                    one,
                ).sum()

            loss = objective(full)
            gradient = torch.autograd.grad(loss, full)[0]
            hessian = torch.autograd.functional.hessian(objective, full)
            eigenvalues = torch.linalg.eigvalsh(hessian)
            record[name] = {
                "population_mse": float(loss),
                "gradient_norm": float(gradient.norm()),
                "min_hessian_eigenvalue": float(eigenvalues.min()),
                "max_hessian_eigenvalue": float(eigenvalues.max()),
                "hessian_geometric_mean": float(
                    eigenvalues.log().mean().exp()
                ),
                "weights": full[:-1].detach().cpu().tolist(),
                "amplitude": float(full[-1]),
            }
        records.append(record)
    return records


def endpoint_diagnostics(plan, output, spec, geometry, device):
    """Separate proximity to a bad minimum from other failed endpoints."""
    bad_weights = torch.tensor(
        [
            item["bad"]["weights"]
            if item["bad"]
            else [np.nan] * plan["inputs"]
            for item in geometry
        ],
        device=device,
        dtype=torch.float64,
    )
    bad_amplitude = torch.tensor(
        [
            item["bad"]["amplitude"] if item["bad"] else np.nan
            for item in geometry
        ],
        device=device,
        dtype=torch.float64,
    )
    verified = torch.tensor(
        [item["bad"] is not None for item in geometry], device=device
    )
    spec = {
        key: value.double() if isinstance(value, torch.Tensor) else value
        for key, value in spec.items()
    }
    summaries = {}
    for path in sorted((output / "evaluation").glob("*/models.npz")):
        saved = np.load(path)
        weights = torch.tensor(
            saved["final_weights"][:, 0],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        amplitude = torch.tensor(
            saved["final_amplitude"][:, 0],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        loss = population_loss(weights, amplitude, spec)
        gradients = torch.autograd.grad(loss.sum(), (weights, amplitude))
        grad_norm = (
            gradients[0].square().sum(2) + gradients[1].square()
        ).sqrt()
        plus = (weights - bad_weights[:, None]).square().sum(2) + (
            amplitude - bad_amplitude[:, None]
        ).square()
        minus = (weights + bad_weights[:, None]).square().sum(2) + (
            amplitude + bad_amplitude[:, None]
        ).square()
        distance = torch.minimum(plus, minus).sqrt()
        known = torch.tensor(
            spec["families"] < plan["families"], device=device
        )
        near = (distance < 0.2) & (grad_norm < 0.02)
        failure = loss >= plan["success_mse"]
        summaries[path.parent.name] = {
            "known_final_attempts": int(loss[known].numel()),
            "known_tasks_with_bad_stationary_point": int(
                (known & verified).sum()
            ),
            "known_final_failures": int(failure[known].sum()),
            "known_final_failures_near_bad_minimum": int(
                (near & failure)[known].sum()
            ),
            "criterion": (
                "distance < 0.2 (joint w,a, allowing sign flip), grad < 0.02"
            ),
            "median_final_gradient_norm": float(grad_norm[known].median()),
            "median_final_distance_to_bad": float(
                distance[known & verified].median()
            ),
        }
        np.savez_compressed(
            path.parent / "endpoint_geometry.npz",
            population=loss.detach().cpu().numpy(),
            gradient_norm=grad_norm.detach().cpu().numpy(),
            distance_to_bad=distance.detach().cpu().numpy(),
        )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=int, choices=(4, 6, 8), default=4)
    args = parser.parse_args()
    plan = json.loads((ROOT / "protocol.json").read_text())
    output = Path(plan["output"])
    if not torch.cuda.is_available():
        raise RuntimeError("Optimization diagnostics require CUDA")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    if args.frequency == 4:
        spec = make_tasks(plan, "evaluation", device)
    else:
        from experiments.basin_structure_transfer_v1.frequency_sweep import (
            new_tasks,
        )

        extension = json.loads((ROOT / "frequency_sweep.json").read_text())
        output = Path(extension["output"]) / f"frequency_{args.frequency}"
        spec = new_tasks(plan, extension, args.frequency, "evaluation", device)
    if not (output / "evaluation" / "complete.json").exists():
        raise RuntimeError("Run frozen evaluation before target diagnostics")
    geometry = stationary_points(spec, device)
    existing = output / "geometry_audit.json"
    if args.frequency == 4 and existing.exists():
        toy = json.loads(existing.read_text())["toy"]
    else:
        toy = toy_geometry(plan, device) if args.frequency == 4 else None
    report = {
        "toy": toy,
        "stationary_points": geometry,
        "endpoints": endpoint_diagnostics(
            plan, output, spec, geometry, device
        ),
    }
    (output / "geometry_audit.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps({"toy": report["toy"], "endpoints": report["endpoints"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
