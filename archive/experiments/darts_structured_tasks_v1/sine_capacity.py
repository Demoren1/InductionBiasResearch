"""Separate representational error from numerical optimization after freezing."""

import json

import numpy as np
from scipy.optimize import minimize_scalar

from .data import BASE, save_json
from .evaluate import sine_population


def basis(indices):
    labels = np.unique(np.abs(indices[indices != 0]))
    vectors = np.stack(
        [
            (indices.ravel() == label).astype(float)
            - (indices.ravel() == -label).astype(float)
            for label in labels
        ],
        -1,
    )
    return vectors / np.linalg.norm(vectors, axis=0, keepdims=True)


def constrained_minimum(indices, direction, specification):
    assignment = basis(indices)
    projection = assignment @ (assignment.T @ direction)
    cosine = float(np.linalg.norm(projection))
    truth = {"direction": direction}
    energy = float(sine_population(np.zeros_like(direction), 0.0, truth, specification))
    low, high = specification["low_frequency"], specification["frequency"]
    alpha = specification["low_amplitude"]

    def objective(radius):
        radius = np.asarray(radius)
        correlation = (
            alpha * np.exp(-(radius**2 + low**2) / 2) * np.sinh(radius * low * cosine)
        )
        correlation += np.exp(-(radius**2 + high**2) / 2) * np.sinh(
            radius * high * cosine
        )
        norm = -0.5 * np.expm1(-2 * radius**2)
        return energy - correlation**2 / np.maximum(norm, 1e-30)

    grid = np.linspace(1e-5, high + 4, 1201)
    loss = objective(grid)
    candidates = np.flatnonzero((loss[1:-1] < loss[:-2]) & (loss[1:-1] < loss[2:])) + 1
    best, radius = float(loss.min()), float(grid[loss.argmin()])
    for point in candidates:
        optimized = minimize_scalar(
            objective,
            bounds=(grid[point - 1], grid[point + 1]),
            method="bounded",
            options={"xatol": 1e-12},
        )
        if optimized.fun < best:
            best, radius = float(optimized.fun), float(optimized.x)
    return {
        "population_mse": best,
        "weight_norm": radius,
        "direction_projection_cosine": cosine,
    }


def main():
    output = BASE / "results/20261104"
    results = {}
    for frequency in (4, 6, 8):
        parent = output / f"sine_{frequency}"
        spec = json.loads((parent / "specification.json").read_text())
        with np.load(parent / "teacher_audit_only.npz") as archive:
            truth = dict(archive)
        direction = truth["direction"].astype(float)
        direction /= np.linalg.norm(direction)
        true_basis = basis(truth["indices"])
        rng = np.random.default_rng(2611048943)
        new_directions = rng.normal(size=(32, true_basis.shape[1])) @ true_basis.T
        new_directions /= np.linalg.norm(new_directions, axis=1, keepdims=True)
        methods = {"oracle": truth["indices"]}
        for name, folder in (
            ("direct", parent),
            ("quantized", output / f"sine_{frequency}_quantized"),
        ):
            with np.load(folder / "selected_states.npz") as archive:
                methods[name] = archive["indices"]
        for name, indices in methods.items():
            source = constrained_minimum(indices, direction, spec)
            transfers = [
                constrained_minimum(indices, new, spec)["population_mse"]
                for new in new_directions
            ]
            overlap = np.linalg.svd(basis(indices).T @ true_basis, compute_uv=False)
            overlap = np.pad(overlap, (0, true_basis.shape[1] - len(overlap)))
            results[f"{frequency}/{name}"] = {
                "source_function_best_population_mse": source["population_mse"],
                "source_projection_cosine": source["direction_projection_cosine"],
                "principal_angle_cosines": overlap.tolist(),
                "new_codebooks_best_risk_mean": float(np.mean(transfers)),
                "new_codebooks_best_risk_under_point1": int(
                    (np.array(transfers) < 0.1).sum()
                ),
                "new_codebooks": 32,
                "new_codebooks_best_risks": transfers,
            }
    save_json(
        output / "sine_capacity_diagnosis.json",
        {
            "status": "post-freeze mathematical diagnostic; not training or hyperparameter selection",
            "assumptions": "Standard Gaussian X, positive low/high teacher amplitudes, collinear teacher frequencies, one sine unit with free amplitude and W constrained to signed group subspace. At fixed weight norm the target correlation increases with |W.v|, so the best direction is the normalized projection of v into that subspace. Optimize exact population risk over its norm and analytically eliminate amplitude.",
            "scope": "Numerical one-dimensional risk minimization, including all minima found on a1201-point grid and bounded refinements. New-codebook values are diagnostic best-achievable risks, not empirically trained transfer accuracies.",
            "true_direction_values": np.unique(direction).tolist(),
            "results": results,
        },
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
