"""Inspect the expected MSE optimized by the streaming-data SGD experiment."""

import json

import torch

from .settings import CONTROL
from .core import population_loss, save_json, task


def main():
    torch.set_num_threads(1)
    empirical = json.loads((CONTROL / "geometry.json").read_text())
    spec = task(0, 0)

    def loss(vector):
        return population_loss(
            vector[:16].reshape(1, 8, 2), vector[16:][None], spec
        ).sum()

    records = {}
    for name in ("broad_low_frequency", "good_high_frequency"):
        vector = torch.tensor(
            empirical[name]["parameters"],
            dtype=torch.float64,
            requires_grad=True,
        )
        optimizer = torch.optim.LBFGS(
            [vector],
            max_iter=500,
            tolerance_grad=1e-12,
            tolerance_change=1e-15,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            value = loss(vector)
            value.backward()
            return value

        optimizer.step(closure)
        value = loss(vector)
        gradient = torch.autograd.grad(value, vector)[0]
        hessian = torch.autograd.functional.hessian(loss, vector)
        eigenvalues = torch.linalg.eigvalsh(hessian)
        records[name] = {
            "population_mse": float(value.detach()),
            "gradient_norm": float(gradient.norm()),
            "hessian_eigenvalues": eigenvalues.tolist(),
            "strict_local_minimum": bool(eigenvalues.min() > 0),
            "geometric_mean_curvature": float(eigenvalues.log().mean().exp()),
            "parameters": vector.detach().tolist(),
        }
    records["curvature_ratio_good_over_broad"] = (
        records["good_high_frequency"]["geometric_mean_curvature"]
        / records["broad_low_frequency"]["geometric_mean_curvature"]
    )
    save_json(CONTROL / "geometry_population.json", records)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
