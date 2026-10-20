"""Certify minima of actual empirical squared error and inspect their geometry."""

from pathlib import Path

import numpy as np
import torch

from core import population_loss, save_json, task, teacher_vectors


def main():
    torch.set_num_threads(1)
    torch.manual_seed(91901)
    destination = Path("tmp/sine_mask_vae")
    spec = task(0, 0)
    low, high, small = teacher_vectors(spec, "cpu", torch.float64)
    x = torch.randn(16384, 8, dtype=torch.float64)
    y = small * torch.sin(x @ low) + torch.sin(x @ high)

    def empirical(vector):
        w, amp = vector[:16].reshape(8, 2), vector[16:]
        return (torch.sin(x @ w) * amp - y).square().mean()

    records, solutions = {}, {}
    for name, w, amp in (
        ("broad_low_frequency", low, small),
        ("good_high_frequency", high, torch.ones_like(small)),
    ):
        vector = torch.cat((w.flatten(), amp)).clone().requires_grad_()
        optimizer = torch.optim.LBFGS(
            [vector],
            max_iter=500,
            tolerance_grad=1e-11,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = empirical(vector)
            loss.backward()
            return loss

        optimizer.step(closure)
        loss = empirical(vector)
        gradient = torch.autograd.grad(loss, vector)[0]
        hessian = torch.autograd.functional.hessian(empirical, vector)
        eigenvalues = torch.linalg.eigvalsh(hessian).detach().numpy()
        records[name] = {
            "empirical_mse": float(loss.detach()),
            "population_mse": float(
                population_loss(
                    vector[:16].reshape(1, 8, 2), vector[16:][None], spec
                ).detach()
            ),
            "gradient_norm": float(gradient.norm()),
            "hessian_eigenvalues": eigenvalues.tolist(),
            "strict_local_minimum": bool(eigenvalues.min() > 0),
            "geometric_mean_curvature": float(
                np.exp(np.log(eigenvalues).mean())
            ),
            "parameters": vector.detach().tolist(),
        }
        solutions[name] = vector.detach().numpy()

    frequencies = np.linspace(0.03, 6, 500)
    # One-output slice: optimize its output amplitude for each input frequency.
    coordinate = torch.tensor(frequencies, dtype=torch.float64)
    activation = torch.sin(x[:, spec["rows"][0], None] * coordinate)
    amplitudes = (activation * y[:, :1]).mean(0) / activation.square().mean(0)
    profile = (activation * amplitudes - y[:, :1]).square().mean(0)
    np.savez_compressed(
        destination / "geometry.npz",
        frequencies=frequencies,
        profile=profile.numpy(),
        **solutions,
    )
    records["curvature_ratio_good_over_broad"] = (
        records["good_high_frequency"]["geometric_mean_curvature"]
        / records["broad_low_frequency"]["geometric_mean_curvature"]
    )
    records["sample_count"] = len(x)
    records["gate_passed"] = (
        all(records[name]["strict_local_minimum"] for name in solutions)
        and records["good_high_frequency"]["population_mse"] < 0.1
    )
    save_json(destination / "geometry.json", records)
    print(records, flush=True)


if __name__ == "__main__":
    main()
