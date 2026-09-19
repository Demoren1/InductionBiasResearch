"""Small end-to-end tests for analytic-lower benchmark runners."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from yeh2022_generated_sharing.gaussian import GaussianConfig, run_gaussian_benchmark
from yeh2022_generated_sharing.gaussian import _release_shared_mean
from yeh2022_generated_sharing.linear import (
    LinearExperimentConfig,
    fit_shared_linear,
    fit_paper_projection,
    labels_to_assignment,
    run_linear_experiment,
)
from yeh2022_generated_sharing.multitask_linear import run_multitask_generated_linear
from yeh2022_generated_sharing.sum_numbers import (
    SumNumbersConfig,
    _neumann_inverse_hvp,
    run_sum_numbers_benchmark,
)
from yeh2022_generated_sharing.tasks import CrossCorrelationSpec, make_cross_correlation
from yeh2022_generated_sharing.tasks import Split


class ExperimentSmokeTest(unittest.TestCase):
    def test_gaussian_soft_lower_matches_weighted_group_means(self) -> None:
        samples = torch.tensor([[[1.0, 5.0], [3.0, 7.0]]])
        assignment = torch.tensor([[[0.75, 0.25], [0.25, 0.75]]])
        empirical = samples.mean(dim=1)
        category = (
            assignment.transpose(-2, -1) @ empirical.unsqueeze(-1)
        ).squeeze(-1) / assignment.sum(dim=-2)
        expected = (assignment @ category.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(_release_shared_mean(samples, assignment), expected)

    def test_exact_shared_linear_lower(self) -> None:
        generator = torch.Generator().manual_seed(9)
        x = torch.randn(12, 3, generator=generator)
        y = torch.randn(12, 2, generator=generator)
        split = Split(x=x, y=y)
        labels = torch.tensor([0, 1, 2, 1, 0, 2])
        assignment = labels_to_assignment(labels, n_categories=4)
        weight = fit_shared_linear(split, assignment, ridge=0.0)
        structured = assignment.reshape(2, 3, -1)
        design = torch.einsum("ni,oic->noc", x, structured).reshape(-1, 4)
        self.assertTrue(torch.all(design[:, 3] == 0))
        active_design = design[:, :3]
        values = torch.linalg.solve(
            active_design.T @ active_design,
            active_design.T @ y.reshape(-1, 1),
        )
        expected = (assignment[:, :3] @ values).reshape(2, 3)
        torch.testing.assert_close(weight, expected, atol=1e-5, rtol=1e-5)

    def test_paper_projection_matches_dense_then_pinv(self) -> None:
        generator = torch.Generator().manual_seed(17)
        split = Split(
            x=torch.randn(10, 3, generator=generator),
            y=torch.randn(10, 2, generator=generator),
        )
        assignment = labels_to_assignment(torch.tensor([0, 1, 2, 0, 1, 2]), n_categories=6)
        projected = fit_paper_projection(split, assignment, ridge=0.0)
        dense = torch.linalg.lstsq(split.x, split.y).solution.T
        expected = assignment @ (torch.linalg.pinv(assignment) @ dense.reshape(-1, 1))
        torch.testing.assert_close(projected, expected.reshape_as(dense), atol=1e-5, rtol=1e-5)

    def test_neumann_inverse_hvp_has_alpha_scale(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        lower_loss = 0.5 * parameter.square()
        train_gradient = (torch.autograd.grad(lower_loss, parameter, create_graph=True)[0],)
        estimate = _neumann_inverse_hvp(
            train_gradient,
            [parameter],
            (torch.tensor(1.0),),
            alpha=0.25,
            iterations=3,
        )[0]
        expected = 0.25 * sum(0.75**power for power in range(4))
        torch.testing.assert_close(estimate, torch.tensor(expected))

    def test_gaussian_smoke(self) -> None:
        config = GaussianConfig.quick(
            seed=3, runs=3, dimensions=4, true_rank=2, epochs=2, log_every=1
        )
        summary, artifacts = run_gaussian_benchmark(config, "cpu")
        self.assertEqual(set(summary["methods"]), {"no_sharing", "oracle", "direct", "generated"})
        self.assertIn("assignment_gt", artifacts)

    def test_cross_correlation_smoke_writes_artifacts(self) -> None:
        benchmark = make_cross_correlation(
            CrossCorrelationSpec(
                input_length=4,
                kernel_length=2,
                train_size=6,
                validation_size=6,
                test_size=8,
            ),
            seed=3,
        )
        config = LinearExperimentConfig(
            seed=3,
            outer_steps=2,
            checkpoint_every=1,
            patience=2,
            generator_width=8,
            latent_dim=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            summary = run_linear_experiment(benchmark, config, output)
            self.assertEqual(set(summary["methods"]), {"no_sharing", "oracle", "direct", "generated"})
            for filename in ("summary.json", "training.pt", "RESULTS.md"):
                self.assertTrue((output / filename).is_file())

    def test_multitask_global_latent_smoke(self) -> None:
        tasks = [
            make_cross_correlation(
                CrossCorrelationSpec(
                    input_length=3,
                    kernel_length=2,
                    train_size=6,
                    validation_size=6,
                    test_size=8,
                ),
                seed=seed,
            )
            for seed in (1, 2)
        ]
        config = LinearExperimentConfig(
            outer_steps=2,
            checkpoint_every=1,
            generator_width=8,
            latent_dim=3,
            restarts=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = run_multitask_generated_linear(
                tasks,
                config,
                Path(directory) / "run",
                shared_latent=True,
            )
            self.assertEqual(summary["latent_mode"], "global")
            self.assertEqual(summary["task_count"], 2)
            self.assertIn("generated", summary["methods"])

    def test_sum_numbers_smoke(self) -> None:
        config = SumNumbersConfig.quick(
            seed=3,
            outer_steps=1,
            inner_steps=1,
            neumann_iterations=1,
            refit_steps=2,
        )
        summary, artifacts = run_sum_numbers_benchmark(
            config,
            "cpu",
            alternating=True,
            train_size=6,
            validation_size=6,
            test_size=8,
        )
        self.assertEqual(set(summary["methods"]), {"no_sharing", "oracle", "direct", "generated"})
        self.assertEqual(summary["methods"]["oracle"]["partition_distance"], 0)
        self.assertIn("methods", artifacts)


if __name__ == "__main__":
    unittest.main()
