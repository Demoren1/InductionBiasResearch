"""Unit tests for assignments, partition metrics, and synthetic data."""

from __future__ import annotations

import unittest

import torch

from yeh2022_generated_sharing.core import (
    CoordinateAssignmentGenerator,
    assignment_from_logits,
    partition_distance,
    solve_shared_mean,
)
from yeh2022_generated_sharing.tasks import (
    CrossCorrelationSpec,
    SumNumbersSpec,
    UnitStepDenoisingSpec,
    make_cross_correlation,
    make_sum_numbers,
    make_unit_step_denoising,
)


class CoreAndTasksTest(unittest.TestCase):
    def test_assignment_modes_and_ste_gradient(self) -> None:
        logits = torch.randn(5, 4, requires_grad=True)
        soft = assignment_from_logits(logits, mode="soft")
        hard = assignment_from_logits(logits, mode="hard")
        ste = assignment_from_logits(logits, mode="ste")
        self.assertTrue(torch.allclose(soft.sum(-1), torch.ones(5)))
        self.assertTrue(torch.allclose(hard, ste.detach()))
        self.assertTrue(torch.all((hard == 0) | (hard == 1)))
        (ste * torch.arange(4)).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_coordinate_generator_accepts_matrix_coordinates(self) -> None:
        coordinates = torch.randn(6, 2)
        generator = CoordinateAssignmentGenerator(
            6, latent_dim=3, width=8, item_coordinates=coordinates
        )
        assignment = generator.assignment(torch.randn(2, 3), mode="hard")
        self.assertEqual(tuple(assignment.shape), (2, 6, 6))
        self.assertTrue(torch.all(assignment.sum(-1) == 1))

    def test_partition_distance_ignores_group_names(self) -> None:
        first = torch.tensor([0, 0, 1, 1, 2])
        renamed = torch.tensor([7, 7, 3, 3, 5])
        different = torch.tensor([7, 3, 7, 3, 5])
        self.assertEqual(partition_distance(first, renamed), 0)
        self.assertGreater(partition_distance(first, different), 0)

    def test_shared_mean_is_cluster_average(self) -> None:
        samples = torch.tensor([[1.0, 3.0, 9.0], [3.0, 5.0, 11.0]])
        labels = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        estimate = solve_shared_mean(samples, labels, ridge=0.0)
        self.assertTrue(torch.allclose(estimate, torch.tensor([3.0, 3.0, 10.0])))

    def test_sum_data_and_oracle(self) -> None:
        standard = make_sum_numbers(
            SumNumbersSpec(sequence_length=4, train_size=3, validation_size=4, test_size=5),
            seed=9,
        )
        alternating = make_sum_numbers(
            SumNumbersSpec(
                sequence_length=4,
                train_size=3,
                validation_size=4,
                test_size=5,
                alternating=True,
            ),
            seed=9,
        )
        self.assertEqual(tuple(standard.splits.train.x.shape), (3, 4))
        self.assertEqual(standard.oracle_categories.unique().numel(), 1)
        self.assertEqual(alternating.oracle_categories.unique().numel(), 2)
        self.assertTrue(
            torch.allclose(
                alternating.splits.test.y,
                alternating.splits.test.x @ torch.tensor([1.0, -1.0, 1.0, -1.0]),
            )
        )

    def test_cross_correlation_oracle_generates_targets(self) -> None:
        benchmark = make_cross_correlation(
            CrossCorrelationSpec(
                input_length=5,
                kernel_length=3,
                train_size=3,
                validation_size=4,
                test_size=5,
            ),
            seed=4,
        )
        self.assertEqual(tuple(benchmark.oracle_weight.shape), (3, 5))
        expected = benchmark.splits.test.x @ benchmark.oracle_weight.T
        self.assertTrue(torch.allclose(benchmark.splits.test.y, expected))
        self.assertEqual(int((benchmark.oracle_categories > 0).sum()), 9)

    def test_denoising_shapes(self) -> None:
        benchmark = make_unit_step_denoising(
            UnitStepDenoisingSpec(
                signal_length=6,
                train_size=3,
                validation_size=4,
                test_size=5,
            ),
            seed=4,
        )
        self.assertEqual(tuple(benchmark.splits.train.x.shape), (3, 6))
        self.assertEqual(tuple(benchmark.splits.test.y.shape), (5, 6))
        self.assertEqual(benchmark.oracle_categories.unique().numel(), 11)


if __name__ == "__main__":
    unittest.main()
