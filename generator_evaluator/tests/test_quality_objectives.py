import unittest
import torch

from generator_evaluator.config import CooperativeConfig
from generator_evaluator.quality_objectives import quality_objective_cost


class QualityObjectiveTests(unittest.TestCase):
    def test_mean_positive_worst_changes_candidate_ranking_and_keeps_gradients(self):
        delta = torch.tensor([[-1.0, 0.4], [-0.05, 0.2]], requires_grad=True)

        legacy = quality_objective_cost(delta, task_dim=1)
        combined = quality_objective_cost(delta, "mean_positive_worst", task_dim=1)

        torch.testing.assert_close(legacy, torch.tensor([0.4, 0.2]))
        torch.testing.assert_close(combined, torch.tensor([0.1, 0.275]))
        self.assertEqual(torch.argsort(legacy).tolist(), [1, 0])
        self.assertEqual(torch.argsort(combined).tolist(), [0, 1])

        combined.sum().backward()
        torch.testing.assert_close(delta.grad, torch.tensor([[0.5, 1.5], [0.5, 1.5]]))

    def test_objective_supports_arbitrary_task_axis_and_preserves_other_dimensions(self):
        delta = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        cost = quality_objective_cost(delta, "mean_positive_worst", task_dim=1)

        self.assertEqual(cost.shape, (2, 4))
        expected = delta.mean(dim=1) + delta.max(dim=1).values
        torch.testing.assert_close(cost, expected)

    def test_unknown_quality_objectives_are_rejected(self):
        for value in ("mean", "", None, 1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "quality_objective"):
                CooperativeConfig(quality_objective=value)

    def test_invalid_task_deltas_and_axes_are_rejected(self):
        cases = (
            (torch.tensor([1.0, float("nan")]), -1),
            (torch.tensor([1.0, float("inf")]), -1),
            (torch.tensor([1.0 + 2.0j]), -1),
            (torch.empty((2, 0)), 1),
            (torch.ones(2), 1),
        )
        for delta, task_dim in cases:
            with self.subTest(delta=delta, task_dim=task_dim), self.assertRaises((TypeError, ValueError)):
                quality_objective_cost(delta, task_dim=task_dim)


if __name__ == "__main__":
    unittest.main()
