"""Contracts for data-derived functional-consensus control proposals."""
from __future__ import annotations

import unittest

import torch

from generator_evaluator.functional_consensus import build_functional_consensus_proposals


def bank(aligned_q_abs: torch.Tensor) -> dict:
    return {"diagnostics": {"aligned_q_abs": aligned_q_abs}}


class FunctionalConsensusTests(unittest.TestCase):
    def test_exact_k_and_balanced_remainder_uses_next_best_column_entries(self) -> None:
        values = torch.tensor([[
            [9., 10., 1.],
            [5., 9., 7.],
            [4., 1., 2.],
            [0., 0., 0.],
        ]])

        proposals = build_functional_consensus_proposals([bank(values)], k=5)

        self.assertEqual(int(proposals.global_topk.sum()), 5)
        self.assertEqual(int(proposals.balanced_per_column.sum()), 5)
        self.assertEqual(proposals.balanced_per_column.sum(0).tolist(), [2., 2., 1.])
        self.assertEqual(proposals.balanced_per_column[1, 0].item(), 1.)
        self.assertEqual(proposals.balanced_per_column[1, 1].item(), 1.)
        self.assertEqual(proposals.balanced_per_column[0, 2].item(), 0.)

    def test_column_scaling_and_teacher_count_do_not_change_equal_bank_weight(self) -> None:
        first_map = torch.tensor([[1., 2.], [3., 1.], [0., 2.]])
        second_map = torch.tensor([[0., 4.], [2., 0.], [1., 2.]])
        first = bank(first_map[None].requires_grad_())
        second = bank(second_map[None])
        expected = build_functional_consensus_proposals([first, second], k=3)

        scaled_repeats = torch.stack([
            first_map * torch.tensor(scales)[None, :]
            for scales in ([3., .25], [.5, 8.], [11., 2.], [.125, 5.])
        ])
        changed = build_functional_consensus_proposals(
            [bank(scaled_repeats), second], k=3)

        torch.testing.assert_close(changed.importance, expected.importance, rtol=0, atol=0)
        self.assertFalse(expected.importance.requires_grad)
        self.assertEqual(expected.importance.device.type, "cpu")
        self.assertEqual(expected.global_topk.device.type, "cpu")
        self.assertEqual(expected.balanced_per_column.device.type, "cpu")
        self.assertEqual(expected.global_topk.dtype, torch.float32)

    def test_zero_columns_and_exact_ties_are_finite_and_deterministic(self) -> None:
        zero_maps = bank(torch.zeros(2, 3, 2))
        first = build_functional_consensus_proposals([zero_maps], k=3)
        second = build_functional_consensus_proposals([zero_maps], k=3)
        expected = torch.tensor([[1., 1.], [1., 0.], [0., 0.]])

        self.assertTrue(torch.isfinite(first.importance).all())
        torch.testing.assert_close(first.importance, torch.zeros(3, 2, dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(first.global_topk, expected, rtol=0, atol=0)
        torch.testing.assert_close(first.balanced_per_column, expected, rtol=0, atol=0)
        torch.testing.assert_close(first.global_topk, second.global_topk, rtol=0, atol=0)
        torch.testing.assert_close(first.balanced_per_column, second.balanced_per_column, rtol=0, atol=0)

        one_active_column = bank(torch.tensor([[[0., 1.], [0., 3.], [0., 0.]]]))
        mixed = build_functional_consensus_proposals([one_active_column], k=2)
        self.assertTrue(torch.isfinite(mixed.importance).all())
        self.assertTrue(torch.equal(mixed.importance[:, 0], torch.zeros(3, dtype=torch.float64)))
        self.assertEqual(int(mixed.global_topk.sum()), 2)
        self.assertEqual(int(mixed.balanced_per_column.sum()), 2)

    def test_arbitrary_planted_support_is_recovered_without_spatial_template(self) -> None:
        values = torch.zeros(1, 4, 4)
        values[0, 0, 0], values[0, 3, 0] = 9., 1.
        values[0, 2, 1], values[0, 0, 1] = 9., 1.
        values[0, 1, 2], values[0, 3, 2] = 9., 1.
        values[0, 3, 3], values[0, 0, 3], values[0, 1, 3] = 9., 8., 1.
        expected = torch.zeros(4, 4)
        expected[0, 0] = expected[2, 1] = expected[1, 2] = 1.
        expected[3, 3] = expected[0, 3] = 1.

        proposals = build_functional_consensus_proposals([bank(values)], k=5)

        torch.testing.assert_close(proposals.global_topk, expected, rtol=0, atol=0)
        torch.testing.assert_close(proposals.balanced_per_column, expected, rtol=0, atol=0)
        self.assertEqual(int(proposals.global_topk.sum()), 5)
        self.assertEqual(int(proposals.balanced_per_column.sum()), 5)

    def test_invalid_maps_and_cardinalities_are_rejected(self) -> None:
        bad_inputs = (
            torch.zeros(3, 2),
            torch.tensor([[[1., -1.], [0., 1.]]]),
            torch.tensor([[[float("nan"), 0.], [0., 1.]]]),
            torch.tensor([[[True, False], [False, True]]]),
            torch.tensor([[[1 + 1j, 0j], [0j, 1 + 0j]]]),
        )
        for values in bad_inputs:
            with self.subTest(dtype=values.dtype, shape=tuple(values.shape)):
                with self.assertRaises(ValueError):
                    build_functional_consensus_proposals([bank(values)], k=1)

        with self.assertRaisesRegex(ValueError, "compatible"):
            build_functional_consensus_proposals(
                [bank(torch.zeros(1, 3, 2)), bank(torch.zeros(1, 4, 2))], k=1)
        with self.assertRaisesRegex(ValueError, "between 0"):
            build_functional_consensus_proposals([bank(torch.zeros(1, 3, 2))], k=7)
        with self.assertRaisesRegex(ValueError, "integer"):
            build_functional_consensus_proposals([bank(torch.zeros(1, 3, 2))], k=1.5)


if __name__ == "__main__":
    unittest.main()
