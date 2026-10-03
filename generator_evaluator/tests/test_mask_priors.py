"""Structural pattern baseline and its task-independent topology contract."""
import itertools
import unittest

import torch

from generator_evaluator.adapters import _pattern_labels, _pattern_table
from generator_evaluator.mask_priors import SlidingWindowMaskPrior
from pattern.task_quality.meta import child_logits


class MaskPriorTests(unittest.TestCase):
    def test_rectangular_toeplitz_has_exact_windows_and_permutation_safe_diagnostics(self):
        prior = SlidingWindowMaskPrior()
        mask = prior.mask()
        self.assertEqual(tuple(mask.shape), (11, 8))
        self.assertEqual(int(mask.sum()), 32)
        for column in range(8):
            self.assertEqual(mask[:, column].nonzero().flatten().tolist(), list(range(column, column + 4)))
        permuted = mask[:, torch.tensor([5, 0, 7, 2, 4, 1, 3, 6])]
        self.assertTrue(prior.matches(permuted))
        self.assertEqual(prior.diagnostics(permuted)["differing_edges"], 0)
        self.assertEqual(prior.diagnostics(permuted)["toeplitz_score"], 1.)
        changed = mask.clone()
        changed[0, 0] = 0
        changed[10, 0] = 1
        self.assertFalse(prior.matches(changed))
        self.assertEqual(prior.diagnostics(changed)["differing_edges"], 2)
        values = prior.diagnostics(changed)
        self.assertEqual(values["matched_edges"], 31)
        self.assertEqual(values["missing_edges"], 1)
        self.assertEqual(values["extra_edges"], 1)
        self.assertEqual(values["toeplitz_score"], 31 / 32)
        torch.testing.assert_close(prior.aligned_mask(permuted), mask)

    def test_dense_and_empty_masks_do_not_get_perfect_active_edge_scores(self):
        prior = SlidingWindowMaskPrior()
        dense = prior.diagnostics(torch.ones(11, 8))
        self.assertEqual(dense["toeplitz_score"], 64 / 120)
        self.assertEqual(dense["missing_edges"], 0)
        self.assertEqual(dense["extra_edges"], 56)
        self.assertEqual(prior.diagnostics(torch.zeros(11, 8))["toeplitz_score"], 0.)

    def test_window_topology_can_represent_every_four_bit_pattern_exactly(self):
        # Constructive expressivity proof only. These hand-coded weights are
        # never used by the real-fit baseline or generator training.
        prior = SlidingWindowMaskPrior()
        mask = prior.mask()
        _, x = _pattern_table()
        for bits in itertools.product("01", repeat=4):
            pattern = "".join(bits)
            signs = torch.tensor([2 * int(bit) - 1 for bit in bits]).float()
            weight = torch.zeros_like(mask)
            for column in range(8):
                weight[column:column + 4, column] = signs
            params = dict(w=weight, b=torch.full((8,), -3.),
                          a=torch.full((8,), 20.), c=torch.tensor(-10.))
            prediction = (child_logits(x, mask, params) > 0).float()
            torch.testing.assert_close(prediction, _pattern_labels(x, pattern), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
