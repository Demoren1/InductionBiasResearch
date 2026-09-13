from __future__ import annotations

import unittest

import torch

from evaluation.hard_mask_sampling import fit_and_score, sample_candidates


class HardMaskSamplingTests(unittest.TestCase):
    def test_candidates_retain_parent_and_respect_radius(self):
        parents = torch.randn(3, 32)
        parents *= 4 / parents.norm(dim=1, keepdim=True)
        values = sample_candidates(parents, (.25, .5, 1.), 4., 123)
        self.assertTrue(torch.equal(values[:, 0], parents))
        self.assertLessEqual(float(values.norm(dim=2).max()), 4.000001)

    def test_paired_identical_masks_have_identical_scores(self):
        base = torch.zeros(2, 8, 8)
        base.reshape(2, -1)[:, :32] = 1
        masks = base[:, None].repeat(1, 3, 1, 1)
        result = fit_and_score(
            masks, "0100", torch.device("cpu"), steps=2,
            model_seed=7, train_seed_base=123000, eval_seed=456000,
        )["bce"]
        self.assertLessEqual(float((result - result[:, :1]).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
