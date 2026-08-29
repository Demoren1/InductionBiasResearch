"""Regression tests for prior sampling."""

import unittest

import torch

import config
from models.cvae import CVAE
from models.mlp import generate_fixed_sparsity_masks


class CVAESamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = CVAE(config.MASK_DIM, latent_dim=4, hidden=16)
        self.pattern = config.pattern_to_pm1("0000").view(1, -1)

    def test_prior_inputs_draw_one_latent_per_sample(self) -> None:
        generator = torch.Generator().manual_seed(11)
        z, condition = self.model.prior_inputs(self.pattern, 5, generator)
        self.assertEqual(z.shape, (5, 4))
        self.assertEqual(condition.shape, (5, 0))
        self.assertEqual(torch.unique(z, dim=0).size(0), 5)

    def test_topk_samples_have_fixed_sparsity(self) -> None:
        masks = self.model.sample_topk(self.pattern, 7, k_active=11,
                                       generator=torch.Generator().manual_seed(13))
        self.assertEqual(masks.shape, (7, config.MASK_DIM))
        self.assertTrue(torch.equal(masks.sum(dim=1), torch.full((7,), 11.0)))

    def test_fixed_sparsity_random_masks_have_exact_cardinality(self) -> None:
        masks = generate_fixed_sparsity_masks(64, 8, 8, 32, seed=23)
        self.assertTrue(torch.equal(masks.sum(dim=(1, 2)), torch.full((64,), 32.0)))
        self.assertGreater(torch.unique(masks.reshape(64, -1), dim=0).size(0), 1)


if __name__ == "__main__":
    unittest.main()
