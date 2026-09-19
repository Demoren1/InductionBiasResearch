"""Guard the two protocol choices that determine the no-z control."""

import unittest

import torch

from pattern.bilevel_mask.generated_sharing import SharingConfig
from scripts.ablate_generated_sharing_z import CoordinateGenerator
from scripts.no_z_multistart_control import select_by_validation


class NoZMultistartControlTest(unittest.TestCase):
    def test_initialization_seed_changes_the_generator(self) -> None:
        config = SharingConfig(seed=42)
        placeholder = torch.zeros(1, 1, config.latent_dim)
        first = CoordinateGenerator(config, placeholder, initialization_seed=42)
        second = CoordinateGenerator(config, placeholder, initialization_seed=43)
        self.assertTrue(any(
            not torch.equal(left, right)
            for left, right in zip(first.parameters(), second.parameters())
        ))

    def test_selection_uses_validation_even_when_test_disagrees(self) -> None:
        records = [
            {"restart": 0, "training": {"selected_training_validation_bce": 0.2},
             "test": {"mean_query_bce": 0.1}},
            {"restart": 1, "training": {"selected_training_validation_bce": 0.1},
             "test": {"mean_query_bce": 0.5}},
        ]
        self.assertEqual(select_by_validation(records)["restart"], 1)


if __name__ == "__main__":
    unittest.main()
