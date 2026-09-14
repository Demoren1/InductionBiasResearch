"""Tests for vectorized length-32 joint structure search."""

import tempfile
import unittest
from pathlib import Path

import torch

from pattern.bilevel_mask.length32_joint import (
    Generator, Length32Config, Weights, analytic_mask, joint_search, make_data, masks, run,
)


class Length32JointTest(unittest.TestCase):
    def tiny_config(self, **changes):
        values = dict(seed=9, restarts=2, outer_steps=1, inner_max_steps=2, eval_inner_max_steps=2, inner_patience=1,
                      weight_steps_per_z=1, frozen_pretrain_steps=1, eval_weight_steps=1,
                      support_per_class=4, validation_per_class=3, query_per_class=3)
        values.update(changes)
        return Length32Config(**values)

    def test_data_and_analytic_cardinality(self) -> None:
        config = self.tiny_config()
        data = make_data(("0011", "1010"), config, torch.device("cpu"))
        self.assertEqual(tuple(data.support_x.shape), (2, 8, 32))
        self.assertEqual(tuple(data.validation_x.shape), (2, 6, 32))
        self.assertEqual(int(analytic_mask(config, torch.device("cpu")).sum()), 128)

    def test_vectorized_joint_search(self) -> None:
        config = self.tiny_config(inner_max_steps=3, inner_patience=2)
        generator = Generator(config)
        data = make_data(("0011", "1010"), config, torch.device("cpu"))
        z = torch.nn.Parameter(torch.randn(2, 2, config.latent_dim))
        weights = Weights(2, 2, config, torch.device("cpu"), seed=3)
        info = joint_search(generator, z, weights, data, config, 0.5)
        hard = masks(generator, z, config, 0.5, "hard")
        self.assertEqual(tuple(hard.shape), (2, 2, 32, 32))
        self.assertTrue(torch.all(hard.sum((-1, -2)) == 128))
        self.assertLessEqual(int(info["steps"]), 3)
        self.assertTrue(torch.isfinite(z).all())

    def test_smoke_run(self) -> None:
        config = self.tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            run(config, output, "cpu")
            self.assertTrue((output / "summary.json").exists())
            self.assertTrue((output / "RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()
