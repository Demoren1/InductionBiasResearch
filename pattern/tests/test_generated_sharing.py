"""Tests for generated parameter-sharing schemes."""

import tempfile
import unittest
from pathlib import Path

import torch

from pattern.bilevel_mask.generated_sharing import (
    SharingConfig, SharingGenerator, adapt_weights, analytic_assignment, assignments,
    run, sharing_forward,
)
from pattern.bilevel_mask.length32_joint import make_data


class GeneratedSharingTest(unittest.TestCase):
    def tiny_config(self, **changes):
        values = dict(
            seed=7, train_restarts=2, eval_restarts=2, outer_steps=1, inner_steps=1,
            latent_search_steps=1, latent_patience=1, final_refit_steps=1,
            support_per_class=3, validation_per_class=2, query_per_class=2,
        )
        values.update(changes)
        return SharingConfig(**values)

    def test_exact_active_assignment_and_oracle(self) -> None:
        config = self.tiny_config()
        generator = SharingGenerator(config)
        z = torch.randn(2, 2, config.latent_dim)
        structure = assignments(generator, z, config, 0.5, "hard")
        self.assertEqual(tuple(structure.shape), (2, 2, 32, 29, 4))
        self.assertTrue(torch.all(structure.sum((-3, -2, -1)) == config.k_active))
        self.assertTrue(torch.all(structure.sum(-3) == 1))
        oracle = analytic_assignment(config, torch.device("cpu"))
        self.assertEqual(int(oracle.sum()), config.k_active)
        self.assertTrue(torch.all(oracle.sum(-1) <= 1))

    def test_hypergradient_reaches_generator(self) -> None:
        config = self.tiny_config()
        generator = SharingGenerator(config)
        z = torch.randn(1, 2, config.latent_dim, requires_grad=True)
        data = make_data(("0011",), config, torch.device("cpu"))
        structure = assignments(generator, z, config, 0.5, "ste")
        weights = adapt_weights(structure, data.support_x, data.support_y, config,
                                steps=1, seed=3, create_graph=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            sharing_forward(data.validation_x, weights, structure),
            data.validation_y[:, None].expand(-1, 2, -1),
        )
        gradients = torch.autograd.grad(loss, (z, *generator.parameters()), allow_unused=False)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)

    def test_smoke_run(self) -> None:
        config = self.tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            run(config, output, "cpu")
            self.assertTrue((output / "summary.json").exists())
            self.assertTrue((output / "RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()
