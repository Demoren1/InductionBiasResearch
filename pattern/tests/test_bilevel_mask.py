"""Tests for direct bilevel structure optimization."""

import math
import tempfile
import unittest
from pathlib import Path

import torch

from pattern.bilevel_mask.core import (
    BilevelMaskConfig,
    MaskStructure,
    _all_sequences,
    balanced_bce,
    task_loss,
    train_structure,
)
from pattern.bilevel_mask.latent_joint import (
    BatchedWeights,
    LatentJointConfig,
    LatentMaskGenerator,
    batched_forward,
    make_task_splits,
    optimize_z_to_convergence,
    relaxed_masks,
    train_latent_generator,
)


class BilevelMaskTest(unittest.TestCase):
    def test_exact_population_and_balanced_loss(self) -> None:
        x, y = _all_sequences("0101", torch.device("cpu"))
        self.assertEqual(tuple(x.shape), (256, 8))
        self.assertGreater(int(y.sum()), 0)
        self.assertLess(int(y.sum()), 256)
        self.assertAlmostEqual(float(balanced_bce(torch.zeros_like(y), y)), 0.693147, places=5)

    def test_relaxations_have_exact_cardinality_and_gradient(self) -> None:
        for relaxation in ("soft", "ste"):
            model = MaskStructure(seed=3)
            forward, soft = model.masks(0.5, 32, relaxation)
            self.assertAlmostEqual(float(soft.sum()), 32.0, places=4)
            if relaxation == "ste":
                self.assertTrue(torch.equal(forward.detach(), forward.detach().round()))
                self.assertEqual(int(forward.detach().sum()), 32)
            (forward.square().sum()).backward()
            self.assertIsNotNone(model.logits.grad)
            self.assertTrue(bool(torch.isfinite(model.logits.grad).all()))

    def test_full_hypergradient_reaches_mask(self) -> None:
        model = MaskStructure(seed=5).double()
        mask, _ = model.masks(0.7, 32, "soft")
        loss, _, _, _ = task_loss(
            "0010", mask, steps=2, lr=0.02, seed=9, create_graph=True,
        )
        loss.backward()
        self.assertIsNotNone(model.logits.grad)
        self.assertGreater(float(model.logits.grad.norm()), 0.0)
        self.assertTrue(bool(torch.isfinite(model.logits.grad).all()))

    def test_smoke_run_writes_reproducible_artifacts(self) -> None:
        config = BilevelMaskConfig(
            seed=7, relaxation="soft", outer_steps=1, inner_steps=1,
            tasks_per_step=1, validate_every=1, validation_restarts=1,
            eval_steps=1, eval_restarts=1, random_masks=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            train_structure(config, output, "cpu")
            self.assertTrue((output / "summary.json").exists())
            artifacts = torch.load(output / "artifacts.pt", weights_only=True)
            self.assertEqual(int(artifacts["learned_mask"].sum()), 32)
            self.assertTrue(torch.equal(artifacts["learned_mask"], artifacts["learned_mask"].round()))


class LatentJointTest(unittest.TestCase):
    def test_vectorized_shapes_cardinality_and_gradients(self) -> None:
        config = LatentJointConfig(restarts=3, outer_steps=1, weight_steps=1,
                                   eval_weight_steps=1, eval_rounds=1, z_max_steps=1,
                                   relaxation="ste")
        generator = LatentMaskGenerator(config.latent_dim, config.generator_width, seed=2)
        z = torch.randn(2, 3, config.latent_dim, requires_grad=True)
        masks, soft, _ = relaxed_masks(generator, z, config, 0.5)
        self.assertEqual(tuple(masks.shape), (2, 3, 8, 8))
        self.assertTrue(torch.equal(masks.detach(), masks.detach().round()))
        self.assertTrue(torch.allclose(soft.sum((-1, -2)), torch.full((2, 3), 32.0), atol=1e-4))
        weights = BatchedWeights(2, 3, seed=4, device=torch.device("cpu"))
        logits = batched_forward(torch.randn(2, 5, 8), weights, masks)
        self.assertEqual(tuple(logits.shape), (2, 3, 5))
        logits.square().mean().backward()
        self.assertGreater(float(z.grad.norm()), 0.0)
        self.assertGreater(sum(float(p.grad.norm()) for p in generator.parameters() if p.grad is not None), 0.0)

    def test_task_splits_are_disjoint_and_have_fixed_shapes(self) -> None:
        config = LatentJointConfig(restarts=2, outer_steps=1, weight_steps=1,
                                   eval_weight_steps=1, eval_rounds=1, z_max_steps=1,
                                   exact_population_loss=False, sample_with_replacement=False,
                                   support_positive=24, support_negative=48,
                                   validation_positive=12, validation_negative=24,
                                   query_positive=12, query_negative=24)
        data = make_task_splits(("0000", "0101"), config, torch.device("cpu"))
        self.assertEqual(tuple(data.support_x.shape), (2, 72, 8))
        self.assertEqual(tuple(data.validation_x.shape), (2, 36, 8))
        self.assertEqual(tuple(data.query_x.shape), (2, 36, 8))
        for task in range(2):
            rows = [set(map(tuple, tensor[task].tolist())) for tensor in
                    (data.support_x, data.validation_x, data.query_x)]
            self.assertFalse(rows[0] & rows[1])
            self.assertFalse(rows[0] & rows[2])
            self.assertFalse(rows[1] & rows[2])

    def test_z_optimizer_is_batched_and_finite(self) -> None:
        config = LatentJointConfig(restarts=2, outer_steps=1, weight_steps=1,
                                   eval_weight_steps=1, eval_rounds=1, z_max_steps=3,
                                   z_patience=2)
        generator = LatentMaskGenerator(config.latent_dim, config.generator_width, seed=2)
        weights = BatchedWeights(2, 2, seed=4, device=torch.device("cpu"))
        z = torch.nn.Parameter(torch.randn(2, 2, config.latent_dim))
        data = make_task_splits(("0000", "0101"), config, torch.device("cpu"))
        result = optimize_z_to_convergence(
            generator, z, weights, data.validation_x, data.validation_y, config, 0.5,
        )
        self.assertLessEqual(int(result["steps"]), 3)
        self.assertTrue(math.isfinite(float(result["validation_bce"])))
        self.assertLessEqual(float(z.norm(dim=-1).max()), config.z_radius + 1e-5)

    def test_latent_joint_smoke_run(self) -> None:
        config = LatentJointConfig(
            seed=8, restarts=2, outer_steps=1, weight_steps=1,
            eval_weight_steps=1, eval_rounds=1, z_max_steps=1, z_patience=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "latent"
            train_latent_generator(config, output, "cpu")
            self.assertTrue((output / "summary.json").exists())
            self.assertTrue((output / "RESULTS.md").exists())
            artifacts = torch.load(output / "training_artifacts.pt", weights_only=True)
            self.assertEqual(tuple(artifacts["latents"].shape), (10, 2, config.latent_dim))


if __name__ == "__main__":
    unittest.main()
