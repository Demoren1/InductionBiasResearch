"""Tests for the held-length full-U interpolation evaluator."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from meta_pattern.calibrate import batched_adapt_v
from meta_pattern.common import make_model
from meta_pattern.config import Config
from meta_pattern.evaluate_interpolation import (
    _validate_protocol,
    load_interpolation_checkpoint,
)
from meta_pattern.models import RandomU, adapt_v


class InterpolationEvaluationTests(unittest.TestCase):
    def test_batched_snapshots_match_scalar_adaptation_at_every_budget(self) -> None:
        """One batched U uses the same updates as independent scalar episodes."""
        u = RandomU(seq_len=5, hidden=3, rank1=2, rank2=2, length_min=2, length_max=2, seed=13)(2)
        x = torch.tensor([
            [[1., -1., 1., -1., 1.], [-1., 1., -1., 1., -1.],
             [1., 1., -1., -1., 1.], [-1., -1., 1., 1., -1.]],
            [[-1., -1., 1., 1., -1.], [1., 1., -1., -1., 1.],
             [-1., 1., -1., 1., -1.], [1., -1., 1., -1., 1.]],
        ])
        y = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]])
        seeds = torch.tensor([11, 29], dtype=torch.int64)
        budgets = (1, 2, 4)
        snapshots = batched_adapt_v(
            u, x, y, steps=max(budgets), lrs=torch.tensor([0.03, 0.03]), seeds=seeds,
            optimizers=["adam", "adam"], init_scales=torch.tensor([0.1, 0.1]),
            batch_size=2, checkpoints=budgets,
        )
        for budget in budgets:
            for index, seed in enumerate(seeds.tolist()):
                expected = adapt_v(
                    u, x[index], y[index], steps=budget, lr=0.03, seed=seed,
                    create_graph=False, batch_size=2, optimizer="adam", init_scale=0.1,
                )
                actual = (snapshots[budget][0][index], snapshots[budget][1][index])
                for observed, reference in zip(actual, expected):
                    self.assertTrue(torch.allclose(observed, reference, rtol=1e-6, atol=1e-7))

    def test_checkpoint_loading_requires_exactly_the_four_training_lengths(self) -> None:
        good = Config(
            lengths=(3, 4, 5, 6, 7, 8), train_lengths=(3, 4, 6, 8),
            width=8, generator_depth=2,
        )
        bad = Config(
            lengths=(3, 4, 5, 6, 7, 8), train_lengths=(3, 4, 5, 6, 8),
            width=8, generator_depth=2,
        )
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            good_model = make_model(good, "cpu")
            good_path = folder / "good.pt"
            torch.save({"config": good.to_dict(), "model": good_model.state_dict(), "step": 7,
                        "source_sha256": {"train.py": "test"}}, good_path)
            state, config, model = load_interpolation_checkpoint(good_path, "cpu")
            self.assertEqual(state["step"], 7)
            self.assertEqual(config.train_lengths, (3, 4, 6, 8))
            self.assertEqual(type(model), type(good_model))

            bad_model = make_model(bad, "cpu")
            bad_path = folder / "bad.pt"
            torch.save({"config": bad.to_dict(), "model": bad_model.state_dict()}, bad_path)
            with self.assertRaisesRegex(ValueError, "exactly lengths"):
                load_interpolation_checkpoint(bad_path, "cpu")
            with self.assertRaisesRegex(ValueError, "exactly lengths"):
                _validate_protocol(bad)


if __name__ == "__main__":
    unittest.main()
