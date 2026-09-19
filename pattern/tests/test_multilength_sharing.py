"""Tests for multi-length generated parameter sharing."""

import tempfile
import unittest
from pathlib import Path

import torch

from pattern.bilevel_mask.generated_sharing import SharingGenerator, assignments
from pattern.bilevel_mask.multilength_sharing import (
    MultiLengthConfig,
    all_tasks,
    cyclic_assignment,
    make_data,
    run,
)


class MultiLengthSharingTest(unittest.TestCase):
    def tiny_config(self, **changes):
        values = dict(
            seed=9, pattern_lengths=(2, 3), filter_dim=3, hidden=8, seq_len=8,
            train_restarts=2, eval_restarts=2, tasks_per_length=1,
            outer_steps=1, inner_steps=1, final_refit_steps=1,
            refit_validate_every=1, eval_task_chunk=4,
            support_per_class=2, validation_per_class=2, query_per_class=2,
            evaluation_support_per_class=2, evaluation_validation_per_class=2,
            evaluation_query_per_class=2,
        )
        values.update(changes)
        return MultiLengthConfig(**values)

    def test_cyclic_structure_and_assignments(self) -> None:
        config = self.tiny_config()
        oracle = cyclic_assignment(config, torch.device("cpu"))
        self.assertEqual(tuple(oracle.shape), (8, 8, 3))
        self.assertTrue(torch.all(oracle.sum(-3) == 1))
        generator = SharingGenerator(config)
        z = torch.randn(2, 2, config.latent_dim)
        learned = assignments(generator, z, config, 0.5, "hard")
        self.assertEqual(tuple(learned.shape), (2, 2, 8, 8, 3))
        self.assertTrue(torch.all(learned.sum(-3) == 1))

    def test_data_contains_both_lengths(self) -> None:
        config = self.tiny_config()
        tasks = all_tasks(config.pattern_lengths)
        data = make_data(tasks, config, torch.device("cpu"), "test")
        self.assertEqual(len(tasks), 12)
        self.assertEqual(tuple(data.support_x.shape), (12, 4, 8))
        self.assertEqual({task.length for task in data.tasks}, {2, 3})

    def test_smoke_both_variants(self) -> None:
        config = self.tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            for variant in ("global", "length_latent"):
                output = Path(directory) / variant
                run(config, variant, output, "cpu")
                self.assertTrue((output / "summary.json").exists())
                self.assertTrue((output / "training.pt").exists())


if __name__ == "__main__":
    unittest.main()
