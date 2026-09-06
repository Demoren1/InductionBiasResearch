"""Focused no-GPU checks for the capacity-study finisher."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from meta_pattern.config import Config
from meta_pattern.scripts.finish_capacity_study import (
    _config_to_train_args,
    _select_architecture,
)


class FinishCapacityStudyTests(unittest.TestCase):
    def test_config_round_trip_renders_resume_cli_fields(self) -> None:
        config = Config(
            train_lengths=(3, 4, 6, 8), width=256, generator_depth=3,
            rank1=16, rank2=4, inner_optimizer="adam", inner_lr=0.1,
            init_scale=0.1, data_seed=20_260_906, all_unseen_patterns=True,
            inner_steps=50, tasks_per_step=4, support_size=1024, query_size=1024,
            batch_size=128, validate_every=100, val_tasks_per_length=0,
        ).to_dict()
        args = _config_to_train_args(config, Path("new-run"), Path("old/latest.pt"), 3000)
        self.assertEqual(args[:3], ["-m", "meta_pattern.train", "--out"])
        self.assertIn("--resume", args)
        self.assertEqual(args[args.index("--resume") + 1], "old/latest.pt")
        self.assertEqual(args[args.index("--outer-steps") + 1], "3000")
        self.assertEqual(args[args.index("--inner-optimizer") + 1], "adam")
        self.assertEqual(args[args.index("--inner-lr") + 1], "0.1")
        self.assertEqual(args[args.index("--init-scale") + 1], "0.1")
        self.assertEqual(args[args.index("--width") + 1], "256")
        self.assertEqual(args[args.index("--generator-depth") + 1], "3")
        begin = args.index("--train-lengths") + 1
        self.assertEqual(args[begin:begin + 4], ["3", "4", "6", "8"])
        self.assertIn("--all-unseen-patterns", args)
        self.assertIn("--data-seed", args)

    def test_selection_uses_only_fake_best_checkpoint_validation(self) -> None:
        config = Config(train_lengths=(3, 4, 6, 8), all_unseen_patterns=True).to_dict()
        # The winner is deliberately unlike lexical order, making this a real
        # exercise of score loading and equal-seed averaging rather than a
        # check of the tie break.
        means = {"small": (0.50, 0.52), "large": (0.35, 0.37), "xlarge": (0.41, 0.40)}
        with tempfile.TemporaryDirectory() as folder:
            outputs = {}
            for architecture, scores in means.items():
                for seed, score in zip((42, 43), scores):
                    output = Path(folder) / f"{architecture}_seed{seed}"
                    output.mkdir()
                    torch.save({"config": config, "val_bce": score}, output / "best.pt")
                    outputs[output.name] = output
            selected, scores, states = _select_architecture(outputs)
        self.assertEqual(selected, "large")
        self.assertAlmostEqual(scores["large"]["equal_seed_mean_best_val_bce"], 0.36)
        self.assertEqual(set(states), {f"{name}_seed{seed}" for name in means for seed in (42, 43)})


if __name__ == "__main__":
    unittest.main()
