"""Small end-to-end checks for train/evaluate artifact and gradient semantics."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

import torch

from meta_pattern.config import Config
from meta_pattern.evaluate import evaluate
from meta_pattern.train import stable_clip_grad_norm_, train
from meta_pattern.common import task_splits


class PipelineTests(unittest.TestCase):
    def test_stable_clipping_scales_large_finite_float32_gradients(self):
        parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
        parameter.grad = torch.tensor([1e23, -2e23], dtype=torch.float32)
        expected = float(torch.linalg.vector_norm(parameter.grad.double()))
        reported = stable_clip_grad_norm_([parameter], max_norm=1.0)
        self.assertTrue(math.isfinite(reported))
        self.assertAlmostEqual(reported / expected, 1.0, places=12)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertAlmostEqual(float(torch.linalg.vector_norm(parameter.grad)), 1.0, places=5)

    def test_stable_clipping_rejects_actual_nonfinite_gradient_elements(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                parameter = torch.nn.Parameter(torch.zeros(1))
                parameter.grad = torch.tensor([value])
                with self.assertRaises(FloatingPointError):
                    stable_clip_grad_norm_([parameter], max_norm=1.0)

    def test_stable_clipping_matches_torch_for_ordinary_gradients(self):
        ours = [torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2))]
        reference = [torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2))]
        gradients = [torch.tensor([2.0, -3.0, 1.5]), torch.tensor([-0.5, 4.0])]
        for left, right, gradient in zip(ours, reference, gradients):
            left.grad = gradient.clone()
            right.grad = gradient.clone()
        ours_norm = stable_clip_grad_norm_(ours, max_norm=0.75)
        torch_norm = torch.nn.utils.clip_grad_norm_(reference, max_norm=0.75)
        self.assertAlmostEqual(ours_norm, float(torch_norm), places=6)
        for left, right in zip(ours, reference):
            self.assertTrue(torch.allclose(left.grad, right.grad, rtol=2e-6, atol=2e-7))

    def test_interpolation_catalogue_excludes_unseen_lengths_from_training(self):
        c = Config(train_lengths=(3, 4, 6, 8), all_unseen_patterns=True)
        splits = task_splits(c)
        self.assertTrue(all(t.length in c.train_lengths for s in ("train", "val") for t in splits[s]))
        self.assertEqual(sum(t.length == 5 for t in splits["test"]), 32)
        self.assertEqual(sum(t.length == 7 for t in splits["test"]), 128)

    def test_resumed_training_matches_uninterrupted_training(self):
        c = Config(lengths=(3,), train_lengths=(3,), outer_steps=2, inner_steps=2,
                   tasks_per_step=1, support_size=16, query_size=16, batch_size=8,
                   val_tasks_per_length=1, validate_every=1, width=8,
                   inner_optimizer="adam", inner_lr=0.01)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train(c, root / "full", "cpu")
            train(replace(c, outer_steps=1), root / "first", "cpu")
            train(c, root / "resumed", "cpu", resume=root / "first/latest.pt")
            expected = torch.load(root / "full/latest.pt", weights_only=False)
            resumed = torch.load(root / "resumed/latest.pt", weights_only=False)
            self.assertEqual(expected["step"], resumed["step"])
            for key, value in expected["model"].items():
                self.assertTrue(torch.equal(value, resumed["model"][key]), key)

    def test_resume_retains_best_that_precedes_latest_checkpoint(self):
        c = Config(lengths=(3,), train_lengths=(3,), outer_steps=2, inner_steps=2,
                   tasks_per_step=1, support_size=16, query_size=16, batch_size=8,
                   val_tasks_per_length=1, validate_every=1, width=8,
                   inner_optimizer="adam", inner_lr=0.01)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            train(replace(c, outer_steps=1), first, "cpu")
            # Model a legitimate historical best at an earlier validation
            # point.  Its score is deliberately lower than every BCE observed
            # in this compact run, so resuming must copy it instead of replacing
            # it with the latest state at step one.
            historical_best = torch.load(first / "latest.pt", weights_only=False)
            historical_best["step"] = 0
            historical_best["val_bce"] = 0.0
            historical_best["best_val_bce"] = 0.0
            torch.save(historical_best, first / "best.pt")

            resumed_dir = root / "resumed"
            train(c, resumed_dir, "cpu", resume=first / "latest.pt")
            resumed_best = torch.load(resumed_dir / "best.pt", weights_only=False)
            resumed_latest = torch.load(resumed_dir / "latest.pt", weights_only=False)
            done = json.loads((resumed_dir / "done.json").read_text())
            self.assertEqual(resumed_best["step"], 0)
            self.assertEqual(resumed_best["val_bce"], 0.0)
            self.assertEqual(resumed_latest["best_val_bce"], 0.0)
            self.assertEqual(done["best_val_bce"], 0.0)

    def test_invalid_protocol_fails_early(self):
        with self.assertRaises(ValueError):
            Config(method="ideal", rank1=8)
        with self.assertRaises(ValueError):
            Config(train_lengths=(3, 3))
        with self.assertRaises(ValueError):
            evaluate("missing.pt", "unused.json", support_size=0)

    def test_training_updates_generator_and_evaluation_is_reproducible(self):
        c = Config(lengths=(3,), train_lengths=(3,), outer_steps=1, inner_steps=2,
                   tasks_per_step=1, support_size=16, query_size=16, batch_size=8,
                   val_tasks_per_length=1, validate_every=1, width=8)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            checkpoint = train(c, folder / "run", "cpu")
            with self.assertRaises(FileExistsError):
                train(c, folder / "run", "cpu")
            history = [json.loads(line) for line in (folder / "run/training.jsonl").read_text().splitlines()]
            self.assertGreater(history[0]["gradient_norm_before_clip"], 0)
            latest = torch.load(folder / "run/latest.pt", weights_only=False)
            self.assertEqual(latest["step"], 1)
            first = evaluate(checkpoint, folder / "first.json", "cpu", steps=(2,), repeats=1,
                             tasks_per_length=1, test_size=32)
            second = evaluate(checkpoint, folder / "second.json", "cpu", steps=(2,), repeats=1,
                              tasks_per_length=1, test_size=32)
            self.assertEqual(first["rows"], second["rows"])
            self.assertEqual({r["regime"] for r in first["aggregates"]}, {"seen_length"})
            self.assertEqual(len(first["rows"]), 2)


if __name__ == "__main__":
    unittest.main()
