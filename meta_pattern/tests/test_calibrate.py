"""Tests for the analytic-U inner-optimizer calibration protocol."""

from __future__ import annotations

import unittest

import torch

from meta_pattern.calibrate import (
    BUDGETS,
    KNOWN_LENGTHS,
    batched_adapt_v,
    grid,
    selected_task_manifest,
)
from meta_pattern.data import PatternTask
from meta_pattern.models import adapt_v, ideal_u


class CalibrationTests(unittest.TestCase):
    def test_protocol_never_selects_heldout_lengths_or_test_patterns(self) -> None:
        manifest = selected_task_manifest(seed=42)
        self.assertEqual(set(map(int, manifest)), set(KNOWN_LENGTHS))
        self.assertNotIn(5, KNOWN_LENGTHS)
        self.assertNotIn(7, KNOWN_LENGTHS)
        self.assertEqual(BUDGETS, (20, 100, 500, 2000))
        for length, splits in manifest.items():
            self.assertEqual(set(splits), {"train", "val"})
            self.assertLessEqual(len(splits["train"]), 4)
            self.assertLessEqual(len(splits["val"]), 4)
            self.assertTrue(splits["train"])
            self.assertTrue(splits["val"])
            # PatternTask also makes this check independent of an accidental
            # change to the textual serialization in the manifest.
            self.assertTrue(all(PatternTask(pattern).length == int(length)
                                for patterns in splits.values() for pattern in patterns))
        self.assertEqual(len(grid()), 12)

    def test_batched_sgd_and_adam_match_independent_scalar_adaptation(self) -> None:
        # Float64 makes this a sensitive equivalence test, while two different
        # optimizer branches make sure selection in the batched kernel cannot
        # accidentally apply one setting to every row.
        u = ideal_u(3, seq_len=8, hidden=6, rank1=5, rank2=2, dtype=torch.float64)
        generator = torch.Generator(device="cpu").manual_seed(101)
        x = (torch.randint(0, 2, (24, 8), generator=generator) * 2 - 1).to(torch.float64)
        y = torch.randint(0, 2, (24,), generator=generator).to(torch.float64)
        snapshots = batched_adapt_v(
            u,
            torch.stack((x, x)),
            torch.stack((y, y)),
            steps=7,
            lrs=torch.tensor((0.2, 0.03), dtype=torch.float64),
            seeds=torch.tensor((37, 41), dtype=torch.int64),
            optimizers=("sgd", "adam"),
            init_scales=torch.tensor((0.1, 1.0), dtype=torch.float64),
            batch_size=8,
            checkpoints=(7,),
        )
        scalar_sgd = adapt_v(
            u, x, y, steps=7, lr=0.2, seed=37, create_graph=False, batch_size=8,
            optimizer="sgd", init_scale=0.1,
        )
        scalar_adam = adapt_v(
            u, x, y, steps=7, lr=0.03, seed=41, create_graph=False, batch_size=8,
            optimizer="adam", init_scale=1.0,
        )
        batched = snapshots[7]
        # bmm and scalar matmul have a different reduction order, so bitwise
        # equality is not a meaningful CPU/GPU requirement.  These bounds are
        # still far tighter than float64 training error and catch any formula,
        # seed, batch schedule, or per-model-mean reduction discrepancy.
        torch.testing.assert_close(batched[0][0], scalar_sgd[0], rtol=1e-8, atol=5e-9)
        torch.testing.assert_close(batched[1][0], scalar_sgd[1], rtol=1e-8, atol=5e-9)
        torch.testing.assert_close(batched[0][1], scalar_adam[0], rtol=1e-8, atol=5e-9)
        torch.testing.assert_close(batched[1][1], scalar_adam[1], rtol=1e-8, atol=5e-9)


if __name__ == "__main__":
    unittest.main()
