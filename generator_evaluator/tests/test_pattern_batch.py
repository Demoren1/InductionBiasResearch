"""Equivalence checks for the vectorised pattern measurement path."""
from __future__ import annotations

import dataclasses
import unittest

import torch

from generator_evaluator.adapters import build_pattern_fixture, measure_mask
from generator_evaluator.data import InnerProtocol
from generator_evaluator.pattern_batch import fit_pattern_batch


def _inputs():
    bank, tasks, _ = build_pattern_fixture(seed=81, bank_steps=1, teacher_count=5,
                                           support_count=8, query_count=8, k=8)
    chosen = [task for task in tasks if task.split == "train"][:2]
    masks = torch.stack((bank.masks[0], bank.masks[1]))
    return masks, chosen


class PatternBatchTests(unittest.TestCase):
    def test_batch_matches_independent_fixed_horizon_children(self) -> None:
        masks, tasks = _inputs()
        protocol = InnerProtocol(steps=3, replicas=2, lr=.01, l2=.03, checkpoint_every=1, seed=29)
        sequential = [measure_mask(mask, task, protocol) for mask, task in zip(masks, tasks)]
        batched = fit_pattern_batch(masks, tasks, protocol)
        self.assertEqual(len(batched), len(sequential))
        for expected, actual, mask, task in zip(sequential, batched, masks, tasks):
            self.assertEqual(actual["protocol_id"], protocol.fingerprint)
            self.assertEqual(actual["task_id"], task.task_id)
            self.assertEqual(actual["label_source"], "fresh_terminal_query")
            self.assertTrue(actual["fixed_horizon"])
            self.assertEqual(actual["seeds"], expected["seeds"])
            self.assertEqual(len(actual["replica_losses"]), len(expected["replica_losses"]))
            for left_loss, right_loss in zip(actual["replica_losses"], expected["replica_losses"]):
                self.assertAlmostEqual(left_loss, right_loss, places=5)
            for left, right in zip(expected["state_dict"], actual["state_dict"]):
                for name in left:
                    torch.testing.assert_close(left[name], right[name], rtol=2e-5, atol=2e-6)
            for left, right in zip(expected["effective_weights"], actual["effective_weights"]):
                torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(left, right * mask.ne(0), rtol=2e-5, atol=2e-6)
            for left_history, right_history in zip(expected["history"], actual["history"]):
                self.assertEqual([row["step"] for row in right_history], [0, 1, 2, 3])
                for left_value, right_value in zip([row["query_bce"] for row in right_history],
                                                    [row["query_bce"] for row in left_history]):
                    self.assertAlmostEqual(left_value, right_value, places=5)
            self.assertEqual(len(actual["optimizer_state"]), protocol.replicas)


    def test_query_labels_never_change_batched_child_states(self) -> None:
        masks, tasks = _inputs()
        protocol = InnerProtocol(steps=2, replicas=2, lr=.01, checkpoint_every=1, seed=91)
        changed = [tasks[0], dataclasses.replace(tasks[1], y_query=1.0 - tasks[1].y_query)]
        original = fit_pattern_batch(masks, tasks, protocol)
        altered = fit_pattern_batch(masks, changed, protocol)
        for before, after in zip(original, altered):
            for state_before, state_after in zip(before["state_dict"], after["state_dict"]):
                for name in state_before:
                    torch.testing.assert_close(state_before[name], state_after[name])
        self.assertNotEqual(original[1]["replica_losses"], altered[1]["replica_losses"])


    def test_rejects_minibatch_or_nonbinary_masks(self) -> None:
        masks, tasks = _inputs()
        with self.assertRaisesRegex(ValueError, "full-batch"):
            fit_pattern_batch(masks, tasks, InnerProtocol(steps=1, replicas=2, batch_size=4))
        masks[0, 0, 0] = .5
        with self.assertRaisesRegex(ValueError, "finite and binary"):
            fit_pattern_batch(masks, tasks, InnerProtocol(steps=1, replicas=2))


if __name__ == "__main__":
    unittest.main()
