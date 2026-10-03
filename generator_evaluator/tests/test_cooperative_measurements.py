"""Batched real-label caching and recovery from partial artifact writes."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from generator_evaluator import cooperative_measurements as module
from generator_evaluator.cooperative_data import build_cooperative_fixture
from generator_evaluator.data import InnerProtocol, RealReplay


class BatchedStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        banks, tasks, selection, _ = build_cooperative_fixture(seed=417, bank_steps=1,
                 teachers_per_pattern=5, support_count=8, query_count=8, selection_count=4, k=8)
        cls.entries = [(banks["0001"].baseline_mask, tasks[0], "candidate"),
                       (banks["0011"].baseline_mask, tasks[1], "candidate")]
        cls.selection = selection
        cls.protocol = InnerProtocol(steps=3, replicas=2, checkpoint_every=1, seed=417)

    def test_complete_cache_skips_all_fits_and_shape_groups_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay = RealReplay(self.protocol)
            store = module.BatchedMeasurementStore(Path(tmp), replay, "cpu")
            entries = self.entries + [(self.entries[0][0], self.selection[0], "selection")]
            with patch.object(module, "fit_pattern_batch", wraps=module.fit_pattern_batch) as fit:
                results = store.measure_many(entries)
                self.assertEqual(fit.call_count, 2)
            with patch.object(module, "fit_pattern_batch", side_effect=AssertionError("must use cache")):
                cached = store.measure_many(entries)
            self.assertEqual([row[0]["quality"] for row in results], [row[0]["quality"] for row in cached])
            self.assertEqual(len(replay.records), 3)
            replay.validate()

    def test_partial_artifact_batch_recovers_same_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            reference = module.BatchedMeasurementStore(out / "reference", RealReplay(self.protocol), "cpu")
            expected = reference.measure_many(self.entries)
            replay = RealReplay(self.protocol)
            store = module.BatchedMeasurementStore(out / "interrupted", replay, "cpu")
            original = module.save_torch
            def interrupt(path, payload):
                original(path, payload)
                raise RuntimeError("simulated partial batch")
            with patch.object(module, "save_torch", interrupt), self.assertRaisesRegex(RuntimeError, "partial batch"):
                store.measure_many(self.entries)
            self.assertEqual(len(replay.records), 0)
            actual = store.measure_many(self.entries)
            for (_, left), (_, right) in zip(expected, actual):
                self.assertEqual(left["replica_losses"], right["replica_losses"])
                for a, b in zip(left["state_dict"], right["state_dict"]):
                    for key in a:
                        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            replay.validate()


if __name__ == "__main__":
    unittest.main()
