from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from generator_evaluator.adapters import build_pattern_fixture, make_pattern_test_tasks
from generator_evaluator.data import InnerProtocol, RealReplay
from generator_evaluator.run import MeasurementStore, SearchConfig, run_experiment


class IntegrationTests(unittest.TestCase):
    def test_deepsets_cycle_uses_packed_store_and_freezes_before_test(self):
        from generator_evaluator.adapters import FunctionalBank
        from generator_evaluator.tests.test_deepsets_batch import _task, _masks
        task, masks = _task(), _masks()
        validation = replace(task, task_id="deepsets:validation:fixture", split="validation",
                             support_ids=task.support_ids + 100, query_ids=task.query_ids + 100)
        test = replace(task, task_id="deepsets:test:fixture", split="test",
                       support_ids=task.support_ids + 200, query_ids=task.query_ids + 200)
        bank = FunctionalBank(torch.randn(1, 2, 3, 12), None, masks, masks[0],
                              {"family": "deepsets"})
        protocol = InnerProtocol(steps=2, replicas=2, checkpoint_every=1, seed=222, metric="nmse")
        config = SearchConfig(seed=222, k=90, generator_epochs=1, updates_per_epoch=1,
            refresh_every=1, acquisition_budget=2, candidates=4, initial_random=2,
            evaluator_epochs=1, width=8, heads=2, layers=1, noise_dim=2,
            ensemble_members=2, smoke=True, direct_control=False)
        with tempfile.TemporaryDirectory() as folder, patch("generator_evaluator.run.write_plots"):
            out = Path(folder)
            def test_factory():
                self.assertTrue((out / "frozen.pt").is_file())
                return [test]
            result = run_experiment(bank, [task, validation], test_factory, out, protocol, config,
                dense_learning_rates=[.01], measurement_devices=["cpu"], measurement_batch_size=2)
            self.assertTrue((out / "COMPLETE").is_file())
            self.assertIn(test.task_id, result["test"])
            RealReplay.load(out / "replay.pt").validate()

    def fixture(self):
        bank, tasks, spec = build_pattern_fixture(seed=71, bank_steps=2, teacher_count=2,
                                                  support_count=8, query_count=8, k=8)
        tasks = [task for task in tasks if task.split == "train"][:2] + [task for task in tasks if task.split == "validation"][:1]
        protocol = InnerProtocol(steps=2, replicas=2, checkpoint_every=1, seed=71)
        config = SearchConfig(seed=71, k=8, generator_epochs=1, updates_per_epoch=1,
            refresh_every=1, acquisition_budget=3, candidates=6, initial_random=2,
            evaluator_epochs=1, width=8, heads=2, layers=1, noise_dim=2,
            ensemble_members=2, smoke=True)
        return bank, tasks, spec, protocol, config

    def test_fresh_draws_spend_real_labels_with_distinct_initializations(self):
        bank, tasks, _, protocol, _ = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            replay = RealReplay(protocol)
            store = MeasurementStore(Path(folder), replay, "cpu")
            a, _ = store.measure(bank.baseline_mask, tasks[0], "initial")
            cached, _ = store.measure(bank.baseline_mask, tasks[0], "cache")
            self.assertEqual(a, cached)
            self.assertEqual(len(replay.records), 1)
            b, states_b = store.measure(bank.baseline_mask, tasks[0], "direct", fresh=True)
            c, states_c = store.measure(bank.baseline_mask, tasks[0], "direct", fresh=True)
            self.assertEqual(len(replay.records), 3)
            self.assertNotEqual(a["seeds"], b["seeds"])
            self.assertNotEqual(b["seeds"], c["seeds"])
            self.assertFalse(torch.equal(states_b["state_dict"][0]["w"], states_c["state_dict"][0]["w"]))
            replay.validate()

    def test_full_cycle_freezes_before_test_and_rejects_changed_bank(self):
        bank, tasks, spec, protocol, config = self.fixture()
        with tempfile.TemporaryDirectory() as folder, patch("generator_evaluator.run.write_plots"):
            out = Path(folder)
            def test_factory():
                self.assertTrue((out / "frozen.pt").is_file())
                return make_pattern_test_tasks(spec)[:1]
            result = run_experiment(bank, tasks, test_factory, out, protocol, config,
                                    dense_learning_rates=[.01])
            self.assertTrue(result["budgets_match"])
            self.assertGreater(result["direct_training_mask_task_budget"], 0)
            self.assertTrue((out / "COMPLETE").exists())
            replay = RealReplay.load(out / "replay.pt")
            self.assertTrue(any(row["split"] == "test" for row in replay.records))
            frozen = torch.load(out / "frozen.pt", weights_only=False)
            self.assertEqual(frozen["masks"]["generator"].sum().item(), config.k)
            with patch("generator_evaluator.run.measure_mask", side_effect=AssertionError("completed run retrained")):
                again = run_experiment(bank, tasks, test_factory, out, protocol, config,
                                       resume=True, dense_learning_rates=[.01])
            self.assertEqual(result, again)
            changed = replace(bank, baseline_mask=bank.baseline_mask.roll(1, dims=0))
            with self.assertRaisesRegex(ValueError, "different inputs"):
                run_experiment(changed, tasks, test_factory, out, protocol, config,
                               resume=True, dense_learning_rates=[.01])

    def test_interrupted_epoch_restores_atomic_state_and_frozen_restart(self):
        from generator_evaluator.artifacts import save_torch
        bank, tasks, spec, protocol, config = self.fixture()
        with tempfile.TemporaryDirectory() as folder, patch("generator_evaluator.run.write_plots"):
            out, reference = Path(folder) / "resume", Path(folder) / "reference"
            factory = lambda: make_pattern_test_tasks(spec)[:1]
            expected = run_experiment(bank, tasks, factory, reference, protocol, config,
                                      dense_learning_rates=[.01])
            def interrupt(path, payload):
                if path.name == "checkpoint.pt" and payload.get("epoch") == 1:
                    raise RuntimeError("simulated interruption after replay export")
                return save_torch(path, payload)
            with patch("generator_evaluator.run.save_torch", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    run_experiment(bank, tasks, factory, out, protocol, config,
                                   dense_learning_rates=[.01])
            checkpoint = torch.load(out / "checkpoint.pt", weights_only=False)
            self.assertEqual(checkpoint["epoch"], 0)
            self.assertGreater(len(RealReplay.load(out / "replay.pt").records), len(checkpoint["replay"].records))
            def fail_test():
                self.assertTrue((out / "frozen.pt").exists())
                raise RuntimeError("simulated test-stage interruption")
            with self.assertRaisesRegex(RuntimeError, "test-stage"):
                run_experiment(bank, tasks, fail_test, out, protocol, config,
                               resume=True, dense_learning_rates=[.01])
            frozen_before = torch.load(out / "frozen.pt", weights_only=False)
            actual = run_experiment(bank, tasks, factory, out, protocol, config,
                                    resume=True, dense_learning_rates=[.01])
            self.assertEqual(actual["test"], expected["test"])
            frozen_after = torch.load(out / "frozen.pt", weights_only=False)
            for key in frozen_before["masks"]:
                torch.testing.assert_close(frozen_before["masks"][key], frozen_after["masks"][key])
            a = torch.load(out / "checkpoint.pt", weights_only=False)
            b = torch.load(reference / "checkpoint.pt", weights_only=False)
            for key in a["generator"]:
                torch.testing.assert_close(a["generator"][key], b["generator"][key])
