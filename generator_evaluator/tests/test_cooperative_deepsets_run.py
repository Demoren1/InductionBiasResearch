"""Small real fits exercise the two-stage DeepSets cooperative pipeline."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from deepsets_vaae.core import Split
from generator_evaluator.cooperative_run import deepsets_config, run_cooperative_experiment
from generator_evaluator.data import InnerProtocol
from generator_evaluator.warm_start import load_cooperative_warm_start


def synthetic_data(row_count=160):
    rng = torch.Generator().manual_seed(783)
    return {f"{family}_{role}": Split(torch.randn(row_count, 784, generator=rng),
            torch.arange(row_count) % 10, torch.arange(row_count) + index * 1000)
            for index, (family, role) in enumerate(
                (family, role) for family in ("source", "target")
                for role in ("train", "validation", "test"))}


class CooperativeDeepSetsIntegrationTests(unittest.TestCase):
    def test_bootstrap_stays_sealed_and_search_imports_live_critic_and_banks(self):
        self._run_cycle(train_count=2, test_count=2)

    def test_three_generators_share_one_critic_and_hold_out_three_tasks(self):
        self._run_cycle(train_count=3, test_count=3)

    def test_six_generators_and_four_heldout_tasks_complete_joint_bootstrap_and_search(self):
        self._run_cycle(train_count=6, test_count=4)

    def _run_cycle(self, train_count, test_count):
        from generator_evaluator.cooperative_deepsets import (build_cooperative_deepsets_fixture,
                                                              make_cooperative_deepsets_test_tasks)
        torch.set_num_threads(1)
        names = tuple(str(index) for index in range(train_count))
        config = deepsets_config(seed=783, hidden=2, k=32, output_budgets=(80, 800),
            train_patterns=names, test_task_count=test_count,
            generator_epochs=1, updates_per_epoch=2, refresh_every=1,
            cooperation_rounds=1, cooperation_updates=1,
            candidates=4, acquisition_budget=3, auxiliary_budget=1, initial_random=2,
            evaluator_epochs=1, width=8, heads=2, noise_dim=2,
            feedback_masks=1, bank_capacity=12, elite_limit=4, tune_dense=False, smoke=True,
            generator_pretrain_epochs=1, pretrain_updates_per_epoch=2)
        protocol = InnerProtocol(steps=1, replicas=2, checkpoint_every=1,
                                 seed=783, metric="nmse", lr=.005, l2=.0001)
        data = synthetic_data(row_count=max(160, 20 * (train_count + test_count)))
        with TemporaryDirectory() as directory, patch("deepsets_vaae.core.load_data", return_value=data), \
                patch("generator_evaluator.cooperative_deepsets.load_data", return_value=data, create=True), \
                patch("generator_evaluator.cooperative_run.write_plots"):
            root = Path(directory)
            fixture = build_cooperative_deepsets_fixture("unused", seed=783, bank_steps=1,
                teachers_per_task=8, bank_candidates=8, teacher_batch_size=4,
                support_count=4, query_count=3, selection_count=3, probe_count=4,
                k=32, hidden=2, out=root / "banks",
                train_task_count=train_count, test_task_count=test_count)
            identity_width = train_count + test_count
            for index, (task, selection) in enumerate(zip(fixture[1], fixture[2])):
                identity = torch.zeros(identity_width)
                identity[index] = 1
                torch.testing.assert_close(task.context[-identity_width:], identity)
                torch.testing.assert_close(selection.context[-identity_width:], identity)
            prepare = replace(config, phase="bootstrap")
            bootstrap = root / "bootstrap"
            result = run_cooperative_experiment(*fixture, bootstrap, protocol, prepare,
                measurement_devices=["cpu"], measurement_batch_size=4,
                test_factory=lambda spec: self.fail("bootstrap opened heldout tasks"))
            self.assertEqual(result["generators"], train_count)
            self.assertFalse(result["test_materialized"])
            self.assertFalse((bootstrap / "frozen.pt").exists())
            source = torch.load(bootstrap / "checkpoint.pt", weights_only=False)
            self.assertEqual(tuple(source["models"]), names)
            self.assertEqual({row["pattern"] for row in source["history"]}, set(names))
            self.assertTrue(any(row["input_density"] != "mixed" for row in source["history"]))
            self.assertEqual({row["stage"] for row in source["history"]}, {"joint"})
            self.assertTrue(all(row["direct_agreement_weight"] > 0 for row in source["history"]))
            self.assertTrue(all("reconstruction_loss" in row for row in source["history"]))
            self.assertEqual(len(source["pretraining_history"]), train_count * 2)
            self.assertTrue(all(row["reconstruction_scope"] == "teacher"
                                for row in source["pretraining_history"]))
            self.assertTrue(source["shared_latent_used"] is False)
            self.assertFalse(source["trainer_state"]["latent_optimizer"]["state"])
            self.assertTrue(all(bank.diagnostics["feedback_rows_added"] > 0
                                for bank in source["banks"].values()))
            torch.testing.assert_close(source["trainer_state"]["shared_latent"],
                                       source["initial_shared_latent"], rtol=0, atol=0)
            self.assertFalse(any(row["task_split"] == "test" for row in source["replay"].records))
            for bank in source["banks"].values():
                self.assertGreater(bank.diagnostics["feedback_rows_added"], 0)
                self.assertLessEqual(len(bank.masks), config.bank_capacity)
            warm = load_cooperative_warm_start(bootstrap, config, protocol)
            for name in config.train_patterns:
                torch.testing.assert_close(warm.banks[name].tokens, source["banks"][name].tokens)
            search = root / "search"
            def test_factory(spec):
                self.assertTrue((search / "frozen.pt").is_file())
                state = torch.load(search / "checkpoint.pt", weights_only=False)
                self.assertFalse(any(row["task_split"] == "test" for row in state["replay"].records))
                return make_cooperative_deepsets_test_tasks(spec)
            completed = run_cooperative_experiment(warm.banks, warm.train_tasks, warm.selection_tasks,
                warm.test_spec, search, protocol, config, warm_start=warm,
                measurement_devices=["cpu"], measurement_batch_size=4, test_factory=test_factory)
            self.assertEqual(completed["generators"], train_count)
            self.assertEqual(len(completed["test_task_ids"]), test_count)
            self.assertTrue(all("query_nmse" in completed["final"][task_id]["common"]
                                for task_id in completed["test_task_ids"]))
            state = torch.load(search / "checkpoint.pt", weights_only=False)
            self.assertEqual(len(state["pretraining_history"]), train_count * 2)
            self.assertTrue(all(row["reconstruction_scope"] == "teacher"
                                for row in state["pretraining_history"]))
            joint = [row for row in state["history"] if row["stage"] == "joint"]
            self.assertTrue(joint)
            self.assertEqual({row["stage"] for row in state["history"]}, {"joint"})
            self.assertTrue(all(row["direct_agreement_weight"] > 0 for row in joint))
            self.assertTrue(all("reconstruction_loss" in row for row in state["history"]))
            self.assertTrue(torch.equal(state["trainer_state"]["shared_latent"],
                                        state["initial_shared_latent"]))
            self.assertFalse(state["trainer_state"]["latent_optimizer"]["state"])
            self.assertEqual(state["stage"], "training_complete")
            for bank in state["banks"].values():
                self.assertGreater(bank.diagnostics["feedback_rows_added"], 0)
            state["replay"].validate()
            replay_contexts = state["replay"].tensors("train")[1]
            self.assertTrue(torch.all(replay_contexts[:, -identity_width:].sum(1) == 1))
            for key, value in source["ensemble"].items():
                torch.testing.assert_close(warm.ensemble_state[key], value)


if __name__ == "__main__":
    unittest.main()
