"""Opt-in tiny two-GPU fit and joint/staged generator integration tests."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from generator_evaluator import cooperative_run as run
from generator_evaluator.cooperative_data import build_cooperative_fixture
from generator_evaluator.data import InnerProtocol
from generator_evaluator.parallel_measurements import ParallelMeasurementStore


@unittest.skipUnless(os.environ.get("GENERATOR_EVALUATOR_GPU_TEST") == "1",
                     "two-GPU integration is opt-in")
class StagedGPUIntegrationTests(unittest.TestCase):
    def test_joint_updates_with_two_spawned_measurement_workers(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        torch.set_num_threads(1)
        fixture = build_cooperative_fixture(
            seed=417, bank_steps=1, teachers_per_pattern=5,
            support_count=8, query_count=8, selection_count=4, k=8,
            probe_count=32, batch_teachers=True)
        config = run.pattern_small_config(
            seed=417, training_mode="joint", k=8, output_budgets=(12,),
            width=8, heads=2, evaluator_epochs=1,
            generator_epochs=1, updates_per_epoch=2,
            generator_pretrain_epochs=1, pretrain_updates_per_epoch=1,
            elite_limit=1, candidates=4, acquisition_budget=2,
            auxiliary_budget=0, smoke=True)
        protocol = InnerProtocol(steps=1, replicas=2, checkpoint_every=1, seed=417)
        worker_devices = []
        original_executor = ParallelMeasurementStore._executor

        def track_executor(store, device):
            worker_devices.append(device)
            return original_executor(store, device)

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(run, "write_plots"), \
                patch.object(ParallelMeasurementStore, "_executor", track_executor), \
                patch.dict(os.environ, GENERATOR_EVALUATOR_PROGRESS="0"):
            result = run.run_cooperative_experiment(
                *fixture, temporary, protocol, config, device="cuda:0",
                generator_devices=["cuda:0", "cuda:1"],
                measurement_devices=["cuda:0", "cuda:1"], measurement_batch_size=128)
            state = torch.load(Path(temporary) / "checkpoint.pt", map_location="cpu", weights_only=False)
            self.assertEqual(state["training_mode"], "joint")
            self.assertEqual({row["stage"] for row in state["history"]}, {"joint"})
            self.assertFalse(state["shared_latent_used"])
            self.assertFalse(state["trainer_state"]["latent_optimizer"]["state"])
            for name in config.train_patterns:
                updates = [row for row in state["history"] if row["pattern"] == name]
                self.assertEqual([row["output_k"] for row in updates], [8, 12])
                self.assertTrue(all(row["direct_agreement_weight"] > 0 for row in updates))
            self.assertEqual(set(worker_devices), {"cuda:0", "cuda:1"})
            state["replay"].validate()
            self.assertTrue(all(len(row["replica_losses"]) == 2
                                for row in state["replay"].records))
            self.assertFalse(result["test_used_for_training"])
            self.assertTrue((Path(temporary) / "frozen.pt").is_file())

    def test_quality_and_cooperation_with_two_spawned_measurement_workers(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        torch.set_num_threads(1)
        fixture = build_cooperative_fixture(
            seed=417, bank_steps=1, teachers_per_pattern=5,
            support_count=8, query_count=8, selection_count=4, k=8,
            probe_count=32, batch_teachers=True)
        config = run.pattern_small_config(
            seed=417, training_mode="staged", k=8, width=8, heads=2, evaluator_epochs=1,
            generator_epochs=1, updates_per_epoch=1,
            cooperation_rounds=1, cooperation_updates=1,
            generator_pretrain_epochs=1, pretrain_updates_per_epoch=1,
            candidates=4, acquisition_budget=2, smoke=True)
        protocol = InnerProtocol(steps=1, replicas=4, checkpoint_every=1, seed=417)
        worker_devices = []
        original_executor = ParallelMeasurementStore._executor

        def track_executor(store, device):
            worker_devices.append(device)
            return original_executor(store, device)

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(run, "write_plots"), \
                patch.object(ParallelMeasurementStore, "_executor", track_executor), \
                patch.dict(os.environ, GENERATOR_EVALUATOR_PROGRESS="0"):
            result = run.run_cooperative_experiment(
                *fixture, temporary, protocol, config, device="cuda:0",
                generator_devices=["cuda:0", "cuda:1"],
                measurement_devices=["cuda:0", "cuda:1"], measurement_batch_size=128)
            state = torch.load(Path(temporary) / "checkpoint.pt", map_location="cpu", weights_only=False)
            self.assertEqual({row["stage"] for row in state["history"]}, {"quality", "cooperation"})
            self.assertEqual(state["stage"], "training_complete")
            latent = state["trainer_state"]["shared_latent"]
            self.assertTrue(torch.isfinite(latent).all())
            latent_state = state["trainer_state"]["latent_optimizer"]["state"]
            self.assertTrue(latent_state)
            self.assertGreater(max(int(row["step"]) for row in latent_state.values()), 0)
            self.assertFalse(result["test_used_for_training"])
            self.assertEqual(set(worker_devices), {"cuda:0", "cuda:1"})
            state["replay"].validate()
            self.assertTrue(all(len(row["replica_losses"]) == 4
                                for row in state["replay"].records))
            self.assertTrue((Path(temporary) / "frozen.pt").is_file())

    def test_resume_after_cooperation_restores_gpu_optimizers_and_shared_latent(self):
        torch.set_num_threads(1)
        fixture = build_cooperative_fixture(
            seed=417, bank_steps=1, teachers_per_pattern=5,
            support_count=8, query_count=8, selection_count=4, k=8,
            probe_count=32, batch_teachers=True)
        config = run.pattern_small_config(
            seed=417, training_mode="staged", k=8, width=8, heads=2, evaluator_epochs=1,
            generator_epochs=1, updates_per_epoch=1,
            cooperation_rounds=2, cooperation_updates=1,
            generator_pretrain_epochs=1, pretrain_updates_per_epoch=1,
            candidates=4, acquisition_budget=2, smoke=True)
        protocol = InnerProtocol(steps=1, replicas=2, checkpoint_every=1, seed=417)
        kwargs = dict(device="cuda:0", generator_devices=["cuda:0", "cuda:1"],
                      measurement_devices=["cuda:0", "cuda:1"], measurement_batch_size=128)
        original_save = run.save_torch

        def interrupt_checkpoint(path, payload):
            original_save(path, payload)
            if (path.name == "checkpoint.pt" and payload.get("stage") == "cooperation"
                    and payload.get("stage_epoch") == 1):
                raise RuntimeError("test interruption after cooperation checkpoint")

        with tempfile.TemporaryDirectory() as temporary, patch.object(run, "write_plots"), \
                patch.dict(os.environ, GENERATOR_EVALUATOR_PROGRESS="0"):
            complete, interrupted = Path(temporary) / "complete", Path(temporary) / "interrupted"
            expected = run.run_cooperative_experiment(*fixture, complete, protocol, config, **kwargs)
            with patch.object(run, "save_torch", interrupt_checkpoint):
                with self.assertRaisesRegex(RuntimeError, "test interruption"):
                    run.run_cooperative_experiment(*fixture, interrupted, protocol, config, **kwargs)
            actual = run.run_cooperative_experiment(
                *fixture, interrupted, protocol, config, resume=True, **kwargs)
            self.assertEqual(actual["best_selection_delta"], expected["best_selection_delta"])
            left = torch.load(complete / "checkpoint.pt", map_location="cpu", weights_only=False)
            right = torch.load(interrupted / "checkpoint.pt", map_location="cpu", weights_only=False)
            torch.testing.assert_close(left["trainer_state"]["shared_latent"],
                                       right["trainer_state"]["shared_latent"], rtol=0, atol=0)
            for name in config.train_patterns:
                for key, value in left["models"][name].items():
                    # Cross-device gradient reductions may differ by one
                    # float32 ULP; masks, quality and latent remain identical.
                    torch.testing.assert_close(value, right["models"][name][key], rtol=1e-6, atol=1e-8)
            for name, rng_state in left["cuda_rng_states"].items():
                torch.testing.assert_close(rng_state, right["cuda_rng_states"][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
