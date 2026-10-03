"""Integration checks for real feedback, sealed test and exact resume."""
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from generator_evaluator import cooperative_run as run
from generator_evaluator.cooperative_data import (bank_input_fingerprint,
                                                  build_cooperative_fixture,
                                                  make_cooperative_test_task)
from generator_evaluator.data import InnerProtocol, tensor_hash


class CooperativeRunTests(unittest.TestCase):
    def test_small_cli_defaults_full_preset_and_explicit_overrides(self):
        args = run.make_parser().parse_args(["--out", "unused"])
        config, protocol, build = run.resolve_run_settings(args)
        self.assertIsNone(args.generator_devices)
        self.assertEqual(config.training_mode, "joint")
        self.assertEqual((config.width, config.heads, config.layers, config.noise_dim), (16, 2, 1, 4))
        self.assertEqual((config.cooperation_rounds, config.cooperation_updates, config.latent_lr), (5, 20, .001))
        self.assertEqual((protocol.steps, build["bank_steps"], build["teachers"], build["probe_count"]), (500, 200, 100, 32))
        self.assertEqual((config.auxiliary_budget, config.output_budgets), (0, ()))
        self.assertTrue(config.batch_children)
        self.assertFalse(config.tune_dense)
        explicit_devices = run.make_parser().parse_args([
            "--out", "unused", "--generator-devices", "auto"])
        self.assertEqual(explicit_devices.generator_devices, ["auto"])
        args = run.make_parser().parse_args(["--out", "unused", "--preset", "full"])
        config, protocol, build = run.resolve_run_settings(args)
        self.assertEqual((config.width, config.layers, protocol.steps, build["probe_count"]), (64, 2, 2000, 128))
        self.assertTrue(config.tune_dense)
        legacy = run.make_parser().parse_args(["--out", "unused", "--training-mode", "staged"])
        self.assertEqual(run.resolve_run_settings(legacy)[0].training_mode, "staged")
        args = run.make_parser().parse_args(["--out", "unused", "--steps", "100", "--width", "8", "--probe-count", "16"])
        config, protocol, build = run.resolve_run_settings(args)
        self.assertEqual((config.width, protocol.steps, build["probe_count"]), (8, 100, 16))

    def test_refresh_cadence_floor_is_configurable_and_validated(self):
        legacy = run.make_parser().parse_args(["--out", "unused"])
        self.assertEqual(run.resolve_run_settings(legacy)[0].minimum_refresh_every, 1)

        args = run.make_parser().parse_args([
            "--out", "unused", "--domain", "deepsets", "--refresh-every", "4",
            "--minimum-refresh-every", "2"])
        config, _, _ = run.resolve_run_settings(args)
        self.assertEqual((config.refresh_every, config.minimum_refresh_every), (4, 2))

        smoke = run.make_parser().parse_args([
            "--out", "unused", "--domain", "deepsets", "--minimum-refresh-every", "2", "--smoke"])
        config, _, _ = run.resolve_run_settings(smoke)
        self.assertEqual((config.refresh_every, config.minimum_refresh_every), (1, 1))

        for minimum, refresh in ((0, 2), (3, 2)):
            args = run.make_parser().parse_args([
                "--out", "unused", "--domain", "deepsets", "--refresh-every", str(refresh),
                "--minimum-refresh-every", str(minimum)])
            with self.subTest(minimum=minimum, refresh=refresh), self.assertRaises(ValueError):
                run.resolve_run_settings(args)

    def test_adaptive_refresh_cadence_respects_configured_floor(self):
        controller = object.__new__(run.CooperativeSearchController)
        controller.config = SimpleNamespace(minimum_refresh_every=2, gap_threshold=.1)
        controller.cadence = 4
        self.assertEqual(controller._adapt_refresh_cadence(.2), 4)
        self.assertEqual(controller.cadence, 2)
        controller.cadence = 3
        controller._adapt_refresh_cadence(.2)
        self.assertEqual(controller.cadence, 2)
        controller._adapt_refresh_cadence(.05)
        self.assertEqual(controller.cadence, 2)

    def test_pattern_cli_accepts_twelve_train_and_four_ordered_test_roles(self):
        heldouts = ("0010", "0100", "1011", "1101")
        training = tuple(pattern for pattern in
                         (f"{index:04b}" for index in range(16)) if pattern not in heldouts)
        args = run.make_parser().parse_args([
            "--out", "unused", "--preset", "full", "--train-patterns", *training,
            "--test-patterns", *heldouts])
        config, protocol, build = run.resolve_run_settings(args)
        self.assertEqual(config.train_patterns, training)
        self.assertEqual(config.effective_test_patterns, heldouts)
        self.assertEqual(config.test_pattern, heldouts[0])
        self.assertEqual((protocol.steps, build["support_count"], build["query_count"],
                          build["selection_count"]), (2000, 128, 128, 64))
        legacy = replace(run.CooperativeConfig(), test_pattern="0010")
        self.assertEqual(legacy.effective_test_patterns, ("0010",))

    def test_small_full_cycle_uses_batched_labels_and_keeps_density_inputs(self):
        fixture = build_cooperative_fixture(seed=417, bank_steps=1, teachers_per_pattern=5,
                    support_count=8, query_count=8, selection_count=4, k=8,
                    probe_count=32, batch_teachers=True)
        small = run.pattern_small_config(seed=417, k=8, evaluator_epochs=1,
                    width=8, heads=2, candidates=3, cooperation_rounds=1,
                    cooperation_updates=1, generator_epochs=1,
                    updates_per_epoch=2, generator_pretrain_epochs=1,
                    pretrain_updates_per_epoch=2, smoke=True)
        # Joint updates alternate the configured auxiliary generator K on odd
        # ordinals while keeping the real-acquisition budget independently zero.
        small = replace(small, output_budgets=(9,), auxiliary_budget=0)
        with tempfile.TemporaryDirectory() as tmp, patch.object(self, "fixture", fixture), patch.object(self, "config", small):
            result = self.invoke(tmp)
            out = Path(tmp)
            checkpoint = torch.load(out / "checkpoint.pt", weights_only=False)
            self.assertEqual(checkpoint["training_mode"], "joint")
            self.assertEqual(checkpoint["algorithm_version"], 5)
            self.assertTrue(checkpoint["pretraining_history"])
            self.assertTrue(all(row["reconstruction_scope"] == "teacher"
                                for row in checkpoint["pretraining_history"]))
            self.assertEqual({row["stage"] for row in checkpoint["history"]}, {run.STAGE_JOINT})
            self.assertTrue(all("direct_agreement_weight" in row and "elite_distillation_loss" in row
                                and "reconstruction_loss" in row for row in checkpoint["history"]))
            self.assertTrue(all(bank.diagnostics["feedback_rows_added"] > 0
                                for bank in checkpoint["banks"].values()))
            self.assertFalse(checkpoint["shared_latent_used"])
            self.assertFalse(checkpoint["trainer_state"]["latent_optimizer"]["state"])
            self.assertEqual(json.loads((out / "run_spec.json").read_text())
                             ["config"]["training_mode"], "joint")
            proposals = torch.load(out / "final_generator_proposals.pt", weights_only=False)
            self.assertIsNone(proposals["shared_latent"])
            self.assertEqual(proposals["latent_mode"], "independent_random_per_generator")
            self.assertEqual(torch.load(out / "proposals/epoch_0001.pt",
                                        weights_only=False)["stage"], run.STAGE_JOINT)
            self.assertEqual({row["output_k"] for row in checkpoint["history"]}, {8, 9})
            self.assertTrue(any(row["input_density"] != "mixed" for row in checkpoint["history"]))
            self.assertTrue(all(bank.tokens.shape[-1] == 76 for bank in checkpoint["banks"].values()))
            self.assertTrue(all({8,9,44,62,88} == set(bank.masks.sum((1,2)).int().tolist()) for bank in checkpoint["banks"].values()))
            self.assertFalse(any(row["origin"].startswith("auxiliary:") for row in checkpoint["replay"].records))
            self.assertEqual(json.loads((out / "dense_tuning.json").read_text())["label_measurements"], 0)
            checkpoint["replay"].validate()
            self.assertEqual(result["generators"], 2)
            self.assertIn("pattern:0101:test", result["final"])
            (out / "COMPLETE").unlink()
            before = (out / "frozen.pt").read_bytes()
            resumed = self.invoke(out, resume=True)
            self.assertEqual(resumed["best_selection_delta"], result["best_selection_delta"])
            self.assertEqual(before, (out / "frozen.pt").read_bytes())

    def test_joint_default_resumes_deterministically_and_rejects_mode_switch(self):
        config = replace(self.config, training_mode="joint", generator_epochs=2,
                         updates_per_epoch=1, refresh_every=1)
        with tempfile.TemporaryDirectory() as temporary, patch.object(self, "config", config):
            complete, interrupted = Path(temporary) / "complete", Path(temporary) / "interrupted"
            expected = self.invoke(complete)
            original = run.save_torch

            def interrupt_after_joint_checkpoint(path, payload):
                original(path, payload)
                if (path.name == "checkpoint.pt" and payload.get("stage") == run.STAGE_JOINT
                        and payload.get("stage_epoch") == 1):
                    raise RuntimeError("simulated stop after joint checkpoint")

            with patch.object(run, "save_torch", interrupt_after_joint_checkpoint):
                with self.assertRaisesRegex(RuntimeError, "simulated stop after joint checkpoint"):
                    self.invoke(interrupted)
            resumed = self.invoke(interrupted, resume=True)
            self.assertEqual(expected["best_selection_delta"], resumed["best_selection_delta"])
            left = torch.load(complete / "checkpoint.pt", weights_only=False)
            right = torch.load(interrupted / "checkpoint.pt", weights_only=False)
            self.assertEqual(left["history"], right["history"])
            self.assertEqual(left["stage"], run.STAGE_TRAINING_COMPLETE)
            self.assertEqual(right["stage"], run.STAGE_TRAINING_COMPLETE)
            for name in config.train_patterns:
                for key in left["models"][name]:
                    torch.testing.assert_close(left["models"][name][key],
                                               right["models"][name][key], rtol=0, atol=0)
                torch.testing.assert_close(left["own_rngs"][name],
                                           right["own_rngs"][name], rtol=0, atol=0)
            incompatible = replace(config, training_mode="staged")
            with patch.object(self, "config", incompatible), \
                    self.assertRaisesRegex(ValueError, "checkpoint training mode differs"):
                self.invoke(complete, resume=True)

    def test_legacy_v4_bootstrap_resume_rejects_default_joint_before_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            torch.save(dict(algorithm_version=4, trainer_state={}), out / "checkpoint.pt")
            config = replace(self.config, phase="bootstrap", training_mode="joint")
            with patch.object(run.MeasurementStore, "measure",
                              side_effect=AssertionError("resume must fail before fitting")):
                with self.assertRaisesRegex(ValueError, "version 4 checkpoints require --training-mode staged"):
                    run.run_cooperative_experiment(*copy.deepcopy(self.fixture), out,
                        self.protocol, config, resume=True)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.fixture = build_cooperative_fixture(seed=417, bank_steps=1, teachers_per_pattern=5,
                         support_count=8, query_count=8, selection_count=4, k=8)
        cls.config = run.CooperativeConfig(seed=417, training_mode="staged", k=8, generator_epochs=2,
                         updates_per_epoch=4, refresh_every=1, acquisition_budget=2,
                         cooperation_rounds=1, cooperation_updates=1,
                         auxiliary_budget=1, candidates=3, initial_random=2,
                         evaluator_epochs=1, width=8, heads=2, layers=1,
                         ensemble_members=2, bank_capacity=12, feedback_masks=1, smoke=True,
                         generator_pretrain_epochs=1, pretrain_updates_per_epoch=2)
        cls.protocol = InnerProtocol(steps=1, replicas=2, checkpoint_every=1, seed=417)

    def invoke(self, folder, **kwargs):
        def plots(out, *args, **kwargs):
            (out / "figures").mkdir(exist_ok=True)
            (out / "figures/CAPTIONS_RU.md").write_text("test")
        with patch.dict(os.environ, GENERATOR_EVALUATOR_PROGRESS="0"), patch.object(run, "write_plots", plots):
            return run.run_cooperative_experiment(*copy.deepcopy(self.fixture), folder,
                       self.protocol, self.config, dense_learning_rates=[.01], **kwargs)

    def write_generator_reconstruction(self, path, *, updates=7):
        banks = self.fixture[0]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9876)
            models = {
                name: run.DensityConditionedGenerator(
                    bank.tokens.shape[-1], self.config.features, self.config.hidden,
                    self.config.width, self.config.heads, self.config.layers,
                    self.config.noise_dim, target_k=self.config.k).state_dict()
                for name, bank in banks.items()
            }
        artifact = dict(models=models,
                        bank_hashes={name: bank_input_fingerprint(bank)
                                     for name, bank in banks.items()},
                        updates=updates, test_used=False)
        torch.save(artifact, path)
        return artifact

    def test_generator_pretrained_import_skips_reconstruction_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_path = root / "generator_reconstruction.pt"
            self.write_generator_reconstruction(artifact_path, updates=7)
            out = root / "search"

            result = self.invoke(out, generator_pretrained_from=artifact_path)

            self.assertEqual(result["generator_pretraining_updates"], 0)
            self.assertEqual(result["generator_pretraining"], dict(
                origin="imported", source_path=str(artifact_path.resolve()),
                source_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                source_updates=7, updates_this_run=0,
                optimizer_states_imported=False,
                optimizer_initialization="fresh_adam"))
            run_spec = json.loads((out / "run_spec.json").read_text())
            self.assertEqual(run_spec["generator_pretrained_from"]["sha256"],
                             hashlib.sha256(artifact_path.read_bytes()).hexdigest())
            self.assertEqual(run_spec["generator_pretrained_from"]["updates"], 7)
            imported = torch.load(out / "generator_reconstruction.pt", weights_only=False)
            self.assertEqual(imported["updates"], 7)
            self.assertFalse(imported["test_used"])
            self.assertEqual(imported["origin"]["kind"], "imported")
            self.assertEqual(imported["origin"]["sha256"],
                             hashlib.sha256(artifact_path.read_bytes()).hexdigest())
            self.assertFalse(imported["optimizer_states_imported"])
            self.assertEqual(imported["optimizer_initialization"], "fresh_adam")
            checkpoint = torch.load(out / "checkpoint.pt", weights_only=False)
            self.assertEqual(checkpoint["pretraining_history"], [])
            self.assertEqual(checkpoint["generator_pretraining"]["source_updates"], 7)
            allowed_budgets = {self.config.k, *self.config.output_budgets}
            self.assertTrue(set(checkpoint["budgets"].values()).issubset(allowed_budgets))
            for optimizer in checkpoint["optimizers"].values():
                steps = [int(state["step"]) for state in optimizer["state"].values()]
                self.assertTrue(steps)
                self.assertEqual(max(steps), self.config.generator_epochs *
                                 self.config.updates_per_epoch + self.config.cooperation_updates)

    def test_generator_pretrained_import_rejects_roles_banks_test_leakage_and_shapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self.write_generator_reconstruction(root / "base.pt")
            variants = []
            wrong_roles = copy.deepcopy(base)
            wrong_roles["models"].pop(self.config.train_patterns[0])
            variants.append(("roles", wrong_roles, "roles must exactly match"))
            wrong_banks = copy.deepcopy(base)
            wrong_banks["bank_hashes"][self.config.train_patterns[0]] = "changed"
            variants.append(("banks", wrong_banks, "fingerprints do not match"))
            leaked_test = copy.deepcopy(base)
            leaked_test["test_used"] = True
            variants.append(("test", leaked_test, "test_used=False"))
            wrong_shape = copy.deepcopy(base)
            role = self.config.train_patterns[0]
            key = next(iter(wrong_shape["models"][role]))
            wrong_shape["models"][role][key] = wrong_shape["models"][role][key][:-1]
            variants.append(("shape", wrong_shape, "state shape mismatch"))

            for name, payload, message in variants:
                with self.subTest(artifact=name):
                    artifact_path = root / f"{name}.pt"
                    torch.save(payload, artifact_path)
                    out = root / f"out-{name}"
                    with self.assertRaisesRegex(ValueError, message):
                        self.invoke(out, generator_pretrained_from=artifact_path)
                    self.assertFalse((out / "run_spec.json").exists())
                    self.assertFalse((out / "children").exists())

    def test_generator_pretrained_import_rejects_bootstrap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_path = root / "generator_reconstruction.pt"
            self.write_generator_reconstruction(artifact_path)
            with patch.object(self, "config", replace(self.config, phase="bootstrap")):
                with self.assertRaisesRegex(ValueError, "search runs"):
                    self.invoke(root / "bootstrap", generator_pretrained_from=artifact_path)

    def test_generator_pretrained_resume_revalidates_source_and_restores_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_path = root / "generator_reconstruction.pt"
            self.write_generator_reconstruction(artifact_path, updates=7)
            complete, interrupted = root / "complete", root / "interrupted"
            expected = self.invoke(complete, generator_pretrained_from=artifact_path)
            original = run.save_torch

            def interrupt_after_quality_checkpoint(path, payload):
                original(path, payload)
                if (path.name == "checkpoint.pt" and payload.get("stage") == run.STAGE_QUALITY
                        and payload.get("stage_epoch") == 0):
                    raise RuntimeError("simulated imported-run interruption")

            with patch.object(run, "save_torch", interrupt_after_quality_checkpoint):
                with self.assertRaisesRegex(RuntimeError, "imported-run interruption"):
                    self.invoke(interrupted, generator_pretrained_from=artifact_path)
            artifact_before = (interrupted / "generator_reconstruction.pt").read_bytes()
            resumed = self.invoke(interrupted, resume=True,
                                  generator_pretrained_from=artifact_path)
            self.assertEqual(expected["best_selection_delta"], resumed["best_selection_delta"])
            self.assertEqual(artifact_before,
                             (interrupted / "generator_reconstruction.pt").read_bytes())
            state = torch.load(interrupted / "checkpoint.pt", weights_only=False)
            self.assertEqual(state["generator_pretraining"]["source_updates"], 7)
            self.assertEqual(state["pretraining_history"], [])

    def test_generator_pretrained_cli_option_is_parsed(self):
        args = run.make_parser().parse_args([
            "--out", "unused", "--generator-pretrained-from", "source/generator_reconstruction.pt"])
        self.assertEqual(args.generator_pretrained_from,
                         Path("source/generator_reconstruction.pt"))

    def test_pipeline_feedback_roles_and_real_elites(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            test_calls = []
            def sealed(spec):
                self.assertTrue((out / "frozen.pt").exists())
                checkpoint = torch.load(out / "checkpoint.pt", weights_only=False)
                self.assertEqual(checkpoint["stage"], run.STAGE_TRAINING_COMPLETE)
                self.assertFalse(any(row["task_split"] == "test" for row in checkpoint["replay"].records))
                test_calls.append(spec["test_pattern"])
                return make_cooperative_test_task(spec)
            result = self.invoke(out, test_factory=sealed)
            self.assertEqual(test_calls, ["0101"])
            self.assertEqual(result["generators"], 2)
            state = torch.load(out / "checkpoint.pt", weights_only=False)
            self.assertEqual(state["stage"], run.STAGE_TRAINING_COMPLETE)
            self.assertEqual(set(state["models"]), {"0001", "0011"})
            pretrain_steps = self.config.generator_pretrain_epochs * self.config.pretrain_updates_per_epoch
            for name, optimizer in state["optimizers"].items():
                own_history = [row for row in state["history"] if row["pattern"] == name]
                quality_rows = [row for row in own_history if row["stage"] == run.STAGE_QUALITY]
                cooperation_rows = [row for row in own_history if row["stage"] == run.STAGE_COOPERATION]
                self.assertTrue(quality_rows)
                self.assertTrue(cooperation_rows)
                self.assertTrue(all("direct_agreement_weight" not in row for row in quality_rows))
                self.assertTrue(all(row["direct_agreement_weight"] > 0 for row in cooperation_rows))
                self.assertTrue(all("reconstruction_loss" in row for row in own_history))
                self.assertTrue(all(row["reconstruction_scope"] == "teacher"
                                    for row in cooperation_rows))
                # Cooperation combines quality, agreement and measured
                # distillation before one Adam step per generator.
                expected_steps = pretrain_steps + len(quality_rows) + len(cooperation_rows)
                steps = [int(value["step"]) for value in optimizer["state"].values()]
                self.assertTrue(steps)
                self.assertEqual(max(steps), expected_steps)
                self.assertTrue(all(step <= expected_steps for step in steps))
            self.assertTrue(any(not torch.equal(state["models"]["0001"][key], state["models"]["0011"][key])
                                for key in state["models"]["0001"]))
            self.assertEqual({row["output_k"] for row in state["history"]}, {8, 9, 44, 62})
            self.assertTrue(any(row["input_density"] != "mixed" for row in state["history"]))
            quality_epochs = {row["epoch"] for row in state["history"] if row["stage"] == run.STAGE_QUALITY}
            cooperation_epochs = {row["epoch"] for row in state["history"]
                                  if row["stage"] == run.STAGE_COOPERATION}
            self.assertEqual(quality_epochs, {1, 2})
            self.assertEqual(cooperation_epochs, {self.config.generator_epochs + 1})
            self.assertEqual(state["trainer_state"]["shared_latent"].shape, (self.config.noise_dim,))
            self.assertFalse(torch.equal(state["trainer_state"]["shared_latent"],
                                         state["initial_shared_latent"]))
            self.assertTrue(state["trainer_state"]["latent_optimizer"]["state"])
            self.assertEqual(len(state["evaluator_history"]), 1 + len(state["refresh_history"]))
            replay = state["replay"]
            replay.validate()
            train_paths = {row["artifact_path"]: row for row in replay.records if row["split"] == "train"}
            for name, bank in state["banks"].items():
                self.assertEqual(set(bank.masks.sum((1, 2)).long().tolist()), {8, 9, 44, 62, 88})
                self.assertGreater(bank.diagnostics["feedback_rows_added"], 0)
                self.assertLessEqual(bank.tokens.shape[1], self.config.bank_capacity)
                feedback = [s for s in bank.states if s["source"]["kind"] == "feedback"]
                self.assertTrue(feedback)
                for teacher in feedback:
                    source = teacher["source"]
                    self.assertEqual(source["task_id"], f"pattern:{name}")
                    self.assertIn(source["artifact_path"], train_paths)
                    payload = torch.load(source["artifact_path"], weights_only=False)
                    for key, value in teacher["state_dict"].items():
                        torch.testing.assert_close(value, payload["result"]["state_dict"][source["replica"]][key])
            masks, quality, rows = run._real_candidates(replay, self.fixture[1], 8)
            self.assertEqual(len(masks), len(quality))
            self.assertTrue(all(len(pair) == 2 and all(row["split"] == "train" for row in pair) for pair in rows))
            self.assertIn("pattern:0101:test", result["final"])
            self.assertFalse(result["test_used_for_training"])
            quality_proposal = torch.load(out / "proposals/quality_0001.pt", weights_only=False)
            proposal = torch.load(out / "proposals/cooperation_0001.pt", weights_only=False)
            self.assertNotIn("paired_exact_agreement", quality_proposal["overlap"])
            self.assertIn("paired_exact_agreement", proposal["overlap"])
            families = [set(proposal["proposal_trace"]["generators"][name]["topology_ids"])
                        for name in self.config.train_patterns]
            self.assertEqual(proposal["overlap"]["intersection_count"], len(families[0] & families[1]))
            cooperation_refresh = next(row for row in result["refreshes"] if row["stage"] == "cooperation")
            self.assertEqual(cooperation_refresh["proposal_overlap"], proposal["overlap"])

    def test_best_common_mask_is_selected_at_each_refresh(self):
        original_select = run.CooperativeSearchController.select
        calls = []

        def select(controller, masks, epoch, *, stage="initial"):
            calls.append(epoch)
            return original_select(controller, masks, epoch, stage=stage)

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(run.CooperativeSearchController, "select", select):
            self.invoke(tmp)
            self.assertEqual(calls, [0, 1, 2, 3])
            state = torch.load(Path(tmp) / "checkpoint.pt", weights_only=False)
            self.assertTrue(state["best_models"])
            self.assertTrue(any(row["origin"] == "selection"
                                for row in state["replay"].records))

    def test_pattern_validation_accepts_four_initializations(self):
        run._validate_pattern_inputs(*self.fixture, self.config,
                                     replace(self.protocol, replicas=4))
        with self.assertRaises(ValueError):
            run._validate_pattern_inputs(*self.fixture, self.config,
                                         replace(self.protocol, replicas=1))

    def test_deepsets_cli_supports_multiple_train_and_heldout_tasks(self):
        args = run.make_parser().parse_args([
            "--out", "unused", "--domain", "deepsets",
            "--train-task-count", "6", "--test-task-count", "4"])
        config, protocol, build = run.resolve_run_settings(args)
        self.assertEqual(config.train_patterns, tuple(str(index) for index in range(6)))
        self.assertEqual(config.test_task_count, 4)
        self.assertEqual(build["train_task_count"], 6)
        self.assertEqual(build["test_task_count"], 4)
        for train_count, test_count in ((1, 4), (6, 0)):
            args = run.make_parser().parse_args([
                "--out", "unused", "--domain", "deepsets",
                "--train-task-count", str(train_count), "--test-task-count", str(test_count)])
            with self.assertRaises(ValueError):
                run.resolve_run_settings(args)

    def test_multiple_sealed_patterns_materialize_only_after_freeze(self):
        heldouts = ("0010", "0100", "1011", "1101")
        fixture = build_cooperative_fixture(
            train_patterns=("0001", "0011"), test_pattern=heldouts,
            seed=417, bank_steps=1, teachers_per_pattern=5, support_count=8,
            query_count=8, selection_count=4, k=8)
        config = replace(
            self.config, test_patterns=heldouts, test_pattern=heldouts[0],
            generator_epochs=1, updates_per_epoch=1, acquisition_budget=1,
            candidates=2, initial_random=1, phase="bootstrap")
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(self, "fixture", fixture), patch.object(self, "config", config):
            bootstrap_out = Path(temporary) / "bootstrap"
            bootstrap = self.invoke(bootstrap_out)
            self.assertFalse(bootstrap["test_materialized"])
            self.assertEqual(bootstrap["test_patterns"], list(heldouts))
            self.assertFalse((bootstrap_out / "test_tasks.pt").exists())

            search_config = replace(config, phase="search")
            with patch.object(self, "config", search_config):
                search_out = Path(temporary) / "search"
                calls = []

                def sealed_factory(spec):
                    self.assertTrue((search_out / "frozen.pt").is_file())
                    checkpoint = torch.load(search_out / "checkpoint.pt", weights_only=False)
                    self.assertFalse(any(row.get("task_split") == "test"
                                         for row in checkpoint["replay"].records))
                    calls.append(tuple(spec["test_patterns"]))
                    return run.make_cooperative_test_tasks(spec)

                result = self.invoke(search_out, test_factory=sealed_factory)
            self.assertEqual(calls, [heldouts])
            self.assertEqual(result["test_patterns"], list(heldouts))
            self.assertEqual(result["test_task_ids"], [f"pattern:{pattern}:test" for pattern in heldouts])
            self.assertTrue(all(task_id in result["final"] for task_id in result["test_task_ids"]))
            self.assertEqual(len(torch.load(search_out / "test_tasks.pt", weights_only=False)), 4)

    def test_selection_uses_every_task_label_for_each_mask(self):
        controller = object.__new__(run.CooperativeSearchController)
        controller.config = self.config
        controller.selection_tasks = [type("Task", (), {"task_id": str(index)})() for index in range(3)]
        controller.dense_rows = {str(index): {"quality": 0.} for index in range(3)}
        controller.models, controller.banks = {}, {}
        controller.ensemble = torch.nn.Linear(1, 1)
        controller.best_cost = float("inf")
        first = torch.zeros(11, 8)
        first[0] = 1
        second = torch.zeros(11, 8)
        second[:8, 0] = 1
        labels = [({"quality": loss}, {}) for loss in (0., 0., 1., .5, .5, .5)]
        with patch.object(controller, "measure_many", return_value=labels, create=True) as measure:
            controller.select([first, second], epoch=7)
        self.assertEqual(len(measure.call_args.args[0]), 6)
        self.assertEqual(controller.best_cost, .5)
        torch.testing.assert_close(controller.best_mask, second)

    def test_composite_selection_retains_actual_mean_and_worst_deltas(self):
        controller = object.__new__(run.CooperativeSearchController)
        controller.config = replace(self.config, quality_objective="mean_positive_worst")
        controller.selection_tasks = [type("Task", (), {"task_id": str(i)})() for i in range(3)]
        controller.dense_rows = {str(i): {"quality": 1.} for i in range(3)}
        controller.models, controller.banks = {}, {}
        controller.ensemble = torch.nn.Linear(1, 1)
        controller.best_cost = float("inf")
        first = torch.zeros(11, 8)
        first[0] = 1
        second = torch.zeros(11, 8)
        second[:8, 0] = 1
        labels = [({"quality": loss}, {}) for loss in (.6, .6, 1.02, .95, .95, .95)]
        with patch.object(controller, "measure_many", return_value=labels, create=True):
            controller.select([first, second], epoch=7)
        torch.testing.assert_close(controller.best_mask, first)
        self.assertAlmostEqual(controller.best_worst_delta, .02)
        self.assertAlmostEqual(controller.best_mean_delta, -.26)
        self.assertAlmostEqual(controller.best_cost, -.24)
        args = run.make_parser().parse_args([
            "--out", "unused", "--quality-objective", "mean_positive_worst"])
        self.assertEqual(run.resolve_run_settings(args)[0].quality_objective, "mean_positive_worst")

    def test_composite_full_cycle_freezes_consistent_objective_metrics(self):
        config = replace(self.config, quality_objective="mean_positive_worst")
        with tempfile.TemporaryDirectory() as temporary, patch.object(self, "config", config):
            summary = self.invoke(temporary)
            frozen = torch.load(Path(temporary) / "frozen.pt", map_location="cpu", weights_only=False)
            frozen_json = json.loads((Path(temporary) / "frozen.json").read_text())
        self.assertEqual(frozen["quality_objective"], config.quality_objective)
        for key in ("best_selection_delta", "best_selection_cost", "best_selection_mean_delta"):
            self.assertEqual(frozen[key], summary[key])
        self.assertEqual(frozen_json["selection_delta"], summary["best_selection_delta"])
        self.assertAlmostEqual(summary["best_selection_cost"],
            summary["best_selection_mean_delta"] + max(0., summary["best_selection_delta"]))

    def test_resume_finishes_pretraining_before_initializing_search(self):
        config = replace(self.config, generator_pretrain_epochs=2)
        with tempfile.TemporaryDirectory() as tmp, patch.object(self, "config", config):
            complete, interrupted = Path(tmp) / "complete", Path(tmp) / "interrupted"
            expected = self.invoke(complete)
            original = run.save_torch
            def crash_after_pretraining_epoch(path, payload):
                original(path, payload)
                if (path.name == "checkpoint.pt" and payload.get("stage") == "reconstruction"
                        and payload.get("pretraining_epoch") == 1):
                    raise RuntimeError("interrupted pretraining")
            with patch.object(run, "save_torch", crash_after_pretraining_epoch):
                with self.assertRaisesRegex(RuntimeError, "interrupted pretraining"):
                    self.invoke(interrupted)
            resumed = self.invoke(interrupted, resume=True)
            self.assertEqual(expected["best_selection_delta"], resumed["best_selection_delta"])
            left = torch.load(complete / "checkpoint.pt", weights_only=False)
            right = torch.load(interrupted / "checkpoint.pt", weights_only=False)
            self.assertEqual(left["pretraining_history"], right["pretraining_history"])
            self.assertEqual(right["pretraining_epoch"], 2)
            self.assertTrue(right["initialization_done"])
            self.assertEqual(right["stage"], run.STAGE_TRAINING_COMPLETE)
            for name in config.train_patterns:
                for key in left["models"][name]:
                    torch.testing.assert_close(left["models"][name][key], right["models"][name][key],
                                               rtol=0, atol=0)

    def test_atomic_checkpoint_resume_and_immutable_freeze(self):
        config = replace(self.config, cooperation_rounds=2, cooperation_updates=1)
        with tempfile.TemporaryDirectory() as tmp, patch.object(self, "config", config):
            complete, interrupted = Path(tmp) / "complete", Path(tmp) / "interrupted"
            expected = self.invoke(complete)
            original = run.save_torch
            def crash_after_checkpoint(path, payload):
                original(path, payload)
                if (path.name == "checkpoint.pt" and payload.get("stage") == run.STAGE_COOPERATION
                        and payload.get("stage_epoch") == 1):
                    raise RuntimeError("simulated stop after atomic checkpoint")
            with patch.object(run, "save_torch", crash_after_checkpoint):
                with self.assertRaisesRegex(RuntimeError, "simulated stop"):
                    self.invoke(interrupted)
            # Secondary exports can be corrupt/newer; checkpoint is authoritative.
            (interrupted / "replay.pt").write_bytes(b"stale secondary export")
            resumed = self.invoke(interrupted, resume=True)
            self.assertEqual(expected["best_selection_delta"], resumed["best_selection_delta"])
            for path in ("checkpoint.pt", "frozen.pt"):
                left = torch.load(complete / path, weights_only=False)
                right = torch.load(interrupted / path, weights_only=False)
                for name in self.config.train_patterns:
                    for key in left["models"][name]:
                        torch.testing.assert_close(left["models"][name][key], right["models"][name][key], rtol=0, atol=0)
                    self.assertEqual(tensor_hash(left["banks"][name].tokens), tensor_hash(right["banks"][name].tokens))
            left = torch.load(complete / "checkpoint.pt", weights_only=False)
            right = torch.load(interrupted / "checkpoint.pt", weights_only=False)
            self.assertEqual(right["stage"], run.STAGE_TRAINING_COMPLETE)
            self.assertEqual(left["history"], right["history"])
            torch.testing.assert_close(left["trainer_state"]["shared_latent"],
                                       right["trainer_state"]["shared_latent"], rtol=0, atol=0)
            left_latent_opt = left["trainer_state"]["latent_optimizer"]
            right_latent_opt = right["trainer_state"]["latent_optimizer"]
            self.assertEqual(left_latent_opt["param_groups"], right_latent_opt["param_groups"])
            self.assertEqual(left_latent_opt["state"].keys(), right_latent_opt["state"].keys())
            for key in left_latent_opt["state"]:
                for field, value in left_latent_opt["state"][key].items():
                    other = right_latent_opt["state"][key][field]
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, other, rtol=0, atol=0)
                    else:
                        self.assertEqual(value, other)
            for name in config.train_patterns:
                self.assertTrue(torch.equal(left["own_rngs"][name], right["own_rngs"][name]))
            for name in expected["final"]["pattern:0101:test"]:
                self.assertEqual(expected["final"]["pattern:0101:test"][name]["query_bce"],
                                 resumed["final"]["pattern:0101:test"][name]["query_bce"])
            # Simulate interruption inside final evaluation after freezing.
            (interrupted / "COMPLETE").unlink()
            before = (interrupted / "frozen.pt").read_bytes()
            again = self.invoke(interrupted, resume=True)
            self.assertEqual(before, (interrupted / "frozen.pt").read_bytes())
            self.assertEqual(again["best_selection_delta"], resumed["best_selection_delta"])

    def test_bad_roles_fail_before_labels(self):
        banks, train, selection, sealed = copy.deepcopy(self.fixture)
        selection[0].query_ids = train[0].query_ids.clone()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "query observations overlap"):
                run.run_cooperative_experiment(banks, train, selection, sealed, tmp,
                                              self.protocol, self.config)
            self.assertFalse((Path(tmp) / "children").exists())

    def test_test_factory_cannot_import_training_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            def leaking_factory(spec):
                task = make_cooperative_test_task(spec)
                task.support_ids = self.fixture[1][0].support_ids.clone()
                return task
            with self.assertRaisesRegex(ValueError, "frozen third-pattern role"):
                self.invoke(out, test_factory=leaking_factory)
            self.assertTrue((out / "frozen.pt").is_file())
            self.assertFalse((out / "COMPLETE").exists())

    def test_selection_must_reuse_actual_support_data(self):
        banks, train, selection, sealed = copy.deepcopy(self.fixture)
        selection[0].y_support = 1 - selection[0].y_support
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "reuse its own train support/context"):
                run.run_cooperative_experiment(banks, train, selection, sealed, tmp,
                                              self.protocol, self.config)
            self.assertFalse((Path(tmp) / "children").exists())

    def test_changed_build_settings_reject_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = dict(bank_steps=1, teachers=5, support_count=8, query_count=8, selection_count=4)
            self.invoke(tmp, build_settings=settings)
            with self.assertRaisesRegex(ValueError, "different code/inputs/settings"):
                self.invoke(tmp, resume=True, build_settings={**settings, "teachers": 6})


if __name__ == "__main__":
    unittest.main()
