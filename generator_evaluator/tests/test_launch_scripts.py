"""Validate launch arguments without starting an experiment or activating CUDA."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from generator_evaluator.cooperative_run import make_parser, resolve_run_settings


class JointLaunchScriptTests(unittest.TestCase):
    def commands(self, script, **overrides):
        scripts = Path(__file__).resolve().parents[2] / "sh_scripts"
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            shutil.copy2(scripts / script, folder / script)
            (folder / "_common.sh").write_text(
                'GE_SEED="${GE_SEED:-4100}"\nGE_STAMP=test\n'
                'ge_run() { "$SCRIPT_TEST_PYTHON" -c '
                "'import json,sys; print(\"ARGS=\"+json.dumps(sys.argv[1:]))' "
                '"$@"; }\n')
            env = {key: value for key, value in os.environ.items()
                   if key not in {"WARM_START_FROM", "TRAIN_PATTERNS", "TEST_PATTERNS",
                                  "GE_OUT", "COOPERATION_ROUNDS", "COOPERATION_UPDATES",
                                  "REFRESH_EVERY", "MINIMUM_REFRESH_EVERY", "AUXILIARY_BUDGET",
                                  "FIXED_TEST_FROM", "TRAIN_TASKS", "TEST_TASKS",
                                  "EVALUATOR_BATCH_SIZE", "ELITE_DISTILLATION_WEIGHT",
                                  "ELITE_LIMIT", "QUALITY_OBJECTIVE", "AGREEMENT_RAMP_EPOCHS",
                                  "GENERATOR_PRETRAIN_EPOCHS", "RECONSTRUCTION_WEIGHT",
                                  "RECONSTRUCTION_BATCH_SIZE", "GENERATOR_EPOCHS",
                                  "UPDATES_PER_EPOCH"}}
            env.update(SCRIPT_TEST_PYTHON=sys.executable, **overrides)
            output = subprocess.run(["bash", str(folder / script)], env=env,
                                    check=True, capture_output=True, text=True).stdout
        return [json.loads(line[5:]) for line in output.splitlines() if line.startswith("ARGS=")]

    def settings(self, command):
        self.assertEqual(command[:4], ["python", "-u", "-m", "generator_evaluator.cooperative_run"])
        args = make_parser().parse_args(command[4:])
        config, protocol, build = resolve_run_settings(args)
        self.assertEqual(args.measurement_devices, ["auto"])
        self.assertEqual(args.generator_devices, ["auto"])
        return config, protocol, build

    def test_both_launchers_bootstrap_then_joint_search(self):
        for script in ("pattern.sh", "deepsets.sh"):
            with self.subTest(script=script):
                commands = self.commands(script, GPU_IDS="1 2 3")
                self.assertEqual(len(commands), 2)
                bootstrap, _, _ = self.settings(commands[0])
                search, _, _ = self.settings(commands[1])
                self.assertEqual(bootstrap.phase, "bootstrap")
                self.assertEqual(bootstrap.generator_pretrain_epochs, 0)
                self.assertEqual(search.phase, "search")
                self.assertGreater(search.generator_pretrain_epochs, 0)
                for config in (bootstrap, search):
                    self.assertEqual(config.training_mode, "joint")
                    self.assertEqual(config.quality_objective, "worst")
                self.assertEqual((search.generator_epochs, search.updates_per_epoch), (20, 10))
                self.assertEqual(search.generator_pretrain_epochs, 5)
                self.assertEqual(search.elite_limit, 8)
                self.assertEqual(search.elite_distillation_weight, .1)
                self.assertEqual(search.reconstruction_weight, .1)
                self.assertEqual(search.agreement_ramp_epochs, 5)
                for command in commands:
                    self.assertNotIn("--cooperation-rounds", command)
                    self.assertNotIn("--cooperation-updates", command)
                    self.assertNotIn("--latent-lr", command)
                if script == "pattern.sh":
                    self.assertEqual(search.minimum_refresh_every, 1)

    def test_deepsets_refresh_floor_and_auxiliary_budget_defaults_and_overrides(self):
        commands = self.commands("deepsets.sh")
        self.assertEqual(len(commands), 2)
        for command in commands:
            config, _, _ = self.settings(command)
            self.assertEqual((config.refresh_every, config.minimum_refresh_every,
                              config.auxiliary_budget), (2, 2, 0))

        commands = self.commands("deepsets.sh", REFRESH_EVERY="2",
                                 MINIMUM_REFRESH_EVERY="1", AUXILIARY_BUDGET="2")
        for command in commands:
            config, _, _ = self.settings(command)
            self.assertEqual((config.refresh_every, config.minimum_refresh_every,
                              config.auxiliary_budget), (2, 1, 2))

    def test_pattern_roles_and_joint_training_overrides_reach_search(self):
        heldouts = ("0010", "0100", "1011", "1101")
        training = tuple(f"{index:04b}" for index in range(16) if f"{index:04b}" not in heldouts)
        commands = self.commands("pattern.sh", TRAIN_PATTERNS=" ".join(training),
                                 TEST_PATTERNS=" ".join(heldouts),
                                 GENERATOR_EPOCHS="7", UPDATES_PER_EPOCH="9",
                                 ELITE_LIMIT="1", ELITE_DISTILLATION_WEIGHT="0.5")
        for command in commands:
            config, _, _ = self.settings(command)
            self.assertEqual(config.train_patterns, training)
            self.assertEqual(config.effective_test_patterns, heldouts)
        search, _, _ = self.settings(commands[1])
        self.assertEqual((search.generator_epochs, search.updates_per_epoch), (7, 9))
        self.assertEqual((search.elite_limit, search.elite_distillation_weight), (1, .5))

    def test_warm_start_skips_only_bootstrap(self):
        for script in ("pattern.sh", "deepsets.sh"):
            commands = self.commands(script, WARM_START_FROM="/saved/bootstrap")
            self.assertEqual(len(commands), 1)
            self.assertIn("/saved/bootstrap", commands[0])
            self.assertEqual(self.settings(commands[0])[0].phase, "search")

    def test_deepsets_fixed_four_tests_and_twelve_train_apply_to_both_phases(self):
        commands = self.commands("deepsets.sh", TRAIN_TASKS="12", TEST_TASKS="4",
                                 FIXED_TEST_FROM="/previous/search", EVALUATOR_BATCH_SIZE="128",
                                 ELITE_DISTILLATION_WEIGHT="0.5")
        self.assertEqual(len(commands), 2)
        for command in commands:
            args = make_parser().parse_args(command[4:])
            config, _, _ = self.settings(command)
            self.assertEqual(len(config.train_patterns), 12)
            self.assertEqual(config.test_task_count, 4)
            self.assertEqual(args.fixed_test_from, Path("/previous/search"))
            self.assertEqual(config.evaluator_batch_size, 128)
            self.assertEqual(config.elite_distillation_weight, .5)


if __name__ == "__main__":
    unittest.main()
