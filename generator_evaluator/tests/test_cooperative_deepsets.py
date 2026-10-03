"""Small synthetic contracts for own-task DeepSets banks and feedback."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from deepsets_vaae.core import Split
from generator_evaluator.adapters import _cost_vectors
from generator_evaluator.cooperative_deepsets import (
    _fixed_test_pools,
    append_deepsets_feedback,
    build_cooperative_deepsets_fixture,
    extract_deepsets_functional_token,
    make_cooperative_deepsets_test_tasks,
    validate_cooperative_deepsets_inputs,
    validate_deepsets_test_tasks,
)
from generator_evaluator.data import InnerProtocol, TaskData, support_context


def _split(first_id: int, count: int, seed: int) -> Split:
    generator = torch.Generator().manual_seed(seed)
    return Split(torch.rand(count, 784, generator=generator), torch.arange(count) % 10,
                 torch.arange(first_id, first_id + count))


def _data() -> dict[str, Split]:
    return {"source_train": _split(0, 120, 1),
            "target_train": _split(1_000, 120, 2),
            "source_validation": _split(2_000, 120, 3),
            "target_validation": _split(3_000, 120, 4),
            "target_test": _split(4_000, 120, 5)}


def _base_task(task_id: str, cost: torch.Tensor, seed: int) -> TaskData:
    generator = torch.Generator().manual_seed(seed)
    xs, ys = torch.rand(2, 5, 784, generator=generator), torch.rand(2, generator=generator)
    xq, yq = torch.rand(2, 5, 784, generator=generator), torch.rand(2, generator=generator)
    ids = torch.arange(seed * 100, seed * 100 + 10).reshape(2, 5)
    query_ids = torch.arange(seed * 100 + 10, seed * 100 + 20).reshape(2, 5)
    return TaskData(task_id, "train", xs, ys, xq, yq, support_context(xs.mean(1), ys),
                    ids, query_ids, {"family": "deepsets", "domain": "deepsets",
                                     "costs": cost.tolist()})


class _FakeStore:
    """Return complete deterministic child artifacts without training a model."""

    def __init__(self, out, replay, device, *, devices, batch_size):
        self.out, self.replay = Path(out), replay
        self.untracked_path = self.out / "children" / "preserve-untracked.txt"
        self.untracked_path.parent.mkdir(parents=True, exist_ok=True)
        self.untracked_path.write_text("not a candidate artifact", encoding="utf-8")

    def measure_many(self, entries, *, desc, initialization_seeds):
        from generator_evaluator.artifacts import save_torch

        output = []
        for (mask, task, origin), init_seed in zip(entries, initialization_seeds):
            candidate_id = int(origin.rsplit(":", 1)[1])
            value, hidden = (candidate_id + 1) / 100., mask.shape[1]
            state = {"weight": torch.full((1, 1, 784, hidden), value),
                     "masks": mask[None, None].clone(),
                     "bias": torch.full((1, 1, hidden), value / 2),
                     "readout": torch.full((1, 1, hidden), .5 + value),
                     "per_image_offset": torch.tensor([[value / 3]])}
            result = {"label_source": "fresh_terminal_query", "fixed_horizon": True,
                      "protocol_id": self.replay.protocol.fingerprint, "task_id": task.task_id,
                      "replica_losses": [float(1000 - candidate_id)], "seeds": [str(init_seed)],
                      "plateau_flags": [False], "state_dict": state,
                      "optimizer_state": {"state": {}, "param_groups": []},
                      "history": {"steps": torch.tensor([0, 1]),
                                  "queryNMSE": torch.zeros(2, 1, 1)}}
            path = self.out / "children" / f"candidate_{candidate_id}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_torch(path, {"result": result})
            record = self.replay.append(mask, task, result, origin=origin, artifact_path=path)
            output.append((record, result))
        return output

    def close(self):
        pass


class _Config:
    domain = "deepsets"
    features = 784
    hidden = 2
    k = 5
    test_pattern = "heldout"

    def __init__(self, train_task_count=2, test_task_count=2):
        self.train_patterns = tuple(str(index) for index in range(train_task_count))
        self.test_task_count = test_task_count


class CooperativeDeepSetsTests(unittest.TestCase):
    def fixture(self, train_task_count=2, test_task_count=2, *, data=None,
                data_root=None, fixed_test_spec=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        data, costs = _data() if data is None else data, _cost_vectors(19, 6)
        base = [_base_task(f"deepsets:train:{index}", costs[index], 10 + index)
                for index in range(2)]
        base_spec = {"family": "deepsets", "costs": costs[4:].tolist()}
        patches = [
            patch("generator_evaluator.cooperative_deepsets.make_deepsets_tasks",
                  lambda *args, **kwargs: (base, base_spec)),
            patch("deepsets_vaae.core.load_data", lambda *args, **kwargs: data),
            patch("generator_evaluator.cooperative_deepsets.ParallelMeasurementStore", _FakeStore),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.bank_root = root / "out" / "banks"
        return build_cooperative_deepsets_fixture(
            root / "fake-data" if data_root is None else data_root,
            seed=19, train_task_count=train_task_count,
            test_task_count=test_task_count, bank_steps=1, teachers_per_task=10,
            bank_candidates=20, teacher_batch_size=2, support_count=2, query_count=2,
            selection_count=2, probe_count=4, k=5, hidden=2, out=root / "out",
            fixed_test_spec=fixed_test_spec)

    def test_expanding_six_to_twelve_preserves_four_tests_and_excludes_their_images(self):
        data = _data()
        data["target_train"] = _split(1_000, 240, 2)
        data["target_validation"] = _split(3_000, 240, 4)
        _, old_train, _, old_spec = self.fixture(6, 4, data=data, data_root="/synthetic")
        _, new_train, _, new_spec = self.fixture(12, 4, data=data, data_root="/synthetic",
                                               fixed_test_spec=old_spec)
        for key in ("costs", "test_support_pools", "test_query_pools"):
            self.assertEqual(old_spec[key], new_spec[key])
        for old, new in zip(old_train, new_train):
            self.assertTrue(torch.equal(old.support_ids, new.support_ids))
            self.assertTrue(torch.equal(old.x_support, new.x_support))
            self.assertTrue(torch.equal(old.x_query, new.x_query))
        reserved = set(new_spec["test_ids"])
        for task in new_train:
            self.assertFalse(reserved.intersection(task.support_ids.flatten().tolist()))
            self.assertFalse(reserved.intersection(task.query_ids.flatten().tolist()))
        old_tests, new_tests = (make_cooperative_deepsets_test_tasks(spec)
                               for spec in (old_spec, new_spec))
        for old, new in zip(old_tests, new_tests):
            for key in ("x_support", "y_support", "x_query", "y_query", "support_ids", "query_ids"):
                self.assertTrue(torch.equal(getattr(old, key), getattr(new, key)), key)
        for key, value in (("seed", 20), ("support_count", 3), ("test_task_count", 3)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                _fixed_test_pools(data, {**old_spec, key: value}, data_root="/synthetic", seed=19,
                                  train_task_count=12, test_task_count=4, support_count=2, query_count=2)

    def test_bank_build_writes_light_selected_cards_and_cleans_only_tracked_candidates(self):
        banks, _, _, _ = self.fixture()
        for name, bank in banks.items():
            bank_out = self.bank_root / name
            children = bank_out / "children"
            cards = bank_out / "maps"
            self.assertTrue((children / "preserve-untracked.txt").is_file())
            self.assertEqual(list(children.glob("candidate_*.pt")), [])
            card_paths = sorted(cards.glob("candidate_*.pt"))
            self.assertEqual(len(card_paths), len(bank.states))
            self.assertEqual(len(card_paths), 10)
            for teacher in bank.states:
                source = teacher["source"]
                artifact = Path(source["artifact_path"])
                self.assertTrue(artifact.is_file())
                self.assertEqual(artifact.parent, cards.resolve())
                card = torch.load(artifact, map_location="cpu", weights_only=False)
                self.assertEqual(card["schema_version"], 1)
                self.assertEqual(card["metadata"]["candidate_id"], source["candidate_id"])
                self.assertEqual(card["metadata"]["initialization_seed"],
                                 source["candidate_id"] * 1_000_003 + bank.provenance["seed"] + 1_000_003)
                self.assertEqual(set(card["state_dict"]),
                                 {"weight", "bias", "readout", "per_image_offset"})
                self.assertNotIn("optimizer_state", card)
                self.assertNotIn("history", card)
                self.assertLess(artifact.stat().st_size, 1_000_000)

    def test_incomplete_candidate_build_keeps_cache_until_selection_succeeds(self):
        original = _FakeStore.measure_many

        def write_then_fail(store, entries, *, desc, initialization_seeds):
            original(store, entries[:2], desc=desc, initialization_seeds=initialization_seeds[:2])
            raise RuntimeError("simulated interrupted candidate selection")

        with patch.object(_FakeStore, "measure_many", write_then_fail):
            with self.assertRaisesRegex(RuntimeError, "interrupted candidate selection"):
                self.fixture()
        children = self.bank_root / "0" / "children"
        self.assertTrue(any(children.glob("candidate_*.pt")))
        self.assertTrue((children / "preserve-untracked.txt").is_file())
        self.assertFalse((self.bank_root / "0" / "maps").exists())

    def test_functional_tokens_are_tanh_contributions_and_image_derivatives(self):
        generator = torch.Generator().manual_seed(41)
        state = {"weight": torch.randn(784, 3, generator=generator),
                 "bias": torch.randn(3, generator=generator),
                 "readout": torch.randn(3, generator=generator),
                 "per_image_offset": torch.zeros(())}
        mask = torch.zeros(784, 3); mask.flatten()[:70] = 1
        probe = torch.rand(4, 784, generator=generator)
        tokens, raw = extract_deepsets_functional_token(state, mask, probe)
        effective = state["weight"] * mask
        activation = torch.tanh(probe @ effective + state["bias"])
        psi = activation * state["readout"]
        gain = (1 - activation.square()) * state["readout"]
        q = probe[:, :, None] * effective[None] * gain[:, None]
        torch.testing.assert_close(raw["psi"], psi)
        torch.testing.assert_close(raw["q_signed_mean"], q.mean(0))
        torch.testing.assert_close(raw["q_abs_mean"], q.abs().mean(0))
        torch.testing.assert_close(raw["q_rms"], q.square().mean(0).sqrt())
        self.assertEqual(tokens.shape, (3, len(probe) + 4 * 784))
        torch.testing.assert_close(tokens[:, :len(probe)],
                                   (psi / psi.square().mean().sqrt().clamp_min(1e-8)).T)
        torch.testing.assert_close(tokens[:, -784:], mask.T)

    def test_own_banks_selection_and_sealed_pools_are_disjoint(self):
        banks, train, selection, spec = self.fixture()
        original_costs = _cost_vectors(19, 6)
        self.assertEqual(tuple(banks), ("0", "1"))
        self.assertIsNot(banks["0"], banks["1"])
        self.assertEqual([task.task_id for task in train], ["deepsets:0", "deepsets:1"])
        self.assertEqual([task.task_id for task in selection],
                         ["deepsets:0:selection", "deepsets:1:selection"])
        for index, task in enumerate(train):
            torch.testing.assert_close(torch.tensor(task.provenance["costs"]), original_costs[index])
        for index, cost in enumerate(spec["costs"]):
            torch.testing.assert_close(torch.tensor(cost), original_costs[index + 4])
        for bank in banks.values():
            self.assertIsNone(bank.quality)
            self.assertEqual(bank.tokens.shape, (1, 10, 2, 4 + 4 * 784))
            self.assertEqual(len(bank.states), 10)
            self.assertEqual(bank.diagnostics["feedback_rows_added"], 0)
            self.assertEqual(len(set(bank.provenance["selected_candidate_ids"])), 10)
            self.assertEqual(len(set(bank.provenance["selected_initialization_seeds"])), 10)
            self.assertEqual(len(bank.provenance["selection_density_buckets"]), 10)
        for task, chosen in zip(train, selection):
            self.assertTrue(torch.equal(task.support_ids, chosen.support_ids))
            torch.testing.assert_close(task.context, chosen.context)
            self.assertEqual(task.provenance["task_id_encoding"], "one_hot")
            self.assertEqual(task.provenance["task_id_width"], 4)
            task_number = int(task.task_id.rsplit(":", 1)[1])
            self.assertEqual(task.provenance["evaluator_task_id"], task_number)
            torch.testing.assert_close(task.context[-4:],
                                       torch.nn.functional.one_hot(torch.tensor(task_number), 4).float())
            self.assertEqual(chosen.provenance["evaluator_task_id"], task_number)
            self.assertTrue(set(task.query_ids.flatten().tolist()).isdisjoint(
                chosen.query_ids.flatten().tolist()))
        evaluator_ids = set().union(*(_task_ids(task) for task in train + selection))
        sealed_ids = set(spec["test_ids"])
        self.assertTrue(evaluator_ids.isdisjoint(sealed_ids))
        for bank in banks.values():
            bank_ids = set().union(*(set(values) for values in bank.provenance["partitions"].values()))
            self.assertTrue(bank_ids.isdisjoint(evaluator_ids | sealed_ids))
        self.assertIs(spec["materialized"], False)
        self.assertNotIn("x", spec)
        self.assertNotIn("y", spec)
        self.assertNotIn("labels", spec)

    def test_fixture_supports_arbitrary_ordered_train_and_heldout_counts(self):
        banks, train, selection, spec = self.fixture(train_task_count=3, test_task_count=3)
        self.assertEqual(tuple(banks), ("0", "1", "2"))
        self.assertEqual([task.task_id for task in train],
                         ["deepsets:0", "deepsets:1", "deepsets:2"])
        self.assertEqual([task.task_id for task in selection],
                         ["deepsets:0:selection", "deepsets:1:selection", "deepsets:2:selection"])
        self.assertEqual(spec["train_patterns"], ["0", "1", "2"])
        self.assertEqual(spec["test_task_count"], 3)
        self.assertEqual(spec["test_conditions"], ["0", "1", "2"])
        self.assertEqual(len(spec["costs"]), 3)
        self.assertEqual(len(spec["test_support_pools"]), 3)
        self.assertEqual(len(spec["test_query_pools"]), 3)
        self.assertEqual(len({id(bank) for bank in banks.values()}), 3)
        costs = [torch.tensor(task.provenance["costs"]) for task in train]
        costs.extend(torch.tensor(value) for value in spec["costs"])
        self.assertEqual(len({tuple(value.tolist()) for value in costs}), 6)
        for index, (task, chosen) in enumerate(zip(train, selection)):
            self.assertEqual(task.provenance["evaluator_task_id"], index)
            self.assertEqual(chosen.provenance["evaluator_task_id"], index)
            torch.testing.assert_close(task.context[-6:],
                                       torch.nn.functional.one_hot(torch.tensor(index), 6).float())
            torch.testing.assert_close(chosen.context[-6:], task.context[-6:])

        tests = make_cooperative_deepsets_test_tasks(spec)
        self.assertEqual([task.task_id for task in tests],
                         ["deepsets:test:0", "deepsets:test:1", "deepsets:test:2"])
        for index, task in enumerate(tests):
            torch.testing.assert_close(task.context[-6:],
                                       torch.nn.functional.one_hot(torch.tensor(index + 3), 6).float())
        validate_deepsets_test_tasks(tests, spec, train, selection)
        validate_cooperative_deepsets_inputs(
            banks, train, selection, spec, _Config(3, 3),
            InnerProtocol(steps=2, replicas=2, metric="nmse"))
        wrong_context = train[0].context.clone()
        wrong_context[-6:] = torch.nn.functional.one_hot(torch.tensor(1), 6).float()
        wrong_train = [replace(train[0], context=wrong_context), *train[1:]]
        wrong_selection = [replace(selection[0], context=wrong_context.clone()), *selection[1:]]
        with self.assertRaisesRegex(ValueError, "one-hot task ID"):
            validate_cooperative_deepsets_inputs(
                banks, wrong_train, wrong_selection, spec, _Config(3, 3),
                InnerProtocol(steps=2, replicas=2, metric="nmse"))

    def test_legacy_bootstrap_infers_only_missing_selection_budget(self):
        banks, train, selection, spec = self.fixture()
        self.assertEqual(spec["selection_count"], 2)
        legacy_spec = {key: value for key, value in spec.items() if key != "selection_count"}
        protocol = InnerProtocol(steps=2, replicas=2, metric="nmse")
        validate_cooperative_deepsets_inputs(
            banks, train, selection, legacy_spec, _Config(), protocol)
        self.assertNotIn("selection_count", legacy_spec)
        with self.assertRaisesRegex(ValueError, "budgets must be positive"):
            validate_cooperative_deepsets_inputs(
                banks, train, selection, {**legacy_spec, "selection_count": 0},
                _Config(), protocol)
        shorter = replace(selection[0], x_query=selection[0].x_query[:1],
                          y_query=selection[0].y_query[:1], query_ids=selection[0].query_ids[:1])
        with self.assertRaisesRegex(ValueError, "selection query budgets must agree"):
            validate_cooperative_deepsets_inputs(
                banks, train, [shorter, selection[1]], legacy_spec, _Config(), protocol)

    def test_validators_reject_query_reuse_and_final_cost_leak(self):
        banks, train, selection, spec = self.fixture()
        bad_selection = replace(selection[0], x_query=train[0].x_query,
                                y_query=train[0].y_query, query_ids=train[0].query_ids)
        with self.assertRaisesRegex(ValueError, "training and selection query image IDs overlap"):
            validate_cooperative_deepsets_inputs(
                banks, train, [bad_selection, selection[1]], spec,
                _Config(), InnerProtocol(steps=2, replicas=2, metric="nmse"))
        tests = make_cooperative_deepsets_test_tasks(spec)
        validate_deepsets_test_tasks(tests, spec, train, selection)
        leaking = replace(tests[0], provenance={**tests[0].provenance,
                                               "costs": train[0].provenance["costs"]})
        with self.assertRaisesRegex(ValueError, "wrong or training cost vector"):
            validate_deepsets_test_tasks([leaking, tests[1]], spec, train, selection)

    def test_feedback_adds_complete_replica_state_once_and_reuses_fixed_probe(self):
        banks, train, _, _ = self.fixture()
        bank = banks["0"]
        base = bank.states[0]["state_dict"]
        replicas, state = 2, {}
        for name, value in base.items():
            if name == "masks":
                state[name] = bank.baseline_mask[None, None].expand(1, replicas, -1, -1).clone()
            else:
                first, second = value.clone(), value.clone()
                if name == "weight":
                    first[0, 0] += .015625
                    second[0, 0] += .03125
                state[name] = torch.stack((first, second)).unsqueeze(0)
        optimizer = {"state": {0: {"step": torch.tensor(3.),
                                   "exp_avg": torch.zeros(1, replicas, 784, 2),
                                   "exp_avg_sq": torch.ones(1, replicas, 784, 2)}},
                     "param_groups": [{"lr": .01}]}
        history = {"steps": torch.tensor([0, 1, 2]),
                   "queryNMSE": torch.tensor([[[.6, .5]], [[.4, .3]], [[.2, .1]]])}
        result = {"label_source": "fresh_terminal_query", "fixed_horizon": True,
                  "task_id": train[0].task_id, "protocol_id": "frozen-protocol",
                  "replica_losses": [.2, .1], "seeds": ["10:0", "10:1"],
                  "plateau_flags": [False, False], "state_dict": state,
                  "optimizer_state": optimizer, "history": history}
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        artifact = Path(directory) / "child.pt"
        torch.save({"result": result}, artifact)
        updated = append_deepsets_feedback(bank, bank.baseline_mask, result,
                                           bank.diagnostics["probe_x"], task_id=train[0].task_id,
                                           artifact_path=artifact, max_teachers=12)
        self.assertEqual(updated.tokens.shape[1], bank.tokens.shape[1] + replicas)
        self.assertEqual(updated.diagnostics["feedback_rows_added"], replicas)
        feedback = [row for row in updated.states if row["source"]["kind"] == "feedback"]
        self.assertEqual(len(feedback), replicas)
        for teacher in feedback:
            self.assertEqual(teacher["source"]["task_id"], "deepsets:0")
            self.assertEqual(teacher["optimizer_state"]["state"][0]["exp_avg"].shape[1], 1)
            self.assertEqual(tuple(teacher["history"]["queryNMSE"].shape), (3, 1, 1))
            self.assertTrue(torch.equal(teacher["source"]["source_mask"], bank.baseline_mask))
        duplicate = append_deepsets_feedback(updated, bank.baseline_mask, result,
                                             bank.diagnostics["probe_x"], task_id=train[0].task_id,
                                             artifact_path=artifact, max_teachers=12)
        self.assertIs(duplicate, updated)
        with self.assertRaisesRegex(ValueError, "held-out topology"):
            append_deepsets_feedback(bank, bank.baseline_mask, result,
                                     bank.diagnostics["probe_x"], task_id=train[0].task_id,
                                     artifact_path=artifact, max_teachers=12, eligible=False)
        with self.assertRaisesRegex(ValueError, "fixed reserved probe"):
            append_deepsets_feedback(bank, bank.baseline_mask, result,
                                     bank.diagnostics["probe_x"] + .1, task_id=train[0].task_id,
                                     artifact_path=artifact, max_teachers=12)


def _task_ids(task: TaskData) -> set[int]:
    return set(task.support_ids.flatten().tolist()) | set(task.query_ids.flatten().tolist())
