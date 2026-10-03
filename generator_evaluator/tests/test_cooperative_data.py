"""Contracts for separate pattern banks and their sealed feedback path."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import Future
from unittest.mock import patch

import torch

from generator_evaluator.cooperative_data import (
    _centroid_align_pattern_maps,
    append_feedback,
    bank_input_fingerprint,
    build_cooperative_fixture,
    extract_pattern_tokens,
    make_cooperative_test_spec,
    make_cooperative_test_task,
    make_cooperative_test_tasks,
)


class CooperativeDataTests(unittest.TestCase):
    def test_infeasible_test_support_rejected_before_teacher_fits(self):
        with patch("generator_evaluator.cooperative_data._build_bank") as build:
            with self.assertRaisesRegex(ValueError, "test support_count cannot be balanced"):
                build_cooperative_fixture(seed=4100, support_count=256, query_count=128,
                                          bank_steps=1, teachers_per_pattern=5)
            build.assert_not_called()

    @classmethod
    def setUpClass(cls) -> None:
        cls.banks, cls.train, cls.selection, cls.spec = build_cooperative_fixture(
            seed=413, bank_steps=1, teachers_per_pattern=5,
            support_count=8, query_count=8, selection_count=4, k=8,
        )

    def test_sparse_functional_maps_and_normalization_are_explicit(self) -> None:
        probe = torch.tensor([[1.] * 11, [-1.] * 11])
        mask = torch.zeros(11, 8); mask[0, 0] = 1
        state = {"w": torch.zeros(11, 8), "b": torch.zeros(8), "a": torch.ones(8), "c": torch.zeros(())}
        state["w"][0, 0] = 2
        tokens, raw = extract_pattern_tokens(state, mask, probe)
        expected_q = probe[:, :, None] * (state["w"] * mask)[None] * state["a"][None, None] * torch.tensor([[[True] * 8], [[False] * 8]])
        torch.testing.assert_close(raw["q_signed"], expected_q.mean(0))
        torch.testing.assert_close(raw["q_abs"], expected_q.abs().mean(0))
        torch.testing.assert_close(raw["q_rms"], expected_q.square().mean(0).sqrt())
        self.assertEqual(tokens.shape, (8, len(probe) + 4 * 11))
        self.assertEqual(raw["q_abs"][0, 0].item(), 1.0)
        self.assertAlmostEqual(tokens[0, len(probe) + 11].item(),
                               raw["q_abs"][0, 0].item() / raw["q_scale"].item(), places=6)
        self.assertEqual(tokens[0, -11].item(), 1.0)

    def test_centroid_alignment_is_invariant_to_hidden_permutations_and_ties(self) -> None:
        # All columns have the same coordinate centroid, so the functional
        # values themselves must provide the deterministic tie break.
        base = torch.zeros(11, 4)
        base[[0, 10], 0] = 1
        base[5, 1] = 2
        base[[2, 8], 2] = 1
        base[[4, 6], 3] = 1
        permutations = ([0, 1, 2, 3], [3, 0, 2, 1], [1, 3, 0, 2])
        maps = torch.stack([base[:, permutation] for permutation in permutations])

        aligned, orders, centroids = _centroid_align_pattern_maps(maps)

        torch.testing.assert_close(aligned[0], aligned[1], rtol=0, atol=0)
        torch.testing.assert_close(aligned[0], aligned[2], rtol=0, atol=0)
        self.assertEqual(orders.shape, (3, 4))
        torch.testing.assert_close(centroids, torch.full((3, 4), 5.0, dtype=torch.float64))

    def test_banks_have_distinct_patterns_shared_probe_and_exact_density_anchors(self) -> None:
        self.assertEqual(set(self.banks), {"0001", "0011"})
        self.assertEqual({task.task_id for task in self.train}, {"pattern:0001", "pattern:0011"})
        self.assertEqual({task.task_id for task in self.selection}, {"pattern:0001:selection", "pattern:0011:selection"})
        self.assertTrue(all(task.split == "validation" for task in self.selection))
        expected = {9, 8, 44, 62, 88}
        probes = []
        for pattern, bank in self.banks.items():
            self.assertEqual(set(bank.masks.sum((1, 2)).long().tolist()), expected)
            self.assertEqual(int(bank.baseline_mask.sum()), 8)
            probes.append(bank.diagnostics["probe_x"])
            self.assertEqual(bank.provenance["accepted_feedback_task_ids"], [f"pattern:{pattern}"])
            self.assertEqual(bank.provenance["functional_alignment_method"], "input_coordinate_centroid")
            self.assertEqual(bank.provenance["functional_alignment_scope"],
                             "baseline only; teacher tokens and masks keep their original pairing")
            self.assertEqual(bank.diagnostics["functional_column_orders"].shape, (5, 8))
            self.assertTrue(all({"w", "b", "a", "c"} <= set(item["state_dict"]) for item in bank.states))
            self.assertTrue(all(item["optimizer_state"] is not None and item["history"] for item in bank.states))
            from generator_evaluator.adapters import _exact_topk
            raw_maps = torch.stack([teacher["q_abs"] for teacher in bank.states])
            aligned, orders, _ = _centroid_align_pattern_maps(raw_maps)
            torch.testing.assert_close(bank.diagnostics["aligned_q_abs"], aligned)
            torch.testing.assert_close(bank.diagnostics["functional_column_orders"], orders)
            torch.testing.assert_close(bank.baseline_mask, _exact_topk(aligned.mean(0), 8), rtol=0, atol=0)
            for row, teacher in enumerate(bank.states):
                rebuilt, _ = extract_pattern_tokens(teacher["state_dict"], bank.masks[row],
                                                    bank.diagnostics["probe_x"])
                torch.testing.assert_close(bank.tokens[0, row], rebuilt, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(probes[0], probes[1])
        self.assertNotIn("labels", self.spec)
        self.assertNotIn("x", self.spec)
        self.assertNotIn("y", self.spec)

    def test_all_data_roles_are_isolated_except_selection_reuses_train_support(self) -> None:
        all_bank = set()
        for bank in self.banks.values():
            parts = bank.provenance["partitions"]
            groups = [set(values) for values in parts.values()]
            self.assertEqual(sum(map(len, groups)), len(set().union(*groups)))
            all_bank |= set().union(*groups[:3])
        for task, chosen in zip(self.train, self.selection):
            self.assertTrue(torch.equal(task.support_ids, chosen.support_ids))
            torch.testing.assert_close(task.context, chosen.context)
            self.assertTrue(set(task.query_ids.tolist()).isdisjoint(chosen.query_ids.tolist()))
            self.assertTrue(all_bank.isdisjoint(task.support_ids.tolist()))
            self.assertTrue(all_bank.isdisjoint(task.query_ids.tolist()))
        sealed = set(self.spec["test_ids"])
        self.assertTrue(all(sealed.isdisjoint(task.support_ids.tolist()) and sealed.isdisjoint(task.query_ids.tolist())
                            for task in self.train + self.selection))

    def test_feedback_adds_actual_replicas_and_preserves_them_at_capacity(self) -> None:
        bank = self.banks["0001"]
        base = bank.states[0]["state_dict"]
        first = {name: value.clone() for name, value in base.items()}
        second = {name: value.clone() for name, value in base.items()}
        first["w"][1, 1] += .125
        second["w"][2, 2] -= .25
        measurement = {"label_source": "fresh_terminal_query", "fixed_horizon": True,
                       "task_id": "pattern:0001", "protocol_id": "unit", "replica_losses": [.2, .3],
                       "seeds": [1, 2], "plateau_flags": [False, False],
                       "state_dict": [first, second], "optimizer_state": [{"step": 1}, {"step": 1}],
                       "history": [[{"step": 1}], [{"step": 1}]]}
        with tempfile.TemporaryDirectory() as folder:
            artifact = Path(folder) / "fresh.pt"; torch.save({"record": "real"}, artifact)
            updated = append_feedback(bank, bank.baseline_mask, measurement, bank.diagnostics["probe_x"],
                                      task_id="pattern:0001", artifact_path=artifact, max_teachers=8)
            self.assertEqual(updated.tokens.shape[1], bank.tokens.shape[1] + 2)
            self.assertTrue({9, 88}.issubset(set(updated.masks.sum((1, 2)).long().tolist())))
            self.assertEqual(updated.diagnostics["feedback_rows_added"], 2)
            self.assertNotEqual(bank_input_fingerprint(updated), bank_input_fingerprint(bank))
            self.assertEqual(sum(item["source"]["kind"] == "feedback" for item in updated.states), 2)
            self.assertTrue(all(item["optimizer_state"] is not None and item["history"]
                                for item in updated.states if item["source"]["kind"] == "feedback"))
            newer = deepcopy(measurement)
            for state in newer["state_dict"]:
                state["w"] += .03125
            capped = append_feedback(updated, bank.baseline_mask, newer, bank.diagnostics["probe_x"],
                                     task_id="pattern:0001", artifact_path=artifact, max_teachers=7)
            self.assertEqual(capped.tokens.shape[1], 7)
            self.assertEqual(set(capped.masks.sum((1, 2)).long().tolist()), {8, 9, 44, 62, 88})
            self.assertEqual(sum(item["source"]["kind"] == "feedback" for item in capped.states), 2)
            for teacher in capped.states:
                if teacher["source"]["kind"] == "feedback":
                    torch.testing.assert_close(teacher["state_dict"]["w"], newer["state_dict"][teacher["source"]["replica"]]["w"])
            with self.assertRaisesRegex(ValueError, "held-out topology"):
                append_feedback(bank, bank.baseline_mask, measurement, bank.diagnostics["probe_x"],
                                task_id="pattern:0001", artifact_path=artifact, eligible=False)

    def test_feedback_preserves_four_replicas_and_checks_record_counts(self) -> None:
        bank = self.banks["0001"]
        states = [deepcopy(bank.states[0]["state_dict"]) for _ in range(4)]
        for index, state in enumerate(states):
            state["w"][0, 0] += (index + 1) * .125
        measurement = dict(label_source="fresh_terminal_query", fixed_horizon=True,
            task_id="pattern:0001", protocol_id="four-replicas", state_dict=states,
            replica_losses=[.2, .3, .4, .5], seeds=[1, 2, 3, 4], plateau_flags=[False] * 4)
        with tempfile.TemporaryDirectory() as folder:
            artifact = Path(folder) / "fresh.pt"
            torch.save(measurement, artifact)
            updated = append_feedback(bank, bank.baseline_mask, measurement,
                bank.diagnostics["probe_x"], task_id="pattern:0001", artifact_path=artifact,
                max_teachers=16)
            feedback = [item for item in updated.states if item["source"]["kind"] == "feedback"]
            self.assertEqual({item["source"]["replica"] for item in feedback}, {0, 1, 2, 3})
            self.assertEqual(updated.diagnostics["feedback_rows_added"], 4)
            self.assertTrue(all(item["optimizer_state"] is None for item in feedback))
            for item in feedback:
                torch.testing.assert_close(item["state_dict"]["w"],
                    states[item["source"]["replica"]]["w"])
            invalid = {**measurement, "seeds": [1, 2]}
            with self.assertRaisesRegex(ValueError, "one seeds record per replica"):
                append_feedback(bank, bank.baseline_mask, invalid, bank.diagnostics["probe_x"],
                    task_id="pattern:0001", artifact_path=artifact, max_teachers=16)

    def test_third_pattern_stays_sealed_and_invalid_role_orbits_fail(self) -> None:
        task = make_cooperative_test_task(self.spec)
        self.assertEqual(task.split, "test")
        self.assertEqual(task.task_id, "pattern:0101:test")
        observed = set().union(*(set(item.support_ids.tolist()) | set(item.query_ids.tolist())
                                 for item in self.train + self.selection))
        self.assertTrue(observed.isdisjoint(task.support_ids.tolist()))
        self.assertTrue(observed.isdisjoint(task.query_ids.tolist()))
        altered_support = dict(self.spec, support_count=16)
        self.assertTrue(torch.equal(task.query_ids, make_cooperative_test_task(altered_support).query_ids))
        with self.assertRaisesRegex(ValueError, "distinct reversal/complement"):
            build_cooperative_fixture(train_patterns=("0001", "1000"), bank_steps=1, teachers_per_pattern=5,
                                      support_count=8, query_count=8, selection_count=4, k=8)
        with self.assertRaisesRegex(ValueError, "test_pattern must belong"):
            build_cooperative_fixture(test_pattern="1110", bank_steps=1, teachers_per_pattern=5,
                                      support_count=8, query_count=8, selection_count=4, k=8)

    def test_test_spec_reserves_independent_pools_without_bank_fitting(self) -> None:
        for seed in (4100, 4101):
            spec = make_cooperative_test_spec(seed=seed, support_count=128, query_count=128)
            task = make_cooperative_test_task(spec)
            self.assertEqual(len(task.support_ids), 128)
            self.assertEqual(len(task.query_ids), 128)
            self.assertTrue(set(task.support_ids.tolist()).isdisjoint(task.query_ids.tolist()))
            self.assertEqual(set(spec["test_ids"]),
                             set(spec["test_support_ids"]) | set(spec["test_query_ids"]))

    def test_multi_pattern_twelve_bank_fixture_and_ordered_sealed_tasks(self) -> None:
        held_out = ("0010", "0100", "1011", "1101")
        train_patterns = tuple(f"{value:04b}" for value in range(16) if f"{value:04b}" not in held_out)
        self.assertEqual(len(train_patterns), 12)
        bank_marker = object()
        with patch("generator_evaluator.cooperative_data._build_bank", return_value=bank_marker) as build_bank:
            banks, train, selection, spec = build_cooperative_fixture(
                train_patterns=train_patterns, test_pattern=held_out, seed=413,
                bank_steps=1, teachers_per_pattern=5, support_count=8,
                query_count=8, selection_count=4, k=8, measurement_devices=("cpu",),
            )
        self.assertEqual(set(banks), set(train_patterns))
        self.assertTrue(all(bank is bank_marker for bank in banks.values()))
        self.assertEqual(build_bank.call_count, 12)
        self.assertTrue(all(call.kwargs["measurement_devices"] == ("cpu",)
                            for call in build_bank.call_args_list))
        self.assertEqual([task.task_id for task in train], [f"pattern:{pattern}" for pattern in train_patterns])
        self.assertEqual([task.task_id for task in selection],
                         [f"pattern:{pattern}:selection" for pattern in train_patterns])
        self.assertEqual(spec["family"], "cooperative_pattern")
        self.assertEqual(spec["test_pattern"], held_out[0])
        self.assertEqual(spec["test_patterns"], list(held_out))
        self.assertEqual(len(spec["test_specs"]), 4)
        self.assertFalse(spec["materialized"])
        self.assertEqual(spec["train_patterns"], list(train_patterns))
        self.assertEqual(len(spec["test_ids"]), len(spec["test_support_ids"]) + len(spec["test_query_ids"]))
        self.assertTrue(all(child["test_pattern"] == pattern
                            for child, pattern in zip(spec["test_specs"], held_out)))
        tasks = make_cooperative_test_tasks(spec)
        self.assertEqual([task.task_id for task in tasks], [f"pattern:{pattern}:test" for pattern in held_out])
        self.assertTrue(all(task.split == "test" for task in tasks))
        observed = set().union(*(set(task.support_ids.tolist()) | set(task.query_ids.tolist())
                                 for task in train + selection))
        self.assertTrue(all(observed.isdisjoint(task.support_ids.tolist()) and
                            observed.isdisjoint(task.query_ids.tolist()) for task in tasks))
        self.assertTrue(all(task.query_ids.tolist() == tasks[0].query_ids.tolist() for task in tasks))
        self.assertNotIn("labels", spec)
        self.assertNotIn("x", spec)
        self.assertNotIn("y", spec)

    def test_composite_spec_rejects_reordered_or_tampered_roles_and_ids(self) -> None:
        patterns = ("0010", "0100", "1011", "1101")
        spec = make_cooperative_test_spec(train_patterns=("0000", "0001", "0011"),
                                          test_pattern=patterns, seed=413,
                                          support_count=8, query_count=8)
        for mutate in (
            lambda value: value["test_specs"].reverse(),
            lambda value: value["test_patterns"].reverse(),
            lambda value: value["test_specs"][1].update(test_pattern="1011"),
            lambda value: value["test_support_ids"].__setitem__(0, value["test_support_ids"][0] + 1),
        ):
            altered = deepcopy(spec)
            mutate(altered)
            with self.assertRaisesRegex(ValueError, "altered"):
                make_cooperative_test_tasks(altered)

    def test_bank_query_budget_adapts_to_sparse_reserved_source_pool(self) -> None:
        from generator_evaluator.cooperative_data import (
            _balanced_support_budget, _build_bank, _cooperative_partitions, _pattern_labels, _pattern_table)
        from generator_evaluator.cooperative_data import _preflight_bank_query_pools
        from generator_evaluator.pattern_fit import PatternFitEngine

        parts = _cooperative_partitions(4100)
        budgets = _preflight_bank_query_pools(("0000", "1111"), parts)
        self.assertEqual(budgets["0000"], 60)
        self.assertEqual(budgets["1111"], 64)

        # Candidate ranking also uses the adapted count and publishes it as bank provenance.
        ids, x = _pattern_table()
        labels = _pattern_labels(x, "0000")
        expected_support_count = _balanced_support_budget(labels, parts["bank_support"])
        state = {"w": torch.zeros(11, 8), "b": torch.zeros(8),
                 "a": torch.ones(8), "c": torch.zeros(())}
        fitted = {"state_dict": [state], "optimizer_state": [{}], "history": [[{"step": 1}]],
                  "replica_losses": [1.0]}
        probe_rows = parts["probe"][:4]
        candidate_support_lengths = []

        def candidate_fit(self, masks, tasks, **kwargs):
            candidate_support_lengths.extend(len(task.x_support) for task in tasks)
            return [fitted] * len(tasks)

        with patch.object(PatternFitEngine, "fit", candidate_fit):
            bank = _build_bank(
                "0000", parts, seed=4100, bank_steps=1, teachers_per_pattern=5, k=8,
                probe_x=x[probe_rows], probe_ids=ids[probe_rows],
                device="cpu", bank_candidates=10,
            )
        self.assertEqual(bank.provenance["bank_query_count"], 60)
        self.assertEqual(bank.provenance["bank_support_count"], expected_support_count)
        self.assertEqual(set(candidate_support_lengths), {expected_support_count})
        self.assertEqual(len(bank.diagnostics["bank_selection_query_ids"]), 60)

        source_query_lengths = []
        source_support_lengths = []
        source_fit = {"state_dict": [state], "optimizer_state": [{}], "history": [[{"step": 1}]]}

        def fake_source_fit(mask, task, protocol, device):
            source_query_lengths.append(len(task.query_ids))
            source_support_lengths.append(len(task.support_ids))
            return source_fit

        with patch("generator_evaluator.cooperative_data._fit_pattern", side_effect=fake_source_fit):
            source_bank = _build_bank(
                "0000", parts, seed=4100, bank_steps=1, teachers_per_pattern=5, k=8,
                probe_x=x[probe_rows], probe_ids=ids[probe_rows], device="cpu",
            )
        self.assertEqual(source_query_lengths, [60] * 5)
        self.assertEqual(source_support_lengths, [expected_support_count] * 5)
        self.assertEqual(source_bank.provenance["bank_query_count"], 60)
        self.assertEqual(source_bank.provenance["bank_support_count"], expected_support_count)

    def test_pattern_candidate_fit_waves_match_single_device_selection_and_states(self) -> None:
        from generator_evaluator.cooperative_data import (
            _build_bank, _cooperative_partitions, _pattern_table)
        from generator_evaluator.parallel_measurements import (
            _decode_payload, _encode_payload, _pattern_worker)

        seed = 419
        parts = _cooperative_partitions(seed)
        ids, x = _pattern_table()
        probe_rows = parts["probe"][:4]
        args = dict(pattern="0001", parts=parts, seed=seed, bank_steps=1,
                    teachers_per_pattern=5, k=8, probe_x=x[probe_rows],
                    probe_ids=ids[probe_rows], device="cpu", bank_candidates=5,
                    teacher_batch_size=128)
        serial = _build_bank(**args, measurement_devices=("cpu",))
        submitted = []

        class FakeExecutor:
            def __init__(self, **kwargs):
                pass
            def submit(self, fn, payload):
                masks, tasks, protocol, device, seeds = _decode_payload(payload)
                submitted.append((device, tuple(masks.shape), seeds))
                future = Future()
                worker_args = (masks, tasks, protocol, "cpu", seeds)
                future.set_result(_encode_payload(_pattern_worker(*worker_args)))
                return future
            def shutdown(self, **kwargs):
                pass

        with patch("generator_evaluator.parallel_measurements.ProcessPoolExecutor", FakeExecutor):
            parallel = _build_bank(**args, measurement_devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"))

        self.assertEqual([row[0] for row in submitted], ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        self.assertEqual([row[1] for row in submitted],
                         [(2, 11, 8), (1, 11, 8), (1, 11, 8), (1, 11, 8)])
        self.assertEqual([row[2] for row in submitted],
                         [[_candidate_seed(seed, candidate) for candidate in start]
                          for start in ((0, 1), (2,), (3,), (4,))])
        self.assertEqual(serial.provenance["selected_candidate_ids"],
                         parallel.provenance["selected_candidate_ids"])
        self.assertEqual(serial.provenance["selected_candidate_query_losses"],
                         parallel.provenance["selected_candidate_query_losses"])
        self.assertEqual(serial.provenance["selected_initialization_seeds"],
                         parallel.provenance["selected_initialization_seeds"])
        torch.testing.assert_close(serial.masks, parallel.masks, rtol=0, atol=0)
        torch.testing.assert_close(serial.tokens, parallel.tokens, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(serial.baseline_mask, parallel.baseline_mask, rtol=0, atol=0)
        for left, right in zip(serial.states, parallel.states):
            self.assertNotIn("history", left)
            self.assertNotIn("optimizer_state", left)
            self.assertEqual(left["source"], right["source"])
            for name, value in left["state_dict"].items():
                torch.testing.assert_close(value, right["state_dict"][name], rtol=1e-6, atol=1e-7)


def _candidate_seed(seed: int, candidate_id: int) -> int:
    return int(seed) + 1_000_003 * (int(candidate_id) + 1)


if __name__ == "__main__":
    unittest.main()
