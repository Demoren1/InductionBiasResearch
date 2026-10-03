from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.stats import t

from generator_evaluator.data import InnerProtocol, TaskData, support_context, tensor_hash
from generator_evaluator.evaluate_frozen import evaluate_frozen


def _task(task_id: str, split: str, role: str, index: int, *, salt: int = 0) -> TaskData:
    generator = torch.Generator().manual_seed(70 + index + salt)
    x_support = torch.randn(2, 5, 784, generator=generator)
    y_support = torch.randn(2, generator=generator)
    x_query = torch.randn(3, 5, 784, generator=generator)
    y_query = torch.randn(3, generator=generator)
    return TaskData(task_id, split, x_support, y_support, x_query, y_query,
        support_context(x_support.mean(1), y_support),
        torch.arange(10).reshape(2, 5) + 10_000 * (index + salt),
        torch.arange(15).reshape(3, 5) + 10_000 * (index + salt) + 100,
        {"family": "deepsets", "domain": "deepsets", "role": role,
         "task_index": index,
         **({"heldout_condition": str(index)} if role == "sealed_test" else {})})


def _masks(common_edges: int = 2):
    common = torch.zeros(784, 2)
    common.view(-1)[:common_edges] = 1
    dense = torch.ones(784, 2)
    return common, dense


def _write_run(path: Path, *, test_count: int = 1, selection_count: int = 1,
               salt: int = 0, common_edges: int = 2) -> dict:
    path.mkdir(parents=True)
    protocol = InnerProtocol(steps=2, replicas=2, lr=.01, l2=.001,
        checkpoint_every=1, seed=314, metric="nmse")
    common, dense = _masks(common_edges)
    selection = [_task(f"deepsets:{i}:selection", "validation", "selection", i, salt=salt)
                 for i in range(selection_count)]
    tests = [_task(f"deepsets:test:{i}", "test", "sealed_test", i, salt=salt)
             for i in range(test_count)]
    test_spec = {"family": "cooperative_deepsets", "materialized": False,
        "test_task_count": test_count, "costs": [[0.2] * 784 for _ in range(test_count)],
        "seed": 31 + salt}
    run_task_fingerprints = {task.task_id: task.fingerprint for task in selection}
    frozen = {"methods": {"common": common, "dense": dense},
              "protocol": protocol.__dict__}
    generator_masks = {}
    for index, edge_count in enumerate((3 + salt % 5, 5 + salt % 5)):
        proposal = torch.zeros_like(common)
        proposal.view(-1)[:edge_count] = 1
        generator_masks[f"generator_final_{index}"] = proposal
    torch.save(frozen, path / "frozen.pt")
    torch.save({"masks": generator_masks}, path / "final_generator_proposals.pt")
    torch.save(tests, path / "test_tasks.pt")
    torch.save({"selection_tasks": selection}, path / "inputs.pt")
    (path / "summary.json").write_text(json.dumps({"domain": "deepsets",
        "final": {"old": "metrics are deliberately irrelevant"}}), encoding="utf-8")
    (path / "run_spec.json").write_text(json.dumps({"test_spec": test_spec,
        "tasks": run_task_fingerprints}), encoding="utf-8")
    (path / "frozen.json").write_text(json.dumps({"mask_hashes": {
        "common": tensor_hash(common), "dense": tensor_hash(dense)}}), encoding="utf-8")
    (path / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return {"selection": selection, "tests": tests, "common": common,
            "dense": dense, "protocol": protocol}


class FrozenEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.current = self.root / "current"
        self.data = _write_run(self.current, test_count=2, selection_count=2)
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def fake_fit(self, masks, task, protocol, device, *, initialization_seeds):
        sums = [int(mask.sum()) for mask in masks]
        self.calls.append({"task_id": task.task_id, "replicas": protocol.replicas,
                           "device": device, "mask_sums": sums,
                           "initialization_seeds": list(initialization_seeds)})
        loss_rows = {2: [2.0, 4.0, 8.0], 4: [1.0, 2.0, 4.0],
                     1568: [3.0, 5.0, 7.0]}
        results = []
        for mask, seed in zip(masks, initialization_seeds):
            values = loss_rows.get(int(mask.sum()), [1.5, 2.5, 3.5])
            losses = values[:protocol.replicas]
            results.append({"replica_losses": losses,
                            "seeds": [f"{seed}:{i}" for i in range(protocol.replicas)],
                            # The evaluator must drop all bulky state fields.
                            "state_dict": {"large_state": torch.ones(100)}})
        return results

    def evaluate(self, *args, **kwargs):
        with patch("generator_evaluator.deepsets_batch.fit_deepsets_batch", self.fake_fit):
            return evaluate_frozen(*args, **kwargs)

    def test_paired_replica_statistics_and_cache_resume(self):
        first = self.evaluate(self.current, replicas=3, devices=["cpu"], partition="test")
        self.assertEqual(len(self.calls), 2)
        row = first["task_results"][0]
        stats = row["comparisons"]["common"]
        expected = np.asarray([2., 4., 8.]) - np.asarray([3., 5., 7.])
        self.assertAlmostEqual(stats["mean_paired_delta"], float(expected.mean()))
        self.assertEqual(stats["paired_deltas"], expected.tolist())
        self.assertEqual(stats["n"], 3)
        half_width = t.ppf(.975, 2) * expected.std(ddof=1) / np.sqrt(3)
        np.testing.assert_allclose(stats["ci95"],
            [expected.mean() - half_width, expected.mean() + half_width])
        self.assertEqual(first["aggregate"]["common"]["n_tasks"], 2)
        aggregate = first["aggregate"]["common"]
        self.assertEqual(aggregate["n"], 3)
        self.assertEqual(aggregate["paired_deltas"], expected.tolist())
        np.testing.assert_allclose(aggregate["ci95"], stats["ci95"])
        self.assertEqual(aggregate["uncertainty_unit"],
                         "fresh initialization averaged over fixed current tasks")
        self.assertTrue((self.current / "frozen_diagnostic" / "paired_deltas.csv").is_file())
        self.assertTrue((self.current / "frozen_diagnostic" / "paired_deltas.png").is_file())
        cached = self.evaluate(self.current, replicas=3, devices=["cpu"], partition="test")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(cached["cache"]["reused_tasks"], 2)
        self.assertEqual(cached["cache"]["newly_fitted_tasks"], 0)

        # Changing a frozen mask changes the cache identity and forces a refit.
        common, dense = _masks(common_edges=3)
        torch.save({"methods": {"common": common, "dense": dense},
                    "protocol": self.data["protocol"].__dict__}, self.current / "frozen.pt")
        (self.current / "frozen.json").write_text(json.dumps({"mask_hashes": {
            "common": tensor_hash(common), "dense": tensor_hash(dense)}}), encoding="utf-8")
        refreshed = self.evaluate(self.current, replicas=3, devices=["cpu"], partition="test")
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(refreshed["cache"]["reused_tasks"], 0)

        # Protocol, task fingerprint and sealed spec changes invalidate the same rows.
        frozen = torch.load(self.current / "frozen.pt", map_location="cpu", weights_only=False)
        frozen["protocol"]["lr"] *= 2
        torch.save(frozen, self.current / "frozen.pt")
        self.evaluate(self.current, replicas=3, devices=["cpu"], partition="test")
        self.assertEqual(len(self.calls), 6)
        tests = torch.load(self.current / "test_tasks.pt", map_location="cpu", weights_only=False)
        tests[0].x_query = tests[0].x_query.clone()
        tests[0].x_query[0, 0, 0] += .25
        torch.save(tests, self.current / "test_tasks.pt")
        self.evaluate(self.current, replicas=3, devices=["cpu"], partition="test")
        self.assertEqual(len(self.calls), 8)
        spec_path = self.current / "run_spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec["test_spec"]["seed"] += 1
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        self.evaluate(self.current, replicas=3, devices=["cpu"], partition="test")
        self.assertEqual(len(self.calls), 10)

    def test_all_masks_share_replica_count_and_initialization_seeds(self):
        previous_path = self.root / "previous"
        _write_run(previous_path, test_count=1, selection_count=1, salt=99,
                   common_edges=4)
        result = self.evaluate(self.current, replicas=3, devices=["cpu"],
            partition="selection", comparison_run=previous_path)
        self.assertEqual(len(result["task_results"]), 2)
        self.assertEqual(len(self.calls), 2)
        for call in self.calls:
            self.assertEqual(call["replicas"], 3)
            self.assertEqual(len(call["mask_sums"]), 3)
            self.assertEqual(call["mask_sums"], [2, 4, 1568])
            self.assertEqual(len(set(call["initialization_seeds"])), 1)
        self.assertEqual(result["comparison_run"]["method"], "previous_common")
        self.assertIn("old task data and metrics were not loaded",
                      result["comparison_run"]["interpretation"])
        self.assertTrue(all(row["task_id"].startswith("deepsets:0:") or
                            row["task_id"].startswith("deepsets:1:")
                            for row in result["task_results"]))

    def test_comparison_run_does_not_require_matching_task_fingerprints(self):
        previous_path = self.root / "different_previous"
        old = _write_run(previous_path, test_count=1, selection_count=1, salt=123,
                         common_edges=4)
        # A different prior test manifest and stale metric block must not be compared.
        self.assertNotEqual(old["tests"][0].fingerprint, self.data["tests"][0].fingerprint)
        result = self.evaluate(self.current, replicas=2, devices=["cpu"],
            partition="test", comparison_run=previous_path)
        self.assertEqual({row["task_id"] for row in result["task_results"]},
                         {task.task_id for task in self.data["tests"]})
        self.assertTrue(all(call["task_id"].startswith("deepsets:test:") for call in self.calls))
        self.assertEqual(result["comparison_run"]["mask_hash"], tensor_hash(old["common"]))

    def test_current_generator_proposals_share_seeds_and_ignore_prior_proposals(self):
        previous_path = self.root / "old_run"
        _write_run(previous_path, test_count=1, selection_count=1, salt=99, common_edges=4)
        result = self.evaluate(self.current, replicas=3, devices=["cpu"],
            partition="selection", comparison_run=previous_path, include_generators=True)
        self.assertTrue(result["include_generators"])
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all(call["mask_sums"] == [2, 3, 5, 4, 1568] for call in self.calls))
        methods = set(result["task_results"][0]["replica_losses"])
        self.assertEqual(methods, {"common", "generator_final_0", "generator_final_1",
                                   "previous_common", "dense"})
        self.assertEqual(tensor_hash(self.data["common"]), result["mask_hashes"]["common"])
        self.assertEqual(len({tuple(call["initialization_seeds"]) for call in self.calls}), 1)

    def test_generator_proposals_require_matching_binary_masks(self):
        artifact_path = self.current / "final_generator_proposals.pt"
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        original = artifact["masks"]["generator_final_0"]
        artifact["masks"]["generator_final_0"] = original[:, :1]
        torch.save(artifact, artifact_path)
        with self.assertRaisesRegex(ValueError, "matching binary mask"):
            evaluate_frozen(self.current, replicas=2, devices=["cpu"],
                            partition="test", include_generators=True)
        artifact["masks"]["generator_final_0"] = original.clone()
        artifact["masks"]["generator_final_0"][0, 0] = .5
        torch.save(artifact, artifact_path)
        with self.assertRaisesRegex(ValueError, "matching binary mask"):
            evaluate_frozen(self.current, replicas=2, devices=["cpu"],
                            partition="test", include_generators=True)

    def test_named_candidate_masks_are_validated_and_part_of_cache_identity(self):
        candidate_path = self.root / "source_candidates.pt"
        candidate = torch.zeros(784, 2)
        candidate.view(-1)[:6] = 1
        second = torch.zeros(784, 2)
        second.view(-1)[:8] = 1
        torch.save({"candidate_a": candidate, "candidate_b": second}, candidate_path)
        first = self.evaluate(self.current, replicas=3, devices=["cpu"],
            partition="selection", candidate_masks=candidate_path)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all(call["mask_sums"] == [2, 6, 8, 1568] for call in self.calls))
        methods = set(first["task_results"][0]["replica_losses"])
        self.assertEqual(methods, {"common", "candidate_a", "candidate_b", "dense"})
        self.assertEqual(first["mask_hashes"]["candidate_a"], tensor_hash(candidate))
        self.assertEqual(first["candidate_mask_file"], str(candidate_path.resolve()))
        self.evaluate(self.current, replicas=3, devices=["cpu"],
                      partition="selection", candidate_masks=candidate_path)
        self.assertEqual(len(self.calls), 2)

        changed = candidate.clone()
        changed.view(-1)[6] = 1
        torch.save({"candidate_a": changed, "candidate_b": second}, candidate_path)
        refreshed = self.evaluate(self.current, replicas=3, devices=["cpu"],
            partition="selection", candidate_masks=candidate_path)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(refreshed["cache"]["reused_tasks"], 0)
        self.assertEqual(refreshed["mask_hashes"]["candidate_a"], tensor_hash(changed))

    def test_candidate_masks_reject_reserved_names_and_invalid_tensors(self):
        candidate_path = self.root / "bad_candidates.pt"
        valid = torch.zeros(784, 2)
        torch.save({"common": valid}, candidate_path)
        with self.assertRaisesRegex(ValueError, "reserved"):
            evaluate_frozen(self.current, replicas=2, devices=["cpu"],
                            partition="test", candidate_masks=candidate_path)
        invalid = valid.clone()
        invalid[0, 0] = .5
        torch.save({"fractional": invalid}, candidate_path)
        with self.assertRaisesRegex(ValueError, "finite binary"):
            evaluate_frozen(self.current, replicas=2, devices=["cpu"],
                            partition="test", candidate_masks=candidate_path)
        torch.save({"wrong_shape": torch.zeros(784, 3)}, candidate_path)
        with self.assertRaisesRegex(ValueError, "shape"):
            evaluate_frozen(self.current, replicas=2, devices=["cpu"],
                            partition="test", candidate_masks=candidate_path)
        self.assertEqual(self.calls, [])

    def test_cuda_jobs_use_one_spawned_worker_per_requested_device(self):
        created, submitted = [], []

        class ImmediateExecutor:
            def __init__(self, **kwargs):
                created.append(kwargs)

            def submit(self, worker, payload):
                args = __import__("generator_evaluator.parallel_measurements",
                                  fromlist=["_decode_payload"])._decode_payload(payload)
                submitted.append((args[4], worker, isinstance(payload, bytes)))
                future = Future()
                future.set_result(worker(payload))
                return future

            def shutdown(self, **kwargs):
                pass

        def spawned_fit(masks, task, protocol, device, *, initialization_seeds):
            return self.fake_fit(masks, task, protocol, device,
                                 initialization_seeds=initialization_seeds)

        with patch("generator_evaluator.evaluate_frozen.ProcessPoolExecutor", ImmediateExecutor), \
             patch("generator_evaluator.evaluate_frozen.torch.cuda.is_available", return_value=True), \
             patch("generator_evaluator.evaluate_frozen.torch.cuda.device_count", return_value=2), \
             patch("generator_evaluator.deepsets_batch.fit_deepsets_batch", spawned_fit):
            result = evaluate_frozen(self.current, replicas=2,
                devices=["cuda:0", "cuda:1"], partition="both")
        self.assertEqual(len(result["task_results"]), 4)
        self.assertIsNone(result["aggregate"])
        self.assertEqual(set(result["partition_aggregates"]), {"selection", "test"})
        self.assertEqual(len(created), 2)
        self.assertTrue(all(kwargs["max_workers"] == 1 and
                            kwargs["initargs"] == (device,)
                            for kwargs, device in zip(created, ("cuda:0", "cuda:1"))))
        self.assertEqual([row[0] for row in submitted],
                         ["cuda:0", "cuda:1", "cuda:0", "cuda:1"])
        self.assertTrue(all(row[2] for row in submitted))
        self.assertTrue(all("state_dict" not in row for result_row in result["task_results"]
                            for row in result_row["replica_losses"].values()))

    def test_worker_failure_preserves_completed_task_cache_for_resume(self):
        class FailedExecutor:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def submit(self, worker, payload):
                future = Future()
                future.set_exception(RuntimeError("injected worker failure"))
                return future

            def shutdown(self, **kwargs):
                pass

        with patch("generator_evaluator.evaluate_frozen.ProcessPoolExecutor", FailedExecutor), \
             patch("generator_evaluator.evaluate_frozen.torch.cuda.is_available", return_value=True), \
             patch("generator_evaluator.evaluate_frozen.torch.cuda.device_count", return_value=1), \
             patch("generator_evaluator.deepsets_batch.fit_deepsets_batch", self.fake_fit):
            with self.assertRaisesRegex(RuntimeError, "injected worker failure"):
                evaluate_frozen(self.current, replicas=2, devices=["cpu", "cuda:0"],
                                partition="test")
        cache_files = sorted((self.current / "frozen_diagnostic" / "cache").glob("*.json"))
        self.assertEqual([path.name for path in cache_files], ["test-0000.json"])
        saved = json.loads(cache_files[0].read_text(encoding="utf-8"))
        self.assertEqual(set(saved), {"identity", "losses", "seeds"})
        resumed = self.evaluate(self.current, replicas=2, devices=["cpu"], partition="test")
        self.assertEqual(resumed["cache"]["reused_tasks"], 1)
        self.assertEqual(resumed["cache"]["newly_fitted_tasks"], 1)
        self.assertEqual([call["task_id"] for call in self.calls],
                         ["deepsets:test:0", "deepsets:test:1"])


if __name__ == "__main__":
    unittest.main()
