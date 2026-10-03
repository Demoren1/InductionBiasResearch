from pathlib import Path
import tempfile
import unittest
from concurrent.futures import Future
from multiprocessing.reduction import ForkingPickler
from unittest.mock import patch

import torch

from generator_evaluator.adapters import build_pattern_fixture
from generator_evaluator.data import InnerProtocol, RealReplay, TaskData, support_context
from generator_evaluator.parallel_measurements import (ParallelMeasurementStore,
    _decode_payload, _encode_payload, _serialized_deepsets_worker,
    _serialized_pattern_worker)


def _result(mask, protocol):
    value = float(mask.sum())
    return dict(label_source="fresh_terminal_query", fixed_horizon=True,
                protocol_id=protocol.fingerprint,
                replica_losses=torch.tensor([value, value + 1]),
                seeds=[101, 202], plateau_flags=torch.tensor([False, False]))


def _deepsets_task() -> TaskData:
    generator = torch.Generator().manual_seed(991)
    xs = torch.randn(5, 5, 784, generator=generator)
    ys = torch.randn(5, generator=generator)
    xq = torch.randn(4, 5, 784, generator=generator)
    yq = torch.randn(4, generator=generator)
    return TaskData("deepsets:train:parallel", "train", xs, ys, xq, yq,
                    support_context(xs.mean(1), ys),
                    torch.arange(25).reshape(5, 5), torch.arange(25, 45).reshape(4, 5),
                    {"family": "deepsets"})


def _deepsets_masks(count=1):
    masks = torch.zeros(count, 784, 3)
    for index in range(count):
        masks[index].flatten()[index * 7:index * 7 + 90] = 1
    return masks


class ParallelMeasurementStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.bank, cls.tasks, _ = build_pattern_fixture(seed=845, bank_steps=1, teacher_count=2,
            support_count=8, query_count=8, k=8)
        cls.task = next(task for task in cls.tasks if task.split == "train")
        cls.protocol = InnerProtocol(steps=2, replicas=2, checkpoint_every=1, seed=845)

    def test_deduplicates_caches_and_returns_requested_order(self):
        calls = []
        def fit(masks, task, protocol, device):
            calls.append((len(masks), task.task_id, device))
            return [_result(mask, protocol) for mask in masks]

        first = self.bank.baseline_mask
        second = first.roll(1, 0)
        with tempfile.TemporaryDirectory() as folder:
            store = ParallelMeasurementStore(Path(folder), RealReplay(self.protocol), "cpu",
                devices=["cpu"], batch_size=8, batch_fit_fn=fit)
            entries = [(second, self.task, "second"), (first, self.task, "first"),
                       (second, self.task, "duplicate")]
            values = store.measure_many(entries, desc="test")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], 2)
            self.assertEqual(len(values), 3)
            self.assertEqual(values[0][0]["origin"], "second")
            self.assertEqual(values[1][0]["origin"], "first")
            self.assertEqual(values[2][0], values[0][0])
            self.assertEqual(len(store.replay.records), 2)
            store.measure_many(entries)
            self.assertEqual(len(calls), 1)
            store.replay.validate()
            store.close()

    def test_chunks_and_rejects_wrong_result_count_before_writing(self):
        masks = [self.bank.baseline_mask.roll(index, 0) for index in range(3)]
        calls = []
        def fit(masks, task, protocol, device):
            calls.append(len(masks))
            return [_result(mask, protocol) for mask in masks]

        with tempfile.TemporaryDirectory() as folder:
            store = ParallelMeasurementStore(Path(folder), RealReplay(self.protocol), "cpu",
                devices=["cpu"], batch_size=2, batch_fit_fn=fit)
            store.measure_many([(mask, self.task, "chunk") for mask in masks])
            self.assertEqual(calls, [2, 1])
            store.close()

        with tempfile.TemporaryDirectory() as folder:
            store = ParallelMeasurementStore(Path(folder), RealReplay(self.protocol), "cpu",
                devices=["cpu"], batch_size=2,
                batch_fit_fn=lambda masks, task, protocol, device: [])
            with self.assertRaisesRegex(ValueError, "result count"):
                store.measure_many([(masks[0], self.task, "bad")])
            self.assertFalse(any((Path(folder) / "children").glob("*.pt")))

    def test_partial_cache_reuses_the_original_batch_shape(self):
        masks = [self.bank.baseline_mask.roll(index, 0) for index in range(2)]
        calls = []
        def fit(batch, task, protocol, device):
            calls.append(len(batch))
            return [_result(mask, protocol) for mask in batch]

        with tempfile.TemporaryDirectory() as folder:
            store = ParallelMeasurementStore(Path(folder), RealReplay(self.protocol), "cpu",
                devices=["cpu"], batch_size=2, batch_fit_fn=fit)
            rows = store.measure_many([(mask, self.task, "initial") for mask in masks])
            cached_path = Path(rows[0][0]["artifact_path"])
            missing_path = Path(rows[1][0]["artifact_path"])
            original_bytes = cached_path.read_bytes()
            missing_path.unlink()
            calls.clear()
            store.measure_many([(mask, self.task, "resume") for mask in masks])
            self.assertEqual(calls, [2])
            self.assertEqual(cached_path.read_bytes(), original_bytes)
            self.assertTrue(missing_path.is_file())
            store.close()

    def test_uninjected_deepsets_cpu_path_uses_packed_fitter(self):
        task = _deepsets_task()
        protocol = InnerProtocol(steps=2, replicas=2, lr=.002, l2=.0001,
                                 checkpoint_every=1, seed=991, metric="nmse")
        with tempfile.TemporaryDirectory() as folder:
            store = ParallelMeasurementStore(Path(folder), RealReplay(protocol), "cpu",
                devices=["cpu"], batch_size=2)
            row, result = store.measure_many([(_deepsets_masks()[0], task, "actual")])[0]
            self.assertEqual(row["origin"], "actual")
            self.assertEqual(len(result["replica_losses"]), 2)
            self.assertIn("effective_weights", result)
            store.replay.validate()
            store.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                store.measure_many([])

    def test_uninjected_pattern_cpu_path_uses_domain_batch_fitter(self):
        task = self.task
        protocol = InnerProtocol(steps=2, replicas=1, lr=.002, l2=.0001,
                                 checkpoint_every=1, seed=845)
        masks = [self.bank.baseline_mask, self.bank.baseline_mask.roll(1, 1)]
        with tempfile.TemporaryDirectory() as folder:
            store = ParallelMeasurementStore(Path(folder), RealReplay(protocol), "cpu",
                devices=["cpu"], batch_size=2)
            rows = store.measure_many([(mask, task, f"candidate-{index}")
                                       for index, mask in enumerate(masks)])
            self.assertEqual(len(rows), 2)
            for index, (record, result) in enumerate(rows):
                self.assertEqual(record["origin"], f"candidate-{index}")
                self.assertEqual(result["task_id"], task.task_id)
                self.assertEqual(result["actual_initialization_seed"], protocol.seed)
                self.assertEqual(len(result["state_dict"]), 1)
                self.assertTrue(torch.isfinite(torch.tensor(result["replica_losses"])).all())
            store.replay.validate()
            store.close()

    def test_dense_teacher_candidates_keep_distinct_initializations_and_cache_paths(self):
        task = _deepsets_task()
        protocol = InnerProtocol(steps=2, replicas=1, checkpoint_every=1, seed=991, metric="nmse")
        mask = torch.ones(784, 3)
        with tempfile.TemporaryDirectory() as folder:
            with ParallelMeasurementStore(Path(folder), RealReplay(protocol), "cpu",
                    devices=["cpu"], batch_size=2) as store:
                entries = [(mask, task, "candidate0"), (mask, task, "candidate1")]
                values = store.measure_many(entries, initialization_seeds=[101, 102])
                self.assertNotEqual(values[0][0]["artifact_path"], values[1][0]["artifact_path"])
                self.assertFalse(torch.equal(values[0][1]["effective_weights"], values[1][1]["effective_weights"]))
                store.replay.validate()
                store.measure_many(entries, initialization_seeds=[101, 102])
                self.assertEqual(len(store.replay.records), 2)

    def test_parallel_routes_chunks_to_each_cuda_worker(self):
        task = _deepsets_task()
        protocol = InnerProtocol(steps=2, replicas=2, checkpoint_every=1, seed=991, metric="nmse")
        created, submitted = [], []

        class FakeExecutor:
            def __init__(self, **kwargs):
                created.append(kwargs)
            def submit(self, fn, payload):
                masks, worker_task, worker_protocol, device = _decode_payload(payload)
                submitted.append((fn, tuple(masks.shape), worker_task, worker_protocol, device))
                future = Future()
                future.set_result(_encode_payload([_result(mask, worker_protocol) for mask in masks]))
                return future
            def shutdown(self, **kwargs):
                pass

        with tempfile.TemporaryDirectory() as folder:
            with patch("generator_evaluator.parallel_measurements.ProcessPoolExecutor", FakeExecutor):
                store = ParallelMeasurementStore(Path(folder), RealReplay(protocol), "cuda:0",
                    devices=["cuda:0", "cuda:1", "cuda:2"], batch_size=1)
                store.measure_many([(mask, task, "parallel") for mask in _deepsets_masks(3)])
                self.assertEqual([item[-1] for item in submitted], ["cuda:0", "cuda:1", "cuda:2"])
                self.assertTrue(all(item[0] is _serialized_deepsets_worker for item in submitted))
                self.assertTrue(all(item[1] == (1, 784, 3)
                                    and item[2].fingerprint == task.fingerprint
                                    and item[2].x_support.device.type == "cpu"
                                    and item[3].fingerprint == protocol.fingerprint for item in submitted))
                self.assertEqual(len(created), 3)
                self.assertTrue(all(item["max_workers"] == 1 and item["initargs"] == (device,)
                                    for item, device in zip(created, ["cuda:0", "cuda:1", "cuda:2"])))
                store.close()

    def test_parallel_routes_pattern_chunks_to_each_cuda_worker_in_order(self):
        protocol = InnerProtocol(steps=2, replicas=2, checkpoint_every=1, seed=845)
        tasks = [task for task in self.tasks if task.provenance.get("family") == "pattern"]
        self.assertGreaterEqual(len(tasks), 1)
        task = tasks[0]
        masks = []
        for index in range(5):
            mask = torch.zeros_like(self.bank.baseline_mask)
            mask.flatten()[index] = 1
            masks.append(mask)
        submitted = []

        class FakeExecutor:
            def __init__(self, **kwargs):
                pass
            def submit(self, fn, payload):
                masks, worker_tasks, worker_protocol, device = _decode_payload(payload)
                submitted.append((fn, tuple(masks.shape), worker_tasks, device))
                future = Future()
                future.set_result(_encode_payload([_result(mask, worker_protocol) for mask in masks]))
                return future
            def shutdown(self, **kwargs):
                pass

        with tempfile.TemporaryDirectory() as folder:
            with patch("generator_evaluator.parallel_measurements.ProcessPoolExecutor", FakeExecutor):
                store = ParallelMeasurementStore(Path(folder), RealReplay(protocol), "cuda:0",
                    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"], batch_size=128)
                rows = store.measure_many([(mask, task, "pattern") for mask in masks])
                self.assertEqual(len(rows), 5)
                self.assertEqual([item[3] for item in submitted],
                                 ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
                self.assertTrue(all(item[0] is _serialized_pattern_worker for item in submitted))
                self.assertEqual([item[1] for item in submitted],
                                 [(2, 11, 8), (1, 11, 8), (1, 11, 8), (1, 11, 8)])
                self.assertTrue(all(len(item[2]) == item[1][0] and
                                    all(candidate.fingerprint == task.fingerprint and
                                        candidate.x_support.device.type == "cpu"
                                        for candidate in item[2]) for item in submitted))
                self.assertEqual([row[0]["origin"] for row in rows], ["pattern"] * 5)
                store.close()

    def test_pattern_tasks_with_equal_shapes_share_a_full_batch(self):
        protocol = InnerProtocol(steps=2, replicas=2, checkpoint_every=1, seed=845)
        tasks = [task for task in self.tasks if task.provenance.get("family") == "pattern"]
        if len(tasks) < 2:
            self.skipTest("fixture needs two pattern tasks")
        calls = []

        def fit(masks, grouped_tasks, protocol, device):
            calls.append((len(masks), [task.task_id for task in grouped_tasks]))
            return [_result(mask, protocol) for mask in masks]

        masks = [self.bank.baseline_mask, self.bank.baseline_mask.roll(1, 0)]
        with tempfile.TemporaryDirectory() as folder:
            store = ParallelMeasurementStore(Path(folder), RealReplay(protocol), "cpu",
                devices=["cpu"], batch_size=8, batch_fit_fn=fit)
            store.measure_many([(masks[0], tasks[0], "first"), (masks[1], tasks[1], "second")])
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], 2)
            self.assertEqual(calls[0][1], [tasks[0].task_id, tasks[1].task_id])
            store.close()

    def test_tensor_archives_do_not_invoke_multiprocessing_fd_reducers(self):
        values = {str(index): torch.tensor([index], dtype=torch.float32) for index in range(1024)}
        def forbidden_reducer(value):
            raise AssertionError("tensor reached the multiprocessing FD transport")
        with patch.dict(ForkingPickler._extra_reducers, {torch.Tensor: forbidden_reducer}):
            archive = _encode_payload(values)
            # Both a request and a result use ordinary byte messages.
            ForkingPickler.dumps((_serialized_deepsets_worker, archive))
            ForkingPickler.dumps(archive)
        decoded = _decode_payload(archive)
        self.assertEqual(len(decoded), len(values))
        for key, value in values.items():
            torch.testing.assert_close(decoded[key], value)

    def test_serialized_worker_preserves_real_child_result_and_optimizer(self):
        task = _deepsets_task()
        protocol = InnerProtocol(steps=2, replicas=2, checkpoint_every=1, seed=991, metric="nmse")
        archive = _encode_payload((_deepsets_masks(2), task, protocol, "cpu"))
        results = _decode_payload(_serialized_deepsets_worker(archive))
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(result["state_dict"]["weight"].shape, (1, 2, 784, 3))
            self.assertEqual(result["optimizer_state"]["state"][0]["exp_avg"].shape, (1, 2, 784, 3))
            self.assertTrue(torch.isfinite(torch.tensor(result["replica_losses"])).all())

    def test_serialized_pattern_worker_preserves_candidate_seed_and_optimizer(self):
        from generator_evaluator.pattern_batch import fit_pattern_batch

        task = self.tasks[0]
        protocol = InnerProtocol(steps=2, replicas=1, lr=.002, checkpoint_every=1, seed=845)
        mask = self.bank.baseline_mask.unsqueeze(0)
        args = (mask, [task], protocol, "cpu", [991])
        results = _decode_payload(_serialized_pattern_worker(_encode_payload(args)))
        reference = fit_pattern_batch(mask, [task], protocol, "cpu", initialization_seeds=[991])
        self.assertEqual(results[0]["actual_initialization_seed"], 991)
        self.assertEqual(results[0]["seeds"], reference[0]["seeds"])
        self.assertEqual(results[0]["history"], reference[0]["history"])
        for key, value in reference[0]["state_dict"][0].items():
            torch.testing.assert_close(results[0]["state_dict"][0][key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
