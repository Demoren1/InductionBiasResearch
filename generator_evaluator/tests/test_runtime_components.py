from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from generator_evaluator.adapters import build_pattern_fixture, measure_mask
from generator_evaluator.data import InnerProtocol, RealReplay
from generator_evaluator.measurements import BatchedMeasurementStore, MeasurementStore
from generator_evaluator.runtime import DenseProtocolSelector, RunSession, cpu_state, random_exact_k


class RuntimeComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.bank, cls.tasks, _ = build_pattern_fixture(seed=733, bank_steps=1, teacher_count=2,
            support_count=8, query_count=8, k=8)
        cls.train = next(task for task in cls.tasks if task.split == "train")
        cls.validation = next(task for task in cls.tasks if task.split == "validation")
        cls.protocol = InnerProtocol(steps=2, replicas=2, checkpoint_every=1, seed=733)

    def test_store_cache_fresh_and_injected_measurement(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = []
            def measured(*args, **kwargs):
                calls.append(kwargs.get("initialization_seed"))
                return measure_mask(*args, **kwargs)
            store = MeasurementStore(Path(folder), RealReplay(self.protocol), "cpu", measure_fn=measured)
            first, _ = store.measure(self.bank.baseline_mask, self.train, "first")
            cached, _ = store.measure(self.bank.baseline_mask, self.train, "cached")
            fresh, _ = store.measure(self.bank.baseline_mask, self.train, "fresh", fresh=True)
            self.assertEqual(first, cached)
            self.assertNotEqual(first["seeds"], fresh["seeds"])
            self.assertEqual(len(calls), 2)
            store.replay.validate()

    def test_batched_partial_cache_and_utility_exact_k(self):
        with tempfile.TemporaryDirectory() as folder:
            store = BatchedMeasurementStore(Path(folder), RealReplay(self.protocol), "cpu")
            entries = [(self.bank.baseline_mask, self.train, "a"),
                       (self.bank.baseline_mask.roll(1, 0), self.train, "b")]
            first = store.measure_many(entries)
            second = store.measure_many(entries)
            self.assertEqual([row[0]["quality"] for row in first], [row[0]["quality"] for row in second])
            masks = random_exact_k(4, 11, 8, 8, torch.Generator().manual_seed(1))
            self.assertEqual(masks.shape, (4, 11, 8))
            self.assertTrue(torch.equal(masks.sum((1, 2)), torch.full((4,), 8.)))

    def test_session_strict_spec_checkpoint_and_dense_selector(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "out"
            source = Path(__file__).resolve().parents[1] / "runtime.py"
            spec = {"protocol": asdict(self.protocol), "source_hashes": RunSession.source_hashes([source], source.parents[1])}
            session = RunSession(out, spec, [source], project=source.parents[1])
            self.assertIsNone(session.prepare(inputs={"fixture": 1}))
            session.save_protocol(self.protocol, role="test")
            self.assertEqual(session.load_protocol().fingerprint, self.protocol.fingerprint)
            replay = RealReplay(self.protocol)
            session.save_checkpoint({"epoch": 0}, replay=replay, history={"rows": []})
            self.assertEqual(session.load_checkpoint()["epoch"], 0)
            with self.assertRaisesRegex(ValueError, "different"):
                RunSession(out, {"changed": True}, [source], project=source.parents[1], resume=True).prepare()
            selector = DenseProtocolSelector(out)
            selected = selector.select([self.validation], torch.ones(11, 8), self.protocol,
                                       "cpu", [.005, .01])
            self.assertIn(selected.lr, (.005, .01))
            self.assertTrue((out / "dense_tuning.json").is_file())

    def test_cpu_state_is_detached(self):
        module = torch.nn.Linear(2, 1)
        state = cpu_state(module)
        self.assertTrue(all(value.device.type == "cpu" for value in state.values()))
        self.assertFalse(any(value.requires_grad for value in state.values()))


if __name__ == "__main__":
    unittest.main()
