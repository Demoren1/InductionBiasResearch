"""Focused checks for train-only cooperative warm-start imports."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from generator_evaluator.data import InnerProtocol, RealReplay, TaskData
from generator_evaluator.warm_start import load_cooperative_warm_start


def _task(domain: str, role: str, split: str, *, task_id: str | None = None,
          evaluator_task_id: int | None = None, task_id_width: int | None = None) -> TaskData:
    task_id = task_id or f"{domain}:{role}" + (":selection" if split == "validation" else "")
    shape = (5, 784) if domain == "deepsets" else (11,)
    support = torch.ones((2, *shape), dtype=torch.float32)
    query = torch.full((2, *shape), 2.0, dtype=torch.float32)
    if domain == "deepsets":
        support_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        query_ids = torch.tensor([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])
    else:
        support_ids, query_ids = torch.tensor([1, 2]), torch.tensor([3, 4])
    if split == "validation":
        query_ids += 100
        query.fill_(3.0)
    provenance = {"family": domain}
    if split == "validation":
        provenance["role"] = "selection"
    context = torch.zeros(3)
    if domain == "deepsets" and task_id_width is not None:
        if evaluator_task_id is None:
            raise ValueError("encoded DeepSets fixture tasks need an explicit task ID")
        from generator_evaluator.data import support_context
        context = support_context(support.mean(1), torch.tensor([0.0, 1.0]))
        context = torch.cat((context, torch.nn.functional.one_hot(
            torch.tensor(evaluator_task_id), task_id_width).float()))
        provenance.update({"evaluator_task_id": evaluator_task_id,
                           "task_id_encoding": "one_hot", "task_id_width": task_id_width})
    return TaskData(task_id, split, support, torch.tensor([0.0, 1.0]), query,
                    torch.tensor([1.0, 0.0]), context, support_ids, query_ids,
                    provenance)


def _fixture(root: Path, domain: str, *, with_protocol_json: bool = True,
             test_replay: bool = False, train_task_count: int = 2,
             test_task_count: int = 2) -> tuple[SimpleNamespace, InnerProtocol, InnerProtocol]:
    roles = tuple(str(index) for index in range(train_task_count)) if domain == "deepsets" else ("0001", "0011")
    heldout = "heldout" if domain == "deepsets" else "0101"
    features, hidden = (784, 32) if domain == "deepsets" else (11, 8)
    config = SimpleNamespace(domain=domain, train_patterns=roles, test_pattern=heldout,
                             seed=19, k=5, features=features, hidden=hidden,
                             width=8, heads=2, layers=1, ensemble_members=2, noise_dim=4,
                             test_task_count=test_task_count)
    requested = InnerProtocol(steps=4, replicas=2, lr=.01, seed=19)
    actual = replace(requested, lr=.015) if with_protocol_json else requested
    task_prefix = "deepsets" if domain == "deepsets" else "pattern"
    task_id_width = train_task_count + test_task_count
    train = [_task(domain, role, "train", task_id=f"{task_prefix}:{role}",
                   evaluator_task_id=index, task_id_width=task_id_width if domain == "deepsets" else None)
             for index, role in enumerate(roles)]
    selection = [_task(domain, role, "validation", task_id=f"{task_prefix}:{role}:selection",
                       evaluator_task_id=index, task_id_width=task_id_width if domain == "deepsets" else None)
                 for index, role in enumerate(roles)]
    family = "cooperative_deepsets" if domain == "deepsets" else "cooperative_pattern"
    test_spec = {"family": family, "train_patterns": list(roles),
                 "test_pattern": heldout, "materialized": False}
    if domain == "deepsets":
        test_spec.update({"test_task_count": test_task_count, "task_id_encoding": "one_hot",
                          "task_id_width": task_id_width,
                          "costs": [[float(index + j) for j in range(10)]
                                    for index in range(100, 100 + test_task_count)],
                          "test_support_pools": [[1000 + i] for i in range(test_task_count)],
                          "test_query_pools": [[2000 + i] for i in range(test_task_count)]})
    baseline = torch.zeros((features, hidden))
    baseline.view(-1)[:config.k] = 1
    banks = {
        role: SimpleNamespace(
            provenance={"domain": domain, "family": f"cooperative_{domain}", "pattern": role},
            masks=torch.zeros((1, features, hidden)),
            baseline_mask=baseline.clone(),
        )
        for role in roles
    }

    root.mkdir(parents=True)
    children = root / "children"
    children.mkdir()
    replay = RealReplay(actual, split_seed=config.seed)
    target_task = train[0]
    mask = torch.zeros((features, hidden))
    mask[0, 0] = 1
    result = {"protocol_id": actual.fingerprint,
              "label_source": "fresh_terminal_query", "fixed_horizon": True,
              "replica_losses": [0.4, 0.6], "seeds": [101, 202],
              "plateau_flags": [False, False]}
    child = children / "referenced.pt"
    torch.save({"result": result}, child)
    replay.append(mask, target_task, result, origin="fixture", artifact_path=child)
    if test_replay:
        test_task = _task(domain, heldout, "test", task_id=f"{task_prefix}:test:{heldout}")
        test_child = children / "test.pt"
        torch.save({"result": result}, test_child)
        replay.append(mask, test_task, result, origin="fixture", artifact_path=test_child)

    inputs = {"banks": banks, "train_tasks": train, "selection_tasks": selection,
              "test_spec": test_spec}
    torch.save(inputs, root / "inputs.pt")
    source_config = vars(config).copy()
    if domain == "pattern":
        source_config.pop("domain")
        source_config.pop("features")
        source_config.pop("hidden")
    run_spec = {"config": source_config, "requested_protocol": asdict(requested)}
    (root / "run_spec.json").write_text(json.dumps(run_spec), encoding="utf-8")
    if with_protocol_json:
        (root / "protocol.json").write_text(json.dumps({
            "inner_protocol": asdict(actual), "protocol_id": actual.fingerprint,
        }), encoding="utf-8")
    torch.save({"epoch": 3, "banks": banks, "replay": replay,
                "ensemble": {"weight": torch.tensor([1.0])},
                "evaluator_training_state": {"updates": 2},
                "best_mask": mask}, root / "checkpoint.pt")
    (children / "orphan.pt").write_bytes(b"unreferenced")
    return config, requested, actual


class WarmStartTests(unittest.TestCase):
    def test_pattern_source_without_protocol_json_keeps_legacy_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            config, requested, _ = _fixture(root, "pattern", with_protocol_json=False)
            warm = load_cooperative_warm_start(root, config, requested)
            self.assertEqual(warm.protocol.fingerprint, requested.fingerprint)
            self.assertEqual(tuple(warm.banks), config.train_patterns)

    def test_pattern_warm_start_binds_ordered_composite_test_roles_and_pools(self):
        heldouts = ("0010", "0100", "1011", "1101")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            config, requested, _ = _fixture(root, "pattern")
            config_fields = vars(config).copy()
            config_fields.update(test_pattern=heldouts[0], test_patterns=heldouts)
            config = SimpleNamespace(**config_fields)
            inputs_path = root / "inputs.pt"
            inputs = torch.load(inputs_path, map_location="cpu", weights_only=False)
            support_ids, query_ids = [100, 101], [200, 201]
            child_specs = [
                {"family": "cooperative_pattern", "seed": config.seed,
                 "test_pattern": pattern, "test_ids": support_ids + query_ids,
                 "test_support_ids": support_ids, "test_query_ids": query_ids,
                 "support_count": 2, "query_count": 2,
                 "train_patterns": list(config.train_patterns), "materialized": False}
                for pattern in heldouts]
            inputs["test_spec"] = {
                "family": "cooperative_pattern", "seed": config.seed,
                "test_pattern": heldouts[0], "test_patterns": list(heldouts),
                "test_specs": child_specs, "test_ids": support_ids + query_ids,
                "test_support_ids": support_ids, "test_query_ids": query_ids,
                "support_count": 2, "query_count": 2,
                "train_patterns": list(config.train_patterns), "materialized": False}
            torch.save(inputs, inputs_path)
            run_spec_path = root / "run_spec.json"
            run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
            run_spec["config"]["test_pattern"] = heldouts[0]
            run_spec["config"]["test_patterns"] = list(heldouts)
            run_spec_path.write_text(json.dumps(run_spec), encoding="utf-8")

            warm = load_cooperative_warm_start(root, config, requested)
            self.assertEqual(warm.test_spec["test_patterns"], list(heldouts))
            wrong_order_fields = vars(config).copy()
            wrong_order_fields.update(
                test_pattern=heldouts[1],
                test_patterns=(heldouts[1], *heldouts[2:], heldouts[0]))
            wrong_order = SimpleNamespace(**wrong_order_fields)
            with self.assertRaisesRegex(ValueError, "ordered test roles"):
                load_cooperative_warm_start(root, wrong_order, requested)

    def test_deepsets_uses_actual_protocol_and_copies_only_replay_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            config, requested, actual = _fixture(root, "deepsets")
            warm = load_cooperative_warm_start(root, config, requested)
            self.assertEqual(warm.protocol.fingerprint, actual.fingerprint)
            self.assertEqual(warm.replay.protocol.fingerprint, actual.fingerprint)
            self.assertEqual(warm.provenance["source_epoch"], 3)
            self.assertTrue(torch.equal(warm.ensemble_state["weight"], torch.tensor([1.0])))
            self.assertEqual(warm.evaluator_training_state, {"updates": 2})
            self.assertEqual(warm.best_mask.shape, (784, 32))
            self.assertEqual([task.task_id for task in warm.train_tasks],
                             ["deepsets:0", "deepsets:1"])
            self.assertEqual([task.task_id for task in warm.selection_tasks],
                             ["deepsets:0:selection", "deepsets:1:selection"])
            destination = Path(temporary) / "target"
            copied = warm.materialize_replay(destination)
            self.assertEqual({path.name for path in (destination / "children").iterdir()},
                             {"referenced.pt"})
            self.assertEqual(Path(copied.records[0]["artifact_path"]).parent,
                             (destination / "children").resolve())
            self.assertFalse((destination / "inputs.pt").exists())

    def test_deepsets_warm_start_accepts_arbitrary_ordered_task_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            config, requested, _ = _fixture(root, "deepsets", train_task_count=4,
                                            test_task_count=3)
            warm = load_cooperative_warm_start(root, config, requested)
            self.assertEqual(tuple(warm.banks), ("0", "1", "2", "3"))
            self.assertEqual([task.task_id for task in warm.train_tasks],
                             ["deepsets:0", "deepsets:1", "deepsets:2", "deepsets:3"])
            self.assertEqual([task.provenance["evaluator_task_id"] for task in warm.train_tasks],
                             [0, 1, 2, 3])
            self.assertEqual([task.provenance["task_id_width"] for task in warm.selection_tasks],
                             [7, 7, 7, 7])

            bad_config = SimpleNamespace(**vars(config))
            bad_config.test_task_count = 2
            with self.assertRaisesRegex(ValueError, "held-out task count"):
                load_cooperative_warm_start(root, bad_config, requested)

    def test_deepsets_warm_start_checks_encoded_task_id_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            config, requested, _ = _fixture(root, "deepsets")
            inputs = torch.load(root / "inputs.pt", map_location="cpu", weights_only=False)
            wrong_context = inputs["train_tasks"][0].context.clone()
            wrong_context[-4:] = torch.nn.functional.one_hot(torch.tensor(1), 4).float()
            inputs["train_tasks"][0].context = wrong_context
            inputs["selection_tasks"][0].context = wrong_context.clone()
            torch.save(inputs, root / "inputs.pt")
            with self.assertRaisesRegex(ValueError, "context task ID"):
                load_cooperative_warm_start(root, config, requested)

    def test_requested_protocol_and_domain_dimensions_and_roles_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            config, requested, _ = _fixture(root, "deepsets")
            with self.assertRaisesRegex(ValueError, "requested protocol"):
                load_cooperative_warm_start(root, config, replace(requested, lr=.02))
            for field, value, message in (
                ("seed", 20, "seed"), ("width", 10, "width"), ("k", 6, "k"),
                ("features", 785, "features"), ("hidden", 31, "hidden"),
                ("domain", "pattern", "domain"),
                ("train_patterns", ("1", "0"), "training-role order"),
                ("test_pattern", "another-heldout", "test role"),
            ):
                with self.subTest(field=field):
                    bad_config = SimpleNamespace(**vars(config))
                    setattr(bad_config, field, value)
                    with self.assertRaisesRegex(ValueError, message):
                        load_cooperative_warm_start(root, bad_config, requested)

    def test_actual_protocol_may_only_change_learning_rate(self):
        for field, value in (("steps", 5), ("replicas", 3), ("metric", "nmse"),
                             ("seed", 20), ("l2", .01)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "source"
                config, requested, _ = _fixture(root, "deepsets")
                saved = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
                saved["inner_protocol"][field] = value
                tampered = InnerProtocol(**saved["inner_protocol"])
                saved["protocol_id"] = tampered.fingerprint
                (root / "protocol.json").write_text(json.dumps(saved), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "only in lr"):
                    load_cooperative_warm_start(root, config, requested)

    def test_deepsets_requires_explicit_source_dimensions_and_matching_selection_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            config, requested, _ = _fixture(root, "deepsets")
            spec_path = root / "run_spec.json"
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            spec["config"].pop("features")
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lacks features"):
                load_cooperative_warm_start(root, config, requested)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            config, requested, _ = _fixture(root, "deepsets")
            inputs = torch.load(root / "inputs.pt", map_location="cpu", weights_only=False)
            inputs["selection_tasks"][0].x_support[0, 0, 0] = 9
            torch.save(inputs, root / "inputs.pt")
            with self.assertRaisesRegex(ValueError, "reuse its training support"):
                load_cooperative_warm_start(root, config, requested)

    def test_final_test_replay_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            config, requested, _ = _fixture(root, "deepsets", test_replay=True)
            with self.assertRaisesRegex(ValueError, "final-test"):
                load_cooperative_warm_start(root, config, requested)

    def test_frozen_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            config, requested, _ = _fixture(root, "pattern")
            (root / "frozen.pt").write_bytes(b"sealed")
            with self.assertRaisesRegex(ValueError, "never frozen.pt"):
                load_cooperative_warm_start(root / "frozen.pt", config, requested)


if __name__ == "__main__":
    unittest.main()
