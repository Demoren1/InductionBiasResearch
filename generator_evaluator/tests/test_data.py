import tempfile
from pathlib import Path
import unittest

import torch

from generator_evaluator.data import InnerProtocol, RealReplay, TaskData, support_context, topology_id


class ReplayTests(unittest.TestCase):
    def task(self, split="train"):
        x = torch.tensor([[1., -1.], [-1., 1.]])
        y = torch.tensor([0., 1.])
        return TaskData("task", split, x, y, -x, 1-y, support_context(x, y),
                        torch.tensor([1, 2]), torch.tensor([3, 4]))

    def result(self, protocol):
        return dict(label_source="fresh_terminal_query", fixed_horizon=True,
                    protocol_id=protocol.fingerprint, replica_losses=torch.tensor([.3, .4]),
                    seeds=[5, 6], plateau_flags=[False, False])

    def test_column_permutations_share_split_and_roundtrip(self):
        p = InnerProtocol(steps=2)
        replay = RealReplay(p)
        mask = torch.tensor([[1., 0., 1.], [0., 1., 1.]])
        reordered = mask[:, [2, 0, 1]]
        self.assertEqual(topology_id(mask), topology_id(reordered))
        self.assertEqual(replay.mask_split(mask), replay.mask_split(reordered))
        with tempfile.TemporaryDirectory() as folder:
            artifact = Path(folder) / "child.pt"
            torch.save(self.result(p), artifact)
            a = replay.append(mask, self.task(), self.result(p), origin="random", artifact_path=artifact)
            b = replay.append(reordered, self.task(), self.result(p), origin="permutation", artifact_path=artifact)
            self.assertEqual(a["split"], b["split"])
            replay.save(Path(folder) / "replay.pt")
            restored = RealReplay.load(Path(folder) / "replay.pt")
            self.assertEqual(restored.records, replay.records)
            torch.testing.assert_close(restored.tensors(a["split"])[2], torch.tensor([.35, .35]))

    def test_reject_prediction_protocol_changes_and_task_leakage(self):
        p = InnerProtocol(steps=2)
        replay = RealReplay(p)
        mask = torch.eye(2)
        with tempfile.TemporaryDirectory() as folder:
            artifact = Path(folder) / "child.pt"
            torch.save({}, artifact)
            result = self.result(p)
            result["label_source"] = "surrogate"
            with self.assertRaises(ValueError):
                replay.append(mask, self.task(), result, origin="bad", artifact_path=artifact)
            result = self.result(InnerProtocol(steps=3))
            with self.assertRaises(ValueError):
                replay.append(mask, self.task(), result, origin="bad", artifact_path=artifact)
            replay.append(mask, self.task(), self.result(p), origin="real", artifact_path=artifact)
            with self.assertRaises(ValueError):
                replay.append(mask, self.task("validation"), self.result(p), origin="bad", artifact_path=artifact)

    def test_query_ids_cannot_overlap_support(self):
        task = self.task()
        with self.assertRaises(ValueError):
            TaskData("overlap", "train", task.x_support, task.y_support,
                     task.x_query, task.y_query, task.context, task.support_ids, task.support_ids)

    def test_observation_ids_must_cover_measured_rows(self):
        task = self.task()
        with self.assertRaisesRegex(ValueError, "every measured"):
            TaskData("missing_id", "train", task.x_support, task.y_support,
                     task.x_query, task.y_query, task.context, task.support_ids[:1], task.query_ids)

    def test_corrupt_replay_cannot_cross_topology_partitions(self):
        protocol = InnerProtocol(steps=2)
        replay = RealReplay(protocol)
        with tempfile.TemporaryDirectory() as folder:
            artifact = Path(folder) / "child.pt"
            torch.save(self.result(protocol), artifact)
            replay.append(torch.eye(2), self.task(), self.result(protocol), origin="real", artifact_path=artifact)
            path = Path(folder) / "replay.pt"
            replay.save(path)
            saved = torch.load(path, weights_only=False)
            saved["records"][0]["split"] = "meta_validation"
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "crossed partitions"):
                RealReplay.load(path)

    def test_solver_budget_changes_target_identity(self):
        self.assertNotEqual(InnerProtocol().fingerprint, InnerProtocol(steps=20).fingerprint)
        self.assertNotEqual(InnerProtocol().fingerprint, InnerProtocol(lr=.02).fingerprint)
