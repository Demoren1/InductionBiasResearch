import io
import unittest

import torch

from generator_evaluator.data import InnerProtocol, TaskData
from generator_evaluator.data import tensor_hash
from generator_evaluator.measurements import _payload, measurement_digest


class CompactArtifactTests(unittest.TestCase):
    def test_payload_serializes_only_mask_and_id_view_contents(self) -> None:
        torch.set_num_threads(1)
        # One candidate view retains a 100 MB [1000, 784, 32] batch unless the
        # artifact explicitly takes ownership of just that candidate.
        mask_batch = torch.zeros(1000, 784, 32, dtype=torch.float32)
        mask = mask_batch[7]
        mask[0, 0] = 1.0
        mask[12, 3] = 1.0

        id_pool = torch.arange(1_000_000, dtype=torch.int64)
        support_ids = id_pool[10:14]
        query_ids = id_pool[10_000:10_003]
        x_support = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y_support = torch.tensor([0.0, 1.0, 1.0, 0.0])
        x_query = torch.tensor([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
        y_query = torch.tensor([0.0, 1.0, 0.0])
        task = TaskData(
            "compact-artifact-fixture", "train", x_support, y_support, x_query, y_query,
            torch.tensor([1.0, -1.0]), support_ids, query_ids,
            {"family": "fixture", "source_hash": "fixture-hash"},
        )
        protocol = InnerProtocol(steps=2, replicas=2, checkpoint_every=1)
        result = {"quality": 0.25, "label_source": "fresh_terminal_query"}
        expected_mask = mask.clone()
        expected_support_ids = support_ids.clone()
        expected_query_ids = query_ids.clone()
        expected_task_fingerprint = task.fingerprint
        expected_mask_key = tensor_hash(mask)
        expected_digest = measurement_digest(mask, task, protocol)

        payload = _payload(mask, task, protocol, result)

        torch.testing.assert_close(payload["mask"], expected_mask)
        torch.testing.assert_close(payload["support_ids"], expected_support_ids)
        torch.testing.assert_close(payload["query_ids"], expected_query_ids)
        self.assertEqual(payload["mask_key"], expected_mask_key)
        self.assertEqual(payload["task_fingerprint"], expected_task_fingerprint)
        self.assertEqual(measurement_digest(payload["mask"], task, protocol), expected_digest)
        self.assertEqual(payload["task_provenance"], task.provenance)
        self.assertIs(payload["result"], result)

        for name in ("mask", "support_ids", "query_ids"):
            value = payload[name]
            self.assertEqual(value.untyped_storage().nbytes(), value.numel() * value.element_size())
        self.assertGreater(mask.untyped_storage().nbytes(), payload["mask"].untyped_storage().nbytes() * 900)
        self.assertGreater(support_ids.untyped_storage().nbytes(), payload["support_ids"].untyped_storage().nbytes() * 100_000)

        buffer = io.BytesIO()
        torch.save(payload, buffer)
        self.assertLess(buffer.tell(), 1_000_000)

        # Cloning for serialization must not alter the source views or their
        # task provenance hashes.
        torch.testing.assert_close(mask, expected_mask)
        torch.testing.assert_close(support_ids, expected_support_ids)
        torch.testing.assert_close(query_ids, expected_query_ids)
        self.assertEqual(task.fingerprint, expected_task_fingerprint)
        self.assertEqual(measurement_digest(mask, task, protocol), expected_digest)


if __name__ == "__main__":
    unittest.main()
