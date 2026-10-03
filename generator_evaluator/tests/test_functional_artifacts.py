import tempfile
from pathlib import Path
import unittest

import torch

from generator_evaluator.functional_artifacts import write_functional_card


class FunctionalArtifactTests(unittest.TestCase):
    def test_card_is_compact_and_preserves_both_model_state_schemas(self) -> None:
        hidden, features, probes = 32, 784, 32
        token = torch.randn(hidden, probes + 4 * features)
        mask_batch = torch.zeros(8, features, hidden)
        mask_batch[3, :10, :4] = 1
        mask = mask_batch[3]
        state_batch = torch.randn(8, features, hidden)
        deep_state = {
            "weight": state_batch[5],
            "bias": torch.randn(hidden),
            "readout": torch.randn(hidden),
            "per_image_offset": torch.tensor(0.125),
        }
        pattern_state = {
            "w": state_batch[6],
            "b": torch.randn(hidden),
            "v": torch.randn(hidden),
            "c": torch.tensor(-0.25),
        }
        metadata = {
            "kind": "initial_teacher",
            "candidate_id": 17,
            "initialization_seed": 1234,
            "score": 0.03125,
            "task_source": {"task_id": "source:fixture", "provenance": {"pool": "source_train"}},
        }
        with tempfile.TemporaryDirectory() as folder:
            for name, state in (("deepsets", deep_state), ("pattern", pattern_state)):
                path = Path(folder) / f"{name}.pt"
                written = write_functional_card(path, token=token, mask=mask,
                                                state=state, metadata=metadata)
                self.assertEqual(written, path)
                card = torch.load(path, map_location="cpu", weights_only=False)
                self.assertEqual(card["schema"], "generator_evaluator.functional_map_card")
                self.assertEqual(card["schema_version"], 1)
                self.assertEqual(set(card["state_dict"]), set(state))
                self.assertTrue(card["mask"].dtype == torch.bool)
                torch.testing.assert_close(card["mask"], mask.bool())
                torch.testing.assert_close(card["token"], token.float())
                for key, value in state.items():
                    self.assertEqual(card["state_dict"][key].dtype, torch.float32)
                    torch.testing.assert_close(card["state_dict"][key], value.float())
                    self.assertEqual(card["state_dict"][key].untyped_storage().nbytes(),
                                     card["state_dict"][key].numel() * 4)
                self.assertNotIn("optimizer_state", card)
                self.assertNotIn("history", card)
                self.assertEqual(card["metadata"], metadata)
                self.assertLess(path.stat().st_size, 1_000_000)

    def test_card_rejects_optimizer_and_history_metadata(self) -> None:
        token = torch.zeros(2, 4)
        mask = torch.zeros(3, 2)
        mask[0, 0] = 1
        state = {"w": torch.zeros(3, 2)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "must_not_exist.pt"
            for field in ("optimizer_state", "training_history"):
                with self.assertRaisesRegex(ValueError, "optimizer or history"):
                    write_functional_card(path, token=token, mask=mask, state=state,
                                          metadata={field: {"step": 1}})
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
