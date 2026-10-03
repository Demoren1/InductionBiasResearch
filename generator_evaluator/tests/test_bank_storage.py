from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from generator_evaluator.adapters import FunctionalBank
from generator_evaluator.bank_storage import externalize_banks
from generator_evaluator.artifacts import save_torch


def _bank(*, view_storage: torch.Tensor | None = None) -> FunctionalBank:
    generator = torch.Generator().manual_seed(73)
    tokens = torch.randn((1, 3, 4, 4096), generator=generator)
    masks = torch.zeros((3, 5, 4))
    masks[0, 0, 0] = 1
    masks[1, 1, 1] = 1
    masks[2, 2, 2] = 1
    states = []
    for row in range(3):
        states.append({
            "q_raw": torch.randn((5, 4), generator=generator, dtype=torch.bfloat16),
            "state_dict": {"weight": torch.randn((5, 4), generator=generator),
                           "bias": torch.tensor([row, -row], dtype=torch.int16)},
            "optimizer_state": {"step": row, "momentum": torch.full((2,), row / 4)},
            "history": [{"loss": float(row) + 0.25}],
            "source": {"kind": "initial_teacher", "candidate_id": row,
                       "source_mask": torch.ones((256, 256), dtype=torch.float32)},
        })
    if view_storage is not None:
        states[0]["state_dict"]["small_view"] = view_storage[17:49]
    diagnostics = {
        "aligned_q_abs": torch.arange(3 * 5 * 4, dtype=torch.float32).reshape(3, 5, 4),
        "large_probe": torch.arange(256 * 256, dtype=torch.float32).reshape(256, 256),
        "ids": torch.tensor([2, 9, 11], dtype=torch.int64),
        "nested": ("kept", {"enabled": True}),
    }
    return FunctionalBank(
        tokens=tokens,
        quality=torch.tensor([[[0.1], [0.2], [0.3]]]),
        masks=masks,
        baseline_mask=masks[0].clone(),
        provenance={"family": "fixture", "seed": 73, "source": {"task_id": "teacher-run"}},
        states=states,
        diagnostics=diagnostics,
    )


def _assert_nested_equal(test: unittest.TestCase, left, right) -> None:
    if torch.is_tensor(left):
        test.assertTrue(torch.is_tensor(right))
        test.assertEqual(left.dtype, right.dtype)
        test.assertEqual(tuple(left.shape), tuple(right.shape))
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertIsInstance(right, dict)
        test.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertIsInstance(right, type(left))
        test.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(test, left_item, right_item)
    else:
        test.assertEqual(left, right)


def _save_externalized(path: Path, payload) -> None:
    save_torch(path, externalize_banks(payload, path))


class BankStorageTests(unittest.TestCase):
    def test_round_trip_is_transparent_and_checkpoint_stays_small(self) -> None:
        bank = _bank()
        payload = {"banks": [bank], "wrapping": ("outer", {"bank": bank})}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plain_path = root / "plain.pt"
            compact_path = root / "checkpoint.pt"
            torch.save(payload, plain_path)
            _save_externalized(compact_path, payload)

            legacy = torch.load(plain_path, map_location="cpu", weights_only=False)
            self.assertIsInstance(legacy["banks"][0], FunctionalBank)
            restored = torch.load(compact_path, map_location="cpu", weights_only=False)
            self.assertIsInstance(restored["banks"][0], FunctionalBank)
            self.assertIs(restored["banks"][0], restored["wrapping"][1]["bank"])
            _assert_nested_equal(self, bank.tokens, restored["banks"][0].tokens)
            _assert_nested_equal(self, bank.quality, restored["banks"][0].quality)
            _assert_nested_equal(self, bank.masks, restored["banks"][0].masks)
            _assert_nested_equal(self, bank.baseline_mask, restored["banks"][0].baseline_mask)
            _assert_nested_equal(self, bank.provenance, restored["banks"][0].provenance)
            _assert_nested_equal(self, bank.states, restored["banks"][0].states)
            _assert_nested_equal(self, bank.diagnostics, restored["banks"][0].diagnostics)
            self.assertLess(compact_path.stat().st_size, plain_path.stat().st_size // 5)

    def test_snapshots_reuse_unchanged_rows_and_keep_changed_rows_independent(self) -> None:
        original = _bank()
        changed = deepcopy(original)
        changed.tokens[0, 1, 0, 0] += 0.5
        changed.diagnostics["aligned_q_abs"][1, 0, 0] += 17
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first_path = root / "checkpoint.pt"
            second_path = root / "best.pt"
            repeat_path = root / "final.pt"
            _save_externalized(first_path, {"bank": original})
            assets_path = root / "bank_assets"
            original_assets = {path.name: path.stat().st_ino for path in assets_path.glob("*.pt")}
            self.assertEqual(len(original_assets), 8)  # rows, aligned maps, probe, and shared source mask

            _save_externalized(second_path, {"bank": changed})
            self.assertEqual(len(list(assets_path.glob("*.pt"))), len(original_assets) + 2)
            second_assets = {path.name: path.stat().st_ino for path in assets_path.glob("*.pt")}
            for asset_name, inode in original_assets.items():
                self.assertEqual(second_assets[asset_name], inode)

            with patch("generator_evaluator.bank_storage.torch.load", wraps=torch.load) as load:
                _save_externalized(repeat_path, {"bank": original})
                self.assertEqual(load.call_count, 0)
            repeated_assets = {path.name: path.stat().st_ino for path in assets_path.glob("*.pt")}
            self.assertEqual(repeated_assets, second_assets)

            restored_first = torch.load(first_path, weights_only=False)["bank"]
            restored_second = torch.load(second_path, weights_only=False)["bank"]
            torch.testing.assert_close(restored_first.tokens, original.tokens, rtol=0, atol=0)
            torch.testing.assert_close(restored_second.tokens, changed.tokens, rtol=0, atol=0)
            torch.testing.assert_close(restored_first.diagnostics["aligned_q_abs"],
                                       original.diagnostics["aligned_q_abs"], rtol=0, atol=0)
            torch.testing.assert_close(restored_second.diagnostics["aligned_q_abs"],
                                       changed.diagnostics["aligned_q_abs"], rtol=0, atol=0)

    def test_teacher_asset_compacts_oversized_cpu_views(self) -> None:
        parent = torch.arange(400_000, dtype=torch.float32)
        bank = _bank(view_storage=parent)
        expected = parent[17:49].clone()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "inputs.pt"
            _save_externalized(path, {"bank": bank})
            restored = torch.load(path, weights_only=False)["bank"]
            view = restored.states[0]["state_dict"]["small_view"]
            torch.testing.assert_close(view, expected, rtol=0, atol=0)
            self.assertEqual(view.device.type, "cpu")
            self.assertEqual(view.untyped_storage().nbytes(), view.numel() * view.element_size())

    def test_missing_and_corrupt_assets_have_clear_errors(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            missing_path = root / "missing.pt"
            _save_externalized(missing_path, {"bank": _bank()})
            asset_path = next((root / "bank_assets").glob("*.pt"))
            asset_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "FunctionalBank asset is missing"):
                torch.load(missing_path, weights_only=False)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            corrupt_path = root / "corrupt.pt"
            _save_externalized(corrupt_path, {"bank": _bank()})
            asset_path = next((root / "bank_assets").glob("*.pt"))
            asset_path.write_bytes(b"broken torch asset")
            with self.assertRaisesRegex(ValueError, "FunctionalBank asset is corrupt or unreadable"):
                torch.load(corrupt_path, weights_only=False)

    def test_corrupt_existing_asset_is_rejected_on_a_repeated_save(self) -> None:
        bank = _bank()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _save_externalized(root / "first.pt", {"bank": bank})
            asset_path = next((root / "bank_assets").glob("*.pt"))
            asset_path.write_bytes(b"broken after initial save")
            with self.assertRaisesRegex(ValueError, "FunctionalBank asset is corrupt or unreadable"):
                _save_externalized(root / "second.pt", {"bank": bank})

    def test_changed_asset_stat_triggers_checksum_verification(self) -> None:
        bank = _bank()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _save_externalized(root / "first.pt", {"bank": bank})
            asset_path = next((root / "bank_assets").glob("*.pt"))
            prior_stat = asset_path.stat()
            os.utime(asset_path, ns=(prior_stat.st_atime_ns, prior_stat.st_mtime_ns + 2_000_000_000))
            with patch("generator_evaluator.bank_storage.torch.load", wraps=torch.load) as load:
                _save_externalized(root / "second.pt", {"bank": bank})
                loaded_paths = [Path(call.args[0]) for call in load.call_args_list]
            self.assertEqual(loaded_paths, [asset_path.resolve()])

    def test_bootstrap_and_search_descendants_share_the_run_asset_directory(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "20261003_fixture"
            bootstrap = run / "bootstrap" / "checkpoint.pt"
            search = run / "search" / "checkpoint.pt"
            role_bank = run / "bootstrap" / "bank_build" / "0" / "roles" / "inputs.pt"
            # Saving the same content through these paths creates one asset set.
            bank = _bank()
            _save_externalized(bootstrap, {"bank": bank})
            _save_externalized(search, {"bank": bank})
            _save_externalized(role_bank, {"bank": bank})
            self.assertEqual(len(list((run / "bank_assets").glob("*.pt"))), 8)

    def test_source_metadata_changes_do_not_create_new_teacher_row_assets(self) -> None:
        original = _bank()
        changed = deepcopy(original)
        changed.states[1]["source"].update({
            "artifact_path": "/new/path/teacher.pt",
            "functional_artifact_path": "/new/path/card.pt",
            "query_error": 0.125,
        })
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first_path = root / "inputs.pt"
            changed_path = root / "checkpoint.pt"
            first_payload = externalize_banks({"bank": original}, first_path)
            first_rows = first_payload["bank"]._manifest["rows"]
            asset_directory = root / "bank_assets"
            row_asset = torch.load(asset_directory / f"{first_rows[1]}.pt", weights_only=False)
            self.assertNotIn("source", row_asset["state"])
            save_torch(first_path, first_payload)

            changed_payload = externalize_banks({"bank": changed}, changed_path)
            changed_rows = changed_payload["bank"]._manifest["rows"]
            self.assertEqual(first_rows, changed_rows)
            save_torch(changed_path, changed_payload)
            self.assertEqual(len(list(asset_directory.glob("*.pt"))), 8)

            restored_first = torch.load(first_path, weights_only=False)["bank"]
            restored_changed = torch.load(changed_path, weights_only=False)["bank"]
            _assert_nested_equal(self, original.states, restored_first.states)
            _assert_nested_equal(self, changed.states, restored_changed.states)
            first_source = restored_first.states[1]["source"]
            changed_source = restored_changed.states[1]["source"]
            self.assertIsNot(first_source, changed_source)
            self.assertIsNot(restored_first.states[0]["source"], restored_first.states[1]["source"])
            self.assertIsNot(first_source["source_mask"], changed_source["source_mask"])

    def test_non_dict_and_source_less_teacher_states_round_trip(self) -> None:
        bank = _bank()
        bank.states = [{"state_dict": {"weight": torch.arange(4)}}, None, ("opaque", 7)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "states.pt"
            _save_externalized(path, {"bank": bank})
            restored = torch.load(path, weights_only=False)["bank"]
            _assert_nested_equal(self, bank.states, restored.states)


if __name__ == "__main__":
    unittest.main()
