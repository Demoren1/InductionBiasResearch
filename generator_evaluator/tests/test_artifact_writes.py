import errno
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from generator_evaluator import artifacts


class ArtifactWriteTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def assert_failed_write_preserved(self, path, previous_bytes, legacy_temp, legacy_bytes, error):
        self.assertEqual(error.errno, errno.ENOSPC)
        self.assertIn("No space left on device", str(error))
        self.assertIn(str(path), str(error))
        self.assertEqual(path.read_bytes(), previous_bytes)
        self.assertEqual(legacy_temp.read_bytes(), legacy_bytes)
        self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_save_json_cleans_its_partial_temp_and_keeps_previous_file(self):
        path = self.tmp_path / "summary.json"
        path.write_text('{"previous": true}\n', encoding="utf-8")
        previous_bytes = path.read_bytes()
        legacy_temp = path.with_suffix(path.suffix + ".tmp")
        legacy_temp.write_bytes(b"another writer's temporary file")
        legacy_bytes = legacy_temp.read_bytes()

        real_fdopen = artifacts.os.fdopen

        class PartialWriter:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def write(self, value):
                self.stream.write(value[:8])
                raise OSError(errno.ENOSPC, "No space left on device")

            def flush(self):
                return self.stream.flush()

        with patch.object(
                artifacts.os,
                "fdopen",
                lambda descriptor, *args, **kwargs: PartialWriter(
                    real_fdopen(descriptor, *args, **kwargs)),
        ):
            with self.assertRaises(OSError) as error:
                artifacts.save_json(path, {"replacement": True})

        self.assert_failed_write_preserved(
            path, previous_bytes, legacy_temp, legacy_bytes, error.exception)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"previous": True})

    def test_save_torch_cleans_partial_temp_and_keeps_previous_checkpoint(self):
        path = self.tmp_path / "checkpoint.pt"
        previous = {"weight": torch.tensor([1.0, 2.0])}
        torch.save(previous, path)
        previous_bytes = path.read_bytes()
        legacy_temp = path.with_suffix(path.suffix + ".tmp")
        legacy_temp.write_bytes(b"another writer's temporary file")
        legacy_bytes = legacy_temp.read_bytes()

        def partial_save(payload, stream):
            stream.write(b"partial checkpoint")
            raise OSError(errno.ENOSPC, "No space left on device")

        with patch.object(artifacts.torch, "save", partial_save):
            with self.assertRaises(OSError) as error:
                artifacts.save_torch(path, {"weight": torch.tensor([9.0])})

        self.assert_failed_write_preserved(
            path, previous_bytes, legacy_temp, legacy_bytes, error.exception)
        restored = torch.load(path, map_location="cpu", weights_only=True)
        self.assertTrue(torch.equal(restored["weight"], previous["weight"]))

    def test_save_json_round_trips_and_removes_temporary_file(self):
        path = self.tmp_path / "summary.json"
        payload = {"unicode": "ёж", "values": [1, 2, 3]}

        artifacts.save_json(path, payload)

        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)
        self.assertEqual(list(self.tmp_path.glob(f".{path.name}.*.tmp")), [])

    def test_pytorch_stream_failure_cleans_temp_and_preserves_checkpoint(self):
        path = self.tmp_path / "checkpoint.pt"
        torch.save({"weight": torch.tensor([1.0])}, path)
        previous = path.read_bytes()

        def partial_save(payload, stream):
            stream.write(b"partial zip archive")
            raise RuntimeError("PytorchStreamWriter failed writing file data/5769")

        with patch.object(artifacts.torch, "save", partial_save):
            with self.assertRaisesRegex(RuntimeError, "checkpoint.pt.*PytorchStreamWriter"):
                artifacts.save_torch(path, {"weight": torch.tensor([2.0])})
        self.assertEqual(path.read_bytes(), previous)
        self.assertEqual(list(self.tmp_path.glob(f".{path.name}.*.tmp")), [])

    def test_replay_write_failure_also_preserves_previous_artifact(self):
        from generator_evaluator.data import InnerProtocol, RealReplay
        path = self.tmp_path / "replay.pt"
        replay = RealReplay(InnerProtocol(steps=1, replicas=2))
        replay.save(path)
        previous = path.read_bytes()

        def partial_save(payload, stream):
            stream.write(b"partial replay")
            raise RuntimeError("PytorchStreamWriter failed writing file")

        with patch.object(artifacts.torch, "save", partial_save):
            with self.assertRaisesRegex(RuntimeError, "replay.pt"):
                replay.save(path)
        self.assertEqual(path.read_bytes(), previous)
        self.assertEqual(list(self.tmp_path.glob(".replay.pt.*.tmp")), [])
        self.assertEqual(RealReplay.load(path).records, [])

    def test_save_torch_round_trips_on_cpu_and_removes_temporary_file(self):
        path = self.tmp_path / "checkpoint.pt"
        payload = {"weight": torch.tensor([[1.0, 2.0]], device="cpu")}

        artifacts.save_torch(path, payload)

        restored = torch.load(path, map_location="cpu", weights_only=True)
        self.assertTrue(torch.equal(restored["weight"], payload["weight"]))
        self.assertEqual(list(self.tmp_path.glob(f".{path.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
