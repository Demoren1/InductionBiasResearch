import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="experiments/dynamic_signal_v1/protocol.json"
    )
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    root = Path(plan["run"])
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(plan["seed"])
    blocks, width = plan["blocks"], plan["block_size"]
    permutation = rng.permutation(blocks * width)
    for role, count, offset in [
        ("calibration", plan["calibration_tasks"], 100),
        ("source", plan["source_tasks"], 200),
        ("target", plan["target_tasks"], 300),
    ]:
        directory = root / role
        directory.mkdir()
        generator = np.random.default_rng(plan["seed"] + offset)
        directions = generator.normal(size=(count, width)).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        for noise_id, sigma in enumerate(plan["noise"]):
            observations = {}
            data_rng = np.random.default_rng(plan["seed"] + offset * 100 + noise_id)
            for split, rows in [
                ("support", max(plan["target_sizes"]) if role != "source" else 16384),
                ("query", plan["query_size"]),
                ("test", plan["test_size"]),
            ]:
                labels = data_rng.integers(0, 2, size=(count, rows)).astype(np.float32)
                positions = data_rng.integers(blocks, size=(count, rows))
                values = (
                    sigma * data_rng.normal(size=(count, rows, blocks, width))
                ).astype(np.float32)
                values[np.arange(count)[:, None], np.arange(rows)[None], positions] += (
                    2 * labels[..., None] - 1
                ) * directions[:, None]
                flat = values.reshape(count, rows, -1)
                observed = np.empty_like(flat)
                observed[..., permutation] = flat
                observations[split + "_x"] = observed
                observations[split + "_y"] = labels
            np.savez_compressed(directory / f"noise_{noise_id}.npz", **observations)
        np.savez_compressed(
            directory / "teacher_audit.npz",
            directions=directions,
            permutation=permutation,
        )
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*.npz")
    }
    (root / "data_manifest.json").write_text(
        json.dumps({"protocol": plan, "hashes": hashes}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
