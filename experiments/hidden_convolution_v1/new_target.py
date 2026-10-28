"""Generate a fresh audit-only teacher and new observations after pilot development."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from benchmark import predict

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/pilot_v4")
    parser.add_argument("--seed", type=int, default=283100)
    args = parser.parse_args()
    if (args.out / "evaluation_frozen.json").exists():
        parser.error("Cannot replace target data after evaluation was frozen")
    layout = np.load(args.out / "source_teacher.npz")
    rng = np.random.default_rng(args.seed)
    choices = rng.integers(4, size=(8, 36))
    indices = np.zeros((8, 64, 36), dtype=np.int64)
    for task in range(8):
        for hidden, window in enumerate(layout["windows"]):
            indices[task, window, hidden] = 1 + 9 * choices[task, hidden] + np.arange(9)
    values = rng.uniform(-1, 1, size=(8, 37)).astype("float32")
    values[:, 0] = 0
    x = rng.normal(size=(256 + 512 + 4096, 64)).astype("float32")
    weights = np.take_along_axis(values[:, :, None], indices, axis=1)
    with torch.no_grad():
        y = predict(torch.from_numpy(x), torch.from_numpy(weights)).numpy()
    np.savez_compressed(args.out / "target_data.npz", x=x, y=y)
    np.savez_compressed(
        args.out / "target_teacher.npz",
        indices=indices,
        values=values,
        choices=choices,
        permutation=layout["permutation"],
        windows=layout["windows"],
    )
    (args.out / "fresh_target_manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "tasks": 8,
                "support": 256,
                "query": 512,
                "test": 4096,
                "purpose": "fresh evaluation after source-only alignment changes following pilot_v2",
                "data_sha256": hashlib.sha256(
                    (args.out / "target_data.npz").read_bytes()
                ).hexdigest(),
                "teacher_sha256": hashlib.sha256(
                    (args.out / "target_teacher.npz").read_bytes()
                ).hexdigest(),
                "generator_only_reads_teacher_layout": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
