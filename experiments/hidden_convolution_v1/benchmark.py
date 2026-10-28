"""Generate a hidden-window teacher; solvers consume separate observation files."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def predict(inputs, weights):
    return torch.einsum("bd,mdh->mbh", inputs, weights).relu().sum(-1) / 6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/pilot_v1")
    parser.add_argument("--source-tasks", type=int, default=32)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(283000)
    permutation = rng.permutation(64)
    windows = np.array(
        [
            permutation[
                (r + np.arange(3)[:, None]) * 8 + c + np.arange(3)[None, :]
            ].reshape(-1)
            for r in range(6)
            for c in range(6)
        ]
    )
    all_choices = []
    for split, tasks, support, query, test in [
        ("source", args.source_tasks, 16384, 2048, 0),
        ("calibration", 2, 256, 512, 0),
        ("target", 8, 256, 512, 4096),
    ]:
        choices = rng.integers(4, size=(tasks, 36))
        for choice in choices:
            while any(np.array_equal(choice, old) for old in all_choices):
                choice[:] = rng.integers(4, size=36)
            all_choices.append(choice.copy())
        indices = np.zeros((tasks, 64, 36), dtype=np.int64)
        values = rng.uniform(-1, 1, size=(tasks, 37)).astype("float32")
        values[:, 0] = 0
        if split == "source":
            # Separated source values help generic scalar quantization; their
            # values and their layout are never supplied to the source fitter.
            for task in range(tasks):
                values[task, 1:] = rng.permutation(np.linspace(-1, 1, 36))
                values[task, 1:] += rng.uniform(-0.003, 0.003, size=36)
        for task in range(tasks):
            for hidden, window in enumerate(windows):
                indices[task, window, hidden] = (
                    1 + choices[task, hidden] * 9 + np.arange(9)
                )
        x = rng.normal(size=(support + query + test, 64)).astype("float32")
        weights = np.take_along_axis(values[:, :, None], indices, axis=1)
        with torch.no_grad():
            y = predict(torch.from_numpy(x), torch.from_numpy(weights)).numpy()
        np.savez_compressed(args.out / f"{split}_data.npz", x=x, y=y)
        np.savez_compressed(
            args.out / f"{split}_teacher.npz",
            indices=indices,
            values=values,
            choices=choices,
            permutation=permutation,
            windows=windows,
        )
    (args.out / "solver_protocol.json").write_text(
        json.dumps(
            {
                "inputs": 64,
                "hidden": 36,
                "codes": 36,
                "fixed_zero_code": 0,
                "fixed_readout": 1 / 6,
                "fixed_bias": 0,
                "source_tasks": args.source_tasks,
                "source_support": 16384,
                "source_query": 2048,
                "target_support": 256,
                "target_query": 512,
                "target_test": 4096,
            },
            indent=2,
        )
    )
    (args.out / "teacher_protocol.json").write_text(
        json.dumps(
            {
                "seed": 283000,
                "latent_grid": [8, 8],
                "window": [3, 3],
                "filters": 4,
                "window_assignments": 36,
                "formal_assignments": str(4**36),
                "source_values": "permuted separated grid plus uniform jitter .003",
                "new_values": "continuous uniform[-1,1]",
                "teacher_access": "benchmark generation, explicit oracle control, post-freeze diagnostics",
            },
            indent=2,
        )
    )
    print("Generated observation files:", args.out, flush=True)


if __name__ == "__main__":
    main()
