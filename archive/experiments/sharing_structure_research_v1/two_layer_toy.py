"""A visible full function whose first coordinate prefix cancels exactly."""

import argparse
import math
from pathlib import Path

import numpy as np

from experiments.darts_indexator_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .guarded_prefix import extract
from .indexed_readout import design, head_metrics, head_path
from .pattern_prefix import prefix_inputs, timestamp
from .unknown_readout import teacher_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve() / "cancellation_control"
    root.mkdir(exist_ok=False)
    spec = {
        "inputs": 2,
        "hidden": 2,
        "codes": 1,
        "outputs": 1,
        "bias": 0.15,
        "slice_points": 2048,
        "prefix_validation": 512,
        "slice_repeats": 4096,
        "slice_scales": [0.05, 0.2, 0.8, 3.2, 12.8, 51.2],
    }
    save_json(
        root / "protocol.json",
        {
            "registered_utc": timestamp(),
            "code_sha256": digest(Path(__file__)),
            "design": "Post-pilot theoretical control. Two neurons with W1 columns (c,c)/(c,-c), true shared W2=(a,-a). First-coordinate prefix is identically zero, full function is nonzero. True C1 supplied only to isolate prefix observability; J2 not supplied. Four tasks, raw noise0/.2 withR4096, old/keep2 unchanged. Not part of primary generalization score.",
            "spec": spec,
        },
    )
    rng = np.random.default_rng(26110631059)
    index = np.array([[1, 1], [1, -1]], dtype=np.int16)
    codes = np.array([[0.7], [1.0], [1.3], [1.6]])
    head = (
        np.array([0.6, 0.8, 1.1, 1.4])[:, None, None] * np.array([1, -1])[None, :, None]
    )
    offset = np.zeros((4, 1))
    order = np.arange(2)
    x, vx = prefix_inputs(rng, order, 2048, spec), prefix_inputs(rng, order, 512, spec)
    train_x, validation_x = rng.normal(size=(4096, 2)), rng.normal(size=(1024, 2))
    zero = teacher_output(np.zeros((1, 2)), index, codes, head, offset, 0.15)[:, :1]
    clean = teacher_output(x, index, codes, head, offset, 0.15) - zero
    vclean = teacher_output(vx, index, codes, head, offset, 0.15) - zero
    noise, vnoise = rng.normal(size=clean.shape), rng.normal(size=vclean.shape)
    arrays, pending = {}, []
    for sigma in (0.0, 0.2):
        data = {
            "order": order,
            "prefix_x_0": x,
            "prefix_vx_0": vx,
            "prefix_y_0": clean + sigma * noise / 64,
            "prefix_vy_0": vclean + sigma * vnoise / 64,
        }
        for task in range(4):
            for mode in ("old", "keep2"):
                found, trace, prefixes = extract(
                    data, codes[task], spec, task, 0, sigma, mode
                )
                key = f"{sigma}_{task}_{mode}"
                arrays[key] = found
                pending.append(
                    {
                        "key": key,
                        "task": task,
                        "mode": mode,
                        "noise": sigma,
                        "trace": trace,
                        "first_prefix_atoms": prefixes[0].tolist() if prefixes else [],
                    }
                )
    np.savez_compressed(root / "indexators.npz", **arrays)
    for row in pending:
        metric, _, columns = align_indices(arrays[row["key"]], index, 1)
        row.update(
            exact=row["trace"]["complete"] and metric["categorical_errors"] == 0,
            categorical_errors=metric["categorical_errors"],
        )
        if row["exact"]:
            task = row["task"]
            phi = design(train_x, arrays[row["key"]], codes[task : task + 1], 0.15)
            vphi = design(
                validation_x, arrays[row["key"]], codes[task : task + 1], 0.15
            )
            y = (
                teacher_output(
                    train_x,
                    index,
                    codes[task : task + 1],
                    head[task : task + 1],
                    offset[:1],
                    0.15,
                )
                - zero[task : task + 1]
            )
            vy = (
                teacher_output(
                    validation_x,
                    index,
                    codes[task : task + 1],
                    head[task : task + 1],
                    offset[:1],
                    0.15,
                )
                - zero[task : task + 1]
            )
            y += row["noise"] * rng.normal(size=y.shape) / math.sqrt(4096)
            vy += row["noise"] * rng.normal(size=vy.shape) / math.sqrt(4096)
            _, selected = head_path(phi, y, vphi, vy)
            row["head"] = head_metrics(
                selected["index"][columns], np.array([[1], [-1]])
            )
    save_json(
        root / "evaluation.json",
        {
            "evaluated_utc": timestamp(),
            "max_clean_first_prefix": float(np.max(np.abs(clean[:, :2048]))),
            "rows": pending,
        },
    )
    print(
        {
            (sigma, mode): sum(
                r["exact"] for r in pending if r["noise"] == sigma and r["mode"] == mode
            )
            for sigma in (0.0, 0.2)
            for mode in ("old", "keep2")
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
