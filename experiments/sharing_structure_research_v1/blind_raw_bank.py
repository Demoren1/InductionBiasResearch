"""Preserve every source proposal and its canonical numerical warm state."""

import argparse
import itertools
import json
import math
from pathlib import Path

from .blind_contract import save_json
from .blind_runtime import ObservationBoundary


def collect(root, case, condition):
    import numpy as np

    from .blind_contract import digest, load_sizes

    sizes = load_sizes(root, case)
    dimensions, hidden, codes, outputs = [
        sizes[k] for k in ("inputs", "hidden", "codes", "outputs")
    ]
    paths = sorted(
        (root / "source").glob(f"{case}_c{condition}_t*/learned.npz"),
        key=lambda p: int(p.parent.name.rsplit("t", 1)[1]),
    )
    bank = {}
    for task, path in enumerate(paths):
        source = dict(np.load(path))
        trace = json.loads(path.with_name("trace.json").read_text())
        parameter = source["dense_parameters"]
        weight = parameter[: dimensions * hidden].reshape(dimensions, hidden)
        norm = np.maximum(np.linalg.norm(weight, axis=0), 1e-12)
        head = (
            parameter[dimensions * hidden + hidden : -outputs].reshape(hidden, outputs)
            * norm[:, None]
        )
        quality = 1 / (
            1
            + trace["dense_validation_mse"]
            / max(trace["noise"]["signal_variance_estimate"], 1e-12)
        )
        for proposal in trace["candidates"]:
            number = proposal["number"]
            index = source[f"index_{number}"]
            best = None
            for permutation in itertools.permutations(range(1, codes + 1)):
                for signs in itertools.product((-1, 1), repeat=codes):
                    labels = np.r_[0, np.asarray(permutation) * signs]
                    renamed = labels[np.abs(index)] * np.sign(index)
                    columns = np.lexsort(renamed[::-1])
                    canonical = renamed[:, columns].astype(np.int16)
                    key = canonical.tobytes()
                    if best is None or key < best[0]:
                        best = key, canonical, labels, columns
            key, canonical, labels, columns = best
            value = bank.setdefault(
                key,
                {
                    "index": canonical,
                    "origins": [],
                    "warm": {},
                    "quality": {},
                    "task_weights": {},
                },
            )
            value["origins"].append(
                {
                    "task": task,
                    "proposal": number,
                    "projection_mse": proposal["projection_mse"],
                }
            )
            value["task_weights"][task] = quality
            if proposal["projection_mse"] >= value["quality"].get(task, math.inf):
                continue
            c = np.zeros(codes)
            for old in range(1, codes + 1):
                c[abs(labels[old]) - 1] = (
                    np.sign(labels[old]) * source[f"codes_{number}"][old - 1]
                )
            scale = np.maximum(source[f"scales_{number}"], 1e-12)
            b = (source["bias"] * scale)[columns]
            a = (head / scale[:, None])[columns]
            warm = np.r_[c, b, a.ravel(), parameter[-outputs:]]
            if np.isfinite(warm).all():
                value["quality"][task] = proposal["projection_mse"]
                value["warm"][task] = warm
        print(
            json.dumps(
                {
                    "case": case,
                    "phase": "preserve_raw",
                    "task": task,
                    "unique": len(bank),
                }
            ),
            flush=True,
        )
    keys = sorted(
        bank,
        key=lambda k: (
            -len(bank[k]["task_weights"]),
            -sum(bank[k]["task_weights"].values()),
            np.mean([p["projection_mse"] for p in bank[k]["origins"]]),
            k,
        ),
    )
    count = codes + hidden + hidden * outputs + outputs
    warm = np.full((len(keys), len(paths), count), np.nan)
    available = np.zeros((len(keys), len(paths)), dtype=bool)
    for row, key in enumerate(keys):
        for task, state in bank[key]["warm"].items():
            warm[row, task] = state
            available[row, task] = True
    folder = root / "fits" / f"{case}_c{condition}"
    np.savez_compressed(
        folder / "raw_preserved_bank.npz",
        indices=np.stack([bank[k]["index"] for k in keys]),
        source_votes=np.array([len(bank[k]["task_weights"]) for k in keys]),
        source_weight=np.array([sum(bank[k]["task_weights"].values()) for k in keys]),
        warm_parameters=warm,
        warm_available=available,
    )
    save_json(
        folder / "raw_preserved_bank.json",
        {
            "teacher_access": False,
            "sources": len(paths),
            "proposals_preserved": sum(len(v["origins"]) for v in bank.values()),
            "unique": len(keys),
            "ranking": "distinct source count, observed source quality, own projection error",
            "origins": [bank[k]["origins"] for k in keys],
            "sha256": digest(folder / "raw_preserved_bank.npz"),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    boundary = ObservationBoundary(root, args.case, args.condition)
    boundary.phase = "preserve_raw_proposals"
    boundary.install()
    status = "failed"
    try:
        collect(root, args.case, args.condition)
        status = "complete"
    finally:
        boundary.save(
            root / "receipts" / f"raw_preserved_{args.case}_c{args.condition}.json",
            status,
        )


if __name__ == "__main__":
    main()
