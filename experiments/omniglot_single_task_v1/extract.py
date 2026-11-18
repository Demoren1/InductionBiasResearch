"""Build complete ratio proposals using one source task at a time."""

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.paper_structure_benchmark_v1.methods.real_project import (
    mapping,
)
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)
from experiments.theorem_pair_recovery_v1.recover import score_magnitudes

from .scalar_pairs import ScalarPairs

THRESHOLDS = np.r_[0.0, np.geomspace(1e-12, 1.0, 97)]


def masks_for(normal, uncertainty):
    masks = [("full", np.ones_like(normal, dtype=bool))]
    for multiplier in (1, 2, 3):
        masks.append(
            (
                f"uncertainty_{multiplier}",
                np.abs(normal)
                > np.maximum(multiplier * uncertainty[None], 0.002),
            )
        )
    for fraction in (0.25, 0.5, 0.75):
        order = np.argsort(np.abs(normal), axis=0, kind="stable")
        mask = np.zeros_like(normal, dtype=bool)
        retained = int(np.ceil(fraction * len(normal)))
        np.put_along_axis(mask, order[-retained:], True, axis=0)
        masks.append((f"top_{fraction}", mask))
    seen = set()
    for name, mask in masks:
        key = mask.tobytes()
        if key not in seen:
            seen.add(key)
            yield name, mask


def signed_index(index, normal):
    index = index * np.where(normal >= 0, 1, -1).astype(np.int16)
    flat = index.ravel()
    for group in range(1, int(np.abs(index).max()) + 1):
        positions = np.flatnonzero(np.abs(flat) == group)
        if flat[positions[0]] < 0:
            flat[positions] *= -1
    return index


def group_diagnostics(index, codes):
    rows, columns = np.where(index)
    labels = np.abs(index[rows, columns]) - 1
    sizes = np.bincount(labels, minlength=codes)
    neurons = [len(np.unique(columns[labels == k])) for k in range(codes)]
    return {
        "group_sizes": sizes.tolist(),
        "neurons_per_group": neurons,
        "largest_group_fraction": float(sizes.max() / len(rows)),
        "singleton_groups": int(np.sum(sizes == 1)),
        "cross_neuron_groups": int(np.sum(np.asarray(neurons) > 1)),
    }


def run(root, output):
    boundary = StudyBoundary(root)
    boundary.install()
    folder = root / "source" / "ratio_proposals"
    folder.mkdir(exist_ok=False)
    source = root / "source" / "dense" / "models.npz"
    dense = dict(np.load(source))
    protocol = json.loads((root / "protocol.json").read_text())
    d, h = protocol["inputs"], protocol["hidden"]
    grid = protocol["independent_code_grid"]
    weights = dense["parameters"][:, : d * h].reshape(-1, d, h)
    weights = weights.astype(np.float64)
    normal = weights / np.maximum(
        np.linalg.norm(weights, axis=1, keepdims=True), 1e-12
    )
    anchors = dense["numerical_loss"].min(0).argsort()[:2]
    save_json(
        folder / "registered.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "dense_sha256": digest(source),
            "anchors": anchors.tolist(),
            "code_grid": grid,
            "thresholds": THRESHOLDS.tolist(),
            "bisection_steps": 24,
            "limit_per_budget": 8,
            "tasks_per_proposal": 1,
            "implementation": "Exact sorted scalar forests, no SVD",
            "code_sha256": {
                str(p): digest(p) for p in Path(__file__).parent.glob("*.py")
            },
        },
    )
    started = time.perf_counter()
    banks = {codes: {} for codes in grid}
    scans, descriptors = [], {}
    for anchor in anchors:
        reference = normal[anchor]
        choices = dense["ensemble_parameters"][anchor, :, : d * h]
        aligned = []
        for item in choices.reshape(-1, d, h).astype(np.float64):
            unit = item / np.maximum(np.linalg.norm(item, axis=0), 1e-12)
            aligned.append(unit[:, mapping(reference, unit)])
        uncertainty = np.maximum(
            np.sqrt(
                np.mean((np.stack(aligned) - reference) ** 2, axis=(0, 1))
            ),
            0.0003,
        )
        descriptors[f"a{anchor}_normal"] = reference
        descriptors[f"a{anchor}_uncertainty"] = uncertainty
        for mask_name, mask in masks_for(reference, uncertainty):
            if mask.sum() < min(grid):
                continue
            cloud = ScalarPairs(reference, mask)
            observations, local = {}, {}

            def evaluate(threshold):
                if threshold in observations:
                    return observations[threshold]
                index, count = cloud.partition(threshold)
                observations[threshold] = count
                if count not in grid:
                    return count
                index = signed_index(index, reference)
                key = index.tobytes()
                if key in local:
                    return count
                active_score, _ = score_magnitudes(
                    index, reference[None], uncertainty[None], count
                )
                omitted = (
                    reference[~mask]
                    / np.broadcast_to(uncertainty, reference.shape)[~mask]
                )
                score = float(
                    (
                        mask.sum() * active_score
                        + np.sum(4 * np.log1p(omitted**2 / 4))
                    )
                    / reference.size
                )
                local[key] = {
                    "index": index,
                    "codes": count,
                    "anchor": int(anchor),
                    "mask": mask_name,
                    "threshold": float(threshold),
                    "score": score,
                    "active": int(mask.sum()),
                    **group_diagnostics(index, count),
                }
                return count

            counts = [evaluate(float(t)) for t in THRESHOLDS]
            for low, high, left, right in zip(
                THRESHOLDS[:-1], THRESHOLDS[1:], counts[:-1], counts[1:]
            ):
                for codes in grid:
                    if (left - codes) * (right - codes) > 0 or left == right:
                        continue
                    lower, upper = float(low), float(high)
                    lower_count = left
                    for _ in range(24):
                        middle = (lower + upper) / 2
                        count = evaluate(middle)
                        if (lower_count - codes) * (count - codes) > 0:
                            lower, lower_count = middle, count
                        elif count == codes and lower_count == codes:
                            lower = middle
                        else:
                            upper = middle
            for codes in grid:
                eligible = sorted(
                    (row for row in local.values() if row["codes"] == codes),
                    key=lambda row: row["score"],
                )[:2]
                for row in eligible:
                    key = row["index"].tobytes()
                    if (
                        key not in banks[codes]
                        or row["score"] < banks[codes][key]["score"]
                    ):
                        banks[codes][key] = row
            record = {
                "anchor": int(anchor),
                "mask": mask_name,
                "active": int(mask.sum()),
                "positive_pairs": len(cloud.delta),
                "scan": [
                    {"threshold": t, "groups": count}
                    for t, count in sorted(observations.items())
                ],
                "complete_partitions": len(local),
                "seconds_since_start": time.perf_counter() - started,
            }
            scans.append(record)
            save_json(folder / "progress.json", {"completed_masks": scans})
            print(
                json.dumps({k: record[k] for k in record if k != "scan"}),
                flush=True,
            )
    arrays, metadata = {}, []
    for codes, bank in banks.items():
        ordered = sorted(bank.values(), key=lambda row: row["score"])
        selected, extra, origins = [], [], set()
        for row in ordered:
            origin = row["anchor"], row["mask"]
            if origin in origins:
                extra.append(row)
            else:
                selected.append(row)
                origins.add(origin)
        for row in (selected + extra)[:8]:
            name = f"candidate_{len(metadata):03d}"
            arrays[name] = row["index"]
            metadata.append(
                {
                    "name": name,
                    **{k: v for k, v in row.items() if k != "index"},
                }
            )
    np.savez_compressed(folder / "candidates.npz", **arrays)
    np.savez_compressed(folder / "descriptors.npz", **descriptors)
    save_json(folder / "candidates.json", {"candidates": metadata})
    available = sorted({row["codes"] for row in metadata})
    save_json(
        folder / "metrics.json",
        {
            "status": "complete",
            "wall_seconds": time.perf_counter() - started,
            "available_budgets": available,
            "missing_budgets": sorted(set(grid) - set(available)),
            "candidates": len(metadata),
            "masks": scans,
            "dense_matches_previous_run": digest(source)
            == json.loads((output / "registration.json").read_text())[
                "previous_dense_sha256"
            ],
        },
    )
    boundary.save(folder / "access_receipt.json", "complete")
    destination = output / "extraction"
    shutil.copytree(folder, destination)
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, output / "code_snapshot" / path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
