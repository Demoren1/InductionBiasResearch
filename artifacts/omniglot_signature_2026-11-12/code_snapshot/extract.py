"""Freeze source-only weighted spaces and recover all complete partitions."""

import argparse
import concurrent.futures
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.paper_structure_benchmark_v1.methods.real_project import mapping
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import StudyBoundary
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.theorem_pair_recovery_v1.recover import THRESHOLDS, task_precision_weights

from .signatures import recover_rank


def prepare_spaces(root, output):
    boundary = StudyBoundary(root)
    boundary.install()
    output.mkdir(parents=True, exist_ok=False)
    source = root / "source" / "dense" / "models.npz"
    dense = dict(np.load(source))
    protocol = json.loads((root / "protocol.json").read_text())
    d, h = protocol["inputs"], protocol["hidden"]
    weights = dense["parameters"][:, :d * h].reshape(-1, d, h)
    normal = weights / np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-12)
    ensemble = dense["ensemble_parameters"][..., :d * h].reshape(len(normal), -1, d, h)
    uncertainty = []
    for task, choices in enumerate(ensemble):
        aligned = []
        for item in choices:
            unit = item / np.maximum(np.linalg.norm(item, axis=0), 1e-12)
            aligned.append(unit[:, mapping(normal[task], unit)])
        uncertainty.append(np.sqrt(np.mean((np.stack(aligned) - normal[task]) ** 2,
                                           axis=(0, 1))))
    uncertainty = np.maximum(uncertainty, 0.0003)
    anchors = dense["numerical_cross_entropy"].min(0).argsort()[:2]
    spaces = root / "source" / "signature_spaces"
    spaces.mkdir()
    records = []
    for anchor in anchors:
        permutations = [mapping(normal[anchor], item) for item in normal]
        aligned = np.stack([item[:, p] for item, p in zip(normal, permutations)])
        errors = np.stack([item[p] for item, p in zip(uncertainty, permutations)])
        evidence = np.abs(aligned) > np.maximum(3 * errors[:, None], 0.002)
        masks = {
            "full": np.ones((d, h), dtype=bool),
            "anchor": evidence[anchor],
            "votes_1": evidence.sum(0) >= 1,
            "votes_3": evidence.sum(0) >= 3,
        }
        np.savez_compressed(spaces / f"anchor_{anchor}.npz",
                            normal=aligned.astype(np.float64),
                            uncertainty=errors.astype(np.float64))
        seen = set()
        for name, mask in masks.items():
            if mask.tobytes() in seen or mask.sum() < min(protocol["independent_code_grid"]):
                continue
            seen.add(mask.tobytes())
            rows, columns = np.where(mask)
            logs = np.log(np.maximum(np.abs(aligned[:, rows, columns].astype(np.float64)),
                                     1e-300)).T
            for neuron in np.unique(columns):
                logs[columns == neuron] -= logs[columns == neuron].mean(0)
            precision = task_precision_weights(errors)
            logs *= precision[None]
            basis, singular, _ = np.linalg.svd(logs, full_matrices=False)
            key = f"a{anchor}_{name}"
            path = spaces / f"{key}.npz"
            np.savez_compressed(path, basis=basis, mask=mask)
            record = {"key": key, "anchor": int(anchor), "mask": name,
                      "active": int(mask.sum()), "file": str(path),
                      "singular_values": singular.tolist(),
                      "task_weights": precision.tolist()}
            records.append(record)
            print(json.dumps({k: record[k] for k in ("key", "active")}), flush=True)
    save_json(output / "spaces.json", {"spaces": records})
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    paths = [*Path(__file__).parent.glob("*.py"), *Path(__file__).parent.glob("*.md")]
    for path in paths:
        shutil.copy2(path, snapshot / path.name)
    save_json(output / "registration.json", {
        "created_utc": datetime.now(UTC).isoformat(),
        "dense_sha256": digest(source), "protocol_sha256": digest(root / "protocol.json"),
        "code_sha256": {str(path): digest(path) for path in paths},
        "signature_core_sha256": digest(Path("experiments/theorem_pair_recovery_v1/recover.py")),
        "weights": "inverse RMS direction uncertainty, floor .0003, normalize maximum to one",
        "thresholds": THRESHOLDS.tolist(), "code_grid": protocol["independent_code_grid"],
        "ranks": "0,...,63; skip only impossible m-r>number of active neurons",
        "early_stopping": "Rigorous component/isolated-vertex certificates only; no pair sampling approximation",
        "selection": "First geometric candidate per code budget; no source-functional reranking",
        "no_target_or_development_access": True,
    })
    boundary.save(output / "space_access_receipt.json", "complete")
    return records, protocol["independent_code_grid"]


def worker(job):
    path, key, rank, grid, output = job
    started = time.perf_counter()
    with np.load(path) as arrays:
        proposals, diagnostic = recover_rank(arrays["basis"], arrays["mask"], rank, grid)
    folder = Path(output) / "ranks" / key
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"r{rank:02d}"
    np.savez_compressed(destination.with_suffix(".npz"), **{
        f"candidate_{i}": row["index"] for i, row in enumerate(proposals)
    })
    metadata = [{k: v for k, v in row.items() if k != "index"} for row in proposals]
    save_json(destination.with_suffix(".json"), {
        "key": key, "rank": rank, "candidates": metadata, "diagnostic": diagnostic,
        "seconds": time.perf_counter() - started,
    })
    return {"key": key, "rank": rank, "candidates": len(proposals),
            "seconds": time.perf_counter() - started}


def run(root, output, workers):
    records, grid = prepare_spaces(root, output)
    jobs = [(row["file"], row["key"], rank, grid, str(output))
            for row in sorted(records, key=lambda row: row["active"])
            for rank in range(max(grid))]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    save_json(output / "partitions_frozen.json", {
        "created_utc": datetime.now(UTC).isoformat(), "jobs": len(jobs),
        "files": {str(p.relative_to(output)): digest(p)
                  for p in sorted((output / "ranks").rglob("*")) if p.is_file()},
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve(), args.workers)


if __name__ == "__main__":
    main()
