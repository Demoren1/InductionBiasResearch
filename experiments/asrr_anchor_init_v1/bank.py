"""Build the two previously selected aSRR indexators per source cohort."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)
from experiments.srr_approx_v1.core import input_forest, signed_forest

from . import ANCHORS, DENSE, OUTPUT, ROOT

ORIGINS = ("independent", *(f"anchor_{number:02d}" for number in ANCHORS))


def extract(origin):
    folder = OUTPUT / "banks"
    folder.mkdir(exist_ok=True)
    metadata = folder / f"{origin}.json"
    if metadata.exists():
        return
    cold = json.loads((DENSE.parent / "polish.json").read_text())["quality"]
    accuracy = np.array(cold["H_accuracy"])
    independent = origin == "independent"
    if independent:
        archive = DENSE
        record = json.loads((OUTPUT / "registration.json").read_text())
        expected = record["independent_dense_sha256"]
    else:
        source = OUTPUT / "source_fits" / origin
        record = json.loads((source / "complete.json").read_text())
        archive = source / "models.npz"
        expected = record["models_sha256"]
        file = ROOT / "source_fits" / origin / "source/dense/polish.json"
        adapted = json.loads(file.read_text())["quality"]["H_accuracy"]
        accuracy[record["adapted_tasks"]] = adapted
    if digest(archive) != expected:
        raise ValueError(f"Changed Dense source archive: {origin}")
    with np.load(archive) as data:
        weights = data["parameters"][:, : 784 * 16].astype(float)
        loss = data["numerical_cross_entropy"].min(0)
    if weights.shape[0] != len(ANCHORS):
        raise ValueError("Each bank needs exactly ten selected Dense models")
    weights = weights.reshape(-1, 784, 16)
    weights /= np.linalg.norm(weights, axis=1, keepdims=True).clip(1e-12)
    reference = int(loss.argmin())
    if independent:
        base = weights[reference].copy()
        for task, model in enumerate(weights):
            _, order = linear_sum_assignment(-base.T @ model)
            weights[task] = model[:, order]

    global_bank, _ = signed_forest(weights, (8192,))
    indexes = {
        "global_m8192": global_bank[8192],
        "input_m2048": input_forest(weights, (2048,))[2048],
    }
    for name, index in indexes.items():
        codes = int(name.rsplit("m", 1)[1])
        if not np.array_equal(
            np.unique(np.abs(index)), np.arange(1, codes + 1)
        ):
            raise ValueError(f"Invalid {name} partition")
    path = folder / f"{origin}.npz"
    np.savez_compressed(path, **indexes)
    cosine = np.einsum("tdh,sdh->tsh", weights, weights)
    save_json(
        metadata,
        {
            "origin": origin,
            "bank_sha256": digest(path),
            "dense_sha256": expected,
            "models_per_bank": len(weights),
            "alignment_reference": reference if independent else None,
            "neuron_correspondence": "matched" if independent else "inherited",
            "mean_corresponding_source_cosine": float(
                cosine[np.triu_indices(len(weights), k=1)].mean()
            ),
            "numerical_validation_accuracy": accuracy.tolist(),
            "numerical_validation_ce": loss.tolist(),
            "candidates": [
                {"origin": origin, "name": "global_m8192", "codes": 8192},
                {"origin": origin, "name": "input_m2048", "codes": 2048},
            ],
        },
    )
    print(json.dumps({"origin": origin, "models": len(weights)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(extract, ORIGINS))
