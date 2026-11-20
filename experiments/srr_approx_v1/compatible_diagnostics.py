"""Evaluate two predictions specified in compatible_diagnostics_plan.json."""

import json
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.compatible import OUTPUT, RelationBank


PLAN = Path(__file__).with_name("compatible_diagnostics_plan.json")


def separation(plan):
    rng = np.random.default_rng(plan["seed"])
    grid = plan["tasks"]
    total_tasks = max(grid)
    train = plan["train"]
    loss_tolerance = plan["tolerance"]
    proposals = (-1, 1)
    bank = RelationBank([{s} for s in proposals])
    results = []
    for p in plan["informative_probability"]:
        counts = {
            t: dict(survives=0, oracle=0, disagreements=0, misses_true=0) for t in grid
        }
        for start in range(0, plan["families"], 64):
            batch = min(64, plan["families"] - start)
            sign = rng.choice(proposals, size=batch)
            informative = rng.random((batch, total_tasks)) < p
            x = rng.normal(size=(batch, total_tasks, train + plan["validation"], 2))
            x[..., 1] *= informative[..., None]
            coefficient = rng.uniform(0.7, 1.3, size=(batch, total_tasks, 1))
            y = coefficient * (x[..., 0] + sign[:, None, None] * x[..., 1])
            y += plan["noise_sd"] * rng.normal(size=y.shape)
            risk = []
            for s in proposals:
                feature = x[..., 0] + s * x[..., 1]
                fit = feature[..., :train]
                slope = (fit * y[..., :train]).sum(axis=-1) / (fit * fit).sum(axis=-1)
                error = slope[..., None] * feature[..., train:] - y[..., train:]
                risk.append(np.minimum(error**2 / 4, 1).mean(axis=-1))
            risk = np.stack(risk, axis=-1)
            for n, truth in enumerate(sign):
                for t in grid:
                    found = bank.infer(risk[n, :t], loss_tolerance)["maximal"]
                    observed = frozenset([-truth]) in found
                    oracle = not informative[n, :t].any()
                    count = counts[t]
                    count["survives"] += int(observed)
                    count["oracle"] += int(oracle)
                    count["disagreements"] += int(observed != oracle)
                    count["misses_true"] += int(frozenset([truth]) not in found)
        results.extend(
            {
                "informative_probability": p,
                "tasks": t,
                "families": plan["families"],
                "theoretical_survival": (1 - p) ** t,
                **counts[t],
            }
            for t in grid
        )
    return results


def misspecification(plan):
    rng = np.random.default_rng(plan["seed"])
    d, k, n = plan["dimension"], plan["constrained_dimension"], plan["train"]
    variance = plan["noise_sd"] ** 2
    rotation = np.eye(d)
    rotation[:2, :2] = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    subspace = rotation[:, [0, *range(2, d)]]
    normal = rotation[:, 1]
    samples = {b: [] for b in plan["distance"]}
    for start in range(0, plan["replicates"], 256):
        batch = min(256, plan["replicates"] - start)
        x = rng.normal(size=(batch, n, d))
        noise = rng.normal(0, plan["noise_sd"], size=(batch, n))
        constrained = x @ subspace
        dense_map = np.linalg.pinv(x)
        constrained_map = subspace @ np.linalg.pinv(constrained)
        for b, values in samples.items():
            w = subspace[:, 0] + b * normal
            y = (x @ w + noise)[..., None]
            dense_error = np.square((dense_map @ y)[..., 0] - w).sum(axis=-1)
            tied_error = np.square((constrained_map @ y)[..., 0] - w).sum(axis=-1)
            values.extend(dense_error - tied_error)
    dense = variance * d / (n - d - 1)
    rows = []
    for b, values in samples.items():
        tied = b**2 + (variance + b**2) * k / (n - k - 1)
        mean = float(np.mean(values))
        half_width = float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))
        rows.append(
            {
                "distance": b,
                "predicted_gain": dense - tied,
                "observed_gain": mean,
                "paired_ci95": [mean - half_width, mean + half_width],
            }
        )
    return rows


if __name__ == "__main__":
    plan = json.loads(PLAN.read_text())
    results = {
        "plan_sha256": digest(PLAN),
        "separation": separation(plan["separation"]),
        "misspecification": misspecification(plan["misspecification"]),
    }
    save_json(OUTPUT / "mechanism_diagnostics.json", results)
