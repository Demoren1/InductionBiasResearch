"""Conditional transfer estimates by period and recovery of hardware equalities."""

import json

import numpy as np
from scipy.stats import t as student

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .compatible_study import interval
from .gas_replicates import CACHE, FOLDER, PERIODS, PLAN
from .trajectory_data import ROOT


def stratified_interval(delta, periods, confidence):
    samples = [delta[periods == t] for t in PERIODS["target"]]
    mean = float(np.mean([s.mean() for s in samples]))
    variances = np.array([s.var(ddof=1) / len(s) for s in samples])
    if not variances.any():
        return [mean, mean, mean]
    df = variances.sum() ** 2 / sum(
        v**2 / (len(s) - 1) for v, s in zip(variances, samples)
    )
    half = float(
        student.ppf((1 + confidence) / 2, df) * np.sqrt(variances.sum()) / len(samples)
    )
    return [mean, mean - half, mean + half]


def tie_recovery(index, hardware):
    def pairs(labels):
        counts = np.unique(labels, return_counts=True, axis=0)[1].astype(np.int64)
        return int((counts * (counts - 1)).sum() // 2)

    predicted = pairs(abs(index).ravel())
    reference = pairs(hardware.ravel())
    overlap = pairs(np.column_stack([hardware.ravel(), index.ravel()]))
    return {
        "precision": overlap / predicted,
        "recall": overlap / reference,
        "signed_ties_recovered": overlap,
        "predicted_pairs": predicted,
        "hardware_pairs": reference,
    }


def summarize():
    freeze = json.loads((FOLDER / "frozen.json").read_text())
    names = freeze["aliases"]
    with np.load(FOLDER / "frozen_indices.npz") as f:
        geo = {
            k: tie_recovery(f[names[k]], f[names["hardware"]])
            for k in [
                "asrr",
                "physical_budget",
                *(f"asrr_perm_{j}" for j in range(4)),
                *(f"physical_budget_perm_{j}" for j in range(4)),
            ]
        }
    with np.load(CACHE / "records.npz") as f:
        corr = np.corrcoef(f["x"][f["fit"]], rowvar=False)
    sensor = np.arange(128) // 8
    feature = np.arange(128) % 8
    model = (sensor % 8) // 2
    i, j = np.triu_indices(128, k=1)
    comparable = (feature[i] == feature[j]) & (model[i] == model[j])
    geo["mean_source_correlation_same_type_feature"] = float(
        corr[i[comparable], j[comparable]].mean()
    )
    geo["mean_source_correlation_all_pairs"] = float(corr[i, j].mean())
    outcomes = {}
    for s in (2, 5):
        with np.load(FOLDER / f"target_{s}_scores.npz") as f:
            unit, batch = f["units"], f["batch"]
            batches = np.unique(batch)
            periods = np.array([unit[batch == b][0] for b in batches])
            rows = {}
            for k, name in names.items():
                (step,) = freeze["methods"][name]["steps"]
                values = f[f"{name}__{step}"]
                rows[k] = np.stack([values[batch == b].mean(0) for b in batches])
        for k in ("asrr", "physical_budget"):
            rows[k + "_permuted"] = np.mean(
                [rows[f"{k}_perm_{j}"] for j in range(4)], axis=0
            )

        def contrast(first, second, confidence=0.95):
            return stratified_interval(
                rows[first][:, 0] - rows[second][:, 0], periods, confidence
            )

        keys = [
            "asrr",
            "dense",
            "asrr_permuted",
            "physical_budget",
            "hardware",
            "physical_budget_permuted",
            "narrow",
            "pooled",
        ]
        outcomes[str(s)] = {
            "accuracy_ce_brier": {k: rows[k].mean(0).tolist() for k in keys},
            "by_period": {
                str(p): {k: float(rows[k][periods == p, 0].mean()) for k in keys}
                for p in PERIODS["target"]
            },
            "asrr_minus_dense_ci95": contrast("asrr", "dense"),
            "asrr_minus_dense_three_dataset": contrast("asrr", "dense", 1 - 0.05 / 3),
            "asrr_minus_controls_ci95": {k: contrast("asrr", k) for k in keys},
            "physical_budget_asrr_minus_dense_ci95": contrast(
                "physical_budget", "dense"
            ),
            "same_budget_asrr_minus_hardware_ci95": contrast(
                "physical_budget", "hardware"
            ),
            "same_budget_asrr_minus_permuted_ci95": contrast(
                "physical_budget", "physical_budget_permuted"
            ),
            "episodes": len(unit),
            "numeric_batches": len(batches),
            "periods": list(PERIODS["target"]),
            "scores_sha256": digest(FOLDER / f"target_{s}_scores.npz"),
        }
    save_json(
        FOLDER / "results.json",
        {
            "plan_sha256": digest(PLAN),
            "frozen_sha256": digest(FOLDER / "frozen.json"),
            "hardware_recovery": geo,
            "shots": outcomes,
        },
    )
    summary = {}
    for d in ("pen", "spoken"):
        result = json.loads((ROOT / d / "results.json").read_text())
        two = result["shots"]["2"]
        summary[d] = {
            "two_shot_asrr_minus_dense_ci9833": interval(
                np.asarray(two["asrr_minus_dense_by_unit"]), 1 - 0.05 / 3
            ),
            "five_shot_asrr_minus_dense_ci95": result["shots"]["5"]["asrr_minus_dense"],
            "results_sha256": digest(ROOT / d / "results.json"),
        }
    summary["gas"] = {
        "two_shot_asrr_minus_dense_ci9833": outcomes["2"][
            "asrr_minus_dense_three_dataset"
        ],
        "five_shot_asrr_minus_dense_ci95": outcomes["5"]["asrr_minus_dense_ci95"],
        "results_sha256": digest(FOLDER / "results.json"),
    }
    save_json(ROOT / "comparison.json", summary)
    print(
        "gas",
        [(s, v["asrr_minus_dense_ci95"]) for s, v in outcomes.items()],
        flush=True,
    )


if __name__ == "__main__":
    summarize()
