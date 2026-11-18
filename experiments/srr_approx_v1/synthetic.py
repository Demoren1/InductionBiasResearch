"""Short controlled development of approximate SRR before real-data tuning."""

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.heterogeneous_index_v1.model import scalar_seed
from experiments.paper_structure_benchmark_v1.methods.scalable_projection import assign
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .core import (
    canonical,
    input_forest,
    local_forest,
    signed_forest,
    synchronize_profiles,
)

DEST = Path("artifacts/srr_approx_synthetic_2026-11-15")
SCENARIOS = ("balanced", "unbalanced", "approximate", "input_groups")
BUDGETS = (8, 16, 24, 32, 64, 96)
NOISE = (0.0, 0.01, 0.04)


def cases():
    return [
        (f"{scenario}_r{rank}_s{rep}_n{level}", scenario, rank, rep, level)
        for scenario, rank, rep, level in itertools.product(
            SCENARIOS, (2, 5), range(4), range(3)
        )
    ]


def prepare(*, destination=DEST, seed_base=3511158700):
    destination.mkdir(parents=True, exist_ok=False)
    for folder in ("public", "private", "predictions"):
        (destination / folder).mkdir()
    save_json(
        destination / "registered.json",
        {
            "protocol_sha256": digest(
                Path(__file__).with_name("SYNTHETIC_PROTOCOL.md")
            ),
            "theory_sha256": digest(
                Path("paper/theory/APPROXIMATE_SRR_THEORY_2026-11-15.md")
            ),
            "core_sha256": digest(Path(__file__).with_name("core.py")),
            "scenarios": SCENARIOS,
            "ranks": [2, 5],
            "independent_structures": 32,
            "paired_conditions": 96,
            "budget_grid": BUDGETS,
            "seed_base": seed_base,
        },
    )
    for name, scenario, rank, rep, level in cases():
        seed = seed_base + SCENARIOS.index(scenario) * 100003 + rank * 1009 + rep * 101
        rng = np.random.default_rng(seed)
        dimensions, hidden, tasks = 24, 4, 8
        index = np.zeros((dimensions, hidden), np.int16)
        if scenario == "input_groups":
            labels = np.repeat(np.arange(6), 4)
            rng.shuffle(labels)
            index = (labels[:, None] * hidden + np.arange(1, hidden + 1)[None]).astype(
                np.int16
            )
        else:
            for neuron in range(hidden):
                if scenario == "unbalanced":
                    labels = np.r_[
                        np.arange(8),
                        rng.choice(8, 16, p=rng.dirichlet(np.full(8, 0.4))),
                    ]
                else:
                    labels = np.tile(np.arange(8), 3)
                rng.shuffle(labels)
                index[:, neuron] = (labels + 1) * rng.choice((-1, 1), dimensions)
        codes, count = int(np.abs(index).max()), index.size
        basis = rng.normal(size=(rank, codes)) / np.sqrt(rank)
        numbers = rng.normal(size=(tasks, rank)) @ basis
        weights = numbers[:, np.abs(index) - 1] * np.sign(index)[None]
        deformation = 0.25 if scenario == "approximate" else 0.0
        weights += rng.normal(0, deformation, weights.shape)
        weights /= np.linalg.norm(weights, axis=1, keepdims=True)
        measurement = rng.normal(size=weights.shape)
        observed = weights + NOISE[level] * measurement
        observed /= np.linalg.norm(observed, axis=1, keepdims=True)
        arrays = {
            "weights": observed,
            "uncertainty": np.full((tasks, hidden), max(NOISE[level], 0.0003)),
        }
        truth = {"index": index, "clean_source": weights}
        for cohort, functions in (("development", 3), ("test", 3)):
            code = (rng.normal(size=(functions, rank)) @ basis) / np.sqrt(count)
            theta = code[:, np.abs(index).ravel() - 1] * np.sign(index).ravel()[None]
            theta += rng.normal(0, deformation / np.sqrt(count), theta.shape)
            truth[f"{cohort}_theta"] = theta
            for part, samples in (
                ("fit", 192),
                ("valid", 96),
                ("query", 256 if cohort == "development" else 512),
            ):
                x = rng.normal(size=(functions, samples, count))
                clean = np.einsum("tnp,tp->tn", x, theta)
                y = clean + rng.normal(0, 0.2, clean.shape)
                arrays[f"{cohort}_{part}_x"] = x
                if cohort == "test" and part == "query":
                    truth["test_query_clean"] = clean
                else:
                    arrays[f"{cohort}_{part}_y"] = y
        np.savez_compressed(destination / "public" / f"{name}.npz", **arrays)
        np.savez_compressed(destination / "private" / f"{name}.npz", **truth)


def numerical(index, data, cohort):
    """Fresh exact ridge in the lookup subspace; H chooses the ridge amount."""
    flat = index.ravel()
    labels = np.abs(flat) - 1
    codes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=codes)
    matrix = np.eye(codes)[labels] * np.sign(flat)[:, None] / np.sqrt(counts)[None]
    x, y, vx, vy, qx = [
        data[f"{cohort}_{part}_{key}"]
        for part, key in (
            ("fit", "x"),
            ("fit", "y"),
            ("valid", "x"),
            ("valid", "y"),
            ("query", "x"),
        )
    ]
    feature, validation, query = [item @ matrix for item in (x, vx, qx)]
    prediction, selected = [], []
    for train, answer, valid, target, test in zip(feature, y, validation, vy, query):
        values, vectors = np.linalg.eigh(train.T @ train)
        transformed = vectors.T @ train.T @ answer
        choices = []
        for penalty in (0.0, 0.1, 1.0, 10.0, 100.0):
            coefficient = vectors @ (transformed / np.maximum(values + penalty, 1e-12))
            choices.append(
                (
                    float(np.mean((valid @ coefficient - target) ** 2)),
                    penalty,
                    test @ coefficient,
                )
            )
        best = min(choices, key=lambda item: item[0])
        selected.append(best[1])
        prediction.append(best[2])
    return np.stack(prediction), selected


def learner(item):
    name, scenario, rank, rep, level = item
    torch.set_num_threads(1)
    started = time.perf_counter()
    data = dict(np.load(DEST / "public" / f"{name}.npz"))
    weights, uncertainty = data["weights"], data["uncertainty"]
    raw, _ = signed_forest(weights, BUDGETS)
    synchronized = synchronize_profiles(raw[24], weights, uncertainty)
    synced, _ = signed_forest(synchronized, BUDGETS)
    local = local_forest(weights, BUDGETS)
    inputs = input_forest(weights, BUDGETS)
    banks = {"asrr": [], "asrr_a1": [], "g34": []}
    for family, bank in (
        ("global", raw),
        ("synchronized", synced),
        ("local", local),
        ("input", inputs),
    ):
        for codes, index in bank.items():
            banks["asrr"].append((f"{family}_{codes}", index))
    banks["asrr_a1"].extend(banks["asrr"])
    rng = np.random.default_rng(3511158750 + rep)
    normal, error = torch.as_tensor(weights), torch.as_tensor(uncertainty)
    for codes in BUDGETS[:-1]:
        mask = np.ones(weights.shape[1:], bool)
        for anchor in (0, 1):
            seed = scalar_seed(weights[anchor].ravel()[:, None], codes).reshape(
                mask.shape
            )
            for restart in (0, 1):
                initial = (
                    seed
                    if restart == 0
                    else rng.integers(1, codes + 1, mask.shape)
                    * np.sign(weights[anchor])
                )
                answer = assign(
                    initial.astype(np.int16), mask, normal, error, codes, steps=12
                )
                if answer is not None:
                    index, _, scales, _, _ = answer
                    index = canonical(index * np.where(scales[0] >= 0, 1, -1)[None])
                    banks["g34"].append((f"a{anchor}_r{restart}_m{codes}", index))
        for family, seed in (("global", raw[codes]), ("synchronized", synced[codes])):
            answer = assign(seed, mask, normal, error, codes, steps=12)
            if answer is not None:
                index, _, scales, _, _ = answer
                banks["asrr_a1"].append(
                    (
                        f"{family}_a1_m{codes}",
                        canonical(index * np.where(scales[0] >= 0, 1, -1)[None]),
                    )
                )
    banks["asrr_a1"].extend((f"g34/{family}", index) for family, index in banks["g34"])
    dense = np.arange(1, weights[0].size + 1, dtype=np.int16).reshape(weights.shape[1:])
    dense_development, _ = numerical(dense, data, "development")
    dense_test, dense_penalty = numerical(dense, data, "test")
    truth = data["development_query_y"]
    arrays = {"dense_prediction": dense_test, "dense_index": dense}
    selected = {"dense": {"codes": dense.size, "penalties": dense_penalty}}
    for method, bank in banks.items():
        unique = {
            canonical(index).tobytes(): (family, canonical(index))
            for family, index in bank
        }
        choices = [
            (float(np.mean((dense_development - truth) ** 2)), "dense_endpoint", dense)
        ]
        for family, index in unique.values():
            prediction, _ = numerical(index, data, "development")
            choices.append((float(np.mean((prediction - truth) ** 2)), family, index))
        score, family, index = min(choices, key=lambda row: row[0])
        prediction, penalty = numerical(index, data, "test")
        arrays[f"{method}_prediction"] = prediction
        arrays[f"{method}_index"] = index
        selected[method] = {
            "family": family,
            "codes": int(np.abs(index).max()),
            "development_mse": score,
            "penalties": penalty,
            "bank_size": len(unique) + 1,
        }
    folder = DEST / "predictions" / name
    folder.mkdir(exist_ok=False)
    np.savez_compressed(folder / "frozen.npz", **arrays)
    save_json(
        folder / "frozen.json",
        {
            "prediction_sha256": digest(folder / "frozen.npz"),
            "selected": selected,
            "scenario": scenario,
            "rank": rank,
            "replicate": rep,
            "noise": NOISE[level],
            "seconds": time.perf_counter() - started,
            "private_truth_opened": False,
        },
    )
    return {"case": name, "seconds": time.perf_counter() - started}


def run():
    with ProcessPoolExecutor(max_workers=8) as executor:
        rows = list(executor.map(learner, cases()))
    save_json(DEST / "complete.json", rows)


def audit():
    rows = []
    for name, scenario, rank, rep, level in cases():
        folder = DEST / "predictions" / name
        record = json.loads((folder / "frozen.json").read_text())
        if digest(folder / "frozen.npz") != record["prediction_sha256"]:
            raise RuntimeError("Changed frozen prediction")
        truth = dict(np.load(DEST / "private" / f"{name}.npz"))
        arrays = dict(np.load(folder / "frozen.npz"))
        true_labels = np.abs(truth["index"]).ravel()
        equal = true_labels[:, None] == true_labels[None]
        np.fill_diagonal(equal, False)
        for method in ("dense", "g34", "asrr", "asrr_a1"):
            index = arrays[f"{method}_index"]
            labels = np.abs(index).ravel()
            proposed = labels[:, None] == labels[None]
            np.fill_diagonal(proposed, False)
            both = int(np.sum(proposed & equal))
            true_sign = np.sign(truth["index"]).ravel()
            predicted_sign = np.sign(index).ravel()
            correct_sign = (true_sign[:, None] * true_sign[None]) == (
                predicted_sign[:, None] * predicted_sign[None]
            )
            signed_both = int(np.sum(proposed & equal & correct_sign))
            count = int(labels.max())
            signs = np.sign(index).ravel()
            counts = np.bincount(labels - 1, minlength=count)
            theta = truth["test_theta"]
            centers = np.stack(
                [
                    np.bincount(labels - 1, row * signs, minlength=count) / counts
                    for row in theta
                ]
            )
            fitted_theta = centers[:, labels - 1] * signs[None]
            bias = float(np.mean(np.sum((fitted_theta - theta) ** 2, axis=1)))
            prediction = arrays[f"{method}_prediction"]
            row = {
                "case": name,
                "scenario": scenario,
                "rank": rank,
                "replicate": rep,
                "noise": NOISE[level],
                "method": method,
                "codes": int(labels.max()),
                "mse": float(np.mean((prediction - truth["test_query_clean"]) ** 2)),
                "pair_precision": both / int(proposed.sum())
                if proposed.any()
                else None,
                "pair_recall": both / int(equal.sum()),
                "signed_pair_precision": signed_both / int(proposed.sum())
                if proposed.any()
                else None,
                "signed_pair_recall": signed_both / int(equal.sum()),
                "target_projection_bias": bias,
                "ols_excess_risk_formula": bias
                + (0.04 + bias) * count / (192 - count - 1),
                **record["selected"][method],
            }
            rows.append(row)
    aggregate = []
    for scenario, level, method in itertools.product(
        SCENARIOS, NOISE, ("dense", "g34", "asrr", "asrr_a1")
    ):
        subset = [
            row
            for row in rows
            if (row["scenario"], row["noise"], row["method"])
            == (scenario, level, method)
        ]
        aggregate.append(
            {
                "scenario": scenario,
                "noise": level,
                "method": method,
                "mean_mse": float(np.mean([row["mse"] for row in subset])),
                "median_codes": float(np.median([row["codes"] for row in subset])),
                "mean_pair_precision": float(
                    np.mean(
                        [
                            row["pair_precision"]
                            for row in subset
                            if row["pair_precision"] is not None
                        ]
                    )
                )
                if any(row["pair_precision"] is not None for row in subset)
                else None,
                "mean_pair_recall": float(
                    np.mean([row["pair_recall"] for row in subset])
                ),
            }
        )
    save_json(DEST / "results.json", {"cases": rows, "aggregate": aggregate})
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "audit"))
    globals()[parser.parse_args().action]()
