"""Add WSR to the existing synthetic indexator comparisons."""

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.asrr_anchor_init_v1.report import interval
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.wsr_omniglot_v1.source import proximal_groups, read_json

from .synthetic import NOISE, cases, numerical
from .synthetic_random_controls import (
    LINEAR,
    OUTPUT,
    RELU_PANELS,
    RELU_SCORES,
    audit_index,
    average,
)

DEST = OUTPUT / "wsr"
PREDICTIONS = LINEAR / "wsr_step_zero"
RELU_WSR = Path(
    "artifacts/local/compact_experiments_2026-11-18/synthetic/targets/case_04_c0/wsr"
)


def fit_case(item):
    name, scenario, *_ = item
    codes = 24 if scenario == "input_groups" else 8
    with np.load(LINEAR / "public" / f"{name}.npz") as archive:
        data = dict(archive)
    candidates = []
    started = time.perf_counter()
    for task, weight in enumerate(data["weights"]):
        _, labels, strength, ties, _ = proximal_groups(
            weight.ravel().astype(np.float32), codes, 0.001, 0.98, break_ties=True
        )
        index = labels.reshape(weight.shape)
        prediction, _ = numerical(index, data, "development")
        loss = float(np.mean(np.square(prediction - data["development_query_y"])))
        candidates.append((loss, task, index, float(strength), ties))
    loss, task, index, strength, ties = min(candidates, key=lambda row: row[:2])
    prediction, penalties = numerical(index, data, "test")
    path = PREDICTIONS / f"{name}.npz"
    np.savez_compressed(path, index=index, prediction=prediction)
    return name, {
        "source_anchor": task,
        "codes": codes,
        "strength": strength,
        "tie_break": ties,
        "development_mse": loss,
        "target_penalties": penalties,
        "prediction_sha256": digest(path),
        "seconds": time.perf_counter() - started,
    }


def relu_comparison():
    compact = RELU_SCORES.parent
    archive = read_json(compact / "predictions_frozen.json")
    prediction = RELU_WSR / "frozen_predictions.npz"
    if digest(prediction) != archive["wsr"]:
        raise ValueError("Changed archived WSR predictions")
    wsr = read_json(RELU_WSR / "registered.json")
    asrr = read_json(RELU_PANELS / "asrr/registered.json")
    for key in ("code_sha256", "adapter_sha256", "numeric_sha256", "budgets"):
        if wsr[key] != asrr[key]:
            raise ValueError(f"Changed ReLU numerical protocol: {key}")
    for key in ("support.npz", "selection.npz", "query_x.npz"):
        if wsr["input_sha256"][key] != asrr["input_sha256"][key]:
            raise ValueError(f"Different ReLU observations: {key}")
    source = read_json(compact / "wsr_source.json")
    scores = read_json(RELU_SCORES)
    random = read_json(OUTPUT / "relu/results.json")["means"]
    candidate = source["candidates"][0]
    return {
        "functions": scores["target_functions"],
        "methods": {
            key: scores["methods"][key] for key in ("dense", "asrr", "g34", "wsr")
        },
        "control_mean_mse": random,
        "wsr_source_step": candidate["step"],
        "wsr_source_largest_group": candidate["largest_group"],
        "wsr_predictions_sha256": archive["wsr"],
    }


def run():
    DEST.mkdir()
    PREDICTIONS.mkdir()
    linear_path = OUTPUT / "linear/results.json"
    save_json(
        DEST / "registration.json",
        {
            "profile_results_sha256": digest(linear_path),
            "source_rule": "Single-source fixed-budget WSR proximal step zero",
            "source_selection": "Development MSE after fitting fresh codes",
            "source_dtype": "float32",
            "target_rule": "Unchanged ridge fit with numerical validation",
            "wsr_groups": "Equal signed scalar weights, without separate assignment signs",
        },
    )
    with ProcessPoolExecutor(max_workers=8) as pool:
        frozen = dict(pool.map(fit_case, cases()))
    save_json(DEST / "predictions_frozen.json", frozen)

    original = read_json(linear_path)
    previous = {row["case"]: row for row in original["per_structure"]}
    rows = []
    for name, _, _, _, level in cases():
        path = PREDICTIONS / f"{name}.npz"
        if digest(path) != frozen[name]["prediction_sha256"]:
            raise ValueError(f"Changed frozen WSR prediction: {name}")
        with (
            np.load(path) as data,
            np.load(LINEAR / "private" / f"{name}.npz") as truth,
        ):
            score = audit_index(
                data["index"],
                truth["index"],
                data["prediction"],
                truth["test_query_clean"],
            )
        row = previous[name]
        if row["noise"] != NOISE[level]:
            raise ValueError(f"Changed original noise level: {name}")
        row["methods"]["wsr_step_zero"] = score
        rows.append(row)

    summary = []
    for group in original["summary"]:
        panel = [
            row["methods"]
            for row in rows
            if row["scenario"] == group["scenario"] and row["noise"] == group["noise"]
        ]
        delta = np.array(
            [row["wsr_step_zero"]["mse"] - row["asrr"]["mse"] for row in panel]
        )
        group["methods"]["wsr_step_zero"] = average(
            [row["wsr_step_zero"] for row in panel]
        )
        group["wsr_step_zero_minus_asrr"] = {
            "mse_difference": float(delta.mean()),
            "ci95_halfwidth": interval(delta),
        }
        summary.append(group)
    save_json(
        DEST / "results.json",
        {
            "linear_conditions": len(rows),
            "linear_summary": summary,
            "linear_per_structure": rows,
            "relu": relu_comparison(),
        },
    )
    print({"linear_conditions": len(rows), "results": str(DEST / "results.json")})


if __name__ == "__main__":
    run()
