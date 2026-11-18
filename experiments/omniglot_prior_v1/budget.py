"""Source-only nested group refinements to separate code capacity from search."""

import heapq
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .model import (
    HIDDEN,
    INPUTS,
    align,
    calibration,
    dense_descriptor,
    numerical_fit,
    project,
)
from .run import OUTPUT, ROOT, SOURCE, episodes, stamp

WORK = ROOT / "budget_followup"
DEST = OUTPUT / "budget_followup"
BUDGETS = (256, 512, 1024, 2048, 4096, 8192)
SINGLE_BUDGETS = (64, *BUDGETS)


def scalar_index(weight, codes):
    """The same 20 Lloyd updates, using sorted scalar nearest neighbours."""
    magnitude = np.abs(weight).ravel()
    centers = np.quantile(magnitude, (np.arange(codes) + 0.5) / codes)
    for _ in range(20):
        unique, original = np.unique(centers, return_index=True)
        right = np.searchsorted(unique, magnitude).clip(0, len(unique) - 1)
        left = (right - 1).clip(0)
        left_distance, right_distance = (
            abs(magnitude - unique[left]),
            abs(magnitude - unique[right]),
        )
        choose_left = (left_distance < right_distance) | (
            (left_distance == right_distance) & (original[left] < original[right])
        )
        labels = np.where(choose_left, original[left], original[right])
        counts = np.bincount(labels, minlength=codes)
        sums = np.bincount(labels, weights=magnitude, minlength=codes)
        centers = np.divide(sums, counts, out=centers.copy(), where=counts > 0)
        distance = abs(magnitude - centers[labels])
        for group in np.flatnonzero(counts == 0):
            position = np.where(counts[labels] > 1, distance, -1).argmax()
            counts[labels[position]] -= 1
            labels[position] = group
            counts[group] += 1
            centers[group] = magnitude[position]
    return ((labels.reshape(weight.shape) + 1) * np.where(weight >= 0, 1, -1)).astype(
        np.int16
    )


def refinements(anchor):
    with np.load(SOURCE) as arrays:
        ranked = arrays["numerical_cross_entropy"].min(0).argsort()[:16]
        raw = arrays["parameters"][ranked, : INPUTS * HIDDEN].reshape(
            -1, INPUTS, HIDDEN
        )
    normal = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    return split_groups(anchor, normal, BUDGETS[1:])


def split_groups(anchor, normal, budgets):
    """Refine an existing signed grouping by calibrated median splits."""
    uncertainty = np.full((len(normal), anchor.shape[1]), 0.03)
    aligned, uncertainty = align(anchor, normal, uncertainty)
    _, scales = calibration(
        np.broadcast_to(anchor, aligned.shape), aligned, uncertainty
    )
    calibrated = aligned * scales[:, None] * np.sign(anchor)[None]
    deviation = np.median(uncertainty * scales, axis=1).clip(1e-6)
    features = (calibrated / deviation[:, None, None]).reshape(len(normal), -1).T
    groups = {
        int(group): np.flatnonzero(np.abs(anchor).ravel() == group)
        for group in np.unique(np.abs(anchor))
        if group
    }
    queue = []

    def push(group):
        positions = groups[group]
        if len(positions) > 1:
            score = np.square(features[positions] - features[positions].mean(0)).sum()
            heapq.heappush(queue, (-float(score), group))

    for group in groups:
        push(group)
    result = {int(np.abs(anchor).max()): anchor.copy()}
    next_group = max(groups) + 1
    for budget in budgets:
        while len(groups) < budget:
            _, group = heapq.heappop(queue)
            positions = groups.pop(group)
            axis = np.var(features[positions], axis=0).argmax()
            order = np.argsort(features[positions, axis], kind="stable")
            half = len(positions) // 2
            groups[group], groups[next_group] = (
                positions[order[:half]],
                positions[order[half:]],
            )
            push(group)
            push(next_group)
            next_group += 1
        labels = np.empty(anchor.size, dtype=np.int16)
        for label, group in enumerate(sorted(groups), 1):
            labels[groups[group]] = label
        result[budget] = labels.reshape(anchor.shape) * np.sign(anchor)
    return result


def fit(name, panels, index, cohort):
    folder = WORK / cohort / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "fit.npz"
    if path.exists():
        with np.load(path) as arrays:
            result = dict(arrays)
    else:
        started = time.perf_counter()
        result = numerical_fit(panels, index=index, precompute=False)
        save_json(folder / "trace.json", result.pop("trace"))
        np.savez_compressed(path, **result)
        if index is not None:
            np.savez_compressed(folder / "index.npz", index=index)
        save_json(
            folder / "complete.json",
            stamp(seconds=time.perf_counter() - started, fit_sha256=digest(path)),
        )
    if "query_y" in panels:
        prediction, y = result["prediction"], panels["query_y"]
        result["ce"] = float(
            (logsumexp(prediction, -1) - (prediction * y).sum(-1)).mean()
        )
        result["accuracy"] = float((prediction.argmax(-1) == y.argmax(-1)).mean())
        print(
            json.dumps(
                {
                    "cohort": cohort,
                    "method": name,
                    "ce": result["ce"],
                    "accuracy": result["accuracy"],
                }
            ),
            flush=True,
        )
    else:
        print(
            json.dumps(
                {
                    "cohort": cohort,
                    "method": name,
                    "predictions_saved": len(panels["train_x"]),
                }
            ),
            flush=True,
        )
    return result


def run():
    torch.set_num_threads(2)
    started = time.perf_counter()
    WORK.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    registration = DEST / "registered.json"
    if not registration.exists():
        snapshot = DEST / "executed_source"
        snapshot.mkdir()
        for name in ("budget.py", "model.py", "run.py"):
            shutil.copy2(Path(__file__).with_name(name), snapshot / name)
        save_json(
            registration,
            stamp(
                budgets=BUDGETS,
                single_budgets=SINGLE_BUDGETS,
                full_budget=INPUTS * HIDDEN,
                original_test_previously_scored=True,
                target_seed=3511158405,
                fresh_test_episodes=128,
                source_rule="Nested median splits along maximum-variance calibrated source trajectory coordinate; inherited signs",
                source_information="Same retained 16 source dense tasks; no target observations for source refinements",
                interpretation="Nested capacity ablation of the selected G34 grouping, not fresh full G34 optimization at every m",
                numerical_backend="Direct signed lookup to bound memory; same Adam rules and six fresh starts",
                original_penalty_conclusion_scope="Previous lambda selection applies to m256 only; not retuned here for all new budgets",
            ),
        )
    choices = json.loads((OUTPUT / "choices_frozen.json").read_text())["choices"]
    with np.load(ROOT / "source_bank.npz") as arrays:
        anchor = arrays[choices["common"]["name"]]
    bank_path = WORK / "bank.npz"
    if bank_path.exists():
        with np.load(bank_path) as arrays:
            bank = {int(key[1:]): arrays[key] for key in arrays.files}
    else:
        bank = refinements(anchor)
        np.savez_compressed(
            bank_path, **{f"m{codes}": index for codes, index in bank.items()}
        )
        save_json(
            DEST / "source_frozen.json",
            stamp(
                bank_sha256=digest(bank_path),
                used_codes={
                    str(codes): len(np.unique(np.abs(index)))
                    for codes, index in bank.items()
                },
                fresh_test_generated=False,
            ),
        )
    with np.load(ROOT / "development.npz") as arrays:
        panels = dict(arrays)
    with np.load(ROOT / "development/dense/fit.npz") as arrays:
        dense = dict(arrays)
    normal, _ = dense_descriptor(dense)
    development = []
    for codes, index in bank.items():
        result = fit(f"shared_m{codes}", panels, index, "development")
        development.append(
            {
                "family": "shared",
                "codes": codes,
                "ce": result["ce"],
                "accuracy": result["accuracy"],
            }
        )
    for codes in SINGLE_BUDGETS:
        index = np.stack([scalar_index(weight, codes) for weight in normal])
        result = fit(f"single_m{codes}", panels, index, "development")
        development.append(
            {
                "family": "single",
                "codes": codes,
                "ce": result["ce"],
                "accuracy": result["accuracy"],
            }
        )
    dense_ce = float(
        (
            logsumexp(dense["prediction"], -1)
            - (dense["prediction"] * panels["query_y"]).sum(-1)
        ).mean()
    )
    dense_accuracy = float(
        (dense["prediction"].argmax(-1) == panels["query_y"].argmax(-1)).mean()
    )
    best = {
        family: min(
            (row for row in development if row["family"] == family),
            key=lambda row: (row["ce"], row["codes"]),
        )
        for family in ("shared", "single")
    }
    dense_row = {
        "family": "dense",
        "codes": INPUTS * HIDDEN,
        "ce": dense_ce,
        "accuracy": dense_accuracy,
    }
    frozen_choices = {
        "shared": best["shared"],
        "single": best["single"],
        "including_unshared": {
            family: min((row, dense_row), key=lambda value: value["ce"])
            for family, row in best.items()
        },
        "development": development,
        "dense_development": dense_row,
    }
    save_json(
        DEST / "choices_frozen.json",
        stamp(choices=frozen_choices, fresh_test_generated=False),
    )
    print(
        json.dumps(
            {
                "selected": best,
                "including_unshared": frozen_choices["including_unshared"],
            }
        ),
        flush=True,
    )
    target_path = WORK / "target_inputs.npz"
    if not target_path.exists():
        panels, provenance = episodes("test", 128, 3511158405)
        private = WORK / "private"
        private.mkdir()
        np.savez_compressed(private / "answers.npz", y=panels.pop("query_y"))
        save_json(private / "provenance.json", provenance)
        np.savez_compressed(target_path, **panels)
        save_json(
            DEST / "test_created.json",
            stamp(
                choices_sha256=digest(DEST / "choices_frozen.json"),
                inputs_sha256=digest(target_path),
                answers_sha256=digest(private / "answers.npz"),
                seed=3511158405,
                new_episodes_on_same_test_characters=True,
            ),
        )
    with np.load(target_path) as arrays:
        panels = dict(arrays)
    dense = fit("dense", panels, None, "test")
    normal, uncertainty = dense_descriptor(dense)
    fit("shared", panels, bank[best["shared"]["codes"]], "test")
    index = np.stack(
        [scalar_index(weight, best["single"]["codes"]) for weight in normal]
    )
    fit("single", panels, index, "test")
    aligned, error = align(anchor, normal, uncertainty)
    index = project(anchor, aligned, error, 0, 1)
    fit("old_adapt_m256", panels, index, "test")
    methods = ("dense", "shared", "single", "old_adapt_m256")
    save_json(
        DEST / "predictions_frozen.json",
        stamp(
            choices_sha256=digest(DEST / "choices_frozen.json"),
            predictions={
                name: digest(WORK / "test" / name / "fit.npz") for name in methods
            },
            fresh_evaluation_answers_opened=False,
        ),
    )
    save_json(
        DEST / "fitting_complete.json", stamp(seconds=time.perf_counter() - started)
    )
    score()


def score():
    frozen = json.loads((DEST / "predictions_frozen.json").read_text())
    if digest(DEST / "choices_frozen.json") != frozen["choices_sha256"]:
        raise RuntimeError("Budget choices changed")
    for method, expected in frozen["predictions"].items():
        if digest(WORK / "test" / method / "fit.npz") != expected:
            raise RuntimeError("Changed predictions")
    save_json(DEST / "scoring_started.json", stamp(predictions_verified=True))
    with np.load(WORK / "private/answers.npz") as arrays:
        truth = arrays["y"]
    from .report import interval

    rows, accuracy_values = [], {}
    for method in frozen["predictions"]:
        with np.load(WORK / "test" / method / "fit.npz") as arrays:
            prediction = arrays["prediction"]
            accuracy = (prediction.argmax(-1) == truth.argmax(-1)).mean(1)
            ce = (logsumexp(prediction, -1) - (prediction * truth).sum(-1)).mean(1)
            rows.append(
                {
                    "method": method,
                    "codes": int(arrays["codes"]),
                    "accuracy": interval(accuracy),
                    "ce": interval(ce),
                }
            )
            accuracy_values[method] = accuracy
    choices = json.loads((DEST / "choices_frozen.json").read_text())["choices"]
    paired = {
        name: interval(value - accuracy_values["dense"])
        for name, value in accuracy_values.items()
        if name != "dense"
    }
    save_json(
        DEST / "summary.json",
        {"methods": rows, "paired_against_dense": paired, "choices": choices},
    )
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ibr-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for family, color in (("shared", "#0284c7"), ("single", "#059669")):
        grid = [row for row in choices["development"] if row["family"] == family]
        axes[0].plot(
            [row["codes"] for row in grid],
            [row["ce"] for row in grid],
            "o-",
            label=family,
            color=color,
        )
    axes[0].axhline(
        choices["dense_development"]["ce"],
        color="gray",
        linestyle="--",
        label="Dense: 12544 weights",
    )
    axes[0].set(
        xscale="log",
        xlabel="Code count",
        ylabel="Development evaluation CE",
        title="Budget selection on 48 development tasks",
    )
    axes[0].legend()
    labels = [f"{row['method']}\nm={row['codes']}" for row in rows]
    axes[1].bar(
        range(len(rows)),
        [row["accuracy"]["mean"] * 100 for row in rows],
        yerr=[
            (row["accuracy"]["upper"] - row["accuracy"]["mean"]) * 100 for row in rows
        ],
        color=["#6b7280", "#0284c7", "#059669", "#e8790c"],
        capsize=3,
    )
    axes[1].set_xticks(range(len(rows)), labels)
    axes[1].set(
        ylabel="Accuracy / macro recall (%)",
        title="128 new test episodes; 95% episode CIs",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        figure.savefig(DEST / f"budget_comparison.{suffix}", dpi=180)
    plt.close(figure)
    save_json(DEST / "complete.json", stamp(tasks=128, code_budgets=BUDGETS))
    print(json.dumps({"results": rows, "paired": paired}, indent=2), flush=True)


if __name__ == "__main__":
    run()
