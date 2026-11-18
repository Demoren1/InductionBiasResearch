"""Single-source recovery with a fixed zero and profiled neuron scales."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)
from experiments.single_task_recovery_v1.audit import index_metrics
from experiments.srr_approx_v1.core import signed_forest


ARTIFACTS = Path("artifacts")
OUTPUT = ARTIFACTS / "srr_theory_pilots_2026-11-19"
LOCAL = ARTIFACTS / "local" / OUTPUT.name
PAPER = ARTIFACTS / "local/paper_asrr_2026-11-17"
ARCHIVE = ARTIFACTS / "local/our_synthetic_dense_recovery_relu_2026-11-15"
ORIGINAL = ARTIFACTS / "single_task_recovery_2026-11-14"
METHODS = ("fixed_zero", "fixed_zero_scales")
CODES = 4


def zero_quantizer(values, codes):
    """Optimal scalar SSE, with a free-length prefix assigned to fixed zero."""
    values = np.sort(values.ravel())
    count = len(values)
    sums = np.r_[0.0, np.cumsum(values)]
    squares = np.r_[0.0, np.cumsum(values**2)]
    previous = squares.copy()
    parents = np.zeros((codes, count + 1), dtype=int)
    for level in range(codes):
        current = np.full(count + 1, np.inf)
        for end in range(level + 1, count + 1):
            starts = np.arange(level, end)
            total = sums[end] - sums[starts]
            sse = squares[end] - squares[starts] - total**2 / (end - starts)
            costs = previous[starts] + np.maximum(sse, 0)
            best = int(costs.argmin())
            parents[level, end] = starts[best]
            current[end] = costs[best]
        previous = current
    centers = np.zeros(codes + 1)
    end = count
    for level in reversed(range(codes)):
        start = parents[level, end]
        centers[level + 1] = values[start:end].mean()
        end = start
    return centers


def fit_scale(values, centers):
    """Minimize over the scale and all assignments for this fixed codebook."""
    if not values.any():
        return 1.0, np.zeros(len(values), dtype=int)
    midpoints = (centers[:-1] + centers[1:]) / 2
    switches = np.unique(values[:, None] / midpoints)
    switches = switches[switches > 0]
    lower = np.r_[0.0, switches]
    upper = np.r_[switches, np.inf]
    probes = np.r_[(lower[:-1] + switches) / 2, 2 * switches[-1]]
    labels = np.searchsorted(midpoints, values[None, :] / probes[:, None])
    levels = centers[labels]
    squared = np.sum(levels**2, axis=1)
    optima = np.divide(levels @ values, squared, out=probes, where=squared > 0)
    scales = np.clip(optima, lower, upper)
    losses = ((values - scales[:, None] * levels) ** 2).sum(axis=1)
    best = losses.argmin()
    return scales[best], labels[best]


def sparse_assignments(weights):
    values = np.abs(weights)
    signs = np.where(weights < 0, -1, 1)
    pooled = zero_quantizer(values, CODES)
    fixed = np.searchsorted((pooled[:-1] + pooled[1:]) / 2, values)
    starts = [pooled, *(zero_quantizer(column, CODES) for column in values.T)]
    best_loss = np.inf
    for start in starts:
        if np.any(np.diff(start) <= 0):
            continue
        centers = start / start[-1]
        previous_loss = np.inf
        for _ in range(40):
            fits = [fit_scale(column, centers) for column in values.T]
            scales = np.array([scale for scale, _ in fits])
            labels = np.column_stack([label for _, label in fits])
            loss = float(np.sum((values - scales * centers[labels]) ** 2))
            counts = np.bincount(labels.ravel(), minlength=CODES + 1)
            if loss < best_loss and np.all(counts[1:]):
                best_loss = loss
                best = labels.copy()
            if np.isfinite(previous_loss) and (
                previous_loss - loss <= 1e-10 * max(previous_loss, 1e-24)
            ):
                break
            previous_loss = loss
            for j in range(1, CODES + 1):
                take = labels == j
                denominator = np.sum(take * scales**2)
                if denominator > 0:
                    centers[j] = np.sum(take * values * scales) / denominator
            centers.sort()
            centers /= centers[-1]
    if not np.isfinite(best_loss):
        raise ValueError("No candidate used the full nonzero code budget")
    return {
        METHODS[0]: (signs * fixed).astype(np.int16),
        METHODS[1]: (signs * best).astype(np.int16),
    }


def sources(polished):
    for number in (4, 5):
        case = f"case_{number:02d}"
        if polished:
            path = ORIGINAL / "source" / f"{case}_c0_t0" / "learned.npz"
            with np.load(path) as data:
                weights = data["normal"]
            ward = signed_forest(weights[None], [CODES])[0][CODES]
            yield f"paper_{case}", weights, ward
            continue
        source = PAPER / f"synthetic_{number}"
        with np.load(source / "source/asrr/profiles.npz") as data:
            weights = data["selected"][0]
        with np.load(source / "fits/structures.npz") as data:
            yield f"paper_{case}", weights, data["index"]
    if polished:
        return
    for number in range(8):
        case = f"case_{number:02d}"
        path = ARCHIVE / "source" / f"{case}_c0_t0" / "learned.npz"
        with np.load(path) as data:
            weights = data["normal"]
        ward = signed_forest(weights[None], [CODES])[0][CODES]
        yield f"archive_{case}", weights, ward


def freeze(output, target_root, *, polished):
    output.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for key, weights, ward in sources(polished):
        indices = {"ward": ward, **sparse_assignments(weights)}
        predictions.update({f"{key}_{k}": v for k, v in indices.items()})
        if not key.startswith("paper_"):
            continue
        case = key.removeprefix("paper_")
        number = int(case.split("_")[-1])
        panel = Path("targets") / f"{case}_c0"
        destination = target_root / panel
        destination.mkdir(parents=True, exist_ok=True)
        names = ("sizes.json", "support.npz", "selection.npz", "query_x.npz")
        for name in names:
            original = PAPER / f"synthetic_{number}" / panel / name
            shutil.copyfile(original, destination / name)
        bases = {}
        for method in indices if polished else METHODS:
            index = indices[method]
            basis = np.eye(CODES + 1)[np.abs(index)][..., 1:]
            basis *= np.sign(index)[..., None]
            bases[method] = basis / np.sqrt(np.square(basis).sum(-1).mean())
        np.savez_compressed(destination / "structures.npz", **bases)
    path = output / "frozen_indices.npz"
    np.savez_compressed(path, **predictions)
    save_json(
        output / "sparse_registration.json",
        {
            "sha256": digest(path),
            "source": "one fitted Dense direction matrix per geometry",
            "paper_source": (
                "archived post-repair, Gauss-Newton-refined Dense"
                if polished
                else "archived aSRR source"
            ),
            "teacher_inputs": "only code budget 4; no mask size or active labels",
            "cases": [4, 5],
            "archive_cases": [] if polished else list(range(8)),
            "codes": CODES,
            "methods": list(METHODS),
            "max_updates": 40,
            "target_rule": "paper regression_target; 256 samples, four functions per case",
        },
    )


def targets(target_root, *, polished):
    methods = ("ward", *METHODS) if polished else METHODS
    jobs = [(number, method) for number in (4, 5) for method in methods]
    environment = dict(os.environ, PYTORCH_NO_CUDA_MEMORY_CACHING="1")

    def fit(job):
        device, (number, method) = job
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.paper_recovery_extension_v1.regression_target",
                "--root",
                str(target_root),
                "--case",
                f"case_{number:02d}",
                "--method",
                method,
                "--device",
                f"cuda:{device}",
            ],
            check=True,
            env=environment,
        )

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        list(pool.map(fit, enumerate(jobs)))


def score(output, target_root, *, polished):
    path = output / "frozen_indices.npz"
    registration = json.loads(
        (output / "sparse_registration.json").read_text()
    )
    if digest(path) != registration["sha256"]:
        raise ValueError("Frozen indices differ from registration")
    rows = {}
    with np.load(path) as frozen:
        cohorts = (
            ("paper", [4, 5]),
            ("archive", registration["archive_cases"]),
        )
        for cohort, cases in cohorts:
            truth_root = ORIGINAL if cohort == "paper" else ARCHIVE
            for number in cases:
                case = f"case_{number:02d}"
                key = f"{cohort}_{case}"
                with np.load(truth_root / "private" / f"{case}.npz") as data:
                    truth = data["index"]
                rows[key] = {
                    method: index_metrics(frozen[f"{key}_{method}"], truth)
                    for method in ("ward", *METHODS)
                }
                if cohort != "paper":
                    continue
                source = PAPER / f"synthetic_{number}"
                panel = Path("targets") / f"{case}_c0"
                with np.load(source / "private/answers.npz") as data:
                    answers = data["y"]
                for method in ("ward", *METHODS):
                    folder = target_root / panel / method
                    if method == "ward" and not polished:
                        folder = source / panel / "asrr"
                    receipt = json.loads((folder / "metrics.json").read_text())
                    prediction = folder / "frozen_predictions.npz"
                    if receipt["predictions_sha256"] != digest(prediction):
                        raise ValueError(
                            f"Changed target prediction: {folder}"
                        )
                    with np.load(prediction) as data:
                        errors = data["prediction_256"] - answers
                    mse = np.square(errors).mean(axis=(1, 2))
                    rows[key][method].update(
                        mse=float(mse.mean()), mse_per_function=mse.tolist()
                    )
    save_json(output / "sparse_results.json", rows)


def plot(output, target_root, *, polished):
    colors = {
        "ward": "#464e60",
        "fixed_zero": "#b35628",
        "fixed_zero_scales": "#16736d",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex="col")
    for number, row in zip((4, 5), axes, strict=True):
        panel = Path("targets") / f"case_{number:02d}_c0"
        for method, color in colors.items():
            folder = target_root / panel / method
            if method == "ward" and not polished:
                folder = PAPER / f"synthetic_{number}" / panel / "asrr"
            with np.load(folder / "curves.npz") as curves:
                tasks = curves["train_history_256"].shape[-1]
                for ax, prefix in zip(row, ("", "gn_"), strict=True):
                    train = curves[f"{prefix}train_history_256"]
                    valid = curves[f"{prefix}validation_history_256"]
                    steps = (
                        np.arange(len(train))
                        if prefix
                        else curves["steps_256"]
                    )
                    train = train.reshape(len(steps), -1, tasks)
                    valid = valid.reshape(train.shape)
                    best_valid = np.full(tasks, np.inf)
                    best_train = np.full(tasks, np.nan)
                    history = []
                    for tr, va in zip(train, valid, strict=True):
                        best = va.argmin(axis=0)
                        columns = np.arange(tasks)
                        improved = va[best, columns] < best_valid
                        best_valid[improved] = va[best, columns][improved]
                        best_train[improved] = tr[best, columns][improved]
                        history.append((best_train.mean(), best_valid.mean()))
                    training, validation = np.array(history).T
                    ax.semilogy(steps, training, color=color)
                    ax.semilogy(steps, validation, "--", color=color)
                    ax.set_xlabel("GN step" if prefix else "Adam step")
            row[0].set_ylabel(f"Case {number:02d}\nMean normalized MSE")
    handles = [
        plt.Line2D([], [], color=color, label=label)
        for color, label in zip(
            colors.values(),
            ("aSRR (Ward)", "Fixed zero", "Fixed zero + neuron scales"),
            strict=True,
        )
    ]
    handles.extend(
        plt.Line2D([], [], color="black", linestyle=style, label=label)
        for style, label in (("-", "Train"), ("--", "Validation"))
    )
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output / "target_loss.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("freeze", "targets", "score", "plot")
    )
    parser.add_argument("--polished-source", action="store_true")
    args = parser.parse_args()
    output, target_root = OUTPUT, LOCAL
    if args.polished_source:
        output, target_root = (
            OUTPUT / "polished_source",
            LOCAL / "polished_source",
        )
    if args.action == "targets":
        targets(target_root, polished=args.polished_source)
    else:
        {"freeze": freeze, "score": score, "plot": plot}[args.action](
            output, target_root, polished=args.polished_source
        )
