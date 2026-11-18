"""Fit source groups with the WSR proximal operator at a fixed code count."""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import isotonic_regression

from experiments.asrr_anchor_init_v1.omniglot import (
    BASELINE,
    CLASSES,
    HIDDEN,
    INPUTS,
    read_json,
)
from experiments.asrr_anchor_init_v1.omniglot import (
    OUTPUT as PREVIOUS_OUTPUT,
)
from experiments.asrr_anchor_init_v1.omniglot import (
    ROOT as PREVIOUS_ROOT,
)
from experiments.paper_recovery_extension_v1.adapter import losses
from experiments.paper_structure_benchmark_v1.fitting.real_dense import (
    predict,
)
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

OUTPUT = Path("artifacts/wsr_omniglot_2026-11-18")
ROOT = OUTPUT.parent / "local" / OUTPUT.name
BANKS = ROOT / "banks/omniglot/models"
PANELS = PREVIOUS_ROOT / "source_panels"
CONNECTIONS, CODES = INPUTS * HIDDEN, 1024
REGISTRATION = OUTPUT / "registration.json"


def prepare():
    previous_path = PREVIOUS_OUTPUT / "registration.json"
    previous = read_json(previous_path)
    reference = PREVIOUS_OUTPUT / "frozen_indexators" / f"{BASELINE}.npz"
    asrr = previous["baseline"]
    if asrr["codes"] != CODES or digest(reference) != asrr["model_sha256"]:
        raise ValueError("Unexpected aSRR reference")
    if len(previous["source_tasks"]) != 16:
        raise ValueError("Expected the previous 16 Omniglot source tasks")
    OUTPUT.mkdir(parents=True)
    BANKS.mkdir(parents=True)
    shutil.copyfile(reference, BANKS / reference.name)
    save_json(
        REGISTRATION,
        {
            "source_tasks": previous["source_tasks"],
            "parent_registration_sha256": digest(previous_path),
            "source_panel_sha256": previous["selected_source_sha256"],
            "initialization": "Own source task's retained Dense fit",
            "source_training": {
                "codes": CODES,
                "steps": 500,
                "checkpoint_every": 25,
                "adam_lr": 0.003,
                "rewinding": 0.98,
                "first_layer_penalty": "Scalar WSR, full support",
                "strength_selection": "Match code count at each update",
            },
            "asrr": asrr,
            "development": {"count": 96, "seed": 7611186200},
            "targets": {"count": 512, "seeds": [7611186300, 7611186301]},
        },
    )


def match_strength(sorted_weight, codes, strength, slope):
    # At the upper bound every adjacent difference is nonpositive.
    upper = np.diff(sorted_weight).max() * (len(sorted_weight) - 1)
    lower = 0.0
    alpha = min(strength, upper)
    for _ in range(64):
        result = isotonic_regression(sorted_weight - alpha * slope)
        sizes = np.diff(result.blocks)
        if len(sizes) == codes:
            return result, sizes, alpha
        if len(sizes) > codes:
            lower = alpha
        else:
            upper = alpha
        alpha = (lower + upper) / 2.0
    return None


def proximal_groups(weight, codes, strength, rewind, break_ties=False):
    """Select the WSR strength by the number of isotonic blocks alone."""
    n = len(weight)
    slope = np.linspace(-1.0, 1.0, n)
    order = np.argsort(weight, kind="stable")
    sorted_weight = weight[order].astype(np.float64)
    match = match_strength(sorted_weight, codes, strength, slope)
    used_tiebreak = match is None
    if used_tiebreak and break_ties:
        # Separate simultaneous merges at the resolution of the input dtype.
        noise = np.random.default_rng(0).uniform(-1.0, 1.0, n)
        unit = np.spacing(np.max(np.abs(weight)))
        for multiplier in (1, 4, 16):
            perturbed = weight.astype(np.float64) + multiplier * float(unit) * noise
            order = np.argsort(perturbed, kind="stable")
            sorted_weight = perturbed[order]
            match = match_strength(sorted_weight, codes, strength, slope)
            if match is not None:
                break
    if match is None:
        raise ValueError("No WSR strength reached the prescribed code count")

    result, sizes, alpha = match
    starts = result.blocks[:-1]
    pooled = np.add.reduceat(sorted_weight, starts) / sizes
    rewound = (rewind * pooled + (1.0 - rewind) * result.x[starts]).astype(weight.dtype)
    separated_levels = 0
    if break_ties and np.any(rewound[1:] <= rewound[:-1]):
        # Distinct blocks can round to the same float32 value after rewinding.
        for j in range(1, len(rewound)):
            if rewound[j] <= rewound[j - 1]:
                rewound[j] = np.nextafter(rewound[j - 1], weight.dtype.type(np.inf))
                separated_levels += 1
    projected = np.empty_like(weight)
    projected[order] = np.repeat(rewound, sizes)
    labels = np.empty(n, dtype=np.int16)
    labels[order] = np.repeat(np.arange(1, codes + 1, dtype=np.int16), sizes)
    return projected, labels, alpha, used_tiebreak, separated_levels


def fit_groups(initial, panels, shape, options, kind):
    (x, y), (vx, vy) = panels
    device = x.device
    tasks = len(initial)
    dimensions, hidden, _ = shape
    connections = dimensions * hidden
    parameter = torch.tensor(initial[None], device=device, requires_grad=True)
    optimizer = torch.optim.Adam([parameter], lr=options["adam_lr"])
    codes = options["codes"]
    steps = options["steps"]
    strengths = [0.001] * tasks
    best_loss = np.full(tasks, np.inf)
    best_index = np.empty((tasks, connections), dtype=np.int16)
    best_weight = np.empty((tasks, connections), dtype=np.float32)
    best_step = np.zeros(tasks, dtype=int)
    best_strength = np.zeros(tasks)
    break_ties = options.get("break_ties", False)
    strength_tiebreaks = np.zeros(tasks, dtype=int)
    separated_levels = np.zeros(tasks, dtype=int)

    for step in range(steps + 1):
        with torch.no_grad():
            weight = parameter[0, :, :connections].cpu().numpy()
            fitted = [
                proximal_groups(row, codes, strength, options["rewinding"], break_ties)
                for row, strength in zip(weight, strengths)
            ]
            weighted, labels, strengths, tie_counts, level_counts = zip(*fitted)
            strength_tiebreaks += tie_counts
            separated_levels += level_counts
            projected = np.stack(weighted)
            parameter[0, :, :connections].copy_(
                torch.as_tensor(projected, device=device)
            )
            if step % options["checkpoint_every"] == 0 or step == steps:
                value = predict(vx, parameter, *shape)
                error = losses(value, vy, kind)[0].cpu().numpy()
                improved = error < best_loss
                best_loss[improved] = error[improved]
                best_index[improved] = np.stack(labels)[improved]
                best_weight[improved] = projected[improved]
                best_step[improved] = step
                best_strength[improved] = np.asarray(strengths)[improved]
                if step % 100 == 0:
                    print(
                        f"Step {step}: median source validation loss "
                        f"{np.median(best_loss):.4f}",
                        flush=True,
                    )
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        losses(predict(x, parameter, *shape), y, kind).sum().backward()
        optimizer.step()

    groups = []
    for i in range(tasks):
        index = best_index[i].reshape(dimensions, hidden)
        _, equal_groups = np.unique(best_weight[i], return_inverse=True)
        if not np.array_equal(equal_groups + 1, index.ravel()):
            raise ValueError(
                f"Source task at position {i} has inconsistent equality groups"
            )
        groups.append(
            (
                index,
                {
                    "codes": codes,
                    "numerical_loss": float(best_loss[i]),
                    "step": int(best_step[i]),
                    "prox_strength": float(best_strength[i]),
                    "largest_group": int(np.bincount(index.ravel()).max()),
                    "strength_tiebreaks": int(strength_tiebreaks[i]),
                    "levels_separated_after_rounding": int(separated_levels[i]),
                },
            )
        )
    return groups


def fit(device):
    config = read_json(REGISTRATION)
    options = config["source_training"]
    if (OUTPUT / "source_fits.json").exists():
        raise FileExistsError(OUTPUT / "source_fits.json")
    torch.set_num_threads(2)
    boundary = StudyBoundary(ROOT)
    boundary.install()
    for name, expected in config["source_panel_sha256"].items():
        if digest(PANELS / f"{name}.npz") != expected:
            raise ValueError(f"Changed source data: {name}")
    with np.load(PANELS / "anchors.npz") as data:
        task_ids = data["task_ids"].tolist()
        initial = data["parameters"]
    if task_ids != config["source_tasks"]:
        raise ValueError("Source task IDs differ from the registration")

    panels = []
    for name in ("support", "numerical"):
        with np.load(PANELS / f"{name}.npz") as panel:
            x, y = (
                torch.as_tensor(panel[k], device=device, dtype=torch.float32)
                for k in ("x", "y")
            )
        panels.append((x / 255, y))
    elapsed = time.perf_counter()
    groups = fit_groups(
        initial, panels, (INPUTS, HIDDEN, CLASSES), options, "cross_entropy"
    )
    records = []
    for task, (index, metrics) in zip(task_ids, groups, strict=True):
        name = f"wsr_source_{task:03d}"
        path = BANKS / f"{name}.npz"
        np.savez_compressed(path, index=index)
        records.append(
            {
                "name": name,
                "source_task": task,
                "numerical_ce": metrics.pop("numerical_loss"),
                **metrics,
                "model_sha256": digest(path),
            }
        )
    boundary.save(OUTPUT / "source_access.json", "complete")
    save_json(
        OUTPUT / "source_fits.json",
        {"candidates": records, "seconds": time.perf_counter() - elapsed},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "fit"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        fit(torch.device(args.device))
