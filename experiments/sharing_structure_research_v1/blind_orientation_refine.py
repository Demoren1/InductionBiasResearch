"""Functional orientation search using estimated, unoriented hyperplanes."""

import argparse
import itertools
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .blind_contract import digest, load_sizes, save_json
from .blind_numeric import (
    calibrate_geometry,
    feature,
    fit_batch,
    numpy_predict,
    prediction,
    profile_head,
)
from .blind_runtime import ObservationBoundary


def graph_components(index, codes):
    parent = np.arange(codes + index.shape[1])

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for d, h in zip(*np.where(index != 0)):
        parent[root(abs(index[d, h]) - 1)] = root(codes + h)
    result = {}
    for code in np.unique(np.abs(index[index != 0])):
        result.setdefault(root(int(code) - 1), []).append(int(code) - 1)
    return list(result.values())


def match_planes(index, code, normal):
    weight = np.sign(index) * np.r_[0, code][np.abs(index)]
    norm = np.linalg.norm(weight, axis=0)
    unit = weight / np.maximum(norm, 1e-12)
    positive = np.sum((unit[:, :, None] - normal[:, None, :]) ** 2, axis=0)
    negative = np.sum((unit[:, :, None] + normal[:, None, :]) ** 2, axis=0)
    cost = np.minimum(positive, negative)
    left, right = linear_sum_assignment(cost)
    mapping = np.full(index.shape[1], -1, dtype=int)
    signs = np.ones(index.shape[1])
    mapping[left] = right
    signs[left] = np.where(positive[left, right] <= negative[left, right], 1, -1)
    return float(cost[left, right].sum()), norm, mapping, signs


@torch.no_grad()
def initial_states(index, source, codes, x, y, vx, vy, device, keep=4):
    hidden, outputs = index.shape[1], y.shape[1]
    components = graph_components(index, codes)
    anchors = {min(component) for component in components}
    relative = [k for k in range(codes) if k not in anchors]
    normal = source["normal"]
    choices = []
    available = [name for name in source if name.startswith("codes_")][:4]
    for name in available:
        for permutation in itertools.permutations(range(codes)):
            for pattern in itertools.product((-1, 1), repeat=len(relative)):
                sign = np.ones(codes)
                sign[relative] = pattern
                code = source[name][list(permutation)] * sign
                code /= max(float(np.linalg.norm(code) / math.sqrt(codes)), 1e-12)
                cost, _, _, _ = match_planes(index, code, normal)
                choices.append((cost, code))
    choices.sort(key=lambda row: row[0])
    seeds, seen = [], set()
    for _, code in choices[:32]:
        for _ in range(2):
            cost, norm, mapping, signs = match_planes(index, code, normal)
            aligned = normal.copy()
            aligned[:, mapping] *= signs
            code = calibrate_geometry(
                index, aligned, source["uncertainty"], mapping, code
            )
        cost, norm, mapping, signs = match_planes(index, code, normal)
        bias = source["bias"][mapping] * signs * norm
        for pattern in itertools.product((-1, 1), repeat=len(components)):
            physical = code.copy()
            physical_bias = bias.copy()
            for component, flip in zip(components, pattern):
                physical[component] *= flip
                columns = np.any(
                    np.isin(np.abs(index), np.array(component) + 1), axis=0
                )
                physical_bias[columns] *= flip
            key = np.round(np.r_[physical, physical_bias], 10).tobytes()
            if key in seen:
                continue
            seen.add(key)
            seeds.append(
                np.r_[physical, physical_bias, np.zeros(hidden * outputs + outputs)]
            )
        if len(seeds) >= 16:
            break
    if not seeds:
        raise ValueError("No observed unoriented numerical states")
    parameters = torch.tensor(np.stack(seeds), device=device, dtype=torch.float64)
    tx, target, tvx, validation = [
        torch.tensor(a, device=device, dtype=torch.float64) for a in (x, y, vx, vy)
    ]
    feat = feature(tx[None], index[None], codes, device).expand(len(seeds), -1, -1, -1)
    vf = feature(tvx[None], index[None], codes, device).expand(len(seeds), -1, -1, -1)
    parameters = profile_head(
        feat,
        parameters,
        target[None].expand(len(seeds), -1, -1),
        codes,
        hidden,
        outputs,
    )
    scores = (
        (prediction(vf, parameters, codes, hidden, outputs) - validation[None])
        .square()
        .mean((1, 2))
    )
    chosen = scores.argsort().cpu().tolist()[:keep]
    while len(chosen) < keep:
        chosen.append(chosen[0])
    return parameters[chosen].cpu().numpy()


def fit(root, case, condition, device, limit):
    torch.set_num_threads(2)
    started = time.monotonic()
    sizes = load_sizes(root, case)
    folder = root / "fits" / f"{case}_c{condition}"
    with np.load(folder / "fitted.npz") as archive:
        base = dict(archive)
    tasks = base["parameters"].shape[1]
    source = [
        dict(np.load(root / "source" / f"{case}_c{condition}_t{task}" / "learned.npz"))
        for task in range(tasks)
    ]
    observed = dict(
        np.load(root / "public" / f"{case}_c{condition}" / "fixed_train.npz")
    )
    selection = dict(
        np.load(root / "public" / f"{case}_c{condition}" / "selection.npz")
    )
    x, y = [
        np.stack([observed[f"{name}_{t}"] for t in range(tasks)]) for name in ("x", "y")
    ]
    vx, vy = [
        np.stack([selection[f"{name}_{t}"] for t in range(tasks)])
        for name in ("x", "y")
    ]
    selected = base["validation_mse"].mean(1).argsort()[:limit]
    initial = []
    for number in selected:
        initial.append(
            [
                initial_states(
                    base["indices"][number],
                    source[task],
                    sizes["codes"],
                    x[task],
                    y[task],
                    vx[task],
                    vy[task],
                    device,
                )
                for task in range(tasks)
            ]
        )
    initial = np.array(initial)
    states, keep = base["parameters"].copy(), initial.shape[2]
    scores = base["validation_mse"].copy()
    history = []
    for begin in range(0, len(selected), 4):
        ids = selected[begin : begin + 4]
        count = len(ids)
        indices = np.repeat(base["indices"][ids], tasks * keep, axis=0)
        parameters = initial[begin : begin + count].reshape(count * tasks * keep, -1)
        panels = [
            np.tile(np.repeat(a, keep, axis=0), (count, 1, 1)) for a in (x, y, vx, vy)
        ]
        fitted, trace = fit_batch(
            indices, parameters, *panels, sizes["codes"], device, steps=48
        )
        fitted = fitted.reshape(count, tasks, keep, -1)
        loss = trace["validation_mse"].reshape(count, tasks, keep)
        best = loss.argmin(2)
        values = loss[np.arange(count)[:, None], np.arange(tasks)[None], best]
        numerical = fitted[np.arange(count)[:, None], np.arange(tasks)[None], best]
        for local, candidate in enumerate(ids):
            improved = values[local] < scores[candidate]
            states[candidate, improved] = numerical[local, improved]
            scores[candidate, improved] = values[local, improved]
        history.append({"candidates": ids.tolist(), "seconds": trace["seconds"]})
        print(
            json.dumps(
                {
                    "case": case,
                    "phase": "unoriented_numeric",
                    "candidates": ids.tolist(),
                    "seconds": trace["seconds"],
                }
            ),
            flush=True,
        )
    np.savez_compressed(
        folder / "orientation_refined.npz",
        indices=base["indices"],
        parameters=states,
        validation_mse=scores,
    )
    save_json(
        folder / "orientation_refined.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "limit": limit,
            "selected_candidates": selected.tolist(),
            "seconds": time.monotonic() - started,
            "history": history,
            "teacher_access": False,
            "sha256": digest(folder / "orientation_refined.npz"),
            "base_sha256": digest(folder / "fitted.npz"),
        },
    )


def select(root, case, condition):
    sizes = load_sizes(root, case)
    folder = root / "fits" / f"{case}_c{condition}"
    fitted = dict(np.load(folder / "orientation_refined.npz"))
    meta = dict(np.load(root / "public" / f"{case}_c{condition}" / "meta.npz"))
    tasks = fitted["parameters"].shape[1]
    losses, predictions = [], []
    for index, parameters in zip(fitted["indices"], fitted["parameters"]):
        values = [
            numpy_predict(
                meta[f"x_{t}"], index, parameters[t], sizes["codes"], sizes["outputs"]
            )
            for t in range(tasks)
        ]
        losses.append(
            [
                float(np.mean((value - meta[f"y_{t}"]) ** 2))
                for t, value in enumerate(values)
            ]
        )
        predictions.append(values)
    losses = np.array(losses)
    chosen = int(losses.mean(1).argmin())
    np.savez_compressed(
        folder / "orientation_selected.npz",
        index=fitted["indices"][chosen],
        parameters=fitted["parameters"][chosen],
        meta_predictions=np.stack(predictions[chosen]),
    )
    save_json(
        folder / "orientation_selected.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "candidate": chosen,
            "meta_mse": float(losses[chosen].mean()),
            "task_meta_mse": losses[chosen].tolist(),
            "all_meta_mse": losses.mean(1).tolist(),
            "sha256": digest(folder / "orientation_selected.npz"),
            "teacher_access": False,
        },
    )
    print(
        json.dumps(
            {
                "case": case,
                "candidate": chosen,
                "meta_mse": float(losses[chosen].mean()),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    root = args.root.resolve()
    (root / "receipts").mkdir(exist_ok=True)
    boundary = ObservationBoundary(root, args.case, args.condition)
    boundary.install()
    status = "failed"
    try:
        boundary.phase = "orientation_refinement"
        fit(root, args.case, args.condition, torch.device(args.device), args.limit)
        boundary.role, boundary.phase = "select", "orientation_meta"
        select(root, args.case, args.condition)
        status = "complete"
    finally:
        boundary.save(
            root / "receipts" / f"orientation_{args.case}_c{args.condition}.json",
            status,
        )


if __name__ == "__main__":
    main()
