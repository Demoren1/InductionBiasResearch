"""Infer common signed groups from several learned, scale-ambiguous models."""

import json
import math
import time
from datetime import UTC, datetime

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .blind_contract import canonical_index, digest, load_sizes, save_json
from .blind_numeric import fit_batch, numpy_predict
from .blind_orientation_refine import graph_components, initial_states


def align_source(index, code, normal, uncertainty):
    weight = np.sign(index) * np.r_[0, code][np.abs(index)]
    unit = weight / np.maximum(np.linalg.norm(weight, axis=0), 1e-12)
    positive = ((unit[:, :, None] - normal[:, None]) ** 2).sum(0)
    negative = ((unit[:, :, None] + normal[:, None]) ** 2).sum(0)
    observed = np.abs(normal) > np.maximum(3 * uncertainty[None], 0.002)
    support = ((index[:, :, None] != 0) != observed[:, None]).sum(0)
    left, right = linear_sum_assignment(support + 0.25 * np.minimum(positive, negative))
    return right[np.argsort(left)]


def scale_projection(index, normal, uncertainty, codes):
    tasks, dimensions, hidden = normal.shape
    parent = np.arange(codes + hidden)

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for d, h in zip(*np.where(index)):
        parent[root(abs(index[d, h]) - 1)] = root(codes + h)
    components = {}
    for h in range(hidden):
        components.setdefault(root(codes + h), []).append(h)
    guesses, scales = [], []
    for task in range(tasks):
        anchors = {
            min(group, key=lambda h: uncertainty[task, h])
            for group in components.values()
        }
        free = [h for h in range(hidden) if h not in anchors]
        locations = {h: codes + j for j, h in enumerate(free)}
        matrix = np.zeros((dimensions * hidden, codes + len(free)))
        target = np.zeros(dimensions * hidden)
        for row, (d, h) in enumerate(np.ndindex(index.shape)):
            if index[d, h]:
                matrix[row, abs(index[d, h]) - 1] = np.sign(index[d, h])
            if h in anchors:
                target[row] = normal[task, d, h]
            else:
                matrix[row, locations[h]] = -normal[task, d, h]
        column_scale = np.ones(hidden)
        for _ in range(3):
            deviation = np.maximum(uncertainty[task], 0.0003)
            deviation *= np.maximum(np.abs(column_scale), 0.01)
            weight = np.tile(1 / deviation, dimensions)
            solution = np.linalg.lstsq(
                matrix * weight[:, None], target * weight, rcond=1e-10
            )[0]
            column_scale[free] = solution[codes:]
        guesses.append(solution[:codes])
        scales.append(column_scale)
    guesses, scales = np.array(guesses), np.array(scales)
    return guesses, scales


def projection_score(index, code, scales, normal, uncertainty):
    weight = np.sign(index)[None] * np.c_[np.zeros(len(code)), code][:, np.abs(index)]
    safe = np.where(scales >= 0, 1, -1) * np.maximum(np.abs(scales), 1e-8)
    error = weight / safe[:, None] - normal
    deviation = np.maximum(uncertainty[:, None], 0.0003)
    return float(np.mean(4 * np.log1p((error / deviation) ** 2 / 4)))


def joint_assign(initial, mask, normal, uncertainty, codes, steps=40):
    index = initial.copy()
    index[~mask] = 0
    positions = np.flatnonzero(mask)
    if len(positions) < codes:
        return None
    missing = index.ravel()[positions] == 0
    index.ravel()[positions[missing]] = np.resize(
        np.arange(1, codes + 1), missing.sum()
    )
    for _ in range(steps):
        code, scales = scale_projection(index, normal, uncertainty, codes)
        symbols = np.r_[np.arange(1, codes + 1), -np.arange(1, codes + 1)]
        centers = np.c_[code, -code]
        target = scales[:, None] * normal
        deviation = np.maximum(uncertainty, 0.0003) * np.maximum(np.abs(scales), 0.01)
        standardized = (
            (target[..., None] - centers[:, None, None]) / deviation[:, None, :, None]
        ) ** 2
        costs = np.mean(4 * np.log1p(standardized / 4), axis=0)
        updated = np.zeros(index.shape, dtype=np.int16)
        updated[mask] = symbols[costs[mask].argmin(1)]
        for label in range(1, codes + 1):
            if np.any(np.abs(updated) == label):
                continue
            counts = np.bincount(np.abs(updated).ravel(), minlength=codes + 1)
            allowed = mask & (counts[np.abs(updated)] > 1)
            if not allowed.any():
                return None
            local = np.minimum(costs[..., label - 1], costs[..., label - 1 + codes])
            selected = int(np.where(allowed, local, np.inf).argmin())
            sign = (
                1
                if costs.reshape(-1, 2 * codes)[selected, label - 1]
                <= costs.reshape(-1, 2 * codes)[selected, label - 1 + codes]
                else -1
            )
            updated.ravel()[selected] = sign * label
        if np.array_equal(updated, index):
            break
        index = updated
    code, scales = scale_projection(index, normal, uncertainty, codes)
    score = projection_score(index, code, scales, normal, uncertainty)
    return index, code, scales, score


def initialize_bank(root, case, condition, device, source_seeds=4):
    sizes = load_sizes(root, case)
    folder = root / "fits" / f"{case}_c{condition}"
    if (folder / "fitted.npz").exists():
        return
    folder.mkdir(exist_ok=True)
    source_paths = sorted(
        (root / "source").glob(f"{case}_c{condition}_t*/learned.npz"),
        key=lambda p: int(p.parent.name.rsplit("t", 1)[1]),
    )
    source = [dict(np.load(p)) for p in source_paths]
    traces = [json.loads(p.with_name("trace.json").read_text()) for p in source_paths]
    quality = [
        t["dense_validation_mse"] / max(t["noise"]["signal_variance_estimate"], 1e-12)
        for t in traces
    ]
    best = np.argsort(quality)[:source_seeds]
    candidates = {}
    for t in best:
        for name in [k for k in source[t] if k.startswith("index_")][:2]:
            index = canonical_index(source[t][name], sizes["codes"])
            candidates[index.tobytes()] = index
    if not candidates:
        raise ValueError("No source-derived complete structure seed")
    train = dict(np.load(root / "public" / f"{case}_c{condition}" / "fixed_train.npz"))
    selection = dict(
        np.load(root / "public" / f"{case}_c{condition}" / "selection.npz")
    )
    states, scores = [], []
    for index in candidates.values():
        task_states, task_scores = [], []
        for t, observed in enumerate(source):
            p = initial_states(
                index,
                observed,
                sizes["codes"],
                train[f"x_{t}"],
                train[f"y_{t}"],
                selection[f"x_{t}"],
                selection[f"y_{t}"],
                device,
            )
            losses = [
                float(
                    np.mean(
                        (
                            numpy_predict(
                                selection[f"x_{t}"],
                                index,
                                row,
                                sizes["codes"],
                                sizes["outputs"],
                            )
                            - selection[f"y_{t}"]
                        )
                        ** 2
                    )
                )
                for row in p
            ]
            chosen = int(np.argmin(losses))
            task_states.append(p[chosen])
            task_scores.append(losses[chosen])
        states.append(task_states)
        scores.append(task_scores)
        print(
            json.dumps(
                {
                    "case": case,
                    "phase": "source_seed",
                    "finished": len(states),
                    "total": len(candidates),
                }
            ),
            flush=True,
        )
    np.savez_compressed(
        folder / "fitted.npz",
        indices=np.stack(list(candidates.values())),
        parameters=np.array(states),
        initial_parameters=np.array(states),
        validation_mse=np.array(scores),
    )
    save_json(
        folder / "seed_initialization.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "teacher_access": False,
            "source_ranking": np.argsort(quality).tolist(),
            "source_seeds": best.tolist(),
            "sources": len(source),
            "sha256": digest(folder / "fitted.npz"),
        },
    )


def propose(root, case, condition, anchors=4, starts=16):
    sizes = load_sizes(root, case)
    codes = sizes["codes"]
    folder = root / "fits" / f"{case}_c{condition}"
    base = dict(np.load(folder / "fitted.npz"))
    tasks = base["parameters"].shape[1]
    source = [
        dict(np.load(root / "source" / f"{case}_c{condition}_t{t}" / "learned.npz"))
        for t in range(tasks)
    ]
    ranked = base["validation_mse"].mean(1).argsort()[:anchors]
    rng = np.random.default_rng(33670019)
    bank, origin = {}, {}
    started = time.monotonic()
    for number in ranked:
        anchor = base["indices"][number]
        mappings = [
            align_source(
                anchor,
                base["parameters"][number, t, :codes],
                source[t]["normal"],
                source[t]["uncertainty"],
            )
            for t in range(tasks)
        ]
        normal = np.stack([s["normal"][:, m] for s, m in zip(source, mappings)])
        uncertainty = np.stack([s["uncertainty"][m] for s, m in zip(source, mappings)])
        evidence = np.abs(normal) > np.maximum(3 * uncertainty[:, None], 0.002)
        for votes in (0, 1, 2, 3):
            mask = anchor != 0 if votes == 0 else evidence.sum(0) >= min(votes, tasks)
            seeds = [anchor]
            for s, m in zip(source, mappings):
                for key in [key for key in s if key.startswith("index_")][:2]:
                    seeds.append(s[key][:, m])
            while len(seeds) < starts:
                label = rng.integers(1, codes + 1, size=mask.shape)
                sign = np.where(normal[len(seeds) % tasks] >= 0, 1, -1)
                seeds.append((label * sign).astype(np.int16))
            for seed in seeds:
                result = joint_assign(seed, mask, normal, uncertainty, codes)
                if result is None:
                    continue
                index, code, scales, score = result
                if not np.isfinite(score):
                    continue
                for reference in range(tasks):
                    oriented = index * np.where(scales[reference] >= 0, 1, -1)[None]
                    canonical = canonical_index(oriented.astype(np.int16), codes)
                    key = canonical.tobytes()
                    if key in bank and bank[key][0] <= score:
                        continue
                    bank[key] = (score, canonical, np.abs(code))
                    origin[key] = {
                        "anchor": int(number),
                        "votes": votes,
                        "reference": reference,
                    }
        print(
            json.dumps(
                {
                    "case": case,
                    "phase": "joint_candidates",
                    "anchor": int(number),
                    "unique": len(bank),
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    if not bank:
        raise ValueError("No complete joint candidates")
    keys = sorted(bank, key=lambda key: bank[key][0])
    np.savez_compressed(
        folder / "joint_bank.npz",
        indices=np.stack([bank[k][1] for k in keys]),
        score=np.array([bank[k][0] for k in keys]),
        code_guesses=np.stack([bank[k][2] for k in keys]),
    )
    save_json(
        folder / "joint_bank.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "seconds": time.monotonic() - started,
            "anchors": ranked.tolist(),
            "starts": starts,
            "projection_loss": "Student4 on own normalized-direction errors",
            "origins": [origin[k] for k in keys],
            "teacher_access": False,
            "sha256": digest(folder / "joint_bank.npz"),
        },
    )


def fit(root, case, condition, device, limit=32, steps=48):
    torch.set_num_threads(2)
    sizes = load_sizes(root, case)
    codes = sizes["codes"]
    folder = root / "fits" / f"{case}_c{condition}"
    base, joint = [
        dict(np.load(folder / name)) for name in ("fitted.npz", "joint_bank.npz")
    ]
    tasks, keep = base["parameters"].shape[1], 4
    source = [
        dict(np.load(root / "source" / f"{case}_c{condition}_t{t}" / "learned.npz"))
        for t in range(tasks)
    ]
    train = dict(np.load(root / "public" / f"{case}_c{condition}" / "fixed_train.npz"))
    selection = dict(
        np.load(root / "public" / f"{case}_c{condition}" / "selection.npz")
    )
    x, y = [
        np.stack([train[f"{name}_{t}"] for t in range(tasks)]) for name in ("x", "y")
    ]
    vx, vy = [
        np.stack([selection[f"{name}_{t}"] for t in range(tasks)])
        for name in ("x", "y")
    ]
    all_indices = list(base["indices"])
    parameters, risks = list(base["parameters"]), list(base["validation_mse"])
    scores = [math.inf] * len(all_indices)
    lookup = {a.tobytes(): i for i, a in enumerate(all_indices)}
    selected, initial = [], []
    for number in range(min(limit, len(joint["indices"]))):
        index = joint["indices"][number]
        key = index.tobytes()
        if key not in lookup:
            lookup[key] = len(all_indices)
            all_indices.append(index)
            parameters.append(np.zeros_like(base["parameters"][0]))
            risks.append(np.full(tasks, np.inf))
            scores.append(math.inf)
        target = lookup[key]
        scores[target] = float(joint["score"][number])
        selected.append(target)
        states = []
        for t in range(tasks):
            observed = {"codes_joint": joint["code_guesses"][number, t], **source[t]}
            states.append(
                initial_states(
                    index, observed, codes, x[t], y[t], vx[t], vy[t], device, keep
                )
            )
        initial.append(states)
    parameters, risks, all_indices = (
        np.array(parameters),
        np.array(risks),
        np.array(all_indices),
    )
    history = []
    for begin in range(0, len(selected), 4):
        ids = selected[begin : begin + 4]
        count = len(ids)
        packed = [
            np.tile(np.repeat(a, keep, axis=0), (count, 1, 1)) for a in (x, y, vx, vy)
        ]
        fitted, trace = fit_batch(
            np.repeat(all_indices[ids], tasks * keep, axis=0),
            np.array(initial[begin : begin + count]).reshape(count * tasks * keep, -1),
            *packed,
            codes,
            device,
            steps=steps,
        )
        fitted = fitted.reshape(count, tasks, keep, -1)
        loss = trace["validation_mse"].reshape(count, tasks, keep)
        chosen = loss.argmin(2)
        values = loss[np.arange(count)[:, None], np.arange(tasks)[None], chosen]
        numerical = fitted[np.arange(count)[:, None], np.arange(tasks)[None], chosen]
        for i, candidate in enumerate(ids):
            improve = values[i] < risks[candidate]
            parameters[candidate, improve] = numerical[i, improve]
            risks[candidate, improve] = values[i, improve]
        history.append(
            {
                "candidates": ids,
                "train": trace["train_history"].tolist(),
                "validation": trace["validation_history"].tolist(),
            }
        )
        print(
            json.dumps(
                {
                    "case": case,
                    "phase": "joint_numeric",
                    "finished": begin + count,
                    "total": len(selected),
                    "seconds": trace["seconds"],
                }
            ),
            flush=True,
        )
    np.savez_compressed(
        folder / "joint_fitted.npz",
        indices=all_indices,
        parameters=parameters,
        validation_mse=risks,
        joint_geometry_score=np.array(scores),
    )
    save_json(
        folder / "joint_fitted.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "teacher_access": False,
            "new_numeric_candidates": selected,
            "steps": steps,
            "history": history,
            "sha256": digest(folder / "joint_fitted.npz"),
        },
    )


def information_criterion(root, case, condition):
    sizes = load_sizes(root, case)
    codes, outputs = sizes["codes"], sizes["outputs"]
    folder = root / "fits" / f"{case}_c{condition}"
    fitted = dict(np.load(folder / "joint_fitted.npz"))
    train = dict(np.load(root / "public" / f"{case}_c{condition}" / "fixed_train.npz"))
    tasks = fitted["parameters"].shape[1]
    bic, dimensions, mse = [], [], []
    for index, parameters in zip(fitted["indices"], fitted["parameters"]):
        active = int(np.any(index != 0, axis=0).sum())
        used = len(np.unique(np.abs(index[index != 0])))
        symmetry = len(graph_components(index, codes))
        effective = used + active * (1 + outputs) + outputs - symmetry
        score, errors = 0.0, []
        for t in range(tasks):
            x, y = train[f"x_{t}"], train[f"y_{t}"]
            value = numpy_predict(x, index, parameters[t], codes, outputs)
            loss = float(np.mean((value - y) ** 2))
            floor = np.finfo(float).eps ** 2 * max(float(np.var(y)), 1)
            score += y.size * math.log(max(loss, floor)) + effective * math.log(y.size)
            errors.append(loss)
        bic.append(score)
        dimensions.append(effective)
        mse.append(errors)
    chosen = int(np.argmin(bic))
    save_json(
        folder / "joint_bic.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "candidate": chosen,
            "effective_dimension_per_task": dimensions,
            "all_bic": bic,
            "all_train_mse": mse,
            "noise_estimate": "profile Gaussian variance from each task's own residuals",
            "teacher_access": False,
        },
    )
    print(
        json.dumps(
            {
                "case": case,
                "bic_candidate": chosen,
                "effective_parameters": dimensions[chosen],
                "train_mse": mse[chosen],
            }
        ),
        flush=True,
    )


def select(root, case, condition):
    sizes = load_sizes(root, case)
    folder = root / "fits" / f"{case}_c{condition}"
    fitted = dict(np.load(folder / "joint_fitted.npz"))
    meta = dict(np.load(root / "public" / f"{case}_c{condition}" / "meta.npz"))
    tasks = fitted["parameters"].shape[1]
    errors, predictions = [], []
    for index, parameters in zip(fitted["indices"], fitted["parameters"]):
        values = np.stack(
            [
                numpy_predict(
                    meta[f"x_{t}"],
                    index,
                    parameters[t],
                    sizes["codes"],
                    sizes["outputs"],
                )
                for t in range(tasks)
            ]
        )
        target = np.stack([meta[f"y_{t}"] for t in range(tasks)])
        errors.append(((values - target) ** 2).mean(2).ravel())
        predictions.append(values)
    errors = np.array(errors)
    risk = errors.mean(1)
    best = int(risk.argmin())
    difference = errors - errors[best]
    se = difference.std(1, ddof=1) / math.sqrt(difference.shape[1])
    eligible = np.flatnonzero(difference.mean(1) <= 2 * se + 1e-12)
    tie = int(min(eligible, key=lambda i: (fitted["joint_geometry_score"][i], risk[i])))
    arrays, records = {}, {}
    choices = [("joint_functional", best), ("joint_paired_geometry", tie)]
    if (folder / "joint_bic.json").exists():
        bic = json.loads((folder / "joint_bic.json").read_text())
        choices.append(("joint_bic", bic["candidate"]))
    for name, chosen in choices:
        arrays[f"{name}_index"] = fitted["indices"][chosen]
        arrays[f"{name}_parameters"] = fitted["parameters"][chosen]
        arrays[f"{name}_meta"] = predictions[chosen]
        records[name] = {
            "candidate": chosen,
            "meta_mse": float(risk[chosen]),
            "joint_geometry_score": float(fitted["joint_geometry_score"][chosen])
            if np.isfinite(fitted["joint_geometry_score"][chosen])
            else None,
        }
    np.savez_compressed(folder / "joint_selected.npz", **arrays)
    save_json(
        folder / "joint_selected.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "methods": records,
            "eligible_paired": eligible.tolist(),
            "all_meta_mse": risk.tolist(),
            "teacher_access": False,
            "sha256": digest(folder / "joint_selected.npz"),
        },
    )
    print(json.dumps({"case": case, "selected": records}), flush=True)
