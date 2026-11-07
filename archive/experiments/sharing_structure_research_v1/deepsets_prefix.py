"""G27: recover hidden DeepSets sharing, then reset C on new functions."""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr

from experiments.darts_indexator_v1.data import digest, load_archive, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .joint_prefix_search import census
from .pattern_prefix import axis_inputs, prefix_inputs, real_output, timestamp
from .shared_structure_extract import prefix_candidates
from .shared_structure_numeric import (
    canonical_index,
    code_prediction,
    features,
    fresh_codes,
)
from .shared_structure_numeric_batch import fit_codes

BASE = Path(__file__).resolve().parent
METHODS = ("bootstrap", "single_fixed")


def register(root):
    previous = BASE / "results/20261105/joint_prefix_confirmation_g25/protocol.json"
    old = json.loads(previous.read_text())
    keys = [
        "axis_points",
        "axis_min",
        "axis_max",
        "axis_repeats",
        "slice_repeats",
        "axis_code_min",
        "axis_code_max",
        "axis_starts",
        "axis_solver_evaluations",
        "dp_chunk",
        "slice_points",
        "prefix_validation",
        "slice_scales",
        "rank_relative",
        "qp_gap_absolute",
        "qp_primal_tolerance",
        "active_iterations",
        "active_zero_tolerance",
        "active_direction_tolerance",
        "integer_moves",
        "integer_decrease",
        "exchange_iterations",
        "composition_chunk",
        "code_refit_steps",
        "code_ridge_relative",
        "code_line_search",
        "numeric_chunk",
        "numeric_code_steps",
        "source_polish_steps",
        "numeric_ridge",
        "numeric_line_search",
        "prefix_settings",
        "bank_per_source",
    ]
    plan = {key: old[key] for key in keys}
    plan.update(
        stage="G27 synthetic DeepSets: hidden point/feature roles and fresh-C transfer",
        seed=26110527017,
        registered_utc=timestamp(),
        inputs=16,
        hidden=16,
        codes=4,
        sources=4,
        targets=8,
        families=4,
        bias=0.15,
        noise=[0.0, 0.2, 2.0],
        fit_validation=2048,
        meta_points=2048,
        fit_repeats=64,
        zero_repeats=32 * old["axis_repeats"],
        target_support=4096,
        target_validation=1024,
        target_test=8192,
        target_repeats=64,
        target_zero_repeats=4096 * 64,
        methods=list(METHODS),
        primary="single_fixed",
        dense={
            "rates": [0.003, 0.01],
            "starts_per_rate": 16,
            "steps": 4096,
            "batch": 256,
            "checkpoint_every": 128,
        },
        parent_protocol_sha256=digest(previous),
        model="Eight 2D points, common affine2->ReLU2 block, sum pooling/readout1/sqrt16. Unknown hidden input-coordinate permutation and hidden columns. Known common b=.15/readout; unknown output offset estimated from f(0), not supplied from teacher.",
        source_policy="One original C=[.8,-1.1,1.3,.45] and three fresh Gaussian C per new hidden geometry. Only real output queries. No I/C/point grouping/edge budget/teacher gradients enter extraction. G25 count solver unchanged. Source calibration keeps combined axis+prefix objective.",
        zero_policy="Estimate d from mean f(0)-sqrt(H)b. R0=2D*Raxis balances anchor-noise variance in the even-axis sum. Independent Gaussian means simulated analytically. Subtract the same observed d from source/target panels; restore d for raw predictions.",
        transfer_policy="Freeze all source choices before teacher audit/new target preparation. Reset all target C and optimizers. Thirty-two signed GN starts, validation selection. Identical canonical I shares identical target fits; never reuse numerical parameters across target functions. Freeze entire target cohort before any clean test/permutation audit.",
        dense_policy="Native torch.optim.Adam, full256 independent weights, same known bias/readout and observed output offset. Two registered rates x16 initializations/task, all chosen on noisy validation. Uses the same target support/validation as C adaptation, but does not receive the source extraction data. No clustering/postprocessing of dense weights.",
        limitations="Restricted one-hidden-layer DeepSets with known common bias/readout and active queries, large averaging budget. Four hidden geometries, not an image experiment. Learned dense W is a functional control, not a categorical I estimator.",
        resource_policy="CPU2threads for prefix/QP; GPU0 for packed source fits; independent batched target workers may use free GPUs0-3. Never GPU6/7 or occupied4/5.",
        software_tests=False,
    )
    root.mkdir(parents=True, exist_ok=False)
    (root / "code_snapshot").mkdir()
    names = [
        "deepsets_prefix.py",
        "shared_structure_extract.py",
        "codebook_counts.py",
        "prefix_census.py",
        "count_projection.py",
        "count_qp_active.py",
        "count_exchanges.py",
        "joint_prefix_search.py",
        "shared_structure_numeric.py",
        "shared_structure_numeric_batch.py",
        "pattern_prefix.py",
    ]
    plan["code_sha256"] = {name: digest(BASE / name) for name in names}
    for name in names:
        shutil.copy2(BASE / name, root / "code_snapshot" / name)
    save_json(root / "protocol.json", plan)
    return plan


def centered_offsets(codes, plan):
    local = codes.reshape(-1, 2, 2)
    sigma = np.linalg.norm(local, axis=1)
    z = plan["bias"] / sigma
    means = sigma * np.exp(-(z**2) / 2) / math.sqrt(2 * math.pi) + plan["bias"] * ndtr(
        z
    )
    return -8 * means.sum(1) / math.sqrt(plan["hidden"])


def prepare(root, plan):
    (root / "private").mkdir()
    (root / "data").mkdir()
    for family in range(plan["families"]):
        rng = np.random.default_rng(plan["seed"] + 100003 * family)
        permutation = rng.permutation(16)
        index = np.zeros((16, 16), dtype=np.int16)
        for point in range(8):
            index[
                np.ix_(
                    permutation[2 * point : 2 * point + 2], [2 * point, 2 * point + 1]
                )
            ] = [[1, 2], [3, 4]]
        index = index[:, rng.permutation(16)]
        codes = np.r_[np.array([[0.8, -1.1, 1.3, 0.45]]), rng.normal(size=(3, 4))]
        offsets = centered_offsets(codes, plan)
        np.savez_compressed(
            root / "private" / f"teacher_{family}.npz",
            index=index,
            codes=codes,
            offsets=offsets,
            permutation=permutation,
        )
        order = rng.permutation(16)
        t = np.geomspace(plan["axis_min"], plan["axis_max"], plan["axis_points"])
        vt = np.sqrt(t[:-1] * t[1:])
        axis_x, axis_vx = axis_inputs(t, 16), axis_inputs(vt, 16)
        prefix_x = prefix_inputs(rng, order, plan["slice_points"], plan)
        panels = {
            "train_x": np.r_[axis_x, prefix_x],
            "axis_validation_x": axis_vx,
            "prefix_validation_x": prefix_inputs(
                rng, order, plan["prefix_validation"], plan
            ),
            "repeat_train_x": prefix_inputs(rng, order, plan["slice_points"], plan),
            "repeat_validation_x": prefix_inputs(
                rng, order, plan["prefix_validation"], plan
            ),
            "fit_validation_x": rng.normal(size=(plan["fit_validation"], 16)),
            "final_validation_x": rng.normal(size=(plan["fit_validation"], 16)),
        }
        arrays = {"order": order, "axis_t": t, "axis_vt": vt, **panels}
        zero = (
            real_output(np.zeros((1, 16)), index, codes, plan["bias"])[:, 0] + offsets
        )
        zero_noise = rng.normal(size=4)
        for condition, sigma in enumerate(plan["noise"]):
            arrays[f"zero_y_{condition}"] = zero + zero_noise * sigma / math.sqrt(
                plan["zero_repeats"]
            )
        for name, x in panels.items():
            clean = real_output(x, index, codes, plan["bias"]) + offsets[:, None]
            noise = rng.normal(size=clean.shape)
            if name == "train_x":
                repeats = np.r_[
                    np.full(len(axis_x), plan["axis_repeats"]),
                    np.full(len(prefix_x), plan["slice_repeats"]),
                ]
            elif name == "axis_validation_x":
                repeats = plan["axis_repeats"]
            elif name in (
                "prefix_validation_x",
                "repeat_train_x",
                "repeat_validation_x",
            ):
                repeats = plan["slice_repeats"]
            else:
                repeats = plan["fit_repeats"]
            for condition, sigma in enumerate(plan["noise"]):
                arrays[f"{name[:-2]}_y_{condition}"] = clean + noise * sigma / np.sqrt(
                    repeats
                )
        np.savez_compressed(root / "data" / f"source_{family}.npz", **arrays)
        x = rng.normal(size=(plan["meta_points"], 16))
        clean = real_output(x, index, codes, plan["bias"]) + offsets[:, None]
        noise = rng.normal(size=clean.shape)
        np.savez_compressed(
            root / "data" / f"meta_{family}.npz",
            x=x,
            **{
                f"y_{condition}": clean + noise * sigma / math.sqrt(plan["fit_repeats"])
                for condition, sigma in enumerate(plan["noise"])
            },
        )
    save_json(
        root / "data_frozen.json",
        {
            "frozen_utc": timestamp(),
            "sha256": {
                str(p.relative_to(root)): digest(p) for p in (root / "data").iterdir()
            },
        },
    )


def initial_codes(plan, family, tasks, device, *, target=False):
    return torch.stack(
        [
            fresh_codes(
                plan,
                family + (70000 if target else 1000),
                task + (plan["sources"] if target else 0),
                device,
            )
            for task in range(tasks)
        ]
    )


def normalized_source(root, plan, family, condition):
    raw = load_archive(root / "data" / f"source_{family}.npz")
    offset = raw[f"zero_y_{condition}"] - math.sqrt(plan["hidden"]) * plan["bias"]
    data = {
        key: value
        for key, value in raw.items()
        if key.endswith("_x") or key in ("order", "axis_t", "axis_vt")
    }
    for key, value in raw.items():
        if key.endswith(f"_y_{condition}") and not key.startswith("zero"):
            data[key[: -len(str(condition)) - 1]] = value - offset[:, None]
    return data, offset


def fit_bank(indices, data, validation, plan, family, device):
    return fit_codes(
        np.stack(indices),
        torch.tensor(data["train_x"], device=device),
        torch.tensor(data["train_y"], device=device),
        torch.tensor(data[validation + "_x"], device=device),
        torch.tensor(data[validation + "_y"], device=device),
        initial_codes(plan, family, plan["sources"], device),
        plan,
    )


def source_case(root, plan, family, condition, device):
    started = time.monotonic()
    folder = root / "source" / f"case_{family}_{condition}"
    folder.mkdir(parents=True, exist_ok=False)
    data, offsets = normalized_source(root, plan, family, condition)
    bank, seen, records = [], set(), []
    for task in range(plan["sources"]):
        observed = {
            key: torch.tensor(value[task] if key.endswith("_y") else value)
            for key, value in data.items()
            if key.endswith(("_x", "_y"))
        }
        observed.update({key: data[key] for key in ("order", "axis_t", "axis_vt")})
        for number, result in enumerate(
            prefix_candidates(observed, plan, plan["seed"] + 1009 * family + 71 * task)
        ):
            name = f"bootstrap_{task}_{number}"
            np.savez_compressed(
                folder / f"{name}.npz",
                indices=result["trace_indices"],
                codebooks=result["trace_fitted_C"],
            )
            records.append(
                {
                    "name": name,
                    "complete": result["complete"],
                    "setting": result["setting"],
                    "history": result["history"],
                    "axis_fit": result["axis_fit"],
                    "polish": result["polish"],
                }
            )
            if result["complete"]:
                index = canonical_index(result["index"], 4)
                if index.tobytes() not in seen:
                    seen.add(index.tobytes())
                    bank.append(index)
    save_json(folder / "bootstrap_search.json", records)
    if not bank:
        save_json(
            folder / "selection.json",
            {
                "family": family,
                "condition": condition,
                "abstained": True,
                "methods": {},
            },
        )
        print(
            json.dumps({"source": [family, condition], "abstained": True}), flush=True
        )
        return
    code, histories, _ = fit_bank(bank, data, "fit_validation", plan, family, device)
    scores = np.array(
        [[record["validation_mse"] for record in row] for row in histories]
    ).mean(1)
    provisional = int(scores.argmin())
    bootstrap, calibrated = bank[provisional], code[provisional].cpu().numpy()
    np.savez_compressed(
        folder / "calibration.npz",
        indices=np.stack(bank),
        codes=code.cpu().numpy(),
        offsets=offsets,
    )
    save_json(
        folder / "calibration.json",
        {"frozen_utc": timestamp(), "selected": provisional, "histories": histories},
    )
    repeat_data = {
        "order": data["order"],
        "search_train_x": data["repeat_train_x"],
        "search_train_y": data["repeat_train_y"],
        "prefix_validation_x": data["repeat_validation_x"],
        "prefix_validation_y": data["repeat_validation_y"],
    }
    banks = {"bootstrap": [bootstrap], "single_fixed": []}
    repeated = []
    for task in range(plan["sources"]):
        result = census(repeat_data, calibrated, [task], False, "unit", plan, "cpu")
        np.savez_compressed(
            folder / f"repeat_{task}.npz",
            indices=result["indices"],
            codebooks=result["codebooks"],
        )
        repeated.append(
            {"task": task, "complete": result["complete"], "history": result["trace"]}
        )
        if result["complete"]:
            index = canonical_index(result["indices"][-1], 4)
            if not any(np.array_equal(index, other) for other in banks["single_fixed"]):
                banks["single_fixed"].append(index)
    save_json(folder / "repeat_search.json", repeated)
    indices, lookup, method_ids = [], {}, {}
    for method, entries in banks.items():
        method_ids[method] = []
        for index in entries:
            if index.tobytes() not in lookup:
                lookup[index.tobytes()] = len(indices)
                indices.append(index)
            method_ids[method].append(lookup[index.tobytes()])
    fitted, histories, _ = fit_bank(
        indices, data, "final_validation", plan, family, device
    )
    np.savez_compressed(
        folder / "predictors.npz",
        indices=np.stack(indices),
        codes=fitted.cpu().numpy(),
        offsets=offsets,
    )
    save_json(
        folder / "predictors_frozen.json",
        {
            "frozen_utc": timestamp(),
            "sha256": digest(folder / "predictors.npz"),
            "histories": histories,
        },
    )
    meta = load_archive(root / "data" / f"meta_{family}.npz")
    x = torch.tensor(meta["x"], device=device)
    y = torch.tensor(meta[f"y_{condition}"] - offsets[:, None], device=device)
    losses = [
        float(
            (code_prediction(features(x, index, 4), values, plan) - y).square().mean()
        )
        for index, values in zip(indices, fitted, strict=True)
    ]
    selected = {
        method: {
            "candidate": min(numbers, key=lambda n: losses[n]),
            "bank_size": len(numbers),
        }
        for method, numbers in method_ids.items()
        if numbers
    }
    save_json(
        folder / "selection.json",
        {
            "frozen_utc": timestamp(),
            "family": family,
            "condition": condition,
            "abstained": False,
            "methods": selected,
            "meta_scores": losses,
            "seconds": time.monotonic() - started,
        },
    )
    print(
        json.dumps(
            {
                "source": [family, condition],
                "bootstrap_bank": len(bank),
                "final_banks": {m: len(v) for m, v in method_ids.items()},
                "seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


def freeze_sources(root, plan):
    paths = [
        root / "source" / f"case_{family}_{condition}" / "selection.json"
        for family in range(plan["families"])
        for condition in range(len(plan["noise"]))
    ]
    if not all(path.exists() for path in paths):
        raise RuntimeError("Source cohort is incomplete")
    save_json(
        root / "source_frozen.json",
        {
            "frozen_utc": timestamp(),
            "sha256": {str(path.relative_to(root)): digest(path) for path in paths},
        },
    )
    rows, arrays = [], {}
    for path in paths:
        selection = json.loads(path.read_text())
        family, condition = selection["family"], selection["condition"]
        teacher = load_archive(root / "private" / f"teacher_{family}.npz")
        if selection["abstained"]:
            rows.append({"family": family, "condition": condition, "abstained": True})
            continue
        predictors = load_archive(path.parent / "predictors.npz")
        for method, chosen in selection["methods"].items():
            metric, matched, _ = align_indices(
                predictors["indices"][chosen["candidate"]], teacher["index"], 4
            )
            rows.append(
                {
                    "family": family,
                    "condition": condition,
                    "noise": plan["noise"][condition],
                    "method": method,
                    "exact": metric["categorical_errors"] == 0,
                    **metric,
                }
            )
            arrays[f"{family}_{condition}_{method}"] = matched
        arrays[f"truth_{family}"] = teacher["index"]
    np.savez_compressed(root / "aligned_indices.npz", **arrays)
    save_json(root / "structure_audit.json", {"audited_utc": timestamp(), "rows": rows})
    print(json.dumps({"structure": rows}), flush=True)


def prepare_targets(root, plan):
    json.loads((root / "source_frozen.json").read_text())
    (root / "target").mkdir()
    for family in range(plan["families"]):
        rng = np.random.default_rng(plan["seed"] + 3000011 + 100003 * family)
        teacher = load_archive(root / "private" / f"teacher_{family}.npz")
        codes = rng.normal(size=(plan["targets"], 4))
        offsets = centered_offsets(codes, plan)
        zero = (
            real_output(np.zeros((1, 16)), teacher["index"], codes, plan["bias"])[:, 0]
            + offsets
        )
        zero_noise = rng.normal(size=len(codes))
        data, private = {}, {"codes": codes, "true_offsets": offsets}
        for condition, sigma in enumerate(plan["noise"]):
            data[f"zero_y_{condition}"] = zero + sigma * zero_noise / math.sqrt(
                plan["target_zero_repeats"]
            )
        for split, count in (
            ("support", plan["target_support"]),
            ("validation", plan["target_validation"]),
            ("test", plan["target_test"]),
        ):
            x = rng.normal(size=(count, 16))
            clean = (
                real_output(x, teacher["index"], codes, plan["bias"]) + offsets[:, None]
            )
            if split == "test":
                private.update(test_x=x, test_y=clean)
            else:
                data[split + "_x"] = x
                noise = rng.normal(size=clean.shape)
                for condition, sigma in enumerate(plan["noise"]):
                    data[f"{split}_y_{condition}"] = clean + sigma * noise / math.sqrt(
                        plan["target_repeats"]
                    )
        base = rng.normal(size=(64, 8, 2))
        permuted = np.stack(
            [[row[rng.permutation(8)] for _ in range(16)] for row in base]
        ).reshape(64, 16, 16)
        observed = np.empty_like(permuted)
        observed[..., teacher["permutation"]] = permuted
        private["permutation_x"] = observed.reshape(-1, 16)
        np.savez_compressed(root / "target" / f"data_{family}.npz", **data)
        np.savez_compressed(root / "private" / f"target_{family}.npz", **private)
    save_json(
        root / "target/data_frozen.json",
        {
            "frozen_utc": timestamp(),
            "source_freeze_sha256": digest(root / "source_frozen.json"),
            "sha256": {p.name: digest(p) for p in (root / "target").glob("data_*.npz")},
        },
    )


def dense_adam(data, plan, family, device):
    """Batch all target tasks/noise levels; keep independent native Adam states."""
    settings = plan["dense"]
    count = settings["starts_per_rate"]
    tasks = plan["targets"] * len(plan["noise"])
    x = torch.tensor(data["support_x"], dtype=torch.float32, device=device)
    vx = torch.tensor(data["validation_x"], dtype=torch.float32, device=device)
    offsets = np.concatenate(
        [
            data[f"zero_y_{c}"] - math.sqrt(plan["hidden"]) * plan["bias"]
            for c in range(len(plan["noise"]))
        ]
    )
    y = torch.tensor(
        np.concatenate([data[f"support_y_{c}"] for c in range(len(plan["noise"]))])
        - offsets[:, None],
        dtype=torch.float32,
        device=device,
    )
    vy = torch.tensor(
        np.concatenate([data[f"validation_y_{c}"] for c in range(len(plan["noise"]))])
        - offsets[:, None],
        dtype=torch.float32,
        device=device,
    )
    rng = np.random.default_rng(plan["seed"] + 5000011 + 997 * family)
    initial = rng.normal(0, 0.1, size=(plan["targets"], count, 16, 16))
    initial = np.tile(initial, (len(plan["noise"]), 1, 1, 1))
    parameters = [
        torch.nn.Parameter(
            torch.tensor(
                initial.reshape(-1, 16, 16), dtype=torch.float32, device=device
            )
        )
        for _ in settings["rates"]
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": [parameter], "lr": rate}
            for parameter, rate in zip(parameters, settings["rates"], strict=True)
        ]
    )
    task_ids = (
        torch.arange(tasks, device=device)
        .repeat_interleave(count)
        .repeat(len(parameters))
    )
    best = x.new_full((len(task_ids),), torch.inf)
    saved = torch.cat(parameters).detach().clone()
    best_step = torch.zeros(len(task_ids), device=device, dtype=torch.long)
    steps, train_history, validation_history = [], [], []
    generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + family * 37 + 6000011
    )

    def prediction(inputs, weights):
        return (torch.einsum("nd,mdh->mnh", inputs, weights) + plan["bias"]).relu().sum(
            -1
        ) / math.sqrt(plan["hidden"])

    for step in range(settings["steps"] + 1):
        weights = torch.cat(parameters)
        if step % settings["checkpoint_every"] == 0 or step == settings["steps"]:
            with torch.no_grad():
                validation = (prediction(vx, weights) - vy[task_ids]).square().mean(-1)
                training = (
                    (prediction(x[:1024], weights) - y[task_ids, :1024])
                    .square()
                    .mean(-1)
                )
                better = validation < best
                saved[better], best[better], best_step[better] = (
                    weights[better],
                    validation[better],
                    step,
                )
                steps.append(step)
                train_history.append(training.cpu().numpy())
                validation_history.append(validation.cpu().numpy())
                if not torch.isfinite(validation).all():
                    raise FloatingPointError("Nonfinite native dense Adam fit")
        if step == settings["steps"]:
            break
        ids = torch.randint(
            len(x), (settings["batch"],), device=device, generator=generator
        )
        loss = (
            (prediction(x[ids], weights) - y[task_ids[:, None], ids])
            .square()
            .mean(-1)
            .sum()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    winners = []
    for task in range(tasks):
        ids = (
            torch.flatnonzero(task_ids == task)
            if hasattr(torch, "flatnonzero")
            else torch.nonzero(task_ids == task).flatten()
        )
        winners.append(int(ids[best[ids].argmin()]))
    return {
        "weights": saved.cpu().numpy(),
        "task_ids": task_ids.cpu().numpy(),
        "winners": np.array(winners),
        "best_validation": best.cpu().numpy(),
        "best_step": best_step.cpu().numpy(),
        "history_steps": np.array(steps),
        "train_history": np.stack(train_history),
        "validation_history": np.stack(validation_history),
        "offsets": offsets,
    }


def target_fit(root, plan, family, device):
    started = time.monotonic()
    data = load_archive(root / "target" / f"data_{family}.npz")
    teacher = load_archive(root / "private" / f"teacher_{family}.npz")
    reference = canonical_index(teacher["index"], 4)
    rng = np.random.default_rng(plan["seed"] + family * 101 + 7100011)
    random_index = rng.integers(1, 5, size=(16, 16)) * rng.choice(
        (-1, 1), size=(16, 16)
    )
    random_index[rng.random((16, 16)) > 0.125] = 0
    folder = root / "target" / f"family_{family}"
    folder.mkdir(exist_ok=False)
    records = []
    for condition, _ in enumerate(plan["noise"]):
        selection = json.loads(
            (
                root / "source" / f"case_{family}_{condition}" / "selection.json"
            ).read_text()
        )
        source = (
            load_archive(
                root / "source" / f"case_{family}_{condition}" / "predictors.npz"
            )
            if selection["methods"]
            else None
        )
        indices, lookup, origins = [], {}, {}
        choices = [
            (method, source["indices"][chosen["candidate"]])
            for method, chosen in selection["methods"].items()
        ] + [("oracle", reference), ("random", random_index)]
        for name, index in choices:
            index = canonical_index(index, 4)
            if index.tobytes() not in lookup:
                lookup[index.tobytes()] = len(indices)
                indices.append(index)
            origins[name] = lookup[index.tobytes()]
        offsets = data[f"zero_y_{condition}"] - math.sqrt(plan["hidden"]) * plan["bias"]
        x, vx = [
            torch.tensor(data[key], device=device)
            for key in ("support_x", "validation_x")
        ]
        y, vy = [
            torch.tensor(data[f"{key}_y_{condition}"] - offsets[:, None], device=device)
            for key in ("support", "validation")
        ]
        fitted, history, elapsed = fit_codes(
            np.stack(indices),
            x,
            y,
            vx,
            vy,
            initial_codes(plan, family, plan["targets"], device, target=True),
            plan,
        )
        path = folder / f"codes_{condition}.npz"
        np.savez_compressed(
            path, indices=np.stack(indices), codes=fitted.cpu().numpy(), offsets=offsets
        )
        save_json(
            folder / f"codes_{condition}.json",
            {
                "origins": origins,
                "histories": history,
                "seconds": elapsed,
                "all_C_reset": True,
                "source_C_used": False,
                "data_sha256": digest(root / "target" / f"data_{family}.npz"),
            },
        )
        records.append(
            {"condition": condition, "origins": origins, "sha256": digest(path)}
        )
    dense = dense_adam(data, plan, family, device)
    np.savez_compressed(folder / "dense.npz", **dense)
    save_json(
        folder / "predictors_frozen.json",
        {
            "frozen_utc": timestamp(),
            "codes": records,
            "dense_sha256": digest(folder / "dense.npz"),
            "seconds": time.monotonic() - started,
            "test_opened": False,
        },
    )
    print(
        json.dumps(
            {
                "target_family": family,
                "seconds": time.monotonic() - started,
                "test_opened": False,
            }
        ),
        flush=True,
    )


def target_audit(root, plan):
    files = [
        root / "target" / f"family_{family}" / "predictors_frozen.json"
        for family in range(plan["families"])
    ]
    if not all(path.exists() for path in files):
        raise RuntimeError("Target cohort incomplete; keep test private")
    save_json(
        root / "target/cohort_frozen.json",
        {
            "frozen_utc": timestamp(),
            "sha256": {str(path.relative_to(root)): digest(path) for path in files},
        },
    )
    rows = []
    for family in range(plan["families"]):
        private = load_archive(root / "private" / f"target_{family}.npz")
        folder = root / "target" / f"family_{family}"
        variance = private["test_y"].var(1)
        dense = load_archive(folder / "dense.npz")
        for condition, sigma in enumerate(plan["noise"]):
            fitted = load_archive(folder / f"codes_{condition}.npz")
            origins = json.loads((folder / f"codes_{condition}.json").read_text())[
                "origins"
            ]
            for method, number in origins.items():
                index, codes = fitted["indices"][number], fitted["codes"][number]
                pred = (
                    real_output(private["test_x"], index, codes, plan["bias"])
                    + fitted["offsets"][:, None]
                )
                perm = real_output(
                    private["permutation_x"], index, codes, plan["bias"]
                ).reshape(8, 64, 16)
                errors = ((pred - private["test_y"]) ** 2).mean(1)
                invariance = np.sqrt(
                    ((perm - perm.mean(-1, keepdims=True)) ** 2).mean((1, 2))
                )
                for task in range(plan["targets"]):
                    rows.append(
                        {
                            "family": family,
                            "condition": condition,
                            "noise": sigma,
                            "task": task,
                            "method": method,
                            "mse": float(errors[task]),
                            "nmse": float(errors[task] / variance[task]),
                            "permutation_rms": float(invariance[task]),
                            "permutation_normalized_rms": float(
                                invariance[task] / math.sqrt(variance[task])
                            ),
                        }
                    )
            for task in range(plan["targets"]):
                combined = condition * plan["targets"] + task
                winner = int(dense["winners"][combined])
                weight = dense["weights"][winner].astype(float)
                pred = (
                    np.maximum(private["test_x"] @ weight + plan["bias"], 0).sum(1)
                    / math.sqrt(plan["hidden"])
                    + dense["offsets"][combined]
                )
                perm = (
                    np.maximum(private["permutation_x"] @ weight + plan["bias"], 0).sum(
                        1
                    )
                    / math.sqrt(plan["hidden"])
                ).reshape(64, 16)
                error = float(np.mean((pred - private["test_y"][task]) ** 2))
                inv = float(
                    np.sqrt(np.mean((perm - perm.mean(-1, keepdims=True)) ** 2))
                )
                rows.append(
                    {
                        "family": family,
                        "condition": condition,
                        "noise": sigma,
                        "task": task,
                        "method": "dense_adam",
                        "mse": error,
                        "nmse": error / float(variance[task]),
                        "permutation_rms": inv,
                        "permutation_normalized_rms": inv
                        / math.sqrt(float(variance[task])),
                        "selected_run": winner,
                        "best_step": int(dense["best_step"][winner]),
                    }
                )
    save_json(
        root / "target/evaluation.json",
        {
            "evaluated_utc": timestamp(),
            "rows": rows,
            "target_functions": plan["families"] * plan["targets"],
            "conditions_paired": True,
        },
    )
    print(json.dumps({"evaluated_rows": len(rows)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "source",
            "freeze_source",
            "prepare_target",
            "target_fit",
            "target_audit",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--family", type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.monotonic()
    if args.action == "prepare":
        plan = register(args.root)
        prepare(args.root, plan)
    else:
        plan = json.loads((args.root / "protocol.json").read_text())
    if args.action == "source":
        families = range(plan["families"]) if args.family is None else [args.family]
        for family in families:
            for condition in range(len(plan["noise"])):
                source_case(args.root, plan, family, condition, args.device)
    elif args.action == "freeze_source":
        freeze_sources(args.root, plan)
    elif args.action == "prepare_target":
        prepare_targets(args.root, plan)
    elif args.action == "target_fit":
        if args.family is None:
            raise ValueError("Specify one independent family per target worker")
        target_fit(args.root, plan, args.family, args.device)
    elif args.action == "target_audit":
        target_audit(args.root, plan)
    print(
        json.dumps({"action": args.action, "seconds": time.monotonic() - started}),
        flush=True,
    )


if __name__ == "__main__":
    main()
