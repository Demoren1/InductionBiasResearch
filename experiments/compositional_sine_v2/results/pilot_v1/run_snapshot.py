"""Run the frozen CPU pilot for transfer across a million compositions."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def kernel(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    dot = left.T @ right
    norm_sum = (left * left).sum(0)[:, None]
    norm_sum = norm_sum + (right * right).sum(0)[None, :]
    return 0.5 * (
        np.exp(-np.maximum(norm_sum - 2 * dot, 0) / 2)
        - np.exp(-np.maximum(norm_sum + 2 * dot, 0) / 2)
    )


def population_mse(weights, amplitudes, task):
    truth_w, truth_a = task["weights"], task["amplitudes"]
    return float(
        max(
            0,
            amplitudes @ kernel(weights, weights) @ amplitudes
            - 2 * amplitudes @ kernel(weights, truth_w) @ truth_a
            + truth_a @ kernel(truth_w, truth_w) @ truth_a,
        )
    )


def sample(task, count, rng):
    x = rng.standard_normal((count, task["weights"].shape[0])).astype(
        np.float32
    )
    y = np.sin(x @ task["weights"]) @ task["amplitudes"]
    return x, y


def make_family(plan, rng):
    groups, width = plan["groups"], plan["width"]
    dimension = groups * width
    motifs = np.array(
        [[1, 1, 1, 1, -1, -1, -1, -1], [1, 1, -1, -1, 1, 1, -1, -1]],
        dtype=np.float32,
    ).T / math.sqrt(width)
    basis = np.zeros((dimension, 2 * groups), dtype=np.float32)
    for group in range(groups):
        basis[
            group * width : (group + 1) * width, 2 * group : 2 * group + 2
        ] = motifs
    basis = basis[rng.permutation(dimension)]
    order = rng.permutation(2 * groups)
    source_frequency = rng.uniform(
        *plan["source"]["frequency_range"], size=2 * groups
    ).astype(np.float32)
    return basis, order, source_frequency


def source_bank(plan, basis, order, frequencies, output):
    spec = plan["source"]
    rng = np.random.default_rng(plan["seed"] + 10)
    torch.manual_seed(plan["seed"] + 11)
    dimension, count = basis.shape
    restarts = spec["restarts"]
    teacher = basis[:, order] * frequencies[order]
    teacher_t = torch.from_numpy(teacher)
    weights = (
        torch.randn(dimension, count * restarts)
        * spec["initial_norm"]
        / math.sqrt(dimension)
    ).requires_grad_()
    amplitudes = torch.ones(count * restarts, requires_grad=True)
    optimizer = torch.optim.SGD(
        [weights, amplitudes], lr=spec["learning_rate"]
    )
    start = time.perf_counter()
    for _ in range(spec["steps"]):
        x = torch.from_numpy(
            rng.standard_normal((spec["batch_size"], dimension)).astype(
                np.float32
            )
        )
        y = torch.sin(x @ teacher_t).repeat_interleave(restarts, dim=1)
        pred = torch.sin(x @ weights) * amplitudes
        loss = (pred - y).square().mean(0).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start
    x = rng.standard_normal((spec["selection_examples"], dimension)).astype(
        np.float32
    )
    fitted = weights.detach().numpy()
    gamma = amplitudes.detach().numpy()
    target = np.sin(x @ teacher).repeat(restarts, axis=1)
    losses = ((np.sin(x @ fitted) * gamma - target) ** 2).mean(0)
    best = losses.reshape(count, restarts).argmin(1)
    indices = np.arange(count) * restarts + best
    fitted = fitted[:, indices] * np.sign(gamma[indices])[None, :]
    gamma = np.abs(gamma[indices])
    losses = losses[indices]
    if np.any(losses >= spec["acceptance_mse"]):
        save_json(
            output / "source_failure.json",
            {"selection_mse": losses.tolist(), "seconds": elapsed},
        )
        raise RuntimeError("Source bank did not pass the frozen MSE gate")
    recovered = np.zeros_like(fitted)
    for column in range(count):
        values = fitted[:, column]
        centers = np.array([values.min(), 0, values.max()])
        for _ in range(20):
            labels = np.abs(values[:, None] - centers[None, :]).argmin(1)
            centers = np.array([values[labels == k].mean() for k in range(3)])
        zero = np.abs(centers).argmin()
        signs = np.sign(centers[labels])
        signs[labels == zero] = 0
        recovered[:, column] = signs / np.linalg.norm(signs)
    supports = {}
    for column in range(count):
        support = tuple(np.flatnonzero(recovered[:, column]))
        supports.setdefault(support, []).append(column)
    if len(supports) != plan["groups"] or any(
        len(pair) != 2 for pair in supports.values()
    ):
        raise RuntimeError("Source solutions did not identify paired parts")
    pairs = np.array(list(supports.values()), dtype=int)
    alignment = np.abs(recovered.T @ basis).max(1)
    statistics = {
        "tasks": count,
        "fits": count * restarts,
        "seconds": elapsed,
        "selection_mse_max": float(losses.max()),
        "selection_mse_mean": float(losses.mean()),
        "exact_parts_diagnostic": int((alignment > 1 - 1e-6).sum()),
        "recovered_groups": len(pairs),
        "source_gradient_examples": (
            count * restarts * spec["steps"] * spec["batch_size"]
        ),
    }
    np.savez_compressed(
        output / "source_bank.npz",
        weights=fitted,
        amplitudes=gamma,
        recovered_basis=recovered,
        recovered_pairs=pairs,
        teacher_basis_diagnostic=basis,
        teacher_order_diagnostic=order,
    )
    save_json(output / "source_summary.json", statistics)
    print("SOURCE " + json.dumps(statistics), flush=True)
    return fitted, recovered, pairs


def make_target(plan, basis, bits, rng):
    groups = plan["groups"]
    indices = 2 * np.arange(groups) + bits
    frequencies = rng.uniform(
        *plan["target"]["frequency_range"], size=groups
    ).astype(np.float32)
    amplitudes = rng.uniform(
        *plan["target"]["amplitude_range"], size=groups
    ).astype(np.float32) / math.sqrt(groups)
    return {
        "weights": basis[:, indices] * frequencies,
        "amplitudes": amplitudes,
        "bits": bits,
    }


def fit_dense(plan, task, settings, restarts, seed, initial=None):
    spec = plan["target"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed + 1)
    dimension = task["weights"].shape[0]
    hidden = spec["hidden_units"]
    if initial is None:
        weights = torch.randn(restarts, dimension, hidden)
        weights *= settings["norm"] / math.sqrt(dimension)
    else:
        weights = torch.from_numpy(initial).repeat(restarts, 1, 1)
    weights.requires_grad_()
    amplitudes = (
        torch.randn(restarts, hidden) / math.sqrt(hidden)
    ).requires_grad_()
    optimizer_class = (
        torch.optim.Adam
        if settings["optimizer"] == "adam"
        else torch.optim.SGD
    )
    optimizer = optimizer_class(
        [weights, amplitudes], lr=settings["learning_rate"]
    )
    truth_w = torch.from_numpy(task["weights"])
    truth_a = torch.from_numpy(task["amplitudes"])
    started = time.perf_counter()
    for _ in range(spec["steps"]):
        x = torch.from_numpy(
            rng.standard_normal((spec["batch_size"], dimension)).astype(
                np.float32
            )
        )
        y = torch.sin(x @ truth_w) @ truth_a
        pred = (torch.sin(x @ weights) * amplitudes[:, None, :]).sum(-1)
        loss = (pred - y).square().mean(1).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    fitted_w = weights.detach().numpy()
    fitted_a = amplitudes.detach().numpy()
    select_x, select_y = sample(
        task,
        spec["selection_examples"],
        np.random.default_rng(seed + 100000),
    )
    errors = (np.sin(select_x @ fitted_w) * fitted_a[:, None, :]).sum(
        -1
    ) - select_y
    selection_losses = (errors * errors).mean(1)
    index = selection_losses.argmin()
    elapsed = time.perf_counter() - started
    return (
        fitted_w[index],
        fitted_a[index],
        {
            "selection_mse": float(selection_losses[index]),
            "seconds": elapsed,
            "gradient_examples": (
                restarts * spec["steps"] * spec["batch_size"]
            ),
            "restarts": restarts,
            "initial_norm": settings["norm"],
            "learning_rate": settings["learning_rate"],
        },
    )


def scan_parts(projected, y, pairs, frequencies):
    chosen, radial = [], []
    for pair in pairs:
        candidates = projected[:, pair, None] * frequencies[None, None, :]
        features = np.sin(candidates)
        correlation = (features * y[:, None, None]).mean(0)
        variance = (features * features).mean(0)
        scores = correlation**2 / np.maximum(variance, 1e-12)
        index, frequency = np.unravel_index(scores.argmax(), scores.shape)
        chosen.append(pair[index])
        radial.append(frequencies[frequency])
    return np.asarray(chosen), np.asarray(radial)


def fit_parts(plan, x, y, basis, pairs):
    spec = plan["structure_search"]
    frequencies = np.linspace(
        spec["frequency_min"],
        spec["frequency_max"],
        spec["frequency_points"],
        dtype=np.float32,
    )
    projected = x @ basis
    chosen, radial = scan_parts(projected, y, pairs, frequencies)
    features = np.sin(projected[:, chosen] * radial)
    amplitudes = np.linalg.lstsq(features, y, rcond=None)[0]
    for _ in range(spec["refinement_rounds"]):
        for group, pair in enumerate(pairs):
            residual = y - features @ amplitudes
            residual += features[:, group] * amplitudes[group]
            local = np.linspace(
                max(
                    spec["frequency_min"],
                    radial[group] - spec["refinement_radius"],
                ),
                min(
                    spec["frequency_max"],
                    radial[group] + spec["refinement_radius"],
                ),
                spec["refinement_points"],
                dtype=np.float32,
            )
            columns, values = scan_parts(projected, residual, [pair], local)
            chosen[group], radial[group] = columns[0], values[0]
            features[:, group] = np.sin(
                projected[:, chosen[group]] * radial[group]
            )
            amplitudes = np.linalg.lstsq(features, y, rcond=None)[0]
    evaluations = sum(len(pair) for pair in pairs) * len(frequencies)
    evaluations += (
        spec["refinement_rounds"]
        * sum(len(pair) for pair in pairs)
        * spec["refinement_points"]
    )
    return (
        basis[:, chosen] * radial,
        amplitudes,
        {
            "frequency_candidates": evaluations,
            "prediction_examples": evaluations * len(x),
            "chosen_parts": chosen.tolist(),
            "frequencies": radial.tolist(),
        },
    )


def fit_blind(plan, task, basis, pairs, x, y, selection_x, selection_y, rng):
    started = time.perf_counter()
    best_loss = float("inf")
    best = None
    candidates = plan["structure_search"]["blind_candidates"]
    codes = []
    total = 0
    for _ in range(candidates):
        bits = rng.integers(0, 2, size=len(pairs))
        codes.append(bits.tolist())
        singleton_pairs = [
            np.array([pair[bit]]) for pair, bit in zip(pairs, bits)
        ]
        weights, amplitudes, info = fit_parts(
            plan, x, y, basis, singleton_pairs
        )
        loss = float(
            (
                (np.sin(selection_x @ weights) @ amplitudes - selection_y) ** 2
            ).mean()
        )
        total += info["prediction_examples"]
        if loss < best_loss:
            best_loss = loss
            best = (weights, amplitudes)
    return (
        *best,
        {
            "selection_mse": best_loss,
            "seconds": time.perf_counter() - started,
            "prediction_examples": total,
            "unique_full_structures": len({tuple(code) for code in codes}),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=HERE / "results/pilot_v1"
    )
    args = parser.parse_args()
    plan_path = HERE / "protocol.json"
    plan = json.loads(plan_path.read_text())
    if plan["width"] != 8 or plan["source"]["tasks"] != 2 * plan["groups"]:
        raise ValueError("This pilot uses two width-eight motifs per group")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    seal = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    save_json(
        output / "seal.json",
        {
            "protocol_sha256": seal,
            "protocol": plan,
            "script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
        },
    )
    torch.set_num_threads(plan["threads"])
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(plan["seed"])
    basis, order, frequencies = make_family(plan, rng)
    bank_path = output / "source_bank.npz"
    if bank_path.exists():
        bank = np.load(bank_path)
        source_weights = bank["weights"]
        recovered, pairs = bank["recovered_basis"], bank["recovered_pairs"]
    else:
        source_weights, recovered, pairs = source_bank(
            plan, basis, order, frequencies, output
        )
    spec = plan["target"]
    count = spec["validation_tasks"] + spec["evaluation_tasks"]
    codes = rng.choice(2 ** plan["groups"], size=count, replace=False)
    bits = ((codes[:, None] >> np.arange(plan["groups"])) & 1).astype(int)
    tasks = [make_target(plan, basis, code, rng) for code in bits]
    validation = tasks[: spec["validation_tasks"]]
    evaluation = tasks[spec["validation_tasks"] :]
    np.savez_compressed(
        output / "target_tasks.npz",
        codes=codes,
        weights=np.stack([task["weights"] for task in tasks]),
        amplitudes=np.stack([task["amplitudes"] for task in tasks]),
    )
    settings_path = output / "selected_settings.json"
    if settings_path.exists():
        selected = json.loads(settings_path.read_text())
    else:
        selected = {}
        tuning_records = []
        for optimizer in ["sgd", "adam"]:
            options = []
            for norm, rate in itertools.product(
                plan["tuning"]["initial_norms"],
                plan["tuning"][f"{optimizer}_learning_rates"],
            ):
                settings = {
                    "optimizer": optimizer,
                    "norm": norm,
                    "learning_rate": rate,
                }
                losses = []
                for index, task in enumerate(validation):
                    _, _, info = fit_dense(
                        plan,
                        task,
                        settings,
                        spec["tuning_restarts"],
                        plan["seed"] + 1000 + index,
                    )
                    losses.append(info["selection_mse"])
                score = float(np.mean(losses))
                row = {**settings, "validation_mse": score}
                options.append(row)
                tuning_records.append(row)
                save_json(output / "tuning.json", tuning_records)
                print("TUNING " + json.dumps(row), flush=True)
            selected[optimizer] = min(
                options, key=lambda row: row["validation_mse"]
            )
        save_json(settings_path, selected)
    wrong_basis = recovered[rng.permutation(len(recovered))]
    records = []
    fitted = {}
    for index, task in enumerate(evaluation):
        task_rng = np.random.default_rng(plan["seed"] + 20000 + index)
        x, y = sample(task, spec["structure_examples"], task_rng)
        selection_x, selection_y = sample(
            task, spec["selection_examples"], task_rng
        )
        methods = [
            "learned_parts",
            "wrong_parts",
            "blind_structures",
            "fixed_source",
            "dense_sgd",
            "dense_adam",
            "source_start_adam",
        ]
        for method in methods:
            started = time.perf_counter()
            if method in ("learned_parts", "wrong_parts"):
                candidates = (
                    recovered if method == "learned_parts" else wrong_basis
                )
                weights, amplitudes, info = fit_parts(
                    plan, x, y, candidates, pairs
                )
            elif method == "blind_structures":
                weights, amplitudes, info = fit_blind(
                    plan,
                    task,
                    recovered,
                    pairs,
                    x,
                    y,
                    selection_x,
                    selection_y,
                    task_rng,
                )
            elif method == "fixed_source":
                features = np.sin(x @ source_weights)
                amplitudes = np.linalg.lstsq(features, y, rcond=None)[0]
                weights = source_weights
                info = {"prediction_examples": len(x) * features.shape[1]}
            else:
                optimizer = "sgd" if method == "dense_sgd" else "adam"
                initial = (
                    source_weights if method == "source_start_adam" else None
                )
                weights, amplitudes, info = fit_dense(
                    plan,
                    task,
                    selected[optimizer],
                    spec["evaluation_restarts"],
                    plan["seed"] + 30000 + index,
                    initial,
                )
            info["seconds"] = time.perf_counter() - started
            info["structure_training_examples"] = (
                len(x)
                if method
                not in ("dense_sgd", "dense_adam", "source_start_adam")
                else 0
            )
            risk = population_mse(weights, amplitudes, task)
            row = {
                "task": index,
                "code": int(codes[index + len(validation)]),
                "method": method,
                "population_mse": risk,
                "success": risk < spec["success_mse"],
                **info,
            }
            if method == "learned_parts":
                match = np.abs(
                    (weights / np.linalg.norm(weights, axis=0)).T
                    @ (
                        task["weights"]
                        / np.linalg.norm(task["weights"], axis=0)
                    )
                ).max(1)
                row["exact_parts"] = int((match > 1 - 1e-6).sum())
            records.append(row)
            fitted[f"{index}_{method}_weights"] = weights
            fitted[f"{index}_{method}_amplitudes"] = amplitudes
            save_json(output / "records.json", records)
            np.savez_compressed(output / "fitted_models.npz", **fitted)
            print("RESULT " + json.dumps(row), flush=True)
    summary = {}
    for method in methods:
        rows = [row for row in records if row["method"] == method]
        summary[method] = {
            "successes": sum(row["success"] for row in rows),
            "tasks": len(rows),
            "mean_mse": float(
                np.mean([row["population_mse"] for row in rows])
            ),
            "mean_seconds": float(np.mean([row["seconds"] for row in rows])),
        }
    save_json(output / "summary.json", summary)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
