"""Synthetic experiments for the weighted merge, block exchange, and rank rules."""

import time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.linalg import null_space
from scipy.optimize import linear_sum_assignment
from scipy.special import ndtr
from sklearn.metrics import adjusted_rand_score

from experiments.asrr_anchor_init_v1.report import interval
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .merge_cost import (
    QuadraticBank,
    constraint_cost,
    indicators,
    optimize,
    relu_bound,
)

OUTPUT = Path("artifacts/asrr_compensated_merging_2026-11-19")
SEED = 7611193300
REPLICATES = 32


def summary(values):
    values = np.asarray(values)
    return {
        "mean": float(values.mean()),
        "ci95_halfwidth": float(interval(values)),
    }


def verify_quadratics(rng):
    errors = {}
    for name, rows in (("full_rank", 18), ("singular", 6)):
        errors[name] = []
        for _ in range(64):
            matrix = rng.normal(size=(rows, 12))
            target = rng.normal(size=rows)
            point = np.linalg.lstsq(matrix, target, rcond=None)[0]
            base = np.linalg.norm(matrix @ point - target) ** 2
            for sign in (1, -1):
                normal = np.zeros(12)
                normal[rng.choice(12, 2, replace=False)] = (1, -sign)
                if name == "singular":
                    # Test both ways in which a singular Gram can occur.
                    normals = (normal, matrix.T @ rng.normal(size=rows))
                else:
                    normals = (normal,)
                for direction in normals:
                    feasible = null_space(direction[None, :])
                    reduced = matrix @ feasible
                    refitted = np.linalg.lstsq(reduced, target, rcond=None)[0]
                    true_cost = (
                        np.linalg.norm(reduced @ refitted - target) ** 2 - base
                    )
                    estimate = constraint_cost(
                        matrix.T @ matrix, point, direction
                    )
                    errors[name].append(abs(true_cost - estimate))
    pair_examples = []
    for gamma in (0.0, 0.5, 0.95, 1.0):
        x = rng.multivariate_normal([0, 0], [[1, gamma], [gamma, 1]], 100_000)
        x -= x.mean(0)
        x /= x.std(0)
        weight = np.array([2.0, -1.0])
        total = x.sum(axis=1)
        c = np.dot(total, x @ weight) / np.dot(total, total)
        pair_examples.append(
            {
                "gamma": gamma,
                "predicted": 0.5
                * (1 - (x.T @ x / len(x))[0, 1])
                * np.diff(weight)[0] ** 2,
                "measured": float(np.mean((x @ weight - c * total) ** 2)),
            }
        )
    left = rng.integers(1, 30, 128)
    right = rng.integers(1, 30, 128)
    ward_errors = []
    for n, k in zip(left, right, strict=True):
        c = rng.normal(size=2)
        for sign in (1, -1):
            ward = n * k / (n + k) * (c[0] - sign * c[1]) ** 2
            estimated = constraint_cost(
                np.diag([n, k]), c, np.array([1, -sign])
            )
            ward_errors.append(abs(ward - estimated))
    images = rng.normal(size=(64, 512, 20)) + rng.normal(size=(64, 1, 20))
    truth = rng.normal(scale=0.3, size=(64, 20, 12))
    candidate = truth + rng.normal(scale=0.09, size=truth.shape)
    head = rng.normal(scale=0.3, size=(64, 12, 4))
    head -= head.mean(-1, keepdims=True)
    kl, bound = relu_bound(
        images,
        truth,
        candidate,
        rng.normal(size=(64, 12)),
        head,
        rng.normal(size=(64, 4)),
    )
    return {
        "two_inputs": pair_examples,
        "merge_max_absolute_error": {
            name: float(max(v)) for name, v in errors.items()
        },
        "merge_cases": {name: len(v) for name, v in errors.items()},
        "signed_ward_max_absolute_error": float(max(ward_errors)),
        "common_sign_example": {"common": 2.0, "independent_per_task": 0.0},
        "relu": {
            "tasks": len(kl),
            "violations": int(np.count_nonzero(kl > bound + 1e-10)),
            "median_kl_over_bound": float(np.median(kl / bound)),
        },
    }


def rank_rule(rng):
    repetitions = 32768
    base, full, n = 4, 8, 10
    noise = rng.normal(size=(repetitions, full)) * np.sqrt(2 / n)
    independent = rng.normal(size=noise.shape) * np.sqrt(2 / n)
    rows = []
    for energy in (0.16, 0.8, 1.6):
        delta = np.r_[
            2.0, np.zeros(base - 1), np.sqrt(energy), np.zeros(full - base - 1)
        ]
        estimate = delta + noise
        projected = estimate.copy()
        projected[:, base:] = 0
        diff = np.square(projected - delta).sum(-1) - np.square(
            estimate - delta
        ).sum(-1)
        cross = np.sum(
            estimate[:, base:] * (delta + independent)[:, base:], axis=-1
        )
        classification = []
        for fitted in (projected, estimate):
            margin = (fitted @ delta) / (2 * np.linalg.norm(fitted, axis=1))
            classification.append(ndtr(-margin))
        rows.append(
            {
                "added_signal": energy,
                "theoretical_mse_gain": energy - 2 * (full - base) / n,
                "measured_mse_gain": summary(diff),
                "crossfit_added_signal": summary(cross),
                "bayes_error_base": float(
                    ndtr(-np.linalg.norm(delta[:base]) / 2)
                ),
                "bayes_error_full": float(ndtr(-np.linalg.norm(delta) / 2)),
                "plug_in_error_gain": summary(
                    classification[0] - classification[1]
                ),
            }
        )
    return {"base_rank": base, "full_rank": full, "per_class": n, "rows": rows}


def structure_case(rng, difficult):
    sizes = (6, 5, 3, 2)
    tasks, width, dimensions = 12, 4, 36
    truth = [rng.permutation(np.arange(dimensions) % k) for k in sizes]
    means = [rng.normal(size=(k, width)) for k in sizes]
    gaussian = np.eye(dimensions)
    if difficult:
        basis = np.linalg.qr(rng.normal(size=(dimensions, dimensions)))[0]
        scale = np.r_[np.ones(12), np.full(24, 0.08)]
        gaussian = basis * scale
    covariance = gaussian @ gaussian.T
    weight = [
        (mean + rng.normal(scale=0.5, size=(tasks, *mean.shape)))[:, labels]
        for mean, labels in zip(means, truth, strict=True)
    ]
    weights = np.concatenate(weight, axis=2)
    if difficult:
        # Put errors of the observed Dense fits mostly in low-variance directions.
        weights += (
            1.1 * basis[:, 12:] @ rng.normal(size=(tasks, 24, 4 * width))
        )
        for task in range(tasks):
            order = np.r_[rng.choice(width, 1) + np.arange(4) * width]
            weights[task, :, order] = weights[task, :, np.roll(order, 1)]
    else:
        weights += rng.normal(scale=0.07, size=weights.shape)
    weights /= np.linalg.norm(weights, axis=1, keepdims=True)
    initial = []
    for block in np.split(weights, 4, axis=2):
        points = block.transpose(1, 0, 2).reshape(dimensions, -1)
        initial.append(
            cut_tree(linkage(points, method="ward"), n_clusters=[4])[:, 0]
        )
    identity = QuadraticBank(
        weights,
        np.broadcast_to(np.eye(dimensions), (tasks, dimensions, dimensions)),
        np.ones((tasks, 4 * width)),
    )
    functional = QuadraticBank(
        weights,
        np.broadcast_to(covariance, (tasks, dimensions, dimensions)),
        identity.importance,
    )
    variants = {"initial": initial}
    discrepancy = []
    for name, bank, exchange, rematch in (
        ("identity_exchange", identity, True, False),
        ("covariance_equal", functional, False, False),
        ("covariance_exchange", functional, True, False),
        ("covariance_rematch", functional, True, True),
    ):
        variants[name], trace = optimize(
            bank,
            initial,
            exchange=exchange,
            rematch=rematch,
            moves=24,
        )
        discrepancy.extend(
            abs(entry["predicted_gain"] - entry["refit_gain"])
            for entry in trace[1:]
        )

    def transfer_data():
        target_codes = [rng.normal(size=(k, width)) for k in sizes]
        target_weights = np.concatenate(
            [c[labels] for c, labels in zip(target_codes, truth, strict=True)],
            axis=1,
        )
        x = rng.normal(size=(40, dimensions)) @ gaussian.T
        y = x @ target_weights + rng.normal(
            scale=0.5, size=(len(x), 4 * width)
        )
        return x, y, target_weights

    held_out = [transfer_data() for _ in range(12)]
    rows = {}
    for name, partitions in {**variants, "oracle": truth}.items():
        match = np.array(
            [[adjusted_rand_score(a, b) for b in partitions] for a in truth]
        )
        order = linear_sum_assignment(-match)[1]
        ordered = [partitions[i] for i in order]
        reconstructed = []
        for x, y, true_weights in held_out:
            pieces = []
            for labels, target in zip(
                ordered, np.split(y, 4, axis=1), strict=True
            ):
                design = indicators(labels)
                fit = np.linalg.lstsq(x @ design, target, rcond=None)[0]
                pieces.append(design @ fit)
            diff = true_weights - np.concatenate(pieces, axis=1)
            reconstructed.append(
                float(np.sum(diff * (covariance @ diff)) / diff.shape[1])
            )
        rows[name] = {
            "ari": float(match[np.arange(4), order].mean()),
            "target_mse": float(np.mean(reconstructed)),
            "exact_budget": sorted(
                int(labels.max()) + 1 for labels in partitions
            )
            == sorted(sizes),
        }
    return rows, max(discrepancy, default=0.0)


def run():
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "synthetic.json"
    if path.exists():
        raise FileExistsError(path)
    save_json(
        OUTPUT / "synthetic_registration.json",
        {
            "seed": SEED,
            "replicates_per_regime": REPLICATES,
            "source_tasks": 12,
            "target_tasks_per_replica": 12,
            "target_examples_per_task": 40,
            "true_groups_per_block": [6, 5, 3, 2],
            "code_sha256": digest(Path(__file__)),
            "merge_implementation_sha256": digest(
                Path(__file__).with_name("merge_cost.py")
            ),
        },
    )
    rng = np.random.default_rng(SEED)
    numerical = verify_quadratics(rng)
    gaussian = rank_rule(rng)
    print(
        "Two-input, quadratic, ReLU and Gaussian simulations finished",
        flush=True,
    )
    structure = {}
    for name, difficult in (
        ("clean", False),
        ("correlated_noisy_permuted", True),
    ):
        rows = [structure_case(rng, difficult) for _ in range(REPLICATES)]
        structure[name] = {
            "max_absolute_gain_error": float(max(x[1] for x in rows)),
            "methods": {
                method: {
                    key: summary([r[0][method][key] for r in rows])
                    for key in ("ari", "target_mse", "exact_budget")
                }
                for method in rows[0][0]
            },
            "paired_against_initial": {
                method: {
                    key: summary(
                        [
                            r[0][method][key] - r[0]["initial"][key]
                            for r in rows
                        ]
                    )
                    for key in ("ari", "target_mse")
                }
                for method in rows[0][0]
                if method not in {"initial", "oracle"}
            },
        }
        print(
            name,
            {
                m: round(r["ari"]["mean"], 3)
                for m, r in structure[name]["methods"].items()
            },
            flush=True,
        )
    save_json(
        path,
        {
            "quadratics": numerical,
            "known_groups": structure,
            "rank_rule": gaussian,
            "seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    run()
