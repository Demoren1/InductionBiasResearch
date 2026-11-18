"""Compensated merges and rank diagnostics on locally connected FC100."""

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.omniglot64_v1.merge_cost import (
    QuadraticBank,
    joint_basis,
    optimize,
    relu_bound,
)
from experiments.omniglot64_v1.merge_cost_omniglot import (
    OPTIONS,
    pair_diagnostic,
)
from experiments.omniglot64_v1.merge_cost_synthetic import OUTPUT, summary
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)
from experiments.srr_approx_v1.core import canonical
from experiments.srr_cluster_v1.clustering import centroid_fit
from experiments.wsr_omniglot_v1.source import read_json

from .data import FC100_SUPERCLASSES, fc100_panel
from .study import (
    DATA,
    LAYOUTS,
    SEED,
    TARGET_STEPS,
    evaluate,
    source_profiles,
    summarize,
)
from .study import OUTPUT as PILOT
from .tuning import PARTITION_SEED

OUTPUT = OUTPUT / "fc100"
LAYOUT = LAYOUTS["fc100"]
PREVIOUS = PILOT / "tuning" / "fc100"
DENSE = DATA.parent / "fc100/dense_fits.npz"
BUDGET = 3072
RANK_BUDGET = 768
TEST_COUNT = 2048
TEST_SEED = SEED + 221001
SOURCE_DIAGNOSTIC_SEED = SEED + 221002
MOVES = 32


@dataclass
class SourceData:
    weights: np.ndarray
    head: np.ndarray
    bias: np.ndarray
    output_bias: np.ndarray
    covariance: np.ndarray
    origin: np.ndarray
    profiles: np.ndarray
    support: np.ndarray
    holdout: np.ndarray

    def bank(self, metric):
        dims, channels = LAYOUT.fan_in, LAYOUT.channels
        cov = self.covariance.reshape(-1, dims, dims)
        if metric == "identity":
            cov = np.broadcast_to(np.eye(dims), cov.shape)
        importance = np.linalg.norm(self.head, axis=-1).reshape(-1, channels)
        return QuadraticBank(
            self.weights.reshape(-1, dims, channels), cov, importance
        )


def source_data(data):
    config = read_json(PREVIOUS / "registration.json")
    if digest(DENSE) != config["source_dense_sha256"]:
        raise ValueError("Changed FC100 source Dense fits")
    source = fc100_panel(data, "source", 16, SEED + 1)
    with np.load(DENSE, allow_pickle=False) as archive:
        if not np.array_equal(source.records, archive["records"]):
            raise ValueError("Different source episode records")
        parameters = archive["parameters"].astype(np.float64)

    profiles = source_profiles(parameters, LAYOUT)
    t, sites, dims, channels = 16, LAYOUT.sites, LAYOUT.fan_in, LAYOUT.channels
    width = LAYOUT.connections
    weights = parameters[:, 0, :width].reshape(t, sites, dims, channels)
    scales = np.maximum(np.linalg.norm(weights, axis=2), 1e-12)
    unit = weights / scales[:, :, None, :]
    order = np.stack(
        [
            [
                linear_sum_assignment(-a.T @ w)[1]
                for a, w in zip(unit[0], task, strict=True)
            ]
            for task in unit
        ]
    )
    aligned = np.take_along_axis(unit, order[:, :, None, :], axis=-1)
    bias = (
        parameters[:, 0, width : width + LAYOUT.hidden].reshape(
            t,
            sites,
            channels,
        )
        / scales
    )
    bias = np.take_along_axis(bias, order, axis=-1)
    head = parameters[:, 0, width + LAYOUT.hidden : -LAYOUT.outputs]
    head = head.reshape(t, sites, channels, LAYOUT.outputs)
    head = (head - head.mean(-1, keepdims=True)) * scales[..., None]
    head = np.take_along_axis(head, order[..., None], axis=2)
    head /= np.sqrt(LAYOUT.hidden)
    support = source.support_x.astype(np.float64)
    origin = support.mean(axis=1, keepdims=True)
    centered = (support - origin).transpose(0, 2, 1, 3)
    covariance = centered.transpose(0, 1, 3, 2) @ centered / support.shape[1]
    pooled = covariance.mean(axis=0)
    ridge = np.trace(pooled, axis1=1, axis2=2) / dims * 0.05
    metric = 0.5 * (covariance + pooled) + ridge[:, None, None] * np.eye(dims)

    rng = np.random.default_rng(SOURCE_DIAGNOSTIC_SEED)
    holdout = []
    for old in source.records:
        rows = []
        for indices in old:
            label = data["fine"][indices[0]]
            available = np.setdiff1d(
                np.flatnonzero(data["fine"] == label), indices
            )
            rows.extend(rng.choice(available, 50, replace=False))
        holdout.append(data["x"][rows])
    return SourceData(
        aligned,
        head,
        bias,
        parameters[:, 0, -LAYOUT.outputs :],
        metric,
        origin,
        profiles,
        support,
        np.asarray(holdout, dtype=np.float64),
    )


def initial_partitions(profiles, blocks, codes):
    tasks, dims, _ = profiles.shape
    slots = profiles.reshape(tasks, dims, LAYOUT.sites, LAYOUT.channels)
    return [
        centroid_fit(
            slab.transpose(1, 0, 2, 3).reshape(dims, -1),
            codes // LAYOUT.hidden,
            "lloyd",
            PARTITION_SEED + b * 1009,
            "cpu",
        )[0]
        for b, slab in enumerate(np.split(slots, blocks, axis=3))
    ]


def make_index(partitions):
    by_channel = np.repeat(
        np.column_stack(partitions),
        LAYOUT.channels // len(partitions),
        axis=1,
    )
    by_neuron = np.tile(by_channel, (1, LAYOUT.sites))
    return canonical(
        by_neuron * LAYOUT.hidden + np.arange(1, LAYOUT.hidden + 1)
    )


def expanded(weight):
    tasks, sites, dims, channels = weight.shape
    whole = np.zeros((tasks, sites, dims, sites, channels))
    for site in range(sites):
        whole[:, site, :, site, :] = weight[:, site]
    return whole.reshape(tasks, sites * dims, sites * channels)


def heldout_bound(source, approximation):
    t, _, _, _ = source.weights.shape
    hidden, dims = LAYOUT.hidden, LAYOUT.sites * LAYOUT.fan_in
    kl, bound = relu_bound(
        source.holdout.reshape(t, -1, dims),
        expanded(source.weights),
        expanded(approximation.reshape(source.weights.shape)),
        source.bias.reshape(t, hidden),
        source.head.reshape(t, hidden, LAYOUT.outputs),
        source.output_bias,
        origin=source.origin.reshape(t, 1, dims),
    )
    return {
        "holdout_kl": float(kl.mean()),
        "holdout_bound": float(bound.mean()),
        "median_kl_over_bound": float(np.median(kl / bound)),
        "violations": int(np.count_nonzero(kl > bound + 1e-10)),
    }


def construct():
    if not (OUTPUT.parent / "synthetic.json").exists():
        raise ValueError("Finish the synthetic benchmark first")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "block_indexators.npz"
    if path.exists():
        raise FileExistsError(path)
    with np.load(DATA / "fc100.npz", allow_pickle=False) as archive:
        src = source_data(dict(archive))
    banks = {key: src.bank(key) for key in ("identity", "covariance")}
    initial = initial_partitions(src.profiles, 4, BUDGET)
    partitions = {
        "four": initial,
        "eight": initial_partitions(src.profiles, 8, BUDGET),
        "four_m768": initial_partitions(src.profiles, 4, RANK_BUDGET),
        "eight_m768": initial_partitions(src.profiles, 8, RANK_BUDGET),
    }
    traces = {}
    for name, (metric, exchange, rematch) in OPTIONS.items():
        started = time.perf_counter()
        partitions[name], trace = optimize(
            banks[metric],
            initial,
            exchange=exchange,
            rematch=rematch,
            moves=MOVES,
        )
        traces[name] = {
            "trace": trace,
            "seconds": time.perf_counter() - started,
            "max_exact_gain_error": max(
                (
                    abs(x["predicted_gain"] - x["refit_gain"])
                    for x in trace[1:]
                ),
                default=0.0,
            ),
        }
        print(name, trace[-1], flush=True)
    indices = {name: make_index(p) for name, p in partitions.items()}
    for name, index in indices.items():
        expected = RANK_BUDGET if name.endswith("_m768") else BUDGET
        if index.max() != expected:
            raise ValueError(f"Wrong budget for {name}: {index.max()}")
    selected = read_json(PREVIOUS / "results.json")["selected"]
    if selected["family"] != "global" or selected["codes"] != BUDGET:
        raise ValueError("Unexpected global Ward budget")
    with np.load(
        PREVIOUS / "frozen_indexators.npz", allow_pickle=False
    ) as bank:
        indices["global_ward"] = bank["asrr"]

    stats = {}
    bank = banks["covariance"]
    for name, groups in partitions.items():
        fits = [bank.fit(labels) for labels in groups]
        match = bank.assignment(fits, True)
        stats[name] = {
            "refit_q": float(bank.losses(fits, match).sum()),
            "groups_per_block": [int(p.max() + 1) for p in groups],
            **heldout_bound(src, bank.directions_from_fits(fits, match)),
        }
    np.savez_compressed(path, **indices)
    save_json(
        OUTPUT / "source_diagnostics.json",
        {
            "heldout_examples_per_task": src.holdout.shape[1],
            "two_pixels": pair_diagnostic(
                src.support[:, ::8, ::2],
                tail_fraction=0.1,
            ),
            "block_fits": stats,
            "search": traces,
        },
    )
    save_json(
        OUTPUT / "registration.json",
        {
            "source_dense_sha256": digest(DENSE),
            "indexators_sha256": digest(path),
            "global_reference_sha256": digest(
                PREVIOUS / "frozen_indexators.npz"
            ),
            "primary_budget": BUDGET,
            "rank_budget": RANK_BUDGET,
            "spatial_sites": LAYOUT.sites,
            "site_channels": LAYOUT.channels,
            "sources": len(src.weights),
            "refinement_moves": MOVES,
            "source_diagnostic_seed": SOURCE_DIAGNOSTIC_SEED,
            "test_count": TEST_COUNT,
            "test_seed": TEST_SEED,
            "target_steps": TARGET_STEPS,
            "source_metric": "Task and same-site pooled covariance (50/50) + ridge",
            "matching": "Within each site and task, balanced across channel blocks",
            "primary": "covariance_rematch minus four, both at 3072",
        },
    )


def rank_statistics(data, target, indices):
    """Evaluate the projected mean-risk in the fixed test class population."""
    sites, dims, channels = LAYOUT.sites, LAYOUT.fan_in, LAYOUT.channels
    test = np.isin(data["coarse"], FC100_SUPERCLASSES["test"])
    classes = np.unique(data["fine"][test])
    samples = [
        data["x"][data["fine"] == c].astype(np.float32) for c in classes
    ]
    counts = np.array([len(s) for s in samples])
    trial = np.searchsorted(classes, data["fine"][target.records[..., 0]])
    shots = target.support_x.shape[1] // LAYOUT.outputs
    trial_means = (
        target.support_x.astype(np.float32)
        .reshape(
            len(target.units),
            LAYOUT.outputs,
            shots,
            sites,
            dims,
        )
        .mean(axis=2)
    )
    a, b = np.triu_indices(LAYOUT.outputs, k=1)

    def projected(basis):
        means, noise = [], []
        for x in samples:
            z = x @ basis
            mean = z.mean(axis=0)
            means.append(mean)
            noise.append(float(np.square(z - mean).sum() / (len(z) - 1)))
        population = np.stack(means)[trial]
        d = population[:, a] - population[:, b]
        residual = trial_means @ basis - population
        sample_error = residual[:, a] - residual[:, b]
        variance = np.asarray(noise)[trial]
        finite = variance * (1 / shots - 1 / counts[trial])
        return {
            "signal": np.square(d).sum(axis=(-2, -1)).mean(axis=1),
            "variance_iid": ((variance[:, a] + variance[:, b]) / shots).mean(
                axis=1
            ),
            "variance_finite": (finite[:, a] + finite[:, b]).mean(axis=1),
            "realized_error": np.square(sample_error)
            .sum(axis=(-2, -1))
            .mean(axis=1),
        }

    report = {}
    for budget, suffix in ((BUDGET, ""), (RANK_BUDGET, "_m768")):
        layouts = {
            name: [
                np.unique(chunk[:, 0], return_inverse=True)[1]
                for chunk in np.split(
                    indices[name + suffix][:, :channels], blocks, axis=1
                )
            ]
            for name, blocks in (("four", 4), ("eight", 8))
        }
        layouts["union"] = layouts["four"] + layouts["eight"]
        bases = {
            n: joint_basis(g).astype(np.float32) for n, g in layouts.items()
        }
        projected_scores = {
            name: projected(basis) for name, basis in bases.items()
        }
        base = projected_scores["four"]
        comparisons = {}
        for name in ("eight", "union"):
            diff = {k: projected_scores[name][k] - base[k] for k in base}
            comparisons[name] = {
                **{k: summary(v) for k, v in diff.items()},
                "iid_mse_gain": summary(diff["signal"] - diff["variance_iid"]),
                "finite_pool_mse_gain": summary(
                    diff["signal"] - diff["variance_finite"]
                ),
                "observed_mse_gain": summary(
                    diff["signal"] - diff["realized_error"]
                ),
            }
        report[budget] = {
            "rank_per_site": {key: u.shape[1] for key, u in bases.items()},
            **comparisons,
        }
    return {"shots": shots, "class_counts": counts.tolist(), "budgets": report}


def transfer(gpus):
    path = OUTPUT / "target_results.json"
    if path.exists():
        raise FileExistsError(path)
    archive_path = OUTPUT / "block_indexators.npz"
    config = read_json(OUTPUT / "registration.json")
    if digest(archive_path) != config["indexators_sha256"]:
        raise ValueError("Block indexators changed after freezing")
    with np.load(archive_path, allow_pickle=False) as data:
        indices = dict(data)
    global_path = OUTPUT / "global_indexators.npz"
    global_hash = digest(global_path)
    if (
        global_hash
        != read_json(OUTPUT / "global_diagnostics.json")["indexators_sha256"]
    ):
        raise ValueError("Global indexators changed after freezing")
    with np.load(global_path, allow_pickle=False) as data:
        indices.update(dict(data))
    seed = read_json(PREVIOUS / "registration.json")["test_fit_seed"]
    save_json(
        OUTPUT / "target_plan.json",
        {
            "block_indexators_sha256": digest(archive_path),
            "global_indexators_sha256": global_hash,
            "methods": [*indices, "dense"],
            "test_seed": TEST_SEED,
            "test_fit_seed": seed,
            "test_episodes": TEST_COUNT,
            "gpus": gpus,
        },
    )
    with np.load(DATA / "fc100.npz", allow_pickle=False) as archive:
        data = dict(archive)
    target = fc100_panel(data, "test", TEST_COUNT, TEST_SEED)
    with ProcessPoolExecutor(
        max_workers=len(gpus),
        mp_context=mp.get_context("spawn"),
    ) as pool:
        fits = evaluate(
            pool, gpus, target, LAYOUT, {**indices, "dense": None}, seed
        )
    pred = {k: v[0] for k, v in fits.items()}
    np.savez_compressed(
        OUTPUT / "target_predictions.npz",
        records=target.records,
        truth=target.query_y,
        **pred,
    )
    np.savez_compressed(
        OUTPUT / "target_losses.npz",
        **{name: row[1] for name, row in fits.items()},
    )
    accuracy = {
        k: (v == target.query_y).mean(axis=-1) for k, v in pred.items()
    }
    rows = {
        k: {
            "accuracy": summarize(accuracy[k], target.units),
            "ce": float(v[1].mean()),
            "max_gpu_shard_seconds": v[2],
            "peak_allocated_bytes": v[3],
        }
        for k, v in fits.items()
    }
    pairs = [
        (a, "four")
        for a in accuracy
        if a not in {"four", "four_m768", "eight_m768"}
    ]
    pairs += [
        ("eight_m768", "four_m768"),
        ("covariance_exchange", "identity_exchange"),
        ("covariance_exchange", "covariance_equal"),
        ("covariance_rematch", "covariance_exchange"),
        ("global_ward", "dense"),
    ]
    pairs += [
        (k, "global_ward")
        for k in accuracy
        if k.startswith("global_") and k != "global_ward"
    ]
    if "global_covariance" in accuracy and "global_identity" in accuracy:
        pairs += [("global_covariance", "global_identity")]
    save_json(
        path,
        {
            "episodes": TEST_COUNT,
            "methods": rows,
            "paired": {
                f"{a}_minus_{b}": {
                    "accuracy": summarize(
                        accuracy[a] - accuracy[b], target.units
                    ),
                    "ce": summarize(fits[a][1] - fits[b][1], target.units),
                }
                for a, b in pairs
            },
        },
    )
    save_json(
        OUTPUT / "rank_diagnostics.json",
        rank_statistics(data, target, indices),
    )
    print(
        {k: round(100 * v["accuracy"]["mean"], 3) for k, v in rows.items()},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("construct", "transfer"))
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    args = parser.parse_args()
    if args.action == "construct":
        construct()
    else:
        transfer(args.gpus)
