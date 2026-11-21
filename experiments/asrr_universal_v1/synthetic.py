"""Compare aSRR bases and signed ties on the article's cross-correlation task."""

import argparse
import json

import numpy as np

from experiments.asrr_benchmark_families_v1.linear import atomic_npz, panel
from experiments.asrr_transfer_refinement_v1.common import OUT
from experiments.asrr_transfer_refinement_v1.reliable import symmetric
from experiments.paper_transfer_v2 import linear
from experiments.paper_transfer_v2.common import OUT as PAPER, SEED
from experiments.paper_transfer_v2.report import confidence
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .adaptive import (
    REVISED_STAGE,
    SHARES,
    affinity_halves,
    combine,
    source_groups,
)
from .fixed_blocks import GRADIENT_WEIGHTS, STAGE as FIXED_STAGE
from .run import STAGE, candidate_bases, guide_bases
from .structure import signed_groups, spectral_basis, unit_frobenius


def regression_signals(x, y):
    window = x[:, :, linear.WINDOW]
    left, right = [
        (features * targets[..., None]).mean(1).reshape(len(x), -1)
        for features, targets in zip(
            np.split(window, 2, axis=1), np.split(y, 2, axis=1), strict=True
        )
    ]
    signal = unit_frobenius(symmetric(left.T @ right))
    noise = np.eye(len(signal))
    halves = np.stack(
        [
            symmetric(left[i::2].T @ right[i::2]) / len(left[i::2])
            for i in range(2)
        ]
    )
    return (
        signal,
        (signal[None], np.zeros_like(signal[None]), noise[None]),
        halves,
    )


def ties(basis, index):
    first, second = np.triu_indices(len(basis), k=1)
    a, b = basis[first], basis[second]
    nonzero = (np.linalg.norm(a, axis=-1) > 1e-8) & (
        np.linalg.norm(b, axis=-1) > 1e-8
    )
    equal = (np.linalg.norm(a - b, axis=-1) <= 1e-8) & nonzero
    opposite = (np.linalg.norm(a + b, axis=-1) <= 1e-8) & nonzero
    found = equal.astype(int) - opposite
    u, v = index[first], index[second]
    truth = np.where(np.abs(u) == np.abs(v), np.sign(u * v), 0)
    correct = np.count_nonzero((truth == found) & (truth != 0))
    predicted = np.count_nonzero(found)
    expected = np.count_nonzero(truth)
    return {
        "precision": correct / max(1, predicted),
        "recall": correct / expected,
        "exact": bool(correct == predicted == expected),
    }


def run(stage):
    output = OUT / "synthetic" / stage
    output.mkdir(parents=True, exist_ok=True)
    cfg = linear.CFG
    span = max(cfg["shots"]) + cfg["query_examples"]
    layout = Layout(1, linear.CONNECTIONS, 1, 1)
    outcomes = []
    for geometry in range(cfg["geometries"]):
        seed = SEED + 2137 * geometry
        index = linear.teacher(np.random.default_rng(seed))
        x, y = panel(
            index,
            cfg["source_tasks"],
            cfg["source_examples"],
            seed + 1,
            cfg["source_noise"],
        )
        weights, _ = linear.source_regression(x, y)
        profiles = weights.reshape(len(weights), -1, 1)
        signal, stats, halves = regression_signals(x, y)
        variants = {}
        if stage == FIXED_STAGE:
            dense = profiles[..., 0].T
            kernel = unit_frobenius(dense @ dense.T)
            guide = guide_bases(*stats, linear.CODES)[0.0][0]
            for beta in GRADIENT_WEIGHTS:
                groups = signed_groups(
                    (1 - beta) * kernel + beta * signal, linear.CODES
                )
                for strength in SHARES[1:]:
                    variants[f"g{beta:g}_mix_{strength:g}"] = spectral_basis(
                        groups, guide, strength
                    )
        elif stage == REVISED_STAGE:
            groups, _, _ = source_groups(
                affinity_halves(profiles),
                {"query": halves[:, None]},
                {"query": 1.0},
                layout,
                linear.CODES,
                calibrated=False,
            )
            guide = guide_bases(*stats, linear.CODES)[0.0][0]
            variants = {
                f"mix_{strength:g}": combine(groups[0], guide, strength)
                for strength in SHARES
                if strength > 0
            }
        else:
            for knobs, basis in candidate_bases(
                profiles,
                stats,
                {"query": signal[None]},
                {"query": 0.0},
                layout,
                linear.CODES,
            ):
                _, gradient, share, moment = knobs
                if moment == 0:
                    variants[f"g{gradient:g}_s{share:g}"] = basis[0]

        dev_x, dev_y = panel(
            index, cfg["development_tasks"], span, seed + 2, 0
        )
        choice, validation = linear.select(dev_x, dev_y, variants)
        save_json(
            output / f"geometry_{geometry:02d}_choices.json",
            {"chosen": choice, "validation": validation},
        )

        test_x, test_y = panel(index, cfg["test_tasks"], span, seed + 3, 0)
        target = test_y[:, max(cfg["shots"]) :]
        truth = index[linear.WINDOW, linear.OUTPUT].ravel()
        per_task = {}
        outcomes.append({"choice": choice, "mse": {}, "ties": {}})
        for shot, entry in choice.items():
            basis = variants[entry["candidate"]]
            prediction = next(
                linear.ridge_predictions(
                    test_x, test_y, int(shot), basis, [entry["ridge"]]
                )
            )
            per_task[shot] = np.square(prediction - target).mean((1, 2))
            outcomes[-1]["mse"][shot] = float(per_task[shot].mean())
            outcomes[-1]["ties"][shot] = ties(basis, truth)
        atomic_npz(output / f"geometry_{geometry:02d}_errors.npz", **per_task)

    summary = {}
    for shot in map(str, cfg["shots"]):
        current = np.asarray([row["mse"][shot] for row in outcomes])
        measures = {"universal": confidence(current)}
        for stage, methods in (
            ("baselines", ("yeh_soft", "msr")),
            ("ward", ("asrr", "oracle")),
        ):
            for method in methods:
                other = np.asarray(
                    [
                        json.loads(
                            (
                                PAPER
                                / "linear"
                                / f"geometry_{i:02d}"
                                / f"{stage}_results.json"
                            ).read_text()
                        )["mse"][f"{method}_{shot}"]
                        for i in range(len(outcomes))
                    ]
                )
                measures[method] = confidence(other)
                measures[f"reduction_vs_{method}"] = confidence(
                    other - current
                )
        summary[shot] = measures
    save_json(
        output / "results.json", {"geometries": outcomes, "summary": summary}
    )
    print("synthetic", summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(STAGE, REVISED_STAGE, FIXED_STAGE),
        default=FIXED_STAGE,
    )
    run(parser.parse_args().stage)
