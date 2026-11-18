"""Decode the manuscript's four-code example by joint log-magnitude residual."""

from itertools import combinations, permutations, product
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import save_json


OUTPUT = Path("artifacts/srr_theory_pilots_2026-11-19/cycle_results.json")
SEED = 2026111902
TRIALS = 256
TASKS = 4


def cycle_basis(labels, neurons):
    incidence = np.column_stack(
        (np.eye(4)[labels], -np.eye(neurons.max() + 1)[neurons])
    )
    left, singular, _ = np.linalg.svd(incidence, full_matrices=True)
    return left[:, np.count_nonzero(singular > 1e-9) :]


def run():
    # The local groups, active positions and alignment are observed.
    # A reference triple fixes code names, leaving three ordered pairs.
    pairs = tuple(permutations(range(4), 2))
    candidates = np.array(
        [
            (0, 1, 2, *first, *second, *third)
            for first, second, third in product(pairs, repeat=3)
        ]
    )
    neurons = np.repeat(np.arange(4), (3, 2, 2, 2))
    truth = np.array([0, 1, 2, 0, 3, 1, 3, 2, 3])
    true_id = np.flatnonzero((candidates == truth).all(axis=1)).item()
    bases = [cycle_basis(labels, neurons) for labels in candidates]
    width = max(basis.shape[1] for basis in bases)
    packed = np.column_stack(
        [
            np.pad(basis, ((0, 0), (0, width - basis.shape[1])))
            for basis in bases
        ]
    )

    generator = np.random.default_rng(SEED)
    codes = generator.normal(size=(TRIALS, TASKS, 4))
    scales = generator.normal(size=(TRIALS, TASKS, 4))
    clean = codes[:, :, truth] - scales[:, :, neurons]
    noise = generator.normal(size=clean.shape)
    results = []
    for sigma in (0.0, 0.05, 0.15):
        noisy = clean + sigma * noise
        projected = noisy.reshape(-1, len(truth)) @ packed
        residuals = (
            projected.reshape(TRIALS, TASKS, len(bases), width) ** 2
        ).sum(axis=-1)
        for tasks in (1, 2, 4):
            scores = residuals[:, :tasks].sum(axis=1)
            minimum = scores.min(axis=1, keepdims=True)
            ties = np.abs(scores - minimum) <= 1e-18 + 1e-10 * minimum
            exact = ties[:, true_id] & (ties.sum(axis=1) == 1)
            # Build one best candidate per task and re-rank their union
            # with the same all-task score used by the joint decoder.
            separate = residuals[:, :tasks].argmin(axis=2)
            union_scores = np.take_along_axis(scores, separate, axis=1)
            winner = separate[np.arange(TRIALS), union_scores.argmin(axis=1)]
            results.append(
                {
                    "log_noise_sd": sigma,
                    "tasks": tasks,
                    "joint_unique_exact": int(exact.sum()),
                    "separate_then_rescore_exact": int(
                        (winner == true_id).sum()
                    ),
                }
            )

    pair_only = np.array(list(combinations(range(4), 2)))
    pair_neurons = np.repeat(np.arange(len(pair_only)), 2)
    first = cycle_basis(pair_only.ravel(), pair_neurons)
    swapped = cycle_basis(pair_only[:, ::-1].ravel(), pair_neurons)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        OUTPUT,
        {
            "seed": SEED,
            "trials": TRIALS,
            "candidates": len(candidates),
            "true_local_codes": [[0, 1, 2], [0, 3], [1, 3], [2, 3]],
            "model": "independent standard normal code and neuron log magnitudes; "
            "all-positive weights; independent Gaussian additive log noise",
            "joint": "minimum total squared cycle residual over all candidates",
            "separate_then_rescore": "one minimum per task, chosen by total residual",
            "results": results,
            "pair_only_reciprocal_projector_distance": float(
                np.linalg.norm(first @ first.T - swapped @ swapped.T, ord=2)
            ),
        },
    )


if __name__ == "__main__":
    run()
