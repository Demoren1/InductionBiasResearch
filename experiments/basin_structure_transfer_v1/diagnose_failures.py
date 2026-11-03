"""Attribute saved search failures without fitting or changing any model."""

import csv
import gzip
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import logsumexp

from experiments.basin_structure_transfer_v1.core import canonicalize
from experiments.basin_structure_transfer_v1.run import load_prior

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/failure_diagnosis_v1"
FIGURES = ROOT.parents[1] / "mds/assets/2026-11-02/basin_structure_transfer"
PERMUTATIONS = np.array(list(itertools.permutations(range(3))))


def sine_kernel(left, right):
    return 0.5 * (
        np.exp(-0.5 * ((left - right) ** 2).sum(-1))
        - np.exp(-0.5 * ((left + right) ** 2).sum(-1))
    )


def profiled_amplitude_risk(weights, teacher):
    """Diagnostic lower bound after optimizing only the output amplitude."""
    direction = teacher["direction"][:, None].astype(float)
    high = direction * teacher["high"][:, None, None]
    low = direction * teacher["low"][:, None, None]
    alpha = teacher["alpha"][:, None]
    energy = (
        alpha**2 * sine_kernel(low, low)
        + sine_kernel(high, high)
        + 2 * alpha * sine_kernel(low, high)
    )
    correlation = alpha * sine_kernel(weights, low) + sine_kernel(
        weights, high
    )
    return energy - correlation**2 / np.maximum(
        sine_kernel(weights, weights), 1e-30
    )


def representation_floor(teacher, library):
    """E Var(y | V_I^T x), a bound for any predictor in each subspace."""
    direction = teacher["direction"].astype(float)
    projected = np.zeros((len(direction), len(library), direction.shape[1]))
    for index, structure in enumerate(library):
        for group in np.unique(structure):
            positions = structure == group
            projected[:, index, positions] = direction[:, positions].mean(
                1, keepdims=True
            )
    residual = ((direction[:, None] - projected) ** 2).sum(2)
    high = teacher["high"][:, None]
    low = teacher["low"][:, None]
    alpha = teacher["alpha"][:, None]
    projected_high = projected * high[:, :, None]
    projected_low = projected * low[:, :, None]
    low_amplitude = alpha * np.exp(-0.5 * low**2 * residual)
    high_amplitude = np.exp(-0.5 * high**2 * residual)
    conditional_energy = (
        low_amplitude**2 * sine_kernel(projected_low, projected_low)
        + high_amplitude**2 * sine_kernel(projected_high, projected_high)
        + 2
        * low_amplitude
        * high_amplitude
        * sine_kernel(projected_low, projected_high)
    )
    original_high = direction[:, None] * high[:, :, None]
    original_low = direction[:, None] * low[:, :, None]
    energy = (
        alpha**2 * sine_kernel(original_low, original_low)
        + sine_kernel(original_high, original_high)
        + 2 * alpha * sine_kernel(original_low, original_high)
    )
    return np.maximum(energy - conditional_energy, 0)


def prior_audit(plan, source, library):
    variants = PERMUTATIONS[:, library].transpose(1, 0, 2).reshape(-1, 24)
    results = []
    for seed in range(len(plan["prior_seeds"])):
        model = load_prior(plan, source, seed, "cpu")
        with torch.no_grad():
            inputs = torch.tensor(variants)
            log_probabilities = model(inputs).double().log_softmax(-1)
            first_probability = log_probabilities[0, 0].exp().numpy()
            log_probabilities = log_probabilities.gather(
                -1, inputs[:, :, None]
            ).squeeze(-1)
        log_joint = log_probabilities.sum(-1).numpy().reshape(-1, 6)
        masses = np.exp(logsumexp(log_joint, axis=1))
        first_zero = PERMUTATIONS[:, 0] == 0
        conditional_mass = (
            np.exp(logsumexp(log_joint[:4, first_zero], axis=1))
            / first_probability[0]
        )
        results.append(
            {
                "seed_index": seed,
                "known_family_mass": masses[:4].tolist(),
                "foreign_family_mass": float(masses[4]),
                "outside_library_mass": float(1 - masses[:4].sum()),
                "expected_correct_attempts": float(32 * masses[:4].mean()),
                "first_token_probability": first_probability.tolist(),
                "known_mass_conditional_first_zero": float(
                    conditional_mass.sum()
                ),
                "family_mass_conditional_first_zero": (
                    conditional_mass.tolist()
                ),
            }
        )
    return results


def method_audit(directory, method, teacher, library, frequency, rows):
    saved = np.load(directory / method / "models.npz")
    geometry = np.load(directory / method / "endpoint_geometry.npz")
    with gzip.open(directory / method / "history.json.gz", "rt") as stream:
        history = json.load(stream)
    indices = canonicalize(saved["indices"])
    exact = (indices == teacher["indices"][:, None]).all(2)
    membership = (indices[:, :, None] == library[None, None, :4]).all(3)
    in_library = membership.any(2)
    query, risk = saved["query"][:, 0], saved["population"][:, 0]
    success = risk < 0.1
    selected = query.argmin(1)
    solved = success[np.arange(len(query)), selected]
    population_history = np.array([item["population"] for item in history])
    ever_success = (population_history < 0.1).any(0)
    near_bad = (geometry["distance_to_bad"] < 0.2) & (
        geometry["gradient_norm"] < 0.02
    )
    count = int((teacher["families"] < 4).sum())
    failed_exact = exact[:count] & ~success[:count]
    final_weights = saved["final_weights"][:, 0].astype(float)
    high = teacher["direction"] * teacher["high"][:, None]
    high_distance = np.minimum(
        np.linalg.norm(final_weights - high[:, None], axis=2),
        np.linalg.norm(final_weights + high[:, None], axis=2),
    )
    amplitude_rescue = (
        profiled_amplitude_risk(saved["weights"][:, 0].astype(float), teacher)
        < 0.1
    )
    early = {}
    for checkpoint in history:
        if checkpoint["step"] not in (0, 256, 512, 1024, 2048):
            continue
        choice = np.array(checkpoint["query"]).argmin(1)
        correct = exact[np.arange(len(query)), choice]
        early[checkpoint["step"]] = {
            "chosen_exact": int(correct[:count].sum()),
            "chosen_exact_on_eventually_unsolved": int(
                (correct & ~solved)[:count].sum()
            ),
        }
    failed_tasks = np.flatnonzero(~solved[:count])
    for task in range(len(query)):
        rows.append(
            {
                "frequency": frequency,
                "method": method,
                "task": task,
                "family": int(teacher["families"][task]),
                "known_family": task < count,
                "solved": bool(solved[task]),
                "selected_mse": float(risk[task, selected[task]]),
                "correct_proposals": int(exact[task].sum()),
                "successful_correct": int((exact & success)[task].sum()),
                "successful_other": int((~exact & success)[task].sum()),
            }
        )
    result = {
        "known_tasks": count,
        "known_tasks_solved": int(solved[:count].sum()),
        "tasks_with_correct_proposal": int(exact[:count].any(1).sum()),
        "failed_task_ids": failed_tasks.tolist(),
        "correct_proposals_on_failed_tasks": exact[failed_tasks]
        .sum(1)
        .tolist(),
        "candidate_counts": {
            "outside_library_failed": int(
                (~in_library & ~success)[:count].sum()
            ),
            "other_library_failed": int(
                (in_library & ~exact & ~success)[:count].sum()
            ),
            "correct_failed": int(failed_exact.sum()),
            "other_successful": int((~exact & success)[:count].sum()),
            "correct_successful": int((exact & success)[:count].sum()),
        },
        "query_missed_successful_checkpoint": int(
            (ever_success & ~success)[:count].sum()
        ),
        "query_missed_successful_candidate": int(
            (success.any(1) & ~solved)[:count].sum()
        ),
        "correct_failures_near_bad_minimum": int(
            (near_bad[:count] & failed_exact).sum()
        ),
        "amplitude_only_rescues_correct_failure": int(
            (amplitude_rescue[:count] & failed_exact).sum()
        ),
        "failed_correct_final_medians": {
            "absolute_amplitude": float(
                np.median(
                    np.abs(saved["final_amplitude"][:count, 0])[failed_exact]
                )
            ),
            "weight_norm": float(
                np.median(
                    np.linalg.norm(final_weights[:count], axis=2)[failed_exact]
                )
            ),
            "distance_to_high_frequency": float(
                np.median(high_distance[:count][failed_exact])
            ),
        },
        "early_query_routing": early,
    }
    if method.startswith("ar_"):
        invalid = indices[:count][~in_library[:count]]
        variants = PERMUTATIONS[:, library[:4]].reshape(-1, 24)
        differences = invalid[:, None] != variants[None]
        nearest = differences.sum(2).argmin(1)
        mismatch = differences[np.arange(len(invalid)), nearest]
        distances = mismatch.sum(1)
        result["outside_library_hamming_distance"] = {
            str(value): int((distances == value).sum())
            for value in np.unique(distances)
        }
        result["single_edit_error_positions"] = (
            mismatch[distances == 1].sum(0).tolist()
        )
        result["nonzero_first_token_outside_library"] = int(
            ((saved["indices"][:count, :, 0] != 0) & ~in_library[:count]).sum()
        )
        oracle = np.load(directory / "oracle/models.npz")
        raw_exact = (
            saved["indices"][:count] == teacher["indices"][:count, None]
        ).all(2)
        initial_difference = np.linalg.norm(
            saved["initial_values"][:count, 0]
            - oracle["initial_values"][:count, 0],
            axis=2,
        )
        oracle_success = oracle["population"][:count, 0] < 0.1
        result["oracle_pairing_limit"] = {
            "same_raw_indexator_initial_max_difference": float(
                initial_difference[raw_exact].max()
            ),
            "same_raw_indexator_success_disagreements": int(
                ((success[:count] != oracle_success) & raw_exact).sum()
            ),
            "interpretation": (
                "Same seeds are not a bitwise paired counterfactual; "
                "do not replay individual oracle outcomes as AR outcomes."
            ),
        }
    return result


def main():
    plan = json.loads((ROOT / "protocol.json").read_text())
    source = ROOT / "results/pilot_v1"
    teacher = np.load(source / "evaluation/teacher_diagnostic.npz")
    library = np.stack(
        [
            teacher["indices"][np.flatnonzero(teacher["families"] == i)[0]]
            for i in range(5)
        ]
    )
    result = {
        "scope": "post-hoc diagnostics; no fitting or hyperparameter changes",
        "prior_probabilities": prior_audit(plan, source, library),
        "regimes": {},
    }
    rows = []
    for frequency in (4, 6, 8):
        run = (
            source
            if frequency == 4
            else (ROOT / f"results/frequency_sweep_v1/frequency_{frequency}")
        )
        directory = run / "evaluation"
        teacher = np.load(directory / "teacher_diagnostic.npz")
        methods = {
            method: method_audit(
                directory, method, teacher, library, frequency, rows
            )
            for method in ("ar_0", "ar_1", "ar_2", "retrieval", "oracle")
        }
        floor = representation_floor(teacher, library[:4])
        correct = np.arange(4)[None] == teacher["families"][:32, None]
        result["regimes"][frequency] = {
            "methods": methods,
            "representation_floor": {
                "known_wrong_pairs_below_threshold": int(
                    ((floor[:32] < 0.1) & ~correct).sum()
                ),
                "known_wrong_pairs_total": int((~correct).sum()),
                "foreign_best_library_floor": floor[32:].min(1).tolist(),
                "known_correct_max": float(floor[:32][correct].max()),
            },
        }
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "diagnosis.json").write_text(json.dumps(result, indent=2))
    with (OUTPUT / "tasks.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = ["#aaaaaa", "#d8b57d", "#c7704c", "#73b8a7", "#267788"]
    labels = [
        "Outside source library; failed",
        "Other source structure; failed",
        "Correct structure; numeric fit failed",
        "Other structure; successful",
        "Correct structure; successful",
    ]
    fig, axis = plt.subplots(figsize=(10, 5))
    keys = list(methods["ar_0"]["candidate_counts"])
    left = np.zeros(3)
    for key, color, label in zip(keys, colors, labels):
        values = np.array(
            [
                sum(
                    result["regimes"][frequency]["methods"][f"ar_{seed}"][
                        "candidate_counts"
                    ][key]
                    for seed in range(3)
                )
                / 3072
                * 100
                for frequency in (4, 6, 8)
            ]
        )
        axis.barh([0, 1, 2], values, left=left, color=color, label=label)
        left += values
    axis.set_yticks([0, 1, 2], ["Frequency 4", "Frequency 6", "Frequency 8"])
    axis.set_xlim(0, 100)
    axis.set_xlabel("Share of AR candidates (%)")
    axis.set_title("Where 32 attempts per function are spent")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "failure_funnel.png", dpi=170)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
