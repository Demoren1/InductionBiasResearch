"""Scientific oracle controls separating the theorem from fitted-model errors.

These diagnostics open generating parameters after the first source freeze.
They never contribute a candidate or a numerical state to a blind learner.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.paper_structure_benchmark_v1.protocol.prepare_general import (
    general_index,
    general_numbers,
)
from experiments.paper_structure_benchmark_v1.reporting.evaluate import write_csv
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .recover import (
    canonical_partition,
    contrast_rank,
    score_magnitudes,
    signature_partitions,
    task_precision_weights,
)


def centered_logs(normal, mask):
    rows, columns = np.where(mask)
    logs = np.log(np.maximum(np.abs(normal[:, rows, columns]), 1e-300)).T
    for neuron in np.unique(columns):
        active = columns == neuron
        logs[active] -= logs[active].mean(0)
    return logs


def inspect(normal, uncertainty, truth, codes, weighting):
    mask = truth != 0
    rows, columns = np.where(mask)
    labels = np.abs(truth[rows, columns]) - 1
    incidence = np.eye(codes)[labels]
    for neuron in np.unique(columns):
        active = columns == neuron
        incidence[active] -= incidence[active].mean(0)
    rank = contrast_rank(truth, codes)
    true_basis = np.linalg.svd(incidence, full_matrices=False)[0][:, :rank]
    weights = (
        task_precision_weights(uncertainty) if weighting == "uncertainty" else None
    )
    logs = centered_logs(normal, mask)
    if weights is not None:
        logs *= weights[None]
    basis, singular, _ = np.linalg.svd(logs, full_matrices=False)
    estimated = basis[:, :rank]
    eta = float(np.linalg.norm(estimated @ estimated.T - true_basis @ true_basis.T, 2))
    maximum_size = int(np.bincount(labels).max())
    limit = 1 / (8 * np.sqrt(maximum_size))
    lower = 2 * np.sqrt(2) * eta
    upper = np.sqrt(2 / maximum_size) / 2 - lower
    true_key = canonical_partition(np.abs(truth), codes).tobytes()
    proposals, _ = signature_partitions(normal, mask, codes, weights)
    eligible = [item for item in proposals if item["rank"] == rank]
    certified = [
        item
        for item in eligible
        if any(lower <= value < upper for value in item["threshold_values"])
    ]
    scores = [
        score_magnitudes(item["index"], normal, uncertainty, codes)[0]
        for item in eligible
    ]
    correct = [
        canonical_partition(item["index"], codes).tobytes() == true_key
        for item in eligible
    ]
    chosen = int(np.argmin(scores)) if scores else None
    return {
        "known_rank": rank,
        "observed_numerical_rank": int(np.linalg.matrix_rank(logs)),
        "eta": eta,
        "sufficient_limit": float(limit),
        "stable_condition": bool(eta < limit),
        "complete_known_rank_candidates": len(eligible),
        "true_partition_proposed": any(correct),
        "source_log_selected_correct": bool(chosen is not None and correct[chosen]),
        "certified_candidates": len(certified),
        "certified_candidates_all_correct": all(
            canonical_partition(item["index"], codes).tobytes() == true_key
            for item in certified
        ),
        "smallest_signal_singular": float(singular[rank - 1]) if rank else 0.0,
    }


def one_case(arguments):
    work, cohort, case, sizes = arguments
    root = Path(work) / cohort
    number = int(case.rsplit("_", 1)[1])
    if cohort == "standard":
        teacher = dict(np.load(root / "private" / f"{case}.npz"))
        truth, code = teacher["index"], teacher["codes"]
    else:
        rng = np.random.default_rng(27111064019 + 100003 * number)
        variant = ("local", "quartet_mix", "random")[number // 2]
        truth = general_index(rng, variant)
        code, _, _, _ = general_numbers(rng, variant, number % 2, 16, sizes["outputs"])
    codes = sizes["codes"]
    weight = np.sign(truth)[None] * np.c_[np.zeros(len(code)), code][:, np.abs(truth)]
    normal = weight / np.maximum(np.linalg.norm(weight, axis=1, keepdims=True), 1e-300)
    uncertainty = np.zeros((len(code), truth.shape[1]))
    family = (
        ("pattern" if number < 3 else "deepsets")
        if cohort == "standard"
        else ("general_local", "general_quartet_mix", "general_random")[number // 2]
    )
    rows = []
    base = {
        "cohort": cohort,
        "case": case,
        "family": family,
        "oracle_support": True,
        "oracle_alignment": True,
    }
    rows.append(
        {
            **base,
            "control": "exact_observations",
            "condition": -1,
            "weighting": "uniform",
            **inspect(normal, uncertainty, truth, codes, "uniform"),
        }
    )
    rng = np.random.default_rng(271112731 + 10007 * number + (cohort == "general"))
    rescaled = normal / rng.lognormal(0, 2, (len(code), 1, truth.shape[1]))
    rows.append(
        {
            **base,
            "control": "exact_positive_rescaling",
            "condition": -1,
            "weighting": "uniform",
            **inspect(rescaled, uncertainty, truth, codes, "uniform"),
        }
    )
    mask = truth != 0
    active_rows, columns = np.where(mask)
    rank = contrast_rank(truth, codes)
    signal = np.linalg.svd(centered_logs(normal, mask), compute_uv=False)[rank - 1]
    maximum_size = int(np.bincount(np.abs(truth[mask]) - 1).max())
    noise_limit = signal / (16 * np.sqrt(maximum_size))
    noise = rng.normal(size=(len(active_rows), len(code)))
    for neuron in np.unique(columns):
        active = columns == neuron
        noise[active] -= noise[active].mean(0)
    noise /= np.linalg.norm(noise, 2)
    for factor in (0.25, 0.5, 2.0, 8.0):
        disturbed = normal.copy()
        disturbed[:, active_rows, columns] *= np.exp((noise * factor * noise_limit).T)
        rows.append(
            {
                **base,
                "control": f"log_noise_{factor:g}_times_bound",
                "condition": -1,
                "weighting": "uniform",
                "noise_to_sufficient_bound": factor,
                **inspect(disturbed, uncertainty, truth, codes, "uniform"),
            }
        )
    for condition in (0, 1):
        sources = [
            dict(
                np.load(
                    root / "source" / f"{case}_c{condition}_t{task}" / "learned.npz"
                )
            )
            for task in range(len(code))
        ]
        fitted, deviations = [], []
        for task, source in enumerate(sources):
            reference, observed = normal[task], source["normal"]
            positive = ((reference[:, :, None] - observed[:, None]) ** 2).sum(0)
            negative = ((reference[:, :, None] + observed[:, None]) ** 2).sum(0)
            left, right = linear_sum_assignment(np.minimum(positive, negative))
            mapping = right[np.argsort(left)]
            fitted.append(observed[:, mapping])
            deviations.append(source["uncertainty"][mapping])
        for weighting in ("uniform", "uncertainty"):
            rows.append(
                {
                    **base,
                    "control": "fitted_dense_oracle_support_alignment",
                    "condition": condition,
                    "weighting": weighting,
                    **inspect(
                        np.stack(fitted), np.stack(deviations), truth, codes, weighting
                    ),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bundle", type=Path, default=Path("artifacts/anchor_start_2026-11-10")
    )
    args = parser.parse_args()
    if not (
        Path("artifacts/theorem_pair_recovery_2026-11-12") / "selection_freeze.json"
    ).exists():
        raise RuntimeError("Oracle diagnostics require frozen blind selections")
    sizes = json.loads((args.bundle / "case_sizes.json").read_text())
    save_json(
        args.output / "controls_registration.json",
        {
            "oracle_only": True,
            "feeds_learner": False,
            "code_sha256": {
                str(path): digest(path)
                for path in (Path(__file__), Path(__file__).with_name("recover.py"))
            },
            "threshold_audit": "Only grid values actually producing the partition are certified; gaps are not interpolated.",
            "noise_factors": [0.25, 0.5, 2.0, 8.0],
            "known_rank_in_controls": True,
        },
    )
    arguments = [
        (str(args.work), *key.split("/"), value) for key, value in sorted(sizes.items())
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for group in executor.map(one_case, arguments):
            rows.extend(group)
            print(
                json.dumps(
                    {
                        "finished_structures": len(rows) // 10,
                        "total_structures": len(arguments),
                    }
                ),
                flush=True,
            )
    write_csv(args.output / "diagnostic_controls.csv", rows)
    save_json(
        args.output / "diagnostic_controls.json",
        {
            "oracle_only": True,
            "feeds_learner": False,
            "unique_structures": len(arguments),
            "rows": rows,
            "guarantee_violations": sum(
                item["certified_candidates"] > 0
                and not item["certified_candidates_all_correct"]
                for item in rows
            ),
        },
    )


if __name__ == "__main__":
    main()
