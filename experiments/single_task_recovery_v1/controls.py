"""Oracle controls for one task, positive scales, and bounded log noise."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .ratios import pair_partition


def correct_partition(index, truth):
    active = truth != 0
    predicted, expected = index[active], np.abs(truth[active])
    return bool(
        np.array_equal(
            predicted[:, None] == predicted[None],
            expected[:, None] == expected[None],
        )
    )


def run(root):
    results = []
    save_json(
        root / "diagnostics" / "controls_registration.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "algorithm_sha256": digest(Path(__file__).with_name("ratios.py")),
            "scaling_trials_per_case": 20,
            "noise_trials_per_level": 8,
            "noise_fractions": [0.0, 0.25, 0.5, 1.0, 2.0],
            "oracle_support": True,
            "noise_threshold": (
                "2 epsilon + 1e-12; pair tolerance twice zero tolerance"
            ),
        },
    )
    for number in (4, 5):
        case = f"case_{number:02d}"
        truth = dict(np.load(root / "private" / f"{case}.npz"))
        index, code = truth["index"], truth["codes"][0]
        mask = index != 0
        weight = np.sign(index) * np.r_[0.0, code][np.abs(index)]
        log_code = np.log(np.abs(code))
        differences = np.array(
            [
                log_code[j] - log_code[k]
                for j in range(4)
                for k in range(4)
                if j != k
            ]
        )
        gamma_zero = float(np.min(np.abs(differences)))
        distance = np.abs(differences[:, None] - differences[None])
        np.fill_diagonal(distance, np.inf)
        gamma_pairs = float(distance.min())
        sufficient_error = min(gamma_zero / 4, gamma_pairs / 8)
        rng = np.random.default_rng(2611143203 + number * 100003)
        scaled_success = 0
        for _ in range(20):
            scaled = weight / rng.lognormal(0, 2, (1, 16))
            candidate, _ = pair_partition(scaled, mask, 1e-11, 2e-11)
            scaled_success += correct_partition(candidate, index)
        noise = []
        for fraction in (0.0, 0.25, 0.5, 1.0, 2.0):
            epsilon = fraction * sufficient_error
            correct, complete, false_merges = 0, 0, 0
            for _ in range(8):
                error = rng.uniform(-1, 1, size=weight.shape)
                error *= epsilon / np.max(np.abs(error[mask]))
                scaled = weight / rng.lognormal(0, 2, (1, 16))
                measured = scaled * np.exp(error)
                threshold = 2 * epsilon + 1e-12
                candidate, groups = pair_partition(
                    measured, mask, threshold, 2 * threshold
                )
                correct += correct_partition(candidate, index)
                complete += groups == 4
                labels, actual = candidate[mask], np.abs(index[mask])
                false_merges += bool(
                    np.any(
                        (labels[:, None] == labels[None])
                        & (actual[:, None] != actual[None])
                    )
                )
            noise.append(
                {
                    "fraction_of_sufficient_error": fraction,
                    "epsilon": epsilon,
                    "correct": correct,
                    "complete": complete,
                    "trials_with_false_merges": false_merges,
                    "trials": 8,
                }
            )
        results.append(
            {
                "case": case,
                "active": int(mask.sum()),
                "positive_rescaling_correct": scaled_success,
                "scaling_trials": 20,
                "gamma_zero": gamma_zero,
                "gamma_pairs": gamma_pairs,
                "sufficient_log_error": sufficient_error,
                "noise": noise,
            }
        )
    save_json(
        root / "diagnostics" / "exact_controls.json",
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "scope": "Oracle diagnostics; not passed to source learners",
            "cases": results,
        },
    )
    print(json.dumps(results), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
