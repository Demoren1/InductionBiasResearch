"""Audit frozen dense candidates against previously opened references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/constructive_dense_recovery_2026-11-14"),
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=Path(
            "artifacts/single_task_recovery_2026-11-14/"
            "recovery_structures.npz"
        ),
    )
    args = parser.parse_args()
    selected = json.loads(
        (args.output / "dense_selected_frozen.json").read_text()
    )
    records = []
    with np.load(args.references, allow_pickle=False) as references:
        for result in selected["results"]:
            case = result["case"]
            truth = references[f"{case}_truth"]
            rows = np.asarray(result["rows"])
            columns = np.asarray(result["columns"])
            for family in result["families"]:
                index = np.zeros(truth.shape, dtype=np.int16)
                index[rows, columns] = (
                    (np.asarray(family["indexator"]) + 1)
                    * np.asarray(family["fixed_signs"])
                )
                cost = (
                    (index[:, :, None] != 0)
                    != (truth[:, None, :] != 0)
                ).sum(axis=0)
                left, right = linear_sum_assignment(cost)
                record = {
                    "case": case,
                    "epsilon": result["epsilon"],
                    "branch": family["branch"],
                    "signed_exact_up_to_code_gauges_and_neuron_permutation": (
                        bool(
                            np.array_equal(
                                canonical_index(index, 4),
                                canonical_index(truth, 4),
                            )
                        )
                    ),
                    "mask_errors_after_neuron_assignment": int(
                        cost[left, right].sum()
                    ),
                    "max_log_residual": family["max_log_residual"],
                    "candidate_seconds": result["seconds"],
                }
                records.append(record)
                print(record, flush=True)
    (args.output / "dense_audit.json").write_text(
        json.dumps(
            {
                "references": str(args.references),
                "same_cases_previously_opened": True,
                "new_training_or_transfer": False,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
