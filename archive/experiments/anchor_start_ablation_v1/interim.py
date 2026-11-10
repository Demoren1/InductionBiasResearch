"""Freeze completed source structures without tuning the remaining run."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.paper_structure_benchmark_v1.reporting.evaluate import (
    align_index,
    write_csv,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("pattern", "deepsets"), default="pattern"
    )
    dataset = parser.parse_args().dataset
    start, codes = (0, 2) if dataset == "pattern" else (3, 4)
    base = Path("/tmp/ibr-anchor-start-20261110")
    output = (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / (
            "anchor_start_interim_2026-11-10"
            if dataset == "pattern"
            else "anchor_start_deepsets_interim_2026-11-10"
        )
    )
    output.mkdir(exist_ok=False)
    methods = ("full", "anchor_fixed", "anchor_free")
    files = []
    for number in range(start, start + 3):
        for condition in (0, 1):
            for method in methods:
                folder = (
                    base
                    / "standard/methods"
                    / method
                    / "fits"
                    / f"case_{number:02d}_c{condition}"
                )
                files.extend(
                    folder / name
                    for name in (
                        "joint_selected.npz",
                        "joint_selected.json",
                        "joint_bank.npz",
                        "joint_bank.json",
                        "fitted.npz",
                    )
                )
    manifest = {
        "utc": datetime.now(UTC).isoformat(),
        "reason": f"User requested intermediate {dataset} results",
        "scope": "18 completed source selections; no target scoring",
        "remaining_fitting_or_selection_changed": False,
        "files": {str(p.relative_to(base)): digest(p) for p in files},
    }
    save_json(output / f"frozen_{dataset}.json", manifest)
    save_json(base / f"interim_{dataset}_amendment.json", manifest)
    rows, arrays = [], {}
    for number in range(start, start + 3):
        case = f"case_{number:02d}"
        truth = np.load(base / "standard/private" / f"{case}.npz")["index"]
        for condition in (0, 1):
            for method in methods:
                folder = (
                    base
                    / "standard/methods"
                    / method
                    / "fits"
                    / f"{case}_c{condition}"
                )
                index = np.load(folder / "joint_selected.npz")[
                    "joint_functional_index"
                ]
                aligned, recovery = align_index(index, truth, codes)
                initial = np.load(folder / "fitted.npz")["indices"]
                metadata = json.loads((folder / "joint_bank.json").read_text())
                rows.append(
                    {
                        "case": case,
                        "source_noise": condition * 0.1,
                        "method": method,
                        **recovery,
                        "selected_in_initial_bank": any(
                            np.array_equal(index, candidate)
                            for candidate in initial
                        ),
                        "projection_calls": metadata["projection_calls"],
                    }
                )
                prefix = f"{case}_c{condition}_{method}"
                arrays[prefix + "_aligned"] = aligned
                arrays[prefix + "_truth"] = truth
    write_csv(output / f"{dataset}_recovery.csv", rows)
    np.savez_compressed(output / f"{dataset}_structures.npz", **arrays)
    save_json(output / "summary.json", {"rows": rows})
    for noise in (0, 0.1):
        for method in methods:
            group = [
                row
                for row in rows
                if row["source_noise"] == noise and row["method"] == method
            ]
            print(
                json.dumps(
                    {
                        "noise": noise,
                        "method": method,
                        "exact": sum(row["exact"] for row in group),
                        "mask_exact": sum(row["mask_exact"] for row in group),
                        "total": len(group),
                        "entry_errors": [
                            row["signed_entry_errors"] for row in group
                        ],
                        "selected_from_initial": sum(
                            row["selected_in_initial_bank"] for row in group
                        ),
                        "projection_calls": [
                            row["projection_calls"] for row in group
                        ],
                    }
                )
            )


if __name__ == "__main__":
    main()
