"""Open structural truth after all source banks and selections are frozen."""

import argparse
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
    digest,
    save_json,
)
from experiments.theorem_pair_recovery_v1.recover import canonical_partition


def index_metrics(index, truth):
    signed = np.array_equal(
        canonical_index(index, 4), canonical_index(truth, 4)
    )
    unsigned = np.array_equal(
        canonical_partition(np.abs(index), 4),
        canonical_partition(np.abs(truth), 4),
    )
    mask_cost = ((index[:, :, None] != 0) != (truth[:, None] != 0)).sum(0)
    left, right = linear_sum_assignment(mask_cost)
    best_error = index.size
    for permutation in itertools.permutations(range(1, 5)):
        for signs in itertools.product((-1, 1), repeat=4):
            labels = np.r_[0, np.array(permutation) * signs]
            renamed = labels[np.abs(index)] * np.sign(index)
            cost = (renamed[:, :, None] != truth[:, None]).sum(0)
            row, column = linear_sum_assignment(cost)
            best_error = min(best_error, int(cost[row, column].sum()))
    return {
        "signed_exact": bool(signed),
        "unsigned_exact": bool(unsigned),
        "signed_entry_errors": best_error,
        "mask_errors": int(mask_cost[left, right].sum()),
        "active_edges": int(np.count_nonzero(index)),
    }


def run(root):
    for case in ("case_04", "case_05"):
        for path in (
            root / "ratio" / f"{case}_c0" / "selected.json",
            root / "fits" / f"{case}_c0" / "joint_selected.json",
        ):
            if not path.exists():
                raise ValueError(f"Source selection not frozen: {path}")
    records = []
    for case in ("case_04", "case_05"):
        truth_data = dict(np.load(root / "private" / f"{case}.npz"))
        truth = truth_data["index"]
        truth_key = canonical_index(truth, 4).tobytes()
        source = dict(
            np.load(root / "source" / f"{case}_c0_t0" / "learned.npz")
        )
        true_weight = (
            np.sign(truth) * np.r_[0, truth_data["codes"][0]][np.abs(truth)]
        )
        true_normal = true_weight / np.linalg.norm(true_weight, axis=0)
        cost = (
            (true_normal[:, :, None] - source["normal"][:, None]) ** 2
        ).sum(0)
        left, right = linear_sum_assignment(cost)
        mapping = right[np.argsort(left)]
        aligned = source["normal"][:, mapping]
        mask = truth != 0
        log_error = float(
            np.max(
                np.abs(
                    np.log(np.abs(aligned[mask]))
                    - np.log(np.abs(true_normal[mask]))
                )
            )
        )
        methods = {}
        for method in ("ratio", "g34"):
            if method == "ratio":
                folder = root / "ratio" / f"{case}_c0"
                bank = dict(np.load(folder / "fitted.npz"))
                selection = json.loads((folder / "selected.json").read_text())
                selected = selection["selected"]
                geometry = selection["geometric_first"]
                extraction = json.loads(
                    (folder / "candidates.json").read_text()
                )
                extraction_seconds = extraction["diagnostics"]["seconds"]
                total_seconds = extraction_seconds + selection["seconds"]
                proposals = extraction["diagnostics"][
                    "unique_complete_partitions"
                ]
            else:
                folder = root / "fits" / f"{case}_c0"
                bank = dict(np.load(folder / "joint_fitted.npz"))
                selection = json.loads(
                    (folder / "joint_selected.json").read_text()
                )
                selected = selection["methods"]["joint_functional"][
                    "candidate"
                ]
                geometry = None
                metrics = json.loads(
                    (root / "metrics" / f"{case}_g34.json").read_text()
                )
                extraction_seconds = sum(
                    stage["seconds"]
                    for stage in metrics["stages"]
                    if stage["stage"] == "propose"
                )
                total_seconds = metrics["seconds"]
                proposals = len(np.load(folder / "joint_bank.npz")["indices"])
            matches = [
                i
                for i, index in enumerate(bank["indices"])
                if canonical_index(index, 4).tobytes() == truth_key
            ]
            methods[method] = {
                "generated_proposals": proposals,
                "evaluated_bank_size": len(bank["indices"]),
                "truth_in_bank": bool(matches),
                "truth_candidate_ids": matches,
                "selected": selected,
                "selection": index_metrics(bank["indices"][selected], truth)
                if selected is not None
                else None,
                "geometric_first": index_metrics(
                    bank["indices"][geometry], truth
                )
                if geometry is not None
                else None,
                "proposal_seconds": extraction_seconds,
                "bank_and_selection_seconds": total_seconds,
            }
        source_metrics = json.loads(
            (root / "metrics" / f"{case}_source.json").read_text()
        )
        records.append(
            {
                "case": case,
                "sources": 1,
                "true_active_edges": int(mask.sum()),
                "source_seconds": source_metrics["seconds"],
                "max_direction_error": float(np.sqrt(cost[left, right].max())),
                "max_log_direction_error_on_active": log_error,
                "methods": methods,
            }
        )
    project = Path(__file__).resolve().parents[2]
    registered = json.loads((root / "registered_sources.json").read_text())
    changed = [
        name
        for name, value in registered["code_sha256"].items()
        if digest(project / name) != value
    ]
    violations = []
    for path in sorted((root / "receipts").glob("*.json")):
        receipt = json.loads(path.read_text())
        violations.extend(receipt.get("violations", []))
    result = {
        "audited_utc": datetime.now(UTC).isoformat(),
        "cases": records,
        "changed_registered_core_files": changed,
        "access_violations": violations,
        "scope": "One new source task per historical geometry; exploratory",
    }
    save_json(root / "structural_audit.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
