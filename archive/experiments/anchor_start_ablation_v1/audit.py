"""Recompute the published control metrics without training or local states."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.paper_structure_benchmark_v1.reporting.evaluate import (
    align_index,
    write_csv,
)
from experiments.sharing_structure_research_v1.blind_contract import digest


def audit(base, output):
    """Verify the compact publication and reconstruct its primary metrics."""
    base = base.resolve()
    manifest = json.loads((base / "published_manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        path = (base / name).resolve()
        if not path.is_relative_to(base) or digest(path) != expected:
            raise RuntimeError(f"Changed publication file: {name}")
    sizes = json.loads((base / "case_sizes.json").read_text())
    tasks = json.loads((base / "target_map.json").read_text())
    expected = json.loads((base / "summary.json").read_text())["rows"]
    references = {
        (
            "general" if row["family"].startswith("general_") else "standard",
            row["case"],
            int(row["source_noise"] > 0),
            row["method"],
        ): row
        for row in expected
    }
    recovery, transfer = [], []
    with (
        np.load(base / "selected_structures.npz") as structures,
        np.load(base / "scoring_answers.npz") as answers,
    ):
        for item in tasks.values():
            cohort, case, condition, method = (
                item[key] for key in ("cohort", "case", "condition", "method")
            )
            reference = references[cohort, case, condition, method]
            prefix = f"{cohort}_{case}_c{condition}_{method}"
            _, metrics = align_index(
                structures[prefix + "_index"],
                structures[prefix + "_truth"],
                sizes[f"{cohort}/{case}"]["codes"],
            )
            prediction_path = (
                base / "predictions" / cohort / f"{item['cache']}.npz"
            )
            with np.load(prediction_path) as predictions:
                losses = (
                    (
                        predictions["prediction"]
                        - answers[f"{cohort}_{case}_query_y"]
                    )
                    ** 2
                ).mean((1, 2))
            if (
                metrics["signed_entry_errors"]
                != reference["signed_entry_errors"]
                or metrics["mask_exact"] != reference["mask_exact"]
                or not np.isclose(
                    losses.mean(),
                    reference["target_mse_1024"],
                    rtol=1e-12,
                    atol=1e-15,
                )
            ):
                raise RuntimeError(f"Published metric differs: {prefix}")
            label = {
                "family": reference["family"],
                "case": case,
                "source_noise": reference["source_noise"],
                "method": method,
            }
            recovery.append(
                {**label, **metrics, "target_mse_1024": float(losses.mean())}
            )
            transfer.extend(
                {**label, "task": task, "mse": float(value)}
                for task, value in enumerate(losses)
            )
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "recovery.csv", recovery)
    write_csv(output / "transfer_per_task.csv", transfer)
    counts = Counter(row["method"] for row in recovery if row["exact"])
    result = {
        "source_selections": len(recovery),
        "target_rows": len(transfer),
        "exact_by_method": dict(counts),
        "published_metrics_reproduced": True,
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.base, arguments.output)))


if __name__ == "__main__":
    main()
