"""Summarize and audit the independent identical-SGD follow-up."""

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.basin_release_v1.report import paired_interval
from experiments.basin_release_v1.run import ROOT


def main():
    protocol = json.loads((ROOT / "sgd_protocol.json").read_text())
    bootstrap = json.loads((ROOT / "protocol.json").read_text())
    output = Path(protocol["output"])
    reports = {}
    for frequency in protocol["frequencies"]:
        run = output / f"frequency_{frequency}"
        complete = json.loads((run / "complete.json").read_text())
        hashes = json.loads((run / "frozen_before_target.json").read_text())
        inputs_match = all(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
            for path, digest in hashes.items()
        )
        if not complete["frozen_inputs_unchanged"] or not inputs_match:
            raise RuntimeError("Follow-up inputs changed")
        teacher = np.load(run / "teacher_diagnostic.npz")
        known = teacher["families"] < 4
        success, metrics = {}, {}
        for name in complete["outcomes"]:
            timing = json.loads((run / name / "timing.json").read_text())
            saved = np.load(run / name / "selected.npz")
            if timing["steps"] * timing["starts"] != 65536:
                raise RuntimeError("Unequal target budget")
            if timing["configs"][0] != complete["config"]:
                raise RuntimeError("Optimizer settings differ")
            if timing["training_seed"] != protocol["fit_seed"] + 100000:
                raise RuntimeError("Training data seed differs")
            if timing["query_seed"] != protocol["query_seed"]:
                raise RuntimeError("Query data seed differs")
            success[name] = (
                saved["population"][known] < protocol["success_mse"]
            ).astype(float)
            metrics[name] = {
                **complete["outcomes"][name],
                "known_mean_population_mse": float(
                    saved["population"][known].mean()
                ),
                "numeric_seconds": timing["seconds"],
            }
        differences = np.mean(
            [
                success[f"tied_ar_{seed}"] - success[f"released_ar_{seed}"]
                for seed in protocol["prior_seeds"]
            ],
            axis=0,
        )
        reports[frequency] = {
            "config": complete["config"],
            "metrics": metrics,
            "tied_minus_released": paired_interval(
                differences, teacher["families"][known], bootstrap
            ),
            "matching": complete["matching"],
            "frozen_inputs_match": inputs_match,
            "budgets_and_settings_match": True,
            "known_tasks": int(known.sum()),
            "foreign_tasks": int((~known).sum()),
        }
    (output / "summary.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
