"""Register a matched finer lambda path before claiming a weight improvement."""

import argparse
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.darts_structured_tasks_v1.data import digest, save_json

BASE = Path(__file__).resolve().parent


def register(root, source):
    parent = json.loads((source / "protocol.json").read_text())
    choices = json.loads((source / "cohort_frozen.json").read_text())["choices"]
    raw = Path(parent["raw_control"])
    raw_choice = json.loads((raw / "hpo_frozen.json").read_text())["settings"]
    settings = {"raw": raw_choice, **{m: c["settings"] for m, c in choices.items()}}
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        **{
            k: parent[k]
            for k in (
                "seed",
                "inputs",
                "outputs",
                "tasks",
                "codes",
                "source_root",
                "g4_root",
                "pilot_source_jobs",
                "solver",
            )
        },
        "stage": "Matched lambda-resolution diagnostic on already-opened pilot; no new parameter-metric HPO or independent confirmation",
        "registered_utc": datetime.now(UTC).isoformat(),
        "weight_metrics": list(settings),
        "rows": {m: [s] for m, s in settings.items()},
        "lambdas": np.unique(np.r_[1e-5, 1e-4, np.geomspace(0.001, 10, 41)]).tolist(),
        "observations": [8192],
        "source_pilot_sha256": digest(source / "cohort_frozen.json"),
        "raw_hpo_sha256": digest(raw / "hpo_frozen.json"),
        "selection": "Keep all gamma/zero-ratio/bandwidth choices from earlier source-only freezes. Only densify the same lambda range; select by same independent source-validation functions. Freeze all16 outputs before scoring true I.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": {
            name: digest(BASE / name)
            for name in (
                "fusion_lambda_resolution.py",
                "calibrated_fusion_pilot.py",
                "joint_fusion.py",
                "fusion_weights.py",
                "competitor_models.py",
                "competitor_pilot.py",
                "gaussian_stats.py",
            )
        },
        "checks": "No software tests; actual optimization-resolution experiment and minimum lint only.",
    }
    plan["jobs"] = [
        {"source_job": job, "n": 8192, "weight_metric": metric}
        for metric, job in itertools.product(
            plan["weight_metrics"], plan["pilot_source_jobs"]
        )
    ]
    save_json(root / "protocol.json", plan)
    shutil.copy2(BASE / "THEORY.md", root / "theory_at_registration.md")
    snapshot = root / "executed_code_snapshot"
    snapshot.mkdir()
    for name in plan["code_sha256"]:
        shutil.copy2(BASE / name, snapshot / name)
    (root / "pilot").mkdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    register(args.root, args.source)


if __name__ == "__main__":
    main()
