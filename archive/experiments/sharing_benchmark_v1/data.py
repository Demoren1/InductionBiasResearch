"""Observation-only task cohorts and separately stored teacher audit states."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.darts_structured_tasks_v1.data import digest, save_json

BASE = Path(__file__).resolve().parent


def task_cohort(rng, index, tasks, support, query, noise):
    inputs = index.shape[1]
    codes = int(index.max())
    coefficient = rng.normal(size=(tasks, codes)).astype(np.float32)
    weights = np.where(index[None] == 0, 0, coefficient[:, index - 1])
    x = rng.normal(size=(tasks, support + query, inputs)).astype(np.float32)
    y = np.einsum("tni,toi->tno", x, weights)
    y += noise * rng.normal(size=y.shape).astype(np.float32)
    return {
        "support_x": x[:, :support],
        "support_y": y[:, :support],
        "query_x": x[:, support:],
        "query_y": y[:, support:],
    }


def prepare_family(root, plan, profile, family):
    spec = {**plan[profile], "profile": profile, "family": family}
    output = root / profile / f"family_{family}"
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(plan["seed"] + family * 100003)
    index = np.zeros((spec["outputs"], spec["inputs"]), dtype=np.int16)
    for position in range(spec["outputs"]):
        index[position, position : position + spec["codes"]] = np.arange(
            1, spec["codes"] + 1
        )
    input_order = np.arange(spec["inputs"])
    output_order = np.arange(spec["outputs"])
    if spec["hidden_permutations"]:
        input_order = rng.permutation(input_order)
        output_order = rng.permutation(output_order)
        index = index[output_order][:, input_order]
    np.savez_compressed(
        output / "teacher_audit_only.npz",
        indices=index,
        input_order=input_order,
        output_order=output_order,
    )
    observed = {}
    for prefix, tasks, support, query in (
        ("train", spec["train_tasks"], spec["train_support"], spec["train_query"]),
        (
            "selection",
            spec["selection_tasks"],
            spec["selection_support"],
            spec["selection_query"],
        ),
    ):
        cohort = task_cohort(rng, index, tasks, support, query, spec["noise"])
        observed.update({f"{prefix}_{name}": value for name, value in cohort.items()})
    np.savez_compressed(output / "observations.npz", **observed)
    save_json(output / "specification.json", spec)
    save_json(
        output / "prepared.json",
        {
            "sha256": {
                name: digest(output / name)
                for name in (
                    "observations.npz",
                    "specification.json",
                    "teacher_audit_only.npz",
                )
            }
        },
    )


def prepare(root):
    plan = json.loads((BASE / "protocol.json").read_text())
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / "protocol.json", plan)
    for family in plan["pilot_families"]:
        prepare_family(root, plan, "hidden", family)
    prepare_family(root, plan, "native", 301)
    snapshot = root / "code_snapshot"
    shutil.copytree(
        BASE, snapshot, ignore=shutil.ignore_patterns("results", "__pycache__")
    )
    save_json(
        root / "registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(root / "protocol.json"),
            "source_sha256": {
                str(path.relative_to(snapshot)): digest(path)
                for path in snapshot.rglob("*")
                if path.is_file()
            },
            "reference_sources": {
                "MSR": "https://github.com/AllanYangZhou/metalearning-symmetries",
                "Yeh": "https://github.com/raymondyeh07/equivariance_discovery",
                "LC": "https://arxiv.org/html/1707.04319",
                "LUT-Q": "https://arxiv.org/html/1911.04951",
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    prepare(args.root)


if __name__ == "__main__":
    main()
