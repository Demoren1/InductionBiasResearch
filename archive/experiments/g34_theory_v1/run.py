"""Prepare and schedule isolated comparisons against the immutable G34."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)


PROJECT = Path(__file__).resolve().parents[2]
REFERENCE = "experiments.paper_structure_benchmark_v1"


def baseline(root, case, condition):
    folder = root / "logs" / f"{case}_c{condition}"
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for phase in ("source", "seeds", "proposals", "fit", "select"):
        with (folder / f"baseline_{phase}.log").open("w") as stream:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    f"{REFERENCE}.current_method",
                    "--root",
                    str(root),
                    "--case",
                    case,
                    "--condition",
                    str(condition),
                    "--phase",
                    phase,
                    "--device",
                    "cuda:0",
                ],
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        print(
            json.dumps({"case": case, "condition": condition, "phase": phase})
        )
    save_json(
        root / "metrics" / f"{case}_c{condition}_baseline.json",
        {"status": "complete", "seconds": time.perf_counter() - started},
    )


def setup(base):
    from experiments.paper_structure_benchmark_v1.protocol.prepare import (
        prepare,
    )

    base.mkdir(parents=True, exist_ok=False)
    confirmation = base / "confirmation"
    prepare(confirmation, "confirmation", 3, 16, seed=27110854013)
    shutil.copy2(Path(__file__).with_name("PROTOCOL.md"), base / "PROTOCOL.md")
    development = base / "development"
    for folder in ("public", "source", "fits", "metrics", "receipts", "logs"):
        (development / folder).mkdir(parents=True, exist_ok=True)
    original = (
        PROJECT
        / "experiments/paper_structure_benchmark_v1/results/20261107"
        / "synthetic_confirmation"
    )
    copied = {}
    for case in ("case_00", "case_03"):
        paths = [original / "public" / f"{case}.json"]
        paths += list((original / "public" / f"{case}_c1").glob("*.npz"))
        paths = [p for p in paths if p.name != "test.npz"]
        paths += list((original / "source").glob(f"{case}_c1_t*/learned.npz"))
        paths += list((original / "source").glob(f"{case}_c1_t*/trace.json"))
        paths += [
            original / "fits" / f"{case}_c1" / name
            for name in (
                "fitted.npz",
                "joint_bank.npz",
                "joint_bank.json",
                "joint_fitted.npz",
                "joint_selected.npz",
                "joint_selected.json",
            )
        ]
        for path in paths:
            target = development / path.relative_to(original)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied[str(target.relative_to(development))] = digest(target)
    save_json(
        development / "source_export.json",
        {
            "origin": str(original),
            "role": "Retrospective development; source observations only",
            "sha256": copied,
            "teacher_access": False,
        },
    )
    jobs = []
    for case in range(6):
        for condition in (0, 1):
            name = f"case_{case:02d}"
            key = f"{name}_c{condition}"
            jobs.append(
                {
                    "id": key,
                    "module": "experiments.g34_theory_v1.run",
                    "args": [
                        "--base",
                        str(confirmation),
                        "--case",
                        name,
                        "--condition",
                        str(condition),
                        "--method",
                        "baseline",
                    ],
                    "metric": str(
                        confirmation / "metrics" / f"{key}_baseline.json"
                    ),
                    "registration": str(
                        confirmation / "dense" / key / "fitted.npz"
                    ),
                    "log": str(confirmation / "logs" / f"{key}_baseline.log"),
                }
            )
    save_json(base / "baseline_plan.json", {"jobs": jobs})
    save_json(
        base / "registered_protocol.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(base / "PROTOCOL.md"),
            "confirmation_seed": 27110854013,
            "development_source_export_sha256": digest(
                development / "source_export.json"
            ),
            "core_sha256": {
                str(p.relative_to(PROJECT)): digest(p)
                for p in (
                    PROJECT / "experiments/sharing_structure_research_v1"
                ).glob("blind_*.py")
            },
            "allowed_gpus": list(range(6)),
            "primary": "paired target MSE at N=1024",
            "target_answers_opened": False,
        },
    )


def variants(root, case, condition, method):
    phases = ("proposals",) if method == "proposals" else ("fit", "select")
    folder = root / "logs" / f"{case}_c{condition}"
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for phase in phases:
        with (folder / f"{method}_{phase}.log").open("w") as stream:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.g34_theory_v1.variant",
                    "--root",
                    str(root),
                    "--case",
                    case,
                    "--condition",
                    str(condition),
                    "--method",
                    "theory" if method == "proposals" else method,
                    "--phase",
                    phase,
                    "--device",
                    "cuda:0",
                ],
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
    save_json(
        root / "metrics" / f"{case}_c{condition}_{method}_pipeline.json",
        {"status": "complete", "seconds": time.perf_counter() - started},
    )


def plan_variants(base, development):
    root = base / ("development" if development else "confirmation")
    cases = (
        ("case_00", "case_03")
        if development
        else (f"case_{n:02d}" for n in range(6))
    )
    conditions = (1,) if development else (0, 1)
    jobs = []
    for case in cases:
        for condition in conditions:
            key = f"{case}_c{condition}"
            for method in (() if development else ("proposals",)) + (
                "theory",
                "diversity",
            ):
                if method == "proposals":
                    dependencies = [root / "metrics" / f"{key}_baseline.json"]
                    registration = root / "fits" / key / "theory_pool.npz"
                else:
                    dependencies = [root / "fits" / key / method / "bank.json"]
                    registration = root / "fits" / key / method / "fitted.npz"
                jobs.append(
                    {
                        "id": f"{key}_{method}",
                        "module": "experiments.g34_theory_v1.run",
                        "args": [
                            "--base",
                            str(root),
                            "--case",
                            case,
                            "--condition",
                            str(condition),
                            "--method",
                            method,
                        ],
                        "dependencies": [str(p) for p in dependencies],
                        "registration": str(registration),
                        "metric": str(
                            root / "metrics" / f"{key}_{method}_pipeline.json"
                        ),
                        "log": str(root / "logs" / f"{key}_{method}.log"),
                    }
                )
    name = "development_plan.json" if development else "variants_plan.json"
    save_json(base / name, {"jobs": jobs})


def register(base):
    root = base / "confirmation"
    path = root / "theory_registration.json"
    if path.exists():
        raise RuntimeError("Confirmation method already registered")
    development = base / "development"
    summaries = {}
    for case in ("case_00", "case_03"):
        folder = development / "fits" / f"{case}_c1"
        summaries[case] = {
            "current": json.loads(
                (folder / "joint_selected.json").read_text()
            )["methods"]["joint_functional"],
            **{
                method: json.loads(
                    (folder / method / "selected.json").read_text()
                )
                for method in ("theory", "diversity")
            },
        }
    files = [
        Path(__file__).with_name(name) for name in ("bank.py", "variant.py")
    ]
    save_json(
        path,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "code_sha256": {p.name: digest(p) for p in files},
            "development_source_selection": summaries,
            "refit_limit": 32,
            "quotas": {
                "original": 8,
                "relations": 12,
                "log_positive": 8,
                "signs": 4,
            },
            "primary": "theory vs current at target N=1024, no post-query tuning",
            "confirmation_outputs_used_to_tune": False,
        },
    )
    snapshots = root / "theory_code_snapshot"
    snapshots.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, snapshots / path.name)


def export_targets(base):
    import numpy as np

    root = base / "confirmation"
    protocol = json.loads((root / "protocol.json").read_text())
    for sizes in protocol["cases"]:
        case = sizes["name"]
        for condition in protocol["conditions"]:
            key = f"{case}_c{condition}"
            source = root / "fits" / key
            folder = root / "targets" / key
            folder.mkdir(exist_ok=False)
            structure_paths = {
                "current": source / "joint_selected.npz",
                "theory": source / "theory" / "selected.npz",
                "diversity": source / "diversity" / "selected.npz",
            }
            structures = {
                name: np.load(path)[
                    "joint_functional_index" if name == "current" else "index"
                ]
                for name, path in structure_paths.items()
            }
            np.savez_compressed(folder / "structures.npz", **structures)
            with np.load(root / "private" / f"{case}_targets.npz") as archive:
                for panel in ("support", "selection"):
                    np.savez_compressed(
                        folder / f"{panel}.npz",
                        x=np.stack(
                            [archive[f"{panel}_x_{t}"] for t in range(8)]
                        ),
                        y=np.stack(
                            [archive[f"{panel}_y_{t}"] for t in range(8)]
                        ),
                    )
                np.savez_compressed(
                    folder / "query_x.npz",
                    x=np.stack([archive[f"query_x_{t}"] for t in range(8)]),
                )
            save_json(
                folder / "sizes.json",
                {
                    k: sizes[k]
                    for k in ("inputs", "hidden", "codes", "outputs")
                },
            )
            save_json(
                folder / "ready.json",
                {
                    "source_frozen_utc": datetime.now(UTC).isoformat(),
                    "source_sha256": {
                        method: digest(p)
                        for method, p in structure_paths.items()
                    },
                    "inputs_sha256": {
                        p.name: digest(p) for p in folder.glob("*.npz")
                    },
                    "target_query_answers_exported": False,
                    "structural_truth_accessed": False,
                },
            )
    jobs = []
    for case in protocol["cases"]:
        for condition in protocol["conditions"]:
            key = f"{case['name']}_c{condition}"
            for method in ("current", "theory", "diversity") + (
                ("dense",) if condition == 0 else ()
            ):
                folder = root / "targets" / key / method
                jobs.append(
                    {
                        "id": f"{key}_{method}",
                        "module": f"{REFERENCE}.fitting.target_fit",
                        "args": [
                            "--root",
                            str(root),
                            "--case",
                            case["name"],
                            "--condition",
                            str(condition),
                            "--method",
                            method,
                        ],
                        "registration": str(folder / "registered.json"),
                        "metric": str(folder / "metrics.json"),
                        "dependencies": [
                            str(root / "targets" / key / "ready.json")
                        ],
                        "log": str(
                            root / "logs" / f"{key}_{method}_target.log"
                        ),
                    }
                )
    save_json(base / "target_plan.json", {"jobs": jobs})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument(
        "--method",
        choices=(
            "setup",
            "baseline",
            "proposals",
            "theory",
            "diversity",
            "plan-development",
            "plan-confirmation",
            "register",
            "export",
        ),
        required=True,
    )
    args = parser.parse_args()
    if args.method == "setup":
        setup(args.base.resolve())
    elif args.method == "baseline":
        baseline(args.base.resolve(), args.case, args.condition)
    elif args.method.startswith("plan-"):
        plan_variants(args.base.resolve(), args.method == "plan-development")
    elif args.method == "register":
        register(args.base.resolve())
    elif args.method == "export":
        export_targets(args.base.resolve())
    else:
        variants(args.base.resolve(), args.case, args.condition, args.method)


if __name__ == "__main__":
    main()
