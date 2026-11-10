"""Register and execute the paired robust-calibration ablation."""

import argparse
import json
import os
import resource
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from experiments.sharing_structure_research_v1 import blind_joint_bank
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)
from experiments.sharing_structure_research_v1.blind_runtime import (
    ObservationBoundary,
)

from . import bank


PROJECT = Path(__file__).resolve().parents[2]
METHODS = ("guard", "irls")


def setup(base, reference):
    base.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__).with_name("PROTOCOL.md"), base / "PROTOCOL.md")
    core_paths = [
        *(PROJECT / "experiments/sharing_structure_research_v1").glob(
            "blind_*.py"
        ),
        PROJECT / "experiments/paper_structure_benchmark_v1/current_method.py",
    ]
    learner_paths = [Path(__file__), Path(__file__).with_name("bank.py")]
    registration = {
        "created_utc": datetime.now(UTC).isoformat(),
        "reference": str(reference),
        "reference_freeze_sha256": digest(
            reference.parent / "report/frozen_results.json"
        ),
        "protocol_sha256": digest(base / "PROTOCOL.md"),
        "core_sha256": {
            str(path.relative_to(PROJECT)): digest(path) for path in core_paths
        },
        "learner_sha256": {
            str(path.relative_to(PROJECT)): digest(path)
            for path in learner_paths
        },
        "primary": "paired target MSE, Adam+GN, N=1024, by geometry and noise",
        "retrospective_data": True,
        "prior_results_known": True,
        "new_target_results_used_to_tune": False,
        "allowed_gpus": list(range(6)),
        "forbidden_gpus": [6, 7],
    }
    save_json(base / "registration.json", registration)
    for path in [*core_paths, *learner_paths]:
        target = base / "code_snapshot" / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    protocol = json.loads((reference / "protocol.json").read_text())
    jobs = []
    for method in METHODS:
        root = base / method
        for name in (
            "public", "source", "fits", "metrics", "logs", "receipts", "targets"
        ):
            (root / name).mkdir(parents=True)
        save_json(root / "registration.json", registration)
        save_json(root / "protocol.json", protocol)
        copied = {}
        for sizes in protocol["cases"]:
            case = sizes["name"]
            paths = [reference / "public" / f"{case}.json"]
            for condition in protocol["conditions"]:
                key = f"{case}_c{condition}"
                paths.extend(
                    reference / "public" / key / f"{fold}.npz"
                    for fold in ("fixed_train", "selection", "meta")
                )
                paths.extend(
                    reference / "source" / f"{key}_t{task}" / name
                    for task in range(protocol["sources"])
                    for name in ("learned.npz", "trace.json")
                )
                paths.append(reference / "fits" / key / "fitted.npz")
                jobs.append(
                    {
                        "id": f"{key}_{method}",
                        "module": "experiments.g34_robust_v1.run",
                        "args": [
                            "--base", str(root),
                            "--case", case,
                            "--condition", str(condition),
                            "--method", method,
                            "--phase", "source",
                        ],
                        "registration": str(
                            root / "fits" / key / "started.json"
                        ),
                        "metric": str(
                            root / "metrics" / f"{key}_pipeline.json"
                        ),
                        "log": str(root / "logs" / f"{key}_pipeline.log"),
                    }
                )
            for path in paths:
                relative = path.relative_to(reference)
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied[str(relative)] = digest(target)
        save_json(
            root / "source_export.json",
            {
                "reference": str(reference),
                "teacher_access": False,
                "copied_sha256": copied,
            },
        )
    save_json(base / "source_plan.json", {"jobs": jobs})
    print(json.dumps({"registered": str(base), "jobs": len(jobs)}))


def source(root, case, condition, method):
    torch.set_num_threads(2)
    registration = json.loads((root / "registration.json").read_text())
    for name, expected in {
        **registration["core_sha256"],
        **registration["learner_sha256"],
    }.items():
        if digest(PROJECT / name) != expected:
            raise ValueError(f"Registered learner changed: {name}")
    boundary = ObservationBoundary(root, case, condition)
    boundary.install()
    key = f"{case}_c{condition}"
    save_json(
        root / "fits" / key / "started.json",
        {
            "utc": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "method": method,
        },
    )
    device = torch.device("cuda:0")
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    status, stages = "failed", {}
    counter = FlopCounterMode(display=False)
    try:
        for phase in ("proposals", "fit", "select"):
            boundary.phase = phase
            boundary.role = "select" if phase == "select" else "train"
            tick = time.perf_counter()
            if phase == "proposals":
                bank.propose(root, case, condition, robust=method == "irls")
            elif phase == "fit":
                with counter:
                    blind_joint_bank.fit(root, case, condition, device)
            else:
                blind_joint_bank.select(root, case, condition)
            torch.cuda.synchronize(device)
            stages[phase] = time.perf_counter() - tick
        status = "complete"
    finally:
        save_json(
            root / "metrics" / f"{key}_pipeline.json",
            {
                "status": status,
                "method": method,
                "wall_seconds": time.perf_counter() - started,
                "stage_seconds": stages,
                "torch_operator_flops": counter.get_total_flops(),
                "flop_scope": (
                    "Functional refit only; excludes NumPy/SciPy geometry"
                ),
                "cpu_peak_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss * 1024,
                "cuda_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device)
                ),
            },
        )
        boundary.save(root / "receipts" / f"{key}.json", status)


def export_targets(base):
    registration = json.loads((base / "registration.json").read_text())
    reference = Path(registration["reference"])
    jobs = []
    for method in METHODS:
        root = base / method
        protocol = json.loads((root / "protocol.json").read_text())
        for sizes in protocol["cases"]:
            case = sizes["name"]
            for condition in protocol["conditions"]:
                key = f"{case}_c{condition}"
                selected = root / "fits" / key / "joint_selected.npz"
                folder = root / "targets" / key
                folder.mkdir(exist_ok=False)
                index = np.load(selected)["joint_functional_index"]
                np.savez_compressed(
                    folder / "structures.npz", **{method: index}
                )
                # These are already exported observed target data; no private
                # archive or query responses are accessed by this operation.
                for name in (
                    "support.npz", "selection.npz", "query_x.npz", "sizes.json"
                ):
                    shutil.copy2(
                        reference / "targets" / key / name, folder / name
                    )
                save_json(
                    folder / "ready.json",
                    {
                        "utc": datetime.now(UTC).isoformat(),
                        "source_sha256": digest(selected),
                        "inputs_sha256": {
                            path.name: digest(path)
                            for path in folder.glob("*.npz")
                        },
                        "target_query_answers_exported": False,
                        "structural_truth_accessed": False,
                    },
                )
                jobs.append(
                    {
                        "id": f"{key}_{method}",
                        "module": (
                            "experiments.paper_structure_benchmark_v1"
                            ".fitting.target_fit"
                        ),
                        "args": [
                            "--root", str(root),
                            "--case", case,
                            "--condition", str(condition),
                            "--method", method,
                        ],
                        "registration": str(
                            folder / method / "registered.json"
                        ),
                        "metric": str(folder / method / "metrics.json"),
                        "dependencies": [str(folder / "ready.json")],
                        "log": str(root / "logs" / f"{key}_target.log"),
                    }
                )
    save_json(base / "target_plan.json", {"jobs": jobs})
    print(json.dumps({"target_jobs": len(jobs)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument(
        "--phase", choices=("setup", "source", "export"), required=True
    )
    args = parser.parse_args()
    if args.phase == "setup":
        setup(args.base.resolve(), args.reference.resolve())
    elif args.phase == "export":
        export_targets(args.base.resolve())
    else:
        source(args.base.resolve(), args.case, args.condition, args.method)


if __name__ == "__main__":
    main()
