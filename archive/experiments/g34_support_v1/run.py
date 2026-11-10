"""Register and execute a bounded support ablation without changing G34."""

import argparse
import json
import resource
import shutil
import subprocess
import sys
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
SOURCE_METHODS = ("free", "full")
TARGET_METHODS = (*SOURCE_METHODS, "learned_zero", "mask_only")
TARGET_CODE = PROJECT / "experiments/paper_structure_benchmark_v1/fitting"
FLEET = "experiments.paper_structure_benchmark_v1.execution.fleet"


def setup(base, reference):
    base.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__).with_name("PROTOCOL.md"), base / "PROTOCOL.md")
    code = [
        *(PROJECT / "experiments/sharing_structure_research_v1").glob("blind_*.py"),
        *(Path(__file__).parent).glob("*.py"),
        *(TARGET_CODE / name for name in (
            "target_fit.py", "adaptation.py", "numerical.py"
        )),
        PROJECT / "experiments/paper_structure_benchmark_v1/execution/fleet.py",
    ]
    registration = {
        "created_utc": datetime.now(UTC).isoformat(),
        "reference": str(reference),
        "reference_freeze_sha256": digest(
            reference.parent / "report/frozen_results.json"
        ),
        "protocol_sha256": digest(base / "PROTOCOL.md"),
        "code_sha256": {str(p.relative_to(PROJECT)): digest(p) for p in code},
        "allowed_gpus": list(range(8)),
        "prior_results_known": True,
        "new_target_answers_used_to_tune": False,
        "primary": "Adam+GN clean target MSE, N=1024, by geometry/noise",
    }
    save_json(base / "registration.json", registration)
    for path in code:
        target = base / "code_snapshot" / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    protocol = json.loads((reference / "protocol.json").read_text())
    protocol["allowed_gpus"], protocol["forbidden_gpus"] = list(range(8)), []
    jobs = []
    for method in TARGET_METHODS:
        root = base / method
        for name in (
            "public", "source", "fits", "metrics", "logs", "receipts", "targets"
        ):
            (root / name).mkdir(parents=True)
        save_json(root / "registration.json", registration)
        save_json(root / "protocol.json", protocol)
        if method not in SOURCE_METHODS:
            continue
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
                jobs.append({
                    "id": f"{key}_{method}",
                    "module": "experiments.g34_support_v1.run",
                    "args": [
                        "--base", str(root), "--phase", "source",
                        "--case", case, "--condition", str(condition),
                        "--method", method,
                    ],
                    "registration": str(root / "fits" / key / "started.json"),
                    "metric": str(root / "metrics" / f"{key}_pipeline.json"),
                    "log": str(root / "logs" / f"{key}_pipeline.log"),
                })
            for path in paths:
                target = root / path.relative_to(reference)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied[str(path.relative_to(reference))] = digest(target)
        save_json(root / "source_export.json", {
            "reference": str(reference), "teacher_access": False,
            "copied_sha256": copied,
        })
    # Interleave methods and cases instead of occupying all GPUs with one mode.
    jobs.sort(key=lambda row: (
        row["id"].split("_c")[0],
        row["id"].split("_c")[1].split("_")[0], row["id"]
    ))
    save_json(base / "source_plan.json", {"jobs": jobs})
    print(json.dumps({"source_jobs": len(jobs), "base": str(base)}))


def source(root, case, condition, method):
    torch.set_num_threads(2)
    registration = json.loads((root / "registration.json").read_text())
    for name, expected in registration["code_sha256"].items():
        if "/g34_support_v1/" in name or "/sharing_structure_research_v1/" in name:
            if digest(PROJECT / name) != expected:
                raise ValueError(f"Registered learner changed: {name}")
    boundary = ObservationBoundary(root, case, condition)
    boundary.install()
    key = f"{case}_c{condition}"
    save_json(root / "fits" / key / "started.json", {
        "utc": datetime.now(UTC).isoformat(), "method": method,
    })
    device = torch.device("cuda:0")
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started, stages, status = time.perf_counter(), {}, "failed"
    counter = FlopCounterMode(display=False)
    try:
        for phase in ("proposals", "fit", "select"):
            boundary.phase = phase
            boundary.role = "select" if phase == "select" else "train"
            tick = time.perf_counter()
            if phase == "proposals":
                bank.propose(root, case, condition, method)
            elif phase == "fit":
                with counter:
                    blind_joint_bank.fit(root, case, condition, device)
            else:
                bank.select(root, case, condition, method)
            torch.cuda.synchronize(device)
            stages[phase] = time.perf_counter() - tick
        status = "complete"
    finally:
        save_json(root / "metrics" / f"{key}_pipeline.json", {
            "status": status, "method": method,
            "wall_seconds": time.perf_counter() - started,
            "stage_seconds": stages,
            "torch_operator_flops": counter.get_total_flops(),
            "flop_scope": "Functional refit only; NumPy/SciPy geometry excluded",
            "cpu_peak_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss * 1024,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        })
        boundary.save(root / "receipts" / f"{key}.json", status)


def export_targets(base):
    registration = json.loads((base / "registration.json").read_text())
    reference = Path(registration["reference"])
    protocol = json.loads((reference / "protocol.json").read_text())
    jobs = []
    for method in TARGET_METHODS:
        root = base / method
        for sizes in protocol["cases"]:
            case = sizes["name"]
            for condition in protocol["conditions"]:
                key = f"{case}_c{condition}"
                origin = root if method in SOURCE_METHODS else reference
                selected = origin / "fits" / key / "joint_selected.npz"
                index = np.load(selected)["joint_functional_index"].copy()
                new_sizes = {
                    name: sizes[name]
                    for name in ("inputs", "hidden", "codes", "outputs")
                }
                if method == "learned_zero":
                    index[index == 0] = sizes["codes"] + 1
                    new_sizes["codes"] += 1
                elif method == "mask_only":
                    mask = index != 0
                    index = np.zeros_like(index)
                    index[mask] = np.arange(1, int(mask.sum()) + 1)
                    new_sizes["codes"] = int(mask.sum())
                folder = root / "targets" / key
                folder.mkdir(exist_ok=False)
                np.savez_compressed(folder / "structures.npz", **{method: index})
                for name in ("support.npz", "selection.npz", "query_x.npz"):
                    shutil.copy2(reference / "targets" / key / name, folder / name)
                save_json(folder / "sizes.json", new_sizes)
                save_json(folder / "ready.json", {
                    "utc": datetime.now(UTC).isoformat(),
                    "source_selected": str(selected),
                    "source_sha256": digest(selected),
                    "method": method,
                    "exported_sha256": {
                        p.name: digest(p) for p in folder.glob("*.npz")
                    },
                    "fresh_numeric_initialization": True,
                    "target_query_answers_exported": False,
                    "teacher_structure_accessed": False,
                })
                jobs.append({
                    "id": f"{key}_{method}",
                    "module": (
                        "experiments.paper_structure_benchmark_v1.fitting.target_fit"
                    ),
                    "args": [
                        "--root", str(root), "--case", case,
                        "--condition", str(condition), "--method", method,
                    ],
                    "registration": str(folder / method / "registered.json"),
                    "metric": str(folder / method / "metrics.json"),
                    "dependencies": [str(folder / "ready.json")],
                    "log": str(root / "logs" / f"{key}_target.log"),
                })
    save_json(base / "target_plan.json", {"jobs": jobs})
    print(json.dumps({"target_jobs": len(jobs)}))


def complete(base):
    for kind in ("source", "target"):
        if kind == "target":
            export_targets(base)
        subprocess.run([
            sys.executable, "-m", FLEET, "--plan",
            str(base / f"{kind}_plan.json"), "--per-gpu", "1",
            "--gpus", *map(str, range(8)),
        ], check=True)
        state = json.loads((base / f"{kind}_plan.finished.json").read_text())
        if state["completed"] != state["total"] or state["failures"]:
            raise RuntimeError(f"Incomplete {kind} plan: {state['failures']}")
    for phase in ("freeze", "score", "render"):
        subprocess.run([
            sys.executable, "-m", "experiments.g34_support_v1.report",
            "--base", str(base), "--phase", phase,
        ], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument("--method", choices=SOURCE_METHODS)
    parser.add_argument(
        "--phase", choices=("setup", "source", "export", "complete"), required=True
    )
    args = parser.parse_args()
    if args.phase == "setup":
        setup(args.base.resolve(), args.reference.resolve())
    elif args.phase == "source":
        source(args.base.resolve(), args.case, args.condition, args.method)
    elif args.phase == "export":
        export_targets(args.base.resolve())
    else:
        complete(args.base.resolve())


if __name__ == "__main__":
    main()
