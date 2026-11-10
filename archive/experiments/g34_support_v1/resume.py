"""Continue only the zero-preserving mask ablation after user clarification."""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .run import FLEET, PROJECT, setup


def export_targets(base):
    reference = Path(
        json.loads((base / "registration.json").read_text())["reference"]
    )
    protocol = json.loads((reference / "protocol.json").read_text())
    root, jobs = base / "free", []
    for sizes in protocol["cases"]:
        case = sizes["name"]
        for condition in protocol["conditions"]:
            key = f"{case}_c{condition}"
            selected = root / "fits" / key / "joint_selected.npz"
            index = np.load(selected)["joint_functional_index"]
            folder = root / "targets" / key
            folder.mkdir(exist_ok=False)
            np.savez_compressed(folder / "structures.npz", free=index)
            for name in (
                "support.npz", "selection.npz", "query_x.npz", "sizes.json"
            ):
                shutil.copy2(reference / "targets" / key / name, folder / name)
            save_json(folder / "ready.json", {
                "utc": datetime.now(UTC).isoformat(),
                "source_sha256": digest(selected),
                "exported_sha256": {
                    p.name: digest(p) for p in folder.glob("*.npz")
                },
                "zero_symbol_retained": True,
                "fresh_numeric_initialization": True,
                "target_query_answers_exported": False,
                "teacher_structure_accessed": False,
            })
            jobs.append({
                "id": f"{key}_free",
                "module": (
                    "experiments.paper_structure_benchmark_v1.fitting.target_fit"
                ),
                "args": [
                    "--root", str(root), "--case", case,
                    "--condition", str(condition), "--method", "free",
                ],
                "registration": str(folder / "free/registered.json"),
                "metric": str(folder / "free/metrics.json"),
                "dependencies": [str(folder / "ready.json")],
                "log": str(root / "logs" / f"{key}_target.log"),
            })
    save_json(base / "target_plan.json", {"jobs": jobs})


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
            raise RuntimeError(f"Incomplete {kind}: {state['failures']}")
    for phase in ("freeze", "score", "render"):
        subprocess.run([
            sys.executable, "-m", "experiments.g34_support_v1.report",
            "--base", str(base), "--phase", phase,
        ], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--phase", choices=("setup", "complete"), default="complete")
    args = parser.parse_args()
    base = args.base.resolve()
    if args.phase == "setup":
        setup(base, args.reference.resolve())
        plan = json.loads((base / "source_plan.json").read_text())
        save_json(base / "source_plan.before_user_clarification.json", plan)
        save_json(base / "source_plan.json", {
            "jobs": [job for job in plan["jobs"] if job["id"].endswith("_free")]
        })
        save_json(base / "scope_amendment.json", {
            "utc": datetime.now(UTC).isoformat(),
            "user_instruction": "Нуль сохранён; проверяется только бинарная маска.",
            "retained_source_jobs": 12,
            "cancelled_methods": ["full", "learned_zero", "mask_only"],
            "target_methods": ["free"], "new_target_answers_opened": False,
            "learner_free_code_changed": False,
        })
    amendment = json.loads((base / "scope_amendment.json").read_text())
    active = Path(__file__).with_name("ACTIVE_PROTOCOL.md")
    shutil.copy2(active, base / active.name)
    paths = [
        Path(__file__), active, Path(__file__).with_name("report.py"),
        Path(__file__).with_name("render.py"),
    ]
    amendment["active_code_sha256"] = {
        str(p.relative_to(PROJECT)): digest(p) for p in paths
    }
    amendment["active_protocol_sha256"] = digest(base / active.name)
    save_json(base / "scope_amendment.json", amendment)
    for path in paths:
        target = base / "scope_code_snapshot" / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    if args.phase == "complete":
        complete(base)


if __name__ == "__main__":
    main()
