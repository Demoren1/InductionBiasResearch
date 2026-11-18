"""Execute the dependent comparison stages, preserving a resumable journal."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json


def run(root, output, device):
    journal = output / "workflow.json"
    state = json.loads(journal.read_text()) if journal.exists() else {"completed": []}

    def stage(name, module, arguments):
        if name in state["completed"]:
            return
        command = [sys.executable, "-m", module, "--root", str(root), *arguments]
        state.update(
            status="running", stage=name, parent_pid=os.getpid(), command=command
        )
        save_json(journal, state)
        subprocess.run(command, check=True)
        state["completed"].append(name)
        save_json(journal, state)

    namespace = "experiments.paper_recovery_comparison_v1."
    common = ["--output", str(output), "--device", device]
    stage("dense", namespace + "source", common)
    stage(
        "g34_proposals",
        "experiments.paper_structure_benchmark_v1.methods.real_project",
        ["--device", device],
    )
    stage(
        "pair_proposals",
        "experiments.omniglot_single_task_v1.extract",
        ["--output", str(output / "pair")],
    )
    for method, proposal in (("g34", "current_proposals"), ("pair", "ratio_proposals")):
        metadata = json.loads(
            (root / "source" / proposal / "candidates.json").read_text()
        )
        for codes in sorted({row["codes"] for row in metadata["candidates"]}):
            stage(
                f"refit_{method}_{codes}",
                namespace + "workers",
                common
                + ["--phase", "refit", "--method", method, "--codes", str(codes)],
            )
    save_json(
        root / "source_choices_frozen.json",
        {
            "source_models": {
                str(p.relative_to(root)): digest(p)
                for p in (root / "fits").glob("*/selected.npz")
            },
            "target_query_labels_used": False,
        },
    )
    for method in ("g34", "pair", "dense"):
        stage(
            f"tune_{method}",
            namespace + "workers",
            common + ["--phase", "tune", "--method", method],
        )
    stage("export", namespace + "workers", common + ["--phase", "export"])
    frozen = json.loads((output / "development_frozen.json").read_text())
    for method in frozen["models"]:
        stage(
            f"target_{method}",
            "experiments.paper_structure_benchmark_v1.fitting.real_target",
            ["--method", method, "--device", device],
        )
    stage("score", namespace + "workers", common + ["--phase", "score"])
    state.update(status="complete", stage="complete")
    save_json(journal, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve(), args.device)


if __name__ == "__main__":
    main()
