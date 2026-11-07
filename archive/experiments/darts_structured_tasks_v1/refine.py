"""Register a source-driven continuation without changing completed runs."""

import argparse
import itertools
import json
import shutil
from pathlib import Path

import torch

from .data import BASE, digest, save_json
from .run import freeze, fresh_codebooks, refit_search
from .train import search


def run(parent, device):
    output = parent.parent / "deepsets_refinement"
    output.mkdir(exist_ok=False)
    plan = json.loads((parent / "protocol.json").read_text())
    specification = json.loads((parent / "specification.json").read_text())
    specification["name"] = "deepsets_refinement"
    plan["search"].update(
        {
            "pilot_steps": 12000,
            "confirmation_steps": 12000,
            "numeric_warmup": 512,
            "minimum_temperature": 0.03,
            "late_entropy": 0.02,
            "pilot_seeds": [41, 42],
            "confirmation_seeds": [43, 44, 45],
        }
    )
    save_json(output / "protocol.json", plan)
    save_json(output / "specification.json", specification)
    for name in ("source.npz", "teacher_audit_only.npz"):
        shutil.copy2(parent / name, output / name)
    (output / "controls").symlink_to(
        Path("..") / parent.name / "controls", target_is_directory=True
    )
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for source in BASE.glob("*.py"):
        shutil.copy2(source, snapshot / source.name)
    candidates = []
    for numeric, architecture, radius, sparsity in itertools.product(
        [0.01, 0.03], [0.03, 0.1], [0.5, 1.5], [0.0, 0.003, 0.01, 0.03]
    ):
        candidates.append(
            {
                "configuration": len(candidates),
                "numeric_lr": numeric,
                "architecture_lr": architecture,
                "radius": radius,
                "sparsity": sparsity,
                "mode": "joint",
            }
        )
    save_json(
        output / "registration.json",
        {
            "reason": "Original source selection NMSE about .079 versus tuned dense Adam .032, with a near-zero exact-I control. Test scores and teacher agreement have not been inspected. Compare longer search, lower final temperature and stronger late entropy; retain both original and new results.",
            "parent_source_sha256": digest(parent / "source.npz"),
            "configurations": candidates,
            "snapshot_sha256": {p.name: digest(p) for p in snapshot.glob("*.py")},
        },
    )
    for phase in ("pilot", "confirmation"):
        selected = candidates
        if phase == "confirmation":
            ids = json.loads((output / "pilot_selection.json").read_text())[
                "top_configurations"
            ]
            selected = [row for row in candidates if row["configuration"] in ids]
        rows = [
            {**row, "seed": seed}
            for row in selected
            for seed in plan["search"][phase + "_seeds"]
        ]
        search(output, plan, specification, rows, phase, device)
        refit_search(output, plan, specification, phase, device)
    fresh_codebooks(output, plan, specification, device)
    freeze(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.parent, torch.device("cuda:0"))


if __name__ == "__main__":
    main()
