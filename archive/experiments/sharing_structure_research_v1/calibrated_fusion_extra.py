"""Frozen G13 weight choices on the same, already-opened G4 cohort."""

import argparse
import json
import shutil
from pathlib import Path

import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .joint_fusion_confirm import audit_case, freeze, register, run_case

BASE = Path(__file__).resolve().parent


def register_all(root):
    plan = json.loads((root / "protocol.json").read_text())
    pilot = json.loads((root / "cohort_frozen.json").read_text())
    if not (root / "audit.json").exists():
        raise RuntimeError("Complete the prespecified pilot audit before extending")
    extra = root / "additional"
    extra.mkdir(exist_ok=False)
    registrations = {}
    for metric in plan["weight_metrics"]:
        folder = extra / metric
        folder.mkdir()
        child = {
            **plan,
            "weight_metric": metric,
            "stage": "G13 additional paired evaluation on already-opened G4; not independent confirmation",
            "parent_pilot_frozen_sha256": digest(root / "cohort_frozen.json"),
        }
        save_json(folder / "protocol.json", child)
        save_json(
            folder / "hpo_frozen.json",
            {
                **pilot["choices"][metric],
                "parent_pilot_frozen_sha256": digest(root / "cohort_frozen.json"),
                "origin": "Only earlier source-only G13 pilot choices; no additional geometry or target selection",
            },
        )
        register(folder)
        registrations[metric] = digest(folder / "confirmation_registration.json")
    save_json(
        root / "additional_registration.json",
        {
            "parent_pilot_sha256": digest(root / "cohort_frozen.json"),
            "method_registrations": registrations,
            "entry_code_sha256": digest(Path(__file__)),
            "scope": "All3 methods on48 G4 source conditions each; whole144 cohort frozen before scoring/regenerating the paired old target cohort. Not new independent confirmation.",
        },
    )
    shutil.copy2(Path(__file__), extra / "executed_entry.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "freeze", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register_all(args.root)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    specs, children = {}, {}
    for metric in plan["weight_metrics"]:
        folder = args.root / "additional" / metric
        children[metric] = json.loads((folder / "protocol.json").read_text())
        specs[metric] = json.loads(
            (folder / "confirmation_registration.json").read_text()
        )
    if args.action == "freeze":
        receipts = {}
        for metric in plan["weight_metrics"]:
            folder = args.root / "additional" / metric
            freeze(folder, specs[metric])
            receipts[metric] = digest(folder / "confirmation_models_frozen.json")
        save_json(
            args.root / "additional_models_frozen.json",
            {
                "sha256": receipts,
                "registration_sha256": digest(
                    args.root / "additional_registration.json"
                ),
            },
        )
        return
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Invalid scientific shard")
    if args.action == "run":
        for metric in plan["weight_metrics"]:
            for name, expected in specs[metric]["code_sha256"].items():
                if digest(BASE / name) != expected:
                    raise RuntimeError("Registered additional method code changed")
    else:
        receipt = json.loads((args.root / "additional_models_frozen.json").read_text())
        for metric in plan["weight_metrics"]:
            folder = args.root / "additional" / metric
            if (
                digest(folder / "confirmation_models_frozen.json")
                != receipt["sha256"][metric]
            ):
                raise RuntimeError("Additional cohort freeze changed")
    jobs = [
        (metric, i)
        for metric in plan["weight_metrics"]
        for i in range(len(specs[metric]["jobs"]))
    ]
    for metric, job_id in jobs[args.shard :: args.shards]:
        folder = args.root / "additional" / metric
        function = run_case if args.action == "run" else audit_case
        function(folder, children[metric], specs[metric], job_id, args.device)


if __name__ == "__main__":
    main()
