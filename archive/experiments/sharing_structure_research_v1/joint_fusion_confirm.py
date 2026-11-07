"""Additional joint-fusion control on already-opened, unchanged G4 cases."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.evaluate import alignment

from .competitor_pilot import regression_validation, save_results
from .confirmation import transfer
from .joint_fusion import train_joint

BASE = Path(__file__).resolve().parent


def register(root):
    plan = json.loads((root / "protocol.json").read_text())
    choice = root / "hpo_frozen.json"
    if not choice.exists():
        raise RuntimeError("Freeze source-only pilot choices before this control")
    source = Path(plan["g4_root"])
    original = json.loads((source / "confirmation_registration.json").read_text())
    jobs = [
        {"source_case": i, "family": job["family"], "regime": job["name"], "n": n}
        for i, job in enumerate(original["jobs"])
        for n in job["train_at"]
    ]
    specification = {
        "registered_utc": datetime.now(UTC).isoformat(),
        "stage": "Additional strong known control on already-opened G4 data; not new independent confirmation",
        "hpo_sha256": digest(choice),
        "g4_frozen_sha256": digest(source / "confirmation_models_frozen.json"),
        "jobs": jobs,
        "transfer": original["transfer"],
        "code_sha256": {
            name: digest(BASE / name)
            for name in (
                "joint_fusion_confirm.py",
                "joint_fusion.py",
                "fusion_weights.py",
                "competitor_models.py",
                "competitor_pilot.py",
            )
        },
        "weight_metric": plan.get("weight_metric", "raw"),
        "policy": "Source models/checkpoints use only source data and source-validation. Freeze whole additional cohort before new structural scoring. Paired transfer regenerates exactly the same G4 target seed/cohort, never uses target for selection.",
    }
    path = root / "confirmation_registration.json"
    if path.exists():
        raise RuntimeError("Additional control already registered")
    save_json(path, specification)
    snapshot = root / "confirmation_executed_code_snapshot"
    snapshot.mkdir()
    for name in specification["code_sha256"]:
        shutil.copy2(BASE / name, snapshot / name)
    (root / "confirmation").mkdir()


def run_case(root, plan, spec, job_id, device):
    job = spec["jobs"][job_id]
    folder = root / "confirmation" / f"job_{job_id:03d}"
    folder.mkdir(exist_ok=False)
    source = Path(plan["g4_root"])
    source_file = (
        source
        / "confirmation"
        / f"case_{job['source_case']:03d}"
        / f"source_n{job['n']}.npz"
    )
    old_freeze = json.loads((source / "confirmation_models_frozen.json").read_text())
    if (
        digest(source_file)
        != old_freeze["sha256"][str(source_file.relative_to(source))]
    ):
        raise RuntimeError("Original frozen G4 source changed")
    with np.load(source_file) as a:
        stats = {name: torch.tensor(a[name], device=device) for name in a.files}
    choice = json.loads((root / "hpo_frozen.json").read_text())
    validation = regression_validation(
        job["family"]["id"], job["family"]["kind"], plan, device
    )
    result = train_joint(
        [choice["settings"]],
        plan["lambdas"],
        stats,
        validation,
        {**plan, "n": job["n"]},
        plan["solver"],
        weight_metric=plan.get("weight_metric", "raw"),
    )
    scores = result["selection_mse"][0]
    path_index = int(scores.argmin())
    returned = bool(np.isfinite(scores[path_index]))
    if returned:
        result["selected_index"] = result["indices"][0, path_index]
        result["selected_weights"] = result["weights"][0, path_index]
    result.update(
        {
            "job": job,
            "returned_index": returned,
            "selected_lambda_index": path_index if returned else None,
            "selected_lambda": plan["lambdas"][path_index] if returned else None,
            "selected_validation_mse": float(scores[path_index]) if returned else None,
            "source_sha256": digest(source_file),
            "registration_sha256": digest(root / "confirmation_registration.json"),
        }
    )
    save_results(folder, "model", result)
    print(
        json.dumps(
            {
                "finished_additional_case": job_id,
                "job": job,
                "returned": returned,
                "seconds": result["seconds"],
            }
        ),
        flush=True,
    )


def freeze(root, spec):
    hashes = {}
    for i in range(len(spec["jobs"])):
        folder = root / "confirmation" / f"job_{i:03d}"
        for name in ("model.npz", "model.json"):
            path = folder / name
            hashes[str(path.relative_to(root))] = digest(path)
    path = root / "confirmation_models_frozen.json"
    if path.exists():
        raise RuntimeError("Additional cohort already frozen")
    save_json(
        path,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "registration_sha256": digest(root / "confirmation_registration.json"),
            "hpo_sha256": digest(root / "hpo_frozen.json"),
            "sha256": hashes,
        },
    )


def audit_case(root, plan, spec, job_id, device):
    frozen = json.loads((root / "confirmation_models_frozen.json").read_text())
    folder = root / "confirmation" / f"job_{job_id:03d}"
    if (folder / "audit.json").exists():
        raise RuntimeError("Additional case already audited")
    for name in ("model.npz", "model.json"):
        path = folder / name
        if digest(path) != frozen["sha256"][str(path.relative_to(root))]:
            raise RuntimeError("Additional frozen model changed")
    metadata = json.loads((folder / "model.json").read_text())
    job = spec["jobs"][job_id]
    source = Path(plan["g4_root"]) / "confirmation" / f"case_{job['source_case']:03d}"
    with np.load(source / "teacher_audit_only.npz") as a:
        truth = a["indices"]
    with np.load(folder / "model.npz") as a:
        models = dict(a)
    result = {
        "job": job,
        "returned_index": metadata["returned_index"],
        "selected_lambda": metadata["selected_lambda"],
        "validation_mse": metadata["selected_validation_mse"],
    }
    if metadata["returned_index"]:
        result["recovery"], aligned = alignment(
            models["selected_index"], truth, plan["codes"]
        )
        result["transfer"] = transfer(
            models["selected_index"],
            truth,
            {**spec, "codes": plan["codes"]},
            761105000 + job["family"]["id"],
            device,
        )
        np.savez_compressed(
            folder / "aligned_audit_only.npz", index=aligned, truth=truth
        )
    masks = models["trace_masks"]
    true_mask = truth != 0
    mask_recall = (masks & true_mask).sum((-1, -2)) / true_mask.sum()
    mask_precision = (masks & true_mask).sum((-1, -2)) / masks.sum((-1, -2)).clip(1)
    result["trace_mask_recall"] = mask_recall.tolist()
    result["trace_mask_precision"] = mask_precision.tolist()
    result["oracle_transfer"] = transfer(
        truth,
        truth,
        {**spec, "codes": plan["codes"]},
        761105000 + job["family"]["id"],
        device,
    )
    result["cohort_frozen_sha256"] = digest(root / "confirmation_models_frozen.json")
    save_json(folder / "audit.json", result)
    print(
        json.dumps(
            {
                "audited_additional_case": job_id,
                "returned": result["returned_index"],
                "exact": result.get("recovery", {}).get("exact_I"),
            }
        ),
        flush=True,
    )


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
        register(args.root)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    spec = json.loads((args.root / "confirmation_registration.json").read_text())
    if args.action == "freeze":
        freeze(args.root, spec)
        return
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Invalid scientific shard")
    if args.action == "run":
        if digest(args.root / "hpo_frozen.json") != spec["hpo_sha256"]:
            raise RuntimeError("Frozen hyperparameters changed")
        for name, expected in spec["code_sha256"].items():
            if digest(BASE / name) != expected:
                raise RuntimeError("Registered control code changed")
    for job_id in range(args.shard, len(spec["jobs"]), args.shards):
        if args.action == "run":
            run_case(args.root, plan, spec, job_id, args.device)
        else:
            audit_case(args.root, plan, spec, job_id, args.device)


if __name__ == "__main__":
    main()
