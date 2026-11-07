"""G12: tune a proper joint adaptive signed-fusion control on G4 pilot data."""

import argparse
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .competitor_pilot import regression_validation, save_results
from .joint_fusion import train_joint

BASE = Path(__file__).resolve().parent


def register(root, source):
    original = json.loads((source / "protocol.json").read_text())
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        **{
            key: original[key]
            for key in ("seed", "inputs", "outputs", "tasks", "codes")
        },
        "stage": "G12 strong known joint/adaptive fusion control; pilot only",
        "registered_utc": datetime.now(UTC).isoformat(),
        "g4_root": str(source.resolve()),
        "source_root": original["source_root"],
        "pilot_source_jobs": [3, 9, 15, 21],
        "observations": [8192, 131072],
        "lambdas": np.geomspace(1e-5, 10.0, 7).tolist(),
        "rows": [
            {"gamma": gamma, "zero_ratio": ratio}
            for gamma, ratio in itertools.product((0, 1, 2, 4, 8), (0.1, 1.0))
        ],
        "solver": {
            "maximum_steps": 4000,
            "checkpoint_every": 100,
            "tolerance": 1e-6,
            "rho_shift": 1.0,
            "adaptive_rho": True,
            "over_relaxation": 1.5,
        },
        "objective": "Joint vector PACS adaptation with full per-function sample curvature; exact norm proximal steps on zero/sum/difference profile constraints. Fixed observed OLS adaptive weights, no teacher I/mask/group sizes.",
        "selection": "Choose one global gamma/zero-ratio by mean pilot validation MSE; each pilot/source dataset chooses its own lambda and converged checkpoint using the same32 independent full-C support4/query128 functions as G4. No teacher score. This is standard dataset-wise regularization selection, not one potentially inappropriate lambda across data budgets.",
        "readout": "Native exact-zero proximal constraints; signed zero-anchor graph, sign cycles force zero. Eligible only if normalized primal/dual residual and feasible relative primal-dual objective gap<=1e-6, native nonzero groups<=knownK. No k-means, tolerance-clustering or teacher mask. Refitting C under the active equality constraints is explicit solver polishing and uses new support in validation.",
        "scope": "Known PACS/convex-clustering objective-level multi-task adaptation; no novelty claimed, no automatic scalar-PACS oracle theorem. Further evaluation on already-opened G4 cases is an extra control, not new independent confirmation.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": {
            name: digest(BASE / name)
            for name in (
                "joint_fusion.py",
                "joint_fusion_pilot.py",
                "competitor_models.py",
                "competitor_pilot.py",
            )
        },
        "checks": "No software tests. Scientific solver/provenance diagnostics and minimum lint only.",
    }
    plan["jobs"] = [
        {"source_job": job, "n": n}
        for job, n in itertools.product(plan["pilot_source_jobs"], plan["observations"])
    ]
    save_json(root / "protocol.json", plan)
    shutil.copy2(BASE / "THEORY.md", root / "theory_at_registration.md")
    snapshot = root / "executed_code_snapshot"
    snapshot.mkdir()
    for name in plan["code_sha256"]:
        shutil.copy2(BASE / name, snapshot / name)
    (root / "pilot").mkdir()


def run_job(root, plan, job_id, device):
    job = plan["jobs"][job_id]
    source = Path(plan["source_root"])
    source_plan = json.loads((source / "protocol.json").read_text())
    family = source_plan["jobs"][job["source_job"]]["family"]
    path = source / "statistics" / f"job_{job['source_job']:03d}_n{job['n']}.npz"
    output = root / "pilot" / f"job_{job_id:03d}"
    if output.with_suffix(".json").exists():
        raise RuntimeError("Pilot output already exists")
    with np.load(path) as a:
        stats = {k: torch.tensor(a[k], device=device) for k in a.files}
    validation = regression_validation(family["id"], family["kind"], plan, device)
    print(
        json.dumps({"starting_job": job_id, "family": family, "n": job["n"]}),
        flush=True,
    )
    result = train_joint(
        plan["rows"],
        plan["lambdas"],
        stats,
        validation,
        {**plan, "n": job["n"]},
        plan["solver"],
    )
    result.update(
        {
            "source_sha256": digest(path),
            "family": family,
            "n": job["n"],
            "protocol_sha256": digest(root / "protocol.json"),
        }
    )
    save_results(root / "pilot", f"job_{job_id:03d}", result)
    print(
        json.dumps({"finished_job": job_id, "seconds": result["seconds"]}), flush=True
    )


def freeze(root, plan):
    scores, hashes = [], {}
    for job in range(len(plan["jobs"])):
        path = root / "pilot" / f"job_{job:03d}.npz"
        with np.load(path) as a:
            scores.append(a["selection_mse"].min(-1))
        hashes[str(path.relative_to(root))] = digest(path)
        hashes[str(path.with_suffix(".json").relative_to(root))] = digest(
            path.with_suffix(".json")
        )
    means = np.mean(scores, axis=0)
    winner = int(means.argmin())
    if not np.isfinite(means[winner]):
        raise RuntimeError("No converged native-budget candidate on all pilot data")
    destination = root / "hpo_frozen.json"
    if destination.exists():
        raise RuntimeError("HPO already frozen")
    save_json(
        destination,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "row": winner,
            "settings": plan["rows"][winner],
            "mean_validation_mse": float(means[winner]),
            "all_mean_validation_mse": [
                None if not np.isfinite(x) else float(x) for x in means
            ],
            "protocol_sha256": digest(root / "protocol.json"),
            "input_sha256": hashes,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "freeze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root, args.source)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    if args.action == "freeze":
        freeze(args.root, plan)
        return
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Invalid scientific shard")
    for name, expected in plan["code_sha256"].items():
        if digest(BASE / name) != expected:
            raise RuntimeError("Registered joint-fusion code changed")
    for job in range(args.shard, len(plan["jobs"]), args.shards):
        run_job(args.root, plan, job, args.device)


if __name__ == "__main__":
    main()
