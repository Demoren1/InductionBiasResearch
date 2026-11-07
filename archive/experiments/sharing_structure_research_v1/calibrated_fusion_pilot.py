"""G13 pilot: separate noise subtraction, floor decay and known heat kernels."""

import argparse
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.evaluate import alignment

from .competitor_pilot import regression_validation, save_results
from .gaussian_stats import sample_index
from .joint_fusion import train_joint

BASE = Path(__file__).resolve().parent


def register(root, source):
    original = json.loads((source / "protocol.json").read_text())
    raw_frozen = json.loads((source / "hpo_frozen.json").read_text())
    if not (source / "pilot_diagnosis.json").exists():
        raise RuntimeError("Measure the raw fusion failure before registering G13")
    root.mkdir(parents=True, exist_ok=False)
    ratio = raw_frozen["settings"]["zero_ratio"]
    rows = [{"gamma": g, "zero_ratio": ratio} for g in (1, 2, 4, 8)]
    plan = {
        **{
            k: original[k]
            for k in (
                "seed",
                "inputs",
                "outputs",
                "tasks",
                "codes",
                "source_root",
                "g4_root",
                "observations",
                "pilot_source_jobs",
                "lambdas",
                "solver",
            )
        },
        "stage": "G13 prospective algorithmic pilot on already-opened G4 pilot data, not independent confirmation",
        "registered_utc": datetime.now(UTC).isoformat(),
        "raw_control": str(source.resolve()),
        "raw_diagnosis_sha256": digest(source / "pilot_diagnosis.json"),
        "weight_metrics": ["raw_shrinking_floor", "debiased", "heat"],
        "rows": {
            "raw_shrinking_floor": rows,
            "debiased": rows,
            "heat": [
                {"gamma": 0, "bandwidth": b, "zero_ratio": ratio}
                for b in (0.1, 0.3, 1.0, 3.0)
            ],
        },
        "selection": "Separate source-only global row for each weighting method; each dataset chooses lambda/converged checkpoint from same seven-point path. Same observed source functions/validation as G12. Ratio fixed from earlier raw-CV winner. No true-I audit until all24 outputs frozen.",
        "controls": "Raw old-floor power weights reuse complete frozen G12 results, no retraining. Raw shrinking floor isolates floor decay. Heat uses known Gaussian kernels, shared observed noise-scale bandwidth and finite log-weight cap[-30,30]. No priority claim for noise correction/convex clustering.",
        "known": "Shape, exactlyK3 signed groups with fixedzero; no teacher mask/positions/group sizes/coefficients",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": {
            name: digest(BASE / name)
            for name in (
                "calibrated_fusion_pilot.py",
                "joint_fusion.py",
                "fusion_weights.py",
                "competitor_models.py",
                "competitor_pilot.py",
                "gaussian_stats.py",
            )
        },
        "checks": "No software tests; necessary scientific solver diagnostics/provenance and lint only.",
    }
    plan["jobs"] = [
        {"source_job": job, "n": n, "weight_metric": metric}
        for metric, job, n in itertools.product(
            plan["weight_metrics"], plan["pilot_source_jobs"], plan["observations"]
        )
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
    destination = root / "pilot" / f"job_{job_id:03d}.json"
    if destination.exists():
        raise RuntimeError("Calibrated fusion pilot already exists")
    with np.load(path) as a:
        stats = {k: torch.tensor(a[k], device=device) for k in a.files}
    validation = regression_validation(family["id"], family["kind"], plan, device)
    print(
        json.dumps({"starting_job": job_id, "job": job, "family": family}), flush=True
    )
    result = train_joint(
        plan["rows"][job["weight_metric"]],
        plan["lambdas"],
        stats,
        validation,
        {**plan, "n": job["n"]},
        plan["solver"],
        weight_metric=job["weight_metric"],
    )
    result.update(
        {
            "job": job,
            "family": family,
            "source_sha256": digest(path),
            "protocol_sha256": digest(root / "protocol.json"),
        }
    )
    save_results(root / "pilot", f"job_{job_id:03d}", result)
    print(
        json.dumps({"finished_job": job_id, "seconds": result["seconds"]}), flush=True
    )


def freeze(root, plan):
    choices, hashes = {}, {}
    for metric in plan["weight_metrics"]:
        scores = []
        for i, job in enumerate(plan["jobs"]):
            if job["weight_metric"] != metric:
                continue
            path = root / "pilot" / f"job_{i:03d}.npz"
            with np.load(path) as a:
                scores.append(a["selection_mse"].min(-1))
            hashes[str(path.relative_to(root))] = digest(path)
            hashes[str(path.with_suffix(".json").relative_to(root))] = digest(
                path.with_suffix(".json")
            )
        mean = np.mean(scores, axis=0)
        row = int(mean.argmin())
        if not np.isfinite(mean[row]):
            raise RuntimeError(f"No native converged pilot candidate for {metric}")
        choices[metric] = {
            "row": row,
            "settings": plan["rows"][metric][row],
            "mean_validation_mse": float(mean[row]),
            "all_mean_validation_mse": [
                None if not np.isfinite(v) else float(v) for v in mean
            ],
        }
    path = root / "cohort_frozen.json"
    if path.exists():
        raise RuntimeError("Calibrated pilot cohort already frozen")
    save_json(
        path,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "choices": choices,
            "protocol_sha256": digest(root / "protocol.json"),
            "sha256": hashes,
        },
    )


def audit(root, plan):
    frozen = json.loads((root / "cohort_frozen.json").read_text())
    path = root / "audit.json"
    if path.exists():
        raise RuntimeError("Calibrated pilot already audited")
    source_plan = json.loads((Path(plan["source_root"]) / "protocol.json").read_text())
    records = []
    for i, job in enumerate(plan["jobs"]):
        file = root / "pilot" / f"job_{i:03d}.npz"
        if digest(file) != frozen["sha256"][str(file.relative_to(root))]:
            raise RuntimeError("Frozen calibrated pilot changed")
        family = source_plan["jobs"][job["source_job"]]["family"]
        truth = sample_index(family["id"], family["kind"], plan, "cpu").numpy()
        row = frozen["choices"][job["weight_metric"]]["row"]
        with np.load(file) as a:
            values = dict(a)
        column = int(values["selection_mse"][row].argmin())
        index = values["indices"][row, column]
        recovery, aligned = alignment(index, truth, plan["codes"])
        meta = json.loads(file.with_suffix(".json").read_text())
        step = int(values["selected_step"][row, column])
        selected = next(h for h in meta["history"] if h["step"] == step)
        records.append(
            {
                "job_id": i,
                "job": job,
                "family": family,
                "recovery": recovery,
                "selected_lambda": plan["lambdas"][column],
                "validation_mse": float(values["selection_mse"][row, column]),
                "selected_step": step,
                "primal_residual": selected["primal_residual"][row],
                "dual_residual": selected["dual_residual"][row],
                "relative_primal_dual_gap": selected["relative_primal_dual_gap"][row],
                "trace_mask_recall": (
                    (values["trace_masks"][:, row] & (truth != 0)).sum((-1, -2))
                    / (truth != 0).sum()
                ).tolist(),
            }
        )
        np.savez_compressed(
            file.with_name(file.stem + "_audit_only.npz"), truth=truth, index=aligned
        )
    save_json(
        path,
        {
            "status": "Already-opened pilot geometry audit after all source outputs/choices frozen; no independent confirmation or target claim",
            "records": records,
            "cohort_frozen_sha256": digest(root / "cohort_frozen.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "freeze", "audit"))
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
    if args.action == "audit":
        audit(args.root, plan)
        return
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Invalid scientific shard")
    for name, expected in plan["code_sha256"].items():
        if digest(BASE / name) != expected:
            raise RuntimeError("Registered calibrated solver changed")
    for i in range(args.shard, len(plan["jobs"]), args.shards):
        run_job(args.root, plan, i, args.device)


if __name__ == "__main__":
    main()
