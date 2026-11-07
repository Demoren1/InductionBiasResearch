"""G21 frozen-C count-metric ablations with a separate hard-prefix C refit."""

import argparse
import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .count_projection import census
from .prefix_census import forward, make_teacher, numeric_prefix, observations

BASE = Path(__file__).resolve().parent
METHODS = ("euclidean", "qp", "qp_integer", "qp_integer_refit")


def register(root, parent):
    source = json.loads((parent / "protocol.json").read_text())
    if not (parent / "report/summary.json").exists():
        raise RuntimeError("G20 diagnosis must be completed first")
    root.mkdir(parents=True, exist_ok=False)
    original = source["parent_plan"]
    plan = {
        **{
            key: original[key]
            for key in (
                "jobs",
                "inputs",
                "hidden",
                "codes",
                "bias",
                "rank_relative",
                "axis_code_min",
                "axis_code_max",
                "slice_points",
                "slice_scales",
                "validation",
            )
        },
        "stage": "G21 opened G20 count-feasibility/function-metric/C-refit pilot; no new native competition or transfer",
        "registered_utc": datetime.now(UTC).isoformat(),
        "seed": 26110521017,
        "parent": str(parent.resolve()),
        "parent_freeze_sha256": digest(parent / "models_frozen.json"),
        "g19_parent": source["parent"],
        "g19_plan": original,
        "methods": list(METHODS),
        "panels": [{"noise": 0.0, "budget": 4096}]
        + [
            {"noise": sigma, "budget": budget}
            for sigma in (0.2, 2.0)
            for budget in (4096, 16777216)
        ],
        "initial_C_method": "continuous_dp_start",
        "initial_C_grid": 1025,
        "prefix_validation": 512,
        "qp_iterations": 1000,
        "qp_ftol": 1e-12,
        "qp_gap_absolute": 1e-8,
        "qp_primal_tolerance": 1e-7,
        "qp_polish_rounds": 20,
        "integer_moves": 4096,
        "integer_decrease": 1e-12,
        "code_refit_steps": 12,
        "code_ridge_relative": 1e-10,
        "code_line_search": 16,
        "known": "Same G19/G20 shape/K/commonb0.15,readout1/sqrt(H),offset0. No teacher I/C/counts/roles. One underlying function per I, opened geometries.",
        "source": "Frozen G20continuous_dp_start C on grid1025, both low/high budgets retained. Exact same G19 slices/noisy means, hash/array verification. Old freeLS baseline read, not rerun. New independent512validation queries per prefix available to all methods; C-refit selects numerical checkpoint there. Count full inherited axis/slice budget plus16*512*r validation labels; no new axis labels.",
        "methods_detail": "Exact separable Euclidean integer projection; convex function-LS constrained by nonnegative parent capacity then same rounding; plus exact-loss unit-count moves (local, notglobalMIQP); plus hard-prefix numerical C GN-refit. Same rank gate for all. QP FW-gap/primal checks before accepting prefix; failedsolve means abstention, always save metrics. Refitted C never sorted relative to frozenlabels.",
        "selection": "All5panels and4methods saved. No teacher/budget/method selection. C-refit checkpoint uses only independent prefix validation; old baseline has same data available only for reporting after frozen choice, no retrospective HPO.",
        "audit": "Freeze all18jobs/360primary states before G21teacher metrics. Then optional oracle count diagnostics fixed trueC for euclidean/qp/qp_integer, once perjob/two noisy levels, not duplicated acrossaxisprecision. No oracle results passed into primary/selection.",
        "limits": "Known numerical convex/QP/coordinate-descent components, no novelty claim. Conditional fixed-C exact quadratic; QP+local solver notglobalintegerMLE. EstimatedC/designbias, irreversible wrongprefix and validation selection prevent automatic finite coverage. Native competitors, multi-source commonI, freebias/readout, misspecifiednoise and transfer remain.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": {
            name: digest(BASE / name)
            for name in ("count_projection.py", "count_metric.py", "prefix_census.py")
        },
    }
    save_json(root / "protocol.json", plan)
    shutil.copy2(BASE / "THEORY.md", root / "theory_at_registration.md")
    snapshot = root / "executed_code_snapshot"
    snapshot.mkdir()
    for name in plan["code_sha256"]:
        shutil.copy2(BASE / name, snapshot / name)
    (root / "models").mkdir()


@torch.no_grad()
def source_data(plan, job_id, device):
    job = plan["jobs"][job_id]
    original = plan["g19_plan"]
    old = observations(job, original, device)
    parent = Path(plan["parent"])
    parent_path = parent / "models" / f"job_{job_id:03d}"
    frozen = json.loads((parent / "models_frozen.json").read_text())
    if digest(parent / "models_frozen.json") != plan["parent_freeze_sha256"]:
        raise RuntimeError("G20 freeze changed")
    for suffix in (".json", ".npz"):
        current = parent_path.with_suffix(suffix)
        if digest(current) != frozen["sha256"][str(current.relative_to(parent))]:
            raise RuntimeError("Frozen G20 source changed")
    g19 = Path(plan["g19_parent"])
    old_path = g19 / "models" / f"job_{job_id:03d}.npz"
    old_frozen = json.loads((g19 / "models_frozen.json").read_text())
    if digest(old_path) != old_frozen["sha256"][str(old_path.relative_to(g19))]:
        raise RuntimeError("Frozen G19 slice observations changed")
    targets = {}
    with np.load(old_path) as archived:
        for sigma in (0.0, 0.2, 2.0):
            condition_id = next(
                i
                for i, row in enumerate(original["conditions"])
                if row == {"noise": sigma, "replicates": 1 if sigma == 0 else 4096}
            )
            y = (
                old["slice_clean"]
                + sigma / math.sqrt(1 if sigma == 0 else 4096) * old["slice_noise"]
            )
            if not np.array_equal(
                archived[f"condition_{condition_id}__slice_y"], y.cpu().numpy()
            ):
                raise RuntimeError("Slice means regenerated differently")
            targets[sigma] = y
    weights, _, _ = make_teacher(job, original, device)
    generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + job["family"] * 83
    )
    vx, vy, vn = [], [], []
    for step in range(1, plan["inputs"] + 1):
        scales = weights.new_tensor(plan["slice_scales"])
        scales = scales[
            torch.arange(plan["prefix_validation"], device=device) % len(scales)
        ]
        local = torch.randn(
            (plan["prefix_validation"], step),
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        local *= scales[:, None] / math.sqrt(step)
        x = weights.new_zeros((plan["prefix_validation"], plan["inputs"]))
        x[:, old["order"][:step]] = local
        y = forward(x, weights, original)
        noise = torch.randn(
            y.shape, generator=generator, device=device, dtype=torch.float64
        )
        vx.append(x)
        vy.append(y)
        vn.append(noise)
    validation_targets = {
        sigma: torch.stack(vy)
        + sigma / math.sqrt(1 if sigma == 0 else 4096) * torch.stack(vn)
        for sigma in targets
    }
    return (
        old,
        targets,
        vx,
        validation_targets,
        json.loads(parent_path.with_suffix(".json").read_text()),
    )


@torch.no_grad()
def run_case(root, plan, job_id, device):
    started = time.monotonic()
    old, targets, vx, vy, parent = source_data(plan, job_id, device)
    arrays, records = {"order": old["order"]}, []
    for sigma, values in vy.items():
        arrays[f"validation_noise{sigma}"] = values.cpu().numpy()
    for panel_id, panel in enumerate(plan["panels"]):
        sigma, budget = panel["noise"], panel["budget"]
        reference = next(
            row
            for row in parent["records"]
            if row["method"] == plan["initial_C_method"]
            and row["panel"]
            == {
                "noise": sigma,
                "grid": plan["initial_C_grid"],
                "coarse_equivalent_repeats": budget,
            }
        )
        validation_labels = (
            plan["inputs"] * plan["prefix_validation"] * (1 if sigma == 0 else 4096)
        )
        for method in METHODS:
            local_started = time.monotonic()
            output = census(
                old["slices"],
                targets[sigma],
                vx,
                vy[sigma],
                old["order"],
                reference["codebook"],
                method,
                plan,
            )
            key = f"panel_{panel_id}__{method}"
            record = {
                "key": key,
                "panel": panel,
                "method": method,
                "initial_C": reference["codebook"],
                "final_C": output["centers"].tolist(),
                "complete": output["complete"],
                "trace": output["trace"],
                "parent_reference_key": reference["key"],
                "full_available_source_labels": reference[
                    "full_available_source_labels"
                ]
                + validation_labels,
                "new_prefix_validation_labels": validation_labels,
                "seconds": time.monotonic() - local_started,
            }
            if output["complete"]:
                weights = numeric_prefix(
                    output["indices"][-1], output["centers"], device
                )
                prediction = forward(old["validation_x"], weights, plan)
                observed_validation = (
                    old["validation_clean"] + sigma * old["validation_noise"]
                )
                record["iid_validation_mse"] = float(
                    (prediction - observed_validation).square().mean()
                )
            arrays[key + "__indices"] = output["indices"]
            arrays[key + "__codebooks"] = output["codebooks"]
            records.append(record)
            print(
                json.dumps(
                    {
                        "job": job_id,
                        "panel": panel,
                        "method": method,
                        "complete": output["complete"],
                        "prefixes": len(output["indices"]),
                        "last_reason": output["trace"][-1]["reason"],
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    path = root / "models" / f"job_{job_id:03d}"
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    save_json(
        path.with_suffix(".json"),
        {
            "job": plan["jobs"][job_id],
            "records": records,
            "seconds": time.monotonic() - started,
            "protocol_sha256": digest(root / "protocol.json"),
        },
    )


def freeze(root, plan):
    hashes = {}
    for job_id in range(len(plan["jobs"])):
        for suffix in (".json", ".npz"):
            path = root / "models" / (f"job_{job_id:03d}" + suffix)
            hashes[str(path.relative_to(root))] = digest(path)
    save_json(
        root / "models_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "sha256": hashes,
            "protocol_sha256": digest(root / "protocol.json"),
        },
    )


def audit(root, plan):
    frozen = json.loads((root / "models_frozen.json").read_text())
    records, arrays = [], {}
    for job_id, job in enumerate(plan["jobs"]):
        path = root / "models" / f"job_{job_id:03d}"
        for suffix in (".json", ".npz"):
            current = path.with_suffix(suffix)
            if digest(current) != frozen["sha256"][str(current.relative_to(root))]:
                raise RuntimeError("G21 frozen primary changed")
        meta = json.loads(path.with_suffix(".json").read_text())
        _, truth, centers = make_teacher(job, plan["g19_plan"], "cpu")
        expected_C = np.sort(np.abs(centers))
        arrays[f"job_{job_id}__truth_I"] = truth
        arrays[f"job_{job_id}__true_C"] = centers
        with np.load(path.with_suffix(".npz")) as data:
            for source in meta["records"]:
                indices = data[source["key"] + "__indices"]
                record = {
                    "job_id": job_id,
                    "job": job,
                    "key": source["key"],
                    "panel": source["panel"],
                    "method": source["method"],
                    "complete": source["complete"],
                    "exact_I": False,
                    "completed_prefixes": len(indices),
                    "initial_C_max_error": float(
                        np.max(np.abs(np.sort(source["initial_C"]) - expected_C))
                    ),
                    "final_C_max_error": float(
                        np.max(np.abs(np.sort(source["final_C"]) - expected_C))
                    ),
                    "full_available_source_labels": source[
                        "full_available_source_labels"
                    ],
                    "iid_validation_mse": source.get("iid_validation_mse"),
                    "last_reason": source["trace"][-1]["reason"],
                }
                if source["complete"]:
                    metric, aligned, _ = align_indices(
                        indices[-1], truth, plan["codes"]
                    )
                    record.update(metric)
                    record["exact_I"] = metric["categorical_errors"] == 0
                    arrays[f"job_{job_id}__{source['key']}__aligned"] = aligned
                records.append(record)
    np.savez_compressed(root / "audit_only.npz", **arrays)
    save_json(
        root / "audit.json",
        {
            "records": records,
            "models_frozen_sha256": digest(root / "models_frozen.json"),
        },
    )


@torch.no_grad()
def oracle_case(root, plan, job_id, device, qp_solver=None):
    if not (root / "models_frozen.json").exists():
        raise RuntimeError("Entire primary cohort must be frozen before oracle")
    old, targets, vx, vy, _ = source_data(plan, job_id, device)
    _, truth, centers = make_teacher(plan["jobs"][job_id], plan["g19_plan"], device)
    records, arrays = [], {}
    for sigma in (0.2, 2.0):
        for method in METHODS[:-1]:
            output = census(
                old["slices"],
                targets[sigma],
                vx,
                vy[sigma],
                old["order"],
                np.sort(np.abs(centers)),
                method,
                plan,
                qp_solver=qp_solver,
            )
            key = f"noise{sigma}__{method}"
            record = {
                "key": key,
                "noise": sigma,
                "method": method,
                "complete": output["complete"],
                "exact_I": False,
                "completed_prefixes": len(output["indices"]),
                "trace": output["trace"],
            }
            if output["complete"]:
                metric, _, _ = align_indices(
                    output["indices"][-1], truth, plan["codes"]
                )
                record.update(metric)
                record["exact_I"] = metric["categorical_errors"] == 0
            records.append(record)
            arrays[key + "__indices"] = output["indices"]
    path = root / "models" / f"job_{job_id:03d}_oracle"
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    save_json(
        path.with_suffix(".json"),
        {
            "job": plan["jobs"][job_id],
            "records": records,
            "models_frozen_sha256": digest(root / "models_frozen.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("register", "run", "freeze", "audit", "oracle")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root, args.parent)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    if args.action in ("freeze", "audit"):
        destination = args.root / (
            "models_frozen.json" if args.action == "freeze" else "audit.json"
        )
        if destination.exists():
            raise RuntimeError("G21 phase already completed")
        if args.action == "freeze":
            freeze(args.root, plan)
        else:
            audit(args.root, plan)
        return
    for job_id in range(args.shard, len(plan["jobs"]), args.shards):
        suffix = "_oracle.json" if args.action == "oracle" else ".json"
        if (args.root / "models" / (f"job_{job_id:03d}" + suffix)).exists():
            raise RuntimeError("Refusing to repeat completed G21 phase")
        if args.action == "run":
            run_case(args.root, plan, job_id, args.device)
        else:
            oracle_case(args.root, plan, job_id, args.device)


if __name__ == "__main__":
    main()
