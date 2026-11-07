"""G20 frozen-panel codebook ablations, followed by unchanged prefix census."""

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

from .codebook_counts import curve_fit, integer_segments
from .prefix_census import census, forward, make_teacher, numeric_prefix, observations

BASE = Path(__file__).resolve().parent
METHODS = ("continuous", "continuous_dp_start", "integer_mass")


def register(root, parent):
    original = json.loads((parent / "protocol.json").read_text())
    if not (parent / "report/summary.json").exists():
        raise RuntimeError("G19 diagnosis must be complete before G20")
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        **original,
        "stage": "G20 opened G19 codebook density/SNR/integer-mass pilot; unchanged prefix search, not competitive confirmation",
        "registered_utc": datetime.now(UTC).isoformat(),
        "parent": str(parent.resolve()),
        "parent_plan": original,
        "parent_models_sha256": digest(parent / "models_frozen.json"),
        "axis_grids": [257, 1025],
        "coarse_equivalent_repeats": [4096, 16777216],
        "noises": [0.0, 0.2, 2.0],
        "methods": list(METHODS),
        "dp_chunk": 24,
        "panels": [
            {"noise": sigma, "grid": grid, "coarse_equivalent_repeats": budget}
            for sigma in (0.0, 0.2, 2.0)
            for grid in (257, 1025)
            for budget in ([4096] if sigma == 0 else [4096, 16777216])
        ],
        "known": original["known"],
        "methods_detail": "Continuous baseline same objective and7starts asG19 but all G20 restart choices use independent interleaved axis-validation. DP-start replaces only first start, same6 others. Integer-mass uses same7starts and fixed positive integer masses from relaxed DP. Raw DP physicality/bound/refit separate; no exact continuous-spline guarantee.",
        "data": "Underlying G19 teachers/C and exact frozen slices, noisy slice repeats4096, no-noise1; hash-verify regeneration. New fixed axis train+midpoint-validation grids and paired Gaussian sample means, count r labels/query. Per-axis budget=(2*257-1)*coarseR; r=floor(budget/(2*grid-1)). Noiseless r1, duplicate precision budgets omitted. Axis cost includes validation queries, plus full old slices; extremely large virtual label budgets explicit.",
        "selection": "Independent noisy axis-validation chooses one of7 numerical starts within each method/panel. No teacher/hard-I/budget/grid selection. Save every registered result. No inferred C or I from known-C oracle controls passed to learner.",
        "audit": "Entire18-job/540-method-state cohort freezes before G20 teacher-C/mass/I metrics. Old G19 upper-controls already opened and kept separate. Check underlying frozen slice hashes before learner; audit masses and numerical bound only after whole freeze.",
        "limits": "Known commonb/readout and one function/I. No new multi-source benchmark, transfer, misspecified-noise guarantee or native competitor comparison. High-noise unchanged prefix solver already failed with trueC. New component not claimed novel or superior.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": {
            name: digest(BASE / name)
            for name in (
                "codebook_counts.py",
                "codebook_refinement.py",
                "prefix_census.py",
            )
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
def axis_observations(job, plan, grid, device):
    original = plan["parent_plan"]
    weights, _, _ = make_teacher(job, original, device)
    t = np.r_[0.0, np.geomspace(plan["axis_min"], plan["axis_max"], grid - 1)]
    vt = (t[:-1] + t[1:]) / 2
    points = torch.tensor(np.r_[t, vt], device=device)
    axes = torch.eye(plan["inputs"], device=device, dtype=torch.float64)
    x = torch.einsum("s,di,n->sdni", weights.new_tensor([-1, 1]), axes, points)
    clean = forward(x, weights, plan)
    generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + job["family"] * 73 + grid * 11
    )
    noise = torch.randn(
        clean.shape, generator=generator, device=device, dtype=torch.float64
    )
    return t, vt, clean, noise


@torch.no_grad()
def run_case(root, plan, job_id, device):
    started = time.monotonic()
    job = plan["jobs"][job_id]
    original = plan["parent_plan"]
    old = observations(job, original, device)
    parent_path = Path(plan["parent"]) / "models" / f"job_{job_id:03d}.npz"
    frozen = json.loads((Path(plan["parent"]) / "models_frozen.json").read_text())
    if (
        digest(parent_path)
        != frozen["sha256"][str(parent_path.relative_to(Path(plan["parent"])))]
    ):
        raise RuntimeError("Parent G19 observations changed")
    slice_data = {}
    with np.load(parent_path) as previous:
        for sigma in plan["noises"]:
            condition_id = next(
                i
                for i, condition in enumerate(original["conditions"])
                if condition
                == {"noise": sigma, "replicates": 1 if sigma == 0 else 4096}
            )
            target = (
                old["slice_clean"]
                + sigma / math.sqrt(1 if sigma == 0 else 4096) * old["slice_noise"]
            )
            if not np.array_equal(
                previous[f"condition_{condition_id}__slice_y"], target.cpu().numpy()
            ):
                raise RuntimeError("G19 slice regeneration mismatch")
            slice_data[sigma] = target
    records, arrays = [], {"order": old["order"]}
    axes = {
        grid: axis_observations(job, plan, grid, device) for grid in plan["axis_grids"]
    }
    for panel_id, panel in enumerate(plan["panels"]):
        sigma, grid, budget = (
            panel["noise"],
            panel["grid"],
            panel["coarse_equivalent_repeats"],
        )
        t, vt, clean, noise = axes[grid]
        repeat = 1 if sigma == 0 else (2 * 257 - 1) * budget // (2 * grid - 1)
        observed = clean + sigma / math.sqrt(repeat) * noise
        curve = (
            observed.sum((0, 1)).cpu().numpy()
            - 2 * plan["inputs"] * math.sqrt(plan["hidden"]) * plan["bias"]
        )
        y, vy = curve[:grid], curve[grid:]
        initial, masses, dp = integer_segments(t, y, plan, device)
        key = f"panel_{panel_id}"
        arrays[key + "__t"] = t
        arrays[key + "__vt"] = vt
        arrays[key + "__axis_observed"] = observed.cpu().numpy()
        axis_labels = observed.numel() * repeat
        slice_labels = (
            plan["inputs"] * plan["slice_points"] * (1 if sigma == 0 else 4096)
        )
        for method in METHODS:
            centers, fit = curve_fit(
                t, y, vt, vy, plan, plan["seed"] + job_id, method, initial, masses
            )
            output = census(
                old["slices"], slice_data[sigma], old["order"], centers, original
            )
            for point in output["trace"]:
                point["label_budget"] = axis_labels + point["prefix_length"] * plan[
                    "slice_points"
                ] * (1 if sigma == 0 else 4096)
            record = {
                "key": key + "__" + method,
                "panel_id": panel_id,
                "panel": panel,
                "method": method,
                "axis_replicates": repeat,
                "axis_train_and_validation_labels": axis_labels,
                "slice_labels": slice_labels,
                "full_available_source_labels": axis_labels + slice_labels,
                "codebook": centers.tolist(),
                "dp": dp,
                "fit": fit,
                "complete": output["complete"],
                "trace": output["trace"],
            }
            if output["complete"]:
                weights = numeric_prefix(output["I"], centers, device)
                prediction = forward(old["validation_x"], weights, original)
                observed_validation = (
                    old["validation_clean"] + sigma * old["validation_noise"]
                )
                record["iid_validation_mse"] = float(
                    (prediction - observed_validation).square().mean()
                )
            arrays[record["key"] + "__indices"] = output["indices"]
            records.append(record)
        print(
            json.dumps(
                {
                    "job": job_id,
                    "panel": panel,
                    "methods_complete": [row["complete"] for row in records[-3:]],
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
            "job": job,
            "records": records,
            "seconds": time.monotonic() - started,
            "protocol_sha256": digest(root / "protocol.json"),
            "parent_npz_sha256": digest(parent_path),
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
                raise RuntimeError("Frozen G20 primary changed")
        meta = json.loads(path.with_suffix(".json").read_text())
        _, truth, center = make_teacher(job, plan["parent_plan"], "cpu")
        true_order = np.argsort(np.abs(center))
        magnitudes = np.abs(center)[true_order]
        true_mass = np.bincount(np.abs(truth).flatten(), minlength=plan["codes"] + 1)[
            1:
        ][true_order]
        arrays[f"job_{job_id}__truth_I"] = truth
        arrays[f"job_{job_id}__true_C"] = center
        with np.load(path.with_suffix(".npz")) as primary:
            for source in meta["records"]:
                indices = primary[source["key"] + "__indices"]
                record = {
                    "job_id": job_id,
                    "job": job,
                    "key": source["key"],
                    "panel": source["panel"],
                    "method": source["method"],
                    "complete": source["complete"],
                    "exact_I": False,
                    "completed_prefixes": len(indices),
                    "code_max_error": float(
                        np.max(np.abs(np.asarray(source["codebook"]) - magnitudes))
                    ),
                    "dp_masses_exact": bool(
                        np.array_equal(source["dp"]["masses_ascending"], true_mass)
                    ),
                    "true_masses": true_mass.tolist(),
                    "full_available_source_labels": source[
                        "full_available_source_labels"
                    ],
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "freeze", "audit"))
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
        output = args.root / (
            "models_frozen.json" if args.action == "freeze" else "audit.json"
        )
        if output.exists():
            raise RuntimeError("G20 phase already completed")
        if args.action == "freeze":
            freeze(args.root, plan)
        else:
            audit(args.root, plan)
        return
    for job_id in range(args.shard, len(plan["jobs"]), args.shards):
        if (args.root / "models" / f"job_{job_id:03d}.json").exists():
            raise RuntimeError("Refusing to repeat completed G20 fit")
        run_case(args.root, plan, job_id, args.device)


if __name__ == "__main__":
    main()
