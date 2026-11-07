"""G19: unknown-codebook prefix census; teacher controls only after freeze."""

import argparse
import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares, nnls

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .functional_observability import teacher as g18_teacher

BASE = Path(__file__).resolve().parent


def register(root, parent):
    old = json.loads((parent / "protocol.json").read_text())
    if not (parent / "audit.json").exists():
        raise RuntimeError("G18 diagnosis must be completed first")
    root.mkdir(parents=True, exist_ok=False)
    conditions = [{"noise": 0.0, "replicates": 1}] + [
        {"noise": sigma, "replicates": repeat}
        for sigma in (0.2, 2.0)
        for repeat in (1, 64, 4096)
    ]
    plan = {
        "stage": "G19 direct signed prefix-count candidate, one-source observability pilot; not competitive multi-task confirmation",
        "registered_utc": datetime.now(UTC).isoformat(),
        "seed": 26110519017,
        "parent": str(parent.resolve()),
        "parent_audit_sha256": digest(parent / "audit.json"),
        "g18_plan": old,
        "jobs": [
            {"family": family, "variant": variant}
            for family in old["families"]
            for variant in old["variants"]
        ]
        + [{"family": family, "variant": "random"} for family in range(8901, 8907)],
        "inputs": 16,
        "hidden": 16,
        "codes": 4,
        "bias": 0.15,
        "conditions": conditions,
        "axis_points": 257,
        "axis_min": 0.001,
        "axis_max": 1024.0,
        "axis_code_min": 1e-4,
        "axis_code_max": 16.0,
        "axis_starts": 7,
        "axis_solver_evaluations": 400,
        "slice_points": 2048,
        "slice_scales": [0.05, 0.2, 0.8, 3.2, 12.8, 51.2],
        "rank_relative": 1e-12,
        "validation": 8192,
        "known": "D,H,K,common bias0.15,readout1/sqrt(H),offset0. Unknown C/mask/counts/input roles. This is a restricted pilot, not free-bias/readout identification.",
        "teacher": "Opened G18 local/mixed coordinate geometries with new fresh Gaussian numeric C, plus six new unrestricted random signed I (Bernoulli0.55 connections, random groups/signs). No true structure/counts/numbers reach learner. One function per I, not multi-source learning.",
        "learner": "Estimate positive code magnitudes from summed even axis queries by ordinary segmented-LS initialization and variable projection. Expand learned prefixes by0/+-codes. Global linear LS for extension counts, nearest integer, abstain on negative/conservation/rank failure. No W clustering or true blocks. Every condition retained.",
        "data": "Fixed random learner coordinate order, fixed Gaussian coordinate-prefix panels with registered scales. Sample mean r independent Gaussian labels simulated exactly as f+sigma/sqrt(r)Z. Count r physical labels/query. Common unit noises make noise/budget comparisons dependent. Queries are not IID Gaussian full-input data. No teacher autograd queries.",
        "audit": "Freeze all18 primary jobs before true C/I. Then factorial diagnostic of estimated/exact C and observed/clean slice labels. Oracle variants are diagnostic only, never primary results or selection. Prefix-I and full-I separate; validation never selects settings/conditions/teacher stops.",
        "limits": "No novelty, competitor win, finite estimated-C/noisy recovery guarantee, multi-source benchmark or fresh-C transfer claim. Known bias/positive common readout, active queries and potentially huge repeat count explicit. No software tests.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": digest(Path(__file__)),
    }
    save_json(root / "protocol.json", plan)
    shutil.copy2(BASE / "THEORY.md", root / "theory_at_registration.md")
    snapshot = root / "executed_code_snapshot"
    snapshot.mkdir()
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    (root / "models").mkdir()


def make_teacher(job, plan, device):
    family, variant = job["family"], job["variant"]
    rng = np.random.default_rng(plan["seed"] + family * 41)
    if variant == "random":
        active = rng.uniform(size=(16, 16)) < 0.55
        index = active * rng.integers(1, 5, size=(16, 16))
        index *= rng.choice([-1, 1], size=(16, 16))
    else:
        _, index, _, _ = g18_teacher(family, variant, plan["g18_plan"], "cpu")
    centers = rng.normal(size=plan["codes"])
    if variant == "quartet_mix":
        centers /= 2
    weights = np.r_[0.0, centers][np.abs(index)] * np.sign(index)
    return torch.tensor(weights, device=device), index.astype(np.int16), centers


def forward(x, weights, plan):
    return (x @ weights + plan["bias"]).relu().sum(-1) / math.sqrt(plan["hidden"])


@torch.no_grad()
def observations(job, plan, device):
    weights, _, _ = make_teacher(job, plan, device)
    generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + job["family"] * 41 + 11
    )
    rng = np.random.default_rng(plan["seed"] + job["family"] * 41 + 19)
    order = rng.permutation(plan["inputs"])
    t = np.r_[
        0.0, np.geomspace(plan["axis_min"], plan["axis_max"], plan["axis_points"] - 1)
    ]
    axes = torch.eye(plan["inputs"], device=device, dtype=torch.float64)
    axis_x = torch.einsum(
        "s,di,n->sdni", weights.new_tensor([-1, 1]), axes, weights.new_tensor(t)
    )
    axis_clean = forward(axis_x, weights, plan)
    axis_noise = torch.randn(
        axis_clean.shape, generator=generator, device=device, dtype=torch.float64
    )
    slices, slice_clean, slice_noise = [], [], []
    for step in range(1, plan["inputs"] + 1):
        scales = weights.new_tensor(plan["slice_scales"])
        scales = scales[torch.arange(plan["slice_points"], device=device) % len(scales)]
        local = torch.randn(
            (plan["slice_points"], step),
            generator=generator,
            device=device,
            dtype=torch.float64,
        )
        local *= scales[:, None] / math.sqrt(step)
        x = weights.new_zeros((plan["slice_points"], plan["inputs"]))
        x[:, order[:step]] = local
        clean = forward(x, weights, plan)
        noise = torch.randn(
            clean.shape, generator=generator, device=device, dtype=torch.float64
        )
        slices.append(x)
        slice_clean.append(clean)
        slice_noise.append(noise)
    vx = torch.randn(
        (plan["validation"], plan["inputs"]),
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    vclean = forward(vx, weights, plan)
    vnoise = torch.randn(
        vclean.shape, generator=generator, device=device, dtype=torch.float64
    )
    return {
        "axis_t": t,
        "axis_clean": axis_clean,
        "axis_noise": axis_noise,
        "order": order,
        "slices": slices,
        "slice_clean": torch.stack(slice_clean),
        "slice_noise": torch.stack(slice_noise),
        "validation_x": vx,
        "validation_clean": vclean,
        "validation_noise": vnoise,
    }


def segmented_start(x, y, codes):
    count = len(x)
    sums = [np.r_[0.0, np.cumsum(value)] for value in (x, x * x, y, y * y, x * y)]
    left, right = np.arange(count)[:, None], np.arange(1, count + 1)[None]
    n = np.maximum(right - left, 1)
    sx, sxx, sy, syy, sxy = [value[right] - value[left] for value in sums]
    variance = sxx - sx * sx / n
    covariance = sxy - sx * sy / n
    cost = syy - sy * sy / n - covariance * covariance / np.maximum(variance, 1e-30)
    cost[right - left < 4] = np.inf
    table = np.full((codes + 2, count + 1), np.inf)
    parent = np.full(table.shape, -1, dtype=int)
    table[1, 4:] = sums[3][4:]
    for segments in range(2, codes + 2):
        for end in range(segments * 4, count + 1):
            candidates = np.arange((segments - 1) * 4, end - 3)
            objective = table[segments - 1, candidates] + cost[candidates, end - 1]
            selected = int(np.argmin(objective))
            parent[segments, end] = candidates[selected]
            table[segments, end] = objective[selected]
    end, boundaries = count, []
    for segments in range(codes + 1, 1, -1):
        end = parent[segments, end]
        boundaries.append(end)
    boundaries.reverse()
    return np.sqrt(np.maximum(x[np.array(boundaries) - 1], 1e-12) * x[boundaries])


def fit_codes(axis_t, observed, plan, seed):
    curve = observed.sum((0, 1)).cpu().numpy()
    curve -= 2 * plan["inputs"] * math.sqrt(plan["hidden"]) * plan["bias"]
    knots = segmented_start(axis_t, curve, plan["codes"])
    starts = [plan["bias"] / knots, [0.15, 0.4, 1.0, 2.5], [0.3, 0.6, 1.0, 1.7]]
    rng = np.random.default_rng(seed)
    while len(starts) < plan["axis_starts"]:
        starts.append(np.exp(rng.uniform(np.log(0.03), np.log(4), plan["codes"])))

    def residual(log_codes, return_amplitude=False):
        values = np.sort(np.exp(log_codes))
        matrix = np.maximum(axis_t[:, None] * values[None] - plan["bias"], 0)
        scale = np.maximum(np.linalg.norm(matrix, axis=0), 1e-30)
        amplitude, _ = nnls(matrix / scale, curve, maxiter=100)
        amplitude /= scale
        error = matrix @ amplitude - curve
        return (error, amplitude) if return_amplitude else error

    best, attempts = None, []
    for initial in starts:
        initial = np.clip(
            initial, plan["axis_code_min"] * 1.001, plan["axis_code_max"] / 1.001
        )
        result = least_squares(
            residual,
            np.log(initial),
            bounds=(math.log(plan["axis_code_min"]), math.log(plan["axis_code_max"])),
            max_nfev=plan["axis_solver_evaluations"],
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        error, amplitude = residual(result.x, return_amplitude=True)
        score = float(np.mean(error**2))
        attempts.append(
            {"mse": score, "evaluations": result.nfev, "success": bool(result.success)}
        )
        if best is None or score < best[0]:
            best = score, np.sort(np.exp(result.x)), amplitude
    return best[1], {
        "axis_even_mse": best[0],
        "amplitude": best[2].tolist(),
        "attempts": attempts,
    }


def numeric_prefix(prefix, centers, device):
    index = torch.tensor(prefix, device=device).long()
    values = torch.as_tensor(centers, device=device, dtype=torch.float64)
    return values[(index.abs() - 1).clamp_min(0)] * index.sign()


@torch.no_grad()
def census(slices, targets, order, centers, plan):
    device = slices[0].device
    prefix = np.zeros((1, 0), dtype=np.int16)
    counts = np.array([plan["hidden"]], dtype=np.int16)
    nonzero = np.r_[np.arange(1, plan["codes"] + 1), -np.arange(1, plan["codes"] + 1)]
    values = numeric_prefix(nonzero, centers, device)
    trace, indices, debug = [], [], []
    for step, (x, y) in enumerate(zip(slices, targets, strict=True), 1):
        weights = numeric_prefix(prefix, centers, device)
        base = x[:, order[: step - 1]] @ weights.T + plan["bias"]
        atom0 = base.relu() / math.sqrt(plan["hidden"])
        matrix = (base[..., None] + x[:, order[step - 1], None, None] * values).relu()
        matrix = (matrix - base.relu()[..., None]).flatten(1) / math.sqrt(
            plan["hidden"]
        )
        target = y - atom0 @ y.new_tensor(counts)
        gram = matrix.T @ matrix
        eigenvalues, vectors = torch.linalg.eigh(gram)
        valid = eigenvalues > eigenvalues[-1].clamp_min(1e-30) * plan["rank_relative"]
        row = {
            "prefix_length": step,
            "parameters": matrix.shape[1],
            "rank": int(valid.sum()),
            "condition": float(eigenvalues[-1] / eigenvalues[0])
            if eigenvalues[0] > 0
            else None,
        }
        if not valid.all():
            trace.append(
                {**row, "accepted": False, "reason": "rank_deficient_dictionary"}
            )
            break
        coefficient = vectors @ ((vectors.T @ (matrix.T @ target)) / eigenvalues)
        other = coefficient.reshape(len(prefix), len(nonzero)).cpu().numpy()
        estimated = np.c_[counts - other.sum(1), other]
        rounded = np.rint(estimated).astype(int)
        residual = matrix @ coefficient - target
        row["slice_mse_continuous"] = float(residual.square().mean())
        row["integrality_gap"] = float(np.max(np.abs(estimated - rounded)))
        feasible = bool((rounded >= 0).all() and np.array_equal(rounded.sum(1), counts))
        debug.append(estimated)
        if not feasible:
            trace.append(
                {**row, "accepted": False, "reason": "infeasible_integer_counts"}
            )
            break
        rounded_other = y.new_tensor(rounded[:, 1:].flatten())
        row["slice_mse_integer"] = float(
            (matrix @ rounded_other - target).square().mean()
        )
        extended = np.c_[
            np.repeat(prefix, len(nonzero) + 1, axis=0),
            np.tile(np.r_[0, nonzero], len(prefix)),
        ]
        keep = rounded.flatten() > 0
        prefix = extended[keep]
        counts = rounded.flatten()[keep]
        index = np.zeros((plan["inputs"], plan["hidden"]), dtype=np.int16)
        index[order[:step]] = np.repeat(prefix, counts, axis=0).T
        indices.append(index)
        trace.append(
            {
                **row,
                "accepted": True,
                "reason": "integer_prefix",
                "distinct_prefixes": len(prefix),
            }
        )
    complete = len(indices) == plan["inputs"]
    return {
        "trace": trace,
        "indices": np.stack(indices)
        if indices
        else np.zeros((0, 16, 16), dtype=np.int16),
        "complete": complete,
        "I": indices[-1] if complete else None,
        "estimated_counts": debug,
    }


@torch.no_grad()
def run_case(root, plan, job_id, device):
    job = plan["jobs"][job_id]
    started = time.monotonic()
    data = observations(job, plan, device)
    records, arrays = [], {"order": data["order"], "axis_t": data["axis_t"]}
    for number, condition in enumerate(plan["conditions"]):
        key = f"condition_{number}"
        sigma, repeat = condition["noise"], condition["replicates"]
        scale = sigma / math.sqrt(repeat)
        axis_y = data["axis_clean"] + scale * data["axis_noise"]
        slice_y = data["slice_clean"] + scale * data["slice_noise"]
        centers, axis_meta = fit_codes(
            data["axis_t"], axis_y, plan, plan["seed"] + job_id
        )
        output = census(data["slices"], slice_y, data["order"], centers, plan)
        axis_labels = repeat * axis_y.numel()
        for point in output["trace"]:
            point["label_budget"] = (
                axis_labels + repeat * point["prefix_length"] * plan["slice_points"]
            )
        record = {
            "key": key,
            **condition,
            "codebook": centers.tolist(),
            "axis": axis_meta,
            "complete": output["complete"],
            "trace": output["trace"],
            "primary_label_budget": axis_labels
            + repeat * sum(plan["slice_points"] for _ in output["trace"]),
        }
        if output["complete"]:
            weights = numeric_prefix(output["I"], centers, device)
            prediction = forward(data["validation_x"], weights, plan)
            vy = data["validation_clean"] + sigma * data["validation_noise"]
            record["validation_mse"] = float((prediction - vy).square().mean())
        arrays[key + "__axis_y"] = axis_y.cpu().numpy()
        arrays[key + "__slice_y"] = slice_y.cpu().numpy()
        arrays[key + "__indices"] = output["indices"]
        for step, count in enumerate(output["estimated_counts"]):
            arrays[key + f"__counts_step_{step + 1}"] = count
        records.append(record)
        print(
            json.dumps(
                {
                    "job": job_id,
                    "condition": condition,
                    "complete": output["complete"],
                    "prefixes": len(output["indices"]),
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


def score_output(output, truth, codes):
    if output["I"] is None:
        return {"complete": False, "exact_I": False}
    recovery, _, _ = align_indices(output["I"], truth, codes)
    return {"complete": True, **recovery}


@torch.no_grad()
def audit_case(root, plan, job_id, device):
    frozen = json.loads((root / "models_frozen.json").read_text())
    path = root / "models" / f"job_{job_id:03d}"
    for suffix in (".npz", ".json"):
        candidate = path.with_suffix(suffix)
        if digest(candidate) != frozen["sha256"][str(candidate.relative_to(root))]:
            raise RuntimeError("Primary frozen census changed")
    meta = json.loads(path.with_suffix(".json").read_text())
    job = plan["jobs"][job_id]
    data = observations(job, plan, device)
    _, truth, true_centers = make_teacher(job, plan, device)
    centers = np.sort(np.abs(true_centers))
    records, arrays = [], {"truth_I": truth, "true_C": true_centers}
    with np.load(path.with_suffix(".npz")) as observed:
        for number, source in enumerate(meta["records"]):
            key = source["key"]
            scale = source["noise"] / math.sqrt(source["replicates"])
            if not np.array_equal(
                observed[key + "__slice_y"],
                (data["slice_clean"] + scale * data["slice_noise"]).cpu().numpy(),
            ):
                raise RuntimeError("Regenerated slice observations differ")
            indices = observed[key + "__indices"]
            primary = {"I": indices[-1] if source["complete"] else None}
            record = {
                "key": key,
                "condition": plan["conditions"][number],
                "primary": score_output(primary, truth, plan["codes"]),
                "max_abs_C_error": float(
                    np.max(np.abs(np.asarray(source["codebook"]) - centers))
                ),
                "relative_C_error": float(
                    np.linalg.norm(np.asarray(source["codebook"]) - centers)
                    / np.linalg.norm(centers)
                ),
                "prefix_recovery": [],
            }
            for step, index in enumerate(indices, 1):
                partial_truth = truth.copy()
                partial_truth[data["order"][step:]] = 0
                metric, _, _ = align_indices(index, partial_truth, plan["codes"])
                record["prefix_recovery"].append({"prefix_length": step, **metric})
            for name, dictionary, targets in (
                (
                    "oracle_codes_observed",
                    centers,
                    torch.tensor(observed[key + "__slice_y"], device=device),
                ),
                ("estimated_codes_clean", source["codebook"], data["slice_clean"]),
                ("oracle_codes_clean", centers, data["slice_clean"]),
            ):
                output = census(
                    data["slices"], targets, data["order"], dictionary, plan
                )
                record[name] = {
                    **score_output(output, truth, plan["codes"]),
                    "completed_prefixes": len(output["indices"]),
                    "trace": output["trace"],
                }
                arrays[key + "__" + name + "__indices"] = output["indices"]
            if source["complete"]:
                weights = numeric_prefix(indices[-1], source["codebook"], device)
                record["primary_clean_validation_mse"] = float(
                    (
                        forward(data["validation_x"], weights, plan)
                        - data["validation_clean"]
                    )
                    .square()
                    .mean()
                )
            records.append(record)
    np.savez_compressed(path.parent / (path.name + "_audit_only.npz"), **arrays)
    save_json(
        path.parent / (path.name + "_audit.json"),
        {
            "job": job,
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
    if args.action == "freeze":
        if (args.root / "models_frozen.json").exists():
            raise RuntimeError("Primary census cohort already frozen")
        freeze(args.root, plan)
        return
    if args.action == "audit" and not (args.root / "models_frozen.json").exists():
        raise RuntimeError("Entire primary cohort must be frozen before any audit")
    for job_id in range(args.shard, len(plan["jobs"]), args.shards):
        suffix = ".json" if args.action == "run" else "_audit.json"
        if (args.root / "models" / (f"job_{job_id:03d}" + suffix)).exists():
            raise RuntimeError("Refusing to repeat completed scientific phase")
        if args.action == "run":
            run_case(args.root, plan, job_id, args.device)
        else:
            audit_case(args.root, plan, job_id, args.device)


if __name__ == "__main__":
    main()
