"""G10: global interval constraints versus quantization and pairwise graphs.

This is an exact linear-Gaussian diagnostic. It is not a substitution for the
pending nonlinear/native-competitor confirmation of the broader research goal.
"""

import argparse
import itertools
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t
from torch.nn import functional as F

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .contrasts import resolve_graph, signed_graph
from .gaussian_stats import draw_increment, fit_stats
from .interval_indexator import interval_indexator

BASE = Path(__file__).resolve().parent
METHODS = (
    "scalar_dp",
    "marginal_information_dp",
    "diagonal_hessian_dp",
    "zero_t_then_information_dp",
    "point_single_linkage",
    "same_interval_graph",
    "full_pair_f_graph",
    "global_interval_certificate",
)


def register(root):
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "stage": "G10 prospective linear-Gaussian diagnostic, not final nonlinear/native-method confirmation",
        "registered_utc": datetime.now(UTC).isoformat(),
        "seed": 26110510017,
        "families": list(range(8601, 8607)),
        "coefficient_seeds": [21, 22, 23],
        "inputs": 16,
        "outputs": 16,
        "codes": 4,
        "tasks": 1,
        "observations": [512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608],
        "regimes": ["isotropic", "axis_condition400", "rotated_condition400"],
        "noise": [0.2, 1.0],
        "alpha": 0.05,
        "alpha_scope": "Per algorithm/family/coefficient/regime/noise run: alpha/8 per registered look. Marginal intervals use E coordinates, full F graph E^2 relations. No claim of simultaneous coverage across the entire benchmark or eight different algorithms.",
        "methods": list(METHODS),
        "teacher_only": {
            "nonzero_group_sizes": [8, 8, 8, 8],
            "codes": "Four sorted independent Uniform(0.4,1.8) magnitudes, not shown to learner; signed positions and coordinate roles random. Group sizes never passed to interval/DP/graph algorithms.",
        },
        "data": "Exact nested sample-Gram Gaussian-regression sufficient-statistic simulation. Covariance axes/rotation random per family; actual sample Grams, not population substitutions. Same standardized noise across sigma regimes. Learners receive observed stats only.",
        "known": "Shape, exactly4 nonempty signed groups and a fixed zero. No teacher W/C/I, mask, active count or group sizes.",
        "selection": "No metric/HPO selected after audit. Run all prespecified sample budgets. Confidence algorithms may stop/abstain using their own observations. Point algorithms evaluated at fixed budgets; first exact budget, if reported, is oracle-audit only.",
        "transfer": {
            "tasks": 100,
            "support": [1, 4, 16, 64],
            "noise": 0.1,
            "evaluate_at": [8192, 32768, 131072],
            "policy": "Freeze whole cohort before true-I audit and fresh independent standardGaussian C generation. Fix I, refit all C from new support. Clean isotropic population MSE=sumWeightSquaredError/outputs. No source numeric state carried.",
        },
        "scientific_example": "Section17 fixed heterogeneous-interval example is illustration only, never performance evidence or tuning data.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": {
            name: digest(BASE / name)
            for name in (
                "interval_pilot.py",
                "interval_indexator.py",
                "gaussian_stats.py",
                "contrasts.py",
            )
        },
        "checks": "No software tests; necessary scientific/provenance guards and lint only.",
    }
    plan["jobs"] = [
        {"family": family, "coefficient_seed": seed, "regime": regime, "noise": sigma}
        for family, seed, regime, sigma in itertools.product(
            plan["families"], plan["coefficient_seeds"], plan["regimes"], plan["noise"]
        )
    ]
    save_json(root / "protocol.json", plan)
    shutil.copy2(BASE / "THEORY.md", root / "theory_at_registration.md")
    snapshot = root / "executed_code_snapshot"
    snapshot.mkdir()
    for name in plan["code_sha256"]:
        shutil.copy2(BASE / name, snapshot / name)
    (root / "cases").mkdir()


def teacher(family, coefficient_seed, plan, device):
    rng = np.random.default_rng(plan["seed"] + family * 19)
    index = np.zeros((plan["outputs"], plan["inputs"]), dtype=np.int16)
    positions = rng.permutation(index.size)
    begin = 0
    for code, count in enumerate(plan["teacher_only"]["nonzero_group_sizes"], 1):
        index.flat[positions[begin : begin + count]] = code * rng.choice((-1, 1), count)
        begin += count
    coefficient_rng = np.random.default_rng(
        plan["seed"] + family * 23 + coefficient_seed * 100003
    )
    codes = np.sort(coefficient_rng.uniform(0.4, 1.8, plan["codes"]))
    weights = np.r_[0.0, codes][np.abs(index)] * np.sign(index)
    return index, codes, torch.tensor(weights[None], device=device)


def design(family, regime, plan):
    if regime == "isotropic":
        return np.eye(plan["inputs"])
    values = np.geomspace(1.0, 400.0, plan["inputs"])
    values /= values.mean()
    if regime == "axis_condition400":
        rng = np.random.default_rng(plan["seed"] + family * 29)
        return np.diag(np.sqrt(values[rng.permutation(len(values))]))
    rng = np.random.default_rng(plan["seed"] + family * 31)
    q, _ = np.linalg.qr(rng.normal(size=(plan["inputs"], plan["inputs"])))
    return np.linalg.cholesky((q * values) @ q.T)


@torch.no_grad()
def weighted_signed_dp(values, importance, codes):
    """Exact scalar weighted k-means control with a fixed zero prefix."""
    roots, length = values.shape
    magnitudes, order = values.abs().double().sort(-1)
    importance = importance.gather(1, order).double()
    prefix_weight = F.pad(importance.cumsum(-1), (1, 0))
    prefix_sum = F.pad((importance * magnitudes).cumsum(-1), (1, 0))
    prefix_square = F.pad((importance * magnitudes.square()).cumsum(-1), (1, 0))
    weight = prefix_weight[:, None, :] - prefix_weight[:, :, None]
    sums = prefix_sum[:, None, :] - prefix_sum[:, :, None]
    intervals = prefix_square[:, None, :] - prefix_square[:, :, None]
    intervals -= sums.square() / weight.clamp_min(1e-30)
    intervals.clamp_min_(0)
    positions = torch.arange(length + 1, device=values.device)
    intervals.masked_fill_(
        (positions[None, :] - positions[:, None])[None] <= 0, torch.inf
    )
    cost = prefix_square
    finishes, pointers = [cost[:, -1]], []
    for _ in range(codes):
        cost, previous = (cost[:, :, None] + intervals).min(1)
        pointers.append(previous)
        finishes.append(cost[:, -1])
    groups = torch.stack(finishes).argmin(0)
    back = torch.stack(pointers)
    end = torch.full((roots,), length, dtype=torch.long, device=values.device)
    root = torch.arange(roots, device=values.device)
    coordinate = torch.arange(length, device=values.device)[None]
    labels = torch.zeros_like(magnitudes, dtype=torch.long)
    for _ in range(codes):
        active = groups > 0
        start = back[groups.clamp_min(1) - 1, root, end]
        selected = (
            active[:, None]
            & (coordinate >= start[:, None])
            & (coordinate < end[:, None])
        )
        labels = torch.where(selected, groups[:, None], labels)
        end = torch.where(active, start, end)
        groups = (groups - 1).clamp_min(0)
    signed = labels * values.gather(1, order).sign().long()
    output = torch.zeros_like(signed).scatter(1, order, signed)
    reconstruction = values.new_zeros(values.shape)
    for code in range(1, codes + 1):
        selected = labels == code
        center = (importance * magnitudes * selected).sum(-1) / (
            importance * selected
        ).sum(-1).clamp_min(1e-30)
        reconstruction += (output.abs() == code) * output.sign() * center[:, None]
    distortion = (
        (values - reconstruction).square() * importance.scatter(1, order, importance)
    ).sum(-1)
    return output.cpu().numpy().astype(np.int16), distortion.cpu().numpy()


def point_single_linkage(weights, codes):
    flat = weights.ravel()
    values = np.r_[0.0, np.abs(flat)]
    order = np.argsort(values, kind="stable")
    cuts = np.argsort(np.diff(values[order]), kind="stable")[-codes:]
    cluster = np.zeros(len(values), dtype=np.int16)
    cluster[order] = np.cumsum(np.r_[0, np.isin(np.arange(len(values) - 1), cuts)])
    return (cluster[1:] * np.sign(flat)).reshape(weights.shape).astype(np.int16)


def interval_graph(lower, upper, codes):
    low, high = lower.ravel(), upper.ravel()
    relations = {
        1: (low[:, None] <= high[None]) & (low[None] <= high[:, None]),
        -1: (low[:, None] <= -low[None]) & (-high[None] <= high[:, None]),
    }
    return resolve_graph((low <= 0) & (high >= 0), relations, codes, lower.shape)


@torch.no_grad()
def run_job(root, plan, job_id, device):
    job = plan["jobs"][job_id]
    folder = root / "cases" / f"job_{job_id:03d}"
    folder.mkdir(exist_ok=False)
    truth, codes, weights = teacher(
        job["family"], job["coefficient_seed"], plan, device
    )
    covariance_root = design(job["family"], job["regime"], plan)
    regime_index = plan["regimes"].index(job["regime"])
    rng = np.random.default_rng(
        plan["seed"]
        + job["family"] * 37
        + job["coefficient_seed"] * 100003
        + regime_index * 1009
    )
    stats, previous, records, arrays = None, 0, [], {}
    started = time.perf_counter()
    for n in plan["observations"]:
        increment = draw_increment(
            n - previous, weights, job["noise"], covariance_root, rng
        )
        stats = (
            increment
            if stats is None
            else {name: stats[name] + increment[name] for name in stats}
        )
        previous = n
        fitted, residual, inverse = fit_stats(stats)
        degrees = plan["outputs"] * (n - plan["inputs"])
        variance = residual.sum() / degrees
        estimate = fitted[0]
        diagonal = inverse[0].diag()[None].expand_as(estimate) * variance
        alpha = plan["alpha"] / len(plan["observations"])
        critical = float(t.isf(alpha / (2 * estimate.numel()), degrees))
        radius = critical * diagonal.sqrt()
        lower, upper = (
            (estimate - radius).cpu().numpy(),
            (estimate + radius).cpu().numpy(),
        )
        clean = estimate.clone()
        clean[estimate.abs() <= radius] = 0
        source_values = torch.stack(
            (
                estimate.flatten(),
                estimate.flatten(),
                estimate.flatten(),
                clean.flatten(),
            )
        )
        marginal = diagonal.flatten().reciprocal()
        hessian_diagonal = stats["gram"][0].diag()[None].expand_as(estimate).flatten()
        importance = torch.stack(
            (torch.ones_like(marginal), marginal, hessian_diagonal, marginal)
        )
        indices, loss = weighted_signed_dp(source_values, importance, plan["codes"])
        candidates = {
            method: index.reshape(estimate.shape)
            for method, index in zip(METHODS[:4], indices)
        }
        candidates["point_single_linkage"] = point_single_linkage(
            estimate.cpu().numpy(), plan["codes"]
        )
        index, metadata, _ = interval_graph(lower, upper, plan["codes"])
        if index is not None:
            candidates["same_interval_graph"] = index
        graph_meta = {"same_interval_graph": metadata}
        index, metadata, _ = signed_graph(
            fitted, inverse, variance, degrees, plan["codes"], alpha
        )
        if index is not None:
            candidates["full_pair_f_graph"] = index
        graph_meta["full_pair_f_graph"] = metadata
        index, metadata = interval_indexator(lower, upper, plan["codes"])
        if index is not None:
            candidates["global_interval_certificate"] = index
        graph_meta["global_interval_certificate"] = metadata
        for method in METHODS:
            key = f"{method}__n{n}"
            record = {
                "method": method,
                "n": n,
                "returned_index": method in candidates,
                "key": key,
            }
            if method in candidates:
                arrays[key] = candidates[method]
            if method in graph_meta:
                record["certificate"] = graph_meta[method]
            if method in METHODS[:4]:
                record["weighted_distortion"] = float(loss[METHODS.index(method)])
            records.append(record)
        arrays[f"lower__n{n}"], arrays[f"upper__n{n}"] = lower, upper
        arrays[f"fitted__n{n}"] = estimate.cpu().numpy()
        arrays[f"variance_diagonal__n{n}"] = diagonal.cpu().numpy()
    np.savez_compressed(folder / "learner_outputs.npz", **arrays)
    save_json(
        folder / "learner_records.json",
        {"job": job, "records": records, "seconds": time.perf_counter() - started},
    )
    np.savez_compressed(folder / "teacher_audit_only.npz", indices=truth, codes=codes)
    save_json(
        folder / "completed.json",
        {
            "protocol_sha256": digest(root / "protocol.json"),
            "sha256": {
                name: digest(folder / name)
                for name in (
                    "learner_outputs.npz",
                    "learner_records.json",
                    "teacher_audit_only.npz",
                )
            },
        },
    )
    print(
        json.dumps({"job": job_id, **job, "seconds": time.perf_counter() - started}),
        flush=True,
    )


def freeze(root, plan):
    paths = []
    for job_id in range(len(plan["jobs"])):
        folder = root / "cases" / f"job_{job_id:03d}"
        completed = json.loads((folder / "completed.json").read_text())
        if completed["protocol_sha256"] != digest(root / "protocol.json"):
            raise RuntimeError("Job protocol changed")
        for name, expected in completed["sha256"].items():
            if digest(folder / name) != expected:
                raise RuntimeError("Completed job changed")
        paths.extend(
            folder / name
            for name in (
                "learner_outputs.npz",
                "learner_records.json",
                "completed.json",
            )
        )
    save_json(
        root / "cohort_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(root / "protocol.json"),
            "sha256": {str(p.relative_to(root)): digest(p) for p in paths},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "freeze"))
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
    if args.action == "freeze":
        freeze(args.root, plan)
        return
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Invalid scientific shard")
    for job_id in range(args.shard, len(plan["jobs"]), args.shards):
        run_job(args.root, plan, job_id, args.device)


if __name__ == "__main__":
    main()
