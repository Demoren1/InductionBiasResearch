"""Prospective G11: certified cores, independent codebooks, full covariance."""

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

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .contrasts import signed_graph
from .core_codebook import CORE_METHODS, independent_codebook_contrasts
from .gaussian_stats import draw_increment, fit_stats
from .interval_indexator import interval_indexator
from .interval_pilot import (
    METHODS,
    design,
    freeze,
    interval_graph,
    point_single_linkage,
    teacher,
    weighted_signed_dp,
)

BASE = Path(__file__).resolve().parent
ALL_METHODS = METHODS + CORE_METHODS


def register(root):
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "stage": "G11 prospective independent-core codebooks; exact linear Gaussian, not final nonlinear competition",
        "registered_utc": datetime.now(UTC).isoformat(),
        "seed": 26110511019,
        "families": list(range(8701, 8707)),
        "coefficient_seeds": [21, 22, 23],
        "inputs": 16,
        "outputs": 16,
        "codes": 4,
        "tasks": 1,
        "observations": [512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608],
        "regimes": ["isotropic", "axis_condition400", "rotated_condition400"],
        "noise": [0.2, 1.0],
        "alpha": 0.05,
        "methods": list(ALL_METHODS),
        "teacher_only": {
            "nonzero_group_sizes": [8, 8, 8, 8],
            "codes": "Sorted Uniform(.4,1.8), hidden from all methods; no known mask/group sizes",
        },
        "known": "Matrix shape, exactly K=4 nonempty signed groups, fixed zero; no teacher numeric state or assignments",
        "data": "Two independent nested halves with exact sampled Gaussian sufficient statistics. Each comparison uses the same total n. All-data controls pool both halves. Core methods learn S from A only, codebooks/contrasts from independent B only.",
        "alpha_scope": "Per method/run across8 looks. Core methods allocate alpha/(2*looks) to simultaneous A intervals and alpha/(2*looks) to B contrasts; all-data controls alpha/looks. No simultaneous benchmark-wide or inter-method guarantee.",
        "selection": "No HPO; all budgets prespecified. No teacher audit or target observations before whole cohort freeze. Abstentions count as failures.",
        "transfer": {
            "tasks": 100,
            "support": [1, 4, 16, 64],
            "noise": 0.1,
            "evaluate_at": [8192, 32768, 131072],
            "policy": "Fresh independent Gaussian C; fix I, fit all C from new support, discard source numeric values. Clean population MSE=sumWeightSquaredError/outputs.",
        },
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": {
            name: digest(BASE / name)
            for name in (
                "core_pilot.py",
                "core_codebook.py",
                "interval_indexator.py",
                "interval_pilot.py",
                "gaussian_stats.py",
                "contrasts.py",
            )
        },
        "checks": "No software tests; scientific invariants, provenance checks and necessary lint only",
    }
    plan["jobs"] = [
        {"family": f, "coefficient_seed": c, "regime": r, "noise": s}
        for f, c, r, s in itertools.product(
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


def intervals(fitted, inverse, residual, n, plan, alpha):
    degrees = plan["outputs"] * (n - plan["inputs"])
    variance = float(residual.sum()) / degrees
    diagonal = inverse[0].diag()[None].expand_as(fitted[0]) * variance
    critical = float(t.isf(alpha / (2 * diagonal.numel()), degrees))
    radius = critical * diagonal.sqrt()
    estimate = fitted[0]
    return (
        (estimate - radius).cpu().numpy(),
        (estimate + radius).cpu().numpy(),
        variance,
        diagonal,
        degrees,
    )


@torch.no_grad()
def run_job(root, plan, job_id, device):
    job = plan["jobs"][job_id]
    folder = root / "cases" / f"job_{job_id:03d}"
    folder.mkdir(exist_ok=False)
    truth, codes, weights = teacher(
        job["family"], job["coefficient_seed"], plan, device
    )
    covariance_root = design(job["family"], job["regime"], plan)
    regime = plan["regimes"].index(job["regime"])
    rngs = [
        np.random.default_rng(
            plan["seed"]
            + job["family"] * 37
            + job["coefficient_seed"] * 100003
            + regime * 1009
            + part * 999983
        )
        for part in range(2)
    ]
    halves, previous, arrays, records = [None, None], 0, {}, []
    started = time.perf_counter()
    for n in plan["observations"]:
        for part in range(2):
            increment = draw_increment(
                (n - previous) // 2, weights, job["noise"], covariance_root, rngs[part]
            )
            halves[part] = (
                increment
                if halves[part] is None
                else {
                    name: halves[part][name] + value
                    for name, value in increment.items()
                }
            )
        previous = n
        total = {name: halves[0][name] + halves[1][name] for name in halves[0]}
        fitted, residual, inverse = fit_stats(total)
        alpha = plan["alpha"] / len(plan["observations"])
        lower, upper, variance, diagonal, degrees = intervals(
            fitted, inverse, residual, n, plan, alpha
        )
        estimate = fitted[0]
        clean = estimate.clone()
        clean[
            (torch.tensor(lower, device=device) <= 0)
            & (torch.tensor(upper, device=device) >= 0)
        ] = 0
        source = torch.stack(
            (
                estimate.flatten(),
                estimate.flatten(),
                estimate.flatten(),
                clean.flatten(),
            )
        )
        precision = diagonal.flatten().reciprocal()
        hessian = total["gram"][0].diag()[None].expand_as(estimate).flatten()
        importance = torch.stack(
            (torch.ones_like(precision), precision, hessian, precision)
        )
        indices, distortion = weighted_signed_dp(source, importance, plan["codes"])
        candidates = {
            method: index.reshape(estimate.shape)
            for method, index in zip(METHODS[:4], indices)
        }
        candidates["point_single_linkage"] = point_single_linkage(
            estimate.cpu().numpy(), plan["codes"]
        )
        metadata = {}
        index, meta, _ = interval_graph(lower, upper, plan["codes"])
        metadata["same_interval_graph"] = meta
        if index is not None:
            candidates["same_interval_graph"] = index
        index, meta, _ = signed_graph(
            fitted, inverse, variance, degrees, plan["codes"], alpha
        )
        metadata["full_pair_f_graph"] = meta
        if index is not None:
            candidates["full_pair_f_graph"] = index
        index, meta = interval_indexator(lower, upper, plan["codes"])
        metadata["global_interval_certificate"] = meta
        if index is not None:
            candidates["global_interval_certificate"] = index
        fitted_a, residual_a, inverse_a = fit_stats(halves[0])
        low_a, high_a, _, _, _ = intervals(
            fitted_a, inverse_a, residual_a, n // 2, plan, alpha / 2
        )
        fitted_b, residual_b, inverse_b = fit_stats(halves[1])
        degrees_b = plan["outputs"] * (n // 2 - plan["inputs"])
        variance_b = float(residual_b.sum()) / degrees_b
        covariance_b = np.kron(np.eye(plan["outputs"]), inverse_b[0].cpu().numpy())
        core_candidates, core_details = independent_codebook_contrasts(
            low_a,
            high_a,
            fitted_b[0].cpu().numpy(),
            covariance_b,
            variance_b,
            degrees_b,
            plan["codes"],
            alpha / 2,
        )
        candidates.update(core_candidates)
        for method in CORE_METHODS:
            metadata[method] = {
                "core_certificate": core_details["core_certificate"],
                **core_details.get("methods", {}).get(
                    method, {"reason": core_details.get("reason")}
                ),
            }
        for method in ALL_METHODS:
            key = f"{method}__n{n}"
            row = {
                "method": method,
                "n": n,
                "key": key,
                "returned_index": method in candidates,
            }
            if method in candidates:
                arrays[key] = candidates[method]
            if method in metadata:
                row["certificate"] = metadata[method]
            if method in METHODS[:4]:
                row["weighted_distortion"] = float(distortion[METHODS.index(method)])
            records.append(row)
        arrays[f"lower__n{n}"], arrays[f"upper__n{n}"] = lower, upper
        arrays[f"fitted__n{n}"] = estimate.cpu().numpy()
        arrays[f"variance_diagonal__n{n}"] = diagonal.cpu().numpy()
        arrays[f"core_lower__n{n}"], arrays[f"core_upper__n{n}"] = low_a, high_a
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
    for name, expected in plan["code_sha256"].items():
        if digest(BASE / name) != expected:
            raise RuntimeError("Registered G11 code changed")
    for job_id in range(args.shard, len(plan["jobs"]), args.shards):
        run_job(args.root, plan, job_id, args.device)


if __name__ == "__main__":
    main()
