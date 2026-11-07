"""Post-freeze G10 structural audit, fresh-C transfer and separate diagnostics."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import t
from torch.nn import functional as F

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices
from experiments.sharing_benchmark_v1.evaluate import alignment

from .core_codebook import CORE_METHODS
from .interval_indexator import interval_indexator
from .interval_pilot import interval_graph


@torch.no_grad()
def transfer_observations(truth, job, plan, device):
    seed = plan["seed"] + 900000 + job["family"] * 43 + job["coefficient_seed"] * 97
    generator = torch.Generator(device=device).manual_seed(seed)
    tasks = plan["transfer"]["tasks"]
    coefficient = torch.randn(
        (tasks, plan["codes"]), generator=generator, device=device, dtype=torch.float64
    )
    true_index = torch.tensor(truth, device=device).long()
    true_basis = (
        F.one_hot(true_index.abs(), plan["codes"] + 1)[..., 1:].double()
        * true_index.sign()[..., None]
    )
    weights = torch.einsum("oik,tk->toi", true_basis, coefficient)
    x = torch.randn(
        (tasks, max(plan["transfer"]["support"]), plan["inputs"]),
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    clean = torch.einsum("tni,toi->tno", x, weights)
    noise = torch.randn(
        clean.shape, generator=generator, device=device, dtype=torch.float64
    )
    return x, clean + plan["transfer"]["noise"] * noise, weights, seed


@torch.no_grad()
def transfer(index, x, y, truth_weights, plan):
    indices = torch.tensor(index, device=x.device).long()
    basis = (
        F.one_hot(indices.abs(), plan["codes"] + 1)[..., 1:].double()
        * indices.sign()[..., None]
    )
    features = torch.einsum("tni,oik->tnok", x, basis)
    result = {}
    for support in plan["transfer"]["support"]:
        design = features[:, :support].flatten(1, 2)
        response = y[:, :support].flatten(1, 2)
        gram = design.transpose(-1, -2) @ design
        cross = design.transpose(-1, -2) @ response[..., None]
        coefficient = (torch.linalg.pinv(gram, hermitian=True, rtol=1e-12) @ cross)[
            ..., 0
        ]
        fitted = torch.einsum("oik,tk->toi", basis, coefficient)
        errors = (fitted - truth_weights).square().sum((-1, -2)) / plan["outputs"]
        result[str(support)] = {
            "mean_mse": float(errors.mean()),
            "std_mse": float(errors.std(unbiased=True)),
            "tasks": len(errors),
        }
    return result


def audit_job(root, plan, frozen, job_id, device):
    folder = root / "cases" / f"job_{job_id:03d}"
    if (folder / "audit.json").exists():
        raise RuntimeError("G10 audit already exists")
    for name in ("learner_outputs.npz", "learner_records.json", "completed.json"):
        path = folder / name
        if digest(path) != frozen["sha256"][str(path.relative_to(root))]:
            raise RuntimeError("Frozen G10 learner changed")
    completed = json.loads((folder / "completed.json").read_text())
    if (
        digest(folder / "teacher_audit_only.npz")
        != completed["sha256"]["teacher_audit_only.npz"]
    ):
        raise RuntimeError("Teacher generation artifact changed")
    source = json.loads((folder / "learner_records.json").read_text())
    with np.load(folder / "learner_outputs.npz") as archive:
        arrays = dict(archive)
    with np.load(folder / "teacher_audit_only.npz") as archive:
        truth, codes = archive["indices"], archive["codes"]
    true_weights = np.r_[0.0, codes][np.abs(truth)] * np.sign(truth)
    x, y, target_weights, target_seed = transfer_observations(
        truth, source["job"], plan, device
    )
    first_return, first_exact, audit_arrays = {}, {}, {"truth_indices": truth}
    coverage = []
    core_coverage = []
    started = time.perf_counter()
    for n in plan["observations"]:
        covered = (arrays[f"lower__n{n}"] <= true_weights) & (
            true_weights <= arrays[f"upper__n{n}"]
        )
        coverage.append(
            {
                "n": n,
                "all_weight_intervals_cover": bool(covered.all()),
                "noncovered_positions": int((~covered).sum()),
            }
        )
        if f"core_lower__n{n}" in arrays:
            low_a, high_a = arrays[f"core_lower__n{n}"], arrays[f"core_upper__n{n}"]
            core_coverage.append(
                {
                    "n": n,
                    "all_weight_intervals_cover": bool(
                        ((low_a <= true_weights) & (true_weights <= high_a)).all()
                    ),
                }
            )
    records = []
    for record in source["records"]:
        row = dict(record)
        method, n = row["method"], row["n"]
        if row["returned_index"]:
            index = arrays[row["key"]]
            row["recovery"], aligned = alignment(index, truth, plan["codes"])
            audit_arrays[row["key"] + "__aligned"] = aligned
            first_return.setdefault(method, n)
            if row["recovery"]["exact_I"]:
                first_exact.setdefault(method, n)
            if n in plan["transfer"]["evaluate_at"] or (
                method
                in (
                    "same_interval_graph",
                    "full_pair_f_graph",
                    "global_interval_certificate",
                )
                + CORE_METHODS
                and first_return[method] == n
            ):
                row["transfer"] = transfer(index, x, y, target_weights, plan)
        records.append(row)
    save_json(
        folder / "audit.json",
        {
            "job": source["job"],
            "records": records,
            "interval_coverage": coverage,
            "independent_core_interval_coverage": core_coverage,
            "first_return": first_return,
            "first_exact_budget_oracle_audit_only": first_exact,
            "oracle_transfer": transfer(truth, x, y, target_weights, plan),
            "target_seed": target_seed,
            "cohort_frozen_sha256": digest(root / "cohort_frozen.json"),
            "seconds": time.perf_counter() - started,
        },
    )
    np.savez_compressed(folder / "aligned_indexators_audit_only.npz", **audit_arrays)
    print(
        json.dumps({"audited_job": job_id, "seconds": time.perf_counter() - started}),
        flush=True,
    )


def illustration(root):
    lower, upper = np.array([0.99, 0.1, 1.8, 2.99]), np.array([1.01, 2.2, 3.9, 3.01])
    index, global_metadata = interval_indexator(lower, upper, 2)
    graph_index, graph_metadata, _ = interval_graph(lower, upper, 2)
    save_json(
        root / "mathematical_illustration.json",
        {
            "illustration_only": True,
            "status": "Fixed theorem17 example, not training data, HPO, software test or performance evidence",
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "known_nonzero_codes": 2,
            "global_index": None if index is None else index.tolist(),
            "global_metadata": global_metadata,
            "graph_index": None if graph_index is None else graph_index.tolist(),
            "graph_metadata": graph_metadata,
        },
    )


def nonlinear_diagnostic(root, nonlinear):
    specification = json.loads((nonlinear / "protocol.json").read_text())
    records, arrays = [], {}
    entries = specification["inputs"] * specification["hidden"]
    alpha = specification["alpha_per_family"] / (
        len(specification["noise"]) * len(specification["initializations"])
    )
    degrees = specification["source"] - entries - 2
    critical = float(t.isf(alpha / (2 * entries), degrees))
    for family in specification["families"]:
        folder = nonlinear / f"family_{family}"
        audit = json.loads((folder / "audit.json").read_text())
        with np.load(folder / "learner_outputs.npz") as archive:
            outputs = dict(archive)
        with np.load(folder / "audit_only.npz") as archive:
            truth, true_weights = archive["truth_indices"], archive["truth_weights"]
        for source in audit["records"]:
            key = source["key"]
            weights = outputs[key + "__theta"][:entries].reshape(truth.shape)
            radius = critical * np.sqrt(
                np.diag(outputs[key + "__covariance"])[:entries].reshape(truth.shape)
            )
            lower, upper = weights - radius, weights + radius
            index, metadata = interval_indexator(lower, upper, specification["codes"])
            costs = ((weights[:, :, None] - true_weights[:, None, :]) ** 2).sum(0)
            source_columns, target_columns = linear_sum_assignment(costs)
            columns = source_columns[np.argsort(target_columns)]
            row = {
                "family": family,
                "initialization": source["method"],
                "noise": source["noise"],
                "g9_inference_eligible": source["numerical"]["inference_eligible"],
                "all_weight_intervals_cover_audit": bool(
                    (
                        (lower[:, columns] <= true_weights)
                        & (true_weights <= upper[:, columns])
                    ).all()
                ),
                "certificate": metadata,
            }
            if index is not None:
                row["recovery"], aligned, _ = align_indices(
                    index, truth, specification["codes"]
                )
                row["recovery"]["exact_I"] = row["recovery"]["categorical_errors"] == 0
                arrays[f"{family}__{key}"] = aligned
            records.append(row)
    save_json(
        root / "frozen_g9_interval_diagnostic.json",
        {
            "status": "Exploratory new interval solver on already-opened nonlinear G9 estimates, ignoring its registered stationarity gate solely for diagnosis. Nonlinear CI coverage unproven; no exact Gaussian/F guarantee or independent confirmation claimed.",
            "records": records,
        },
    )
    np.savez_compressed(root / "frozen_g9_interval_diagnostic_indices.npz", **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "diagnostics"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--nonlinear", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "diagnostics":
        illustration(args.root)
        if args.nonlinear is not None:
            nonlinear_diagnostic(args.root, args.nonlinear)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    frozen = json.loads((args.root / "cohort_frozen.json").read_text())
    if frozen["protocol_sha256"] != digest(args.root / "protocol.json"):
        raise RuntimeError("Frozen protocol changed")
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Invalid scientific shard")
    for job_id in range(args.shard, len(plan["jobs"]), args.shards):
        audit_job(args.root, plan, frozen, job_id, args.device)


if __name__ == "__main__":
    main()
