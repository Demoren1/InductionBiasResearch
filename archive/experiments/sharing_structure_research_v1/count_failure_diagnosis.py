"""Diagnose frozen true-C integer failures without changing any fitted I."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .count_metric import source_data
from .prefix_census import forward, numeric_prefix


def canonical_truth(index, code):
    order = np.argsort(np.abs(code))
    rename = np.zeros(len(code) + 1, dtype=np.int16)
    rename[order + 1] = np.arange(1, len(code) + 1) * np.sign(code[order])
    return rename[np.abs(index)] * np.sign(index)


def column_multiset(index, order, step):
    return sorted(map(tuple, index[order[:step]].T))


def extension_counts(index, order, step, prefixes, choices):
    lookup = {tuple(row): i for i, row in enumerate(prefixes)}
    labels = {int(value): i for i, value in enumerate(choices)}
    counts = np.zeros((len(prefixes), len(choices)), dtype=np.int16)
    for column in index[order[:step]].T:
        counts[lookup[tuple(column[:-1])], labels[int(column[-1])]] += 1
    return counts


@torch.no_grad()
def diagnose(root, device):
    destination = root / "report/failure_diagnosis.json"
    if destination.exists():
        raise RuntimeError("Diagnosis already saved")
    plan = json.loads((root / "protocol.json").read_text())
    final = root / "numeric_continuation/face_stationarity"
    if not (final / "audit.json").exists():
        raise RuntimeError("Whole primary audit must precede teacher diagnosis")
    records = []
    with np.load(root / "audit_only.npz") as truth:
        for job_id, job in enumerate(plan["jobs"]):
            path = final / "models" / f"job_{job_id:03d}_oracle"
            meta = json.loads(path.with_suffix(".json").read_text())
            source = next(
                r
                for r in meta["records"]
                if r["noise"] == 2.0 and r["method"] == "qp_integer"
            )
            if source["exact_I"]:
                continue
            with np.load(path.with_suffix(".npz")) as data:
                indices = data[source["key"] + "__indices"]
            expected = canonical_truth(
                truth[f"job_{job_id}__truth_I"], truth[f"job_{job_id}__true_C"]
            )
            centers = np.sort(np.abs(truth[f"job_{job_id}__true_C"]))
            old, targets, vx, vy, _ = source_data(plan, job_id, device)
            order = old["order"]
            step = next(
                s
                for s, index in enumerate(indices, 1)
                if column_multiset(index, order, s)
                != column_multiset(expected, order, s)
            )
            if step == 1:
                prefix, totals = (
                    np.zeros((1, 0), dtype=np.int16),
                    np.array([plan["hidden"]]),
                )
            else:
                previous = indices[step - 2][order[: step - 1]].T
                unique, first, counts = np.unique(
                    previous, axis=0, return_index=True, return_counts=True
                )
                appearance = np.argsort(first)
                prefix, totals = unique[appearance], counts[appearance]
            choices = np.r_[
                0, np.arange(1, plan["codes"] + 1), -np.arange(1, plan["codes"] + 1)
            ]
            true_counts = extension_counts(expected, order, step, prefix, choices)
            found_counts = extension_counts(
                indices[step - 1], order, step, prefix, choices
            )
            x, y = old["slices"][step - 1], targets[2.0][step - 1]
            base = (
                x[:, order[: step - 1]] @ numeric_prefix(prefix, centers, device).T
                + plan["bias"]
            )
            atom = (
                base[..., None]
                + x[:, order[step - 1], None, None]
                * numeric_prefix(choices[1:], centers, device)
            ).relu()
            matrix = (atom - base.relu()[..., None]).flatten(1) / math.sqrt(
                plan["hidden"]
            )
            target = y - base.relu() @ y.new_tensor(totals) / math.sqrt(plan["hidden"])
            a = y.new_tensor(true_counts[:, 1:].flatten())
            b = y.new_tensor(found_counts[:, 1:].flatten())
            delta = b - a
            gram, cross = matrix.T @ matrix / len(y), matrix.T @ target / len(y)
            quadratic = float((gram @ a - cross) @ delta + 0.5 * delta @ gram @ delta)
            direct = float(
                0.5
                * (
                    (matrix @ b - target).square().mean()
                    - (matrix @ a - target).square().mean()
                )
            )
            selected = indices[step - 1]
            true_prefix = expected.copy()
            true_prefix[order[step:]] = 0
            validation_x, validation_y = vx[step - 1], vy[2.0][step - 1]
            found_prediction = forward(
                validation_x, numeric_prefix(selected, centers, device), plan
            )
            true_prediction = forward(
                validation_x, numeric_prefix(true_prefix, centers, device), plan
            )
            record = {
                "job_id": job_id,
                "job": job,
                "first_numeric_wrong_prefix": step,
                "observed_half_mse_found_minus_truth": direct,
                "quadratic_difference": quadratic,
                "identity_absolute_error": abs(direct - quadratic),
                "noiseless_design_mse_separation": float(
                    (matrix @ delta).square().mean()
                ),
                "validation_mse_found_minus_truth": float(
                    (found_prediction - validation_y).square().mean()
                    - (true_prediction - validation_y).square().mean()
                ),
                "integer_metadata": source["trace"][step - 1]["integer"],
                "chosen_counts": found_counts.tolist(),
                "true_counts": true_counts.tolist(),
                "interpretation": "local integer barrier: teacher feasible and has lower observed loss"
                if direct > 0
                else "noise misranking: chosen beats teacher observed loss, not proof of global integer optimum",
                "oracle_artifact_sha256": digest(path.with_suffix(".json")),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    save_json(
        destination,
        {
            "records": records,
            "scope": "Post-freeze true-C diagnosis; no source fit or structure changed",
            "code_sha256": digest(Path(__file__)),
            "final_source_freeze_sha256": digest(final / "models_frozen.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    diagnose(args.root, args.device)


if __name__ == "__main__":
    main()
