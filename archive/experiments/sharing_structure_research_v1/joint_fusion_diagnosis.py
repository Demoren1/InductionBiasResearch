"""Post-selection G12 pilot diagnosis, before proposing weight corrections."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.evaluate import alignment

from .gaussian_stats import sample_index


def diagnose(root):
    plan = json.loads((root / "protocol.json").read_text())
    frozen = json.loads((root / "hpo_frozen.json").read_text())
    source = Path(plan["source_root"])
    source_plan = json.loads((source / "protocol.json").read_text())
    records, distances = [], []
    for job_id, job in enumerate(plan["jobs"]):
        path = root / "pilot" / f"job_{job_id:03d}.npz"
        if digest(path) != frozen["input_sha256"][str(path.relative_to(root))]:
            raise RuntimeError("Frozen pilot changed")
        family = source_plan["jobs"][job["source_job"]]["family"]
        truth = sample_index(family["id"], family["kind"], plan, "cpu").numpy()
        with np.load(path) as a:
            values = dict(a)
        metadata = json.loads(path.with_suffix(".json").read_text())
        row = frozen["row"]
        column = int(values["selection_mse"][row].argmin())
        index = values["indices"][row, column]
        recovery, _ = alignment(index, truth, plan["codes"])
        step = int(values["selected_step"][row, column])
        selected = next(h for h in metadata["history"] if h["step"] == step)
        records.append(
            {
                "job_id": job_id,
                "family": family,
                "n": job["n"],
                "selection_mse": float(values["selection_mse"][row, column]),
                "lambda": plan["lambdas"][column],
                "recovery": recovery,
                "false_negative_by_truecode": {
                    str(k): int(((np.abs(truth) == k) & (index == 0)).sum())
                    for k in range(1, plan["codes"] + 1)
                },
                "selected_step": step,
                "relative_primal_dual_gap": selected["relative_primal_dual_gap"][row],
                "primal_residual": selected["primal_residual"][row],
                "dual_residual": selected["dual_residual"][row],
            }
        )
        with np.load(
            source / "statistics" / f"job_{job['source_job']:03d}_n{job['n']}.npz"
        ) as a:
            gram, cross, square = a["gram"], a["cross"], a["square_y"]
        inverse = np.linalg.inv(gram)
        estimate = np.linalg.solve(gram, cross.swapaxes(-1, -2)).swapaxes(-1, -2)
        variance = (square - (estimate * cross).sum(-1)).sum() / (
            plan["tasks"] * plan["outputs"] * (job["n"] - plan["inputs"])
        )
        flat = estimate.reshape(plan["tasks"], -1)
        labels = truth.ravel()
        entries = len(labels)
        left, right = np.triu_indices(entries, 1)
        row_left, row_right = left // plan["inputs"], right // plan["inputs"]
        il, ir = left % plan["inputs"], right % plan["inputs"]
        diagonal = np.tile(inverse.diagonal(axis1=-2, axis2=-1), (1, plan["outputs"]))
        off = inverse[:, il, ir] * (row_left == row_right)[None]
        for sign in (1, -1):
            observed = (flat[:, left] - sign * flat[:, right]) ** 2
            expected_noise = variance * (
                diagonal[:, left] + diagonal[:, right] - 2 * sign * off
            ).mean(0)
            raw = observed.mean(0)
            corrected = raw - expected_noise
            equal_nonzero = (labels[left] == sign * labels[right]) & (labels[left] != 0)
            weak_vs_zero = (
                (np.abs(labels[left]) == plan["codes"]) & (labels[right] == 0)
            ) | ((np.abs(labels[right]) == plan["codes"]) & (labels[left] == 0))
            for name, selected_relations in (
                ("true_nonzero_equalities", equal_nonzero),
                ("weak_code_vs_zero", weak_vs_zero),
            ):
                distances.append(
                    {
                        "job_id": job_id,
                        "n": job["n"],
                        "family": family["id"],
                        "relation": name,
                        "sign": sign,
                        "pairs": int(selected_relations.sum()),
                        "median_observed_squared_rms": float(
                            np.median(raw[selected_relations])
                        )
                        if selected_relations.any()
                        else None,
                        "median_expected_noise": float(
                            np.median(expected_noise[selected_relations])
                        )
                        if selected_relations.any()
                        else None,
                        "median_noise_corrected_squared_rms": float(
                            np.median(corrected[selected_relations])
                        )
                        if selected_relations.any()
                        else None,
                    }
                )
    save_json(
        root / "pilot_diagnosis.json",
        {
            "status": "Post-HPO frozen pilot audit, not independent confirmation; no target data used",
            "records": records,
            "distance_diagnosis": distances,
        },
    )
    print(
        json.dumps(
            {
                "selected_settings": frozen["settings"],
                "pilot": records,
                "distance_examples": distances[:8],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    torch.set_num_threads(1)
    diagnose(parser.parse_args().root)


if __name__ == "__main__":
    main()
