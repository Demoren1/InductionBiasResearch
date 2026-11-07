"""Post-freeze distinction between outer-objective overfit and missed structures."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.linear import (
    adaptation_errors,
    adapted,
    index_basis,
)

from .run import METHODS, cohort, result_path


@torch.no_grad()
def audit(root, device):
    plan = json.loads((root / "protocol.json").read_text())
    frozen = root / "models_frozen.json"
    records, spectra = [], []
    for family in plan["confirmation_families"]:
        folder = root / "hidden" / f"family_{family}"
        with np.load(folder / "teacher_audit_only.npz") as archive:
            truth = archive["indices"]
        for tasks in plan["task_budgets"]:
            data = cohort(folder, tasks, device)
            indices, keys = [truth], [("oracle", 0)]
            for method in METHODS:
                for seed in plan["confirmation_seeds"]:
                    with np.load(
                        result_path(folder, "confirmation", tasks, method, seed)
                    ) as archive:
                        indices.append(archive["indices"][0])
                    keys.append((method, seed))
            basis = index_basis(
                torch.tensor(np.stack(indices), device=device), plan["hidden"]["codes"]
            )
            scores = adaptation_errors(
                basis,
                data["train_support_x"],
                data["train_support_y"],
                data["train_query_x"],
                data["train_query_y"],
            ).mean(-1)
            x = torch.cat((data["train_support_x"], data["train_query_x"]), 1)
            y = torch.cat((data["train_support_y"], data["train_query_y"]), 1)
            full = adaptation_errors(basis, x, y, x, y).mean(-1)
            centers = adapted(basis[:1].double(), x.double(), y.double())[0]
            singular = torch.linalg.svdvals(centers)
            distances = []
            for left in range(centers.shape[-1]):
                distances.append(centers[:, left].square().mean())
                for right in range(left):
                    distances.extend(
                        (
                            (centers[:, left] - centers[:, right]).square().mean(),
                            (centers[:, left] + centers[:, right]).square().mean(),
                        )
                    )
            spectra.append(
                {
                    "family": family,
                    "tasks": tasks,
                    "fitted_oracle_code_singular_values": singular.cpu().tolist(),
                    "minimum_signed_profile_squared_distance": float(
                        torch.stack(distances).min()
                    ),
                }
            )
            for i, (method, seed) in enumerate(keys):
                record = {
                    "family": family,
                    "tasks": tasks,
                    "method": method,
                    "seed": seed,
                    "selected_source_query_mse": float(scores[i]),
                    "selected_all_source_fit_mse": float(full[i]),
                    "oracle_source_query_mse": float(scores[0]),
                    "oracle_all_source_fit_mse": float(full[0]),
                    "oracle_beats_selected_source": bool(scores[0] < scores[i] - 1e-8),
                }
                if method == "query_edits":
                    state = json.loads(
                        result_path(folder, "confirmation", tasks, method, seed)
                        .with_suffix(".json")
                        .read_text()
                    )
                    record["last_search_source_query_mse"] = state["history"][-1][
                        "source_query_mse"
                    ][0]
                    record["oracle_beats_last_search_source"] = (
                        float(scores[0]) < record["last_search_source_query_mse"] - 1e-8
                    )
                records.append(record)
    save_json(
        root / "posthoc_objectives.json",
        {
            "models_frozen_sha256": digest(frozen),
            "posthoc_only": True,
            "records": records,
            "source_code_profiles": spectra,
        },
    )
    for tasks in plan["task_budgets"]:
        for method in METHODS:
            rows = [r for r in records if r["tasks"] == tasks and r["method"] == method]
            print(
                tasks,
                method,
                "source",
                float(np.mean([r["selected_source_query_mse"] for r in rows])),
                "source_fit",
                float(np.mean([r["selected_all_source_fit_mse"] for r in rows])),
                "oracle_better",
                sum(r["oracle_beats_selected_source"] for r in rows),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit(args.root, args.device)


if __name__ == "__main__":
    main()
