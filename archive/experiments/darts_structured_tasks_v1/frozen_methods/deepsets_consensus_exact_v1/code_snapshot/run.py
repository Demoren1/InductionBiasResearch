"""Run source-only selection before opening a new held-out set."""

import argparse
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .data import BASE, create_test, digest, prepare, save_json
from .model import hard_indices
from .train import configurations, fit, save_fit, search


def refit_search(output, plan, specification, phase, device):
    folder = output / phase
    roots = json.loads((folder / "rows.json").read_text())
    with np.load(folder / "snapshots.npz") as archive:
        snapshots = dict(archive)
    rows, indices, numeric = [], [], []
    for snapshot, step in enumerate(snapshots["steps"]):
        proposed = hard_indices(
            torch.tensor(snapshots["probabilities"][snapshot])
        ).numpy()
        for number, row in enumerate(roots):
            rows.append(
                {**row, "search_step": int(step), "root": number, "snapshot": snapshot}
            )
            indices.append(proposed[number])
            numeric.append(snapshots["numeric"][snapshot, number])
    fresh_multistart = specification.get("hard_refit") == "fresh_multistart"
    if fresh_multistart:
        fresh_rows, fresh_indices = [], []
        for candidate, (row, index) in enumerate(zip(rows, indices)):
            for radius in specification["hard_refit_radii"]:
                fresh_rows.append(
                    {
                        **row,
                        "source_radius": row["radius"],
                        "radius": radius,
                        "aligned_initial_codes": False,
                        "base_candidate": candidate,
                    }
                )
                fresh_indices.append(index)
        rows, indices = fresh_rows, fresh_indices
    result = fit(
        output,
        plan,
        specification,
        rows,
        device,
        indices=np.array(indices),
        initial=None if fresh_multistart else np.array(numeric),
        restarts=specification.get("hard_refit_restarts_per_radius"),
    )
    result["indices"] = np.array(indices)
    save_fit(folder / "hard_fits", result, rows)
    best_by_seed = {}
    eligible = range(len(rows))
    if fresh_multistart:
        query_selected = {}
        for number, row in enumerate(rows):
            candidate = row["base_candidate"]
            query = float(result["query_mse"][number].min())
            if candidate not in query_selected or query < query_selected[candidate][0]:
                query_selected[candidate] = query, number
        eligible = [entry[1] for entry in query_selected.values()]
    for number in eligible:
        row = rows[number]
        key = (row["configuration"], row["seed"])
        restart = int(result["query_mse"][number].argmin())
        value = float(
            result["selection_mse"][number, restart]
            if fresh_multistart
            else result["selection_mse"][number].mean()
        )
        previous = best_by_seed.get(key)
        if previous is None or value < previous["selection_mse"]:
            best_by_seed[key] = {"row": number, "selection_mse": value, **row}
    ranking = []
    for configuration in sorted({row["configuration"] for row in rows}):
        entries = [
            entry for key, entry in best_by_seed.items() if key[0] == configuration
        ]
        ranking.append(
            {
                "configuration": configuration,
                "selection_mse": float(
                    np.mean([entry["selection_mse"] for entry in entries])
                ),
                "per_seed": entries,
            }
        )
    ranking.sort(key=lambda row: row["selection_mse"])
    save_json(folder / "ranking.json", ranking)
    if phase == "pilot":
        save_json(
            output / "pilot_selection.json",
            {
                "top_configurations": [
                    row["configuration"]
                    for row in ranking[: plan["search"]["top_configurations"]]
                ],
                "ranking": ranking,
            },
        )
    else:
        winner = min(ranking[0]["per_seed"], key=lambda row: row["selection_mse"])
        save_json(
            output / "architecture_selection.json",
            {
                "configuration": ranking[0]["configuration"],
                "mean_confirmation_selection_mse": ranking[0]["selection_mse"],
                "primary": winner,
                "ranking": ranking,
            },
        )
    print(specification["name"], phase, "BEST", ranking[0], flush=True)


def controls(output, plan, specification, device):
    directory = output / "controls"
    directory.mkdir(exist_ok=False)
    rows = []
    for optimizer in ("adam", "sgd", "sgdm"):
        rates = (
            specification["numeric_rates"]
            if optimizer == "adam"
            else [0.01, 0.03, 0.1, 0.3]
        )
        for radius, rate in itertools.product(specification["radii"], rates):
            rows.append(
                {
                    "optimizer": "sgd" if optimizer == "sgdm" else optimizer,
                    "method": optimizer,
                    "momentum": 0.9 if optimizer == "sgdm" else 0.0,
                    "numeric_lr": rate,
                    "radius": radius,
                    "seed": 71,
                    "configuration": len(rows),
                    "aligned_initial_codes": False,
                }
            )
    # Teacher assignments are read only by this isolated diagnostic control.
    with np.load(output / "teacher_audit_only.npz") as archive:
        indices = archive["indices"]
    variants = ["oracle", "dense_shared"]
    if specification["kind"] == "deepsets":
        variants.append("dense_full")
    else:
        variants.extend(["dense_moment", "dense_long"])
    selection = {}
    for variant in variants:
        local = {
            **specification,
            "head": "full" if variant == "dense_full" else "shared",
        }
        direction = None
        if variant == "dense_moment":
            with np.load(output / "source.npz") as archive:
                direction = (archive["support_x"] * archive["support_y"][:, None]).mean(
                    0
                )
                direction /= max(np.linalg.norm(direction), 1e-12)
        result = fit(
            output,
            plan,
            local,
            rows,
            device,
            indices=np.repeat(indices[None], len(rows), axis=0)
            if variant == "oracle"
            else None,
            restarts=2 if variant == "dense_long" else 8,
            steps=plan["fit"]["steps"] * 4 if variant == "dense_long" else None,
            initial_direction=direction,
        )
        if variant == "oracle":
            result["indices"] = np.repeat(indices[None], len(rows), axis=0)
        save_fit(directory / variant, result, rows)
        chosen = {}
        for method in ("adam", "sgd", "sgdm"):
            options = []
            for number, row in enumerate(rows):
                if row["method"] != method:
                    continue
                restart = int(result["query_mse"][number].argmin())
                options.append(
                    {
                        "row": number,
                        "restart": restart,
                        "selection_mse": float(
                            result["selection_mse"][number, restart]
                        ),
                        "settings": row,
                    }
                )
            chosen[method] = min(options, key=lambda row: row["selection_mse"])
        selection[variant] = chosen
        save_json(directory / "selection.json", selection)
        print(specification["name"], variant, "SELECTED", chosen, flush=True)


def fresh_codebooks(output, plan, specification, device):
    choice = json.loads((output / "architecture_selection.json").read_text())
    with np.load(output / "confirmation/hard_fits.npz") as archive:
        indices = archive["indices"][choice["primary"]["row"]]
    rows = []
    for radius, rate in itertools.product(
        specification["radii"], specification["numeric_rates"]
    ):
        rows.append(
            {
                "radius": radius,
                "numeric_lr": rate,
                "seed": 81,
                "aligned_initial_codes": False,
                "configuration": len(rows),
            }
        )
    result = fit(
        output,
        plan,
        specification,
        rows,
        device,
        indices=np.repeat(indices[None], len(rows), axis=0),
        restarts=8,
    )
    save_fit(output / "fresh_calibration", result, rows)
    if specification["kind"] == "sine":
        query_starts = result["query_mse"].argmin(-1)
        losses = result["selection_mse"][np.arange(len(rows)), query_starts]
        criterion = (
            "selection of query-selected restart, matching multistart deployment"
        )
    else:
        losses = result["selection_mse"].mean(-1)
        criterion = "mean selection over numerical restarts"
    selected = int(losses.argmin())
    evaluation_rows = [{**rows[selected], "seed": 82}]
    result = fit(
        output,
        plan,
        specification,
        evaluation_rows,
        device,
        indices=indices[None],
        restarts=plan["fit"]["fresh_restarts"],
    )
    result["indices"] = indices[None]
    save_fit(output / "fresh_evaluation", result, evaluation_rows)
    save_json(
        output / "fresh_selection.json",
        {
            "calibration_row": selected,
            "settings": evaluation_rows[0],
            "query_selected_restart": int(result["query_mse"][0].argmin()),
            "calibration_criterion": criterion,
        },
    )


def freeze(output):
    paths = [
        "source.npz",
        "specification.json",
        "protocol.json",
        "architecture_selection.json",
        "confirmation/hard_fits.npz",
        "confirmation/hard_fits.json",
        "controls/selection.json",
        "fresh_selection.json",
        "fresh_evaluation.npz",
    ]
    paths += [
        str(path.relative_to(output)) for path in (output / "controls").glob("*.npz")
    ]
    save_json(
        output / "frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "sha256": {name: digest(output / name) for name in paths},
            "test_created": False,
        },
    )
    create_test(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("deepsets", "sine"), required=True)
    parser.add_argument("--frequency", type=float, default=6)
    parser.add_argument(
        "--stage",
        choices=("prepare", "pilot", "confirmation", "controls", "fresh", "freeze"),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan = json.loads((BASE / "protocol.json").read_text())
    if args.stage == "prepare":
        print(prepare(plan, args.task, args.frequency), flush=True)
        return
    name = args.task if args.task == "deepsets" else f"sine_{int(args.frequency)}"
    output = Path(plan["output"]) / name
    plan = json.loads((output / "protocol.json").read_text())
    spec = json.loads((output / "specification.json").read_text())
    device = torch.device(args.device)
    snapshot = output / "execution_snapshots" / args.stage
    snapshot.mkdir(parents=True, exist_ok=False)
    for name in ("run.py", "train.py", "model.py", "data.py"):
        shutil.copy2(BASE / name, snapshot / name)
    save_json(
        snapshot / "manifest.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "sha256": {p.name: digest(p) for p in snapshot.glob("*.py")},
        },
    )
    if args.stage in ("pilot", "confirmation"):
        candidates = configurations(spec)
        if args.stage == "confirmation":
            selected = json.loads((output / "pilot_selection.json").read_text())[
                "top_configurations"
            ]
            candidates = [row for row in candidates if row["configuration"] in selected]
        rows = [
            {**row, "seed": seed}
            for row in candidates
            for seed in plan["search"][args.stage + "_seeds"]
        ]
        search(output, plan, spec, rows, args.stage, device)
        refit_search(output, plan, spec, args.stage, device)
    elif args.stage == "controls":
        controls(output, plan, spec, device)
    elif args.stage == "fresh":
        fresh_codebooks(output, plan, spec, device)
    else:
        freeze(output)


if __name__ == "__main__":
    main()
