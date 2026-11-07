"""Evaluate frozen structural methods with freshly initialized coefficients."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.mlp_structure_discovery_v1.fit import (
    evaluate,
    fit_models,
    read_observations,
)
from experiments.random_structure_search_v1.prior import load_prior, proposals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="experiments/random_structure_search_v1/protocol.json"
    )
    parser.add_argument("--stage", choices=["audit", "target"], default="target")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    root = Path(plan["output"])
    frozen = json.loads((root / "frozen.json").read_text())
    for path, expected in frozen["sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen input changed: {path}")
    output = root / args.stage
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if args.stage == "audit":
        path, selected_tasks = plan["source_data"], plan["audit_tasks"]
    else:
        path = root / "benchmark" / "target.npz"
        with np.load(path) as archive:
            selected_tasks = list(range(len(archive["support_y"])))
    data = read_observations(path, selected_tasks, plan["transfer_support"])
    tasks = len(data["support_y"])
    restarts = plan["target_restarts"]
    bank, kind, model, probabilities = load_prior(root, plan, device)
    settings = frozen["selected"]
    methods = list(settings)
    torch.manual_seed(plan["seed"] + (920 if args.stage == "target" else 919))
    rng = np.random.default_rng(plan["seed"] + (922 if args.stage == "target" else 921))
    candidates = [
        proposals(
            method,
            settings[method]["temperature"],
            tasks * restarts,
            bank,
            kind,
            model,
            probabilities,
            rng,
        )
        for method in methods
    ]
    rows = [
        {
            "method": method,
            "task": task,
            "restart": restart,
            "lr": settings[method]["lr"],
            "weight_decay": settings[method]["weight_decay"],
        }
        for method in methods
        for task in range(tasks)
        for restart in range(restarts)
    ]
    indices = np.concatenate(candidates)
    arrays = fit_models(data, indices, rows, plan, output / "fit", device)
    selected = []
    for method in methods:
        for task in range(tasks):
            ids = np.array(
                [
                    i
                    for i, row in enumerate(rows)
                    if row["method"] == method and row["task"] == task
                ]
            )
            selected.append(int(ids[arrays["query_loss"][ids].argmin()]))
    selected = np.array(selected)
    parameters = [
        torch.as_tensor(arrays[f"parameter_{i}"][selected], device=device)
        for i in range(4)
    ]
    task_ids = torch.tensor([rows[i]["task"] for i in selected], device=device)
    # Test observations are opened only after all starts/checkpoints are chosen.
    with np.load(path) as archive:
        test_x = archive["test_x"][selected_tasks]
        test_y = archive["test_y"][selected_tasks]
    test_loss, test_accuracy = evaluate(test_x, test_y, parameters, task_ids, device)
    test_loss, test_accuracy = test_loss.cpu().numpy(), test_accuracy.cpu().numpy()
    records = []
    for position, selected_id in enumerate(selected):
        index = indices[selected_id]
        record = dict(rows[selected_id])
        record.update(
            fit_row=int(selected_id),
            query_bce=float(arrays["query_loss"][selected_id]),
            query_accuracy=float(arrays["query_accuracy"][selected_id]),
            test_bce=float(test_loss[position]),
            test_accuracy=float(test_accuracy[position]),
            nonzeros=int(np.count_nonzero(index)),
            numerical_parameters=int(
                len(np.unique(index[index > 0])) + 2 * index.shape[1] + 1
            ),
            step=int(arrays["step"][selected_id]),
        )
        records.append(record)
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        method: {
            metric: float(
                np.mean([row[metric] for row in records if row["method"] == method])
            )
            for metric in [
                "query_bce",
                "query_accuracy",
                "test_bce",
                "test_accuracy",
                "nonzeros",
                "numerical_parameters",
            ]
        }
        for method in methods
    }
    best_values = np.array(
        [row["test_accuracy"] for row in records if row["method"] == "ar"]
    )
    paired = {}
    for method in methods:
        if method == "ar":
            continue
        reference = np.array(
            [row["test_accuracy"] for row in records if row["method"] == method]
        )
        differences = best_values - reference
        paired[method] = {
            "mean_accuracy_difference": float(differences.mean()),
            "positive_tasks": int((differences > 0).sum()),
            "tasks": tasks,
        }
    (output / "summary.json").write_text(
        json.dumps(
            {
                "stage": args.stage,
                "kind": kind,
                "tasks": tasks,
                "methods": summary,
                "ar_paired": paired,
                "independence": "One source bank; target tasks are independent conditional on the frozen bank.",
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(
        output / "selected_indexators.npz",
        indices=indices[selected].astype(np.int16),
        fit_rows=selected,
        method_ids=np.repeat(np.arange(len(methods)), tasks),
        task_ids=np.tile(np.arange(tasks), len(methods)),
    )
    print("RESULT", args.stage, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
