"""Select random-search structures by held-out source-task performance."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from experiments.mlp_structure_discovery_v1.extract import select_settings
from experiments.mlp_structure_discovery_v1.fit import fit_models, read_observations
from experiments.random_structure_search_v1.search import decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="experiments/random_structure_search_v1/protocol.json"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    root = Path(plan["output"])
    output = root / "selection"
    output.mkdir(parents=True, exist_ok=False)
    if not (root / "search" / "complete.json").exists():
        raise RuntimeError("Random structural search is incomplete")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    with np.load(root / "search" / "shortlist.npz") as archive:
        tokens = archive["tokens"]
    records = json.loads((root / "search" / "shortlist.json").read_text())
    candidates = np.stack(
        [decode(value, row["kind"]) for value, row in zip(tokens, records)]
    )
    candidates = np.concatenate(
        [candidates, decode(np.ones(tokens.shape[1:], dtype=int), "mask")[None]]
    )
    descriptions = [dict(row) for row in records] + [
        {"kind": "dense", "generation": -1}
    ]
    data = read_observations(
        plan["source_data"], plan["selection_tasks"], plan["transfer_support"]
    )
    configs = list(itertools.product(plan["learning_rates"], plan["weight_decays"]))
    rows = [
        {
            "candidate": candidate,
            "config": config,
            "task": task,
            "restart": restart,
            "lr": rate,
            "weight_decay": decay,
        }
        for candidate in range(len(candidates))
        for config, (rate, decay) in enumerate(configs)
        for task in range(len(data["support_y"]))
        for restart in range(plan["selection_restarts"])
    ]
    arrays = fit_models(
        data,
        candidates[[row["candidate"] for row in rows]],
        rows,
        plan,
        output / "candidate_fit",
        device,
        save_parameters=False,
    )
    grid = select_settings(arrays, rows)
    chosen = {}
    for kind in ["mask", "coded", "dense"]:
        eligible = [
            row for row in grid if descriptions[row["candidate"]]["kind"] == kind
        ]
        chosen[kind] = dict(min(eligible, key=lambda row: row["query_loss"]))
        chosen[kind]["structure"] = descriptions[chosen[kind]["candidate"]]
    for kind in ["mask", "coded"]:
        initial = [
            row
            for row in grid
            if descriptions[row["candidate"]]["kind"] == kind
            and descriptions[row["candidate"]]["generation"] == 0
        ]
        chosen["initial_" + kind] = dict(
            min(initial, key=lambda row: row["query_loss"])
        )
    selected_kind = min(["mask", "coded"], key=lambda kind: chosen[kind]["query_loss"])
    selected = chosen[selected_kind]
    selected_id = selected["candidate"]
    eligible = [i for i, row in enumerate(records) if row["kind"] == selected_kind]
    eligible.sort(key=lambda i: records[i]["fitness"])
    retained = [selected_id] + [i for i in eligible if i != selected_id][
        : plan["bank_records"] - 1
    ]
    source = read_observations(
        plan["source_data"], plan["discovery_tasks"], plan["transfer_support"]
    )
    bank_rows = [
        {
            "candidate": candidate,
            "task": task,
            "restart": 0,
            "lr": selected["lr"],
            "weight_decay": selected["weight_decay"],
        }
        for candidate in range(len(retained))
        for task in range(len(source["support_y"]))
    ]
    bank_indices = np.repeat(candidates[retained], len(source["support_y"]), axis=0)
    refit = fit_models(
        source,
        bank_indices,
        bank_rows,
        plan,
        output / "bank_refit",
        device,
        save_parameters=False,
    )
    losses = refit["query_loss"].reshape(len(retained), -1)
    np.savez_compressed(
        output / "bank.npz",
        tokens=tokens[retained],
        best=tokens[selected_id],
        source_candidate_ids=np.array(retained),
        source_query_loss=losses,
        source_query_accuracy=refit["query_accuracy"].reshape(len(retained), -1),
    )
    result = {
        "kind": selected_kind,
        "chosen": chosen,
        "grid": grid,
        "bank_records": len(retained),
        "bank_unique": len(np.unique(tokens[retained], axis=0)),
        "bank_mean_source_bce": float(losses.mean()),
    }
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "SELECTED",
        json.dumps({key: value for key, value in result.items() if key != "grid"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
