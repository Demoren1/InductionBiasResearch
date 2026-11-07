"""Discover arbitrary weight partitions from source solvers, without geometry."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from experiments.mlp_structure_discovery_v1.fit import (
    fit_models,
    read_observations,
    untied_indices,
)


def align_solvers(weights, readout, query_loss, task_ids):
    """Fix only the positive scaling and permutation symmetries of ReLU MLPs."""
    norms = np.linalg.norm(weights, axis=1).clip(1e-9)
    normalized = weights / norms[:, None]
    salience = np.abs(normalized)
    signs = np.sign(readout)
    reference_id = int(query_loss.argmin())
    reference = salience[reference_id].copy()
    reference_signs = signs[reference_id].copy()
    importance = np.zeros(len(weights))
    for task in np.unique(task_ids):
        ids = task_ids == task
        importance[ids] = 1 / ids.sum() / len(np.unique(task_ids))
    for _ in range(4):
        orders = []
        for values, current_signs in zip(salience, signs):
            similarity = values.T @ reference
            similarity -= 0.1 * (current_signs[:, None] != reference_signs[None])
            old, new = linear_sum_assignment(-similarity)
            order = np.empty(weights.shape[-1], dtype=int)
            order[new] = old
            orders.append(order)
        aligned = np.stack(
            [value[:, order] for value, order in zip(normalized, orders)]
        )
        reference = (np.abs(aligned) * importance[:, None, None]).sum(0)
        reference /= np.linalg.norm(reference, axis=0).clip(1e-9)
    profiles = np.stack(
        [aligned[task_ids == task].mean(0) for task in np.unique(task_ids)], -1
    )
    profiles = profiles.reshape(-1, profiles.shape[-1])
    profiles /= np.linalg.norm(profiles, axis=1, keepdims=True).clip(1e-9)
    masks = aligned != 0
    frequency = (masks * importance[:, None, None]).sum(0)
    return profiles, masks, frequency, np.array(orders), reference_id


def select_settings(arrays, rows):
    summaries = []
    combinations = sorted({(row["candidate"], row["config"]) for row in rows})
    for candidate, config in combinations:
        ids = np.array(
            [
                i
                for i, row in enumerate(rows)
                if (row["candidate"], row["config"]) == (candidate, config)
            ]
        )
        selected = []
        for task in sorted({rows[i]["task"] for i in ids}):
            task_rows = ids[[rows[i]["task"] == task for i in ids]]
            selected.append(int(task_rows[arrays["query_loss"][task_rows].argmin()]))
        summaries.append(
            {
                "candidate": candidate,
                "config": config,
                "lr": rows[ids[0]]["lr"],
                "weight_decay": rows[ids[0]]["weight_decay"],
                "query_loss": float(arrays["query_loss"][selected].mean()),
                "query_accuracy": float(arrays["query_accuracy"][selected].mean()),
                "selected_rows": selected,
            }
        )
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="experiments/mlp_structure_discovery_v1/protocol.json"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    root = Path(plan["output"])
    output = root / "extraction"
    output.mkdir(parents=True, exist_ok=False)
    if not (root / "source" / "complete.json").exists():
        raise RuntimeError("Source collection is incomplete")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    source_rows = json.loads((root / "source" / "rows.json").read_text())
    task_ids = np.array([row["task"] for row in source_rows])
    candidates, descriptions = [], []
    for level, fraction in enumerate(plan["retained_fractions"]):
        with np.load(root / "source" / f"level_{level}.npz") as raw:
            profiles, masks, frequency, orders, reference = align_solvers(
                raw["parameter_0"], raw["parameter_2"], raw["query_loss"], task_ids
            )
        shape = frequency.shape
        retained = max(1, round(frequency.size * fraction))
        chosen = np.argsort(-frequency.ravel(), kind="stable")[:retained]
        consensus_mask = np.zeros(frequency.size, dtype=bool)
        consensus_mask[chosen] = True
        consensus_mask = consensus_mask.reshape(shape)
        labels = []
        for requested in plan["code_counts"]:
            codes = min(requested, retained)
            with threadpool_limits(limits=2):
                clustering = KMeans(
                    codes, random_state=plan["seed"] + level, n_init=5
                ).fit(profiles[consensus_mask.ravel()])
                membership = clustering.predict(profiles) + 1
            # Canonical labels refer to first occurrence, never teacher roles.
            first = list(dict.fromkeys(membership[consensus_mask.ravel()].tolist()))
            first.extend(value for value in range(1, codes + 1) if value not in first)
            mapping = np.zeros(codes + 1, dtype=np.int64)
            mapping[first] = np.arange(1, codes + 1)
            membership = mapping[membership].reshape(shape)
            candidates.append(membership * consensus_mask)
            labels.append(membership)
            descriptions.append(
                {
                    "kind": "coded",
                    "level": level,
                    "fraction": fraction,
                    "requested_codes": requested,
                    "actual_codes": int(
                        np.unique(candidates[-1][candidates[-1] > 0]).size
                    ),
                    "nonzeros": retained,
                }
            )
        candidates.append(untied_indices(consensus_mask))
        descriptions.append(
            {
                "kind": "dense" if level == 0 else "mask",
                "level": level,
                "fraction": fraction,
                "nonzeros": retained,
            }
        )
        np.savez_compressed(
            output / f"level_{level}.npz",
            profiles=profiles,
            masks=masks,
            frequency=frequency,
            orders=orders,
            reference_id=reference,
            consensus_mask=consensus_mask,
            memberships=np.array(labels),
        )
        print("EXTRACTED", level, retained, flush=True)
    candidates = np.stack(candidates)
    np.savez_compressed(output / "candidates.npz", indices=candidates.astype(np.int32))
    (output / "candidates.json").write_text(json.dumps(descriptions, indent=2) + "\n")
    data = read_observations(
        plan["source_data"], plan["selection_tasks"], plan["transfer_support"]
    )
    configs = list(itertools.product(plan["learning_rates"], plan["weight_decays"]))
    rows = [
        {
            "candidate": candidate,
            "config": cid,
            "task": task,
            "restart": restart,
            "lr": rate,
            "weight_decay": decay,
        }
        for candidate in range(len(candidates))
        for cid, (rate, decay) in enumerate(configs)
        for task in range(len(data["support_y"]))
        for restart in range(plan["selection_restarts"])
    ]
    indices = candidates[[row["candidate"] for row in rows]]
    arrays = fit_models(data, indices, rows, plan, output / "selection_fit", device)
    summaries = select_settings(arrays, rows)
    chosen = {}
    for kind in ["coded", "mask", "dense"]:
        eligible = [
            row for row in summaries if descriptions[row["candidate"]]["kind"] == kind
        ]
        chosen[kind] = min(eligible, key=lambda row: row["query_loss"])
        chosen[kind]["structure"] = descriptions[chosen[kind]["candidate"]]
    coded_level = chosen["coded"]["structure"]["level"]
    same_mask = [
        row
        for row in summaries
        if descriptions[row["candidate"]]["level"] == coded_level
        and descriptions[row["candidate"]]["kind"] in {"mask", "dense"}
    ]
    chosen["same_mask"] = min(same_mask, key=lambda row: row["query_loss"])
    (output / "selection.json").write_text(
        json.dumps({"chosen": chosen, "grid": summaries}, indent=2) + "\n"
    )
    print("SELECTED", json.dumps(chosen), flush=True)
    selected = chosen["coded"]
    structure = selected["structure"]
    with np.load(output / f"level_{structure['level']}.npz") as archive:
        member_id = plan["code_counts"].index(structure["requested_codes"])
        bank_indices = archive["masks"] * archive["memberships"][member_id]
    source_data = read_observations(plan["source_data"], plan["discovery_tasks"])
    bank_rows = [
        {
            "task": row["task"],
            "restart": row["restart"],
            "lr": selected["lr"],
            "weight_decay": selected["weight_decay"],
        }
        for row in source_rows
    ]
    refit = fit_models(
        source_data, bank_indices, bank_rows, plan, output / "bank_refit", device
    )
    retained = []
    for task in range(len(source_data["support_y"])):
        positive = float(source_data["support_y"][task].mean())
        positive = np.clip(positive, 1e-6, 1 - 1e-6)
        labels = source_data["query_y"][task]
        constant_loss = float(
            -(labels * np.log(positive) + (1 - labels) * np.log(1 - positive)).mean()
        )
        ids = np.flatnonzero((task_ids == task) & (refit["query_loss"] < constant_loss))
        retained.extend(
            ids[np.argsort(refit["query_loss"][ids])[: plan["bank_keep_per_task"]]]
        )
    retained = np.array(retained, dtype=int)
    if not len(retained):
        raise RuntimeError("No compressed source solver beat a constant predictor")
    np.savez_compressed(
        output / "bank.npz",
        indices=bank_indices[retained].astype(np.int16),
        task_ids=task_ids[retained],
        source_rows=retained,
        query_loss=refit["query_loss"][retained],
        query_accuracy=refit["query_accuracy"][retained],
        consensus=candidates[selected["candidate"]].astype(np.int16),
    )
    (output / "complete.json").write_text(
        json.dumps(
            {
                "retained": len(retained),
                "unique": len(np.unique(bank_indices[retained], axis=0)),
                "selected": selected,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
