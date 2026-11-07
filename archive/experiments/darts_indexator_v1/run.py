"""Register, select and freeze DARTS-style indexator experiments."""

import argparse
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .data import (
    BASE,
    canonical_indices,
    digest,
    load_archive,
    prepare,
    save_json,
)
from .fit import fit_fixed
from .search import roots


def collect(plan, output, phase):
    candidates, records, lookup = [], [], {}
    count = 0
    for folder in sorted((output / phase / "search").glob("shard_*")):
        json.loads((folder / "complete.json").read_text())
        specifications = json.loads((folder / "roots.json").read_text())
        count += len(specifications)
        snapshots = load_archive(folder / "snapshots.npz")
        for snapshot, step in enumerate(snapshots["steps"]):
            for root, row in enumerate(specifications):
                indices = canonical_indices(
                    snapshots["alpha"][snapshot, root].argmax(-1)
                )
                key = indices.tobytes()
                if key not in lookup:
                    lookup[key] = len(candidates)
                    candidates.append(indices)
                    records.append({"candidate": len(records), "origins": []})
                origin = {
                    **row,
                    "step": int(step),
                    "shard": str(folder.relative_to(output)),
                    "snapshot": snapshot,
                    "root": root,
                    "soft_query": snapshots["soft_query"][snapshot, root]
                    .mean((0, 1))
                    .tolist(),
                    "inherited_hard_query": snapshots["inherited_hard_query"][
                        snapshot, root
                    ]
                    .mean((0, 1))
                    .tolist(),
                }
                records[lookup[key]]["origins"].append(origin)
    if count != len(roots(plan, output, phase)):
        raise ValueError("All predeclared search roots must finish before selection")
    destination = output / phase
    np.savez_compressed(destination / "candidates.npz", indices=np.stack(candidates))
    save_json(destination / "candidates.json", records)
    if phase == "pilot":
        settings = [
            {"lr": rate, "norm": norm}
            for rate, norm in itertools.product(
                plan["fit"]["learning_rates"], plan["fit"]["initial_norms"]
            )
        ]
        restarts = plan["fit"]["pilot_restarts_per_setting"]
    else:
        selection = json.loads((output / "pilot_selection.json").read_text())
        settings = [selection["numeric_setting"]]
        restarts = plan["fit"]["selection_restarts"]
    rows = [
        {
            "candidate": candidate,
            "task": task,
            "setting": setting_id,
            "restart": restart,
            "kind": "shared",
            **setting,
        }
        for candidate in range(len(candidates))
        for task in range(len(plan["selection_patterns"]))
        for setting_id, setting in enumerate(settings)
        for restart in range(restarts)
    ]
    save_json(destination / "fit_rows.json", rows)
    save_json(destination / "settings.json", settings)
    print("CANDIDATES", phase, len(candidates), "FRESH FITS", len(rows), flush=True)


def calibrate(plan, output, phase, device, shard, shards):
    folder = output / phase
    rows = json.loads((folder / "fit_rows.json").read_text())
    ids = [
        number for number, row in enumerate(rows) if row["candidate"] % shards == shard
    ]
    selected = [rows[number] for number in ids]
    candidates = load_archive(folder / "candidates.npz")["indices"]
    data = load_archive(output / "selection.npz")
    seed = plan["seed"] + (8000001 if phase == "pilot" else 9000001)
    result = fit_fixed(
        data,
        candidates[[row["candidate"] for row in selected]],
        selected,
        plan["fit"],
        seed,
        device,
    )
    np.savez_compressed(folder / f"fits_{shard}.npz", row_ids=np.array(ids), **result)
    print("FITTED", phase, shard, len(selected), float(result["seconds"]), flush=True)


def combined(folder):
    pieces = [load_archive(path) for path in sorted(folder.glob("fits_*.npz"))]
    if not pieces:
        raise ValueError("No numerical fits have completed")
    order = np.argsort(np.concatenate([piece["row_ids"] for piece in pieces]))
    rows = json.loads((folder / "fit_rows.json").read_text())
    if not np.array_equal(
        np.sort(np.concatenate([piece["row_ids"] for piece in pieces])),
        np.arange(len(rows)),
    ):
        raise ValueError("Missing or duplicated numerical fit rows")
    result = {}
    for key in pieces[0]:
        if key in ("history_steps", "seconds"):
            result[key] = pieces[0][key]
        elif key.endswith("history"):
            result[key] = np.concatenate([piece[key] for piece in pieces], axis=1)[
                :, order
            ]
        else:
            result[key] = np.concatenate([piece[key] for piece in pieces], axis=0)[
                order
            ]
    return rows, result


def select(plan, output, phase):
    folder = output / phase
    rows, result = combined(folder)
    records = json.loads((folder / "candidates.json").read_text())
    settings = json.loads((folder / "settings.json").read_text())
    ranks = []
    for candidate, setting in itertools.product(
        range(len(records)), range(len(settings))
    ):
        selected = [
            number
            for number, row in enumerate(rows)
            if row["candidate"] == candidate and row["setting"] == setting
        ]
        ranks.append(
            {
                "candidate": candidate,
                "setting": setting,
                "selection_bce": float(result["selection_metrics"][selected, 0].mean()),
                "selection_balanced_accuracy": float(
                    result["selection_metrics"][selected, 1].mean()
                ),
                "selection_recall": float(
                    result["selection_metrics"][selected, 2].mean()
                ),
            }
        )
    ranks.sort(key=lambda row: (row["selection_bce"], row["candidate"], row["setting"]))
    save_json(folder / "ranking.json", ranks)
    if phase == "pilot":
        config_scores = {}
        for rank in ranks:
            if rank["setting"] != ranks[0]["setting"]:
                continue
            for origin in records[rank["candidate"]]["origins"]:
                config_scores.setdefault(origin["id"], rank["selection_bce"])
        top = sorted(config_scores, key=lambda key: (config_scores[key], key))[
            : plan["search"]["confirmation_top_configs"]
        ]
        selection = {
            "criterion": "Mean selection BCE over tasks and all independently query-selected fresh starts",
            "best": ranks[0],
            "numeric_setting": settings[ranks[0]["setting"]],
            "top_configurations": top,
            "configuration_best_scores": config_scores,
            "target_labels_constructed": False,
        }
        save_json(output / "pilot_selection.json", selection)
    else:
        per_root = {}
        for rank in ranks:
            for origin in records[rank["candidate"]]["origins"]:
                key = (origin["id"], origin["order"], origin["search_seed"])
                per_root.setdefault(key, {**rank, "origin": origin})
        groups = []
        for config, order in sorted({key[:2] for key in per_root}):
            selected = [
                per_root[(config, order, seed)]
                for seed in plan["search"]["confirmation_seeds"]
            ]
            groups.append(
                {
                    "configuration": config,
                    "order": order,
                    "mean_selection_bce": float(
                        np.mean([row["selection_bce"] for row in selected])
                    ),
                    "seeds": selected,
                }
            )
        groups.sort(
            key=lambda row: (
                row["mean_selection_bce"],
                row["configuration"],
                row["order"],
            )
        )
        selection = {
            "groups": groups,
            "selected_group": groups[0],
            "target_labels_constructed": False,
        }
        save_json(output / "confirmation_selection.json", selection)
    print(json.dumps(selection, indent=2), flush=True)


def freeze(plan, output):
    selection = json.loads((output / "confirmation_selection.json").read_text())
    seeds = selection["selected_group"]["seeds"]
    primary = min(seeds, key=lambda row: row["selection_bce"])
    candidates = load_archive(output / "confirmation/candidates.npz")["indices"]
    np.savez_compressed(
        output / "frozen_indexators.npz",
        primary=candidates[primary["candidate"]],
        by_seed=np.stack([candidates[row["candidate"]] for row in seeds]),
        seed_ids=np.array([row["origin"]["search_seed"] for row in seeds]),
    )
    save_json(
        output / "frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "selected_group": selection["selected_group"],
            "primary": primary,
            "numeric_setting": json.loads(
                (output / "pilot_selection.json").read_text()
            )["numeric_setting"],
            "protocol_sha256": digest(output / "protocol.json"),
            "indexators_sha256": digest(output / "frozen_indexators.npz"),
            "target_labels_constructed": False,
        },
    )
    print("FROZEN", primary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "collect", "calibrate", "select", "freeze")
    )
    parser.add_argument("--protocol", type=Path, default=BASE / "protocol.json")
    parser.add_argument(
        "--output", type=Path, default=BASE / "results/pattern_20261104"
    )
    parser.add_argument("--phase", choices=("pilot", "confirmation"), default="pilot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.protocol)
        return
    torch.set_num_threads(2)
    plan = json.loads((args.output / "protocol.json").read_text())
    if args.stage == "collect":
        collect(plan, args.output, args.phase)
    elif args.stage == "calibrate":
        calibrate(
            plan,
            args.output,
            args.phase,
            torch.device(args.device),
            args.shard,
            args.shards,
        )
    elif args.stage == "select":
        select(plan, args.output, args.phase)
    else:
        freeze(plan, args.output)


if __name__ == "__main__":
    main()
