"""Harden, fine-tune existing codebooks, then audit fresh-codebook transfer."""

import argparse
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .data import BASE, canonical_indices, digest, load_archive, observations, save_json
from .fit import fit_fixed
from .run import combined


def untied(indices):
    return np.where(indices > 0, np.arange(1, 89).reshape(11, 8), 0).astype(np.int16)


def controls_prepare(plan, output):
    frozen = json.loads((output / "frozen.json").read_text())
    archive = load_archive(output / "frozen_indexators.npz")
    if frozen["indexators_sha256"] != digest(output / "frozen_indexators.npz"):
        raise ValueError("Frozen indexators were modified")
    primary = archive["primary"]
    rng = np.random.default_rng(plan["seed"] + 11000001)
    permuted = primary.copy()
    permuted[primary > 0] = rng.permutation(primary[primary > 0])
    shuffled = rng.permutation(primary.ravel()).reshape(11, 8)
    origin = frozen["primary"]["origin"]
    snapshots = load_archive(output / origin["shard"] / "snapshots.npz")
    initial = canonical_indices(snapshots["alpha"][0, origin["root"]].argmax(-1))
    rows = np.arange(11)[:, None]
    columns = np.arange(8)[None]
    oracle = np.where((rows >= columns) & (rows < columns + 4), rows - columns + 1, 0)
    oracle = canonical_indices(oracle)
    dense = np.arange(1, 89, dtype=np.int16).reshape(11, 8)
    methods = [
        {"name": "darts_primary", "kind": "shared", "indices": primary},
        {"name": "learned_binary", "kind": "untied", "indices": untied(primary)},
        {
            "name": "random_sharing_on_support",
            "kind": "shared",
            "indices": canonical_indices(permuted),
        },
        {
            "name": "random_same_code_counts",
            "kind": "shared",
            "indices": canonical_indices(shuffled),
        },
        {"name": "initial_indexator", "kind": "shared", "indices": initial},
        {"name": "dense_shared", "kind": "dense_shared", "indices": dense},
        {"name": "dense_full", "kind": "dense_full", "indices": dense},
        {"name": "oracle_shared", "kind": "shared", "indices": oracle},
        {"name": "oracle_binary", "kind": "untied", "indices": untied(oracle)},
    ]
    for seed, indices in zip(archive["seed_ids"], archive["by_seed"]):
        if int(seed) != origin["search_seed"]:
            methods.append(
                {"name": f"darts_seed_{seed}", "kind": "shared", "indices": indices}
            )
    np.savez_compressed(
        output / "audit_indexators.npz",
        indices=np.stack([row["indices"] for row in methods]),
    )
    for row in methods:
        indices = row.pop("indices")
        row["active_edges"] = int((indices > 0).sum())
        row["used_codes"] = len(np.unique(indices[indices > 0]))
        row["numeric_parameters"] = row["used_codes"] + (
            17 if row["kind"] == "dense_full" else 3
        )
    save_json(output / "audit_methods.json", methods)
    calibration = output / "controls"
    calibration.mkdir(exist_ok=False)
    specifications = [
        {"lr": rate, "norm": scale}
        for rate, scale in itertools.product(
            plan["fit"]["dense_learning_rates"], plan["fit"]["dense_initial_scales"]
        )
    ]
    selected = [
        number
        for number, method in enumerate(methods)
        if method["name"] in ("learned_binary", "dense_shared", "dense_full")
    ]
    fit_rows = [
        {
            "candidate": candidate,
            "task": task,
            "setting": setting,
            "restart": restart,
            "kind": methods[candidate]["kind"],
            **specification,
        }
        for candidate in selected
        for task in range(len(plan["selection_patterns"]))
        for setting, specification in enumerate(specifications)
        for restart in range(plan["fit"]["selection_restarts"])
    ]
    save_json(calibration / "fit_rows.json", fit_rows)
    save_json(calibration / "settings.json", specifications)
    save_json(
        output / "evaluation_protocol.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "architecture_frozen_sha256": digest(output / "frozen.json"),
            "source_codebook_fine_tuning": "Keep the selected raw hard I and inherited numeric values; reset Adam moments; use the original search numeric rate for 4096 further support steps; query selects each replica checkpoint.",
            "user_steering": "Explicit user description requested codebook fine-tuning after fixing I. This phase is separate from fresh-codebook transfer.",
            "dense_and_untied_calibration": "Six rate/scale settings, eight starts on each of the same two selection functions; mean selection BCE selects each control setting, without target labels.",
            "random_controls": "One fixed label permutation inside learned support and one fixed permutation of all positions; same code-count multiset; no target-based choice.",
            "target_labels_constructed": False,
            "software_tests_run": False,
        },
    )


def control_fit(plan, output, device):
    folder = output / "controls"
    rows = json.loads((folder / "fit_rows.json").read_text())
    indices = load_archive(output / "audit_indexators.npz")["indices"]
    result = fit_fixed(
        load_archive(output / "selection.npz"),
        indices[[row["candidate"] for row in rows]],
        rows,
        plan["fit"],
        plan["seed"] + 12000001,
        device,
    )
    np.savez_compressed(folder / "fits_0.npz", row_ids=np.arange(len(rows)), **result)
    methods = json.loads((output / "audit_methods.json").read_text())
    specifications = json.loads((folder / "settings.json").read_text())
    selected = {}
    for candidate in sorted({row["candidate"] for row in rows}):
        scores = []
        for setting, specification in enumerate(specifications):
            chosen = [
                number
                for number, row in enumerate(rows)
                if row["candidate"] == candidate and row["setting"] == setting
            ]
            scores.append(
                {
                    **specification,
                    "setting": setting,
                    "selection_bce": float(
                        result["selection_metrics"][chosen, 0].mean()
                    ),
                }
            )
        selected[methods[candidate]["name"]] = min(
            scores, key=lambda row: row["selection_bce"]
        )
    save_json(output / "control_settings.json", selected)
    print("CONTROL SETTINGS", json.dumps(selected), flush=True)


def fine_tune(plan, output, device):
    frozen = json.loads((output / "frozen.json").read_text())
    origin = frozen["primary"]["origin"]
    snapshots = load_archive(output / origin["shard"] / "snapshots.npz")
    raw_indices = snapshots["alpha"][origin["snapshot"], origin["root"]].argmax(-1)
    inherited = snapshots["numeric"][origin["snapshot"], origin["root"]]
    rows, initial = [], []
    for task in range(len(plan["source_patterns"])):
        for restart in range(plan["search"]["numeric_restarts"]):
            value = inherited[task, restart]
            parameters = np.zeros(105, dtype=np.float32)
            parameters[:4] = value[:4]
            parameters[88], parameters[96], parameters[104] = value[4:7]
            initial.append(parameters)
            rows.append(
                {
                    "candidate": 0,
                    "task": task,
                    "restart": restart,
                    "kind": "shared",
                    "lr": origin["numeric_lr"],
                    "norm": plan["search"]["initial_codebook_norm"],
                }
            )
    result = fit_fixed(
        load_archive(output / "source.npz"),
        np.broadcast_to(raw_indices, (len(rows), 11, 8)).copy(),
        rows,
        plan["fit"],
        plan["seed"] + 12500001,
        device,
        initial_parameters=np.stack(initial),
    )
    np.savez_compressed(
        output / "source_hard_fine_tune.npz",
        initial_parameters=np.stack(initial),
        **result,
    )
    save_json(output / "source_hard_fine_tune_rows.json", rows)
    save_json(
        output / "source_hard_fine_tune.json",
        {
            "origin": origin,
            "inherited_soft_query": origin["soft_query"],
            "inherited_hard_query": origin["inherited_hard_query"],
            "after_fine_tune_query": result["query_metrics"].mean(0).tolist(),
            "after_fine_tune_selection": result["selection_metrics"].mean(0).tolist(),
            "numeric_initialization": "Inherited codebooks, shared bias/readout and offset, no renaming or teacher alignment; Adam moments reset",
            "steps": plan["fit"]["steps"],
        },
    )
    print("FINE TUNED", result["query_metrics"].mean(0).tolist(), flush=True)


def target_prepare(plan, output):
    frozen = json.loads((output / "frozen.json").read_text())
    settings = json.loads((output / "control_settings.json").read_text())
    json.loads((output / "source_hard_fine_tune.json").read_text())
    methods = json.loads((output / "audit_methods.json").read_text())
    pilot = json.loads((output / "pilot_selection.json").read_text())
    pilot_indices = load_archive(output / "pilot/candidates.npz")["indices"][
        pilot["best"]["candidate"]
    ]
    archive = load_archive(output / "audit_indexators.npz")["indices"]
    methods.append(
        {
            "name": "pilot_best_reference",
            "kind": "shared",
            "active_edges": int((pilot_indices > 0).sum()),
            "used_codes": len(np.unique(pilot_indices[pilot_indices > 0])),
            "numeric_parameters": len(np.unique(pilot_indices[pilot_indices > 0])) + 3,
        }
    )
    np.savez_compressed(
        output / "audit_indexators.npz",
        indices=np.concatenate([archive, pilot_indices[None]], axis=0),
    )
    save_json(output / "audit_methods.json", methods)
    save_json(
        output / "pilot_reference_frozen.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "selection": pilot["best"],
            "criterion": "The pilot winner selected before confirmation by source selection BCE only",
            "scope": "Additional descriptive reference; the predeclared confirmation-based primary indexator remains unchanged",
            "reason": "Keep the source-selected pilot result visible when independent search seeds show variable quality",
            "target_labels_constructed": False,
        },
    )
    rows = []
    for candidate, method in enumerate(methods):
        if method["kind"] == "shared":
            setting = frozen["numeric_setting"]
        elif method["kind"] == "untied":
            setting = settings["learned_binary"]
        else:
            setting = settings[method["name"]]
        rows.extend(
            {
                "candidate": candidate,
                "task": task,
                "restart": restart,
                "kind": method["kind"],
                "lr": setting["lr"],
                "norm": setting["norm"],
            }
            for task in range(len(plan["target_patterns"]))
            for restart in range(plan["fit"]["target_restarts"])
        )
    folder = output / "target"
    folder.mkdir(exist_ok=False)
    save_json(folder / "fit_rows.json", rows)
    save_json(
        output / "evaluation_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "architecture_sha256": digest(output / "frozen_indexators.npz"),
            "all_audit_indexators_sha256": digest(output / "audit_indexators.npz"),
            "control_settings_sha256": digest(output / "control_settings.json"),
            "fit_rows_sha256": digest(folder / "fit_rows.json"),
            "all_methods": methods,
            "target_labels_constructed": False,
        },
    )
    data = observations(plan["target_patterns"], plan["seed"] + 13000001, plan)
    np.savez_compressed(output / "target_observations.npz", **data)
    save_json(
        folder / "opened.json",
        {
            "opened_utc": datetime.now(UTC).isoformat(),
            "data_sha256": digest(output / "target_observations.npz"),
        },
    )
    print("TARGET OPENED", len(rows), "fresh fits", flush=True)


def target_fit(plan, output, device, shard, shards):
    folder = output / "target"
    rows = json.loads((folder / "fit_rows.json").read_text())
    ids = [
        number for number, row in enumerate(rows) if row["candidate"] % shards == shard
    ]
    chosen = [rows[number] for number in ids]
    indices = load_archive(output / "audit_indexators.npz")["indices"]
    data = load_archive(output / "target_observations.npz")
    # The fitter only reads its third split after all optimization and query selection.
    # Here that split is the union of the 1024 disjoint target test observations.
    for axis in ("x", "y"):
        dimension = 0 if axis == "x" else 1
        data["selection_" + axis] = np.concatenate(
            [data["selection_" + axis], data["test_" + axis]], axis=dimension
        )
    result = fit_fixed(
        data,
        indices[[row["candidate"] for row in chosen]],
        chosen,
        plan["fit"],
        plan["seed"] + 14000001,
        device,
    )
    np.savez_compressed(folder / f"fits_{shard}.npz", row_ids=np.array(ids), **result)
    print("TARGET FITTED", shard, len(chosen), float(result["seconds"]), flush=True)


def score(plan, output):
    rows, result = combined(output / "target")
    methods = json.loads((output / "audit_methods.json").read_text())
    summary = {}
    for candidate, method in enumerate(methods):
        chosen = []
        for task in range(len(plan["target_patterns"])):
            ids = [
                number
                for number, row in enumerate(rows)
                if row["candidate"] == candidate and row["task"] == task
            ]
            chosen.append(ids[int(np.argmin(result["query_metrics"][ids, 0]))])
        all_ids = [
            number for number, row in enumerate(rows) if row["candidate"] == candidate
        ]
        test = result["selection_metrics"]
        summary[method["name"]] = {
            **method,
            "mean_start_test_bce": float(test[all_ids, 0].mean()),
            "mean_start_test_balanced_accuracy": float(test[all_ids, 1].mean()),
            "mean_start_test_recall": float(test[all_ids, 2].mean()),
            "selected_test_bce": float(test[chosen, 0].mean()),
            "selected_test_balanced_accuracy": float(test[chosen, 1].mean()),
            "selected_test_recall": float(test[chosen, 2].mean()),
            "selected_rows": chosen,
            "by_task": [
                {
                    "pattern": pattern,
                    "test_metrics": test[number].tolist(),
                    "restart": rows[number]["restart"],
                    "step": int(result["best_step"][number]),
                }
                for pattern, number in zip(plan["target_patterns"], chosen)
            ],
        }
    save_json(output / "target_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "controls_prepare",
            "control_fit",
            "fine_tune",
            "target_prepare",
            "target_fit",
            "score",
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=BASE / "results/pattern_20261104"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan = json.loads((args.output / "protocol.json").read_text())
    if args.stage == "controls_prepare":
        controls_prepare(plan, args.output)
    elif args.stage == "control_fit":
        control_fit(plan, args.output, torch.device(args.device))
    elif args.stage == "fine_tune":
        fine_tune(plan, args.output, torch.device(args.device))
    elif args.stage == "target_prepare":
        target_prepare(plan, args.output)
    elif args.stage == "target_fit":
        target_fit(
            plan, args.output, torch.device(args.device), args.shard, args.shards
        )
    else:
        score(plan, args.output)


if __name__ == "__main__":
    main()
