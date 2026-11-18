"""Prepare fresh targets and reuse the original numerical adapter."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
    digest,
    save_json,
)


def prepare(root):
    from experiments.paper_structure_benchmark_v1.protocol import (
        prepare_general,
    )
    from experiments.paper_structure_benchmark_v1.protocol.prepare import (
        outputs_for,
    )

    plans = []
    for number in (4, 5):
        case = f"case_{number:02d}"
        ratio_folder = root / "ratio" / f"{case}_c0"
        ratio_selection = json.loads(
            (ratio_folder / "selected.json").read_text()
        )
        g34 = dict(
            np.load(root / "fits" / f"{case}_c0" / "joint_selected.npz")
        )
        structures = {"g34": canonical_index(g34["joint_functional_index"], 4)}
        if ratio_selection["selected"] is not None:
            ratio = np.load(ratio_folder / "fitted.npz")
            structures["ratio"] = canonical_index(
                ratio["indices"][ratio_selection["selected"]], 4
            )
        aliases, unique = {}, {}
        for name in ("ratio", "g34"):
            if name not in structures:
                continue
            key = structures[name].tobytes()
            aliases[name] = unique.setdefault(key, name)
        folder = root / "targets" / f"{case}_c0"
        folder.mkdir(exist_ok=False)
        np.savez_compressed(folder / "structures.npz", **structures)
        sizes = json.loads((root / "public" / f"{case}.json").read_text())
        save_json(folder / "sizes.json", sizes)
        rng = np.random.default_rng(2611146541 + 100003 * number)
        code, bias, head, offset = prepare_general.general_numbers(
            rng, "random", number % 2, 4, sizes["outputs"]
        )
        teacher = np.load(root / "private" / f"{case}.npz")["index"]
        for name, count in (
            ("support", 1024),
            ("selection", 256),
            ("query", 2048),
        ):
            x = rng.normal(size=(4, count, 16))
            y = np.stack(
                [
                    outputs_for(
                        x[t], teacher, code[t], bias[t], head[t], offset[t]
                    )
                    for t in range(4)
                ]
            )
            if name == "query":
                np.savez_compressed(folder / "query_x.npz", x=x)
                np.savez_compressed(
                    root / "private" / f"{case}_target_answers.npz", y=y
                )
            else:
                np.savez_compressed(folder / f"{name}.npz", x=x, y=y)
        save_json(
            folder / "plan.json",
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "aliases": aliases,
                "execute": [*unique.values(), "dense"],
                "tasks": 4,
                "budgets": [64, 256, 1024],
                "adapter": "Unchanged target_fit.run: 18 Adam/2000 + 4 GN/48",
                "fresh_numbers": True,
                "deduplication": (
                    "Identical canonical structures share one numerical fit"
                ),
            },
        )
        plans.append(
            {
                "case": case,
                "aliases": aliases,
                "execute": [*unique.values(), "dense"],
            }
        )
    save_json(root / "transfer_plan.json", {"cases": plans})
    print(json.dumps(plans), flush=True)


def fit(root, case, device):
    from experiments.paper_structure_benchmark_v1.fitting.target_fit import run

    plan = json.loads(
        (root / "targets" / f"{case}_c0" / "plan.json").read_text()
    )
    for method in plan["execute"]:
        run(root, case, 0, method, device)


def score(root):
    plan = json.loads((root / "transfer_plan.json").read_text())
    frozen = {}
    for item in plan["cases"]:
        for method in item["execute"]:
            folder = root / "targets" / f"{item['case']}_c0" / method
            metrics = json.loads((folder / "metrics.json").read_text())
            if metrics["status"] != "complete":
                raise ValueError(f"Unfrozen target predictions: {folder}")
            frozen[str(folder.relative_to(root))] = digest(
                folder / "frozen_predictions.npz"
            )
    save_json(
        root / "target_prediction_freeze.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "sha256": frozen,
        },
    )
    rows = []
    for item in plan["cases"]:
        case = item["case"]
        answer = np.load(root / "private" / f"{case}_target_answers.npz")["y"]
        for method in ("ratio", "g34", "dense"):
            actual = item["aliases"].get(method, method)
            if actual not in item["execute"]:
                continue
            path = (
                root
                / "targets"
                / f"{case}_c0"
                / actual
                / "frozen_predictions.npz"
            )
            result = np.load(path)
            for budget in (64, 256, 1024):
                mse = ((result[f"prediction_{budget}"] - answer) ** 2).mean(
                    (1, 2)
                )
                rows.append(
                    {
                        "case": case,
                        "method": method,
                        "actual_fit": actual,
                        "budget": budget,
                        "mean_mse": float(mse.mean()),
                        "per_task_mse": mse.tolist(),
                    }
                )
    save_json(
        root / "transfer_results.json",
        {
            "scored_utc": datetime.now(UTC).isoformat(),
            "rows": rows,
            "source_parameters_transferred": False,
        },
    )
    print(json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("prepare", "fit", "score"), required=True
    )
    parser.add_argument("--case", choices=("case_04", "case_05"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == "prepare":
        prepare(root)
    elif args.phase == "score":
        score(root)
    else:
        if args.case is None:
            parser.error("--case is required for target fitting")
        fit(root, args.case, torch.device(args.device))


if __name__ == "__main__":
    main()
