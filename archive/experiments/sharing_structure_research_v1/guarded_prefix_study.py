"""G31: locate prefix loss and compare paired scoring with larger pools."""

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.darts_indexator_v1.data import digest, load_archive, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .guarded_prefix import extract
from .pattern_prefix import timestamp
from .unknown_readout import observations

BASE = Path(__file__).resolve().parent
PARENT = BASE / "results/20261106/unknown_readout_synthetic_g30"
MODES = ("old", "minimum", "paired", "keep1", "keep2", "keep4")


def prefix_audit(prefixes, teacher, code, order, data, task, spec):
    true_c = teacher["codes"][task]
    ranks = np.argsort(np.argsort(np.abs(true_c))) + 1
    mapping = np.r_[0, np.sign(true_c).astype(int) * ranks]
    truth = np.sign(teacher["index"]) * mapping[np.abs(teacher["index"])]
    out = []
    for step, found in enumerate(prefixes, 1):
        patterns, inverse = np.unique(
            truth[order[:step]].T, axis=0, return_inverse=True
        )
        active = np.any(patterns, axis=1)
        amplitudes = np.zeros((len(patterns), spec["outputs"]))
        np.add.at(amplitudes, inverse, teacher["readout"][task])
        retained = {tuple(row) for row in found}
        missing = [
            i
            for i, row in enumerate(patterns)
            if active[i] and tuple(row) not in retained
        ]
        losses = []
        if missing:
            selected = np.flatnonzero(active)
            true_u = np.sort(np.abs(true_c))
            weights = (
                np.sign(patterns[selected])
                * np.r_[0, true_u][np.abs(patterns[selected])]
            )
            block = slice(
                (step - 1) * spec["slice_points"], step * spec["slice_points"]
            )
            x = data["prefix_x_0"][block][:, order[:step]]
            matrix = (
                np.maximum(x @ weights.T + spec["bias"], 0) - max(spec["bias"], 0)
            ) / math.sqrt(spec["hidden"])
            for i in missing:
                column = int(np.flatnonzero(selected == i)[0])
                others = np.delete(matrix, column, axis=1)
                residual = (
                    matrix[:, column]
                    - others
                    @ np.linalg.lstsq(others, matrix[:, column], rcond=1e-12)[0]
                )
                losses.append(float(np.mean(residual**2) * np.mean(amplitudes[i] ** 2)))
        out.append(
            {
                "step": step,
                "true_prefixes": int(active.sum()),
                "retained_true": int(active.sum()) - len(missing),
                "missing": len(missing),
                "recall": (int(active.sum()) - len(missing))
                / max(int(active.sum()), 1),
                "minimum_missing_amplitude": None
                if not missing
                else float(min(np.linalg.norm(amplitudes[i]) for i in missing)),
                "minimum_missing_deletion_mse": min(losses, default=None),
            }
        )
    return out


def run_job(job):
    root, case, condition, plan = job
    torch.set_num_threads(2)
    data = observations(PARENT, case, condition)
    code = load_archive(
        PARENT / "source" / f"{case['name']}_{condition}" / "bootstrap_bank.npz"
    )["axis_codes"]
    arrays, rows, supports = {}, [], {}
    for task in range(plan["sources"]):
        for mode in MODES:
            index, trace, prefixes = extract(
                data,
                code[task],
                {**plan, **case},
                task,
                0,
                plan["noise"][condition],
                mode,
            )
            key = f"{mode}_{task}"
            arrays[key] = index
            supports[key] = prefixes
            for step, values in enumerate(prefixes, 1):
                arrays[f"{key}_prefix_{step}"] = values
            rows.append({"key": key, "task": task, "mode": mode, **trace})
    path = root / "pilot" / f"{case['name']}_{condition}"
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    # Search decisions above used observations only. Teacher is opened after saving.
    teacher = load_archive(PARENT / "private" / f"{case['name']}.npz")
    for row in rows:
        metric, _, _ = align_indices(
            arrays[row["key"]], teacher["index"], case["codes"]
        )
        row.update(
            exact=row["complete"] and metric["categorical_errors"] == 0,
            categorical_errors=metric["categorical_errors"],
            mask_f1=metric["mask_f1"],
        )
        row["prefix_audit"] = prefix_audit(
            supports[row["key"]],
            teacher,
            code[row["task"]],
            data["order"],
            data,
            row["task"],
            {**plan, **case},
        )
        row["first_loss"] = next(
            (r["step"] for r in row["prefix_audit"] if r["missing"]), None
        )
    save_json(path.with_suffix(".json"), {"rows": rows})
    result = {
        mode: sum(r["exact"] for r in rows if r["mode"] == mode) for mode in MODES
    }
    print(
        json.dumps({"case": case["name"], "condition": condition, "exact": result}),
        flush=True,
    )
    return {"case": case["name"], "condition": condition, "exact_tasks": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(exist_ok=False)
    (root / "pilot").mkdir()
    plan = json.loads((PARENT / "protocol.json").read_text())
    plan.update(
        stage="G31 paired validation, prefix visibility and indexed W2 investigation",
        registered_utc=timestamp(),
        parent_root=str(PARENT),
        parent_freeze_sha256=digest(PARENT / "source_frozen.json"),
        modes=list(MODES),
        pilot_policy="Open exploratory G30 replay, all six geometries and both noises. Old baseline is exactly imported. minimum/paired refit small supports stably; keep1/2/4 use noise-scaled ridge and retain H/2H/4H intermediate atoms, exactly at most H final atoms, no minimum-cardinality noise tolerance. All prefixes saved before teacher audit. Follow with newly frozen unseen geometries and free/tied heads, then optional functional J search. No teacher roles or head group counts to learner.",
    )
    save_json(root / "protocol.json", plan)
    for name in ("guarded_prefix.py", "guarded_prefix_study.py"):
        (root / "code_snapshot").mkdir(exist_ok=True)
        (root / "code_snapshot" / name).write_text((BASE / name).read_text())
    jobs = [
        (root, case, condition, plan)
        for case in plan["cases"]
        for condition in range(len(plan["noise"]))
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_job, jobs))
    save_json(
        root / "pilot_summary.json", {"completed_utc": timestamp(), "rows": results}
    )


if __name__ == "__main__":
    main()
