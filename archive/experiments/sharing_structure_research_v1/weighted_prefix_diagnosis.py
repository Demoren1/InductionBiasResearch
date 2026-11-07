"""Post-freeze factorial diagnosis of code error versus prefix observation noise."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.darts_indexator_v1.data import digest, load_archive, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .pattern_prefix import timestamp
from .unknown_readout import observations, teacher_output
from .unknown_readout_synthetic import wide_axis
from .weighted_prefix_search import extract


def run_job(job):
    root, case, condition, plan = job
    torch.set_num_threads(2)
    observed = observations(root, case, condition)
    teacher = load_archive(root / "private" / f"{case['name']}.npz")
    estimated = load_archive(
        root / "source" / f"{case['name']}_{condition}" / "bootstrap_bank.npz"
    )["axis_codes"]
    clean = dict(observed)
    zero = teacher_output(
        np.zeros((1, case["inputs"])),
        teacher["index"],
        teacher["codes"],
        teacher["readout"],
        teacher["offset"],
        case["bias"],
    )[:, 0]
    for key, xkey in [
        ("prefix_y_0", "prefix_x_0"),
        ("prefix_vy_0", "prefix_vx_0"),
        ("axis_y", "axis_x"),
        ("axis_v_y", "axis_v_x"),
    ]:
        clean[key] = (
            teacher_output(
                observed[xkey],
                teacher["index"],
                teacher["codes"],
                teacher["readout"],
                teacher["offset"],
                case["bias"],
            )
            - zero[:, None]
        )
    rows, arrays = [], {}
    for task in range(4):
        axis_y = (
            clean["axis_y"][task]
            .reshape(case["inputs"], -1, case["outputs"])
            .transpose(1, 0, 2)
        )
        axis_vy = (
            clean["axis_v_y"][task]
            .reshape(case["inputs"], -1, case["outputs"])
            .transpose(1, 0, 2)
        )
        clean_estimate, _ = wide_axis(
            clean["axis_s"],
            axis_y,
            clean["axis_vs"],
            axis_vy,
            {**plan, **case},
            plan["seed"] + task,
        )
        cells = [
            ("estimated_observed", estimated[task], observed, plan["noise"][condition]),
            (
                "true_observed",
                teacher["codes"][task],
                observed,
                plan["noise"][condition],
            ),
            ("estimated_clean", estimated[task], clean, 0.0),
            ("true_clean", teacher["codes"][task], clean, 0.0),
            ("clean_estimated_clean", clean_estimate, clean, 0.0),
        ]
        for label, code, data, sigma in cells:
            index, trace = extract(data, code, {**plan, **case}, task, 0, sigma)
            metric, _, _ = align_indices(index, teacher["index"], case["codes"])
            row = {
                "case": case["name"],
                "noise": plan["noise"][condition],
                "task": task,
                "cell": label,
                "complete": trace["complete"],
                "categorical_errors": metric["categorical_errors"],
                "mask_f1": metric["mask_f1"],
                "exact": trace["complete"] and metric["categorical_errors"] == 0,
                "axis_max_abs_error": float(
                    np.max(
                        np.abs(
                            np.sort(np.abs(code))
                            - np.sort(np.abs(teacher["codes"][task]))
                        )
                    )
                ),
                "search": trace,
            }
            rows.append(row)
            arrays[f"{label}_{task}"] = index
    path = root / "diagnosis"
    np.savez_compressed(path / f"{case['name']}_{condition}.npz", **arrays)
    save_json(path / f"{case['name']}_{condition}.json", {"rows": rows})
    brief = {
        cell: sum(r["exact"] for r in rows if r["cell"] == cell)
        for cell in sorted({r["cell"] for r in rows})
    }
    print(
        json.dumps(
            {"case": case["name"], "condition": condition, "exact_tasks": brief}
        ),
        flush=True,
    )
    return {"case": case["name"], "condition": condition, "exact_tasks": brief}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "protocol.json").read_text())
    (root / "diagnosis").mkdir(exist_ok=False)
    save_json(
        root / "diagnosis/registration.json",
        {
            "registered_utc": timestamp(),
            "source_freeze_sha256": digest(root / "source_frozen.json"),
            "design": "All six families and both noises, original bootstrap queries/order. Five paired cells: estimatedC+observed; trueC+observed; estimatedC+clean; trueC+clean; signed-axis estimation on clean responses+clean prefixes. True C and denoised labels are oracle diagnostics only. No primary I, source predictor or source choice changes. W2 remains unknown to every extractor.",
            "code_sha256": digest(Path(__file__)),
        },
    )
    jobs = [
        (root, case, condition, plan)
        for case in plan["cases"]
        for condition in range(len(plan["noise"]))
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(run_job, jobs))
    save_json(
        root / "diagnosis/summary.json", {"completed_utc": timestamp(), "rows": rows}
    )


if __name__ == "__main__":
    main()
