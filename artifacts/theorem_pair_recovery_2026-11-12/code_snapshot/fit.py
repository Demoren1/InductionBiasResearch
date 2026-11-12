"""Evaluate signature candidates on source observations, with fresh numbers."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    load_sizes,
    save_json,
)
from experiments.sharing_structure_research_v1.blind_numeric import (
    fit_batch,
    numpy_predict,
)
from experiments.sharing_structure_research_v1.blind_orientation_refine import (
    initial_states,
)


class SourceBoundary:
    """Prevent numerical fitting from opening private or meta responses."""

    def __init__(self):
        self.role = "fit"
        self.opened = set()

    def audit(self, event, arguments):
        if event != "open" or not arguments or isinstance(arguments[0], int):
            return
        try:
            path = Path(os.fsdecode(arguments[0])).resolve()
        except (TypeError, ValueError, OSError):
            return
        if path.suffix not in (".npz", ".npy"):
            return
        if "private" in path.parts or path.name in (
            "test.npz",
            "selected_structures.npz",
            "scoring_answers.npz",
        ):
            raise PermissionError(f"Sealed observations: {path}")
        if path.name == "meta.npz" and self.role != "select":
            raise PermissionError(f"Meta responses unavailable during fitting: {path}")
        self.opened.add(str(path))


def evaluate(work, output, cohort, case, condition, device):
    torch.set_num_threads(2)
    started = time.monotonic()
    folder = output / cohort / f"{case}_c{condition}"
    if (folder / "functional.json").exists():
        return
    boundary = SourceBoundary()
    sys.addaudithook(boundary.audit)
    root = work / cohort
    key = f"{case}_c{condition}"
    sizes = load_sizes(root, case)
    codes, outputs = sizes["codes"], sizes["outputs"]
    candidate = dict(np.load(folder / "candidates.npz"))
    baseline = dict(
        np.load(root / "methods" / "full" / "fits" / key / "joint_fitted.npz")
    )
    indices = list(baseline["indices"])
    parameters = list(baseline["parameters"])
    validation_mse = list(baseline["validation_mse"])
    lookup = {index.tobytes(): number for number, index in enumerate(indices)}
    tasks, keep = baseline["parameters"].shape[1], 4
    sources = [
        dict(np.load(root / "source" / f"{key}_t{task}" / "learned.npz"))
        for task in range(tasks)
    ]
    panels = [
        dict(np.load(root / "public" / key / f"{name}.npz"))
        for name in ("fixed_train", "selection")
    ]
    x, y, vx, vy = [
        np.stack([panel[f"{name}_{task}"] for task in range(tasks)])
        for panel in panels
        for name in ("x", "y")
    ]
    packed = [np.repeat(value, keep, axis=0) for value in (x, y, vx, vy)]
    selected, new_count, reused = [], 0, 0
    for number, index in enumerate(candidate["indices"][:32]):
        identity = index.tobytes()
        if identity in lookup:
            selected.append(lookup[identity])
            reused += 1
            continue
        initial = np.stack(
            [
                initial_states(
                    index,
                    {
                        "codes_joint": candidate["code_guesses"][number, task],
                        **sources[task],
                    },
                    codes,
                    x[task],
                    y[task],
                    vx[task],
                    vy[task],
                    device,
                    keep,
                )
                for task in range(tasks)
            ]
        )
        fitted, trace = fit_batch(
            np.repeat(index[None], tasks * keep, axis=0),
            initial.reshape(tasks * keep, -1),
            *packed,
            codes,
            device,
            steps=48,
        )
        fitted = fitted.reshape(tasks, keep, -1)
        losses = trace["validation_mse"].reshape(tasks, keep)
        chosen = losses.argmin(1)
        selected.append(len(indices))
        lookup[identity] = len(indices)
        indices.append(index)
        parameters.append(fitted[np.arange(tasks), chosen])
        validation_mse.append(losses[np.arange(tasks), chosen])
        new_count += 1
        print(
            json.dumps(
                {
                    "case": case,
                    "condition": condition,
                    "finished": number + 1,
                    "total": min(len(candidate["indices"]), 32),
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    indices, parameters, validation_mse = map(
        np.asarray, (indices, parameters, validation_mse)
    )
    np.savez_compressed(
        folder / "functional_states.npz",
        indices=indices,
        parameters=parameters,
        validation_mse=validation_mse,
        signature_ids=np.asarray(selected, dtype=int),
        baseline_count=len(baseline["indices"]),
    )
    boundary.role = "select"
    meta = dict(np.load(root / "public" / key / "meta.npz"))
    risks = np.array(
        [
            [
                float(
                    np.mean(
                        (
                            numpy_predict(
                                meta[f"x_{task}"], index, state[task], codes, outputs
                            )
                            - meta[f"y_{task}"]
                        )
                        ** 2
                    )
                )
                for task in range(tasks)
            ]
            for index, state in zip(indices, parameters)
        ]
    )
    mean_risk = risks.mean(1)
    choices = {
        "baseline_full": int(mean_risk[: len(baseline["indices"])].argmin()),
        "augmented": int(mean_risk.argmin()),
    }
    if selected:
        choices["signature_only"] = int(selected[int(mean_risk[selected].argmin())])
    np.savez_compressed(
        folder / "selection.npz",
        **{f"{name}_index": indices[number] for name, number in choices.items()},
        meta_mse=risks,
    )
    save_json(
        folder / "functional.json",
        {
            "choices": {
                name: {"candidate": number, "source_meta_mse": float(mean_risk[number])}
                for name, number in choices.items()
            },
            "baseline_candidates": len(baseline["indices"]),
            "signature_evaluated": len(selected),
            "new_fits": new_count,
            "reused_fits": reused,
            "seconds": time.monotonic() - started,
            "gn_steps": 48,
            "starts": keep,
            "tasks": tasks,
            "teacher_access": False,
            "opened_arrays": sorted(boundary.opened),
            "states_sha256": digest(folder / "functional_states.npz"),
            "selection_sha256": digest(folder / "selection.npz"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", choices=("standard", "general"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    evaluate(
        args.work,
        args.output,
        args.cohort,
        args.case,
        args.condition,
        torch.device(args.device),
    )


if __name__ == "__main__":
    main()
