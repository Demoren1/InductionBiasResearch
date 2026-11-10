"""Compare native hard indexators using a common numerical fitting budget."""

import argparse
import json
import resource
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
    digest,
    load_sizes,
    save_json,
)
from experiments.sharing_structure_research_v1.blind_numeric import (
    fit_batch,
    initial_states,
    numpy_predict,
)
from experiments.sharing_structure_research_v1.blind_runtime import ObservationBoundary


def candidate_bank(folder, codes, limit):
    native = dict(np.load(folder / "native.npz"))
    indices = native["indices"]
    scores = native["numerical_selection_mse"]
    if indices.ndim == 4:
        scale = native["normalization_scale"]
        ranking = scores / scale[None] ** 2
        order = []
        for rank in range(indices.shape[0]):
            for task in range(indices.shape[1]):
                scenario = ranking[:, task].argsort()[rank]
                order.append((scenario, task))
        proposals = [indices[row] for row in order]
    else:
        proposals = list(indices[np.argsort(scores)])
    seen, kept = set(), []
    for proposal in proposals:
        canonical = canonical_index(proposal.astype(np.int16), codes)
        key = canonical.tobytes()
        if key not in seen:
            seen.add(key)
            kept.append(canonical)
    return np.stack(kept[:limit]), len(kept)


def run(root, case, condition, method, limit, device):
    all_started = time.perf_counter()
    torch.set_num_threads(2)
    folder = root / "competitors" / f"{case}_c{condition}" / method
    if (folder / "refitted.npz").exists():
        raise RuntimeError("Existing numerical results require explicit recovery")
    boundary = ObservationBoundary(root, case, condition)
    boundary.phase = f"{method}_refit"
    boundary.install()
    sizes = load_sizes(root, case)
    codes, outputs = sizes["codes"], sizes["outputs"]
    fixed = dict(np.load(root / "public" / f"{case}_c{condition}" / "fixed_train.npz"))
    validation = dict(
        np.load(root / "public" / f"{case}_c{condition}" / "selection.npz")
    )
    tasks, keep = len(fixed) // 2, 4
    indices, distinct = candidate_bank(folder, codes, limit)
    source = [
        dict(np.load(root / "source" / f"{case}_c{condition}_t{t}" / "learned.npz"))
        for t in range(tasks)
    ]
    x, y, vx, vy = [
        np.stack([panel[f"{name}_{task}"] for task in range(tasks)])
        for panel, name in (
            (fixed, "x"),
            (fixed, "y"),
            (validation, "x"),
            (validation, "y"),
        )
    ]
    save_json(
        folder / "refit_registered.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "code_sha256": digest(Path(__file__)),
            "native_sha256": digest(folder / "native.npz"),
            "candidates": len(indices),
            "distinct_native_candidates": distinct,
            "starts_per_task": keep,
            "gn_steps": 48,
            "initialization": "Same task-local learned dense descriptors as ours; no true numerical parameters. Shared preprocessing cost must be reported.",
            "post_clustering": False,
        },
    )
    shutil.copy2(__file__, folder / "refit_executed_code.py")
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    counter = FlopCounterMode(display=False)
    parameters, risks, train_history, validation_history = [], [], [], []
    status = "failed"
    try:
        with counter:
            for begin in range(0, len(indices), 4):
                batch = indices[begin : begin + 4]
                initial = np.stack(
                    [
                        np.stack(
                            [
                                initial_states(
                                    index,
                                    source[t],
                                    codes,
                                    x[t],
                                    y[t],
                                    vx[t],
                                    vy[t],
                                    keep,
                                    device,
                                )
                                for t in range(tasks)
                            ]
                        )
                        for index in batch
                    ]
                )
                packed = [
                    np.tile(np.repeat(a, keep, axis=0), (len(batch), 1, 1))
                    for a in (x, y, vx, vy)
                ]
                fitted, trace = fit_batch(
                    np.repeat(batch, tasks * keep, axis=0),
                    initial.reshape(len(batch) * tasks * keep, -1),
                    *packed,
                    codes,
                    device,
                    steps=48,
                )
                fitted = fitted.reshape(len(batch), tasks, keep, -1)
                loss = trace["validation_mse"].reshape(len(batch), tasks, keep)
                best = loss.argmin(2)
                parameters.append(
                    fitted[np.arange(len(batch))[:, None], np.arange(tasks)[None], best]
                )
                risks.append(
                    loss[np.arange(len(batch))[:, None], np.arange(tasks)[None], best]
                )
                train_history.append(
                    trace["train_history"].reshape(49, len(batch), tasks, keep)
                )
                validation_history.append(
                    trace["validation_history"].reshape(49, len(batch), tasks, keep)
                )
                print(
                    json.dumps(
                        {
                            "method": method,
                            "case": case,
                            "fitted_candidates": begin + len(batch),
                            "total": len(indices),
                        }
                    ),
                    flush=True,
                )
        torch.cuda.synchronize(device)
        parameters = np.concatenate(parameters)
        risks = np.concatenate(risks)
        np.savez_compressed(
            folder / "refitted.npz",
            indices=indices,
            parameters=parameters,
            numerical_selection_mse=risks,
            train_history=np.concatenate(train_history, axis=1),
            validation_history=np.concatenate(validation_history, axis=1),
        )
        status = "complete"
    finally:
        save_json(
            folder / "refit_metrics.json",
            {
                "status": status,
                "wall_seconds": time.perf_counter() - all_started,
                "fitting_wall_seconds": time.perf_counter() - started,
                "torch_operator_flops": counter.get_total_flops(),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                "refitted_sha256": digest(folder / "refitted.npz")
                if status == "complete"
                else None,
            },
        )
        boundary.save(folder / "refit_access_receipt.json", status)
    boundary.role, boundary.phase = "select", f"{method}_select"
    meta = dict(np.load(root / "public" / f"{case}_c{condition}" / "meta.npz"))
    errors = []
    for index, states in zip(indices, parameters):
        error = [
            np.mean(
                (
                    numpy_predict(meta[f"x_{t}"], index, states[t], codes, outputs)
                    - meta[f"y_{t}"]
                )
                ** 2
            )
            for t in range(tasks)
        ]
        errors.append(error)
    risk = np.array(errors).mean(1)
    best = int(risk.argmin())
    np.savez_compressed(
        folder / "selected.npz",
        index=indices[best],
        parameters=parameters[best],
        source_meta_mse=np.array(errors)[best],
    )
    save_json(
        folder / "selected.json",
        {
            "candidate": best,
            "source_meta_mse": float(risk[best]),
            "all_source_meta_mse": risk.tolist(),
            "selected_sha256": digest(folder / "selected.npz"),
            "frozen_utc": datetime.now(UTC).isoformat(),
            "true_parameters_accessed": False,
        },
    )
    boundary.save(folder / "selection_access_receipt.json", "complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument("--method", default="lutq")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(
        args.root.resolve(),
        args.case,
        args.condition,
        args.method,
        args.limit,
        torch.device(args.device),
    )


if __name__ == "__main__":
    main()
