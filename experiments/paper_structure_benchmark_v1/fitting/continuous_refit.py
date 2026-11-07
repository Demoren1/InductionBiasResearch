"""Refit native continuous bases without assigning or clustering their rows."""

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
    digest,
    load_sizes,
    save_json,
)
from experiments.sharing_structure_research_v1.blind_numeric import prediction
from experiments.sharing_structure_research_v1.blind_runtime import ObservationBoundary

from .numerical import fit_features, random_states


def run(root, case, condition, method, device):
    all_started = time.perf_counter()
    torch.set_num_threads(2)
    folder = root / "competitors" / f"{case}_c{condition}" / method
    boundary = ObservationBoundary(root, case, condition)
    boundary.phase = f"{method}_continuous_refit"
    boundary.install()
    sizes = load_sizes(root, case)
    h, k, o = [sizes[name] for name in ("hidden", "codes", "outputs")]
    native = dict(np.load(folder / "continuous.npz"))
    bases, native_parameters = native["basis"], native["source_parameters"]
    if (folder / "selected_continuous.npz").exists():
        raise RuntimeError("Existing frozen continuous result")
    fixed = dict(np.load(root / "public" / f"{case}_c{condition}" / "fixed_train.npz"))
    validation = dict(
        np.load(root / "public" / f"{case}_c{condition}" / "selection.npz")
    )
    tasks, keep = len(fixed) // 2, 4
    x, y, vx, vy = [
        torch.tensor(
            np.stack([panel[f"{name}_{task}"] for task in range(tasks)]),
            device=device,
            dtype=torch.float64,
        )
        for panel, name in (
            (fixed, "x"),
            (fixed, "y"),
            (validation, "x"),
            (validation, "y"),
        )
    ]
    save_json(
        folder / "continuous_refit_registered.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "native_sha256": digest(folder / "continuous.npz"),
            "code_sha256": digest(Path(__file__)),
            "numerical_code_sha256": digest(Path(__file__).with_name("numerical.py")),
            "starts": keep,
            "gn_steps": 48,
            "representation": "Original continuous basis, no discretization",
            "initialization": "One native task-specific state and three fresh states",
        },
    )
    shutil.copy2(__file__, folder / "continuous_refit_executed_code.py")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    counter = FlopCounterMode(display=False)
    parameters, risks, training, validation_history = [], [], [], []
    status = "failed"
    try:
        with counter:
            for begin in range(0, len(bases), 4):
                basis = torch.tensor(
                    bases[begin : begin + 4], device=device, dtype=torch.float64
                )
                count = len(basis)
                feature = (
                    torch.einsum("tnd,rdhk->rtnhk", x, basis)
                    .flatten(0, 1)
                    .repeat_interleave(keep, 0)
                )
                vf = (
                    torch.einsum("tnd,rdhk->rtnhk", vx, basis)
                    .flatten(0, 1)
                    .repeat_interleave(keep, 0)
                )
                initial = random_states(
                    count * tasks * keep, k, h, o, device, 749001 + begin
                )
                initial.reshape(count, tasks, keep, -1)[:, :, 0] = torch.tensor(
                    native_parameters[begin : begin + count], device=device
                )
                target = y.repeat(count, 1, 1).repeat_interleave(keep, 0)
                vtarget = vy.repeat(count, 1, 1).repeat_interleave(keep, 0)
                fitted, trace = fit_features(
                    feature, vf, initial, target, vtarget, k, steps=48
                )
                fitted = fitted.cpu().numpy().reshape(count, tasks, keep, -1)
                score = trace["numerical_selection_mse"].reshape(count, tasks, keep)
                best = score.argmin(2)
                parameters.append(
                    fitted[np.arange(count)[:, None], np.arange(tasks)[None], best]
                )
                risks.append(
                    score[np.arange(count)[:, None], np.arange(tasks)[None], best]
                )
                training.append(trace["train_history"].reshape(49, count, tasks, keep))
                validation_history.append(
                    trace["validation_history"].reshape(49, count, tasks, keep)
                )
                print(
                    json.dumps(
                        {
                            "method": method,
                            "case": case,
                            "continuous_refitted": begin + count,
                            "total": len(bases),
                        }
                    ),
                    flush=True,
                )
        torch.cuda.synchronize(device)
        parameters = np.concatenate(parameters)
        np.savez_compressed(
            folder / "refitted_continuous.npz",
            basis=bases,
            parameters=parameters,
            numerical_selection_mse=np.concatenate(risks),
            train_history=np.concatenate(training, axis=1),
            validation_history=np.concatenate(validation_history, axis=1),
        )
        status = "complete"
    finally:
        save_json(
            folder / "continuous_refit_metrics.json",
            {
                "status": status,
                "wall_seconds": time.perf_counter() - all_started,
                "fitting_wall_seconds": time.perf_counter() - started,
                "torch_operator_flops": counter.get_total_flops(),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                "refitted_sha256": digest(folder / "refitted_continuous.npz")
                if status == "complete"
                else None,
            },
        )
        boundary.save(folder / "continuous_refit_access_receipt.json", status)
    boundary.role, boundary.phase = "select", f"{method}_continuous_selection"
    meta = dict(np.load(root / "public" / f"{case}_c{condition}" / "meta.npz"))
    mx = torch.tensor(
        np.stack([meta[f"x_{t}"] for t in range(tasks)]),
        device=device,
        dtype=torch.float64,
    )
    my = torch.tensor(
        np.stack([meta[f"y_{t}"] for t in range(tasks)]),
        device=device,
        dtype=torch.float64,
    )
    errors = []
    with torch.no_grad():
        for basis, states in zip(bases, parameters):
            feature = torch.einsum(
                "tnd,dhk->tnhk",
                mx,
                torch.tensor(basis, device=device, dtype=torch.float64),
            )
            value = prediction(feature, torch.tensor(states, device=device), k, h, o)
            errors.append((value - my).square().mean((1, 2)).cpu().numpy())
    risk = np.array(errors).mean(1)
    best = int(risk.argmin())
    np.savez_compressed(
        folder / "selected_continuous.npz",
        basis=bases[best],
        parameters=parameters[best],
        source_meta_mse=np.array(errors)[best],
    )
    save_json(
        folder / "selected_continuous.json",
        {
            "candidate": best,
            "source_meta_mse": float(risk[best]),
            "all_source_meta_mse": risk.tolist(),
            "frozen_utc": datetime.now(UTC).isoformat(),
            "selected_sha256": digest(folder / "selected_continuous.npz"),
            "post_clustering": False,
        },
    )
    boundary.save(folder / "continuous_selection_access_receipt.json", "complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument(
        "--method", choices=("yeh", "yeh_strong", "msr", "msr_maml"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(
        args.root.resolve(),
        args.case,
        args.condition,
        args.method,
        torch.device(args.device),
    )


if __name__ == "__main__":
    main()
