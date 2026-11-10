"""Fit only fresh target numbers, then freeze predictions before query scoring."""

import argparse
import json
import os
import resource
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from .adaptation import feature, fit_adam, signed_basis, values
from .numerical import fit_features


class TargetBoundary:
    def __init__(self, root, folder):
        self.root, self.folder = root, folder
        self.project = Path(__file__).resolve().parents[3]
        self.opened, self.violations = set(), []

    def install(self):
        sys.addaudithook(self.audit)

    def audit(self, event, args):
        if event != "open" or not args or isinstance(args[0], int):
            return
        try:
            path = Path(os.fsdecode(args[0])).resolve()
        except (TypeError, ValueError, OSError):
            return
        if not path.is_relative_to(self.project):
            return
        forbidden = "private" in path.parts
        forbidden |= "results" in path.parts and not path.is_relative_to(self.folder)
        if forbidden:
            self.violations.append(str(path))
            raise PermissionError(f"Forbidden source numbers or target answer: {path}")
        self.opened.add(str(path))


def run(root, case, condition, method, device, *, timing_only=False):
    torch.set_num_threads(2)
    folder = root / "targets" / f"{case}_c{condition}"
    boundary = TargetBoundary(root, folder)
    boundary.install()
    output = folder / method / "unprofiled_timing" if timing_only else folder / method
    output.mkdir(parents=True, exist_ok=False)
    sizes = json.loads((folder / "sizes.json").read_text())
    d, h, k, o = [sizes[key] for key in ("inputs", "hidden", "codes", "outputs")]
    structures = np.load(folder / "structures.npz")
    basis = (
        None
        if method == "dense"
        else (
            torch.tensor(structures[method], device=device, dtype=torch.float64)
            if method in ("yeh_soft", "msr")
            else signed_basis(structures[method], k, device)
        )
    )
    arrays = [
        dict(np.load(folder / name))
        for name in ("support.npz", "selection.npz", "query_x.npz")
    ]
    registration = {
        "code_sha256": digest(Path(__file__)),
        "adapter_sha256": digest(Path(__file__).with_name("adaptation.py")),
        "numeric_sha256": digest(Path(__file__).with_name("numerical.py")),
        "input_sha256": {
            name: digest(folder / name)
            for name in (
                "structures.npz",
                "support.npz",
                "selection.npz",
                "query_x.npz",
            )
        },
        "budgets": [1024] if timing_only else [64, 256, 1024],
        "timing_only": timing_only,
        "adam_steps": 2000,
        "restarts": 18,
        "gauss_newton_starts": 4,
        "gauss_newton_steps": 48,
        "initialization": "Fresh random codebook/bias/readout/offset; bias activity quantiles from target support inputs only",
        "numerical_selection": "Target support/4 additional observations select numeric checkpoints; query responses never opened",
        "dense_controls": "Dense Adam checkpoint and Adam+same GN refinement both saved",
    }
    save_json(output / "registered.json", registration)
    for name in ("target_fit.py", "adaptation.py", "numerical.py"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    counter = None if timing_only else FlopCounterMode(display=False)
    result, trace_all = {}, {}
    status = "failed"
    try:
        with counter if counter is not None else nullcontext():
            for budget in (1024,) if timing_only else (64, 256, 1024):
                tick = time.perf_counter()
                support, selection, query = arrays
                x, y, vx, vy, qx = [
                    torch.tensor(a, device=device, dtype=torch.float32)
                    for a in (
                        support["x"][:, :budget],
                        support["y"][:, :budget],
                        selection["x"][:, : budget // 4],
                        selection["y"][:, : budget // 4],
                        query["x"],
                    )
                ]
                center = y.mean(1)
                scale = y.std((1, 2), correction=0).clamp_min(1e-8)
                y = (y - center[:, None]) / scale[:, None, None]
                vy = (vy - center[:, None]) / scale[:, None, None]
                fb = None if basis is None else basis.float()
                initial, trace = fit_adam(x, y, vx, vy, fb, h, o, seed=763001 + budget)
                torch.cuda.synchronize(device)
                result[f"adam_seconds_{budget}"] = np.asarray(
                    time.perf_counter() - tick
                )
                if timing_only:
                    result[f"adam_peak_allocated_bytes_{budget}"] = np.asarray(
                        torch.cuda.max_memory_allocated(device)
                    )
                adam_value = values(qx, fb, initial[:1], h, o)[0]
                result[f"adam_prediction_{budget}"] = (
                    (adam_value * scale[:, None, None] + center[:, None])
                    .detach()
                    .cpu()
                    .numpy()
                )
                keep, tasks = initial.shape[:2]
                gb = (
                    basis
                    if basis is not None
                    else torch.eye(d * h, device=device, dtype=torch.float64).reshape(
                        d, h, d * h
                    )
                )
                tx, tvx = x.double(), vx.double()
                feat = feature(tx, gb)[None].expand(keep, -1, -1, -1, -1).flatten(0, 1)
                vf = feature(tvx, gb)[None].expand(keep, -1, -1, -1, -1).flatten(0, 1)
                gn, gn_trace = fit_features(
                    feat,
                    vf,
                    initial.double().flatten(0, 1),
                    y.double().repeat(keep, 1, 1),
                    vy.double().repeat(keep, 1, 1),
                    gb.shape[-1],
                    steps=48,
                )
                loss = gn_trace["numerical_selection_mse"].reshape(keep, tasks)
                gn = gn.reshape(keep, tasks, -1)
                numeric = gn[loss.argmin(0), torch.arange(tasks, device=device)]
                qvalue = values(qx.double(), gb, numeric[None], h, o)[0]
                raw = numeric.detach().clone()
                raw[..., gb.shape[-1] + h : -o] *= scale[:, None]
                raw[..., -o:] = raw[..., -o:] * scale[:, None] + center
                result[f"parameters_{budget}"] = raw.cpu().numpy()
                result[f"prediction_{budget}"] = (
                    (qvalue * scale[:, None, None] + center[:, None])
                    .detach()
                    .cpu()
                    .numpy()
                )
                trace_all.update(
                    {
                        f"{key}_{budget}": val
                        for key, val in trace.items()
                        if key != "settings"
                    }
                )
                trace_all.update(
                    {
                        f"gn_{key}_{budget}": val
                        for key, val in gn_trace.items()
                        if isinstance(val, np.ndarray)
                    }
                )
                torch.cuda.synchronize(device)
                result[f"seconds_{budget}"] = np.asarray(time.perf_counter() - tick)
                print(
                    json.dumps(
                        {
                            "method": method,
                            "case": case,
                            "budget": budget,
                            "median_H0_normalized_mse": float(np.median(loss.min(0))),
                            "seconds": float(result[f"seconds_{budget}"]),
                        }
                    ),
                    flush=True,
                )
        if timing_only:
            save_json(
                output / "timing.json",
                {
                    key: float(value)
                    for key, value in result.items()
                    if "seconds" in key or "peak_allocated_bytes" in key
                },
            )
        else:
            np.savez_compressed(output / "frozen_predictions.npz", **result)
            np.savez_compressed(output / "curves.npz", **trace_all)
        status = "complete"
    finally:
        save_json(
            output / "metrics.json",
            {
                "status": status,
                "wall_seconds": time.perf_counter() - started,
                "torch_operator_flops": counter.get_total_flops()
                if counter is not None
                else None,
                "timing_only": timing_only,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                "predictions_sha256": digest(output / "frozen_predictions.npz")
                if status == "complete" and not timing_only
                else None,
            },
        )
        save_json(
            output / "access_receipt.json",
            {
                "status": status,
                "violations": boundary.violations,
                "opened_workspace_files": sorted(boundary.opened),
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument("--method", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    run(
        args.root.resolve(),
        args.case,
        args.condition,
        args.method,
        torch.device(args.device),
        timing_only=args.timing_only,
    )


if __name__ == "__main__":
    main()
