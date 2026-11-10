"""Evaluate native MSR learned-initialization transfer as a separate control."""

import argparse
import json
import resource
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from ..methods.bilevel import predict
from ..methods.real_competitors import adapt
from .target_fit import TargetBoundary


def run(root, device, *, timing_only=False):
    torch.set_num_threads(2)
    folder = root / "targets"
    boundary = TargetBoundary(root, folder)
    boundary.install()
    output = folder / "msr_native"
    protocol = json.loads((folder / "sizes.json").read_text())
    h, o, kind = protocol["hidden"], protocol["outputs"], protocol["loss"]
    archive = np.load(output / "model.npz")
    basis, initial, rates = [
        torch.tensor(archive[key], device=device, dtype=torch.float32)[None]
        for key in ("basis", "initial", "rates")
    ]
    support, query = [
        dict(np.load(folder / name)) for name in ("support.npz", "query_x.npz")
    ]
    count = 25 if kind == "cross_entropy" else 512
    model_folder = output
    if timing_only:
        output = output / "unprofiled_timing"
        output.mkdir(exist_ok=False)
    save_json(
        output / "registered.json",
        {
            "code_sha256": digest(Path(__file__)),
            "model_sha256": digest(model_folder / "model.npz"),
            "target_inner_steps": 3,
            "support_count": count,
            "transferred": "U plus learned numerical initialization and per-group SGD rates",
            "not_structure_only": True,
            "target_query_answers_used": False,
        },
    )
    shutil.copy2(__file__, output / "executed_code.py")
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    counter = None if timing_only else FlopCounterMode(display=False)
    values, states = [], []
    status = "failed"
    try:
        with counter if counter is not None else nullcontext():
            for begin in range(0, len(support["x"]), 64):
                local = slice(begin, begin + 64)
                x, y, qx = [
                    torch.tensor(a, device=device, dtype=torch.float32)
                    for a in (
                        support["x"][local, :count],
                        support["y"][local, :count],
                        query["x"][local],
                    )
                ]
                x, qx = x / 255, qx / 255
                center, scale = y.mean(1), y.std((1, 2), correction=0).clamp_min(1e-8)
                if kind == "cross_entropy":
                    center, scale = torch.zeros_like(center), torch.ones_like(scale)
                y = (y - center[:, None]) / scale[:, None, None]
                numeric = adapt(x, y, basis, initial, rates, h, o, kind, 3, False)
                with torch.no_grad():
                    value = predict(qx, basis, numeric, h, o)[0]
                    values.append(
                        (value * scale[:, None, None] + center[:, None]).cpu().numpy()
                    )
                    raw = numeric[0].clone()
                    codes = basis.shape[-1]
                    raw[:, codes + h : -o] *= scale[:, None]
                    raw[:, -o:] = raw[:, -o:] * scale[:, None] + center
                    states.append(raw.cpu().numpy())
        if not timing_only:
            np.savez_compressed(
                output / "frozen_predictions.npz",
                prediction=np.concatenate(values),
                parameters=np.concatenate(states),
            )
        status = "complete"
    finally:
        torch.cuda.synchronize(device)
        save_json(
            output / "metrics.json",
            {
                "status": status,
                "wall_seconds": time.perf_counter() - started,
                "tasks": len(support["x"]),
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
                "not_structure_only": True,
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    run(args.root.resolve(), torch.device(args.device), timing_only=args.timing_only)


if __name__ == "__main__":
    main()
