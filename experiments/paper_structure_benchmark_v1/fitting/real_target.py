"""Fresh support-only real-task adaptation with a frozen global update rule."""

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
from .adaptation import fit_adam, values
from .target_fit import TargetBoundary


def run(root, method, device, *, timing_only=False):
    torch.set_num_threads(2)
    folder = root / "targets"
    boundary = TargetBoundary(root, folder)
    boundary.install()
    output = folder / method
    protocol = json.loads((folder / "sizes.json").read_text())
    selection = json.loads((output / "selected.json").read_text())
    h, o, kind = protocol["hidden"], protocol["outputs"], protocol["loss"]
    model = np.load(output / "model.npz")
    basis = (
        None
        if method in ("dense", "mask_only", "mask_random")
        else torch.tensor(model["basis"], device=device, dtype=torch.float32)
    )
    mask = (
        torch.tensor(model["mask"], device=device, dtype=torch.float32)
        if "mask" in model.files
        else None
    )
    support, query = [
        dict(np.load(folder / name)) for name in ("support.npz", "query_x.npz")
    ]
    count = selection["support_count"]
    settings, updates = [selection["global_setting"]], selection["numeric_updates"]
    model_folder = output
    if timing_only:
        output = output / "unprofiled_timing"
        output.mkdir(exist_ok=False)
    save_json(
        output / "registered.json",
        {
            "code_sha256": digest(Path(__file__)),
            "adapter_sha256": digest(Path(__file__).with_name("adaptation.py")),
            "model_sha256": digest(model_folder / "model.npz"),
            "selection_sha256": digest(model_folder / "selected.json"),
            "support_sha256": digest(folder / "support.npz"),
            "query_inputs_sha256": digest(folder / "query_x.npz"),
            "numeric_parameters": "One fresh random restart per task; fixed global learning rate/activity quantile/update count from disjoint development tasks",
            "support_count": count,
            "query_labels_used": False,
        },
    )
    for name in ("real_target.py", "adaptation.py"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    counter = None if timing_only else FlopCounterMode(display=False)
    predictions, parameters, training = [], [], []
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
                fitted, trace = fit_adam(
                    x,
                    y,
                    None,
                    None,
                    basis,
                    h,
                    o,
                    loss=kind,
                    seed=771003 + begin,
                    fixed_step=updates,
                    settings=settings,
                    mask=mask,
                )
                with torch.no_grad():
                    value = values(qx, basis, fitted[:1], h, o, mask=mask)[0]
                    predictions.append(
                        (value * scale[:, None, None] + center[:, None]).cpu().numpy()
                    )
                    raw = fitted[0].clone()
                    codes = raw.shape[-1] - h - h * o - o
                    raw[:, codes + h : -o] *= scale[:, None]
                    raw[:, -o:] = raw[:, -o:] * scale[:, None] + center
                    if mask is not None:
                        raw = torch.cat(
                            (raw[:, :codes][:, mask.flatten().bool()], raw[:, codes:]),
                            -1,
                        )
                    parameters.append(raw.cpu().numpy())
                    training.append(trace["train_history"][:, 0])
                print(
                    json.dumps(
                        {
                            "method": method,
                            "target_tasks_done": min(begin + 64, len(support["x"])),
                            "numeric_updates": updates,
                            "seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        if not timing_only:
            np.savez_compressed(
                output / "frozen_predictions.npz",
                prediction=np.concatenate(predictions),
                parameters=np.concatenate(parameters),
                steps=trace["steps"],
                train_history=np.concatenate(training, -1),
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
    parser.add_argument("--method", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    run(
        args.root.resolve(),
        args.method,
        torch.device(args.device),
        timing_only=args.timing_only,
    )


if __name__ == "__main__":
    main()
