"""Calibrate ratio proposals and select on independent source responses."""

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from experiments.paper_structure_benchmark_v1.fitting.adaptation import (
    fit_adam,
    losses,
    signed_basis,
    values,
)
from experiments.paper_structure_benchmark_v1.fitting.real_bank_fit import (
    dense_initial,
)
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)


def run(root, output, codes, device):
    torch.set_num_threads(2)
    boundary = StudyBoundary(root)
    boundary.install()
    folder = root / "fits" / f"ratio_k{codes}"
    folder.mkdir(exist_ok=False)
    proposals = root / "source" / "ratio_proposals"
    metadata = json.loads((proposals / "candidates.json").read_text())
    rows = [r for r in metadata["candidates"] if r["codes"] == codes]
    if not rows:
        raise RuntimeError(
            f"No naturally complete partition for budget {codes}"
        )
    candidates = dict(np.load(proposals / "candidates.npz"))
    protocol = json.loads((root / "protocol.json").read_text())
    h, o = protocol["hidden"], protocol["outputs"]
    save_json(
        folder / "registered.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "codes": codes,
            "candidates": [r["name"] for r in rows],
            "proposal_sha256": digest(proposals / "candidates.npz"),
            "code_sha256": digest(Path(__file__)),
            "adapter_sha256": digest(
                Path("experiments/paper_structure_benchmark_v1/fitting")
                / "adaptation.py"
            ),
            "adam_steps": 2000,
            "starts": 18,
            "numeric_selection": "H per-task CE",
            "structure_selection": "Independent source M mean CE",
            "initialization": "Same dense projection plus 17 fresh states",
        },
    )
    support, numerical = [
        dict(np.load(root / "public" / f"{name}.npz"))
        for name in ("support", "numerical")
    ]
    x, y, vx, vy = [
        torch.tensor(panel[key], device=device, dtype=torch.float32)
        for panel, key in (
            (support, "x"),
            (support, "y"),
            (numerical, "x"),
            (numerical, "y"),
        )
    ]
    x, vx = x / 255, vx / 255
    dense = torch.tensor(
        np.load(root / "source" / "dense" / "models.npz")["parameters"],
        device=device,
    )
    bases, indices, parameters, traces = [], [], [], []
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    counter = FlopCounterMode(display=False)
    status = "failed"
    try:
        with counter:
            for row in rows:
                index = candidates[row["name"]]
                basis = signed_basis(index, codes, device, torch.float32)
                basis /= basis.square().sum(-1).mean().sqrt().clamp_min(1e-12)
                initial = dense_initial(basis, dense, row["anchor"], h, o)
                fitted, trace = fit_adam(
                    x,
                    y,
                    vx,
                    vy,
                    basis,
                    h,
                    o,
                    loss="cross_entropy",
                    steps=2000,
                    initial=initial,
                )
                bases.append(basis.cpu().numpy())
                indices.append(index)
                parameters.append(fitted[0].cpu().numpy())
                traces.append(trace)
                print(
                    json.dumps(
                        {
                            "codes": codes,
                            "candidate": row["name"],
                            "done": len(parameters),
                            "total": len(rows),
                            "seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        boundary.role = "select"
        structure = dict(np.load(root / "public" / "structure.npz"))
        mx = torch.tensor(structure["x"], device=device) / 255
        my = torch.tensor(structure["y"], device=device)
        risks = []
        with torch.no_grad():
            for basis, numeric in zip(bases, parameters):
                prediction = values(
                    mx,
                    torch.tensor(basis, device=device),
                    torch.tensor(numeric, device=device)[None],
                    h,
                    o,
                )
                risks.append(
                    losses(prediction, my, "cross_entropy")[0].cpu().numpy()
                )
        risks = np.stack(risks)
        chosen = int(risks.mean(1).argmin())
        np.savez_compressed(
            folder / "fitted.npz",
            indices=np.stack(indices),
            parameters=np.stack(parameters),
            source_M_loss=risks,
            steps=traces[0]["steps"],
            H_loss=np.stack([t["numerical_selection_loss"] for t in traces]),
            train_history=np.stack([t["train_history"] for t in traces]),
            validation_history=np.stack(
                [t["validation_history"] for t in traces]
            ),
        )
        np.savez_compressed(
            folder / "selected.npz",
            basis=bases[chosen],
            index=indices[chosen],
            source_parameters=parameters[chosen],
            source_M_loss=risks[chosen],
        )
        save_json(
            folder / "selected.json",
            {
                "candidate": rows[chosen]["name"],
                "codes": codes,
                "source_M_loss": float(risks[chosen].mean()),
                "all_source_M_loss": risks.mean(1).tolist(),
                "all_candidates": [r["name"] for r in rows],
                "selected_sha256": digest(folder / "selected.npz"),
                "origin": rows[chosen],
            },
        )
        status = "complete"
    finally:
        torch.cuda.synchronize(device)
        save_json(
            folder / "metrics.json",
            {
                "status": status,
                "wall_seconds": time.perf_counter() - started,
                "torch_operator_flops": counter.get_total_flops(),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(
                    device
                ),
            },
        )
        boundary.save(folder / "access_receipt.json", status)
    destination = output / "source_selection" / f"k{codes}"
    destination.mkdir(parents=True)
    for name in (
        "selected.json",
        "registered.json",
        "metrics.json",
        "access_receipt.json",
    ):
        shutil.copy2(folder / name, destination / name)
    np.savez_compressed(
        destination / "selected.npz",
        index=indices[chosen],
        basis=bases[chosen],
        source_M_loss=risks[chosen],
        all_source_M_loss=risks,
    )
    shutil.copy2(__file__, output / "code_snapshot" / Path(__file__).name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codes", type=int, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    run(
        args.root.resolve(),
        args.output.resolve(),
        args.codes,
        torch.device(args.device),
    )


if __name__ == "__main__":
    main()
