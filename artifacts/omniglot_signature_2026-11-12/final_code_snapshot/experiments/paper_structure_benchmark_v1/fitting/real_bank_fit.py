"""Common numerical source calibration and independent functional bank selection."""

import argparse
import json
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.flop_counter import FlopCounterMode

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from .adaptation import fit_adam, losses, signed_basis, values
from ..protocol.real_boundary import StudyBoundary


def candidates(root, method, codes):
    if method in ("current", "current_functional"):
        folder = root / "source" / f"{method}_proposals"
        arrays = np.load(folder / "candidates.npz")
        metadata = json.loads((folder / "candidates.json").read_text())["candidates"]
        return [
            (row["name"], arrays[row["name"]], row.get("anchor", 0))
            for row in metadata
            if row["codes"] == codes
        ]
    folder = (
        root / "source" / f"{'yeh' if method.startswith('yeh') else method}_k{codes}"
    )
    native = np.load(folder / "native.npz")
    score = native["H0_loss"]
    if method == "lutq":
        ranked_tasks = score.min(0).argsort()[:4]
        ranked = [
            (score[:, task].argsort()[rank], task)
            for rank in range(score.shape[0])
            for task in ranked_tasks
        ]
        arrays = native["indices"]
        seen, result = set(), []
        for scenario, task in ranked:
            item = arrays[scenario, task]
            if item.tobytes() in seen:
                continue
            seen.add(item.tobytes())
            result.append((f"r{scenario}_t{task}", item, int(task)))
            if len(result) == 8:
                break
        return result
    key = "indices" if method == "yeh_map" else "basis"
    ordered = [
        (f"r{scenario}", native[key][scenario], 0) for scenario in score.argsort()
    ]
    if method != "yeh_map":
        return ordered
    seen, unique = set(), []
    for item in ordered:
        if item[1].tobytes() not in seen:
            seen.add(item[1].tobytes())
            unique.append(item)
    return unique


def dense_initial(basis, dense, reference, hidden, outputs):
    dimensions, _, codes = basis.shape
    design = basis.reshape(dimensions * hidden, codes)
    gram = design.T @ design
    regularized = (
        gram
        + torch.eye(codes, device=basis.device, dtype=basis.dtype)
        * gram.diagonal().max().clamp_min(1e-12)
        * 1e-8
    )
    original = dense[:, : dimensions * hidden].reshape(-1, dimensions, hidden)
    c0 = torch.linalg.solve(regularized, design.T @ original[reference].flatten())
    proxy = torch.einsum("dhk,k->dh", basis, c0)
    proxy = (proxy / proxy.norm(dim=0).clamp_min(1e-12)).cpu().numpy()
    aligned, biases, heads = [], [], []
    for row, weight in zip(dense, original):
        normal = (weight / weight.norm(dim=0).clamp_min(1e-12)).cpu().numpy()
        cost = ((proxy[:, :, None] - normal[:, None]) ** 2).sum(0)
        left, right = linear_sum_assignment(cost)
        mapping = torch.tensor(right[np.argsort(left)], device=basis.device)
        aligned.append(weight[:, mapping].flatten())
        biases.append(row[dimensions * hidden : dimensions * hidden + hidden][mapping])
        heads.append(
            row[dimensions * hidden + hidden : -outputs]
            .reshape(hidden, outputs)[mapping]
            .flatten()
        )
    code = torch.linalg.solve(regularized, design.T @ torch.stack(aligned).T).T
    return torch.cat(
        (code, torch.stack(biases), torch.stack(heads), dense[:, -outputs:]), -1
    )[None]


def run(root, method, codes, device, steps):
    torch.set_num_threads(2)
    boundary = StudyBoundary(root)
    boundary.install()
    folder = root / "fits" / f"{method}_k{codes}"
    folder.mkdir(exist_ok=False)
    protocol = json.loads((root / "protocol.json").read_text())
    h, o, kind = protocol["hidden"], protocol["outputs"], protocol["loss"]
    proposals = candidates(root, method, codes)
    save_json(
        folder / "registered.json",
        {
            "method": method,
            "codes": codes,
            "code_sha256": digest(Path(__file__)),
            "adapter_sha256": digest(Path(__file__).with_name("adaptation.py")),
            "candidates": [row[0] for row in proposals],
            "adam_steps": steps,
            "restarts": 18,
            "initialization": "One common source-data-derived dense projection and17 fresh states; identical solver for every discrete/continuous bank",
            "selection": "H0 chooses per-task numeric states, independent source M chooses shared structure; no second discretization/clustering of competitors",
        },
    )
    for name in ("real_bank_fit.py", "adaptation.py"):
        shutil.copy2(Path(__file__).with_name(name), folder / name)
    source, numerical = [
        dict(np.load(root / "public" / f"{name}.npz"))
        for name in ("support", "numerical")
    ]
    x, y, vx, vy = [
        torch.tensor(panel[key], device=device, dtype=torch.float32)
        for panel, key in (
            (source, "x"),
            (source, "y"),
            (numerical, "x"),
            (numerical, "y"),
        )
    ]
    x, vx = x / 255, vx / 255
    center, scale = y.mean(1), y.std((1, 2), correction=0).clamp_min(1e-8)
    if kind == "cross_entropy":
        center, scale = torch.zeros_like(center), torch.ones_like(scale)
    y, vy = (
        (y - center[:, None]) / scale[:, None, None],
        (vy - center[:, None]) / scale[:, None, None],
    )
    dense = torch.tensor(
        np.load(root / "source" / "dense" / "models.npz")["parameters"], device=device
    )
    dense[:, protocol["inputs"] * h + h : -o] /= scale[:, None]
    dense[:, -o:] = (dense[:, -o:] - center) / scale[:, None]
    parameters, bases, indices, traces = [], [], [], []
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    counter = FlopCounterMode(display=False)
    status = "failed"
    try:
        with counter:
            for name, array, reference in proposals:
                basis = (
                    signed_basis(array, codes, device, torch.float32)
                    if array.ndim == 2
                    else torch.tensor(array, device=device, dtype=torch.float32)
                )
                # A global rescaling changes the code coordinates, not the model family.
                basis = basis / basis.square().sum(-1).mean().sqrt().clamp_min(1e-12)
                initial = dense_initial(basis, dense, reference, h, o)
                fitted, trace = fit_adam(
                    x, y, vx, vy, basis, h, o, steps=steps, loss=kind, initial=initial
                )
                best = fitted[0]
                best[:, codes + h : -o] *= scale[:, None]
                best[:, -o:] = best[:, -o:] * scale[:, None] + center
                parameters.append(best.cpu().numpy())
                bases.append(basis.cpu().numpy())
                indices.append(
                    array
                    if array.ndim == 2
                    else np.zeros((protocol["inputs"], h), dtype=np.int16)
                )
                traces.append(trace)
                print(
                    json.dumps(
                        {
                            "method": method,
                            "codes": codes,
                            "candidate": name,
                            "finished": len(parameters),
                            "total": len(proposals),
                            "seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        np.savez_compressed(
            folder / "fitted.npz",
            parameters=np.stack(parameters),
            basis=np.stack(bases),
            indices=np.stack(indices),
            H0_loss=np.stack([row["numerical_selection_loss"] for row in traces]),
            steps=traces[0]["steps"],
            train_history=np.stack([row["train_history"] for row in traces]),
            validation_history=np.stack([row["validation_history"] for row in traces]),
        )
        boundary.role = "select"
        structure = dict(np.load(root / "public" / "structure.npz"))
        mx, my = [
            torch.tensor(structure[key], device=device, dtype=torch.float32)
            for key in ("x", "y")
        ]
        mx /= 255
        risks = []
        with torch.no_grad():
            for basis, numeric in zip(bases, parameters):
                value = values(
                    mx,
                    torch.tensor(basis, device=device),
                    torch.tensor(numeric, device=device)[None],
                    h,
                    o,
                )
                risks.append(losses(value, my, kind)[0].cpu().numpy())
        risk = np.stack(risks)
        chosen = int(risk.mean(1).argmin())
        np.savez_compressed(
            folder / "selected.npz",
            basis=bases[chosen],
            index=indices[chosen],
            source_parameters=parameters[chosen],
            source_M_loss=risk[chosen],
        )
        save_json(
            folder / "selected.json",
            {
                "candidate": proposals[chosen][0],
                "source_M_loss": float(risk[chosen].mean()),
                "all_source_M_loss": risk.mean(1).tolist(),
                "codes": codes,
                "native_continuous": method in ("yeh_soft", "msr"),
                "selected_sha256": digest(folder / "selected.npz"),
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
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
            },
        )
        boundary.save(folder / "access_receipt.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--codes", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(
        args.root.resolve(),
        args.method,
        args.codes,
        torch.device(args.device),
        args.steps,
    )


if __name__ == "__main__":
    main()
