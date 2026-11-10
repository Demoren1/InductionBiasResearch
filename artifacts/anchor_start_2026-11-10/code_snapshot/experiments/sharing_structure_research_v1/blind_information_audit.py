"""Inspect identifiable directions of learned dense models, without teacher data."""

import argparse
from pathlib import Path

from .blind_contract import load_sizes, save_json
from .blind_runtime import ObservationBoundary


def run(root, case, condition, device):
    import math

    import numpy as np
    import torch

    from .blind_dense_extract import values

    torch.set_num_threads(2)
    sizes = load_sizes(root, case)
    dimensions, hidden, outputs = [sizes[k] for k in ("inputs", "hidden", "outputs")]
    observed = dict(
        np.load(root / "public" / f"{case}_c{condition}" / "fixed_train.npz")
    )
    paths = sorted(
        (root / "source").glob(f"{case}_c{condition}_t*/learned.npz"),
        key=lambda p: int(p.parent.name.rsplit("t", 1)[1]),
    )
    source = [dict(np.load(p)) for p in paths]
    parameters = np.stack([s["dense_parameters"] for s in source])
    count = dimensions * hidden
    weight = parameters[:, :count].reshape(-1, dimensions, hidden)
    norms = np.maximum(np.linalg.norm(weight, axis=1), 1e-12)
    parameters[:, :count] = (weight / norms[:, None]).reshape(-1, count)
    parameters[:, count : count + hidden] /= norms
    parameters[:, count + hidden : -outputs] = (
        parameters[:, count + hidden : -outputs].reshape(-1, hidden, outputs)
        * norms[..., None]
    ).reshape(len(source), -1)
    rows = []
    with torch.no_grad():
        for begin in range(0, len(source), 4):
            end = min(begin + 4, len(source))
            tx = torch.tensor(
                np.stack([observed[f"x_{t}"] for t in range(begin, end)]),
                device=device,
                dtype=torch.float64,
            )
            ty = torch.tensor(
                np.stack([observed[f"y_{t}"] for t in range(begin, end)]),
                device=device,
                dtype=torch.float64,
            )
            p = torch.tensor(parameters[begin:end], device=device, dtype=torch.float64)
            prediction, jac = values(tx, p, dimensions, hidden, outputs, jacobian=True)
            gram = jac.transpose(1, 2) @ jac
            eigenvalues, vectors = torch.linalg.eigh(gram)
            for local, task in enumerate(range(begin, end)):
                keep = eigenvalues[local] > eigenvalues[local, -1] * 1e-10
                rank = int(keep.sum())
                variance = float(
                    (ty[local] - prediction[local]).square().sum()
                    / max(ty[local].numel() - rank, 1)
                )
                basis = vectors[local, :, keep]
                covariance = (
                    ((basis / eigenvalues[local, keep]) @ basis.T * variance)
                    .cpu()
                    .numpy()
                )
                null = (
                    (
                        torch.eye(len(gram[local]), device=device, dtype=torch.float64)
                        - basis @ basis.T
                    )
                    .cpu()
                    .numpy()
                )
                direction = parameters[task, :count].reshape(dimensions, hidden)
                fractions, deviations = [], []
                for h in range(hidden):
                    selected = np.arange(dimensions) * hidden + h
                    tangent = np.eye(dimensions) - np.outer(
                        direction[:, h], direction[:, h]
                    )
                    fraction = float(
                        np.trace(tangent @ null[np.ix_(selected, selected)] @ tangent.T)
                        / max(float(np.sum(tangent**2)), 1e-12)
                    )
                    cov = tangent @ covariance[np.ix_(selected, selected)] @ tangent.T
                    deviation = math.sqrt(max(float(np.max(cov.diagonal())), 0))
                    fractions.append(max(fraction, 0))
                    deviations.append(
                        max(deviation, 1.0) if fraction > 1e-6 else deviation
                    )
                normal = parameters[task, :count].reshape(dimensions, hidden)
                activation = (
                    tx[local].cpu().numpy() @ normal
                    + parameters[task, count : count + hidden]
                )
                row = {
                    "task": task,
                    "empirical_rank": rank,
                    "full_dimension": p.shape[1],
                    "variance_estimate": variance,
                    "old_uncertainty": source[task]["uncertainty"].tolist(),
                    "canonical_pseudoinverse_uncertainty": deviations,
                    "unidentified_angular_fraction": fractions,
                    "unidentified_neurons": int(
                        np.count_nonzero(np.array(fractions) > 1e-6)
                    ),
                    "active_fraction": (activation > 0).mean(0).tolist(),
                    "teacher_access": False,
                }
                save_json(paths[task].with_name("information_diagnostic.json"), row)
                rows.append(row)
    save_json(
        root / "fits" / f"{case}_c{condition}" / "information_diagnostic.json",
        {
            "teacher_access": False,
            "sources": rows,
            "unidentified_neurons": sum(r["unidentified_neurons"] for r in rows),
        },
    )
    print(
        {
            "case": case,
            "condition": condition,
            "unidentified_neurons": sum(r["unidentified_neurons"] for r in rows),
            "tasks": len(rows),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = args.root.resolve()
    boundary = ObservationBoundary(root, args.case, args.condition)
    boundary.phase = "own_geometry_information"
    boundary.install()
    status = "failed"
    try:
        run(root, args.case, args.condition, args.device)
        status = "complete"
    finally:
        boundary.save(
            root / "receipts" / f"information_{args.case}_c{args.condition}.json",
            status,
        )


if __name__ == "__main__":
    main()
