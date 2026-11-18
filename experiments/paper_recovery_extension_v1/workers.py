"""Matched calibration, development and support-only target fitting."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from experiments.paper_structure_benchmark_v1.fitting.real_bank_fit import (
    candidates as native_candidates,
)
from experiments.paper_structure_benchmark_v1.fitting.real_bank_fit import (
    dense_initial,
)
from experiments.paper_structure_benchmark_v1.fitting.target_fit import TargetBoundary
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .adapter import fit_adam, losses, signed_basis, values
from .prepare import BASE, GRID


def load_basis(model, device):
    if bool(model.get("unshared", False)):
        return None
    if "basis" in model and model["basis"].size:
        return torch.tensor(model["basis"], device=device, dtype=torch.float32)
    index = torch.tensor(model["index"], device=device, dtype=torch.long)
    return {
        "index": index,
        "codes": int(model["codes"]),
        "scale": float(model["scale"]),
    }


def index_initial(basis, dense, reference, hidden, outputs):
    """Exact diagonal least squares for a signed one-hot design."""
    index, scale, codes = basis["index"], basis["scale"], basis["codes"]
    dimensions = index.shape[0]
    active = index.flatten() != 0
    label = index.abs().flatten()[active] - 1
    sign = index.sign().flatten()[active]
    counts = torch.bincount(label, minlength=codes).clamp_min(1)
    original = dense[:, : dimensions * hidden].reshape(-1, dimensions, hidden)
    sums = dense.new_zeros(codes).scatter_add_(
        0, label, original[reference].flatten()[active] * sign
    )
    c0 = sums / counts / scale
    proxy = c0[(index.abs() - 1).clamp_min(0)] * index.sign() * scale
    proxy = (proxy / proxy.norm(dim=0).clamp_min(1e-12)).cpu().numpy()
    aligned, biases, heads = [], [], []
    for row, weight in zip(dense, original):
        normal = (weight / weight.norm(dim=0).clamp_min(1e-12)).cpu().numpy()
        cost = ((proxy[:, :, None] - normal[:, None]) ** 2).sum(0)
        left, right = linear_sum_assignment(cost)
        mapping = torch.tensor(right[np.argsort(left)], device=dense.device)
        aligned.append(weight[:, mapping].flatten())
        biases.append(row[dimensions * hidden : dimensions * hidden + hidden][mapping])
        heads.append(
            row[dimensions * hidden + hidden : -outputs]
            .reshape(hidden, outputs)[mapping]
            .flatten()
        )
    matrix = torch.stack(aligned)[:, active] * sign
    code = dense.new_zeros(len(dense), codes)
    code.scatter_add_(1, label[None].expand(len(dense), -1), matrix)
    code /= counts * scale
    return torch.cat(
        (code, torch.stack(biases), torch.stack(heads), dense[:, -outputs:]), -1
    )[None]


def refit(root, output, method, codes, device, *, proposals=None):
    boundary = StudyBoundary(root)
    boundary.install()
    protocol = json.loads((root / "protocol.json").read_text())
    h, o = protocol["hidden"], protocol["outputs"]
    if proposals is not None:
        proposals = list(proposals)
    elif method in ("g34", "pair"):
        folder = (
            root
            / "source"
            / ("current_proposals" if method == "g34" else "ratio_proposals")
        )
        metadata = json.loads((folder / "candidates.json").read_text())["candidates"]
        arrays = dict(np.load(folder / "candidates.npz"))
        proposals = [
            (r["name"], arrays[r["name"]], r["anchor"])
            for r in metadata
            if r["codes"] == codes
        ]
    else:
        proposals = native_candidates(root, method, codes)[:8]
    folder = root / "fits" / f"{method}_k{codes}"
    folder.mkdir(exist_ok=False)
    save_json(
        folder / "registered.json",
        {
            "method": method,
            "codes": codes,
            "starts": 18,
            "steps": 2000,
            "candidates": [p[0] for p in proposals],
            "adapter_sha256": digest(Path(__file__).with_name("adapter.py")),
            "source_dense_sha256": digest(root / "source/dense/models.npz"),
        },
    )
    if not proposals:
        save_json(folder / "missing.json", {"reason": "no complete native candidates"})
        boundary.save(folder / "access_receipt.json", "no_candidates")
        return
    arrays = [
        dict(np.load(root / "public" / f"{n}.npz")) for n in ("support", "numerical")
    ]
    x, y, vx, vy = [
        torch.tensor(a[k], device=device, dtype=torch.float32)
        for a, k in (
            (arrays[0], "x"),
            (arrays[0], "y"),
            (arrays[1], "x"),
            (arrays[1], "y"),
        )
    ]
    x, vx = x / 255, vx / 255
    dense = torch.tensor(
        np.load(root / "source/dense/models.npz")["parameters"],
        device=device,
        dtype=torch.float32,
    )
    models, parameters, traces = [], [], []
    for name, array, reference in proposals:
        if array.ndim == 2:
            scale = float(np.sqrt(array.size / max(1, np.count_nonzero(array))))
            model = {
                "index": array,
                "codes": np.asarray(codes),
                "scale": np.asarray(scale),
            }
            # Preserve the existing feature evaluation at modest budgets.
            if codes <= 128:
                model["basis"] = (
                    (signed_basis(array, codes, device, torch.float32) * scale)
                    .cpu()
                    .numpy()
                )
        else:
            model = {
                "index": np.empty(0, dtype=np.int16),
                "basis": array
                / max(float(np.sqrt(np.mean(np.sum(array**2, -1)))), 1e-12),
            }
        basis = load_basis(model, device)
        initial = (
            index_initial(basis, dense, reference, h, o)
            if isinstance(basis, dict)
            else dense_initial(basis, dense, reference, h, o)
        )
        fitted, trace = fit_adam(
            x, y, vx, vy, basis, h, o, loss="cross_entropy", steps=2000, initial=initial
        )
        models.append(model)
        parameters.append(fitted[0].cpu().numpy())
        traces.append(trace)
        print(
            json.dumps(
                {
                    "method": method,
                    "codes": codes,
                    "candidate": name,
                    "done": len(models),
                    "total": len(proposals),
                }
            ),
            flush=True,
        )
    boundary.role = "select"
    panel = dict(np.load(root / "public/structure.npz"))
    mx = torch.tensor(panel["x"], device=device, dtype=torch.float32) / 255
    my = torch.tensor(panel["y"], device=device, dtype=torch.float32)
    with torch.no_grad():
        risk = np.stack(
            [
                losses(
                    values(
                        mx,
                        load_basis(m, device),
                        torch.tensor(p, device=device)[None],
                        h,
                        o,
                    ),
                    my,
                    "cross_entropy",
                )[0]
                .cpu()
                .numpy()
                for m, p in zip(models, parameters)
            ]
        )
    selected = int(risk.mean(1).argmin())
    np.savez_compressed(
        folder / "selected.npz", **models[selected], source_M_loss=risk[selected]
    )
    np.savez_compressed(
        folder / "curves.npz",
        **{
            f"candidate_{i}_{key}": trace[key]
            for i, trace in enumerate(traces)
            for key in ("steps", "train_history", "validation_history")
        },
    )
    save_json(
        folder / "selected.json",
        {
            "method": method,
            "codes": codes,
            "candidate": proposals[selected][0],
            "source_M_loss": float(risk[selected].mean()),
            "all_source_M_losses": risk.mean(1).tolist(),
            "selected_sha256": digest(folder / "selected.npz"),
        },
    )
    boundary.save(folder / "access_receipt.json", "complete")


def tune(root, output, method, device):
    boundary = StudyBoundary(root, "validation")
    boundary.install()
    folder = root / "validation_fit" / method
    folder.mkdir(exist_ok=False)
    protocol = json.loads((root / "protocol.json").read_text())
    h, o = protocol["hidden"], protocol["outputs"]
    with np.load(root / "validation/validation.npz") as panel:
        x, y, vx, vy = [
            torch.tensor(panel[k], device=device, dtype=torch.float32)
            for k in ("support_x", "support_y", "query_x", "query_y")
        ]
    x, vx = x / 255, vx / 255
    budgets = [0] if method == "dense" else list(GRID)
    previous = BASE / root.name / "validation_fit" / method
    cached = (
        dict(np.load(previous / "curves.npz"))
        if (previous / "curves.npz").exists()
        else {}
    )
    records, curves, best = [], {}, None
    for codes in budgets:
        model = {
            "unshared": np.asarray(True),
            "index": np.empty(0, dtype=np.int16),
            "basis": np.empty(0, dtype=np.float32),
        }
        if codes:
            path = root / "fits" / f"{method}_k{codes}/selected.npz"
            if not path.exists():
                continue
            model = {
                k: v for k, v in dict(np.load(path)).items() if k != "source_M_loss"
            }
        basis = load_basis(model, device)
        if f"k{codes}_validation_history" in cached:
            trace = {
                k.removeprefix(f"k{codes}_"): v
                for k, v in cached.items()
                if k.startswith(f"k{codes}_")
            }
            settings = [
                {"lr": lr, "quantile": q, "restart": r}
                for lr in (0.003, 0.01, 0.03)
                for q in (0.1, 0.5, 0.9)
                for r in range(2)
            ]
        else:
            _, trace = fit_adam(
                x, y, vx, vy, basis, h, o, loss="cross_entropy", steps=2000, seed=769003
            )
            settings = trace["settings"]
        score = trace["validation_history"].mean(-1).reshape(-1, 9, 2).mean(-1)
        step, setting = np.unravel_index(score.argmin(), score.shape)
        record = {
            "method": method,
            "codes": codes,
            "validation_mean_loss": float(score[step, setting]),
            "numeric_updates": int(trace["steps"][step]),
            "global_setting": settings[2 * setting],
            "support_count": int(x.shape[1]),
            "source_selection": {"codes": codes},
            "unshared": codes == 0,
        }
        records.append(record)
        budget_folder = folder / "budgets" / f"k{codes}"
        budget_folder.mkdir(parents=True)
        np.savez_compressed(budget_folder / "model.npz", **model)
        save_json(budget_folder / "selected.json", record)
        curves.update(
            {f"k{codes}_{k}": v for k, v in trace.items() if isinstance(v, np.ndarray)}
        )
        if (
            best is None
            or record["validation_mean_loss"] < best["validation_mean_loss"]
        ):
            best = record
            np.savez_compressed(folder / "model.npz", **model)
        print(json.dumps({"development": record}), flush=True)
    if best is None:
        raise RuntimeError(f"No model available for {method}")
    # Keep the best sharing model and the exact common no-sharing endpoint distinct.
    save_json(
        folder / "selected.json",
        {
            **best,
            "all_budget_choices": records,
            "model_sha256": digest(folder / "model.npz"),
        },
    )
    np.savez_compressed(folder / "curves.npz", **curves)
    boundary.save(folder / "access_receipt.json", "complete")


def target(root, output, method, device):
    boundary = TargetBoundary(root, root / "targets")
    boundary.install()
    folder = root / "targets" / method
    protocol = json.loads((root / "targets/sizes.json").read_text())
    selected = json.loads((folder / "selected.json").read_text())
    h, o = protocol["hidden"], protocol["outputs"]
    model = dict(np.load(folder / "model.npz"))
    basis = load_basis(model, device)
    support, query = [
        dict(np.load(root / "targets" / n)) for n in ("support.npz", "query_x.npz")
    ]
    predictions, parameters = [], []
    started = time.perf_counter()
    save_json(
        folder / "registered.json",
        {
            "model_sha256": digest(folder / "model.npz"),
            "selection_sha256": digest(folder / "selected.json"),
            "adapter_sha256": digest(Path(__file__).with_name("adapter.py")),
            "query_answers_used": False,
        },
    )
    for begin in range(0, len(support["x"]), 32):
        local = slice(begin, begin + 32)
        x, y, qx = [
            torch.tensor(a, device=device, dtype=torch.float32)
            for a in (support["x"][local], support["y"][local], query["x"][local])
        ]
        x, qx = x / 255, qx / 255
        fitted, _ = fit_adam(
            x,
            y,
            None,
            None,
            basis,
            h,
            o,
            loss="cross_entropy",
            seed=771003 + begin,
            fixed_step=selected["numeric_updates"],
            settings=[selected["global_setting"]],
        )
        with torch.no_grad():
            predictions.append(values(qx, basis, fitted[:1], h, o)[0].cpu().numpy())
        parameters.append(fitted[0].cpu().numpy())
    np.savez_compressed(
        folder / "frozen_predictions.npz",
        prediction=np.concatenate(predictions),
        parameters=np.concatenate(parameters),
    )
    save_json(
        folder / "access_receipt.json",
        {
            "status": "complete",
            "violations": boundary.violations,
            "opened_workspace_files": sorted(boundary.opened),
        },
    )
    save_json(
        folder / "metrics.json",
        {
            "status": "complete",
            "wall_seconds": time.perf_counter() - started,
            "predictions_sha256": digest(folder / "frozen_predictions.npz"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("refit", "tune", "target"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--codes", type=int)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.phase == "refit":
        refit(
            args.root.resolve(),
            args.output.resolve(),
            args.method,
            args.codes,
            torch.device(args.device),
        )
    elif args.phase == "tune":
        tune(
            args.root.resolve(),
            args.output.resolve(),
            args.method,
            torch.device(args.device),
        )
    else:
        target(
            args.root.resolve(),
            args.output.resolve(),
            args.method,
            torch.device(args.device),
        )
