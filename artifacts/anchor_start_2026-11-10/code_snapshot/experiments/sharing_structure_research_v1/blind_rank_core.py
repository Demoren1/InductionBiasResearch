"""Task-level uncertainty and gauge-invariant codebook diversity control."""

import json
import math
from datetime import UTC, datetime

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .blind_contract import digest, load_sizes, save_json
from .blind_numeric import numpy_predict
from .blind_orientation_refine import graph_components
from .blind_report import dense_predict


def diverse_codebook(code, index, codes):
    components = graph_components(index, codes)
    used = np.unique(np.abs(index[index != 0])) - 1
    if len(used) != codes:
        return {
            "logdet": -100.0 * (codes - len(used)),
            "eigenvalues": [],
            "components": len(components),
            "used": len(used),
        }
    task_count = len(code)
    membership = np.empty(codes, dtype=int)
    for group, component in enumerate(components):
        membership[component] = group

    def objective(values):
        logscale = np.zeros((task_count, len(components)))
        if len(components) > 1:
            logscale[:, 1:] = values.reshape(task_count, -1)
        logscale -= logscale.max(1, keepdims=True)
        weight = code * np.exp(logscale[:, membership])
        norm = np.maximum(np.sqrt(np.mean(weight**2, axis=1, keepdims=True)), 1e-12)
        row = weight / norm
        deviation = np.maximum(np.sqrt(np.mean(row**2, axis=0)), 1e-12)
        whitened = row / deviation
        gram = whitened.T @ whitened / task_count + 1e-8 * np.eye(codes)
        value = -float(np.linalg.slogdet(gram)[1])
        derivative = -2 * whitened @ np.linalg.inv(gram) / task_count
        row_grad = (
            derivative / deviation
            - row * np.mean(derivative * row, axis=0) / deviation**3
        )
        weight_grad = (
            row_grad - row * np.mean(row_grad * row, axis=1, keepdims=True)
        ) / norm
        products = weight_grad * weight
        gradient = np.stack(
            [
                products[:, membership == group].sum(1)
                for group in range(len(components))
            ],
            axis=1,
        )
        return value, gradient[:, 1:].ravel(), gram

    count = task_count * (len(components) - 1)
    if count:
        rng = np.random.default_rng(33681013)
        starts = [np.zeros(count), rng.normal(0, 0.5, count), rng.normal(0, 0.5, count)]
        best = None
        for initial in starts:
            result = minimize(
                lambda a: objective(a)[:2],
                initial,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 80, "gtol": 1e-6},
            )
            if best is None or result.fun < best.fun:
                best = result
        value, _, gram = objective(best.x)
    else:
        value, _, gram = objective(np.empty(0))
    return {
        "logdet": -value,
        "eigenvalues": np.linalg.eigvalsh(gram).tolist(),
        "components": len(components),
        "used": len(used),
    }


def select(root, case, condition):
    sizes = load_sizes(root, case)
    folder = root / "fits" / f"{case}_c{condition}"
    fitted = dict(np.load(folder / "joint_fitted.npz"))
    meta = dict(np.load(root / "public" / f"{case}_c{condition}" / "meta.npz"))
    tasks = fitted["parameters"].shape[1]
    risks, predictions, point_errors = [], [], []
    target = np.stack([meta[f"y_{t}"] for t in range(tasks)])
    for index, parameters in zip(fitted["indices"], fitted["parameters"]):
        value = np.stack(
            [
                numpy_predict(
                    meta[f"x_{t}"],
                    index,
                    parameters[t],
                    sizes["codes"],
                    sizes["outputs"],
                )
                for t in range(tasks)
            ]
        )
        predictions.append(value)
        errors = ((value - target) ** 2).mean(2)
        point_errors.append(errors)
        risks.append(errors.mean(1))
    risks = np.array(risks)
    mean = risks.mean(1)
    best = int(mean.argmin())
    difference = risks - risks[best]
    standard_error = difference.std(1, ddof=1) / math.sqrt(tasks)
    clustered = difference.mean(1) <= 2 * standard_error + 1e-12
    dense = np.load(root / "dense" / f"{case}_c{condition}" / "repaired.npz")[
        "parameters"
    ]
    dense_errors = np.stack(
        [
            ((dense_predict(meta[f"x_{t}"], dense[t], sizes) - target[t]) ** 2).mean(1)
            for t in range(tasks)
        ]
    )
    excess = np.array(point_errors) - dense_errors[None]
    query_error = excess.std(2, ddof=1) / math.sqrt(excess.shape[2])
    threshold = float(norm.ppf(1 - 0.05 / tasks))
    passes_dense_guard = np.all(
        excess.mean(2) <= threshold * query_error + 1e-12, axis=1
    )
    eligible = np.flatnonzero(clustered & passes_dense_guard)
    fallback = not len(eligible)
    if fallback:
        eligible = np.array([best])
    quality = {
        int(i): diverse_codebook(
            fitted["parameters"][i, :, : sizes["codes"]],
            fitted["indices"][i],
            sizes["codes"],
        )
        for i in eligible
    }
    chosen = int(
        max(
            eligible,
            key=lambda i: (
                quality[int(i)]["used"],
                quality[int(i)]["logdet"],
                -mean[i],
            ),
        )
    )
    np.savez_compressed(
        folder / "task_rank_selected.npz",
        index=fitted["indices"][chosen],
        parameters=fitted["parameters"][chosen],
        meta_predictions=predictions[chosen],
    )
    save_json(
        folder / "task_rank_selected.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "candidate": chosen,
            "meta_mse": float(mean[chosen]),
            "task_meta_mse": risks[chosen].tolist(),
            "best_functional_candidate": best,
            "eligible": eligible.tolist(),
            "dense_guard_threshold": threshold,
            "passes_dense_guard": np.flatnonzero(passes_dense_guard).tolist(),
            "functional_fallback": fallback,
            "quality": quality,
            "teacher_access": False,
            "assumption": "Among functionally similar candidates, prefer using K varying code directions across tasks, after function-preserving positive component scales. Generic non-degeneracy preference; not a universal identifiability guarantee.",
            "sha256": digest(folder / "task_rank_selected.npz"),
        },
    )
    print(
        json.dumps(
            {
                "case": case,
                "condition": condition,
                "task_rank_candidate": chosen,
                "meta_mse": float(mean[chosen]),
                "eligible": len(eligible),
                "quality": quality[chosen],
            }
        ),
        flush=True,
    )
