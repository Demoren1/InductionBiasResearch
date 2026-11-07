"""Exact Gaussian regression sufficient statistics with nested sample budgets.

No observed matrix is approximated by its population covariance. Bartlett draws
sample Gram matrices; conditional Gaussian projected noise and chi-square
residual noise complete the exact distribution of X'X, Y'X, and diag(Y'Y).
"""

import numpy as np
import torch


def sample_index(family, kind, plan, device):
    rng = np.random.default_rng(plan["seed"] + family * 5 + 1)
    index = np.zeros((plan["outputs"], plan["inputs"]), dtype=np.int16)
    if kind == "conv":
        for output in range(plan["outputs"]):
            index[output, output : output + plan["codes"]] = np.arange(
                1, plan["codes"] + 1
            )
        index = index[rng.permutation(len(index))][:, rng.permutation(index.shape[1])]
    elif kind == "signed":
        positions = rng.permutation(index.size)
        begin = 0
        for code, count in enumerate((48, 12, 3), 1):
            index.flat[positions[begin : begin + count]] = code * rng.choice(
                (-1, 1), count
            )
            begin += count
    else:
        raise ValueError(f"Unknown geometry kind: {kind}")
    return torch.tensor(index, device=device).long()


def source_parameters(family, kind, excitation, plan, device):
    truth = sample_index(family, kind, plan, device)
    rng = np.random.default_rng(plan["seed"] + family * 5 + 2)
    codes = rng.normal(size=(plan["tasks"], plan["codes"]))
    if excitation == "weak":
        codes *= np.array((1.0, 0.3, 0.03))
    elif excitation == "collision":
        codes[:, 1] = codes[:, 0]
    elif excitation == "collinear":
        codes[:, 1] = 2 * codes[:, 0]
        codes[:, 2] = 3 * codes[:, 0]
    elif excitation != "full":
        raise ValueError(f"Unknown excitation: {excitation}")
    centers = torch.tensor(codes, device=device, dtype=torch.float64)
    weights = centers[:, truth.abs().clamp_min(1) - 1] * truth.sign()[None]
    return truth, weights


def design_root(family, condition, plan):
    rng = np.random.default_rng(plan["seed"] + family * 5 + 3)
    q, _ = np.linalg.qr(rng.normal(size=(plan["inputs"], plan["inputs"])))
    values = np.geomspace(1, condition, plan["inputs"])
    values /= values.mean()
    covariance = (q * values[None]) @ q.T
    return np.linalg.cholesky(covariance)


def draw_increment(n, weights, sigma, design, rng):
    tasks, outputs, inputs = weights.shape
    if n <= inputs:
        raise ValueError(
            "Each registered statistics increment must exceed input dimension"
        )
    bartlett = np.tril(rng.normal(size=(tasks, inputs, inputs)), k=-1)
    diagonal = np.sqrt(rng.chisquare(n - np.arange(inputs), size=(tasks, inputs)))
    bartlett[:, np.arange(inputs), np.arange(inputs)] = diagonal
    root = torch.tensor(design @ bartlett, device=weights.device, dtype=weights.dtype)
    gram = root @ root.transpose(-1, -2)
    standard = torch.tensor(
        rng.normal(size=(tasks, outputs, inputs)),
        device=weights.device,
        dtype=weights.dtype,
    )
    projected = sigma * (standard @ root.transpose(-1, -2))
    cross = weights @ gram + projected
    residual = torch.tensor(
        rng.chisquare(n - inputs, size=(tasks, outputs)),
        device=weights.device,
        dtype=weights.dtype,
    )
    square_y = (
        (weights @ gram * weights).sum(-1)
        + 2 * (weights * projected).sum(-1)
        + sigma**2 * (standard.square().sum(-1) + residual)
    )
    return {"gram": gram, "cross": cross, "square_y": square_y}


def nested_observations(family, kind, condition, excitation, noise, plan, device):
    truth, weights = source_parameters(family, kind, excitation, plan, device)
    design = design_root(family, condition, plan)
    rng = np.random.default_rng(plan["seed"] + family * 5 + 4)
    previous, halves = 0, [None, None]
    for n in plan["observations"]:
        for part in range(2):
            increment = draw_increment((n - previous) // 2, weights, noise, design, rng)
            if halves[part] is None:
                halves[part] = increment
            else:
                halves[part] = {
                    name: halves[part][name] + value
                    for name, value in increment.items()
                }
        total = {name: halves[0][name] + halves[1][name] for name in halves[0]}
        previous = n
        yield n, truth, total, halves


def fit_stats(stats):
    gram, cross = stats["gram"], stats["cross"]
    fitted = torch.linalg.solve(gram, cross.transpose(-1, -2)).transpose(-1, -2)
    residual = stats["square_y"] - (fitted * cross).sum(-1)
    if float(residual.min()) < -1e-6:
        raise FloatingPointError(
            "Negative regression residual from sufficient statistics"
        )
    return fitted, residual.clamp_min(0), torch.linalg.inv(gram)
