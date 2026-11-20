"""Partial cross-correlation with an equal, known receptive field for all methods."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from experiments.asrr_benchmark_families_v1.groups import signed_bank
from experiments.asrr_benchmark_families_v1.linear import (
    atomic_npz,
    panel,
    structural,
)
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .common import CONFIG, OUT, SEED, freeze, register
from .hard import fit_indexators

CFG = CONFIG["synthetic"]
INPUTS, OUTPUTS, WIDTH = 70, 68, 3
WINDOW = np.arange(OUTPUTS)[:, None] + np.arange(WIDTH)
OUTPUT = np.broadcast_to(np.arange(OUTPUTS)[:, None], WINDOW.shape)
CONNECTIONS = OUTPUTS * WIDTH
CODES = WIDTH * (1 + CFG["exceptions_per_offset"])
VALIDATION_EXAMPLES = 16


def teacher(rng):
    index = np.zeros((INPUTS, OUTPUTS), dtype=np.int32)
    labels = np.broadcast_to(np.arange(1, WIDTH + 1), WINDOW.shape).copy()
    for offset in range(WIDTH):
        private = rng.choice(
            OUTPUTS, CFG["exceptions_per_offset"], replace=False
        )
        start = WIDTH + 1 + offset * len(private)
        labels[private, offset] = np.arange(start, start + len(private))
    index[WINDOW, OUTPUT] = labels
    return index


def index_matrix(labels):
    index = np.zeros((INPUTS, OUTPUTS), dtype=np.int32)
    index[WINDOW, OUTPUT] = labels.reshape(OUTPUTS, WIDTH)
    return index


def signed_design(labels):
    labels = np.asarray(labels, dtype=np.int64)
    return (
        np.eye(np.abs(labels).max() + 1)[np.abs(labels), 1:]
        * np.sign(labels)[:, None]
    )


def budgeted_map(probability):
    """Maximize assignment scores while occupying every code at least once."""
    ordinary = np.argmax(probability, axis=-1)
    if len(np.unique(ordinary)) == probability.shape[1]:
        return ordinary + 1
    cost = probability.max(axis=-1, keepdims=True) - probability
    codes, positions = linear_sum_assignment(cost.T)
    ordinary[positions] = codes
    return ordinary + 1


def source_regression(x, y):
    fit_count = x.shape[1] - VALIDATION_EXAMPLES
    fit = x[:, :fit_count, WINDOW].transpose(0, 2, 1, 3)
    y_fit = y[:, :fit_count].transpose(0, 2, 1)[..., None]
    xt = fit.swapaxes(-1, -2)
    weight = np.linalg.solve(xt @ fit, xt @ y_fit)[..., 0]
    prediction = np.einsum("tnhw,thw->tnh", x[:, fit_count:, WINDOW], weight)
    error = np.square(prediction - y[:, fit_count:]).mean((1, 2))
    return weight, error


def hard_candidates(method, x, y, weights, quality):
    settings = CONFIG["hard_source"]
    tasks = np.argsort(quality)[: settings["tasks"]]
    inputs = torch.from_numpy(x[tasks][:, :, WINDOW].astype(np.float32))
    answers = torch.from_numpy(y[tasks].astype(np.float32))
    initial = torch.from_numpy(
        weights[tasks].reshape(len(tasks), -1).astype(np.float32)
    )
    fit_count = x.shape[1] - VALIDATION_EXAMPLES

    def loss(weight, _tail, *, validation):
        span = slice(fit_count, None) if validation else slice(None, fit_count)
        prediction = (
            inputs[:, span] * weight.reshape(len(tasks), 1, OUTPUTS, WIDTH)
        ).sum(-1)
        return (prediction - answers[:, span]).square().mean((1, 2))

    result = {}
    for rate in settings["rates"]:
        labels, _, steps = fit_indexators(
            method,
            initial,
            CONNECTIONS,
            CODES,
            loss,
            {**settings, "rate": rate},
        )
        for task, row, step in zip(tasks, labels, steps, strict=True):
            result[f"task{task}_rate{rate:g}_step{step}"] = row
    return result


def design(x, basis):
    return torch.einsum("tnhw,hwk->tnhk", x, basis.reshape(OUTPUTS, WIDTH, -1))


def inner_ridge(a, y, strength):
    energy = a.square().sum((1, 2)) / a.shape[-1]
    gram = a.mT @ a
    stabilizer = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    return torch.linalg.solve(
        gram + strength * energy[:, None, None].clamp_min(1e-8) * stabilizer,
        a.mT @ y,
    )


def continuous_candidates(method, x, y, weights, quality, seed):
    data = torch.tensor(x[:, :, WINDOW], dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    source = weights.reshape(len(weights), -1)
    svd = torch.from_numpy(
        np.linalg.svd(source, full_matrices=False)[2][:CODES].T
    ).float()
    exemplar = source[np.argmin(quality)]
    centers = np.quantile(exemplar, np.linspace(0, 1, CODES))
    scale = max(exemplar.std() / 3, 1e-6)
    warm = torch.tensor(
        -0.5 * ((exemplar[:, None] - centers) / scale) ** 2,
        dtype=torch.float32,
    ).clamp_min(-16)
    choices = {}
    for rate, init in product(CFG["outer_rates"], ("cold", "source")):
        parameter = torch.randn(CONNECTIONS, CODES, generator=generator)
        if init == "cold":
            parameter = (
                torch.linalg.qr(parameter).Q
                if method == "msr"
                else parameter * 0.2
            )
        else:
            parameter = svd.clone() if method == "msr" else warm.clone()
        parameter.requires_grad_()
        optimizer = torch.optim.Adam([parameter], lr=rate)
        for step in range(CFG["outer_steps"] + 1):
            basis = parameter if method == "msr" else parameter.softmax(-1)
            if step in (0, 200, CFG["outer_steps"]):
                with torch.no_grad():
                    candidate = (
                        torch.linalg.qr(basis).Q if method == "msr" else basis
                    )
                    choices[f"{init}_rate{rate:g}_step{step}"] = (
                        candidate.detach().numpy().copy()
                    )
            if step == CFG["outer_steps"]:
                break
            tasks = torch.randint(len(x), (8,), generator=generator)
            sample = torch.randperm(x.shape[1], generator=generator)[:17]
            shot = CFG["shots"][step % len(CFG["shots"])]
            values = design(data[tasks][:, sample], basis)
            ys = target[tasks][:, sample]
            fit = values[:, :shot].reshape(len(tasks), -1, CODES)
            coefficient = inner_ridge(
                fit, ys[:, :shot].reshape(len(tasks), -1, 1), 0.003
            )
            prediction = (
                values[:, shot:] * coefficient[:, None, None, :, 0]
            ).sum(-1)
            risk = (prediction - ys[:, shot:]).square().mean()
            if method == "yeh":
                entropy = (
                    -(basis * basis.clamp_min(1e-12).log()).sum(-1).mean()
                )
                risk = risk + 0.002 * entropy
            optimizer.zero_grad(set_to_none=True)
            risk.backward()
            optimizer.step()
    return choices


def ridge_predictions(x, y, shot, basis, strengths):
    sx = x[:, :shot, WINDOW]
    qx = x[:, max(CFG["shots"]) :, WINDOW]
    if basis is None:
        a = sx.transpose(0, 2, 1, 3)
        response = y[:, :shot].transpose(0, 2, 1)[..., None]
    else:
        b = basis.reshape(OUTPUTS, WIDTH, -1).astype(np.float64)
        a = np.einsum("tnhw,hwk->tnhk", sx, b, optimize=True)
        a = a.reshape(len(x), -1, basis.shape[-1])
        response = y[:, :shot].reshape(len(x), -1, 1)
    energy = np.square(a).sum((-2, -1), keepdims=True) / a.shape[-1]
    energy = np.maximum(energy, 1e-12)
    xt = a.swapaxes(-1, -2)
    dual = a.shape[-2] < a.shape[-1]
    if dual:
        gram, rhs = a @ xt, response
    else:
        gram, rhs = xt @ a, xt @ response
    identity = np.eye(gram.shape[-1])
    for strength in strengths:
        coefficients = np.linalg.solve(
            gram + (strength * energy) * identity, rhs
        )
        if dual:
            coefficients = xt @ coefficients
        if basis is None:
            weight = coefficients[..., 0]
        else:
            weight = np.einsum("hwk,tk->thw", b, coefficients[..., 0])
        yield np.einsum("tnhw,thw->tnh", qx, weight)


def select(x, y, variants, *, average=False):
    choices, held_out = {}, {}
    truth = y[:, max(CFG["shots"]) :]
    for shot in CFG["shots"]:
        scores = []
        for name, basis in variants.items():
            candidates = basis if average else [basis]
            error = np.stack(
                [
                    [
                        np.square(p - truth).mean()
                        for p in ridge_predictions(
                            x, y, shot, value, CFG["ridge"]
                        )
                    ]
                    for value in candidates
                ]
            ).mean(0)
            scores.extend(
                {"candidate": name, "ridge": reg, "mse": float(val)}
                for reg, val in zip(CFG["ridge"], error, strict=True)
            )
        held_out[str(shot)] = scores
        choices[str(shot)] = min(scores, key=lambda row: row["mse"])
    return choices, held_out


def phase(geometry, stage):
    torch.set_num_threads(1)
    folder = OUT / "linear" / f"geometry_{geometry:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stage}_results.json"
    if path.exists():
        return
    seed = SEED + 2137 * geometry
    index = teacher(np.random.default_rng(seed))
    x, y = panel(
        index,
        CFG["source_tasks"],
        CFG["source_examples"],
        seed + 1,
        CFG["source_noise"],
    )
    dev_x, dev_y = panel(
        index,
        CFG["development_tasks"],
        max(CFG["shots"]) + CFG["query_examples"],
        seed + 2,
        0,
    )
    weights, quality = source_regression(x, y)
    indices, bases = {}, {}
    if stage == "baselines":
        bases["dense"] = {"untied": None}
        for method in ("wsr", "lutq"):
            indices[method] = hard_candidates(method, x, y, weights, quality)
        yeh = continuous_candidates("yeh", x, y, weights, quality, seed + 5)
        indices["yeh_map"] = {
            key: budgeted_map(row) for key, row in yeh.items()
        }
        bases["yeh_soft"] = yeh
        bases["msr"] = continuous_candidates(
            "msr", x, y, weights, quality, seed + 5
        )
    else:
        profiles = np.zeros((len(weights), *index.shape))
        profiles[:, WINDOW, OUTPUT] = weights
        ward = signed_bank(profiles, index != 0, (CODES,), partial=False)[
            CODES
        ]
        indices["asrr"] = {"ward": ward[WINDOW, OUTPUT].ravel()}
        indices["oracle"] = {"teacher": index[WINDOW, OUTPUT].ravel()}
        rng = np.random.default_rng(seed + 4)
        signs = np.sign(ward[WINDOW, OUTPUT].ravel())
        labels = np.abs(ward[WINDOW, OUTPUT].ravel())
        shuffle = [
            signs * rng.permutation(labels)
            for _ in range(CONFIG["target"]["shuffle_replicates"])
        ]
        bases["shuffled"] = {
            "four_permutations": [signed_design(row) for row in shuffle]
        }
    for method, candidates in indices.items():
        bases[method] = {
            key: signed_design(row) for key, row in candidates.items()
        }
    choices, grid = {}, {}
    for method, variants in bases.items():
        choices[method], grid[method] = select(
            dev_x, dev_y, variants, average=method == "shuffled"
        )
        print(f"linear {geometry} {method}: frozen", flush=True)
    freeze(folder / f"{stage}_choices.json", choices)
    save_json(folder / f"{stage}_validation.json", grid)
    test_x, test_y = panel(
        index,
        CFG["test_tasks"],
        max(CFG["shots"]) + CFG["query_examples"],
        seed + 3,
        0,
    )
    per_task, recovery = {}, {}
    for method, rows in choices.items():
        for shot, row in rows.items():
            name = row["candidate"]
            values = bases[method][name]
            if method != "shuffled":
                values = [values]
            error = [
                np.square(
                    next(
                        ridge_predictions(
                            test_x, test_y, int(shot), basis, [row["ridge"]]
                        )
                    )
                    - test_y[:, max(CFG["shots"]) :]
                ).mean((1, 2))
                for basis in values
            ]
            per_task[f"{method}_{shot}"] = np.mean(error, axis=0)
            if method in indices:
                recovery[f"{method}_{shot}"] = structural(
                    index_matrix(indices[method][name]), index
                )
    atomic_npz(folder / f"{stage}_per_task.npz", **per_task)
    save_json(
        path,
        {
            "mse": {k: float(v.mean()) for k, v in per_task.items()},
            "structure": recovery,
        },
    )
    print(f"linear {geometry} {stage}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("baselines", "ward"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    register()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(
            pool.map(
                phase,
                range(CFG["geometries"]),
                [args.stage] * CFG["geometries"],
            )
        )
