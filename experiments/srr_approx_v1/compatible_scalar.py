"""Blind scalar-output ReLU fits, local signed-group proposals and joint support."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import combinations, product

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from experiments.sharing_structure_research_v1.blind_contract import save_json
from experiments.srr_approx_v1.compatible import OUTPUT, RelationBank
from experiments.srr_approx_v1.compatible_synthetic import (
    CASES,
    D,
    HOLDOUT,
    M,
    TASKS,
    TOLERANCE,
    alternatives,
    design,
    exact_population_risk,
    extracted,
    generate,
    matchings,
)
from experiments.srr_approx_v1.core import signed_forest


FOLDER = OUTPUT / "scalar_relu"
SOURCES = 12
SEED = 94811930
NUMERICAL_STEPS = 384
TRAIN = 256
LOCAL_VISITS, LOCAL_KEEP = 48, 6


def greedy(weight):
    magnitude = np.linalg.norm(weight, axis=0, keepdims=True)
    z = weight / np.maximum(magnitude, 1e-12)
    available = set(range(len(z)))
    group = []
    while available:
        edges = [
            (i, j, s) for i, j in combinations(sorted(available), 2) for s in (-1, 1)
        ]
        edge = min(edges, key=lambda e: np.sum((z[e[0]] - e[2] * z[e[1]]) ** 2))
        group.append(edge)
        available -= set(edge[:2])
    return frozenset(group)


@lru_cache(maxsize=256)
def neighbors(model):
    result = set()
    for i, j, s in model:
        result.add((model - {(i, j, s)}) | {(i, j, -s)})
    for a, b in combinations(model, 2):
        shared = model - {a, b}
        for k, l in ((b[0], b[1]), (b[1], b[0])):
            i, j = sorted((a[0], k))
            p, q = sorted((a[1], l))
            for s, t in product((-1, 1), repeat=2):
                result.add(shared | {(i, j, s), (p, q, t)})
    return tuple(sorted(result, key=lambda m: tuple(sorted(m))))


def propose(bank, fitted, training_cost):
    """Generate locally. Only score visited states, and use no validation data."""
    lookup = {m: i for i, m in enumerate(bank.full)}
    pooled = set()
    for fits, scores in zip(fitted, training_cost, strict=True):
        cost = {}

        def visit(model):
            if model not in cost:
                cost[model] = scores[lookup[model]]

        for w in fits:
            visit(greedy(w))
        expanded = set()
        while len(cost) < LOCAL_VISITS:
            usable = [m for m in cost if m not in expanded]
            if not usable:
                break
            current = min(usable, key=lambda m: (cost[m], lookup[m]))
            expanded.add(current)
            for nxt in neighbors(current):
                visit(nxt)
                if len(cost) == LOCAL_VISITS:
                    break
        pooled.update(
            lookup[m]
            for m in sorted(cost, key=lambda m: (cost[m], lookup[m]))[:LOCAL_KEEP]
        )
    return sorted(pooled)


def adam(features, response, w, bias, head, offset, steps):
    params = [v.detach().clone().requires_grad_() for v in (w, bias, head, offset)]
    optimizer = torch.optim.Adam(params, lr=0.025)
    for _ in range(steps):
        w, bias, head, offset = params
        z = (features @ w + bias[..., None, :]).relu()
        pred = (z * head[..., None, :]).sum(-1) + offset[..., None]
        risk = (pred - response[None, :, None]).square().mean(-1)
        optimizer.zero_grad(set_to_none=True)
        risk.sum().backward()
        optimizer.step()
    with torch.no_grad():
        z = (features @ w + bias[..., None, :]).relu()
        pred = (z * head[..., None, :]).sum(-1) + offset[..., None]
        risk = (pred - response[None, :, None]).square().mean(-1)
    return [v.detach() for v in params], risk


def best(params, loss, keep):
    order = loss.argsort(dim=0)[:keep]
    selected = []
    for p in params:
        idx = order.reshape(*order.shape, *([1] * (p.ndim - 3)))
        selected.append(torch.gather(p, 0, idx.expand(-1, *p.shape[1:])))
    return selected


def trained_bank(x, y, vx, vy, basis, seed, device):
    rng = torch.Generator(device=device).manual_seed(seed)

    def normal(shape):
        return torch.randn(shape, device=device, generator=rng)

    x, y, vx, vy, b = [
        torch.as_tensor(v, dtype=torch.float32, device=device)
        for v in (x, y, vx, vy, basis)
    ]
    tasks, total, h = len(x), len(b), 2
    full = (10, tasks, 1)
    params, ce = adam(
        x[:, None],
        y,
        normal((*full, D, h)) / np.sqrt(D),
        normal((*full, h)) * 0.2,
        normal((*full, h)),
        torch.zeros(full, device=device),
        NUMERICAL_STEPS,
    )
    w, bias, head, offset = best(params, ce, 4)
    fit = x[:, None] @ b
    holdout = vx[:, None] @ b
    dense_train = x[:, None] @ w
    # Positive column scales and hidden permutations need not be shared between tasks.
    projection = torch.linalg.pinv(fit, rcond=1e-5) @ dense_train
    proj_cost = (fit @ projection - dense_train).square().mean(dim=(-1, -2)).amin(dim=0)
    random = (2, tasks, total)
    start_w = torch.cat([projection, normal((*random, M, h)) / np.sqrt(D)])
    start_bias = torch.cat([bias.expand(-1, -1, total, -1), normal((*random, h)) * 0.2])
    start_head = torch.cat([head.expand(-1, -1, total, -1), normal((*random, h))])
    start_offset = torch.cat(
        [offset.expand(-1, -1, total), torch.zeros(random, device=device)]
    )
    all_params, train = adam(
        fit, y, start_w, start_bias, start_head, start_offset, NUMERICAL_STEPS
    )
    fw, fb, fh, fo = best(all_params, train, 1)
    pred = (
        ((holdout @ fw + fb[..., None, :]).relu() * fh[..., None, :]).sum(-1)
        + fo[..., None]
    )[0]
    heldout = ((pred - vy[:, None]).square() / 4).clamp_max(1).mean(-1)
    return (
        heldout.cpu().numpy(),
        proj_cost.cpu().numpy(),
        w[:, :, 0].permute(1, 0, 2, 3).cpu().numpy(),
        float(ce.amin(dim=0).mean()),
    )


def ordinary(dense):
    z = dense / np.maximum(np.linalg.norm(dense, axis=2, keepdims=True), 1e-12)
    profiles = []
    for fits in z:
        representative = fits[0]
        within = [w[:, linear_sum_assignment(-representative.T @ w)[1]] for w in fits]
        order = linear_sum_assignment(-z[0, 0].T @ representative)[1]
        profiles.append(np.mean(within, axis=0)[:, order])
    return signed_forest(np.stack(profiles), [2 * M])[0][2 * M]


def serialize(family):
    return [sorted(s) for s in family]


def trial(bank, basis, number, case, device, *, confirmation=False):
    seed = (80419190 if confirmation else SEED) + number
    rng = np.random.default_rng(seed)
    pairs, planted = alternatives(bank, rng, case)
    tx, ty, qx, qy, pop = [], [], [], [], []
    for pair in pairs:
        proj, w = generate(rng, basis, pair, 2, 0)
        head = rng.choice((-1, 1), 2) * rng.uniform(0.75, 1.25, 2)
        bias = rng.uniform(-0.35, 0.35, 2)
        offset = rng.normal(0, 0.2)
        x = rng.normal(size=(TRAIN + HOLDOUT, D)) @ proj
        signal = np.maximum(x @ w + bias, 0) @ head + offset
        y = signal + rng.normal(0, 0.035, size=len(x))
        tx.append(x[:TRAIN])
        ty.append(y[:TRAIN])
        qx.append(x[TRAIN:])
        qy.append(y[TRAIN:])
        pop.append(exact_population_risk(proj, w, basis))
    risk, training, dense, dense_mse = trained_bank(
        *map(np.stack, (tx, ty, qx, qy)), basis, seed, device
    )
    pool = propose(bank, dense, training)
    matched = sorted(rng.choice(len(basis), len(pool), replace=False))
    oracle = bank.infer(np.stack(pop), 1e-10)
    result = {
        "dense_train_mse": dense_mse,
        "bank_size": len(pool),
        "oracle_count": len(oracle["maximal"]),
        "oracle_is_planted": set(oracle["maximal"]) == set(planted["maximal"]),
        "ordinary_ward_is_maximal": extracted(ordinary(dense))
        in set(oracle["maximal"]),
    }
    strategies = {
        "local": pool,
        "exhaustive": list(range(len(basis))),
        "uniform": matched,
    }
    if confirmation:
        limit = 48
        strategies["witness_cover"] = bank.witness_cover(training, limit)
        # Equal refit budget. Cycle through the ranked lists of source tasks.
        order = training.argsort(axis=1, kind="stable").T.ravel()
        strategies["round_robin"] = list(dict.fromkeys(order.tolist()))[:limit]
        strategies["uniform48"] = sorted(rng.choice(len(basis), limit, replace=False))
    for name, ids in strategies.items():
        subset = RelationBank([bank.full[j] for j in ids])
        found = subset.infer(risk[:, ids], TOLERANCE)
        true = set(oracle["maximal"])
        result[name] = {
            "population_family_exact": set(found["maximal"]) == true,
            "planted_family_exact": set(found["maximal"]) == set(planted["maximal"]),
            "family": serialize(found["maximal"]),
            "mandatory": sorted(found["necessary"]),
            "mandatory_population_exact": found["necessary"] == oracle["necessary"],
            "complete_witness_per_task": float(
                np.mean(np.min(np.stack(pop)[:, ids], axis=1) < 1e-10)
            ),
            "full_true_frontier_covered": all(
                all(
                    any(s <= bank.full[j] and pop[t][j] < 1e-10 for j in ids)
                    for t in range(TASKS)
                )
                for s in true
            ),
            "mean_solutions": float(found["solutions_per_task"].mean()),
            "erm_is_maximal": subset.full[int(risk[:, ids].mean(axis=0).argmin())]
            in true,
            "vote_is_maximal": subset.full[
                np.bincount(risk[:, ids].argmin(axis=1), minlength=len(ids)).argmax()
            ]
            in true,
        }
    result["population_family"] = serialize(oracle["maximal"])
    result["pool"] = pool
    return result, risk, training


def run(gpus, confirmation=False):
    folder = OUTPUT / "scalar_confirmation" if confirmation else FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    bank = RelationBank(matchings(list(range(D))))
    basis = np.stack([design(s) for s in bank.full])
    count = 16 if confirmation else SOURCES
    first = 80419190 if confirmation else SEED
    jobs = list(product(range(count), CASES))
    save_json(
        folder / "plan.json",
        {
            "seeds": list(range(first, first + count)),
            "cases": CASES,
            "source_tasks": TASKS,
            "fit": TRAIN,
            "held_out": HOLDOUT,
            "noise_sd": 0.035,
            "scalar_output": True,
            "architecture": "two ReLUs, fitted biases, output weights and intercept",
            "starts_dense": 10,
            "retained_dense": 4,
            "starts_constrained": 6,
            "adam_steps_each": NUMERICAL_STEPS,
            "signed_pairs_per_index": M,
            "tolerance": TOLERANCE,
            "coverage": 1,
            "local_visits": LOCAL_VISITS,
            "local_retained_per_task": LOCAL_KEEP,
            "full_bank": len(basis),
            "confirmation": confirmation,
            "fixed_budget_if_confirming": 48,
        },
    )
    torch.set_num_threads(1)

    def worker(slot):
        device = f"cuda:{gpus[slot]}"
        torch.cuda.set_device(device)
        for number, case in jobs[slot :: len(gpus)]:
            key = f"{number:02d}_{case}"
            dest = folder / f"{key}.json"
            if dest.exists():
                continue
            result, risk, training = trial(
                bank, basis, number, case, device, confirmation=confirmation
            )
            np.save(folder / f"{key}.npy", risk)
            if confirmation:
                np.save(folder / f"{key}_proxy.npy", training)
            save_json(dest, result)
            print(key, result["local"]["population_family_exact"], flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, range(len(gpus))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--confirmation", action="store_true")
    args = parser.parse_args()
    run(args.gpus, args.confirmation)
