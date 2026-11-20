"""Continuous tasks with several exactly equivalent signed weight assignments."""

import argparse
import itertools

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import save_json
from experiments.srr_approx_v1.compatible import OUTPUT, RelationBank
from experiments.srr_approx_v1.core import signed_forest


CASES = ("unique", "optional_core", "necessary_core", "two_full", "exclusive")
TASKS, D, M = 8, 6, 3
REPLICATES = 32
FIT, HOLDOUT = 192, 256
TOLERANCE = 0.001
SEED = 11974011


def matchings(items):
    if not items:
        yield ()
        return
    i, *others = items
    for j in others:
        for sign in (1, -1):
            for tail in matchings([k for k in others if k != j]):
                yield ((i, j, sign), *tail)


def design(relations):
    """Any unconstrained position gets a separate free code."""
    columns, used = [], set()
    for i, j, s in sorted(relations):
        v = np.zeros(D)
        v[i], v[j] = 1, s
        columns.append(v)
        used.update((i, j))
    columns.extend(np.eye(D)[j] for j in range(D) if j not in used)
    return np.stack(columns, axis=1)


def as_index(relations, hidden=1):
    basis = design(relations)
    row, label = np.nonzero(basis)
    return (
        np.sign(basis[row, label, None])
        * (label[:, None] * hidden + np.arange(1, hidden + 1))
    ).astype(np.int16)


def extracted(index):
    atoms = set()
    for i, j in itertools.combinations(range(len(index)), 2):
        for sign in (1, -1):
            if np.array_equal(index[i], sign * index[j]):
                atoms.add((i, j, sign))
    return frozenset(atoms)


def alternatives(bank, rng, case):
    anchor = int(rng.integers(len(bank.full)))
    base = bank.full[anchor]
    which = rng.permutation(len(base))
    a, b = [sorted(base)[j] for j in which[:2]]
    ids = np.arange(len(bank.full))
    ina = np.array([a in row for row in bank.full])
    inb = np.array([b in row for row in bank.full])
    overlap = np.array([len(base & row) for row in bank.full])
    if case == "two_full":
        alternative = int(rng.choice(ids[overlap == 1]))
    for _ in range(1000):
        if case == "unique":
            pairs = [(anchor, int(rng.choice(ids[overlap == 0]))) for _ in range(TASKS)]
            expected = (base,)
        elif case == "two_full":
            pairs = [(anchor, alternative)] * TASKS
            expected = (base, bank.full[alternative])
        elif case == "necessary_core":
            pairs = [
                tuple(rng.choice(ids[ina], 2, replace=False)) for _ in range(TASKS)
            ]
            expected = (frozenset([a]),)
        elif case == "optional_core":
            pairs = [
                (int(rng.choice(ids[ina])), int(rng.choice(ids[~ina])))
                for _ in range(TASKS)
            ]
            expected = (frozenset([a]),)
        elif case == "exclusive":
            pairs = [
                (int(rng.choice(ids[ina & ~inb])), int(rng.choice(ids[inb & ~ina])))
                for _ in range(TASKS)
            ]
            expected = (frozenset([a]), frozenset([b]))
        else:
            raise ValueError(case)
        # Generate a family with precisely the requested combinatorial scenario.
        # No model scores, features or examples have been generated at this point.
        fake = np.ones((TASKS, len(ids)))
        fake[np.arange(TASKS)[:, None], pairs] = 0
        truth = bank.infer(fake, 0)
        if set(truth["maximal"]) == set(expected):
            return np.array(pairs), truth
    raise RuntimeError("Could not draw the prescribed structure family")


def generate(rng, basis, pair, hidden, jitter):
    # Null directions make both signed indexators represent the same function.
    # Regressions may use different numeric codes in the same structure.
    w0 = basis[pair[0]] @ rng.normal(size=(M, hidden))
    w1 = basis[pair[1]] @ rng.normal(size=(M, hidden)) * 0.75
    q = np.linalg.svd(w0 - w1, full_matrices=False)
    null = q[0][:, q[1] > 1e-10]
    projected = np.eye(D) - null @ null.T
    scales = np.linalg.norm(projected @ w0, axis=0).clip(1e-8)
    w0 = w0 / scales
    transform = projected + jitter * (np.eye(D) - projected)
    return transform, w0


def observations(rng, transform, w, n, link):
    x = rng.normal(size=(n, D)) @ transform
    y = x @ w
    if link == "relu":
        y = np.maximum(y, 0) * np.exp(0.05 * rng.normal(size=y.shape) - 0.05**2 / 2)
    else:
        y += 0.05 * rng.normal(size=y.shape)
    return x, y


def least_squares(basis, gram, rhs):
    g = basis.transpose(0, 2, 1) @ gram @ basis
    b = basis.transpose(0, 2, 1) @ rhs
    return (basis @ (np.linalg.pinv(g, rcond=1e-10, hermitian=True) @ b))[..., 0]


def fit_bank(basis, x, y, link, rng):
    """Per-task numeric refit. Positive ReLU responses provide linear observations."""
    models, dense = [], []
    for target in y.T:
        active = target > 0 if link == "relu" else np.ones(len(x), dtype=bool)
        xx, yy = x[active], target[active]
        models.append(least_squares(basis, xx.T @ xx, xx.T @ yy[:, None]))
        inverse = np.linalg.pinv(xx, rcond=1e-10)
        # Exactly the LS gradient-flow limits from six independent initial weights.
        null = np.eye(D) - inverse @ xx
        dense.append(inverse @ yy + rng.normal(0, 0.6, size=(6, D)) @ null)
    return np.stack(models, axis=-1), np.stack(dense, axis=-1)


def losses(w, x, y, link):
    pred = x @ w
    if link == "relu":
        pred = np.maximum(pred, 0)
    return np.minimum((pred - y) ** 2 / 4, 1).mean(axis=(1, 2))


def exact_population_risk(transform, w, basis):
    gram = transform.T @ transform
    fit = np.stack(
        [least_squares(basis, gram, gram @ v[:, None]) for v in w.T], axis=-1
    )
    return np.square(transform @ (fit - w)).sum(axis=1).mean(axis=1)


def representations(dense, basis):
    norm = dense / np.maximum(np.linalg.norm(dense, axis=2, keepdims=True), 1e-12)
    z = norm.mean(axis=1)
    hidden = z.shape[-1]
    ward = signed_forest(z, (M * hidden,))[0][M * hidden]
    # A matched-bank control uses precisely the same candidate pairings.
    projected = basis @ (basis.transpose(0, 2, 1) / 2)
    error = ((np.eye(D) - projected)[:, None] @ z[None]) ** 2
    return ward, int(error.mean(axis=(1, 2, 3)).argmin())


def target_mse(rng, truth, structures, link, episodes=48):
    """Fresh full-rank tasks with exactly the chosen transferable constraints."""
    true_design = design(truth)
    hidden = 2 if link == "relu" else 1
    trials = []
    for _ in range(episodes):
        w = true_design @ rng.normal(size=(true_design.shape[1], hidden))
        w /= np.linalg.norm(w, axis=0).clip(1e-8)
        x, y = observations(rng, np.eye(D), w, 16, link)
        qx = rng.normal(size=(256, D))
        qy = qx @ w
        if link == "relu":
            qy = np.maximum(qy, 0)
        row = []
        for b in structures:
            cols = []
            for target in y.T:
                active = target > 0 if link == "relu" else np.ones(len(y), dtype=bool)
                u = x[active] @ b
                # The same weak ridge controls rare small positive-response samples.
                beta = np.linalg.solve(
                    u.T @ u + 0.05 * np.eye(u.shape[1]), u.T @ target[active]
                )
                cols.append(b @ beta)
            pred = qx @ np.stack(cols, axis=1)
            if link == "relu":
                pred = np.maximum(pred, 0)
            row.append(float(np.mean((pred - qy) ** 2)))
        trials.append(row)
    return np.mean(trials, axis=0).tolist()


def run_one(bank, basis, seed, case, link, jitter):
    rng = np.random.default_rng(seed)
    pairs, planted = alternatives(bank, rng, case)
    targets, dense, exact = [], [], []
    hidden = 2 if link == "relu" else 1
    for pair in pairs:
        transform, w = generate(rng, basis, pair, hidden, jitter)
        x, y = observations(rng, transform, w, FIT, link)
        vx, vy = observations(rng, transform, w, HOLDOUT, link)
        fits, dense_fits = fit_bank(basis, x, y, link, rng)
        targets.append(losses(fits, vx, vy, link))
        dense.append(dense_fits)
        exact.append(exact_population_risk(transform, w, basis))
    risk = np.stack(targets)
    found = bank.infer(risk, TOLERANCE)
    oracle = bank.infer(np.array(exact), 1e-10)
    family = set(planted["maximal"])
    inferred = set(found["maximal"])
    ward, match = representations(np.stack(dense), basis)
    winner = risk.argmin(axis=1)
    vote = int(np.bincount(winner, minlength=len(basis)).argmax())
    erm = int(risk.mean(axis=0).argmin())
    marginal = set()
    for atom in set.union(*(set(s) for s in bank.full)):
        contains = np.array([atom in s for s in bank.full])
        if np.all(np.min(risk[:, contains], axis=1) <= risk.min(axis=1) + TOLERANCE):
            marginal.add(atom)
    fixed_rows = [
        bank.full[found["fixed"]],
        bank.full[erm],
        bank.full[vote],
        bank.full[match],
    ]
    out = {
        "family_exact": inferred == family,
        "oracle_family_is_planted": set(oracle["maximal"]) == family,
        "family_matches_exact_population": inferred == set(oracle["maximal"]),
        "necessary_exact": found["necessary"] == planted["necessary"],
        "maximal_count": len(inferred),
        "necessary_count": len(found["necessary"]),
        "mean_witnesses": float(np.mean(found["solutions_per_task"])),
        "mean_population_witnesses": float(np.mean(oracle["solutions_per_task"])),
        "full_common_count": int(np.sum(found["full_coverage"] == 1)),
        "marginal_union_infeasible": not any(
            marginal <= candidate for candidate in family
        ),
        "ward_is_maximal": extracted(ward) in family,
        "fixed_is_maximal": [row in family for row in fixed_rows],
        "fixed_coverage": float(found["full_coverage"][found["fixed"]]),
        "sample_size_sensitivity": {
            str(t): set(bank.infer(risk[:t], TOLERANCE)["maximal"]) == family
            for t in (2, 4, 8)
        },
        "tolerance_sensitivity": {
            str(e): set(bank.infer(risk, e)["maximal"]) == family
            for e in (1e-6, 0.0001, 0.001, 0.01)
        },
    }
    if len(family) == 1:
        gold = next(iter(family))
        committed = min(found["maximal"], key=lambda s: (-len(s), tuple(sorted(s))))
        other = [p for p in bank.partial if len(p) == len(committed) and p != committed]
        random = other[rng.integers(len(other))] if other else committed
        out["target_mse"] = target_mse(
            rng,
            gold,
            [
                design(row)
                for row in (committed, random, gold, frozenset(), bank.full[erm])
            ],
            link,
        )
        out["partial_codes_per_neuron"] = D - len(committed)
    out["example"] = {
        "pairs": pairs.tolist(),
        "inferred": [sorted(s) for s in found["maximal"]],
        "planted": [sorted(s) for s in planted["maximal"]],
        "necessary": sorted(found["necessary"]),
        "constraint_only_index": [
            as_index(s, hidden).tolist() for s in found["maximal"]
        ],
    }
    return out, risk


def simulation(link):
    folder = OUTPUT / "synthetic"
    folder.mkdir(parents=True, exist_ok=True)
    bank = RelationBank(matchings(list(range(D))))
    basis = np.stack([design(s) for s in bank.full])
    plan = {
        "family_seeds": list(range(SEED, SEED + REPLICATES)),
        "tasks": TASKS,
        "candidate_complete": len(basis),
        "candidate_partial": len(bank.partial),
        "fit": FIT,
        "holdout": HOLDOUT,
        "primary_tolerance": TOLERANCE,
        "coverage": 1,
        "link": link,
        "cases": CASES,
        "jitter": [0, 0.025],
        "noise_scale": 0.05,
        "target_mse_columns": [
            "inferred_partial",
            "equal_budget_random",
            "oracle_partial",
            "dense",
            "mean_risk_full",
        ],
        "fixed_columns": [
            "joint_fixed",
            "mean_risk",
            "single_winner_vote",
            "dense_matching",
        ],
        "loss": "mean min(squared residual / 4, 1)",
    }
    save_json(folder / f"{link}_plan.json", plan)
    example, raw, scores = {}, {}, {}
    for jitter in plan["jitter"]:
        for case in CASES:
            group = f"{case}_j{jitter}"
            records, risks = [], []
            for k in range(REPLICATES):
                row, risk = run_one(bank, basis, SEED + k, case, link, jitter)
                detail = row.pop("example")
                if k == 0:
                    example[group] = detail
                records.append(row)
                risks.append(risk)
            raw[group] = records
            scores[group] = np.stack(risks)
            print(link, group, sum(r["family_exact"] for r in records), flush=True)
    save_json(folder / f"{link}_examples.json", example)
    save_json(folder / f"{link}_results.json", raw)
    np.savez_compressed(folder / f"{link}_risks.npz", **scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("link", choices=("linear", "relu"))
    simulation(parser.parse_args().link)
