"""Population risks of the actual fitted synthetic candidates and a signed-map audit."""

import numpy as np
from scipy.special import ndtr

from experiments.sharing_structure_research_v1.blind_contract import save_json
from experiments.srr_approx_v1.compatible import OUTPUT, RelationBank, common_frontier
from experiments.srr_approx_v1.compatible_synthetic import (
    CASES,
    D,
    FIT,
    HOLDOUT,
    REPLICATES,
    SEED,
    TOLERANCE,
    alternatives,
    as_index,
    design,
    extracted,
    fit_bank,
    generate,
    matchings,
    observations,
)


def normal_risk(sigma):
    limit = 2 / sigma
    tail = 2 * ndtr(-limit)
    density = np.exp(-(limit**2) / 2) / np.sqrt(2 * np.pi)
    return (sigma**2 / 4) * (1 - tail - 2 * limit * density) + tail


def population():
    folder = OUTPUT / "synthetic"
    bank = RelationBank(matchings(list(range(D))))
    basis = np.stack([design(s) for s in bank.full])
    summary = {}
    with np.load(folder / "linear_risks.npz") as saved:
        for jitter in (0, 0.025):
            for case in CASES:
                key = f"{case}_j{jitter}"
                recorded = saved[key]
                rows = []
                for k in range(REPLICATES):
                    rng = np.random.default_rng(SEED + k)
                    pairs, planted = alternatives(bank, rng, case)
                    true, observed = [], []
                    for pair in pairs:
                        a, w = generate(rng, basis, pair, 1, jitter)
                        x, y = observations(rng, a, w, FIT, "linear")
                        vx, vy = observations(rng, a, w, HOLDOUT, "linear")
                        fitted, _ = fit_bank(basis, x, y, "linear", rng)
                        sd = np.sqrt(np.sum((a @ (fitted - w)) ** 2, axis=1) + 0.05**2)
                        true.append(normal_risk(sd).ravel())
                        observed.append(
                            np.minimum((vx @ fitted - vy) ** 2 / 4, 1).mean(axis=(1, 2))
                        )
                    if not np.allclose(recorded[k], observed, atol=1e-12, rtol=1e-12):
                        raise ValueError(
                            "Population audit must address the same fitted predictors"
                        )
                    oracle = bank.infer(np.stack(true), TOLERANCE)
                    sample = bank.infer(recorded[k], TOLERANCE)
                    rows.append(
                        {
                            "same_maximal": set(oracle["maximal"])
                            == set(sample["maximal"]),
                            "same_necessary": oracle["necessary"]
                            == sample["necessary"],
                            "population_equals_planted": set(oracle["maximal"])
                            == set(planted["maximal"]),
                            "population_task_solutions": float(
                                oracle["solutions_per_task"].mean()
                            ),
                        }
                    )
                summary[key] = rows
    save_json(folder / "linear_population_audit.json", summary)
    # Compare both representations on real version sets including incompatible ties.
    signed = {}
    for link in ("linear", "relu"):
        hidden = 1 if link == "linear" else 2
        complete = [as_index(s, hidden) for s in bank.full]
        with np.load(folder / f"{link}_risks.npz") as risk:
            for case in CASES:
                r = risk[f"{case}_j0"][0]
                good = r <= r.min(axis=1, keepdims=True) + TOLERANCE
                maps = common_frontier(
                    [[i for i, keep in zip(complete, row) if keep] for row in good]
                )
                reference = bank.infer(r, TOLERANCE)["maximal"]
                signed[f"{link}_{case}"] = {
                    "complete_frontier_equal": {extracted(c) for c in maps}
                    == set(reference),
                    "maximal_maps": [c.tolist() for c in maps],
                }
    save_json(folder / "signed_maps.json", signed)
    print(
        {k: sum(row["same_maximal"] for row in rs) for k, rs in summary.items()},
        flush=True,
    )


if __name__ == "__main__":
    population()
