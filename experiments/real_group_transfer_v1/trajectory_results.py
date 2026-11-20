"""Report paired estimates and source geometry for the fixed trajectory study."""

import argparse
import json

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .compatible_study import interval
from .trajectory_data import CHANNELS, LOCAL, ROOT


def summarize(dataset):
    folder = ROOT / dataset
    frozen = json.loads((folder / "frozen.json").read_text())
    channels = CHANNELS[dataset]
    with np.load(LOCAL / dataset / "records.npz") as f:
        x = f["x"][f["source_fit"].ravel()]
    correlation = np.corrcoef(x, rowvar=False)
    with np.load(folder / "frozen_indices.npz") as bank:

        def adjacent(name):
            index = bank[frozen["aliases"][name]]
            return float((index[channels:] == index[:-channels]).mean())

        permuted = [adjacent(f"permutation_{j}") for j in range(4)]
        geometry = {
            "adjacent_source_correlation": float(
                np.diag(correlation, k=channels).mean()
            ),
            "all_pairs_source_correlation": float(
                correlation[np.triu_indices_from(correlation, k=1)].mean()
            ),
            "adjacent_tie_asrr": adjacent("asrr"),
            "adjacent_tie_permutations": permuted,
        }
    outcomes = {}
    for shots in (2, 5):
        phase = f"target_{shots}"
        with np.load(folder / f"{phase}_scores.npz") as f:
            units = f["units"]
            ids = np.unique(units)
            rows = {}
            for name, key in frozen["aliases"].items():
                (step,) = frozen["methods"][key]["steps"]
                values = f[f"{key}__{step}"]
                rows[name] = np.stack([values[units == u].mean(0) for u in ids])
        for family in ("permutation", "mask"):
            rows[family] = np.mean([rows[f"{family}_{i}"] for i in range(4)], axis=0)
        primary = rows["asrr"][:, 0] - rows["dense"][:, 0]
        contrasts = {
            k: interval(rows["asrr"][:, 0] - v[:, 0], 0.95) for k, v in rows.items()
        }
        outcomes[str(shots)] = {
            "accuracy_brier_ce": {k: v.mean(0).tolist() for k, v in rows.items()},
            "asrr_minus_dense": interval(primary, 0.975 if shots == 2 else 0.95),
            "asrr_minus_each_ci95": contrasts,
            "strict_minus_erm_ci95": interval(
                rows["strict"][:, 0] - rows["erm"][:, 0], 0.95
            ),
            "strict_minus_dense_ci95": interval(
                rows["strict"][:, 0] - rows["dense"][:, 0], 0.95
            ),
            "units": len(ids),
            "episodes": len(units),
            "units_with_asrr_gain": int((primary > 0).sum()),
            "unit_ids": ids.tolist(),
            "asrr_minus_dense_by_unit": primary.tolist(),
            "scores_sha256": digest(folder / f"{phase}_scores.npz"),
        }
    save_json(
        folder / "results.json",
        {
            "frozen_sha256": digest(folder / "frozen.json"),
            "geometry": geometry,
            "shots": outcomes,
        },
    )
    print(
        dataset,
        [(s, row["asrr_minus_dense"]) for s, row in outcomes.items()],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("pen", "spoken"))
    summarize(parser.parse_args().dataset)
