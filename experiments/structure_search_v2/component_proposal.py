"""Use source feature dependence and source-network utility to propose a pool.

The covariance graph groups dependent inputs; ablations of partial solutions
identify which group is useful. Ground-truth block IDs are never read here.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering

ROOT = Path("experiments/structure_search_v2/runs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], required=True)
    args = parser.parse_args()
    out = ROOT / ("distractor_" + args.dataset + "_v4")
    shutil.copy2(__file__, out / "component_source.py")
    source, bank = (
        np.load(out / "source_observations.npz"),
        np.load(out / "good_bank.npz"),
    )
    x = np.concatenate([source["x"], source["query_x"]], 1).astype(np.float64)
    x -= x.mean(1, keepdims=True)
    matrix = x.reshape(-1, x.shape[-1])
    variance = np.square(matrix).mean(0)
    active = np.flatnonzero(variance > 1e-6)
    standardized = matrix[:, active] / np.sqrt(variance[active])[None]
    correlation = standardized.T @ standardized / len(matrix)
    affinity = np.square(correlation)
    affinity = np.maximum(0, affinity - 1 / (len(matrix) - len(x)))
    np.fill_diagonal(affinity, 0)
    labels = SpectralClustering(
        n_clusters=8,
        affinity="precomputed",
        assign_labels="cluster_qr",
        random_state=281510,
    ).fit_predict(affinity)
    confidence = np.maximum(0, np.log(2) - bank["quality"])
    confidence[bank["role"] != 0] = 0
    for task in np.unique(bank["task"]):
        ids = (bank["task"] == task) & (bank["role"] == 0)
        confidence[ids] /= confidence[ids].sum()
    relevance = (bank["tokens"][:, :, 2] * confidence[:, None]).sum(0)
    scores = [
        relevance[active[labels == group]].sum() / np.sqrt((labels == group).sum())
        for group in range(8)
    ]
    selected_group = int(np.argmax(scores))
    pool = active[labels == selected_group]
    if len(pool) < 64:
        raise ValueError(
            "The inferred useful component is too small for the mask family"
        )
    np.savez_compressed(
        out / "component_proposal.npz",
        pool=pool,
        active_coordinates=active,
        groups=labels,
        component_scores=np.array(scores),
        relevance=relevance,
        variance=variance,
    )
    (out / "component_complete.json").write_text(
        json.dumps(
            {
                "input_pool_size": len(pool),
                "scores": [float(v) for v in scores],
                "selection": "source-train-lineage network ablation utility per source-derived covariance component",
                "source_only": True,
                "ground_truth_layout_used": False,
            },
            indent=2,
        )
    )
    print(args.dataset, "COMPONENT", len(pool), "scores", scores, flush=True)


if __name__ == "__main__":
    main()
