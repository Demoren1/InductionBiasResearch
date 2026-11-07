"""Independent source-data moment proposal, not a bank or NF extractor.

For Gaussian inputs, the second Hermite moment exposes block interactions.
This is a diagnostic alternative proposer for the synthetic family only.
"""

import json
import shutil
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering


def propose(x, y, seed):
    design = np.c_[np.ones(len(x)), x]
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    moments = np.array(
        [x.T @ (x * residual[:, task, None]) / len(x) for task in range(y.shape[1])]
    )
    affinity = np.sqrt(np.square(moments).mean(0))
    np.fill_diagonal(affinity, 0)
    labels = SpectralClustering(
        n_clusters=8,
        affinity="precomputed",
        assign_labels="cluster_qr",
        random_state=seed,
    ).fit_predict(affinity)
    if not np.array_equal(np.bincount(labels, minlength=8), np.full(8, 8)):
        raise ValueError("Moment proposal did not identify eight equal groups")
    groups = np.array([np.flatnonzero(labels == group) for group in range(8)])
    blocks = np.stack([moments[:, group[:, None], group[None, :]] for group in groups])
    signatures = np.concatenate(
        [
            np.sort(blocks, axis=-1).transpose(0, 2, 1, 3).reshape(8, 8, -1),
            np.diagonal(blocks, axis1=-2, axis2=-1).transpose(0, 2, 1),
        ],
        axis=-1,
    )
    best = None
    for reference in range(8):
        orders = []
        for group in range(8):
            cost = np.square(
                signatures[group, :, None] - signatures[reference, None, :]
            ).sum(-1)
            old, new = linear_sum_assignment(cost)
            order = np.empty(8, dtype=int)
            order[new] = old
            orders.append(order)
        for _ in range(30):
            aligned = np.stack(
                [block[:, order][:, :, order] for block, order in zip(blocks, orders)]
            )
            prototype = aligned.mean(0)
            updated = []
            for block, order in zip(blocks, orders):
                cost = np.square(
                    block[:, :, order].transpose(1, 0, 2)[:, None]
                    - prototype.transpose(1, 0, 2)[None]
                ).sum((2, 3))
                old, new = linear_sum_assignment(cost)
                order = np.empty(8, dtype=int)
                order[new] = old
                updated.append(order)
            orders = updated
        aligned = np.stack(
            [block[:, order][:, :, order] for block, order in zip(blocks, orders)]
        )
        error = float(np.square(aligned - aligned.mean(0)).sum())
        if best is None or error < best[0]:
            best = error, np.array(orders)
    permutation = np.stack(
        [group[order] for group, order in zip(groups, best[1])]
    ).flatten()
    return {
        "permutation": permutation,
        "groups": groups,
        "moments": moments,
        "affinity": affinity,
        "alignment_error": np.array(best[0]),
    }


def main():
    out = Path("experiments/structure_search_v2/runs/scrambled_pattern_source_pilot_v1")
    shutil.copy2(__file__, out / "moment_proposal_source.py")
    spec = json.loads((out / "protocol.json").read_text())
    data = np.load(out / "source_data.npz")
    result = propose(
        data["x"][: spec["support"]].astype(float),
        data["y"][:, : spec["support"]].T.astype(float),
        spec["seed"],
    )
    np.savez_compressed(out / "moment_proposal.npz", **result)
    (out / "moment_proposal_manifest.json").write_text(
        json.dumps(
            {
                "source_support_only": True,
                "uses_teacher_indices": False,
                "method": "source-data moment control; not NF+VAE and not bank extraction",
                "group_sizes": [8] * 8,
                "alignment_error": float(result["alignment_error"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
