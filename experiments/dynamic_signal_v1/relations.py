"""Propose indexators from permutation-invariant relations in source weights.

Known budgets: 16 input groups, eight shared coordinate roles, two filters/group.
Neither teacher permutations nor observations are read by this extractor.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, SpectralClustering

from experiments.dynamic_signal_v1.fit import fit


def propose(raw, seed, count=16):
    task_ids = raw["task_ids"]
    training = task_ids < 8
    weights = raw["parameter_0"][training]
    readout = raw["parameter_2"][training]
    masks = (weights != 0).astype(float)
    # Sum of input-to-output paths is unchanged by hidden column permutation
    # or function-preserving positive rescaling of a ReLU neuron.
    paths = (weights * readout[:, None]).sum(-1).reshape(8, 4, 128)
    rng = np.random.default_rng(seed)
    proposals = []
    reference_groups = reference_profiles = None
    for sample in range(count):
        selected = (
            np.arange(len(weights))
            if sample == 0
            else rng.integers(len(weights), size=len(weights))
        )
        affinity = np.einsum("nij,nkj->ik", masks[selected], masks[selected])
        np.fill_diagonal(affinity, 0)
        groups = SpectralClustering(
            16, affinity="precomputed", random_state=seed + sample, n_init=20
        ).fit_predict(affinity)
        starts = (
            np.tile(np.arange(4), (8, 1))
            if sample == 0
            else rng.integers(4, size=(8, 4))
        )
        profiles = np.stack(
            [paths[task, starts[task]].mean(0) for task in range(8)], -1
        )
        profiles /= np.maximum(np.linalg.norm(profiles, axis=-1, keepdims=True), 1e-9)
        roles = KMeans(8, n_init=20, random_state=seed + sample).fit_predict(profiles)
        centers = np.stack([profiles[roles == role].mean(0) for role in range(8)])
        if reference_groups is None:
            order = sorted(
                range(16), key=lambda group: np.flatnonzero(groups == group)[0]
            )
            mapping = np.empty(16, int)
            mapping[order] = np.arange(16)
            groups = mapping[groups]
            role_order = sorted(
                range(8), key=lambda role: np.flatnonzero(roles == role)[0]
            )
            role_mapping = np.empty(8, int)
            role_mapping[role_order] = np.arange(8)
            roles = role_mapping[roles]
            reference_profiles = centers[role_order]
            reference_groups = groups.copy()
        else:
            old, new = linear_sum_assignment(
                -np.array(
                    [
                        [
                            (groups == a).__and__(reference_groups == b).sum()
                            for b in range(16)
                        ]
                        for a in range(16)
                    ]
                )
            )
            mapping = np.empty(16, int)
            mapping[old] = new
            groups = mapping[groups]
            old, new = linear_sum_assignment(
                ((centers[:, None] - reference_profiles[None]) ** 2).sum(-1)
            )
            mapping = np.empty(8, int)
            mapping[old] = new
            roles = mapping[roles]
        indices = np.zeros((128, 32), dtype=np.int64)
        indices[np.arange(128), 2 * groups] = roles + 1
        indices[np.arange(128), 2 * groups + 1] = roles + 9
        proposals.append(indices)
    unique = np.unique(np.stack(proposals), axis=0)
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-id", type=int, required=True)
    parser.add_argument(
        "--protocol", default="experiments/dynamic_signal_v1/protocol.json"
    )
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    root = Path(plan["run"])
    source = (
        root
        / ("banks_large_tuned" if args.noise_id < 2 else "banks")
        / f"noise_{args.noise_id}"
    )
    output = root / "banks_relations" / f"noise_{args.noise_id}"
    output.mkdir(parents=True, exist_ok=False)
    proposals = propose(np.load(source / "round_256.npz"), plan["seed"])
    np.savez_compressed(output / "proposals.npz", indices=proposals)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    data = dict(np.load(root / "source" / f"noise_{args.noise_id}.npz"))
    configs = [
        {
            "optimizer": "adam",
            "lr": rate,
            "weight_decay": 0.0,
            "momentum": 0.0,
            "initial_scale": 1.0,
        }
        for rate in plan["grids"]["codebook_lr"]
    ]
    tasks, candidates = len(data["support_y"]), len(proposals)
    indices = torch.tensor(
        np.tile(proposals, (len(configs) * tasks, 1, 1)), device=device
    )
    arrays = fit(
        plan,
        data,
        "relation_source",
        configs,
        16384,
        candidates,
        device,
        output / "codebook_refit",
        indices=indices,
    )
    # Numerical HPO per proposal; retain two good proposals per source task.
    loss = arrays["query_loss"].reshape(len(configs), tasks, candidates)
    best_config = loss.argmin(0)
    chosen = (
        best_config * tasks * candidates
        + np.arange(tasks)[:, None] * candidates
        + np.arange(candidates)[None]
    )
    losses = arrays["query_loss"][chosen]
    kept, task_ids = [], []
    for task in range(tasks):
        acceptable = np.flatnonzero(losses[task] < 0.65)
        good = acceptable[np.argsort(losses[task, acceptable])[:2]]
        kept.extend(chosen[task, good])
        task_ids.extend([task] * len(good))
    kept = np.array(kept, int)
    task_ids = np.array(task_ids, int)
    role = np.where(task_ids < 8, 0, np.where(task_ids < 10, 1, 2))
    np.savez_compressed(
        output / "bank.npz",
        indices=arrays["indices"][kept],
        task_ids=task_ids,
        role=role,
        query_loss=arrays["query_loss"][kept],
        query_accuracy=arrays["query_accuracy"][kept],
    )
    # Consensus is chosen solely by aggregate first-eight-source query BCE.
    consensus_id = int(losses[:8].mean(0).argmin())
    np.savez_compressed(output / "extracted.npz", consensus=proposals[consensus_id])
    (output / "complete.json").write_text(
        json.dumps(
            {
                "source_bank": str(source),
                "candidate_count": candidates,
                "retained": len(kept),
                "retained_query_accuracy": float(arrays["query_accuracy"][kept].mean()),
                "consensus_id": consensus_id,
                "consensus_source_query_accuracy": float(
                    arrays["query_accuracy"][chosen[:8, consensus_id]].mean()
                ),
                "source_inputs": "Sparse first-layer weights and readout weights only, first eight tasks. No teacher or observations used for clustering.",
                "given_budgets": {
                    "input_groups": 16,
                    "roles": 8,
                    "filters_per_group": 2,
                },
                "motivation": "Column-by-column alignment loses cross-network relations. Input co-occurrence affinity and summed input-output paths are invariant to hidden-column permutation and positive ReLU rescaling.",
            },
            indent=2,
        )
        + "\n"
    )
    print("BANK", (output / "complete.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
