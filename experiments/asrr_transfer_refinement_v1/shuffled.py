"""Test the contribution of group directions in a frozen continuous basis."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.paper_transfer_v2.common import CONFIG
from experiments.paper_transfer_v2.report import confidence, grouped
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .allocation import input_groups, spectrum, unequal_model
from .bases import factor_model
from .common import OUT, SEED
from .contrasts import scatter_matrices
from .fusion import align_blocks, chosen_factor, fuse, guides
from .study import evaluate


def fusion_controls(root, dataset, layout, policy, rng):
    name = policy["candidate"]
    bank = json.loads((root / "fusion" / "bank.json").read_text())
    row = bank[name]
    weight = float(name.rsplit("_", 2)[1])
    groups = chosen_factor(root / "known", row["group_source"])
    first, second, _ = guides(root, dataset)
    for _ in range(4):
        shuffled = np.stack(
            [rng.permutation(group, axis=0) for group in groups]
        )
        yield factor_model(
            [
                fuse(*site, weight, row["fixed_directions"])
                for site in align_blocks(shuffled, first, second)
            ],
            layout,
        )


def allocation_controls(root, dataset, layout, policy, rng, device):
    name = policy["candidate"]
    bank = json.loads((root / "allocation" / "bank.json").read_text())
    row = bank[name]
    fixed = row["fixed_groups"]
    ridge = float(name.rsplit("_", 1)[1])
    rank = CONFIG["real"][dataset]["codes"] // layout.hidden
    groups = input_groups(dataset, device, layout, rank)
    signal, within = scatter_matrices(dataset, layout, device)
    for _ in range(4):
        bases = []
        ranks = row["rank_by_site"]
        for a, b, labels, size in zip(
            signal, within, groups, ranks, strict=True
        ):
            _, anchors, directions = spectrum(
                a, b, rng.permutation(labels), fixed, ridge
            )
            basis = np.column_stack((anchors, directions[:, : size - fixed]))
            bases.append(np.linalg.qr(basis)[0])
        yield unequal_model(bases, layout)


def run(dataset, replicate, device):
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    root = OUT / dataset
    folder = root / (
        f"confirmation_{replicate}" if replicate else "confirmation"
    )
    policy = json.loads((folder / "choices.json").read_text())["policies"][
        "refined"
    ]
    data = read_data(folder / "per_episode.npz")
    layout = Layout(*CONFIG["real"][dataset]["layout"])
    seed = SEED + 4511
    rng = np.random.default_rng(seed)
    if policy["study"] == "fusion":
        controls = fusion_controls(root, dataset, layout, policy, rng)
    elif policy["study"] == "allocation":
        controls = allocation_controls(
            root, dataset, layout, policy, rng, device
        )
    else:
        raise ValueError(
            "The selected structure has no matching shuffled control"
        )

    destination = folder / "shuffled"
    destination.mkdir(exist_ok=True)
    paths = {}
    for i, control in enumerate(controls):
        name = f"shuffled_{i}"
        paths[name] = Path(destination / f"{name}.npz")
        atomic_npz(paths[name], **control)
    values = evaluate(
        dataset,
        "test",
        paths,
        device,
        rates=(policy["rate"],),
        steps=(policy["steps"],),
        replicates=2,
        folder=folder / "test",
    )
    accuracy = np.stack([value["query"][:, 0] for value in values.values()])
    difference = data["refined_query"] - accuracy.mean(0)
    results = {
        "shuffle_seed": seed,
        "accuracy": confidence(grouped(accuracy.mean(0), data["units"])),
        "gain": confidence(grouped(difference, data["units"])),
    }
    save_json(destination / "results.json", results)
    atomic_npz(
        destination / "per_episode.npz", accuracy=accuracy, gain=difference
    )
    print(dataset, results, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("fc100", "mnist_torus", "wisdm"))
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.dataset, args.replicate, torch.device(args.device))
