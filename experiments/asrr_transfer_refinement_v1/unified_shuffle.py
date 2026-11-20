"""Permute only the group map of the selected partial-sharing models."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_tuning_v1.data import episodes
from experiments.paper_transfer_v2.common import CONFIG
from experiments.paper_transfer_v2.linear import signed_design
from experiments.paper_transfer_v2.report import confidence, grouped
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.sharing_structure_research_v1.blind_contract import save_json
from experiments.srr_approx_v1.synthetic_random_controls import shuffle_groups

from .bases import factor_model
from .common import OUT, SEED
from .fusion import align_blocks, fuse
from .study import evaluate
from .unified import source_signals, span
from .unified_global import HAR_LAYOUT, har_guides, har_test


def real_controls(dataset, folder, policy, destination, device, rng):
    config = CONFIG["real"][dataset]
    layout = Layout(*config["layout"])
    bank = json.loads((folder.parent / "unified" / "bank.json").read_text())
    model = bank[policy["candidate"]]
    family = model["family"].removeprefix("projected_")
    key = model["group_source"]
    groups = span(read_data(Path(bank[key]["path"]))["factor"])
    rank = config["codes"] // layout.hidden
    _, _, _, guides = source_signals(dataset, layout, rank, device)
    first, second, alpha = guides[family]
    paths = {}
    for i in range(4):
        permuted = [rng.permutation(group, axis=0) for group in groups]
        basis = [
            fuse(*site, alpha, 1)
            for site in align_blocks(permuted, first, second)
        ]
        key = f"shuffle_{i}"
        paths[key] = destination / f"{key}.npz"
        atomic_npz(paths[key], **factor_model(basis, layout))
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
    return np.stack([row["query"][:, 0] for row in values.values()])


def har_controls(folder, settings, destination, device, rng):
    policy = settings["policies"]["refined"]
    model = folder / "work" / "banks" / "har" / "models" / "asrr.npz"
    index = read_data(model)["index"]
    guide = dict(har_guides())[policy["name"].removeprefix("projected_")]
    observations, _ = episodes("har3", "test", 360, settings["panel_seed"])
    inputs = [observations[key] for key in ("train_x", "train_y", "query_x")]
    predictions = []
    for i in range(4):
        permuted = shuffle_groups(index, rng)
        group = span(signed_design(permuted.ravel()))
        basis = fuse(group, guide, guide, 0.0, 1)
        basis = basis.reshape(HAR_LAYOUT.fan_in, HAR_LAYOUT.hidden, -1)
        path = destination / f"shuffle_{i}.npz"
        atomic_npz(path, basis=basis.astype(np.float32))
        predictions.append(har_test(inputs, path, policy["rule"], device))
    truth = observations["query_y"].argmax(-1)
    return (np.stack(predictions) == truth).mean(-1)


def run(dataset, device):
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    is_har = dataset == "har"
    folder = OUT / dataset / ("unified" if is_har else "confirmation_2")
    settings = json.loads((folder / "choices.json").read_text())
    destination = folder / "shuffled"
    destination.mkdir(exist_ok=True)
    seed = SEED + 25120
    rng = np.random.default_rng(seed)
    if is_har:
        values = har_controls(folder, settings, destination, device, rng)
    else:
        values = real_controls(
            dataset,
            folder,
            settings["policies"]["refined"],
            destination,
            device,
            rng,
        )
    data = read_data(folder / "per_episode.npz")
    acc = data["refined" if is_har else "refined_query"]
    delta = acc - values.mean(0)
    report = {
        "seed": seed,
        "accuracy": confidence(grouped(values.mean(0), data["units"])),
        "gain": confidence(grouped(delta, data["units"])),
    }
    save_json(destination / "results.json", report)
    atomic_npz(destination / "per_episode.npz", accuracy=values, gain=delta)
    print(dataset, report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", choices=("mnist_torus", "fc100", "wisdm", "har")
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.dataset, torch.device(args.device))
