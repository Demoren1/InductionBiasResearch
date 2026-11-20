"""Measure the portion of each competitor basis represented by Ward groups."""

import json
from pathlib import Path

import numpy as np

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.paper_transfer_v2.common import CONFIG, OUT as PAPER
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .common import DATASETS, OUT


def model(dataset, name):
    choice = json.loads((PAPER / dataset / f"{name}_choice.json").read_text())
    return read_data(Path(choice["model"]))


def overlap(subspace, index, counts):
    sign = np.sign(index)
    ids = np.abs(index)
    projected = np.zeros((len(counts), subspace.shape[-1]))
    scale = sign / np.sqrt(np.maximum(counts[ids], 1))
    np.add.at(projected, ids, scale[:, None] * subspace)
    return float(np.square(projected).sum())


def diagnostic(dataset):
    layout = Layout(*CONFIG["real"][dataset]["layout"])
    index = model(dataset, "asrr")["index"].reshape(
        layout.sites, layout.fan_in, layout.channels
    )
    counts = np.bincount(np.abs(index).ravel())
    yeh = model(dataset, "yeh_soft")["basis"]
    yeh_ids = index.transpose(0, 2, 1).reshape(len(yeh), yeh.shape[1])
    yeh_trace = sum(
        overlap(basis, ids, counts)
        for basis, ids in zip(yeh, yeh_ids, strict=True)
    )
    factor = model(dataset, "msr")["factor"]
    sites = min(layout.sites, len(factor))
    blocks = len(factor) // sites
    reshaped = factor.reshape(sites, blocks, layout.fan_in, -1)
    factors = np.broadcast_to(reshaped, (layout.sites, *reshaped.shape[1:]))
    per_neuron = np.repeat(factors, layout.channels // blocks, axis=1)
    msr_trace = sum(
        overlap(basis, ids, counts)
        for basis, ids in zip(
            per_neuron.reshape(layout.hidden, layout.fan_in, -1),
            index.transpose(0, 2, 1).reshape(layout.hidden, layout.fan_in),
            strict=True,
        )
    )
    return {
        "yeh_soft_fraction": yeh_trace / (len(yeh) * yeh.shape[-1]),
        "msr_fraction": msr_trace / (layout.hidden * factor.shape[-1]),
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    scores = {dataset: diagnostic(dataset) for dataset in DATASETS}
    save_json(OUT / "overlap.json", scores)
    print(scores)
