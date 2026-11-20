"""Test source fitting from the same first layer with and without matching."""

import json

import numpy as np

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.paper_transfer_v2.common import CONFIG
from experiments.paper_transfer_v2.source import dense_source, source_tensors
from experiments.real_group_transfer_v1.optimization import Layout, fit_dense
from experiments.real_group_transfer_v1.study import source_profiles
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .bank import block_profiles
from .bases import partition_model, ward_profiles
from .common import OUT, SEED

STEPS = {"fc100": 800, "wisdm": 450, "mnist_torus": 500, "aug_omniglot": 500}


def build_anchor(dataset, device):
    root = OUT / dataset / "anchor"
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "bank.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    settings = CONFIG["real"][dataset]
    layout = Layout(*settings["layout"])
    file = root / "source_fits.npz"
    if file.exists():
        archive = read_data(file)
        weights, errors = archive["weights"], archive["errors"]
    else:
        dense, ce = dense_source(dataset, device)
        first = dense[ce[:, 0].argmin(), 0, : layout.connections]
        weights, errors, _ = fit_dense(
            source_tensors(dataset, device),
            layout,
            steps=STEPS[dataset],
            restarts=dense.shape[1],
            device=device,
            seed=SEED + 57,
            anchor=first,
        )
        atomic_npz(file, weights=weights, errors=errors)
    order = np.argsort(errors[:, 0])
    aligned = source_profiles(weights[order], layout)
    array = (
        weights[..., : layout.connections]
        .reshape(
            *weights.shape[:2], layout.sites, layout.fan_in, layout.channels
        )
        .astype(float)
    )
    array /= np.linalg.norm(array, axis=-2, keepdims=True).clip(1e-12)
    unaligned = array.mean(1).transpose(0, 2, 1, 3).reshape(aligned.shape)
    bank = {}
    rank = settings["codes"] // layout.hidden
    for family, profiles in (("anchor", unaligned), ("anchor_align", aligned)):
        for blocks in (1, 4, 8, 16):
            groups = [
                ward_profiles(x, rank)
                for x in block_profiles(profiles, blocks)
            ]
            model = partition_model(groups, layout, normalized=True)
            name = f"{family}_{blocks}"
            path = root / f"{name}.npz"
            atomic_npz(path, **model)
            bank[name] = {"path": str(path), "family": family}
    save_json(manifest, bank)
    return bank
