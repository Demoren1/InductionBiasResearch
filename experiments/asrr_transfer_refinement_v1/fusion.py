"""Keep directions in the learned group span while combining source signals."""

import json
from pathlib import Path

import numpy as np

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.paper_transfer_v2.common import CONFIG
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .bases import factor_model
from .common import OUT


def chosen_factor(folder, family):
    choice = json.loads((folder / "choices.json").read_text())
    name = choice["family"][family]["candidate"]
    bank = json.loads((folder / "bank.json").read_text())
    factor = read_data(Path(bank[name]["path"]))["factor"]
    return np.linalg.qr(factor.astype(np.float64))[0]


def fuse(group, first, second, weight, anchors):
    rank = group.shape[1]
    signal = np.column_stack(
        (np.sqrt(1 - weight) * first, np.sqrt(weight) * second)
    )
    left = np.linalg.svd(group.T @ signal, full_matrices=False)[0]
    fixed = group @ left[:, :anchors]
    residual = signal - fixed @ (fixed.T @ signal)
    flexible = np.linalg.svd(residual, full_matrices=False)[0]
    return np.column_stack((fixed, flexible[:, : rank - anchors]))


def align_blocks(*factors):
    count = max(len(factor) for factor in factors)
    return zip(
        *(
            np.repeat(factor, count // len(factor), axis=0)
            for factor in factors
        ),
        strict=True,
    )


def guides(root, dataset):
    if dataset == "mnist_torus":
        first = chosen_factor(root / "reliable", "reliable_basis")
        second = chosen_factor(root / "reliable", "reliable_with_groups")
        references = ("ward", "repartition")
    elif dataset == "wisdm":
        first = chosen_factor(root / "allocation", "few_group_directions")
        second = chosen_factor(root / "ridge", "ward_warm_ridge")
        references = ("ward", "lloyd")
    else:
        raise ValueError(f"Unsupported group fusion dataset {dataset}")
    return first, second, references


def build_fusion(dataset):
    root = OUT / dataset
    folder = root / "fusion"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "bank.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    layout = Layout(*CONFIG["real"][dataset]["layout"])
    first, second, references = guides(root, dataset)
    bank = {}
    for reference in references:
        groups = chosen_factor(root / "known", reference)
        for weight in (0.0, 0.25, 0.5):
            for anchors in (1, 2):
                bases = [
                    fuse(*site, weight, anchors)
                    for site in align_blocks(groups, first, second)
                ]
                name = f"{reference}_{weight:g}_g{anchors}"
                path = folder / f"{name}.npz"
                atomic_npz(path, **factor_model(bases, layout))
                bank[name] = {
                    "path": str(path),
                    "family": "fused_group_span",
                    "group_source": reference,
                    "fixed_directions": anchors,
                }
    save_json(manifest, bank)
    return bank
