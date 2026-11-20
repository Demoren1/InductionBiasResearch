"""Select the number of fixed group directions and allocate rank across sites."""

import json

import numpy as np
from scipy.linalg import eigh, null_space

from experiments.asrr_benchmark_families_v1.gradient_groups import (
    orthogonal_groups,
)
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.paper_transfer_v2.common import CONFIG
from experiments.paper_transfer_v2.source import dense_source
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.real_group_transfer_v1.study import source_profiles
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .bank import block_profiles
from .bases import factor_model, ward_profiles
from .common import OUT
from .contrasts import scatter_matrices


def spectrum(signal, within, groups, anchors, ridge):
    size = len(signal)
    penalty = ridge * max(np.trace(within) / size, 1e-8)
    noise = within + penalty * np.eye(size)
    group_basis = orthogonal_groups(groups)
    score = (group_basis * (signal @ group_basis)).sum(0) / (
        (group_basis * (noise @ group_basis)).sum(0)
    )
    fixed = group_basis[:, np.argsort(-score)[:anchors]]
    orthogonal = null_space(fixed.T) if anchors else np.eye(size)
    a = orthogonal.T @ signal @ orthogonal
    b = orthogonal.T @ noise @ orthogonal
    values, vectors = eigh(a, b)
    return values[::-1], fixed, orthogonal @ vectors[:, ::-1]


def unequal_rank(values, rank, anchors):
    minimum = max(anchors + 1, rank // 2)
    ranks = np.full(len(values), minimum)
    options = [
        (float(v), site, i)
        for site, a in enumerate(values)
        for i, v in enumerate(a)
        if i >= minimum - anchors
    ]
    options.sort(key=lambda x: (-x[0], x[1], x[2]))
    for _, site, _ in options[: len(values) * (rank - minimum)]:
        ranks[site] += 1
    return ranks


def unequal_model(bases, layout):
    maximum = max(x.shape[1] for x in bases)
    factor = np.zeros((layout.sites, layout.fan_in, maximum), dtype=np.float32)
    index = np.zeros((layout.sites, maximum, layout.channels), dtype=np.int32)
    scales = []
    offset = 0
    for site, basis in enumerate(bases):
        rank = basis.shape[1]
        width = rank * layout.channels
        factor[site, :, :rank] = basis
        index[site, :rank] = np.arange(offset + 1, offset + 1 + width).reshape(
            rank, layout.channels
        )
        scales.extend([np.sqrt(2 / rank)] * width)
        offset += width
    return {
        "factor": factor,
        "index": index.ravel(),
        "initial_scale": np.asarray(scales, dtype=np.float32),
    }


def input_groups(dataset, device, layout, rank):
    weights, validation = dense_source(dataset, device)
    profiles = source_profiles(weights[np.argsort(validation[:, 0])], layout)
    return [
        ward_profiles(x, rank) for x in block_profiles(profiles, layout.sites)
    ]


def build_allocation(dataset, device):
    folder = OUT / dataset / "allocation"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "bank.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    cfg = CONFIG["real"][dataset]
    layout = Layout(*cfg["layout"])
    rank = cfg["codes"] // layout.hidden
    signal, within = scatter_matrices(dataset, layout, device)
    groups = input_groups(dataset, device, layout, rank)
    bank = {}
    for ridge in (0.03, 0.3):
        counts = sorted({0, 1, max(2, rank // 8), rank // 4})
        for anchors in counts:
            data = [
                spectrum(a, w, g, anchors, ridge)
                for a, w, g in zip(signal, within, groups, strict=True)
            ]
            budget = unequal_rank([row[0] for row in data], rank, anchors)
            for family, limits in (
                ("few_group_directions", np.full(len(data), rank)),
                ("unequal_site_rank", budget),
            ):
                if anchors == 0 and family == "few_group_directions":
                    continue
                bases = [
                    np.linalg.qr(
                        np.column_stack(
                            (fixed, rest[:, : int(size) - anchors])
                        )
                    )[0]
                    for (_, fixed, rest), size in zip(
                        data, limits, strict=True
                    )
                ]
                if (
                    sum(x.shape[1] for x in bases) * layout.channels
                    != cfg["codes"]
                ):
                    raise ValueError(
                        "The total first-layer code count has changed"
                    )
                name = f"{family}_{anchors}_{ridge:g}"
                path = folder / f"{name}.npz"
                model = (
                    unequal_model(bases, layout)
                    if family == "unequal_site_rank"
                    else factor_model(bases, layout)
                )
                atomic_npz(path, **model)
                bank[name] = {
                    "path": str(path),
                    "family": family,
                    "rank_by_site": limits.tolist(),
                    "fixed_groups": anchors,
                }
    save_json(manifest, bank)
    return bank
