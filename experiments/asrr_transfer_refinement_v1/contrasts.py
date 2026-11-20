"""Find transfer directions that preserve reproducible source class contrasts."""

import json

import numpy as np
from scipy.linalg import eigh, null_space

from experiments.asrr_benchmark_families_v1.gradient_groups import (
    gradient_refine,
    gradient_tree,
    orthogonal_groups,
)
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.paper_transfer_v2.common import CONFIG
from experiments.paper_transfer_v2.source import dense_source, source_tensors
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.real_group_transfer_v1.study import source_profiles
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .bank import block_profiles
from .bases import factor_model, partition_model, ward_profiles
from .common import OUT


def means_and_noise(x, y, classes):
    means = np.stack(
        [
            np.stack(
                [
                    task[labels == category].mean(0)
                    for category in range(classes)
                ]
            )
            for task, labels in zip(x, y, strict=True)
        ]
    )
    centered = x - means[np.arange(len(y))[:, None], y]
    centered = centered.astype(np.float64).transpose(2, 0, 1, 3)
    within = centered.reshape(x.shape[2], -1, x.shape[-1])
    covariance = within.mT @ within / within.shape[1]
    contrasts = means - means.mean(1, keepdims=True)
    return contrasts.astype(np.float64), covariance


def scatter_matrices(dataset, layout, device):
    sx, sy, qx, qy = [
        value.cpu().numpy() for value in source_tensors(dataset, device)
    ]
    a, sa = means_and_noise(sx, sy, layout.outputs)
    b, sb = means_and_noise(qx, qy, layout.outputs)
    cross = np.einsum("tcsi,tcsj->sij", a, b, optimize=True) / (
        len(a) * layout.outputs
    )
    return (cross + cross.mT) / 2, (sa + sb) / 2


def discriminant_basis(signal, noise, rank, ridge, fixed):
    dimension = signal.shape[0]
    covariance = noise + ridge * max(
        np.trace(noise) / dimension, 1e-8
    ) * np.eye(dimension)
    if fixed.shape[1]:
        complement = null_space(fixed.T)
    else:
        complement = np.eye(dimension)
    a = complement.T @ signal @ complement
    b = complement.T @ covariance @ complement
    remaining = rank - fixed.shape[1]
    _, directions = eigh(
        a, b, subset_by_index=(len(a) - remaining, len(a) - 1)
    )
    combined = np.column_stack((fixed, complement @ directions))
    return np.linalg.qr(combined)[0]


def build_contrasts(dataset, device):
    root = OUT / dataset / "contrasts"
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "bank.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    cfg = CONFIG["real"][dataset]
    layout = Layout(*cfg["layout"])
    rank = cfg["codes"] // layout.hidden
    signal, noise = scatter_matrices(dataset, layout, device)
    source, error = dense_source(dataset, device)
    profiles = source_profiles(source[np.argsort(error[:, 0])], layout)
    bank = {}
    for count in sorted({1, layout.sites}):
        slab = block_profiles(profiles, count)
        k = signal.reshape(count, -1, layout.fan_in, layout.fan_in).mean(1)
        covariance = noise.reshape(
            count, -1, layout.fan_in, layout.fan_in
        ).mean(1)
        groups = [ward_profiles(x, rank) for x in slab]
        hard = [gradient_refine(a, gradient_tree(a, rank)) for a in k]
        name = f"contrast_groups_{count}"
        path = root / f"{name}.npz"
        atomic_npz(path, **partition_model(hard, layout, normalized=True))
        bank[name] = {"family": "contrast_groups", "path": str(path)}
        for ridge in (0.03, 0.3, 3.0):
            for anchors, family in (
                (rank // 3, "anchored_contrast"),
                (0, "contrast_basis"),
            ):
                output = []
                for a, within, labels in zip(
                    k, covariance, groups, strict=True
                ):
                    basis = orthogonal_groups(labels)
                    scores = (basis * (a @ basis)).sum(0) / (
                        (basis * (within @ basis)).sum(0) + 1e-8
                    )
                    fixed = basis[:, np.argsort(-scores)[:anchors]]
                    output.append(
                        discriminant_basis(a, within, rank, ridge, fixed)
                    )
                name = f"{family}_{count}_{ridge:g}"
                path = root / f"{name}.npz"
                atomic_npz(path, **factor_model(output, layout))
                bank[name] = {"family": family, "path": str(path)}
    save_json(manifest, bank)
    return bank
