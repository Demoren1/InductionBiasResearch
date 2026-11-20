"""Use independent subjects or second moments to estimate transferable contrasts."""

import json

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.gradient_groups import (
    orthogonal_groups,
)
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.paper_transfer_v2.common import CONFIG
from experiments.paper_transfer_v2.source import dense_source, source_tensors
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.real_group_transfer_v1.study import source_profiles
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .bank import block_profiles
from .bases import factor_model, ward_profiles
from .common import OUT
from .contrasts import discriminant_basis, means_and_noise, scatter_matrices


def symmetric(matrix):
    return 0.5 * (matrix + matrix.mT)


def subject_signals(dataset, layout, device):
    sx, sy, qx, qy = [
        value.cpu().numpy() for value in source_tensors(dataset, device)
    ]
    a, sa = means_and_noise(sx, sy, layout.outputs)
    b, sb = means_and_noise(qx, qy, layout.outputs)
    classes = layout.outputs
    within = np.einsum("tcsi,tcsj->sij", a, b, optimize=True)
    across = (
        np.einsum("csi,csj->sij", a.sum(0), b.sum(0), optimize=True) - within
    )
    a -= a.mean(0, keepdims=True)
    b -= b.mean(0, keepdims=True)
    nuisance = np.einsum("tcsi,tcsj->sij", a, a, optimize=True) + np.einsum(
        "tcsi,tcsj->sij", b, b, optimize=True
    )
    noise = (sa + sb) / 2 + nuisance / (2 * len(a) * classes)
    same = symmetric(within / (len(a) * classes))
    shared = symmetric(across / (len(a) * (len(a) - 1) * classes))
    return noise, {
        f"subjects_{share:g}": (1 - share) * same + share * shared
        for share in (0.5, 1.0)
    }


def covariance_differences(features, labels, layout):
    order = labels.argsort(dim=1, stable=True)
    tasks = torch.arange(len(features), device=features.device)[:, None]
    values = (
        features[tasks, order]
        .reshape(
            len(features), layout.outputs, -1, layout.sites, layout.fan_in
        )
        .transpose(2, 3)
    )
    centered = values - values.mean(-2, keepdim=True)
    covariance = centered.mT @ centered / centered.shape[-2]
    return covariance - covariance.mean(1, keepdim=True)


def moment_signals(dataset, layout, device):
    sx, sy, qx, qy = source_tensors(dataset, device)
    a = covariance_differences(sx, sy, layout)
    b = covariance_differences(qx, qy, layout)
    product = symmetric((a @ b).mean((0, 1))).cpu().numpy()
    means, noise = scatter_matrices(dataset, layout, device)
    means /= max(np.linalg.norm(means), 1e-12)
    product /= max(np.linalg.norm(product), 1e-12)
    return noise, {
        f"moments_{share:g}": (1 - share) * means + share * product
        for share in (0.25, 0.75, 1.0)
    }


def activity_signals(dataset, layout, device):
    sx, sy, qx, qy = source_tensors(dataset, device)
    a = covariance_differences(sx, sy, layout)
    b = covariance_differences(qx, qy, layout)
    same = (a @ b).sum(0)
    shared = (a.sum(0) @ b.sum(0) - same) / (len(a) - 1)
    noise, means = subject_signals(dataset, layout, device)
    mean = means["subjects_1"]
    mean /= max(np.linalg.norm(mean), 1e-12)
    result = {}
    for name, value in (("same", same), ("shared", shared)):
        moment = symmetric(value.mean(0)).cpu().numpy()
        moment /= max(np.linalg.norm(moment), 1e-12)
        for share in (0.25, 0.75):
            result[f"{name}_moments_{share:g}"] = (
                1 - share
            ) * mean + share * moment
    return noise, result


def build_reliable(dataset, device, *, stage):
    folder = OUT / dataset / stage
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "bank.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    settings = CONFIG["real"][dataset]
    layout = Layout(*settings["layout"])
    rank = settings["codes"] // layout.hidden
    if stage == "second_moments" and dataset == "wisdm":
        noise, signals = activity_signals(dataset, layout, device)
    elif dataset == "wisdm":
        noise, signals = subject_signals(dataset, layout, device)
    elif dataset == "mnist_torus":
        noise, signals = moment_signals(dataset, layout, device)
    else:
        raise ValueError(f"Unsupported reliability hypothesis for {dataset}")
    weights, validation = dense_source(dataset, device)
    profiles = source_profiles(weights[np.argsort(validation[:, 0])], layout)
    bank = {}
    for count in sorted({1, layout.sites}):
        within = noise.reshape(count, -1, layout.fan_in, layout.fan_in).mean(1)
        groups = [
            ward_profiles(x, rank) for x in block_profiles(profiles, count)
        ]
        for name, matrix in signals.items():
            signal = matrix.reshape(
                count, -1, layout.fan_in, layout.fan_in
            ).mean(1)
            for fixed in sorted({0, 1, max(2, rank // 8)}):
                for ridge in (0.03, 0.3):
                    values = []
                    for a, b, labels in zip(
                        signal, within, groups, strict=True
                    ):
                        basis = orthogonal_groups(labels)
                        score = (basis * (a @ basis)).sum(0) / (
                            (basis * (b @ basis)).sum(0) + 1e-8
                        )
                        anchors = basis[:, np.argsort(-score)[:fixed]]
                        values.append(
                            discriminant_basis(a, b, rank, ridge, anchors)
                        )
                    tag = f"{name}_b{count}_g{fixed}_r{ridge:g}"
                    path = folder / f"{tag}.npz"
                    atomic_npz(path, **factor_model(values, layout))
                    family = (
                        "reliable_with_groups" if fixed else "reliable_basis"
                    )
                    bank[tag] = {"path": str(path), "family": family}
    save_json(manifest, bank)
    return bank
