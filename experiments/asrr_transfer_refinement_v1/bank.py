"""Construct known aSRR variants and ratio-weighted follow-up candidates."""

import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.gradient_groups import (
    gradient_refine,
    gradient_tree,
    orthogonal_groups,
)
from experiments.asrr_benchmark_families_v1.groups import ROOT as FAMILY
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.paper_transfer_v2.common import CONFIG, OUT as PAPER
from experiments.paper_transfer_v2.source import dense_source
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.real_group_transfer_v1.study import source_profiles
from experiments.sharing_structure_research_v1.blind_contract import save_json
from experiments.srr_cluster_v1.clustering import centroid_fit

from .bases import factor_model, local_basis, partition_model, ward_profiles
from .common import BLOCK_COUNTS, OUT, SEED
from .gradient import source_kernels


def block_profiles(profiles, count):
    return [
        x.transpose(1, 0, 2).reshape(x.shape[1], -1)
        for x in np.split(profiles, count, axis=-1)
    ]


def decode_groups(index, count, layout):
    index = index.reshape(layout.sites, layout.fan_in, layout.channels)
    inputs = index.transpose(1, 0, 2).reshape(layout.fan_in, layout.hidden)
    return np.stack(
        [
            np.unique(inputs[:, i], return_inverse=True)[1]
            for i in range(0, layout.hidden, layout.hidden // count)
        ]
    )


def known_gradient_models(dataset, layout):
    result = {}
    path = FAMILY / "gradient_asrr" / dataset / "bank.npz"
    if path.exists():
        for key, value in read_data(path).items():
            if key.startswith("s5_"):
                result[f"gradient_saved_{key}"] = (
                    "gradient",
                    factor_model(value, layout),
                )
            if key == "normalized_threshold":
                result["threshold"] = (
                    "threshold",
                    factor_model(value, layout),
                )
    if dataset == "aug_omniglot":
        path = FAMILY / "gradient_mixture" / dataset / "bank.npz"
        for key, value in read_data(path).items():
            if key.startswith("s5_") and key.rsplit("_", 1)[-1] in {
                "0000",
                "0250",
                "0500",
                "0700",
                "1000",
            }:
                result[f"mixture_{key}"] = (
                    "gradient_mixture",
                    factor_model(value, layout),
                )
        path = FAMILY / "gradient_blocks" / dataset / "bank.npz"
        for key, value in read_data(path).items():
            result[f"mix_blocks_{key}"] = (
                "gradient_blocks",
                factor_model(value, layout),
            )
    return result


def fc100_controls(layout, codes):
    result = {}
    paths = {
        "compensation": Path(
            "artifacts/asrr_compensated_merging_2026-11-19/fc100/global_indexators.npz"
        ),
        "compatibility": Path(
            "artifacts/asrr_compatible_fc100_ninapro_2026-11-19/fc100/fixed_indices.npz"
        ),
    }
    for family, path in paths.items():
        for key, index in read_data(path).items():
            if (
                index.shape == (layout.fan_in, layout.hidden)
                and len(np.unique(abs(index))) == codes
            ):
                flat = index.reshape(
                    layout.fan_in, layout.sites, layout.channels
                )
                result[f"{family}_{key}"] = (
                    family,
                    {"index": flat.transpose(1, 0, 2).ravel()},
                )
    return result


def current_model(dataset, layout):
    choice = json.loads((PAPER / dataset / "asrr_choice.json").read_text())
    path = Path(choice["model"])
    if not path.stem.startswith("input"):
        raise ValueError("The primary comparisons use input partitions")
    count = int(path.stem.removeprefix("input"))
    return count, decode_groups(read_data(path)["index"], count, layout)


def known_candidates(dataset, layout, profiles, rank, device):
    old_count, old_groups = current_model(dataset, layout)
    raw = partition_model(old_groups, layout, normalized=False)
    normalized = partition_model(old_groups, layout, normalized=True)
    result = {
        "ward_fresh_coordinates": ("ward", raw),
        "ward_unit_norm": ("normalization", normalized),
        **known_gradient_models(dataset, layout),
    }
    for count in BLOCK_COUNTS:
        slabs = block_profiles(profiles, count)
        if count != old_count:
            groups = [ward_profiles(x, rank) for x in slabs]
            result[f"ward_blocks_{count}"] = (
                "repartition",
                partition_model(groups, layout, normalized=True),
            )
        groups = [
            centroid_fit(
                x, rank, "lloyd", SEED + 197 * i + 73 * count, device
            )[0]
            for i, x in enumerate(slabs)
        ]
        result[f"lloyd_blocks_{count}"] = (
            "lloyd",
            partition_model(groups, layout, normalized=True),
        )
    if dataset == "fc100":
        result.update(fc100_controls(layout, rank * layout.hidden))
    return result


def gradient_candidates(layout, profiles, rank, kernels, counts):
    result = {}
    for count in counts:
        slabs = block_profiles(profiles, count)
        groups = [ward_profiles(x, rank) for x in slabs]
        for name, matrix in kernels.items():
            values = matrix.reshape(
                count, -1, layout.fan_in, layout.fan_in
            ).sum(1)
            labels = [
                gradient_refine(k, g, passes=3)
                for k, g in zip(values, groups, strict=True)
            ]
            result[f"gradient_{name}_b{count}"] = (
                "gradient",
                partition_model(labels, layout, normalized=True),
            )
            if name in ("initial", "adapted"):
                labels = [
                    gradient_refine(k, gradient_tree(k, rank), passes=3)
                    for k in values
                ]
                result[f"gradient_tree_{name}_b{count}"] = (
                    "gradient",
                    partition_model(labels, layout, normalized=True),
                )
    return result


def relaxed_candidates(layout, profiles, rank, kernels, counts):
    result = {}
    for count in counts:
        slabs = block_profiles(profiles, count)
        covariance = [x @ x.T for x in slabs]
        per_input = [
            x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)
            for x in slabs
        ]
        for profiles_to_cluster, name in (
            (slabs, "euclidean"),
            (per_input, "angular"),
        ):
            partition = [ward_profiles(x, rank) for x in profiles_to_cluster]
            basis = [
                local_basis(s, g, rank)
                for s, g in zip(covariance, partition, strict=True)
            ]
            result[f"ratio_{name}_b{count}"] = (
                "ratio",
                factor_model(basis, layout),
            )
        if rank > 2:
            labels = [
                cut_tree(linkage(x, method="ward"), rank // 2).ravel()
                for x in slabs
            ]
            signals = {"covariance": covariance}
            for name in ("initial", "adapted"):
                matrix = kernels[name]
                signals[name] = matrix.reshape(
                    count, -1, layout.fan_in, layout.fan_in
                ).sum(1)
            for name, signal in signals.items():
                basis = [
                    local_basis(s, g, rank)
                    for s, g in zip(signal, labels, strict=True)
                ]
                result[f"local_rank_{name}_b{count}"] = (
                    "local_rank",
                    factor_model(basis, layout),
                )
    return result


def build_bank(dataset, stage, device):
    folder = OUT / dataset / stage
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "bank.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    config = CONFIG["real"][dataset]
    layout = Layout(*config["layout"])
    rank = config["codes"] // layout.hidden
    state, validation = dense_source(dataset, device)
    profiles = source_profiles(state[np.argsort(validation[:, 0])], layout)
    count, groups = current_model(dataset, layout)
    kpath = OUT / dataset / "kernels.npz"
    if kpath.exists():
        kernels = read_data(kpath)
    else:
        factor = factor_model(
            [orthogonal_groups(row) for row in groups], layout
        )["factor"]
        kernels = source_kernels(
            dataset, layout, config["shot"], factor, device
        )
        atomic_npz(kpath, **kernels)
    counts = tuple(sorted({1, count, 4, 8, min(16, layout.sites * 4)}))
    if stage == "known":
        candidates = known_candidates(dataset, layout, profiles, rank, device)
        candidates.update(
            gradient_candidates(
                layout,
                profiles,
                rank,
                kernels,
                (count, 4) if count != 4 else (4,),
            )
        )
    else:
        candidates = relaxed_candidates(
            layout, profiles, rank, kernels, counts
        )
    serial = {}
    for key, (family, representation) in candidates.items():
        path = folder / f"{key}.npz"
        atomic_npz(path, **representation)
        serial[key] = {"family": family, "path": str(path)}
    save_json(manifest, serial)
    print(dataset, stage, "source candidates", len(serial), flush=True)
    return serial
