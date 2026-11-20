"""Select the same group-and-subspace family by mean post-refit accuracy."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.paper_transfer_v2.common import CONFIG
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .allocation import input_groups, spectrum, unequal_model, unequal_rank
from .bank import build_bank
from .bases import factor_model
from .common import DATASETS, OUT
from .contrasts import discriminant_basis, scatter_matrices
from .fusion import align_blocks, fuse
from .reliable import moment_signals
from .study import select

HARD_FAMILIES = {
    "ward",
    "normalization",
    "repartition",
    "lloyd",
    "gradient",
    "gradient_mixture",
    "gradient_blocks",
}
BLOCKS = (1, 4, 8, 16, 32)
RIDGE = 0.03
MOMENT_WEIGHT = 0.25


def span(factor):
    return np.linalg.qr(factor.astype(np.float64))[0]


def discriminants(signal, noise, labels, rank, fixed):
    return [
        spectrum(a, b, g, fixed, RIDGE)
        for a, b, g in zip(signal, noise, labels, strict=True)
    ]


def class_guides(signal, noise, labels, rank):
    fixed = max(2, rank // 8)
    base = discriminants(signal, noise, labels, rank, 0)
    grouped = discriminants(signal, noise, labels, rank, fixed)
    return (
        span(np.stack([free[:, :rank] for _, _, free in base])),
        span(
            np.stack(
                [
                    np.column_stack((g, free[:, : rank - fixed]))
                    for _, g, free in grouped
                ]
            )
        ),
    )


def moment_guides(signal, noise, labels, rank):
    base, grouped = [], []
    fixed = max(2, rank // 8)
    for a, b, g in zip(signal, noise, labels, strict=True):
        head = np.eye(rank)[g] / np.sqrt(np.bincount(g))
        scores = (head * (a @ head)).sum(0) / (
            (head * (b @ head)).sum(0) + 1e-8
        )
        base.append(discriminant_basis(a, b, rank, RIDGE, head[:, :0]))
        grouped.append(
            discriminant_basis(
                a, b, rank, RIDGE, head[:, np.argsort(-scores)[:fixed]]
            )
        )
    return np.stack(base), np.stack(grouped)


def source_signals(dataset, layout, rank, device):
    labels = input_groups(dataset, device, layout, rank)
    mean, noise = scatter_matrices(dataset, layout, device)
    _, moments = moment_signals(dataset, layout, device)
    second = moments[f"moments_{MOMENT_WEIGHT:g}"]
    mean_pure, mean_grouped = class_guides(mean, noise, labels, rank)
    moment_pure, moment_grouped = moment_guides(second, noise, labels, rank)
    guides = {
        "means": (mean_pure, mean_grouped, 0.0),
        "means_seeded": (mean_grouped, mean_pure, 0.0),
        "moments": (moment_pure, moment_grouped, 0.0),
        "moments_seeded": (moment_pure, moment_grouped, MOMENT_WEIGHT),
    }
    return labels, noise, {"means": mean, "moments": second}, guides


def build(dataset, folder, device):
    manifest = folder / "bank.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    config = CONFIG["real"][dataset]
    layout = Layout(*config["layout"])
    rank = config["codes"] // layout.hidden
    sites = layout.sites
    source = build_bank(dataset, "known", device)
    bank = {
        key: {**row, "group_directions": rank}
        for key, row in source.items()
        if row["family"] in HARD_FAMILIES
    }

    def add(name, family, model, **metadata):
        path = folder / f"{name}.npz"
        atomic_npz(path, **model)
        bank[name] = {
            "path": str(path),
            "family": family,
            "group_directions": 1,
            **metadata,
        }

    labels, noise, signals, guides = source_signals(
        dataset, layout, rank, device
    )
    for key, row in source.items():
        if row["family"] not in {"ward", "repartition", "lloyd"}:
            continue
        groups = read_data(Path(row["path"]))["factor"]
        if len(groups) not in {*BLOCKS, sites}:
            continue
        groups = span(groups)
        for kind, (first, second_guide, alpha) in guides.items():
            values = [
                fuse(*matrices, alpha, 1)
                for matrices in align_blocks(groups, first, second_guide)
            ]
            add(
                f"{key}_{kind}",
                f"projected_{kind}",
                factor_model(values, layout),
                group_source=key,
            )

    for kind, signal in signals.items():
        parts = discriminants(signal, noise, labels, rank, 1)
        limits = unequal_rank([row[0] for row in parts], rank, 1)
        for name, ranks in (
            ("uniform", [rank] * sites),
            ("allocated", limits.tolist()),
        ):
            if name == "allocated" and ranks == [rank] * sites:
                continue
            bases = [
                np.column_stack((g, free[:, : r - 1]))
                for (_, g, free), r in zip(parts, ranks, strict=True)
            ]
            model = (
                unequal_model([span(b) for b in bases], layout)
                if name == "allocated"
                else factor_model(span(np.stack(bases)), layout)
            )
            add(f"{kind}_{name}", "direct", model, rank_by_site=ranks)
    save_json(manifest, bank)
    return bank


def develop(dataset, device):
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    folder = OUT / dataset / "unified"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "choices.json").exists():
        return
    bank = build(dataset, folder, device)
    print(dataset, "unified candidates", len(bank), flush=True)
    select(dataset, bank, folder, device, average_views=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    develop(args.dataset, torch.device(args.device))
