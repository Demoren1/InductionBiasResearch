"""Fit the same signed-group and spectral model on each source panel."""

import argparse
import json

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.gradient_mixture import (
    clean_source_queries,
)
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_transfer_refinement_v1.common import DATASETS, OUT
from experiments.asrr_transfer_refinement_v1.contrasts import (
    discriminant_basis,
)
from experiments.asrr_transfer_refinement_v1.study import select
from experiments.paper_transfer_v2.bilevel import stratified_sources
from experiments.paper_transfer_v2.common import CONFIG
from experiments.paper_transfer_v2.source import dense_source, source_tensors
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.real_group_transfer_v1.study import source_profiles
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .signals import (
    block_gradients,
    class_signals,
    gradient_signals,
    profile_kernels,
)
from .structure import signed_groups, spectral_basis

STAGE = "universal_v1"
BLOCKS = (1, 2, 4, 8, 16)
GRADIENT_WEIGHTS = (0.0, 0.5, 1.0)
GROUP_WEIGHTS = (0.125, 0.5, 0.875, 1.0)
MOMENT_WEIGHTS = (0.0, 0.25)
RIDGE = 0.03


def guide_bases(signal, covariance, noise, rank, *, ridge=RIDGE):
    fixed = np.empty((signal.shape[-1], 0))
    return {
        moment: np.stack(
            [
                discriminant_basis(a, b, rank, ridge, fixed)
                for a, b in zip(
                    (1 - moment) * signal + moment * covariance,
                    noise,
                    strict=True,
                )
            ]
        )
        for moment in MOMENT_WEIGHTS
    }


def candidate_bases(profiles, signals, gradient, difficulty, layout, rank):
    guides = guide_bases(*signals, rank)
    for moment, guide in guides.items():
        yield (1, 0.0, 0.0, moment), guide

    for blocks in BLOCKS:
        if layout.channels % blocks:
            continue
        count = layout.sites * blocks
        dense = profile_kernels(profiles, count)
        cross = block_gradients(gradient, difficulty, count)
        for beta in GRADIENT_WEIGHTS:
            groups = np.stack(
                [
                    signed_groups((1 - beta) * a + beta * k, rank)
                    for a, k in zip(dense, cross, strict=True)
                ]
            )
            for share in GROUP_WEIGHTS:
                for moment, guide in guides.items():
                    if share == 1 and moment != 0:
                        continue
                    repeated = np.repeat(guide, blocks, axis=0)
                    factor = np.stack(
                        [
                            spectral_basis(g, u, share)
                            for g, u in zip(groups, repeated, strict=True)
                        ]
                    )
                    yield (blocks, beta, share, moment), factor


def model_bank(folder, profiles, data, ordered, layout, rank, shot):
    manifest = folder / "bank.json"
    sx, sy, queries, qy = data
    ref_blocks = max(k for k in (1, 2, 4) if layout.channels % k == 0)
    initial = np.stack(
        [
            signed_groups(kernel, rank)
            for kernel in profile_kernels(profiles, layout.sites * ref_blocks)
        ]
    )
    stats = class_signals(sx, sy, queries, qy, layout)[:3]
    gradient, difficulty = gradient_signals(
        ordered, queries, qy, layout, shot, initial
    )
    bank = {}
    for knobs, factor in candidate_bases(
        profiles, stats, gradient, difficulty, layout, rank
    ):
        blocks, beta, share, moment = knobs
        name = f"b{blocks}_g{beta:g}_s{share:g}_m{moment:g}"
        path = folder / f"{name}.npz"
        atomic_npz(path, factor=factor.astype(np.float32))
        bank[name] = {
            "path": str(path),
            "family": f"group_weight_{share:g}",
            "blocks_per_site": blocks,
            "gradient_weight": beta,
            "group_weight": share,
            "moment_weight": moment,
        }
    save_json(manifest, bank)
    save_json(folder / "source_views.json", difficulty)
    return bank


def classification_source(dataset, layout, device):
    sx, sy, qx, qy = source_tensors(dataset, device)
    ordered, _, _ = stratified_sources(dataset, device, layout)
    queries = {"query": qx}
    if dataset == "aug_omniglot":
        clean = clean_source_queries()[:, -qx.shape[1] :]
        queries["clean"] = (
            torch.as_tensor(clean, device=device, dtype=torch.float32) / 255
        )
    states, losses = dense_source(dataset, device)
    profiles = source_profiles(states[np.argsort(losses[:, 0])], layout)
    return profiles, (sx, sy, queries, qy), ordered


def run(dataset, device):
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    settings = CONFIG["real"][dataset]
    layout = Layout(*settings["layout"])
    rank = settings["codes"] // layout.hidden
    folder = OUT / dataset / STAGE
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "bank.json"
    if manifest.exists():
        bank = json.loads(manifest.read_text())
    else:
        profiles, data, ordered = classification_source(
            dataset, layout, device
        )
        bank = model_bank(
            folder, profiles, data, ordered, layout, rank, settings["shot"]
        )
        print(f"{dataset} built {len(bank)} candidates", flush=True)
    if not (folder / "choices.json").exists():
        select(dataset, bank, folder, device, average_views=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    run(args.dataset, args.device)
