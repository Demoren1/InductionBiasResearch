"""Estimate a shared aSRR basis with a fixed block rule and a small grid."""

import argparse
import json

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_transfer_refinement_v1.common import DATASETS, OUT
from experiments.paper_transfer_v2.common import CONFIG
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .adaptive import SHARES, validation, view_allocation
from .run import classification_source, guide_bases
from .signals import class_signals, gradient_signals, profile_kernels
from .structure import signed_groups, spectral_basis, unit_frobenius

STAGE = "universal_fixed"
GRADIENT_WEIGHTS = (0.0, 0.5)
MAX_BLOCKS_PER_SITE = 16


def model_bank(folder, profiles, data, ordered, layout, rank, shot):
    sx, sy, queries, qy = data
    blocks = max(
        b
        for b in range(1, MAX_BLOCKS_PER_SITE + 1)
        if layout.channels % b == 0
    )
    ref_count = layout.sites * max(
        b for b in (1, 2, 4) if layout.channels % b == 0
    )
    reference = np.stack(
        [signed_groups(k, rank) for k in profile_kernels(profiles, ref_count)]
    )
    kernels, difficulty = gradient_signals(
        ordered, queries, qy, layout, shot, reference
    )
    hardest = max(difficulty, key=difficulty.__getitem__)
    view_mass = {view: 0.5 / len(difficulty) for view in difficulty}
    view_mass[hardest] += 0.5
    allocation = view_allocation(view_mass, blocks)
    dense = profile_kernels(profiles, layout.sites * blocks)
    cross = [np.split(k, layout.sites * blocks) for k in kernels.values()]
    paired = [
        sum(
            weight * unit_frobenius(view[i].sum(0))
            for weight, view in zip(allocation[i % blocks], cross, strict=True)
        )
        for i in range(len(dense))
    ]
    guides = guide_bases(*class_signals(sx, sy, queries, qy, layout)[:3], rank)
    bank = {}

    def store(name, factors, moment, strength, beta, groups):
        path = folder / f"{name}.npz"
        atomic_npz(path, factor=np.asarray(factors, np.float32))
        bank[name] = {
            "path": str(path),
            "guide": f"mix_0_m{moment:g}",
            "groups": groups,
            "family": f"group_weight_{strength:g}",
            "group_weight": strength,
            "moment_weight": moment,
            "gradient_weight": beta,
        }

    for moment, guide in guides.items():
        store(f"mix_0_m{moment:g}", guide, moment, 0.0, 0.0, None)
    for beta in GRADIENT_WEIGHTS:
        groups = np.stack(
            [
                signed_groups((1 - beta) * a + beta * k, rank)
                for a, k in zip(dense, paired, strict=True)
            ]
        )
        hard = f"g{beta:g}_mix_1"
        store(hard, groups, 0.0, 1.0, beta, hard)
        for strength in SHARES:
            if strength in (0, 1):
                continue
            for moment, guide in guides.items():
                sources = np.repeat(guide, blocks, axis=0)
                factors = [
                    spectral_basis(g, u, strength)
                    for g, u in zip(groups, sources, strict=True)
                ]
                store(
                    f"g{beta:g}_mix_{strength:g}_m{moment:g}",
                    factors,
                    moment,
                    strength,
                    beta,
                    hard,
                )
    save_json(folder / "bank.json", bank)
    save_json(
        folder / "source.json",
        {
            "blocks_per_site": blocks,
            "views": list(queries),
            "allocation": allocation.tolist(),
        },
    )
    return bank


def run(dataset, device):
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    folder = OUT / dataset / STAGE
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "choices.json").exists():
        return
    if dataset == "har":
        from .har import development

        development(folder, device, build=model_bank)
        return
    config = CONFIG["real"][dataset]
    layout = Layout(*config["layout"])
    rank = config["codes"] // layout.hidden
    path = folder / "bank.json"
    if path.exists():
        bank = json.loads(path.read_text())
    else:
        bank = model_bank(
            folder,
            *classification_source(dataset, layout, device),
            layout,
            rank,
            config["shot"],
        )
    validation(dataset, bank, folder, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=(*DATASETS, "har"))
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    run(args.dataset, args.device)
