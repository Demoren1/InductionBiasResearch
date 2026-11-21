"""Estimate the shared basis in one spectral problem with a signed-group prior."""

import argparse
import json

import numpy as np
import torch
from scipy.linalg import cholesky, eigh, solve_triangular

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_transfer_refinement_v1.common import DATASETS, OUT
from experiments.paper_transfer_v2.common import CONFIG
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .adaptive import (
    REVISED_STAGE,
    SHARES,
    affinity_halves,
    source_groups,
    validation,
)
from .run import MOMENT_WEIGHTS, RIDGE, classification_source
from .signals import class_signals, gradient_signals, profile_kernels
from .structure import signed_groups, unit_frobenius

STAGE = "universal_spectral"


def whitening(signal, noise):
    scale = max(float(np.trace(noise) / len(noise)), 1e-8)
    factor = cholesky(noise + RIDGE * scale * np.eye(len(noise)), lower=True)
    left = solve_triangular(factor, signal, lower=True)
    matrix = solve_triangular(factor, left.T, lower=True).T
    matrix = unit_frobenius((matrix + matrix.T) / 2)
    return factor, matrix, np.sqrt(scale)


def solve_basis(groups, whitener, rank, strength):
    factor, signal, scale = whitener
    if strength > 0:
        group_span = np.linalg.qr(factor.T @ groups)[0]
        if strength == 1:
            return scale * solve_triangular(factor.T, group_span)
        signal = (1 - strength) * signal + strength * (
            group_span @ group_span.T
        )
    _, directions = eigh(
        signal, subset_by_index=(len(signal) - rank, len(signal) - 1)
    )
    return scale * solve_triangular(factor.T, directions[:, ::-1])


def shuffled_bases(operators, moment, strength, rng):
    matrices = read_data(operators)
    groups = matrices["groups"]
    signal = (1 - moment) * matrices["mean"] + moment * matrices["covariance"]
    systems = [
        whitening(s, c) for s, c in zip(signal, matrices["noise"], strict=True)
    ]
    channels = len(groups) // len(systems)
    return np.stack(
        [
            solve_basis(
                g[rng.permutation(len(g))],
                systems[h // channels],
                g.shape[1],
                strength,
            )
            for h, g in enumerate(groups)
        ]
    )


def fit_basis_bank(folder, groups, features, layout, rank):
    mean, covariance, noise = features
    atomic_npz(
        folder / "operators.npz",
        groups=groups,
        mean=mean,
        covariance=covariance,
        noise=noise,
    )
    bank = {}
    for moment in MOMENT_WEIGHTS:
        systems = [
            whitening(signal, within)
            for signal, within in zip(
                (1 - moment) * mean + moment * covariance, noise, strict=True
            )
        ]
        for strength in SHARES:
            name = (
                f"mix_{strength:g}"
                if strength == 1
                else f"mix_{strength:g}_m{moment:g}"
            )
            if name in bank:
                continue
            if strength == 0:
                factors = [
                    solve_basis(None, system, rank, strength)
                    for system in systems
                ]
            else:
                factors = [
                    solve_basis(
                        g, systems[h // layout.channels], rank, strength
                    )
                    for h, g in enumerate(groups)
                ]
            path = folder / f"{name}.npz"
            atomic_npz(path, factor=np.asarray(factors, np.float32))
            bank[name] = {
                "path": str(path),
                "family": f"group_weight_{strength:g}",
                "group_weight": strength,
                "moment_weight": moment,
                "guide": f"mix_0_m{moment:g}",
            }
    save_json(folder / "bank.json", bank)
    return bank


def model_bank(folder, profiles, data, ordered, layout, rank, shot):
    existing = folder.parent / REVISED_STAGE / "mix_1.npz"
    sx, sy, queries, qy = data
    if existing.exists():
        groups = read_data(existing)["factor"].astype(np.float64)
    else:
        reference = np.stack(
            [
                signed_groups(k, rank)
                for k in profile_kernels(profiles, layout.sites)
            ]
        )
        groups, _, _ = source_groups(
            affinity_halves(profiles),
            *gradient_signals(
                ordered, queries, qy, layout, shot, reference, splits=2
            ),
            layout,
            rank,
            calibrated=False,
        )
    stats = class_signals(sx, sy, queries, qy, layout)[:3]
    return fit_basis_bank(folder, groups, stats, layout, rank)


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
