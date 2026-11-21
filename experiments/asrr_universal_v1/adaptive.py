"""Construct one shared basis with a single validated structural weight."""

import argparse
import json

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_transfer_refinement_v1.common import DATASETS, OUT
from experiments.asrr_transfer_refinement_v1.contrasts import (
    discriminant_basis,
)
from experiments.asrr_transfer_refinement_v1.study import (
    evaluate,
    ordering,
    validation_rows,
)
from experiments.paper_transfer_v2.common import CONFIG, freeze
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .run import RIDGE, classification_source, guide_bases
from .signals import class_signals, gradient_signals, profile_kernels
from .structure import signed_groups, unit_frobenius

STAGE = "universal_v2"
REVISED_STAGE = "universal_v3"
SHARES = (0.0, 0.125, 0.5, 1.0)


def signal_to_noise(agreement):
    renormalized = np.clip(agreement, 0, 1 - 1e-6)
    return renormalized / (1 - renormalized)


def affinity_halves(profiles):
    estimates = []
    for side in (profiles[::2], profiles[1::2]):
        rows = side.transpose(2, 1, 0)
        estimates.append(rows @ rows.mT / len(side))
    return np.stack(estimates)


def pool_neurons(halves):
    """Shrink the between-neuron differences by their split-half reliability."""
    first, second = halves
    empirical = (first + second) / 2
    center = empirical.mean(0, keepdims=True)
    u = first - first.mean(0, keepdims=True)
    v = second - second.mean(0, keepdims=True)
    variation = max(float(np.vdot(u, v)), 0.0)
    noise = float(np.square(u - v).sum() / 4)
    local_weight = variation / max(variation + noise, 1e-12)
    # Diagonal terms contain energy but provide no evidence about pair similarity.
    diagonal = np.arange(first.shape[-1])
    a = first.copy()
    b = second.copy()
    a[:, diagonal, diagonal] = 0
    b[:, diagonal, diagonal] = 0
    cosine = np.vdot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    return center + local_weight * (empirical - center), signal_to_noise(
        cosine
    )


def view_allocation(difficulty, neurons):
    losses = np.asarray(list(difficulty.values()))
    weight = losses / max(losses.sum(), 1e-12)
    if neurons < len(weight):
        return np.broadcast_to(weight, (neurons, len(weight)))
    ids = list(range(len(weight)))
    counts = np.ones(len(weight), dtype=int)
    for i in range(len(ids), neurons):
        selected = int(np.argmax((i + 1) * weight - counts))
        ids.append(selected)
        counts[selected] += 1
    return np.eye(len(weight))[ids]


def source_groups(
    weights, gradients, difficulty, layout, rank, *, calibrated=True
):
    panels = list(gradients)
    allocation = view_allocation(difficulty, layout.channels)
    groups = []
    weights_used = []
    for site in range(layout.sites):
        section = slice(site * layout.channels, (site + 1) * layout.channels)
        dense, quality = pool_neurons(weights[:, section])
        cross = {
            name: pool_neurons(gradients[name][:, section]) for name in panels
        }
        if not calibrated:
            quality = 1.0
        for h, proportions in enumerate(allocation):
            kernel = quality * unit_frobenius(dense[h])
            total = quality
            for name, proportion in zip(panels, proportions, strict=True):
                current, strength = cross[name]
                weight = proportion * (strength if calibrated else 1.0)
                kernel += weight * unit_frobenius(current[h])
                total += weight
            if total == 0:
                kernel = dense[h]
            groups.append(signed_groups(kernel, rank))
            weights_used.append(1 - quality / total if total else 0.0)
    return np.stack(groups), weights_used, allocation


def guide_basis(signal, covariance, noise, agreement, rank):
    reliability = signal_to_noise(agreement)
    alpha = float(reliability[1] / max(reliability.sum(), 1e-12))
    mixed = (1 - alpha) * signal + alpha * covariance
    fixed = np.empty((signal.shape[-1], 0))
    guide = np.stack(
        [
            discriminant_basis(a, b, rank, RIDGE, fixed)
            for a, b in zip(mixed, noise, strict=True)
        ]
    )
    return guide, alpha


def combine(groups, guide, strength):
    """Leading eigenspace of a convex combination of two projectors."""
    if strength == 1:
        return groups
    if strength == 0:
        return guide
    factors = np.column_stack(
        (np.sqrt(1 - strength) * guide, np.sqrt(strength) * groups)
    )
    left, _, _ = np.linalg.svd(factors, full_matrices=False)
    basis = left[:, : groups.shape[1]]
    first, _, second = np.linalg.svd(basis.T @ guide, full_matrices=False)
    return basis @ (first @ second)


def model_bank(
    folder, profiles, data, ordered, layout, rank, shot, *, calibrated=True
):
    sx, sy, queries, qy = data
    reference = np.stack(
        [
            signed_groups(k, rank)
            for k in profile_kernels(profiles, layout.sites)
        ]
    )
    groups, gradient_weights, allocation = source_groups(
        affinity_halves(profiles),
        *gradient_signals(
            ordered, queries, qy, layout, shot, reference, splits=2
        ),
        layout,
        rank,
        calibrated=calibrated,
    )
    stats = class_signals(sx, sy, queries, qy, layout)
    if calibrated:
        guide, covariance_weight = guide_basis(*stats, rank)
        guides = {covariance_weight: guide}
    else:
        guides = guide_bases(*stats[:3], rank)
    bank = {}
    for moment, guide in guides.items():
        repeated = np.repeat(guide, layout.channels, axis=0)
        for strength in SHARES:
            name = f"mix_{strength:g}"
            guide_name = "mix_0"
            if not calibrated and strength != 1:
                name += f"_m{moment:g}"
            if not calibrated:
                guide_name += f"_m{moment:g}"
            if name in bank:
                continue
            path = folder / f"{name}.npz"
            factors = [
                combine(g, u, strength)
                for g, u in zip(groups, repeated, strict=True)
            ]
            if strength == 0:
                factors = guide
            atomic_npz(path, factor=np.asarray(factors, np.float32))
            bank[name] = {
                "path": str(path),
                "family": f"group_weight_{strength:g}",
                "group_weight": strength,
                "moment_weight": moment,
                "guide": guide_name,
            }
    save_json(folder / "bank.json", bank)
    save_json(
        folder / "source.json",
        {
            "gradient_weight": gradient_weights,
            "covariance_weights": list(guides),
            "views": list(queries),
            "view_weights": allocation.tolist(),
        },
    )
    return bank


def validation(dataset, bank, folder, device):
    cfg = CONFIG["target"]
    rows = validation_rows(
        evaluate(
            dataset,
            "dev",
            {name: row["path"] for name, row in bank.items()},
            device,
            rates=cfg["rates"],
            steps=cfg["steps"],
            replicates=cfg["seed_replicates"],
        ),
        None,
    )
    families = {record["family"] for record in bank.values()}
    chosen = {
        family: min(
            (
                row
                for row in rows
                if bank[row["candidate"]]["family"] == family
            ),
            key=ordering,
        )
        for family in sorted(families)
    }
    best = min(
        (key for key in chosen if key != "group_weight_0"),
        key=lambda key: ordering(chosen[key]),
    )
    save_json(folder / "validation.json", rows)
    guide = bank[chosen[best]["candidate"]]["guide"]
    control = min(
        (row for row in rows if row["candidate"] == guide), key=ordering
    )
    freeze(
        folder / "choices.json",
        {"family": chosen, "selected": best, "guide": control},
    )
    print(dataset, chosen, best, flush=True)


def run(dataset, device, stage):
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    folder = OUT / dataset / stage
    folder.mkdir(parents=True, exist_ok=True)
    calibrated = stage == STAGE
    if dataset == "har":
        from functools import partial

        from .har import development

        development(
            folder, device, build=partial(model_bank, calibrated=calibrated)
        )
        return
    settings = CONFIG["real"][dataset]
    layout = Layout(*settings["layout"])
    rank = settings["codes"] // layout.hidden
    path = folder / "bank.json"
    if path.exists():
        bank = json.loads(path.read_text())
    else:
        bank = model_bank(
            folder,
            *classification_source(dataset, layout, device),
            layout,
            rank,
            settings["shot"],
            calibrated=calibrated,
        )
    if not (folder / "choices.json").exists():
        validation(dataset, bank, folder, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=(*DATASETS, "har"))
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--stage", choices=(STAGE, REVISED_STAGE), default=REVISED_STAGE
    )
    args = parser.parse_args()
    run(args.dataset, args.device, args.stage)
