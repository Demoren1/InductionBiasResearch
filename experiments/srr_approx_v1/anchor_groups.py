"""Test inherited neuron identities after copying a common source first layer."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from experiments.omniglot64_v1.asrr_functional import cluster_neuron_blocks
from experiments.omniglot64_v1.source import ROOT as OMNI
from experiments.paper_structure_benchmark_v1.fitting.real_dense import run as fit_dense
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.core import input_forest
from experiments.srr_approx_v1.group_pattern import (
    ARCHIVE,
    REFERENCE,
    recover,
    source_data,
)
from experiments.srr_approx_v1.group_transfer import LOCAL, OUT
from experiments.srr_approx_v1.pattern import extract, historical_file, score


def normalize(w):
    return w / np.maximum(np.linalg.norm(w, axis=-2, keepdims=True), 1e-12)


def profiles(weights, selected, anchor):
    """Compare inherited and assigned indices from identical per-task ensembles."""
    unit = normalize(weights)
    selected = normalize(selected)
    aligned, changed, across, margin = [], [], [], []
    reference = selected[anchor]
    for t, fits in enumerate(unit):
        cross = linear_sum_assignment(-reference.T @ selected[t])[1]
        across.append(np.mean(cross != np.arange(len(cross))))
        within = []
        for fit in fits:
            similarity = selected[t].T @ fit
            order = linear_sum_assignment(-similarity)[1]
            changed.append(np.mean(order != np.arange(len(order))))
            within.append(fit[:, order])
        aligned.append(np.mean(within, axis=0)[:, cross])
        gram = reference.T @ selected[t]
        rivals = gram.copy()
        np.fill_diagonal(rivals, -np.inf)
        margin.extend((gram.diagonal() - rivals.max(axis=1)).tolist())
    return (
        {"native": unit.mean(axis=1), "aligned": np.stack(aligned)},
        {
            "within_task_hungarian_moved": float(np.mean(changed)),
            "cross_task_hungarian_moved": float(np.mean(across)),
            "anchor_row_strict_identity_fraction": float(np.mean(np.array(margin) > 0)),
        },
    )


def prepare_omni(device):
    root = LOCAL / "omni_anchor_source"
    (root / "public").mkdir(exist_ok=True, parents=True)
    (root / "source").mkdir(exist_ok=True)
    dense = OMNI / "source/dense/models.npz"
    with np.load(dense) as data:
        prior = data["parameters"]
        anchor = int(data["numerical_cross_entropy"].min(axis=0).argmin())
    others = np.delete(np.arange(len(prior)), anchor)
    save_json(root / "protocol.json", json.loads((OMNI / "protocol.json").read_text()))
    for name in ("support", "numerical"):
        with np.load(OMNI / f"public/{name}.npz") as data:
            np.savez_compressed(
                root / f"public/{name}.npz", **{k: data[k][others] for k in ("x", "y")}
            )
    save_json(
        OUT / "omniglot10_anchor_plan.json",
        {
            "anchor": anchor,
            "rule": "lowest independent source numerical CE",
            "source": str(dense),
            "source_sha256": digest(dense),
            "steps": 2500,
            "retained": 4,
            "hidden": 64,
            "codes": 4096,
            "descendants": others.tolist(),
            "indexators": ["anchor_native", "anchor_aligned", "asrr"],
            "target_fit": "same 24 starts, 500 updates, top four; eight paired 96-episode panels",
        },
    )
    fit_dense(
        root, torch.device(device), 2500, initial_hidden=prior[anchor, : (784 + 1) * 64]
    )


def omni_bank():
    plan = json.loads((OUT / "omniglot10_anchor_plan.json").read_text())
    anchor, others = plan["anchor"], plan["descendants"]
    n, d, h = len(others) + 1, 784, 64
    with np.load(plan["source"]) as store:
        selected = store["parameters"][:, : d * h].reshape(n, d, h)
    path = LOCAL / "omni_anchor_source/source/dense/models.npz"
    with np.load(path) as data:
        raw = data["ensemble_parameters"]
        weights = np.empty((n, raw.shape[1], d, h))
        weights[anchor] = selected[anchor]
        weights[others] = raw[..., : d * h].reshape(len(others), -1, d, h)
        selected[others] = data["parameters"][:, : d * h].reshape(-1, d, h)
        loss = float(data["numerical_cross_entropy"].min(axis=0).mean())
    views, diagnostic = profiles(weights, selected, anchor)
    bank = {"anchor_" + k: cluster_neuron_blocks(z, 4) for k, z in views.items()}
    np.savez_compressed(OUT / "omniglot10_anchor_indices.npz", **bank)
    save_json(
        OUT / "omniglot10_anchor_source.json",
        {
            **diagnostic,
            "anchor": anchor,
            "descendant_mean_best_ce": loss,
            "model_sha256": digest(path),
        },
    )


def mnist_bank():
    prior = Path("artifacts/asrr_anchor_init_2026-11-18")
    choices = json.loads((prior / "choices_frozen.json").read_text())["policies"][
        "mnist16"
    ]
    with np.load(prior / "frozen_indexators.npz") as data:
        bank = {
            "anchor_native": data[choices["warm"]["model"]],
            "independent_single_fit": data[choices["independent"]["model"]],
        }
    np.savez_compressed(OUT / "mnist16_anchor_indices.npz", **bank)
    save_json(
        OUT / "mnist16_anchor_plan.json",
        {
            "source": str(prior),
            "anchor": choices["warm"]["origin"],
            "input_groups": 128,
            "codes": 2048,
            "hidden": 16,
            "aligned_control": "One fit per task, all neurons in one input partition; per-task column permutations preserve every input distance.",
        },
    )


def contrast_banks():
    for setting in ("mnist16", "omniglot10"):
        if setting == "mnist16":
            prior = Path("artifacts/asrr_anchor_init_2026-11-18/source_fits/anchor_06")
            with np.load(prior / "models.npz") as data:
                z = normalize(
                    data["parameters"][:, : 784 * 16].astype(float).reshape(-1, 784, 16)
                )
            centered = input_forest(z - z.mean(axis=0), (2048,))[2048]
        else:
            plan = json.loads((OUT / "omniglot10_anchor_plan.json").read_text())
            with np.load(plan["source"]) as prior:
                z = normalize(
                    prior["parameters"][:, : 784 * 64]
                    .astype(float)
                    .reshape(-1, 784, 64)
                )
            path = LOCAL / "omni_anchor_source/source/dense/models.npz"
            with np.load(path) as fits:
                w = fits["ensemble_parameters"][..., : 784 * 64].astype(float)
                z[plan["descendants"]] = normalize(w.reshape(len(w), -1, 784, 64)).mean(
                    1
                )
            centered = cluster_neuron_blocks(z - z.mean(axis=0), 4)
        with np.load(OUT / f"{setting}_indices.npz") as original:
            base = original["asrr"]
        with np.load(OUT / f"{setting}_anchor_indices.npz") as anchor:
            native = anchor["anchor_native"]
        path = OUT / f"{setting}_contrast_indices.npz"
        np.savez_compressed(path, centered=centered, asrr=base, anchor_native=native)
        save_json(
            OUT / f"{setting}_contrast_plan.json",
            {
                "hypothesis": "Subtract the across-task mean of inherited normalized profiles before clustering",
                "index_archive_sha256": digest(path),
                "target_episodes_per_seed": 96,
                "target_seeds": list(range(481210, 481218)),
                "selection": "single fixed variant",
                "primary": "centered minus ordinary aSRR",
                "secondary": "centered minus uncentered warm",
            },
        )


def binary_train(x, y, split, spec, seed, device, initial=None):
    """Original balanced BCE, optimizer and stopping rule on unpruned Dense fits."""
    tasks, starts, d, h = len(y), 8, x.shape[1], 8
    rng = torch.Generator(device=device).manual_seed(seed)
    n = tasks * starts
    params = [
        torch.randn(n, d, h, generator=rng, device=device) * 0.1,
        torch.zeros(n, h, device=device),
        torch.randn(n, h, generator=rng, device=device) * 0.1,
        torch.zeros(n, device=device),
    ]
    if initial is not None:
        for p, value in zip(params[:2], initial, strict=True):
            p.copy_(value)
    for p in params:
        p.requires_grad_()
    opt = torch.optim.Adam(params, lr=spec["learning_rate"])
    answers = y.repeat_interleave(starts, 0)
    train, valid = split

    def loss(ids):
        w, b, readout, offset = params
        logits = ((x[ids] @ w + b[:, None]).relu() * readout[:, None]).sum(-1) + offset[
            :, None
        ]
        target = answers[:, ids]
        p = target.mean(1, keepdim=True).clamp(1e-6, 1 - 1e-6)
        balanced = 0.5 * (target / p + (1 - target) / (1 - p))
        return (
            F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            * balanced
        ).mean(1)

    best = torch.full((n,), float("inf"), device=device)
    layer = [p.detach().clone() for p in params[:2]]
    for step in range(spec["steps_per_round"] + 1):
        if step % spec["checkpoint_every"] == 0:
            with torch.no_grad():
                ce = loss(valid)
                improved = ce < best
                best[improved] = ce[improved]
                for kept, p in zip(layer, params[:2], strict=True):
                    kept[improved] = p[improved]
        if step < spec["steps_per_round"]:
            objective = loss(train).sum() + 0.5 * spec["l2"] * sum(
                p.square().sum() for p in params
            )
            opt.zero_grad(set_to_none=True)
            objective.backward()
            opt.step()
    return [p.reshape(tasks, starts, *p.shape[1:]) for p in layer], best.reshape(
        tasks, starts
    )


def pattern(device):
    binary, labels, train, _ = source_data()
    with np.load(
        io.BytesIO(historical_file("runs/codebook_20261028_v4/splits.npz"))
    ) as data:
        valid = data["query"]
    spec = json.loads(historical_file("codebook/protocol_v4.json"))["bank"]
    x = torch.as_tensor(2 * binary - 1, device=device).float()
    y = torch.as_tensor(labels, device=device).float()
    splits = (train, valid)
    tasks = torch.arange(len(labels), device=device)
    records, indices = {}, {}
    seeds = (391101, 391102, 391103)
    save_json(
        OUT / "pattern_anchor_plan.json",
        {
            "seeds": seeds,
            "starts": 8,
            "retained": 4,
            "anchor": "lowest independent source balanced validation BCE",
            "comparisons": ["independent", "warm"],
            "views": ["native", "aligned"],
            "source_fit_rule": spec,
            "task_inputs": int(len(train)),
        },
    )
    for seed in seeds:
        layers, losses = binary_train(x, y, splits, spec, seed, device)
        anchor, start = np.unravel_index(int(losses.argmin()), losses.shape)
        initial = [p[anchor, start] for p in layers]
        warm, warm_loss = binary_train(x, y, splits, spec, seed, device, initial)
        for p, value in zip(warm, initial, strict=True):
            p[anchor] = value
        warm_loss[anchor] = losses[anchor, start]
        for family, w, ce in (
            ("independent", layers[0], losses),
            ("warm", warm[0], warm_loss),
        ):
            order = ce.argsort(1)[:, :4]
            fits = w[tasks[:, None], order].cpu().numpy()
            views, diagnostic = profiles(fits, fits[:, 0], anchor)
            prefix = f"{seed}_{family}"
            records[prefix] = dict(
                **diagnostic,
                anchor=int(anchor),
                mean_retained_bce=float(ce[tasks[:, None], order].mean()),
            )
            for alignment, z in views.items():
                for method, index in extract(z, 4).items():
                    indices[f"{prefix}_{alignment}_{method}"] = index
            indices[f"{prefix}_pursuit"] = recover(
                fits, binary[train], labels[:, train], 4
            )[0]
            print(prefix, records[prefix], flush=True)
    np.savez_compressed(OUT / "pattern_anchor_indices.npz", **indices)
    with np.load(REFERENCE) as oracle:
        scores = score(indices, oracle["analytic_indices"], 4)
    save_json(
        OUT / "pattern_anchor_results.json", {"source": records, "recovery": scores}
    )


def nonlocal_controls():
    """Blind input permutations and new networks with unordered, unequal supports."""
    torch.set_num_threads(3)
    rng = np.random.default_rng(391211)
    bits, y, train, _ = source_data()
    with np.load(ARCHIVE) as archive:
        fits = archive["binary_281067_weights"]
    order = rng.permutation(bits.shape[1])
    shuffled = fits[:, :, order, :].copy()
    for matrix in shuffled.reshape(-1, *shuffled.shape[-2:]):
        matrix[:] = matrix[:, rng.permutation(matrix.shape[1])]
    recovered = recover(shuffled, bits[train][:, order], y[:, train], 4)[0]
    with np.load(REFERENCE) as store:
        truth = store["analytic_indices"]
    records = {
        "shuffled_pattern": score({"index": recovered}, truth[order], 4)["index"]
    }
    bank = {"shuffled_pattern": recovered, "shuffled_pattern_row_order": order}

    for seed in (391212, 391213, 391214):
        rng = np.random.default_rng(seed)
        hidden, dimensions, tasks, budget = 8, 13, 12, 4
        sizes = rng.permutation([3, 3, 3, 4, 4, 4, 5, 5])
        index = np.zeros((dimensions, hidden), dtype=np.int16)
        for h, size in enumerate(sizes):
            support = rng.choice(dimensions, size, replace=False)
            labels = np.resize(rng.permutation(np.arange(1, budget + 1)), size)
            index[support, h] = labels * rng.choice([-1, 1], size)
        population = (np.arange(2**dimensions)[:, None] >> np.arange(dimensions)) & 1
        codes = (np.arange(2**budget)[:, None] >> np.arange(budget)) & 1
        codes = (2 * codes - 1)[rng.choice(len(codes), tasks, replace=False)]
        z = np.sign(index) * np.pad(codes, ((0, 0), (1, 0)))[:, abs(index)]
        labels = np.any((2 * population - 1) @ z == sizes, axis=-1)
        ids = rng.permutation(len(population))
        train, valid = ids[:768], ids[768:1024]
        spec = json.loads(historical_file("codebook/protocol_v4.json"))["bank"]
        x = torch.as_tensor(2 * population - 1).float()
        y = torch.as_tensor(labels).float()
        layer, ce = binary_train(x, y, (train, valid), spec, seed, "cpu")
        kept = layer[0][torch.arange(tasks)[:, None], ce.argsort(1)[:, :4]].numpy()
        guessed = recover(kept, population[train], labels[:, train], budget)[0]
        bank[f"nonlocal_{seed}"] = guessed
        records[str(seed)] = score({"index": guessed}, index, budget)["index"]
        records[str(seed)]["active_per_neuron"] = sizes.tolist()
        records[str(seed)]["signed_entry_errors_out_of"] = int(index.size)
        print(seed, records[str(seed)]["signed_entry_errors"], flush=True)
    np.savez_compressed(OUT / "nonlocal_indices.npz", **bank)
    save_json(OUT / "nonlocal_results.json", records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "pattern",
            "fit_omni",
            "omni_bank",
            "mnist_bank",
            "nonlocal",
            "contrast",
        ),
    )
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    if args.phase == "pattern":
        pattern(args.device)
    elif args.phase == "fit_omni":
        prepare_omni(args.device)
    elif args.phase == "omni_bank":
        omni_bank()
    elif args.phase == "nonlocal":
        nonlocal_controls()
    elif args.phase == "contrast":
        contrast_banks()
    else:
        mnist_bank()
