"""Apply joint compatibility to the existing signed, zero-allowing Pattern bank."""

import argparse
from itertools import product

import numpy as np
import torch

from experiments.paper_structure_benchmark_v1.reporting.evaluate import align_index
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.compatible import OUTPUT, common_frontier, common_index
from experiments.srr_approx_v1.group_pattern import (
    ARCHIVE,
    REFERENCE,
    recover,
    source_data,
)
from experiments.srr_approx_v1.pattern import SEEDS, extract
from experiments.srr_approx_v1.anchor_groups import profiles
from experiments.srr_approx_v1.core import canonical


FOLDER = OUTPUT / "pattern"


def prepare():
    FOLDER.mkdir(parents=True, exist_ok=True)
    bits, y, train, _ = source_data()
    rng = np.random.default_rng(691113)
    subsets = [
        np.arange(len(y)),
        *(rng.choice(len(y), k, replace=False) for k in (2, 3, 4, 5) for _ in range(2)),
    ]
    candidates = []
    with np.load(ARCHIVE) as data:
        for si, seed in enumerate(SEEDS):
            w = data[f"binary_{seed}_weights"]
            loss = data[f"binary_{seed}_losses"]
            for subset in subsets if si == 0 else subsets[:1]:
                index = recover(w[subset], bits[train], y[subset][:, train], 4)[0]
                candidates.append(index)
            selected = w[np.arange(len(w)), loss.argmin(axis=1)]
            z = profiles(w, selected, int(loss.min(axis=1).argmin()))[0]["aligned"]
            candidates.extend(extract(z, 4).values())
    anchor = candidates[0]
    banks = {}
    for i, index in enumerate(candidates):
        # Use a source proposal as the common coordinate reference. No planted
        # matrix enters inference. Resolve code gauges and one hidden permutation.
        matched = canonical(align_index(index, anchor, 4)[0])
        banks.setdefault(matched.tobytes(), (f"proposal_{i}", matched))
    path = FOLDER / "bank.npz"
    np.savez_compressed(path, **dict(banks.values()))
    save_json(
        FOLDER / "plan.json",
        {
            "source_ensemble_sha256": digest(ARCHIVE),
            "bank_sha256": digest(path),
            "train": len(train),
            "structural_holdout": 384,
            "refit": "32 orthant/quantile starts; learn four codes, biases and head on training BCE",
            "tolerance_binary_brier": 0.01,
            "coverage": 1,
            "candidates": len(banks),
            "source_task_subsets": [r.tolist() for r in subsets],
        },
    )


def refit(index, x, y, train, validation, device):
    rng = torch.Generator(device=device).manual_seed(691119)
    ids = torch.as_tensor(index, device=device)
    m = int(abs(index).max())
    signs = torch.tensor(list(product((-1.0, 1.0), repeat=m)), device=device)
    c = signs.repeat(2, 1)[:, None].expand(-1, len(y), -1).clone()

    def decode(codes):
        return codes[..., (ids.abs() - 1).clamp_min(0).long()] * ids.sign()

    pre = x[train] @ decode(c)
    half = len(signs)
    b = torch.cat(
        [
            -torch.quantile(v, q, dim=2)
            for v, q in zip((pre[:half], pre[half:]), (0.65, 0.90), strict=True)
        ]
    )
    head = torch.randn(b.shape, generator=rng, device=device) * 0.1
    bias = torch.zeros_like(head[..., 0])
    params = [v.requires_grad_() for v in (c, b, head, bias)]
    opt = torch.optim.Adam(params, lr=0.05)

    def logits(ids):
        return ((x[ids] @ decode(c) + b[..., None, :]).relu() * head[..., None, :]).sum(
            -1
        ) + bias[..., None]

    for _ in range(300):
        p = logits(train)
        objective = (torch.nn.functional.softplus(p) - y[:, train] * p).mean(-1).sum()
        opt.zero_grad(set_to_none=True)
        objective.backward()
        opt.step()
    with torch.no_grad():
        p = logits(train)
        loss = (torch.nn.functional.softplus(p) - y[:, train] * p).mean(-1)
        best = loss.argmin(0)
        v = logits(validation)[best, torch.arange(len(y), device=device)]
        return (v.sigmoid() - y[:, validation]).square().mean(-1).cpu().numpy()


def run(device):
    bits, raw, train, validation = source_data()
    x = torch.as_tensor(2 * bits - 1, device=device).float()
    y = torch.as_tensor(raw, device=device).float()
    with np.load(FOLDER / "bank.npz") as data:
        names, arrays = data.files, list(dict(data).values())
    risk = np.stack(
        [refit(idx, x, y, train, validation, device) for idx in arrays], axis=1
    )
    good = risk <= risk.min(axis=1, keepdims=True) + 0.01
    frontier = common_frontier(
        [[idx for idx, keep in zip(arrays, row, strict=True) if keep] for row in good]
    )
    candidates = [
        i
        for i, idx in enumerate(arrays)
        if any(np.array_equal(common_index(c, idx), c) for c in frontier)
    ]
    selected = min(candidates, key=lambda j: (-good[:, j].sum(), risk[:, j].mean(), j))
    np.savez_compressed(
        FOLDER / "frozen.npz",
        fixed=arrays[selected],
        **{f"partial_{i}": v for i, v in enumerate(frontier)},
    )
    np.save(FOLDER / "held_out_brier.npy", risk)
    with np.load(REFERENCE) as data:
        truth = data["analytic_indices"]
    errors = align_index(arrays[selected], truth, 4)[1]["signed_entry_errors"]
    save_json(
        FOLDER / "results.json",
        {
            "proposals": len(arrays),
            "maximal_count": len(frontier),
            "selected": names[selected],
            "exact_entry_errors": errors,
            "sources_with_selected_witness": int(good[:, selected].sum()),
            "good_per_task": good.sum(axis=1).tolist(),
            "maximal_code_counts": [int(abs(c).max()) for c in frontier],
        },
    )
    print("Pattern", len(arrays), len(frontier), errors, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "fit"))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.phase == "prepare":
        prepare()
    else:
        run(args.device)
