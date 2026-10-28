"""Evolve image feature indexators; train only their task codebooks inside."""

import argparse
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from distractor_data import image_table, layout, observations
from feature_bank import predict
from torch.nn import functional as F
from torchvision.datasets import MNIST, FashionMNIST

ROOT = Path("experiments/structure_search_v2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--pool-from")
    args = parser.parse_args()
    spec = json.loads((ROOT / "distractor_protocol.json").read_text())
    if args.balanced:
        spec.update(
            sampling="balanced_without_replacement",
            pool_size=5000,
            source_train=[0, 2000],
            source_query=[2000, 4000],
            calibration_support=[4000, 4500],
            calibration_query=[4500, 5000],
            test_per_class=512,
        )
    if args.pool_from:
        spec["proposal_pool"] = args.pool_from
        spec["generations"] = 4
    out = ROOT / "runs" / ("distractor_" + args.dataset + "_" + args.version)
    out.mkdir(parents=True, exist_ok=False)
    for file in [
        "distractor_bank.py",
        "distractor_data.py",
        "distractor_protocol.json",
        "feature_bank.py",
    ]:
        shutil.copy2(ROOT / file, out / file)
    (out / "distractor_protocol.json").write_text(json.dumps(spec, indent=2))
    torch.set_num_threads(4)
    dataset = {"MNIST": MNIST, "FashionMNIST": FashionMNIST}[args.dataset](
        ROOT / "datasets", train=True
    )
    pixels = image_table(dataset)
    rng = np.random.default_rng(spec["seed"])
    pools = {}
    for cls in spec["source_classes"]:
        ids = rng.permutation(np.flatnonzero(dataset.targets.numpy() == cls))
        pools[cls] = ids[: spec.get("pool_size", 1000)]
    pairs = [
        p
        for p in itertools.combinations(spec["source_classes"], 2)
        if list(p) not in spec["reserved_pairs"]
    ]
    permutation = layout(spec["seed"] + 1, spec["blocks"])
    xs, qs, ids, qids = [], [], [], []
    for task, pair in enumerate(pairs):
        if args.balanced:
            train_pool = np.concatenate([pools[c][:2000] for c in pair])
            query_pool = np.concatenate([pools[c][2000:4000] for c in pair])
            support = np.concatenate(
                [rng.choice(pools[c][:2000], 24, replace=False) for c in pair]
            )
            query = np.concatenate(
                [rng.choice(pools[c][2000:4000], 128, replace=False) for c in pair]
            )
        else:
            support = np.concatenate([pools[c][:24] for c in pair])
            query = np.concatenate([pools[c][500:628] for c in pair])
            train_pool, query_pool = support, query
        labels = dataset.targets.numpy() if args.balanced else None
        x, sample_ids = observations(
            pixels,
            support,
            train_pool,
            permutation,
            spec["seed"] + 100 + task,
            donor_labels=labels,
        )
        q, query_ids = observations(
            pixels,
            query,
            query_pool,
            permutation,
            spec["seed"] + 200 + task,
            donor_labels=labels,
        )
        xs.append(x)
        qs.append(q)
        ids.append(sample_ids)
        qids.append(query_ids)
    # The search below receives only observations. The layout is a generator
    # artifact stored separately for explicit oracle checks, not an input prior.
    np.savez_compressed(
        out / "source_splits.npz",
        support=np.array(ids),
        query=np.array(qids),
        **{f"pool_{c}": v for c, v in pools.items()},
    )
    np.savez_compressed(
        out / "hidden_layout.npz",
        permutation=permutation,
        informative=np.flatnonzero(permutation < 196),
    )
    device = torch.device("cuda")
    x, qx = torch.stack(xs).to(device), torch.stack(qs).to(device)
    tasks, population, d, h = (
        len(pairs),
        spec["population"],
        len(permutation),
        spec["hidden"],
    )
    y = torch.tensor([[0.0] * 24 + [1.0] * 24], device=device).expand(tasks, -1)
    qy = torch.tensor([[0.0] * 128 + [1.0] * 128], device=device).expand(tasks, -1)
    np.savez_compressed(
        out / "source_observations.npz",
        x=x.cpu().numpy(),
        query_x=qx.cpu().numpy(),
        y=y.cpu().numpy(),
        query_y=qy.cpu().numpy(),
    )
    torch.manual_seed(spec["seed"] + 2)
    masks = torch.zeros(tasks, population, d, device=device)
    pool = (
        torch.tensor(np.load(args.pool_from)["pool"], device=device)
        if args.pool_from
        else torch.arange(d, device=device)
    )
    scores = torch.full_like(masks, -torch.inf)
    scores[:, :, pool] = torch.rand(tasks, population, len(pool), device=device)
    masks.scatter_(-1, scores.topk(spec["slots"], -1).indices, 1.0)
    parameters = [
        torch.randn(tasks, population, d, h, device=device) / np.sqrt(spec["slots"]),
        torch.zeros(tasks, population, h, device=device),
        torch.randn(tasks, population, h, device=device) / np.sqrt(h),
        torch.zeros(tasks, population, device=device),
    ]
    role = torch.repeat_interleave(
        torch.arange(3, device=device), torch.tensor(spec["roles"], device=device)
    )
    saved = {
        key: []
        for key in [
            "tokens",
            "indices",
            "codebook",
            "bias",
            "readout",
            "offset",
            "quality",
            "accuracy",
            "task",
            "role",
            "generation",
        ]
    }
    history = []
    started = time.time()
    for generation in range(spec["generations"]):
        parameters = [p.detach().clone().requires_grad_() for p in parameters]
        optimizer = torch.optim.Adam(parameters, lr=spec["learning_rate"])
        best = [p.detach().clone() for p in parameters]
        best_loss = torch.full((tasks, population), torch.inf, device=device)
        for step in range(spec["inner_steps"] + 1):
            if step % 32 == 0:
                with torch.no_grad():
                    logits = predict(qx, parameters, masks)
                    loss = F.binary_cross_entropy_with_logits(
                        logits, qy[:, None].expand_as(logits), reduction="none"
                    ).mean(-1)
                    better = loss < best_loss
                    best_loss[better] = loss[better]
                    for stored, current in zip(best, parameters):
                        stored[better] = current[better]
            if step == spec["inner_steps"]:
                break
            logits = predict(x, parameters, masks)
            loss = (
                F.binary_cross_entropy_with_logits(
                    logits, y[:, None].expand_as(logits), reduction="none"
                )
                .mean(-1)
                .sum()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        parameters = best
        w = parameters[0].clone().requires_grad_()
        logits = predict(x, [w, *parameters[1:]], masks)
        loss = (
            F.binary_cross_entropy_with_logits(
                logits, y[:, None].expand_as(logits), reduction="none"
            )
            .mean(-1)
            .sum()
        )
        gradient = torch.autograd.grad(loss, w)[0]
        with torch.no_grad():
            effective = parameters[0] * masks[..., None]
            activated = (
                torch.einsum("tbd,tmdh->tmbh", x, effective) + parameters[1][:, :, None]
                > 0
            ).float()
            response = (
                (effective * parameters[2][:, :, None]).abs()
                * torch.einsum("tbd,tmbh->tmdh", x.abs(), activated)
                / x.shape[1]
            )
            tokens = torch.stack(
                [
                    effective.square().mean(-1).sqrt(),
                    gradient.square().mean(-1).sqrt(),
                    response.sum(-1),
                ],
                -1,
            )
            tokens /= tokens.amax(2, keepdim=True).clamp_min(1e-8)
            indices = masks.topk(spec["slots"], -1).indices.sort(-1).values
            compact = effective.gather(2, indices[..., None].expand(-1, -1, -1, h))
            logits = predict(qx, parameters, masks)
            accuracy = ((logits >= 0) == qy[:, None].bool()).float().mean(-1)
        for key, value in [
            ("tokens", tokens),
            ("indices", indices),
            ("codebook", compact),
            ("bias", parameters[1]),
            ("readout", parameters[2]),
            ("offset", parameters[3]),
            ("quality", best_loss),
            ("accuracy", accuracy),
        ]:
            saved[key].append(value.detach().flatten(0, 1).cpu().numpy())
        saved["task"].append(np.repeat(np.arange(tasks), population))
        saved["role"].append(np.tile(role.cpu().numpy(), tasks))
        saved["generation"].append(np.full(tasks * population, generation))
        record = {
            "generation": generation,
            "mean_source_bce": float(best_loss.mean()),
            "best_source_bce": float(best_loss.min(1).values.mean()),
            "mean_accuracy": float(accuracy.mean()),
        }
        history.append(record)
        print(args.dataset, record, flush=True)
        if generation + 1 == spec["generations"]:
            break
        parents = torch.empty(tasks, population, dtype=torch.long, device=device)
        elites = torch.zeros(tasks, population, dtype=torch.bool, device=device)
        for group, count in enumerate(spec["elite_per_role"]):
            positions = (role == group).nonzero().flatten()
            choices = positions[best_loss[:, positions].argsort(1)[:, :count]]
            parents[:, positions] = choices.gather(
                1, torch.randint(count, (tasks, len(positions)), device=device)
            )
            parents[:, positions[:count]] = choices
            elites[:, positions[:count]] = True
        rows = torch.arange(tasks, device=device)[:, None]
        masks = masks[rows, parents].clone()
        parameters = [p[rows, parents].detach().clone() for p in parameters]
        for task in range(tasks):
            for candidate in range(population):
                if elites[task, candidate]:
                    continue
                on = (masks[task, candidate] > 0).nonzero().flatten()
                off = pool[masks[task, candidate, pool] == 0]
                masks[
                    task,
                    candidate,
                    on[torch.randperm(len(on), device=device)[: spec["mutations"]]],
                ] = 0
                masks[
                    task,
                    candidate,
                    off[torch.randperm(len(off), device=device)[: spec["mutations"]]],
                ] = 1
    raw = {key: np.concatenate(value) for key, value in saved.items()}
    selected = []
    for task in range(tasks):
        for group in range(3):
            ids = np.flatnonzero((raw["task"] == task) & (raw["role"] == group))
            ids = ids[np.argsort(raw["quality"][ids])]
            selected.extend(ids[: max(1, len(ids) // 4)])
    selected = np.array(selected)
    np.savez_compressed(
        out / "good_bank.npz", **{key: v[selected] for key, v in raw.items()}
    )
    np.savez_compressed(
        out / "raw_mask_scores.npz",
        **{
            key: raw[key]
            for key in ["indices", "quality", "task", "role", "generation"]
        },
    )
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "bank_complete.json").write_text(
        json.dumps(
            {
                "seconds": time.time() - started,
                "dataset": args.dataset,
                "pairs": pairs,
                "dimension": d,
                "selected_models": len(selected),
                "target_classes_used": False,
                "layout_used_by_search": False,
                "nf_features": "hidden-permutation-invariant row moments of W, loss gradient and functional response; compact masks have identical hidden-column support",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
