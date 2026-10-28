"""Collect quality-ranked feature-mask solutions on known image datasets."""

import argparse
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import MNIST, FashionMNIST


def predict(x, parameters, masks):
    weight, bias, readout, offset = parameters
    hidden = (
        torch.einsum("tbd,tmdh->tmbh", x, weight * masks[..., None]) + bias[:, :, None]
    ).relu()
    return (hidden * readout[:, :, None]).sum(-1) + offset[:, :, None]


def source_rows(dataset, spec):
    generator = torch.Generator().manual_seed(spec["seed"])
    pools, queries = {}, {}
    for cls in spec["source_classes"]:
        ids = (dataset.targets == cls).nonzero().flatten()
        ids = ids[torch.randperm(len(ids), generator=generator)]
        pools[cls] = ids[: spec["source_pool_per_class"]]
        queries[cls] = ids[
            spec["source_pool_per_class"] : spec["source_pool_per_class"]
            + spec["source_query_pool_per_class"]
        ]
    pairs = [
        pair
        for pair in itertools.combinations(spec["source_classes"], 2)
        if list(pair) not in spec["reserved_pairs"]
    ]
    support_ids = torch.stack(
        [
            torch.cat(
                [
                    pools[a][: spec["source_support_per_class"]],
                    pools[b][: spec["source_support_per_class"]],
                ]
            )
            for a, b in pairs
        ]
    )
    query_ids = torch.stack(
        [
            torch.cat(
                [
                    queries[a][: spec["source_query_per_class"]],
                    queries[b][: spec["source_query_per_class"]],
                ]
            )
            for a, b in pairs
        ]
    )
    support_y = torch.tensor(
        [
            [0.0] * spec["source_support_per_class"]
            + [1.0] * spec["source_support_per_class"]
        ]
    ).expand(len(pairs), -1)
    query_y = torch.tensor(
        [
            [0.0] * spec["source_query_per_class"]
            + [1.0] * spec["source_query_per_class"]
        ]
    ).expand(len(pairs), -1)
    return pairs, support_ids, query_ids, support_y, query_y, pools, queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], required=True)
    args = parser.parse_args()
    root = Path("experiments/structure_search_v2")
    spec = json.loads((root / "feature_protocol.json").read_text())
    out = root / "runs" / ("feature_bank_" + args.dataset + "_v1")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / "source.py")
    shutil.copy2(root / "feature_protocol.json", out / "protocol.json")
    dataset = {"MNIST": MNIST, "FashionMNIST": FashionMNIST}[args.dataset](
        root / "datasets", train=True
    )
    pairs, ids, qids, sy, qy, pools, queries = source_rows(dataset, spec)
    np.savez_compressed(
        out / "source_splits.npz",
        support=ids.numpy(),
        query=qids.numpy(),
        **{f"pool_{cls}": value.numpy() for cls, value in pools.items()},
        **{f"query_pool_{cls}": value.numpy() for cls, value in queries.items()},
    )
    torch.set_num_threads(4)
    device = torch.device("cuda")
    x = F.avg_pool2d(dataset.data[ids.flatten(), None].to(device).float() / 255, 2)
    x = x.flatten(1).reshape(len(pairs), len(ids[0]), 196)
    qx = F.avg_pool2d(dataset.data[qids.flatten(), None].to(device).float() / 255, 2)
    qx = qx.flatten(1).reshape(len(pairs), len(qids[0]), 196)
    sy, qy = sy.to(device), qy.to(device)
    tasks, population, d, h = len(pairs), spec["population"], 196, spec["hidden"]
    torch.manual_seed(spec["seed"] + 1)
    masks = torch.zeros(tasks, population, d, device=device)
    selected = (
        torch.rand(tasks, population, d, device=device)
        .topk(spec["active_features"], -1)
        .indices
    )
    masks.scatter_(-1, selected, 1.0)
    parameters = [
        torch.randn(tasks, population, d, h, device=device) / np.sqrt(d),
        torch.zeros(tasks, population, h, device=device),
        torch.randn(tasks, population, h, device=device) / np.sqrt(h),
        torch.zeros(tasks, population, device=device),
    ]
    history = []
    arrays = {
        key: []
        for key in [
            "weights",
            "bias",
            "readout",
            "offset",
            "mask",
            "gradient",
            "functional",
            "quality",
            "accuracy",
            "task",
            "role",
            "generation",
        ]
    }
    role = torch.repeat_interleave(
        torch.arange(3, device=device),
        torch.tensor(spec["source_roles"], device=device),
    )
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
                    for stored, value in zip(best, parameters):
                        stored[better] = value[better]
            if step == spec["inner_steps"]:
                break
            logits = predict(x, parameters, masks)
            loss = (
                F.binary_cross_entropy_with_logits(
                    logits, sy[:, None].expand_as(logits), reduction="none"
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
                logits, sy[:, None].expand_as(logits), reduction="none"
            )
            .mean(-1)
            .sum()
        )
        gradient = torch.autograd.grad(loss, w)[0]
        with torch.no_grad():
            effective = parameters[0] * masks[..., None]
            active = (
                torch.einsum("tbd,tmdh->tmbh", x, effective) + parameters[1][:, :, None]
                > 0
            ).float()
            response = torch.einsum("tbd,tmbh->tmdh", x.abs(), active) / x.shape[1]
            functional = (effective * parameters[2][:, :, None]).abs() * response
            logits = predict(qx, parameters, masks)
            accuracy = ((logits >= 0) == qy[:, None].bool()).float().mean(-1)
        for key, value in zip(["weights", "bias", "readout", "offset"], parameters):
            arrays[key].append(value.detach().flatten(0, 1).cpu().numpy())
        for key, value in [
            ("mask", masks),
            ("gradient", gradient),
            ("functional", functional),
            ("quality", best_loss),
            ("accuracy", accuracy),
        ]:
            arrays[key].append(value.detach().flatten(0, 1).cpu().numpy())
        arrays["task"].append(np.repeat(np.arange(tasks), population))
        arrays["role"].append(np.tile(role.cpu().numpy(), tasks))
        arrays["generation"].append(np.full(tasks * population, generation))
        record = {
            "generation": generation,
            "best_source_bce": float(best_loss.min(1).values.mean()),
            "mean_source_bce": float(best_loss.mean()),
            "mean_source_accuracy": float(accuracy.mean()),
        }
        history.append(record)
        print(args.dataset, record, flush=True)
        if generation + 1 == spec["generations"]:
            break
        parent_ids = torch.empty(tasks, population, device=device, dtype=torch.long)
        elites = torch.zeros(tasks, population, device=device, dtype=torch.bool)
        for group, count in enumerate(spec["elite_per_role"]):
            positions = (role == group).nonzero().flatten()
            choices = best_loss[:, positions].argsort(1)[:, :count]
            choices = positions[choices]
            draw = torch.randint(count, (tasks, len(positions)), device=device)
            parent_ids[:, positions] = choices.gather(1, draw)
            parent_ids[:, positions[:count]] = choices
            elites[:, positions[:count]] = True
        rows = torch.arange(tasks, device=device)[:, None]
        masks = masks[rows, parent_ids].clone()
        parameters = [p[rows, parent_ids].detach().clone() for p in parameters]
        for task in range(tasks):
            for candidate in range(population):
                if elites[task, candidate]:
                    continue
                current = masks[task, candidate]
                off = (current == 0).nonzero().flatten()
                on = (current > 0).nonzero().flatten()
                count = spec["mutations"]
                current[on[torch.randperm(len(on), device=device)[:count]]] = 0
                current[off[torch.randperm(len(off), device=device)[:count]]] = 1
    raw = {key: np.concatenate(value) for key, value in arrays.items()}
    np.savez_compressed(out / "raw_bank.npz", **raw)
    selected = []
    for task in range(tasks):
        for group in range(3):
            eligible = np.flatnonzero((raw["task"] == task) & (raw["role"] == group))
            ordered = eligible[np.argsort(raw["quality"][eligible])]
            selected.extend(ordered[: max(1, len(ordered) // 4)])
    selected = np.array(selected)
    bank = {key: value[selected] for key, value in raw.items()}
    np.savez_compressed(out / "good_bank.npz", **bank)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "complete.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "pairs": pairs,
                "raw_models": len(raw["mask"]),
                "good_models": len(selected),
                "seconds": time.time() - started,
                "target_classes_used": False,
                "official_test_used": False,
                "alignment": "feature masks are identical across hidden columns; NF input columns will be aligned before prior fitting",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
