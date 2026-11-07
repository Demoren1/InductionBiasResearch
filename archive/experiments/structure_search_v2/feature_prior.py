"""Align good source networks and learn a compact feature-indexator prior."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from nf_prior import NFVAE, normalized_features
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F


def assignment(logits, straight_through=False):
    rows, columns = logits.shape[-2:]
    flat = logits.reshape(-1, rows, columns)
    indices = []
    for scores in flat.detach().cpu().numpy():
        old, new = linear_sum_assignment(-scores)
        labels = np.empty(rows, dtype=np.int64)
        labels[old] = new
        indices.append(labels)
    indices = torch.tensor(np.array(indices), device=logits.device)
    hard = F.one_hot(indices, columns).to(logits.dtype)
    if straight_through:
        soft = flat.softmax(-1)
        hard = hard + soft - soft.detach()
    return hard.reshape_as(logits)


def prepare(data):
    masks = data["mask"]
    weights = data["weights"] * masks[:, :, None]
    gradients = data["gradient"]
    functional = data["functional"]
    tasks = int(data["task"].max()) + 1
    orders = np.empty((len(masks), weights.shape[-1]), dtype=np.int64)
    for task in range(tasks):
        train = np.flatnonzero((data["task"] == task) & (data["role"] == 0))
        reference_id = train[data["quality"][train].argmin()]
        reference = functional[reference_id]
        reference = reference / np.linalg.norm(reference, axis=0, keepdims=True).clip(
            1e-8
        )
        for index in np.flatnonzero(data["task"] == task):
            profile = functional[index]
            profile = profile / np.linalg.norm(profile, axis=0, keepdims=True).clip(
                1e-8
            )
            old, new = linear_sum_assignment(-(profile.T @ reference))
            order = np.empty(weights.shape[-1], dtype=np.int64)
            order[new] = old
            orders[index] = order

    def align(values):
        return np.stack([value[:, order] for value, order in zip(values, orders)])

    return align(weights), align(gradients), align(functional), orders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], required=True)
    args = parser.parse_args()
    root = Path("experiments/structure_search_v2")
    out = root / "runs" / ("feature_bank_" + args.dataset + "_v1")
    shutil.copy2(__file__, out / "prior_source.py")
    shutil.copy2(root / "nf_prior.py", out / "nf_prior_source.py")
    data = np.load(out / "good_bank.npz")
    weights, gradient, response, orders = prepare(data)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    features = normalized_features(
        torch.tensor(weights, device=device),
        torch.tensor(gradient, device=device),
        torch.tensor(response, device=device),
    )
    targets = torch.tensor(
        np.stack([np.flatnonzero(mask) for mask in data["mask"]]), device=device
    )
    task_ids, role = data["task"], data["role"]
    tasks, inputs, slots = int(task_ids.max()) + 1, weights.shape[1], targets.shape[1]
    train = [
        torch.tensor(np.flatnonzero((task_ids == task) & (role == 0)), device=device)
        for task in range(tasks)
    ]
    val = torch.stack(
        [
            torch.tensor(
                np.flatnonzero((task_ids == task) & (role == 1)), device=device
            )
            for task in range(tasks)
        ]
    )
    torch.manual_seed(281220)
    model = NFVAE(tasks, inputs, slots * inputs).to(device)
    torch.save(model.state_dict(), out / "untrained_prior.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best, history = float("inf"), []
    for epoch in range(301):
        model.eval()
        with torch.no_grad():
            scores, _, _ = model.encode(features[val], sample=False)
            validation = F.cross_entropy(
                scores.reshape(-1, inputs), targets[val].flatten()
            )
            if float(validation) < best:
                best = float(validation)
                torch.save(model.state_dict(), out / "prior.pt")
        if epoch == 300:
            break
        model.train()
        for _ in range(2):
            selected = torch.stack(
                [pool[torch.randint(len(pool), (16,), device=device)] for pool in train]
            )
            scores, mu, logvar = model.encode(features[selected])
            ce = F.cross_entropy(
                scores.reshape(-1, inputs), targets[selected].flatten()
            )
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
            loss = ce + 0.1 * min(1, (epoch + 1) / 50) * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "validation_ce": float(validation),
                "ce": float(ce.detach()),
                "kl": float(kl.detach()),
            }
        )
        if epoch % 25 == 0:
            print(args.dataset, "PRIOR", history[-1], flush=True)
    np.savez_compressed(
        out / "aligned_bank.npz",
        features=features.cpu().numpy(),
        indices=targets.cpu().numpy(),
        orders=orders,
        task=task_ids,
        role=role,
        quality=data["quality"],
    )
    (out / "prior_history.json").write_text(json.dumps(history))
    (out / "prior_complete.json").write_text(
        json.dumps(
            {
                "tasks": tasks,
                "inputs": inputs,
                "slots": slots,
                "latent": 16,
                "validation_reconstruction_ce": best,
                "hidden_alignment": "source-only Hungarian on functional profiles",
                "codebook_nonzero_values": slots * weights.shape[-1],
                "mask_family": "ordered selection of 64 distinct input pixels; compact codebook C[64,16]",
                "target_classes_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
