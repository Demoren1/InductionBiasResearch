"""Explicit stages of the source-only calibrated real-image AR experiment."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.real_image_ar.common import fit_masks, images, load_plan, prepare


def audit_bank(plan, name, device):
    root = Path(plan["run"]) / name
    out = root / "bank_audit"
    if (out / "complete.json").exists():
        return
    out.mkdir(exist_ok=True)
    spec = plan["bank_audit"]
    bank = dict(np.load(root / "original_bank.npz"))
    data = dict(np.load(root / "source.npz"))
    configs = [
        {
            "optimizer": "adam",
            "lr": lr,
            "weight_decay": 0.001,
            "momentum": 0.0,
            "initial_scale": 1.0,
        }
        for lr in spec["learning_rates"]
    ]
    quality = np.zeros(len(bank["mask"]))
    accuracy = np.zeros_like(quality)
    for start in range(0, len(quality), spec["chunk_size"]):
        stop = min(start + spec["chunk_size"], len(quality))
        path = out / f"chunk_{start}"
        if path.with_suffix(".npz").exists():
            arrays = dict(np.load(path.with_suffix(".npz")))
        else:
            masks = np.repeat(bank["mask"][start:stop], spec["restarts"], axis=0)
            arrays = fit_masks(
                plan,
                data,
                masks,
                "bank_audit",
                configs,
                spec["restarts"],
                path,
                device,
                case_ids=bank["task"][start:stop],
                steps=spec["steps"],
            )
        for local in range(stop - start):
            ids = np.flatnonzero(arrays["task_ids"] == local)
            best = ids[arrays["query_loss"][ids].argmin()]
            quality[start + local] = arrays["query_loss"][best]
            accuracy[start + local] = arrays["query_accuracy"][best]
    kept = []
    for task, role in itertools.product(np.unique(bank["task"]), range(3)):
        ids = np.flatnonzero((bank["task"] == task) & (bank["role"] == role))
        count = max(1, int(np.ceil(len(ids) * spec["keep_fraction"])))
        kept.extend(ids[np.argsort(quality[ids])[:count]])
    kept = np.asarray(kept)
    np.savez_compressed(
        root / "bank.npz",
        mask=bank["mask"][kept].astype(np.uint8),
        task=bank["task"][kept],
        role=bank["role"][kept],
        original_ids=kept,
        query_loss=quality[kept],
        query_accuracy=accuracy[kept],
    )
    result = {
        "original_rows": len(quality),
        "kept_rows": len(kept),
        "role_counts": [int((bank["role"][kept] == r).sum()) for r in range(3)],
        "fresh_query_accuracy_before": float(accuracy.mean()),
        "fresh_query_accuracy_after": float(accuracy[kept].mean()),
        "quality_rule": "Best 75 percent within each source task and lineage role; finite 4-start numeric audit, not a representability certificate.",
    }
    (out / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print("BANK", name, result, flush=True)


def pretrain(plan, name, device):
    root = Path(plan["run"]) / name
    if (root / "pretrain.json").exists():
        return
    values, labels = images(plan, name)
    ids = np.load(root / "pretrain_data.npz")
    train_x = torch.tensor(values[ids["train_ids"]], device=device)
    train_y = torch.tensor(labels[ids["train_ids"]], device=device)
    val_x = torch.tensor(values[ids["validation_ids"]], device=device)
    val_y = torch.tensor(labels[ids["validation_ids"]], device=device)
    configs = list(itertools.product([0.001, 0.003, 0.01], plan["weight_decay"]))
    options = [config for config in configs for _ in range(2)]
    count, hidden = len(options), plan["hidden"]
    torch.manual_seed(plan["seed"] + 700)
    parameters = [
        torch.randn(count, 196, hidden, device=device) * np.sqrt(2 / 196),
        torch.zeros(count, hidden, device=device),
        torch.randn(count, hidden, 6, device=device) / np.sqrt(hidden),
        torch.zeros(count, 6, device=device),
    ]
    for parameter in parameters:
        parameter.requires_grad_()
    optimizer = torch.optim.Adam(parameters, lr=1.0)
    rates = torch.tensor([row[0] for row in options], device=device)
    decays = torch.tensor([row[1] for row in options], device=device)
    best_loss = torch.full((count,), float("inf"), device=device)
    best = [value.detach().clone() for value in parameters]
    best_step = torch.zeros(count, dtype=torch.long, device=device)
    history = []

    def logits(x):
        hidden_values = (
            torch.einsum("bd,ndh->nbh", x, parameters[0]) + parameters[1][:, None]
        ).relu()
        return torch.bmm(hidden_values, parameters[2]) + parameters[3][:, None]

    for step in range(4097):
        if step % 64 == 0:
            with torch.no_grad():
                output = logits(val_x)
                validation = (
                    F.cross_entropy(
                        output.flatten(0, 1), val_y.repeat(count), reduction="none"
                    )
                    .reshape(count, -1)
                    .mean(1)
                )
                better = validation < best_loss
                best_loss[better] = validation[better]
                best_step[better] = step
                for stored, value in zip(best, parameters):
                    stored[better] = value[better]
                train_output = logits(train_x)
                training = (
                    F.cross_entropy(
                        train_output.flatten(0, 1),
                        train_y.repeat(count),
                        reduction="none",
                    )
                    .reshape(count, -1)
                    .mean(1)
                )
                train_recall, val_recall = [], []
                for cls in range(6):
                    train_recall.append(
                        (train_output[:, train_y == cls].argmax(-1) == cls)
                        .float()
                        .mean(-1)
                    )
                    val_recall.append(
                        (output[:, val_y == cls].argmax(-1) == cls).float().mean(-1)
                    )
                history.append(
                    {
                        "step": step,
                        "train_loss": training.cpu().tolist(),
                        "validation_loss": validation.cpu().tolist(),
                        "train_macro_recall": torch.stack(train_recall)
                        .mean(0)
                        .cpu()
                        .tolist(),
                        "validation_macro_recall": torch.stack(val_recall)
                        .mean(0)
                        .cpu()
                        .tolist(),
                    }
                )
        if step == 4096:
            break
        batch = torch.randint(len(train_x), (128,), device=device)
        output = logits(train_x[batch])
        loss = (
            F.cross_entropy(
                output.flatten(0, 1), train_y[batch].repeat(count), reduction="none"
            )
            .reshape(count, -1)
            .mean(1)
            .sum()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        old = [value.detach().clone() for value in parameters]
        optimizer.step()
        with torch.no_grad():
            for before, value in zip(old, parameters):
                shape = (-1,) + (1,) * (value.ndim - 1)
                value.copy_(
                    before
                    + (value - before) * rates.reshape(shape)
                    - before * (rates * decays).reshape(shape)
                )
    chosen = int(best_loss.argmin())
    np.savez_compressed(
        root / "pretrained.npz",
        weight=best[0][chosen].cpu().numpy(),
        bias=best[1][chosen].cpu().numpy(),
    )
    (root / "pretrain_history.json").write_text(json.dumps(history) + "\n")
    (root / "pretrain.json").write_text(
        json.dumps(
            {
                "configs": configs,
                "restarts": 2,
                "chosen_row": chosen,
                "validation_loss": best_loss.cpu().tolist(),
                "best_steps": best_step.cpu().tolist(),
                "source_observations_only": True,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], required=True)
    parser.add_argument("--stage", choices=["prepare", "source"], required=True)
    args = parser.parse_args()
    plan = load_plan()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare(plan, args.dataset)
    else:
        device = torch.device("cuda")
        audit_bank(plan, args.dataset, device)
        pretrain(plan, args.dataset, device)


if __name__ == "__main__":
    main()
