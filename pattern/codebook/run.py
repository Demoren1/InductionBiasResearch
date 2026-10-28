"""Run the source bank and the learned categorical Pattern prior."""

import argparse
import gzip
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from pattern.codebook.bank import collect, infer_indices
from pattern.codebook.models import IndexVAE, index_probabilities
from pattern.datasets import input_table, labels_for, partitions


def train_prior(plan, bank, targets, output, device):
    spec = plan["vae"]
    tasks = len(plan["source_patterns"])
    networks = plan["bank"]["networks_per_task"]
    train = plan["bank"]["train_networks"]
    val = plan["bank"]["validation_networks"]
    features = torch.tensor(bank["features"], device=device)
    targets = torch.tensor(targets, device=device)
    if "task_ids" in bank:
        task_ids = torch.tensor(bank["task_ids"], device=device)
        split_ids = torch.tensor(bank["split_ids"], device=device)
    else:
        ids = torch.arange(len(features), device=device)
        task_ids = ids // networks
        split_ids = torch.where(
            ids % networks < train, 0, torch.where(ids % networks < train + val, 1, 2)
        )
    pools = [
        ((task_ids == task) & (split_ids == 0)).nonzero().flatten()
        for task in range(tasks)
    ]
    active = torch.tensor([len(pool) > 0 for pool in pools], device=device).float()
    fallback = (split_ids == 0).nonzero().flatten()[0:1]
    pools = [pool if len(pool) else fallback for pool in pools]
    validation_ids, validation_weights = [], []
    for task in range(tasks):
        pool = ((task_ids == task) & (split_ids == 1)).nonzero().flatten()
        size = min(val, len(pool))
        padded = torch.cat([pool[:size], pools[task][0:1].expand(val - size)])
        weights = torch.zeros(val, device=device)
        weights[:size] = 1 / max(1, size)
        validation_ids.append(padded)
        validation_weights.append(weights)
    validation_ids = torch.stack(validation_ids)
    validation_weights = torch.stack(validation_weights)
    if not validation_weights.any():
        raise ValueError("No good validation solutions; collect more source solutions")
    torch.manual_seed(plan["seed"] + 2)
    model = IndexVAE(tasks, spec).to(device)
    torch.save(model.state_dict(), output / "untrained_prior.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"])
    class_weight = torch.tensor([1, 7, 7, 7, 7], device=device).float()
    best = float("inf")
    history = []
    for epoch in range(spec["epochs"] + 1):
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(features[validation_ids], sample=False)
            validation_losses = F.cross_entropy(
                logits.flatten(0, -2),
                targets[validation_ids].flatten(),
                weight=class_weight,
                reduction="none",
            )
            validation_losses = validation_losses.reshape(tasks, val, 88).mean(-1)
            validation = (validation_losses * validation_weights).sum() / (
                validation_weights.sum().clamp_min(1)
            )
            if float(validation) < best:
                best = float(validation)
                torch.save(model.state_dict(), output / "prior.pt")
        if epoch == spec["epochs"]:
            break
        model.train()
        total_ce, total_kl = 0.0, 0.0
        batches = 0
        batch_size = spec["batch_per_task"]
        for _ in range(2):
            sampled = torch.stack(
                [
                    pool[torch.randint(len(pool), (batch_size,), device=device)]
                    for pool in pools
                ]
            )
            logits, mu, logvar = model(features[sampled])
            ce_by_task = (
                F.cross_entropy(
                    logits.flatten(0, -2),
                    targets[sampled].flatten(),
                    weight=class_weight,
                    reduction="none",
                )
                .reshape(tasks, batch_size, 88)
                .mean((1, 2))
            )
            ce = (ce_by_task * active).sum() / active.sum()
            kl_by_task = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean((1, 2))
            kl = (kl_by_task * active).sum() / active.sum()
            beta = spec["beta"] * min(1, (epoch + 1) / spec["kl_warmup_epochs"])
            optimizer.zero_grad(set_to_none=True)
            (ce + beta * kl).backward()
            optimizer.step()
            total_ce += float(ce.detach())
            total_kl += float(kl.detach())
            batches += 1
        history.append(
            {
                "epoch": epoch,
                "ce": total_ce / batches,
                "kl": total_kl / batches,
                "validation_ce": float(validation),
            }
        )
        if epoch % 20 == 0:
            print(f"PRIOR epoch={epoch} val_ce={float(validation):.4f}", flush=True)
    model.load_state_dict(torch.load(output / "prior.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        z = torch.randn(1024, spec["latent_dim"], device=device)
        sampled = index_probabilities(model.decode(z)).argmax(-1).cpu().numpy()
    np.savez_compressed(output / "prior_samples.npz", indices=sampled)
    (output / "prior_history.json").write_text(json.dumps(history))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["bank", "prior"], required=True)
    parser.add_argument("--protocol", default="pattern/codebook/protocol.json")
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    output = Path(plan["run"])
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / ("source_" + args.stage + "_" + str(time.time_ns()))
    snapshot.mkdir(exist_ok=False)
    files = [Path(args.protocol), *Path("pattern/codebook").glob("*.py")]
    hashes = {}
    for source in files:
        shutil.copy2(source, snapshot / source.name)
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (snapshot / "hashes.json").write_text(json.dumps(hashes, indent=2))
    torch.set_num_threads(4)
    device = torch.device("cuda")
    started = time.time()
    if args.stage == "bank":
        _, x = input_table()
        x = x.to(device)
        y = torch.stack([labels_for(x, p) for p in plan["source_patterns"]])
        splits = {
            key: value.to(device) for key, value in partitions(plan["seed"]).items()
        }
        np.savez(
            output / "splits.npz", **{k: v.cpu().numpy() for k, v in splits.items()}
        )
        if plan["version"] >= 2:
            collection = output / "raw_collection"
            collection.mkdir(exist_ok=False)
        else:
            collection = output
        bank, history = collect(plan, x, y, splits, collection)
        with gzip.open(output / "bank_history.json.gz", "wt") as stream:
            json.dump(history, stream)
        if plan["version"] >= 2:
            from pattern.codebook.extract import good_aligned_bank

            bank, indices, targets, details = good_aligned_bank(plan, bank)
            np.savez_compressed(output / "bank.npz", **bank)
        else:
            indices, targets, details = infer_indices(plan, bank)
        np.savez_compressed(
            output / "extracted.npz", indices=indices, targets=targets, **details
        )
        print("EXTRACTED indices:", indices.tolist(), flush=True)
    else:
        bank = dict(np.load(output / "bank.npz"))
        targets = np.load(output / "extracted.npz")["targets"]
        train_prior(plan, bank, targets, output, device)
    (output / (args.stage + "_complete.json")).write_text(
        json.dumps(
            {
                "seconds": time.time() - started,
                "stage": args.stage,
                "hashes": hashes,
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
