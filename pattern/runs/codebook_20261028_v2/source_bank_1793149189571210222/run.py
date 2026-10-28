"""Run the source bank and the learned categorical Pattern prior."""

import argparse
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
    features = torch.tensor(bank["features"], device=device).reshape(
        tasks, networks, 11, 8, 3
    )
    targets = torch.tensor(targets, device=device).reshape(tasks, networks, 11, 8)
    eligible = torch.tensor(
        bank.get("eligible", np.ones(tasks * networks, dtype=bool)), device=device
    ).reshape(tasks, networks)
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
            logits, _, _ = model(features[:, train : train + val], sample=False)
            validation_losses = F.cross_entropy(
                logits.flatten(0, -2),
                targets[:, train : train + val].flatten(),
                weight=class_weight,
                reduction="none",
            )
            validation_losses = validation_losses.reshape(tasks, val, 88).mean(-1)
            validation_weights = eligible[:, train : train + val].float()
            validation_weights /= validation_weights.sum(1, keepdim=True).clamp_min(1)
            validation = (validation_losses * validation_weights).sum() / (
                validation_weights.sum().clamp_min(1)
            )
            if float(validation) < best:
                best = float(validation)
                torch.save(model.state_dict(), output / "prior.pt")
        if epoch == spec["epochs"]:
            break
        model.train()
        order = torch.randperm(train, device=device)
        total_ce, total_kl = 0.0, 0.0
        batches = 0
        for ids in order.split(spec["batch_per_task"]):
            if "eligible" in bank:
                per_task = []
                for task in range(tasks):
                    pool = eligible[task, :train].nonzero().flatten()
                    if not len(pool):
                        raise ValueError(f"No good training masks for source task {task}")
                    per_task.append(pool[torch.randint(len(pool), (len(ids),), device=device)])
                sampled = torch.stack(per_task)
                task_ids = torch.arange(tasks, device=device)[:, None]
                batch_features = features[task_ids, sampled]
                batch_targets = targets[task_ids, sampled]
            else:
                batch_features, batch_targets = features[:, ids], targets[:, ids]
            logits, mu, logvar = model(batch_features)
            ce = F.cross_entropy(
                logits.flatten(0, -2), batch_targets.flatten(), weight=class_weight
            )
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
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
        (output / "bank_history.json").write_text(json.dumps(history))
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
