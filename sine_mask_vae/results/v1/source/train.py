"""Train the original and accepted partwise NF/VAE on the same elite bank."""

import copy
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bank import DIRECTORY, RUN, SEEDS, digest
from core import save_json

REFERENCE = DIRECTORY.parent / "random_mask_vae"
sys.path.insert(1, str(REFERENCE))
sys.path.insert(2, str(DIRECTORY.parents[1]))

from improved_models import ImprovedVAE, objective, partwise_forward  # noqa: E402


def load_cards():
    manifest = json.loads((RUN / "bank/manifest.json").read_text())
    roles = [[], []]
    for record in manifest:
        path = RUN / "bank" / record["cards"]
        if digest(path) != record["sha256"]:
            raise RuntimeError("Changed elite cards")
        with np.load(path) as data:
            for role in range(2):
                selected = data["roles"] == role
                roles[role].append(
                    (
                        torch.tensor(data["x"][selected]),
                        torch.tensor(data["masks"][selected]),
                    )
                )
    return manifest, roles


def sampled_batch(tasks, count, generator=None, fixed=False):
    features, labels = [], []
    for x, y in tasks:
        indices = (
            torch.arange(count) % len(x)
            if fixed
            else torch.randint(len(x), (count,), generator=generator)
        )
        features.append(x[indices])
        labels.append(y[indices])
    return torch.stack(features), torch.stack(labels)


def validation(model, tasks):
    x, target = sampled_batch(tasks, max(len(x) for x, _ in tasks), fixed=True)
    with torch.no_grad():
        output = model(x, sample=False)
        metrics = []
        for index, (features, _) in enumerate(tasks):
            sliced = tuple(
                t[index : index + 1, : len(features)] for t in output
            )
            _, values = objective(
                sliced,
                target[index : index + 1, : len(features)],
                beta=0.05,
                canonical=True,
            )
            metrics.append(values)
    return {
        key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]
    }


def restore(record):
    path = RUN / record["checkpoint"]
    if digest(path) != record["sha256"]:
        raise RuntimeError("Frozen checkpoint changed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ImprovedVAE(checkpoint["tasks"])
    model.load_state_dict(checkpoint["state"])
    return model.eval().requires_grad_(False)


def main():
    torch.set_num_threads(1)
    if not (RUN / "BANK_COMPLETE.json").exists():
        raise RuntimeError("Bank must finish first")
    manifest, roles = load_cards()
    source = RUN / "source"
    shutil.copy2(__file__, source / "train.py")
    for path, name in (
        (REFERENCE / "models.py", "reference_models.py"),
        (REFERENCE / "improved_models.py", "reference_improved_models.py"),
        (
            DIRECTORY.parents[1] / "overlapping_groups/models/vae.py",
            "repository_nf.py",
        ),
    ):
        shutil.copy2(path, source / name)
    records = []
    for variant in ("vanilla", "partwise_training"):
        for seed in SEEDS:
            destination = RUN / "models" / f"{variant}_{seed}"
            destination.mkdir(parents=True, exist_ok=True)
            summary_path = destination / "summary.json"
            if summary_path.exists():
                records.append(json.loads(summary_path.read_text()))
                continue
            started = time.perf_counter()
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed + 17)
            model = ImprovedVAE(len(manifest))
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.001, weight_decay=0.01
            )
            best_loss, best_epoch, best_state = float("inf"), 0, None
            history = []
            for epoch in range(1, 601):
                model.train()
                x, target = sampled_batch(roles[0], 16, generator)
                optimizer.zero_grad(set_to_none=True)
                output = (
                    partwise_forward(model, x)
                    if variant == "partwise_training"
                    else model(x)
                )
                loss, _ = objective(
                    output,
                    target,
                    beta=0.05 * min(1, epoch / 90),
                    canonical=True,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if epoch % 5 and epoch != 1:
                    continue
                model.eval()
                values = validation(model, roles[1])
                history.append({"epoch": epoch, **values})
                if values["loss"] < best_loss:
                    best_loss, best_epoch = values["loss"], epoch
                    best_state = copy.deepcopy(model.state_dict())
                if epoch - best_epoch >= 120:
                    break
            model.load_state_dict(best_state)
            path = destination / "best.pt"
            torch.save(
                {"tasks": len(manifest), "state": model.state_dict()}, path
            )
            record = {
                "variant": variant,
                "seed": seed,
                "checkpoint": str(path.relative_to(RUN)),
                "sha256": digest(path),
                "epochs": epoch,
                "best_epoch": best_epoch,
                "validation": validation(model, roles[1]),
                "seconds": time.perf_counter() - started,
            }
            save_json(summary_path, record)
            save_json(destination / "history.json", history)
            records.append(record)
            print(record, flush=True)
    save_json(RUN / "models.json", records)
    save_json(RUN / "TRAIN_COMPLETE.json", {"models": len(records)})


if __name__ == "__main__":
    main()
