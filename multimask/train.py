"""Collect independent IMP networks when needed, then train NF/VAE encoders."""

import argparse
import math
from pathlib import Path

import torch

from pattern.io import ROOT, device_name, new_directory

from .common import (
    DEFAULT_CONFIG,
    HIDDEN,
    ROWS,
    K,
    Model,
    bce,
    digest,
    fit,
    initial,
    load_bank,
    load_config,
    make_data,
    matches,
    progress,
    save_history,
    save_json,
    save_maps,
    save_torch,
    target,
    tasks,
    topk,
    vae_loss,
)


def collect(config, bank, device, enabled):
    if not bank.resolve().is_relative_to((ROOT / "data").resolve()):
        raise ValueError("banks must be under data/")
    bank.mkdir(parents=True, exist_ok=False)
    data = make_data(config)
    options = config["bank"]
    manifest = {
        "schema": "multimask.v1",
        "status": "running",
        **{
            key: config[key]
            for key in ("seed", "data", "bank", "train_patterns")
        },
        "tasks": tasks(config),
        "hashes": {},
        "shape": [ROWS, HIDDEN, 3],
        "k": K,
        "channels": ["weights", "BCE_gradient", "functional_map"],
        "gradient_loss": "ordinary mean support soft-target BCE, excluding L2",
        "history_fields": [
            "active_edges",
            "support_bce",
            "query_bce",
            "selected_step",
        ],
    }
    save_torch(bank / "data.pt", data)
    manifest["hashes"]["data.pt"] = digest(bank / "data.pt")
    save_json(bank / "manifest.json", manifest)
    xs, xq, xp = (
        data[key].to(device) for key in ("support", "query", "probe")
    )
    for task_index, task in enumerate(
        progress(tasks(config), enabled, desc="Collect tasks", unit="task")
    ):
        ys, yq = target(xs, task), target(xq, task)
        chunks = []
        count = options["maps_per_task"]
        for start in progress(
            range(0, count, options["batch_size"]),
            enabled,
            desc=task,
            unit="batch",
            leave=False,
        ):
            seeds = [
                config["seed"] + 100003 * task_index + 1000003 * (index + 1)
                for index in range(
                    start, min(count, start + options["batch_size"])
                )
            ]
            original = initial(seeds, device)
            mask = torch.ones_like(original["w"])
            rounds = []
            while True:
                active = int(mask[0].sum())
                state, selected, _ = fit(
                    original,
                    mask,
                    (xs, ys),
                    (xq, yq),
                    options["steps"],
                    options["lr"],
                    options["l2"],
                    options["select_every"],
                    enabled,
                    options["select_every"],
                )
                with torch.no_grad():
                    scores = torch.stack(
                        (bce(state, mask, xs, ys), bce(state, mask, xq, yq))
                    ).cpu()
                rounds.append(
                    torch.stack(
                        (torch.full_like(selected, active), *scores, selected),
                        -1,
                    )
                )
                if active == K:
                    break
                keep = max(
                    K,
                    min(
                        active - 1,
                        math.ceil(active * (1 - options["prune_fraction"])),
                    ),
                )
                values = (
                    state["w"]
                    .abs()
                    .masked_fill(mask == 0, -torch.inf)
                    .flatten(1)
                    .cpu()
                )
                winners = values.argsort(dim=-1, descending=True, stable=True)[
                    :, :keep
                ].to(device)
                mask = (
                    torch.zeros_like(values, device=device)
                    .scatter_(1, winners, 1)
                    .reshape_as(mask)
                )
            weights = state["w"].detach().clone().requires_grad_()
            (gradient,) = torch.autograd.grad(
                bce({**state, "w": weights}, mask, xs, ys).sum(), weights
            )
            with torch.no_grad():
                activity = (
                    (torch.matmul(xp, state["w"]) + state["b"][:, None] > 0)
                    .float()
                    .mean(1)
                )
                functional = (
                    (state["w"] * mask).abs()
                    * state["a"].abs()[:, None]
                    * activity[:, None]
                )
                raw = torch.stack(
                    (state["w"] * mask, gradient, functional), -1
                )
                mass = functional.sum(1)
                centroid = (
                    functional
                    * torch.arange(ROWS, device=device)[None, :, None]
                ).sum(1) / mass.clamp_min(1e-8)
                centroid = centroid.masked_fill(mass == 0, float("inf"))
                order = centroid.cpu().argsort(dim=-1, stable=True).to(device)
                features = raw.gather(
                    2, order[:, None, :, None].expand_as(raw)
                )
                features = features / features.abs().amax(
                    (1, 2), keepdim=True
                ).clamp_min(1e-8)
                chunks.append(
                    {
                        "features": features.cpu(),
                        "masks": mask.gather(
                            2, order[:, None].expand_as(mask)
                        ).cpu(),
                        "raw_channels": raw.cpu(),
                        "column_orders": order.cpu(),
                        "seeds": torch.tensor(seeds),
                        "history": torch.stack(rounds, 1),
                        **{key: value.cpu() for key, value in state.items()},
                    }
                )
            print(
                f"{task}: {start + len(seeds)}/{count} maps, 32 connections",
                flush=True,
            )
        card = {
            key: torch.cat([chunk[key] for chunk in chunks])
            for key in chunks[0]
        }
        allocation = torch.randperm(
            count,
            generator=torch.Generator().manual_seed(
                config["seed"] + task_index + 71
            ),
        )
        size = max(1, count // 8)
        card["split"] = torch.zeros(count, dtype=torch.long)
        card["split"][allocation[:size]] = 2
        card["split"][allocation[size : 2 * size]] = 1
        iou, variant, _ = matches(card["masks"], task)
        save_torch(bank / f"{task}.pt", card)
        manifest["hashes"][f"{task}.pt"] = digest(bank / f"{task}.pt")
        manifest.setdefault("quality", {})[task] = {
            "query_bce": float(card["history"][:, -1, 2].mean()),
            "oracle_query_bce": float(
                torch.nn.functional.binary_cross_entropy(yq, yq)
            ),
            "nearest_iou": float(iou.mean()),
            "exact_fraction": float((iou == 1).float().mean()),
            "exact_variant_counts": torch.bincount(
                variant[iou == 1], minlength=4
            ).tolist(),
        }
        save_json(bank / "manifest.json", manifest)
    manifest["status"] = "complete"
    save_json(bank / "manifest.json", manifest)
    return bank


def train(config, bank, run_id, device, enabled):
    if (ROOT / "multimask/runs" / run_id).exists():
        raise ValueError("run already exists; choose a new RUN_ID")
    if not bank.exists():
        collect(config, bank, device, enabled)
    bank, _, cards, _ = load_bank(bank, config)
    torch.manual_seed(config["seed"] + 13)
    model = Model(tasks(config), config["model"]).to(device)
    data = {
        task: {
            split: tuple(
                card[key][card["split"] == role].to(device)
                for key in ("features", "masks")
            )
            for split, role in (("train", 0), ("validation", 1))
        }
        for task, card in cards.items()
    }
    options = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options["lr"],
        weight_decay=options["weight_decay"],
    )
    destination = new_directory(ROOT / "multimask/runs", run_id)
    reference = {"path": str(bank), "sha256": digest(bank / "manifest.json")}
    save_json(destination / "config.json", config)
    save_json(destination / "bank_reference.json", reference)
    generator = torch.Generator().manual_seed(config["seed"] + 17)
    history = []
    best = (-1.0, -float("inf"))
    best_epoch = 0
    loop = progress(
        range(1, options["epochs"] + 1),
        enabled,
        desc=f"Train ({device})",
        unit="epoch",
    )
    for epoch in loop:
        model.train()
        beta = options["beta"] * min(1, epoch / max(1, options["warmup"]))
        indices = {
            task: torch.randperm(len(card["train"][0]), generator=generator)
            for task, card in data.items()
        }
        for start in range(
            0, len(next(iter(indices.values()))), options["batch_size"]
        ):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for task, order in indices.items():
                rows = order[start : start + options["batch_size"]].to(device)
                x, masks = data[task]["train"]
                losses.append(
                    vae_loss(
                        model(task, x[rows], True),
                        masks[rows],
                        beta,
                        options["hard_weight"],
                    )[0]
                )
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
        model.eval()
        row = {
            "epoch": epoch,
            "beta": options["beta"],
            "optimization_beta": beta,
        }
        with torch.no_grad():
            for split in ("train", "validation"):
                totals = []
                ious = []
                for task, card in data.items():
                    x, masks = card[split]
                    output = model(task, x)
                    loss, kl = vae_loss(
                        output, masks, options["beta"], options["hard_weight"]
                    )
                    totals.append(torch.stack((loss, kl)))
                    ious.append(
                        float(matches(topk(output[0]), task)[0].mean())
                    )
                loss, kl = torch.stack(totals).mean(0).cpu().tolist()
                row.update(
                    {
                        f"{split}_loss": loss,
                        f"{split}_kl": kl,
                        f"{split}_beta_kl": options["beta"] * kl,
                        f"{split}_iou": sum(ious) / len(ious),
                    }
                )
        history.append(row)
        save_json(destination / "history.json", history)
        checkpoint = {
            "config": config,
            "model": model.state_dict(),
            "epoch": epoch,
            "bank": reference,
        }
        save_torch(destination / "checkpoints/last.pt", checkpoint)
        rank = (row["validation_iou"], -row["validation_loss"])
        if rank > best:
            best = rank
            best_epoch = epoch
            save_torch(destination / "checkpoints/best.pt", checkpoint)
            save_torch(
                destination / "checkpoints/shared_decoder.pt",
                model.decoder.state_dict(),
            )
        loop.set_postfix(
            loss=f"{row['validation_loss']:.5f}", iou=f"{rank[0]:.4f}"
        )
        print(
            f"epoch {epoch}: validation loss={row['validation_loss']:.5f}, "
            f"IoU={rank[0]:.4f}, KL={row['validation_kl']:.4f}",
            flush=True,
        )
        if options["patience"] and epoch - best_epoch >= options["patience"]:
            break
    best_checkpoint = torch.load(
        destination / "checkpoints/best.pt",
        weights_only=True,
        map_location="cpu",
    )
    model.load_state_dict(best_checkpoint["model"])
    save_history(destination, history)
    with torch.no_grad():
        for task, card in data.items():
            save_maps(
                destination, task, model(task, card["validation"][0])[0], task
            )
    save_json(
        destination / "COMPLETE.json",
        {
            "best_epoch": best_epoch,
            "validation_iou": best[0],
            "tasks": tasks(config),
        },
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--bank",
        type=Path,
        default=ROOT / "data/multimask/banks/imp32_1024_v1",
    )
    parser.add_argument("--run-id", default="nf_vae_v1")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-progress", action="store_true")
    for flag in (
        "maps-per-task",
        "bank-steps",
        "bank-batch",
        "epochs",
        "batch-size",
    ):
        parser.add_argument("--" + flag, type=int)
    for flag in ("lr", "beta"):
        parser.add_argument("--" + flag, type=float)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    config = load_config(args.config, args)
    device = device_name(args.device)
    print(f"Device: {device}", flush=True)
    destination = train(
        config, args.bank, args.run_id, device, not args.no_progress
    )
    print(f"Run saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
