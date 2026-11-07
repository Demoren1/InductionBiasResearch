"""Manually launched CUDA training on sealed W/G bank cards."""

import argparse
import json
import time
from pathlib import Path

import torch

from .datasets import load_source
from .io import (
    DEFAULT_BANK,
    load_manifest,
    save_json,
    save_torch,
    sha256,
    snapshot,
    source_hashes,
)
from .io import PACKAGE as HERE
from .models.vae import MaskVAE, aligned_iou, objective, topk
from .problem import INPUTS
from .runtime import configure_cuda


def batches(data, batch_size, orders=None):
    tasks, samples = data["x"].shape[:2]
    task_indices = torch.arange(tasks, device=data["x"].device)[:, None]
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        if orders is None:
            yield {key: value[:, start:stop] for key, value in data.items()}
        else:
            indices = orders[:, start:stop]
            yield {
                key: value[task_indices, indices]
                for key, value in data.items()
            }


def summarize(parts):
    count = sum(size for size, _ in parts)
    return {
        key: float(sum(values[key] * size for size, values in parts) / count)
        for key in parts[0][1]
    }


@torch.no_grad()
def evaluate(model, data, options):
    model.eval()
    parts = []
    for batch in batches(data, options["batch_per_task"]):
        output = model(batch["x"], sample=False)
        _, metrics = objective(
            output, batch["mask"], options["beta"], options["hard_weight"]
        )
        metrics["iou_to_imp"] = aligned_iou(output[0], batch["mask"]).mean()
        metrics["mu_variance"] = (
            output[1].flatten(0, 1).var(0, unbiased=False).mean()
        )
        parts.append((batch["mask"].shape[1], metrics))
    return summarize(parts)


@torch.no_grad()
def save_prior(model, config, destination, seed, device):
    generator = torch.Generator().manual_seed(seed + 3001)
    z = torch.randn(
        config["prior_samples"],
        config["model"]["latent_dim"],
        generator=generator,
    ).to(device)
    scores = model.decoder(z)
    masks = topk(scores)
    zero_scores = model.decoder(torch.zeros_like(z[:1]))
    zero_mask = topk(zero_scores)
    # Canonical column codes count distinct structures modulo permutations.
    powers = 1 << torch.arange(INPUTS, device=device, dtype=torch.int64)
    codes = (
        (masks.to(torch.int64) * powers[None, :, None]).sum(1).sort(-1).values
    )
    zero_codes = (
        (zero_mask.to(torch.int64) * powers[None, :, None])
        .sum(1)
        .sort(-1)
        .values
    )
    summary = {
        "samples": len(z),
        "distinct_structures_modulo_columns": len(torch.unique(codes, dim=0)),
        "fraction_different_from_zero_modulo_columns": float(
            (codes != zero_codes).any(-1).float().mean()
        ),
        "mean_logit_std": float(scores.std(0).mean()),
        "scope": "Prior variation only; no analytic masks or held-out families used.",
    }
    save_torch(
        destination / "prior.pt",
        {
            "z": z.cpu(),
            "logits": scores.cpu(),
            "masks": masks.cpu(),
            "zero_logits": zero_scores.cpu(),
            "zero_mask": zero_mask.cpu(),
        },
    )
    save_json(destination / "prior.json", summary)
    return summary


def plot_history(destination, history):
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/overlapping-vae-plots")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5), layout="constrained")
    epochs = [row["epoch"] for row in history]
    for split in ["train", "validation"]:
        axes[0].plot(
            epochs,
            [row[split]["reconstruction"] for row in history],
            label=split,
        )
        axes[1].plot(
            epochs, [row[split]["kl"] for row in history], label=split
        )
    axes[2].plot(epochs, [row["validation"]["iou_to_imp"] for row in history])
    for ax, title in zip(
        axes, ["Matched reconstruction", "KL", "Validation IoU to IMP"]
    ):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
    axes[0].legend()
    figure.savefig(destination / "training.png", dpi=160)
    plt.close(figure)


def train(args):
    torch.set_num_threads(1)
    config = json.loads(args.config.read_text())
    options = config["training"]
    bank = args.bank.resolve()
    manifest = load_manifest(bank)
    protocol = {
        "schema": "overlapping-groups.nf-vae.v1",
        "config": config,
        "seed": args.seed,
        "representation": "wg",
        "bank": str(bank),
        "bank_sha256": sha256(bank / "manifest.json"),
        "encoder": "Independent encoder per source function; grouped GEMM; invariant column pooling.",
        "decoder": "One shared unconditional decoder, z -> 48x8 logits; top-32 mask.",
        "matching": "Hungarian minimum of weighted BCE + hard-MSE against actual IMP target only.",
        "selection": "Fixed final-beta loss on held-out validation seeds of source functions.",
        "held_out_families": "Not loaded or used for fitting, selection, or generation diagnostics.",
        "memory_budget_gib": args.gpu_memory_gib,
        "implementation_sha256": source_hashes(),
    }
    destination = args.run.resolve()
    if (destination / "COMPLETE.json").exists():
        previous = json.loads((destination / "protocol.json").read_text())
        if any(previous.get(key) != value for key, value in protocol.items()):
            raise ValueError(
                "Completed run has different settings; choose another run path"
            )
        print(f"Already complete: {destination}", flush=True)
        return
    if destination.exists():
        raise ValueError(
            f"Run directory already exists; choose a new --run: {destination}"
        )
    device = configure_cuda(args.device, args.gpu_memory_gib)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    task_names, data = load_source(bank, manifest, device)
    model = MaskVAE(len(task_names), config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options["lr"],
        weight_decay=options["weight_decay"],
    )
    protocol.update(
        {
            "tasks": task_names,
            "split_cards": {
                role: int(group["x"].shape[0] * group["x"].shape[1])
                for role, group in data.items()
            },
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": str(torch.__version__),
        }
    )
    destination.mkdir(parents=True)
    snapshot(destination / "source", args.config)
    save_json(destination / "protocol.json", protocol)
    generator = torch.Generator().manual_seed(args.seed + 17)
    history, best, best_epoch = [], float("inf"), 0
    started = time.monotonic()
    print(
        json.dumps(
            {
                "run": str(destination),
                "tasks": len(task_names),
                "parameters": protocol["parameters"],
                "device": str(device),
            }
        ),
        flush=True,
    )
    try:
        for epoch in range(1, options["epochs"] + 1):
            model.train()
            tasks, samples = data["train"]["x"].shape[:2]
            orders = torch.stack(
                [
                    torch.randperm(samples, generator=generator)
                    for _ in range(tasks)
                ]
            ).to(device)
            beta = options["beta"] * min(
                1, epoch / max(1, options["warmup_epochs"])
            )
            parts = []
            for batch in batches(
                data["train"], options["batch_per_task"], orders
            ):
                optimizer.zero_grad(set_to_none=True)
                output = model(batch["x"], sample=True)
                loss, metrics = objective(
                    output, batch["mask"], beta, options["hard_weight"]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    options["clip_norm"],
                    error_if_nonfinite=True,
                )
                optimizer.step()
                parts.append((batch["mask"].shape[1], metrics))
            validation = evaluate(model, data["validation"], options)
            row = {
                "epoch": epoch,
                "beta": beta,
                "train": summarize(parts),
                "validation": validation,
                "elapsed_seconds": time.monotonic() - started,
            }
            if not all(
                torch.isfinite(torch.tensor(value))
                for values in [row["train"], validation]
                for value in values.values()
            ):
                raise RuntimeError("Nonfinite training or validation metrics")
            history.append(row)
            save_json(destination / "history.json", history)
            if validation["loss"] < best:
                best, best_epoch = validation["loss"], epoch
                save_torch(
                    destination / "best.pt",
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "validation": validation,
                        "config": config,
                        "tasks": task_names,
                        "representation": "wg",
                        "seed": args.seed,
                        "bank_sha256": protocol["bank_sha256"],
                    },
                )
                save_torch(
                    destination / "decoder.pt",
                    {
                        "model": model.decoder.state_dict(),
                        "config": config["model"],
                        "epoch": epoch,
                    },
                )
            print(json.dumps(row), flush=True)
            if (
                epoch >= options["warmup_epochs"]
                and epoch - best_epoch >= options["patience"]
            ):
                break
        checkpoint = torch.load(
            destination / "best.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        result = {
            "status": "complete",
            "selected_epoch": best_epoch,
            "epochs": len(history),
            "validation": evaluate(model, data["validation"], options),
            "source_test_seeds": evaluate(model, data["test"], options),
            "prior": save_prior(model, config, destination, args.seed, device),
            "elapsed_seconds": time.monotonic() - started,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device)
            / 1024**3,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device)
            / 1024**3,
        }
        plot_history(destination, history)
        save_json(destination / "COMPLETE.json", result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        save_json(
            destination / "FAILED.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument(
        "--config", type=Path, default=HERE / "configs/vae.json"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpu-memory-gib", type=float, default=12)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    options = config["training"]
    if (
        min(config["model"].values()) <= 0
        or min(
            options["epochs"], options["batch_per_task"], options["patience"]
        )
        <= 0
    ):
        parser.error(
            "Model dimensions, epoch count, batch and patience must be positive"
        )
    if (
        options["lr"] <= 0
        or options["beta"] < 0
        or options["weight_decay"] < 0
        or options["warmup_epochs"] < 0
        or options["clip_norm"] <= 0
        or not 0 <= options["hard_weight"] <= 1
    ):
        parser.error("Invalid optimization settings")
    if config["prior_samples"] < 2:
        parser.error("At least two prior samples are required")
    train(args)


if __name__ == "__main__":
    main()
