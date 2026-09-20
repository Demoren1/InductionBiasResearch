"""Fine-tune a random-task MoE for the ordinary MNIST8m digit sum.

The coordinate-generated first-layer U is frozen.  The downstream layers can
optionally be trained.  The training and test sets are the same masked,
nonzero-digit sets used for the paper MLP control.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .meta_u_first_v_moe import VExpertMoE
from .run import TEST_LENGTHS, batch, load_split


CENTER = 4.5
SCALE = math.sqrt(8.25)


def predict(model: VExpertMoE, images: torch.Tensor, mask: torch.Tensor,
            frozen_u: torch.Tensor) -> torch.Tensor:
    """Sum per-image predictions while skipping padded images."""
    count, capacity = mask.shape
    valid = mask.reshape(-1)
    pixels = images.reshape(-1, 784)[valid]
    projections, route = model.prepare(pixels, frozen_u)
    coefficients = model.initial_coefficients
    effective_v = (route * (coefficients * math.sqrt(model.n_experts))) @ model.v_experts
    first = torch.tanh((projections * effective_v[:, None, :]).sum(-1))
    second = torch.tanh(model.base.second(first))
    third = torch.tanh(model.base.third(second))
    per_image = F.pad(third, (0, 1), value=1.0) @ model.base.initial_readout
    owners = torch.arange(count, device=images.device)[:, None].expand(-1, capacity)
    return per_image.new_zeros(count).index_add_(
        0, owners.reshape(-1)[valid], per_image)


@torch.no_grad()
def evaluate(model: VExpertMoE, frozen_u: torch.Tensor,
             images: np.ndarray, split: dict[str, np.ndarray],
             device: torch.device, batch_size: int, center: float,
             scale: float) -> dict[str, float]:
    model.eval()
    absolute = 0.0
    exact = 0
    count = len(split["targets"])
    for start in range(0, count, batch_size):
        ids = np.arange(start, min(start + batch_size, count))
        x, mask, target = batch(images, split, ids, device, 255.0)
        raw_prediction = predict(model, x, mask, frozen_u) * scale + center * mask.sum(1)
        absolute += (raw_prediction - target).abs().sum().item()
        exact += (raw_prediction.round() == target).sum().item()
    return {"mae": absolute / count, "exact_round_accuracy": exact / count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router", choices=("conv", "mlp"), default="conv")
    parser.add_argument("--trainable", choices=("router_v", "router_v_coeff",
                                              "router_v_readout", "router_v_head"),
                        default="router_v")
    parser.add_argument("--unfreeze-middle", action="store_true",
                        help="Also fine-tune the second and third image layers; U stays frozen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mnist8m"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path(
        "deepsets_z/mnist8m/outputs/v_moe_digit_sum_finetune"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--adam-eps", type=float, default=0.001)
    parser.add_argument("--target-center", type=float, default=CENTER)
    parser.add_argument("--target-scale", type=float, default=SCALE)
    parser.add_argument("--probe-sets", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (min(args.epochs, args.patience, args.batch_size, args.probe_sets) < 1
            or args.lr <= 0 or args.adam_eps <= 0 or args.target_scale <= 0):
        parser.error("Epochs, patience, batch size, probe sets, LR, and Adam epsilon must be positive")
    if args.checkpoint is None:
        directory = ("meta_u_first_v_moe_router_conv_10k" if args.router == "conv"
                     else "meta_u_first_v_moe_router_mlp_10k")
        stem = "moe_k96_ortho0p2_seed42" + ("_mlp" if args.router == "mlp" else "")
        args.checkpoint = Path("deepsets_z/mnist8m/outputs") / directory / f"{stem}_best.pt"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    images = np.load(args.data_dir / "images.npy", mmap_mode="r")
    sets_dir = args.data_dir / "sets_authors"
    train = load_split(sets_dir / "train.npz")
    validation = load_split(sets_dir / "validation.npz")
    probes = {length: {key: value[:args.probe_sets]
                       for key, value in load_split(sets_dir / f"test_{length}.npz").items()}
              for length in (5, 20, 50)}
    model = VExpertMoE(96, args.seed, args.router).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=True))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.router.parameters():
        parameter.requires_grad_(True)
    model.v_experts.requires_grad_(True)
    if args.unfreeze_middle:
        for layer in (model.base.second, model.base.third):
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
    if args.trainable in ("router_v_coeff", "router_v_head"):
        model.initial_coefficients.requires_grad_(True)
    if args.trainable in ("router_v_readout", "router_v_head"):
        model.base.initial_readout.requires_grad_(True)
    with torch.no_grad():
        frozen_u = model.base.u().detach()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr, eps=args.adam_eps)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    rng = np.random.default_rng(args.seed + 3_000_000)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.router}_{args.trainable}_seed{args.seed}"
    if args.unfreeze_middle:
        stem += "_unfrozen_middle"
    result_path = args.out / f"{stem}.json"
    latest_path = args.out / f"{stem}_latest.pt"
    best_path = args.out / f"{stem}_best.pt"
    if result_path.exists() and not args.resume:
        raise FileExistsError(result_path)
    start_epoch, best_epoch, best_mae, steps = 0, 0, math.inf, 0
    history = []
    previous_seconds = 0.0
    if args.resume:
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rng.bit_generator.state = saved["rng"]
        start_epoch, best_epoch, best_mae = (
            saved["epoch"], saved["best_epoch"], saved["best_mae"])
        history, steps = saved["history"], saved["steps"]
        previous_seconds = saved["elapsed_seconds"]
    started = time.monotonic() - previous_seconds
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train["targets"]))
        train_mae = 0.0
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            x, mask, target = batch(images, train, ids, device, 255.0)
            normalized_target = (target - args.target_center * mask.sum(1)) / args.target_scale
            prediction = predict(model, x, mask, frozen_u)
            mae = (prediction - normalized_target).abs().mean()
            loss = mae + 0.2 * model.orthogonality_loss()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            train_mae += mae.detach().item() * len(ids) * args.target_scale
        train_mae /= len(train["targets"])
        val = evaluate(model, frozen_u, images, validation, device, args.batch_size,
                       args.target_center, args.target_scale)
        scheduler.step(val["mae"])
        probe = {str(length): evaluate(model, frozen_u, images, split,
                                       device, args.batch_size,
                                       args.target_center, args.target_scale)
                 for length, split in probes.items()}
        row = {"epoch": epoch, "steps": steps, "train_mae": train_mae,
               "validation": val, "probe": probe,
               "lr": optimizer.param_groups[0]["lr"],
               "elapsed_seconds": time.monotonic() - started}
        history.append(row)
        if val["mae"] < best_mae - 1e-5:
            best_mae, best_epoch = val["mae"], epoch
            torch.save(model.state_dict(), best_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "rng": rng.bit_generator.state,
                    "epoch": epoch, "best_epoch": best_epoch, "best_mae": best_mae,
                    "history": history, "steps": steps,
                    "elapsed_seconds": row["elapsed_seconds"]}, latest_path)
        result = {"config": {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()},
                  "frozen_u": True,
                  "frozen_middle_layers": not args.unfreeze_middle,
                  "trained_parameters": sum(p.numel() for p in trainable),
                  "train_sets": len(train["targets"]),
                  "best_epoch": best_epoch, "best_validation_mae": best_mae,
                  "history": history}
        temporary = result_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(result_path)
        print(f"{stem} epoch={epoch} steps={steps} train_mae={train_mae:.3f} "
              f"val_mae={val['mae']:.3f} acc5={probe['5']['exact_round_accuracy']:.3%} "
              f"acc50={probe['50']['exact_round_accuracy']:.3%} "
              f"best={best_epoch} elapsed={row['elapsed_seconds']:.0f}s", flush=True)
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    result["stopping"] = {"epoch": epoch,
                          "reason": "validation_plateau" if epoch - best_epoch >= args.patience
                          else "epoch_limit"}
    result["test"] = {str(length): evaluate(
        model, frozen_u, images, load_split(sets_dir / f"test_{length}.npz"),
        device, args.batch_size, args.target_center, args.target_scale)
        for length in TEST_LENGTHS}
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"DONE {stem} best={best_epoch} "
          f"test50_acc={result['test']['50']['exact_round_accuracy']:.3%}", flush=True)


if __name__ == "__main__":
    main()
