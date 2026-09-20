"""Image digit-sum benchmark: paper MLP and generated sharing in its last image layer.

All arms see only images and labels of whole sets. The generated arm replaces the
paper's 100x30 image-layer matrix by W=Uv. U is a hard assignment of every
weight coordinate to one of 16 shared scalars; there is no set position input.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ARMS = ("paper_mlp", "learned_z", "fixed_z", "no_z")
TEST_LENGTHS = tuple(range(5, 51, 5))


def weight_coordinates(device: torch.device) -> torch.Tensor:
    incoming = torch.arange(100, device=device).float() / 99
    outgoing = torch.arange(30, device=device).float() / 29
    i, j = torch.meshgrid(incoming, outgoing, indexing="ij")
    coords = [i, j]
    for frequency in (1, 3):
        for coordinate in (i, j):
            coords.extend((torch.sin(2 * math.pi * frequency * coordinate),
                           torch.cos(2 * math.pi * frequency * coordinate)))
    return torch.stack(coords, dim=-1).reshape(-1, 10)


class CoordinateGenerator(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim + 10, 16), nn.Tanh(),
                                 nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 16))
        nn.init.normal_(self.net[-1].weight, std=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, coordinates: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        if z is not None:
            expanded = z[None].expand(coordinates.shape[0], -1)
            coordinates = torch.cat((expanded, coordinates), dim=-1)
        return self.net(coordinates)


class GeneratedLayer(nn.Module):
    def __init__(self, arm: str, seed: int) -> None:
        super().__init__()
        if arm not in ("learned_z", "fixed_z", "no_z"):
            raise ValueError(arm)
        torch.manual_seed(seed + 10)
        full = CoordinateGenerator(latent_dim=4)
        initial_z = torch.randn(4)
        if arm == "no_z":
            generator = CoordinateGenerator(latent_dim=0)
            with torch.no_grad():
                first, folded = full.net[0], generator.net[0]
                folded.weight.copy_(first.weight[:, 4:])
                folded.bias.copy_(first.bias + first.weight[:, :4] @ initial_z)
                for index in (2, 4):
                    generator.net[index].load_state_dict(full.net[index].state_dict())
            self.z = None
        else:
            generator = full
            if arm == "learned_z":
                self.z = nn.Parameter(initial_z)
            else:
                self.register_buffer("z", initial_z)
        self.generator = generator
        torch.manual_seed(seed + 11)
        self.values = nn.Parameter(torch.linspace(-0.14, 0.14, 16)
                                   + 0.005 * torch.randn(16))
        self.bias = nn.Parameter(torch.zeros(30))
        self.register_buffer("coordinates", weight_coordinates(torch.device("cpu")))

    def weight(self) -> torch.Tensor:
        logits = self.generator(self.coordinates, self.z)
        hard = F.one_hot(logits.argmax(-1), num_classes=16).float()
        if self.training:
            soft = logits.softmax(-1)
            assignment = hard + soft - soft.detach()
        else:
            assignment = hard
        # Row order: input coordinate first, output coordinate second.
        return (assignment @ self.values).reshape(100, 30).T.contiguous()


class ImageSum(nn.Module):
    def __init__(self, arm: str, seed: int, paper_initialization: bool = False,
                 padding_mode: str = "masked") -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        if padding_mode not in ("masked", "notebook"):
            raise ValueError(padding_mode)
        torch.manual_seed(seed)
        self.padding_mode = padding_mode
        self.first = nn.Linear(784, 300)
        self.second = nn.Linear(300, 100)
        self.readout = nn.Linear(30, 1)
        self.third = (nn.Linear(100, 30) if arm == "paper_mlp"
                      else GeneratedLayer(arm, seed))
        if paper_initialization:
            layers = [self.first, self.second, self.readout]
            if isinstance(self.third, nn.Linear):
                layers.append(self.third)
            for layer in layers:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        count, capacity = images.shape[:2]
        valid = mask.reshape(-1)
        pixels = images.reshape(-1, 784)[valid]
        weight = self.third.weight() if isinstance(self.third, GeneratedLayer) else None

        def encode(input_pixels: torch.Tensor) -> torch.Tensor:
            hidden = torch.tanh(self.first(input_pixels))
            hidden = torch.tanh(self.second(hidden))
            if weight is not None:
                hidden = F.linear(hidden, weight, self.third.bias)
            else:
                hidden = self.third(hidden)
            return torch.tanh(hidden)

        x = encode(pixels)
        owners = torch.arange(count, device=images.device)[:, None].expand(-1, capacity)
        summed = x.new_zeros((count, 30)).index_add_(0, owners.reshape(-1)[valid], x)
        if self.padding_mode == "notebook" and (~valid).any():
            # The released Keras Lambda sums encoded index-0 images even at
            # padded positions. Encode that same image once per batch.
            pad_pixel = images.reshape(-1, 784)[~valid][:1]
            pad_feature = encode(pad_pixel)
            pad_counts = (~mask).sum(1).to(summed.dtype)
            summed = summed + pad_counts[:, None] * pad_feature
        return self.readout(summed).squeeze(-1)


def load_split(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in ("indices", "targets", "lengths")}


def batch(images: np.ndarray, split: dict[str, np.ndarray], indices: np.ndarray,
          device: torch.device, pixel_scale: float):
    chosen = split["indices"][indices]
    present = chosen >= 0
    pixels = np.ascontiguousarray(images[np.maximum(chosen, 0)])
    x = torch.from_numpy(pixels).to(device=device, non_blocking=True).float()
    if pixel_scale != 1:
        x.div_(pixel_scale)
    mask = torch.from_numpy(present).to(device=device, non_blocking=True)
    target = torch.from_numpy(split["targets"][indices]).to(device=device,
                                                                non_blocking=True)
    return x, mask, target


@torch.no_grad()
def evaluate(model: ImageSum, images: np.ndarray, split: dict[str, np.ndarray],
             device: torch.device, batch_size: int, pixel_scale: float) -> dict[str, float]:
    model.eval()
    absolute = 0.0
    squared = 0.0
    exact = 0
    n = len(split["targets"])
    for start in range(0, n, batch_size):
        ids = np.arange(start, min(start + batch_size, n))
        x, mask, target = batch(images, split, ids, device, pixel_scale)
        prediction = model(x, mask)
        error = prediction - target
        absolute += error.abs().sum().item()
        squared += error.square().sum().item()
        exact += (prediction.round() == target).sum().item()
    return {"mae": absolute / n, "rmse": math.sqrt(squared / n),
            "exact_round_accuracy": exact / n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mnist8m"))
    parser.add_argument("--sets-subdir", default="sets")
    parser.add_argument("--output", type=Path, default=Path("deepsets_z/mnist8m/results"))
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-sets", type=int, default=100_000)
    parser.add_argument("--pixel-scale", type=float, default=1.0)
    parser.add_argument("--paper-initialization", action="store_true")
    parser.add_argument("--padding-mode", choices=("masked", "notebook"),
                        default="masked", help="Whether to sum padded index-0 images as in Keras")
    parser.add_argument("--evaluation-checkpoint", choices=("best", "last"),
                        default="best", help="The authors evaluate the last epoch")
    parser.add_argument("--probe-epochs", default="",
                        help="Comma-separated epochs for per-length held-out curves")
    parser.add_argument("--probe-set-count", type=int, default=1000)
    parser.add_argument("--resume", action="store_true",
                        help="Continue from the best checkpoint and its previous history")
    args = parser.parse_args()
    if args.pixel_scale <= 0:
        raise ValueError("pixel-scale must be positive")
    probe_epochs = {int(value) for value in args.probe_epochs.split(",") if value}
    if any(epoch < 1 or epoch > args.max_epochs for epoch in probe_epochs):
        raise ValueError("probe epochs must lie within 1..max_epochs")
    if args.probe_set_count < 1:
        raise ValueError("probe-set-count must be positive")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    images = np.load(args.data_dir / "images.npy", mmap_mode="r")
    sets_dir = args.data_dir / args.sets_subdir
    train = load_split(sets_dir / "train.npz")
    train = {key: value[:args.train_sets] for key, value in train.items()}
    validation = load_split(sets_dir / "validation.npz")
    probes = ({length: {key: value[:args.probe_set_count]
                        for key, value in load_split(sets_dir / f"test_{length}.npz").items()}
               for length in TEST_LENGTHS} if probe_epochs else {})
    model = ImageSum(args.arm, args.seed, args.paper_initialization,
                     args.padding_mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6)
    generator = np.random.default_rng(args.seed + 3_000_000)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.arm}_seed{args.seed}"
    result_path = args.output / f"{stem}.json"
    checkpoint = args.output / f"{stem}.pt"
    if result_path.exists() and not args.resume:
        print(f"SKIP {result_path}", flush=True)
        return
    best_mae = math.inf
    best_epoch = 0
    history = []
    start_epoch = 0
    previous_seconds = 0.0
    optimizer_steps = 0
    if args.resume:
        previous = json.loads(result_path.read_text())
        if any(previous["config"].get(key) != value for key, value in
               (("train_sets", args.train_sets), ("pixel_scale", args.pixel_scale),
                ("sets_subdir", args.sets_subdir),
                ("paper_initialization", args.paper_initialization),
                ("padding_mode", args.padding_mode),
                ("evaluation_checkpoint", args.evaluation_checkpoint))):
            raise ValueError("Resume requires identical data and initialization")
        stored = torch.load(checkpoint, map_location=device, weights_only=False)
        if "model" in stored:
            model.load_state_dict(stored["model"])
            optimizer.load_state_dict(stored["optimizer"])
            scheduler.load_state_dict(stored["scheduler"])
        else:
            model.load_state_dict(stored)
        best_epoch = previous["best_epoch"]
        start_epoch = best_epoch
        best_mae = previous["metrics"]["validation"]["mae"]
        history = [row for row in previous["history"] if row["epoch"] <= best_epoch]
        previous_seconds = previous["training_seconds"]
        optimizer_steps = start_epoch * math.ceil(len(train["targets"]) / args.batch_size)
        print(f"RESUME {stem} best_epoch={best_epoch} best_val_mae={best_mae:.4f}", flush=True)
    started = time.monotonic()
    for epoch in range(start_epoch + 1, args.max_epochs + 1):
        model.train()
        order = generator.permutation(len(train["targets"]))
        losses = []
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            x, mask, target = batch(images, train, ids, device, args.pixel_scale)
            predicted = model(x, mask)
            loss = (predicted - target).abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(loss.detach()))
        val_metrics = evaluate(model, images, validation, device,
                               min(args.batch_size, 256), args.pixel_scale)
        scheduler.step(val_metrics["mae"])
        row = {"epoch": epoch, "optimizer_steps": optimizer_steps,
               "elapsed_seconds": time.monotonic() - started,
               "train_mae_batches": float(np.mean(losses)),
               "validation": val_metrics, "learning_rate": optimizer.param_groups[0]["lr"]}
        if epoch in probe_epochs:
            row["probe"] = {str(length): evaluate(model, images, split, device,
                                                   min(args.batch_size, 256), args.pixel_scale)
                            for length, split in probes.items()}
            print(f"{stem} probe epoch={epoch} steps={optimizer_steps} "
                  f"acc50={row['probe']['50']['exact_round_accuracy']:.3f}", flush=True)
        history.append(row)
        print(f"{stem} epoch={epoch} train_mae={np.mean(losses):.4f} "
              f"val_mae={val_metrics['mae']:.4f} val_exact={val_metrics['exact_round_accuracy']:.3f} "
              f"elapsed={time.monotonic()-started:.1f}s", flush=True)
        if val_metrics["mae"] < best_mae - 1e-5:
            best_mae = val_metrics["mae"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch}, checkpoint)
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            break
    if args.evaluation_checkpoint == "best":
        stored = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(stored["model"] if "model" in stored else stored)
        evaluated_epoch = best_epoch
    else:
        evaluated_epoch = epoch
        torch.save({"model": model.state_dict(), "epoch": epoch},
                   args.output / f"{stem}_last.pt")
    metrics = {"validation": evaluate(model, images, validation, device,
                                       min(args.batch_size, 256), args.pixel_scale)}
    for length in TEST_LENGTHS:
        split = load_split(sets_dir / f"test_{length}.npz")
        metrics[f"test_{length}"] = evaluate(model, images, split, device,
                                             min(args.batch_size, 256), args.pixel_scale)
    result = {
        "arm": args.arm, "seed": args.seed, "best_epoch": best_epoch,
        "evaluated_epoch": evaluated_epoch,
        "training_seconds": previous_seconds + time.monotonic() - started,
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "config": {"max_epochs": args.max_epochs, "min_epochs": args.min_epochs,
                   "patience": args.patience, "batch_size": args.batch_size,
                   "train_sets": args.train_sets, "pixel_scale": args.pixel_scale,
                   "sets_subdir": args.sets_subdir,
                   "paper_initialization": args.paper_initialization,
                   "padding_mode": args.padding_mode,
                   "evaluation_checkpoint": args.evaluation_checkpoint,
                   "probe_epochs": sorted(probe_epochs),
                   "probe_set_count": args.probe_set_count if probe_epochs else 0,
                   "optimizer": "Adam(lr=0.001, eps=0.001)", "loss": "MAE"},
        "metrics": metrics, "history": history,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"DONE {stem} best_epoch={best_epoch} "
          f"test50_exact={metrics['test_50']['exact_round_accuracy']:.3f}", flush=True)


if __name__ == "__main__":
    main()
