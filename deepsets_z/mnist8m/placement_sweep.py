"""One-seed, masked-padding sweep over generated-layer placement and capacity.

Run ``python -m deepsets_z.mnist8m.placement_sweep --dry-run`` to inspect the
matrix. Without ``--dry-run``, complete runs are skipped and interrupted runs
resume from their latest full-epoch checkpoint. Automatic device selection
reads GPU utilization only; it never queries GPU memory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .run import ImageSum, TEST_LENGTHS, batch, evaluate, load_split


@dataclass(frozen=True)
class Spec:
    name: str
    layer: str | None
    mode: str = "dense"
    hidden: int = 0
    values: int = 0


SPECS = (
    Spec("dense", None),
    Spec("last_small", "third", "discrete", 16, 16),
    Spec("last_medium", "third", "discrete", 32, 16),
    Spec("last_direct", "third", "direct", 32),
    Spec("middle_small", "second", "discrete", 16, 16),
    Spec("middle_medium", "second", "discrete", 64, 64),
    Spec("middle_direct", "second", "direct", 64),
    Spec("first_small", "first", "discrete", 16, 16),
    Spec("first_medium", "first", "discrete", 64, 64),
    Spec("first_large", "first", "discrete", 64, 256),
    Spec("first_direct", "first", "direct", 64),
)
BY_NAME = {spec.name: spec for spec in SPECS}
LAYER_DIMS = {"first": (784, 300), "second": (300, 100),
              "third": (100, 30)}


def weight_coordinates(layer: str, incoming: int, outgoing: int) -> torch.Tensor:
    """Input-major coordinates, matching the transpose in ``weight``."""
    axes = []
    input_index = torch.arange(incoming, dtype=torch.float32)
    if layer == "first":
        axes.extend(((input_index // 28) / 27, (input_index % 28) / 27))
    else:
        axes.append(input_index / max(incoming - 1, 1))
    output_axis = torch.arange(outgoing, dtype=torch.float32) / max(outgoing - 1, 1)
    expanded = [axis[:, None].expand(incoming, outgoing) for axis in axes]
    expanded.append(output_axis[None, :].expand(incoming, outgoing))
    features = []
    for axis in expanded:
        features.append(axis)
        for frequency in (1, 3):
            features.append(torch.sin(2 * math.pi * frequency * axis))
            features.append(torch.cos(2 * math.pi * frequency * axis))
    return torch.stack(features, dim=-1).reshape(incoming * outgoing, -1)


class GeneratedLinear(nn.Module):
    def __init__(self, layer: str, mode: str, hidden: int,
                 num_values: int, seed: int) -> None:
        super().__init__()
        if mode not in ("discrete", "direct"):
            raise ValueError(mode)
        self.in_features, self.out_features = LAYER_DIMS[layer]
        self.mode = mode
        coordinates = weight_coordinates(layer, self.in_features, self.out_features)
        self.register_buffer("coordinates", coordinates, persistent=False)
        torch.manual_seed(seed + 101)
        output_size = num_values if mode == "discrete" else 1
        self.generator = nn.Sequential(
            nn.Linear(coordinates.shape[1], hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, output_size),
        )
        nn.init.normal_(self.generator[-1].weight, std=0.1)
        nn.init.zeros_(self.generator[-1].bias)
        self.values = (nn.Parameter(torch.linspace(-1, 1, num_values))
                       if mode == "discrete" else None)
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        self._calibrate_initial_weight()

    def weight(self) -> torch.Tensor:
        generated = self.generator(self.coordinates)
        if self.mode == "discrete":
            selected = self.values[generated.argmax(-1)]
            if self.training:
                # Same straight-through derivative as one_hot + soft -
                # soft.detach(), without materializing a huge one-hot matrix.
                soft_value = generated.softmax(-1) @ self.values.detach()
                flat = selected + soft_value - soft_value.detach()
            else:
                flat = selected
        else:
            flat = generated.squeeze(-1)
        return flat.reshape(self.in_features, self.out_features).T.contiguous()

    @torch.no_grad()
    def _calibrate_initial_weight(self) -> None:
        """Match a dense Glorot layer's initial mean and standard deviation."""
        desired_std = math.sqrt(2 / (self.in_features + self.out_features))
        was_training = self.training
        self.eval()
        try:
            for attempt in range(4):
                initial = self.weight()
                mean = initial.mean()
                std = initial.std(unbiased=False)
                if std > 1e-6:
                    factor = desired_std / std
                    if self.mode == "discrete":
                        self.values.sub_(mean).mul_(factor)
                    else:
                        last = self.generator[-1]
                        last.weight.mul_(factor)
                        last.bias.sub_(mean).mul_(factor)
                    return
                nn.init.normal_(self.generator[-1].weight, std=0.1 * (attempt + 2))
            raise RuntimeError("Generator assigned every coordinate the same weight")
        finally:
            self.train(was_training)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight(), self.bias)


def make_model(spec: Spec, seed: int) -> ImageSum:
    # Construct the same dense initialization first for every run. Only the
    # chosen layer is replaced, so the other layers start identically.
    model = ImageSum("paper_mlp", seed, paper_initialization=True,
                     padding_mode="masked")
    if spec.layer is not None:
        setattr(model, spec.layer, GeneratedLinear(
            spec.layer, spec.mode, spec.hidden, spec.values, seed))
    return model


def num_parameters(spec: Spec) -> int:
    if spec.layer is None:
        return 268_661
    incoming, outgoing = LAYER_DIMS[spec.layer]
    axes = 3 if spec.layer == "first" else 2
    input_width = 5 * axes
    output_width = spec.values if spec.mode == "discrete" else 1
    generator_count = ((input_width + 1) * spec.hidden +
                       (spec.hidden + 1) * spec.hidden +
                       (spec.hidden + 1) * output_width)
    codebook_count = spec.values if spec.mode == "discrete" else 0
    return 268_661 - (incoming * outgoing + outgoing) + generator_count + codebook_count + outgoing


def parse_epochs(value: str, last_epoch: int) -> list[int]:
    epochs = sorted({int(part) for part in value.split(",") if part})
    if any(epoch < 1 for epoch in epochs):
        raise ValueError("Probe epochs must be positive")
    return [epoch for epoch in epochs if epoch <= last_epoch]


def config_for(args: argparse.Namespace, spec: Spec, probes: list[int]) -> dict:
    return {"spec": asdict(spec), "seed": args.seed,
            "max_epochs": args.max_epochs, "train_sets": args.train_sets,
            "batch_size": args.batch_size, "pixel_scale": args.pixel_scale,
            "test_set_count": args.test_set_count,
            "probe_set_count": args.probe_set_count, "probe_epochs": probes,
            "sets_subdir": "sets_authors", "padding_mode": "masked",
            "optimizer": "Adam(lr=0.001, eps=0.001)", "loss": "MAE",
            "checkpoint_selection": "last"}


def atomic_torch_save(data: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def complete_result_matches(path: Path, config: dict) -> bool:
    if not path.exists():
        return False
    saved = json.loads(path.read_text())
    if saved["config"] != config:
        raise ValueError(f"Existing result has different settings: {path}; "
                         "choose another --results directory")
    return True


def run_one(args: argparse.Namespace, spec: Spec) -> int:
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    args.results.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.name}_seed{args.seed}"
    result_path = args.results / f"{stem}.json"
    progress_path = args.results / f"{stem}_progress.pt"
    probes = parse_epochs(args.probe_epochs, args.max_epochs)
    config = config_for(args, spec, probes)
    if complete_result_matches(result_path, config):
        print(f"SKIP completed {stem}", flush=True)
        return 0
    images = np.load(args.data_dir / "images.npy", mmap_mode="r")
    sets_dir = args.data_dir / "sets_authors"
    train = load_split(sets_dir / "train.npz")
    train = {key: value[:args.train_sets] for key, value in train.items()}
    validation = load_split(sets_dir / "validation.npz")
    probe_splits = ({length: {key: value[:args.probe_set_count]
                              for key, value in load_split(
                                  sets_dir / f"test_{length}.npz").items()}
                     for length in TEST_LENGTHS} if probes else {})
    model = make_model(spec, args.seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6)
    generator = np.random.default_rng(args.seed + 3_000_000)
    history = []
    start_epoch = 0
    elapsed_before = 0.0
    if progress_path.exists():
        saved = torch.load(progress_path, map_location=device, weights_only=False)
        if saved["config"] != config:
            raise ValueError(f"Checkpoint settings differ: {progress_path}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        generator.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device=device)
        history = saved["history"]
        start_epoch = saved["epoch"]
        elapsed_before = saved["elapsed_seconds"]
        print(f"RESUME {stem} from epoch {start_epoch}", flush=True)
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()

    def save_progress(epoch: int) -> None:
        data = {"config": config, "epoch": epoch,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "numpy_rng": generator.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "history": history,
                "elapsed_seconds": elapsed_before + time.monotonic() - started}
        atomic_torch_save(data, progress_path)

    steps_per_epoch = math.ceil(len(train["targets"]) / args.batch_size)
    for epoch in range(start_epoch + 1, args.max_epochs + 1):
        model.train()
        order = generator.permutation(len(train["targets"]))
        losses = []
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            x, mask, target = batch(images, train, ids, device, args.pixel_scale)
            prediction = model(x, mask)
            loss = (prediction - target).abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_metrics = evaluate(model, images, validation, device,
                                      args.batch_size, args.pixel_scale)
        scheduler.step(validation_metrics["mae"])
        row = {"epoch": epoch, "optimizer_steps": epoch * steps_per_epoch,
               "train_mae_batches": float(np.mean(losses)),
               "validation": validation_metrics,
               "learning_rate": optimizer.param_groups[0]["lr"],
               "elapsed_seconds": elapsed_before + time.monotonic() - started}
        if epoch in probes:
            row["probe"] = {str(length): evaluate(
                model, images, split, device, args.batch_size, args.pixel_scale)
                for length, split in probe_splits.items()}
        history.append(row)
        if epoch % 10 == 0 or epoch in probes or epoch == args.max_epochs:
            print(f"{stem} epoch={epoch} train_mae={row['train_mae_batches']:.4f} "
                  f"val_mae={validation_metrics['mae']:.4f} "
                  f"lr={row['learning_rate']:.6g}", flush=True)
        if epoch % args.save_every == 0 or epoch == args.max_epochs or stop_requested:
            save_progress(epoch)
        if stop_requested:
            print(f"STOPPED {stem} after epoch {epoch}; restart to resume", flush=True)
            return 130
    model.eval()
    validation_metrics = history[-1]["validation"]
    test_metrics = {}
    for length in TEST_LENGTHS:
        split = load_split(sets_dir / f"test_{length}.npz")
        split = {key: value[:args.test_set_count] for key, value in split.items()}
        test_metrics[str(length)] = evaluate(
            model, images, split, device, args.batch_size, args.pixel_scale)
    result = {"config": config,
              "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "training_seconds": elapsed_before + time.monotonic() - started,
              "validation": validation_metrics, "test": test_metrics,
              "history": history}
    if result["parameter_count"] != num_parameters(spec):
        raise AssertionError("Parameter-count formula does not match model")
    temporary = result_path.with_name(result_path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, result_path)
    print(f"DONE {stem} accuracy50={test_metrics['50']['exact_round_accuracy']:.4f} "
          f"mae50={test_metrics['50']['mae']:.4f}", flush=True)
    return 0


def idle_devices() -> list[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    result = []
    for line in output.splitlines():
        index, utilization = (piece.strip() for piece in line.split(","))
        if int(utilization) == 0:
            result.append(int(index))
    return result


def run_all(args: argparse.Namespace, specs: list[Spec]) -> int:
    devices = idle_devices() if args.devices == "auto" else [
        int(item) for item in args.devices.split(",") if item.strip()]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Supply distinct idle GPU indices with --devices")
    probes = parse_epochs(args.probe_epochs, args.max_epochs)
    pending = [spec for spec in specs if not complete_result_matches(
        args.results / f"{spec.name}_seed{args.seed}.json",
        config_for(args, spec, probes))]
    print(f"Selected devices: {devices}; pending runs: {len(pending)}", flush=True)
    active: dict[int, tuple[Spec, subprocess.Popen, object]] = {}
    interrupted = False
    failed = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while pending or active:
            for gpu in devices:
                if interrupted or failed or not pending or gpu in active:
                    continue
                spec = pending.pop(0)
                stem = f"{spec.name}_seed{args.seed}"
                logs = args.results / "logs"
                logs.mkdir(parents=True, exist_ok=True)
                log = (logs / f"{stem}.log").open("a")
                command = [sys.executable, "-m", __package__ + ".placement_sweep",
                           "--run-id", spec.name, "--device", f"cuda:{gpu}",
                           "--seed", str(args.seed), "--data-dir", str(args.data_dir),
                           "--results", str(args.results), "--max-epochs", str(args.max_epochs),
                           "--train-sets", str(args.train_sets),
                           "--batch-size", str(args.batch_size),
                           "--pixel-scale", str(args.pixel_scale),
                           "--test-set-count", str(args.test_set_count),
                           "--probe-set-count", str(args.probe_set_count),
                           "--probe-epochs", args.probe_epochs,
                           "--save-every", str(args.save_every)]
                process = subprocess.Popen(command, stdout=log,
                                           stderr=subprocess.STDOUT,
                                           start_new_session=True)
                active[gpu] = (spec, process, log)
                print(f"START {stem} on GPU {gpu}", flush=True)
            for gpu, (spec, process, log) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                log.close()
                del active[gpu]
                print(f"END {spec.name}_seed{args.seed} GPU {gpu} exit={status}",
                      flush=True)
                if status != 0:
                    failed = True
            if interrupted or failed:
                break
            time.sleep(0.5)
    finally:
        if active:
            for _spec, process, _log in active.values():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 20
            for _spec, process, log in active.values():
                try:
                    process.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                log.close()
    return 130 if interrupted else (1 if failed else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", choices=BY_NAME)
    parser.add_argument("--configs", default="all",
                        help="Comma-separated run IDs, or all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--devices", default="auto",
                        help="auto selects GPUs with 0%% utilization; or comma-separated indices")
    parser.add_argument("--device", default="cpu",
                        help="Device for one --run-id invocation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mnist8m"))
    parser.add_argument("--results", type=Path, default=Path(
        "deepsets_z/mnist8m/results_placement_sweep"))
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--train-sets", type=int, default=148_148)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pixel-scale", type=float, default=255)
    parser.add_argument("--test-set-count", type=int, default=10_000)
    parser.add_argument("--probe-set-count", type=int, default=1_000)
    parser.add_argument("--probe-epochs", default="1,2,5,10,20,50,100,200,350,500")
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()
    if args.max_epochs < 1 or args.batch_size < 1 or args.save_every < 1:
        parser.error("Epochs, batch size, and save frequency must be positive")
    if (args.pixel_scale <= 0 or args.train_sets < 1 or
            args.test_set_count < 1 or args.probe_set_count < 1):
        parser.error("Pixel scale and set counts must be positive")
    if args.run_id:
        return run_one(args, BY_NAME[args.run_id])
    names = list(BY_NAME) if args.configs == "all" else [
        name.strip() for name in args.configs.split(",") if name.strip()]
    if not names or len(set(names)) != len(names) or any(name not in BY_NAME for name in names):
        parser.error("--configs must contain unique known run IDs")
    specs = [BY_NAME[name] for name in names]
    if args.dry_run:
        for spec in specs:
            print(f"{spec.name:15} layer={str(spec.layer):6} "
                  f"mode={spec.mode:8} hidden={spec.hidden:3} "
                  f"values={spec.values:3} parameters={num_parameters(spec)}")
        return 0
    return run_all(args, specs)


if __name__ == "__main__":
    raise SystemExit(main())
