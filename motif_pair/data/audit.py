"""Audits that rule out simple data-generation shortcuts.

The primary audit trains a linear classifier on raw position bits plus the
global one-count.  It must remain near chance: all information about the label
is in the relative A/B offset, which a linear feature map cannot express.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

try:
    from motif_pair.data.generate import configure_compute_device, compute_device, sample_balanced
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.generate import configure_compute_device, compute_device, sample_balanced  # type: ignore


def shortcut_features(dataset: dict) -> torch.Tensor:
    """Raw 0/1 positions concatenated with an explicit global one-count."""
    bits = (dataset["x"] + 1.0) / 2.0
    one_count = dataset["ones_count"].to(torch.float32).unsqueeze(1)
    return torch.cat((bits, one_count), dim=1)


def linear_shortcut_accuracy(task: str, *, n_samples: int = 4_096,
                             seed: int = 42, steps: int = 400) -> float:
    """Fit/evaluate a deterministic linear raw-bit shortcut probe."""
    train = sample_balanced(task, n_samples, seed=seed)
    test = sample_balanced(task, n_samples, seed=seed + 1)
    x_train, y_train = shortcut_features(train), train["y"]
    x_test, y_test = shortcut_features(test), test["y"]
    # Count is intentionally explicit despite being linearly recoverable from
    # bits.  Standardisation makes its audit coefficient numerically stable.
    mean, std = x_train.mean(dim=0, keepdim=True), x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    device = compute_device()
    fork_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed + 10_000_019)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + 10_000_019)
        model = torch.nn.Linear(x_train.size(1), 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(steps):
            logits = model(x_train).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, y_train)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        prediction = (model(x_test).squeeze(1) >= 0).to(y_test.dtype)
    return float((prediction == y_test).float().mean())


def audit_split(split_path: Path, *, n_samples: int = 4_096,
                seed: int = 42, steps: int = 400,
                device: str | None = None) -> dict:
    """Run the raw-bit linear audit over every held-out task in a manifest."""
    if device is not None:
        configure_compute_device(device)
    payload = json.loads(split_path.read_text())
    tasks = payload.get("test_tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("split must contain nonempty test_tasks")
    accuracies = {
        task: linear_shortcut_accuracy(task, n_samples=n_samples,
                                       seed=seed + ordinal * 100_000, steps=steps)
        for ordinal, task in enumerate(tasks)
    }
    values = list(accuracies.values())
    return {
        "split_path": str(split_path.resolve()),
        "seed": seed,
        "n_samples": n_samples,
        "steps": steps,
        "linear_raw_bits_plus_count_accuracy": accuracies,
        "mean_accuracy": sum(values) / len(values),
        "max_accuracy": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=4_096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    result = audit_split(args.split, n_samples=args.n_samples, seed=args.seed,
                         steps=args.steps, device=args.device)
    text = json.dumps(result, indent=2) + "\n"
    if args.out is None:
        print(text, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"audit -> {args.out}; mean={result['mean_accuracy']:.4f}, max={result['max_accuracy']:.4f}")


if __name__ == "__main__":
    main()
