"""Prepare an explicitly reduced, multitask MNIST image-sum benchmark."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import MNIST

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json


def panel(images, labels, mapping, count, rng, permutation):
    selected = rng.integers(len(images), size=(len(mapping), count, 2))
    x = images[selected].reshape(len(mapping), count, -1)[..., permutation]
    y = np.take_along_axis(mapping[:, None, :], labels[selected], axis=2).sum(
        -1, keepdims=True
    )
    return x, y.astype(np.float32), selected


def prepare(root, raw):
    root.mkdir(parents=True, exist_ok=False)
    for name in (
        "public",
        "validation",
        "private",
        "source",
        "fits",
        "metrics",
        "receipts",
        "logs",
    ):
        (root / name).mkdir()
    train = MNIST(str(raw), train=True, download=False)
    test = MNIST(str(raw), train=False, download=False)
    images = [
        F.adaptive_avg_pool2d(data.data[:, None].float(), (7, 7))[:, 0]
        .round()
        .to(torch.uint8)
        .numpy()
        .reshape(-1, 49)
        for data in (train, test)
    ]
    labels = [data.targets.numpy() for data in (train, test)]
    rng = np.random.default_rng(26110778013)
    order = rng.permutation(len(images[0]))
    pools = {
        "source": order[:40000],
        "validation": order[40000:50000],
        "target": order[50000:],
    }
    permutation = rng.permutation(98)
    metadata = {}
    for name, tasks, counts in (
        ("source", 16, (4096, 512, 1024)),
        ("validation", 32, (512, 256)),
        ("target", 32, (2048, 2048)),
    ):
        mapping = rng.normal(size=(tasks, 10))
        data, provenance = {}, {}
        for panel_name, count in zip(
            ("support", "numerical", "structure")
            if name == "source"
            else ("support", "query"),
            counts,
        ):
            pool = pools[name]
            px, py = images[0][pool], labels[0][pool]
            if name == "target" and panel_name == "query":
                px, py = images[1], labels[1]
            x, y, selected = panel(px, py, mapping, count, rng, permutation)
            data[f"{panel_name}_x"], data[f"{panel_name}_y"] = x, y
            provenance[panel_name] = selected
            if name == "source":
                np.savez_compressed(root / "public" / f"{panel_name}.npz", x=x, y=y)
        if name != "source":
            folder = root / ("validation" if name == "validation" else "private")
            np.savez_compressed(folder / f"{name}.npz", **data)
        np.savez_compressed(
            root / "private" / f"{name}_generation.npz", mapping=mapping, **provenance
        )
        metadata[name] = {
            "tasks": tasks,
            "image_pool": "MNIST train disjoint 40k/10k/10k; target query uses official test10k",
        }
    np.savez_compressed(
        root / "private" / "coordinate_permutation.npz",
        permutation=permutation,
        image_pool_order=order,
    )
    save_json(
        root / "protocol.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "dataset": "Paired MNIST digit-function sums, reduced Deep-Sets-inspired transfer benchmark",
            "literature": "https://arxiv.org/pdf/1703.06114",
            "inputs": 98,
            "hidden": 32,
            "outputs": 1,
            "independent_code_grid": [16, 32, 64, 128],
            "loss": "mse",
            "source_tasks": 16,
            "validation_tasks": 32,
            "target_tasks": 32,
            "pixel_encoding": "7x7 average-pool uint8 per digit; two flattened images, shared hidden random coordinate permutation; divide255 only",
            "task_family": "Each task maps ten digits to independent sampled scalar values and sums the two image contributions; task mappings and coordinate roles are hidden from learners",
            "comparison_scope": "A reduced multitask extension, not the published variable-size M=5..50 InfiMNIST benchmark. No published scores are directly comparable.",
            "target_policy": "Only I/U transfers. Numeric parameters fresh; support-only target budget selected on separate validation tasks; official MNIST test responses remain sealed until predictions are frozen.",
            "partitions": metadata,
            "code_sha256": digest(Path(__file__)),
            "public_sha256": {
                p.name: digest(p) for p in (root / "public").glob("*.npz")
            },
        },
    )
    shutil.copy2(__file__, root / "prepare_executed_code.py")
    print(json.dumps({"prepared": str(root)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.raw.resolve())


if __name__ == "__main__":
    main()
