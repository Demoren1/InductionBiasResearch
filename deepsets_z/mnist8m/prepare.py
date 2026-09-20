"""Convert downloaded MNIST8m images and labels into NumPy memory maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


SOURCE = "https://huggingface.co/datasets/MarveenLee/MNIST8M"


def convert_images(raw: Path, converted: Path, *, chunk: int = 100_000) -> int:
    with h5py.File(raw, "r") as h5:
        source = h5["MNIST8M"]
        if source.shape[0] == 784:
            n = source.shape[1]
            transposed = True
        elif source.shape[1] == 784:
            n = source.shape[0]
            transposed = False
        else:
            raise ValueError(f"Unexpected image shape {source.shape}")
        if converted.exists():
            existing = np.load(converted, mmap_mode="r")
            if existing.shape == (n, 784) and existing.dtype == np.uint8:
                return n
            raise ValueError(f"Unexpected converted image shape {existing.shape}")
        temporary = converted.with_suffix(".partial.npy")
        destination = np.lib.format.open_memmap(temporary, mode="w+", dtype="uint8",
                                                shape=(n, 784))
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            block = source[:, start:stop].T if transposed else source[start:stop]
            destination[start:stop] = block
            destination.flush()
            print(f"images {stop}/{n}", flush=True)
        del destination
        temporary.replace(converted)
        return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mnist8m"))
    args = parser.parse_args()
    directory = args.data_dir
    n = convert_images(directory / "MNIST8M_data.h5", directory / "images.npy")
    labels_path = directory / "labels.npy"
    if labels_path.exists():
        labels = np.load(labels_path, mmap_mode="r")
    else:
        labels = np.fromfile(directory / "labels.txt", sep=" ", dtype=np.uint8)
        if labels.shape != (n,) or labels.max() > 9:
            raise ValueError(f"Unexpected labels shape/range: {labels.shape}")
        np.save(labels_path, labels)
    if n != 8_100_000 or len(labels) != n:
        raise ValueError(f"Expected 8.1M aligned images/labels, got {n}/{len(labels)}")
    manifest = {
        "source": SOURCE, "images": n, "source_file": "MNIST8M_data.h5",
        "pixel_range": [0, 255], "label_range": [0, 9],
    }
    (directory / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"images": n, "data_dir": str(directory)}), flush=True)


if __name__ == "__main__":
    main()
