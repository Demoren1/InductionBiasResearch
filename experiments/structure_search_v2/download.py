"""Download the two official vision datasets with torchvision checksum checks."""

import hashlib
import json
from pathlib import Path

from torchvision.datasets import MNIST, FashionMNIST

root = Path("experiments/structure_search_v2/datasets")
FashionMNIST.mirrors = [
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
]
manifest = {}
for dataset in (MNIST, FashionMNIST):
    for train in (True, False):
        data = dataset(root=root, train=train, download=True)
        manifest[f"{dataset.__name__}/{'train' if train else 'test'}"] = {
            "images": len(data),
            "shape": list(data.data.shape),
            "pixels_sha256": hashlib.sha256(data.data.numpy().tobytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(data.targets.numpy().tobytes()).hexdigest(),
        }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
