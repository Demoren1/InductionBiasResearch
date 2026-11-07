"""Create fixed Aug-Omniglot episodes from the standard Vinyals class split."""

import argparse
import functools
import json
import shutil
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

SPLIT_URL = "https://raw.githubusercontent.com/tristandeleu/pytorch-meta/master/torchmeta/datasets/assets/omniglot"


@functools.lru_cache(maxsize=8192)
def image_at(path):
    with Image.open(path) as image:
        return ImageOps.invert(image.convert("L")).copy()


def classes_for(raw, split):
    directory = raw.parent / "omniglot_splits"
    directory.mkdir(exist_ok=True)
    path = directory / f"{split}.json"
    if not path.exists():
        urllib.request.urlretrieve(f"{SPLIT_URL}/{split}.json", path)
    definitions = json.loads(path.read_text())
    result = []
    for part, alphabets in definitions.items():
        for alphabet, characters in alphabets.items():
            for character in characters:
                folder = raw / "omniglot-py" / f"images_{part}" / alphabet / character
                files = tuple(sorted(folder.glob("*.png")))
                if len(files) != 20:
                    raise ValueError(f"Expected 20 images in {folder}")
                for angle in (0, 90, 180, 270):
                    result.append((files, angle))
    return result, path


def episodes(raw, split, count, seed):
    classes, path = classes_for(raw, split)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    augment = transforms.Compose(
        [
            transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(30, interpolation=InterpolationMode.BILINEAR),
        ]
    )
    panels = {name: {"x": [], "y": []} for name in ("support", "query")}
    identifiers = []
    for task in range(count):
        task_panels = {name: {"x": [], "y": []} for name in panels}
        chosen = rng.choice(len(classes), 5, replace=False)
        identifiers.append(chosen.tolist())
        for label, number in enumerate(chosen):
            files, angle = classes[number]
            for index, position in enumerate(rng.permutation(20)):
                image = image_at(str(files[position])).rotate(angle)
                image = image.resize((28, 28), resample=Image.Resampling.BILINEAR)
                panel = "support" if index < 5 else "query"
                if panel == "query":
                    image = augment(image)
                task_panels[panel]["x"].append(
                    np.asarray(image, dtype=np.uint8).reshape(-1)
                )
                task_panels[panel]["y"].append(np.eye(5, dtype=np.float32)[label])
        for name, arrays in task_panels.items():
            order = rng.permutation(len(arrays["x"]))
            for kind in ("x", "y"):
                panels[name][kind].append(np.stack(arrays[kind])[order])
    packed = {
        f"{name}_{kind}": np.stack(value)
        for name, arrays in panels.items()
        for kind, value in arrays.items()
    }
    return packed, {
        "split": split,
        "base_classes": len(classes) // 4,
        "virtual_rotated_classes": len(classes),
        "class_split_sha256": digest(path),
        "task_classes": identifiers,
    }


def prepare(root, raw, sources, validation, targets):
    root.mkdir(parents=True, exist_ok=False)
    for directory in (
        "public",
        "validation",
        "private",
        "logs",
        "source",
        "fits",
        "metrics",
        "receipts",
    ):
        (root / directory).mkdir()
    seed = 26110772013
    provenance = {}
    for name, split, count, offset in (
        ("source", "train", sources, 0),
        ("validation", "val", validation, 100003),
        ("target", "test", targets, 200003),
    ):
        data, details = episodes(raw, split, count, seed + offset)
        provenance[name] = details
        if name == "source":
            selected_x, selected_y, structure_x, structure_y = [], [], [], []
            for task in range(count):
                label = data["query_y"][task].argmax(1)
                first = np.concatenate(
                    [np.flatnonzero(label == group)[:5] for group in range(5)]
                )
                second = np.concatenate(
                    [np.flatnonzero(label == group)[5:] for group in range(5)]
                )
                selected_x.append(data["query_x"][task, first])
                selected_y.append(data["query_y"][task, first])
                structure_x.append(data["query_x"][task, second])
                structure_y.append(data["query_y"][task, second])
            np.savez_compressed(
                root / "public" / "source.npz",
                support_x=data["support_x"],
                support_y=data["support_y"],
                numerical_x=np.stack(selected_x),
                numerical_y=np.stack(selected_y),
                structure_x=np.stack(structure_x),
                structure_y=np.stack(structure_y),
            )
        else:
            directory = root / ("validation" if name == "validation" else "private")
            np.savez_compressed(directory / f"{name}.npz", **data)
        print(
            json.dumps(
                {
                    "prepared": name,
                    "tasks": count,
                    "base_classes": details["base_classes"],
                }
            ),
            flush=True,
        )
    save_json(root / "private" / "provenance.json", provenance)
    save_json(
        root / "protocol.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "dataset": "Aug-Omniglot, fixed-cohort fully connected comparison",
            "literature": "https://arxiv.org/pdf/2007.02933",
            "class_split": "Standard Vinyals 1028/172/423 base classes; four 90-degree virtual classes per base character",
            "inputs": 784,
            "hidden": 16,
            "outputs": 5,
            "independent_code_grid": [8, 16, 32, 64],
            "loss": "cross_entropy",
            "source_tasks": sources,
            "validation_tasks": validation,
            "target_tasks": targets,
            "ways": 5,
            "shots": 5,
            "queries_per_class": 15,
            "source_query_partition": {
                "numerical_per_class": 5,
                "structure_per_class": 10,
            },
            "query_only_augmentation": "Resize28, RandomResizedCrop28(scale .8..1), VFlip .5, HFlip .5, Rotation±30 bilinear",
            "pixel_encoding": "Inverted grayscale uint8; learners divide by255, no pretrained encoder",
            "comparison_scope": "Same fixed source episode cohort for all rerun methods. Published MSR CNN/init-transfer results are separate context, not matched FC/structure-only numbers.",
            "target_policy": "Fresh numerical parameters; support only. Target queries stay private until all selected models/predictions are frozen.",
            "unique_information": "Source cohort size and class coverage differ from the author's 60000-episode-update experiment and must be disclosed.",
            "code_sha256": digest(Path(__file__)),
            "public_sha256": digest(root / "public" / "source.npz"),
        },
    )
    shutil.copy2(__file__, root / "prepare_executed_code.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--sources", type=int, default=128)
    parser.add_argument("--validation", type=int, default=128)
    parser.add_argument("--targets", type=int, default=1000)
    args = parser.parse_args()
    prepare(
        args.root.resolve(),
        args.raw.resolve(),
        args.sources,
        args.validation,
        args.targets,
    )


if __name__ == "__main__":
    main()
