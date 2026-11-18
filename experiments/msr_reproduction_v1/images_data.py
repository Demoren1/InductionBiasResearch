"""Official class splits and independently sampled augmented image tasks."""

import argparse
import io
import json
import pickle
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

from .linear import save_json


def prepare_mini(root):
    folder = root / "images" / "miniimagenet"
    folder.mkdir(parents=True, exist_ok=True)
    archive = root / "miniimagenet" / "mini-imagenet.tar.gz"
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if not name.startswith("mini-imagenet-cache-") or not name.endswith(".pkl"):
                continue
            split = name.removeprefix("mini-imagenet-cache-").removesuffix(".pkl")
            panel = pickle.load(tar.extractfile(member), encoding="latin1")
            classes = sorted(panel["class_dict"])
            data = np.lib.format.open_memmap(folder / f"{split}.npy", mode="w+", dtype="uint8",
                                             shape=(len(classes), 600, 3, 84, 84))
            for position, label in enumerate(classes):
                images = panel["image_data"][panel["class_dict"][label]]
                data[position] = images.transpose(0, 3, 1, 2)
            data.flush()
            save_json(folder / f"{split}_classes.json", classes)
            print("MiniImageNet", split, data.shape, flush=True)
            del data, panel
    save_json(folder / "registration.json", {
        "dataset": "TorchMeta/Ren MiniImageNet, Ravi class split",
        "classes": {"train": 64, "val": 16, "test": 20},
        "source_archive_md5": "b38f1eb4251fb9459ecc8e7febf9b2eb",
        "source_url": "https://drive.google.com/uc?export=download&id=16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY",
        "image_size": 84, "normalization": "ToTensor [0,1]", "class_rotations": False,
    })


def prepare_omniglot(root):
    folder = root / "images" / "omniglot"
    folder.mkdir(parents=True, exist_ok=True)
    previous = Path("artifacts/local/omniglot_single_task_2026-11-14/raw/omniglot-py")
    archives = {name: zipfile.ZipFile(previous / f"images_{name}.zip")
                for name in ("background", "evaluation")}
    for split in ("train", "val", "test"):
        labels = json.loads((root / "references" / f"omniglot_{split}.json").read_text())
        classes = sorted((name, alphabet, character) for name, alphabets in labels.items()
                         for alphabet, characters in alphabets.items() for character in characters)
        data = np.lib.format.open_memmap(folder / f"{split}.npy", mode="w+", dtype="uint8",
                                         shape=(len(classes), 20, 1, 28, 28))
        for position, (name, alphabet, character) in enumerate(classes):
            name = name.removeprefix("images_")
            archive = archives[name]
            prefix = f"images_{name}/{alphabet}/{character}/"
            members = sorted(member for member in archive.namelist()
                             if member.startswith(prefix) and member.endswith(".png"))
            for index, member in enumerate(members):
                image = ImageOps.invert(Image.open(io.BytesIO(archive.read(member))).convert("L"))
                data[position, index, 0] = np.asarray(image.resize((28, 28), Image.Resampling.BILINEAR))
        data.flush()
        save_json(folder / f"{split}_classes.json", classes)
        print("Omniglot", split, data.shape, flush=True)
    for archive in archives.values():
        archive.close()
    save_json(folder / "registration.json", {
        "dataset": "TorchMeta Vinyals Omniglot class split",
        "classes": {"train": 1028, "val": 172, "test": 423}, "image_size": 28,
        "normalization": "ImageOps.invert, ToTensor [0,1]", "class_rotations": [0, 90, 180, 270],
        "support_query_classes": "Common within task, disjoint across train/val/test",
    })


def augment_query(images, generator):
    """Independent torchvision crop/flip/rotate geometry for each query image.

    GPU bilinear transforms avoid a CPU data pipeline limiting meta-batches.
    Integer crop boxes and 8-bit rounding match the PIL convention. This is a
    disclosed modern tensor implementation, not bitwise PIL reproduction.
    """
    count, _, height, width = images.shape
    options = {"device": images.device, "generator": generator}
    crop_h = torch.zeros(count, device=images.device)
    crop_w = crop_h.clone()
    pending = torch.ones(count, device=images.device, dtype=torch.bool)
    for _ in range(10):
        area = height * width * (0.8 + 0.2 * torch.rand(count, **options))
        aspect = torch.exp(torch.empty(count, device=images.device).uniform_(np.log(0.75), np.log(4/3), generator=generator))
        w, h = (area * aspect).sqrt().round(), (area / aspect).sqrt().round()
        accepted = pending & (w <= width) & (h <= height) & (w > 0) & (h > 0)
        crop_w[accepted], crop_h[accepted] = w[accepted], h[accepted]
        pending &= ~accepted
    crop_h[pending], crop_w[pending] = height, width
    left = (torch.rand(count, **options) * (width - crop_w + 1)).floor()
    top = (torch.rand(count, **options) * (height - crop_h + 1)).floor()
    theta = images.new_zeros((count, 2, 3))
    theta[:, 0, 0], theta[:, 1, 1] = crop_w / width, crop_h / height
    theta[:, 0, 2] = (2 * left + crop_w) / width - 1
    theta[:, 1, 2] = (2 * top + crop_h) / height - 1
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    # Resize samples the cropped image: its border, rather than pixels outside
    # the crop, supplies the half-pixel interpolation boundary.
    minimum_x = (2 * left + 1) / width - 1
    maximum_x = (2 * (left + crop_w) - 1) / width - 1
    minimum_y = (2 * top + 1) / height - 1
    maximum_y = (2 * (top + crop_h) - 1) / height - 1
    grid[..., 0] = torch.maximum(torch.minimum(grid[..., 0], maximum_x[:, None, None]),
                                 minimum_x[:, None, None])
    grid[..., 1] = torch.maximum(torch.minimum(grid[..., 1], maximum_y[:, None, None]),
                                 minimum_y[:, None, None])
    images = F.grid_sample(images, grid, align_corners=False, padding_mode="border")
    images = (images * 255).round() / 255
    angle = (torch.rand(count, **options) - 0.5) * (np.pi / 3)
    horizontal = torch.where(torch.rand(count, **options) < 0.5, -1, 1)
    vertical = torch.where(torch.rand(count, **options) < 0.5, -1, 1)
    theta.zero_()
    theta[:, 0, 0], theta[:, 0, 1] = angle.cos() * horizontal, -angle.sin() * vertical
    theta[:, 1, 0], theta[:, 1, 1] = angle.sin() * horizontal, angle.cos() * vertical
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return (F.grid_sample(images, grid, align_corners=False) * 255).round() / 255


class Episodes:
    def __init__(self, folder, split, ways, shots, device, seed, query=15):
        self.data = torch.as_tensor(np.load(folder / f"{split}.npy"), device=device)
        self.ways, self.shots, self.query = ways, shots, query
        self.omniglot = folder.name == "omniglot"
        self.generator = torch.Generator(device=device).manual_seed(seed)

    @torch.no_grad()
    def next(self, batch):
        data, ways, shots, query = self.data, self.ways, self.shots, self.query
        options = {"device": data.device, "generator": self.generator}
        classes = torch.stack([torch.randperm(len(data) * (4 if self.omniglot else 1), **options)[:ways]
                               for _ in range(batch)])
        base = classes // 4 if self.omniglot else classes
        positions = torch.rand(batch, ways, data.shape[1], **options).argsort(-1)[..., :shots + query]
        images = data[base[..., None], positions].float() / 255
        if self.omniglot:
            rotations = classes % 4
            for rotation in (1, 2, 3):
                mask = rotations == rotation
                images[mask] = images[mask].rot90(rotation, (-2, -1))
        support = images[:, :, :shots].flatten(1, 2)
        target = images[:, :, shots:].flatten(1, 2)
        shape = target.shape
        target = augment_query(target.flatten(0, 1), self.generator).reshape(shape)
        labels = torch.arange(ways, device=data.device)
        sy = labels.repeat_interleave(shots).expand(batch, -1)
        qy = labels.repeat_interleave(query).expand(batch, -1)
        return support, sy, target, qy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    prepare_omniglot(args.root.resolve())
    prepare_mini(args.root.resolve())


if __name__ == "__main__":
    main()
