"""Read verified W/G channels and IMP targets from source-family cards."""

import torch

from .io import checked_load
from .problem import HIDDEN, INPUTS

ROLES = ("train", "validation", "test")


def load_source(bank, manifest, device):
    """Only source-family cards and their IMP masks enter the trainer."""
    tasks = sorted(
        [
            t
            for t in manifest["completed_tasks"].values()
            if t["split"] == "train"
        ],
        key=lambda task: task["task"],
    )
    grouped = {role: {key: [] for key in ("x", "mask")} for role in ROLES}
    for task in tasks:
        path = bank / task["path"] / "cards.pt"
        card = checked_load(path, task["hashes"]["cards.pt"])
        features = card["features"] / card["feature_scale"]
        masks = card["targets"].float()
        if features.shape[1:] != (INPUTS, HIDDEN, 2):
            raise ValueError("Unexpected bank feature shape")
        if not torch.all(masks.sum((1, 2)) == 32):
            raise ValueError("IMP targets must have exactly 32 edges")
        for role in ROLES:
            indices = [
                i
                for i, name in enumerate(card["network_roles"])
                if name == role
            ]
            if not indices:
                raise ValueError(f"Missing {role} cards in {task['task']}")
            for key, value in [
                ("x", features),
                ("mask", masks),
            ]:
                grouped[role][key].append(value[indices])
    if not tasks:
        raise ValueError("No source tasks in bank")
    data = {
        role: {
            key: torch.stack(values).to(device)
            for key, values in groups.items()
        }
        for role, groups in grouped.items()
    }
    return [task["task"] for task in tasks], data
