"""Known-image classification with identically distributed distractor images."""

import numpy as np
import torch
from torch.nn import functional as F


def image_table(dataset):
    return F.avg_pool2d(dataset.data[:, None].float() / 255, 2).flatten(1)


def layout(seed, blocks=8):
    return np.random.default_rng(seed).permutation(196 * blocks)


def observations(
    pixels, main_ids, donor_pool, permutation, seed, blocks=8, donor_labels=None
):
    rng = np.random.default_rng(seed)
    donor_pool = np.asarray(donor_pool)
    if donor_labels is None:
        donor_ids = rng.choice(donor_pool, (len(main_ids), blocks - 1))
    else:
        labels = np.asarray(donor_labels)[donor_pool]
        classes = np.unique(labels)
        if len(main_ids) % len(classes):
            raise ValueError("Every block must have the same balanced class counts")
        per_class = len(main_ids) // len(classes)
        columns = []
        for _ in range(blocks - 1):
            chosen = np.concatenate(
                [
                    rng.choice(donor_pool[labels == c], per_class, replace=False)
                    for c in classes
                ]
            )
            columns.append(rng.permutation(chosen))
        donor_ids = np.stack(columns, axis=1)
    ids = np.c_[np.asarray(main_ids), donor_ids]
    # Balanced sampling gives each block the same class counts and no repeated
    # image IDs within a block. Only block 0 has a relation to the row's label.
    result = pixels[torch.tensor(ids)].flatten(1)[:, torch.tensor(permutation)]
    return result, ids
