"""Known-image classification with identically distributed distractor images."""

import numpy as np
import torch
from torch.nn import functional as F


def image_table(dataset):
    return F.avg_pool2d(dataset.data[:, None].float() / 255, 2).flatten(1)


def layout(seed, blocks=8):
    return np.random.default_rng(seed).permutation(196 * blocks)


def observations(pixels, main_ids, donor_pool, permutation, seed, blocks=8):
    rng = np.random.default_rng(seed)
    donor_ids = rng.choice(np.asarray(donor_pool), (len(main_ids), blocks - 1))
    ids = np.c_[np.asarray(main_ids), donor_ids]
    # Every block uses the same empirical image distribution. Only its label
    # relation differs. The learner is not told which coordinates form block 0.
    result = pixels[torch.tensor(ids)].flatten(1)[:, torch.tensor(permutation)]
    return result, ids
