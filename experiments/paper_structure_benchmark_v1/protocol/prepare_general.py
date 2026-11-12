"""Numerical families for the general local/quartet/random benchmark."""

import numpy as np


def general_index(rng, variant):
    """Use the three geometries of the original general synthetic benchmark."""
    if variant == "random":
        index = (rng.uniform(size=(16, 16)) < 0.55) * rng.integers(1, 5, size=(16, 16))
        index *= rng.choice((-1, 1), size=(16, 16))
    else:
        index = np.zeros((16, 16), dtype=np.int16)
        for point in range(8):
            for other in range(8):
                if variant == "local":
                    sign = int(point == other)
                else:
                    sign = (-1 if point == other else 1) * int(point // 4 == other // 4)
                index[2 * point : 2 * point + 2, 2 * other : 2 * other + 2] = (
                    sign * np.arange(1, 5).reshape(2, 2)
                )
    return index[rng.permutation(16)][:, rng.permutation(16)].astype(np.int16)


def general_numbers(rng, variant, replicate, count, outputs):
    code = rng.normal(size=(count, 4))
    if variant == "quartet_mix":
        code /= 2
    if replicate == 0:
        bias = np.repeat(rng.normal(0, 0.8, (count, 1)), 16, axis=1)
    else:
        bias = rng.normal(0, 1.7, (count, 16)) + rng.uniform(-1, 1, (count, 1))
    head = rng.normal(size=(count, 16, outputs))
    offset = rng.normal(0, 0.3, (count, outputs))
    return code, bias, head, offset
