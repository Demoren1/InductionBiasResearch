"""Public observation schema, signed indexator representation and provenance."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

PUBLIC_SIZES = frozenset(("inputs", "hidden", "codes", "outputs"))


def load_sizes(root, case):
    sizes = json.loads((Path(root) / "public" / f"{case}.json").read_text())
    if set(sizes) != PUBLIC_SIZES:
        raise ValueError("Learner metadata must contain only D/H/K/O")
    if any(type(value) is not int or value <= 0 for value in sizes.values()):
        raise ValueError("Public sizes must be positive integers")
    return sizes


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def canonical_index(index, codes):
    best, result = None, None
    for permutation in itertools.permutations(range(1, codes + 1)):
        for signs in itertools.product((-1, 1), repeat=codes):
            labels = np.r_[0, np.asarray(permutation) * signs]
            renamed = labels[np.abs(index)] * np.sign(index)
            renamed = renamed[:, np.lexsort(renamed[::-1])]
            key = renamed.tobytes()
            if best is None or key < best:
                best, result = key, renamed.copy()
    return result.astype(np.int16)
