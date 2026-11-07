"""Known structural blocks, held-out combinations, and bank recombination."""

from itertools import product

import numpy as np
import torch

from .masks import assignments, canonical_mask
from .problem import HIDDEN, INPUTS, reference


def block_variants(generator, count, offset):
    """Four anchored neurons partition eight variable inputs into pairs."""
    result, codes = [], []
    for _ in range(100000):
        groups = generator.permutation(8).reshape(4, 2)
        code = np.empty(8, dtype=int)
        for column, pair in enumerate(groups):
            code[pair] = column
        if any(np.count_nonzero(code == old) > 4 for old in codes):
            continue
        result.append((groups + offset).tolist())
        codes.append(code)
        if len(result) == count:
            return result
    raise ValueError("Too many block variants for the separation constraint")


def families(config):
    """Every block occurs in train; complete validation/test masks do not."""
    count = config["composition"]["variants"]
    if count != 6 or config["families"] != {
        "train": 24,
        "validation": 6,
        "test": 6,
    }:
        raise ValueError("This design uses six variants and a 24/6/6 split")
    generator = np.random.default_rng(config["seed"])
    blocks = {
        "a": block_variants(generator, count, 16),
        "b": block_variants(generator, count, 24),
    }
    result = []
    for split in config["families"]:
        for a, b in product(range(count), repeat=2):
            offset = (b - a) % count
            role = (
                "train"
                if offset < 4
                else "validation"
                if offset == 4
                else "test"
            )
            if role != split:
                continue
            result.append(
                {
                    "id": len(result),
                    "split": split,
                    "a": a,
                    "b": b,
                    "groups": blocks["a"][a] + blocks["b"][b],
                }
            )
    sources = [family for family in result if family["split"] == "train"]
    masks = {
        family["id"]: reference(family, "cpu").bool() for family in result
    }
    source_codes = {
        canonical_mask(masks[family["id"]][None]).numpy().tobytes()
        for family in sources
    }
    composition = []
    for family in result:
        if family["split"] == "train":
            continue
        target = masks[family["id"]]
        if canonical_mask(target[None]).numpy().tobytes() in source_codes:
            raise ValueError("A held-out mask duplicates a training mask")
        parents_a = [old for old in sources if old["a"] == family["a"]]
        parents_b = [old for old in sources if old["b"] == family["b"]]
        left, right = max(
            product(parents_a, parents_b),
            key=lambda pair: int(
                (masks[pair[0]["id"]] | masks[pair[1]["id"]]).sum()
            ),
        )
        union = masks[left["id"]] | masks[right["id"]]
        combined = torch.cat(
            (masks[left["id"]][:, :4], masks[right["id"]][:, 4:]), -1
        )
        if not torch.equal(combined, target) or (target & ~union).any():
            raise ValueError("Invalid parent decomposition")
        composition.append(
            {
                "family": family["id"],
                "split": family["split"],
                "a": family["a"],
                "b": family["b"],
                "parent_a": left["id"],
                "parent_b": right["id"],
                "union_edges": int(union.sum()),
                "target_edges": int(target.sum()),
                "missing_from_union": int((target & ~union).sum()),
                "exact_train_match": False,
            }
        )
    return result, {
        "kind": "compositional",
        "blocks": blocks,
        "common_edges": 16,
        "block_edges": 16,
        "train_combinations": 24,
        "validation_combinations": 6,
        "test_combinations": 6,
        "block_train_occurrences": 4,
        "decompositions": composition,
        "scope": "Analytical parents are generator diagnostics only, never search inputs.",
    }


def recombine(pool, count, seed):
    """Mix four columns from two IMP cards without labels or parent IDs."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(pool), (count, 2), generator=generator)
    left, right = pool[indices[:, 0]], pool[indices[:, 1]]
    overlap = torch.bmm(left.transpose(1, 2), right)
    order = assignments(-overlap)
    right = right.gather(2, order[:, None].expand(-1, INPUTS, -1))
    columns = torch.rand(count, HIDDEN, generator=generator).argsort(-1)
    selector = torch.zeros(count, HIDDEN, dtype=torch.bool)
    selector.scatter_(1, columns[:, :4], True)
    mixed = torch.where(selector[:, None], left, right)
    # IMP can have unequal column sizes; correct the budget using parent votes.
    votes = left + right
    for index in range(count):
        active = mixed[index].flatten().nonzero().flatten()
        if len(active) > 32:
            rank = (
                votes[index]
                .flatten()[active]
                .argsort(descending=True, stable=True)
            )
            mixed[index].flatten()[active[rank[32:]]] = 0
        elif len(active) < 32:
            candidates = (
                ((votes[index] > 0) & (mixed[index] == 0))
                .flatten()
                .nonzero()
                .flatten()
            )
            mixed[index].flatten()[candidates[: 32 - len(active)]] = 1
    if not torch.all(mixed.sum((1, 2)) == 32):
        raise ValueError("Recombined masks must retain 32 edges")
    return canonical_mask(mixed), indices
