"""Checks for the shared, position-free digit-sum construction."""

import torch

from .run import (BitGenerator, Config, IndependentBitGenerators, assignment, design)


def test_example_3_5_7() -> None:
    digits = torch.tensor([3, 5, 7])
    bits = ((digits[:, None] >> torch.arange(4)) & 1).float()[None]
    torch.testing.assert_close(bits.sum(1)[0], torch.tensor([3., 2., 2., 0.]))
    value = design(bits, torch.eye(4)[None])[0] @ torch.tensor([1., 2., 4., 8.])
    torch.testing.assert_close(value, torch.tensor([15.]))
    torch.testing.assert_close(design(bits, torch.eye(4)[None]),
                               design(bits.flip(1), torch.eye(4)[None]))


def test_every_permutation_can_represent_the_sum() -> None:
    bits = torch.randint(0, 2, (9, 7, 4)).float()
    target = bits @ torch.tensor([1., 2., 4., 8.])
    logits = torch.randn(24, 4, 4)
    u = assignment(logits, temperature=1, straight_through=False)
    torch.testing.assert_close(u.sum(-1), torch.ones(24, 4))
    torch.testing.assert_close(u.sum(-2), torch.ones(24, 4))
    v = torch.einsum("cdq,d->cq", u, torch.tensor([1., 2., 4., 8.]))
    prediction = torch.einsum("cbq,cq->cb", design(bits, u), v)
    torch.testing.assert_close(prediction, target.sum(-1)[None].expand(24, -1))


def test_fixed_z_folds_into_no_z_generator() -> None:
    cfg = Config()
    z = torch.tensor([[0.2, -0.7, 0.4, 1.2]])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        original = BitGenerator(cfg)
    folded = IndependentBitGenerators(cfg, [73], z)
    torch.testing.assert_close(original(z), folded(torch.device("cpu")),
                               atol=1e-6, rtol=1e-6)
