"""Small mathematical checks for the generated-sharing benchmark."""

import torch

from .run_position_dependent import (Config, IndependentGenerators, LatentGenerator, assignment,
                  coordinate_grid, design, make_sets)


def test_global_z_folds_exactly_into_bias() -> None:
    cfg = Config()
    z = torch.tensor([[0.2, -0.7, 0.4, 1.2]])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        original = LatentGenerator(cfg)
    folded = IndependentGenerators(cfg, [73], z)
    grid = coordinate_grid(10, 10, torch.device("cpu"))
    torch.testing.assert_close(original(grid, z), folded(grid), atol=1e-6, rtol=1e-6)


def test_generated_structure_is_one_permutation_per_slot() -> None:
    logits = torch.randn(3, 7, 4, 4)
    u = assignment(logits, temperature=1.0, straight_through=False)
    torch.testing.assert_close(u.sum(-1), torch.ones(3, 7, 4))
    torch.testing.assert_close(u.sum(-2), torch.ones(3, 7, 4))


def test_oracle_sharing_computes_exact_digit_sum() -> None:
    rng = torch.Generator().manual_seed(62)
    bits, target, _ = make_sets(40, 10, rng, torch.device("cpu"), slot_capacity=100)
    u = torch.eye(4)[None, None].expand(1, 100, -1, -1)
    prediction = design(bits, u)[0] @ torch.tensor([1.0, 2.0, 4.0, 8.0])
    torch.testing.assert_close(prediction, target, atol=0, rtol=0)
