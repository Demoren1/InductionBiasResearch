"""Focused checks for the MNIST8m generated-sharing comparison."""

import torch
from torch.nn import functional as F

from .run import GeneratedLayer, ImageSum


def test_no_z_starts_at_exactly_the_fixed_z_function() -> None:
    fixed = ImageSum("fixed_z", 42).eval()
    without = ImageSum("no_z", 42).eval()
    torch.testing.assert_close(fixed.third.weight(), without.third.weight(),
                               rtol=1e-6, atol=1e-6)
    images = torch.rand(3, 5, 784)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]],
                        dtype=torch.bool)
    torch.testing.assert_close(fixed(images, mask), without(images, mask),
                               rtol=1e-5, atol=1e-5)


def test_any_learned_z_can_be_folded_into_no_z() -> None:
    learned = ImageSum("learned_z", 42).eval()
    without = ImageSum("no_z", 42).eval()
    with torch.no_grad():
        learned.third.z.add_(torch.tensor([0.7, -1.3, 0.2, 1.1]))
        learned.third.generator.net[0].weight.add_(0.01)
        source = learned.third.generator.net
        target = without.third.generator.net
        target[0].weight.copy_(source[0].weight[:, 4:])
        target[0].bias.copy_(source[0].bias + source[0].weight[:, :4] @ learned.third.z)
        for index in (2, 4):
            target[index].load_state_dict(source[index].state_dict())
        without.third.values.copy_(learned.third.values)
        without.third.bias.copy_(learned.third.bias)
    images = torch.rand(3, 5, 784)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]],
                        dtype=torch.bool)
    torch.testing.assert_close(learned.third.weight(), without.third.weight(),
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(learned(images, mask), without(images, mask),
                               rtol=1e-5, atol=1e-5)


def test_prediction_does_not_depend_on_set_order() -> None:
    images = torch.rand(2, 5, 784)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    order = torch.tensor([4, 2, 0, 1, 3])
    for arm in ("paper_mlp", "learned_z", "no_z"):
        model = ImageSum(arm, 47).eval()
        torch.testing.assert_close(model(images, mask),
                                   model(images[:, order], mask[:, order]),
                                   rtol=1e-5, atol=1e-5)


def test_notebook_padding_matches_explicit_dense_sum() -> None:
    images = torch.rand(3, 5, 784)
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [0, 0, 0, 1, 1]],
                        dtype=torch.bool)
    images[~mask] = images[0, 0].clone()
    for arm in ("paper_mlp", "learned_z", "no_z"):
        model = ImageSum(arm, 42, paper_initialization=True,
                         padding_mode="notebook").eval()
        hidden = torch.tanh(model.first(images))
        hidden = torch.tanh(model.second(hidden))
        if isinstance(model.third, GeneratedLayer):
            hidden = F.linear(hidden, model.third.weight(), model.third.bias)
        else:
            hidden = model.third(hidden)
        naive = model.readout(torch.tanh(hidden).sum(1)).squeeze(-1)
        torch.testing.assert_close(model(images, mask), naive,
                                   rtol=1e-5, atol=1e-5)
