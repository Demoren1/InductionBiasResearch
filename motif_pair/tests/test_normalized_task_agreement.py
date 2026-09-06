"""CPU checks for per-ordinal gradient normalization."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.normalized_task_agreement import combine_gradients


def test_alpha_zero_is_exact_task_gradient_and_normalized_ratio_is_alpha():
    torch.manual_seed(12)
    task = torch.randn(5, 2, 3, 7)
    pair = torch.randn_like(task)
    direction, info = combine_gradients(task, pair)
    assert torch.equal(direction[0], task[0])
    assert torch.allclose(info["weighted_agreement_ratio"][1:4], torch.tensor([.1, .3, 1.])[:, None])
    assert torch.allclose(info["effective_coefficient"][1:4] * info["agreement_grad_norm"][1:4],
                          torch.tensor([.1, .3, 1.])[:, None] * info["task_grad_norm"][1:4], atol=2e-7)


def test_norm_is_independent_per_ordinal_and_zero_norm_does_not_amplify():
    task = torch.ones(5, 2, 3, 4)
    pair = torch.ones_like(task)
    pair[1, :, 0] *= 9
    pair[2, :, 2] = 0
    task[3, :, 1] = 0
    direction, info = combine_gradients(task, pair)
    # Changing ordinal 0's pair norm cannot change ordinal 1's coefficient.
    expected = .1 * task[1, :, 1].norm() / pair[1, :, 1].norm()
    assert torch.allclose(info["effective_coefficient"][1, 1], expected)
    assert info["effective_coefficient"][2, 2] == 0 and info["effective_coefficient"][3, 1] == 0
    assert torch.equal(direction[2, :, 2], task[2, :, 2])
    assert torch.equal(direction[3, :, 1], task[3, :, 1])


def test_cosine_uses_the_product_of_the_individual_clamped_norms():
    task = torch.zeros(5, 2, 1, 2)
    pair = torch.zeros_like(task)
    task[:, 0, 0, 0] = 2e-7
    pair[:, 0, 0, 0] = 3e-7
    _, info = combine_gradients(task, pair)
    assert torch.allclose(info["cosine"], torch.ones_like(info["cosine"]))
