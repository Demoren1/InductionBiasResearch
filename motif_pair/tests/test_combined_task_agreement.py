"""Check grouped task gradients and ordinal pairing against independent MLPs."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.combined_task_agreement import TaskBank, per_network_bce
from models.mlp import BatchedMaskedMLP


TASKS = ["A000_B110_G05", "A001_B101_G05"]


def test_grouped_mlp_matches_independent_outputs_and_mask_gradients():
    bank = TaskBank(TASKS, 2, 2, 0, 3, 12345, "cpu")
    torch.manual_seed(17)
    masks = torch.rand(2, 2, 3, 16, 16, requires_grad=True)
    x = torch.randn(2, 7, 16)
    labels = torch.randint(2, (2, 7)).float()
    output = bank(x, masks)
    loss = per_network_bce(output, labels).mean((1, 2)).sum()
    gradient = torch.autograd.grad(loss, masks)[0]
    reference_masks = masks.detach().clone().requires_grad_()
    reference_loss = 0.
    for variant in range(2):
        for decoder in range(2):
            for task in range(2):
                model = BatchedMaskedMLP(3, 16, 16)
                with torch.no_grad():
                    for name in ("w1", "b1", "w2", "b2"):
                        value = getattr(bank, name)[variant, decoder, task]
                        parameter = getattr(model, name)
                        parameter.copy_(value.reshape_as(parameter))
                    model.load_masks(reference_masks[variant, decoder].detach())
                assert torch.allclose(model(x[task]).T, output[variant, decoder, task], atol=1e-7)
                # Existing live-mask forward preserves the independent network's mask gradient.
                from evaluation.single_z_mlp import _forward_live_mask
                logits = _forward_live_mask(model, x[task], reference_masks[variant, decoder])
                target = labels[task, :, None].expand_as(logits)
                reference_loss = reference_loss + F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none").mean(0).sum() / 4
    reference_gradient = torch.autograd.grad(reference_loss, reference_masks)[0]
    assert torch.allclose(loss, reference_loss, atol=1e-6)
    assert torch.allclose(gradient, reference_gradient, atol=1e-7, rtol=1e-5)


def test_initial_weights_are_paired_and_shard_invariant():
    full = TaskBank(TASKS, 3, 2, 0, 64, 777, "cpu")
    for start, stop in ((0, 32), (32, 64)):
        shard = TaskBank(TASKS, 3, 2, start, stop, 777, "cpu")
        for (name, whole), (_, part) in zip(full.named_parameters(), shard.named_parameters()):
            assert torch.equal(whole[:, :, :, start:stop], part), name
            assert torch.equal(part[0, 0], part[1, 1]), name
    assert not torch.equal(full.w1[0, 0, 0], full.w1[0, 0, 1])
