import torch

from evaluation.multi_decoder_agreement import optimize_multi_agreement
from evaluation.run_multi_pattern_agreement import group_patterns


class TinyDecoder(torch.nn.Module):
    def __init__(self, shift: float):
        super().__init__()
        self.latent_dim = 2
        self.mask_dim = 4
        self.cond_dim = 0
        self.register_buffer("weight", torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
        ]))
        self.register_buffer("bias", torch.tensor([shift, 0.0, -shift, 0.0]))

    def decode(self, z, condition):
        return z @ self.weight.T + self.bias


def test_multi_agreement_shapes_and_best_selection():
    models = [TinyDecoder(x) for x in (-0.3, 0.0, 0.3)]
    result = optimize_multi_agreement(
        models, n_starts=5, steps=8, lr=.05, seed=7, temperature=.5,
        radius=3., device="cpu", k=2, progress_every=0,
    )
    assert result["initial_z"].shape == (3, 5, 2)
    assert result["final_masks"].shape == (3, 5, 2, 2)
    assert torch.all(result["final_masks"].sum(dim=(2, 3)) == 2)
    assert torch.all(result["final_loss"] <= result["initial_loss"] + 1e-7)
    assert torch.all(result["final_z"].norm(dim=2) <= 3.00001)


def test_nested_groups_are_distinct_and_balanced_at_even_sizes():
    groups = [group_patterns(i) for i in range(8)]
    assert all(len(group) == len(set(group)) == 10 for group in groups)
    for size in (2, 4, 6, 8, 10):
        counts = {pattern: 0 for pattern in (f"{i:04b}" for i in range(16))}
        for group in groups:
            for pattern in group[:size]:
                counts[pattern] += 1
        assert set(counts.values()) == {size // 2}
