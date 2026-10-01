"""CPU checks for the utility graph field and independent FM helpers."""

from __future__ import annotations

import unittest

import torch

from deepsets_vaae.utility_graph_models import (
    UtilityGraphField,
    exact_topk,
    flow_matching_loss,
    flow_matching_losses,
    mask_to_flow_endpoint,
    sample_flow,
)


def _model(features: int = 11, hidden: int = 8) -> UtilityGraphField:
    torch.manual_seed(23)
    return UtilityGraphField(
        features=features,
        hidden=hidden,
        node_dim=13,
        edge_dim=5,
        task_dim=7,
        width=24,
        edge_width=8,
        depth=2,
    )


def _contexts(batch: int = 2, features: int = 11, hidden: int = 8):
    generator = torch.Generator().manual_seed(81)
    state = torch.randn((batch, features, hidden), generator=generator)
    time = torch.rand((batch,), generator=generator)
    nodes = torch.randn((batch, hidden, 13), generator=generator)
    edges = torch.randn((batch, features, hidden, 5), generator=generator)
    tasks = torch.randn((batch, 7), generator=generator)
    return state, time, nodes, edges, tasks


class UtilityGraphModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._old_threads)

    def test_joint_hidden_permutation_equivariance(self):
        model = _model()
        state, time, nodes, edges, tasks = _contexts()
        permutation = torch.randperm(state.shape[-1], generator=torch.Generator().manual_seed(7))
        actual = model(state[:, :, permutation], time, nodes[:, permutation],
                       edges[:, :, permutation, :], tasks)
        expected = model(state, time, nodes, edges, tasks)[:, :, permutation]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_identical_hidden_contexts_collapse_logits_from_zero_state(self):
        model = _model()
        batch, features, hidden = 2, 11, 8
        nodes = torch.randn(batch, 1, 13).expand(-1, hidden, -1).contiguous()
        edges = torch.randn(batch, features, 1, 5).expand(-1, -1, hidden, -1).contiguous()
        state = torch.zeros(batch, features, hidden)
        time = torch.tensor([0.0, 0.7])
        tasks = torch.randn(batch, 7)
        logits = model(state, time, nodes, edges, tasks)
        torch.testing.assert_close(logits, logits[:, :, :1].expand_as(logits), atol=1e-6, rtol=1e-6)

    def test_exact_topk_count_shift_invariance_and_tie_contract(self):
        scores = torch.randn(4, 11, 8, generator=torch.Generator().manual_seed(2))
        mask = exact_topk(scores, 23)
        shifted = exact_topk(scores + 19.25, 23)
        self.assertTrue(torch.equal(mask, shifted))
        self.assertTrue(torch.equal(mask.sum((1, 2)), torch.full((4,), 23.0)))
        # Tied cutoff membership is implementation-selected; only exact count is promised.
        tied = exact_topk(torch.zeros(2, 11, 8), 23)
        self.assertTrue(torch.equal(tied.sum((1, 2)), torch.full((2,), 23.0)))

    def test_flow_loss_backward_and_sampling_with_joint_noise_permutation(self):
        model = _model()
        state, time, nodes, edges, tasks = _contexts()
        endpoint = mask_to_flow_endpoint(exact_topk(torch.randn_like(state), 23))
        losses = flow_matching_losses(model, endpoint, nodes, edges, tasks,
                                      generator=torch.Generator().manual_seed(50))
        self.assertEqual(tuple(losses.shape), (state.shape[0],))
        loss = flow_matching_loss(model, endpoint, nodes, edges, tasks,
                                  generator=torch.Generator().manual_seed(50))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))

        permutation = torch.randperm(state.shape[-1], generator=torch.Generator().manual_seed(19))
        noise = torch.randn_like(state)
        sample = sample_flow(model, nodes, edges, tasks, steps=12, initial_noise=noise)
        permuted_sample = sample_flow(
            model,
            nodes[:, permutation],
            edges[:, :, permutation, :],
            tasks,
            steps=12,
            initial_noise=noise[:, :, permutation],
        )
        torch.testing.assert_close(permuted_sample, sample[:, :, permutation], atol=3e-6, rtol=3e-6)
        self.assertEqual(tuple(sample.shape), tuple(state.shape))

    def test_deepsets_scale_smoke_forward_backward(self):
        # Standard 784x32 graph: about 25k active bipartite edges.
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            model = UtilityGraphField(
                features=784,
                hidden=32,
                node_dim=16,
                edge_dim=4,
                task_dim=9,
                width=32,
                edge_width=8,
                depth=2,
            )
            state = torch.randn(1, 784, 32, requires_grad=True)
            nodes = torch.randn(1, 32, 16)
            edges = torch.randn(1, 784, 32, 4)
            tasks = torch.randn(1, 9)
            output = model(state, torch.tensor([0.4]), nodes, edges, tasks)
            self.assertEqual(tuple(output.shape), (1, 784, 32))
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertIsNotNone(state.grad)
            self.assertTrue(torch.isfinite(state.grad).all())
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
