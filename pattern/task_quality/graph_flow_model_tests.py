"""Focused CPU tests for graph edge scoring and conditional flow helpers."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from .graph_flow_models import (
    GraphVelocity,
    elite_mask_bce_loss,
    flow_matching_coupling,
    flow_matching_loss,
    mask_to_flow_endpoint,
    sample_flow_endpoint,
    score_to_mask,
)


class GraphFlowModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def _inputs(self, batch: int = 3):
        torch.manual_seed(241)
        model = GraphVelocity(node_dim=6, edge_dim=5, task_dim=7, width=32, layers=3)
        edge_state = torch.randn(batch, 11, 8)
        time = torch.rand(batch)
        # Distinct hidden contexts ensure the equivariance check exercises the
        # aligned context path rather than relying on identical columns.
        node_context = torch.randn(batch, 8, 6)
        edge_context = torch.randn(batch, 11, 8, 5)
        task_context = torch.randn(batch, 7)
        return model, edge_state, time, node_context, edge_context, task_context

    def test_hidden_permutation_equivariance(self) -> None:
        model, state, time, nodes, edges, task = self._inputs()
        permutation = torch.stack([torch.randperm(8) for _ in range(state.shape[0])])
        permuted_state = state.gather(2, permutation[:, None, :].expand(-1, 11, -1))
        permuted_nodes = nodes.gather(1, permutation[:, :, None].expand(-1, -1, nodes.size(-1)))
        permuted_edges = edges.gather(
            2, permutation[:, None, :, None].expand(-1, 11, -1, edges.size(-1))
        )
        with torch.no_grad():
            scores = model(state, time, nodes, edges, task)
            permuted_scores = model(permuted_state, time, permuted_nodes, permuted_edges, task)
        expected = scores.gather(2, permutation[:, None, :].expand(-1, 11, -1))
        self.assertEqual(tuple(scores.shape), (state.shape[0], 11, 8))
        self.assertTrue(torch.allclose(permuted_scores, expected, atol=2e-6, rtol=2e-6))

    def test_null_contexts_force_equal_hidden_logits(self) -> None:
        torch.manual_seed(713)
        batch = 2
        model = GraphVelocity(node_dim=6, edge_dim=5, task_dim=7, width=32, layers=3)
        node_seed = torch.randn(batch, 1, 6)
        edge_seed = torch.randn(batch, 11, 1, 5)
        nodes = node_seed.expand(-1, 8, -1).contiguous()
        edges = edge_seed.expand(-1, -1, 8, -1).contiguous()
        zero_state = torch.zeros(batch, 11, 8)
        task = torch.randn(batch, 7)
        with torch.no_grad():
            logits = model(zero_state, torch.zeros(batch), nodes, edges, task)
        self.assertTrue(torch.allclose(logits, logits[:, :, :1].expand_as(logits), atol=1e-6, rtol=0))

    def test_exact_topk_shift_invariance_and_endpoint_encodings(self) -> None:
        scores = torch.randn(4, 11, 8)
        mask = score_to_mask(scores)
        shifted = score_to_mask(scores + torch.tensor([[-13.0], [0.25], [8.0], [100.0]])[:, None])
        self.assertTrue(torch.equal(mask, shifted))
        self.assertTrue(torch.equal(mask.sum(dim=(1, 2)), torch.full((4,), 32.0)))
        self.assertTrue(torch.all((mask == 0) | (mask == 1)))
        signed = mask_to_flow_endpoint(mask)
        self.assertTrue(torch.equal(signed, mask.mul(2).sub(1)))
        standardized = mask_to_flow_endpoint(mask, encoding="standardized")
        self.assertTrue(torch.allclose(standardized.mean(dim=(1, 2)), torch.zeros(4), atol=1e-6))
        self.assertTrue(torch.allclose(standardized.std(dim=(1, 2), unbiased=False), torch.ones(4)))

    def test_flow_and_elite_bce_losses_backpropagate(self) -> None:
        model, _, _, nodes, edges, task = self._inputs(batch=2)
        mask = score_to_mask(torch.randn(2, 11, 8))
        noise = torch.randn_like(mask)
        time = torch.tensor([0.2, 0.73])
        endpoint = mask_to_flow_endpoint(mask)
        state, sampled_time, velocity = flow_matching_coupling(
            mask, noise=noise, time=time
        )
        self.assertEqual(tuple(state.shape), (2, 11, 8))
        self.assertTrue(torch.allclose(sampled_time, time))
        self.assertTrue(torch.allclose(velocity, endpoint - noise))

        model.zero_grad(set_to_none=True)
        loss = flow_matching_loss(
            model, mask, nodes, edges, task, noise=noise, time=time
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))

        model.zero_grad(set_to_none=True)
        bce = elite_mask_bce_loss(model, mask, nodes, edges, task)
        self.assertTrue(torch.isfinite(bce))
        bce.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))

    def test_sampling_uses_bounded_default_euler_and_heun_steps(self) -> None:
        class TimeDependentVelocity(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def forward(self, state, time, node_context, edge_context, task_context):
                self.calls += 1
                return torch.ones_like(state) * (1.0 + time[:, None, None])

        model = TimeDependentVelocity()
        nodes = torch.zeros(2, 8, 1)
        edges = torch.zeros(2, 11, 8, 1)
        task = torch.zeros(2, 1)
        noise = torch.zeros(2, 11, 8)
        euler = sample_flow_endpoint(model, nodes, edges, task, noise=noise)
        self.assertEqual(model.calls, 12)
        self.assertTrue(torch.allclose(euler, torch.full_like(euler, 1.0 + 11.0 / 24.0)))

        model.calls = 0
        heun = sample_flow_endpoint(model, nodes, edges, task, noise=noise, solver="heun")
        self.assertEqual(model.calls, 16)
        self.assertTrue(torch.allclose(heun, torch.full_like(heun, 1.5), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
