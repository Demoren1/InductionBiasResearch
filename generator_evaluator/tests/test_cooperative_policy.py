import unittest

import torch
from torch import nn

from generator_evaluator.adapters import FunctionalBank
from generator_evaluator.data import topology_id
from generator_evaluator.cooperative_policy import (
    DensityConditionedGenerator,
    cooperative_generator_update,
    mixed_acquisition,
    propose_shared_pool,
    rank_shared_pool,
    select_common_elites,
)
import generator_evaluator.cooperative_policy as policy


def generator() -> DensityConditionedGenerator:
    return DensityConditionedGenerator(3, 2, 2, width=8, heads=2, layers=1, noise_dim=3, target_k=2)


def bank(seed: int) -> FunctionalBank:
    rng = torch.Generator().manual_seed(seed)
    masks = torch.zeros(3, 2, 2)
    masks[:, 0, 0] = 1
    masks[:, 1, 1] = 1
    return FunctionalBank(torch.randn(1, 3, 2, 3, generator=rng), None, masks, masks[0])


class FakeEnsemble(nn.Module):
    def predict(self, masks: torch.Tensor, contexts: torch.Tensor):
        return masks[:, 0].sum(1) + contexts[:, 0], contexts[:, 1].abs()


class CooperativePolicyTests(unittest.TestCase):
    def test_generators_are_separate_and_density_conditioned(self) -> None:
        first, second = generator(), generator()
        self.assertIsNot(next(first.parameters()), next(second.parameters()))
        tokens, noise = torch.randn(1, 3, 2, 3), torch.randn(1, 3)
        first.eval()
        low = first(tokens, noise, density=.25)
        high = first(tokens, noise, density=.75)
        self.assertFalse(torch.allclose(low, high))
        permuted = tokens.index_select(1, torch.tensor([2, 0, 1])).index_select(2, torch.tensor([1, 0]))
        torch.testing.assert_close(low, first(permuted, noise, density=.25), rtol=1e-5, atol=1e-6)

    def test_pool_has_exact_budget_and_canonical_deduplication(self) -> None:
        models, banks = {"a": generator(), "b": generator()}, {"a": bank(1), "b": bank(2)}
        pool, sources = propose_shared_pool(models, banks, 2, 3, torch.Generator().manual_seed(3),
                                            random_count=2, mutation_count=0)
        self.assertEqual(len(pool), len(sources))
        self.assertTrue(bool((pool.sum((1, 2)) == 2).all()))
        self.assertEqual(len({tuple(sorted(tuple(c.tolist()) for c in row.T)) for row in pool}), len(pool))

    def test_proposal_trace_keeps_both_generator_attributions_before_pool_deduplication(self) -> None:
        models, banks = {"a": generator(), "b": generator()}, {"a": bank(1), "b": bank(2)}
        # The second proposal differs only by a hidden-column permutation.
        duplicate_family = torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])
        original = policy.propose_candidates
        policy.propose_candidates = lambda *args, **kwargs: duplicate_family.clone()
        trace = {}
        try:
            pool, _ = propose_shared_pool(models, banks, 2, 2, torch.Generator().manual_seed(1),
                                          random_count=0, mutation_count=0, proposal_trace=trace)
        finally:
            policy.propose_candidates = original
        self.assertEqual(len(pool), 1)
        first = trace["generators"]["a"]
        second = trace["generators"]["b"]
        self.assertEqual(first["sampled_count"], 2)
        self.assertEqual(second["sampled_count"], 2)
        self.assertEqual(first["unique_count"], 1)
        self.assertEqual(second["unique_count"], 1)
        self.assertEqual(first["topology_ids"], second["topology_ids"])

    def test_paired_proposals_use_the_same_latent_and_reach_the_pool(self):
        first, second = generator(), generator()
        second.load_state_dict(first.state_dict())
        trace = {}
        pool, _ = propose_shared_pool({"a": first, "b": second}, {"a": bank(9), "b": bank(9)},
            2, 4, torch.Generator().manual_seed(8), random_count=0, mutation_count=0,
            paired_proposals=True, proposal_trace=trace)
        paired = trace["generators"]["a"]["paired_topology_ids"]
        self.assertEqual(len(paired), 2)
        self.assertEqual(paired, trace["generators"]["b"]["paired_topology_ids"])
        from generator_evaluator.data import topology_id
        self.assertTrue(set(paired).issubset({topology_id(mask) for mask in pool}))
        self.assertTrue(bool((pool.sum((1, 2)) == 2).all()))

    def test_learned_shared_noise_keeps_half_stochastic_proposals(self):
        first, second = generator(), generator()
        second.load_state_dict(first.state_dict())
        learned_neighborhood = torch.tensor([[0.0, 0.0, 0.0], [0.8, -0.4, 0.2]])
        trace = {}
        pool, _ = propose_shared_pool(
            {"a": first, "b": second}, {"a": bank(19), "b": bank(19)},
            2, 4, torch.Generator().manual_seed(13), random_count=0, mutation_count=0,
            paired_proposals=True, shared_noise=learned_neighborhood, proposal_trace=trace,
        )
        self.assertEqual(trace["generators"]["a"]["sampled_count"], 4)
        self.assertEqual(len(trace["generators"]["a"]["paired_topology_ids"]), 2)
        self.assertEqual(trace["generators"]["a"]["paired_topology_ids"],
                         trace["generators"]["b"]["paired_topology_ids"])
        self.assertTrue(set(trace["generators"]["a"]["paired_topology_ids"]).issubset(
            {topology_id(mask) for mask in pool}))
        self.assertTrue(bool((pool.sum((1, 2)) == 2).all()))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA generator")
    def test_cpu_pool_accepts_cuda_generator(self) -> None:
        models = {"a": generator().cuda(), "b": generator().cuda()}
        banks = {"a": bank(9), "b": bank(10)}
        rng = torch.Generator(device="cuda").manual_seed(11)
        pool, _ = propose_shared_pool(models, banks, 2, 2, rng, random_count=2, mutation_count=0)
        self.assertEqual(pool.device.type, "cpu")
        self.assertTrue(bool((pool.sum((1, 2)) == 2).all()))
        ranks = {"worst_delta": torch.zeros(len(pool)), "max_std": torch.zeros(len(pool))}
        indices, labels, _ = mixed_acquisition(pool, ["x"] * len(pool), ranks, 2, rng)
        self.assertEqual(len(indices), len(labels))

    def test_global_ranking_and_mixed_acquisition_keep_all_task_interpretations(self) -> None:
        masks = torch.tensor([[[1., 0.], [0., 0.]], [[1., 1.], [0., 0.]]])
        ranks = rank_shared_pool(masks, FakeEnsemble(), torch.tensor([[0., .2], [3., .4]]), torch.zeros(2))
        self.assertEqual(tuple(ranks["mean"].shape), (2, 2))
        torch.testing.assert_close(ranks["worst_delta"], torch.tensor([4., 5.]))
        selected, labels, trace = mixed_acquisition(masks, ["generator:a", "generator:b"], ranks, 2,
                                                    torch.Generator().manual_seed(4))
        self.assertEqual(len(selected), len(labels))
        self.assertEqual(len(selected), len(trace))
        self.assertIn("promising", labels)

    def test_mixed_acquisition_promising_slot_uses_objective_cost_when_available(self) -> None:
        masks = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1)
        ranks = {
            "worst_delta": torch.tensor([0., 1., 2., 3.]),
            "objective_cost": torch.tensor([2., 0., 3., 4.]),
            "max_std": torch.zeros(4),
        }
        indices, labels, _ = mixed_acquisition(masks, ["x"] * len(masks), ranks, 1)
        self.assertEqual(indices.tolist(), [1])
        self.assertEqual(labels, ["promising"])

    def test_pool_and_common_elite_ranking_use_configured_objective(self) -> None:
        masks = torch.tensor([
            [[1., 0.], [0., 0.]],
            [[1., 1.], [0., 0.]],
        ])
        contexts = torch.tensor([[0., .2], [3., .4]])
        ranks = rank_shared_pool(
            masks, FakeEnsemble(), contexts, torch.zeros(2),
            quality_objective="mean_positive_worst",
        )
        torch.testing.assert_close(ranks["objective_cost"], torch.tensor([6.5, 8.5]))
        torch.testing.assert_close(ranks["worst_delta"], torch.tensor([4., 5.]))

        real = torch.tensor([[-1., .4], [-.05, .2]])
        dense = torch.zeros(2)
        legacy = select_common_elites(masks, real, dense, margin=.5, limit=2)
        combined = select_common_elites(
            masks, real, dense, margin=.5, limit=2,
            quality_objective="mean_positive_worst",
        )
        torch.testing.assert_close(legacy[0], masks[1])
        torch.testing.assert_close(combined[0], masks[0])

    def test_own_task_policy_update_and_measured_common_elites(self) -> None:
        captured = {}
        original = policy.generator_update
        def stub(model, ensemble, tokens, quality, contexts, dense, optimizer, k, rng, **kwargs):
            captured["contexts"], captured["dense"] = contexts.clone(), dense.clone()
            return {"loss": 0.0}
        policy.generator_update = stub
        try:
            cooperative_generator_update(generator(), FakeEnsemble(), bank(5).tokens, None,
                                         torch.tensor([[1., 0.], [2., 0.]]), torch.tensor([.1, .2]),
                                         torch.optim.Adam(generator().parameters()), 2,
                                         task_index=1, agreement_weight=0.)
        finally:
            policy.generator_update = original
        self.assertEqual(tuple(captured["contexts"].shape), (1, 2))
        torch.testing.assert_close(captured["contexts"], torch.tensor([[2., 0.]]))
        masks = torch.tensor([[[1., 0.], [0., 1.]], [[1., 1.], [0., 0.]]])
        kept = select_common_elites(masks, torch.tensor([[.1, .2], [.1, .8]]), torch.tensor([.2, .3]))
        self.assertEqual(len(kept), 1)
        empty = select_common_elites(masks, torch.tensor([[.3, .2], [.1, .8]]), torch.tensor([.2, .3]))
        self.assertEqual(len(empty), 0)

    def test_exact_k_elite_distillation_updates_same_model_and_skips_other_density(self) -> None:
        model = generator()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        original = policy.generator_update
        policy.generator_update = lambda *args, **kwargs: {"loss": 0.0}
        try:
            before = [parameter.detach().clone() for parameter in model.parameters()]
            logs = cooperative_generator_update(
                model, FakeEnsemble(), bank(7).tokens, None, torch.tensor([[0., 0.]]), torch.zeros(1),
                optimizer, 2, torch.Generator().manual_seed(6), task_index=0,
                elite_masks=torch.tensor([[[1., 0.], [0., 1.]]]), agreement_weight=.5,
            )
            self.assertGreater(logs["elite_distillation_loss"], 0.)
            self.assertTrue(torch.isfinite(torch.tensor(logs["elite_distillation_loss"])))
            self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))
            after = [parameter.detach().clone() for parameter in model.parameters()]
            skipped = cooperative_generator_update(
                model, FakeEnsemble(), bank(7).tokens, None, torch.tensor([[0., 0.]]), torch.zeros(1),
                optimizer, 2, torch.Generator().manual_seed(7), task_index=0,
                elite_masks=torch.ones(1, 2, 2), agreement_weight=.5,
            )
        finally:
            policy.generator_update = original
        self.assertEqual(skipped["elite_count"], 0.)
        self.assertEqual(skipped["elite_distillation_loss"], 0.)
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(after, model.parameters())))


if __name__ == "__main__":
    unittest.main()
