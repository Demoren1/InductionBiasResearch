import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from generator_evaluator.adapters import FunctionalBank
from generator_evaluator.joint_training import joint_generator_update
from generator_evaluator.staged_training import StagedGeneratorTrainer


class TinyGenerator(nn.Module):
    def __init__(self, offset: float = 0.0, slope: float = 1.0) -> None:
        super().__init__()
        self.features, self.hidden, self.noise_dim, self.token_dim = 2, 2, 2, 1
        self.logits = nn.Parameter(torch.tensor([
            [1.0 + offset, -0.7], [-0.5, 0.2 - offset],
        ]))
        self.slope = nn.Parameter(torch.tensor([
            [slope, -0.3 * slope], [0.15 * slope, -0.8 * slope],
        ]))
        self.noise_history = []
        self.target_k = None
        self.budget_history = []

    def set_budget(self, k):
        self.target_k = k
        self.budget_history.append(k)

    def _validate_bank(self, tokens, quality):
        if tokens.ndim != 4 or tokens.shape[0] != 1 or tokens.shape[-1] != self.token_dim:
            raise ValueError("invalid test bank")
        if quality is not None and quality.shape[:2] != tokens.shape[:2]:
            raise ValueError("invalid test quality")
        return tokens.shape[0], tokens.shape[1]

    def forward(self, tokens, noise, quality=None, *, density=None):
        self.last_noise = noise.detach().clone()
        self.noise_history.append(self.last_noise)
        return self.logits.unsqueeze(0) + noise[:, :1, None] * self.slope


class TinyCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def predict(self, masks, contexts):
        score = self.scale * (masks[:, 0, 0] + 0.3 * masks[:, 1, 1])
        return score + contexts[:, 0], torch.zeros_like(score)


def _bank(seed: int) -> FunctionalBank:
    rng = torch.Generator().manual_seed(seed)
    tokens = torch.randn((1, 3, 2, 1), generator=rng)
    masks = torch.zeros((3, 2, 2))
    masks[:, 0, 0] = 1
    masks[:, 1, 1] = 1
    return FunctionalBank(tokens, None, masks, masks[0])


def _trainer(reconstruction_weight: float = 0.0, reconstruction_batch_size: int = 8,
             quality_objective: str = "worst"):
    models = {"a": TinyGenerator(0.0, 1.0), "b": TinyGenerator(0.3, -0.8)}
    optimizers = {name: torch.optim.SGD(model.parameters(), lr=0.03)
                  for name, model in models.items()}
    critic = TinyCritic()
    trainer = StagedGeneratorTrainer(
        models, {name: _bank(index + 10) for index, name in enumerate(models)},
        optimizers, critic, torch.tensor([[0.0], [0.7]]), torch.zeros(2),
        {name: torch.Generator().manual_seed(100 + index) for index, name in enumerate(models)},
        torch.Generator().manual_seed(77), quality_sample_count=16, agreement_sample_count=3,
        permutation_weight=0.0, seed=9,
        quality_objective=quality_objective,
        reconstruction_weight=reconstruction_weight,
        reconstruction_batch_size=reconstruction_batch_size,
    )
    return trainer


def _snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


class StagedTrainingTests(unittest.TestCase):
    def test_joint_update_sets_each_alternating_budget_before_policy_and_agreement(self):
        trainer = _trainer()
        policy_budgets = []
        agreement_budgets = []

        def policy_update(model, ensemble, tokens, quality, context, dense_quality,
                          optimizer, k, rng, **kwargs):
            self.assertEqual(model.target_k, k)
            policy_budgets.append(k)
            return {"quality_loss": 0.0}

        def agreement_update(models, banks, optimizers, k, **kwargs):
            self.assertTrue(all(model.target_k == k for model in models))
            agreement_budgets.append(k)
            return {"agreement_loss": 0.0}

        with patch("generator_evaluator.joint_training.generator_update",
                   side_effect=policy_update), \
                patch("generator_evaluator.joint_training.joint_mask_agreement",
                      side_effect=agreement_update), \
                patch("generator_evaluator.joint_training.reconstruct_bank_masks",
                      return_value={"reconstruction_loss": 0.0}):
            for ordinal, output_k in enumerate((1, 2)):
                joint_generator_update(
                    models=trainer.models, banks=trainer.banks, optimizers=trainer.optimizers,
                    training_views=trainer.banks, ensemble=trainer.ensemble,
                    contexts=trainer.contexts, dense_quality=trainer.dense_quality,
                    targets=torch.empty((0, 2, 2)), output_k=output_k,
                    own_rngs=trainer.rngs, agreement_rng=torch.Generator().manual_seed(302),
                    update_ordinal=ordinal, updates_per_epoch=2, agreement_weight=0.1,
                    agreement_ramp_epochs=1, elite_weight=0.0, elite_limit=8,
                    reconstruction_weight=0.0, reconstruction_batch_size=1,
                    permutation_weight=0.0)

        self.assertEqual(policy_budgets, [1, 1, 2, 2])
        self.assertEqual(agreement_budgets, [1, 2])
        self.assertTrue(all(model.budget_history == [1, 2] for model in trainer.models.values()))

    def test_joint_update_interleaves_own_quality_random_elites_full_bank_reconstruction_and_shared_noise(self):
        from generator_evaluator.generator_objectives import joint_mask_agreement, reconstruct_bank_masks
        from generator_evaluator.training import generator_update

        trainer = _trainer(reconstruction_weight=0.2, quality_objective="mean_positive_worst")
        views = {}
        for name, bank in trainer.banks.items():
            views[name] = FunctionalBank(
                # Joint updates normalize view tensors to the owning generator's
                # device and dtype before both policy and elite-distillation work.
                tokens=bank.tokens[:, :1].double(),
                quality=None if bank.quality is None else bank.quality[:, :1].double(),
                masks=bank.masks[:1], baseline_mask=bank.baseline_mask,
                provenance=bank.provenance, states=bank.states[:1], diagnostics=bank.diagnostics)
        elites = torch.stack([bank.masks[0] for bank in trainer.banks.values()]).double()
        latent_before = trainer.shared_latent.detach().clone()
        agreement_rng = torch.Generator().manual_seed(302)

        with patch("generator_evaluator.joint_training.generator_update",
                   wraps=generator_update) as quality, \
                patch("generator_evaluator.joint_training.reconstruct_bank_masks",
                      wraps=reconstruct_bank_masks) as reconstruction, \
                patch("generator_evaluator.joint_training.joint_mask_agreement",
                      wraps=joint_mask_agreement) as agreement:
            rows = joint_generator_update(
                models=trainer.models, banks=trainer.banks, optimizers=trainer.optimizers,
                training_views=views, ensemble=trainer.ensemble,
                contexts=trainer.contexts, dense_quality=trainer.dense_quality,
                targets=elites, output_k=2, own_rngs=trainer.rngs,
                agreement_rng=agreement_rng, update_ordinal=0, updates_per_epoch=4,
                agreement_weight=0.1, agreement_ramp_epochs=2, elite_weight=0.5,
                elite_limit=8, reconstruction_weight=0.2,
                reconstruction_batch_size=3, permutation_weight=0.0)

        self.assertEqual(quality.call_count, 2)
        self.assertTrue(all(call.args[2].dtype == next(trainer.models[name].parameters()).dtype
                            for call, name in zip(quality.call_args_list, trainer.models)))
        self.assertTrue(all(call.args[2].device == next(trainer.models[name].parameters()).device
                            for call, name in zip(quality.call_args_list, trainer.models)))
        torch.testing.assert_close(quality.call_args_list[0].args[4], trainer.contexts[:1])
        torch.testing.assert_close(quality.call_args_list[1].args[4], trainer.contexts[1:])
        self.assertEqual(quality.call_args_list[0].kwargs["quality_objective"], "worst")
        self.assertEqual(reconstruction.call_count, 2)
        self.assertTrue(all(call.args[1] is trainer.banks[name]
                            for call, name in zip(reconstruction.call_args_list, trainer.models)))
        self.assertTrue(all(call.kwargs.get("bank_consensus", False) is False
                            for call in reconstruction.call_args_list))
        self.assertNotIn("shared_noise", agreement.call_args.kwargs)
        self.assertEqual(agreement.call_args.kwargs["sample_count"], 2)
        self.assertTrue(all(bank is trainer.banks[name] for bank, name in
                            zip(agreement.call_args.args[1], trainer.models)))
        self.assertAlmostEqual(rows["a"]["direct_agreement_weight"], 0.0125)
        self.assertEqual(rows["a"]["distillation_target_count"], 2.0)
        self.assertGreater(rows["a"]["elite_distillation_loss"], 0.0)
        for name in trainer.models:
            distillation_noise = trainer.models[name].noise_history[3]
            agreement_noise = trainer.models[name].noise_history[-1]
            self.assertEqual(tuple(distillation_noise.shape), (2, 2))
            self.assertEqual(torch.unique(distillation_noise, dim=0).shape[0], 2)
            self.assertFalse(torch.equal(distillation_noise, agreement_noise))
        torch.testing.assert_close(trainer.models["a"].last_noise,
                                   trainer.models["b"].last_noise)
        torch.testing.assert_close(latent_before, trainer.shared_latent.detach(), rtol=0, atol=0)

    def test_measured_elite_sampling_uses_sampled_indices_for_one_or_many_targets(self):
        from generator_evaluator.joint_training import _distil_measured_targets
        from generator_evaluator.cooperative_policy import align_elite_to_logits

        trainer = _trainer()
        model, bank = trainer.models["a"], trainer.banks["a"]
        targets = torch.stack((bank.masks[0], torch.flip(bank.masks[0], dims=(0,))))
        cases = ((targets[:1], torch.tensor([0, 0]), [targets[0], targets[0]]),
                 (targets, torch.tensor([1, 0]), [targets[1], targets[0]]))
        for target_set, selected, expected in cases:
            model.noise_history.clear()
            with patch("generator_evaluator.joint_training.torch.randint",
                       return_value=selected), \
                    patch("generator_evaluator.joint_training.align_elite_to_logits",
                          wraps=align_elite_to_logits) as align:
                loss, count = _distil_measured_targets(
                    model, bank.tokens, bank.quality, trainer.optimizers["a"],
                    target_set, k=2, rng=trainer.rngs["a"], weight=0.5, limit=8)
            self.assertEqual(count, len(target_set))
            self.assertGreater(loss, 0.0)
            self.assertEqual(len(align.call_args_list), 2)
            for call, target in zip(align.call_args_list, expected):
                torch.testing.assert_close(call.args[0], target)
            self.assertEqual(tuple(model.noise_history[-1].shape), (2, 2))
            self.assertEqual(torch.unique(model.noise_history[-1], dim=0).shape[0], 2)

    def test_staged_quality_update_passes_selected_quality_objective(self):
        from generator_evaluator.training import generator_update

        trainer = _trainer(quality_objective="mean_positive_worst")
        with patch("generator_evaluator.staged_training.generator_update",
                   wraps=generator_update) as update:
            trainer.quality_update("a", 2)
        self.assertEqual(update.call_args.kwargs["quality_objective"], "mean_positive_worst")

    def test_quality_stage_updates_policy_but_not_latent_or_critic(self):
        trainer = _trainer()
        latent_before = trainer.shared_latent.detach().clone()
        critic_before = _snapshot(trainer.ensemble)
        generator_before = _snapshot(trainer.models["a"])

        with patch("generator_evaluator.staged_training.joint_mask_agreement",
                   side_effect=AssertionError("quality-only stage must not run agreement")), \
             patch("generator_evaluator.staged_training.reconstruct_bank_masks",
                   side_effect=AssertionError("default reconstruction weight must stay disabled")):
            logs = trainer.quality_update("a", 2, auxiliary_budgets=[1, 3], ordinal=4)

        self.assertEqual(logs["output_k"], 2.0)
        self.assertEqual(logs["ordinal"], 4.0)
        self.assertEqual(logs["sample_count"], 16.0)
        self.assertEqual(logs["reconstruction_scope"], "disabled")
        self.assertGreater(torch.unique(trainer.models["a"].last_noise, dim=0).shape[0], 1)
        self.assertTrue(any(not torch.equal(before, after) for before, after in
                            zip(generator_before.values(), trainer.models["a"].state_dict().values())))
        torch.testing.assert_close(trainer.shared_latent, latent_before)
        self.assertIsNone(trainer.shared_latent.grad)
        for name, value in trainer.ensemble.state_dict().items():
            torch.testing.assert_close(value, critic_before[name])
        self.assertTrue(all(parameter.grad is None for parameter in trainer.ensemble.parameters()))

    def test_quality_policy_objective_can_train_shared_latent_in_cooperation(self):
        trainer = _trainer()
        before = trainer.shared_latent.detach().clone()
        trainer.quality_update("a", 2, train_shared_latent=True)
        self.assertIsNotNone(trainer.shared_latent.grad)
        self.assertFalse(torch.equal(before, trainer.shared_latent.detach()))

    def test_hungarian_agreement_updates_generators_and_shared_latent_without_critic_grads(self):
        trainer = _trainer()
        with torch.no_grad():
            trainer.models["b"].logits.copy_(torch.tensor([[1.0, 1.0], [-1.0, -1.0]]))
        generator_before = {name: _snapshot(model) for name, model in trainer.models.items()}
        latent_before = trainer.shared_latent.detach().clone()
        critic_before = _snapshot(trainer.ensemble)

        # Isolate the joint agreement step from cooperation's quality updates.
        trainer.quality_update = lambda *args, **kwargs: {"loss": 0.0}  # type: ignore[method-assign]
        rows = trainer.cooperation_update(2, agreement_weight=0.7)

        self.assertEqual(set(rows), {"a", "b"})
        self.assertTrue(all(row["direct_agreement_weight"] == 0.7 for row in rows.values()))
        self.assertFalse(torch.equal(latent_before, trainer.shared_latent.detach()))
        for name, model in trainer.models.items():
            self.assertTrue(any(not torch.equal(old, new) for old, new in
                                zip(generator_before[name].values(), model.state_dict().values())))
        for name, value in trainer.ensemble.state_dict().items():
            torch.testing.assert_close(value, critic_before[name])
        self.assertTrue(all(parameter.grad is None for parameter in trainer.ensemble.parameters()))

    def test_measured_target_distillation_is_available_only_in_cooperation(self):
        trainer = _trainer()
        target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        latent_before = trainer.shared_latent.detach().clone()
        rows = trainer.cooperation_update(2, agreement_weight=0.0,
                                          targets=target, elite_weight=0.5)
        self.assertTrue(all(row["measured_target_count"] == 1.0 for row in rows.values()))
        self.assertTrue(all(row["measured_distillation_loss"] > 0.0 for row in rows.values()))
        self.assertFalse(torch.equal(latent_before, trainer.shared_latent.detach()))

    def test_cooperation_steps_latent_once_for_combined_all_task_objective(self):
        trainer = _trainer()
        target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        from generator_evaluator.training import generator_update
        with patch("generator_evaluator.staged_training.generator_update",
                   wraps=generator_update) as update:
            trainer.cooperation_update(2, agreement_weight=0.5,
                                       targets=target, elite_weight=0.2)
        self.assertEqual(update.call_count, 2)
        for call in update.call_args_list:
            self.assertEqual(len(call.args[4]), 2)
            self.assertTrue(call.kwargs["accumulate"])
            self.assertEqual(call.kwargs["loss_scale"], 0.5)
        state = trainer.latent_optimizer.state[trainer.shared_latent]
        self.assertEqual(int(state["step"]), 1)
        self.assertTrue(all(p.grad is None for p in trainer.ensemble.parameters()))

    def test_state_restore_covers_latent_adam_rng_and_proposal_noise(self):
        trainer = _trainer()
        trainer.quality_update("a", 2, train_shared_latent=True)
        state = copy.deepcopy(trainer.state_dict())
        expected_proposals = trainer.proposal_noise(3)
        restored = _trainer()
        restored.load_state_dict(state)
        actual_proposals = restored.proposal_noise(3)
        torch.testing.assert_close(actual_proposals, expected_proposals)
        torch.testing.assert_close(restored.shared_latent, trainer.shared_latent)
        expected_state = trainer.latent_optimizer.state_dict()["state"]
        restored_state = restored.latent_optimizer.state_dict()["state"]
        self.assertEqual(expected_state.keys(), restored_state.keys())
        for parameter_id in expected_state:
            for key in expected_state[parameter_id]:
                torch.testing.assert_close(restored_state[parameter_id][key], expected_state[parameter_id][key])

    def test_proposal_noise_keeps_exact_latent_and_samples_its_neighborhood(self):
        trainer = _trainer()
        noise = trainer.proposal_noise(4, perturbation=0.25)
        torch.testing.assert_close(noise[0], trainer.shared_latent.detach().cpu())
        self.assertFalse(torch.equal(noise[1:], noise[0].expand_as(noise[1:])))

    def test_alternating_bank_views_reuse_prepared_copies_until_feedback_invalidation(self):
        trainer = _trainer()
        original = trainer.banks["a"]
        original_prepared = trainer._prepared_bank("a")
        view = FunctionalBank(original.tokens[:, :1], None, original.masks[:1], original.baseline_mask)
        trainer.set_bank("a", view)
        view_prepared = trainer._prepared_bank("a")
        self.assertIsNot(view_prepared, original_prepared)
        trainer.set_bank("a", original)
        self.assertIs(trainer._prepared_bank("a"), original_prepared)
        trainer.invalidate_bank_cache("a")
        self.assertIsNot(trainer._prepared_bank("a"), original_prepared)

        cached = trainer._prepared_bank("a")
        baseline = trainer.banks["a"].baseline_mask
        replacement = baseline.clone()
        replacement[0, 0] = 1.0 - replacement[0, 0]
        baseline.copy_(replacement)
        refreshed = trainer._prepared_bank("a")
        self.assertIsNot(refreshed, cached)
        torch.testing.assert_close(refreshed.baseline_mask, replacement)

    def test_reconstruction_regularizer_shares_one_standalone_optimizer_step(self):
        class CountingSGD(torch.optim.SGD):
            def __init__(self, params, **kwargs):
                super().__init__(params, **kwargs)
                self.step_count = 0

            def step(self, closure=None):
                self.step_count += 1
                return super().step(closure)

        trainer = _trainer(reconstruction_weight=0.4, reconstruction_batch_size=3)
        optimizer = CountingSGD(trainer.models["a"].parameters(), lr=0.03)
        trainer.optimizers["a"] = optimizer
        from generator_evaluator.generator_objectives import reconstruct_bank_masks
        with patch("generator_evaluator.staged_training.reconstruct_bank_masks",
                   wraps=reconstruct_bank_masks) as reconstruction:
            logs = trainer.quality_update("a", 2, ordinal=1, loss_scale=0.5)

        self.assertEqual(optimizer.step_count, 1)
        reconstruction.assert_called_once()
        self.assertEqual(reconstruction.call_args.kwargs["batch_size"], 3)
        self.assertAlmostEqual(reconstruction.call_args.kwargs["weight"], 0.2)
        self.assertTrue(reconstruction.call_args.kwargs["accumulate"])
        self.assertTrue(reconstruction.call_args.kwargs["bank_consensus"])
        self.assertEqual(logs["reconstruction_scope"], "bank_consensus")
        self.assertGreater(logs["reconstruction_loss"], 0.0)
        self.assertGreaterEqual(logs["reconstruction_overlap"], 0.0)

    def test_every_own_task_view_restores_consensus_and_cooperation_restores_teachers(self):
        trainer = _trainer(reconstruction_weight=0.2)
        from generator_evaluator.generator_objectives import reconstruct_bank_masks
        with patch("generator_evaluator.staged_training.reconstruct_bank_masks",
                   wraps=reconstruct_bank_masks) as reconstruction:
            own_consensus = trainer.quality_update("a", 2, ordinal=0, accumulate=True)
            sparse_view_consensus = trainer.quality_update("a", 2, ordinal=1, accumulate=True)
            all_task_teacher = trainer.quality_update(
                "a", 2, ordinal=1, accumulate=True, all_tasks=True
            )

        self.assertEqual(own_consensus["reconstruction_scope"], "bank_consensus")
        self.assertEqual(sparse_view_consensus["reconstruction_scope"], "bank_consensus")
        self.assertEqual(all_task_teacher["reconstruction_scope"], "teacher")
        self.assertTrue(reconstruction.call_args_list[0].kwargs["bank_consensus"])
        self.assertTrue(reconstruction.call_args_list[1].kwargs["bank_consensus"])
        self.assertFalse(reconstruction.call_args_list[2].kwargs["bank_consensus"])

    def test_reconstruction_adds_no_critic_or_shared_latent_gradient(self):
        regularized = _trainer(reconstruction_weight=0.5)
        policy_only = _trainer(reconstruction_weight=0.0)
        regularized_before = _snapshot(regularized.models["a"])
        policy_only_before = _snapshot(policy_only.models["a"])
        regularized.quality_update(
            "a", 2, ordinal=0, train_shared_latent=True, accumulate=True, loss_scale=0.4
        )
        policy_only.quality_update(
            "a", 2, ordinal=0, train_shared_latent=True, accumulate=True, loss_scale=0.4
        )

        torch.testing.assert_close(regularized.shared_latent.grad, policy_only.shared_latent.grad)
        self.assertIsNotNone(regularized.shared_latent.grad)
        self.assertTrue(all(parameter.grad is None for parameter in regularized.ensemble.parameters()))
        for current, saved in zip(regularized.models["a"].state_dict().values(),
                                  regularized_before.values()):
            torch.testing.assert_close(current, saved)
        for current, saved in zip(policy_only.models["a"].state_dict().values(),
                                  policy_only_before.values()):
            torch.testing.assert_close(current, saved)
