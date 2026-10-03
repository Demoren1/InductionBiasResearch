import copy

import torch

from generator_evaluator.data import topology_id
from generator_evaluator.models import MaskQualityEvaluator, QualityEnsemble, TransformerMaskGenerator
from generator_evaluator.training import (
    evaluator_metrics,
    generator_update,
    propose_candidates,
    select_acquisition,
    train_evaluators,
)


def _generator() -> TransformerMaskGenerator:
    return TransformerMaskGenerator(3, 2, 2, width=8, heads=2, layers=1, noise_dim=3, quality_dim=1)


def _ensemble(context_dim: int = 2) -> QualityEnsemble:
    return QualityEnsemble(evaluators=[
        MaskQualityEvaluator(2, context_dim, width=8, heads=2, layers=1),
        MaskQualityEvaluator(2, context_dim, width=8, heads=2, layers=1),
    ])


def test_proposals_are_exact_k_masks() -> None:
    generator = _generator()
    masks = propose_candidates(
        generator, torch.randn(1, 3, 2, 3), torch.randn(1, 3, 1),
        k=2, count=5, rng=torch.Generator().manual_seed(4),
    )
    assert masks.shape == (5, 2, 2)
    torch.testing.assert_close(masks.flatten(1).sum(1), torch.full((5,), 2.0))
    assert set(masks.unique().tolist()) <= {0.0, 1.0}


def test_generator_update_freezes_evaluator_and_uses_worst_task_cost() -> None:
    torch.manual_seed(1)
    generator, ensemble = _generator(), _ensemble()
    before = copy.deepcopy(ensemble.state_dict())

    # Make the required max over tasks observable independently of a learned
    # evaluator.  The hard-policy gradient still flows through log probabilities.
    def predicted_cost(_masks: torch.Tensor, contexts: torch.Tensor):
        return contexts[:, 0], torch.zeros_like(contexts[:, 0])

    ensemble.predict = predicted_cost  # type: ignore[method-assign]
    logs = generator_update(
        generator, ensemble, torch.randn(1, 3, 2, 3), torch.randn(1, 3, 1),
        torch.tensor([[1.0, 0.0], [4.0, 0.0]]), torch.tensor([0.0, 1.0]),
        torch.optim.Adam(generator.parameters(), lr=1e-3), 2,
        torch.Generator().manual_seed(7), permutation_weight=0.0,
    )
    assert logs["sample_count"] == 2.0
    # deltas are 1 and 3, so every independent draw's cost is max(1, 3).
    assert logs["predicted_cost"] == 3.0
    assert logs["policy_gradient_loss"] == logs["gradient_surrogate"]
    for name, parameter in ensemble.state_dict().items():
        torch.testing.assert_close(parameter, before[name])
    assert all(parameter.grad is None for parameter in ensemble.parameters())


def test_generator_update_can_use_mean_plus_positive_worst_cost() -> None:
    generator, ensemble = _generator(), _ensemble()
    ensemble.predict = lambda _masks, contexts: (contexts[:, 0], torch.zeros_like(contexts[:, 0]))  # type: ignore[method-assign]
    logs = generator_update(
        generator, ensemble, torch.randn(1, 3, 2, 3), torch.randn(1, 3, 1),
        torch.tensor([[1.0, 0.0], [4.0, 0.0]]), torch.tensor([0.0, 1.0]),
        torch.optim.SGD(generator.parameters(), lr=0.01), 2,
        torch.Generator().manual_seed(12), permutation_weight=0.0,
        quality_objective="mean_positive_worst",
    )
    # The task deltas are 1 and 3: mean=2, with a positive worst penalty of 3.
    assert logs["predicted_cost"] == 5.0
    assert logs["predicted_objective"] == 5.0


def test_generator_update_scales_only_the_accumulated_gradient() -> None:
    torch.manual_seed(18)
    base = _generator()
    unscaled, scaled = copy.deepcopy(base), copy.deepcopy(base)
    ensemble = _ensemble()
    tokens, quality = torch.randn(1, 3, 2, 3), torch.randn(1, 3, 1)
    contexts = torch.tensor([[0.3, 0.0], [1.4, 0.0]])
    dense_quality = torch.tensor([0.0, 0.2])
    unscaled_logs = generator_update(
        unscaled, ensemble, tokens, quality, contexts, dense_quality,
        torch.optim.SGD(unscaled.parameters(), lr=0.01), 2,
        torch.Generator().manual_seed(47), permutation_weight=0.2,
        accumulate=True,
    )
    scaled_logs = generator_update(
        scaled, ensemble, tokens, quality, contexts, dense_quality,
        torch.optim.SGD(scaled.parameters(), lr=0.01), 2,
        torch.Generator().manual_seed(47), permutation_weight=0.2,
        accumulate=True, loss_scale=0.25,
    )

    for first, second in zip(unscaled.parameters(), scaled.parameters()):
        assert first.grad is not None and second.grad is not None
        torch.testing.assert_close(second.grad, first.grad * 0.25, rtol=2e-5, atol=2e-7)
    assert scaled_logs["loss_scale"] == 0.25
    assert unscaled_logs["loss_scale"] == 1.0
    assert scaled_logs["loss"] == unscaled_logs["loss"]


def test_generator_update_rejects_invalid_loss_scale() -> None:
    generator, ensemble = _generator(), _ensemble()
    args = (
        generator, ensemble, torch.randn(1, 3, 2, 3), torch.randn(1, 3, 1),
        torch.zeros(1, 2), torch.zeros(1),
        torch.optim.SGD(generator.parameters(), lr=0.01), 2,
    )
    for value in (-0.1, float("nan"), float("inf"), True):
        try:
            generator_update(*args, loss_scale=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"loss_scale={value!r} should be rejected")


def test_acquisition_mixes_sources_and_deduplicates_topologies() -> None:
    ensemble = _ensemble()
    # The first two masks differ only by hidden-column order.
    masks = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[1.0, 1.0], [0.0, 0.0]],
        [[1.0, 0.0], [1.0, 0.0]],
    ])
    selected, origins = select_acquisition(
        masks, ensemble, torch.randn(2, 2), torch.zeros(2), budget=3,
        rng=torch.Generator().manual_seed(3),
    )
    assert len(selected) == len(origins) == 3
    assert set(origins) == {"promising", "uncertain", "random"}
    assert len({topology_id(mask) for mask in selected}) == len(selected)


def test_metrics_are_tie_safe() -> None:
    metrics = evaluator_metrics(torch.ones(3), torch.tensor([1.0, 2.0, 3.0]))
    assert metrics["spearman"] == 0.0
    assert metrics["top_candidate_error"] >= 0.0
    within = evaluator_metrics(
        torch.tensor([0.0, 2.0, 10.0, 12.0]), torch.tensor([0.0, 1.0, 5.0, 6.0]),
        task_ids=["a", "a", "b", "b"],
    )
    assert abs(within["within_task_spearman"] - 1.0) < 1e-8
    assert abs(within["within_task_top_candidate_error"]) < 1e-8


def test_evaluator_fit_only_optimizes_train_partition() -> None:
    class Replay:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def tensors(self, split: str):
            self.calls.append(split)
            if split == "joint_validation":
                raise ValueError("no joint validation rows")
            marker = {"train": 0.0, "mask_validation": 11.0, "meta_validation": 22.0}[split]
            return (torch.full((3, 2, 2), marker), torch.zeros(3, 2), torch.arange(3.0))

    replay, ensemble = Replay(), _ensemble()
    seen: list[tuple[bool, float]] = []
    member = ensemble.evaluators[0]
    original_forward = member.forward

    def wrapped(masks: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        seen.append((member.training, float(masks.mean())))
        return original_forward(masks, contexts)

    member.forward = wrapped  # type: ignore[method-assign]
    history = train_evaluators(ensemble, replay, epochs=1, batch_size=2, lr=1e-3, seed=5)
    assert history[0]["train_mse"] >= 0
    assert "mask_validation_mse" in history[0] and "meta_validation_spearman" in history[0]
    assert replay.calls == ["train", "mask_validation", "meta_validation", "joint_validation"]
    assert all(marker == 0.0 for training, marker in seen if training)
    state = ensemble.training_state
    assert state["member_count"] == 2
    assert len(state["optimizer_states"]) == len(state["bootstrap_rng_states"]) == 2
