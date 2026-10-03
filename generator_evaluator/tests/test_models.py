import torch

from generator_evaluator.models import (
    MaskQualityEvaluator,
    QualityEnsemble,
    TransformerMaskGenerator,
    permute_bank,
)


def _generator() -> TransformerMaskGenerator:
    return TransformerMaskGenerator(
        token_dim=5, features=4, hidden=3, width=16, heads=4, layers=1, noise_dim=6, quality_dim=2
    )


def test_generator_is_invariant_to_matched_bank_permutations() -> None:
    torch.manual_seed(7)
    model = _generator().eval()
    tokens = torch.randn(2, 4, 3, 5)
    quality = torch.randn(2, 4, 2)
    noise = torch.randn(2, 6)
    shuffled_tokens, shuffled_quality = permute_bank(
        tokens, quality, generator=torch.Generator().manual_seed(8)
    )
    original = model(tokens, noise, quality)
    shuffled = model(shuffled_tokens, noise, shuffled_quality)
    torch.testing.assert_close(original, shuffled, rtol=2e-5, atol=2e-6)
    # ``encode_bank`` keeps solution rows, hence is equivariant to their order;
    # its pooled representation is invariant.
    torch.testing.assert_close(
        model.encode_bank(tokens, quality).mean(dim=1),
        model.encode_bank(shuffled_tokens, shuffled_quality).mean(dim=1),
        rtol=2e-5,
        atol=2e-6,
    )


def test_generator_is_invariant_to_independent_hidden_permutations_per_solution() -> None:
    torch.manual_seed(9)
    model = _generator().eval()
    batch, solutions, hidden = 2, 4, 3
    tokens = torch.randn(batch, solutions, hidden, 5)
    quality = torch.randn(batch, solutions, 2)
    noise = torch.randn(batch, 6)
    solution_orders = torch.stack([torch.randperm(solutions) for _ in range(batch)])
    orders = torch.stack([
        torch.stack([torch.randperm(hidden) for _ in range(solutions)])
        for _ in range(batch)
    ])
    permuted = torch.stack([
        torch.stack([tokens[b, solution_orders[b, r]].index_select(0, orders[b, r])
                     for r in range(solutions)])
        for b in range(batch)
    ])
    permuted_quality = torch.stack([
        quality[b].index_select(0, solution_orders[b]) for b in range(batch)
    ])

    original = model(tokens, noise, quality)
    reordered = model(permuted, noise, permuted_quality)
    torch.testing.assert_close(original, reordered, rtol=2e-5, atol=2e-6)


def test_permute_bank_preserves_quality_solution_association() -> None:
    tokens = torch.zeros(1, 5, 4, 2)
    quality = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
    for solution in range(5):
        tokens[:, solution, :, 0] = float(solution)
    shuffled_tokens, shuffled_quality = permute_bank(
        tokens, quality, generator=torch.Generator().manual_seed(23)
    )
    torch.testing.assert_close(shuffled_tokens[:, :, 0, 0], shuffled_quality[:, :, 0])


def test_evaluator_is_column_permutation_invariant_and_accepts_varying_hidden_counts() -> None:
    torch.manual_seed(3)
    evaluator = MaskQualityEvaluator(features=4, context_dim=3, width=16, heads=4, layers=1).eval()
    masks = torch.randn(2, 4, 5)
    context = torch.randn(2, 3)
    reordered = masks[:, :, torch.tensor([3, 0, 4, 1, 2])]
    torch.testing.assert_close(evaluator(masks, context), evaluator(reordered, context), rtol=2e-5, atol=2e-6)
    assert evaluator(torch.randn(2, 4, 2), context).shape == (2,)


def test_models_respond_to_inputs_and_backpropagate() -> None:
    torch.manual_seed(11)
    generator = _generator()
    tokens = torch.randn(2, 3, 3, 5, requires_grad=True)
    quality = torch.randn(2, 3, 2, requires_grad=True)
    noise = torch.randn(2, 6, requires_grad=True)
    logits = generator(tokens, noise, quality)
    changed_logits = generator(tokens.detach() + 0.5, noise.detach(), quality.detach())
    assert logits.shape == (2, 4, 3)
    assert not torch.allclose(logits.detach(), changed_logits)
    logits.square().mean().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert noise.grad is not None and torch.isfinite(noise.grad).all()
    assert generator.token_projection.weight.grad is not None

    evaluator = MaskQualityEvaluator(features=4, context_dim=3, width=16, heads=4, layers=1)
    masks = torch.randn(2, 4, 3, requires_grad=True)
    context = torch.randn(2, 3, requires_grad=True)
    scores = evaluator(masks, context)
    assert not torch.allclose(scores.detach(), evaluator(masks.detach(), context.detach() + 0.5))
    scores.sum().backward()
    assert masks.grad is not None and context.grad is not None


def test_multihead_key_bias_gradients_are_zero_but_query_value_remain_active() -> None:
    torch.manual_seed(21)
    generator = _generator()
    tokens = torch.randn(2, 3, 3, 5)
    quality = torch.randn(2, 3, 2)
    noise = torch.randn(2, 6)
    generator(tokens, noise, quality).square().mean().backward()
    generator_attentions = [
        module for module in generator.modules()
        if isinstance(module, torch.nn.MultiheadAttention)
    ]
    assert generator_attentions
    generator_qv_gradient = 0.0
    for attention in generator_attentions:
        gradient = attention.in_proj_bias.grad
        assert gradient is not None
        width = attention.embed_dim
        assert torch.count_nonzero(gradient[width:2 * width]) == 0
        generator_qv_gradient += float(gradient[:width].abs().sum() + gradient[2 * width:].abs().sum())
    assert generator_qv_gradient > 0.0

    critic = MaskQualityEvaluator(features=4, context_dim=3, width=16, heads=4, layers=1)
    masks = torch.randn(2, 4, 5)
    context = torch.randn(2, 3)
    critic(masks, context).square().mean().backward()
    critic_attentions = [
        module for module in critic.modules()
        if isinstance(module, torch.nn.MultiheadAttention)
    ]
    assert critic_attentions
    critic_qv_gradient = 0.0
    for attention in critic_attentions:
        gradient = attention.in_proj_bias.grad
        assert gradient is not None
        width = attention.embed_dim
        assert torch.count_nonzero(gradient[width:2 * width]) == 0
        critic_qv_gradient += float(gradient[:width].abs().sum() + gradient[2 * width:].abs().sum())
    assert critic_qv_gradient > 0.0


def test_ensemble_predicts_mean_and_population_standard_deviation() -> None:
    torch.manual_seed(13)
    ensemble = QualityEnsemble(features=4, context_dim=3, num_members=3, width=16, heads=4, layers=1)
    masks = torch.randn(2, 4, 3)
    context = torch.randn(2, 3)
    mean, std = ensemble.predict(masks, context)
    raw = torch.stack([member(masks, context) for member in ensemble.evaluators])
    torch.testing.assert_close(mean, raw.mean(dim=0))
    torch.testing.assert_close(std, raw.std(dim=0, unbiased=False))
    assert mean.shape == std.shape == (2,)
    assert torch.all(std >= 0)
