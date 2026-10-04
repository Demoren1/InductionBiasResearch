"""Permutation-respecting Transformer models for mask generation and scoring.

The bank axes (solutions and their hidden neurons) deliberately have no
positional embeddings.  The output coordinates, on the other hand, are
represented by learned fixed queries and feature embeddings.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _validate_architecture(width: int, heads: int, layers: int) -> None:
    if width <= 0 or heads <= 0 or layers <= 0:
        raise ValueError("width, heads, and layers must be positive")
    if width % heads:
        raise ValueError("width must be divisible by heads")


def _validate_floating_tensor(value: Tensor, name: str, dimensions: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions, got {value.ndim}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _zero_mha_key_bias_gradients(module: nn.Module) -> None:
    """Remove roundoff gradients for the softmax-invariant MHA key bias.

    ``MultiheadAttention.in_proj_bias`` packs Q, K, and V biases in that
    order.  The K bias adds the same query-dependent constant to every key
    score, which softmax cancels exactly.  Floating-point roundoff can leave a
    tiny nonzero gradient; Adam can amplify it across resumed runs, so suppress
    only the redundant middle third while retaining the Q and V gradients.
    This changes no parameter names or checkpoint contents.
    """
    for attention in module.modules():
        if not isinstance(attention, nn.MultiheadAttention):
            continue
        bias = attention.in_proj_bias
        width = attention.embed_dim
        if bias is None or bias.numel() != 3 * width:
            continue

        def clear_key_component(gradient: Tensor, *, width: int = width) -> Tensor:
            return torch.cat((gradient[:width], torch.zeros_like(gradient[width:2 * width]),
                              gradient[2 * width:]))

        bias.register_hook(clear_key_component)


class TransformerMaskGenerator(nn.Module):
    """Generate all feature-to-hidden mask logits from a solution bank.

    ``tokens`` holds per-neuron functional profiles with shape ``[B, R, H, D]``.
    The neuron encoder is shared among solutions.  Its per-neuron states are
    retained for the output decoder, while their mean gives each solution a
    global summary.  The decoder reads both memories, so it can use local
    functional structure without relying on a source-neuron index.  All
    attention stages are set Transformers: none receives a position for a
    bank solution or source hidden neuron.
    """

    def __init__(
        self,
        token_dim: int,
        features: int,
        hidden: int,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        noise_dim: int = 16,
        quality_dim: int = 1,
        *,
        generator_bilinear_head: bool = False,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or features <= 0 or hidden <= 0 or noise_dim <= 0 or quality_dim <= 0:
            raise ValueError("token_dim, features, hidden, noise_dim, and quality_dim must be positive")
        if not isinstance(generator_bilinear_head, bool):
            raise TypeError("generator_bilinear_head must be a bool")
        _validate_architecture(width, heads, layers)
        self.token_dim = token_dim
        self.features = features
        self.hidden = hidden
        self.noise_dim = noise_dim
        self.quality_dim = quality_dim
        self.width = width
        self.generator_bilinear_head = generator_bilinear_head

        self.token_projection = nn.Linear(token_dim, width)
        self.quality_projection = nn.Linear(quality_dim, width, bias=False)
        self.noise_projection = nn.Linear(noise_dim, width, bias=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            dropout=0.0, activation="gelu", batch_first=True,
        )
        self.neuron_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        bank_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            dropout=0.0, activation="gelu", batch_first=True,
        )
        self.bank_encoder = nn.TransformerEncoder(bank_layer, num_layers=layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            dropout=0.0, activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)

        self.output_queries = nn.Parameter(torch.empty(hidden, width))
        self.feature_embeddings = nn.Parameter(torch.empty(features, width))
        self.output_head = nn.Sequential(
            nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1)
        )
        nn.init.normal_(self.output_queries, std=width ** -0.5)
        nn.init.normal_(self.feature_embeddings, std=width ** -0.5)
        _zero_mha_key_bias_gradients(self)

    def _validate_bank(self, tokens: Tensor, quality: Tensor | None) -> tuple[int, int]:
        _validate_floating_tensor(tokens, "tokens", 4)
        batch, solutions, source_hidden, token_dim = tokens.shape
        if solutions == 0 or source_hidden == 0:
            raise ValueError("tokens must contain at least one solution and one hidden neuron")
        if token_dim != self.token_dim:
            raise ValueError(f"tokens last dimension must be {self.token_dim}, got {token_dim}")
        if quality is not None:
            _validate_floating_tensor(quality, "quality", 3)
            if quality.shape != (batch, solutions, self.quality_dim):
                raise ValueError(
                    "quality must have shape "
                    f"[{batch}, {solutions}, {self.quality_dim}], got {list(quality.shape)}"
                )
        return batch, solutions

    def encode_bank(self, tokens: Tensor, quality: Tensor | None = None) -> Tensor:
        """Return invariantly encoded solution memory with shape ``[B, R, width]``."""
        batch, solutions = self._validate_bank(tokens, quality)
        return self._encode_bank_validated(tokens, quality, batch, solutions)

    def _encode_bank_components_validated(
        self,
        tokens: Tensor,
        quality: Tensor | None,
        batch: int | None = None,
        solutions: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode global solution summaries and their per-neuron memories."""
        if batch is None or solutions is None:
            batch, solutions = tokens.shape[:2]
        source_hidden = tokens.shape[2]
        neurons = self.token_projection(tokens).reshape(batch * solutions, source_hidden, self.width)
        encoded_neurons = self.neuron_encoder(neurons)
        encoded_neurons = encoded_neurons.reshape(batch, solutions, source_hidden, self.width)
        solution_tokens = encoded_neurons.mean(dim=2)
        if quality is not None:
            solution_tokens = solution_tokens + self.quality_projection(quality)
        solution_memory = self.bank_encoder(solution_tokens)
        return solution_memory, encoded_neurons

    def _encode_bank_validated(self, tokens: Tensor, quality: Tensor | None,
                              batch: int | None = None, solutions: int | None = None) -> Tensor:
        """Return invariant solution summaries after public validation."""
        solution_memory, _ = self._encode_bank_components_validated(
            tokens, quality, batch, solutions
        )
        return solution_memory

    def forward(self, tokens: Tensor, noise: Tensor, quality: Tensor | None = None) -> Tensor:
        """Return connection logits in the fixed target coordinates ``[B, F, H]``."""
        batch, _ = self._validate_bank(tokens, quality)
        _validate_floating_tensor(noise, "noise", 2)
        if noise.shape != (batch, self.noise_dim):
            raise ValueError(
                f"noise must have shape [{batch}, {self.noise_dim}], got {list(noise.shape)}"
            )

        return self._forward_validated(tokens, noise, quality, batch)

    def _forward_validated(self, tokens: Tensor, noise: Tensor, quality: Tensor | None,
                           batch: int | None = None) -> Tensor:
        """Fast path matching :meth:`forward` after argument validation."""
        if batch is None:
            batch = tokens.shape[0]
        solution_memory, neuron_memory = self._encode_bank_components_validated(
            tokens, quality, batch
        )
        return self._decode_from_memories_validated(solution_memory, neuron_memory, noise, batch)

    def _decode_from_memories_validated(self, solution_memory: Tensor, neuron_memory: Tensor,
                                        noise: Tensor, batch: int | None = None) -> Tensor:
        """Decode per-row noise from reusable solution and neuron memories."""
        if batch is None:
            batch = noise.shape[0]
        noise_memory = self.noise_projection(noise).unsqueeze(1)
        solution_memory = solution_memory + noise_memory
        # Attach each neuron's within-solution profile to its contextualized
        # teacher summary.  Flattening only changes storage order: decoder
        # cross-attention treats these memories as a set.
        neuron_memory = neuron_memory + solution_memory.unsqueeze(2)
        neuron_memory = neuron_memory.reshape(
            batch, -1, self.width
        )
        memory = torch.cat((solution_memory, neuron_memory), dim=1)
        queries = self.output_queries.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        hidden_terms = decoded.unsqueeze(1).expand(-1, self.features, -1, -1)
        feature_terms = self.feature_embeddings.unsqueeze(0).unsqueeze(2).expand(
            batch, -1, self.hidden, -1
        )
        logits = self.output_head(torch.cat((hidden_terms, feature_terms), dim=-1)).squeeze(-1)
        if self.generator_bilinear_head:
            logits = logits + (hidden_terms * feature_terms).sum(dim=-1) / (self.width ** 0.5)
        return logits


class MaskQualityEvaluator(nn.Module):
    """Predict a scalar quality for a mask and a task context.

    Each hidden-column is a set item.  A distinguished context token reads the
    column set after Transformer attention, so changing column order cannot
    change the prediction and the number of columns may vary between calls.
    """

    def __init__(
        self,
        features: int,
        context_dim: int,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if features <= 0 or context_dim <= 0:
            raise ValueError("features and context_dim must be positive")
        _validate_architecture(width, heads, layers)
        self.features = features
        self.context_dim = context_dim
        self.width = width
        self.column_projection = nn.Linear(features, width)
        self.context_projection = nn.Linear(context_dim, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            dropout=0.0, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        _zero_mha_key_bias_gradients(self)

    def forward(self, masks: Tensor, context: Tensor) -> Tensor:
        _validate_floating_tensor(masks, "masks", 3)
        _validate_floating_tensor(context, "context", 2)
        batch, features, hidden = masks.shape
        if features != self.features:
            raise ValueError(f"masks feature dimension must be {self.features}, got {features}")
        if hidden == 0:
            raise ValueError("masks must contain at least one hidden column")
        if context.shape != (batch, self.context_dim):
            raise ValueError(
                f"context must have shape [{batch}, {self.context_dim}], got {list(context.shape)}"
            )
        return self._forward_validated(masks, context)

    def _forward_validated(self, masks: Tensor, context: Tensor) -> Tensor:
        """Fast path matching :meth:`forward` after argument validation."""
        columns = self.column_projection(masks.transpose(1, 2))
        context_token = self.context_projection(context).unsqueeze(1)
        encoded = self.encoder(torch.cat((context_token, columns), dim=1))
        return self.readout(encoded[:, 0]).squeeze(-1)


class QualityEnsemble(nn.Module):
    """Independent quality evaluators with a mean and population uncertainty."""

    def __init__(
        self,
        features: int | Sequence[MaskQualityEvaluator] | None = None,
        context_dim: int | None = None,
        *,
        num_members: int = 3,
        members: int | None = None,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        evaluators: Sequence[MaskQualityEvaluator] | None = None,
    ) -> None:
        super().__init__()
        if evaluators is None and isinstance(features, Sequence) and not isinstance(features, (str, bytes)):
            evaluators = features
            features = None
        if evaluators is not None:
            if len(evaluators) == 0:
                raise ValueError("evaluators must not be empty")
            if not all(isinstance(evaluator, MaskQualityEvaluator) for evaluator in evaluators):
                raise TypeError("evaluators must contain MaskQualityEvaluator instances")
            first = evaluators[0]
            if any(
                evaluator.features != first.features or evaluator.context_dim != first.context_dim
                for evaluator in evaluators
            ):
                raise ValueError("all evaluators must use the same input dimensions")
            self.evaluators = nn.ModuleList(evaluators)
            return
        if not isinstance(features, int) or not isinstance(context_dim, int):
            raise TypeError("features and context_dim are required when evaluators are not supplied")
        count = num_members if members is None else members
        if count <= 0:
            raise ValueError("num_members must be positive")
        self.evaluators = nn.ModuleList(
            MaskQualityEvaluator(features, context_dim, width, heads, layers)
            for _ in range(count)
        )

    def predict(self, masks: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        predictions = torch.stack([evaluator(masks, context) for evaluator in self.evaluators], dim=0)
        return predictions.mean(dim=0), predictions.std(dim=0, unbiased=False)

    def _predict_validated(self, masks: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        """Avoid repeated finite/shape scans during an already validated update."""
        predictions = torch.stack([
            evaluator._forward_validated(masks, context)
            if isinstance(evaluator, MaskQualityEvaluator) else evaluator(masks, context)
            for evaluator in self.evaluators
        ], dim=0)
        return predictions.mean(dim=0), predictions.std(dim=0, unbiased=False)

    def forward(self, masks: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        return self.predict(masks, context)


def permute_bank(
    tokens: Tensor,
    quality: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Jointly shuffle bank solutions and source hidden-neuron columns.

    One permutation is used for every batch member, which makes the operation
    suitable for a direct matched-noise consistency comparison.  Quality rows
    follow their corresponding solution rows exactly.
    """
    _validate_floating_tensor(tokens, "tokens", 4)
    batch, solutions, hidden, _ = tokens.shape
    if solutions == 0 or hidden == 0:
        raise ValueError("tokens must contain at least one solution and one hidden neuron")
    if quality is not None:
        _validate_floating_tensor(quality, "quality", 3)
        if quality.shape[0] != batch or quality.shape[1] != solutions:
            raise ValueError(
                f"quality must start with shape [{batch}, {solutions}], got {list(quality.shape)}"
            )
    solution_order = torch.randperm(solutions, generator=generator, device=tokens.device)
    hidden_order = torch.randperm(hidden, generator=generator, device=tokens.device)
    permuted_tokens = tokens.index_select(1, solution_order).index_select(2, hidden_order)
    permuted_quality = None if quality is None else quality.index_select(1, solution_order)
    return permuted_tokens, permuted_quality
