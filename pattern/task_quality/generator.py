"""Hidden-column/map-set mask generators for task-conditioned MLP masks."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .core import ACTIVE_EDGES, HIDDEN, SEQ_LEN


PROBE_SIZE = 128
FEATURE_DIM = PROBE_SIZE + 5 * SEQ_LEN + 4


def exact_topk_ste(logits: torch.Tensor, k: int = ACTIVE_EDGES) -> torch.Tensor:
    """Hard global top-k mask in the forward pass, sigmoid surrogate backward."""
    if logits.shape[-2:] != (SEQ_LEN, HIDDEN):
        raise ValueError(f"mask logits must end in [{SEQ_LEN}, {HIDDEN}]")
    flat = logits.reshape(*logits.shape[:-2], SEQ_LEN * HIDDEN)
    if not 0 <= k <= flat.size(-1):
        raise ValueError("invalid top-k count")
    indices = flat.topk(k, dim=-1, sorted=False).indices
    hard = torch.zeros_like(flat).scatter_(-1, indices, 1.0)
    soft = torch.sigmoid(flat)
    # Parenthesized subtraction is required: hard + soft - soft.detach()
    # can round away from {0,1} in float32 and reduced precision.
    straight_through = hard + (soft - soft.detach())
    return straight_through.reshape_as(logits)


def permute_hidden_columns(feature: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """Jointly relabel all features of the eight teacher hidden columns."""
    if feature.ndim != 3 or feature.size(1) != HIDDEN or feature.size(-1) != FEATURE_DIM:
        raise ValueError(f"feature must have shape [maps, {HIDDEN}, {FEATURE_DIM}]")
    permutation = permutation.to(device=feature.device, dtype=torch.long)
    expected = torch.arange(HIDDEN, device=feature.device)
    if permutation.shape == (HIDDEN,):
        if not torch.equal(permutation.sort().values, expected):
            raise ValueError("permutation must contain every hidden column exactly once")
        return feature.index_select(1, permutation)
    if permutation.shape == (feature.size(0), HIDDEN):
        if not torch.equal(permutation.sort(dim=1).values,
                           expected.expand(feature.size(0), -1)):
            raise ValueError("each map permutation must contain every hidden column once")
        return feature.gather(1, permutation[:, :, None].expand_as(feature))
    raise ValueError("permutation must have shape [8] or [maps, 8]")


class _NoPositionEncoder(nn.Module):
    """Self-attention stack with no token positions and no dropout."""

    def __init__(self, width: int = 64, layers: int = 1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=2,
            dim_feedforward=128,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                             enable_nested_tensor=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)


class Generator(nn.Module):
    """Produce a fixed-cardinality mask from a functional bank and support set.

    Teacher hidden columns are encoded with shared token weights, attention
    without hidden positions, and mean pooling.  Maps are aggregated through
    learned inducing queries, so both hidden-column and map permutations leave
    the bank representation unchanged.  Input positions remain ordered and
    receive explicit decoder identities.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, mode: str = "transformer_mask"):
        super().__init__()
        if mode not in ("transformer_mask", "free_mask"):
            raise ValueError("mode must be 'transformer_mask' or 'free_mask'")
        if feature_dim != FEATURE_DIM:
            raise ValueError(f"expected feature_dim={FEATURE_DIM}, got {feature_dim}")
        self.feature_dim = int(feature_dim)
        self.mode = mode
        if mode == "free_mask":
            self.mask_logits = nn.Parameter(torch.zeros(SEQ_LEN, HIDDEN))
            return

        self.token_mlp = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU()
        )
        self.map_encoder = _NoPositionEncoder(width=64, layers=1)
        self.inducing_queries = nn.Parameter(torch.randn(16, 64) * 0.02)
        self.bank_set_attention = nn.MultiheadAttention(64, 2, dropout=0.0, batch_first=True)
        self.bank_norm = nn.LayerNorm(64)

        self.support_mlp = nn.Sequential(
            nn.Linear(SEQ_LEN + 1, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU()
        )
        self.support_encoder = _NoPositionEncoder(width=64, layers=1)
        self.support_norm = nn.LayerNorm(64)
        self.context_mlp = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU()
        )

        self.decoder_queries = nn.Parameter(torch.randn(HIDDEN, 64) * 0.02)
        self.input_position_embeddings = nn.Parameter(torch.randn(SEQ_LEN, 64) * 0.02)
        self.edge_decoder = nn.Sequential(
            nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 1)
        )

    def _bank_context(self, bank_feature: torch.Tensor) -> torch.Tensor:
        if bank_feature.ndim != 3 or bank_feature.size(1) != HIDDEN:
            raise ValueError(f"bank_feature must have shape [maps, {HIDDEN}, {FEATURE_DIM}]")
        if bank_feature.size(-1) != self.feature_dim or bank_feature.size(0) < 1:
            raise ValueError("bank feature dimension or map count is invalid")
        map_tokens = self.token_mlp(bank_feature)
        map_tokens = self.map_encoder(map_tokens).mean(dim=1)
        queries = self.inducing_queries.unsqueeze(0)
        induced, _ = self.bank_set_attention(queries, map_tokens.unsqueeze(0),
                                             map_tokens.unsqueeze(0), need_weights=False)
        return self.bank_norm(induced.mean(dim=1).squeeze(0))

    def _support_context(self, context_x: torch.Tensor,
                         context_y: torch.Tensor) -> torch.Tensor:
        if context_x.ndim == 2:
            context_x = context_x.unsqueeze(0)
            context_y = context_y.unsqueeze(0)
        if context_x.ndim != 3 or context_x.size(-1) != SEQ_LEN:
            raise ValueError(f"context_x must have shape [batch, support, {SEQ_LEN}]")
        if context_y.shape != context_x.shape[:2]:
            raise ValueError("context_y must align with [batch, support]")
        batch, n_support, _ = context_x.shape
        labels = context_y.to(device=context_x.device, dtype=context_x.dtype).unsqueeze(-1)
        examples = torch.cat((context_x, labels), dim=-1)
        tokens = self.support_mlp(examples)
        tokens = self.support_encoder(tokens)
        return self.support_norm(tokens.mean(dim=1))

    def forward(
        self,
        bank_feature: Optional[torch.Tensor],
        context_x: torch.Tensor,
        context_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context_x.ndim == 2:
            context_x = context_x.unsqueeze(0)
            context_y = context_y.unsqueeze(0)
        batch = context_x.size(0)
        if self.mode == "free_mask":
            logits = self.mask_logits.unsqueeze(0).expand(batch, -1, -1)
            return exact_topk_ste(logits), logits
        if bank_feature is None:
            raise ValueError("transformer_mask requires a functional bank feature tensor")
        bank_feature = bank_feature.to(device=context_x.device, dtype=context_x.dtype)
        bank_context = self._bank_context(bank_feature).unsqueeze(0).expand(batch, -1)
        support_context = self._support_context(context_x, context_y)
        context = self.context_mlp(torch.cat((bank_context, support_context), dim=-1))

        context_token = context[:, None, None, :].expand(batch, SEQ_LEN, HIDDEN, 64)
        position = self.input_position_embeddings[None, :, None, :].expand(
            batch, SEQ_LEN, HIDDEN, 64)
        hidden_query = self.decoder_queries[None, None, :, :].expand(
            batch, SEQ_LEN, HIDDEN, 64)
        edge_features = torch.cat((context_token, position, hidden_query), dim=-1)
        logits = self.edge_decoder(edge_features).squeeze(-1)
        return exact_topk_ste(logits), logits


def generate(
    model: Generator,
    bank_feature: Optional[torch.Tensor],
    context_x: torch.Tensor,
    context_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Public functional API shared by meta-training and evaluation."""
    return model(bank_feature, context_x, context_y)
