"""Train-only raw-teacher tokens and a hidden-column set encoder.

The functional bank artifact stores per-teacher moments after a label-free
hidden alignment.  This module selects only saved train rows, reverses that
alignment, and keeps each teacher as an independent set element.  No task
quality or audit metric is used unless source-query quality is explicitly
enabled; audit fields are never read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


FUNCTIONAL_CHANNELS = (
    "psi_norm",
    "q_signed_norm",
    "q_abs_norm",
    "q_rms_norm",
    "training_mask",
)


def _torch_load_cpu(path: Path) -> dict:
    """Load a tensor artifact with lazy CPU-backed storage when supported."""
    try:
        result = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        result = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(result, dict):
        raise ValueError(f"expected a dictionary artifact at {path}")
    return result


def _state_tensor(bank: dict, name: str) -> Tensor:
    state = bank.get("state_dict", bank)
    value = state.get(name) if isinstance(state, dict) else None
    if value is None:
        value = bank.get(name)
    if not torch.is_tensor(value):
        raise ValueError(f"source bank lacks tensor field {name!r}")
    return value


def _undo_saved_alignment(values: Tensor, destination_to_source: Tensor) -> Tensor:
    """Restore raw hidden order from values gathered into aligned order."""
    if values.ndim != 3 or destination_to_source.ndim != 2:
        raise ValueError("values and alignment orders must have shapes [N,F,H] and [N,H]")
    if values.shape[0] != destination_to_source.shape[0] or values.shape[-1] != destination_to_source.shape[-1]:
        raise ValueError("alignment-order dimensions do not match teacher values")
    expected = torch.arange(values.shape[-1], dtype=torch.long)
    if not torch.equal(destination_to_source.sort(dim=-1).values, expected.expand_as(destination_to_source)):
        raise ValueError("saved alignment order is not a hidden-column permutation")
    source_to_destination = destination_to_source.argsort(dim=-1)
    gather_index = source_to_destination[:, None, :].expand_as(values)
    return values.gather(-1, gather_index)


@dataclass(frozen=True)
class TeacherBatch:
    """A sampled teacher set and its provenance."""

    tokens: Tensor
    references: Tensor
    source_query_quality: Tensor | None


class TrainOnlyFunctionalTeacherBank:
    """Lazy registry of individual teachers from a saved functional context.

    ``gather`` and ``sample`` return teacher tokens in the original raw hidden
    column order. References are ``[source_task_index, teacher_row]`` pairs;
    every reference is checked against the artifact's ``train_rows`` before
    any features are returned.

    Token channel order is ``psi_norm``, ``q_signed_norm``, ``q_abs_norm``,
    ``q_rms_norm``, then the individual teacher's raw training mask when
    ``include_training_masks`` is true. Each channel is arranged as
    ``[teacher, hidden, channel_width]`` before concatenation.
    """

    def __init__(
        self,
        context_path: str | Path,
        *,
        include_source_query_quality: bool = False,
        include_training_masks: bool = True,
    ) -> None:
        self.context_path = Path(context_path).resolve()
        if not self.context_path.is_file():
            raise FileNotFoundError(self.context_path)
        self._context = _torch_load_cpu(self.context_path)
        required = (
            "psi", "q_signed_mean", "q_abs_mean", "q_rms", "train_rows",
            "alignment_orders", "bank_references",
        )
        missing = [key for key in required if key not in self._context]
        if missing:
            raise ValueError(f"functional context lacks required fields: {missing}")

        self.include_source_query_quality = bool(include_source_query_quality)
        self.include_training_masks = bool(include_training_masks)
        self._task_count, self._teacher_count, self.probe_count, self.hidden = self._shape(
            self._context["psi"], 4, "psi"
        )
        if self._context["psi"].shape[1] != self._teacher_count:
            raise ValueError("psi teacher dimension is inconsistent")
        self.features = int(self._context["q_signed_mean"].shape[2])
        for name in ("q_signed_mean", "q_abs_mean", "q_rms"):
            value = self._context[name]
            if tuple(value.shape) != (self._task_count, self._teacher_count, self.features, self.hidden):
                raise ValueError(f"{name} has an incompatible shape")
        if tuple(self._context["train_rows"].shape[:1]) != (self._task_count,):
            raise ValueError("train_rows task dimension does not match functional features")
        if tuple(self._context["alignment_orders"].shape) != (
            self._task_count, self._teacher_count, self.hidden
        ):
            raise ValueError("alignment_orders has an incompatible shape")
        references = self._context["bank_references"]
        if len(references) != self._task_count:
            raise ValueError("bank_references task dimension does not match functional features")

        train_rows = self._context["train_rows"].detach().cpu().long()
        self._train_rows = [set(int(row) for row in rows.tolist()) for rows in train_rows]
        if any(not rows for rows in self._train_rows):
            raise ValueError("each source task must have at least one train teacher")
        all_refs = [
            (task, row)
            for task, rows in enumerate(train_rows)
            for row in rows.tolist()
        ]
        self.train_references = torch.tensor(all_refs, dtype=torch.long)

        self._source_banks: list[dict] = []
        for task, reference in enumerate(references):
            path_value = reference.get("path") if isinstance(reference, dict) else None
            if not path_value:
                raise ValueError(f"bank_references[{task}] lacks a source bank path")
            bank_path = Path(path_value)
            if not bank_path.is_file():
                raise FileNotFoundError(bank_path)
            bank = _torch_load_cpu(bank_path)
            masks = _state_tensor(bank, "masks") if self.include_training_masks else None
            if masks is not None and tuple(masks.shape) != (
                self._teacher_count, self.features, self.hidden
            ):
                raise ValueError(f"source mask shape does not match context task {task}")
            if self.include_source_query_quality:
                query = bank.get("queryNMSE")
                if not torch.is_tensor(query) or tuple(query.shape) != (self._teacher_count,):
                    raise ValueError(
                        f"source bank {task} lacks per-teacher queryNMSE for source-query quality input"
                    )
            self._source_banks.append(bank)

        self.token_dim = self.probe_count + 3 * self.features
        if self.include_training_masks:
            self.token_dim += self.features
        self.channel_slices = self._make_channel_slices()

        self.source_query_mean: Tensor | None = None
        self.source_query_std: Tensor | None = None
        if self.include_source_query_quality:
            quality = torch.cat([
                self._source_banks[task]["queryNMSE"][rows].detach().float().cpu()
                for task, rows in enumerate(train_rows)
            ])
            if not torch.isfinite(quality).all():
                raise ValueError("train-row source queryNMSE contains non-finite values")
            self.source_query_mean = quality.mean()
            self.source_query_std = quality.std(unbiased=False).clamp_min(1e-8)

    @staticmethod
    def _shape(value: Tensor, ndim: int, name: str) -> tuple[int, ...]:
        if not torch.is_tensor(value) or value.ndim != ndim:
            raise ValueError(f"{name} must have {ndim} dimensions")
        return tuple(int(dim) for dim in value.shape)

    def _make_channel_slices(self) -> dict[str, slice]:
        widths = [self.probe_count, self.features, self.features, self.features]
        names = list(FUNCTIONAL_CHANNELS[:4])
        if self.include_training_masks:
            widths.append(self.features)
            names.append(FUNCTIONAL_CHANNELS[4])
        start = 0
        slices: dict[str, slice] = {}
        for name, width in zip(names, widths):
            slices[name] = slice(start, start + width)
            start += width
        return slices

    def _validate_references(self, references: Tensor) -> Tensor:
        refs = torch.as_tensor(references, dtype=torch.long, device="cpu")
        if refs.ndim != 3 or refs.shape[-1] != 2:
            raise ValueError("references must have shape [B,T,2] as (task,row) pairs")
        if refs.numel() == 0:
            raise ValueError("references cannot be empty")
        for task, row in refs.reshape(-1, 2).tolist():
            if not 0 <= task < self._task_count:
                raise ValueError(f"source task index {task} is out of range")
            if row not in self._train_rows[task]:
                raise ValueError(f"teacher row {row} for task {task} is not in saved train_rows")
        return refs

    def gather(self, references: Tensor) -> TeacherBatch:
        """Gather explicitly selected train teachers into ``[B,T,H,D]`` tokens."""
        refs = self._validate_references(references)
        batch, teacher_count, _ = refs.shape
        flat_refs = refs.reshape(-1, 2)
        token_rows = torch.empty(
            (flat_refs.shape[0], self.hidden, self.token_dim), dtype=torch.float32
        )
        quality_rows = (
            torch.empty((flat_refs.shape[0],), dtype=torch.float32)
            if self.include_source_query_quality else None
        )

        for task in flat_refs[:, 0].unique(sorted=True).tolist():
            positions = torch.nonzero(flat_refs[:, 0] == task, as_tuple=False).reshape(-1)
            rows = flat_refs[positions, 1]
            context = self._context
            alignment = context["alignment_orders"][task, rows].detach().long().cpu()
            raw_psi = context["psi"][task, rows].detach().float().cpu()
            raw_q_signed = context["q_signed_mean"][task, rows].detach().float().cpu()
            raw_q_abs = context["q_abs_mean"][task, rows].detach().float().cpu()
            raw_q_rms = context["q_rms"][task, rows].detach().float().cpu()

            psi = _undo_saved_alignment(raw_psi, alignment)
            q_signed = _undo_saved_alignment(raw_q_signed, alignment)
            q_abs = _undo_saved_alignment(raw_q_abs, alignment)
            q_rms = _undo_saved_alignment(raw_q_rms, alignment)

            # Per-teacher scales keep moment channels numerically comparable;
            # unlike the old pooled context, every teacher remains separate.
            psi_scale = psi.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
            q_scale = q_rms.flatten(1).amax(dim=1).clamp_min(1e-8)[:, None, None]
            channels = [
                (psi / psi_scale).transpose(1, 2),
                (q_signed / q_scale).transpose(1, 2),
                (q_abs / q_scale).transpose(1, 2),
                (q_rms / q_scale).transpose(1, 2),
            ]
            if self.include_training_masks:
                raw_mask = _state_tensor(self._source_banks[task], "masks")[rows].detach().float().cpu()
                channels.append(raw_mask.transpose(1, 2))
            values = torch.cat(channels, dim=-1)
            if not torch.isfinite(values).all():
                raise ValueError(f"non-finite functional features in source task {task}")
            token_rows[positions] = values

            if quality_rows is not None:
                assert self.source_query_mean is not None and self.source_query_std is not None
                source_query = self._source_banks[task]["queryNMSE"][rows].detach().float().cpu()
                quality_rows[positions] = (source_query - self.source_query_mean) / self.source_query_std

        tokens = token_rows.reshape(batch, teacher_count, self.hidden, self.token_dim)
        quality = quality_rows.reshape(batch, teacher_count) if quality_rows is not None else None
        return TeacherBatch(tokens=tokens, references=refs, source_query_quality=quality)

    def sample(
        self,
        batch_size: int,
        teacher_count: int,
        *,
        generator: torch.Generator | None = None,
        task_indices: Sequence[int] | None = None,
    ) -> TeacherBatch:
        """Sample train teachers with replacement, uniformly over eligible rows."""
        if batch_size < 1 or teacher_count < 1:
            raise ValueError("batch_size and teacher_count must be positive")
        if task_indices is None:
            eligible = self.train_references
        else:
            selected_tasks = sorted(set(int(task) for task in task_indices))
            if not selected_tasks or any(task < 0 or task >= self._task_count for task in selected_tasks):
                raise ValueError("task_indices must contain valid source task indices")
            task_mask = torch.zeros(len(self.train_references), dtype=torch.bool)
            for task in selected_tasks:
                task_mask |= self.train_references[:, 0] == task
            eligible = self.train_references[task_mask]
        indices = torch.randint(
            len(eligible), (batch_size, teacher_count), generator=generator, device="cpu"
        )
        return self.gather(eligible[indices])


def permute_teacher_hidden_columns(
    tokens: Tensor,
    *,
    permutation: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Jointly permute hidden-column tokens independently for each teacher.

    ``tokens`` has shape ``[B,T,H,D]``. A returned permutation uses
    destination-to-source indexing and has shape ``[B,T,H]``. All token
    channels move together, preserving each teacher's feature association.
    Decoder output slots are not involved in this operation.
    """
    if tokens.ndim != 4:
        raise ValueError("tokens must have shape [B,T,H,D]")
    batch, teachers, hidden, width = tokens.shape
    if min(batch, teachers, hidden, width) < 1:
        raise ValueError("tokens cannot contain an empty dimension")
    if permutation is None:
        keys = torch.rand(
            (batch, teachers, hidden),
            device=tokens.device,
            generator=generator,
        )
        permutation = keys.argsort(dim=-1)
    else:
        permutation = torch.as_tensor(permutation, dtype=torch.long, device=tokens.device)
    if tuple(permutation.shape) != (batch, teachers, hidden):
        raise ValueError("permutation must have shape [B,T,H]")
    expected = torch.arange(hidden, dtype=torch.long, device=tokens.device)
    if not torch.equal(permutation.sort(dim=-1).values, expected.expand_as(permutation)):
        raise ValueError("each teacher column mapping must be a permutation of [0,H)")
    gather_index = permutation[..., None].expand(batch, teachers, hidden, width)
    return tokens.gather(dim=2, index=gather_index), permutation


def permutation_consistency_loss(logits_a: Tensor, logits_b: Tensor) -> Tensor:
    """Mean squared difference between fixed-slot outputs for two input orders."""
    if logits_a.shape != logits_b.shape or logits_a.ndim != 3:
        raise ValueError("both logits tensors must have the same [B,F,H] shape")
    return F.mse_loss(logits_a, logits_b)


class RawFunctionalBankEncoder(nn.Module):
    """Set encoder from raw teacher/neuron tokens to fixed hidden decoder slots.

    The teacher and input hidden-column axes are exchangeable by default. A
    trainable column-position bias can be enabled for experiments that need a
    non-vacuous permutation-consistency ablation. The decoder's 32 slots and
    784 labelled feature identities stay fixed in either mode.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        task_context_dim: int = 1570,
        features: int = 784,
        hidden: int = 32,
        width: int = 64,
        attention_heads: int = 4,
        teacher_pool_slots: int = 4,
        column_position_bias: bool = False,
    ) -> None:
        super().__init__()
        if min(token_dim, task_context_dim, features, hidden, width, attention_heads,
               teacher_pool_slots) < 1:
            raise ValueError("all dimensions must be positive")
        if width % attention_heads:
            raise ValueError("width must be divisible by attention_heads")
        self.token_dim = token_dim
        self.task_context_dim = task_context_dim
        self.features = features
        self.hidden = hidden
        self.width = width
        self.teacher_pool_slots = teacher_pool_slots
        self.column_position_bias_enabled = bool(column_position_bias)

        self.token_encoder = nn.Sequential(
            nn.Linear(token_dim, width),
            nn.SiLU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(task_context_dim, width),
            nn.SiLU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
        )
        self.quality_encoder = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.slot_queries = nn.Parameter(torch.empty(hidden, width))
        nn.init.normal_(self.slot_queries, std=width ** -0.5)
        self.teacher_pool_queries = nn.Parameter(torch.empty(teacher_pool_slots, width))
        nn.init.normal_(self.teacher_pool_queries, std=width ** -0.5)
        self.column_position_bias = (
            nn.Parameter(torch.empty(hidden, width)) if column_position_bias else None
        )
        if self.column_position_bias is not None:
            nn.init.normal_(self.column_position_bias, std=0.02)
        self.neuron_pool = nn.MultiheadAttention(
            width, attention_heads, dropout=0.0, batch_first=True
        )
        self.bank_pool = nn.MultiheadAttention(
            width, attention_heads, dropout=0.0, batch_first=True
        )
        self.teacher_norm = nn.LayerNorm(width)
        self.slot_norm = nn.LayerNorm(width)
        self.feature_embedding = nn.Parameter(torch.empty(features, width))
        nn.init.normal_(self.feature_embedding, std=width ** -0.5)
        self.feature_decoder = nn.Linear(width, width, bias=False)
        self.slot_decoder = nn.Linear(width, width, bias=False)
        self.feature_bias = nn.Parameter(torch.zeros(features))
        self.slot_bias = nn.Parameter(torch.zeros(hidden))

    def encode(
        self,
        tokens: Tensor,
        task: Tensor,
        teacher_quality: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return global embedding and fixed-slot mask logits."""
        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [B,T,H,D]")
        batch, teacher_count, hidden, token_dim = tokens.shape
        if teacher_count < 1 or hidden != self.hidden or token_dim != self.token_dim:
            raise ValueError(
                f"tokens must have shape [B,T,{self.hidden},{self.token_dim}] with T >= 1"
            )
        if task.shape != (batch, self.task_context_dim):
            raise ValueError(f"task must have shape [B,{self.task_context_dim}]")
        if not torch.isfinite(tokens).all() or not torch.isfinite(task).all():
            raise ValueError("tokens and task must be finite")

        encoded = self.token_encoder(tokens)
        if self.column_position_bias is not None:
            encoded = encoded + self.column_position_bias[None, None, :, :]
        condition = self.task_encoder(task)
        encoded = encoded + condition[:, None, None, :]

        if teacher_quality is not None:
            if teacher_quality.shape != (batch, teacher_count):
                raise ValueError(f"teacher_quality must have shape [B,{teacher_count}]")
            if not torch.isfinite(teacher_quality).all():
                raise ValueError("teacher_quality must be finite")
            quality_embedding = self.quality_encoder(teacher_quality[..., None])
            encoded = encoded + quality_embedding[:, :, None, :]

        # Pool each teacher independently before combining teachers. This
        # keeps teacher membership observable while remaining invariant to a
        # permutation of neurons within a teacher.
        teacher_condition = condition[:, None, :].expand(batch, teacher_count, self.width)
        if teacher_quality is not None:
            teacher_condition = teacher_condition + quality_embedding
        neuron_keys_and_values = encoded.reshape(batch * teacher_count, hidden, self.width)
        neuron_queries = (
            self.teacher_pool_queries[None, None, :, :]
            + teacher_condition[:, :, None, :]
        ).reshape(batch * teacher_count, self.teacher_pool_slots, self.width)
        teacher_summary, _ = self.neuron_pool(
            neuron_queries,
            neuron_keys_and_values,
            neuron_keys_and_values,
            need_weights=False,
        )
        teacher_tokens = self.teacher_norm(neuron_queries + teacher_summary).reshape(
            batch, teacher_count * self.teacher_pool_slots, self.width
        )

        queries = self.slot_queries[None, :, :] + condition[:, None, :]
        pooled, _ = self.bank_pool(queries, teacher_tokens, teacher_tokens, need_weights=False)
        slots = self.slot_norm(queries + pooled)
        embedding = slots.mean(dim=1)

        feature = self.feature_decoder(self.feature_embedding)
        slot = self.slot_decoder(slots)
        logits = torch.einsum("fw,bhw->bfh", feature, slot) / (self.width ** 0.5)
        logits = logits + self.feature_bias[None, :, None] + self.slot_bias[None, None, :]
        return embedding, logits

    def forward(
        self,
        tokens: Tensor,
        task: Tensor,
        teacher_quality: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.encode(tokens, task, teacher_quality)
