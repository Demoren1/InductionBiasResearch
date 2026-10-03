"""Auditable adapters that turn the two research pilots into evaluator data.

The module deliberately keeps the bank-building data apart from ``TaskData``
support/query rows.  In particular, a saved teacher's validation metric is
never presented as a target-mask quality label: :func:`measure_mask` always
fits a fresh child and reports its terminal query loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from .data import InnerProtocol, TaskData, support_context
from .progress import progress


@dataclass
class FunctionalBank:
    """Raw functional teacher profiles and fixed-cardinality candidate masks."""

    tokens: Tensor                         # [1, R, H, D]
    quality: Tensor | None                  # [1, R, 1], if explicitly allowed
    masks: Tensor                           # [R, F, H]
    baseline_mask: Tensor                   # [F, H]
    provenance: dict[str, Any] = field(default_factory=dict)
    states: list[Any] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tokens = torch.as_tensor(self.tokens, dtype=torch.float32).cpu().contiguous()
        self.masks = torch.as_tensor(self.masks, dtype=torch.float32).cpu().contiguous()
        self.baseline_mask = torch.as_tensor(self.baseline_mask, dtype=torch.float32).cpu().contiguous()
        if self.tokens.ndim != 4 or self.tokens.shape[0] != 1 or min(self.tokens.shape[1:]) < 1:
            raise ValueError("tokens must have shape [1, teachers, hidden, token_dim]")
        if self.masks.ndim != 3 or self.masks.shape[0] != self.tokens.shape[1]:
            raise ValueError("masks must have one [features, hidden] row per teacher")
        if self.masks.shape[2] != self.tokens.shape[2] or self.baseline_mask.shape != self.masks.shape[1:]:
            raise ValueError("bank masks must agree with teacher and baseline dimensions")
        if not torch.isfinite(self.tokens).all() or not torch.isfinite(self.masks).all():
            raise ValueError("functional-bank tensors must be finite")
        if not ((self.masks == 0) | (self.masks == 1)).all() or not ((self.baseline_mask == 0) | (self.baseline_mask == 1)).all():
            raise ValueError("bank masks must be binary")
        if self.quality is not None:
            self.quality = torch.as_tensor(self.quality, dtype=torch.float32).cpu().contiguous()
            if self.quality.shape != (1, self.tokens.shape[1], 1) or not torch.isfinite(self.quality).all():
                raise ValueError("quality must have shape [1, teachers, 1]")


def _exact_topk(scores: Tensor, k: int) -> Tensor:
    scores = torch.as_tensor(scores, dtype=torch.float32).cpu()
    if scores.ndim != 2 or not 1 <= k <= scores.numel():
        raise ValueError("K must be within the number of mask edges")
    # Stable tie breaking makes a bank reproducible across runs.
    values = scores.flatten() + torch.arange(scores.numel(), dtype=torch.float32) * 1e-12
    result = torch.zeros_like(values)
    result[values.topk(k).indices] = 1.0
    return result.reshape_as(scores)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dense_deepsets_token(state: dict[str, Tensor], row: int, probe_x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Extract the same normalized raw functional channels as the context builder."""
    required = ("weight", "masks", "bias", "readout", "per_image_offset")
    if any(name not in state or not torch.is_tensor(state[name]) for name in required):
        raise ValueError("dense source state lacks a complete DeepSets child state")
    selected = {name: state[name][row].detach().float().cpu().clone() for name in required}
    mask = selected["masks"]
    if mask.ndim != 2 or mask.shape[0] != 784 or not bool((mask == 1).all()):
        raise ValueError("a dense teacher must have actual mask density 1.0")
    effective = selected["weight"] * mask
    activation = torch.tanh(probe_x @ effective + selected["bias"])
    psi = activation * selected["readout"]
    gain = (1.0 - activation.square()) * selected["readout"]
    probes = len(probe_x)
    signed = effective * torch.einsum("pf,ph->fh", probe_x, gain) / probes
    absolute = effective.abs() * torch.einsum("pf,ph->fh", probe_x.abs(), gain.abs()) / probes
    rms = effective.abs() * torch.einsum("pf,ph->fh", probe_x.square(), gain.square()).div(probes).sqrt()
    psi_norm = psi / psi.square().mean().sqrt().clamp_min(1e-8)
    q_scale = rms.amax().clamp_min(1e-8)
    token = torch.cat((psi_norm.T, (signed / q_scale).T, (absolute / q_scale).T,
                       (rms / q_scale).T, mask.T), dim=1)
    selected.update({"psi": psi, "q_signed_mean": signed, "q_abs_mean": absolute, "q_rms": rms})
    return token, mask, selected


def load_deepsets_bank(context_path: str | Path, k: int = 7526, teacher_count: int = 32,
                       seed: int = 4100, *, include_dense: bool = True) -> FunctionalBank:
    """Load train-only raw DeepSets teacher profiles from a saved context.

    Source-query quality is intentionally disabled.  ``mean_score`` is the
    saved train-only functional baseline, rather than a newly recomputed or
    target-task statistic.
    """
    from deepsets_vaae.permutation_bank_encoder import TrainOnlyFunctionalTeacherBank

    context_path = Path(context_path).resolve()
    hash_cache: dict[Path, str] = {}

    def artifact_hash(path: Path) -> str:
        path = path.resolve()
        if path not in hash_cache:
            hash_cache[path] = _file_sha256(path)
        return hash_cache[path]
    registry = TrainOnlyFunctionalTeacherBank(context_path, include_source_query_quality=False,
                                               include_training_masks=True)
    refs = registry.train_references
    if teacher_count < 1 or teacher_count > len(refs):
        raise ValueError("teacher_count must fit saved train-only references")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if include_dense and teacher_count < 1:
        raise ValueError("teacher_count must reserve one row for a dense teacher")
    sparse_count = teacher_count - int(include_dense)
    # Keep the saved sparse-density mixture visible to the generator.  This is a
    # selection of train rows only; no target observations or teacher query
    # metric participates in it.
    density_groups: dict[int, list[int]] = {}
    for index, (task, row) in enumerate(refs.tolist()):
        source = registry._source_banks[task]
        state = source.get("state_dict", source)
        count = int(torch.as_tensor(state["masks"])[row].sum())
        density_groups.setdefault(count, []).append(index)
    groups = sorted(density_groups)
    selected_indices: list[int] = []
    for group in groups[:sparse_count]:
        candidates = density_groups[group]
        selected_indices.append(candidates[int(torch.randint(len(candidates), (), generator=generator))])
    remaining = [index for index in range(len(refs)) if index not in selected_indices]
    if len(selected_indices) < sparse_count:
        fill = torch.randperm(len(remaining), generator=generator)[:sparse_count - len(selected_indices)].tolist()
        selected_indices.extend(remaining[index] for index in fill)
    chosen = refs[torch.tensor(selected_indices, dtype=torch.long)]
    batch = registry.gather(chosen[None]) if sparse_count else None
    # The last token channel is the raw training mask, represented [H,F].
    mask_slice = registry.channel_slices["training_mask"]
    masks = (batch.tokens[0, :, :, mask_slice].permute(0, 2, 1).round().float()
             if batch is not None else torch.empty(0, registry.features, registry.hidden))
    payload = torch.load(context_path, map_location="cpu", weights_only=False)
    mean_score = payload.get("mean_score")
    if not torch.is_tensor(mean_score):
        raise ValueError("functional context lacks saved train-only mean_score")
    tokens = batch.tokens if batch is not None else torch.empty(1, 0, registry.hidden, registry.token_dim)
    teacher_sources: list[dict[str, Any]] = []
    sparse_states: list[dict[str, Any]] = []
    for task, row in chosen.tolist():
        # The original bank reference is stable even if a loader did not retain
        # a private source-path field.
        reference_path = Path(payload["bank_references"][task]["path"]).resolve()
        source_state = registry._source_banks[task].get("state_dict", registry._source_banks[task])
        total_rows = int(source_state["masks"].shape[0])
        row_state = {name: value[row].detach().cpu().clone()
                     for name, value in source_state.items()
                     if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == total_rows}
        source_hash = artifact_hash(reference_path)
        teacher_sources.append({"kind": "sparse", "reference": [task, row], "path": str(reference_path),
                                "sha256": source_hash, "active_edges": int(source_state["masks"][row].sum())})
        sparse_states.append({"kind": "sparse", "reference": [task, row], "path": str(reference_path),
                              "sha256": source_hash, "state_dict": row_state,
                              "optimizer_state_ref": str(reference_path)})
    dense_states: list[dict[str, Any]] = []
    if include_dense:
        dense_paths = sorted(context_path.parent.glob("dense_*.pt"))
        if not dense_paths:
            raise FileNotFoundError("production DeepSets context has no dense_<task>.pt source state")
        candidates: list[tuple[Path, int, dict]] = []
        for path in dense_paths:
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            state = artifact.get("source_state_dict", artifact.get("state_dict"))
            if not isinstance(state, dict) or "masks" not in state:
                continue
            for row in range(len(state["masks"])):
                if bool((state["masks"][row] == 1).all()):
                    candidates.append((path.resolve(), row, artifact))
        if not candidates:
            raise ValueError("production DeepSets context has no teacher with actual mask density 1.0")
        path, row, artifact = candidates[int(torch.randint(len(candidates), (), generator=generator))]
        state = artifact.get("source_state_dict", artifact.get("state_dict"))
        dense_token, dense_mask, dense_state = _dense_deepsets_token(state, row, payload["probe_x"].detach().float().cpu())
        tokens = torch.cat((tokens, dense_token[None, None]), dim=1)
        masks = torch.cat((masks, dense_mask[None]), dim=0)
        dense_states.append({"kind": "dense", "path": str(path), "sha256": artifact_hash(path), "row": row,
                             "state_dict": dense_state, "optimizer_state": artifact.get("optimizer_state"),
                             "history": artifact.get("history")})
        teacher_sources.append({"kind": "dense", "path": str(path), "sha256": artifact_hash(path), "row": row,
                                "active_edges": int(dense_mask.sum()), "density": 1.0})
    if masks.shape[0] != teacher_count:
        raise AssertionError("selected teacher count differs from requested total")
    baseline = _exact_topk(mean_score, k)
    density_counts = {int(v): int((masks.sum((1, 2)) == v).sum())
                      for v in torch.unique(masks.sum((1, 2)).long()).tolist()}
    provenance = {
        "family": "deepsets", "context_path": str(context_path), "seed": int(seed),
        "train_references": chosen.tolist(), "teacher_sources": teacher_sources,
        "context_sha256": artifact_hash(context_path), "probe_ids": payload.get("probe_ids", torch.empty(0, dtype=torch.long)).tolist(),
        "quality_source": None, "baseline_source": "saved_train_only_mean_score",
        "baseline_k": int(k), "teacher_density_counts": density_counts,
        "source_task_count": int(registry._task_count),
    }
    return FunctionalBank(tokens, None, masks, baseline, provenance, states=sparse_states + dense_states)


def _pattern_table() -> tuple[Tensor, Tensor]:
    ids = torch.arange(1 << 11, dtype=torch.long)
    shifts = torch.arange(10, -1, -1, dtype=torch.long)
    return ids, (((ids[:, None] >> shifts) & 1).float() * 2 - 1)


def _pattern_labels(x: Tensor, pattern: str) -> Tensor:
    bits = torch.tensor([int(v) for v in pattern], dtype=x.dtype)
    return ((x.add(1).div(2).unfold(1, len(pattern), 1) == bits).all(-1).any(-1)).float()


def _pattern_partitions(seed: int) -> dict[str, Tensor]:
    """Four frozen observation partitions, independent of task labels."""
    ids, _ = _pattern_table()
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(int(seed) + 97))
    # The first partition belongs only to source-teacher fitting/probing.
    return {"bank": order[:512], "support": order[512:1280],
            "query": order[1280:1792], "test": order[1792:]}


def _align_pattern_columns(reference: Tensor, values: Tensor) -> Tensor:
    """Return label-free Hungarian alignment of ``values`` to ``reference``.

    Both inputs are [features, hidden] non-negative functional summaries.  A
    small dynamic-programming assignment keeps this fixture dependency-free
    and exact for its H=8 architecture.
    """
    if reference.shape != values.shape or reference.ndim != 2:
        raise ValueError("functional alignment needs matching [features, hidden] maps")
    hidden = values.shape[1]
    if hidden > 16:
        raise ValueError("pattern fixture alignment supports at most 16 hidden units")
    ref = reference / reference.square().sum(0).sqrt().clamp_min(1e-12)
    candidate = values / values.square().sum(0).sqrt().clamp_min(1e-12)
    score = ref.T @ candidate
    # dp[mask] is the best score after assigning the first popcount(mask)
    # reference columns to candidate columns contained in mask.
    dp = {0: (0.0, ())}
    for _ in range(hidden):
        next_dp: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (total, assignment) in dp.items():
            target = len(assignment)
            for source in range(hidden):
                bit = 1 << source
                if used & bit:
                    continue
                proposal = (total + float(score[target, source]), assignment + (source,))
                key = used | bit
                if key not in next_dp or proposal[0] > next_dp[key][0]:
                    next_dp[key] = proposal
        dp = next_dp
    order = torch.tensor(dp[(1 << hidden) - 1][1], dtype=torch.long)
    return values[:, order]


def _balanced_indices(y: Tensor, candidates: Tensor, count: int, seed: int) -> Tensor:
    if count < 2:
        raise ValueError("support/query counts must be at least two")
    gen = torch.Generator().manual_seed(int(seed))
    pos, neg = candidates[y[candidates] > .5], candidates[y[candidates] <= .5]
    want_pos = count // 2
    want_neg = count - want_pos
    if len(pos) < want_pos or len(neg) < want_neg:
        raise ValueError("partition lacks enough examples of both classes")
    result = torch.cat((pos[torch.randperm(len(pos), generator=gen)[:want_pos]],
                        neg[torch.randperm(len(neg), generator=gen)[:want_neg]]))
    return result[torch.randperm(len(result), generator=gen)]


def _uniform_indices(candidates: Tensor, count: int, seed: int) -> Tensor:
    if count < 1:
        raise ValueError("query count must be positive")
    count = min(count, len(candidates))
    return candidates[torch.randperm(len(candidates), generator=torch.Generator().manual_seed(int(seed)))[:count]]


def _make_pattern_task(pattern: str, split: str, partitions: dict[str, Tensor], support_count: int,
                       query_count: int, seed: int, *, query_partition: str = "query",
                       query_balanced: bool = True) -> TaskData:
    ids, x = _pattern_table()
    y = _pattern_labels(x, pattern)
    support_idx = _balanced_indices(y, partitions["support"], support_count, seed + 1)
    query_idx = (_balanced_indices(y, partitions[query_partition], query_count, seed + 2)
                 if query_balanced else _uniform_indices(partitions[query_partition], query_count, seed + 2))
    return TaskData(task_id=f"pattern:{pattern}", split=split, x_support=x[support_idx], y_support=y[support_idx],
                    x_query=x[query_idx], y_query=y[query_idx], context=support_context(x[support_idx], y[support_idx]),
                    support_ids=ids[support_idx], query_ids=ids[query_idx],
                    provenance={"family": "pattern", "pattern": pattern,
                                "support_partition": "evaluator_support", "query_partition": query_partition,
                                "query_sampling": "balanced" if query_balanced else "uniform_heldout"})


def _random_pattern_masks(count: int, k: int, seed: int) -> Tensor:
    if not 1 <= k <= 88:
        raise ValueError("pattern K must be in [1, 88]")
    gen = torch.Generator().manual_seed(int(seed))
    scores = torch.rand(count, 88, generator=gen)
    result = torch.zeros_like(scores)
    result.scatter_(1, scores.topk(k, dim=1).indices, 1)
    return result.reshape(count, 11, 8)


def _pattern_teacher_masks(count: int, k: int, seed: int) -> Tensor:
    """Small density-diverse source bank with a true dense anchor when possible."""
    if count == 1:
        return _random_pattern_masks(1, k, seed)
    rows = [_random_pattern_masks(1, k, seed)]
    alternatives = (max(1, round(.1 * 88)), k, round(.5 * 88), round(.7 * 88))
    for row in range(1, count - 1):
        rows.append(_random_pattern_masks(1, alternatives[(row - 1) % len(alternatives)], seed + row))
    rows.append(torch.ones(1, 11, 8))
    return torch.cat(rows)


def _fit_pattern(mask: Tensor, task: TaskData, protocol: InnerProtocol, device: str) -> dict[str, Any]:
    """Fixed-horizon ReLU children; gradients use support labels only.

    The scalar adapter intentionally delegates to the packed engine so bank
    building, cache misses, and explicit batch acquisition all share the same
    solver and serialization path.
    """
    from .pattern_fit import PatternFitEngine
    return PatternFitEngine(protocol, device).fit_one(mask, task)


def measure_mask(mask: Tensor, task: TaskData, protocol: InnerProtocol, device: str = "cpu", *,
                 initialization_seed: int | None = None) -> dict[str, Any]:
    """Measure a mask with a fresh, auditable initialization seed.

    ``protocol_id`` remains the caller's frozen solver specification.  The
    optional nonce-derived seed is recorded separately and controls the child
    initialization (and, when enabled, its deterministic support minibatches).
    """
    return _measure_mask(mask, task, protocol, device, initialization_seed)


def _measure_mask(mask: Tensor, task: TaskData, protocol: InnerProtocol, device: str,
                  initialization_seed: int | None = None) -> dict[str, Any]:
    original_protocol = protocol
    actual_seed = protocol.seed if initialization_seed is None else int(initialization_seed)
    fitting_protocol = replace(protocol, seed=actual_seed)
    if task.x_support.ndim == 3:
        result = _measure_deepsets_mask(mask, task, fitting_protocol, device)
    else:
        result = _fit_pattern(mask, task, fitting_protocol, device)
    result.update(label_source="fresh_terminal_query", fixed_horizon=True,
                  protocol_id=original_protocol.fingerprint, task_id=task.task_id,
                  actual_initialization_seed=actual_seed,
                  solver_protocol_seed=original_protocol.seed,
                  minibatch_seed_base=actual_seed if fitting_protocol.batch_size is not None else None)
    return result


def _measure_deepsets_mask(mask: Tensor, task: TaskData, protocol: InnerProtocol, device: str) -> dict[str, Any]:
    """Use the established batched DeepSets child utility without re-scaling it."""
    from deepsets_vaae.utility_graph_child import fit_children
    mask = torch.as_tensor(mask, dtype=torch.float32)
    if mask.ndim != 2 or mask.shape[0] != 784:
        raise ValueError("DeepSets masks must have shape [784, hidden]")
    if protocol.metric != "nmse":
        raise ValueError("the DeepSets adapter requires protocol.metric='nmse'")
    replicas = list(range(protocol.replicas))
    fitted = fit_children(mask[None].expand(protocol.replicas, -1, -1).clone(),
                          task.x_support[None], task.y_support[None], task.x_query[None], task.y_query[None],
                          [protocol.seed], replicas, protocol.steps, protocol.lr, protocol.l2, device,
                          reference_models=max(20, protocol.replicas), checkpoint_every=protocol.checkpoint_every,
                          lr_decay_every=protocol.lr_decay_every, lr_floor=protocol.lr_floor,
                          batch_size=protocol.batch_size,
                          plateau_tolerance=protocol.plateau_tolerance)
    state = fitted["state_dict"]
    weights = (state["weight"] * mask.cpu()[None, None]).squeeze(0)
    return {"label_source": "fresh_terminal_query", "fixed_horizon": True,
            "protocol_id": protocol.fingerprint, "task_id": task.task_id,
            "replica_losses": fitted["query_loss"].reshape(-1).tolist(),
            "seeds": [f"{protocol.seed}:{replica}" for replica in replicas],
            "initialization": {"base_seed": protocol.seed, "reference_replica_ids": replicas},
            "plateau_flags": fitted["plateau_flags"].reshape(-1).tolist(),
            "state_dict": state, "optimizer_state": fitted["optimizer_state"],
            "history": fitted["history"], "effective_weights": weights}


def build_pattern_fixture(seed: int = 4100, bank_steps: int = 40, teacher_count: int = 8,
                          support_count: int = 32, query_count: int = 64, k: int = 32,
                          device: str = "cpu") -> tuple[FunctionalBank, list[TaskData], dict[str, Any]]:
    """Build a tiny source-only pattern bank plus evaluator train/validation tasks."""
    from meta_pattern.data import build_task_splits
    if min(bank_steps, teacher_count, support_count, query_count) < 1:
        raise ValueError("fixture sizes must be positive")
    splits = build_task_splits([4], seed=seed)
    parts = _pattern_partitions(seed)
    source_patterns = [task.pattern for task in splits["train"]]
    if not source_patterns:
        raise ValueError("no source patterns")
    masks = _pattern_teacher_masks(teacher_count, k, seed + 7)
    # Teacher support, teacher diagnostic query, and the common functional
    # probe are three disjoint portions of the bank-only observation partition.
    ids, x = _pattern_table()
    bank_train_idx, bank_query_idx, probe_idx = parts["bank"][:256], parts["bank"][256:384], parts["bank"][384:]
    token_rows: list[Tensor] = []
    raw_profiles: list[dict[str, Tensor]] = []
    bank_protocol = InnerProtocol(steps=bank_steps, replicas=1, lr=.03, l2=.001,
                                  checkpoint_every=max(1, bank_steps // 4), seed=seed)
    for row in progress(range(teacher_count), desc="Functional bank teachers", unit="teacher"):
        pattern = source_patterns[row % len(source_patterns)]
        y = _pattern_labels(x, pattern)
        train_idx = _balanced_indices(y, bank_train_idx, min(96, len(bank_train_idx)), seed + row)
        query_idx = _balanced_indices(y, bank_query_idx, min(48, len(bank_query_idx)), seed + 10_000 + row)
        bank_task = TaskData(f"bank:{row}", "train", x[train_idx], y[train_idx], x[query_idx], y[query_idx],
                             support_context(x[train_idx], y[train_idx]), ids[train_idx], ids[query_idx])
        fitted = _fit_pattern(masks[row], bank_task, bank_protocol, device)
        state = fitted["state_dict"][0]
        with torch.no_grad():
            pre = x[probe_idx] @ (state["w"] * masks[row]) + state["b"]
            psi = F.relu(pre) * state["a"]
            effective = state["w"] * masks[row]
            q = (x[probe_idx, :, None] * effective[None] * state["a"][None, None]
                 * (pre > 0)[:, None]).cpu()
            signed, q_abs, q_rms = q.mean(0), q.abs().mean(0), q.square().mean(0).sqrt()
            token_rows.append(torch.cat((psi.T, signed.T, q_abs.T, q_rms.T, masks[row].T), dim=1))
            raw_profiles.append({"psi": psi.cpu(), "q_signed": signed.cpu(), "q_abs": q_abs.cpu(), "q_rms": q_rms.cpu(),
                                 "state_dict": state, "optimizer_state": fitted["optimizer_state"][0], "history": fitted["history"][0]})
    reference_q_abs = raw_profiles[0]["q_abs"]
    aligned_q_abs = torch.stack([reference_q_abs] + [_align_pattern_columns(reference_q_abs, item["q_abs"])
                                                       for item in raw_profiles[1:]])
    baseline = _exact_topk(aligned_q_abs.mean(0), k)
    bank = FunctionalBank(torch.stack(token_rows)[None], None, masks, baseline,
                          {"family": "pattern", "seed": seed, "source_patterns": source_patterns,
                           "reserved_bank_ids": ids[parts["bank"]].tolist(), "teacher_support_ids": ids[bank_train_idx].tolist(),
                           "teacher_query_ids": ids[bank_query_idx].tolist(), "probe_ids": ids[probe_idx].tolist(),
                           "evaluator_support_ids": ids[parts["support"]].tolist(), "evaluator_query_ids": ids[parts["query"]].tolist(),
                           "baseline_alignment": "exact label-free Hungarian assignment to teacher 0 by cosine q_abs",
                           "baseline_k": k, "quality_source": None,
                           "teacher_density_counts": {int(edges): int((masks.sum((1, 2)) == edges).sum())
                                                      for edges in torch.unique(masks.sum((1, 2)).long()).tolist()}},
                          states=raw_profiles,
                          diagnostics={"aligned_q_abs": aligned_q_abs})
    tasks: list[TaskData] = []
    for split_name, task_split in (("train", "train"), ("val", "validation")):
        for number, item in enumerate(splits[split_name]):
            tasks.append(_make_pattern_task(item.pattern, task_split, parts, support_count, query_count,
                                            seed + 1_000 * (number + 1) + (0 if split_name == "train" else 500)))
    test_spec = {"family": "pattern", "seed": seed, "patterns": [item.pattern for item in splits["test"]],
                 "partitions": {name: ids[value].tolist() for name, value in parts.items()},
                 "support_count": support_count, "query_count": query_count, "materialized": False}
    return bank, tasks, test_spec


def make_pattern_test_tasks(test_spec: dict[str, Any]) -> list[TaskData]:
    """Materialize held-out pattern task labels only after model selection."""
    if test_spec.get("family") != "pattern":
        raise ValueError("not a pattern test specification")
    seed = int(test_spec["seed"])
    parts = {key: torch.tensor(value, dtype=torch.long) for key, value in test_spec["partitions"].items()}
    # Test tasks use held-out task membership but the evaluator's frozen
    # support/query observation partitions; IDs remain disjoint.
    return [_make_pattern_task(pattern, "test", parts, int(test_spec["support_count"]),
                               int(test_spec["query_count"]), seed + 90_000 + index,
                               query_partition="test", query_balanced=False)
            for index, pattern in enumerate(test_spec["patterns"])]


def _cost_vectors(seed: int, count: int) -> Tensor:
    gen = torch.Generator().manual_seed(int(seed))
    values = torch.randn(count, 10, generator=gen)
    return (values - values.mean(1, keepdim=True)) / values.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)


def _deepsets_sets(split: Any, costs: Tensor, count: int, generator: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
    """The core sampler plus the actual source image IDs used in each set."""
    indices = torch.randint(len(split.features), (count, 5), generator=generator)
    x = split.features[indices]
    y = costs[split.digits[indices]].sum(dim=1)
    return x, y, split.source_ids[indices].detach().cpu()


def make_deepsets_tasks(data_root: str | Path, seed: int = 4100, support_count: int = 205,
                        query_count: int = 51) -> tuple[list[TaskData], dict[str, Any]]:
    """Create source/target train-validation tasks; target-test stays delayed."""
    from deepsets_vaae.core import load_data
    data = load_data(data_root, seed, "cpu")
    if min(support_count, query_count) < 1:
        raise ValueError("support_count and query_count must be positive")
    # Cost vectors are task identities.  Source replay, meta-validation and
    # final test therefore receive disjoint vectors rather than merely new
    # sampled sets for the same target function.
    costs = _cost_vectors(seed, 6)
    output: list[TaskData] = []
    # The teacher bank uses source pools.  Every real replay label below uses
    # target pools, including the known source-cost replay tasks, so a teacher
    # fit or its source diagnostics cannot leak into evaluator labels.
    specs = (("target_train", "target_validation", "train", 0),
             ("target_train", "target_validation", "validation", 1))
    for train_key, query_key, split, offset in specs:
        split_costs = costs[:2] if split == "train" else costs[2:4]
        for task_index, cost in enumerate(split_costs):
            gen = torch.Generator().manual_seed(seed + offset * 10_000 + task_index)
            xs, ys, support_ids = _deepsets_sets(data[train_key], cost, support_count, gen)
            xq, yq, query_ids = _deepsets_sets(data[query_key], cost, query_count, gen)
            output.append(TaskData(f"deepsets:{split}:{task_index}", split, xs, ys, xq, yq,
                                   support_context(xs.mean(1), ys), support_ids, query_ids,
                                   {"family": "deepsets", "replay_role": "known_source_cost" if split == "train" else "novel_validation_cost",
                                    "support_pool": train_key, "query_pool": query_key,
                                    "costs": cost.tolist(), "set_size": 5}))
    spec = {"family": "deepsets", "data_root": str(Path(data_root).resolve()), "seed": seed,
            "support_count": support_count, "query_count": query_count, "costs": costs[4:].tolist(),
            "test_pool": "target_test", "materialized": False}
    return output, spec


def materialize_deepsets_test_tasks(test_spec: dict[str, Any]) -> list[TaskData]:
    """Build target-test tasks only at the final evaluation boundary."""
    from deepsets_vaae.core import load_data
    if test_spec.get("family") != "deepsets":
        raise ValueError("not a DeepSets test specification")
    data = load_data(test_spec["data_root"], int(test_spec["seed"]), "cpu")
    result: list[TaskData] = []
    for index, values in enumerate(test_spec["costs"]):
        cost = torch.tensor(values)
        gen = torch.Generator().manual_seed(int(test_spec["seed"]) + 80_000 + index)
        xs, ys, support_ids = _deepsets_sets(data["target_train"], cost, int(test_spec["support_count"]), gen)
        xq, yq, query_ids = _deepsets_sets(data["target_test"], cost, int(test_spec["query_count"]), gen)
        result.append(TaskData(f"deepsets:test:{index}", "test", xs, ys, xq, yq,
                               support_context(xs.mean(1), ys),
                               support_ids, query_ids,
                               {"family": "deepsets", "support_pool": "target_train", "query_pool": "target_test",
                                "costs": cost.tolist(), "set_size": 5}))
    return result
