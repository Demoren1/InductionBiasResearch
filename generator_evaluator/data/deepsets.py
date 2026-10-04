"""Own-task DeepSets functional banks and sealed cooperative task data.

Every generator receives a bank fitted to its own digit-cost function. Source
images used to fit and rank those maps are partitioned away from evaluator
observations, and held-out task data are materialized only by the final test
factory.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from generator_evaluator.data.adapters import (FunctionalBank, _cost_vectors, _deepsets_sets,
                       _exact_topk, make_deepsets_tasks)
from generator_evaluator.data.types import InnerProtocol, RealReplay, TaskData, support_context, tensor_hash
from generator_evaluator.storage.functional import write_functional_card
from generator_evaluator.evaluation.parallel import ParallelMeasurementStore
from generator_evaluator.storage.progress import progress


_FEATURES = 784
_SET_SIZE = 5
_DENSE_FRACTION_BUCKETS = (.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.)


def _candidate_initialization_seed(seed: int, candidate_id: int) -> int:
    return int(seed) + 1_000_003 * (int(candidate_id) + 1)


def _require_count(name: str, value: int, maximum: int, *, minimum: int = 1) -> None:
    if not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}], got {value!r}")


def _density_buckets(k: int, teachers: int, edges: int) -> tuple[int, ...]:
    """Use ten mixed-density strata, retaining the smoke fixture's five anchors."""
    if teachers >= 10:
        buckets = [round(edges * fraction) for fraction in _DENSE_FRACTION_BUCKETS]
    else:
        buckets = [round(edges * .1), k, round(edges * .5), round(edges * .7), edges]
    if k not in buckets:
        nearest = min(range(len(buckets)), key=lambda index: (abs(buckets[index] - k), index))
        buckets[nearest] = k
    return tuple(sorted(set(int(value) for value in buckets)))


def _stratum_counts(total: int, count: int) -> list[int]:
    quotient, remainder = divmod(total, count)
    return [quotient + int(index < remainder) for index in range(count)]


def _candidate_masks(candidate_count: int, k: int, seed: int, features: int,
                     hidden: int, buckets: tuple[int, ...]) -> tuple[Tensor, list[int], list[int]]:
    edges = features * hidden
    counts = _stratum_counts(candidate_count, len(buckets))
    masks: list[Tensor] = []
    strata: list[int] = []
    identifiers: list[int] = []
    for density, amount in zip(buckets, counts):
        for _ in range(amount):
            candidate_id = len(identifiers)
            generator = torch.Generator(device="cpu").manual_seed(int(seed) + candidate_id)
            mask = torch.zeros(edges, dtype=torch.float32)
            mask[torch.randperm(edges, generator=generator)[:density]] = 1.
            masks.append(mask.reshape(features, hidden))
            strata.append(density)
            identifiers.append(candidate_id)
    return torch.stack(masks), identifiers, strata


def _split_rows(split: Any, rows: Tensor) -> Any:
    rows = torch.as_tensor(rows, dtype=torch.long, device=split.features.device)
    return type(split)(split.features.index_select(0, rows), split.digits.index_select(0, rows),
                       split.source_ids.index_select(0, rows))


def _partition_split(split: Any, sizes: Sequence[int], seed: int) -> list[Any]:
    if any(int(size) < 1 for size in sizes) or sum(map(int, sizes)) > len(split.features):
        raise ValueError("requested disjoint image pools do not fit the supplied DeepSets split")
    order = torch.randperm(len(split.features), generator=torch.Generator().manual_seed(int(seed)))
    result, offset = [], 0
    for size in sizes:
        size = int(size)
        result.append(_split_rows(split, order[offset:offset + size]))
        offset += size
    return result


def _split_by_source_ids(split: Any, source_ids: Sequence[int]) -> Any:
    requested = [int(value) for value in source_ids]
    if len(requested) != len(set(requested)):
        raise ValueError("reserved image pool contains duplicate source IDs")
    positions = {int(value): index for index, value in enumerate(split.source_ids.detach().cpu().tolist())}
    missing = set(requested) - set(positions)
    if missing:
        raise ValueError("reserved image pool contains IDs outside the named data split")
    rows = torch.tensor([positions[value] for value in requested], dtype=torch.long,
                        device=split.features.device)
    return _split_rows(split, rows)


def _fixed_test_pools(data, spec, *, data_root, seed, train_task_count,
                      test_task_count, support_count, query_count):
    """Reserve prior sealed pools before allocating any additional train roles."""
    expected = {"family": "cooperative_deepsets", "domain": "deepsets", "seed": seed,
                "test_task_count": test_task_count, "support_count": support_count,
                "query_count": query_count, "materialized": False}
    if any(spec.get(key) != value for key, value in expected.items()) or (
            Path(spec.get("data_root", "")).resolve() != Path(data_root).resolve()):
        raise ValueError("fixed test specification must match domain, data root, seed and test budgets")
    expected_costs = torch.stack([_fixture_cost_vector(seed, "heldout", index)
                                 for index in range(test_task_count)])
    if not torch.equal(torch.as_tensor(spec.get("costs", []), dtype=torch.float32), expected_costs):
        raise ValueError("fixed test specification has different held-out cost vectors")
    support_pools, query_pools = spec.get("test_support_pools", []), spec.get("test_query_pools", [])
    if len(support_pools) != test_task_count or len(query_pools) != test_task_count:
        raise ValueError("fixed test specification has the wrong number of image pools")
    all_pools = support_pools + query_pools
    if (any(len(pool) != size for pools, size in
            ((support_pools, support_count * _SET_SIZE), (query_pools, query_count * _SET_SIZE))
            for pool in pools) or
            len(set(value for pool in all_pools for value in pool)) != sum(map(len, all_pools))):
        raise ValueError("fixed test image pools must have the requested sizes and be disjoint")
    heldout_support = [_split_by_source_ids(data["target_train"], pool) for pool in support_pools]
    heldout_query = [_split_by_source_ids(data["target_test"], pool) for pool in query_pools]
    split = data["target_train"]
    reserved = set(value for pool in support_pools for value in pool)
    order = torch.randperm(len(split.features), generator=torch.Generator().manual_seed(seed + 31_117))
    ids = split.source_ids.detach().cpu().tolist()
    available = torch.tensor([row for row in order.tolist() if ids[row] not in reserved], dtype=torch.long)
    rows = support_count * _SET_SIZE
    if len(available) < train_task_count * rows:
        raise ValueError("train image pools do not fit after reserving fixed held-out supports")
    train_support = [_split_rows(split, available[index * rows:(index + 1) * rows])
                     for index in range(train_task_count)]
    return train_support + heldout_support, heldout_query


def _task_ids(task: TaskData) -> set[int]:
    return set(torch.as_tensor(task.support_ids).detach().cpu().reshape(-1).tolist()) | set(
        torch.as_tensor(task.query_ids).detach().cpu().reshape(-1).tolist())


def _cost(task: TaskData) -> Tensor:
    value = torch.as_tensor(task.provenance.get("costs"), dtype=torch.float32)
    if value.shape != (10,) or not torch.isfinite(value).all():
        raise ValueError(f"{task.task_id} lacks its ten-value task cost vector")
    return value


def _fixture_cost_vector(seed: int, role: str, index: int) -> Tensor:
    """Keep the legacy four costs while deriving stable vectors for extra roles."""
    if role == "train" and index < 2:
        return _cost_vectors(seed, 6)[index].clone()
    if role == "heldout" and index < 2:
        return _cost_vectors(seed, 6)[4 + index].clone()
    if role == "train":
        derived_seed = int(seed) + 1_000_003 * (int(index) + 1)
    elif role == "heldout":
        derived_seed = int(seed) + 2_000_003 * (int(index) + 1)
    else:
        raise ValueError(f"unknown DeepSets task-cost role: {role!r}")
    return _cost_vectors(derived_seed, 1)[0]


def _make_task(task_id: str, split_name: str, support_split: Any, query_split: Any,
               costs: Tensor, support_count: int, query_count: int, seed: int,
               *, role: str, task_index: int, task_count: int | None = None,
               evaluator_task_id: int | None = None) -> TaskData:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    x_support, y_support, support_ids = _deepsets_sets(support_split, costs, support_count, generator)
    x_query, y_query, query_ids = _deepsets_sets(query_split, costs, query_count, generator)
    context = support_context(x_support.mean(1), y_support)
    provenance = {"family": "deepsets", "domain": "deepsets", "task_index": int(task_index),
                  "costs": costs.tolist(), "role": role, "support_pool": "target_train",
                  "query_pool": "target_validation" if split_name != "test" else "target_test",
                  "set_size": _SET_SIZE}
    if task_count is not None:
        encoded_task_id = task_index if evaluator_task_id is None else int(evaluator_task_id)
        if (not isinstance(task_count, int) or task_count < 1 or
                not 0 <= encoded_task_id < task_count):
            raise ValueError("DeepSets evaluator task ID must fit its configured one-hot width")
        identity = F.one_hot(torch.tensor(encoded_task_id), num_classes=task_count).to(context.dtype)
        context = torch.cat((context, identity))
        provenance.update({"evaluator_task_id": encoded_task_id,
                           "task_id_encoding": "one_hot", "task_id_width": int(task_count)})
    return TaskData(task_id, split_name, x_support, y_support, x_query, y_query,
                    context, support_ids, query_ids, provenance)


def _task_costs(seed: int, train_tasks: Sequence[TaskData], heldout_count: int = 2) -> list[Tensor]:
    if len(train_tasks) < 2 or heldout_count < 1:
        raise ValueError("the cooperative DeepSets fixture needs at least two train tasks and one held-out task")
    train = [_cost(task) for task in train_tasks]
    if any(not torch.equal(cost, _fixture_cost_vector(seed, "train", index))
           for index, cost in enumerate(train)):
        raise ValueError("DeepSets train tasks do not match their seeded cost-vector identities")
    heldout = [_fixture_cost_vector(seed, "heldout", index) for index in range(heldout_count)]
    all_costs = [*train, *heldout]
    if any(torch.equal(all_costs[left], all_costs[right])
           for left in range(len(all_costs)) for right in range(left + 1, len(all_costs))):
        raise ValueError("DeepSets task cost vectors must be distinct")
    return all_costs


def _unwrap_model_state(state: dict[str, Tensor], *, row: int = 0, rows: int = 1) -> dict[str, Tensor]:
    """Select one candidate and replica from the packed child-state schema."""
    result: dict[str, Tensor] = {}
    for name, value in state.items():
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().float().cpu()
        if tensor.ndim >= 2 and tensor.shape[0] == 1 and tensor.shape[1] == rows:
            tensor = tensor[0, row]
        elif tensor.ndim >= 1 and tensor.shape[0] == 1 and rows == 1:
            tensor = tensor[0]
        result[name] = tensor.contiguous().clone()
    return result


def extract_deepsets_functional_token(state: dict[str, Tensor], mask: Tensor,
                                      probe_x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Build normalized tanh contributions and feature derivatives on images."""
    required = ("weight", "bias", "readout", "per_image_offset")
    if not isinstance(state, dict) or any(name not in state for name in required):
        raise ValueError("DeepSets child state lacks weight, bias, readout or per_image_offset")
    weight = torch.as_tensor(state["weight"], dtype=torch.float32).detach().cpu()
    bias = torch.as_tensor(state["bias"], dtype=torch.float32).detach().cpu()
    readout = torch.as_tensor(state["readout"], dtype=torch.float32).detach().cpu()
    offset = torch.as_tensor(state["per_image_offset"], dtype=torch.float32).detach().cpu()
    mask = torch.as_tensor(mask, dtype=torch.float32).detach().cpu()
    probe_x = torch.as_tensor(probe_x, dtype=torch.float32).detach().cpu()
    if weight.ndim != 2 or weight.shape[0] != _FEATURES or mask.shape != weight.shape:
        raise ValueError("DeepSets weight and mask must share shape [784, hidden]")
    if bias.shape != (weight.shape[1],) or readout.shape != bias.shape or offset.numel() != 1:
        raise ValueError("DeepSets bias/readout must match hidden width and offset must be scalar")
    if probe_x.ndim != 2 or probe_x.shape[1] != _FEATURES or len(probe_x) < 1:
        raise ValueError("probe_x must have shape [probe_rows, 784]")
    if not all(torch.isfinite(value).all() for value in (weight, bias, readout, offset, mask, probe_x)):
        raise ValueError("functional profile inputs must be finite")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("DeepSets masks must be binary")

    effective = weight * mask
    activation = torch.tanh(probe_x @ effective + bias)
    psi = activation * readout
    gain = (1. - activation.square()) * readout
    count = len(probe_x)
    signed = effective * torch.einsum("pf,ph->fh", probe_x, gain) / count
    absolute = effective.abs() * torch.einsum("pf,ph->fh", probe_x.abs(), gain.abs()) / count
    rms = effective.abs() * torch.einsum("pf,ph->fh", probe_x.square(), gain.square()).div(count).sqrt()
    psi_scale = psi.square().mean().sqrt().clamp_min(1e-8)
    q_scale = rms.amax().clamp_min(1e-8)
    tokens = torch.cat((psi.div(psi_scale).T, signed.div(q_scale).T,
                        absolute.div(q_scale).T, rms.div(q_scale).T, mask.T), dim=1).contiguous()
    raw = {"psi": psi.contiguous(), "q_signed_mean": signed.contiguous(),
           "q_abs_mean": absolute.contiguous(), "q_rms": rms.contiguous(),
           "psi_scale": psi_scale.reshape(1), "q_scale": q_scale.reshape(1),
           "effective_weights": effective.contiguous()}
    return tokens, raw


def _state_hash(state: dict[str, Tensor], mask: Tensor) -> str:
    digest = hashlib.sha256()
    for name in ("weight", "bias", "readout", "per_image_offset"):
        if name not in state:
            raise ValueError("terminal DeepSets state is incomplete")
        digest.update(tensor_hash(torch.as_tensor(state[name], dtype=torch.float32)).encode())
    digest.update(tensor_hash(torch.as_tensor(mask, dtype=torch.float32)).encode())
    return digest.hexdigest()


def _align_hidden_columns(reference: Tensor, values: Tensor) -> Tensor:
    """Align a raw feature-by-hidden map to a reference by cosine similarity."""
    from scipy.optimize import linear_sum_assignment

    if reference.shape != values.shape or reference.ndim != 2:
        raise ValueError("DeepSets functional alignment needs matching [784, hidden] maps")
    ref = reference / reference.square().sum(0).sqrt().clamp_min(1e-12)
    candidate = values / values.square().sum(0).sqrt().clamp_min(1e-12)
    rows, columns = linear_sum_assignment(-(ref.T @ candidate).detach().cpu().numpy())
    order = torch.empty(reference.shape[1], dtype=torch.long)
    order[torch.as_tensor(rows, dtype=torch.long)] = torch.as_tensor(columns, dtype=torch.long)
    return values.index_select(1, order)


def _optimizer_replica(state: dict[str, Any], row: int, rows: int) -> dict[str, Any]:
    selected = {"state": {}, "param_groups": deepcopy(state.get("param_groups", []))}
    for parameter, values in state.get("state", {}).items():
        selected["state"][parameter] = {}
        for name, value in values.items():
            if torch.is_tensor(value):
                tensor = value.detach().cpu()
                if tensor.ndim >= 2 and tensor.shape[0] == 1 and tensor.shape[1] == rows:
                    tensor = tensor[0:1, row:row + 1]
                selected["state"][parameter][name] = tensor.clone()
            else:
                selected["state"][parameter][name] = deepcopy(value)
    return selected


def _history_replica(history: dict[str, Any], row: int, rows: int) -> dict[str, Any]:
    selected = {}
    for name, value in history.items():
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if tensor.ndim >= 3 and tensor.shape[1] == 1 and tensor.shape[2] == rows:
                tensor = tensor[:, :, row:row + 1]
            selected[name] = tensor.clone()
        else:
            selected[name] = deepcopy(value)
    return selected


def _available_measurement_devices(device: str,
                                  measurement_devices: Sequence[str] | None) -> tuple[str, tuple[str, ...]]:
    requested = tuple(str(value) for value in measurement_devices) if measurement_devices else (str(device),)
    if not requested:
        requested = (str(device),)
    usable: list[str] = []
    for value in requested:
        try:
            target = torch.device(value)
        except (ValueError, RuntimeError):
            continue
        if target.type == "cuda" and (not torch.cuda.is_available() or
                                      (target.index is not None and target.index >= torch.cuda.device_count())):
            continue
        if target.type in ("cpu", "cuda"):
            usable.append(str(target))
    if not usable:
        usable = ["cpu"]
    # A mixed CPU/GPU list cannot use the store's spawned-GPU path. Preserve
    # the explicit first viable device for its direct batched fallback.
    if any(value.startswith("cuda") for value in usable) and any(value == "cpu" for value in usable):
        usable = [usable[0]]
    return usable[0], tuple(usable)


def _build_bank(task_index: int, task: TaskData, support_split: Any, query_split: Any,
                probe_x: Tensor, probe_ids: Tensor, *, seed: int, bank_steps: int,
                bank_candidates: int, teachers_per_task: int, support_count: int,
                query_count: int, teacher_batch_size: int, k: int, hidden: int,
                device: str, measurement_devices: tuple[str, ...], bank_out: Path,
                persist_artifacts: bool = False) -> FunctionalBank:
    from deepsets_vaae.core import MaskedDeepSets

    features = _FEATURES
    edge_count = features * hidden
    buckets = _density_buckets(k, teachers_per_task, edge_count)
    quotas = _stratum_counts(teachers_per_task, len(buckets))
    candidate_counts = _stratum_counts(bank_candidates, len(buckets))
    if any(available < keep for available, keep in zip(candidate_counts, quotas)):
        raise ValueError("bank_candidates must provide enough candidates in every density stratum")
    masks, candidate_ids, strata = _candidate_masks(bank_candidates, k, seed, features, hidden, buckets)
    cost = _cost(task)
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_007)
    x_support, y_support, support_ids = _deepsets_sets(support_split, cost, support_count, generator)
    x_query, y_query, query_ids = _deepsets_sets(query_split, cost, query_count, generator)
    if (set(support_ids.reshape(-1).tolist()) & set(probe_ids.tolist()) or
            set(support_ids.reshape(-1).tolist()) & set(query_ids.reshape(-1).tolist()) or
            set(probe_ids.tolist()) & set(query_ids.reshape(-1).tolist())):
        raise ValueError("source teacher support/query/probe raw image IDs overlap")
    source_task = TaskData(f"deepsets:{task_index}:bank", "train", x_support, y_support,
                           x_query, y_query, support_context(x_support.mean(1), y_support),
                           support_ids, query_ids,
                           {"family": "deepsets", "domain": "deepsets", "role": "bank_teacher",
                            "task_id": task.task_id, "task_index": task_index,
                            "costs": cost.tolist(), "support_pool": "source_train",
                            "query_pool": "source_validation", "set_size": _SET_SIZE})
    protocol = InnerProtocol(steps=bank_steps, replicas=1, lr=.03, l2=.001,
                             checkpoint_every=max(1, bank_steps // 4), seed=seed,
                             metric="nmse")

    best_by_density: dict[int, list[tuple[float, int, Tensor, dict[str, Any], dict[str, Any]]]] = {
        density: [] for density in buckets
    }
    if persist_artifacts:
        bank_out.mkdir(parents=True, exist_ok=True)
    replay = RealReplay(protocol)
    store = ParallelMeasurementStore(bank_out, replay, device, devices=measurement_devices,
                                     batch_size=teacher_batch_size,
                                     persist_artifacts=False)
    try:
        # Chunking limits live fit tensors. Source candidates always stay in
        # memory; ``persist_artifacts`` controls only selected functional cards.
        execution_batch_size = teacher_batch_size * max(1, len(measurement_devices))
        for start in progress(range(0, bank_candidates, execution_batch_size),
                              desc=f"DeepSets bank candidates {task_index}", unit="batch"):
            stop = min(start + execution_batch_size, bank_candidates)
            entries = [(masks[candidate_id], source_task,
                        f"source-bank-candidate:{candidate_id}")
                       for candidate_id in range(start, stop)]
            initialization_seeds = [_candidate_initialization_seed(seed, candidate_id)
                                    for candidate_id in range(start, stop)]
            measured = store.measure_many(entries, desc=f"Bank {task_index} fits",
                                          initialization_seeds=initialization_seeds,
                                          retain_results=True)
            for candidate_id, (record, result) in zip(range(start, stop), measured):
                density = strata[candidate_id]
                score = float(result["replica_losses"][0])
                retained = best_by_density[density]
                retained_result = result
                if not persist_artifacts:
                    # Ranking and tokenization need only the terminal model
                    # weights and score; Adam moments and fit histories are
                    # never used to train a source teacher again.
                    retained_result = {"state_dict": result["state_dict"],
                                       "replica_losses": result["replica_losses"]}
                retained.append((score, candidate_id, masks[candidate_id], retained_result, record))
                retained.sort(key=lambda row: (row[0], row[1]))
                del retained[quotas[buckets.index(density)]:]
    finally:
        store.close()
    store.last_results.clear()
    del measured
    del record, result

    selected: list[tuple[float, int, Tensor, dict[str, Any], dict[str, Any]]] = []
    for density, quota in zip(buckets, quotas):
        rows = best_by_density[density]
        if len(rows) != quota:
            raise ValueError("source candidate selection did not retain the requested density quota")
        selected.extend(rows)
    selected.sort(key=lambda row: row[1])

    tokens: list[Tensor] = []
    selected_masks: list[Tensor] = []
    states: list[dict[str, Any]] = []
    for teacher_index, (score, candidate_id, mask, result, record) in enumerate(selected):
        state = _unwrap_model_state(result["state_dict"], row=0, rows=1)
        if "masks" in state and not torch.equal(state["masks"], mask):
            raise ValueError("source child state mask differs from its candidate mask")
        token, raw = extract_deepsets_functional_token(state, mask, probe_x)
        row_hash = _state_hash(state, mask)
        initialization_seed = _candidate_initialization_seed(seed, candidate_id)
        card_path = (bank_out / "maps" / f"candidate_{candidate_id:06d}_init_{initialization_seed}.pt"
                     if persist_artifacts else None)
        card_state = {name: state[name] for name in ("weight", "bias", "readout", "per_image_offset")}
        if persist_artifacts:
            write_functional_card(
                card_path, token=token, mask=mask, state=card_state,
                metadata={
                    "kind": "initial_teacher",
                    "candidate_id": int(candidate_id),
                    "initialization_seed": int(initialization_seed),
                    "score_name": "source_query_nmse",
                    "score": float(score),
                    "row_hash": row_hash,
                    "task_source": {
                        "task_id": task.task_id,
                        "task_index": int(task_index),
                        "bank_task_id": source_task.task_id,
                        "task_provenance": deepcopy(task.provenance),
                        "bank_task_provenance": deepcopy(source_task.provenance),
                        "probe_ids": probe_ids.detach().cpu().tolist(),
                        "probe_fingerprint": tensor_hash(probe_x),
                    },
                    "measurement": {
                        "protocol_id": protocol.fingerprint,
                        "mask_key": record.get("mask_key"),
                        "active_edges": int(mask.sum()),
                        "density_stratum": int(strata[candidate_id]),
                        "candidate_seed_rule": "seed + 1000003 * (candidate_id + 1)",
                    },
                })
        tokens.append(token)
        selected_masks.append(mask.detach().cpu().clone())
        teacher_state = {**raw, "state_dict": state, "row_hash": row_hash,
                         "source": {"kind": "initial_teacher", "task_id": task.task_id,
                                    "task_index": task_index, "teacher": teacher_index,
                                    "candidate_id": candidate_id, "density_stratum": strata[candidate_id],
                                    "active_edges": int(mask.sum()), "source_query_nmse": score,
                                    "artifact_path": (str(card_path.resolve()) if card_path is not None else None),
                                    "artifact_storage": ("file" if card_path is not None else "memory"),
                                    "card_schema": ("generator_evaluator.functional_map_card:v1"
                                                    if card_path is not None else None)}}
        if persist_artifacts:
            teacher_state["optimizer_state"] = deepcopy(result["optimizer_state"])
            teacher_state["history"] = deepcopy(result["history"])
        states.append(teacher_state)

    selected_masks_tensor = torch.stack(selected_masks)
    selected_masks.clear()
    reference_q_abs = states[0]["q_abs_mean"]
    aligned_q_abs = torch.stack([reference_q_abs] + [
        _align_hidden_columns(reference_q_abs, row["q_abs_mean"]) for row in states[1:]])
    mean_q_abs = aligned_q_abs.mean(0)
    baseline = _exact_topk(mean_q_abs, k)
    density_counts = {int(density): int((selected_masks_tensor.sum((1, 2)) == density).sum())
                      for density in buckets}
    selected_ids = [int(row[1]) for row in selected]
    selected_scores = {str(row[1]): row[0] for row in selected}
    selected_initialization_seeds = [_candidate_initialization_seed(seed, candidate_id)
                                     for candidate_id in selected_ids]
    # Release the search-only candidate matrix and optimizer results once the
    # retained terminal states have been materialized into the bank.
    best_by_density.clear()
    selected.clear()
    del masks, mask, result, record
    bank_tokens = torch.stack(tokens)[None]
    tokens.clear()
    del token
    partitions = {"bank_support_ids": support_ids.reshape(-1).unique().tolist(),
                  "bank_query_ids": query_ids.reshape(-1).unique().tolist(),
                  "probe_ids": probe_ids.detach().cpu().tolist()}
    provenance = {"family": "cooperative_deepsets", "domain": "deepsets",
                  "pattern": str(task_index), "task_id": task.task_id, "seed": int(seed),
                  "baseline_k": int(k), "probe_ids": partitions["probe_ids"],
                  "probe_fingerprint": tensor_hash(probe_x), "partitions": partitions,
                  "bank_task_id": source_task.task_id,
                  "teacher_density_counts": density_counts,
                  "candidate_count": int(bank_candidates),
                  "candidate_density_counts": {str(density): int(candidate_counts[index])
                                               for index, density in enumerate(buckets)},
                  "selected_candidate_ids": selected_ids,
                  "selected_initialization_seeds": selected_initialization_seeds,
                  "candidate_seed_rule": "seed + 1000003 * (candidate_id + 1)",
                  "selected_candidate_query_nmse": selected_scores,
                  "selection_density_buckets": list(buckets),
                  "selection_density_quotas": quotas,
                  "baseline_alignment": "label-free Hungarian cosine alignment of q_abs to teacher 0",
                  "selection_rule": "lowest fixed-horizon source-query NMSE per density; candidate ID breaks ties",
                  "quality_source": None, "bank_support_count": support_count,
                  "bank_query_count": query_count, "accepted_feedback_task_ids": [task.task_id],
                  "feedback_hashes": [], "persist_artifacts": bool(persist_artifacts),
                  "artifact_storage": "file" if persist_artifacts else "memory"}
    bank = FunctionalBank(bank_tokens, None, selected_masks_tensor, baseline,
                          provenance, states=states,
                          diagnostics={"probe_x": probe_x.detach().cpu().clone(),
                                       "aligned_q_abs": aligned_q_abs,
                                       "bank_support_ids": support_ids.detach().cpu().clone(),
                                       "bank_query_ids": query_ids.detach().cpu().clone(),
                                       "feedback_rows_added": 0})
    return bank


def build_cooperative_deepsets_fixture(
    data_root: str | Path,
    *,
    seed: int = 4100,
    train_task_count: int = 2,
    test_task_count: int = 2,
    bank_steps: int = 4000,
    teachers_per_task: int = 1024,
    bank_candidates: int = 4096,
    teacher_batch_size: int = 64,
    support_count: int | None = None,
    query_count: int | None = None,
    selection_count: int | None = None,
    probe_count: int = 32,
    bank_support_count: int | None = None,
    bank_query_count: int | None = None,
    k: int = 7526,
    hidden: int = 32,
    device: str = "cpu",
    measurement_devices: Sequence[str] | None = None,
    out: str | Path | None = None,
    fixed_test_spec: dict[str, Any] | None = None,
    persist_artifacts: bool = False,
) -> tuple[dict[str, FunctionalBank], list[TaskData], list[TaskData], dict[str, Any]]:
    """Build own-task banks, matched train/selection tasks and sealed tests.

    Source fit artifacts are memory-only unless ``persist_artifacts`` is
    explicitly enabled.
    """
    _require_count("train_task_count", train_task_count, 100_000, minimum=2)
    _require_count("test_task_count", test_task_count, 100_000)
    _require_count("bank_steps", bank_steps, 10_000)
    _require_count("teachers_per_task", teachers_per_task, 4096)
    _require_count("bank_candidates", bank_candidates, 100_000, minimum=teachers_per_task)
    _require_count("teacher_batch_size", teacher_batch_size, 100_000)
    from deepsets_vaae.core import load_data
    data = load_data(data_root, seed, "cpu")
    if support_count is None:
        support_count = (int(fixed_test_spec["support_count"]) if fixed_test_spec is not None else
                         len(data["target_train"].features) //
                         (_SET_SIZE * (train_task_count + test_task_count)))
    _require_count("support_count", support_count, 100_000)
    for name, value in (("bank_support_count", bank_support_count),
                        ("bank_query_count", bank_query_count)):
        if value is not None:
            _require_count(name, value, 100_000)
    if fixed_test_spec is not None and query_count is None:
        query_count = int(fixed_test_spec["query_count"])
    query_budget = len(data["target_validation"].features) // (_SET_SIZE * train_task_count)
    if query_count is None and selection_count is None:
        query_count = selection_count = min(51, query_budget // 2)
    elif query_count is None:
        query_count = min(51, query_budget - selection_count)
    elif selection_count is None:
        selection_count = min(51, query_budget - query_count)
    _require_count("query_count", query_count, 100_000)
    _require_count("selection_count", selection_count, 100_000)
    _require_count("probe_count", probe_count, 4096)
    _require_count("hidden", hidden, 4096)
    _require_count("k", k, _FEATURES * hidden)
    edges = _FEATURES * hidden
    if k > edges:
        raise ValueError("k exceeds the DeepSets mask size")
    buckets = _density_buckets(k, teachers_per_task, edges)
    candidates_per_density = _stratum_counts(bank_candidates, len(buckets))
    keep_per_density = _stratum_counts(teachers_per_task, len(buckets))
    if any(available < keep for available, keep in zip(candidates_per_density, keep_per_density)):
        raise ValueError("bank_candidates must provide enough candidates in every density stratum")

    # Reuse the adapter's task builder for a compatibility check. Its first
    # two train costs remain the fixture's first two costs; larger fixtures
    # extend the same seeded sequence with one independent vector per role.
    base_tasks, base_test_spec = make_deepsets_tasks(data_root, seed, support_count, query_count)
    base_train_tasks = [task for task in base_tasks if task.split == "train"]
    if [task.task_id for task in base_train_tasks] != ["deepsets:train:0", "deepsets:train:1"]:
        raise ValueError("DeepSets adapter returned an unexpected source task order")
    generated_costs = _cost_vectors(seed, 6)
    costs = [_fixture_cost_vector(seed, "train", index) for index in range(train_task_count)]
    if any(not torch.equal(_cost(task), costs[index])
           for index, task in enumerate(base_train_tasks)):
        raise ValueError("DeepSets adapter returned unexpected seeded train costs")
    adapter_heldout_costs = [torch.as_tensor(value, dtype=torch.float32)
                             for value in base_test_spec["costs"]]
    if len(adapter_heldout_costs) != 2:
        raise ValueError("DeepSets adapter must reserve two held-out cost vectors")
    if any(not torch.equal(adapter_heldout_costs[index], generated_costs[4 + index])
           for index in range(2)):
        raise ValueError("DeepSets adapter returned unexpected seeded held-out costs")

    support_rows = support_count * _SET_SIZE
    query_rows = query_count * _SET_SIZE
    selection_rows = selection_count * _SET_SIZE
    if fixed_test_spec is None:
        train_support_splits = _partition_split(data["target_train"],
                                                [support_rows] * (train_task_count + test_task_count),
                                                seed + 31_117)
        test_query_splits = _partition_split(data["target_test"], [query_rows] * test_task_count,
                                             seed + 31_121)
    else:
        train_support_splits, test_query_splits = _fixed_test_pools(
            data, fixed_test_spec, data_root=data_root, seed=seed, train_task_count=train_task_count,
            test_task_count=test_task_count, support_count=support_count, query_count=query_count)
    validation_splits = _partition_split(data["target_validation"],
                                         [size for _ in range(train_task_count)
                                          for size in (query_rows, selection_rows)],
                                         seed + 31_119)

    train_tasks: list[TaskData] = []
    selection_tasks: list[TaskData] = []
    task_id_width = train_task_count + test_task_count
    for index in range(train_task_count):
        task = _make_task(f"deepsets:{index}", "train", train_support_splits[index],
                          validation_splits[index * 2], costs[index], support_count,
                          query_count, seed + 40_000 + index, role="train", task_index=index,
                          task_count=task_id_width, evaluator_task_id=index)
        selection = _make_task(f"deepsets:{index}:selection", "validation",
                               train_support_splits[index], validation_splits[index * 2 + 1],
                               costs[index], support_count, selection_count,
                               seed + 50_000 + index, role="selection", task_index=index,
                               task_count=task_id_width, evaluator_task_id=index)
        # Selection has the same support draw and context as its own training
        # task; only its query pool and query draw differ.
        selection = replace(selection, x_support=task.x_support, y_support=task.y_support,
                            support_ids=task.support_ids, context=task.context)
        train_tasks.append(task)
        selection_tasks.append(selection)

    # Source bank pools are disjoint from one another and from all target
    # evaluator and held-out pools. Each bank also reserves a private probe.
    source_train_count = len(data["source_train"].features)
    source_validation_count = len(data["source_validation"].features)
    source_train_sizes = ([source_train_count // 2, source_train_count - source_train_count // 2]
                          if train_task_count == 2 else _stratum_counts(source_train_count, train_task_count))
    source_validation_sizes = (
        [source_validation_count // 2, source_validation_count - source_validation_count // 2]
        if train_task_count == 2 else _stratum_counts(source_validation_count, train_task_count))
    source_train_splits = _partition_split(data["source_train"], source_train_sizes, seed + 31_123)
    source_validation_splits = _partition_split(
        data["source_validation"], source_validation_sizes, seed + 31_127)
    bank_probe_x: list[Tensor] = []
    bank_probe_ids: list[Tensor] = []
    bank_support_splits: list[Any] = []
    for index, source_split in enumerate(source_train_splits):
        if len(source_split.features) <= probe_count + 1:
            raise ValueError("source training pool is too small for the reserved probe")
        probe_rows = torch.randperm(len(source_split.features), generator=torch.Generator().manual_seed(
            seed + 60_001 + index))[:probe_count]
        probe_mask = torch.ones(len(source_split.features), dtype=torch.bool)
        probe_mask[probe_rows] = False
        bank_probe_x.append(source_split.features.index_select(0, probe_rows).detach().cpu())
        bank_probe_ids.append(source_split.source_ids.index_select(0, probe_rows).detach().cpu())
        bank_support_splits.append(_split_rows(source_split, torch.nonzero(probe_mask).flatten()))

    primary_device, devices = _available_measurement_devices(device, measurement_devices)
    bank_root = Path(out).resolve() / "banks" if out is not None else None
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if bank_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="cooperative-deepsets-banks-")
        bank_root = Path(temporary.name) / "banks"
    banks: dict[str, FunctionalBank] = {}
    try:
        for index in range(train_task_count):
            bank = _build_bank(index, train_tasks[index], bank_support_splits[index],
                               source_validation_splits[index], bank_probe_x[index], bank_probe_ids[index],
                               seed=seed + 70_003 * (index + 1), bank_steps=bank_steps,
                               bank_candidates=bank_candidates, teachers_per_task=teachers_per_task,
                               support_count=(bank_support_count if bank_support_count is not None else
                                              len(bank_support_splits[index].features) // _SET_SIZE),
                               query_count=(bank_query_count if bank_query_count is not None else
                                            len(source_validation_splits[index].features) // _SET_SIZE),
                               teacher_batch_size=teacher_batch_size, k=k, hidden=hidden,
                               device=primary_device, measurement_devices=devices,
                               bank_out=bank_root / str(index),
                               persist_artifacts=persist_artifacts)
            banks[str(index)] = bank
    finally:
        if temporary is not None:
            temporary.cleanup()
    if temporary is not None:
        for bank in banks.values():
            for teacher in bank.states:
                teacher.get("source", {}).pop("artifact_path", None)

    test_support_pools = [split.source_ids.detach().cpu().tolist()
                          for split in train_support_splits[train_task_count:]]
    test_query_pools = [split.source_ids.detach().cpu().tolist() for split in test_query_splits]
    all_eval_ids: set[int] = set()
    for task in train_tasks + selection_tasks:
        all_eval_ids |= _task_ids(task)
    sealed_pool_ids = set().union(*(set(pool) for pool in test_support_pools + test_query_pools))
    if all_eval_ids & sealed_pool_ids:
        raise ValueError("reserved held-out image pools overlap evaluator observations")
    all_costs = _task_costs(seed, train_tasks, test_task_count)
    heldout_costs = all_costs[train_task_count:]
    test_spec = {"family": "cooperative_deepsets", "domain": "deepsets",
                 "data_root": str(Path(data_root).resolve()), "seed": int(seed),
                 "train_patterns": [str(index) for index in range(train_task_count)],
                 "test_pattern": "heldout", "test_task_count": int(test_task_count),
                 "task_id_encoding": "one_hot", "task_id_width": int(task_id_width),
                 "test_conditions": [str(index) for index in range(test_task_count)],
                 "costs": [value.tolist() for value in heldout_costs],
                 "support_count": int(support_count), "query_count": int(query_count),
                 "selection_count": int(selection_count),
                 "test_support_pools": test_support_pools,
                 "test_query_pools": test_query_pools,
                 "test_ids": sorted(sealed_pool_ids),
                 "test_support_ids": sorted(set().union(*(set(pool) for pool in test_support_pools))),
                 "test_query_ids": sorted(set().union(*(set(pool) for pool in test_query_pools))),
                 "support_pool": "target_train", "query_pool": "target_test",
                 "materialized": False}
    validate_cooperative_deepsets_inputs(
        banks, train_tasks, selection_tasks, test_spec,
        _FixtureConfig(k=k, hidden=hidden, train_task_count=train_task_count,
                       test_task_count=test_task_count),
        InnerProtocol(steps=1, replicas=4, metric="nmse"))
    return banks, train_tasks, selection_tasks, test_spec


class _FixtureConfig:
    """Minimal shape contract used for the fixture's own preflight validation."""

    domain = "deepsets"
    test_pattern = "heldout"
    features = _FEATURES

    def __init__(self, *, k: int, hidden: int, train_task_count: int = 2,
                 test_task_count: int = 2):
        self.k = int(k)
        self.hidden = int(hidden)
        self.train_patterns = tuple(str(index) for index in range(train_task_count))
        self.test_task_count = int(test_task_count)


def _validate_task_id_context(task: TaskData, task_id: int, width: int) -> None:
    expected_support_context = support_context(task.x_support.mean(1), task.y_support).to(task.context.device)
    expected_identity = F.one_hot(torch.tensor(task_id), num_classes=width).to(
        device=task.context.device, dtype=task.context.dtype)
    provenance = task.provenance
    if (provenance.get("evaluator_task_id") != task_id or
            provenance.get("task_id_encoding") != "one_hot" or
            provenance.get("task_id_width") != width or
            task.context.shape != (expected_support_context.numel() + width,) or
            not torch.equal(task.context[:-width], expected_support_context) or
            not torch.equal(task.context[-width:], expected_identity)):
        raise ValueError("DeepSets task context must end with its provenance-matched one-hot task ID")


def validate_cooperative_deepsets_inputs(banks: dict[str, FunctionalBank],
                                         train_tasks: Sequence[TaskData],
                                         selection_tasks: Sequence[TaskData],
                                         test_spec: dict[str, Any], config: Any,
                                         protocol: InnerProtocol) -> None:
    """Check role identity, dimensions, costs, and raw-image isolation."""
    roles = tuple(getattr(config, "train_patterns", ()))
    expected_roles = tuple(str(index) for index in range(len(roles)))
    if (getattr(config, "domain", None) != "deepsets" or len(roles) < 2 or
            roles != expected_roles or getattr(config, "features", None) != _FEATURES or
            len(train_tasks) != len(roles) or len(selection_tasks) != len(roles) or
            not isinstance(banks, dict) or tuple(banks) != roles):
        raise ValueError("cooperative DeepSets inputs require at least two ordered task roles 0 through N-1")
    hidden = int(getattr(config, "hidden", 0))
    k = int(getattr(config, "k", 0))
    if hidden < 1 or not 1 <= k <= _FEATURES * hidden:
        raise ValueError("cooperative DeepSets mask dimensions or K are invalid")
    if protocol.metric != "nmse" or protocol.replicas < 2:
        raise ValueError("cooperative DeepSets evaluation requires at least two NMSE replicas")
    if (not isinstance(test_spec, dict) or test_spec.get("family") != "cooperative_deepsets" or
            test_spec.get("materialized") is not False or
            tuple(test_spec.get("train_patterns", ())) != roles or
            test_spec.get("test_pattern") != getattr(config, "test_pattern", None) or
            test_spec.get("domain") != "deepsets"):
        raise ValueError("held-out DeepSets tests must remain sealed in the fixture specification")
    spec_test_count = test_spec.get("test_task_count", len(test_spec.get("costs", ())))
    test_task_count = getattr(config, "test_task_count", spec_test_count)
    if (not isinstance(test_task_count, int) or test_task_count < 1 or
            spec_test_count != test_task_count or len(test_spec.get("costs", ())) != test_task_count or
            len(test_spec.get("test_support_pools", ())) != test_task_count or
            len(test_spec.get("test_query_pools", ())) != test_task_count):
        raise ValueError("the sealed specification must retain the configured held-out cost and image-pool pairs")
    expected_conditions = [str(index) for index in range(test_task_count)]
    if ("test_conditions" in test_spec and
            list(test_spec["test_conditions"]) != expected_conditions):
        raise ValueError("sealed DeepSets test conditions must be ordered from 0 through M-1")
    task_id_width = len(roles) + test_task_count
    if test_spec.get("task_id_encoding") == "one_hot":
        if test_spec.get("task_id_width") != task_id_width:
            raise ValueError("sealed DeepSets task ID width must cover all train and held-out roles")
    elif (test_spec.get("task_id_encoding") is not None or len(roles) != 2 or
          test_task_count != 2 or "test_task_count" in test_spec):
        raise ValueError("new cooperative DeepSets inputs require one-hot task ID context encoding")
    if any(torch.as_tensor(value).shape != (10,) or not torch.isfinite(
            torch.as_tensor(value, dtype=torch.float32)).all() for value in test_spec["costs"]):
        raise ValueError("held-out conditions require ten-value digit-cost vectors")
    support_count = int(test_spec.get("support_count", 0))
    query_count = int(test_spec.get("query_count", 0))
    if "selection_count" in test_spec:
        selection_count = int(test_spec["selection_count"])
    else:
        # Early bootstrap snapshots omitted this budget; the saved selection
        # sets retain its exact value. Explicit invalid budgets still fail.
        selection_counts = {len(task.x_query) for task in selection_tasks}
        if len(selection_counts) != 1:
            raise ValueError("DeepSets selection query budgets must agree across tasks")
        selection_count = selection_counts.pop()
    if min(support_count, query_count, selection_count) < 1:
        raise ValueError("DeepSets support and query budgets must be positive")
    sealed_pools = [set(map(int, pool)) for pool in
                    test_spec["test_support_pools"] + test_spec["test_query_pools"]]
    if (any(not pool for pool in sealed_pools) or
            any(sealed_pools[left] & sealed_pools[right]
                for left in range(len(sealed_pools)) for right in range(left + 1, len(sealed_pools))) or
            set(map(int, test_spec["test_ids"])) != set().union(*sealed_pools) or
            set(map(int, test_spec.get("test_support_ids", ()))) != set().union(*sealed_pools[:test_task_count]) or
            set(map(int, test_spec.get("test_query_ids", ()))) != set().union(*sealed_pools[test_task_count:])):
        raise ValueError("sealed DeepSets image pools must be pairwise disjoint and correctly indexed")
    if len({id(bank) for bank in banks.values()}) != len(roles):
        raise ValueError("each DeepSets generator needs its own functional bank")

    bank_ids: set[int] = set()
    task_ids: set[int] = set()
    role_observation_ids: set[int] = set()
    train_costs: list[Tensor] = []
    for index, (role, task, selection) in enumerate(zip(roles, train_tasks, selection_tasks)):
        bank = banks[role]
        expected = ("deepsets:" + role, "deepsets:" + role + ":selection")
        if (task.task_id != expected[0] or task.split != "train" or
                selection.task_id != expected[1] or selection.split != "validation" or
                task.provenance.get("family") != "deepsets" or
                selection.provenance.get("family") != "deepsets" or
                task.provenance.get("domain") != "deepsets" or
                selection.provenance.get("domain") != "deepsets" or
                task.provenance.get("task_index") != index or
                selection.provenance.get("task_index") != index or
                selection.provenance.get("role") != "selection"):
            raise ValueError("DeepSets task identity, order, or role is corrupt")
        if (bank.provenance.get("family") != "cooperative_deepsets" or
                bank.provenance.get("domain") != "deepsets" or
                bank.provenance.get("pattern") != role or
                bank.provenance.get("task_id") != task.task_id or
                bank.provenance.get("accepted_feedback_task_ids") != [task.task_id] or
                bank.tokens.shape[2:] != (hidden, bank.tokens.shape[3]) or
                bank.masks.shape != (bank.tokens.shape[1], _FEATURES, hidden) or
                bank.baseline_mask.shape != (_FEATURES, hidden) or
                int(bank.baseline_mask.sum()) != k or bank.quality is not None):
            raise ValueError("DeepSets functional bank identity or dimensions are invalid")
        if (task.x_support.shape[1:] != (_SET_SIZE, _FEATURES) or
                task.x_query.shape[1:] != (_SET_SIZE, _FEATURES) or
                selection.x_support.shape != task.x_support.shape or
                selection.x_query.shape[1:] != (_SET_SIZE, _FEATURES) or
                len(task.x_support) != support_count or len(task.x_query) != query_count or
                len(selection.x_support) != support_count or len(selection.x_query) != selection_count):
            raise ValueError("DeepSets evaluator sets must have shape [count, 5, 784]")
        if (not torch.equal(task.support_ids, selection.support_ids) or
                not torch.equal(task.x_support, selection.x_support) or
                not torch.equal(task.y_support, selection.y_support) or
                not torch.equal(task.context, selection.context)):
            raise ValueError("selection must reuse its own train support and context")
        if test_spec.get("task_id_encoding") == "one_hot":
            _validate_task_id_context(task, index, task_id_width)
            _validate_task_id_context(selection, index, task_id_width)
        if set(task.query_ids.reshape(-1).tolist()) & set(selection.query_ids.reshape(-1).tolist()):
            raise ValueError("training and selection query image IDs overlap")
        this_role_ids = _task_ids(task) | _task_ids(selection)
        if role_observation_ids & this_role_ids:
            raise ValueError("DeepSets train/selection observations overlap across task roles")
        role_observation_ids |= this_role_ids
        task_cost, selection_cost = _cost(task), _cost(selection)
        if not torch.equal(task_cost, selection_cost):
            raise ValueError("selection cost vector differs from its own training task")
        if any(torch.equal(task_cost, previous) for previous in train_costs):
            raise ValueError("DeepSets training tasks must have distinct cost vectors")
        train_costs.append(task_cost)
        task_ids |= _task_ids(task) | _task_ids(selection)
        parts = bank.provenance.get("partitions", {})
        required_parts = ("bank_support_ids", "bank_query_ids", "probe_ids")
        if any(name not in parts for name in required_parts):
            raise ValueError("DeepSets bank is missing its reserved source IDs")
        this_bank_ids = [set(map(int, parts[name])) for name in required_parts]
        if any(this_bank_ids[left] & this_bank_ids[right]
               for left in range(3) for right in range(left + 1, 3)):
            raise ValueError("bank support/query/probe image IDs overlap")
        if bank_ids & set().union(*this_bank_ids):
            raise ValueError("own-task source banks reuse raw image IDs")
        bank_ids |= set().union(*this_bank_ids)
        if set(bank.provenance.get("probe_ids", ())) != this_bank_ids[2]:
            raise ValueError("DeepSets bank probe identity is inconsistent")
        if tensor_hash(bank.diagnostics.get("probe_x", torch.empty(0))) != bank.provenance.get("probe_fingerprint"):
            raise ValueError("DeepSets bank probe data differ from their fingerprint")

    sealed_pool_ids = set(map(int, test_spec["test_ids"]))
    if task_ids & sealed_pool_ids:
        raise ValueError("held-out DeepSets image pools overlap evaluator observations")
    if bank_ids & (task_ids | sealed_pool_ids):
        raise ValueError("source-bank raw image IDs overlap evaluator or held-out pools")
    if any(set(map(int, pool)) & task_ids for pool in test_spec["test_support_pools"] +
           test_spec["test_query_pools"]):
        raise ValueError("held-out DeepSets role pools overlap evaluator observations")
    heldout_costs = [torch.as_tensor(value, dtype=torch.float32) for value in test_spec["costs"]]
    if (any(torch.equal(heldout_costs[left], heldout_costs[right])
            for left in range(len(heldout_costs)) for right in range(left + 1, len(heldout_costs))) or
            any(torch.equal(cost, train_cost) for cost in heldout_costs for train_cost in train_costs)):
        raise ValueError("held-out DeepSets cost vectors must be new task identities")


def make_cooperative_deepsets_test_tasks(test_spec: dict[str, Any]) -> list[TaskData]:
    """Materialize sealed conditions at the final evaluation boundary."""
    if not isinstance(test_spec, dict) or test_spec.get("family") != "cooperative_deepsets" or \
            test_spec.get("materialized") is not False:
        raise ValueError("not an unmaterialized cooperative DeepSets test specification")
    from deepsets_vaae.core import load_data
    data = load_data(test_spec["data_root"], int(test_spec["seed"]), "cpu")
    result: list[TaskData] = []
    train_task_count = len(test_spec.get("train_patterns", ()))
    task_id_width = test_spec.get("task_id_width", train_task_count + len(test_spec["costs"]))
    for index, costs in enumerate(test_spec["costs"]):
        support_split = _split_by_source_ids(data["target_train"], test_spec["test_support_pools"][index])
        query_split = _split_by_source_ids(data["target_test"], test_spec["test_query_pools"][index])
        task = _make_task(f"deepsets:test:{index}", "test", support_split, query_split,
                          torch.as_tensor(costs, dtype=torch.float32),
                          int(test_spec["support_count"]), int(test_spec["query_count"]),
                          int(test_spec["seed"]) + 80_000 + index,
                          role="sealed_test", task_index=index,
                          task_count=task_id_width if test_spec.get("task_id_encoding") == "one_hot" else None,
                          evaluator_task_id=train_task_count + index)
        task.provenance.update({"heldout_condition": str(index), "support_pool": "target_train",
                                "query_pool": "target_test"})
        result.append(task)
    return result


def validate_deepsets_test_tasks(test_tasks: Sequence[TaskData], test_spec: dict[str, Any],
                                 train_tasks: Sequence[TaskData],
                                 selection_tasks: Sequence[TaskData]) -> None:
    """Verify final factory outputs have the sealed conditions and no leakage."""
    if not isinstance(test_spec, dict):
        raise ValueError("final DeepSets factory needs a sealed specification")
    test_count = int(test_spec.get("test_task_count", len(test_spec.get("costs", ()))))
    if (not isinstance(test_tasks, (list, tuple)) or len(test_tasks) != test_count or test_count < 1 or
            test_spec.get("family") != "cooperative_deepsets" or test_spec.get("materialized") is not False or
            len(train_tasks) < 2 or len(selection_tasks) != len(train_tasks) or
            len(test_spec.get("costs", ())) != test_count or
            len(test_spec.get("test_support_pools", ())) != test_count or
            len(test_spec.get("test_query_pools", ())) != test_count):
        raise ValueError("final DeepSets factory must produce the configured tasks from a sealed spec")
    if ("test_conditions" in test_spec and
            list(test_spec["test_conditions"]) != [str(index) for index in range(test_count)]):
        raise ValueError("final DeepSets conditions must be ordered from 0 through M-1")
    expected_support = int(test_spec["support_count"])
    expected_query = int(test_spec["query_count"])
    evaluator_ids: set[int] = set()
    train_costs = [_cost(task) for task in train_tasks]
    for task in list(train_tasks) + list(selection_tasks):
        evaluator_ids |= _task_ids(task)
    seen_costs: list[Tensor] = []
    seen_test_ids: set[int] = set()
    for index, task in enumerate(test_tasks):
        expected_id = f"deepsets:test:{index}"
        expected_cost = torch.as_tensor(test_spec["costs"][index], dtype=torch.float32)
        if (not isinstance(task, TaskData) or task.task_id != expected_id or task.split != "test" or
                task.provenance.get("family") != "deepsets" or
                task.provenance.get("domain") != "deepsets" or
                task.provenance.get("role") != "sealed_test" or
                task.provenance.get("heldout_condition") != str(index) or
                task.provenance.get("support_pool") != "target_train" or
                task.provenance.get("query_pool") != "target_test"):
            raise ValueError("final DeepSets test task has an invalid ID or data role")
        if test_spec.get("task_id_encoding") == "one_hot":
            task_id_width = int(test_spec.get("task_id_width", -1))
            expected_width = len(test_spec.get("train_patterns", ())) + test_count
            if task_id_width != expected_width:
                raise ValueError("final DeepSets task ID width does not cover all configured roles")
            _validate_task_id_context(task, len(train_tasks) + index, task_id_width)
        if (len(task.x_support) != expected_support or len(task.x_query) != expected_query or
                task.x_support.shape[1:] != (_SET_SIZE, _FEATURES) or
                task.x_query.shape[1:] != (_SET_SIZE, _FEATURES)):
            raise ValueError("final DeepSets test task has an invalid support/query shape or count")
        cost = _cost(task)
        if not torch.equal(cost, expected_cost) or any(torch.equal(cost, train_cost) for train_cost in train_costs):
            raise ValueError("final DeepSets task exposes a wrong or training cost vector")
        if any(torch.equal(cost, previous) for previous in seen_costs):
            raise ValueError("final DeepSets conditions must have distinct cost vectors")
        seen_costs.append(cost)
        support_ids = set(task.support_ids.reshape(-1).tolist())
        query_ids = set(task.query_ids.reshape(-1).tolist())
        allowed_support = set(map(int, test_spec["test_support_pools"][index]))
        allowed_query = set(map(int, test_spec["test_query_pools"][index]))
        if not support_ids <= allowed_support or not query_ids <= allowed_query:
            raise ValueError("final DeepSets task uses image IDs outside its sealed pools")
        if (support_ids & query_ids or (support_ids | query_ids) & evaluator_ids or
                (support_ids | query_ids) & seen_test_ids):
            raise ValueError("final DeepSets task leaks image IDs across roles or conditions")
        seen_test_ids |= support_ids | query_ids


def _stratified_rows(bank: FunctionalBank, max_teachers: int) -> list[int]:
    groups: dict[int, list[int]] = {}
    for index, mask in enumerate(bank.masks):
        groups.setdefault(int(mask.sum()), []).append(index)
    if max_teachers < len(groups):
        raise ValueError("max_teachers must retain a representative for every density anchor")

    def is_feedback(index: int) -> bool:
        return bank.states[index].get("source", {}).get("kind") == "feedback"

    target = int(bank.provenance.get("baseline_k", -1))
    priorities = []
    for density in (min(groups), target, max(groups)):
        if density in groups and density not in priorities:
            priorities.append(density)
    priorities.extend(density for density in sorted(groups) if density not in priorities)
    selected: list[int] = []
    for density in priorities:
        anchor = next((index for index in groups[density] if not is_feedback(index)), groups[density][0])
        groups[density].remove(anchor)
        selected.append(anchor)
    newest_feedback = [index for index in range(len(bank.masks) - 1, -1, -1)
                       if is_feedback(index) and index not in selected]
    newest_feedback = newest_feedback[:max(0, max_teachers - len(selected))]
    selected.extend(newest_feedback)
    for index in newest_feedback:
        groups[int(bank.masks[index].sum())].remove(index)
    while len(selected) < max_teachers:
        choices = [density for density, rows in groups.items() if rows]
        if not choices:
            break
        counts = {density: sum(int(bank.masks[index].sum()) == density for index in selected)
                  for density in choices}
        density = min(choices, key=lambda value: (counts[value], value))
        selected.append(groups[density].pop(0))
    return selected


def append_deepsets_feedback(bank: FunctionalBank, mask: Tensor, measurement: dict[str, Any],
                             probe_x: Tensor, *, task_id: str,
                             artifact_path: str | Path | None = None,
                             max_teachers: int = 100, eligible: bool = True) -> FunctionalBank:
    """Add distinct terminal training replicas as normalized functional rows."""
    if not isinstance(bank, FunctionalBank) or bank.provenance.get("family") != "cooperative_deepsets":
        raise ValueError("feedback requires a cooperative DeepSets functional bank")
    if not eligible:
        raise ValueError("held-out topology measurements are not eligible for feedback")
    if task_id not in bank.provenance.get("accepted_feedback_task_ids", ()):
        raise ValueError("feedback is eligible only for this bank's own training task")
    if (not isinstance(measurement, dict) or measurement.get("label_source") != "fresh_terminal_query" or
            not measurement.get("fixed_horizon") or measurement.get("task_id") != task_id or
            not isinstance(measurement.get("protocol_id"), str) or not measurement["protocol_id"]):
        raise ValueError("feedback requires a fresh fixed-horizon measurement from the own training task")
    states = measurement.get("state_dict")
    if not isinstance(states, dict) or "weight" not in states or not torch.is_tensor(states["weight"]):
        raise ValueError("feedback lacks the complete terminal DeepSets state")
    mask = torch.as_tensor(mask, dtype=torch.float32).detach().cpu()
    if (mask.shape != bank.baseline_mask.shape or not torch.isfinite(mask).all() or
            not bool(((mask == 0) | (mask == 1)).all())):
        raise ValueError("feedback mask must be a binary [784, hidden] matrix")
    replica_count = int(states["weight"].shape[1]) if states["weight"].ndim >= 4 else 0
    if replica_count < 2:
        raise ValueError("feedback requires terminal states from at least two prescribed replicas")
    for field in ("replica_losses", "seeds", "plateau_flags"):
        if not isinstance(measurement.get(field), (list, tuple)) or len(measurement[field]) != replica_count:
            raise ValueError(f"feedback requires a protocol record for all {field}")
    if not torch.isfinite(torch.as_tensor(measurement["replica_losses"], dtype=torch.float32)).all():
        raise ValueError("feedback replica losses must be finite")
    path = Path(artifact_path).resolve() if artifact_path is not None else None
    if path is not None and not path.is_file():
        raise ValueError("feedback artifact_path must name the saved real measurement")
    probe_x = torch.as_tensor(probe_x, dtype=torch.float32).detach().cpu()
    if tensor_hash(probe_x) != bank.provenance.get("probe_fingerprint"):
        raise ValueError("feedback must use the bank's fixed reserved probe")
    _require_count("max_teachers", max_teachers, 100_000,
                   minimum=len(set(bank.masks.sum((1, 2)).long().tolist())))
    existing = set(bank.provenance.get("feedback_hashes", ()))
    existing.update(item.get("row_hash") for item in bank.states
                    if isinstance(item, dict) and item.get("row_hash"))
    tokens, masks, additions = [], [], []
    if "masks" in states:
        state_masks = torch.as_tensor(states["masks"], dtype=torch.float32).detach().cpu()
        if state_masks.shape != (1, replica_count, *mask.shape) or not torch.equal(
                state_masks[0], mask.expand_as(state_masks[0])):
            raise ValueError("feedback terminal state mask differs from the measured mask")
    for replica in range(replica_count):
        state = _unwrap_model_state(states, row=replica, rows=replica_count)
        row_hash = _state_hash(state, mask)
        if row_hash in existing:
            continue
        token, raw = extract_deepsets_functional_token(state, mask, probe_x)
        tokens.append(token)
        masks.append(mask.clone())
        addition = {**raw, "state_dict": state, "row_hash": row_hash,
                    "source": {"kind": "feedback", "task_id": task_id,
                               "replica": replica,
                               "artifact_path": str(path) if path is not None else None,
                               "artifact_storage": "file" if path is not None else "memory",
                               "source_mask": mask.clone(),
                               "protocol_id": measurement["protocol_id"]}}
        if bank.provenance.get("persist_artifacts", False):
            addition["optimizer_state"] = _optimizer_replica(
                measurement.get("optimizer_state", {}), replica, replica_count)
            addition["history"] = _history_replica(measurement.get("history", {}),
                                                    replica, replica_count)
        additions.append(addition)
        existing.add(row_hash)
    if not tokens:
        return bank
    combined = FunctionalBank(torch.cat((bank.tokens, torch.stack(tokens)[None]), dim=1), None,
                              torch.cat((bank.masks, torch.stack(masks)), dim=0), bank.baseline_mask,
                              deepcopy(bank.provenance), states=[*bank.states, *additions],
                              diagnostics=deepcopy(bank.diagnostics))
    keep = _stratified_rows(combined, max_teachers)
    provenance = deepcopy(combined.provenance)
    provenance["feedback_hashes"] = sorted(existing)
    provenance["teacher_density_counts"] = {
        int(density): int((combined.masks[keep].sum((1, 2)) == density).sum())
        for density in torch.unique(combined.masks[keep].sum((1, 2)).long()).tolist()}
    diagnostics = deepcopy(combined.diagnostics)
    diagnostics["feedback_rows_added"] = int(diagnostics.get("feedback_rows_added", 0)) + len(additions)
    return FunctionalBank(combined.tokens[:, keep], None, combined.masks[keep], combined.baseline_mask,
                          provenance, states=[combined.states[index] for index in keep], diagnostics=diagnostics)
