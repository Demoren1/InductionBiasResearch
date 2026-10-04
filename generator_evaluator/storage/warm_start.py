"""Validated import of a cooperative critic and its train-only search state.

This module deliberately imports an *epoch checkpoint*, rather than a frozen
selection artifact.  The returned bundle is suitable for a fresh generator
search: it contains the live functional banks and real train/validation
measurements, but never materializes or imports final-test measurements.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import torch

from generator_evaluator.data.types import InnerProtocol, RealReplay, TaskData, support_context, topology_id


_ARCHITECTURE_FIELDS = ("width", "heads", "layers", "ensemble_members")
_EVALUATOR_POLICY = "initial_bank_only"
_SUPPORTED_EVALUATOR_POLICIES = (_EVALUATOR_POLICY, "initial_bank_plus_acquisition")
_EVALUATOR_ROW_FIELDS = (
    "topology_id", "mask_key", "task_id", "task_split", "split",
    "protocol_id", "task_fingerprint", "label_source", "quality",
    "replica_losses", "seeds", "active_edges", "density",
)
_REQUIRED_CHECKPOINT_FIELDS = (
    "epoch", "banks", "replay", "ensemble", "evaluator_training_state",
)


def _evaluator_row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only immutable training identity and labels from a replay row."""
    if not isinstance(row, Mapping):
        raise ValueError("evaluator bank rows must be mappings")
    missing = [field for field in _EVALUATOR_ROW_FIELDS if field not in row]
    if missing:
        raise ValueError("evaluator bank row lacks identity or label metadata")
    result = {field: row[field] for field in _EVALUATOR_ROW_FIELDS}
    result["quality"] = float(result["quality"])
    result["replica_losses"] = [float(value) for value in result["replica_losses"]]
    result["seeds"] = list(result["seeds"])
    result["active_edges"] = int(result["active_edges"])
    result["density"] = float(result["density"])
    return result


def _evaluator_row_key(row: Mapping[str, Any]) -> str:
    return json.dumps(_evaluator_row_identity(row), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def evaluator_bank_fingerprint(topology_ids: Sequence[str],
                               rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash an immutable evaluator-bank manifest, independent of copied paths."""
    if isinstance(topology_ids, (str, bytes)):
        raise ValueError("evaluator topology IDs must be a sequence")
    ids = list(topology_ids)
    if any(not isinstance(identity, str) or not identity for identity in ids):
        raise ValueError("evaluator topology IDs must be nonempty strings")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("evaluator bank rows must be a sequence")
    sorted_ids = sorted(set(ids))
    row_payloads = [_evaluator_row_identity(row) for row in rows]
    row_payloads.sort(key=lambda row: json.dumps(
        row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    payload = {"topology_ids": sorted_ids, "rows": row_payloads}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths(source: str | Path) -> tuple[Path, Path]:
    source = Path(source).expanduser().resolve()
    if source.name == "frozen.pt":
        raise ValueError("warm start must use a live checkpoint.pt, never frozen.pt")
    if source.is_dir():
        checkpoint = source / "checkpoint.pt"
        root = source
    else:
        checkpoint = source
        root = checkpoint.parent
    if checkpoint.name != "checkpoint.pt" or not checkpoint.is_file():
        raise ValueError("warm start source must name an existing checkpoint.pt")
    return root, checkpoint


def _config_domain(config: Any) -> str:
    """Return the configured cooperative domain, keeping old pattern configs valid."""
    return getattr(config, "domain", "pattern")


def _roles(config: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    train = tuple(getattr(config, "train_patterns", ()))
    test = getattr(config, "test_pattern", None)
    domain = _config_domain(config)
    if domain == "deepsets":
        expected = tuple(str(index) for index in range(len(train)))
        if len(train) < 2 or train != expected:
            raise ValueError("warm-start DeepSets configuration has invalid training-role order; expected 0 through N-1")
        if not isinstance(test, str) or not test:
            raise ValueError("warm-start configuration needs a held-out test role")
        return train, (test,)
    if len(train) < 2 or not all(isinstance(value, str) and value for value in train):
        raise ValueError("warm-start configuration needs at least two ordered training roles")
    tests = tuple(getattr(config, "test_patterns", ())) or (test,)
    if not tests or not all(isinstance(value, str) and value for value in tests):
        raise ValueError("warm-start configuration needs ordered held-out test roles")
    if not isinstance(test, str) or not test or test != tests[0]:
        raise ValueError("warm-start scalar test role must match the first ordered test role")
    return train, tests


def _require_config_match(source_config: dict[str, Any], config: Any) -> None:
    domain = _config_domain(config)
    if domain not in ("pattern", "deepsets"):
        raise ValueError(f"unsupported warm-start domain: {domain!r}")
    source_domain = source_config.get("domain", "pattern")
    if source_domain != domain:
        raise ValueError("warm-start domain differs from requested configuration")

    for name in _ARCHITECTURE_FIELDS:
        if name not in source_config or not hasattr(config, name):
            raise ValueError(f"warm-start critic architecture lacks {name}")
        if source_config[name] != getattr(config, name):
            raise ValueError(f"warm-start critic {name} differs from requested configuration")

    for name in ("k", "features", "hidden"):
        if hasattr(config, name):
            if name not in source_config:
                # Original pattern run specs predate explicit dimensions. Its
                # bank contract was fixed at 11-by-8, which the config now
                # records directly.
                inferred = {"features": 11, "hidden": 8}.get(name)
                if domain == "pattern" and inferred is not None and getattr(config, name) == inferred:
                    continue
                raise ValueError(f"warm-start configuration lacks {name}")
            if source_config[name] != getattr(config, name):
                raise ValueError(f"warm-start {name} differs from requested configuration")

    requested_train, requested_tests = _roles(config)
    source_train = tuple(source_config.get("train_patterns", ()))
    if source_train != requested_train:
        raise ValueError("warm-start training-role order differs from requested configuration")
    if domain == "pattern":
        source_tests = tuple(source_config.get("test_patterns", ())) or (source_config.get("test_pattern"),)
        if source_tests != requested_tests:
            raise ValueError("warm-start ordered test roles differ from requested configuration")
        if source_config.get("test_pattern") != requested_tests[0]:
            raise ValueError("warm-start test role differs from requested configuration")
    elif source_config.get("test_pattern") != requested_tests[0]:
        raise ValueError("warm-start test role differs from requested configuration")
    if domain == "deepsets":
        requested_test_count = getattr(config, "test_task_count", 2)
        source_test_count = source_config.get("test_task_count", 2)
        if (not isinstance(requested_test_count, int) or requested_test_count < 1 or
                source_test_count != requested_test_count):
            raise ValueError("warm-start held-out task count differs from requested configuration")
    if source_config.get("seed") != getattr(config, "seed", None):
        raise ValueError("warm-start search seed differs from requested configuration")


def _validate_banks(banks: Any, config: Any) -> None:
    patterns, _ = _roles(config)
    domain = _config_domain(config)
    if not isinstance(banks, dict) or tuple(banks) != patterns:
        raise ValueError("warm-start banks do not match ordered training roles")
    features = getattr(config, "features", 11 if domain == "pattern" else None)
    hidden = getattr(config, "hidden", 8 if domain == "pattern" else None)
    k = getattr(config, "k", None)
    for pattern, bank in banks.items():
        provenance = getattr(bank, "provenance", None)
        if not isinstance(provenance, dict):
            raise ValueError("warm-start bank lacks role provenance")
        if provenance.get("domain", domain) != domain:
            raise ValueError("warm-start bank domain provenance is corrupt")
        if provenance.get("pattern") != pattern:
            raise ValueError("warm-start bank training-role provenance is corrupt")
        if provenance.get("family") not in {domain, f"cooperative_{domain}"}:
            raise ValueError("warm-start bank family provenance is corrupt")
        masks = getattr(bank, "masks", None)
        if (features is not None and hidden is not None and
                (masks is None or masks.ndim < 2 or tuple(masks.shape[-2:]) != (features, hidden))):
            raise ValueError("warm-start bank dimensions differ from requested features/hidden")
        if features is not None and hidden is not None and k is not None:
            baseline = getattr(bank, "baseline_mask", None)
            if baseline is None:
                raise ValueError("warm-start bank lacks its configured baseline mask")
            baseline = torch.as_tensor(baseline)
            if (tuple(baseline.shape) != (features, hidden) or
                    not torch.isfinite(baseline).all() or
                    not ((baseline == 0) | (baseline == 1)).all() or
                    int(baseline.sum()) != k):
                raise ValueError("warm-start bank baseline differs from requested k/features/hidden")


def _validate_task_id_context(task: TaskData, task_id: int, width: int) -> None:
    expected_support_context = support_context(task.x_support.mean(1), task.y_support).to(task.context.device)
    expected_identity = torch.nn.functional.one_hot(torch.tensor(task_id), num_classes=width).to(
        device=task.context.device, dtype=task.context.dtype)
    provenance = task.provenance
    if (provenance.get("evaluator_task_id") != task_id or
            provenance.get("task_id_encoding") != "one_hot" or
            provenance.get("task_id_width") != width or
            task.context.shape != (expected_support_context.numel() + width,) or
            not torch.equal(task.context[:-width], expected_support_context) or
            not torch.equal(task.context[-width:], expected_identity)):
        raise ValueError("warm-start DeepSets context task ID is inconsistent with provenance")


def _validate_inputs(inputs: dict[str, Any], config: Any) -> tuple[dict, list[TaskData], list[TaskData], dict]:
    required = ("banks", "train_tasks", "selection_tasks", "test_spec")
    if not isinstance(inputs, dict) or any(name not in inputs for name in required):
        raise ValueError("warm-start inputs.pt is incomplete")
    if set(inputs) != set(required):
        raise ValueError("warm-start inputs must contain only banks, train/selection tasks and test_spec")
    banks = inputs["banks"]
    train_tasks = inputs["train_tasks"]
    selection_tasks = inputs["selection_tasks"]
    test_spec = inputs["test_spec"]
    patterns, test_patterns = _roles(config)
    test_pattern = test_patterns[0]
    domain = _config_domain(config)
    task_id_width = len(patterns) + getattr(config, "test_task_count", 2) if domain == "deepsets" else None
    _validate_banks(banks, config)
    if len(train_tasks) != len(patterns) or len(selection_tasks) != len(patterns):
        raise ValueError("warm-start task count does not match ordered training roles")
    task_prefix = "pattern" if domain == "pattern" else "deepsets"
    for index, (pattern, train, selection) in enumerate(zip(patterns, train_tasks, selection_tasks)):
        if not isinstance(train, TaskData) or not isinstance(selection, TaskData):
            raise ValueError("warm-start task is not TaskData")
        if (train.task_id != f"{task_prefix}:{pattern}" or
                selection.task_id != f"{task_prefix}:{pattern}:selection"):
            raise ValueError("warm-start task order or domain role identity is corrupt")
        if train.split != "train" or selection.split != "validation":
            raise ValueError("warm-start task partitions are corrupt")
        if (train.provenance.get("domain", domain) != domain or
                selection.provenance.get("domain", domain) != domain):
            raise ValueError("warm-start task domain provenance is corrupt")
        accepted_families = {domain, f"cooperative_{domain}"}
        if (train.provenance.get("family") not in accepted_families or
                selection.provenance.get("family") not in accepted_families):
            raise ValueError("warm-start task family provenance is corrupt")
        if selection.provenance.get("role", "selection") != "selection":
            raise ValueError("warm-start selection task role is corrupt")
        if (not torch.equal(train.support_ids, selection.support_ids) or
                not torch.equal(train.x_support, selection.x_support) or
                not torch.equal(train.y_support, selection.y_support) or
                not torch.equal(train.context, selection.context)):
            raise ValueError("warm-start selection must reuse its training support/context")
        if torch.isin(train.query_ids.cpu(), selection.query_ids.cpu()).any():
            raise ValueError("warm-start training and selection queries overlap")
        if domain == "deepsets":
            arrays = (train.x_support, train.x_query, selection.x_support, selection.x_query)
            if any(value.ndim != 3 or value.shape[1] != 5 or
                   value.shape[-1] != getattr(config, "features", None) for value in arrays):
                raise ValueError("warm-start DeepSets task inputs have invalid set features")
            if test_spec.get("task_id_encoding") == "one_hot":
                if test_spec.get("task_id_width") != task_id_width:
                    raise ValueError("warm-start test specification has a different task ID width")
                _validate_task_id_context(train, index, task_id_width)
                _validate_task_id_context(selection, index, task_id_width)
            elif (test_spec.get("task_id_encoding") is not None or len(patterns) != 2 or
                  getattr(config, "test_task_count", 2) != 2 or "test_task_count" in test_spec):
                raise ValueError("warm-start DeepSets tasks require one-hot IDs for nonlegacy configurations")
    family = "cooperative_pattern" if domain == "pattern" else "cooperative_deepsets"
    if not isinstance(test_spec, dict) or test_spec.get("family") != family:
        raise ValueError("warm-start test specification is corrupt")
    if test_spec.get("materialized") is not False:
        raise ValueError("warm-start must use an unmaterialized test specification")
    if tuple(test_spec.get("train_patterns", ())) != patterns:
        raise ValueError("warm-start test specification has a different training-role order")
    if test_spec.get("test_pattern") != test_pattern:
        raise ValueError("warm-start test specification has a different test role")
    if domain == "pattern":
        source_tests = tuple(test_spec.get("test_patterns", ())) or (test_spec.get("test_pattern"),)
        if source_tests != test_patterns:
            raise ValueError("warm-start test specification has a different ordered test-role list")
        _validate_pattern_test_pools(test_spec, patterns, test_patterns)
    if domain == "deepsets":
        test_count = getattr(config, "test_task_count", 2)
        if (not isinstance(test_count, int) or test_count < 1 or
                test_spec.get("test_task_count", len(test_spec.get("costs", ()))) != test_count or
                len(test_spec.get("costs", ())) != test_count or
                len(test_spec.get("test_support_pools", ())) != test_count or
                len(test_spec.get("test_query_pools", ())) != test_count):
            raise ValueError("warm-start test specification has a different held-out task count")
    return banks, train_tasks, selection_tasks, test_spec


def _validate_pattern_test_pools(test_spec: dict[str, Any], train_patterns: tuple[str, ...],
                                 test_patterns: tuple[str, ...]) -> None:
    """Bind a composite pattern holdout spec to each ordered child pool."""
    children = test_spec.get("test_specs")
    if children is None:
        if len(test_patterns) != 1:
            raise ValueError("warm-start composite test specification lacks child roles")
        return  # Legacy single-pattern specs predate explicit child metadata.
    if not isinstance(children, (list, tuple)) or len(children) != len(test_patterns):
        raise ValueError("warm-start test child count differs from ordered test roles")

    support_count, query_count = test_spec.get("support_count"), test_spec.get("query_count")
    if (not isinstance(support_count, int) or support_count < 1 or
            not isinstance(query_count, int) or query_count < 1):
        raise ValueError("warm-start composite test counts are corrupt")

    def ids_for(spec: dict[str, Any], name: str) -> set[int]:
        try:
            values = torch.as_tensor(spec[name], dtype=torch.long)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("warm-start test observation pools are corrupt") from error
        if values.ndim != 1 or len(torch.unique(values)) != len(values):
            raise ValueError("warm-start test observation pools must be unique vectors")
        return set(values.tolist())

    all_ids, all_support_ids, all_query_ids = set(), set(), set()
    parent_seed = test_spec.get("seed")
    for pattern, child in zip(test_patterns, children):
        if (not isinstance(child, dict) or child.get("family") != "cooperative_pattern" or
                child.get("test_pattern") != pattern or child.get("materialized") is not False or
                tuple(child.get("train_patterns", ())) != train_patterns or
                child.get("support_count") != support_count or child.get("query_count") != query_count or
                (parent_seed is not None and child.get("seed") != parent_seed)):
            raise ValueError("warm-start test child metadata does not match ordered roles")
        child_ids = ids_for(child, "test_ids")
        child_support_ids = ids_for(child, "test_support_ids")
        child_query_ids = ids_for(child, "test_query_ids")
        if (not child_support_ids.isdisjoint(child_query_ids) or
                child_support_ids | child_query_ids != child_ids or
                support_count > len(child_support_ids) or query_count > len(child_query_ids)):
            raise ValueError("warm-start test child support/query pools are corrupt")
        all_ids.update(child_ids)
        all_support_ids.update(child_support_ids)
        all_query_ids.update(child_query_ids)

    if (ids_for(test_spec, "test_ids") != all_ids or
            ids_for(test_spec, "test_support_ids") != all_support_ids or
            ids_for(test_spec, "test_query_ids") != all_query_ids or
            not all_support_ids.isdisjoint(all_query_ids) or
            all_support_ids | all_query_ids != all_ids or
            support_count > len(all_support_ids) or query_count > len(all_query_ids)):
        raise ValueError("warm-start test parent pools differ from ordered child pools")


def _validate_replay(replay: RealReplay, protocol: InnerProtocol,
                     train_tasks: list[TaskData], selection_tasks: list[TaskData], root: Path) -> None:
    if not isinstance(replay, RealReplay):
        raise ValueError("warm-start checkpoint lacks a RealReplay")
    if replay.protocol.fingerprint != protocol.fingerprint:
        raise ValueError("warm-start replay protocol differs from source's actual protocol")
    allowed = {task.task_id: task for task in [*train_tasks, *selection_tasks]}
    children = (root / "children").resolve()
    for row in replay.records:
        # A final-test row is forbidden even if a malicious split field tries
        # to disguise it as train data.
        if row.get("task_split") == "test" or row.get("split") == "test":
            raise ValueError("warm-start replay contains final-test measurements")
        task = allowed.get(row.get("task_id"))
        if task is None:
            raise ValueError("warm-start replay refers to a task outside source inputs")
        if row.get("task_split") != task.split or row.get("task_fingerprint") != task.fingerprint:
            raise ValueError("warm-start replay task fingerprint or split differs from source inputs")
        artifact = Path(row.get("artifact_path", "")).resolve()
        if artifact.parent != children:
            raise ValueError("warm-start replay artifact is outside source children")
    # Validates every real artifact, its fixed-horizon provenance, masks and
    # stored losses before any state is returned to the caller.
    replay.validate()


def _validated_evaluator_metadata(source_spec: dict, checkpoint: dict,
                                  replay: RealReplay, train_tasks: list[TaskData],
                                  protocol: InnerProtocol
                                  ) -> tuple[str | None, str | None, tuple[str, ...], list[dict[str, Any]]]:
    """Authorize critic reuse only for a valid immutable initial-bank manifest."""
    saved_policy = checkpoint.get("evaluator_policy")
    spec_policy = source_spec.get("evaluator_policy")
    if saved_policy is not None and spec_policy is not None and saved_policy != spec_policy:
        raise ValueError("warm-start evaluator policy differs between checkpoint and run spec")
    policy = saved_policy if saved_policy is not None else spec_policy
    if policy is None:
        # Older online-evaluator runs have no immutable-bank provenance, so
        # their critic state must be trained afresh by the caller.
        return None, None, (), []
    if policy not in _SUPPORTED_EVALUATOR_POLICIES:
        raise ValueError("warm-start evaluator policy is unsupported")

    def metadata_value(name: str):
        value = checkpoint.get(name)
        return source_spec.get(name) if value is None else value

    fingerprint = metadata_value("evaluator_bank_fingerprint")
    topology_ids = metadata_value("evaluator_bank_topology_ids")
    rows = metadata_value("evaluator_bank_rows")
    if (not isinstance(fingerprint, str) or not fingerprint or
            not isinstance(topology_ids, (list, tuple)) or not topology_ids or
            not isinstance(rows, (list, tuple)) or not rows):
        raise ValueError("warm-start initial-bank evaluator metadata is incomplete")
    identities = tuple(topology_ids)
    if (any(not isinstance(identity, str) or not identity for identity in identities) or
            identities != tuple(sorted(set(identities)))):
        raise ValueError("warm-start evaluator topology IDs are invalid or unsorted")
    manifest_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    if len(manifest_rows) != len(rows):
        raise ValueError("warm-start evaluator rows must be mappings")
    if evaluator_bank_fingerprint(identities, manifest_rows) != fingerprint:
        raise ValueError("warm-start evaluator bank fingerprint is corrupt")

    train_by_id = {task.task_id: task for task in train_tasks if task.split == "train"}
    topology_set = set(identities)
    requested_rows: Counter[str] = Counter()
    metadata_topologies = set()
    has_train_label = False
    for row in manifest_rows:
        row_identity = _evaluator_row_identity(row)
        if row_identity["topology_id"] not in topology_set:
            raise ValueError("warm-start evaluator row is outside its topology manifest")
        task = train_by_id.get(row_identity["task_id"])
        if (task is None or row_identity["task_split"] != "train" or
                row_identity["task_fingerprint"] != task.fingerprint):
            raise ValueError("warm-start evaluator row refers to a different task identity")
        if row_identity["protocol_id"] != protocol.fingerprint:
            raise ValueError("warm-start evaluator row protocol differs from source replay")
        if row_identity["label_source"] != "fresh_terminal_query":
            raise ValueError("warm-start evaluator row lacks a real terminal-query label")
        metadata_topologies.add(row_identity["topology_id"])
        has_train_label |= row_identity["split"] == "train"
        requested_rows[_evaluator_row_key(row_identity)] += 1
    if metadata_topologies != topology_set or not has_train_label:
        raise ValueError("warm-start evaluator rows do not cover the saved initial-bank topologies")

    replay_rows: Counter[str] = Counter()
    replay_topologies = set()
    for row in replay.records:
        if (row.get("topology_id") in topology_set and row.get("task_id") in train_by_id and
                row.get("task_split") == "train"):
            replay_topologies.add(row["topology_id"])
            replay_rows[_evaluator_row_key(row)] += 1
    if not topology_set.issubset(replay_topologies):
        raise ValueError("warm-start evaluator topology manifest is outside source replay")
    if any(count > replay_rows[key] for key, count in requested_rows.items()):
        raise ValueError("warm-start evaluator rows are not a subset of source replay")
    return policy, fingerprint, identities, manifest_rows


@dataclass
class CooperativeWarmStart:
    """Imported state for a new cooperative run, with an explicit artifact copy."""

    banks: dict
    train_tasks: list[TaskData]
    selection_tasks: list[TaskData]
    test_spec: dict
    protocol: InnerProtocol
    replay: RealReplay
    ensemble_state: dict
    evaluator_training_state: dict
    best_mask: torch.Tensor
    provenance: dict[str, Any]
    evaluator_policy: str | None = None
    evaluator_bank_fingerprint: str | None = None
    evaluator_bank_topology_ids: tuple[str, ...] = ()
    evaluator_bank_rows: list[dict[str, Any]] = field(default_factory=list)
    source_bank_topology_ids: tuple[str, ...] = ()
    source_bank_masks: tuple[torch.Tensor, ...] = ()

    @property
    def reuse_evaluator(self) -> bool:
        """True only when the loader validated a versioned immutable bank."""
        return (self.evaluator_policy in _SUPPORTED_EVALUATOR_POLICIES and
                bool(self.evaluator_bank_fingerprint) and
                bool(self.evaluator_bank_topology_ids) and
                bool(self.evaluator_bank_rows))

    def materialize_replay(self, destination: str | Path) -> RealReplay:
        """Copy only replay-referenced source children and retarget this replay.

        Checkpoint records retain cache-relevant topology and task digests; only
        their path changes.  The source run is never modified.
        """
        destination = Path(destination).resolve()
        children = destination / "children"
        children.mkdir(parents=True, exist_ok=True)
        replay = deepcopy(self.replay)
        copied: dict[Path, Path] = {}
        for row in replay.records:
            source = Path(row["artifact_path"]).resolve()
            target = children / source.name
            if source not in copied:
                if target.exists():
                    if not target.is_file() or _sha256(target) != _sha256(source):
                        raise ValueError(f"destination child conflicts with imported artifact: {target}")
                else:
                    shutil.copy2(source, target)
                copied[source] = target
            row["artifact_path"] = str(copied[source])
        replay.validate()
        return replay


def load_cooperative_warm_start(source: str | Path, config: Any,
                                protocol: InnerProtocol) -> CooperativeWarmStart:
    """Load a critic checkpoint after binding it to the requested new run.

    ``source`` may be the completed output directory or its ``checkpoint.pt``.
    The caller's requested protocol must match the source run request. Replay
    validation uses the source's saved, actual protocol, which can differ when
    dense selection tuned the solver before measurements were collected.
    """
    if not isinstance(protocol, InnerProtocol):
        raise TypeError("protocol must be InnerProtocol")
    root, checkpoint_path = _source_paths(source)
    run_spec_path, inputs_path = root / "run_spec.json", root / "inputs.pt"
    if not run_spec_path.is_file() or not inputs_path.is_file():
        raise ValueError("warm-start source needs run_spec.json and inputs.pt")
    source_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    if not isinstance(source_spec, dict):
        raise ValueError("warm-start source run specification is corrupt")
    source_config = source_spec.get("config")
    if not isinstance(source_config, dict):
        raise ValueError("warm-start source run specification lacks configuration")
    _require_config_match(source_config, config)
    try:
        requested_source_protocol = InnerProtocol(**source_spec["requested_protocol"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("warm-start source run specification has an invalid protocol") from error
    if requested_source_protocol.fingerprint != protocol.fingerprint:
        raise ValueError("warm-start requested protocol differs from source request")

    actual_protocol = requested_source_protocol
    protocol_path = root / "protocol.json"
    if protocol_path.is_file():
        try:
            saved_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            if not isinstance(saved_protocol, dict):
                raise ValueError("protocol must be a JSON object")
            actual_protocol = InnerProtocol(**saved_protocol["inner_protocol"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("warm-start source has an invalid saved protocol") from error
        if saved_protocol.get("protocol_id", actual_protocol.fingerprint) != actual_protocol.fingerprint:
            raise ValueError("warm-start source protocol fingerprint is corrupt")
        requested_fields, actual_fields = (asdict(requested_source_protocol),
                                            asdict(actual_protocol))
        requested_fields.pop("lr")
        actual_fields.pop("lr")
        if requested_fields != actual_fields:
            raise ValueError("warm-start actual protocol may differ from requested protocol only in lr")

    inputs = torch.load(inputs_path, map_location="cpu", weights_only=False)
    source_input_banks, train_tasks, selection_tasks, test_spec = _validate_inputs(inputs, config)
    source_masks_by_topology = {}
    for bank in source_input_banks.values():
        for mask in bank.masks:
            cpu_mask = torch.as_tensor(mask).detach().cpu().float().clone()
            source_masks_by_topology.setdefault(topology_id(cpu_mask), cpu_mask)
    source_bank_topology_ids = tuple(sorted(source_masks_by_topology))
    if not source_bank_topology_ids:
        raise ValueError("warm-start inputs contain no original functional bank masks")
    source_bank_masks = tuple(source_masks_by_topology[identity]
                              for identity in source_bank_topology_ids)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (not isinstance(checkpoint, dict) or
            any(name not in checkpoint for name in (*_REQUIRED_CHECKPOINT_FIELDS, "best_mask"))):
        raise ValueError("warm-start checkpoint is incomplete")
    if not isinstance(checkpoint["epoch"], int) or checkpoint["epoch"] < 0:
        raise ValueError("warm-start checkpoint epoch is invalid")

    replay = deepcopy(checkpoint["replay"])
    _validate_replay(replay, actual_protocol, train_tasks, selection_tasks, root)
    (evaluator_policy, evaluator_fingerprint, evaluator_topology_ids,
     evaluator_rows) = _validated_evaluator_metadata(
         source_spec, checkpoint, replay, train_tasks, actual_protocol)
    if getattr(config, "seed", replay.split_seed) != replay.split_seed:
        raise ValueError("warm-start search seed must retain the source replay partition")
    banks = deepcopy(checkpoint["banks"])
    _validate_banks(banks, config)

    counts = {split: sum(row["split"] == split for row in replay.records)
              for split in sorted({row["split"] for row in replay.records})}
    provenance = {
        "kind": "cooperative_warm_start",
        "source": str(root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "source_epoch": checkpoint["epoch"],
        "train_counts": counts,
        "requested_protocol_id": protocol.fingerprint,
        "protocol_id": actual_protocol.fingerprint,
    }
    return CooperativeWarmStart(banks=banks, train_tasks=deepcopy(train_tasks),
                                selection_tasks=deepcopy(selection_tasks), test_spec=deepcopy(test_spec),
                                protocol=actual_protocol, replay=replay,
                                ensemble_state=deepcopy(checkpoint["ensemble"]),
                                evaluator_training_state=deepcopy(checkpoint["evaluator_training_state"]),
                                best_mask=torch.as_tensor(checkpoint["best_mask"]).detach().cpu().clone(),
                                provenance=provenance,
                                evaluator_policy=evaluator_policy,
                                evaluator_bank_fingerprint=evaluator_fingerprint,
                                evaluator_bank_topology_ids=evaluator_topology_ids,
                                evaluator_bank_rows=evaluator_rows,
                                source_bank_topology_ids=source_bank_topology_ids,
                                source_bank_masks=source_bank_masks)
