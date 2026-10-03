"""Content-addressed storage for :class:`FunctionalBank` snapshots.

The artifact pickle contains a small proxy and ordinary bank metadata. Teacher
rows and large diagnostic tensors live in sibling ``bank_assets`` files,
shared by every save in that run directory. The proxy hydrates to the
original ``FunctionalBank`` class during ``torch.load(..., weights_only=False)``.

Asset paths are absolute because Python's pickle restore hook is not given the
checkpoint filename. Moving an artifact therefore requires moving its asset
directory to the same absolute path, or rewriting the proxy references.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import pickle
import struct
from threading import RLock
from typing import Any

import torch
from torch import Tensor


_SCHEMA = 1
_ASSET_DIRECTORY = "bank_assets"
_LARGE_TENSOR_BYTES = 64 * 1024
_VERIFIED_ASSET_LIMIT = 4096
_VERIFIED_ASSET_STATS: OrderedDict[Path, tuple[int, int, int, int]] = OrderedDict()
_VERIFIED_ASSET_LOCK = RLock()


@dataclass(frozen=True)
class _TensorAssetReference:
    asset: str


@dataclass(frozen=True)
class _TensorRowsAssetReference:
    shape: tuple[int, ...]
    rows: tuple[str, ...]


class _FunctionalBankProxy:
    """Pickle handle that restores a normal FunctionalBank when loaded."""

    def __init__(self, manifest: dict[str, Any]):
        self._manifest = manifest

    def __reduce__(self):
        return _restore_functional_bank, (self._manifest,)


def _tensor_bytes(value: Tensor) -> bytes:
    """Return exact logical tensor bytes for common dense CPU tensor dtypes."""
    cpu = value.detach().to(device="cpu").contiguous()
    if cpu.layout != torch.strided or cpu.is_quantized:
        raise TypeError(f"unsupported tensor layout for FunctionalBank storage: {cpu.layout}")
    # Reinterpretation avoids NumPy's incomplete dtype support (notably bfloat16).
    return cpu.reshape(-1).view(torch.uint8).numpy().tobytes()


def _feed_hash(digest, value: Any) -> None:
    """Hash a nested value deterministically, including tensor type and shape."""
    def tag(name: str) -> None:
        data = name.encode("utf-8")
        digest.update(struct.pack("!Q", len(data)))
        digest.update(data)

    if value is None:
        tag("none")
    elif isinstance(value, bool):
        tag("bool")
        digest.update(b"\x01" if value else b"\x00")
    elif isinstance(value, int):
        tag("int")
        data = str(value).encode("ascii")
        digest.update(struct.pack("!Q", len(data)))
        digest.update(data)
    elif isinstance(value, float):
        tag("float")
        digest.update(struct.pack("!d", value))
    elif isinstance(value, str):
        tag("str")
        data = value.encode("utf-8")
        digest.update(struct.pack("!Q", len(data)))
        digest.update(data)
    elif isinstance(value, bytes):
        tag("bytes")
        digest.update(struct.pack("!Q", len(value)))
        digest.update(value)
    elif isinstance(value, Path):
        tag("path")
        _feed_hash(digest, str(value))
    elif torch.is_tensor(value):
        tag("tensor")
        cpu = value.detach().to(device="cpu").contiguous()
        tag(str(cpu.dtype))
        tag(str(cpu.layout))
        digest.update(struct.pack("!Q", cpu.ndim))
        for dimension in cpu.shape:
            digest.update(struct.pack("!q", int(dimension)))
        data = _tensor_bytes(cpu)
        digest.update(struct.pack("!Q", len(data)))
        digest.update(data)
    elif isinstance(value, dict):
        tag("dict")
        digest.update(struct.pack("!Q", len(value)))
        pairs = [(_stable_digest(key), key, child) for key, child in value.items()]
        for _, key, child in sorted(pairs, key=lambda row: row[0]):
            _feed_hash(digest, key)
            _feed_hash(digest, child)
    elif isinstance(value, (tuple, list)):
        tag("tuple" if isinstance(value, tuple) else "list")
        digest.update(struct.pack("!Q", len(value)))
        for child in value:
            _feed_hash(digest, child)
    elif isinstance(value, (set, frozenset)):
        tag("frozenset" if isinstance(value, frozenset) else "set")
        children = sorted(_stable_digest(child) for child in value)
        digest.update(struct.pack("!Q", len(children)))
        for child_digest in children:
            digest.update(child_digest)
    else:
        # Teacher state records are ordinary Python containers. Keep support for
        # any additional pickleable scalar metadata without silently hashing its
        # repr, which can contain process-specific addresses.
        tag("pickle:" + type(value).__module__ + "." + type(value).__qualname__)
        encoded = pickle.dumps(value, protocol=5)
        digest.update(struct.pack("!Q", len(encoded)))
        digest.update(encoded)


def _stable_digest(value: Any) -> bytes:
    digest = hashlib.sha256()
    _feed_hash(digest, value)
    return digest.digest()


def _digest_name(value: Any) -> str:
    return _stable_digest(value).hex()


def _compact_copy(value: Any) -> Any:
    """Clone nested tensors into compact CPU storage while preserving values."""
    if torch.is_tensor(value):
        return value.detach().to(device="cpu").contiguous().clone()
    if isinstance(value, dict):
        copied = value.copy()
        for key, child in value.items():
            copied[key] = _compact_copy(child)
        return copied
    if isinstance(value, list):
        return [_compact_copy(child) for child in value]
    if isinstance(value, tuple):
        children = [_compact_copy(child) for child in value]
        if hasattr(value, "_fields"):
            return type(value)(*children)
        return tuple(children)
    if isinstance(value, set):
        return {_compact_copy(child) for child in value}
    if isinstance(value, frozenset):
        return frozenset(_compact_copy(child) for child in value)
    return value


def _asset_directory(destination: Path) -> Path:
    parent = destination.expanduser().resolve().parent
    # Cooperative runs place related artifacts under bootstrap/search trees.
    # Share assets at that tree's parent so later stages reuse teacher rows.
    for directory in (parent, *parent.parents):
        if directory.name in {"bootstrap", "search"}:
            return directory.parent / _ASSET_DIRECTORY
    return parent / _ASSET_DIRECTORY


def _store_asset(asset_directory: Path, value: Any) -> str:
    """Atomically save one compact asset, reusing an existing content hash."""
    # Hash logical values before cloning. Repeated checkpoints still pay the
    # necessary content scan, but do not allocate copies of unchanged rows.
    asset_id = _digest_name(value)
    path = asset_directory / f"{asset_id}.pt"
    if not path.exists():
        compact = _compact_copy(value)
        # Keep the public artifact writer as the only atomic torch-save path.
        # The integration hook may call this module recursively; this payload
        # contains no FunctionalBank and therefore terminates immediately.
        from generator_evaluator.storage.artifacts import save_torch
        save_torch(path, compact)
        _remember_verified_asset(path)
    else:
        _verify_existing_asset(asset_directory, path, asset_id)
    return asset_id


def _asset_file_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns),
            int(stat.st_ctime_ns))


def _remember_verified_asset(path: Path) -> None:
    path = path.resolve()
    signature = _asset_file_signature(path)
    with _VERIFIED_ASSET_LOCK:
        _VERIFIED_ASSET_STATS[path] = signature
        _VERIFIED_ASSET_STATS.move_to_end(path)
        while len(_VERIFIED_ASSET_STATS) > _VERIFIED_ASSET_LIMIT:
            _VERIFIED_ASSET_STATS.popitem(last=False)


def _verify_existing_asset(asset_directory: Path, path: Path, asset_id: str) -> None:
    path = path.resolve()
    signature = _asset_file_signature(path)
    with _VERIFIED_ASSET_LOCK:
        if _VERIFIED_ASSET_STATS.get(path) == signature:
            _VERIFIED_ASSET_STATS.move_to_end(path)
            return

    _read_asset(asset_directory, asset_id)
    verified_signature = _asset_file_signature(path)
    with _VERIFIED_ASSET_LOCK:
        _VERIFIED_ASSET_STATS[path] = verified_signature
        _VERIFIED_ASSET_STATS.move_to_end(path)
        while len(_VERIFIED_ASSET_STATS) > _VERIFIED_ASSET_LIMIT:
            _VERIFIED_ASSET_STATS.popitem(last=False)


def _externalize_nested(value: Any, asset_directory: Path) -> Any:
    """Externalize large nested tensors and compact any small inline tensors."""
    if torch.is_tensor(value):
        if value.numel() * value.element_size() >= _LARGE_TENSOR_BYTES:
            return _TensorAssetReference(_store_asset(asset_directory, value))
        return _compact_copy(value)
    if isinstance(value, dict):
        copied = value.copy()
        for key, child in value.items():
            copied[key] = _externalize_nested(child, asset_directory)
        return copied
    if isinstance(value, list):
        return [_externalize_nested(child, asset_directory) for child in value]
    if isinstance(value, tuple):
        children = [_externalize_nested(child, asset_directory) for child in value]
        if hasattr(value, "_fields"):
            return type(value)(*children)
        return tuple(children)
    return _compact_copy(value)


def _externalize_aligned_maps(value: Tensor, teachers: int,
                              asset_directory: Path) -> _TensorRowsAssetReference:
    rows = [_store_asset(asset_directory, value[index]) for index in range(teachers)]
    return _TensorRowsAssetReference(tuple(value.shape), tuple(rows))


def _bank_manifest(bank, asset_directory: Path) -> dict[str, Any]:
    teacher_count = int(bank.tokens.shape[1])
    row_states = isinstance(bank.states, list) and len(bank.states) == teacher_count
    rows: list[str] = []
    state_sources: list[dict[str, Any]] | None = [] if row_states else None
    for index in range(teacher_count):
        row = {"token": bank.tokens[0, index], "mask": bank.masks[index]}
        if row_states:
            state = bank.states[index]
            if type(state) is dict and "source" in state:
                source_keys = tuple(state)
                row_state = state.copy()
                source = row_state.pop("source")
                state_sources.append({
                    "present": True,
                    "keys": source_keys,
                    "value": _externalize_nested(source, asset_directory),
                })
                row["state"] = row_state
            else:
                state_sources.append({"present": False})
                row["state"] = state
        rows.append(_store_asset(asset_directory, row))

    states = None if row_states else _externalize_nested(bank.states, asset_directory)
    diagnostics: dict[str, Any] = {}
    for key, value in bank.diagnostics.items():
        if key == "aligned_q_abs" and torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == teacher_count:
            diagnostics[key] = _externalize_aligned_maps(value, teacher_count, asset_directory)
        else:
            diagnostics[key] = _externalize_nested(value, asset_directory)

    return {
        "schema": _SCHEMA,
        "asset_directory": str(asset_directory.resolve()),
        "rows": rows,
        "states_in_rows": row_states,
        "state_sources": state_sources,
        "states": states,
        "quality": _compact_copy(bank.quality),
        "baseline_mask": _compact_copy(bank.baseline_mask),
        "provenance": _externalize_nested(bank.provenance, asset_directory),
        "diagnostics": diagnostics,
    }


def _map_payload(payload: Any, asset_directory: Path,
                 bank_cache: dict[int, _FunctionalBankProxy]) -> Any:
    from generator_evaluator.data.adapters import FunctionalBank

    if isinstance(payload, FunctionalBank):
        identity = id(payload)
        if identity not in bank_cache:
            bank_cache[identity] = _FunctionalBankProxy(_bank_manifest(payload, asset_directory))
        return bank_cache[identity]
    if isinstance(payload, dict):
        copied = payload.copy()
        for key, child in payload.items():
            copied[key] = _map_payload(child, asset_directory, bank_cache)
        return copied
    if isinstance(payload, list):
        return [_map_payload(child, asset_directory, bank_cache) for child in payload]
    if isinstance(payload, tuple):
        children = [_map_payload(child, asset_directory, bank_cache) for child in payload]
        if hasattr(payload, "_fields"):
            return type(payload)(*children)
        return tuple(children)
    return payload


def externalize_banks(payload: Any, destination: Path) -> Any:
    """Replace nested FunctionalBanks with small, transparently hydrated proxies.

    Assets are content-addressed and shared by all artifacts whose destination
    files have the same parent directory. Existing assets are never rewritten.
    The returned value is ready for ``torch.save`` and loads as the original
    payload shape with original ``FunctionalBank`` instances.
    """
    destination = Path(destination)
    asset_directory = _asset_directory(destination)
    return _map_payload(payload, asset_directory, {})


def _read_asset(asset_directory: Path, asset_id: str) -> Any:
    if (not isinstance(asset_id, str) or len(asset_id) != 64 or
            any(char not in "0123456789abcdef" for char in asset_id)):
        raise ValueError(f"invalid FunctionalBank asset reference {asset_id!r}")
    asset_directory = asset_directory.expanduser().resolve()
    path = (asset_directory / f"{asset_id}.pt").resolve()
    if path.parent != asset_directory:
        raise ValueError(f"FunctionalBank asset reference escapes its asset directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"FunctionalBank asset is missing: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
        actual = _digest_name(value)
    except Exception as error:
        raise ValueError(f"FunctionalBank asset is corrupt or unreadable: {path}") from error
    if actual != asset_id:
        raise ValueError(f"FunctionalBank asset checksum mismatch (corrupt): {path}")
    return value


def _hydrate_nested(value: Any, asset_directory: Path) -> Any:
    if isinstance(value, _TensorAssetReference):
        return _read_asset(asset_directory, value.asset)
    if isinstance(value, _TensorRowsAssetReference):
        rows = [_read_asset(asset_directory, reference) for reference in value.rows]
        try:
            tensor = torch.stack(rows)
        except (RuntimeError, TypeError) as error:
            raise ValueError("FunctionalBank diagnostic row assets cannot be stacked") from error
        if tuple(tensor.shape) != value.shape:
            raise ValueError("FunctionalBank diagnostic row assets have an unexpected shape")
        return tensor
    if isinstance(value, dict):
        copied = value.copy()
        for key, child in value.items():
            copied[key] = _hydrate_nested(child, asset_directory)
        return copied
    if isinstance(value, list):
        return [_hydrate_nested(child, asset_directory) for child in value]
    if isinstance(value, tuple):
        children = [_hydrate_nested(child, asset_directory) for child in value]
        if hasattr(value, "_fields"):
            return type(value)(*children)
        return tuple(children)
    return value


def _restore_functional_bank(manifest: dict[str, Any]):
    from generator_evaluator.data.adapters import FunctionalBank

    if manifest.get("schema") != _SCHEMA:
        raise ValueError(f"unsupported FunctionalBank storage schema: {manifest.get('schema')!r}")
    asset_directory = Path(manifest["asset_directory"])
    rows = [_read_asset(asset_directory, reference) for reference in manifest["rows"]]
    try:
        tokens = torch.stack([row["token"] for row in rows]).unsqueeze(0)
        masks = torch.stack([row["mask"] for row in rows])
    except (KeyError, RuntimeError, TypeError) as error:
        raise ValueError("FunctionalBank teacher row assets have invalid contents") from error
    if manifest["states_in_rows"]:
        try:
            states = [row["state"] for row in rows]
        except KeyError as error:
            raise ValueError("FunctionalBank teacher row asset is missing state") from error
        source_records = manifest.get("state_sources")
        if source_records is not None:
            if len(source_records) != len(states):
                raise ValueError("FunctionalBank source metadata does not match its teacher rows")
            restored_states = []
            for state, source_record in zip(states, source_records):
                if not source_record.get("present"):
                    restored_states.append(state)
                    continue
                if type(state) is not dict:
                    raise ValueError("FunctionalBank teacher row with source metadata is not a dictionary")
                source = _hydrate_nested(source_record["value"], asset_directory)
                rebuilt = {}
                try:
                    for key in source_record["keys"]:
                        rebuilt[key] = source if key == "source" else state[key]
                except KeyError as error:
                    raise ValueError("FunctionalBank teacher row fields do not match source metadata") from error
                if len(rebuilt) != len(state) + 1:
                    raise ValueError("FunctionalBank teacher row fields do not match source metadata")
                restored_states.append(rebuilt)
            states = restored_states
    else:
        states = _hydrate_nested(manifest["states"], asset_directory)
    quality = _hydrate_nested(manifest["quality"], asset_directory)
    baseline_mask = _hydrate_nested(manifest["baseline_mask"], asset_directory)
    provenance = _hydrate_nested(manifest["provenance"], asset_directory)
    diagnostics = _hydrate_nested(manifest["diagnostics"], asset_directory)
    try:
        return FunctionalBank(tokens, quality, masks, baseline_mask, provenance,
                              states=states, diagnostics=diagnostics)
    except (TypeError, ValueError) as error:
        raise ValueError("FunctionalBank assets do not reconstruct a valid bank") from error
