"""Compact, pickle-free persistence for prepared functional banks.

Each bank is stored under a deterministic ``bank-<sha256(name)>`` directory.
The JSON manifest is the commit marker: it is removed before replacing assets
and atomically installed only after every referenced file is complete.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from generator_evaluator.data.adapters import FunctionalBank


_SCHEMA = "generator_evaluator.binary_functional_banks:v1"
_SIGNED_KEYS = ("q_signed_mean", "q_signed")
_ABS_KEYS = ("q_abs_mean", "q_abs")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    """Convert small metadata values, including tensors, to strict JSON."""
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            raise TypeError("object arrays are not valid bank metadata")
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("bank metadata cannot contain non-finite numbers")
        return value
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return _json_value(enum_value)
    raise TypeError(f"unsupported bank metadata value: {type(value).__name__}")


def _array(value: Any, *, float32: bool = False) -> np.ndarray:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu()
        if float32:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.contiguous().numpy()
    result = np.asarray(value)
    if result.dtype.kind == "O":
        raise TypeError("object arrays cannot be saved in a functional bank")
    if float32:
        result = np.asarray(result, dtype=np.float32)
    return result


def _write_array(root: Path, relative: str, value: Any, *, float32: bool = False) -> dict[str, Any]:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = _array(value, float32=float32)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                                   dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"file": relative, "sha256": _sha256(destination),
            "shape": list(array.shape), "dtype": array.dtype.str}


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return None
    if tensor.numel() != 1:
        return None
    result = float(tensor[0])
    return result if math.isfinite(result) and result > 0.0 else None


def _state_map(state: Any, names: tuple[str, ...]) -> Any | None:
    if isinstance(state, Mapping):
        for name in names:
            if name in state:
                return state[name]
    return None


def _infer_scale(raw: Any | None, normalized: Tensor) -> float:
    """Infer a missing scale from the duplicated raw map when possible."""
    if raw is None:
        return 1.0
    try:
        raw_tensor = torch.as_tensor(raw, dtype=torch.float32).detach().cpu()
    except (TypeError, ValueError, RuntimeError):
        return 1.0
    if raw_tensor.shape != normalized.shape:
        if raw_tensor.ndim == normalized.ndim and raw_tensor.transpose(-1, -2).shape == normalized.shape:
            raw_tensor = raw_tensor.transpose(-1, -2)
        else:
            return 1.0
    raw64, normalized64 = raw_tensor.double(), normalized.double()
    nonzero = normalized64 != 0
    if not bool(nonzero.any()):
        return 1.0
    ratios = raw64[nonzero] / normalized64[nonzero]
    scale = float(ratios.median())
    if not math.isfinite(scale) or scale <= 0.0:
        return 1.0
    tolerance = max(1e-7, float(raw64.abs().max()) * 2e-6)
    if torch.allclose(raw64, normalized64 * scale, rtol=2e-5, atol=tolerance):
        return scale
    return 1.0


def _row_metadata(bank: FunctionalBank, row: int, token: Tensor, probe_width: int,
                  features: int) -> dict[str, Any]:
    state = bank.states[row] if row < len(bank.states) else {}
    psi_raw = _state_map(state, ("psi",))
    q_raw = _state_map(state, ("q_rms", "q_abs_mean", "q_abs", "q_signed_mean", "q_signed"))
    psi_norm = token[:, :probe_width].transpose(0, 1)
    q_rms_norm = token[:, probe_width + 2 * features:probe_width + 3 * features].transpose(0, 1)
    psi_scale = _scalar(state.get("psi_scale")) if isinstance(state, Mapping) else None
    q_scale = _scalar(state.get("q_scale")) if isinstance(state, Mapping) else None
    if psi_scale is None:
        psi_scale = _infer_scale(psi_raw, psi_norm)
    if q_scale is None:
        q_scale = _infer_scale(q_raw, q_rms_norm)

    source = state.get("source") if isinstance(state, Mapping) else None
    row_hash = state.get("row_hash") if isinstance(state, Mapping) else None
    signed_name = next((name for name in _SIGNED_KEYS if isinstance(state, Mapping) and name in state),
                       "q_signed_mean")
    abs_name = next((name for name in _ABS_KEYS if isinstance(state, Mapping) and name in state),
                    "q_abs_mean")
    has_maps = ((psi_raw is not None and q_raw is not None) or
                (isinstance(state, Mapping) and
                 _scalar(state.get("psi_scale")) is not None and
                 _scalar(state.get("q_scale")) is not None))
    result: dict[str, Any] = {"psi_scale": float(psi_scale), "q_scale": float(q_scale),
                              "has_raw_maps": bool(has_maps),
                              "map_keys": {"signed": signed_name, "absolute": abs_name}}
    if source is not None:
        result["source"] = _json_value(source)
    if row_hash is not None:
        result["row_hash"] = _json_value(row_hash)
    return result


def save_banks(path: Path, banks: dict[str, FunctionalBank]) -> None:
    """Write functional banks as float32 NPY assets plus an atomic JSON manifest."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    # A failed overwrite must never leave an older manifest describing new assets.
    manifest_path.unlink(missing_ok=True)

    entries: dict[str, Any] = {}
    for name, bank in banks.items():
        if not isinstance(name, str) or not isinstance(bank, FunctionalBank):
            raise TypeError("banks must map string names to FunctionalBank instances")
        rows, features, hidden = bank.masks.shape
        if bank.tokens.shape[1] != rows:
            raise ValueError(f"bank {name!r} token and mask row counts differ")
        probe_width = bank.tokens.shape[-1] - 4 * features
        if probe_width < 1:
            raise ValueError(f"bank {name!r} tokens do not contain the expected functional channels")
        if bank.states and len(bank.states) != rows:
            raise ValueError(f"bank {name!r} must have one state record per token row")

        directory = f"bank-{hashlib.sha256(name.encode('utf-8')).hexdigest()}"
        arrays: dict[str, Any] = {
            "tokens": _write_array(root, f"{directory}/tokens.npy", bank.tokens, float32=True),
            "baseline_mask": _write_array(root, f"{directory}/baseline_mask.npy",
                                           bank.baseline_mask, float32=True),
        }
        token_masks = bank.tokens[0, :, :, -features:].transpose(1, 2).contiguous()
        masks_from_tokens = torch.equal(token_masks, bank.masks)
        if masks_from_tokens:
            arrays["masks_from_tokens"] = True
        else:
            arrays["masks"] = _write_array(root, f"{directory}/masks.npy", bank.masks, float32=True)
        if bank.quality is not None:
            arrays["quality"] = _write_array(root, f"{directory}/quality.npy", bank.quality,
                                              float32=True)

        diagnostic_entries: dict[str, Any] = {}
        for index, (key, value) in enumerate(bank.diagnostics.items()):
            if isinstance(value, (Tensor, np.ndarray, np.generic)):
                filename = ("probe_x.npy" if key == "probe_x" else
                            "aligned_q_abs.npy" if key == "aligned_q_abs" else
                            f"diagnostics-{index:04d}.npy")
                diagnostic_entries[str(key)] = {"array": _write_array(
                    root, f"{directory}/{filename}", value)}
            else:
                diagnostic_entries[str(key)] = {"value": _json_value(value)}

        states = [_row_metadata(bank, row, bank.tokens[0, row], probe_width, features)
                  for row in range(rows)]
        entries[name] = {
            "arrays": arrays,
            "states": states,
            "provenance": _json_value(bank.provenance),
            "diagnostics": diagnostic_entries,
        }

    manifest = {"schema": _SCHEMA, "bank_order": list(banks), "banks": entries}
    descriptor, temporary_name = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _asset_path(root: Path, record: Any) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping) or not isinstance(record.get("file"), str):
        raise ValueError("invalid functional-bank array record")
    relative = Path(record["file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("functional-bank asset path escapes its directory")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("functional-bank asset path escapes its directory")
    if not path.is_file() or _sha256(path) != record.get("sha256"):
        raise ValueError(f"missing or incomplete functional-bank asset: {relative}")
    return path, dict(record)


def _read_array(root: Path, record: Any) -> np.ndarray:
    path, metadata = _asset_path(root, record)
    try:
        result = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid functional-bank array: {metadata['file']}") from error
    if (not isinstance(result, np.ndarray) or list(result.shape) != metadata.get("shape") or
            result.dtype.str != metadata.get("dtype")):
        raise ValueError(f"functional-bank array metadata mismatch: {metadata['file']}")
    return result


def _tensor(root: Path, record: Any) -> Tensor:
    array = _read_array(root, record)
    if array.dtype != np.dtype(np.float32):
        raise ValueError("functional-bank core arrays must be float32")
    return torch.from_numpy(array)


def _rebuild_states(tokens: Tensor, metadata: Any, features: int) -> list[dict[str, Any]]:
    rows, hidden = tokens.shape[1:3]
    probe_width = tokens.shape[-1] - 4 * features
    if (not isinstance(metadata, list) or len(metadata) != rows or probe_width < 1):
        raise ValueError("functional-bank state metadata does not match token rows")
    states: list[dict[str, Any]] = []
    for row, item in enumerate(metadata):
        if not isinstance(item, Mapping):
            raise ValueError("invalid functional-bank row metadata")
        state: dict[str, Any] = {}
        if "source" in item:
            source = item["source"]
            if isinstance(source, Mapping) and "source_mask" in source:
                source = dict(source)
                source_mask = source["source_mask"]
                if not isinstance(source_mask, Tensor):
                    source["source_mask"] = torch.as_tensor(source_mask, dtype=torch.float32).cpu().contiguous()
            state["source"] = source
        if "row_hash" in item:
            state["row_hash"] = item["row_hash"]
        if item.get("has_raw_maps"):
            psi_scale, q_scale = float(item["psi_scale"]), float(item["q_scale"])
            if (not math.isfinite(psi_scale) or psi_scale <= 0.0 or
                    not math.isfinite(q_scale) or q_scale <= 0.0):
                raise ValueError("invalid functional-bank row scales")
            token = tokens[0, row]
            state["psi"] = token[:, :probe_width].transpose(0, 1).contiguous() * psi_scale
            state["q_rms"] = token[:, probe_width + 2 * features:
                                   probe_width + 3 * features].transpose(0, 1).contiguous() * q_scale
            state["q_signed_mean"] = token[:, probe_width:probe_width + features].transpose(0, 1).contiguous() * q_scale
            state["q_abs_mean"] = token[:, probe_width + features:
                                       probe_width + 2 * features].transpose(0, 1).contiguous() * q_scale
            keys = item.get("map_keys", {})
            signed_name = keys.get("signed", "q_signed_mean") if isinstance(keys, Mapping) else "q_signed_mean"
            abs_name = keys.get("absolute", "q_abs_mean") if isinstance(keys, Mapping) else "q_abs_mean"
            if signed_name != "q_signed_mean":
                state[signed_name] = state.pop("q_signed_mean")
            if abs_name != "q_abs_mean":
                state[abs_name] = state.pop("q_abs_mean")
            state["psi_scale"] = torch.tensor([psi_scale], dtype=torch.float32)
            state["q_scale"] = torch.tensor([q_scale], dtype=torch.float32)
        states.append(state)
    return states


def load_banks(path: Path) -> dict[str, FunctionalBank]:
    """Load and validate the complete bank set from its manifest directory."""
    root = Path(path)
    manifest_path = root / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("functional-bank manifest is missing or incomplete") from error
    if (not isinstance(manifest, Mapping) or manifest.get("schema") != _SCHEMA or
            not isinstance(manifest.get("banks"), Mapping) or
            not isinstance(manifest.get("bank_order"), list)):
        raise ValueError("unsupported or incomplete functional-bank manifest")

    bank_order = manifest["bank_order"]
    bank_entries = manifest["banks"]
    if (any(not isinstance(name, str) for name in bank_order) or
            len(bank_order) != len(set(bank_order)) or set(bank_order) != set(bank_entries)):
        raise ValueError("functional-bank manifest order does not match its bank keys")

    result: dict[str, FunctionalBank] = {}
    for name in bank_order:
        entry = bank_entries[name]
        if not isinstance(name, str) or not isinstance(entry, Mapping):
            raise ValueError("invalid functional-bank manifest entry")
        arrays = entry.get("arrays")
        if not isinstance(arrays, Mapping):
            raise ValueError(f"bank {name!r} has incomplete array metadata")
        tokens = _tensor(root, arrays.get("tokens"))
        baseline = _tensor(root, arrays.get("baseline_mask"))
        if arrays.get("masks_from_tokens") is True:
            if tokens.ndim != 4:
                raise ValueError(f"bank {name!r} tokens have invalid dimensions")
            features = baseline.shape[0]
            if baseline.ndim != 2 or tokens.shape[-1] < 4 * features + 1:
                raise ValueError(f"bank {name!r} token mask block is invalid")
            masks = tokens[0, :, :, -features:].transpose(1, 2).contiguous()
        else:
            masks = _tensor(root, arrays.get("masks"))
            features = masks.shape[1] if masks.ndim == 3 else 0
        quality = _tensor(root, arrays["quality"]) if "quality" in arrays else None

        diagnostics: dict[str, Any] = {}
        raw_diagnostics = entry.get("diagnostics", {})
        if not isinstance(raw_diagnostics, Mapping):
            raise ValueError(f"bank {name!r} diagnostics metadata is invalid")
        for key, value in raw_diagnostics.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"bank {name!r} diagnostic {key!r} is invalid")
            if "array" in value:
                diagnostics[key] = torch.from_numpy(_read_array(root, value["array"]))
            elif "value" in value:
                diagnostics[key] = value["value"]
            else:
                raise ValueError(f"bank {name!r} diagnostic {key!r} is incomplete")

        provenance = dict(entry.get("provenance", {}))
        density_counts = provenance.get("teacher_density_counts")
        if isinstance(density_counts, Mapping):
            try:
                provenance["teacher_density_counts"] = {int(key): value
                                                          for key, value in density_counts.items()}
            except (TypeError, ValueError) as error:
                raise ValueError(f"bank {name!r} teacher density keys are invalid") from error
        try:
            bank = FunctionalBank(tokens, quality, masks, baseline, provenance,
                                  states=_rebuild_states(tokens, entry.get("states"), features),
                                  diagnostics=diagnostics)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"bank {name!r} assets do not reconstruct a valid FunctionalBank") from error
        result[name] = bank
    return result
