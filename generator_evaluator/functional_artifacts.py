"""Compact, portable cards for individual functional maps."""
from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .artifacts import save_torch


_SCHEMA = "generator_evaluator.functional_map_card"
_SCHEMA_VERSION = 1
_FORBIDDEN_METADATA_KEY_PARTS = ("optimizer", "adam", "history")
_MAX_METADATA_TENSOR_ELEMENTS = 65_536


def _compact_metadata(value: Any, path: str = "metadata") -> Any:
    """Clone small metadata values and refuse training-state payloads."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            name = str(key)
            normalized = name.lower()
            if any(part in normalized for part in _FORBIDDEN_METADATA_KEY_PARTS):
                raise ValueError(f"functional cards cannot contain optimizer or history field {path}.{name}")
            result[key] = _compact_metadata(item, f"{path}.{name}")
        return result
    if isinstance(value, list):
        return [_compact_metadata(item, f"{path}[]") for item in value]
    if isinstance(value, tuple):
        return tuple(_compact_metadata(item, f"{path}[]") for item in value)
    if isinstance(value, Tensor):
        if value.numel() > _MAX_METADATA_TENSOR_ELEMENTS:
            raise ValueError(f"functional card {path} tensor is too large for metadata")
        compact = value.detach().cpu().clone(memory_format=torch.contiguous_format)
        if compact.is_floating_point() and not bool(torch.isfinite(compact).all()):
            raise ValueError(f"functional card {path} tensor must be finite")
        return compact
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"functional card {path} number must be finite")
        return value
    raise TypeError(f"functional card {path} has unsupported metadata value {type(value).__name__}")


def _compact_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("functional card state must be a nonempty tensor mapping")
    result: dict[str, Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError("functional card state must map string names to tensors")
        compact = value.detach().to(device="cpu", dtype=torch.float32).clone(
            memory_format=torch.contiguous_format)
        if not bool(torch.isfinite(compact).all()):
            raise ValueError(f"functional card state {name!r} must be finite")
        result[name] = compact
    return result


def write_functional_card(
    path: str | Path,
    *,
    token: Tensor,
    mask: Tensor,
    state: Mapping[str, Tensor],
    metadata: Mapping[str, Any],
    diagnostics: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write one compact functional map and its terminal state.

    State names are preserved so pattern (``w/b/v/c``) and DeepSets models
    (``weight/bias/readout/per_image_offset``) use the same card schema.
    Optimizer state and training histories are deliberately excluded.
    """
    if not isinstance(token, Tensor) or token.ndim != 2 or not token.is_floating_point():
        raise ValueError("functional card token must be a floating [hidden, channels] tensor")
    if not bool(torch.isfinite(token).all()):
        raise ValueError("functional card token must be finite")
    if not isinstance(mask, Tensor) or mask.ndim != 2:
        raise ValueError("functional card mask must be a [features, hidden] tensor")
    if mask.is_floating_point() and not bool(torch.isfinite(mask).all()):
        raise ValueError("functional card mask must be finite")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("functional card mask must be binary")
    if token.shape[0] != mask.shape[1]:
        raise ValueError("functional card token and mask hidden dimensions differ")
    if not isinstance(metadata, Mapping):
        raise TypeError("functional card metadata must be a mapping")
    if diagnostics is not None and not isinstance(diagnostics, Mapping):
        raise TypeError("functional card diagnostics must be a mapping")

    payload = {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "mask": mask.detach().to(device="cpu", dtype=torch.bool).clone(
            memory_format=torch.contiguous_format),
        "token": token.detach().to(device="cpu", dtype=torch.float32).clone(
            memory_format=torch.contiguous_format),
        "state_dict": _compact_state(state),
        "metadata": _compact_metadata(metadata),
    }
    if diagnostics is not None:
        payload["diagnostics"] = _compact_metadata(diagnostics, "diagnostics")
    destination = Path(path)
    save_torch(destination, payload)
    return destination
