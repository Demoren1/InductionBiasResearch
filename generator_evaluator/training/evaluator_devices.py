"""Device assignment for independent evaluator ensemble members."""

from collections.abc import Sequence

import torch


def _canonical_device(value: str, *, label: str) -> tuple[str, str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} device must be a non-empty string")
    name = value.strip()
    try:
        device = torch.device(name)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"invalid {label} device: {name}") from error

    if device.type == "cpu":
        if device.index is not None:
            raise ValueError(f"invalid {label} device: {name}")
        return "cpu", "cpu"
    if device.type != "cuda":
        raise ValueError(f"unsupported {label} device: {name}")

    index = 0 if device.index is None else device.index
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise ValueError(f"unavailable {label} device: {name}")
    return "cuda", f"cuda:{index}"


def _repeat_devices(devices: Sequence[str], count: int) -> tuple[str, ...]:
    return tuple(devices[index % len(devices)] for index in range(count))


def select_evaluator_devices(values: Sequence[str] | None, primary: str,
                             count: int) -> tuple[str, ...]:
    """Assign evaluator members to available devices in stable order.

    ``None`` and ``("auto",)`` choose the primary CPU or rank visible CUDA
    devices by currently free memory. Explicit devices retain their first-use
    order, with duplicates removed, then cycle when there are fewer devices
    than evaluator members.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("evaluator member count must be a positive integer")

    primary_type, _ = _canonical_device(primary, label="primary")
    if values is None:
        requested = None
    elif isinstance(values, str):
        requested = (values,)
    else:
        requested = tuple(values)

    automatic = requested is None
    if requested is not None:
        if not requested:
            raise ValueError("evaluator devices must contain at least one device")
        if (len(requested) == 1 and isinstance(requested[0], str) and
                requested[0].strip().lower() == "auto"):
            automatic = True
        elif any(isinstance(value, str) and value.strip().lower() == "auto"
                 for value in requested):
            raise ValueError("auto cannot be combined with explicit evaluator devices")

    if automatic:
        if primary_type == "cpu":
            return ("cpu",) * count

        available = torch.cuda.device_count()
        ranked = []
        for index in range(available):
            try:
                free_memory, _ = torch.cuda.mem_get_info(index)
                free_memory = int(free_memory)
            except Exception:
                # Memory telemetry is optional; keep unavailable readings last.
                free_memory = -1
            ranked.append((index, free_memory))
        if not ranked:
            raise ValueError("automatic evaluator device selection found no visible CUDA devices")
        ranked.sort(key=lambda item: (-item[1], item[0]))
        best_devices = tuple(f"cuda:{index}" for index, _ in ranked[:count])
        return _repeat_devices(best_devices, count)

    devices = []
    seen = set()
    for value in requested:
        device_type, device = _canonical_device(value, label="evaluator")
        if device_type != primary_type:
            raise ValueError("evaluator devices must use the same device type as the primary device")
        if device not in seen:
            seen.add(device)
            devices.append(device)
    return _repeat_devices(devices, count)
