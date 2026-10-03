"""Compatibility entry point for the shared batched measurement store."""
from .artifacts import save_torch
from .parallel_measurements import ParallelMeasurementStore as _ParallelMeasurementStore
from .pattern_batch import fit_pattern_batch


class BatchedMeasurementStore(_ParallelMeasurementStore):
    """Preserve the cooperative entry point with injectable fit/write hooks."""

    def __init__(self, out, replay, device, *, devices=None, batch_size=128):
        selected_devices = tuple(devices) if devices else (device,)

        def fit_pattern(masks, tasks, protocol, target, **kwargs):
            if not isinstance(tasks, (list, tuple)):
                tasks = [tasks] * len(masks)
            return fit_pattern_batch(masks, tasks, protocol, target, **kwargs)

        spawned_multi_gpu = (len(selected_devices) > 1 and
                             all(str(target).startswith("cuda") for target in selected_devices))
        super().__init__(out, replay, device, devices=selected_devices, batch_size=batch_size,
                         pattern_batch_fit_fn=None if spawned_multi_gpu else fit_pattern,
                         save_torch_fn=lambda *args, **kwargs: save_torch(*args, **kwargs))
