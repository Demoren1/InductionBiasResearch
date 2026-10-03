"""Independent generator work grouped by device, without shared-model updates."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import torch


class PerDeviceGeneratorExecutor:
    """Serialize generators on one card and run different cards concurrently.

    Callbacks must own independent models, optimizers, and RNGs. A callback
    must not mutate a shared evaluator or append to shared training history.
    CPU and single-device runs retain their original sequential ordering.
    """

    def __init__(self, names, devices):
        if len(names) != len(devices):
            raise ValueError("each generator must have one execution device")
        self.names = tuple(names)
        self.groups = {}
        for name, device in zip(names, devices):
            self.groups.setdefault(str(torch.device(device)), []).append(name)
        self.pool = None

    def __enter__(self):
        if len(self.groups) > 1 and all(torch.device(d).type == "cuda" for d in self.groups):
            self.pool = ThreadPoolExecutor(max_workers=len(self.groups))
        return self

    def _group(self, device, names, callback):
        context = torch.cuda.device(device) if torch.device(device).type == "cuda" else nullcontext()
        with context:
            return {name: callback(name) for name in names}

    def map(self, callback):
        if self.pool is None:
            return {name: callback(name) for name in self.names}
        futures = [self.pool.submit(self._group, device, names, callback)
                   for device, names in self.groups.items()]
        combined = {}
        for future in futures:
            combined.update(future.result())
        return {name: combined[name] for name in self.names}

    def __exit__(self, *exception):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
